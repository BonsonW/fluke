# Factored-LSTM Gate Kernel — Optimization & Porting Guide

A hardware-agnostic guide to the int8 factored-LSTM **gate recurrence** kernel
(`factored_lstm_gate_i8.cu`), how it reaches its current performance, how to reproduce that on
other GPUs (NVIDIA arches and AMD/HIP), and the traps that cost us time.

**Milestone:** 18.0 µs/step at N=1536, T=2048 on an A100 (sm_80), bit-exact forward + reverse.
This is the **gate** half of the recurrence; the down-projection is a separate kernel. Closing
the remaining gap (fusing the two and lifting tensor-core utilisation past the occupancy wall) is
future work — see *Remaining gap* at the end.

---

## 1. The problem

Per timestep the factored-LSTM computes 4 gates for every (batch row × hidden channel):

```
gate[t] = A[t] @ W^T          # A[t] = [N, KC]  ,  W = [4H, KC]   (int8 x int8 -> int32)
h[t], c[t] = lstm_cell(gate[t], c[t-1])        # sigmoid-hard / tanh / cell update -> int8 h
```

with H=1024, 4 gates (i,f,g,o), KC=256 (the concatenated recurrent-down + input contribution).
The recurrence is **strictly sequential in t** (`h[t]` feeds the next step), and each step's GEMM
is *short-K* (KC=256) with a wide output (4H=4096). This shape is **latency-bound**, not
throughput-bound — the central fact that drives every decision below.

---

## 2. Optimization principles (ground-up)

Build in this order. Each step is independently measurable; don't move on until the current one
is bit-exact and profiled.

### 2.1 Persistent single-launch
Loop all T timesteps *inside one kernel launch* rather than launching per step. The recurrence
forbids overlapping steps, so there's nothing to gain from many launches, and a persistent kernel
lets per-thread state (below) live in registers across t. Grid `(GX, N/BMG)`: `GX` CTAs split the
4H gate-output columns, `N/BMG` splits batch rows. Size the grid to about **one wave** (≈ #SMs) so
all CTAs are co-resident.

### 2.2 Tensor-core MMA for the gate GEMM
Use the hardware int8 matrix instruction (`mma.sync.m16n8k32.s8` on NVIDIA sm_80). Everything else
serves keeping this instruction fed.

### 2.3 Operand placement — the highest-leverage decision
Two operands, two different best paths:

- **Activation A** → stage a small tile in shared memory, load fragments with the matrix-load
  instruction (`ldmatrix`). A is small and reused across *all* output channels within a step, so
  the smem round-trip is cheap and amortised.
- **Weights** → stream directly from L2 into registers via the **read-only / constant-cache path**
  (`__ldg` → `LDG.E.CONSTANT`). Do **not** put weights in shared memory. Weights are large; putting
  them on the shared-memory pipe saturates it (that pipe is the scarce resource). The read-only
  path keeps them off it, and they stay hot in L2 (≈98% hit) so the loads are cheap and their
  latency hides behind the MMAs.
- **Reuse (register blocking):** each loaded weight fragment is reused across `MSET` row-tiles;
  each A fragment is reused across all channels. Reuse is what makes the streamed-weight approach
  win — see the trap in §5 about chasing occupancy instead.

### 2.4 Keep the recurrent cell resident in registers
The cell state `c` is read and written every step. Keep it in **registers** across the whole
T-loop (each thread owns a fixed set of (row,channel) cells) so it never round-trips global
memory. This was individually the single largest speedup in our history. Store it at the precision
the reference uses (f16 here) so it stays bit-exact.

### 2.5 Bank-conflict-free shared memory
A row-major tile whose row stride is a multiple of the bank count makes every row of a matrix-load
tile hit the same banks. **Pad the row stride** so consecutive rows step across banks
(here: stride KC+16 bytes = 68 words ≢ 0 mod 32). Verify with the profiler (§4) — conflicts should
read zero.

### 2.6 Get constants out of the inner loop
The per-channel dequant scale and the fixed activation scale (1/127) multiply *every* accumulator
in the epilogue. Fold them together **once**, when the per-channel scales are loaded into smem,
instead of recomputing the product per element on the epilogue's dependency chain.

### 2.7 Coalesced, wide, packed output stores
Never write the int8 output one byte at a time — scattered sub-word global stores are expensive and
sit on the critical path. Instead:
1. stage the output tile in shared memory, then write it out as **128-bit** coalesced stores;
2. when staging, **pack adjacent elements** into the widest store you can (two int8 → one 16-bit
   store here). Per-byte shared stores were, on their own, ~25% of this kernel's runtime.

---

## 3. Reproduce / port — the method (not the answer)

The optimizations above are the *destination*. The way to get there on new hardware is a
disciplined loop; do **not** copy tile sizes blindly.

1. **Write a scalar reference first.** A trivial, obviously-correct kernel computing the same math.
   Every optimized variant is checked bit-exact against it (forward *and* reverse). This is
   non-negotiable — most speed bugs are silent correctness bugs. (`ref_kernel` + `--reverse` in the
   .cu.)
2. **Compute the roofline.** int8 compute time vs the traffic floor (bytes moved / bandwidth). If
   compute ≪ traffic, you're memory/latency-bound → optimize placement & latency-hiding, not FLOPs.
3. **Find the actual bound with the profiler.** Look at the Speed-of-Light section (which pipe is
   highest), then the *stall reasons*. Don't optimize a pipe that isn't the bottleneck.
4. **Decompose the cost.** Build strip-down variants to attribute time to phases — here `-DCORE`
   (MMA only) and `-DNOWRITE` (MMA + epilogue, no writeout). This told us the writeout was ~4.4 µs
   of 22.6 µs *before* we spent effort on it. Always know where the time is before cutting.
5. **Tune the tile config for the GPU.** `GX` (output-column split) and `BMG` (rows/CTA, sets the
   weight-reuse factor `MSET`) are the two knobs. Sweep them watching register spill and the grid's
   wave count. On a different GPU the sweet spot moves.
6. **Iterate 3–5** until stalls flatten or you hit an occupancy/register wall.

### Profiling recipes (NVIDIA, `ncu`)
Generalize the metric *names* per arch, but these are the ones that mattered:

```
# Speed-of-light: which pipe is the ceiling
ncu --metrics \
 sm__throughput.avg.pct_of_peak_sustained_elapsed,\
 l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
 lts__throughput.avg.pct_of_peak_sustained_elapsed,\
 gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
 sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
 sm__warps_active.avg.pct_of_peak_sustained_active <bin> --N 1536 --T 128

# Stall reasons (what the idle cycles are waiting on)
ncu --metrics \
 smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
 smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
 smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
 smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
 smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio <bin> --N 1536 --T 128

# Shared-memory bank conflicts (must be 0 after padding)
ncu --metrics \
 l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
 l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum <bin> --N 1536 --T 128

# Register/spill at compile time
nvcc ... -Xptxas -v      # want 0 bytes spill; note the register count vs the 255 cap
```

Reading them (rules of thumb): `long_scoreboard` = global/L2 latency (hide with prefetch / more
reuse); `short_scoreboard` = smem→MMA operand latency; `mio_throttle` = smem/LSU pipe throughput
(too much LDS/ldmatrix/STS); `wait` = fixed-latency-instruction (MMA) pipeline not hidden — needs
more independent work in flight (ILP or occupancy); `barrier` = sync. If **no** single stall
dominates and occupancy is low, you're latency-bound at low occupancy — the structural wall.

### Warm the clocks before timing
Always run the kernel a few times before measuring, and take a median. Cold-clock runs read
20–40% slow and have produced false conclusions here more than once. On a shared machine, pin to an
idle device and re-check it stayed idle (contention doubles the numbers).

---

## 4. Porting to other NVIDIA architectures (sm_86 / 89 / 90)
The kernel is written in raw PTX (`mma.sync`, `ldmatrix`, `cp.async`, `__ldg`) and is arch-portable
as-is, but retune:
- **Shared-memory opt-in cap** differs (A100 ≈ 164 KB). It bounds tile size / CTAs-per-SM.
- **#SMs** differs → the "one wave" grid size (and thus the sweet-spot batch N) changes.
- **sm_90** adds thread-block clusters, TMA, and larger async-copy facilities. The
  streamed-weight + resident-cell structure still applies; cluster barriers could enable a cheaper
  fused down+gate than is possible on sm_80 (see *Remaining gap*).
- Re-sweep `GX`/`BMG` and re-check spill — the register file is the same size but the tile that
  fits without spilling can shift with the arch's scheduler.

## 5. Porting to AMD / HIP
The *principles* map directly; the *instructions* change. Equivalences:

| NVIDIA (this kernel)        | AMD / HIP equivalent                               |
|-----------------------------|----------------------------------------------------|
| `mma.sync.m16n8k32.s8`      | `__builtin_amdgcn_mfma_*_i8` (MFMA; CDNA) / WMMA (RDNA3) |
| `ldmatrix` from smem        | `ds_read_b128` patterns into VGPRs (no ldmatrix; lay out the tile so `ds_read` feeds MFMA lanes) |
| `__ldg` / `LDG.E.CONSTANT`  | `buffer_load` / `__builtin_nontemporal_load` (keep weights off LDS, in VGPRs from L2) |
| `cp.async`                  | CDNA async global→LDS where available; else plain global load + LDS store |
| shared-memory bank conflicts| LDS bank conflicts — same idea, **32 banks**; pad likewise |
| warp = 32 threads           | **wavefront = 64** on CDNA — retile fragment ownership and the 8×8/16×16 lane maps accordingly |
| `__syncthreads`             | `__syncthreads` (works in HIP)                     |

Key AMD-specific cautions: (1) the MFMA fragment layouts differ from NVIDIA's — re-derive the
per-lane (row,col) ownership and validate the scalar-load path against the reference *first*, then
the MFMA path. (2) Wavefront 64 changes how many rows/cols a "warp" owns; the tile geometry
(`WARPS`, `MSET`, the epilogue index math) must be redone. (3) There is no `ldmatrix`; you place
the activation tile in LDS and issue `ds_read` in the layout MFMA expects. (4) Profile with
`rocprof`/Omniperf — the equivalent counters are LDS bank conflicts, VALU/MFMA busy, and issue
stalls; the same "find the bound before optimizing" discipline applies.

The fluke DSL kernels (`cute/…`, `fly/…`) are the intended production form; this .cu is the
hand-tuned reference that establishes the achievable number and the recipe.

---

## 6. Traps to avoid (fLSTM-specific — we hit these)
- **Chasing occupancy.** More CTAs/SM (2+) *lowers* performance here: it needs a smaller register
  budget, which forces dropping the resident cell or spilling — and it floods the read-only cache
  path with more concurrent weight loads. Once weights are off the shared-memory pipe, **weight
  reuse at 1 CTA/SM beats higher occupancy.** Don't optimize occupancy for its own sake.
- **Putting weights in shared memory.** It looks like it should help (load once, reuse) but it
  saturates the smem/LSU pipe — the exact resource you're trying to protect. Keep weights on the
  read-only/L2 path.
- **int4.** Do not use int4 for this recurrence. It halves traffic and doubles TOPS on paper, but
  the cumulative accuracy loss across the stacked recurrent layers is unacceptable. Stay int8.
- **Sub-word / scattered stores.** Per-byte `STG`/`STS` are silently expensive. Always stage +
  coalesce + pack.
- **Optimizing a pipe that isn't the bound.** Bank-conflict removal here *reduced* a profiler
  number (L1TEX 65→58%) but did **not** change wall-clock, because the kernel was latency-bound,
  not L1TEX-throughput-bound. Measure the bound first; a high SOL % can be inflated by conflicts
  yet still not be the binding constraint.
- **Cross-step pipelining of the gate.** In a *standalone* gate benchmark the next step's inputs
  are precomputed, so you can "overlap" steps and get a great-looking number — but it's a mirage:
  in the real recurrence the next step's input depends on this step's output. Optimize *within*
  a step; only the down+gate fusion (future) exposes real cross-phase overlap, and even that is
  bounded by the sequential dependency.
- **Trusting cold-clock or contended timings.** Warm the GPU, take medians, pin to an idle device.

---

## 7. Remaining gap (future work)
At the milestone the gate is ~18 µs and dominated by MMA-pipeline latency at low occupancy
(1 CTA/SM) — the structural wall. The realistic next levers, none of them "basic":
- **Fuse the down-projection into the gate** as a single persistent step. The dependency chain
  (down→gate→h→down) is sequential, so the win is a correct, cheap producer/consumer handoff and a
  high-utilisation *within-step* pipeline — not cross-step overlap. Needs a real cross-CTA
  handshake (release/acquire), which on sm_80 costs; sm_90 cluster barriers may help.
- **Lift tensor-core utilisation past the occupancy wall.** More independent MMAs in flight needs
  more registers (at the cap) or more warps (a regression here). This is the genuine open problem
  and likely needs a from-scratch register-blocked mainloop.
- Establish the AMD number by porting per §5.
