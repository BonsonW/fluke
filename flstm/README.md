# Fused factorised-LSTM step (int8) — sm_80 / A100

`cuda/sm80_factored_lstm_fused_i8.cu` — one persistent CUDA kernel that computes the full
factorised-LSTM **recurrent step** in int8: the recurrent down-projection, the 4-gate GEMM, and
the LSTM cell, fused into a single launch that loops all timesteps. Built and tuned for NVIDIA
sm_80 (A100); written to be a portable template (the design principles carry to other arches,
the tile knobs re-tune per arch).

The in-file header comment is the authoritative design document — start there. This README is the
operational summary and porting guide.

## What it computes (per step t, hidden H=1024, batch N)

    A_t     = concat( dp(h_{t-1}) , x_dp[t] )      # [N x 256]   recurrent-down half + input half
    gate    = A_t @ Wg^T                           # [N x 4096]  int8·int8 -> int32, gates i,f,g,o
    h_t     = lstm_cell(gate, c_{t-1})             # hard-sigmoid / tanh / cell -> int8
    dp(h_t) = clamp( (h_t @ Wdn^T) * comb )        # [N x 128]   the rank-128 recurrent down-projection

The recurrent weight is low-rank factorised (`W_rec = Wg[:,0:128] @ Wdn`, rank 128), so each step
is a memory-bound down-projection feeding a short-K (K=256) wide-output (4H=4096) gate GEMM. The
recurrence is strictly sequential in t.

## Performance

- **19.3 µs/step** @ N=1536, T=2048 on A100 (warm, 1410 MHz), int8, bit-consistent (±1 int8 vs the
  f32 reference; exact-f32 build available).
- **Batch-adaptive occupancy** (automatic): 1 CTA/SM while the grid fits one wave (`GX*N/BMG ≤ #SMs`,
  i.e. N ≤ ~1664), 2 CTA/SM above that to co-reside the overflow instead of paying a second wave.
  N=2048 → 33.8 µs/step. `-DFORCECP=n` overrides.

## Build / test / bench

    NVCC="nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177,20013,550"
    $NVCC cuda/sm80_factored_lstm_fused_i8.cu -o flstm

    ./flstm --N 1536 --T 8              # correctness: compares kernel vs scalar ref -> PASS (maxd<=1)
    ./flstm --N 1536 --T 2048 --bench   # warm bench: median us/step

Correctness is checked in-kernel against `ref_kernel` (a trivial, obviously-correct scalar
implementation). Every configuration must pass before it is trusted — most speed bugs are silent
correctness bugs. Always warm the GPU before timing (cold-clock reads 20–40% slow).

### Build knobs

| flag | default | meaning |
|------|---------|---------|
| `-DGX=n`   | 8 | CTAs per group (splits the 4H gate output; HX = H/GX per CTA) |
| `-DBMG=n`  | 128 | batch rows per group |
| `-DWROW=n` `-DWCOL=n` | 4, 2 | warp split across rows × channels (must multiply to 8 warps) |
| `-DF16EPI=0` | (on) | use the exact f32 cell/epilogue (bit-exact vs ref) instead of the f16-packed one |
| `-DTANHAPPROX=0` | (on) | use accurate `tanhf` instead of `tanh.approx` (portability) |
| `-DFORCECP=n` | (auto) | force 1 or 2 CTAs/SM instead of the batch-adaptive choice |

## Design principles (the porting recipe)

These are the transferable decisions; re-tune the tile (`GX/BMG/WROW/WCOL`) and re-check register
spill per arch. Full rationale is in the kernel header.

1. **Channel-split the gate** so each CTA keeps a large row-tile → each streamed weight fragment is
   reused across many rows. A row-split gate re-reads the whole weight per CTA and is memory-bound.
2. **Stream weights from L2 via the read-only path** (`__ldg`), not shared memory — keep the scarce
   smem/L1TEX pipe for activations. Weights stay ~98% L2-hit and hide behind the MMAs.
3. **Keep the cell state resident in registers** across the whole T-loop (f16) — never round-trip it.
4. **Pad smem row strides** to be bank-conflict-free for `ldmatrix`.
5. **Fold the dequant scale × (1/127) once** at load time, off the epilogue's dependency chain.
6. **Coalesce + pack all I/O**: stage the int8 output in smem, then 128-bit coalesced stores; pack
   two int8 per 16-bit store. Never scatter sub-word stores to global memory.
7. **Interleave the epilogue with the MMAs** (software-pipeline the gate n-loop so epilogue(nn)
   co-issues with MMA(nn+1)); size the tile so the double-buffered accumulator fits the register budget.
8. **Batch-adaptive occupancy**: one wave at 1 CTA/SM; co-reside 2 CTA/SM when the grid overflows.

## Why 19.3 µs is the ceiling on sm_80 (hardware framing)

The step is **latency-bound, not throughput-bound**. At H=1024 the per-step work is tiny (~0.9 GMAC),
but the sequential recurrence exposes the IMMA accumulate-chain **result latency** at the 8-warp /
1-CTA-per-SM occupancy this tile requires. Measured decomposition (A100): gate-MMA floor ≈ 11.3 µs,
+epilogue +2.7, +down +5.2 (the cross-CTA h round-trip), all synchronisation ≈ 1.3. Larger tiles,
higher occupancy (needs >108 CTAs, impossible at N≤1536), operand prefetching, and cp.async
multistaging were all measured and do **not** move it — the wall is MMA-result latency at low
occupancy, which is a hardware property of sm_80's synchronous `mma.sync`, not a code deficiency.

The direct fix is **sm_90**: `wgmma` (asynchronous warpgroup MMA) decouples MMA issue from the
result-wait, and TMA / thread-block clusters remove the cross-CTA handoff cost. Port there for lower
latency. On AMD/CDNA, apply the same principles with MFMA + `ds_read` (see the header's porting notes).
