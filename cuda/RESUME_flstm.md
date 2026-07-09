# fLSTM recurrence optimization — resume handoff

Full history is in Claude memory (`dorado-factorised-lstm-kernel.md`, 37 rounds). This is the
in-flight state so a new session continues without re-deriving.

## Where we are (2026-07-09)
- Target: match NVIDIA koi's factorised-LSTM gate. koi = **13.9 µs/step @ N=1536** (measured),
  100% int8 (no int4), one persistent kernel (8 gate CTAs + 1 down-proj CTA), grid(9,12).
- **Shippable safe winners (bit-exact fwd+reverse):**
  - C++ `fused_cutlass.cu` = **37.05 µs/step @ N=2048** (gate 30.2 + down 8.7); 26 µs @ N=1536. 1.7× over the 64 µs two-kernel baseline.
  - DSL `../cute/ampere/factored_lstm/factored_lstm_gate_fused_rl.py` = **33.4 µs gate** (parity).
- Every lever measured (all in `.cu` headers here): resident-weights (smem-pipe bound, self-defeating),
  8:1 fragment-reuse via ldmatrix (koi8_gate.cu — achieved but 3× slower, A-LDS bound), persistence
  (sync-bound / smem-pipe bound), L2-pin (0), batch (N fixed ≤2048). int4 dropped (needs per-token
  activation quant = serial bottleneck; up-proj int4 is accuracy-OK per slorado/scripts/sensitivity_results.tsv
  but the quant cost eats it).

## RESOLVED (38th round, 2026-07-09): koi_gate.cu — "the untried lever" WORKED
koi's gate SASS (scratchpad `koicubin/g07.txt`): 320 IMMA : 40 LDSM : 128 LDG.E.CONSTANT.
Built the one operand-role combination never built — and it BEAT fused_cutlass:
- **koi_gate.cu = 25.9µs gate @N=1536, bit-exact fwd+reverse, vs fused_cutlass 26.2µs** — FIRST hand kernel to beat CUTLASS on this shape.
- ncu (win cfg GX=16/BMG=256/MSET=2/1-CTA-SM/255 regs): imma 22.4% (2× fused_cutlass's 11%), L1TEX 74% (weights off the pipe), long_scoreboard 2.98 (LDG hidden), L2-hit 97.9%, DRAM ~0.5MB/step. SASS 64 IMMA : 16 LDSM(A) : ~300 LDG(weights) — matches koi.
- The 4 levers, none combined before: ldmatrix-A + `__ldg` weights L2→reg (OFF smem pipe) + weight-M-reuse (MSET) + **CELL RESIDENT in registers across T (the breakthrough, 36.5→25.9)**.
- REFINEMENT: once weights are off the smem pipe, **weight-M-reuse is the lever, NOT occupancy** — MSET=2@1-CTA/SM (25.9) beats MSET=1@2-CTA/SM (34-38).
- Plateau ~26µs; residual = weight-LDG L1TEX transit at forced 12.5% occ. Did NOT reach koi's ~8µs gate / 13.9µs full-step.

**M2 (full-step fusion): NOT built — gate-dominated.** Recurrence strictly serial; fused ≈ down(9)+gate(26)+2 barriers ≈ 36µs = persistent_cutlass (37.86), not 13.9. koi's 13.9 implies ~8µs gate; ours plateaus at 26 → fusion can't recover it.

**Shippable epilogue win: fused_cutlass_epi.cu = gate 23.82→22.43µs (+5.8%), bit-exact.** 3a+3b: channel-outer restructure hoists the m_it-invariant 8 scale/bias smem reads + as-multiply out of the IterM loop (once/channel vs 4×) via epi_elem2. 3c f16 tanh.approx.f16 (flagged OFF): not faster + ~9% elements differ, dropped.

### Round 39 (2026-07-09, autonomous, int8-only)
- **int4 is OFF THE TABLE** (user: accuracy bad in practice; koi is 100% int8). `koi_gate_i4.cu` built + bit-exact (23.2µs) but rejected — datapoint only. See memory `flstm-int4-rejected`.
- **Bank conflicts fixed**: koi_gate's ldmatrix-A had 44M shared-load bank conflicts (A tile stride KC=256B≡0 mod 32). `koi_gate_pad.cu` (APAD=16→272B stride) → 0 conflicts, L1TEX 65→58%, bit-exact. **But wall-clock unchanged (26.6µs): latency-bound at 12.5% occ, NOT L1TEX-bound.** Keep the padding (hygiene + headroom). This is the ship-candidate base now.
- Re-confirmed: MSET wall (MSET=4 spills), weights already on constant-cache path, 2 CTA/SM blocked by reg cap, epilogue hoist a wash on koi_gate.
- **Only remaining int8 lever = cross-step software-pipelined mainloop** (overlap step t+1 load/ldmatrix/MMA with step t epilogue at fixed 1-CTA/SM). Hard from-scratch rewrite; NOT attempted (won't converge in one sitting).

### Rounds 40-41 (2026-07-09)
- **Cross-step pipeline on full fused kernel = DEAD END.** `koi_fused_pipe.cu` (koi_flstm structure + koi_gate's best gate mainloop grafted in) = 189µs, same as before. Dependency chain G[t]→h[t]→D[t+1] forbids D/G overlap; kernel is sync/alternation-bound (barrier-stall 40/issue), NOT gate-compute-bound. NO_BARRIER (unsafe) = 150µs, still 10× koi. Confirms fused_cutlass (per-step graph launches) is the right structure.
- **★ COALESCED h WRITEOUT: gate 26→23.1µs, bit-exact.** `koi_gate_co.cu` stages h into smem then STG.128 (was 64 scattered STG.E.U8/step). Win is latency (global stores off critical path). Pad sHo to kill staging conflicts (swizzle equivalent, smem not the limiter).
- **★ CODE-SMELL SWEEP: gate 23.1→22.6µs, bit-exact.** `koi_gate_sm.cu` folds as=1/127 into resident scales (FMUL 448→197). Non-wins: tanhf already MUFU.TANH; register spill already ~0 (8B — coalescing+as-fold freed it, was 176-308B); double-buffer sHo neutral; PIPE=1 overflows smem. **Ship-candidate base = koi_gate_sm.cu.**

- **★ util-plan Phase 0+2a (see memory flstm-gate-util-plan): gate 22.6→18.0µs, bit-exact.** Phase 0 decomposed 22.6µs = MMA 14 (37% imma, wait-bound) + epi-math 4.2 + writeout 4.4. Phase 2a (`koi_gate_w16.cu`): the 64 scattered STS.U8 byte-stores WERE the writeout cost → pack adjacent pairs into 32 STS.U16 → writeout now free. **NEW BEST = koi_gate_w16.cu 18.0µs.** Remaining: MMA 14 (wait-bound, Phase 1 MMA-ILP next, likely low-yield per CORE evidence) + epi-math 4.

### Standing recommendation
Ship `fused_cutlass` + the 3a+3b epilogue hoist. The best research gate is now **`koi_gate_w16.cu` (18.0µs @N1536)** — lean (tanh=MUFU, weights constant-path, A conflict-free, output coalesced STG.128, scales folded, ~0 spill), latency-bound at 12.5% occ. ~1.6× off koi's 13.9µs full step. Progression: fused_cutlass 26.2 → koi_gate 26 → +conflict-free-A → +coalesced-writeout 23.1 → +as-fold 22.6.

### ⚠️ UNFINISHED at handoff — the next continuation step
The N=1536 isolated-gate epi win (23.82→22.43) is CONFIRMED; the **epi-opt at N=2048 + graph full-step numbers were NOT run**. Do that first: build `fused_cutlass_epi.cu` at N=2048, graph-captured full-step, confirm +5.8% survives and stays bit-exact. THEN endgame policy (DSL port or ship C++).

## Rules
New files only (never modify fused_cutlass.cu / koi8_gate.cu / DSL kernels). Bit-exact fwd+reverse
(except the f16-tanh flag). Warm GPU + median timing. Prefix runs `CUDA_VISIBLE_DEVICES=<n>`.
Build: `nvcc -arch=sm_80 -O3 --use_fast_math -I../thirdparty/cutlass/include -DSTG=4 -DFP16CELL koi_gate.cu -o koi_gate`.

## To measure e2e (paused integration)
slorado `/data/bonwon/slorado`, build `export CUDA_ROOT=/usr/local/cuda-12.6 && make -j cuda=1 cxx11_abi=1`.
Bench: `./scripts/bench/bench.sh slorado cuda:N` (500k, hac@v6, --quant=int8, auto-accuracy vs /genome/hg38noAlt.idx).
Integrate winner into fluke/src/fused_cuda.cpp; gate_w needs the interleave; down-proj stays int8.
Endgame policy: ship DSL _rl (delete thirdparty/cutlass + .cu) if e2e parity holds, else ship C++.
