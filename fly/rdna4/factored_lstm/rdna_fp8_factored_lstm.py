"""Fused factored-LSTM step for RDNA4 (gfx120x, wave32).

Reconstruction of the fp8/RDNA source that the CUDA int8 port (cute/ampere/factored_lstm/
factored_lstm_i8.py) was written from. Computes one LSTM timestep with BOTH projections kept
low-rank (no K=H GEMM):

    gates[B, 4H] = hh_down[B,K_hh] @ up_hh[4H,K_hh]^T
                 + x_down[B,R]    @ up_ih[4H,R]^T   + bias_hh + bias_ih
    i,f,o = sighard(gate) = clamp(0.2*gate + 0.5, 0, 1)
    g     = clamp(gate, -1, 1)
    c_new = f*c + i*g                          (f32, written in place)
    h_new = o * tanh(c_new)  in [-1,1]  ->  int8 at fixed scale 1/127

Faithful to the reference: the two up-projections stay f16 (RDNA4 WMMA f16 -> f32 accumulate);
only the hidden-state output is quantized (int8 @ 1/127, h in [-1,1]). The recurrent
down-projection is a SEPARATE fp8 GEMM (rdna_fp8_preshuffle_gemm, fp8 -> f16), exported
alongside this step.

The two projections are MERGED along K so a single K=K_hh+R GEMM feeds all four gates:
    A      = concat([hh_down, x_down], dim=1)          -> [B, Kc], Kc = K_hh + R   (f16)
    W_gate = concat([up_hh[gate], up_ih[gate]], dim=1) -> [H, Kc]  per gate (f16)
The four gate weights W_i/f/g/o are four [H, Kc] tensors. All four accumulators cover the SAME
(batch, hidden) output element, so the LSTM epilogue combines them elementwise (like the
2-accumulator silu(gate)*up dual kernel, generalized to 4).

Structure mirrors kernels/rdna_f16_gemm.py: 2x2-ish warp LDS kernel, double-buffered K-loop,
row-major A[B,Kc] / W[H,Kc] staged through LDS, WMMA operands read from LDS. Extended from one
B operand to four gate weights + the fused LSTM register epilogue (no gate/cell intermediates
touch DRAM).
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.ir import InsertionPoint
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
H_OUT_SCALE = 1.0 / 127.0   # fixed int8 hidden-output scale (h in [-1,1])

GATES = ("i", "f", "g", "o")


@functools.lru_cache(maxsize=32)
def compile_fp8_factored_lstm(
    *,
    H: int,          # hidden dim (output N)
    K_hh: int,       # hidden rank
    R: int,          # input rank; Kc = K_hh + R (merged contraction)
    B: int = None,   # trace-only batch (M is runtime-dynamic)
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
    a_k_pad: int = 8,
    b_k_pad: int = 8,
):
    """Compile the fused factored-LSTM step for RDNA4.

    Returns launcher(h_out_i8, cell_f32, A_f16, Bi, Bf, Bg, Bo, bias_i, bias_f, bias_g, bias_o,
                     m, stream) where A is [M, Kc], Bx are [H, Kc], biases are [H], cell/h are
    [M, H]. M must be a runtime multiple of tile_m; H a multiple of tile_n; Kc a multiple of
    tile_k.
    """
    Kc = K_hh + R
    WAVE_SIZE = 32
    reg_m = tile_m // WMMA_M          # 32/16 = 2
    reg_n = tile_n // WMMA_N          # 32/16 = 2
    reg_k = tile_k // WMMA_K          # 32/16 = 2
    assert tile_m % WMMA_M == 0 and tile_n % WMMA_N == 0 and tile_k % WMMA_K == 0
    assert reg_k >= 2 and reg_k % 2 == 0
    assert Kc % tile_k == 0, f"Kc={Kc} must be a multiple of tile_k={tile_k}"
    assert H % tile_n == 0, f"H={H} must be a multiple of tile_n={tile_n}"

    # Single wave (32 threads) per block — correctness-first; keeps register/LDS pressure low
    # with four gate accumulator sets.
    waves_m, waves_n = 1, 1
    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    BLOCK_M, BLOCK_N, BLOCK_K = tile_m, tile_n, tile_k

    LOAD_VEC = 8                      # 8 f16 = 128-bit buffer_load
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B
    LDS_ONE_BUF = LDS_A_SIZE + 4 * LDS_B_SIZE      # A + four gate weight tiles
    LDS_TOTAL = 2 * LDS_ONE_BUF                    # double-buffered

    A_TILE_ELEMS = BLOCK_M * BLOCK_K
    B_TILE_ELEMS = BLOCK_N * BLOCK_K
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)
    assert NUM_A_LOADS >= 1 and NUM_B_LOADS >= 1

    num_k_tiles = Kc // BLOCK_K
    assert num_k_tiles >= 2, "need >=2 K-tiles for the prefetch pipeline"
    grid_n = H // BLOCK_N

    gpu_arch = get_rocm_arch()
    elem_bytes = 2
    allocator = SmemAllocator(None, arch=gpu_arch)
    lds_byte_offset = allocator._align(allocator.ptr, elem_bytes)
    allocator.ptr = lds_byte_offset + LDS_TOTAL * elem_bytes

    @flyc.kernel
    def kernel_factored_lstm(
        arg_h: fx.Tensor,        # [M, H]  int8  out
        arg_cell: fx.Tensor,     # [M, H]  f32   in/out
        arg_a: fx.Tensor,        # [M, Kc] f16
        arg_bi: fx.Tensor, arg_bf: fx.Tensor, arg_bg: fx.Tensor, arg_bo: fx.Tensor,  # [H, Kc] f16
        arg_bias_i: fx.Tensor, arg_bias_f: fx.Tensor, arg_bias_g: fx.Tensor, arg_bias_o: fx.Tensor,  # [H] f32
        arg_grid_m: fx.Int32,
    ):
        v8_f16 = T.vec(8, T.f16)
        lds_base = allocator.get_base()
        lds = SmemPtr(lds_base, lds_byte_offset, v8_f16, shape=(LDS_TOTAL // LOAD_VEC,))

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")
        lane = tid % 32
        lane16 = lane % 16
        klane = lane // 16
        base8 = klane * 8

        # 1D block grid over (grid_m * grid_n); row-major
        grid_n_c = fx.arith.constant(grid_n, type=fx.T.i32())
        pid_i32 = fx.arith.index_cast(fx.T.i32(), pid)
        bid_m = fx.arith.index_cast(fx.T.index(), pid_i32 // grid_n_c)
        bid_n = fx.arith.index_cast(fx.T.index(), pid_i32 % grid_n_c)
        tile_m0 = bid_m * BLOCK_M
        tile_n0 = bid_n * BLOCK_N

        a_rsrc  = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        bi_rsrc = buffer_ops.create_buffer_resource(arg_bi, max_size=True)
        bf_rsrc = buffer_ops.create_buffer_resource(arg_bf, max_size=True)
        bg_rsrc = buffer_ops.create_buffer_resource(arg_bg, max_size=True)
        bo_rsrc = buffer_ops.create_buffer_resource(arg_bo, max_size=True)
        b_rsrcs = [bi_rsrc, bf_rsrc, bg_rsrc, bo_rsrc]
        h_rsrc    = buffer_ops.create_buffer_resource(arg_h, max_size=True)
        cell_rsrc = buffer_ops.create_buffer_resource(arg_cell, max_size=True)
        bias_rsrcs = [buffer_ops.create_buffer_resource(t, max_size=True)
                      for t in (arg_bias_i, arg_bias_f, arg_bias_g, arg_bias_o)]

        # LDS store maps (A tile, then 4 B tiles). LDS regions: A at 0, B[gate] at
        # LDS_A_SIZE + gate*LDS_B_SIZE within a buffer.
        a_lds_info = []
        for al in range_constexpr(NUM_A_LOADS):
            lin = tid * LOAD_VEC + al * THREADS_PER_BLOCK * LOAD_VEC
            row, col = lin // BLOCK_K, lin % BLOCK_K
            a_lds_info.append((tile_m0 + row, col, row * BLOCK_K_PAD_A + col))
        b_lds_info = []
        for bl in range_constexpr(NUM_B_LOADS):
            lin = tid * LOAD_VEC + bl * THREADS_PER_BLOCK * LOAD_VEC
            row, col = lin // BLOCK_K, lin % BLOCK_K
            b_lds_info.append((tile_n0 + row, col, row * BLOCK_K_PAD_B + col))

        def _gmem_load(k_base):
            raw = []
            for al in range_constexpr(NUM_A_LOADS):
                g_row, col, _ = a_lds_info[al]
                off = (g_row * Kc + k_base + col) // 2
                raw.append(buffer_ops.buffer_load(a_rsrc, off, vec_width=4, dtype=fx.Float32))
            for gate in range_constexpr(4):
                for bl in range_constexpr(NUM_B_LOADS):
                    g_row, col, _ = b_lds_info[bl]
                    off = (g_row * Kc + k_base + col) // 2
                    raw.append(buffer_ops.buffer_load(b_rsrcs[gate], off, vec_width=4, dtype=fx.Float32))
            return raw

        def _lds_store(raw, buf_off):
            idx = 0
            for al in range_constexpr(NUM_A_LOADS):
                _, _, rel = a_lds_info[al]
                lds.store(raw[idx].bitcast(fx.Float16), [(buf_off + rel) // 8]); idx += 1
            for gate in range_constexpr(4):
                gbase = LDS_A_SIZE + gate * LDS_B_SIZE
                for bl in range_constexpr(NUM_B_LOADS):
                    _, _, rel = b_lds_info[bl]
                    lds.store(raw[idx].bitcast(fx.Float16), [(buf_off + gbase + rel) // 8]); idx += 1

        def _load_a(rk, buf_off, rm):
            col = 16 * rk + base8
            row = 16 * rm + lane16
            return lds.load([(buf_off + row * BLOCK_K_PAD_A + col) // 8])

        def _load_b(rk, buf_off, gate, rn):
            col = 16 * rk + base8
            row = 16 * rn + lane16
            gbase = LDS_A_SIZE + gate * LDS_B_SIZE
            return lds.load([(buf_off + gbase + row * BLOCK_K_PAD_B + col) // 8])

        def _barrier():
            gpu.barrier()

        def _compute(accs_in, rk, buf_off):
            # accs_in layout: [gate*reg_m*reg_n + rm*reg_n + rn]
            accs = list(accs_in)
            for gate in range_constexpr(4):
                b_vecs = [_load_b(rk, buf_off, gate, rn) for rn in range_constexpr(reg_n)]
                for rm in range_constexpr(reg_m):
                    a_vec = _load_a(rk, buf_off, rm)
                    for rn in range_constexpr(reg_n):
                        idx = gate * (reg_m * reg_n) + rm * reg_n + rn
                        accs[idx] = rocdl.wmma_f32_16x16x16_f16(accs[idx].type, a_vec, b_vecs[rn], accs[idx]).result
            return accs

        n_acc = 4 * reg_m * reg_n
        zero = fx.full(8, 0.0, fx.Float32)
        accs = [zero for _ in range_constexpr(n_acc)]

        stride = LDS_ONE_BUF
        _lds_store(_gmem_load(0), 0)
        _barrier()

        for iv, state in range(0, num_k_tiles - 1, 1, init=list(accs)):
            s_accs = list(state[:n_acc])
            read_off = (iv % 2) * stride
            write_off = (1 - iv % 2) * stride
            nxt = _gmem_load((iv + 1) * BLOCK_K)
            for rk in range_constexpr(reg_k):
                s_accs = _compute(s_accs, rk, read_off)
            _lds_store(nxt, write_off)
            _barrier()
            results = yield list(s_accs)
        accs = list(results[:n_acc])

        last_off = ((num_k_tiles - 1) % 2) * stride
        for rk in range_constexpr(reg_k):
            accs = _compute(accs, rk, last_off)

        # ── LSTM epilogue: gates, cell update (f32 in place), int8 hidden out ──
        zero_f = arith.constant(0.0, type=T.f32)

        def clampf(x, lo, hi):
            # cute.arch/rocdl expose only fmax; min(y,hi) = -fmax(-y,-hi).
            lo_c = arith.constant(lo, type=T.f32)
            neg_hi = arith.constant(-float(hi), type=T.f32)
            m = ArithValue(x).maximumf(lo_c)                 # max(x, lo)
            neg_capped = (ArithValue(zero_f) - m).maximumf(neg_hi)   # max(-m, -hi) = -min(m, hi)
            return ArithValue(zero_f) - neg_capped           # min(max(x,lo), hi)

        c02 = arith.constant(0.2, type=T.f32)
        c05 = arith.constant(0.5, type=T.f32)
        c127 = arith.constant(127.0, type=T.f32)
        c1_f32 = arith.constant(1.0, type=T.f32)
        c1e30 = arith.constant(1e30, type=T.f32)
        neg_2log2e = arith.constant(-2.8853900817779268, type=T.f32)   # -2 * log2(e); for tanh via exp2

        def tanhf(x):
            # tanh(x) = 2*sigmoid(2x) - 1; sigmoid(2x) = 1/(1+exp(-2x)), exp via exp2.
            emu = ArithValue(rocdl.exp2(fx.T.f32(), ArithValue(x) * neg_2log2e))   # exp(-2x)
            sig = ArithValue(rocdl.rcp(fx.T.f32(), c1_f32 + emu))                  # 1/(1+exp(-2x))
            return sig * arith.constant(2.0, type=T.f32) - c1_f32
        for rm in range_constexpr(reg_m):
            for rn in range_constexpr(reg_n):
                # per-N (hidden col) bias: load the 8 columns this lane owns? WMMA output
                # layout: lane holds 8 rows (base8+si) at column lane16. So the N column is
                # fixed per (rn, lane16); bias is indexed by that column.
                g_col = tile_n0 + 16 * rn + lane16
                bias_vals = [ArithValue(buffer_ops.buffer_load(bias_rsrcs[gate], g_col, vec_width=1, dtype=fx.Float32))
                             for gate in range_constexpr(4)]
                ai = accs[0 * reg_m * reg_n + rm * reg_n + rn]
                af = accs[1 * reg_m * reg_n + rm * reg_n + rn]
                ag = accs[2 * reg_m * reg_n + rm * reg_n + rn]
                ao = accs[3 * reg_m * reg_n + rm * reg_n + rn]
                for si in range_constexpr(8):
                    g_row = tile_m0 + 16 * rm + base8 + si
                    raw_i = ArithValue(ai[si]) + bias_vals[0]
                    raw_f = ArithValue(af[si]) + bias_vals[1]
                    raw_g = ArithValue(ag[si]) + bias_vals[2]
                    raw_o = ArithValue(ao[si]) + bias_vals[3]
                    i_a = clampf(raw_i * c02 + c05, 0.0, 1.0)
                    f_a = clampf(raw_f * c02 + c05, 0.0, 1.0)
                    o_a = clampf(raw_o * c02 + c05, 0.0, 1.0)
                    g_a = clampf(raw_g, -1.0, 1.0)
                    cell_off = g_row * H + g_col
                    c_old = ArithValue(buffer_ops.buffer_load(cell_rsrc, cell_off, vec_width=1, dtype=fx.Float32))
                    c_new = f_a * c_old + i_a * g_a
                    buffer_ops.buffer_store(c_new, cell_rsrc, cell_off)
                    h_new = o_a * tanhf(c_new)
                    # int8 = round(h_new * 127), round-half-away-from-zero (h in [-1,1]):
                    # fptosi truncates toward zero, so add sign(hs)*0.5 first.
                    hs = h_new * c127
                    half = clampf(hs * c1e30, -0.5, 0.5)
                    q = arith.fptosi(fx.T.i8(), hs + half)
                    buffer_ops.buffer_store(q, h_rsrc, cell_off)

    @flyc.jit
    def launch_factored_lstm(
        arg_h: fx.Tensor, arg_cell: fx.Tensor, arg_a: fx.Tensor,
        arg_bi: fx.Tensor, arg_bf: fx.Tensor, arg_bg: fx.Tensor, arg_bo: fx.Tensor,
        arg_bias_i: fx.Tensor, arg_bias_f: fx.Tensor, arg_bias_g: fx.Tensor, arg_bias_o: fx.Tensor,
        m: fx.Int32, stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        c1 = 1
        dyn_grid_m = m // BLOCK_M
        total_blocks = dyn_grid_m * grid_n
        launcher = kernel_factored_lstm(
            arg_h, arg_cell, arg_a, arg_bi, arg_bf, arg_bg, arg_bo,
            arg_bias_i, arg_bias_f, arg_bias_g, arg_bias_o, dyn_grid_m)
        launcher.launch(grid=(total_blocks, c1, c1), block=(THREADS_PER_BLOCK, c1, c1), stream=stream)

    return launch_factored_lstm


__all__ = ["compile_fp8_factored_lstm", "H_OUT_SCALE"]
