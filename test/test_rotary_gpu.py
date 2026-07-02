"""Build-and-run test for the GPU rotary embedding kernel.

Builds lib/libfluke.so via `make` (fluke_lib) and calls the real fluke_rotary_emb_gpu
through ctypes on q (chunk 0) and k (chunk 1) of a qkv tensor, then diffs against a
torch reference -> PASS/FAIL. Also benchmarks with CUDA events.

    ./pyvenv/bin/python test/test_rotary_gpu.py

Requires a CUDA-enabled torch (see test/requirements.txt) and a CUDA toolkit.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # fluke root (fluke_lib)
import fluke_lib

# TxModel.cpp constants (MultiHeadAttentionImpl / RotaryEmbeddingImpl).
THETA = 10000.0
MAX_SEQ_LEN = 2048
ROTARY_DIM = 32


def run_rotary(lib, qkv, sin, cos, rotary_dim):
    """Rotate q (chunk 0) and k (chunk 1) of qkv in place, passing the *full qkv*
    strides exactly as TxModel.cpp does. v (chunk 2) is untouched.
    qkv: [N, seq_len, 3, n_heads, head_dim] fp16 CUDA contiguous; sin/cos [max_seq, head_dim/2]."""
    N, seq_len, _, n_heads, head_dim = qkv.shape
    sb, ss, sh = qkv.stride(0), qkv.stride(1), qkv.stride(3)
    chunk_stride = qkv.stride(2)                  # = n_heads*head_dim when contiguous
    elem = qkv.element_size()
    base = qkv.data_ptr()
    for chunk in (0, 1):                          # q, k
        lib.fluke_rotary_emb_gpu(
            base + chunk * chunk_stride * elem, sin.data_ptr(), cos.data_ptr(),
            N, seq_len, n_heads, head_dim, rotary_dim, sb, ss, sh)


def bench(lib, qkv, sin, cos, rotary_dim, warmup, iters):
    for _ in range(warmup):
        run_rotary(lib, qkv, sin, cos, rotary_dim)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_rotary(lib, qkv, sin, cos, rotary_dim)
    stop.record(); torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def build_cos_sin(head_dim, device):
    inv_freq = torch.pow(THETA, torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim).reciprocal()
    freqs = torch.outer(torch.arange(MAX_SEQ_LEN, dtype=torch.float64), inv_freq)
    cos = torch.cos(freqs).to(torch.float32).contiguous().to(device)
    sin = torch.sin(freqs).to(torch.float32).contiguous().to(device)
    return cos, sin


def reference(qkv, cos, sin, rotary_dim):
    """Rotate-half RoPE on q and k chunks; fp32 math, fp16 output (matches kernel)."""
    out = qkv.clone()
    seq_len = qkv.size(1)
    c = cos[:seq_len, :rotary_dim].view(1, seq_len, 1, rotary_dim)
    s = sin[:seq_len, :rotary_dim].view(1, seq_len, 1, rotary_dim)
    for chunk in (0, 1):
        x = out[:, :, chunk, :, :]
        x0 = x[..., :rotary_dim].to(torch.float32)
        x1 = x[..., rotary_dim:2 * rotary_dim].to(torch.float32)
        x[..., :rotary_dim] = (x0 * c - x1 * s).to(torch.float16)
        x[..., rotary_dim:2 * rotary_dim] = (x0 * s + x1 * c).to(torch.float16)
    return out


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")

    lib = fluke_lib.load()
    dev = "cuda"
    head_dim = 2 * ROTARY_DIM  # = 64, matching the model
    cos, sin = build_cos_sin(head_dim, dev)

    # ---- correctness ----
    torch.manual_seed(0)
    batch, seq_len, n_heads = 4, 128, 8
    qkv = torch.randn(batch, seq_len, 3, n_heads, head_dim, device=dev).to(torch.float16).contiguous()
    expected = reference(qkv, cos, sin, ROTARY_DIM)

    got = qkv.clone()
    run_rotary(lib, got, sin, cos, ROTARY_DIM)
    torch.cuda.synchronize()

    # v chunk must be untouched.
    assert torch.equal(got[:, :, 2], qkv[:, :, 2]), "v chunk was modified"

    max_diff = (got.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    # fp16 rounding: a single fp16 ulp near magnitude ~4 is ~4e-3, and GPU
    # FMA/rounding order can differ from torch by ~1 ulp, so allow a small margin.
    tol = 5e-3
    print(f"kernel vs reference max abs diff: {max_diff:.3e} (tol {tol:.1e})")
    ok = max_diff < tol
    print("PASS" if ok else "FAIL")

    # ---- timing ----
    print("\n>> timing (mean over 100 iters, after 20 warmup)")
    print(f"{'N':>5} {'seq':>6} {'heads':>6} {'ms/call':>10} {'GB/s':>8}")
    for batch, seq_len, n_heads in [(4, 128, 8), (64, 512, 8), (256, 1666, 8), (512, 2048, 8)]:
        qkv = torch.randn(batch, seq_len, 3, n_heads, head_dim, device=dev).to(torch.float16).contiguous()
        ms = bench(lib, qkv, sin, cos, ROTARY_DIM, 20, 100)
        # 2 chunks * (read+write) * 2*rotary_dim halves(2B) + cos/sin reads (2*rotary_dim*4B)
        elems = batch * seq_len * n_heads
        bytes_moved = 2 * elems * (2 * (2 * ROTARY_DIM) * 2 + (2 * ROTARY_DIM) * 4)
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(f"{batch:>5} {seq_len:>6} {n_heads:>6} {ms:>10.4f} {gbps:>8.1f}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
