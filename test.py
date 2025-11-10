import torch
import triton
import triton.language as tl
import os
import numpy as np

# -------------------------------
# Triton kernel: GEMM
# -------------------------------
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 当前 block 的起始索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算行列索引
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化累加
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load A block
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # Load B block
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # Block matmul
        acc += tl.dot(a, b)

    # Store result
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_gemm(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K, N = B.shape

    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C

M, K, N = 1024, 4096, 2048
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)

C_triton = triton_gemm(A, B)
C_torch = A @ B

print("Max error:", (C_triton - C_torch).abs().max())

# === 保存输出 ===
save_dir = "./tensor_dump"
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "x.npy"), C_triton.detach().cpu().numpy())
np.save(os.path.join(save_dir, "ref_x.npy"), C_torch.detach().cpu().numpy())
print(f"Tensors saved to {save_dir}")
