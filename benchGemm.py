#!/usr/bin/env python3
import torch
import lightop
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python benchGemm.py M K N")
        sys.exit(1)

    # 读取矩阵大小
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机生成矩阵
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)

    # PyTorch GEMM
    C_torch = torch.mm(A, B)

    # LightOp 浮点 GEMM
    C_lightop = lightop.gemm(A, B.T.contiguous())

if __name__ == "__main__":
    main()
