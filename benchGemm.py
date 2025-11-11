#!/usr/bin/env python3
import torch
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python benchGemm.py M K N")
        sys.exit(1)

    # Read parameters
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])

    # Create random matrices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    # Perform matrix multiplication
    C = torch.matmul(A, B)

if __name__ == "__main__":
    main()
