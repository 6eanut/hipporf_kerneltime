#!/usr/bin/env python3
import torch
import lightop
import os

# === 配置参数 ===
SIZES = []

# === 不同 K 值 sweep ===
for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 9216]:
    for K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 9216]:
        for N in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 9216]:
            if M*K*N < 1024*1024:
                SIZES.append((M, K, N))

print(f"Total test cases: {len(SIZES)}")


# === 结果输出目录 ===
os.makedirs("temp", exist_ok=True)
log_file = os.path.join("temp", "verify_results.txt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_one_case(M, K, N, atol=1e-3, rtol=1e-3):
    """测试一个矩阵规模，返回是否通过"""
    try:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)

        C_torch = torch.mm(A, B)
        C_lightop = lightop.gemm(A, B.T.contiguous())

        ok = torch.allclose(C_torch, C_lightop, atol=atol, rtol=rtol)
        return ok, None if ok else (C_torch, C_lightop)

    except Exception as e:
        return False, str(e)


def main():
    passed = []
    failed = []

    with open(log_file, "w") as f:
        for (M, K, N) in SIZES:
            ok, info = test_one_case(M, K, N)
            if ok:
                print(f"✅ PASS: M={M}, K={K}, N={N}")
                f.write(f"PASS: M={M}, K={K}, N={N}\n")
                passed.append((M, K, N))
            else:
                print(f"❌ FAIL: M={M}, K={K}, N={N}")
                f.write(f"FAIL: M={M}, K={K}, N={N}\n")
                if isinstance(info, str):
                    f.write(f"  Exception: {info}\n")
                failed.append((M, K, N))

    print("\n=== 测试完成 ===")
    print(f"✅ 通过 {len(passed)} 个 / ❌ 失败 {len(failed)} 个")
    print(f"详细结果已写入 {log_file}")

    # 汇总结论
    if passed:
        max_m = max([M for (M, K, N) in passed])
        max_k = max([K for (M, K, N) in passed])
        max_n = max([N for (M, K, N) in passed])
        print(f"\n🟢 LightOp.gemm 正确的最大已验证规模：M ≤ {max_m}, K ≤ {max_k}, N ≤ {max_n}")
    else:
        print("\n🔴 所有测试均失败。")


if __name__ == "__main__":
    main()
