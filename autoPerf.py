#!/usr/bin/env python3
import subprocess
import re
import os
import statistics
import csv

# === 配置参数 ===
SIZES = []

# === 不同 K 值 sweep ===
for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 9216]:
    for K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 9216]:
        for N in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 9216]:
            if M*K*N < 1024*1024:
                SIZES.append((M, K, N))

print(f"Total test cases: {len(SIZES)}")

REPEAT = 10
TEMP_DIR = "temp"
CSV_FILE = os.path.join(TEMP_DIR, "gemm_benchmark.csv")

os.makedirs(TEMP_DIR, exist_ok=True)

# 正则匹配 kernel time
RE_GEMM = re.compile(r'kernel-name:"[^"]*gemm[^"]*".*?kernel time\s+([\d.]+)\(s\)', re.S | re.I)
RE_CIJK = re.compile(r'kernel-name:"Cijk_Ailk_Bljk[^"]*".*?kernel time\s+([\d.]+)\(s\)', re.S)

def run_benchmark(m, k, n):
    """运行一次 hipprof 并解析结果"""
    print(f"==> Running M={m}, K={k}, N={n}")
    cmd = ["hipprof", "--pmc", "python", "benchGemm.py", str(m), str(k), str(n)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 找到 pmc_results 文件
    pmc_files = [f for f in os.listdir(".") if f.startswith("pmc_results_") and f.endswith(".txt")]
    if not pmc_files:
        print("No pmc_results file found.")
        return None, None

    pmc_file = max(pmc_files, key=os.path.getmtime)
    target_path = os.path.join(TEMP_DIR, f"M{m}_K{k}_N{n}_{len(os.listdir(TEMP_DIR))}.txt")
    os.rename(pmc_file, target_path)

    with open(target_path, "r") as f:
        text = f.read()

    # 提取两个kernel的时间
    cijk_times = [float(x) for x in RE_CIJK.findall(text)]
    gemm_times = [float(x) for x in RE_GEMM.findall(text)]

    cijk_time = min(cijk_times) if cijk_times else None
    gemm_time = min(gemm_times) if gemm_times else None

    return cijk_time, gemm_time


def filter_and_average(values):
    """去掉最大最小后取平均"""
    if len(values) <= 2:
        return statistics.mean(values)
    vals = sorted(values)[1:-1]
    return statistics.mean(vals)


# === 主流程 ===
results = []

for (m, k, n) in SIZES:
    cijk_all = []
    gemm_all = []

    for i in range(REPEAT):
        cijk, gemm = run_benchmark(m, k, n)
        if cijk is not None:
            cijk_all.append(cijk)
        if gemm is not None:
            gemm_all.append(gemm)

    debug_file = os.path.join(TEMP_DIR, f"M{m}_K{k}_N{n}_debug.txt")
    with open(debug_file, "w") as f:
        f.write(f"Cijk_Ailk_Bljk times: {cijk_all}\n")
        f.write(f"gemm times: {gemm_all}\n")

    if cijk_all and gemm_all:
        cijk_avg = filter_and_average(cijk_all)
        gemm_avg = filter_and_average(gemm_all)
        faster = "Cijk" if cijk_avg < gemm_avg else "gemm"
        results.append([m, k, n, cijk_avg, gemm_avg, faster])
        print(f"[{m},{k},{n}]  Cijk:{cijk_avg:.6f}s  gemm:{gemm_avg:.6f}s  => {faster} faster")
    else:
        results.append([m, k, n, "N/A", "N/A", "N/A"])


# === 写出 CSV 文件 ===
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["M", "K", "N", "Cijk_time_avg(s)", "gemm_time_avg(s)", "Faster"])
    writer.writerows(results)

print(f"\n✅ All done. Results saved to {CSV_FILE}")
