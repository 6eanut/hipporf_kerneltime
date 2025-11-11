#!/usr/bin/env python3
import subprocess
import re
import csv
import os

# === Configuration ===

# Square matrix test configuration
SQUARE_START = 256
SQUARE_END = 4096
SQUARE_STEP = 128

# Non-square matrix test configuration [(M, K, N), ...]
NON_SQUARE_SIZES = [
    (1, 7168, 9216),
    (1, 7168, 1536),
    (1, 7168, 1024),
    (1, 5120, 6144),
    (1, 5120, 3584),
    (1, 5120, 3072),
    (1, 5120, 768),
    (1, 5120, 256),
    (1, 4608, 7168),
    (1, 4096, 7168),
    (1, 3072, 5120),
    (1, 1536, 6144),
    (1, 512, 7168),
    (1, 384, 5120)
]

KERNEL_LIST_FILE = "kernel_list.txt"
FINAL_CSV = "gemm_sweep_results.csv"

# === Utility functions ===
def extract_average_from_csv(file_path):
    """Extract the average kernel time from getKernelTime.py CSV output"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Average,"):
            parts = line.strip().split(",")
            return [float(x) if x not in ("N/A", "") else None for x in parts[1:]]
    return None

# === Main function ===
def main():
    square_sizes = list(range(SQUARE_START, SQUARE_END + 1, SQUARE_STEP))

    # Open final CSV
    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "K", "N", "AverageKernelTime(s)"])

        # 1. Square matrix tests
        for size in square_sizes:
            print(f"\n===== Running square GEMM {size}x{size}x{size} =====")
            cmd = f"python getKernelTime.py 'python benchGemm.py {size} {size} {size}' {KERNEL_LIST_FILE}"
            subprocess.run(cmd, shell=True, check=True)

            csv_files = [f for f in os.listdir(".") if f.startswith("hipprof_results_") and f.endswith(".csv")]
            latest_csv = max(csv_files, key=os.path.getmtime)

            averages = extract_average_from_csv(latest_csv)
            avg_time = averages[0] if averages else None
            writer.writerow([size, size, size, avg_time])
            print(f"[OK] {size}x{size}x{size}: average kernel time = {avg_time:.6f}s")

        # 2. Non-square matrix tests
        for M, K, N in NON_SQUARE_SIZES:
            print(f"\n===== Running non-square GEMM {M}x{K} * {K}x{N} =====")
            cmd = f"python getKernelTime.py 'python benchGemm.py {M} {K} {N}' {KERNEL_LIST_FILE}"
            subprocess.run(cmd, shell=True, check=True)

            csv_files = [f for f in os.listdir(".") if f.startswith("hipprof_results_") and f.endswith(".csv")]
            latest_csv = max(csv_files, key=os.path.getmtime)

            averages = extract_average_from_csv(latest_csv)
            avg_time = averages[0] if averages else None
            writer.writerow([M, K, N, avg_time])
            print(f"[OK] {M}x{K}x{N}: average kernel time = {avg_time:.6f}s")

    print(f"\nAll GEMM test results saved to: {FINAL_CSV}")

if __name__ == "__main__":
    main()
