#!/usr/bin/env python3
import subprocess
import re
import csv
import os
import time
import shutil
import sys
from datetime import datetime

def read_kernel_list(file_path):
    """Read kernel names or patterns from a file (one per line)."""
    with open(file_path, "r") as f:
        kernels = [line.strip() for line in f if line.strip()]
    return kernels

def run_hipprof(run_cmd, kernel_list_file, repeat=10):
    kernel_names = read_kernel_list(kernel_list_file)
    if not kernel_names:
        print("Error: kernel list file is empty or not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"hipprof_results_{timestamp}.csv"
    os.makedirs("temp", exist_ok=True)
    all_results = {k: [] for k in kernel_names}

    for i in range(1, repeat + 1):
        print(f"\n[Run {i}/{repeat}] Running: hipprof --pmc {run_cmd}")
        cmd = f"hipprof --pmc {run_cmd}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output, _ = process.communicate()

        # Extract process id
        match = re.search(r"HIP_PROF:process id '(\d+)'", output)
        if not match:
            print("Error: process id not found, skipping this run.")
            print(output)
            continue
        pid = match.group(1)
        pmc_file = f"pmc_results_{pid}.txt"

        # Wait for pmc file to appear
        for _ in range(30):
            if os.path.exists(pmc_file):
                break
            time.sleep(0.5)
        else:
            print(f"Error: {pmc_file} not generated, skipping.")
            continue

        # Parse pmc file
        with open(pmc_file, "r") as f:
            text = f.read()

        for kernel_pattern in kernel_names:
            # Support wildcard "*" → regex ".*"
            regex_pattern = re.escape(kernel_pattern).replace(r'\*', '.*')
            pattern = rf'kernel-name:"{regex_pattern}".*?kernel time\s+([\d.]+)\(s\)'
            matches = re.findall(pattern, text, re.S)

            if len(matches) == 1:
                kernel_time = float(matches[0])
                all_results[kernel_pattern].append(kernel_time)
                print(f"[OK] {kernel_pattern} run {i}: {kernel_time:.6f}s")
            elif len(matches) > 1:
                print(f"[WARN] Multiple matches for {kernel_pattern}, using first one.")
                kernel_time = float(matches[0])
                all_results[kernel_pattern].append(kernel_time)
            else:
                print(f"[WARN] kernel {kernel_pattern} not found in pmc file.")
                all_results[kernel_pattern].append(None)

        # Move pmc file to temp/
        dst_file = os.path.join("temp", f"pmc_results_{pid}.txt")
        shutil.move(pmc_file, dst_file)
        time.sleep(1)

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Run"] + kernel_names
        writer.writerow(header)

        # Per-run results
        for i in range(repeat):
            row = [i + 1]
            for k in kernel_names:
                val = all_results[k][i] if i < len(all_results[k]) else None
                row.append(val if val is not None else "N/A")
            writer.writerow(row)

        # Average (excluding max and min)
        writer.writerow([])
        writer.writerow(["Average (no max/min)"])
        avg_row = ["Average"]
        for k in kernel_names:
            valid = [v for v in all_results[k] if v is not None]
            if len(valid) >= 3:
                valid.sort()
                trimmed = valid[1:-1]
                avg = sum(trimmed) / len(trimmed)
                avg_row.append(avg)
                print(f"[AVG] {k}: average (trimmed) = {avg:.6f}s")
            else:
                avg_row.append("N/A")
        writer.writerow(avg_row)

    print(f"\nAll results saved to: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python getKernelTime.py '<run_cmd>' <kernel_list.txt>")
        print("Example: python getKernelTime.py 'python test.py' kernel_list.txt")
        sys.exit(1)

    run_cmd = sys.argv[1]
    kernel_list_file = sys.argv[2]
    run_hipprof(run_cmd, kernel_list_file)
