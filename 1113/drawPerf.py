#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === CSV 文件路径 ===
CSV_FILE = "test.csv"

# === 读取数据 ===
df = pd.read_csv(CSV_FILE)

# 去掉无效数据
df = df[df["Cijk_time_avg(s)"] != "N/A"]
df = df[df["gemm_time_avg(s)"] != "N/A"]
df["Cijk_time_avg(s)"] = df["Cijk_time_avg(s)"].astype(float)
df["gemm_time_avg(s)"] = df["gemm_time_avg(s)"].astype(float)

# 计算速度比
df["speedup"] = df["Cijk_time_avg(s)"] / df["gemm_time_avg(s)"]

# 限制极端值（防止某些异常点颜色失真）
df["speedup_clamped"] = df["speedup"].clip(0.5, 2.0)

# 获取所有 M 值
m_values = sorted(df["M"].unique())
num_m = len(m_values)

# 创建子图
fig, axes = plt.subplots(nrows=num_m, figsize=(14, 5 * num_m))

if num_m == 1:
    axes = [axes]

for ax, m in zip(axes, m_values):
    df_m = df[df["M"] == m]
    pivot = df_m.pivot(index="N", columns="K", values="speedup_clamped")
    
    # 绘制热力图
    sns.heatmap(
        pivot,
        cmap="RdYlGn",     # 红->黄->绿
        center=1.0,        # 1.0 代表两者性能相等
        annot=False,
        cbar=True,
        linewidths=0.4,
        linecolor="gray",
        ax=ax
    )

    ax.set_title(f"GEMM vs Cijk Performance Ratio (M={m})", fontsize=14)
    ax.set_xlabel("K")
    ax.set_ylabel("N")

plt.tight_layout()

# 保存和显示
output_file = "gemm_perf_heatmap.png"
plt.savefig(output_file, dpi=300)
print(f"✅ Heatmap saved as {output_file}")
plt.show()
