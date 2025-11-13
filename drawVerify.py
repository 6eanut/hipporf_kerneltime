#!/usr/bin/env python3
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 日志文件路径
log_file = "/workspace/temp/verify_results.txt"

# 解析日志
data = []
pattern = re.compile(r"(PASS|FAIL): M=(\d+), K=(\d+), N=(\d+)")
with open(log_file, "r") as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            status, M, K, N = match.groups()
            data.append({
                "M": int(M),
                "K": int(K),
                "N": int(N),
                "Result": status
            })

# 转为 DataFrame
df = pd.DataFrame(data)

# 获取所有唯一的 M 值
m_values = sorted(df['M'].unique())
num_m = len(m_values)

# 创建子图，按 M 值分行
fig, axes = plt.subplots(nrows=num_m, figsize=(14, 6 * num_m))

# 如果只有一个 M，axes 不是列表，需要特殊处理
if num_m == 1:
    axes = [axes]

for ax, m_value in zip(axes, m_values):
    df_m = df[df['M'] == m_value]
    pivot = df_m.pivot(index='N', columns='K', values='Result')
    pivot_numeric = pivot.replace({'PASS': 1, 'FAIL': 0})
    
    sns.heatmap(
        pivot_numeric, 
        annot=pivot, 
        fmt='', 
        cmap='RdYlGn', 
        cbar=False, 
        linewidths=.5, 
        linecolor='gray',
        ax=ax
    )
    ax.set_title(f"GEMM Test Results Heatmap (M={m_value})")
    ax.set_xlabel("K")
    ax.set_ylabel("N")

plt.tight_layout()

# 保存图片
output_file = "gemm_verify_heatmap.png"
plt.savefig(output_file, dpi=300)
print(f"Heatmap saved as {output_file}")

# 显示图片
plt.show()

