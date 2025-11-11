# HippoProf Kernel Time Measurement for GEMM

这个仓库包含用于测量GEMM（General Matrix Multiplication，通用矩阵乘法）内核运行时间的工具集，基于HippoProf性能分析工具。通过自动化脚本，可以对不同尺寸的矩阵乘法进行性能测试，并收集内核运行时间数据。

## 目录结构

- `benchGemm.py`: 矩阵乘法基准测试脚本，用于生成随机矩阵并执行乘法运算
- `getKernelTime.py`: 利用HippoProf收集指定内核的运行时间
- `kernel_list.txt`: 包含需要监测的内核名称模式（支持通配符 `*`）
- `run.py`: 自动化测试脚本，用于批量执行不同尺寸的矩阵乘法测试

## 功能说明

1. **基准测试**：`benchGemm.py`接收矩阵维度参数(M, K, N)，生成随机矩阵并使用PyTorch执行矩阵乘法
2. **内核时间收集**：`getKernelTime.py`使用HippoProf工具监测指定内核的运行时间，并将结果保存到CSV文件
3. **批量测试**：`run.py`自动化执行一系列预定义尺寸的矩阵乘法测试，包括正方形矩阵和非正方形矩阵，并汇总结果

## 配置参数

在 `run.py`中可以配置以下测试参数：

- 正方形矩阵测试：

  - `SQUARE_START`: 起始尺寸
  - `SQUARE_END`: 结束尺寸
  - `SQUARE_STEP`: 步长
- 非正方形矩阵测试：

  - `NON_SQUARE_SIZES`: 自定义的(M, K, N)三元组列表

## 使用方法

### 前提条件

- Python 3.x
- PyTorch（支持CUDA或CPU）
- HippoProf性能分析工具

### 基本使用

1. 单个矩阵乘法测试（例如32x32x32）：

   ```bash
   python benchGemm.py 32 32 32
   ```
2. 测量特定内核的运行时间：

   ```bash
   python getKernelTime.py 'python benchGemm.py 32 32 32' kernel_list.txt
   ```
3. 执行批量测试：

   ```bash
   python run.py
   ```

## 结果输出

- 每次 `getKernelTime.py`运行会生成一个时间戳命名的CSV文件（如 `hipprof_results_20231001_123456.csv`）
- 批量测试的汇总结果会保存到 `gemm_sweep_results.csv`
- 结果包含每次运行的时间数据和修剪后的平均值（去除最大值和最小值）

## 内核匹配

`kernel_list.txt`文件中定义需要监测的内核名称模式，支持通配符 `*`进行模糊匹配。默认配置监测与 `Cijk_Ailk_Bljk*`模式匹配的内核。

## 注意事项

- 确保HippoProf工具正确安装并能正常运行
- 对于CUDA设备，需要确保PyTorch正确配置了CUDA支持
- 测试结果会受到系统负载等因素影响，建议在稳定的环境中运行
- 临时文件会保存在 `temp`目录中，可在测试完成后手动清理
