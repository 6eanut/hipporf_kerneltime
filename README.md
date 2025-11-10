
# hipporf_kerneltime

一个用于测量 Triton 内核执行时间的工具集，基于 `hipprof` 性能分析工具，可自动化运行内核测试并生成时间统计结果。

## 目录结构

```
hipporf_kerneltime/
├── kernel_list.txt       # 待测量的内核名称列表
├── getKernelTime.py      # 核心脚本，用于运行测试并收集时间数据
├── test.py               # 示例 Triton 内核实现（GEMM）
└── temp/                 # 临时存储 pmc 结果文件（自动生成）
```

## 功能说明

- 自动读取内核名称列表，循环执行指定命令测量内核时间
- 支持多次重复测试，计算去除最大最小值后的平均时间
- 生成 CSV 格式的结果文件，包含每次运行的时间和统计平均值
- 示例代码包含一个 Triton 实现的 GEMM（通用矩阵乘法）内核

## 依赖环境

- Python
- hipprof（性能分析工具）

## 使用方法

1. **准备内核列表**在 `kernel_list.txt` 中添加需要测量的内核名称（每行一个），例如：

   ```
   gemm_kernel
   ```
2. **运行测量脚本**使用以下命令执行测试，其中 `<run_cmd>` 是启动内核的命令，`<kernel_list.txt>` 是内核列表文件路径：

   ```bash
   python getKernelTime.py '<run_cmd>' kernel_list.txt
   ```

   示例（运行自带的 GEMM 内核测试）：

   ```bash
   python getKernelTime.py 'python test.py' kernel_list.txt
   ```
3. **查看结果**测试完成后，会生成名为 `hipprof_results_<timestamp>.csv` 的结果文件，包含：

   - 每次运行的具体时间
   - 去除最大最小值后的平均时间（trimmed average）

## 示例代码说明

`test.py` 包含一个使用 Triton 实现的 GEMM 内核，主要功能：

- 定义 `gemm_kernel` 函数实现矩阵乘法
- 提供 `triton_gemm` 接口函数，设置网格和块大小
- 对比 Triton 实现与 PyTorch 内置矩阵乘法的结果（验证正确性）
- 将计算结果保存到 `./tensor_dump` 目录

## 结果解释

CSV 文件中：

- 第一列是运行次数（Run）
- 后续列是对应内核的每次运行时间（单位：秒）
- 末尾包含去除最大最小值后的平均值，用于减少极端值对结果的影响

## 注意事项

- 确保 `hipprof` 工具已正确安装并配置
- 测试过程中会自动创建 `temp` 目录存储临时文件
- 若内核未被正确检测到，结果会显示 "N/A"，请检查内核名称是否与列表一致
- 建议至少运行 3 次以上以获得更稳定的平均结果
