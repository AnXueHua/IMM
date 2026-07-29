# IMM-LLM: Inverted Mamba, Memory & Large Language Models for Time-Series

**IMM-LLM** 是一种针对工业环境中超长多元时间序列分析设计的全新多模态融合架构。它将先进的 `Mamba` 序列模型与大型语言模型 (LLM) 进行深度融合，同时支持**长时序预测 (Long-term Forecasting)** 与**异常诊断 (Anomaly Diagnosis)** 任务。

本架构在 ETTh2 数据集上取得了 **Test MSE = 0.1870** 的突破性成绩，显著优于 ICLR 24 的 TimeLLM (约 0.27) 的 SOTA 水平。

## 🎯 核心架构创新

与传统的 `TimeLLM` (采用 Patching 机制) 不同，本架构保留了传感器/变量间的空间拓扑约束，并通过双路模态协同解决了大模型预测中的表征坍塌问题。核心设计包括：

1. **倒置嵌入层 (Inverted Embedding)**：
   将传统的多变量时序张量 `[Batch, Time, Vars]` 转置并线性投影，实现“变量即 Token”。大模型看到的不再是切碎的时间块，而是每一个完整的传感器趋势实体，完美保留了时间拓扑的零信息丢失。
2. **混合基座编码器 (Hybrid Backbone)**：
   利用受 Jamba 启发的 Mamba2 和 Attention 的混合结构，以 $O(V)$ 的复杂度沿着变量维度进行扫描，提取并融合传感器之间的拓扑耦合关系。
3. **记忆增强的语义 Sidecar (Memory-Augmented Semantic Sidecar)**：
   引入神经记忆矩阵 (Neural Memory Matrix, NMM) 在线提取时序片段的异常惊奇度 (Surprise Score)。LLM (如 Qwen/Qwen3.5) 读取 NMM、retrieval evidence 与层级 time-state 的语义摘要，输出仅用于记录、诊断和后续动态控制的 sidecar 状态；当前 v2 不再把 LLM hidden state 作为残差偏置直接加回数值预测主链。
4. **可逆实例归一化 (RevIN)**：
   直接在网络入口处进行实例级归一化，消除分布漂移，大幅提升了对非平稳时间序列的预测稳定性。

## 🚀 快速上手与环境配置

本项目基于 `PyTorch` 和 `Transformers` 构建，时序特征提取强依赖于 `mamba-ssm` 引擎。

```bash
# 1. 创建虚拟环境
conda create -n imamba python=3.10 -y
conda activate imamba

# 2. 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 HuggingFace 核心库
pip install transformers accelerate peft

# 4. 安装 Mamba 依赖
pip install mamba-ssm[causal-conv1d] --no-build-isolation
```

> **注意**：Windows 环境下编译 `mamba-ssm` 和 `causal-conv1d` 需要 C++ 工具链和对应的 `nvcc` 环境。
### Windows 源码编译指南 (`causal-conv1d` & `mamba-ssm`)

由于原版代码对 Windows (MSVC) 环境的支持不够完善，本项目已在本地 `causal-conv1d` 和 `mamba` 文件夹中对源码进行了以下适配性修改。如果你需要手动编译，请参考以下说明：

使用x64 Native Tools Command Prompt for VS 2022，进入源代码路径，激活虚拟环境

```
set DISTUTILS_USE_SDK=1
```

#### 1. `causal-conv1d` 的修改与编译
- **语法适配 (MSVC)**：在 `csrc/causal_conv1d.cpp` 中，将 `TORCH_CHECK` 宏里的逻辑运算符 `and` 替换为 `&&`，解决了 MSVC 默认不识别 `and` 导致的编译报错。
如：
```bash
TORCH_CHECK(x.stride(2) % 8 == 0 and x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");

TORCH_CHECK(x.stride(2) % 8 == 0 and x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");

TORCH_CHECK(dout.stride(2) % 8 == 0 and dout.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (dout.stride(0) and dout.stride(2)) to be multiples of 8");
```
改为：
```bash
TORCH_CHECK(x.stride(2) % 8 == 0 && x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");

TORCH_CHECK(x.stride(2) % 8 == 0 && x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");

TORCH_CHECK(dout.stride(2) % 8 == 0 && dout.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (dout.stride(0) and dout.stride(2)) to be multiples of 8");
```
- **移除 ROCM 干扰**：清理了 `.cu` 源码中针对 AMD ROCM 平台的冗余条件编译代码，强制走纯净的 CUDA 流程。
如：
```bash
#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_reduce.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif
```
改为：
```bash
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
```
- **算力架构锁定**：在 `setup.py` 中，注释了多个冗余算力的编译选项，仅保留了 `compute_89,code=sm_89` (对应 RTX 40 系显卡)。这避免了不支持架构的报错，并大幅缩短了编译时间。

**编译与安装命令**：
```bash
cd causal-conv1d
pip install . --no-build-isolation --no-cache-dir -v
```

#### 2. `mamba` 的修改与编译
- **宏定义与 Lambda 表达式修复**：在 `csrc/selective_scan/` 下的 CUDA 核函数头文件中手动补充了 `M_LOG2E` 的宏定义；在 `static_switch.h` 及核函数中，将 `constexpr` 修改为 `static constexpr`，修复了 MSVC 编译器在 Lambda 表达式中捕获外部 constexpr 变量的报错。在 csrc/selective_scan/selective_scan_bwd_kernel.cuh 和 csrc/selective_scan/selective_scan_fwd_kernel.cuh 文件开头加入：
```bash
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif
```
- **依赖替换**：在 `pyproject.toml` 中，将官方仅支持 Linux 的 `triton` 库依赖替换为了第三方移植版本 `triton-windows`。
- **算力架构锁定**：同样在 `setup.py` 中锁定了 `compute_89` (RTX 40 系) 算力，屏蔽了其他兼容性不佳的架构配置。
- **移除 ROCM 干扰**：清理了 `.cu` 源码中针对 AMD ROCM 平台的冗余条件编译代码，强制走纯净的 CUDA 流程。

**编译与安装命令**：
```bash
cd mamba
pip install . --no-build-isolation --no-cache-dir -v
```

> **💡 算力配置提示**：本项目的源码目前硬编码适配了 RTX 40 系列显卡 (`sm_89`)。如果你的显卡是其他型号，请在两个文件夹的 `setup.py` 中，将 `cc_flag.append("arch=compute_89,code=sm_89")` 修改为匹配你显卡架构的值（例如：RTX 30 系列修改为 `sm_86`，RTX 20 系列修改为 `sm_75`）。

## 📁 项目解耦与文件结构

本项目遵循松耦合、高扩展性的模块化设计原则：

```text
run/
  ├── conf/                  # Hydra / OmegaConf 配置入口
  │   └── config.yaml        # 单次与批量实验参数
  ├── pipeline/              # 薄入口与运行期 I/O
  │   ├── run_experiments.py # 统一 single / batch CLI
  │   └── runtime_io.py      # 日志、曲线图、stdout tee、异常输出
  └── outputs/               # Hydra 管理运行的输出目录（预留）
exp/
  ├── exp_config.py          # config -> Namespace -> ExpConfig
  ├── exp_basic.py           # seed、device、Accelerate 公共逻辑
  ├── exp_forecasting.py     # 单次 forecasting 训练/测试
  └── exp_batch.py           # batch profile 调度
src/
  ├── data_module/
  │   └── dataset/           # 数据集加载器 (TSDataset)
  ├── model_module/
  │   ├── layers/            # 核心组件库
  │   ├── models/            # 核心模型架构
  │   │   ├── imamba_memory.py # 时序编码器
  │   │   └── imm_llm.py       # IMM-LLM 语义桥接包装器
  │   └── utils/             # 模型辅助工具
  └── trainer_module/
      └── forecasting_trainer.py # 包含 CosineAnnealingLR 和 EarlyStopping 的训练循环
data/
  └── dataset/ETT/           # 放置公开基准数据集 (ETTh2.csv 等)
```

## 🧪 运行测试与模型训练

项目已完美集成 `Qwen3.5-0.8B` 作为语义特征调制器，并使用 `Jamba` 作为底层时序特征提取器。

**执行端到端训练评估：**

```bash
conda activate imamba
export PYTHONPATH=$PWD

# WSL / Linux 环境下运行单次预测实验
# 具体实验参数统一在 run/conf/config.yaml 中修改
python run/pipeline/run_experiments.py --mode single

# 批量实验入口
python run/pipeline/run_experiments.py --mode batch_070
```

Backbone 版本在 `run/conf/config.yaml` 中通过 `ssm.mamba_version` 控制，可选 `mamba`、`mamba2`、`mamba3`。默认保留 `mamba2`，以兼容已有实验结果；若要使用 Mamba3，把该字段改为 `mamba3` 即可。`ssm` 里同时保留 Mamba v1、Mamba2、Mamba3 的官方参数面，运行时只会向当前版本传递对应参数。Mamba3 的 MIMO 由 `mamba3_is_mimo` 与 `mamba3_mimo_rank` 控制。数据加载 CPU worker 数量由 `runtime.num_workers` 控制。

预期输出将展示模型在 ETTh2 数据集上的训练收敛过程，并在最终的测试集上输出极低的 MSE/MAE 指标。

