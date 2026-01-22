# HONPAS-MLP Automation Platform

本项目是一个基于 Python 构建的自动化工作流系统，旨在连接国产材料模拟软件 **HONPAS** 与机器学习势能（DeepMD-kit / GPUMD）训练框架。

平台实现了从**初始结构微扰采样**、**高通量 DFT 任务调度**、**数据清洗**、**相空间分析**、**模型训练**到**精度评估**的全流程自动化，规范了数据集构建过程。

## 1. 核心功能

*   **多模式任务生成**：支持单点能 (SCF)、结构优化 (Relax) 和从头算分子动力学 (AIMD) 任务的批量生成与调度。
*   **物理预筛选**：在生成阶段基于共价半径自动剔除原子重叠结构，减少无效计算。
*   **双格式数据流**：数据提取阶段同时生成 DeepMD (`.npy`) 和 GPUMD (`.xyz`) 格式，并自动修正 HONPAS 维里 (Virial) 符号差异。
*   **质量控制 (QC)**：基于 Z-score 和最大受力阈值自动清洗脏数据。
*   **训练准备自动化**：支持 DeepMD 和 GPUMD 的训练集/验证集自动划分、物理拆分及配置文件生成。
*   **模型全生命周期管理**：提供实时的训练 Loss 监控与最终模型的冻结、压缩及精度评估（Parity Plot）以及gpumd主动学习策略。

## 2. 环境依赖

本项目运行于 Python 3 环境。建议使用 Conda 创建独立环境。

### 2.1 Python 库依赖
请确保安装以下核心库：

```bash
# 基础数据处理
pip install numpy ase dpdata

# 可视化与分析 (Stage 3, 6, 7 需要)
pip install matplotlib seaborn dscribe scikit-learn
```

### 2.2 外部软件依赖
*   **HONPAS**: 需在 HPC 集群上配置好环境变量。
*   **DeepMD-kit**: 用于模型训练（支持 GPU）。
*   **GPUMD (NEP)**: 用于 NEP 模型训练（支持 GPU）。
*   **Slurm**: 用于作业调度。

---

## 3. 目录结构

代码经过模块化重构，逻辑更加清晰：

```text
HAP_project_v2/
├── config.py              # 全局配置 (微扰参数、QC阈值、HONPAS模板路径)
├── config_train.py        # 训练配置 (DeepMD/GPUMD 超参数模板)
├── config_active.py       # 主动学习配置 (MD探索参数)
├── main.py                # 主程序入口 (CLI)
├── modules/               # 核心功能库
│   ├── workflows.py       # [总控] 阶段流程封装
│   ├── generation/        # [生成端]
│   │   ├── sampler.py     # 结构微扰与超胞处理
│   │   ├── wrapper.py     # HONPAS 输入文件生成
│   │   └── scheduler.py   # 任务目录管理与提交
│   ├── data/              # [数据端]
│   │   ├── extractor.py   # 结果提取
│   │   ├── cleaner.py     # 数据清洗 (QC)
│   │   ├── merger.py      # 数据集合并
│   │   ├── converter.py   # 格式转换
│   │   └── validator.py   # 几何校验工具
│   ├── analysis/          # [分析端]
│   │   ├── analyzer.py    # SOAP-PCA 核心分析
│   │   └── visualizer.py  # 基础绘图工具
│   ├── training/          # [训练端]
│   │   ├── trainer.py     # 训练准备与配置生成
│   │   ├── monitor.py     # 训练监控
│   │   └── evaluator.py   # 模型评估
│   └── active_learning/   # 主动学习迭代模块 (Ensemble & Explore)
│       ├── trainer.py     # 构建多模型系综训练环境。
│       ├── explorer.py    # 自动化配置带主动学习判据的 MD 任务
│       └── selector.py    # 从 active.xyz 中筛选用于标注的代表性结构
├── templates/             # 输入模板 (.in) 及 赝势库 (psfs/)
└── data/                  # 数据流转目录
    ├── perturbed/         # 微扰结构备份
    ├── training/          # 清洗后的训练集
    └── analysis/          # 分析报告
```

---

## 4. 使用流程 (Workflow)

通过 `main.py` 的 `--stage` 参数控制执行阶段。

### Stage 1: 生成与提交 (Generate & Submit)
读取基态结构，生成微扰超胞，创建任务目录并提交。

```bash
# 1. 结构优化任务 (Relax) - Dry Run (仅生成文件)
python main.py --mode relax --stage 1

# 2. 正式提交到 Slurm
python main.py --mode relax --stage 1 --submit

# 3. 分子动力学任务 (AIMD)
python main.py --mode aimd --stage 1 --submit
```

### Stage 2: 收集与清洗 (Collect & Clean)
等待任务完成后，提取结果并进行质量控制。

```bash
python main.py --mode relax --stage 2
```
*   **输出**：`data/training/set_relax_时间戳/` (包含 npy 和 xyz)。

### Stage 3: 可视化分析 (Analyze)
计算数据集的全局 SOAP 描述符并进行 PCA 降维，评估相空间覆盖度。

```bash
python main.py --mode relax --stage 3
```

### Stage 4: 数据集合并 (Merge)
将分散的多个数据集（如不同温度的 AIMD、Relax 数据）合并为一个主数据集。

```bash
python main.py --stage 4
```

### Stage 5: 训练准备 (Train Preparation)
生成模型训练所需的配置文件和数据目录。

```bash
# 准备 DeepMD 训练 (自动拆分训练/验证集)
python main.py --stage 5 --model_type deepmd --data_path data/training/merged_master_xxx --val_ratio 0.2

# 准备 GPUMD 训练
python main.py --stage 5 --model_type gpumd --data_path data/training/merged_master_xxx
```

### Stage 6: 训练监控 (Monitor)
在模型训练过程中，实时查看 Loss 收敛曲线。

```bash
python main.py --stage 6 --model_type deepmd
```

### Stage 7: 模型评估 (Evaluate)
训练完成后，对模型进行冻结、压缩，并评估其精度（RMSE）。

```bash
# DeepMD 评估 (自动调用 dp test 并绘图)
python main.py --stage 7 --model_type deepmd
```

### Stage 8: GPUMD 主动学习迭代
通过多模型系综 在线评估不确定度，自动捕捉势能面缺失区域。

#### 8.1 准备系综训练
生成 $N$ 个（默认 4 个）独立的 NEP 训练目录。
```bash
python main.py --stage 8 --sub 1 --data_path data/training/merged_master_xxx
```
*   **输出**：`al_gpumd_train_时间戳/`。手动在各子目录运行 `nep` 得到 4 个 `nep.txt`。

#### 8.2 准备 MD 探索
链接已训练模型，生成带有 `active` 关键字的 GPUMD `run.in`。
```bash
# --path 指向上步生成的训练根目录，--data_path 指向 MD 初始结构
python main.py --stage 8 --sub 2 --path al_gpumd_train_xxx --data_path data/raw/model.xyz
```
*   **特性**：自动适配 `npt_mttk` 或 `nvt_ber` 语法，支持自定义 `dump_xyz` 属性。

#### 8.3 候选结构提取
解析探索产生的 `active.xyz`，筛选高偏差构型送往 HONPAS 进行标注。
```bash
# --path 指向上步生成的训练根目录，--path 指向active.xyz
python main.py --stage 8 --sub 3 --path al_gpumd_explore_xxx 
---


---

## 5. 关键参数配置 (`config.py`)

| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `SUPERCELL_SIZE` | 扩胞倍数，`[1,1,1]` 为不扩胞 | `[2, 2, 2]` |
| `PERTURB_CONFIG` | 微扰幅度 (晶格形变率, 原子位移) | `0.03`, `0.1` |
| `QC_SIGMA_E` | 能量清洗阈值 (标准差倍数) | `4.0` |
| `QC_MAX_FORCE` | 最大受力容忍值 (eV/Å) | `100.0` |
| `VIS_CONFIG` | SOAP 分析参数 (rcut, nmax, lmax) | `6.0`, `8`, `6` |

---

## 6. 模块功能详解

平台的业务逻辑按功能域进行了划分：

###  生成端 (`modules/generation`)
| 模块 | 职责 |
| :--- | :--- |
| **sampler.py** | **结构生成器**：基于 `dpdata` 实现基态结构的超胞扩充与随机微扰。 |
| **wrapper.py** | **模板引擎**：将结构数据注入 HONPAS 模板，自动映射元素。 |
| **scheduler.py** | **任务调度**：管理任务目录、分发赝势、生成 `run.sh` 并提交作业。 |

###  数据端 (`modules/data`)
| 模块 | 职责 |
| :--- | :--- |
| **extractor.py** | **结果采集**：解析 `output.log` 提取能量、力、维里及坐标。 |
| **cleaner.py** | **质量控制**：执行后处理清洗，剔除能量与受力的离群数据。 |
| **validator.py** | **物理预筛选**：基于共价半径检测原子重叠，拦截不合理构型。 |
| **converter.py** | **格式转换**：实现 DeepMD/GPUMD 格式互转，修正维里符号。 |
| **merger.py** | **数据聚合**：支持将多个分散的数据集无损合并。 |

###  分析端 (`modules/analysis`)
| 模块 | 职责 |
| :--- | :--- |
| **analyzer.py** | **特征分析**：计算 SOAP 描述符并执行 PCA 降维。 |
| **visualizer.py** | **绘图组件**：提供基础统计图表支持。 |

###  训练端 (`modules/training`)
| 模块 | 职责 |
| :--- | :--- |
| **trainer.py** | **训练准备**：数据拆分、生成配置文件 (`input.json`/`nep.in`)。 |
| **monitor.py** | **训练监控**：解析日志并绘制 Loss 曲线。 |
| **evaluator.py** | **模型评估**：模型冻结压缩、测试推理及 Parity Plot 绘制。 |

###  核心控制
| 模块 | 职责 |
| :--- | :--- |
| **workflows.py** | **流程编排**：串联各子模块，定义 Stage 1~7 的标准化逻辑。 |

---

### 7. 注意事项 (Notes)

1. **扩胞与计算量控制**：
   - 在 Stage 1 执行时，程序会检测 `config.py` 中的 `SUPERCELL_SIZE`。
   - **提醒**：扩胞会指数级增加原子数，请务必根据 **HONPAS** 的内存限制合理设置。建议先用小体系测试单步 SCF 耗时。

2. **AIMD 数据提取前提**：
   - 使用 `--mode aimd` 模式时，`extractor` 模块依赖 `dpdata` 的 `siesta/aimd_output` 格式。
   - **必须确保**：每个任务目录下除了 `output.log`，还须包含同名的 **`.ANI`** (坐标轨迹) 和 **`.FA`** (受力轨迹) 文件，否则将无法解析有效帧。

3. **原子类型顺序一致性 (Species Consistency)**：
   - 在执行 Stage 4 合并不同 batch 或模式（如 SCF 与 AIMD 合并）时，`dpdata` 会严格校验原子类型的顺序。
   - **要求**：请确保所有计算任务使用完全一致的 `SPECIES_MAP` 定义。如果原子顺序不匹配，合并将失败或导致势函数物理意义错误。

4. **可视化依赖**：
   - Stage 3 的 PCA 分析依赖 `scikit-learn` 和 `dscribe` 库。对于样本量超过 2000 帧的数据集，降维计算可能需要 1-3 分钟。

5. **物理预筛选拦截**：
   - 若生成的任务数少于 `NUM_TASKS` 设置值，通常是因为 `validator` 拦截了过多原子重叠构型。请尝试减小微扰幅度或微调 `QC_OVERLAP_THRESHOLD`。

6. **维里符号修正**：
   - HONPAS 输出的 Virial 与 GPUMD 定义符号相反。本平台在导出 `train.xyz` 时已自动乘以 `-1.0` 进行修正，DeepMD 格式保持原样。