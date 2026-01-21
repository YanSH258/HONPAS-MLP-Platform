# HONPAS-MLP-Platform

本项目是一个基于 Python 构建的自动化工作流系统，旨在连接国产材料模拟软件 **HONPAS** 与机器学习势能（DeepMD-kit / GPUMD）训练框架。

平台实现了从**初始结构微扰采样**、**高通量 DFT 任务调度**、**数据清洗**、**相空间分析**到**模型训练文件构建**的全流程自动化，规范了数据集构建过程。

## 1. 核心功能

*   **多模式任务生成**：支持单点能 (SCF)、结构优化 (Relax) 和从头算分子动力学 (AIMD) 任务的批量生成与调度。
*   **物理预筛选**：在生成阶段基于共价半径自动剔除原子重叠结构，减少无效计算。
*   **双格式数据流**：数据提取阶段同时生成 DeepMD (`.npy`) 和 GPUMD (`.xyz`) 格式，并自动修正 HONPAS 维里 (Virial) 符号差异。
*   **质量控制 (QC)**：基于 Z-score 和最大受力阈值自动清洗脏数据。
*   **训练准备自动化**：支持 DeepMD 和 GPUMD 的训练集/验证集自动划分、物理拆分及配置文件生成。

## 2. 环境依赖

本项目运行于 Python 3 环境。建议使用 Conda 创建独立环境。

### 2.1 Python 库依赖
请确保安装以下核心库：

```bash
# 基础数据处理
pip install numpy ase dpdata

# 可视化与分析 (Stage 3 需要)
pip install matplotlib seaborn dscribe scikit-learn
```

### 2.2 外部软件依赖
*   **HONPAS**: 需在 HPC 集群上配置好环境变量。
*   **DeepMD-kit**: 用于模型训练（支持 GPU）。
*   **GPUMD (NEP)**: 用于 NEP 模型训练（支持 GPU）。
*   **Slurm**: 用于作业调度。

---

## 3. 目录结构

```text
HAP_project_v2/
├── config.py              # 全局配置 (微扰参数、QC阈值、HONPAS模板路径)
├── config_train.py        # 训练配置 (DeepMD/GPUMD 超参数模板)
├── main.py                # 主程序入口 (CLI)
├── modules/               # 功能模块库
│   ├── sampler.py         # 结构微扰与超胞处理
│   ├── validator.py       # 几何合理性校验
│   ├── wrapper.py         # HONPAS 输入文件生成
│   ├── scheduler.py       # 任务目录管理与 Slurm 提交
│   ├── extractor.py       # 结果提取 (log解析)
│   ├── cleaner.py         # 数据清洗 (QC)
│   ├── converter.py       # 格式转换 (DeepMD -> GPUMD XYZ)
│   ├── merger.py          # 数据集合并
│   ├── analyzer.py        # SOAP-PCA 可视化分析
│   ├── trainer.py         # 训练文件生成与数据拆分
│   └── workflows.py       # 阶段流程封装
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
*注：`config.py` 中可配置超胞尺寸 `SUPERCELL_SIZE`。*

### Stage 2: 收集与清洗 (Collect & Clean)
等待任务完成后，提取结果并进行质量控制。

```bash
python main.py --mode relax --stage 2
```
*   **输出**：`data/training/set_relax_时间戳/`
*   **包含**：DeepMD 格式 (`set.000/*.npy`) 和 GPUMD 格式 (`train.xyz`)。

### Stage 3: 可视化分析 (Analyze)
计算数据集的全局 SOAP 描述符并进行 PCA 降维，评估相空间覆盖度。

```bash
python main.py --mode relax --stage 3
```
*   **输出**：`data/analysis/report_relax/soap_pca.png` (能量-描述符分布图)。

### Stage 4: 数据集合并 (Merge)
将分散的多个数据集（如不同温度的 AIMD、Relax 数据）合并为一个主数据集。

```bash
python main.py --stage 4
```
*   程序会自动扫描 `data/training/` 下的所有数据集。
*   **输出**：`data/training/merged_master_时间戳/` (包含合并后的 npy 和 xyz)。

### Stage 5: 训练准备 (Train Preparation)
生成模型训练所需的配置文件和数据目录。

#### 模式 A: 准备 DeepMD 训练
```bash
# 指定合并后的数据集，按 8:2 拆分
python main.py --stage 5 --model_type deepmd --data_path data/training/merged_master_xxx --val_ratio 0.2
```
*   **执行动作**：
    1.  在当前目录创建 `train_work_deepmd_xxx`。
    2.  将数据**物理拆分**为 `data_train` 和 `data_val` 子文件夹。
    3.  生成 `input.json`，使用相对路径指向上述文件夹。
*   **后续操作**：进入目录执行 `dp train input.json`。

#### 模式 B: 准备 GPUMD (NEP) 训练
```bash
python main.py --stage 5 --model_type gpumd --data_path data/training/merged_master_xxx
```
*   **执行动作**：
    1.  在当前目录创建 `train_work_gpumd_xxx`。
    2.  将数据随机拆分并写入 `train.xyz` 和 `test.xyz`。
    3.  生成 `nep.in` 配置文件。
*   **后续操作**：进入目录执行 `nep` 。

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


### 6. 模块功能详解

平台的业务逻辑高度模块化，各组件职责如下表所示：

| 模块名称 | 核心职责说明 |
| :--- | :--- |
| **sampler.py** | **结构生成器**：基于 `dpdata` 实现基态结构的超胞扩充与随机微扰 。 |
| **validator.py** | **物理预筛选**：基于共价半径检测原子重叠，在提交计算前拦截不合理的构型，节省算力。 |
| **wrapper.py** | **模板引擎**：将结构数据注入 HONPAS 模板 (.fdf)，自动映射元素并关联对应的 PSF 赝势文件。 |
| **scheduler.py** | **任务调度**：自动化管理任务目录、分发赝势、生成 Slurm 脚本 (`run.sh`) 并批量提交作业。 |
| **extractor.py** | **结果采集**：支持 SCF/Relax/AIMD 模式，解析 `output.log` 提取能量、力、维里及坐标信息。 |
| **cleaner.py** | **质量控制 (QC)**：执行后处理清洗，利用 Z-score 和最大受力阈值自动剔除离群数据 (Outliers)。 |
| **converter.py** | **格式转换**：实现 DeepMD (`.npy`) 与 GPUMD (`.xyz`) 格式互转，自动修正 HONPAS 维里符号差异。 |
| **merger.py** | **数据聚合**：支持将多个分散的数据集（如不同温度/批次）无损合并为统一的主数据集。 |
| **analyzer.py** | **特征分析**：计算全局 SOAP 描述符并执行 PCA 降维，可视化评估数据集对势能面的覆盖度。 |
| **trainer.py** | **训练准备**：自动划分训练集/验证集，生成 DeepMD (`input.json`) 或 GPUMD (`nep.in`) 配置文件。 |
| **workflows.py** | **流程编排**：作为顶层逻辑控制器，串联各独立模块，定义 Stage 1~5 的标准化操作流程。 |

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
   - Stage 3 的 UMAP 分析依赖 `umap-learn` 库。若运行报错，请检查是否已执行 `pip install umap-learn`。
   - 对于样本量超过 2000 帧的数据集，降维计算可能需要 1-3 分钟。

5. **物理预筛选拦截**：
   - 若生成的任务数少于 `NUM_TASKS` 设置值，通常是因为 `validator` 拦截了过多原子重叠构型。请尝试减小微扰幅度或微调 `QC_OVERLAP_THRESHOLD`。
6.   **维里符号修正**：
   - HONPAS 输出的 Virial 与 GPUMD 定义符号相反。本平台在导出 `train.xyz` 时已自动乘以 `-1.0` 进行修正，DeepMD 格式保持原样。