**HAP-MLP-Platform** 是一款专为机器学习势函数（MLP）训练定制的自动化数据生产与分析平台。

本平台以 **HONPAS**  作为底层高精度密度泛函理论（DFT）计算引擎。通过深度集成 HONPAS 的线性缩放计算特性，平台实现了从大规模构型采样到高性能数据集生成的全流程自动化，显著提升了训练数据的生产效率。

---

## 🌟 核心特色 (Core Features)

*   **HONPAS 原生集成**：针对 HONPAS 的输入输出格式进行深度适配，支持单点能（SCF）、结构优化（Relax）以及分子动力学（AIMD）轨迹的自动解析与数据提取。
*   **智能结构采样**：支持基于 `dpdata` 的扩胞 (Supercell) 与微扰 (Perturbation) 逻辑，自动适配 HONPAS 赝势 (PSF) 文件。
*   **物理准则质量控制**：内置基于元素共价半径的碰撞检测逻辑，确保输入 HONPAS 计算的构型具有物理合理性。
*   **构型空间可视化**：集成 SOAP 描述符与 UMAP 非线性降维算法，直观展示 HONPAS 生成数据的构型空间覆盖率。
*   **一键式工作流**：通过 Stage 1~4 的标准化指令，完成从“POSCAR 结构”到“DeepMD 训练集”的完整转换。

---

## 快速上手

项目通过 `main.py` 进行阶段化管理，主要分为四个阶段（Stage）。

### 0. 配置参数
在运行前，修改项目根目录下的 `config.py`：
*   `INPUT_PATH`: 初始结构文件（如 POSCAR）。
*   `SUPERCELL_SIZE`: 扩胞尺寸，如 `[2, 2, 2]`。
*   `NUM_TASKS`: 扰动生成的任务数量。
*   `QC_OVERLAP_THRESHOLD`: 原子重叠阈值（共价半径之和的倍数，建议 0.5）。

### Stage 1: 任务生成与提交
生成微扰结构并提交至超算集群（支持 Slurm）。

```bash
# 生成 SCF 采样任务 (默认 DryRun，仅生成文件不提交)
python main.py --stage 1 --mode scf

# 生成任务并直接提交到 Slurm 队列
python main.py --stage 1 --mode scf --submit

# 同样支持 aimd 模式的初始结构准备
python main.py --stage 1 --mode aimd --submit
```

### Stage 2: 数据收集与清洗
计算完成后，提取 `output.log` 等结果，进行质量控制并转换为 DeepMD 格式。

```bash
# 收集 SCF 单点能数据 (提取每一文件夹的最后一帧)
python main.py --stage 2 --mode scf

# 收集 AIMD 轨迹数据 (从 .ANI/.FA 文件提取完整轨迹)
python main.py --stage 2 --mode aimd
```
清洗后的数据将存放在 `data/training/set_{mode}_{timestamp}`。

### Stage 3: 构型空间可视化分析
利用 SOAP 描述符和 UMAP 算法分析数据集的构型覆盖范围。

```bash
# 分析最新的 SCF 数据集
python main.py --stage 3 --mode scf

# 分析指定的任意数据集路径
python main.py --stage 3 --path data/training/merged_master_20260120
```
图表保存于 `data/analysis/` 目录下。

### Stage 4: 数据集大合并
将多个 batch 的数据（如不同温度的 AIMD 和不同幅度的 SCF）合并为一个最终训练集。

```bash
# 自动搜索 data/training/ 下所有 set_* 文件夹并合并
python main.py --stage 4
```

---

### 📂 目录结构说明

```text
├── config.py                # 全局核心配置文件
├── main.py                  # 工作流总控程序入口
├── data/
│   ├── raw/                 # 存放原始输入结构 (如 POSCAR)
│   ├── training/            # 存放清洗后的 DeepMD 训练数据集
│   └── analysis/            # 存放 UMAP/PCA 分析报告及图表
├── templates/               # 计算软件输入模板 (HONPAS/SIESTA/PSF等)
└── modules/                 # 平台核心逻辑功能模块
```

### ⚙️ 模块功能详解

平台的业务逻辑高度模块化，各组件职责如下表所示：

| 模块名称 | 核心职责说明 |
| :--- | :--- |
| **sampler.py** | **结构处理中心**：负责原子结构的扩胞 (Supercell) 和几何微扰 (Perturbation)。 |
| **validator.py** | **物理预筛选**：在提交计算前拦截原子重叠构型，减少无效计算成本。 |
| **wrapper.py** | **HONPAS 模板引擎**：将采样结构自动填充至模板并关联相应的 PSF 赝势文件。 |
| **scheduler.py** | **任务调度管家**：管理 Slurm 脚本生成、`sbatch` 提交及 HONPAS 作业状态监控。 |
| **extractor.py** | **结果自动采集**：支持 SCF 与 AIMD 模式，解析 `output.log` 及轨迹文件提取能量/力。 |
| **cleaner.py** | **质量控制 (QC)**：基于共价半径检查原子碰撞，自动剔除能量与力的统计离群帧。 |
| **analyzer.py** | **特征分析**：利用 SOAP + UMAP/t-SNE 算法评估数据集对势能面的覆盖程度。 |
| **merger.py** | **数据集聚合**：支持将不同采样模式下的 HONPAS 计算结果无损合并为统一训练集。 |
| **visualizer.py** | **绘图组件**：提供能量/力分布直方图、降维空间散点图等可视化支持。 |
| **workflows.py** | **工作流编排**：将各独立模块按业务逻辑串联，定义 Stage 1~4 的标准化操作。 |

---

### ⚠️ 注意事项 (Notes)

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