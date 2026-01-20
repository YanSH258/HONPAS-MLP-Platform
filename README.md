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
| **validator.py** | **物理预筛选**：在任务提交前进行拦截，确保没有严重的原子重叠构型。 |
| **wrapper.py** | **模板引擎**：将采样后的结构数据自动填充进计算软件（如 HONPAS）的输入模板。 |
| **scheduler.py** | **任务调度管家**：管理 Slurm 任务脚本生成、`sbatch` 提交及作业状态监控。 |
| **extractor.py** | **结果自动采集**：支持 SCF 与 AIMD 模式，从 output 及轨迹文件中解析能量与力。 |
| **cleaner.py** | **质量控制 (QC)**：基于共价半径检查原子碰撞，自动剔除能量与力的统计离群帧。 |
| **analyzer.py** | **高维特征分析**：计算 SOAP 化学描述符，并执行 PCA 或非线性 UMAP 降维。 |
| **merger.py** | **数据聚合工具**：无损合并不同 batch 或不同模式（SCF/AIMD）生成的训练集。 |
| **visualizer.py** | **绘图组件**：提供能量/力分布直方图、降维空间散点图等可视化支持。 |
| **workflows.py** | **工作流编排**：将各独立模块按业务逻辑串联，定义 Stage 1~4 的标准化操作。 |
```
---

## 注意事项

1.  **扩胞逻辑**：在 Stage 1 执行时，程序会检查 `SUPERCELL_SIZE`。若任意维度 > 1，则会自动执行扩胞后再进行扰动。
2.  **AIMD 提取**：使用 `--mode aimd` 时，请确保任务目录下存在轨迹文件（如 `.ANI` 或 `.FA`），否则将无法提取受力信息。
3.  **原子类型顺序**：合并数据集时，`dpdata` 会严格检查原子种类顺序。请确保所有 batch 使用相同的 `SPECIES_MAP` 配置。

---
