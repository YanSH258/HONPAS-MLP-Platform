import os
import shutil
import time
import glob
import config as cfg  # 导入配置文件

# 导入各个功能模块
from modules.sampler import StructureSampler
from modules.wrapper import InputWrapper
from modules.scheduler import TaskScheduler
from modules.extractor import ResultExtractor
from modules.cleaner import DataQualityControl
from modules.analyzer import SOAPSpaceAnalyzer

# ==============================================================================
# Stage 1: 生成与提交
# ==============================================================================
def run_stage_1_generate(mode="scf", dry_run=True):
    print(f"\n=== Stage 1: 生产与提交 (Mode: {mode.upper()}) ===")
    
    # 1. 检查配置
    if mode not in cfg.TEMPLATE_MAP:
        raise ValueError(f"Unknown mode: {mode}")
    template_path = cfg.TEMPLATE_MAP[mode]

    # 2. 采样 (Sampler 内部已集成 Validator 预筛选)
    print(f"[Workflow] 读取结构: {cfg.INPUT_PATH}")
    sampler = StructureSampler(cfg.INPUT_PATH)

    # --- 优化点：智能处理扩胞逻辑 ---
    target_size = cfg.SUPERCELL_SIZE
    
    # 检查是否真的需要扩胞 (只要有一个维度 > 1)
    if any(s > 1 for s in target_size):
        print(f"[Workflow] 检测到扩胞配置 {target_size}，正在执行扩胞...")
        sampler.apply_supercell(target_size)
    else:
        print(f"[Workflow] 扩胞配置为 {target_size}，保持输入结构不变。")
        print(f"           当前原子数: {sampler.current_system.get_natoms()}")

    # 执行微扰
    systems = sampler.generate_perturbations(
        num_perturbed=cfg.NUM_TASKS, 
        pert_config=cfg.PERTURB_CONFIG
    )
    
    # 3. 备份数据
    backup_dir = os.path.join("data/perturbed", f"batch_{mode}")
    if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
    systems.to("deepmd/npy", backup_dir)
    print(f"[Workflow] 数据已备份至 {backup_dir}")

    # 4. 生成任务
    current_workspace = f"{cfg.WORKSPACE_PREFIX}_{mode}"
    wrapper = InputWrapper(template_path, cfg.SPECIES_MAP)
    scheduler = TaskScheduler(current_workspace, submit_cmd="sbatch")
    
    print(f"[Workflow] 构建任务至 {current_workspace}...")
    for i in range(len(systems)):
        # 这里的 psf_dir 指向 templates/psfs
        scheduler.setup_task(i, wrapper, systems[i].data, psf_dir=cfg.TEMPLATE_DIR)
        
    # 5. 提交
    print(f"[Workflow] 提交作业 (DryRun={dry_run})...")
    scheduler.submit_all(dry_run=dry_run)


# ==============================================================================
# Stage 2: 收集与清洗
# ==============================================================================
def run_stage_2_collect(mode="scf"):
    print(f"\n=== Stage 2: 收集与清洗 (Mode: {mode.upper()}) ===")
    current_workspace = f"{cfg.WORKSPACE_PREFIX}_{mode}"
    
    if not os.path.exists(current_workspace):
        print(f"❌ 目录不存在: {current_workspace}")
        return

    # 1. 提取
    extractor = ResultExtractor(current_workspace)
    raw_data = extractor.collect_data()
    
    if not raw_data:
        print("❌ 无有效数据。")
        return

    # 2. 清洗 (DataQualityControl)
    qc = DataQualityControl(raw_data)
    
    # 物理合理性检查 (使用共价半径倍数阈值)
    qc.check_atom_overlap(threshold_factor=cfg.QC_OVERLAP_THRESHOLD)  
    
    # 统计离群值检查
    qc.check_outliers(sigma_n=cfg.QC_SIGMA_E, max_force_tol=cfg.QC_MAX_FORCE)
    
    # 3. 导出
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"data/training/set_{mode}_{timestamp}"
    
    final_data = qc.generate_report(output_path)
    
    if final_data:
        final_data.to("deepmd/npy", output_path)
        final_data.to("deepmd/raw", output_path)
        print(f"✅ 最终数据集: {output_path} (帧数: {len(final_data)})")


# ==============================================================================
# Stage 3: 可视化分析
# ==============================================================================
def run_stage_3_analysis(mode="scf"):
    print(f"\n=== Stage 3: 可视化分析 (Mode: {mode.upper()}) ===")
    
    # 1. 自动查找 data/training 下最新的数据集
    search_pattern = f"data/training/set_{mode}_*"
    found_dirs = sorted(glob.glob(search_pattern))
    
    if not found_dirs:
        print(f"❌ 错误: 在 data/training/ 下未找到 mode={mode} 的数据集。")
        print("   请先运行 Stage 2 (收集与清洗) 生成数据。")
        return

    # 取最新的一个文件夹
    data_path = found_dirs[-1]
    print(f"[Workflow] 锁定数据集: {data_path}")
    
    # 2. 准备输出目录
    output_dir = os.path.join("data/analysis", f"report_{mode}")
    
    # 3. 调用分析器 (强制读取 npy 格式)
    try:
        analyzer = SOAPSpaceAnalyzer(data_path, output_dir)
        analyzer.compute_and_plot(cfg.VIS_CONFIG, cfg.SPECIES_MAP)
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()