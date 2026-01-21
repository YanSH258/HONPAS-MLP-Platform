import os
import shutil
import time
import glob
import config as cfg  # 导入配置文件

# 导入各个功能模块
# --- Generation Modules ---
from modules.generation.sampler import StructureSampler
from modules.generation.wrapper import InputWrapper
from modules.generation.scheduler import TaskScheduler

# --- Data Modules ---
from modules.data.extractor import ResultExtractor
from modules.data.cleaner import DataQualityControl
from modules.data.merger import DatasetMerger
from modules.data.converter import NEPConverter

# --- Analysis Modules ---
from modules.analysis.analyzer import SOAPSpaceAnalyzer

# --- Training Modules ---
from modules.training.trainer import ModelTrainer
from modules.training.monitor import TrainingMonitor
from modules.training.evaluator import ModelEvaluator

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
    raw_data = extractor.collect_data(mode=mode)
    
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
        # 1. 保存 DeepMD 格式 (npy)
        final_data.to("deepmd/npy", output_path)
        final_data.to("deepmd/raw", output_path)
        print(f"✅ DeepMD 数据集已保存: {output_path} (帧数: {len(final_data)})")
        
        # 2. 保存 GPUMD 格式 (train.xyz)
        xyz_filename = os.path.join(output_path, "train.xyz")
        try:
            NEPConverter.save_as_xyz(final_data, xyz_filename)
            print(f"✅ GPUMD 数据集已生成: {xyz_filename}")
        except Exception as e:
            print(f"⚠️ GPUMD 格式转换失败: {e}")
            import traceback
            traceback.print_exc()

# ==============================================================================
# Stage 3: 可视化分析
# ==============================================================================
def run_stage_3_analysis(mode="scf", custom_path=None):
    print(f"\n=== Stage 3: 可视化分析 (Mode: {mode.upper()}) ===")
    
    if custom_path:
        # 如果手动指定了路径，直接使用它
        data_path = custom_path
    else:
        # 否则，自动查找 data/training 下该 mode 最新的数据集
        search_pattern = f"data/training/set_{mode}_*"
        found_dirs = sorted(glob.glob(search_pattern))
        if not found_dirs: 
            print(f"❌ 错误: 未找到 mode={mode} 的数据集。请通过 --path 指定。")
            return
        data_path = found_dirs[-1]
    
    print(f"[Workflow] 锁定数据集: {data_path}")

    
    # 2. 准备输出目录 (如果是合并数据，存入 report_merged)
    if "merged" in data_path:
        output_dir = os.path.join("data/analysis", "report_merged")
    else:
        output_dir = os.path.join("data/analysis", f"report_{mode}")
    
    # 3. 调用分析器
    try:
        analyzer = SOAPSpaceAnalyzer(data_path, output_dir)
        analyzer.compute_and_plot(cfg.VIS_CONFIG, cfg.SPECIES_MAP)
    except Exception as e:
        print(f"❌ 分析出错: {e}")

# ==============================================================================
# Stage 4: 数据集大合并
# ==============================================================================
def run_stage_4_merge():
    print(f"\n=== Stage 4: 数据集大合并 ===")
    
    # 1. 自动搜索 data/training/ 下所有的有效数据集目录
    # 你也可以手动指定具体的目录列表
    search_pattern = "data/training/set_*"
    all_sets = sorted(glob.glob(search_pattern))
    
    if not all_sets:
        print("❌ 未在 data/training/ 下找到任何待合并的数据集。")
        return

    print(f"[Workflow] 发现以下数据集:")
    for s in all_sets:
        print(f"   - {s}")
    
    confirm = input("\n是否确认合并以上所有数据集? (y/n): ")
    if confirm.lower() != 'y':
        print("操作取消。")
        return

    # 2. 执行合并
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/training/merged_master_{timestamp}"
    
    merger = DatasetMerger()
    merger.merge_all(all_sets, output_dir)

# ==============================================================================
# Stage 5: 训练输入文件准备
# ==============================================================================
def run_stage_5_train(model_type="deepmd", data_path=None, val_ratio=0.2):
    print(f"\n=== Stage 5: 训练准备 (Model: {model_type.upper()}) ===")
    print(f"   目标: 生成配置文件与数据集链接")
    print(f"   数据源: {data_path if data_path else '自动扫描 data/training/set_*'}")
    print(f"   验证集比例: {val_ratio}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = f"train_work_{model_type}_{timestamp}"
    
    trainer = ModelTrainer(work_dir)
    
    try:
        if model_type == "deepmd":
            trainer.prepare_deepmd(source_path=data_path, val_ratio=val_ratio)
        elif model_type == "gpumd":
            trainer.prepare_gpumd(source_path=data_path, val_ratio=val_ratio)
        else:
            print(f"❌ 未知模型: {model_type}")
            return
            
        print(f"\n✅ 准备工作完成！")
        print(f"   请进入目录: {work_dir}")
        if model_type == "deepmd":
            print("   执行命令: dp train input.json")
        else:
            print("   执行命令: nep") 
            
    except Exception as e:
        print(f"❌ 准备失败: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# Stage 6: 训练监控 (Monitor) 
# ==============================================================================
def run_stage_6_monitor(model_type="deepmd", work_dir=None):
    print(f"\n=== Stage 6: 训练监控 (Model: {model_type.upper()}) ===")
    
    if not work_dir:
        import glob
        pattern = f"train_work_{model_type}_*"
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"❌ 未找到训练目录。")
            return
        work_dir = found[-1]
        
    print(f"[Workflow] 监控目标: {work_dir}")
    monitor = TrainingMonitor(work_dir)
    
    if model_type == "deepmd":
        monitor.plot_deepmd_lcurve()
    elif model_type == "gpumd":
        monitor.plot_gpumd_loss()

# ==============================================================================
# Stage 7: 模型评估 (Evaluate) 
# ==============================================================================
def run_stage_7_eval(model_type="deepmd", work_dir=None):
    print(f"\n=== Stage 7: 模型评估 (Model: {model_type.upper()}) ===")
    
    if not work_dir:
        import glob
        pattern = f"train_work_{model_type}_*"
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"❌ 未找到训练目录 {pattern}。请通过 --path 指定。")
            return
        work_dir = found[-1]
    
    print(f"[Workflow] 评估目标: {work_dir}")
    evaluator = ModelEvaluator(work_dir)
    
    if model_type == "deepmd":
        evaluator.eval_deepmd()
    elif model_type == "gpumd":
        evaluator.eval_gpumd()