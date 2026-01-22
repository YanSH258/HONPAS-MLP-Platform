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

# --- Active_learning Moudules
from modules.active_learning.trainer import ALTrainer
from modules.active_learning.explorer import ALExplorer
from modules.active_learning.selector import ActiveSelector

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

    # 1. 提取 (返回字典: {atom_counts: system})
    extractor = ResultExtractor(current_workspace)
    grouped_data = extractor.collect_data(mode=mode)
    
    if not grouped_data:
        print("❌ 无有效数据。")
        return

    # 准备 GPUMD 的总输出文件 (放在 data/training/gpumd_merged_xyz 目录下，或者跟第一个set放一起)
    # 为了方便管理，我们把总的 train.xyz 放在一个独立的汇总目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    global_xyz_dir = f"data/training/gpumd_xyz_{mode}_{timestamp}"
    if not os.path.exists(global_xyz_dir): os.makedirs(global_xyz_dir)
    global_xyz_path = os.path.join(global_xyz_dir, "train.xyz")
    
    # 清空/新建 train.xyz
    open(global_xyz_path, 'w').close() 
    print(f"[Workflow] GPUMD 汇总文件路径: {global_xyz_path}")

    # 2. 遍历每组不同原子数的数据
    for atom_counts, raw_sys in grouped_data.items():
        n_atoms = sum(atom_counts)
        print(f"\n>> 处理分组: 原子数 {n_atoms}, 帧数 {len(raw_sys)}")
        
        # --- 清洗 ---
        qc = DataQualityControl(raw_sys)
        qc.check_atom_overlap(threshold_factor=cfg.QC_OVERLAP_THRESHOLD)
        qc.check_outliers(sigma_n=cfg.QC_SIGMA_E, max_force_tol=cfg.QC_MAX_FORCE)
        
        # 定义 DeepMD 输出目录 (加上原子数后缀以区分)
        # 例如: set_aimd_2026..._N352
        output_path = f"data/training/set_{mode}_{timestamp}_N{n_atoms}"
        
        # 生成报告并获取清洗后的数据
        final_data = qc.generate_report(output_path)
        
        if final_data and len(final_data) > 0:
            # (A) 保存 DeepMD 格式 (必须分开存)
            final_data.to("deepmd/npy", output_path)
            final_data.to("deepmd/raw", output_path)
            print(f"   ✅ DeepMD set 保存至: {output_path}")
            
            # (B) 追加到 GPUMD train.xyz (合并存)
            try:
                NEPConverter.save_as_xyz(final_data, global_xyz_path, mode='a')
            except Exception as e:
                print(f"   ⚠️ 追加 train.xyz 失败: {e}")
        else:
            print(f"   ⚠️ 分组 N={n_atoms} 清洗后无剩余数据，跳过。")

    print(f"\n✅ Stage 2 完成！")
    print(f"   - DeepMD 数据: 分散在 data/training/set_{mode}_{timestamp}_N*")
    print(f"   - GPUMD 数据: 汇总在 {global_xyz_path}")
    
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

# ==============================================================================
# Stage 8: 主动学习 (active) 
# ==============================================================================
def run_stage_8_al_gpumd(sub_stage, data_path=None, work_dir=None,  dry_run=True):
    print(f"\n=== Stage 8: GPUMD 主动学习 (Sub-Stage: 8.{sub_stage}) ===")

    # 8.1 准备系综训练目录
    if sub_stage == 1:
        if not data_path:
            print("❌ 错误: 请使用 --data_path 指定合并后的训练集目录 (含 train.xyz)")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        train_dir = f"al_gpumd_train_{timestamp}"
        
        trainer = ALTrainer(train_dir)
        trainer.prepare_ensemble(data_path)
        print(f"\n✅ 系综训练目录已生成: {train_dir}")
        print(f"   下一步: 在 model_xx 目录下运行 nep 训练出 nep.txt。")

    # 8.2 准备探索任务 (MD)
    elif sub_stage == 2:
        if not work_dir:
            print("❌ 错误: 请使用 --path 指定 Stage 8.1 生成的训练根目录")
            return
        if not data_path:
            print("❌ 错误: 请使用 --data_path 指定 MD 的初始结构文件 (如 model.xyz)")
            return

        # 查找包含 nep.txt 的子目录
        model_dirs = sorted(glob.glob(os.path.join(work_dir, "model_*")))
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        explore_dir = f"al_gpumd_explore_{timestamp}"
        
        explorer = ALExplorer(explore_dir)
        explorer.prepare_exploration(model_dirs, data_path)
        print(f"\n✅ 探索目录已生成: {explore_dir}")

    # ------------------------------------------------------------------
    # 8.3 收集 active.xyz -> 保存 npy -> 生成 HONPAS 任务
    # ------------------------------------------------------------------
    elif sub_stage == 3:
        if not work_dir:
            print("❌ 错误: 请使用 --path 指定探索目录 (al_gpumd_explore_xxx)")
            return

        # 1. 筛选并备份 (npy)
        selector = ActiveSelector(work_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"al_candidates_{timestamp}"
        
        candidates_sys = selector.select_and_save(backup_name)

        if not candidates_sys:
            return

        # 2. 构建 HONPAS 计算任务 (标注)
        # 我们可以单独建立一个 workspace 以便区分
        dft_workspace = f"workspace_al_dft_{timestamp}"
        
        from modules.generation.wrapper import InputWrapper
        from modules.generation.scheduler import TaskScheduler
        
        # 使用 SCF 模式进行标注
        template_path = cfg.TEMPLATE_MAP["scf"]
        wrapper = InputWrapper(template_path, cfg.SPECIES_MAP)
        scheduler = TaskScheduler(dft_workspace, submit_cmd="sbatch")

        print(f"[Workflow] 正在生成 HONPAS 标注任务至: {dft_workspace} ...")
        
        count = 0
        for i in range(len(candidates_sys)):
            frame_data = candidates_sys[i].data
            # 自动复制赝势
            scheduler.setup_task(i, wrapper, frame_data, psf_dir=cfg.TEMPLATE_DIR)
            count += 1
            
        print(f"✅ 任务构建完成，共 {count} 个。")
        
        # 3. 提交
        # 根据命令行参数决定是否真实提交
        scheduler.submit_all(dry_run=dry_run)