import argparse
import os

from modules.workflows import (
    run_stage_1_generate,
    run_stage_2_collect,
    run_stage_3_analysis,
    run_stage_4_merge,
    run_stage_5_train,
    run_stage_6_monitor, 
    run_stage_7_eval,   
    run_stage_8_al_gpumd 
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HONPAS-MLP Automation Platform (V2)")

    # --- 基础参数 ---
    parser.add_argument("--mode", type=str, default="scf", 
                        choices=["scf", "relax", "aimd"], 
                        help="[Stage 1-3] 计算模式 (scf/relax/aimd)")
    
    parser.add_argument("--stage", type=int, default=1, 
                        choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                        help="阶段: 1=生成提交, 2=收集清洗, 3=数据分析, 4=数据合并, " \
                        "5=训练准备, 6=训练监控, 7=模型评估, 8=主动学习" )
    
    parser.add_argument("--submit", action="store_true", 
                        help="[Stage 1] 是否真实提交作业 (不加则为Dry Run)")
    
    # --- 路径相关参数 ---
    parser.add_argument("--path", type=str, default=None, 
                        help="[Stage 3/6/7] 指定目标工作目录")
    
    parser.add_argument("--data_path", type=str, default=None,
                        help="[Stage 5] 指定用于训练的数据源路径")

    # --- 模型相关参数 ---
    parser.add_argument("--model_type", type=str, default="deepmd",
                        choices=["deepmd", "gpumd"],
                        help="[Stage 5-7] 模型类型")

    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="[Stage 5] 验证集划分比例")
    
    parser.add_argument("--sub", type=int, default=1, help="Stage 8 子阶段")
    
    args = parser.parse_args()

    # ================= 逻辑调度 =================
    
    if args.stage == 1:
        run_stage_1_generate(mode=args.mode, dry_run=not args.submit)
        
    elif args.stage == 2:
        run_stage_2_collect(mode=args.mode)
        
    elif args.stage == 3:
        run_stage_3_analysis(mode=args.mode, custom_path=args.path)

    elif args.stage == 4:
        run_stage_4_merge()
        
    elif args.stage == 5:
        run_stage_5_train(
            model_type=args.model_type, 
            data_path=args.data_path,
            val_ratio=args.val_ratio
        )
        
    elif args.stage == 6:
        # 训练监控 (Loss Curve)
        run_stage_6_monitor(
            model_type=args.model_type, 
            work_dir=args.path
        )
        
    elif args.stage == 7:
        # 模型评估 (Freeze/Compress/Test/ParityPlot)
        run_stage_7_eval(
            model_type=args.model_type, 
            work_dir=args.path
        )

    elif args.stage == 8:
        from modules.workflows import run_stage_8_al_gpumd
        run_stage_8_al_gpumd(
            sub_stage=args.sub,
            data_path=args.data_path,
            work_dir=args.path,
            dry_run=not args.submit
        )