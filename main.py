import argparse

# 从 workflows 导入所有阶段的入口函数
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

    # --- 基础控制参数 ---
    parser.add_argument("--stage", type=int, default=1, 
                        choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                        help="执行阶段: 1=生成提交, 2=收集清洗, 3=分析, 4=合并, 5=训练准备, 6=监控, 7=评估, 8=主动学习")
    
    parser.add_argument("--mode", type=str, default="scf", 
                        choices=["scf", "relax", "aimd"], 
                        help="计算模式 (仅限Stage 1-3)")
    
    parser.add_argument("--submit", action="store_true", 
                        help="是否真实提交作业 (Stage 1/8有效，不加则为Dry Run)")
    
    # --- 路径与数据源参数 ---
    parser.add_argument("--path", type=str, default=None, 
                        help="指定目标工作目录 (Stage 3/6/7/8)")
    
    parser.add_argument("--data_path", type=str, default=None,
                        help="指定训练/探索的数据源路径 (Stage 5/8)")

    # --- 模型与训练参数 ---
    parser.add_argument("--model_type", type=str, default="deepmd",
                        choices=["deepmd", "gpumd"],
                        help="模型框架选择 (Stage 5-7)")

    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集/测试集划分比例 (Stage 5/8)")
    
    parser.add_argument("--sub", type=int, default=1, choices=[1, 2, 3],
                        help="主动学习子阶段 (仅限Stage 8)")

    parser.add_argument("--iter", type=int, default=0,
                        help="当前迭代轮次 (例如: 1 代表 iter_01)")

    args = parser.parse_args()

    # ================= 逻辑调度 (Stage Dispatcher) =================
    
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
        run_stage_6_monitor(model_type=args.model_type, work_dir=args.path)
        
    elif args.stage == 7:
        run_stage_7_eval(model_type=args.model_type, work_dir=args.path)

    elif args.stage == 8:
        run_stage_8_al_gpumd(
            sub_stage=args.sub,
            data_path=args.data_path,
            work_dir=args.path,
            dry_run=not args.submit,
            iteration=args.iter 
        )