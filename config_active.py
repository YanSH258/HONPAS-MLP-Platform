# config_active.py

# ==============================================================================
# GPUMD 主动学习 (Active Learning) 参数配置
# ==============================================================================

# 1. 系综训练设置
# 训练多少个 NEP 模型用于误差评估
N_ENSEMBLE = 4

# 2. MD 探索参数 (控制 run.in)
EXPLORE_CONFIG = {
    "time_step": 1.0,          # 时间步长 (fs)
    "steps": 100000,           # MD 总步数
    
    # 温度设置 (可以是单个值，也可以是列表用于多轮迭代)
    # 如果是列表，Workflow 需要逻辑去轮询，这里暂定为单次运行的温度
    "temperature": 300,        
    
    # 系综设置: 'nvt_ber', 'npt_berendsen', 'nve' 等
    # 格式: "ensemble_keyword parameters"
    # 例如 NVT: "nvt_ber 300 300 100" (T_start, T_end, Tau)
    # 例如 NPT: "npt_scr 300 300 100 0 0 0 100 100 100 1000" 
    "ensemble_str": "nvt_bdp 300 300 100",
    
    # 初始速度
    "velocity_temp": 300       # 初始速度对应的温度
}

# 3. 主动学习筛选策略 (active 关键字)
ACTIVE_STRATEGY = {
    "interval": 10,            # 每多少步检查一次
    "has_velocity":0,         # 不输出速度
    "has_force":0,            # 不输出力
    "has_uncertainty"         #输出不确定度
    "threshold": 0.05,         # 力误差阈值 (eV/A), 超过此值保存结构
}

# 4. 采样限制
# 从 active.xyz 中最多挑选多少个结构送去算 DFT
# 避免一次性生成太多任务把集群算爆
MAX_SELECTION = 100