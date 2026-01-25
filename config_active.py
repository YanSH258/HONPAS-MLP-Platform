# config_active.py

# ==============================================================================
# GPUMD 主动学习 (Active Learning) 
# ==============================================================================

# 1. 基础 MD 设置
EXPLORE_CONFIG = {
    "time_step": 1.0,          # fs
    "steps": 10000,            # 探索步数
    "velocity_temp": 300,      # 初始速度对应温度
    "dump_thermo": 100,        # 热力学输出间隔
    "dump_xyz_interval": 1000, # 轨迹输出间隔
    "dump_xyz_properties": ["velocity", "force"] 
}

# 2. 系综配置 (Ensemble)
# 默认使用 nvt_ber，支持切换到 npt_mttk
ENSEMBLE_CONFIG = {
    "method": "nvt_ber",       # 默认 NVT
    
    # --- 温度参数 ---
    "T_start": 300,
    "T_end": 300,
    "T_coupling": 100,         # 对于 mttk 对应 tperiod
    
    # --- 压力参数 (仅 npt_mttk 需要) ---
    "p_direction": "iso",      # iso, aniso, tri, x, y, z...
    "p_start": 0.0,            # P1 (GPa)
    "p_end": 0.0,              # P2 (GPa)
    "p_coupling": 1000         # pperiod
}

# 3. 主动学习筛选策略 (active 关键字)
ACTIVE_STRATEGY = {
    "interval": 10,            # 每 10 步检查一次
    "threshold": 0.05,         # 力误差阈值 (eV/A)
    "has_velocity": 0,
    "has_force": 0,
    "has_uncertainty": 1
}

N_ENSEMBLE = 4
MAX_SELECTION = 100