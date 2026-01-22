# config.py

# 输入与输出
INPUT_PATH = "data/raw/POSCAR"
WORKSPACE_PREFIX = "workspace"

# 模板配置
TEMPLATE_DIR = "templates/psfs"
TEMPLATE_MAP = {
    "scf":   "templates/honpas_scf.in",
    "relax": "templates/honpas_relax.in",
    "aimd":  "templates/honpas_aimd.in"
}

# 元素映射
SPECIES_MAP = {
    20: {'index': 1, 'label': 'Ca'},
    15: {'index': 2, 'label': 'P'},
    8:  {'index': 3, 'label': 'O'},
    1:  {'index': 4, 'label': 'H'}
}

# 超胞配置
SUPERCELL_SIZE = [2, 2, 2] 

# 微扰配置
PERTURB_CONFIG = {
    "cell_pert_fraction": 0.03,
    "atom_pert_distance": 0.1,  # 如果预筛选剔除率太高，可以适当减小这个值
    "atom_pert_style": "normal"
}

# stage1 任务数量
NUM_TASKS = 20

# 质量控制配置
QC_OVERLAP_THRESHOLD = 0.5  # 共价半径之和的倍数
QC_MAX_FORCE = 100         # eV/Ang
QC_SIGMA_E = 4.0            # Energy Z-score

# stage3 可视化与分析配置
VIS_CONFIG = {
    "soap_rcut": 6.0,    # 截断半径 
    "soap_nmax": 8,      # 径向基函数数量
    "soap_lmax": 6,      # 球谐函数阶数
    "soap_sigma": 0.5,   # 高斯扩宽宽度
}