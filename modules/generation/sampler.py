import dpdata
import os
import numpy as np
from modules.data.validator import StructureValidator 

class StructureSampler:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] 找不到文件: {file_path}")
        
        self.base_system = dpdata.System(file_path, fmt='vasp/poscar')
        # 为了兼容 main.py 中的 sampler.current_system 调用
        self.current_system = self.base_system

    def apply_supercell(self, size):
        """
        执行扩胞操作
        """
        if size is None:
            return
            
        if any(s > 1 for s in size):
            try:
                # 执行扩胞，更新 base_system
                self.base_system = self.base_system.replicate(size)
                # 同步更新 current_system 供 main.py 使用
                self.current_system = self.base_system
                
                print(f"[Sampler] 扩胞成功，当前尺寸: {size}")
                # 修正方法名：使用 get_natoms()
                print(f"[Sampler] 扩胞后原子总数: {self.base_system.get_natoms()}")
            except Exception as e:
                print(f"[Error] 扩胞失败: {e}")
                raise
        else:
            print("[Sampler] 扩胞配置为 [1, 1, 1]，跳过扩胞。")

    def generate_perturbations(self, num_perturbed=5, pert_config=None):
        """
        生成微扰结构 (基于当前 base_system)
        """
        if pert_config is None:
            pert_config = {"cell_pert_fraction": 0.03, "atom_pert_distance": 0.1, "atom_pert_style": "normal"}
        
        print(f"[Sampler] 目标: 生成 {num_perturbed} 个合格的微扰结构...")
        
        valid_systems = []
        attempts = 0
        max_attempts = num_perturbed * 15 
        
        while len(valid_systems) < num_perturbed and attempts < max_attempts:
            attempts += 1
            
            # 使用最新的 base_system 进行微扰
            tmp_sys = self.base_system.perturb(pert_num=1, **pert_config)
            
            # 物理检查
            ase_atoms = tmp_sys.to_ase_structure()[0] 
            is_valid, msg = StructureValidator.check_overlap(ase_atoms, threshold_factor=0.5)
            
            if is_valid:
                valid_systems.append(tmp_sys)
                
        if len(valid_systems) < num_perturbed:
            print(f"⚠️ 警告: 尝试了 {attempts} 次，仅生成 {len(valid_systems)} 个合格结构。")
        else:
            print(f"[Sampler] 成功生成 {len(valid_systems)} 个结构 (总尝试次数: {attempts})")
        
        if not valid_systems:
            raise RuntimeError("未能生成任何有效结构。")

        final_system = valid_systems[0]
        for s in valid_systems[1:]:
            final_system.append(s)
            
        return final_system