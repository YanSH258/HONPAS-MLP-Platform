import os
import numpy as np

class NEPConverter:
    @staticmethod
    def save_as_xyz(system, filename):
        """
        将 dpdata.System/LabeledSystem 对象保存为 GPUMD (NEP) 标准的 Extended XYZ 格式
        :param system: dpdata 系统对象 (已加载数据)
        :param filename: 输出文件名 (通常为 train.xyz)
        """
        print(f"[Converter] 正在导出 GPUMD 格式 XYZ: {filename}")
        
        # 1. 获取基本信息
        n_frames = system.get_nframes()
        atom_names = system['atom_names']
        atom_types = system['atom_types']
        natoms = len(atom_types)
        
        # 2. 准备数据引用
        coords = system['coords']
        cells = system['cells']
        
        data_dict = system.data
        has_energy = 'energies' in data_dict
        has_force = 'forces' in data_dict
        has_virial = 'virials' in data_dict
        
        energies = system['energies'] if has_energy else np.zeros(n_frames)
        forces = system['forces'] if has_force else np.zeros((n_frames, natoms, 3))
        virials = system['virials'] if has_virial else None

        # 3. 写入文件
        with open(filename, 'w') as f:
            for i in range(n_frames):
                # --- Line 1: 原子数 ---
                f.write(f"{natoms}\n")
                
                # --- Line 2: Header ---
                
                # 晶胞
                lattice_str = " ".join(f"{x:.9f}" for x in cells[i].flatten())
                
                # 属性定义
                props = "species:S:1:pos:R:3"
                if has_force:
                    props += ":force:R:3"
                
                # 组装 Header
                header = f'Lattice="{lattice_str}" Properties={props} energy={energies[i]:.9f} config_type=train pbc="T T T"'
                
                # 处理维里 (关键修改点！)
                if has_virial:
                    # HONPAS 计算出的 Virial 与 GPUMD 约定相差负号
                    # 因此这里乘以 -1.0
                    gpumd_virial = -1.0 * virials[i]
                    
                    vir_str = " ".join(f"{x:.9f}" for x in gpumd_virial.flatten())
                    header += f' virial="{vir_str}"'
                
                f.write(header + "\n")
                
                # --- Body: 原子数据 ---
                for j in range(natoms):
                    species = atom_names[atom_types[j]]
                    x, y, z = coords[i][j]
                    line = f"{species} {x:.9f} {y:.9f} {z:.9f}"
                    
                    if has_force:
                        fx, fy, fz = forces[i][j]
                        line += f" {fx:.9f} {fy:.9f} {fz:.9f}"
                    
                    f.write(line + "\n")
                    
        print(f"  -> 转换完成，已生成 {n_frames} 帧数据 (HONPS位力与GPUMD约定相反，添加负号调整)。")