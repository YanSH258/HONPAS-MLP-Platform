import os
import numpy as np

class NEPConverter:
    @staticmethod
    def save_as_xyz(system, filename, mode='w'):
        """
        将 dpdata 系统保存为 GPUMD XYZ 格式
        :param mode: 'w' 覆盖写入, 'a' 追加写入
        """
        # 如果是覆盖模式且文件已存在，先提示一下
        action = "覆盖导出" if mode == 'w' else "追加导出"
        print(f"[Converter] 正在{action} GPUMD 格式 XYZ: {filename}")
        
        n_frames = system.get_nframes()
        atom_names = system['atom_names']
        atom_types = system['atom_types']
        natoms = len(atom_types)
        
        coords = system['coords']
        cells = system['cells']
        
        data_dict = system.data
        has_energy = 'energies' in data_dict
        has_force = 'forces' in data_dict
        has_virial = 'virials' in data_dict
        
        energies = system['energies'] if has_energy else np.zeros(n_frames)
        forces = system['forces'] if has_force else np.zeros((n_frames, natoms, 3))
        virials = system['virials'] if has_virial else None

        # 使用传入的 mode 打开文件
        with open(filename, mode) as f:
            for i in range(n_frames):
                # --- Line 1 ---
                f.write(f"{natoms}\n")
                
                # --- Line 2 ---
                lattice_str = " ".join(f"{x:.9f}" for x in cells[i].flatten())
                props = "species:S:1:pos:R:3"
                if has_force: props += ":force:R:3"
                
                header = f'Lattice="{lattice_str}" Properties={props} energy={energies[i]:.9f} config_type=train pbc="T T T"'
                
                if has_virial:
                    # HONPAS -> GPUMD 符号修正
                    gpumd_virial = -1.0 * virials[i]
                    vir_str = " ".join(f"{x:.9f}" for x in gpumd_virial.flatten())
                    header += f' virial="{vir_str}"'
                
                f.write(header + "\n")
                
                # --- Body ---
                for j in range(natoms):
                    species = atom_names[atom_types[j]]
                    x, y, z = coords[i][j]
                    line = f"{species} {x:.9f} {y:.9f} {z:.9f}"
                    if has_force:
                        fx, fy, fz = forces[i][j]
                        line += f" {fx:.9f} {fy:.9f} {fz:.9f}"
                    f.write(line + "\n")
                    
        print(f"  -> {action}完成，写入 {n_frames} 帧(HONPS位力与GPUMD约定相反，添加负号调整)。")