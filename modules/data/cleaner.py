import numpy as np
import matplotlib.pyplot as plt
import os
import dpdata
from ase import Atoms
from ase.geometry import get_distances
from ase.data import covalent_radii, atomic_numbers 

class DataQualityControl:
    def __init__(self, dp_system: dpdata.LabeledSystem):
        self.system = dp_system
        self.n_frames = len(dp_system)
        self.valid_mask = np.ones(self.n_frames, dtype=bool)
        self.rejected_reasons = []

    def check_atom_overlap(self, threshold_factor=0.5):
        """
        优化后的逻辑：基于共价半径之和的倍数检查重叠
        threshold = threshold_factor * (R_i + R_j)
        """
        print(f"[QC] 正在检查原子重叠 (阈值因子: {threshold_factor} * 共价半径之和)...")
        
        coords = self.system['coords']
        cells = self.system['cells']
        atom_types = self.system['atom_types']
        atom_names = self.system['atom_names']
        
        # 1. 预计算该体系的阈值矩阵
        # 获取每个元素的共价半径
        species_radii = [covalent_radii[atomic_numbers[name]] for name in atom_names]
        # 映射到每一个原子
        atom_radii = np.array([species_radii[t] for t in atom_types])
        # 计算 N x N 的阈值矩阵: T_ij = factor * (r_i + r_j)
        threshold_matrix = threshold_factor * (atom_radii[:, None] + atom_radii[None, :])
        
        symbols = [atom_names[i] for i in atom_types]
        count = 0

        for i in range(self.n_frames):
            if not self.valid_mask[i]:
                continue

            # 构建 ASE 对象处理周期性边界 (MIC)
            atoms = Atoms(symbols=symbols, positions=coords[i], cell=cells[i], pbc=True)
            
            # 获取所有原子对的最小镜像距离矩阵
            dists = atoms.get_all_distances(mic=True)
            
            # 排除自身距离
            np.fill_diagonal(dists, np.inf)
            
            # 核心判断：如果任何原子对距离 < 阈值矩阵对应的元素
            if np.any(dists < threshold_matrix):
                self.valid_mask[i] = False
                self.rejected_reasons.append(f"Frame {i}: Atomic overlap detected (below radii sum factor)")
                count += 1
                
        print(f"  -> 发现 {count} 帧存在严重原子重叠。")
        return self

    def check_outliers(self, sigma_n=4.0, max_force_tol=100.0):
        """
        统计检查：剔除能量/力异常的帧
        :param sigma_n: 能量剔除阈值 (平均值 +/- N * 标准差)
        :param max_force_tol: 力的绝对上限 (eV/Ang)
        """
        print(f"[QC] 正在检查数值异常 (MaxForce > {max_force_tol}, Energy Outliers)...")
        
        energies = self.system['energies']
        forces = self.system['forces']
        natoms = self.system.get_natoms()
        
        # 1. 检查力 (绝对值)
        max_forces = np.max(np.abs(forces), axis=(1, 2)) # 每帧的最大力
        
        # 2. 检查能量 (每原子能量)
        e_per_atom = energies / natoms
        mean_e = np.mean(e_per_atom)
        std_e = np.std(e_per_atom)
        
        count_f = 0
        count_e = 0
        
        for i in range(self.n_frames):
            if not self.valid_mask[i]: continue

            # 力检查
            if max_forces[i] > max_force_tol:
                self.valid_mask[i] = False
                self.rejected_reasons.append(f"Frame {i}: Max force {max_forces[i]:.2f} > {max_force_tol}")
                count_f += 1
                continue

            # 能量统计检查 (Z-score)
            # 注意：如果体系变化很大(如相变)，能量分布可能不是正态的，这里仅用于剔除极端离群值
            if abs(e_per_atom[i] - mean_e) > sigma_n * std_e:
                self.valid_mask[i] = False
                self.rejected_reasons.append(f"Frame {i}: Energy outlier (Z-score > {sigma_n})")
                count_e += 1

        print(f"  -> 剔除 {count_f} 帧 (力过大), {count_e} 帧 (能量离群)。")
        return self

    def generate_report(self, output_dir):
        """
        生成清洗后的数据系统，并绘制分布图
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 获取有效索引
        valid_indices = np.where(self.valid_mask)[0]
        n_kept = len(valid_indices)
        
        print(f"[QC] 总结: 原始 {self.n_frames} 帧 -> 保留 {n_kept} 帧 (剔除率 {(self.n_frames-n_kept)/self.n_frames:.1%})")
        
        # 2. 生成新的 dpdata.System
        if n_kept == 0:
            print("❌ 警告: 所有数据均被剔除！")
            return None
            
        cleaned_system = self.system.sub_system(valid_indices)
        
        # 3. 绘图 (Energy & Force Histograms)
        self._plot_hist(cleaned_system, output_dir)
        
        # 4. 保存剔除日志
        with open(os.path.join(output_dir, "qc_rejected_log.txt"), "w") as f:
            f.write("\n".join(self.rejected_reasons))
            
        return cleaned_system

    def _plot_hist(self, system, output_dir):
        energies = system['energies'] / system.get_natoms()
        forces = system['forces'].flatten()
        
        plt.figure(figsize=(12, 5))
        
        # Energy Plot
        plt.subplot(1, 2, 1)
        plt.hist(energies, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Energy per Atom Distribution (N={len(energies)})")
        plt.xlabel("Energy (eV/atom)")
        plt.ylabel("Count")
        
        # Force Plot
        plt.subplot(1, 2, 2)
        plt.hist(forces, bins=50, color='salmon', edgecolor='black', alpha=0.7)
        plt.title("Force Component Distribution")
        plt.xlabel("Force (eV/Ang)")
        plt.ylabel("Count")
        plt.yscale('log') # 力通常跨度大，用对数坐标
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "qc_distribution.png"), dpi=150)
        plt.close()
        print(f"  -> 质量分布图已保存至: {output_dir}/qc_distribution.png")