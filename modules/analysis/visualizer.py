import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dpdata
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from ase import Atoms

class DatasetVisualizer:
    def __init__(self, dp_system: dpdata.LabeledSystem, output_dir):
        self.system = dp_system
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def plot_dataset_stats(self):
        """
        绘制基础统计图：能量与力分布
        """
        print("[Visualizer] 正在绘制能量与力分布直方图...")
        energies = self.system['energies']
        forces = self.system['forces'].flatten()
        natoms = self.system.get_natoms()
        
        # 转换为 per atom
        e_per_atom = energies / natoms
        
        plt.figure(figsize=(14, 6))
        
        # 1. Energy
        plt.subplot(1, 2, 1)
        sns.histplot(e_per_atom, kde=True, color='skyblue', bins=30)
        plt.title(f"Energy Distribution (N={len(energies)})")
        plt.xlabel("Energy (eV/atom)")
        plt.ylabel("Count")
        
        # 2. Force
        plt.subplot(1, 2, 2)
        # 取力的 log10 magnitude 以便观察量级分布
        f_mag = np.linalg.norm(self.system['forces'], axis=2).flatten()
        sns.histplot(f_mag, kde=True, color='salmon', bins=50, log_scale=True)
        plt.title("Force Magnitude Distribution (log scale)")
        plt.xlabel("Force Magnitude (eV/Ang)")
        plt.ylabel("Count")
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "stat_energy_force.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> 保存至: {save_path}")

    def analyze_soap(self, soap_config, species_map):
        """
        利用 SOAP + PCA 分析局域环境分布
        """
        print("[Visualizer] 正在计算 SOAP 描述符 (这可能需要一点时间)...")
        
        # 1. 准备 ASE Atoms 列表
        # dpdata 转 ASE
        ase_frames = self.system.to_ase_structure()
        
        # 获取元素列表 (例如 ['H', 'O', 'P', 'Ca'])
        # 注意：dscribe 需要 species 列表
        species_labels = [v['label'] for k, v in species_map.items()]
        
        # 2. 初始化 SOAP
        soap = SOAP(
            species=species_labels,
            periodic=True,
            rcut=soap_config['soap_rcut'],
            nmax=soap_config['soap_nmax'],
            lmax=soap_config['soap_lmax'],
            sigma=soap_config['soap_sigma'],
            sparse=False
        )
        
        # 3. 批量计算 SOAP
        # create 返回 (n_frames, n_atoms, n_features)
        # 我们需要将其 reshape 为 (n_total_atoms, n_features)
        # 以便分析所有帧中所有原子的环境
        features = soap.create(ase_frames, n_jobs=4) # n_jobs 并行计算
        
        n_frames, n_atoms, n_feats = features.shape
        flat_features = features.reshape(-1, n_feats)
        
        print(f"  -> SOAP 特征维度: {flat_features.shape}")
        
        # 4. PCA 降维
        print("[Visualizer] 正在执行 PCA 降维...")
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(flat_features)
        
        # 5. 准备绘图数据 (为每个点标记元素类型)
        # 获取所有帧的所有原子类型
        all_atom_types = self.system['atom_types'].flatten() # (n_frames * n_atoms,)
        # 将类型索引转换为元素符号
        # species_map: {20: {'label': 'Ca', ...}}
        # dpdata 的 atom_names: ['Ca', 'H', 'O', 'P'] (顺序可能不同，需注意)
        
        dp_atom_names = self.system['atom_names']
        point_labels = [dp_atom_names[idx] for idx in all_atom_types]
        
        # 6. 绘图
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_features[:, 0], 
            y=reduced_features[:, 1], 
            hue=point_labels,
            alpha=0.6,
            s=10,
            palette="deep"
        )
        
        plt.title(f"SOAP Descriptor PCA Analysis\n(rcut={soap_config['soap_rcut']}, Frames={n_frames})")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "analysis_soap_pca.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> SOAP PCA 图已保存至: {save_path}")