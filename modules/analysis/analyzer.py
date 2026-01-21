import os
import numpy as np
import matplotlib.pyplot as plt
import dpdata
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold

# 导入 UMAP
try:
    import umap
except ImportError:
    print("❌ 未找到 umap-learn 库，请执行: pip install umap-learn")

class SOAPSpaceAnalyzer:
    def __init__(self, data_path, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"[Analyzer] 正在加载数据: {data_path}")
        self.system = dpdata.LabeledSystem(data_path, fmt='deepmd/npy')
        print(f"   -> 成功加载 {len(self.system)} 帧。")

    def compute_and_plot(self, soap_config, species_map):
        if len(self.system) == 0: return

        # 1. 计算 SOAP
        print("[Analyzer] 1. 计算全局 SOAP...")
        ase_frames = self.system.to_ase_structure()
        species_labels = [v['label'] for k, v in species_map.items()]

        soap = SOAP(
            species=species_labels,
            periodic=True,
            r_cut=soap_config.get("soap_rcut", 6.0),
            n_max=soap_config.get("soap_nmax", 8),
            l_max=soap_config.get("soap_lmax", 6),
            sigma=soap_config.get("soap_sigma", 0.5),
            average="inner",
            sparse=False
        )

        descriptors = soap.create(ase_frames, n_jobs=-1)
        
        # 特征筛选：剔除无用特征
        selector = VarianceThreshold(threshold=0) 
        descriptors = selector.fit_transform(descriptors)

        # 预处理
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(descriptors)

        # 2. PCA 诊断 (保留之前的逻辑，用于查看系统复杂性)
        pca_full = PCA().fit(X_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        plt.figure(figsize=(6, 4))
        plt.plot(cumulative_variance, marker='o', markersize=4)
        plt.axhline(y=0.9, color='r', linestyle='--')
        plt.title("Cumulative Explained Variance (PCA)")
        plt.savefig(os.path.join(self.output_dir, "pca_variance_check.png"))
        plt.close()

        # 3. 执行 UMAP 降维
        print("[Analyzer] 3. 执行 UMAP 降维 (非线性)...")
        # n_neighbors: 邻居数量，越大越体现全局结构，建议 15-50
        # min_dist: 点之间的紧密程度，建议 0.1
        reducer = umap.UMAP(
            n_neighbors=30, 
            min_dist=0.1, 
            n_components=2, 
            random_state=42,
            metric='euclidean'
        )
        X_embedded = reducer.fit_transform(X_scaled)

        # 4. 绘图优化
        print("[Analyzer] 4. 绘图...")
        energies = self.system['energies']
        natoms = self.system.get_natoms()
        e_relative = (energies / natoms) - np.min(energies / natoms)

        plt.figure(figsize=(11, 8))
        
        # 绘制点，减小 s (size)，增加 alpha，使用更美观的 cmap
        sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                         c=e_relative, cmap='viridis', 
                         s=15, alpha=0.6, edgecolors='none')
        
        # 如果是 AIMD，可以画一条极淡的线连接相邻帧，显示演化路径
        plt.plot(X_embedded[:, 0], X_embedded[:, 1], color='grey', lw=0.5, alpha=0.2)

        cbar = plt.colorbar(sc)
        cbar.set_label('Relative Energy (eV/atom)')
        
        plt.title(f"SOAP Manifold Visualization (UMAP)\nDataset: {len(self.system)} frames (AIMD)")
        plt.xlabel("UMAP dimension 1")
        plt.ylabel("UMAP dimension 2")
        
        save_path = os.path.join(self.output_dir, "soap_umap_map.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ UMAP 降维图已保存: {save_path}")