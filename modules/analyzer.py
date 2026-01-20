import os
import numpy as np
import matplotlib.pyplot as plt
import dpdata
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # 引入 t-SNE
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold

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
        
        # --- 改进点 A: 特征筛选 (剔除几乎不变的特征列) ---
        selector = VarianceThreshold(threshold=0) 
        descriptors = selector.fit_transform(descriptors)
        print(f"   -> 筛选后特征维度: {descriptors.shape}")

        # --- 改进点 B: 换用 MaxAbsScaler (保持稀疏性且防止噪声放大) ---
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(descriptors)

        # 2. PCA 分析 (用于诊断)
        print("[Analyzer] 2. 执行 PCA 诊断...")
        pca_full = PCA().fit(X_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # 绘制方差解释曲线
        plt.figure(figsize=(6, 4))
        plt.plot(cumulative_variance, marker='o')
        plt.axhline(y=0.9, color='r', linestyle='--')
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Variance Ratio")
        plt.savefig(os.path.join(self.output_dir, "pca_variance_check.png"))
        plt.close()

        # --- 改进点 C: 使用 t-SNE 进行非线性降维可视化 ---
        print("[Analyzer] 3. 执行 t-SNE 降维 (非线性)...")
        # perplexity 建议设为样本数的 1/5 到 1/10 左右
        tsne = TSNE(n_components=2, perplexity=min(30, len(self.system)//5), 
                    init='pca', learning_rate='auto', random_state=42)
        X_embedded = tsne.fit_transform(X_scaled)

        # 3. 绘图
        print("[Analyzer] 4. 绘图...")
        energies = self.system['energies']
        natoms = self.system.get_natoms()
        e_relative = (energies / natoms) - np.min(energies / natoms)

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=e_relative, 
                         cmap='plasma', s=30, alpha=0.8, edgecolor='grey', lw=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label('Relative Energy (eV/atom)')
        
        plt.title(f"SOAP Manifold Visualization (t-SNE)\nDataset: {len(self.system)} frames")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        
        save_path = os.path.join(self.output_dir, "soap_tsne_map.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ t-SNE 降维图已保存: {save_path}")
        print(f"✅ 方差诊断图已保存: {self.output_dir}/pca_variance_check.png")