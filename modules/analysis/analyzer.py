import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dpdata
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class SOAPSpaceAnalyzer:
    def __init__(self, data_path, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"[Analyzer] 正在扫描数据路径: {data_path}")
        self.systems = []
        self.frame_labels = [] 

        for root, dirs, files in os.walk(data_path):
            if "type.raw" in files and "set.000" in dirs:
                sub_path = os.path.abspath(root)
                try:
                    s = dpdata.LabeledSystem(sub_path, fmt='deepmd/npy')
                    formula = "".join([f"{n}{c}" for n, c in zip(s['atom_names'], s['atom_numbs'])])
                    self.systems.append(s)
                    self.frame_labels.extend([formula] * len(s))
                except Exception as e:
                    print(f"      [跳过] 加载失败: {e}")
        
        if not self.systems:
            raise ValueError(f"未找到有效数据集")

    def _get_aligned_energies(self):
        """
        核心物理修正：
        通过最小二乘法计算各元素的参考能量 e_i，使得 E_total ≈ sum(n_i * e_i)
        然后返回 (E_total - sum(n_i * e_i)) / N_total，即消除基准偏差后的相对能量。
        """
        print("[Analyzer] 正在执行能量基准对齐 (Atomic Energy Alignment)...")
        
        X_counts = [] # 存储每个原子的数量 [n_Ca, n_P, n_O, n_H, n_Ni]
        Y_energies = [] # 存储总能量
        total_atoms_list = []
        
        # 统一获取所有元素名并排序，确保矩阵列对应一致
        all_names = sorted(self.systems[0]['atom_names'])
        
        for sys in self.systems:
            # 构造数量矩阵
            counts = np.zeros(len(all_names))
            for i, name in enumerate(all_names):
                if name in sys['atom_names']:
                    idx = sys['atom_names'].index(name)
                    counts[i] = sys['atom_numbs'][idx]
            
            for e in sys['energies']:
                X_counts.append(counts)
                Y_energies.append(e)
                total_atoms_list.append(np.sum(counts))
        
        X_counts = np.array(X_counts)
        Y_energies = np.array(Y_energies)
        total_atoms_array = np.array(total_atoms_list)
        
        # 求解线性方程组 X * beta = Y -> beta 就是每个原子的参考能
        # 使用最小二乘法
        atomic_refs, residuals, rank, s = np.linalg.lstsq(X_counts, Y_energies, rcond=None)
        
        # 打印计算出的参考能 (验证用)
        for name, energy in zip(all_names, atomic_refs):
            print(f"   -> 识别到元素 {name} 参考能: {energy:.4f} eV")

        # 计算修正后的能量偏差 (Residuals)
        # E_residual = E_total - E_linear_fit
        e_fitted = np.dot(X_counts, atomic_refs)
        e_diff = Y_energies - e_fitted
        
        # 归一化为每原子偏差 (eV/atom)
        e_final = e_diff / total_atoms_array
        return e_final

    def compute_and_plot(self, soap_config, species_map):
        all_ase_frames = []
        for sys in self.systems:
            all_ase_frames.extend(sys.to_ase_structure())

        # --- 获取对齐后的能量 ---
        # 之前的 (energies / natoms) 会导致巨大的组分偏差，现在改用对齐后的残差
        e_per_atom_aligned = self._get_aligned_energies()
        
        # 将最小值平移到 0，方便颜色观察
        e_relative = e_per_atom_aligned - np.min(e_per_atom_aligned)
        
        print(f"   -> 修正后能量分布范围: {np.min(e_relative):.4f} ~ {np.max(e_relative):.4f} eV/atom")

        # --- SOAP 计算 ---
        species_labels = [v['label'] for k, v in sorted(species_map.items())]
        soap = SOAP(
            species=species_labels, periodic=True,
            r_cut=soap_config.get("soap_rcut", 6.0),
            n_max=4, l_max=3, sigma=1.0, average="inner", sparse=False
        )

        print("[Analyzer] 计算 SOAP 描述符并降维...")
        descriptors = soap.create(all_ase_frames, n_jobs=-1)
        X_scaled = StandardScaler().fit_transform(descriptors)
        
        # PCA & UMAP
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        X_umap = None
        try:
            import umap
            X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_scaled)
        except ImportError: pass

        # --- 2x2 绘图逻辑 (内容同前，但 c 参数换成了对齐后的能量) ---
        print("[Analyzer] 正在生成综合分析报告...")
        sns.set_theme(style="ticks", context="paper")
        labels = np.array(self.frame_labels)
        unique_labels = sorted(list(set(labels)))
        formula_palette = sns.color_palette("Set2", len(unique_labels))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        scatter_kwargs = {"s": 10, "alpha": 0.5, "edgecolor": 'none'}

        # (0,0) PCA - Aligned Energy
        sc00 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=e_relative, cmap='plasma', **scatter_kwargs)
        axes[0, 0].set_title("PCA: Aligned Energy Gradient", fontsize=10)
        
        # (0,1) UMAP - Aligned Energy
        if X_umap is not None:
            axes[0, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=e_relative, cmap='plasma', **scatter_kwargs)
            axes[0, 1].set_title("UMAP: Aligned Energy Gradient", fontsize=10)

        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        cbar = fig.colorbar(sc00, cax=cbar_ax)
        cbar.set_label('Relative Local Energy (eV/atom)', fontsize=9)

        # (1,0) PCA - Formula
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, ax=axes[1, 0], 
                        palette=formula_palette, legend=False, **scatter_kwargs)
        axes[1, 0].set_title("PCA: Chemical Formula", fontsize=10)

        # (1,1) UMAP - Formula
        if X_umap is not None:
            sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, ax=axes[1, 1], 
                            palette=formula_palette, legend=True, **scatter_kwargs)
            axes[1, 1].set_title("UMAP: Chemical Formula", fontsize=10)
            axes[1, 1].legend(title="Systems", bbox_to_anchor=(1.2, 1), loc='upper left', fontsize='x-small')

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        save_path = os.path.join(self.output_dir, "dataset_aligned_report.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 修正后的分析报告已保存: {save_path}")