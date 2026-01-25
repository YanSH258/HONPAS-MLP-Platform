import numpy as np
from ase.data import covalent_radii
from ase.geometry import get_distances

class StructureValidator:
    @staticmethod
    def check_overlap(atoms, threshold_factor=0.5):
        """
        检查 ASE Atoms 对象是否存在原子重叠
        判据: dist_ij < threshold_factor * (r_i + r_j)
        :param atoms: ase.Atoms 对象
        :param threshold_factor: 阈值系数 (默认 0.5)
        :return: (is_valid, message)
        """
        # 获取原子序数
        numbers = atoms.get_atomic_numbers()
        
        # 获取所有原子对的距离 (考虑周期性边界条件 mic=True)
        # 这是一个扁平化的距离矩阵
        # 注意: get_distances 较慢，大规模可用 neighbor list 优化，但几十个原子无所谓
        dist_matrix = atoms.get_all_distances(mic=True)
        
        # 将对角线(自己到自己)设为无穷大，避免误判
        np.fill_diagonal(dist_matrix, np.inf)
        
        # 获取原子对的索引
        n_atoms = len(atoms)
        
        # 遍历上三角矩阵检查
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d_ij = dist_matrix[i, j]
                
                # 计算动态阈值
                r_i = covalent_radii[numbers[i]]
                r_j = covalent_radii[numbers[j]]
                limit = threshold_factor * (r_i + r_j)
                
                if d_ij < limit:
                    return False, f"Overlap detected: Atom {i}({numbers[i]}) - Atom {j}({numbers[j]}) dist={d_ij:.3f} < {limit:.3f}"
                    
        return True, "Pass"