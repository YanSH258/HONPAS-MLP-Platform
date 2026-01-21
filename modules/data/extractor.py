import os
import glob
import dpdata
import numpy as np

class ResultExtractor:
    def __init__(self, workspace_root):
        self.workspace_root = workspace_root

    def collect_data(self, mode="scf"):
        """
        遍历目录，提取数据，并按原子数量/配比分组
        返回: dict { tuple(atom_numbs): dpdata.LabeledSystem, ... }
        """
        # 1. 确定格式
        fmt = "siesta/aimd_output" if mode == "aimd" else "siesta/output"
        print(f"[Extractor] 当前模式: {mode.upper()}, 使用格式: {fmt}")

        # 2. 找到文件
        search_pattern = os.path.join(self.workspace_root, "task_*", "output.log")
        files = sorted(glob.glob(search_pattern))
        
        print(f"[Extractor] 在 {self.workspace_root} 中找到 {len(files)} 个输出文件。")
        
        # 3. 分组容器
        grouped_systems = {} 

        for f in files:
            try:
                # 读取单个任务
                ls = dpdata.LabeledSystem(f, fmt=fmt)
                if len(ls) > 0:
                    # 获取该系统的特征键 (原子数量列表)
                    # 例如: [10, 6, 2, 26] -> tuple (不可变，可做字典key)
                    atom_counts = tuple(ls['atom_numbs'])
                    
                    if atom_counts not in grouped_systems:
                        grouped_systems[atom_counts] = ls
                    else:
                        grouped_systems[atom_counts].append(ls)
                    
                    # print(f"  [OK] 任务 {os.path.basename(os.path.dirname(f))} 提取到 {len(ls)} 帧")
            except Exception as e:
                # print(f"  [Error] 解析失败 {f}: {e}")
                pass

        if not grouped_systems:
            print("[Extractor] 没有收集到任何有效数据！")
            return None

        # 4. 汇报分组情况
        print(f"[Extractor] 数据提取完成，共发现 {len(grouped_systems)} 种不同的体系结构:")
        for key, sys in grouped_systems.items():
            # --- 修复点：手动拼接化学式 ---
            names = sys['atom_names'] # e.g. ['Ca', 'P', 'O', 'H']
            numbs = sys['atom_numbs'] # e.g. [10, 6, 26, 2]
            formula = "".join([f"{n}{c}" for n, c in zip(names, numbs)])
            
            print(f"  - 体系 {formula} (原子数 {sum(key)}): 共 {len(sys)} 帧")

        return grouped_systems