import os
import shutil
import numpy as np
import dpdata
from ase.io import read as ase_read
import config as cfg
import config_active as cfg_al

class ActiveSelector:
    def __init__(self, explore_dir):
        self.explore_dir = explore_dir

    def select_and_save_direct(self, full_save_path):
        """
        [重构版] 读取 active.xyz，筛选结构，并保存至指定的全路径。
        """
        active_file = os.path.join(self.explore_dir, "active.xyz")
        if not os.path.exists(active_file):
            print(f"[Selector] 未找到 {active_file}。没有需要标注的候选结构。")
            return None

        try:
            print(f"[Selector] 正在解析候选结构 (使用 ASE 逐帧模式)...")
            
            # 1. 使用 ASE 读取所有帧
            ase_frames = ase_read(active_file, index=':')
            if not ase_frames:
                print("⚠️ active.xyz 为空。")
                return None
            
            # 2. 转换为 dpdata.System (处理多帧)
            # dpdata.System 接收单个 atoms 对象，所以我们循环追加
            full_system = dpdata.System(ase_frames[0], fmt='ase/structure')
            if len(ase_frames) > 1:
                for frame in ase_frames[1:]:
                    temp_sys = dpdata.System(frame, fmt='ase/structure')
                    full_system.append(temp_sys)
            
            total_frames = len(full_system)
            print(f"  -> 共解析到 {total_frames} 帧结构。")

            # 3. 采样限制 (MAX_SELECTION)
            limit = cfg_al.MAX_SELECTION
            if total_frames > limit:
                print(f"  -> 触发采样限制: 从 {total_frames} 帧中随机抽取 {limit} 帧。")
                indices = np.arange(total_frames)
                np.random.seed(42)
                selected_indices = np.random.choice(indices, limit, replace=False)
                selected_indices.sort()
                final_system = full_system.sub_system(selected_indices)
            else:
                final_system = full_system

            # 4. 数据落地 (npy 格式)
            # 这里的 full_save_path 对应 iter_01/03.select
            if os.path.exists(full_save_path):
                shutil.rmtree(full_save_path)
            os.makedirs(full_save_path, exist_ok=True)
            
            print(f"[Selector] 正在备份候选结构至: {full_save_path}")
            final_system.to("deepmd/npy", full_save_path)
            
            return final_system

        except Exception as e:
            print(f"❌ 候选结构处理失败: {e}")
            return None

    def select_and_save(self, backup_name):
        """
        保留旧接口，映射到 data/perturbed/ 下，兼容旧版逻辑
        """
        backup_path = os.path.join("data", "perturbed", backup_name)
        return self.select_and_save_direct(backup_path)