import os
import random
import numpy as np
import dpdata
import config as cfg
import config_active as cfg_al

class ActiveSelector:
    def __init__(self, explore_dir):
        self.explore_dir = explore_dir

    def select_candidates(self):
        """
        读取 active.xyz，进行随机下采样
        返回: dpdata.System 对象 (包含选中的帧)
        """
        active_file = os.path.join(self.explore_dir, "active.xyz")
        if not os.path.exists(active_file):
            print(f"[Selector] 未找到 active.xyz。可能是 MD 未运行或没有结构触发阈值。")
            return None

        print(f"[Selector] 正在分析候选结构: {active_file}")
        
        try:
            # 读取数据
            # 使用 quip/gap/xyz 读取扩展 XYZ 格式
            sorted_species = sorted(cfg.SPECIES_MAP.items(), key=lambda x: x[0])
            type_map = [item[1]['label'] for item in sorted_species]
            
            # dpdata 读取 MultiSystems (返回 System 列表)
            ms = dpdata.MultiSystems.from_file(active_file, fmt='quip/gap/xyz', type_map=type_map)
            
            if len(ms) == 0:
                print("⚠️ active.xyz 为空。")
                return None
            
            # 通常 active.xyz 里的原子数是一样的，所以 ms[0] 就是我们要的 System
            # 如果原子数会变 (比如巨正则系综)，这里需要额外处理，但通常 MD 不变
            full_system = ms[0]
            total_frames = len(full_system)
            
            print(f"  -> 捕获到 {total_frames} 个高不确定度结构。")
            
            # 采样限制
            limit = cfg_al.MAX_SELECTION
            if total_frames > limit:
                print(f"  -> 触发采样限制 (Max={limit})，正在随机抽取...")
                indices = np.arange(total_frames)
                np.random.seed(42)
                selected_indices = np.random.choice(indices, limit, replace=False)
                selected_indices.sort() # 保持时序
                
                # 提取子集
                final_system = full_system.sub_system(selected_indices)
            else:
                final_system = full_system
                
            print(f"✅ 最终筛选出 {len(final_system)} 个结构用于 DFT 标注。")
            return final_system

        except Exception as e:
            print(f"❌ 读取/筛选失败: {e}")
            return None