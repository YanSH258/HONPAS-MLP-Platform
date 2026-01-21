import os
import glob
import dpdata
import time
from modules.data.converter import NEPConverter

class DatasetMerger:
    @staticmethod
    def merge_all(dataset_dirs, output_path):
        """
        dataset_dirs: 一个包含多个数据集路径的列表
        output_path: 合并后的保存路径
        """
        if not dataset_dirs:
            print("❌ 没有提供需要合并的路径。")
            return None

        print(f"[Merger] 准备合并 {len(dataset_dirs)} 个数据集...")
        
        # 1. 加载第一个作为基准
        try:
            # 假设之前保存的都是 deepmd/npy 格式
            final_system = dpdata.LabeledSystem(dataset_dirs[0], fmt='deepmd/npy')
            print(f"   -> [1/{len(dataset_dirs)}] 加载: {dataset_dirs[0]} ({len(final_system)} 帧)")
        except Exception as e:
            print(f"❌ 加载基础数据集失败: {e}")
            return None

        # 2. 循环追加剩下的
        for i, ddir in enumerate(dataset_dirs[1:], 2):
            try:
                next_sys = dpdata.LabeledSystem(ddir, fmt='deepmd/npy')
                final_system.append(next_sys)
                print(f"   -> [{i}/{len(dataset_dirs)}] 合并: {ddir} ({len(next_sys)} 帧)")
            except Exception as e:
                print(f"⚠️ 跳过数据集 {ddir}，原因: {e}")

        # 3. 保存结果
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        print(f"[Merger] 正在写入硬盘: {output_path}")

        # (A) 保存 DeepMD 格式
        final_system.to('deepmd/npy', output_path)
        final_system.to('deepmd/raw', output_path)
        print(f"   -> DeepMD (npy/raw) 已保存")
        
        # (B) 保存 GPUMD 格式 (train.xyz) -> 新增功能
        xyz_out = os.path.join(output_path, "train.xyz")
        try:
            # 调用转换器，它会自动处理维里符号修正
            NEPConverter.save_as_xyz(final_system, xyz_out)
            print(f"   -> GPUMD (train.xyz) 已保存")
        except Exception as e:
            print(f"⚠️ 生成 train.xyz 失败: {e}")
        
        print(f"✅ 合并全流程完成！总帧数: {len(final_system)}")
        return final_system