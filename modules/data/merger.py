import os
import glob
import dpdata
import shutil
from modules.data.converter import NEPConverter

class DatasetMerger:
    def merge_all(self, dataset_dirs, output_root):
        """
        dataset_dirs: 待合并的源目录列表
        output_root: 输出的根目录 (例如 data/training/merged_master_xxx)
        """
        if not dataset_dirs:
            print("❌ 没有提供需要合并的路径。")
            return

        print(f"[Merger] 准备处理 {len(dataset_dirs)} 个数据集...")
        
        # 1. 准备输出目录和总XYZ文件
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        
        global_xyz_path = os.path.join(output_root, "train.xyz")
        # 清空/新建 train.xyz
        open(global_xyz_path, 'w').close()

        # 2. 按化学式分组读取
        # grouped_data = { "Ca44...": [System1, System2], "Ca352...": [System3] }
        grouped_data = {}

        for ddir in dataset_dirs:
            try:
                # 读取数据
                sys = dpdata.LabeledSystem(ddir, fmt='deepmd/npy')
                
                # 获取化学式作为Key
                atom_names = sys['atom_names']
                atom_numbs = sys['atom_numbs']
                formula = "".join([f"{n}{c}" for n, c in zip(atom_names, atom_numbs)])
                
                if formula not in grouped_data:
                    grouped_data[formula] = []
                grouped_data[formula].append(sys)
                
            except Exception as e:
                print(f"⚠️ 无法读取 {ddir}, 跳过。原因: {e}")

        # 3. 遍历分组进行合并与保存
        print(f"[Merger] 识别到 {len(grouped_data)} 种不同的体系结构，开始分别合并...")

        for formula, sys_list in grouped_data.items():
            # --- 合并同类项 ---
            merged_sys = sys_list[0]
            for s in sys_list[1:]:
                merged_sys.append(s)
            
            n_frames = len(merged_sys)
            n_atoms = sum(merged_sys['atom_numbs'])
            print(f"  >> 处理分组 {formula} (N={n_atoms}): 共 {n_frames} 帧")

            # --- (A) 保存 DeepMD 格式 (存入子文件夹) ---
            # 目录名: merged_master_xxx/Ca10P6...
            sub_dir = os.path.join(output_root, formula)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            
            merged_sys.to('deepmd/npy', sub_dir)
            # merged_sys.to('deepmd/raw', sub_dir) # 可选，省空间可不存
            print(f"     ✅ DeepMD数据保存至: {sub_dir}")

            # --- (B) 追加到总 GPUMD train.xyz ---
            try:
                NEPConverter.save_as_xyz(merged_sys, global_xyz_path, mode='a')
                print(f"     ✅ 追加到总 train.xyz")
            except Exception as e:
                print(f"     ⚠️ XYZ 转换失败: {e}")

        print(f"✅ Stage 4 全部完成！")
        print(f"   - DeepMD: 子文件夹 ({list(grouped_data.keys())})")
        print(f"   - GPUMD:  {global_xyz_path}")