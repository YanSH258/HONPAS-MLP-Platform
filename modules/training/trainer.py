import os
import json
import glob
import random
import shutil
import numpy as np
import dpdata
import config as cfg
import config_train as cfg_tr

class ModelTrainer:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _get_type_map(self):
        """获取元素列表 ['H', 'O', 'P', 'Ca']"""
        sorted_species = sorted(cfg.SPECIES_MAP.items(), key=lambda x: x[0])
        return [item[1]['label'] for item in sorted_species]

    def _locate_source_xyz(self, source_path):
        """定位源 train.xyz 文件"""
        if source_path and os.path.exists(os.path.join(source_path, "train.xyz")):
            target_xyz = os.path.join(source_path, "train.xyz")
            print(f"[Trainer] 锁定源数据文件: {target_xyz}")
            return target_xyz
        else:
            print("❌ 错误: 未找到 train.xyz。请先运行 Stage 4 (Merge) 生成合并数据。")
            if source_path: print(f"   当前路径: {source_path}")
            raise FileNotFoundError("Source train.xyz not found")

    def _split_xyz_file(self, source_xyz, val_ratio, out_train_name, out_val_name):
        """
        核心逻辑：读取大 XYZ，随机打乱，物理写入两个新文件
        """
        print(f"[Trainer] 正在读取并拆分数据 (Ratio={val_ratio})...")
        
        with open(source_xyz, 'r') as f:
            lines = f.readlines()
        
        frames = []
        i = 0
        while i < len(lines):
            try:
                line = lines[i].strip()
                if not line: # 跳过空行
                    i += 1
                    continue
                natoms = int(line)
                chunk_size = natoms + 2
                frames.append(lines[i : i+chunk_size])
                i += chunk_size
            except ValueError:
                i += 1
                continue
        
        total_frames = len(frames)
        print(f"   -> 识别到总帧数: {total_frames}")
        
        # 随机打乱
        random.seed(42)
        random.shuffle(frames)
        
        n_val = int(total_frames * val_ratio)
        val_frames = frames[:n_val]
        train_frames = frames[n_val:]
        
        # 路径
        path_train = os.path.join(self.work_dir, out_train_name)
        path_val = os.path.join(self.work_dir, out_val_name)

        # 写入训练集
        with open(path_train, 'w') as f:
            for frame in train_frames:
                f.writelines(frame)
        
        # 写入验证/测试集
        with open(path_val, 'w') as f:
            for frame in val_frames:
                f.writelines(frame)
                
        print(f"   -> 拆分完成:")
        print(f"      Train: {len(train_frames)} 帧 -> {out_train_name}")
        print(f"      Val/Test: {len(val_frames)} 帧 -> {out_val_name}")
        
        return path_train, path_val

    # ========================== DeepMD 流程 ==========================
    def prepare_deepmd(self, source_path=None, val_ratio=0.2):
        print(f"\n[Trainer] >>> 正在生成 DeepMD 配置文件 <<<")
        
        # 1. 定位源数据
        source_xyz = self._locate_source_xyz(source_path)

        # 2. 物理拆分 XYZ (保留 train.xyz 和 valid.xyz 在工作目录)
        xyz_train, xyz_val = self._split_xyz_file(source_xyz, val_ratio, "train.xyz", "valid.xyz")

        # 3. 将拆分后的 XYZ 转为 DeepMD npy 格式
        print(f"[Trainer] 正在将 XYZ 转换为 DeepMD npy 格式...")
        
        dir_train_root = os.path.join(self.work_dir, "data_train")
        dir_val_root = os.path.join(self.work_dir, "data_val")
        
        self._xyz_to_deepmd_folder(xyz_train, dir_train_root)
        self._xyz_to_deepmd_folder(xyz_val, dir_val_root)

        # 4. 获取所有生成的子系统路径 (递归查找 type.raw)
        # 这里的路径是绝对路径
        abs_train_systems = self._get_leaf_systems(dir_train_root)
        abs_val_systems = self._get_leaf_systems(dir_val_root)
        
        # 转为相对路径 (./data_train/system_0 ...) 写入 json，方便迁移
        rel_train_systems = [f"./{os.path.relpath(p, self.work_dir)}" for p in abs_train_systems]
        rel_val_systems = [f"./{os.path.relpath(p, self.work_dir)}" for p in abs_val_systems]

        # 5. 生成配置 input.json
        jdata = cfg_tr.DEEPMD_TEMPLATE.copy()
        jdata["model"]["type_map"] = self._get_type_map()
        
        jdata["training"]["training_data"] = {
            "systems": rel_train_systems, "batch_size": "auto"
        }
        jdata["training"]["validation_data"] = {
            "systems": rel_val_systems, "batch_size": "auto", "numb_btch": 1
        }

        json_path = os.path.join(self.work_dir, "input.json")
        with open(json_path, 'w') as f:
            json.dump(jdata, f, indent=4)
            
        print(f"✅ DeepMD 配置已生成: {json_path}")
        print(f"   - 训练集文件夹: data_train/ (含 {len(rel_train_systems)} 个子系统)")
        print(f"   - 验证集文件夹: data_val/ (含 {len(rel_val_systems)} 个子系统)")

    def _xyz_to_deepmd_folder(self, xyz_file, output_folder):
        """利用 dpdata 读取 XYZ 并保存为 DeepMD npy 目录结构"""
        try:
            type_map = self._get_type_map()
            # 使用 quip/gap/xyz 读取 Extended XYZ
            ms = dpdata.MultiSystems.from_file(xyz_file, fmt='quip/gap/xyz', type_map=type_map)
            
            if os.path.exists(output_folder): shutil.rmtree(output_folder)
            ms.to_deepmd_npy(output_folder)
            
        except Exception as e:
            print(f"❌ XYZ转DeepMD失败 ({xyz_file}): {e}")
            raise e

    def _get_leaf_systems(self, root_dir):
        """递归查找包含 type.raw 的目录"""
        systems = []
        for root, dirs, files in os.walk(root_dir):
            if "type.raw" in files:
                systems.append(os.path.abspath(root))
        return sorted(systems)

    # ========================== GPUMD (NEP) 流程 ==========================
    def prepare_gpumd(self, source_path=None, val_ratio=0.2):
        print(f"\n[Trainer] >>> 正在生成 GPUMD (NEP) 配置文件 <<<")
        
        # 1. 定位源数据
        source_xyz = self._locate_source_xyz(source_path)

        # 2. 物理拆分 XYZ (GPUMD 习惯叫 train.xyz 和 test.xyz)
        self._split_xyz_file(source_xyz, val_ratio, "train.xyz", "test.xyz")

        # 3. 生成 nep.in
        template = cfg_tr.GPUMD_NEP_TEMPLATE
        type_map = self._get_type_map()
        type_line = f"type          {len(type_map)} {' '.join(type_map)}"
        
        nep_path = os.path.join(self.work_dir, "nep.in")
        with open(nep_path, 'w') as f:
            f.write(f"{type_line}  # mandatory\n")
            for key, val in template.items():
                f.write(f"{key:<13} {val}\n")
                
        print(f"✅ GPUMD 配置已生成: {nep_path}")
        print(f"   - 训练集: train.xyz")
        print(f"   - 测试集: test.xyz")