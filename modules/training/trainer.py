import os
import json
import glob
import random
import shutil
import numpy as np
import dpdata
import config as cfg
import config_train as cfg_tr
from modules.data.converter import NEPConverter

class ModelTrainer:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _get_type_map(self):
        """获取元素列表 ['H', 'O', 'P', 'Ca']"""
        sorted_species = sorted(cfg.SPECIES_MAP.items(), key=lambda x: x[0])
        return [item[1]['label'] for item in sorted_species]

    def _scan_and_split_data(self, source_path=None, val_ratio=0.2):
        """扫描数据路径"""
        valid_dirs = []

        if source_path:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"指定的数据路径不存在: {source_path}")
            
            if os.path.exists(os.path.join(source_path, "type.raw")):
                valid_dirs = [os.path.abspath(source_path)]
            else:
                sub_dirs = sorted(glob.glob(os.path.join(source_path, "*")))
                valid_dirs = [os.path.abspath(d) for d in sub_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "type.raw"))]
        else:
            search_path = os.path.join("data", "training", "set_*")
            all_dirs = sorted(glob.glob(search_path))
            valid_dirs = [os.path.abspath(d) for d in all_dirs if os.path.isdir(d)]

        if not valid_dirs:
            raise FileNotFoundError("未找到有效的数据集目录。")

        # 如果只有一个数据集，直接返回，后续逻辑处理拆分
        if len(valid_dirs) == 1:
            return valid_dirs, valid_dirs

        # 如果有多个数据集，按目录随机划分
        random.seed(42)
        random.shuffle(valid_dirs)
        n_val = int(len(valid_dirs) * val_ratio)
        if n_val == 0 and len(valid_dirs) > 1: n_val = 1
        
        val_sets = valid_dirs[:n_val]
        train_sets = valid_dirs[n_val:]
        
        print(f"[Trainer] 目录级划分: 训练集 {len(train_sets)} 个, 验证集 {len(val_sets)} 个")
        return train_sets, val_sets

    # ========================== DeepMD 流程 ==========================
    def prepare_deepmd(self, source_path=None, val_ratio=0.2):
        print(f"[Trainer] 正在生成 DeepMD 配置文件...")
        
        raw_train_paths, raw_val_paths = self._scan_and_split_data(source_path, val_ratio)
        
        # 判断是否为单一数据源
        is_single_source = (len(raw_train_paths) == 1 and raw_train_paths == raw_val_paths)

        final_train_paths = []
        final_val_paths = []

        if is_single_source:
            print(f"[Trainer] 检测到单一合并数据集，正在执行【物理拆分】并保存至工作目录...")
            # 执行拆分，并获取相对路径 (例如 ["./data_train"], ["./data_val"])
            rel_train, rel_val = self._split_and_save_deepmd(raw_train_paths[0], val_ratio)
            final_train_paths = rel_train
            final_val_paths = rel_val
        else:
            print(f"[Trainer] 使用多个散乱数据集，使用绝对路径链接...")
            final_train_paths = raw_train_paths
            final_val_paths = raw_val_paths

        # 写入配置
        jdata = cfg_tr.DEEPMD_TEMPLATE.copy()
        jdata["model"]["type_map"] = self._get_type_map()
        
        # 注意：systems 必须是列表
        jdata["training"]["training_data"] = {
            "systems": final_train_paths, "batch_size": "auto"
        }
        jdata["training"]["validation_data"] = {
            "systems": final_val_paths, "batch_size": "auto", "numb_btch": 1
        }

        json_path = os.path.join(self.work_dir, "input.json")
        with open(json_path, 'w') as f:
            json.dump(jdata, f, indent=4)
            
        print(f"✅ DeepMD 配置已生成: {json_path}")
        if is_single_source:
            print(f"   数据已物理拆分至: {self.work_dir}/data_train 和 data_val")

    def _split_and_save_deepmd(self, dataset_path, val_ratio):
        """
        读取单一数据集，打乱拆分，保存到工作目录下的 data_train 和 data_val
        返回相对路径列表
        """
        try:
            # 1. 加载
            print(f"  -> 加载原始数据: {dataset_path}")
            system = dpdata.LabeledSystem(dataset_path, fmt="deepmd/npy")
            n_frames = len(system)
            
            # 2. 打乱索引
            indices = np.arange(n_frames)
            np.random.seed(42)
            np.random.shuffle(indices)
            
            # 3. 切分
            n_val = int(n_frames * val_ratio)
            idx_val = indices[:n_val]
            idx_train = indices[n_val:]
            
            print(f"  -> 总帧数 {n_frames} | 训练集 {len(idx_train)} | 验证集 {len(idx_val)}")
            
            # 4. 定义本地路径
            train_dir_name = "data_train"
            val_dir_name = "data_val"
            
            abs_train_dir = os.path.join(self.work_dir, train_dir_name)
            abs_val_dir = os.path.join(self.work_dir, val_dir_name)
            
            # 清理旧数据
            if os.path.exists(abs_train_dir): shutil.rmtree(abs_train_dir)
            if os.path.exists(abs_val_dir): shutil.rmtree(abs_val_dir)
            
            # 5. 保存
            if len(idx_train) > 0:
                sub_train = system.sub_system(idx_train)
                sub_train.to("deepmd/npy", abs_train_dir)
                
            if len(idx_val) > 0:
                sub_val = system.sub_system(idx_val)
                sub_val.to("deepmd/npy", abs_val_dir)
            
            # 6. 返回相对路径 (DeepMD 识别 ./data_train)
            return [f"./{train_dir_name}"], [f"./{val_dir_name}"]

        except Exception as e:
            print(f"❌ DeepMD 数据拆分失败: {e}")
            raise e

    # ========================== GPUMD (NEP) 流程 ==========================
    def prepare_gpumd(self, source_path=None, val_ratio=0.2):
        print(f"[Trainer] 正在生成 GPUMD (NEP) 配置文件...")
        
        train_paths, val_paths = self._scan_and_split_data(source_path, val_ratio)
        is_single_source = (len(train_paths) == 1 and train_paths == val_paths)

        if is_single_source:
            print(f"[Trainer] 检测到单一合并数据集，正在执行【帧级别】随机拆分...")
            # 这里的逻辑和 DeepMD 类似，但是保存为 xyz
            self._split_single_dataset_frames_xyz(train_paths[0], val_ratio)
        else:
            print(f"[Trainer] 检测到多个数据集，执行【目录级别】合并...")
            self._merge_files_physically(train_paths, "train.xyz")
            self._merge_files_physically(val_paths, "test.xyz")

        # 生成 nep.in
        template = cfg_tr.GPUMD_NEP_TEMPLATE
        type_map = self._get_type_map()
        type_line = f"type          {len(type_map)} {' '.join(type_map)}"
        
        nep_path = os.path.join(self.work_dir, "nep.in")
        with open(nep_path, 'w') as f:
            f.write(f"{type_line}  # mandatory\n")
            for key, val in template.items():
                f.write(f"{key:<13} {val}\n")
        print(f"  -> nep.in 已生成。")
        print(f"✅ GPUMD 准备就绪: {self.work_dir}")

    def _split_single_dataset_frames_xyz(self, dataset_path, val_ratio):
        """GPUMD 专用的拆分逻辑 (保存为 XYZ)"""
        try:
            print(f"  -> 正在加载数据: {dataset_path}")
            system = dpdata.LabeledSystem(dataset_path, fmt="deepmd/npy")
            n_frames = len(system)
            
            indices = np.arange(n_frames)
            np.random.seed(42)
            np.random.shuffle(indices)
            
            n_val = int(n_frames * val_ratio)
            idx_val = indices[:n_val]
            idx_train = indices[n_val:]
            
            print(f"  -> 拆分: 训练集 {len(idx_train)} | 测试集 {len(idx_val)}")
            
            if len(idx_train) > 0:
                sub_train = system.sub_system(idx_train)
                NEPConverter.save_as_xyz(sub_train, os.path.join(self.work_dir, "train.xyz"))
            
            if len(idx_val) > 0:
                sub_test = system.sub_system(idx_val)
                NEPConverter.save_as_xyz(sub_test, os.path.join(self.work_dir, "test.xyz"))
                
        except Exception as e:
            print(f"❌ 帧拆分失败: {e}")
            raise e

    def _merge_files_physically(self, dir_list, output_name):
        out_file = os.path.join(self.work_dir, output_name)
        print(f"  -> 正在合并生成 {output_name}...")
        with open(out_file, 'w') as outfile:
            for d in dir_list:
                xyz_file = os.path.join(d, "train.xyz")
                if os.path.exists(xyz_file):
                    with open(xyz_file, 'r') as infile:
                        shutil.copyfileobj(infile, outfile)