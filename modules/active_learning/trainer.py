import os
import shutil
import config as cfg
import config_train as cfg_tr
import config_active as cfg_al

class ALTrainer:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.n_models = cfg_al.N_ENSEMBLE
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _get_type_map(self):
        sorted_species = sorted(cfg.SPECIES_MAP.items(), key=lambda x: x[0])
        return [item[1]['label'] for item in sorted_species]

    def prepare_ensemble(self, source_path):
        """仅生成目录和输入文件，不生成提交脚本"""
        print(f"[AL-Trainer] 构建 GPUMD 系综训练目录 (N={self.n_models})...")
        
        source_xyz = os.path.join(source_path, "train.xyz")
        if not os.path.exists(source_xyz):
            raise FileNotFoundError(f"未找到合并的 train.xyz: {source_xyz}")

        # 构造 nep.in
        template = cfg_tr.GPUMD_NEP_TEMPLATE
        type_map = self._get_type_map()
        type_line = f"type          {len(type_map)} {' '.join(type_map)}"
        
        nep_lines = [type_line]
        for key, val in template.items():
            nep_lines.append(f"{key:<13} {val}")
        nep_str = "\n".join(nep_lines)

        for i in range(self.n_models):
            task_dir = os.path.join(self.work_dir, f"model_{i:02d}")
            os.makedirs(task_dir, exist_ok=True)
            
            # 复制数据
            shutil.copy(source_xyz, os.path.join(task_dir, "train.xyz"))
            # 写入配置
            with open(os.path.join(task_dir, "nep.in"), 'w') as f:
                f.write(nep_str)
            
            print(f"  -> {task_dir} 已创建。")