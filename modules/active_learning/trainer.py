import os
import shutil
import config as cfg
import config_train as cfg_tr
import config_active as cfg_al  

class ALTrainer:
    """
    主动学习训练准备模块 (仅 GPUMD)
    负责生成 N 个独立的训练目录，利用 NEP 的随机初始化特性构建系综。
    """
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.n_models = cfg_al.N_ENSEMBLE
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _get_type_map(self):
        sorted_species = sorted(cfg.SPECIES_MAP.items(), key=lambda x: x[0])
        return [item[1]['label'] for item in sorted_species]

    def prepare_ensemble(self, source_path):
        """
        准备系综训练目录
        :param source_path: 包含 train.xyz 的源数据目录
        """
        print(f"[AL-Trainer] 正在构建 NEP 系综训练目录 (N={self.n_models})...")
        
        # 1. 检查源数据
        source_xyz = os.path.join(source_path, "train.xyz")
        if not os.path.exists(source_xyz):
            raise FileNotFoundError(f"在 {source_path} 下未找到 train.xyz，无法训练。")

        # 2. 生成 nep.in 内容
        template = cfg_tr.GPUMD_NEP_TEMPLATE
        type_map = self._get_type_map()
        type_line = f"type          {len(type_map)} {' '.join(type_map)}"
        
        nep_content = [f"{type_line}"]
        for key, val in template.items():
            nep_content.append(f"{key:<13} {val}")
        nep_str = "\n".join(nep_content)

        # 3. 循环创建子目录
        for i in range(self.n_models):
            task_dir = os.path.join(self.work_dir, f"model_{i:02d}")
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            
            # (A) 复制数据
            shutil.copy(source_xyz, os.path.join(task_dir, "train.xyz"))
            
            # (B) 写入 nep.in
            with open(os.path.join(task_dir, "nep.in"), 'w') as f:
                f.write(nep_str)
            
            print(f"  -> Model {i} 准备就绪: {task_dir}")

        print(f"✅ 系综环境构建完成！请进入各子目录运行 'nep' 。")