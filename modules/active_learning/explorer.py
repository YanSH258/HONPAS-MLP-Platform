import os
import shutil
import glob
import config_active as cfg_al

class ALExplorer:
    """
    主动学习探索模块 (仅 GPUMD)
    负责生成带有 active 关键字的 run.in，并链接系综势函数。
    """
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def prepare_exploration(self, trained_model_dirs, init_structure_path):
        """
        :param trained_model_dirs: 包含 nep.txt 的模型目录列表
        :param init_structure_path: 用户的初始结构文件 (model.xyz)
        """
        print(f"[AL-Explorer] 正在构建探索任务...")
        
        # 1. 链接势函数 (nep0.txt, nep1.txt ...)
        linked_models = self._link_potentials(trained_model_dirs)
        if len(linked_models) < 2:
            print("❌ 错误: 有效模型少于 2 个，无法进行系综不确定度计算。")
            return

        # 2. 准备初始结构 (model.xyz)
        if not os.path.exists(init_structure_path):
            raise FileNotFoundError(f"初始结构不存在: {init_structure_path}")
        
        # 复制到工作目录
        target_model = os.path.join(self.work_dir, "model.xyz")
        shutil.copy(init_structure_path, target_model)
        print(f"  -> 初始结构已就位: {os.path.basename(init_structure_path)}")

        # 3. 生成 run.in
        self._write_run_in(linked_models)
        
        print(f"✅ 探索目录构建完成: {self.work_dir}")
        print(f"   请运行 'nep' (GPUMD) 开始探索。")

    def _link_potentials(self, model_dirs):
        linked = []
        count = 0
        for md in model_dirs:
            src = os.path.join(md, "nep.txt")
            if os.path.exists(src):
                dst_name = f"nep{count}.txt"
                dst_path = os.path.join(self.work_dir, dst_name)
                if os.path.exists(dst_path): os.remove(dst_path)
                os.symlink(os.path.abspath(src), dst_path)
                linked.append(dst_name)
                count += 1
            else:
                print(f"⚠️ 警告: {md} 下未找到 nep.txt")
        return linked

    def _write_run_in(self, models):
        conf = cfg_al.EXPLORE_CONFIG
        act = cfg_al.ACTIVE_STRATEGY
        
        run_file = os.path.join(self.work_dir, "run.in")
        with open(run_file, 'w') as f:
            # --- Potentials ---
            for m in models:
                f.write(f"potential {m}\n")
            f.write("\n")
            
            # --- MD Settings (from config_active) ---
            f.write(f"velocity {conf['velocity_temp']}\n")
            f.write(f"time_step {conf['time_step']}\n")
            f.write(f"ensemble {conf['ensemble_str']}\n")
            
            # --- Active Learning Strategy ---
            # active <interval> <v> <f> <u> <thresh>
            # 这里的 0 0 代表不保存速度和力到 active.xyz，只保存结构
            cmd = f"active {act['interval']} 0 0 {act['has_uncertainty']} {act['threshold']}"
            f.write(f"{cmd}\n")
            
            # --- Run ---
            # 输出轨迹供参考
            f.write(f"dump_exyz 1000 trajectory.xyz\n")
            f.write(f"run {conf['steps']}\n")