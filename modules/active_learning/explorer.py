import os
import shutil
import config_active as cfg_al

class ALExplorer:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def prepare_exploration(self, trained_model_dirs, init_structure_path):
        """生成探索任务文件夹"""
        print(f"[AL-Explorer] 正在构建探索任务目录: {self.work_dir}")
        
        # 1. 链接势函数 (nep0.txt, nep1.txt ...)
        linked_models = self._link_potentials(trained_model_dirs)
        
        # 2. 准备初始结构 (model.xyz)
        target_model = os.path.join(self.work_dir, "model.xyz")
        if os.path.exists(target_model): os.remove(target_model)
        shutil.copy(init_structure_path, target_model)

        # 3. 生成 run.in
        self._write_run_in(linked_models)
        print(f"✅ 探索配置已生成。")

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
        return linked

    def _build_ensemble_line(self):
        """组装系综指令"""
        ec = cfg_al.ENSEMBLE_CONFIG
        method = ec['method']
        
        if method == "nvt_ber":
            return f"ensemble nvt_ber {ec['T_start']} {ec['T_end']} {ec['T_coupling']}"
        
        if method == "npt_mttk":
            return (f"ensemble npt_mttk temp {ec['T_start']} {ec['T_end']} "
                    f"{ec['p_direction']} {ec['p_start']} {ec['p_end']} "
                    f"tperiod {ec['T_coupling']} pperiod {ec['p_coupling']}")
        return f"ensemble {method}"

    def _write_run_in(self, models):
        md = cfg_al.EXPLORE_CONFIG
        act = cfg_al.ACTIVE_STRATEGY
        
        run_file = os.path.join(self.work_dir, "run.in")
        with open(run_file, 'w') as f:
            # 1. Potential
            for m in models:
                f.write(f"potential {m}\n")
            f.write("\n")
            
            # 2. Basic MD
            f.write(f"time_step {md['time_step']}\n")
            f.write(f"velocity {md['velocity_temp']}\n")
            
            # 3. Ensemble
            f.write(f"{self._build_ensemble_line()}\n")
            f.write("\n")
            
            # 4. Active Learning
            # active <interval> <has_v> <has_f> <has_u> <threshold>
            f.write(f"active {act['interval']} {act['has_velocity']} {act['has_force']} {act['has_uncertainty']} {act['threshold']}\n")
            f.write("\n")
            
            # 5. Output Control
            f.write(f"dump_thermo {md['dump_thermo']}\n")
            
            # --- 修正后的 dump_xyz 语法 ---
            # 语法: dump_xyz <grouping_method> <group_id> <interval> <filename> {properties}
            # -1 1 代表全系统输出
            interval = md['dump_xyz_interval']
            props = " ".join(md.get('dump_xyz_properties', []))
            f.write(f"dump_xyz -1 1 {interval} dump.xyz {props}\n")
            
            f.write(f"run {md['steps']}\n")