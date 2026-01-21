# HAP_MLP_Project/modules/scheduler.py
import os
import subprocess
import shutil
import glob  

class TaskScheduler:
    def __init__(self, workspace_root, submit_cmd="sbatch"):
        self.workspace_root = workspace_root
        self.submit_cmd = submit_cmd
        if not os.path.exists(self.workspace_root):
            os.makedirs(self.workspace_root)

    def generate_job_script(self, task_dir, task_name, num_cores=40):
        # 请根据你的集群实际情况调整这里
        script_content = f"""#!/bin/bash
#SBATCH --job-name={task_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_cores}
#SBATCH --partition=batch
#SBATCH -n {num_cores}
#SBATCH --nodelist=compute-0-[5-7,14-15]

echo "Job started at $(date)"
echo "Work dir: {task_dir}"

source /opt/cluster_share/profile.d/honpas
# 进入工作目录
mpirun -np $SLURM_NTASKS honpas < INPUT.fdf > output.log

echo "Job finished at $(date)"
"""
        script_path = os.path.join(task_dir, "run.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        return script_path

    def _copy_pseudopotentials(self, target_dir, psf_source_dir):
        """
        内部方法：将源目录下的所有 .psf 文件复制到任务目录
        """
        # 查找所有 .psf 文件
        psf_files = glob.glob(os.path.join(psf_source_dir, "*.psf"))
        
        if not psf_files:
            print(f"  [Warning] 在 {psf_source_dir} 未找到任何 .psf 文件！")
            return

        for psf in psf_files:
            shutil.copy(psf, target_dir)

    def setup_task(self, task_id, wrapper, frame_data, psf_dir=None): # <--- 新增参数 psf_dir
        """
        建立单个任务
        """
        task_name = f"task_{task_id:04d}"
        task_dir = os.path.join(self.workspace_root, task_name)
        
        # 1. 生成 INPUT.fdf
        wrapper.write_input(frame_data, task_dir, label=f"HAP_{task_id}")
        
        # 2. 生成 run.sh
        self.generate_job_script(task_dir, task_name)

        # 3. --- 新增：复制赝势文件 ---
        if psf_dir and os.path.exists(psf_dir):
            self._copy_pseudopotentials(task_dir, psf_dir)
        
        return task_dir

    def submit_all(self, dry_run=True):
        tasks = sorted([d for d in os.listdir(self.workspace_root) if d.startswith("task_")])
        print(f"[Scheduler] 发现 {len(tasks)} 个任务目录。")

        for t in tasks:
            t_path = os.path.join(self.workspace_root, t)
            script_path = "run.sh"
            
            if dry_run:
                print(f"  [DryRun] Would execute: cd {t_path} && {self.submit_cmd} {script_path}")
            else:
                try:
                    subprocess.run([self.submit_cmd, script_path], cwd=t_path, check=True)
                    print(f"  [Submitted] {t}")
                except subprocess.CalledProcessError as e:
                    print(f"  [Error] 提交失败 {t}: {e}")