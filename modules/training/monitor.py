import os
import numpy as np
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            raise FileNotFoundError(f"找不到训练目录: {self.work_dir}")

    def plot_deepmd_lcurve(self, filename="lcurve.out"):
        """
        DeepMD 训练曲线绘制
        """
        log_path = os.path.join(self.work_dir, filename)
        if not os.path.exists(log_path):
            print(f"⚠️ 未找到 {filename}，跳过绘图。")
            return

        print(f"[Monitor] 正在绘制 DeepMD 训练曲线: {log_path}")
        try:
            # 读取数据
            data = np.genfromtxt(log_path, names=True)
            
            plt.figure(figsize=(8, 6))
            # 绘制除去 step 之外的所有列
            for name in data.dtype.names[1:-1]:
                plt.plot(data['step'], data[name], label=name)
            
            plt.legend()
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.xscale('symlog') # 对数坐标
            plt.yscale('log')
            plt.title(f"DeepMD Training Log ({os.path.basename(self.work_dir)})")
            
            save_path = os.path.join(self.work_dir, "monitor_lcurve.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ DeepMD 曲线已保存: {save_path}")
        except Exception as e:
            print(f"❌ 绘图失败: {e}")

    def plot_gpumd_loss(self, filename="loss.out"):
        """
        GPUMD 训练曲线绘制 
        """
        log_path = os.path.join(self.work_dir, filename)
        if not os.path.exists(log_path):
            print(f"⚠️ 未找到 {filename}，跳过绘图。")
            return

        print(f"[Monitor] 正在绘制 GPUMD 训练曲线: {log_path}")
        try:
            loss = np.loadtxt(log_path)
            
            plt.figure(figsize=(8, 6))
            
            # 判断坐标轴
            if loss.shape[0] > 1 and loss[1, 0] - loss[0, 0] == 100:
                xlabel = 'Generation/100'
                # loss columns: Total, L1, L2, E-train, F-train, V-train
                plot_cols = range(1, min(7, loss.shape[1])) 
                labels = ['Total', 'L1-Reg', 'L2-Reg', 'Energy', 'Force', 'Virial']
            else:
                xlabel = 'Epoch'
                plot_cols = range(1, min(5, loss.shape[1]))
                labels = ['Total', 'Energy', 'Force', 'Virial']

            # 绘图
            for i, col_idx in enumerate(plot_cols):
                label_name = labels[i] if i < len(labels) else f"Col_{col_idx}"
                plt.loglog(loss[:, col_idx], '-', linewidth=2, label=label_name)

            plt.xlabel(xlabel)
            plt.ylabel('Loss functions')
            plt.legend()
            plt.title(f"GPUMD Training Log ({os.path.basename(self.work_dir)})")
            
            save_path = os.path.join(self.work_dir, "monitor_loss.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ GPUMD 曲线已保存: {save_path}")
        except Exception as e:
            print(f"❌ 绘图失败: {e}")