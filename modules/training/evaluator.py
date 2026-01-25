import os
import subprocess
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt

# 设置全局样式
try:
    plt.style.use('seaborn-v0_8-poster')
except:
    try:
        plt.style.use('seaborn-poster')
    except:
        pass

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.8,
})

class ModelEvaluator:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            raise FileNotFoundError(f"找不到工作目录: {self.work_dir}")

    # ========================== DeepMD 评估流程 ==========================
    def eval_deepmd(self, test_target="data_train"):
        print(f"[Evaluator] 开始评估 DeepMD 模型: {self.work_dir}")
        
        # 1. 冻结
        if not self._dp_freeze(): return
        
        # 2. 压缩
        if not self._dp_compress(): return
        
        # 3. 测试
        target_path = os.path.join(self.work_dir, test_target)
        if not os.path.exists(target_path):
            print(f"⚠️ 找不到测试目标: {target_path}，尝试使用 data_val...")
            target_path = os.path.join(self.work_dir, "data_val")
            if not os.path.exists(target_path):
                print("❌ 无法找到任何数据文件夹，无法测试。")
                return

        self._dp_test(target_path)
        
        # 4. 绘图
        self._plot_deepmd_custom()

    def _dp_freeze(self):
        print("  -> [1/4] 正在冻结模型 (Freeze)...")
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        try:
            if not os.path.exists("model.ckpt.index"):
                print("❌ 找不到 model.ckpt，训练可能未开始或失败。")
                return False
            subprocess.run(["dp", "freeze", "-o", "graph.pb"], check=True)
            print("     ✅ graph.pb 已生成")
            return True
        except Exception as e:
            print(f"     ❌ 冻结失败: {e}")
            return False
        finally:
            os.chdir(cwd)

    def _dp_compress(self):
        print("  -> [2/4] 正在压缩模型 (Compress)...")
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        try:
            cmd = ["dp", "compress", "-i", "graph.pb", "-o", "graph-compress.pb"]
            subprocess.run(cmd, check=True)
            print("     ✅ graph-compress.pb 已生成")
            return True
        except Exception as e:
            print(f"     ⚠️ 压缩失败，尝试使用未压缩模型测试。")
            import shutil
            if os.path.exists("graph.pb"):
                shutil.copy("graph.pb", "graph-compress.pb")
                return True
            return False
        finally:
            os.chdir(cwd)

    def _dp_test(self, data_path):
        print(f"  -> [3/4] 正在执行推断 (Test on {os.path.basename(data_path)})...")
        
        eval_subdir_name = "eval_results"
        eval_out_dir = os.path.join(self.work_dir, eval_subdir_name)
        
        if os.path.exists(eval_out_dir):
            shutil.rmtree(eval_out_dir)
        os.makedirs(eval_out_dir)
            
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        try:
            model_file = "graph-compress.pb"
            # 使用 -d result 生成到当前目录
            cmd = ["dp", "test", "-m", model_file, "-s", data_path, "-d", "result"]
            subprocess.run(cmd, check=True)
            
            # 移动文件到 eval_results
            generated_files = glob.glob("result.*.out")
            if not generated_files:
                print("     ⚠️ 未检测到测试输出文件！")
            
            for f in generated_files:
                shutil.move(f, os.path.join(eval_subdir_name, f))
                
            print(f"     ✅ 测试完成，结果已移动至: {eval_subdir_name}/")
            
        except Exception as e:
            print(f"     ❌ 测试运行失败: {e}")
        finally:
            os.chdir(cwd)

    def _plot_deepmd_custom(self):
        """绘图逻辑 (全量数据绘制，无采样)"""
        eval_dir = os.path.join(self.work_dir, "eval_results")
        e_file = os.path.join(eval_dir, "result.e_peratom.out")
        f_file = os.path.join(eval_dir, "result.f.out")

        if not os.path.exists(e_file) or not os.path.exists(f_file):
            print(f"⚠️ 找不到测试结果文件 ({e_file})，可能测试失败，跳过绘图。")
            return

        print("  -> [4/4] 正在绘制评估报告...")

        def read_data(file_path):
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    parts = line.split()
                    data.append([float(x) for x in parts])
            return np.array(data)

        def calculate_rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        try:
            # 读取
            e_data = read_data(e_file)
            f_data = read_data(f_file)

            # 计算 RMSE
            e_rmse = calculate_rmse(e_data[:, 0], e_data[:, 1]) * 1000  # meV
            total_f_rmse = calculate_rmse(f_data[:, :3].flatten(), f_data[:, 3:].flatten()) * 1000 # meV/A

            # 绘图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
            fig.suptitle('DeepMD Model Performance Evaluation', y=1.02, fontsize=16)

            # Energy
            ax1.scatter(e_data[:, 0], e_data[:, 1], alpha=0.7, 
                        c='#1f77b4', edgecolors='none', s=40, label='Data points')
            min_e, max_e = np.min(e_data), np.max(e_data)
            margin = (max_e - min_e) * 0.05
            ax1.plot([min_e-margin, max_e+margin], [min_e-margin, max_e+margin], '--', 
                     color='#ff7f0e', linewidth=2, label='Ideal')
            
            rmse_text = f'RMSE = {e_rmse:.2f} meV/atom'
            ax1.text(0.05, 0.95, rmse_text, transform=ax1.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#dddddd'))
            ax1.set_xlabel('DFT Energy (eV/atom)', fontweight='bold')
            ax1.set_ylabel('DP Energy (eV/atom)', fontweight='bold')
            ax1.set_title('Energy Comparison', pad=12)
            ax1.legend(loc='lower right', framealpha=0.9)
            ax1.grid(True, linestyle='--', alpha=0.6)

            # Force - 移除采样，绘制所有点
            colors = ['#d62728', '#2ca02c', '#9467bd']
            labels = ['Fx', 'Fy', 'Fz']
            for i, (c, l) in enumerate(zip(colors, labels)):
                ax2.scatter(f_data[:, i], f_data[:, i+3], alpha=0.5, 
                        color=c, edgecolors='none', s=30, label=l)

            min_f, max_f = np.min(f_data[:, :3]), np.max(f_data[:, :3])
            margin_f = (max_f - min_f) * 0.05
            ax2.plot([min_f-margin_f, max_f+margin_f], [min_f-margin_f, max_f+margin_f], '--', 
                     color='#ff7f0e', linewidth=2, label='Ideal')

            rmse_text = f'Total RMSE = {total_f_rmse:.2f} meV/Å'
            ax2.text(0.05, 0.95, rmse_text, transform=ax2.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#dddddd'))
            ax2.set_xlabel('DFT Force (eV/Å)', fontweight='bold')
            ax2.set_ylabel('DP Force (eV/Å)', fontweight='bold')
            ax2.set_title('Force Components Comparison', pad=12)
            ax2.legend(loc='lower right', framealpha=0.9)
            ax2.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.25)

            save_path = os.path.join(self.work_dir, 'deepmd_results.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"     ✅ 评估报告已生成: {save_path}")

        except Exception as e:
            print(f"     ❌ 绘图失败: {e}")
            import traceback
            traceback.print_exc()

    # ========================== GPUMD 评估流程 ==========================
    def eval_gpumd(self):
        print(f"[Evaluator] 开始评估 GPUMD 模型: {self.work_dir}")
        self._plot_gpumd_report()

    def _plot_gpumd_report(self):
        """
        整合了你提供的 GPUMD 绘图脚本 (4合1图)
        """
        print("  -> 正在绘制 GPUMD 综合评估报告...")
        cwd = os.getcwd()
        os.chdir(self.work_dir)
        
        try:
            # 检查文件是否存在
            required = ['loss.out', 'energy_train.out', 'force_train.out', 'stress_train.out']
            if not all(os.path.exists(f) for f in required):
                print(f"     ⚠️ 缺少必要的输出文件 ({required})，跳过绘图。")
                return

            # --- 以下是你提供的绘图代码逻辑 ---
            loss = np.loadtxt('loss.out')
            energy_data = np.loadtxt('energy_train.out')
            force_data = np.loadtxt('force_train.out')
            stress_data = np.loadtxt('stress_train.out')

            # Filter stress
            valid_rows = ~np.any(np.abs(stress_data[:, :12]) > 1e6, axis=1)
            stress_data = stress_data[valid_rows]

            def calculate_rmse(pred, actual):
                return np.sqrt(np.mean((pred - actual) ** 2))

            def calculate_limits(train_data, padding=0.08):
                data_min = np.min(train_data)
                data_max = np.max(train_data)
                data_range = data_max - data_min
                return data_min - padding * data_range, data_max + padding * data_range

            fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=150)

            # 1. Loss
            if loss.shape[0] > 1 and loss[1, 0] - loss[0, 0] == 100:
                xlabel = 'Generation/100'
                plot_cols = slice(1, 7)
                legend_labels = ['Total', 'L1', 'L2', 'E-train', 'F-train', 'V-train']
            else:
                xlabel = 'Epoch'
                plot_cols = slice(1, 5)
                legend_labels = ['Total', 'Energy', 'Force', 'Virial']
            
            # 安全处理：确保 plot_cols 不越界
            max_col = loss.shape[1]
            if plot_cols.stop > max_col: plot_cols = slice(1, max_col)
            
            axs[0, 0].loglog(loss[:, plot_cols], '-', linewidth=2)
            axs[0, 0].set_xlabel(xlabel)
            axs[0, 0].set_ylabel('Loss functions')
            axs[0, 0].legend(legend_labels[:loss.shape[1]-1], prop={'size': 8})

            # 2. Energy Parity
            xmin, xmax = calculate_limits(energy_data[:, 1])
            axs[0, 1].plot(energy_data[:, 1], energy_data[:, 0], '.', markersize=5)
            axs[0, 1].plot([xmin, xmax], [xmin, xmax], 'k--')
            e_rmse = calculate_rmse(energy_data[:, 0], energy_data[:, 1]) * 1000
            axs[0, 1].set_xlabel('DFT Energy (eV/atom)')
            axs[0, 1].set_ylabel('NEP Energy (eV/atom)')
            axs[0, 1].text(0.05, 0.9, f'RMSE: {e_rmse:.2f} meV/atom', transform=axs[0, 1].transAxes)

            # 3. Force Parity
            # force_train.out: [fx_nep, fy_nep, fz_nep, fx_dft, fy_dft, fz_dft]
            f_nep = force_data[:, 0:3].flatten()
            f_dft = force_data[:, 3:6].flatten()
            xmin, xmax = calculate_limits(f_dft)
            axs[1, 0].plot(f_dft, f_nep, '.', markersize=5, alpha=0.5)
            axs[1, 0].plot([xmin, xmax], [xmin, xmax], 'k--')
            f_rmse = calculate_rmse(f_nep, f_dft)
            axs[1, 0].set_xlabel('DFT Force (eV/A)')
            axs[1, 0].set_ylabel('NEP Force (eV/A)')
            axs[1, 0].text(0.05, 0.9, f'RMSE: {f_rmse:.3f} eV/A', transform=axs[1, 0].transAxes)

            # 4. Stress Parity
            # stress_train.out: [xx_nep...zx_nep, xx_dft...zx_dft]
            s_nep = stress_data[:, 0:6].flatten()
            s_dft = stress_data[:, 6:12].flatten()
            xmin, xmax = calculate_limits(s_dft)
            axs[1, 1].plot(s_dft, s_nep, '.', markersize=5, alpha=0.5)
            axs[1, 1].plot([xmin, xmax], [xmin, xmax], 'k--')
            s_rmse = calculate_rmse(s_nep, s_dft)
            axs[1, 1].set_xlabel('DFT Stress (GPa)')
            axs[1, 1].set_ylabel('NEP Stress (GPa)')
            axs[1, 1].text(0.05, 0.9, f'RMSE: {s_rmse:.3f} GPa', transform=axs[1, 1].transAxes)

            plt.tight_layout()
            plt.savefig("eval_report_gpumd.png")
            print(f"     ✅ 综合评估报告已保存: {os.path.join(self.work_dir, 'eval_report_gpumd.png')}")

        except Exception as e:
            print(f"     ❌ GPUMD 绘图失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            os.chdir(cwd)