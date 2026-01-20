# HAP_MLP_Project/modules/extractor.py
import os
import glob
import dpdata

class ResultExtractor:
    def __init__(self, workspace_root):
        self.workspace_root = workspace_root

    def collect_data(self, mode="scf"): # 增加 mode 参数，默认为 scf
        """
        遍历目录，读取 output.log，合并为一个 LabeledSystem
        """
        # 根据 mode 决定 dpdata 的解析格式
        # SCF 模式用 siesta/output, AIMD 模式用 siesta/aimd_output
        data_fmt = 'siesta/aimd_output' if mode.lower() == 'aimd' else 'siesta/output'
        
        print(f"[Extractor] 当前模式: {mode.upper()}, 使用格式: {data_fmt}")

        # 1. 找到所有 output.log
        search_pattern = os.path.join(self.workspace_root, "task_*", "output.log")
        files = sorted(glob.glob(search_pattern))
        
        print(f"[Extractor] 在 {self.workspace_root} 中找到 {len(files)} 个输出文件。")
        
        valid_systems = []
        
        # 2. 逐个读取
        for f in files:
            try:
                # 使用动态确定的 data_fmt
                # 注意：siesta/aimd_output 要求 output.log 旁边必须有 .ANI 或 .FA 文件
                ls = dpdata.LabeledSystem(f, fmt=data_fmt)
                
                if len(ls) > 0:
                    valid_systems.append(ls)
                    # 如果是 AIMD 模式，打印一下每个任务抓到了多少帧
                    if mode.lower() == 'aimd':
                        print(f"  [OK] 任务 {os.path.basename(os.path.dirname(f))} 提取到 {len(ls)} 帧数据")
                else:
                    print(f"  [Warn] 文件为空或无帧数据: {f}")
                    
            except Exception as e:
                print(f"  [Error] 解析失败 {f}: {e}")

        if not valid_systems:
            print("[Extractor] 没有收集到任何有效数据！")
            return None

        # 3. 合并数据
        print(f"[Extractor] 正在合并 {len(valid_systems)} 个系统的结果...")
        merged_system = valid_systems[0]
        for s in valid_systems[1:]:
            try:
                merged_system.append(s)
            except Exception as e:
                print(f"  [Error] 合并时出错 (可能是原子数量/类型不一致): {e}")

        print(f"[Extractor] 合并完成！总帧数: {len(merged_system)}")
        return merged_system