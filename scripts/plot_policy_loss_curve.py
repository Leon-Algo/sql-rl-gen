import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_policy_loss_curve():
    # 定义文件路径
    base_dir = Path(__file__).resolve().parents[1]
    txt_path = base_dir / "output" / "model_spider_train" / "scores.txt"
    
    # 检查文件是否存在
    if not txt_path.exists():
        raise FileNotFoundError(f"scores.txt not found at {txt_path}")
    
    # 读取并解析TXT文件
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    
    # 解析表头
    header = lines[0].strip().split('\t')
    
    # 检查average_policy_loss列是否存在
    if 'average_policy_loss' not in header:
        raise ValueError("Column 'average_policy_loss' not found in the TXT file")
    
    # 找到average_policy_loss列的索引
    policy_loss_idx = header.index('average_policy_loss')
    
    # 解析数据
    policy_losses = []
    steps = []
    
    # 从第二行开始是数据行
    for line in lines[1:]:
        values = line.strip().split('\t')
        if len(values) > policy_loss_idx:
            try:
                policy_loss = float(values[policy_loss_idx])
                policy_losses.append(policy_loss)
                
                # 同时获取steps用于x轴
                steps.append(int(values[0]))  # steps是第一列
            except ValueError:
                # 跳过无法解析的行
                continue
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(steps, policy_losses, linewidth=1, color='green')
    plt.title('Policy Loss Curve During Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Policy Loss')
    plt.grid(True, alpha=0.3)
    
    # 添加移动平均线以更好地观察趋势
    window_size = 50
    if len(policy_losses) >= window_size:
        moving_avg = pd.Series(policy_losses).rolling(window=window_size).mean()
        plt.plot(steps, moving_avg, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')
        plt.legend()
    
    # 保存图表
    output_path = base_dir / "output" / "policy_loss_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Policy loss curve saved to: {output_path}")
    
    # 输出一些统计信息
    print(f"Total data points: {len(policy_losses)}")
    print(f"Average policy loss: {np.mean(policy_losses):.6f}")
    print(f"Min policy loss: {np.min(policy_losses):.6f}")
    print(f"Max policy loss: {np.max(policy_losses):.6f}")
    print(f"Final policy loss: {policy_losses[-1]:.6f}")

if __name__ == "__main__":
    plot_policy_loss_curve()