import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_reward_curve():
    # 定义文件路径
    base_dir = Path(__file__).resolve().parents[1]
    csv_path = base_dir / "output" / "model_spider_train" / "training_metrics.csv"
    
    # 检查文件是否存在
    if not csv_path.exists():
        raise FileNotFoundError(f"training_metrics.csv not found at {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查reward列是否存在
    if 'reward' not in df.columns:
        raise ValueError("Column 'reward' not found in the CSV file")
    
    # 获取reward数据
    rewards = df['reward'].values
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, linewidth=0.5, color='blue', alpha=0.7)
    plt.title('Reward Curve During Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # 添加移动平均线以更好地观察趋势
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        plt.plot(moving_avg, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')
        plt.legend()
    
    # 保存图表
    output_path = base_dir / "output" / "reward_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reward curve saved to: {output_path}")
    
    # 输出一些统计信息
    print(f"Total training steps: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Max reward: {np.max(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}")
    print(f"Final reward: {rewards[-1]:.4f}")

if __name__ == "__main__":
    plot_reward_curve()