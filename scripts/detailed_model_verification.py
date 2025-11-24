#!/usr/bin/env python3
"""
详细验证RL模型权重加载情况的脚本
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import torch

from sql_rl_gen.generation.envs.utils import find_device
from configs.config import ROOT_PATH


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_model_files(model_path):
    """检查模型文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"检查模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False
        
    files = os.listdir(model_path)
    logger.info(f"模型目录中的文件: {files}")
    
    model_file = os.path.join(model_path, "model.pt")
    if not os.path.exists(model_file):
        logger.error(f"模型文件不存在: {model_file}")
        return False
        
    # 检查文件大小
    file_size = os.path.getsize(model_file)
    logger.info(f"模型文件大小: {file_size} 字节 ({file_size / 1024 / 1024:.2f} MB)")
    
    # 尝试加载模型文件
    try:
        checkpoint = torch.load(model_file, map_location='cpu')
        logger.info(f"模型检查点类型: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            logger.info(f"检查点键值: {list(checkpoint.keys())}")
        else:
            logger.info(f"检查点内容类型: {type(checkpoint)}")
    except Exception as e:
        logger.error(f"加载模型文件失败: {e}")
        return False
        
    return True


def compare_model_states(model_path):
    """比较模型状态"""
    logger = logging.getLogger(__name__)
    logger.info("比较模型状态...")
    
    model_file = os.path.join(model_path, "model.pt")
    try:
        checkpoint = torch.load(model_file, map_location='cpu')
        logger.info("模型检查点加载成功")
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("找到model_state_dict")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info("找到state_dict")
            else:
                state_dict = checkpoint
                logger.info("使用整个检查点作为状态字典")
        else:
            state_dict = checkpoint
            logger.info("检查点本身就是状态字典")
            
        logger.info(f"状态字典键值数量: {len(state_dict.keys())}")
        logger.info(f"前几个键值: {list(state_dict.keys())[:5]}")
        
        # 显示一些参数的值
        for i, (key, value) in enumerate(state_dict.items()):
            if i >= 3:  # 只显示前3个参数
                break
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if value.numel() > 0:
                logger.info(f"    前5个值: {value.flatten()[:5]}")
                
    except Exception as e:
        logger.error(f"比较模型状态失败: {e}")
        return False
        
    return True


def main():
    parser = argparse.ArgumentParser(description="详细验证RL模型权重加载情况")
    parser.add_argument("--trained_agent_path", type=str, required=True, help="训练好的RL agent路径")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # 获取项目根目录
    base_dir = Path(__file__).resolve().parents[1]
    trained_agent_path = base_dir / args.trained_agent_path
    
    logger.info("="*50)
    logger.info("开始详细模型验证")
    logger.info("="*50)
    
    # 检查模型文件
    if not check_model_files(str(trained_agent_path)):
        logger.error("模型文件检查失败")
        return False
        
    # 比较模型状态
    if not compare_model_states(str(trained_agent_path)):
        logger.error("模型状态比较失败")
        return False
        
    logger.info("="*50)
    logger.info("模型验证完成")
    logger.info("="*50)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)