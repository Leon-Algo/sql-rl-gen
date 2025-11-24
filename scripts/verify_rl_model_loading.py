#!/usr/bin/env python3
"""
验证RL模型权重是否可以正确加载和应用的脚本
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sql_rl_gen.generation.envs.utils import find_device, prepare_observation_list_and_dataset_to_pass
from sql_rl_gen.generation.envs.sql_generation_environment import SQLRLEnv
from sql_rl_gen.generation.rllib.custom_actor import CustomActor
from data_preprocess.data_utils import get_dataset
from configs.data_args import DataArguments, DatasetName
from configs.config import SPIDER_DATABASES_PATH, WIKISQL_PATH, BIRD_DATABASES_DEV_PATH, ROOT_PATH


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


def get_dataset_path(dataset_name):
    """根据数据集名称获取对应的路径"""
    if dataset_name == DatasetName.SPIDER.value:
        return SPIDER_DATABASES_PATH
    elif dataset_name == DatasetName.WIKISQL.value:
        return WIKISQL_PATH
    else:
        return BIRD_DATABASES_DEV_PATH


def main():
    parser = argparse.ArgumentParser(description="验证RL模型权重是否可以正确加载和应用")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="基础模型名称或路径")
    parser.add_argument("--trained_agent_path", type=str, required=True, help="训练好的RL agent路径")
    parser.add_argument("--dataset", type=str, default="example_text2sql_spider_dev", help="数据集名称")
    parser.add_argument("--dataset_name", type=str, default="spider", help="数据集类型名称")
    parser.add_argument("--template", type=str, default="llama3", help="模板名称")
    parser.add_argument("--max_samples", type=int, default=5, help="使用的样本数量")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # 获取项目根目录
    base_dir = Path(__file__).resolve().parents[1]
    trained_agent_path = base_dir / args.trained_agent_path
    
    if not trained_agent_path.exists():
        logger.error(f"训练好的模型路径不存在: {trained_agent_path}")
        return False
    
    logger.info(f"验证模型加载情况: {trained_agent_path}")
    
    # 准备数据参数
    data_args = DataArguments(
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        dataset_dir=f"{ROOT_PATH}/data_preprocess/data/",
        template=args.template,
        max_samples=args.max_samples
    )
    data_args.init_for_training()
    
    # 加载数据集
    dataset = get_dataset(data_args)
    dataset = dataset.take(args.max_samples)
    
    # 获取数据集路径
    dataset_path = get_dataset_path(args.dataset_name)
    
    # 准备观察列表和数据传递列表
    observation_list, data_list_to_pass, columns_names_mismatch = prepare_observation_list_and_dataset_to_pass(dataset)
    
    # 加载模型和tokenizer
    device = find_device()
    logger.info(f"使用设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    
    # 获取模型参数的初始状态（前几个参数值）
    logger.info("模型参数初始状态:")
    initial_params = {}
    param_count = 0
    for name, param in model.named_parameters():
        if param_count >= 3:  # 只显示前3个参数
            break
        initial_params[name] = param.data.clone()
        logger.info(f"  {name}: {param.data.flatten()[:5]}")  # 显示前5个值
        param_count += 1
    
    # 创建RL环境和actor
    logger.info("创建RL环境和actor...")
    env = SQLRLEnv(
        model=model,
        tokenizer=tokenizer,
        dataset=data_list_to_pass,
        dataset_path=dataset_path,
        output_dir=str(trained_agent_path),
        logger=logger,
        environment_name="verification",
        columns_names_mismatch=columns_names_mismatch,
        observation_input=observation_list,
        compare_sample=1,
    )
    
    actor = CustomActor(
        env=env,
        model=model,
        tokenizer=tokenizer,
        temperature=0.0,
        top_k=0,
        top_p=1.0
    )
    
    # 创建PPO agent
    agent = actor.agent_ppo(
        update_interval=1,
        minibatch_size=1,
        epochs=1,
        lr=1e-5
    )
    
    # 检查加载前的agent状态
    logger.info("加载模型前的agent状态检查")
    
    # 加载训练好的模型
    logger.info(f"正在加载训练好的模型: {trained_agent_path}")
    try:
        agent.load(str(trained_agent_path))
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False
    
    # 检查加载后的模型参数
    logger.info("模型参数加载后状态:")
    params_changed = False
    param_count = 0
    for name, param in model.named_parameters():
        if param_count >= 3:  # 只检查前3个参数
            break
        initial_param = initial_params[name]
        current_param = param.data
        logger.info(f"  {name}: {current_param.flatten()[:5]}")  # 显示前5个值
        
        # 检查参数是否发生变化
        if not torch.allclose(initial_param, current_param, atol=1e-6):
            logger.info(f"  参数 {name} 发生变化")
            params_changed = True
        param_count += 1
    
    if params_changed:
        logger.info("模型参数已成功更新")
    else:
        logger.warning("模型参数未发生变化，可能未正确加载权重")
    
    # 测试预测功能
    if observation_list:
        logger.info("测试预测功能...")
        try:
            test_observation = observation_list[0]
            logger.info(f"测试输入: {test_observation['input'][:100]}...")
            
            # 基础模型预测
            with torch.no_grad():
                input_ids = tokenizer(
                    test_observation["input"],
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True
                ).input_ids.to(device)
                
                base_model_output = model.generate(
                    input_ids,
                    max_length=120,
                    min_length=10,
                )
                base_result = tokenizer.decode(base_model_output[0], skip_special_tokens=True)
                logger.info(f"基础模型输出: {base_result}")
            
            # RL模型预测
            rl_result = actor.predict(test_observation)[0][:-4]  # 移除结尾的特殊token
            logger.info(f"RL模型输出: {rl_result}")
            
            if base_result != rl_result:
                logger.info("基础模型和RL模型输出不同，模型加载成功")
                return True
            else:
                logger.warning("基础模型和RL模型输出相同，可能存在问题")
                return False
                
        except Exception as e:
            logger.error(f"预测测试失败: {e}")
            return False
    else:
        logger.warning("没有可用的测试数据")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)