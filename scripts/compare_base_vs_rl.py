import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sql_rl_gen.generation.envs.utils import (
    find_device,
    prepare_observation_list_and_dataset_to_pass,
)
from sql_rl_gen.generation.evaluate_model import load_trained_agent


def load_base_model(model_name_or_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return tokenizer, model


def generate_sql_base(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()


def generate_sql_rl(trained_agent_path: Path, observation: str) -> str:
    """使用已训练的 RL agent 对单条 observation 生成 SQL 文本。

    这里复用 evaluate_model.py 中的 load_trained_agent 封装，
    假设该函数返回的 agent 提供 act(observation) 接口并输出文本。"""

    agent = load_trained_agent(str(trained_agent_path))
    # agent.act 可能返回的是序列/张量，这里假设它直接返回文本或单元素列表
    out = agent.act(observation)
    if isinstance(out, (list, tuple)) and out:
        return str(out[0]).strip()
    return str(out).strip()


def main():
    parser = argparse.ArgumentParser(description="对比基模 vs 强化学习后模型在同一批问题上的 SQL 输出")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="基座模型名称或路径")
    parser.add_argument("--trained_agent_path", type=str, required=True, help="RL 训练后 agent 的路径（best 模型目录）")
    parser.add_argument("--input_questions", type=str, required=True, help="包含自然语言问题的文本文件，每行一条")
    parser.add_argument("--template", type=str, default="llama3", help="prompt 模板名称，与训练时保持一致")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    device = find_device()

    # 加载基座模型
    tokenizer, base_model = load_base_model(args.model_name_or_path, device)

    # 载入问题列表
    questions_path = Path(args.input_questions)
    if not questions_path.exists():
        raise FileNotFoundError(f"input_questions file not found: {questions_path}")

    questions: List[str] = [
        line.strip() for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    # 这里为了简单，使用 question 本身作为 observation；
    # 在更严谨的设置中，应当复用 utils.py 中的模板构造 schema+question 的 observation。
    trained_agent_dir = base_dir / args.trained_agent_path

    print("| 问题 | 基座模型 SQL | RL 后模型 SQL |")
    print("| ---- | ------------ | ------------- |")

    for q in questions:
        obs = q
        sql_base = generate_sql_base(base_model, tokenizer, obs, device)
        sql_rl = generate_sql_rl(trained_agent_dir, obs)
        # 为避免表格过长，这里做简单截断展示前 120 字符
        def trunc(s: str, max_len: int = 120) -> str:
            if len(s) <= max_len:
                return s
            return s[: max_len - 3] + "..."

        print(f"| {q} | `{trunc(sql_base)}` | `{trunc(sql_rl)}` |")


if __name__ == "__main__":
    main()
