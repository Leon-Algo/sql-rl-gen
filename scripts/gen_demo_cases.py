import csv
from pathlib import Path


def clean_output(s: str) -> str:
    s = s.strip()
    # evaluation_metrics.csv 中的 output 形如 "('SQL ...',)"，这里抽取中间 SQL
    if s.startswith("('") and s.endswith("',)"):
        return s[2:-3]
    return s


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    eval_csv = base_dir / "output" / "model_spider_eval_smoke" / "evaluation_metrics.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"evaluation_metrics.csv not found at {eval_csv}")

    rows = []
    with eval_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    good_cases = []
    partial_cases = []
    error_cases = []

    for r in rows:
        try:
            acc = float(r.get("accuracy") or 0)
        except ValueError:
            acc = 0.0
        try:
            reward = float(r.get("reward") or 0)
        except ValueError:
            reward = 0.0
        err = (r.get("error_reason") or "").strip()
        expected = (r.get("expected") or "").strip()
        output_raw = (r.get("output") or "").strip()
        output = clean_output(output_raw)

        # 跳过没有 expected 且没有 output 的行
        if not expected and not output:
            continue

        record = {
            "expected": expected,
            "output": output,
            "accuracy": acc,
            "reward": reward,
            "error_reason": err,
        }

        if err:
            error_cases.append(record)
        elif acc == 1.0 and reward >= 10.0:
            good_cases.append(record)
        elif reward < 0:
            partial_cases.append(record)

    # 选出代表性样本
    demo_good_simple = good_cases[0] if good_cases else None
    demo_good_complex = good_cases[2] if len(good_cases) > 2 else (good_cases[1] if len(good_cases) > 1 else None)
    demo_partial = partial_cases[0] if partial_cases else None
    demo_error = error_cases[0] if error_cases else None

    demo_md = base_dir / "RL_DEMO_SHOWCASE.md"
    with demo_md.open("w", encoding="utf-8") as f:
        f.write("# Text-to-SQL 强化学习效果小样例（演示用）\n\n")
        f.write("> 通过真实评估结果中的几条样本，直观展示模型在 SQL 生成上的能力和改进空间。\n\n")
        f.write("---\n\n")

        f.write("## 代表性样本对比一览\n\n")
        f.write("| 类型 | 标准 SQL（gold） | 模型生成 SQL | accuracy | reward | 错误原因 |\n")
        f.write("| ---- | ---------------- | ------------ | -------- | ------ | -------- |\n")

        def write_row(label: str, rec: dict) -> None:
            err = rec["error_reason"] or "-"
            f.write(
                f"| {label} | `{rec['expected']}` | `{rec['output']}` | {rec['accuracy']} | {rec['reward']} | {err} |\n"
            )

        if demo_good_simple:
            write_row("简单且完全正确", demo_good_simple)
        if demo_good_complex:
            write_row("稍复杂但完全正确", demo_good_complex)
        if demo_partial:
            write_row("语义接近但不完全正确", demo_partial)
        if demo_error:
            write_row("触发 SQL 运行错误", demo_error)

        f.write("\n---\n\n")
        f.write("> 上表可以直接用于 PPT 或汇报文档，按行口头解释每一类案例即可。\n\n")

        f.write("## Mermaid 风格概要对比（可选展示）\n\n")
        f.write("```mermaid\n")
        f.write("flowchart TB\n")
        f.write("    Q[自然语言问题] -->|提示词构造| O[Observation: 问题+表结构]\n")
        f.write("    O -->|基座大模型推理| S0[未强化学习 SQL]\n")
        f.write("    O -->|RL 后策略推理| S1[强化学习后 SQL]\n")
        f.write("    S0 -->|执行+比对| R0[reward_0]\n")
        f.write("    S1 -->|执行+比对| R1[reward_1]\n")
        f.write("    R0 -->|PPO 更新| P[策略参数]\n")
        f.write("    R1 --> P\n")
        f.write("```\n")

        f.write("\n该图适合用来解释：同一个问题下，基模和强化后的策略都会生成 SQL，再基于执行结果的奖励来统一更新策略参数。\n")

    print(f"Demo markdown generated at: {demo_md}")


if __name__ == "__main__":
    main()
