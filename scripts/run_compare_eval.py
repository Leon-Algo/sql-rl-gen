import argparse
from pathlib import Path
import csv
import ast


BASE_EVAL_DIR = Path("output/model_spider_eval_base")
RL_EVAL_DIR = Path("output/model_spider_eval_rl")


def run_eval(model_name: str, dataset: str, template: str, dataset_name: str, trained_agent_path: str | None, outdir: Path, n_rows: int) -> None:
    """调用官方 evaluate_model.py 脚本，运行一次评估（基模或 RL）。"""

    from subprocess import check_call

    args = [
        "python",
        "sql_rl_gen/generation/evaluate_model.py",
        "--model_name_or_path",
        model_name,
        "--dataset",
        dataset,
        "--template",
        template,
        "--outdir",
        str(outdir),
        "--dataset_name",
        dataset_name,
        "--number_of_rows_to_use",
        str(n_rows),
    ]
    if trained_agent_path is not None:
        args.extend(["--trained_agent_path", trained_agent_path])

    print("Running:", " ".join(args))
    check_call(args)


def load_eval_csv(csv_path: Path) -> list[dict]:
    """读取 statistics_metrics 并拆成逐样本的行。

    原始文件里每一列是一个 Python list 的字符串表示，
    比如 accuracy 列是 "[1.0, 0.0, ...]"，output 列是
    "[(...) , (...)]"，expected 列是 "['sql1', 'sql2', ...]"。

    这里把这些字符串安全地解析成 Python 对象，然后按索引
    展开为多行：[{"accuracy": float, "output": str, ...}, ...]
    """

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_in_file = list(reader)

    if not rows_in_file:
        return []

    # 这个文件通常只有两行：一行是整体统计，一行是逐样本列表。
    # 我们取最后一行作为逐样本记录来源。
    row = rows_in_file[-1]

    def parse_list(value: str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []

    accuracies = parse_list(row.get("accuracy", "[]"))
    outputs = parse_list(row.get("output", "[]"))
    expecteds = parse_list(row.get("expected", "[]"))

    n = min(len(accuracies), len(outputs), len(expecteds))

    per_sample: list[dict] = []
    for i in range(n):
        out_i = outputs[i]
        # output 是形如 ("SQL ...",) 的 tuple
        if isinstance(out_i, tuple) and out_i:
            out_str = out_i[0]
        else:
            out_str = str(out_i)

        per_sample.append(
            {
                "accuracy": float(accuracies[i]),
                "output": out_str,
                "expected": str(expecteds[i]),
            }
        )

    return per_sample


def build_comparison_table(base_rows: list[dict], rl_rows: list[dict]) -> str:
    """按行号对齐基模与 RL 评估结果，生成 Markdown 表。"""

    lines: list[str] = []
    lines.append("# 基座模型 vs 强化学习后模型 对比样本表\n")
    lines.append("| 样本序号 | gold SQL (expected) | 基座模型 SQL (output) | 基座 accuracy | RL SQL (output) | RL accuracy |")
    lines.append("| -------- | ------------------- | --------------------- | ------------- | --------------- | ----------- |")

    n = min(len(base_rows), len(rl_rows))
    for idx in range(n):
        b = base_rows[idx]
        r = rl_rows[idx]
        expected = (b.get("expected") or r.get("expected") or "").replace("|", "\\|")
        base_out = (b.get("output") or "").replace("|", "\\|")
        rl_out = (r.get("output") or "").replace("|", "\\|")
        base_acc = b.get("accuracy")
        rl_acc = r.get("accuracy")
        lines.append(
            f"| {idx+1} | `{expected}` | `{base_out}` | {base_acc} | `{rl_out}` | {rl_acc} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行基模和 RL 评估，并生成对比 Markdown 表格")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="example_text2sql_spider_dev")
    parser.add_argument("--dataset_name", type=str, default="spider")
    parser.add_argument("--template", type=str, default="llama3")
    parser.add_argument("--trained_agent_path", type=str, required=True)
    parser.add_argument("--number_of_rows_to_use", type=int, default=50)
    parser.add_argument("--out_md", type=str, default="output/compare_base_vs_rl.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    # 1) 运行基模评估
    BASE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    run_eval(
        model_name=args.model_name_or_path,
        dataset=args.dataset,
        template=args.template,
        dataset_name=args.dataset_name,
        trained_agent_path=None,
        outdir=root / BASE_EVAL_DIR,
        n_rows=args.number_of_rows_to_use,
    )

    # 2) 运行 RL 评估
    RL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    run_eval(
        model_name=args.model_name_or_path,
        dataset=args.dataset,
        template=args.template,
        dataset_name=args.dataset_name,
        trained_agent_path=args.trained_agent_path,
        outdir=root / RL_EVAL_DIR,
        n_rows=args.number_of_rows_to_use,
    )

    # 3) 读取两次评估的逐样本统计文件并生成对比表
    base_csv = root / BASE_EVAL_DIR / "statistics_metrics"
    rl_csv = root / RL_EVAL_DIR / "statistics_metrics"

    if not base_csv.exists():
        raise FileNotFoundError(f"基模评估结果未找到: {base_csv}")
    if not rl_csv.exists():
        raise FileNotFoundError(f"RL 评估结果未找到: {rl_csv}")

    base_rows = load_eval_csv(base_csv)
    rl_rows = load_eval_csv(rl_csv)

    md = build_comparison_table(base_rows, rl_rows)

    out_md_path = root / args.out_md
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(md, encoding="utf-8")
    print(f"对比 Markdown 表格已生成: {out_md_path}")


if __name__ == "__main__":
    main()
