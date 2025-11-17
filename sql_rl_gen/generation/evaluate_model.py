import logging
import os.path
import sys
from os import listdir
from statistics import mode

import torch
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSeq2SeqLM

from configs.config import BIRD_DATABASES_DEV_PATH, SPIDER_DATABASES_PATH, WIKISQL_PATH
from configs.data_args import DataArguments, DatasetName
from configs.rl_args import EvaluateArguments, EvaluationMethod
from data_preprocess.data_utils import get_dataset
from sklearn.model_selection import KFold

from sql_rl_gen.generation.envs.sql_generation_environment import SQLRLEnv
from sql_rl_gen.generation.envs.utils import (
    find_device,
    sql_query_execution_feedback_on_dataset,
    prepare_observation_list_and_dataset_to_pass,
    save_dict_csv,
)
from sql_rl_gen.generation.rllib.custom_actor import CustomActor
from sql_rl_gen.generation.rllib.custom_trainer import train_evaulate_agent


# 让 HuggingFace 的 tokenizer 在多线程下工作，提高预处理速度
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 在 GPU / MPS / CPU 中自动选择一个可用设备，后续模型与张量都复用这个 device
device = find_device()


def prediction_feedback(statistics, data_list_to_pass, dataset_path, observation, result, columns_names_mismatch):
    """给定一条模型输出 SQL，执行并把得到的反馈写入统计字典。

    参数说明：
    - statistics: dict[str, list]，全局累积的逐样本指标，例如
      accuracy / precision / recall / error_type 等，每个键对应一个
      长度为样本数的 list。
    - data_list_to_pass: 来自 `prepare_observation_list_and_dataset_to_pass`
      的结构化数据，包含真实答案 SQL、数据库名等辅助信息。
    - dataset_path: 当前数据集对应的数据库根目录（Spider/WikiSQL/BIRD）。
    - observation: 单个样本的观测字典，至少包含 'input'（prompt）。
    - result: 当前模型或 RL agent 生成的 SQL 字符串。
    - columns_names_mismatch: 列名映射/修正信息，用于容错执行 SQL。
    """

    # 调用通用的执行与打分逻辑：内部会连接 sqlite 数据库，执行 SQL，
    # 并与 gold SQL 的结果做比较，返回一个字典形式的反馈。
    feedback = sql_query_execution_feedback_on_dataset(
        data_list_to_pass,
        dataset_path,
        observation["input"],
        result,
        columns_names_mismatch,
    )

    # 把当前样本的反馈 append 到 statistics 这个“列式存储”的 dict 中。
    for k, v in feedback.items():
        if k in statistics:
            statistics[k].append(v)
        else:
            statistics[k] = [v]
        logging.info(f"Appending to {k} value: {v}")

    # 为了避免长时间评估导致 GPU 显存碎片化，手动清理缓存。
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
    elif device == torch.device("mps"):
        torch.mps.empty_cache()

    return feedback


def construct_file(statistics):
    """把逐样本的统计 `statistics` 汇总成更高层的 summary。

    - 输入 statistics: 形如 {"accuracy": [0/1,...], "error_type": [..]}。
    - 输出 feedback: 对每个 key 计算 max/mean/min（数值型），或者
      计算众数及其出现次数（离散型），用于写入 feedback_metrics 文件。
    """

    feedback = {}
    for k, v in statistics.items():
        # 如果第一项是 float，认为整个列表是数值型指标
        if isinstance(v[0], float):
            feedback[k] = {}
            feedback[k]["maximum"] = max(v)
            feedback[k]["mean"] = sum(v) / len(v)
            feedback[k]["minimum"] = min(v)
        else:
            # 非数值型指标（例如 error_type、error_reason）使用众数
            mode_of_stat = mode(v)
            feedback[k] = {}
            feedback[k]["most_common"] = mode_of_stat
            feedback[k]["number_of_occurrence"] = len(
                list(filter(lambda x: x is mode_of_stat, v))
            )
    return feedback


def create_data_from_indexes(kfolds, observation_list):
    """根据 KFold 的 index 列表，从 observation_list 中切出 train/test 数据。

    KFold.split 返回的是一个 (train_idx, test_idx) 的列表，本函数
    按照这些下标从 observation_list 中抽取对应样本，返回：

        [(train_data_fold0, test_data_fold0), (train_data_fold1, test_data_fold1), ...]
    """

    to_return_data = []
    for i in range(len(kfolds)):
        test_data = []
        train_data = []
        # 根据 train index 采样出训练 fold 的 observation
        for j in range(len(kfolds[i][0])):
            train_data.append(observation_list[kfolds[i][0][j]])
        # 根据 test index 采样出当前 fold 对应的测试 observation
        for j in range(len(kfolds[i][1])):
            test_data.append(observation_list[kfolds[i][1][j]])
        to_return_data.append((train_data, test_data))
    return to_return_data


def prepare_for_evaluate(eval_args: EvaluateArguments, data_args: DataArguments):
    """准备评估所需的所有静态资源：数据集、DB 路径、模型和 tokenizer。

    主要步骤：
    1. 调用 `get_dataset` 根据 DataArguments 加载 HF dataset；
    2. 根据 eval_args.number_of_rows_to_use 截取前 N 条样本；
    3. 根据 dataset_name 选择数据库路径（Spider / WikiSQL / BIRD）；
    4. 利用 `prepare_observation_list_and_dataset_to_pass` 把 dataset
       转成环境使用的 observation_list（包含 prompt 等）以及
       data_list_to_pass（执行 SQL 时需要的结构化信息）；
    5. 从 HuggingFace 加载 encoder-decoder 模型和 tokenizer；
    6. 把模型移动到指定 device，并切到 eval 模式；
    7. 创建输出目录。
    """

    # 从预处理产出的 HF Dataset 加载原始样本
    dataset = get_dataset(data_args)
    # 为了节省时间，可以只评估前 N 条（smoke / 子集评估）
    dataset = dataset.take(eval_args.number_of_rows_to_use)

    # 根据数据集名字挑选对应的 sqlite 数据库目录
    if data_args.dataset_name == DatasetName.SPIDER.value:
        dataset_path = SPIDER_DATABASES_PATH
    elif data_args.dataset_name == DatasetName.WIKISQL.value:
        dataset_path = WIKISQL_PATH
    else:
        dataset_path = BIRD_DATABASES_DEV_PATH

    # 把 HF dataset 转成 RL 环境和执行器更方便使用的结构：
    # - observation_list: 逐条样本的 prompt, db_id 等
    # - data_list_to_pass: SQL 执行时需要的 ground truth 结果等
    # - columns_names_mismatch: 列名映射，用于宽松匹配
    (
        observation_list,
        data_list_to_pass,
        columns_names_mismatch,
    ) = prepare_observation_list_and_dataset_to_pass(dataset)

    # 从 HuggingFace hub 或本地路径加载 tokenizer 和 Seq2Seq 模型
    tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(eval_args.model_name_or_path)

    # 把模型移动到 GPU / CPU，并切换为 eval 模式（关闭 dropout 等）
    model.to(device)
    model.eval()

    # 确保评估输出目录存在
    os.makedirs(eval_args.outdir, exist_ok=True)

    return (
        dataset_path,
        observation_list,
        data_list_to_pass,
        columns_names_mismatch,
        model,
        tokenizer,
    )


def evaluate_models(eval_args: EvaluateArguments, data_args: DataArguments):
    """单次评估入口：支持基座模型或加载好的 RL agent。

    - 当 `eval_args.trained_agent_path` 非空时：
      1. 构造 SQLRLEnv（evaluation 模式），复用 PPO actor。
      2. 从 checkpoint 中 load RL agent 权重。
      3. 对每个 observation 调用 `actor.predict` 生成 SQL。

    - 当 `trained_agent_path` 为空时：
      1. 直接用 HuggingFace Seq2Seq 模型做 greedy/beam search 生成。

    两种模式都会把结果送到 `prediction_feedback`，最终生成：
    - feedback_metrics: 各个指标的 mean/max/min；
    - statistics_metrics: 逐样本的原始指标列表。
    """

    (
        dataset_path,
        observation_list,
        data_list_to_pass,
        columns_names_mismatch,
        model,
        tokenizer,
    ) = prepare_for_evaluate(eval_args, data_args)

    model.eval()

    # 把日志打印到 stdout，方便在命令行直接观察每条样本的反馈
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")
    logger = logging.getLogger("Test")

    # 如果提供了 RL checkpoint，就构造一个 RL 环境与 agent
    if eval_args.trained_agent_path:
        # SQLRLEnv 封装了 Text-to-SQL 的单步 MDP：一次“做题”
        # 就是一个 episode。这里用 "evaluation" 模式，只做前向预测
        # 不更新参数。
        env = SQLRLEnv(
            model,
            tokenizer,
            data_list_to_pass,
            dataset_path,
            eval_args.outdir,
            logger,
            "evaluation",
            columns_names_mismatch=columns_names_mismatch,
            observation_input=observation_list,
            compare_sample=1,
        )

        # CustomActor 负责把 transformer 模型封装到 PPO agent 接口中，
        # 内部会处理状态编码、动作空间（token）的采样等。
        actor = CustomActor(
            env,
            model,
            tokenizer,
            temperature=eval_args.temperature,
            top_k=eval_args.top_k,
            top_p=eval_args.top_p,
        )

        # 根据 EvaluateArguments 中的 PPO 超参创建 agent
        agent = actor.agent_ppo(
            update_interval=eval_args.update_interval,
            minibatch_size=eval_args.minibatch_size,
            epochs=eval_args.epochs,
            lr=eval_args.lr,
        )

        # 从磁盘加载已训练好的 RL 策略参数
        agent.load(eval_args.trained_agent_path)

    statistics = {}

    # 逐条 observation 遍历，生成 SQL 并执行评价
    for observation in observation_list:
        if eval_args.trained_agent_path:
            # actor.predict 返回 (generated_sql_with_eos, info)
            # 这里用 [0][:-4] 去掉结尾的特殊 token（一般是 "</s>" 之类）
            result = actor.predict(observation)[0][:-4]
        else:
            # 基座模型路径：直接用 HuggingFace Seq2Seq 生成 SQL
            input_ids = tokenizer(
                observation["input"],
                return_tensors="pt",
                max_length=1024,
            ).input_ids
            input_ids = input_ids.to(device)

            # 这里使用默认解码策略（通常是 greedy search），
            # 通过 max_length/min_length 控制 SQL 序列长度范围。
            generated_ids = model.generate(
                input_ids,
                max_length=120,
                min_length=10,
            )
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # 注释中的 GC 指的是 Python 垃圾回收；由于大 batch 生成
            # 容易导致 GPU 显存无法及时释放，这里手动删中间变量。
            del input_ids
            del generated_ids

        # 无论来自基座模型还是 RL agent，统一走 feedback 流程
        prediction_feedback(
            statistics,
            data_list_to_pass,
            dataset_path,
            observation,
            result,
            columns_names_mismatch,
        )

    # 把逐样本 statistics 汇总成一个高层 feedback（均值/最大值等）
    feedback = construct_file(statistics)

    # 写出两个 CSV/文本文件：一个 summary，一个逐样本明细
    save_dict_csv(feedback, eval_args.outdir, "feedback_metrics")
    save_dict_csv(statistics, eval_args.outdir, "statistics_metrics")

    return feedback, statistics


def cross_validation_evaluate(eval_args: EvaluateArguments, data_args: DataArguments):
    """使用 K 折交叉验证评估 RL agent 的泛化能力。

    流程概览：
    1. 把 observation_list 划分为 K 个 fold；
    2. 对于每个 fold i：
       - 把前 i 个 fold 视为“已经训练过”的数据（通过 checkpoint 继承）；
       - 在第 i 个训练 fold 上继续训练 RL agent；
       - 在对应的测试 fold 上评估，并记录反馈；
    3. 最终把每个 fold 的 feedback/statistics 聚合并写出文件。
    """

    (
        dataset_path,
        observation_list,
        data_list_to_pass,
        columns_names_mismatch,
        model,
        tokenizer,
    ) = prepare_for_evaluate(eval_args, data_args)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")
    logger = logging.getLogger("Cross Validation")

    ksplit = 5
    # 设置 random_state 保证 KFold 划分可复现
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
    kfolds = list(kf.split(observation_list))
    kfolds_data = create_data_from_indexes(kfolds, observation_list)

    all_feedback = []
    all_statistics = []

    # latest_checkpoint 跟踪最近一次训练结果所在目录，用于 warm-start
    latest_checkpoint = f"./train/0_checkpoint"

    for i in range(len(kfolds_data)):
        # ------------------------- 训练阶段 -------------------------
        model.train()

        # 使用全部 observation_list 做训练环境，只是 RL 逻辑内部会
        # 根据 compare_sample 和 data_list_to_pass 控制采样。
        env = SQLRLEnv(
            model,
            tokenizer,
            data_list_to_pass,
            dataset_path,
            "./train/",
            logger,
            "training",
            columns_names_mismatch=columns_names_mismatch,
            observation_input=observation_list,
            compare_sample=1,
        )

        actor = CustomActor(
            env,
            model,
            tokenizer,
            temperature=eval_args.temperature,
            top_k=eval_args.top_k,
            top_p=eval_args.top_p,
        )
        agent = actor.agent_ppo(
            update_interval=eval_args.update_interval,
            minibatch_size=eval_args.minibatch_size,
            epochs=eval_args.epochs,
            lr=eval_args.lr,
        )

        # 第 0 个 fold 使用用户提供的初始 checkpoint，后续 fold 则
        # 用上一轮训练后的 "800_finish" 作为 warm-start。
        agent.load(
            f"{latest_checkpoint}/800_finish"
            if i > 0
            else eval_args.trained_agent_path
        )

        latest_checkpoint = f"./train/{i}_checkpoint"

        try:
            # 这里调用通用的 RL 训练 + 周期性评估函数。
            train_evaulate_agent(
                agent,
                env,
                steps=len(kfolds_data[i][0]),
                outdir=latest_checkpoint,
                eval_n_steps=None,
                eval_n_episodes=5,
                train_max_episode_len=1000,
                eval_interval=10,
            )
        except Exception as e:
            # 训练过程偶尔会因为个别样本报错，这里简单打印并继续
            print(e)

        # ------------------------- 测试阶段 -------------------------
        model.eval()

        env = SQLRLEnv(
            model,
            tokenizer,
            data_list_to_pass,
            dataset_path,
            "./test/",
            logger,
            "testing",
            columns_names_mismatch=columns_names_mismatch,
            observation_input=observation_list,
            compare_sample=1,
        )

        actor = CustomActor(
            env,
            model,
            tokenizer,
            temperature=eval_args.temperature,
            top_k=eval_args.top_k,
            top_p=eval_args.top_p,
        )
        agent = actor.agent_ppo(
            update_interval=eval_args.update_interval,
            minibatch_size=eval_args.minibatch_size,
            epochs=eval_args.epochs,
            lr=eval_args.lr,
        )

        # 自动从最新 checkpoint 目录中选择一个“最好”的文件：
        # 如果有 "*_finish" 文件，就用它；否则回退到默认 "best"。
        filename = "best"
        list_directory = listdir(f"{latest_checkpoint}")
        for file_name in list_directory:
            if "finish" in file_name:
                filename = file_name

        agent.load(f"{latest_checkpoint}/{filename}")

        statistics = {}
        # 只在当前 fold 的测试样本上做预测与反馈
        for observation in kfolds_data[i][1]:
            result = actor.predict(observation)[0][:-4]
            prediction_feedback(
                statistics,
                data_list_to_pass,
                dataset_path,
                observation,
                result,
                columns_names_mismatch,
            )

        feedback = construct_file(statistics)
        all_feedback.append(feedback)
        all_statistics.append(statistics)

    # 把每个 fold 的 summary/明细整体写出
    save_dict_csv(all_feedback, eval_args.outdir, "feedback_metrics")
    save_dict_csv(all_statistics, eval_args.outdir, "statistics_metrics")

    return all_feedback, all_statistics


def compare_models_rl():
    """命令行入口：解析参数并选择评估模式（单次 or KFold）。"""

    # HfArgumentParser 可以一次性把多个 dataclass 解析出来，
    # 这里对应 EvaluateArguments 和 DataArguments。
    parser = HfArgumentParser((EvaluateArguments, DataArguments))
    eval_args, data_args = parser.parse_args_into_dataclasses()

    # 根据 DataArguments（如 dataset_name/dataset_dir）做额外初始化，
    # 包括从 dataset_info.json 里解析出数据路径等。
    data_args.init_for_training()

    if eval_args.evaluation_method == EvaluationMethod.KFOLD.value:
        cross_validation_evaluate(eval_args, data_args)
    else:
        evaluate_models(eval_args, data_args)


if __name__ == "__main__":
    # 直接 `python evaluate_model.py ...` 时，从命令行读取参数并执行
    compare_models_rl()
