# SQL-RL-Gen 强化学习训练学习与改造指南

本指南面向二次开发者，聚焦：基模位置、RL 训练如何组织、奖励函数在哪里、如何迁移到新 SQL 任务/数据。

## 1. 你当前的理解（确认）
- 数据预处理 → RL 环境 + 基模 + PPO → 评估：管线完整且可跑通（已验证）。
- Eureka 奖励函数自动搜索：属于“高级自动迭代”模块；在已有基础奖励机制之上，自动探索更优 `compute_reward` 实现。
- 原始奖励机制：环境内置了默认 `compute_reward`，无需 Eureka 也可训练/评估。

以上理解正确。本指南专注基础 RL 流程与可改造点。

## 2. 基础训练流程总览
```
数据（JSON） -> 载入数据集 -> 构造 observation_list/data_list_to_pass
              -> 加载基模与 tokenizer
              -> 构建 SQLRLEnv（包含 compute_reward）
              -> CustomActor 封装生成策略
              -> PPO agent（pfrl）训练
              -> 日志/指标/检查点输出
```

核心入口：`sql_rl_gen/generation/sql_generation.py`
- `run()`：解析 `TrainingArguments/DataArguments`，初始化后调用 `train_llm()`
- `train_llm()` 主要步骤：
  - `dataset = get_dataset(data_args)` → 截取样本数 → `prepare_observation_list_and_dataset_to_pass(...)`
  - `tokenizer = AutoTokenizer.from_pretrained(...)`
  - `model = AutoModelForSeq2SeqLM.from_pretrained(...)`（基模）
  - 送入设备（`envs/utils.py` 的 `find_device()`），统计可训练参数
  - 构建环境：`env = SQLRLEnv(...)`
  - 构建策略：`actor = CustomActor(env, model, tokenizer, temperature=..., top_k=..., top_p=...)`
  - 构建 PPO：`agent = actor.agent_ppo(update_interval=..., minibatch_size=..., epochs=..., lr=...)`
  - 训练：`train_evaulate_agent(agent, env, steps=..., ...)`

命令示例（短程 Smoke 训练）：
```bash
python sql_rl_gen/generation/sql_generation.py \
  --model_name_or_path juierror/flan-t5-text2sql-with-schema-v2 \
  --dataset example_text2sql_spider_train \
  --template llama3 \
  --steps_n 200 \
  --dataset_name spider \
  --outdir ./output/model_spider_train_smoke
```

## 3. 基模（Base Model）与策略（Policy）
- 基模位置：Hugging Face 路径通过 `--model_name_or_path` 指定（默认：`juierror/flan-t5-text2sql-with-schema-v2`）。
- 加载方式：`AutoModelForSeq2SeqLM` + `AutoTokenizer`（Transformers）。
- 策略封装：`sql_rl_gen/generation/rllib/custom_actor.py`
  - 负责将 LLM 的生成（温度、top-k、top-p 等）与 PPO 对接，返回 `agent_ppo(...)`。
  - 学习率、minibatch、epochs 等训练超参由 `TrainingArguments` 控制（见 `configs/rl_args.py`）。

切换基模：只需改 `--model_name_or_path` 为任意可兼容的 Seq2Seq 模型（建议同类 T5/FLAN 族）并调整 tokenizer 分隔符等细节（代码中默认 `sep_token=';'`）。

## 4. 环境与奖励函数（Reward）
文件：`sql_rl_gen/generation/envs/sql_generation_environment.py`
- 类：`SQLRLEnv`
  - `get_reward(self, input_item, predicted_list, finish)`：外部访问入口；在 episode 结束 (`finish=True`) 时调用 `compute_reward`。
  - `compute_reward(self, input_item, predicted_text) -> Tuple[float, Dict]`：默认奖励实现（基础版本）。
- 默认奖励机制要点：
  - 对生成 SQL 与黄金 SQL 做对比，结合关键词惩罚（例如禁止 DROP/DELETE 等危险词）
  - 利用 SQLite 执行生成 SQL 与标准 SQL（见 `envs/utils.py`），统计执行是否成功/结果一致性
  - 返回 `(reward_value, metrics_dict)`，metrics 常含 `accuracy/precision/recall/f1/reward` 等
- 环境依赖：
  - `envs/utils.py`：设备选择、prompt 构造、SQLite 执行、指标收集与 CSV 保存
  - 数据库路径：由 `configs/config.py` 暴露（如 Spider DB 路径），通过 `SQLRLEnv(..., dataset_path=...)` 注入

扩展奖励：
- 直接编辑 `compute_reward(...)`，增加你的指标或权重
- 确保返回值为 `Tuple[float, Dict]` 且 Dict 键名稳定，方便评估与日志解析

## 5. 数据与 Observation 构造
- 预处理：`data_preprocess/spider_sql_data_process.py` 生成 `example_text2sql_spider_{train,dev}.json`
- `get_dataset(...)` 与 `prepare_observation_list_and_dataset_to_pass(...)`（在 `envs/utils.py`）会：
  - 将问题 + schema 拼装为模型输入（observation_list）
  - 保留结构化字段给奖励函数使用（data_list_to_pass）
  - 收集列名不一致集合 `columns_names_mismatch`（奖励可引用）

如需改 Prompt 模板：
- 入口参数 `--template`（默认 `llama3`）会影响输入格式；可在 `utils.py` 查看/扩展格式化逻辑。

## 6. 评估（Evaluation）
入口：`sql_rl_gen/generation/evaluate_model.py`
- 可加载已训练策略（`--trained_agent_path ./output/.../best`）或直接用基模做贪心/采样预测
- 逐样本执行 SQL、记录指标，输出到 `evaluation_metrics.csv`、`feedback_metrics/`、`statistics_metrics/`

示例：
```bash
python sql_rl_gen/generation/evaluate_model.py \
  --model_name_or_path juierror/flan-t5-text2sql-with-schema-v2 \
  --dataset example_text2sql_spider_dev \
  --template llama3 \
  --trained_agent_path ./output/model_spider_train_smoke/best \
  --outdir ./output/model_spider_eval_smoke \
  --dataset_name spider
```

## 7. 迁移到你的 SQL 任务/数据集（分步）
1) 准备原始数据与 SQLite 数据库
- 至少包含：`question`、`db_id`、`gold_sql`；并有与 `db_id` 对应的数据库文件及 schema 描述（类似 Spider 的 `tables.json`）。

2) 编写数据预处理脚本
- 参考 `data_preprocess/spider_sql_data_process.py`，将你的原始数据转为 `example_text2sql_<name>_{train,dev}.json`
- 字段对齐：确保后续 `prepare_observation_list_and_dataset_to_pass` 能拼装 observation（问题 + schema）

3) 注册数据集与路径
- `configs/data_args.py`：在 `DatasetName` 中新增你的名称；默认 `dataset_dir` 指向 `data_preprocess/data/`
- `configs/config.py`：为你的数据库路径添加常量（仿照 Spider/WikiSQL），并在训练入口分支选择正确路径

4) 放置数据库与 schema
- 将 SQLite 数据库放到你在 `config.py` 指定的目录层级；按 `db_id` 可定位到对应文件

5) 调整模板/Prompt（可选）
- 如果 schema 组织差异较大，可以定制 `utils.py` 中的输入模板或增加 `--template` 变体

6) 调整/扩展奖励函数（可选但推荐）
- 在 `compute_reward` 中加入适合你任务的指标：如列覆盖率、聚合/排序合理性、JOIN 数量/类型约束等
- 确保异常 SQL 执行时返回合理负奖励，避免策略学习不稳定

7) 训练与评估
- 用你的数据集名与路径参数启动训练/评估；确认输出目录生成检查点与指标 CSV

## 8. 常见问题与排查
- IndexError（训练开始时）：样本太少导致代码中固定访问索引（如 35）报错 → 增大 `--number_of_rows_to_use`
- SQLite OperationalError：SQL 语法或表/列名不匹配 → 加强 schema 对齐与奖励中的列名检查
- 指标缺失：自定义奖励未产出完整指标字典 → 初始化默认键，避免下游解析失败

## 9. 可进一步的工程化建议
- 奖励函数策略化（Strategy Pattern）：便于在配置中切换/热替换
- Prompt 配置化：将 `descriptions/` 文本替换为 YAML/JSON 并做 A/B 实验
- 训练日志接入 TensorBoard/W&B
- PPO 实现替换/并行策略试验（可探索 `trl`）

## 10. 你可能关心的代码坐标
- 训练入口：`sql_rl_gen/generation/sql_generation.py`（`train_llm`/`run`）
- 环境/奖励：`sql_rl_gen/generation/envs/sql_generation_environment.py`（`SQLRLEnv`/`compute_reward`）
- 工具与执行：`sql_rl_gen/generation/envs/utils.py`
- 策略/Agent：`sql_rl_gen/generation/rllib/custom_actor.py`、`custom_trainer.py`
- 数据处理：`data_preprocess/*.py`、`data_preprocess/data/`
- 配置：`configs/config.py`、`configs/data_args.py`、`configs/rl_args.py`

---
如需我基于你的目标数据集提供一个预处理脚手架（含字段映射与最小可运行示例），告诉我你的原始样本/库文件组织方式与期望字段名，我可以直接补齐对应文件与命令。 
下面补充一节更偏「直觉 + 流程图」的说明，方便日后复习。

## 11. 流程图版直觉说明（单步 MDP + PPO）

### 11.1 顶层总流程（从数据到 RL 训练再到评估）

可以把整个项目想成三层流水线：

1. 数据准备层（预处理）
2. RL 训练层（环境 + 基模 + PPO）
3. 评估层（在 dev 集上检验效果）

用文字画出来就是：

```text
原始 Spider 数据
  │
  ▼
[数据预处理脚本]
  │  生成
  ▼
训练集 / 开发集 JSON
(question + schema + gold_sql)
  │
  ▼
[RL 训练脚本 sql_generation.py]
  │
  │  1) 构造 observation_list (提示词文本)
  │  2) 构造 data_list_to_pass (结构化信息)
  │  3) 加载基座 LLM (flan-t5-text2sql-...)
  │  4) 构造 RL 环境 SQLRLEnv
  │  5) 用 PPO 训练策略
  ▼
训练好的策略模型（带 RL 权重）
  │
  ▼
[评估脚本 evaluate_model.py]
  │
  ▼
在 dev 集上生成 SQL、算指标并导出结果
```

### 11.2 单条样本是如何参与一次 RL 训练的

单条样本 `(question, schema, gold_sql)` 被看成一个「单步 MDP」的 episode：

```text
一条训练样本
(question, schema, gold_sql)
  │
  ▼
[observation 构造]
  │  把 question + schema 拼成提示词 prompt
  │  （自然语言问题 + 表结构说明）
  ▼
observation（提示词文本）
  │
  ▼
[策略网络 = LLM + PPO 头]
  │  接收 observation，按当前参数
  │  生成一个完整 SQL 序列 (action)
  ▼
生成的 predicted_sql
  │
  ▼
[环境 SQLRLEnv.compute_reward()]
  │  1) 用 gold_sql 在 SQLite 中执行 → gold_result
  │  2) 用 predicted_sql 在 SQLite 中执行 → pred_result
  │  3) 比较结果、报错情况 → 算出一个 reward (标量)
  ▼
reward （比如 -1 ~ +1 之间的一个数）
  │
  ▼
[PPO 更新阶段使用]
  │  记下 (observation, action, reward, old_log_prob)
  │  等攒够一批样本后，一起算梯度，更新策略参数
  ▼
更新后的策略网络（LLM 权重被微调了一点）
```

在这个视角下：

- 状态 $s$：编码后的问题 + schema 文本（observation）。
- 动作 $a$：整句 SQL 文本（序列级行动，内部对应 token 序列的 log prob）。
- 奖励 $r$：执行 SQL + 与 gold SQL 比对得到的标量。
- 下一个状态：无，episode 直接 `done=True`，属于单步 MDP。

### 11.3 训练循环内部的 PPO 流程

上面的过程是“单条样本”的视角，实际训练是对很多样本不断重复，并在攒够一批样本之后做一次参数更新：

```text
初始化策略网络 (LLM + PPO 头)
  │
  ▼
for t in range(total_steps):                   # 训练大循环
   │
   ├─ 从训练集取一条样本 (question, schema, gold_sql)
   │
   ├─ 构造 observation（提示词文本）
   │
   ├─ 策略网络生成 predicted_sql
   │
   ├─ 环境算 reward：
   │      - 执行 gold_sql → gold_result
   │      - 执行 predicted_sql → pred_result
   │      - 比较，得出 reward
   │
   ├─ 把 (observation, action, reward, old_log_prob) 存入 buffer
   │
   └─ 如果 buffer 里攒够 N 条样本：
       │
       ▼
      [PPO 更新步骤]
       │  用当前 buffer 里的所有样本
       │  计算 advantage，构造 PPO clipped loss
       │  对策略网络做若干次梯度更新
       ▼
      清空 buffer，继续采样下一批
```

类比：

- 采样很多样本 = 做很多道题。
- 攒一批再更新 = 老师讲一次“总结课”，统一纠正写作习惯。

### 11.4 PPO 如何体现“试错”和“记忆”

**试错发生在：**

- 策略网络当前参数下随机/采样生成的 SQL（可能对也可能错）。
- 环境执行 SQL 后给出 reward，告诉策略“这次做得如何”。

**记忆发生在：**

- PPO 更新阶段对参数的持久性改动：
  - reward 高的动作，其对应的生成模式（参数）被整体“奖励”，以后更容易再次出现；
  - reward 低的动作，其对应模式被整体“惩罚”，以后概率下降。
- 这些参数改动会影响之后所有的样本，包括未来再次遇到同一条 question 时的生成行为。

关键点：

- 策略**不记住每一道题的历史**，而是记在**参数里**：
  - 你可以理解为：模型不断调整一套“写 SQL 的通用习惯”。
  - 当同一条样本在经过多次训练后再次出现时，
   它走的是已经被前面无数题目“纠偏过”的新参数，
   因此更有可能生成更合理的 SQL。

### 11.5 样本重复采样与 epoch 的直觉

在实现上，样本重复出现的方式一般有两种（代码可任选其一、或结合）：

1. **严格 epoch 式遍历**：
  - 第 1 个 epoch：把训练集打乱后整遍扫一遍，每个样本用一次；
  - 第 2 个 epoch：再次打乱后再扫一遍；
  - 多个 epoch 下，同一条样本被多次“拿出来做题”。

2. **流式随机采样**：
  - 每次需要一个样本，就从训练集中随机抽一个；
  - 训练时间足够长时，每条样本被抽到很多次是必然事件。

无论哪种方式，本质都是：

- 同一个 `(question, schema, gold_sql)` 会在训练期间**多次参与采样**，
- 每一次参与时，模型已经在前面的若干次 PPO 更新中慢慢改变了参数，
  于是对同一条样本的生成会逐渐越来越合理。

### 11.6 与代码的对应关系速查

当你想把这套流程和代码对应起来时，可以按下面的坐标快速跳转：

- 单条样本 → observation 构造：
  - `sql_rl_gen/generation/envs/utils.py` 中的 `prepare_observation_list_and_dataset_to_pass`

- 环境与 reward：
  - `sql_rl_gen/generation/envs/sql_generation_environment.py` 中的 `SQLRLEnv`、`compute_reward`

- 策略与 PPO agent：
  - `sql_rl_gen/generation/rllib/custom_actor.py`：封装 LLM 为策略网络
  - `sql_rl_gen/generation/rllib/custom_trainer.py`：训练循环 + PPO 更新

- 顶层训练入口：
  - `sql_rl_gen/generation/sql_generation.py`：`train_llm` / `run`

如果以后再有“为什么 RL 在这里能工作”“单步 MDP 怎么体现试错”等问题，可以直接从本节开始往回读，对照上面几份源码一起看，会更直观。