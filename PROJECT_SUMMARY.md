# SQL-RL-Gen 项目总结与架构说明

## 1. 项目目的 (Purpose)
SQL-RL-Gen 旨在通过强化学习 (RL) 和语言模型 (LLM) 技术提升 Text-to-SQL 生成质量。项目不仅提供基础的训练与评估管线，还集成 “Eureka” 风格的奖励函数自动搜索，通过迭代生成与验证奖励函数代码，优化 RL 环境中的 `compute_reward` 逻辑，从而提升模型生成 SQL 的正确性、执行稳定性与通用性。

核心目标：
- 构建一个可扩展的 RL 环境用于 Text-to-SQL 任务。
- 支持多维度反馈（执行正确性、schema 匹配、关键词惩罚等）。
- 自动搜索并演化奖励函数以提高训练信号质量。
- 提供数据预处理、训练、评估、奖励搜索的全流程脚本化入口。

## 2. 整体功能概述 (High-Level Overview)
流程主线：
1. 数据预处理：将 Spider（或其他）原始数据转换为适合 SFT/RL 的样本 JSON。
2. 环境构建：`SQLRLEnv` 封装模型、tokenizer、数据与奖励计算逻辑。
3. 强化学习训练：使用 PPO（基于 PFRL）对模型生成进行策略优化。
4. 评估：按样本或批量生成 SQL，计算多种指标并保存。
5. Eureka 奖励搜索：利用 LLM（Ollama 本地推理）迭代生成新的奖励函数代码，插入环境并快速训练以评估其效果，选择最优版本。

## 3. 架构总览 (Architecture Overview)
逻辑分层：
- 数据层 (data_preprocess)
- 配置与参数层 (configs)
- 环境与工具层 (generation/envs)
- RL 算法层 (generation/rllib)
- 驱动脚本层 (scripts & 顶级 python 入口)
- 奖励搜索层 (eureka_sql.py + descriptions 提示词集合)

关键组件交互：
```
Spider 原始数据 ──> data_preprocess/*.py ──> 训练/评估数据 JSON
                                    ↓
configs/*.py 提供路径/参数 ──> sql_generation.py 构建 SQLRLEnv + PPO Agent
                                    ↓
environment.compute_reward() 基础奖励 ──> 迭代训练输出日志与指标
                                    ↓
eureka_sql.py 使用 Ollama 生成新 reward 函数并替换 compute_reward，快速试训评估
                                    ↓
evaluate_model.py 对训练好的策略或基模型做定性/定量评估
```

## 4. 关键模块详解 (Key Modules)
### 4.1 `configs/`
- `config.py`：集中管理路径常量（数据根目录、输出目录、Spider/WikiSQL 路径等）。我们已修正为基于仓库相对路径，避免硬编码跨机器失效。
- `data_args.py`：定义数据相关 CLI/参数（数据集名称、数据目录、子集行数限制）。
- `rl_args.py`：定义训练相关参数（steps、学习率、更新频率、top-k/p、温度等）。

### 4.2 `data_preprocess/`
- `spider_sql_data_process.py`：读取原始 Spider `train_*.json` + `tables.json`，构建包含问题、数据库 schema、标准 SQL 的训练样本。
- `wikisql_sql_data_process.py`：类似处理 WikiSQL（保留扩展能力）。
- `data_utils.py`：抽象获取数据集、格式标准化、与枚举 `DatasetName` 结合。
- 输出：`example_text2sql_spider_{train,dev}.json` 等文件，通过脚本生成。

### 4.3 `sql_rl_gen/generation/`
- `sql_generation.py`：训练入口。核心步骤：加载模型 → 构造环境 `SQLRLEnv` → 构建 `CustomActor` → 生成 PPO `agent` → 调用 `train_evaulate_agent` 循环。
- `evaluate_model.py`：评估入口。可加载已训练策略或直接用基模型做预测，收集反馈与执行指标。
- `envs/sql_generation_environment.py`：主环境；实现 `get_reward` 与默认 `compute_reward`，对模型生成 SQL 进行执行与度量。
- `envs/sql_generation_environment_obs.py`：观测/调试版本，作为 Eureka 奖励注入模板源。
- `envs/utils.py`：设备选择、Prompt 构造、SQL 执行（SQLite）、指标统计与 CSV 保存。

### 4.4 `rllib/`
- `custom_actor.py`：封装模型生成逻辑（采样策略、温度、top-k/p），并构建与 PPO 对接的接口。
- `custom_trainer.py`：训练循环（与 pfrl 集成），包含保存检查点、评估节奏等。

### 4.5 `eureka_sql.py`
实现奖励函数自动搜索：
1. 组装 system/user prompt（来自 `descriptions/*.txt`）。
2. 调用 Ollama Chat，采样若干次，提取代码块 → 找到 `def compute_reward...` 函数。
3. 将新函数插入环境文件，生成临时训练脚本副本。
4. 启动短程 PPO 训练（步骤数较小）验证奖励质量，记录执行成功率与最大“成功度”（如平均 accuracy）。
5. 最优代码进入下一轮迭代上下文或最终评估，多次评估统计写入 `final_eval.npz`。
（我们已修正训练完成/失败日志判定与评估阶段聚合。）

### 4.6 `descriptions/`
Prompt 模板素材：初始系统说明、用户说明、奖励函数签名、执行错误回馈、策略反馈、输出格式限制等。通过格式化注入，使 LLM 输出更趋近纯函数代码。

### 4.7 `scripts/`
- `generate_data.sh`：调用对应数据处理脚本（参数 spider / wikisql）。
- `run_train.sh`：快速训练统一入口（传数据集名）。
- `evaluate_model.sh`：评估入口，加载指定训练输出目录的 `best` 权重。
- `eureka_sql.sh`：包装 Eureka 搜索脚本（样本数、迭代数、模型名）。

## 5. 数据架构 (Data Architecture)
### 5.1 原始 Spider 数据
- `train_spider.json` / `train_others.json` / `dev.json`：包含 (question, db_id, sql) 等。`tables.json` 提供字段与关系描述。

### 5.2 处理后样本字段（典型结构）
```
{
  "question": "查询问题文本",
  "db_id": "数据库名称",
  "gold": "标准 SQL 语句",
  "schema": {
      "tables": [...],
      "columns": [...],
      ...
  },
  "extra": {... 可扩展辅助信息 ...}
}
```

### 5.3 环境内部数据拆分
- `observation_list`: 直接用于模型生成的输入（拼接问题 + schema 等）。
- `data_list_to_pass`: 结构化对照项（用于奖励计算、执行比较）。
- 列名不匹配集合 `columns_names_mismatch`：用于奖励中惩罚或提示。

## 6. 奖励函数机制 (Reward Function Mechanics)
### 6.1 默认 `compute_reward`
- 解析模型生成 SQL 与黄金 SQL。
- 执行 SQLite：捕获 `OperationalError` 等作为负反馈。
- 指标：accuracy、precision、recall、f1、自定义 reward（整合关键词惩罚、执行成功加成等）。
- 返回：`Tuple[float, Dict]`，第一个为主奖励数值，第二个为指标字典。

### 6.2 Eureka 替换流程
- LLM 输出新的 `compute_reward` 函数代码（必须匹配签名）。
- 插入环境文件后进行短程试训，统计平均 accuracy（或相关成功度）。
- 多样本对比 → 选最优 → 评估阶段复核多次稳定性。

### 6.3 改进点（Todo）
- 强制输出模板：只允许单一 ```python 代码块，减少解析失败。
- 增加 AST 校验：确保函数返回类型与使用的键完整。
- 增加回滚策略：新函数异常时自动恢复上一次成功版本。

## 7. 已完成成果 (Current Achievements)
| 阶段 | 状态 | 说明 |
|------|------|------|
| 环境/依赖安装 | ✅ | requirements 安装 + 可编辑模式 setup.py |
| 配置路径修复 | ✅ | `config.py` 与 `data_args.py` 统一用仓库相对路径 |
| 数据生成 | ✅ | 生成 Spider 训练/开发 JSON（example_text2sql_spider_train/dev）|
| Smoke 训练 | ✅ | 短程 PPO（减少 steps，验证管线）产出 checkpoints |
| Smoke 评估 | ✅ | 指标与反馈 CSV 正常生成 |
| Eureka 初次运行 | ✅(部分) | 成功执行一轮并改进日志；仍需 GPU 性能支持与稳定代码格式收敛 |
| GPU 支持 | ⚠ | 当前 Ollama 未识别 GPU（需宿主 Docker + NVIDIA Toolkit）|

输出示例：
- 训练输出目录：`./output/model_spider_train_smoke/` 包含中间与 `_finish` 检查点。
- 评估目录：`./output/model_spider_eval_smoke/` 含 `evaluation_metrics.csv`、`feedback_metrics/`、`statistics_metrics/`。
- Eureka 目录：`outputs/eureka_sql/<timestamp>/summary.png`、`summary.npz`、`final_eval.npz`。

## 8. 后续建议 (Next Steps)
1. GPU 加速：在宿主机使用 Docker 部署 GPU 版 Ollama，提升生成吞吐与响应速度。
2. 强化奖励稳定性：增加生成代码的单元测试（对 mock 的 SQL/指标结构做断言）。
3. 指标扩展：增加“schema 覆盖率”“列顺序一致性”“聚合函数使用正确性”“join 数量合理性”等特征嵌入奖励。
4. 数据集扩展：接入 BIRD / WikiSQL，统一抽象字段映射。
5. 多任务训练：通过多数据源环境切换提升泛化能力。
6. 长期：引入 Curriculum（简单查询 → 复杂多表 join）提高学习稳定度。

## 9. 二次开发建议 (Secondary Development Guide)
- 插件化奖励：抽象 `compute_reward` 为策略类（Strategy Pattern），支持动态加载 Python 文件。
- Prompt 版本管理：为 `descriptions/` 引入 YAML/JSON 配置以做 A/B 测试。
- 可视化：将训练与评估指标导入 TensorBoard 或 Weights & Biases。
- 更换 PPO 实现：可对接 `trl` 库以获得更丰富 RLHF 组件。
- 模型切换：将 `model_name_or_path` 抽象通过配置文件列表迭代尝试多模型基座。

## 10. 常见故障排查 (Troubleshooting)
| 问题 | 可能原因 | 解决思路 |
|------|----------|---------|
| 训练脚本 IndexError | 使用行数过少，访问固定索引 35 | 提高 `--number_of_rows_to_use` 或移除该调试输出 |
| SQLite OperationalError 多 | 生成 SQL 语法/表名错误 | 增加 schema 字段对齐/列名映射修正 |
| Eureka 解析失败 | LLM 输出含解释文本或无函数签名 | 强化 prompt + 限制只输出函数体代码块 |
| Ollama 低 vram 模式 | GPU 库不可见或容器未启用 NVIDIA runtime | 安装 NVIDIA Container Toolkit + Docker GPU 参数 |
| 指标空或格式错 | 新奖励函数未写完整字典键 | 在函数中添加默认键初始化、单元测试校验 |

## 11. 快速命令汇总 (Command Cheat-Sheet)
```bash
# 数据生成
./scripts/generate_data.sh spider

# 短程训练 (示例)
python sql_rl_gen/generation/sql_generation.py \
  --model_name_or_path juierror/flan-t5-text2sql-with-schema-v2 \
  --dataset example_text2sql_spider_train \
  --template llama3 \
  --steps_n 200 \
  --dataset_name spider \
  --outdir ./output/model_spider_train_smoke

# 评估
python sql_rl_gen/generation/evaluate_model.py \
  --model_name_or_path juierror/flan-t5-text2sql-with-schema-v2 \
  --dataset example_text2sql_spider_dev \
  --template llama3 \
  --trained_agent_path ./output/model_spider_train_smoke/best \
  --outdir ./output/model_spider_eval_smoke \
  --dataset_name spider

# Eureka 奖励搜索（示例：2 样本 1 迭代）
python sql_rl_gen/eureka_sql.py env=sql_query_generator sample=2 iteration=1 model='llama3.2:3b' temperature=0.2

# 更高质量搜索（推荐在 GPU Ollama）
python sql_rl_gen/eureka_sql.py env=sql_query_generator sample=3 iteration=2 model='llama3.2:3b' temperature=0.2
```

## 12. 当前修改摘要 (Recent Internal Adjustments)
- 修复路径：`configs/config.py`, `configs/data_args.py` 改为仓库相对路径。
- 优化 Eureka：统一大小写日志判定，聚合评估结果，避免误判 stuck。
- 增强可扩展性建议：奖励函数抽象、Prompt 结构化。

## 13. 结语
此文档可作为继续调试与扩展的基础索引。建议首先解决 GPU 推理性能，再做奖励函数稳定化与指标扩展；最终可接入更多数据集与更强模型形成通用 Text-to-SQL RL 研究平台。
