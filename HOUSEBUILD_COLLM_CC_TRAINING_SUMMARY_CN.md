# HouseBuild CoLLM-CC 训练细节总结

本文只总结当前目标 CoLLM-CC，不把 MAGRPO baseline 当主线。

## 算法

源码中的 CoLLM-CC 对应 `MAACTrainer`：

```text
house_build/train/train_maac.py
house_build/configs/house_build_maac_config.yaml
```

结构是 2 个 actor LLM + 1 个 centralized critic LLM。critic 输入由所有 agent 的 prompt 拼成 joint prompt；当前配置 `critic_type: v`，所以 critic 学的是 joint history value，不把 joint action 文本拼进 critic 输入。actor 是 decentralized execution，训练时 shared critic 提供 value/advantage 信号。

actor 和 critic 是一起训练的。每次更新里 actor loss 先反传并 `agent_optimizer.step()`，shared critic 的 value loss 再反传并 `critic_optimizer.step()`。

## 模型

源码复现默认：

```text
actor agent 0: Qwen/Qwen3-4B-Instruct-2507
actor agent 1: Qwen/Qwen3-4B-Instruct-2507
critic:        Qwen/Qwen3-4B-Instruct-2507
dtype:         bf16
```

Slurm 默认用 3 GPU：`cuda:0` actor0，`cuda:1` actor1，`cuda:2` centralized critic。

## 数据集

本仓库 HouseBuild 数据在：

```text
house_build/dataset/data.json
```

默认 split：

```text
train: HouseBuild[:8]   共 8 条 task
eval:  HouseBuild[8:]   共 2 条 task
total: 10 条 task
```

每条 task 是一个房屋蓝图，包含 `inventory`、多层 y-axis slices、bbox、资源限制和 spider/player 数值设定。

## 训练步数

源码默认：

```text
num_train_epochs: 150
num_turns:        4
num_agents:       2
train tasks:      8
```

没有 early stop 的上限估算：

```text
joint turn steps = 150 epochs * 8 tasks * 4 turns = 4800
actor samples    = 4800 joint steps * 2 agents = 9600
```

实际步数可能更少，因为 multi-turn MAAC 支持 `early_termination_threshold: 0.0`，当某一 turn 平均 reward 超过阈值时，该 task 的后续 turn 会提前停止。

## 关键超参

按源码默认 CoLLM-CC/MAAC 配置：

```text
agent_learning_rate:  5e-6
critic_learning_rate: 5e-6
value_loss_coef:      0.6
discount:             0.9
rollout_buffer_size:  1
train_batch_size:     1
max_new_tokens:       512
eval_interval:        10
eval_num_samples:     2
```

当前公开 `comlrl==1.3.7` 多轮 MAAC 要求 `num_generations == 1`。论文附录 Minecraft 超参里写过 `num_generations=2`、CoLLM-CC epochs/lr 也和本地 upstream 配置存在差异；这里按源码可运行配置优先。

## Token 规模

用 `Qwen/Qwen3-4B-Instruct-2507` tokenizer 对本仓库 HouseBuild prompt 的离线统计：

```text
首轮单 agent prompt tokens:
  min / mean / p50 / max = 1107 / 1166.5 / 1171 / 1234

短响应 4-turn prompt tokens:
  每个 agent 约 4933 prompt tokens / episode

若每轮输出打满 max_new_tokens=512:
  每个 agent 约 6981 tokens / episode
  两个 actors 合计约 13962 tokens / episode
```

critic 是 centralized critic，`critic_type=v` 时每个 turn 的 critic prompt 约等于两个 agent prompt 拼接再加少量 `[Agent i]` 标签，因此 critic 单次输入大致是 actor 单次 prompt 的 2 倍量级。

## 奖励和反馈

reward 是 Python 模拟环境给的 verifiable reward，不连接真实 Minecraft server：

```text
model outputs Minecraft-style commands
-> Python parses /fill and /kill
-> validate bbox, allowed blocks, resource limits
-> simulate block map
-> compare against target house
-> score_mean - spider penalty
```

multi-turn 使用 `external.mode: score_feedback`，每轮把上一轮 reward 和上一轮命令拼回下一轮 prompt。

