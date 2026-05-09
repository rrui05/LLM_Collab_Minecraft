# LLM_Collab_Minecraft 本地复现说明

仓库：<https://github.com/OpenMLRL/LLM_Collab_Minecraft>

当前本地 commit：

```text
584cf6f v0
```

这份说明记录的是我在你这台 Windows 机器上实际验证过的复现路径，以及这个仓库的训练机制。重点结论先放前面：

- 这个仓库当前版本没有连接真实 Minecraft server。
- README 里写了 `npm install`，但仓库根目录没有 `package.json`，源码里也没有 Mineflayer、Node bot、socket、localhost/server 连接逻辑。
- 训练环境是 Python 写的“Minecraft 指令模拟打分环境”：模型输出 MC 风格命令，Python 校验命令、模拟方块状态、计算 reward。
- 它仍然可以是 on-policy RL：因为每次训练 step 都用当前 policy 现场生成动作，再用当前动作得到 reward/return，然后马上更新当前 policy。
- 我确实在你电脑上下载了一个小 Qwen 模型做 smoke test：`Qwen/Qwen2.5-0.5B-Instruct`，Hugging Face 缓存约 `0.931 GB`。没有保存训练后的 checkpoint。

## 1. 本地环境

不要用 base。已经创建并验证了新的 conda 环境：

```powershell
conda create -y -n llm-collab-mc python=3.11 pip
conda activate llm-collab-mc
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install comlrl==1.3.7 datasets wandb accelerate openai
```

已验证：

```text
Python: C:\Users\Win11\anaconda3\envs\llm-collab-mc\python.exe
Python version: 3.11.15
Torch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5060
CoMLRL: 1.3.7
Transformers: 5.8.0
```

PowerShell 里建议直接用该环境的 Python 绝对路径，避免 Windows 上并行 `conda run` 偶发临时文件锁：

```powershell
$PY="C:\Users\Win11\anaconda3\envs\llm-collab-mc\python.exe"
```

验证环境：

```powershell
& $PY -c "import torch, comlrl, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(getattr(comlrl,'__version__','no_version'), transformers.__version__)"
```

我验证到的输出要点：

```text
2.11.0+cu128 True NVIDIA GeForce RTX 5060
1.3.7 5.8.0
```

## 2. 它有没有连接真实 Minecraft？

没有。至少当前仓库 snapshot 没有。

我检查过：

- README 只有一处 `npm install`。
- 根目录没有 `package.json`。
- 全仓库搜索不到 Mineflayer、Minecraft server、Node bot、socket、localhost、subprocess 启动 MC 等逻辑。
- 训练脚本只 import Python 依赖：`datasets`、`transformers`、`torch`、`comlrl` 和本仓库的 `utils/rewards/external`。

因此这里的“Minecraft environments”不是运行一个真实 MC 客户端/服务端，而是用 Minecraft 命令格式和方块规则做了一个轻量 Python 模拟环境。

## 3. 没有真实 MC，它怎么训练？

核心链路是：

```text
dataset task
  -> prompt formatter
  -> 当前 LLM policy 生成 Minecraft 风格命令
  -> Python 解析命令
  -> Python 校验命令是否合法
  -> Python 模拟命令执行后的方块状态
  -> Python 对比目标结构并算 reward
  -> CoMLRL trainer 用当前 rollout 的 return 更新模型
```

### StrBuild

数据位置：

```text
str_build/dataset/data.csv
```

数据是字符串任务，比如：

```csv
string,difficulty
"I",1
"C",1
"M",1
```

`str_build/utils/str_builder.py` 里把字母渲染成 5x5 字符网格。例如字母 `I` 会变成目标 mask，再映射到局部坐标。

模型需要输出：

```text
/setblock <x> <y> <z> <block>
```

Python 环境做这些事：

- `extract_command_lines`：从模型输出里提取命令，支持去掉 markdown fence/list bullet。
- `validate_and_normalize_mc_commands`：只接受 `/setblock`，要求绝对整数坐标，不允许 `~`，坐标必须在 bbox 内，block 必须在该 agent 允许列表里。
- `simulate_commands_to_scan_blocks`：不用 MC，直接用 Python dict 维护 `(x,y,z)->block` 状态。
- `score_str_builder`：计算覆盖率、额外方块惩罚、相邻同色惩罚、是否用到各 agent palette 等。

StrBuild reward 大致是：

```text
score_total = 2.0 * coverage_ratio
              - 1.5 * extra_ratio
              - 1.0 * adjacent_same_color_ratio
              - missing_palette_penalty
```

所以它训练的是“让模型学会输出正确的 `/setblock` 命令”，不是训练一个真实 MC 里的机器人移动、放块、避障。

### HouseBuild

数据位置：

```text
house_build/dataset/data.json
```

数据里包含：

- `inventory`：字符到方块的映射，例如 `C=white_concrete`、`O=obsidian`、`A=air`。
- `target_spec.layers`：按 y 层描述房子的目标结构。
- `bbox`/`size`：结构范围。

模型需要输出：

```text
/fill <x1> <y1> <z1> <x2> <y2> <z2> <block>
/kill ...
```

Python 环境做这些事：

- 只接受 `/fill` 和 `/kill`。
- `/fill` 必须使用绝对整数坐标，不能带 `replace/keep` 等 fill mode。
- 坐标必须在 bbox 内。
- block 必须在 allowed blocks 内。
- 如果 `limited_resource=true`，会按目标结构计算每个 agent 的资源上限，超量命令被拒绝。
- `simulate_commands_to_scan_blocks` 直接展开 fill 区域，更新 Python dict 里的方块状态。
- `score_house_builder` 对比模拟结果和目标结构，算 exact match 比例和 IoU。
- spider/player 不是一个真实战斗环境，而是 reward 里的数值惩罚逻辑：如果配置里有 spider damage，且 agent 没有输出 `/kill`，reward 会扣最多 `0.1`。

HouseBuild reward 主体是：

```text
reward = score_match - spider_penalty
```

其中 `score_match` 是 bbox 内每个位置是否完全匹配目标方块的比例。

## 4. on-policy 具体体现在哪里？

CoMLRL 的 `MAGRPOTrainer` 在 `comlrl/trainers/reinforce/magrpo.py` 里做 on-policy 采样和更新。

核心过程：

1. `train()` 每个 epoch 遍历当前 train dataset。
2. 对每个 batch item，调用 `_train_step_returns()`。
3. `_train_step_returns()` 里的 `build_node()` 用当前 agent model 生成 `num_generations` 个 completion。
4. 当前 completion 立刻传给 reward function 得到 reward。
5. 多 turn 时，`external_transition` 基于上一轮 completion 和模拟反馈生成下一轮 prompt，再继续生成。
6. rollout tree 完成后，递归计算 return：

```text
return_t = reward_t + discount * mean(child_returns)
```

7. 每个 agent 的样本进入 rollout buffer。
8. `_update_from_samples()` 对这些刚采样出来的样本做反传更新。
9. loss 是 policy-gradient 风格：

```text
loss = - sequence_log_prob * advantage
```

这就是 on-policy：reward/return 来自当前 policy 现场生成的样本，而不是固定离线轨迹。这里的“环境 step”不是 MC server step，而是“生成命令 -> Python 模拟打分”的一次交互。

## 5. 多 turn external feedback 是什么？

当 `num_turns > 1` 时，CoMLRL 要求传入 `external_transition`。这个仓库在训练脚本里注册了 context resolver，然后由：

```text
str_build/external/*
house_build/external/*
```

生成下一轮 prompt。

它不是调用外部大模型，也不是连 MC。它是规则化反馈模块。

StrBuild 支持：

- `perfect_feedback`
- `position_feedback`
- `score_feedback`

HouseBuild 支持：

- `perfect_feedback`
- `position_feedback`
- `position_modification`
- `rect_modification`
- `resource_schedule`
- `score_feedback`

例子：`score_feedback` 会根据上一轮 completion 重新跑 Python reward，然后把 `Previous reward: ...` 写进下一轮 prompt。`position_feedback` 会列出目标坐标或需要关注的位置。`previous_response=true` 时，还会把 agent 上轮输出附到下一轮 prompt 里。

所以 multi-turn 训练本质是：模型输出命令，Python 规则环境评价/总结，再把反馈拼回 prompt，让模型下一轮改进。

## 6. 我在你电脑上到底下载和训练了什么？

我为了验证复现链路，下载了两个 Hugging Face 模型缓存：

```text
C:\Users\Win11\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct
大小约 0.931 GB

C:\Users\Win11\.cache\huggingface\hub\models--sshleifer--tiny-gpt2
大小约 6.09 MB
```

真正跑通 StrBuild 和 HouseBuild smoke test 的是：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

我没有用 README 默认的：

```text
Qwen/Qwen3-4B-Instruct-2507
```

原因是你这张 RTX 5060 只有 8GB 显存，默认配置是 2 agents、4 turns、较长生成、20/150 epochs，完整跑 4B 级模型和 IAC/MAAC critic 非常容易爆显存。烟测目标是验证代码链路，不是复现实验指标。

我跑的 smoke test 做了真实训练 step：

- 加载 Qwen2.5-0.5B-Instruct。
- 用 GPU `cuda:0`。
- 取 1 条数据。
- 生成 2 个 samples。
- 算 reward。
- 做一次 policy-gradient 更新。

但命令里设置了：

```text
save_final_model=false
wandb.enabled=false
```

所以：

- 模型权重确实在进程内被更新过。
- 进程结束后没有保存训练后的 checkpoint。
- 仓库里没有生成 `output*` 目录。
- 你的磁盘上留下的是 Hugging Face 原始模型缓存，不是训练产物。

如果要删除我下载的模型缓存：

```powershell
Remove-Item -Recurse -Force "C:\Users\Win11\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
Remove-Item -Recurse -Force "C:\Users\Win11\.cache\huggingface\hub\models--sshleifer--tiny-gpt2"
```

如果要删除 conda 环境：

```powershell
conda remove -y -n llm-collab-mc --all
```

## 7. 已验证 smoke test

下面两个命令已经在 `llm-collab-mc` + RTX 5060 上通过，退出码 0。

### StrBuild MAGRPO

```powershell
& $PY str_build\train\train_magrpo.py --config str_build\configs\str_build_magrpo_config.yaml --override `
  "agent_model.name='Qwen/Qwen2.5-0.5B-Instruct'" `
  "agent_model.dtype='bf16'" `
  "magrpo.num_agents=1" `
  "magrpo.num_turns=1" `
  "magrpo.num_train_epochs=1" `
  "magrpo.num_generations=2" `
  "magrpo.max_new_tokens=4" `
  "magrpo.rollout_buffer_size=1" `
  "magrpo.train_batch_size=1" `
  "magrpo.eval_interval=0" `
  "magrpo.agent_devices='cuda:0'" `
  "dataset.train_split='[:1]'" `
  "dataset.eval_split=None" `
  "wandb.enabled=false" `
  "output.verbose=false" `
  "reward_processor.enabled=false"
```

### HouseBuild MAGRPO

```powershell
& $PY house_build\train\train_magrpo.py --config house_build\configs\house_build_magrpo_config.yaml --override `
  "agent_model.name='Qwen/Qwen2.5-0.5B-Instruct'" `
  "agent_model.dtype='bf16'" `
  "magrpo.num_agents=1" `
  "magrpo.num_turns=1" `
  "magrpo.num_train_epochs=1" `
  "magrpo.num_generations=2" `
  "magrpo.max_new_tokens=4" `
  "magrpo.rollout_buffer_size=1" `
  "magrpo.train_batch_size=1" `
  "magrpo.eval_interval=0" `
  "magrpo.agent_devices='cuda:0'" `
  "dataset.train_split='[:1]'" `
  "dataset.eval_split=None" `
  "wandb.enabled=false" `
  "output.verbose=false" `
  "reward_processor.enabled=false"
```

注意：`sshleifer/tiny-gpt2` 可以跑 StrBuild smoke test，但不适合 HouseBuild，因为 HouseBuild prompt 会超过 GPT-2 的 1024 token 上下文，导致生成失败。HouseBuild 至少要用长上下文 instruction model。

## 8. 完整训练命令

README 默认完整训练如下。

StrBuild:

```powershell
& $PY str_build\train\train_magrpo.py --config str_build\configs\str_build_magrpo_config.yaml
& $PY str_build\train\train_iac.py    --config str_build\configs\str_build_iac_config.yaml
& $PY str_build\train\train_maac.py   --config str_build\configs\str_build_maac_config.yaml
```

HouseBuild:

```powershell
& $PY house_build\train\train_magrpo.py --config house_build\configs\house_build_magrpo_config.yaml
& $PY house_build\train\train_iac.py    --config house_build\configs\house_build_iac_config.yaml
& $PY house_build\train\train_maac.py   --config house_build\configs\house_build_maac_config.yaml
```

但这不是我建议你在 8GB 5060 上直接跑的命令。默认 config 里：

- 模型是 `Qwen/Qwen3-4B-Instruct-2507`。
- MAGRPO 默认 2 agents、4 turns、20 epochs。
- IAC/MAAC 默认 150 epochs，并且会涉及 critic model。
- HouseBuild prompt 更长，显存和上下文压力更大。

本机建议按下面顺序放大：

1. 保持 `Qwen/Qwen2.5-0.5B-Instruct`，先把 `dataset.train_split='[:1]'` 放到 `[:8]`。
2. 把 `max_new_tokens=4` 放到 64，再到 128，再到 512。
3. 把 `num_turns=1` 放到 2，再到 4。
4. 把 `num_agents=1` 放到 2。
5. 再增加 epochs。
6. 最后才换更大的模型。

## 9. 如何保存训练后的模型

默认 config 里：

```yaml
output:
  save_final_model: false
```

如果你要真的保存训练结果，必须改成：

```powershell
"output.save_final_model=true" "output.save_path='output_magrpo_str_build/final_model'"
```

示例：

```powershell
& $PY str_build\train\train_magrpo.py --config str_build\configs\str_build_magrpo_config.yaml --override `
  "agent_model.name='Qwen/Qwen2.5-0.5B-Instruct'" `
  "agent_model.dtype='bf16'" `
  "magrpo.agent_devices='cuda:0'" `
  "dataset.train_split='[:8]'" `
  "wandb.enabled=false" `
  "output.save_final_model=true" `
  "output.save_path='output_magrpo_str_build/final_model'"
```

## 10. 复现实验指标还缺什么？

这个仓库能跑，但要复现实验曲线/论文结果，还需要确认这些外部信息：

- 论文或作者实际使用的模型 checkpoint。
- CoMLRL 版本是否就是 PyPI `1.3.7`，还是某个 Git commit。
- 训练硬件和显存。
- wandb project 里的原始 run config。
- 是否有未公开的脚本，例如批量 seed、SLURM、评估汇总脚本。
- README 里的 `npm install` 是否来自一个旧版本/未上传的 Mineflayer 环境。

当前仓库本身已经足够复现“Python 模拟 Minecraft 指令环境上的 on-policy CoMLRL 训练链路”；不足以证明它连接真实 Minecraft。

