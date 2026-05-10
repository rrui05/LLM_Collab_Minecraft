# HouseBuild 在 Slurm SuperPOD 上复现

本文档面向你 fork 后的仓库：

```text
https://github.com/rrui05/LLM_Collab_Minecraft
```

目标先只复现 HouseBuild 的 MAGRPO 训练。当前 CoMLRL 的 `parallel_training=mp` 不是 FSDP/DeepSpeed 数据并行，而是把不同 agent/critic 分配到不同 GPU。HouseBuild MAGRPO 默认 `num_agents=2`，所以这里用 1 个节点、2 张 H800：Agent 0 放 `cuda:0`，Agent 1 放 `cuda:1`。

## 1. 准备代码

```bash
git clone https://github.com/rrui05/LLM_Collab_Minecraft.git
cd LLM_Collab_Minecraft
git status
```

如果集群需要代理、私有 Hugging Face cache 或统一 scratch 路径，先按集群规范设置环境变量。

## 2. 创建 conda 环境

先加载集群的 conda/anaconda 模块。不同 SuperPOD 命令可能不同，例如：

```bash
module load Anaconda3
```

然后创建环境：

```bash
bash scripts/setup_superpod_env.sh
```

如果集群的 conda/anaconda 需要通过 environment module 提供，也可以直接让脚本加载：

```bash
MODULES="Anaconda3" bash scripts/setup_superpod_env.sh
```

默认环境名是 `llm-collab-mc`。如果 H800 节点的驱动/集群镜像要求不同 CUDA wheel，可以改：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
CONDA_ENV=llm-collab-mc \
bash scripts/setup_superpod_env.sh
```

脚本会安装：

- `torch`
- `comlrl==1.3.7`
- `datasets`
- `wandb`
- `accelerate`
- `openai`

并打印 `torch.cuda.is_available()` 和 GPU 名称。

## 3. 配置 W&B

训练脚本默认连 W&B online。先登录：

```bash
conda activate llm-collab-mc
wandb login
```

也可以在提交任务时直接传：

```bash
export WANDB_API_KEY=你的key
```

默认 W&B project 是 `house_build`。如果你有自己的 entity：

```bash
export WANDB_ENTITY=你的wandb实体名
```

如果不设置 `WANDB_ENTITY`，sbatch 会把 `wandb.entity=None` 传给训练脚本，避免误用原 config 里的 `OpenMLRL`。

## 4. 提交 2 卡训练

先按你们集群改 Slurm 头部：

```bash
vim slurm/house_build_magrpo_2gpu.sbatch
```

重点确认：

- `#SBATCH --partition=...`
- `#SBATCH --account=...`
- `#SBATCH --gpus-per-node=2`
- `#SBATCH --constraint=h800` 是否需要
- `--mem` 和 `--time` 是否符合队列限制

提交默认 HouseBuild MAGRPO：

```bash
bash scripts/submit_housebuild_slurm_job.sh
```

也可以使用和 `rrui05/MAGRPO` 类似的配置文件入口：

```bash
bash scripts/submit_housebuild_slurm_job.sh configs/cluster/house_build_2gpu.env
```

如果 Slurm batch 作业内默认找不到 conda，不要在 base 里跑；提交时显式传 module 或 conda 初始化脚本：

```bash
bash scripts/submit_housebuild_slurm_job.sh MODULES=Anaconda3
```

或者：

```bash
bash scripts/submit_housebuild_slurm_job.sh CONDA_SETUP=/path/to/conda.sh
```

默认训练配置：

```text
model: Qwen/Qwen3-4B-Instruct-2507
algorithm: MAGRPO
task: HouseBuild
num_agents: 2
num_turns: 4
num_generations: 2
max_new_tokens: 512
epochs: 20
train split: [:8]
eval split: [8:]
parallel_training: mp
agent_devices: ['cuda:0', 'cuda:1']
wandb: enabled
save_final_model: true
```

## 5. 推荐先跑 smoke test

正式训练前可以先提交一个 Slurm dry run。它会进入 batch 环境、激活 conda、记录环境和生成完整训练命令，但不会启动训练：

```bash
bash scripts/submit_housebuild_slurm_job.sh DRY_RUN=true RUN_SUFFIX=dryrun
```

或：

```bash
bash scripts/submit_housebuild_slurm_job.sh configs/cluster/house_build_2gpu.env DRY_RUN=true RUN_SUFFIX=dryrun
```

第一次不要直接跑完整 4B 配置。建议先提交一个小模型 smoke test，验证环境、W&B、日志和数据链路：

```bash
bash scripts/submit_housebuild_slurm_job.sh \
  AGENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  RUN_SUFFIX=smoke \
  TRAIN_SPLIT='[:1]' \
  EVAL_SPLIT=None \
  NUM_EPOCHS=1 \
  NUM_TURNS=1 \
  MAX_NEW_TOKENS=32 \
  EVAL_INTERVAL=0 \
  SAVE_FINAL_MODEL=false
```

smoke test 通过后，再跑默认完整配置。

## 6. 常用覆盖参数

换模型：

```bash
bash scripts/submit_housebuild_slurm_job.sh AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct RUN_SUFFIX=qwen25_15b
```

改 epoch：

```bash
bash scripts/submit_housebuild_slurm_job.sh NUM_EPOCHS=5 RUN_SUFFIX=epoch5
```

改 turn 数：

```bash
bash scripts/submit_housebuild_slurm_job.sh NUM_TURNS=2 RUN_SUFFIX=turn2
```

改 W&B project：

```bash
bash scripts/submit_housebuild_slurm_job.sh WANDB_PROJECT=llm-collab-housebuild RUN_SUFFIX=wandb_test
```

## 7. 日志位置

每次运行会生成：

```text
runs/house_build/<run_name>/
  logs/
    stdout.log
    stderr.log
    env.txt
    pip-freeze.txt
    command.txt
    overrides.txt
    slurm_submit_command.txt
    nvidia-smi.log
  wandb/
  output/
  final_model/   # SAVE_FINAL_MODEL=true 时生成
```

Hugging Face cache 默认不放进每次 run 目录，而是复用：

```text
${SCRATCH}/llm_collab_minecraft/hf_home
```

如果集群没有 `SCRATCH`，会退回到仓库内 `.cache/llm_collab_minecraft/hf_home`。也可以提交前显式设置：

```bash
export HF_HOME=/path/to/shared/hf_home
export TRANSFORMERS_CACHE=/path/to/shared/hf_home/transformers
```

同时 Slurm 会在提交目录生成：

```text
slurm-hb-magrpo-2gpu-<jobid>.out
slurm-hb-magrpo-2gpu-<jobid>.err
```

建议训练中看：

```bash
tail -f runs/house_build/<run_name>/logs/stdout.log
tail -f runs/house_build/<run_name>/logs/stderr.log
tail -f runs/house_build/<run_name>/logs/nvidia-smi.log
```

## 8. 训练到底在做什么

这个仓库当前没有真实 Minecraft server/Mineflayer 运行闭环。HouseBuild 训练过程是：

```text
house_build/dataset/data.json
  -> 解析 target_spec.layers
  -> 转成 prompt 里的 y-axis slices / rectangles
  -> 两个 agent 分别生成 /fill 或 /kill 命令
  -> Python 校验命令合法性
  -> Python 模拟方块填充结果
  -> 与目标结构逐 voxel 比较
  -> 得到 score_match / iou / spider_penalty
  -> MAGRPO 用当前 rollout 更新模型
```

主要 reward：

```text
reward = score_match - spider_penalty
```

配置里还有：

```yaml
reward_processor:
  enabled: true
  scale_factor: 1.0
  shift: -1.0
```

因此训练器实际看到的 reward 会做 shift。

## 9. W&B 指标

训练会记录：

- turn 级别 reward/return
- HouseBuild reward function 中的 `iou`
- `level_1`: `score_match`
- `level_2`: `-spider_penalty`
- `level_total`: raw reward
- eval metrics

W&B run name 默认：

```text
house_build_magrpo_2gpu_<suffix>_<timestamp>
```

## 10. 失败排查

如果 W&B 报 entity 无权限：

```bash
export WANDB_ENTITY=
```

或显式指定你自己的 entity。

如果 Hugging Face 下载失败：

```bash
export HF_TOKEN=你的token
```

如果 4B 模型显存不足，先用：

```bash
AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

如果要完全离线跑 W&B：

```bash
bash scripts/submit_housebuild_slurm_job.sh WANDB_MODE=offline RUN_SUFFIX=offline_test
```

不过正式复现建议 `WANDB_MODE=online`，方便记录完整曲线和环境信息。

## 11. 训练后直接推理

训练脚本默认把最终模型保存为：

```text
runs/house_build/<run_name>/final_model/agent_0
runs/house_build/<run_name>/final_model/agent_1
```

我另外放了一个不在默认训练集 `[:8]`、也不在默认验证集 `[8:]` 里的全新任务：

```text
house_build/dataset/unseen_eval.json
```

提交 2 卡推理作业：

```bash
bash scripts/submit_housebuild_infer_slurm_job.sh
```

它默认会找最新的：

```text
runs/house_build/*/final_model
```

然后对 `unseen_eval.json` 做 4-turn 推理，输出两个 agent 的命令、每轮 raw reward、processed reward、match、iou，并把完整结果写到：

```text
runs/house_build_infer/<run_name>/results/inference.jsonl
```

如果要指定某次训练出的模型：

```bash
bash scripts/submit_housebuild_infer_slurm_job.sh \
  MODEL_DIR=/project/gzmcagent/LLM_Collab_Minecraft/runs/house_build/<run_name>/final_model \
  RUN_SUFFIX=unseen_test
```

如果想看默认 `data.json` 里的验证集 `[8:]`：

```bash
bash scripts/submit_housebuild_infer_slurm_job.sh \
  DATASET_JSON=/project/gzmcagent/LLM_Collab_Minecraft/house_build/dataset/data.json \
  SPLIT='[8:]' \
  RUN_SUFFIX=default_eval
```
