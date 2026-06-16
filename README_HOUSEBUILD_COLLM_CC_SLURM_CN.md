# HouseBuild CoLLM-CC 在 Slurm SuperPOD 上复现

本文档是当前主目标：复现论文中的 HouseBuild CoLLM-CC。MAGRPO 只作为同仓库 baseline，入口见 `README_HOUSEBUILD_SLURM_CN.md`。

## 1. 源码对应关系

论文里 CoLLM-CC 指 centralized critic 的 multi-agent actor-critic。公开源码中没有单独叫 `train_collm_cc.py` 的文件；HouseBuild CoLLM-CC 对应：

```text
house_build/train/train_maac.py
house_build/configs/house_build_maac_config.yaml
comlrl.trainers.actor_critic.MAACTrainer
```

核心结构是 2 个 decentralized actor agents + 1 个 shared centralized critic。当前 Slurm 入口只做运行包装，不改算法实现。

## 2. 默认训练配置

源码默认 HouseBuild CoLLM-CC 配置：

```text
agent_model:  Qwen/Qwen3-4B-Instruct-2507
critic_model: Qwen/Qwen3-4B-Instruct-2507
train split:  HouseBuild[:8]
eval split:   HouseBuild[8:]
num_agents:   2
num_turns:    4
num_generations: 2
critic_type:  v
epochs:       120
actor lr:     5e-6
critic lr:    3e-6
rollout_buffer_size: 1
train_batch_size:    1
max_new_tokens:      512
eval_num_samples:    2
```

注意：HouseBuild CoLLM-CC 入口使用 `PaperAlignedMAACTrainer`，已覆盖上游 `comlrl` 多轮 MAAC 对 `num_generations == 1` 的旧限制。当前配置使用 `num_turns=4`、`num_generations=2`，按 aligned joint generations 分支收集 multi-turn rollouts，并使用 shared centralized critic 更新。

## 3. 创建环境

```bash
git clone https://github.com/rrui05/LLM_Collab_Minecraft.git
cd LLM_Collab_Minecraft

module load Anaconda3
conda create -y -n llm-collab-mc python=3.11 pip
conda activate llm-collab-mc
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install comlrl==1.3.7 datasets wandb accelerate transformers pyyaml
```

如果集群 PyTorch CUDA 版本不同，按集群镜像安装对应 wheel。

## 4. 预检

预检不加载 4B 权重，只检查配置、数据集、CoMLRL 参数、feedback 链路和可见 GPU 数：

```bash
python scripts/preflight_housebuild_collm_cc.py \
  --config house_build/configs/house_build_maac_config.yaml \
  --expect-cuda-devices 3 \
  --token-stats
```

无 GPU 的登录节点可以先去掉 `--expect-cuda-devices 3`。

## 5. 提交 Slurm

默认 1 节点 3 GPU：`cuda:0/1` 放两个 actor，`cuda:2` 放 centralized critic。

```bash
bash scripts/submit_housebuild_collm_cc_slurm_job.sh \
  configs/cluster/house_build_collm_cc_3gpu.env
```

干跑检查：

```bash
bash scripts/submit_housebuild_collm_cc_slurm_job.sh \
  configs/cluster/house_build_collm_cc_3gpu.env \
  DRY_RUN=true RUN_SUFFIX=dryrun
```

短 smoke：

```bash
bash scripts/submit_housebuild_collm_cc_slurm_job.sh \
  configs/cluster/house_build_collm_cc_3gpu.env \
  NUM_EPOCHS=1 TRAIN_SPLIT='[:1]' EVAL_SPLIT=None MAX_NEW_TOKENS=64 \
  EVAL_INTERVAL=0 RUN_SUFFIX=smoke
```

## 6. 产物

默认产物目录：

```text
runs/house_build_collm_cc/<run_name>/
  logs/
    stdout.log
    stderr.log
    env.txt
    pip-freeze.txt
    command.txt
    overrides.txt
    nvidia-smi.log
  output/
  final_model/
    agent_0/
    agent_1/
    critic/
```

`final_model/critic/value_head.pt` 是 centralized critic 的 value head。推理时通常只需要 actor agents；critic 用于训练。
