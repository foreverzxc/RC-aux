# RC-aux + Planner — 修改说明

基于 https://github.com/Guang000/RC-aux，增加了 PlannerDecoder 训练和可视化。

## 环境

RC-aux 使用独立的 uv 虚拟环境（Python 3.10.20，所有依赖已安装）。

```bash
cd /root/RC-aux
source .venv/bin/activate
```

关键包版本：
| 包 | 版本 | 备注 |
|---|------|------|
| torch | 2.12.0+cu130 | CUDA 13.0 |
| transformers | 5.5.4 | **必须此版本**，5.9.0 会导致 `ViTEncoder` 属性错误 |
| stable-worldmodel | 0.0.6 | 包含 Cube/PushT 环境和 HDF5Dataset |
| stable-pretraining | 0.1.6 | DataModule, Compose, Callback |
| lightning | 2.6.5 | |
| hydra | 1.3.2 | |

如果缺包，用 `uv pip install <pkg>` 安装。

## 目录结构

```
RC-aux/
├── CLAUDE.md                        # 本文件
├── .venv/                           # uv 虚拟环境（Python 3.10）
├── planner.py                       # PlannerDecoder + PlannerLoss + planner_rollout
├── train_planner.py                 # Planner 训练入口（独立 Lightning，不依赖 spt.Module）
├── train.py                         # WM 训练入口（ablation 实验用）
├── checkpoints/                     # → /root/autodl-tmp/rcaux-checkpoints/ (symlink)
│   ├── lewm_cube_object.ckpt        # LE-WM 预训练权重
│   └── rcaux_cube_object.ckpt       # RC-aux 预训练权重
├── data/                            # → /root/autodl-fs/ (symlink，数据集)
├── experiments/archive/             # 历史实验快照
├── config/train/
│   ├── lewm.yaml                    # 基础配置（全部训练共享）
│   └── planner_ft.yaml             # Planner 训练配置（继承 lewm.yaml）
├── scripts/viz/                     # 可视化脚本
└── Makefile                         # Ablation 实验快捷命令
```

### Symlink 说明

| 路径 | 目标 | 用途 |
|------|------|------|
| `checkpoints/` | `/root/autodl-tmp/rcaux-checkpoints/` | 预训练权重 + 训练输出权重 |
| `data/` | `/root/autodl-fs/` | 数据集（cube_single_expert.h5 等） |

这样在不同服务器上只需修改 symlink 目标即可兼容。

## 输出目录

所有训练输出写入两个位置：

| 内容 | 路径 |
|------|------|
| 模型权重 (checkpoints) | `checkpoints/` → `/root/autodl-tmp/rcaux-checkpoints/` |
| Hydra 日志/配置 | `/root/autodl-tmp/rcaux-outputs/<YYYY-MM-DD_HH-MM-SS>/` |

Hydra 输出目录由 `config/train/lewm.yaml` 中的 `hydra.run.dir` 控制。
每次训练自动创建时间戳子目录，包含 `.hydra/` 配置快照和 `lightning_logs/`。

## 运行方式

### Planner 训练

```bash
cd /root/RC-aux
source .venv/bin/activate

python train_planner.py --config-name planner_ft data=ogb \
    planner.ckpt_path=checkpoints/rcaux_cube_object.ckpt \
    trainer.max_epochs=200 wandb.enabled=false max_samples=50
```

- `checkpoints/` 是 symlink，权重实际保存在 `/root/autodl-tmp/rcaux-checkpoints/`
- `HDF5_PLUGIN_PATH` 由 `train_planner.py` 自动设置，无需手动 export

### 可视化

```bash
source .venv/bin/activate

python scripts/viz/run.py cube \
    --wm checkpoints/rcaux_cube_object.ckpt \
    --planner checkpoints/planner_overfit_cube.pt --idx 58794
```

输出文件：
- `output/viz/planner_cube_idx{idx}.gif` — 6 列动画对比
- `output/viz/planner_cube_idx{idx}_final.png` — 最终帧静态对比

## 数据集

Cube 数据通过 symlink 访问：

```bash
mkdir -p ~/.stable_worldmodel/datasets/ogbench
ln -sf /root/autodl-fs/cube_single_expert.h5 ~/.stable_worldmodel/datasets/ogbench/cube_single_expert.h5
```

或直接用项目内的 `data/` symlink：`data/cube_single_expert.h5`

## 关键设计决策

### 为什么不用 `spt.Module`
`spt.Module` 在 `training_step` 中调用 `self.manual_backward(state["loss"])`，但 loss 从
`planner_forward` 返回的 dict 中取出后 `requires_grad=False`（怀疑是 dict 遍历时触发副作用）。
改用纯 `pl.LightningModule` + `automatic_optimization=False` + 手动 `manual_backward` 解决。

### 为什么强制 `self.wm.eval()`
Lightning 在每个 epoch 开始时调用 `module.train()`，递归设置所有子模块为 training mode。
WM 的 projector/pred_proj 含 BatchNorm1d，train mode 会用 batch 统计量替代 running stats，
使 embedding 漂移。`PlannerLightningModule.train()` 覆盖为 `wm.eval()` 保持 WM 冻结。

### 为什么 WM 加载到 cuda 而非 cpu
RC-aux checkpoint 是 `torch.save(model, path)` 的完整对象，加载到 cpu 后再被 Lightning
移到 GPU 会导致梯度链断裂（`requires_grad` 丢失）。直接 `map_location="cuda"` 解决。

### 为什么单独写 `planner_ft.yaml`
Planner 参数（num_queries, horizon, action_substeps 等）和 WM 训练参数不应混在一起。
`planner_ft.yaml` 通过 `defaults: - lewm` 继承所有基础参数，只覆盖 planner 和 loader。

## 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| `transformers>=5.9.0` 导致 `ViTEncoder` 反序列化失败 | 已修复 | 锁定 `transformers==5.5.4` |
| HDF5 blosc 压缩插件找不到 | 已修复 | `train_planner.py` 自动设置 `HDF5_PLUGIN_PATH` |
| `pl.callbacks.ModelCheckpoint` 导入失败 (lightning 2.6.x) | 已修复 | 改用显式 `import lightning.pytorch.callbacks` |