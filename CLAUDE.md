# RC-aux + Planner — 修改说明

基于 https://github.com/Guang000/RC-aux，增加了 PlannerDecoder 训练和可视化。

## 环境

RC-aux 使用独立的 uv 虚拟环境（Python 3.10.20，所有依赖已安装）。

```bash
cd /home/cyborg/WM/RC-aux
source .venv/bin/activate
```

关键包版本：
| 包 | 版本 | 备注 |
|---|------|------|
| torch | 2.6.0+cu124 | CUDA 12.4 |
| transformers | 5.5.4 | **必须此版本**，5.9.0 会导致 `ViTEncoder` 属性错误 |
| stable-worldmodel | 0.0.6 | 包含 Cube/PushT 环境和 HDF5Dataset |
| stable-pretraining | 0.1.6 | DataModule, Compose, Callback |
| lightning | 2.5.2 | |
| mujoco | 3.7.0 | |
| ogbench | 1.2.1 | Cube 仿真后端 |

如果缺包，用 `uv pip install <pkg>` 安装。

## 新增文件

```
RC-aux/
├── CLAUDE.md                        # 本文件
├── .venv/                           # uv 虚拟环境（Python 3.10）
├── planner.py                       # PlannerDecoder + PlannerLoss + planner_rollout（从 le-wm 复制）
├── train_planner.py                 # Planner 训练入口（独立 Lightning，不依赖 spt.Module）
├── models/                           # → my_models/RC-aux/rcaux/ (symlink，预训练权重)
├── checkpoints/
│   └── planner_overfit_cube.pt      # Cube 过拟合权重（700 epoch, 10 samples）
├── experiments/archive/             # 历史实验快照（environment/requirements/loss）
├── config/train/planner_ft.yaml     # Planner 训练配置（继承 lewm.yaml）
├── scripts/
│   └── viz/
│       ├── __init__.py
│       ├── core.py                  # 通用可视化管线
│       ├── pusht.py                 # PushT 环境适配
│       ├── cube.py                  # Cube 环境适配（MuJoCo state + 目标隐藏）
│       └── run.py                   # 统一入口
└── output/viz/                      # 可视化输出（gitignored）
```

## 运行方式

### Planner 训练

```bash
cd /home/cyborg/WM/RC-aux
source .venv/bin/activate

python train_planner.py --config-name planner_ft data=ogb \
    planner.ckpt_path=models/checkpoints/pixel_control/cube_rcaux/rcaux_cube_object.ckpt \
    trainer.max_epochs=200 wandb.enabled=false max_samples=50
```

### 可视化

```bash
source .venv/bin/activate

# 训练集样本
python scripts/viz/run.py cube \
    --wm models/checkpoints/pixel_control/cube_rcaux/rcaux_cube_object.ckpt \
    --planner checkpoints/planner_overfit_cube.pt --idx 58794

# 未见过样本
python scripts/viz/run.py cube \
    --wm .../cube_rcaux/rcaux_cube_object.ckpt \
    --planner checkpoints/planner_overfit_cube.pt --idx 405000
```

输出文件：
- `output/viz/planner_cube_idx{idx}.gif` — 6 列动画对比（Dataset | GT | Plan#7 | Plan#0 | Plan#4 | Plan#5）
- `output/viz/planner_cube_idx{idx}_final.png` — 最终帧静态对比

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

## Cube 仿真环境修复

Cube 环境（`swm/OGBCube-v0`）的数据重放需要以下特殊处理：

| 问题 | 修复 |
|------|------|
| qpos 中 cube free joint 被清零 | 从 `privileged_block_0_pos/quat` 补入 qpos[6:13] |
| Quaternion 格式 xyzw vs MuJoCo wxyz | 转换：`qw=bq[3], qx=bq[0], qy=bq[1], qz=bq[2]` |
| 每次 `gym.make()` 第一帧渲染不一致 | `make_env()` 创建单例，reset + render 预热后共享 |
| 仿真出现目标 ghost cube | `_render_goal=False` + geom alpha=0 + mocap_pos=[0,0,-999] |
| 各列初始状态不一致 | 共用同一 env 实例，`reset()` + 状态设置之间 `_hide_targets()` |

## Checkpoint 路径

预训练权重通过 symlink `models/ → my_models/RC-aux/rcaux/` 访问：

| 任务 | 路径 |
|------|------|
| Cube | `models/checkpoints/pixel_control/cube_rcaux/rcaux_cube_object.ckpt` |
| Wall | `models/checkpoints/pixel_control/wall_rcaux/rcaux_wall_object.ckpt` |
| Reacher | `models/checkpoints/pixel_control/reacher_rcaux/rcaux_reacher_object.ckpt` |
| TwoRoom | `models/checkpoints/pixel_control/tworoom_rcaux/rcaux_tworoom_weights.ckpt` |

## 数据集路径

Cube 数据需要 symlink：
```bash
mkdir -p ~/.stable_worldmodel/ogbench
ln -s ~/.stable_worldmodel/cube_single_expert.h5 ~/.stable_worldmodel/ogbench/cube_single_expert.h5
```

## 实验结果

### Cube 过拟合（10 samples）

| Epoch | Train Loss | Query 7 WM cost | Query 7 Sim cost |
|-------|-----------|-----------------|------------------|
| 200 | 0.009 | 0.558 | **0.644** ← 最佳 |
| 700 | 0.005 | 0.587 | 0.817 |

WM loss 下降但 Sim cost 上升 → Planner 过拟合到 WM embedding 空间。
这恰好验证了 RC-aux 核心论点：WM latent distance ≠ 物理可达性。

### 添加新仿真环境

在 `scripts/viz/` 下创建 `xxx.py`，只需提供：
- `RAW_ACT`, `FS`, `CTX_LEN`, `DATASET_NAME`
- `make_env()` → 创建并预热环境
- `run_sim(env, *state, actions_2d)` → 从给定状态运行仿真
- `get_state(ds_raw, idx)` → 提取初始状态
- `get_gt_actions(item_raw, horizon)` → 提取 GT 动作

然后 `python scripts/viz/run.py xxx --wm ... --planner ...` 即可。
