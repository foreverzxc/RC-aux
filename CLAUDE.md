# RC-aux + Planner — 修改说明

基于 https://github.com/Guang000/RC-aux，增加了 PlannerDecoder 训练和可视化。

## 环境

RC-aux 使用独立的 uv 虚拟环境（Python 3.10.20，所有依赖已安装）。

```bash
cd <PROJECT_ROOT>
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
├── checkpoints/                     # 训练输出权重（本地目录）
├── data/                            # 全是软链接 → LIBERO-datasets / stable_worldmodel
├── experiments/archive/             # 历史实验快照
├── config/train/
│   ├── lewm.yaml                    # 基础配置（全部训练共享）
│   └── planner_ft.yaml              # Planner 训练配置（继承 lewm.yaml）
├── scripts/viz/                     # 可视化脚本
├── output/viz/                      # 可视化输出（GIF/PNG）
└── Makefile                         # 实验快捷命令
```

### 数据集中存储

所有大规模数据建议统一放在一个目录下（本例为 `/home/cyborg/WM/LIBERO-datasets/`，其他机器自行调整）：

```
<DATA_ROOT>/
├── libero_10/          # LIBERO-10  数据集
├── libero_90/          # LIBERO-90  数据集
├── libero_goal/        # LIBERO-Goal 数据集
├── libero_object/      # LIBERO-Object 数据集（10 tasks 的 HDF5）
├── libero_spatial/     # LIBERO-Spatial 数据集
└── libero_assets/      # LIBERO 3D 资产（~408 MB）
```

`data/` 下的所有文件/目录都是**软链接**：

新机器上只需修改 `data/` 下的软链接目标即可兼容。

## 输出目录

所有训练输出写入两个位置：

| 内容 | 路径 |
|------|------|
| 模型权重 (checkpoints) | `checkpoints/`|
| Hydra 日志/配置 | `<OUTPUT_BASE>/<YYYY-MM-DD_HH-MM-SS>/`（由 `RC_OUTPUT_BASE` 环境变量控制） |

Hydra 输出目录由 `config/train/lewm.yaml` 中的 `hydra.run.dir` 控制。
每次训练自动创建时间戳子目录，包含 `.hydra/` 配置快照和 `lightning_logs/`。

## 运行方式

### Planner 训练

```bash
cd <PROJECT_ROOT>
source .venv/bin/activate

python train_planner.py --config-name planner_ft data=ogb \
    planner.ckpt_path=checkpoints/rcaux_cube_object.ckpt \
    trainer.max_epochs=200 wandb.enabled=false max_samples=50
```

- `checkpoints/` 可能是 symlink，在不同机器上指向不同存储位置
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

所有数据集通过 `data/` 下的软链接访问：

新建软链接：
```bash
ln -sf /path/to/actual/data data/<name>
```

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

## LIBERO 仿真环境

### 在新机器上安装

```bash
# 1. 安装 Python 包
source .venv/bin/activate
uv pip install libero robosuite==1.4.0

# 2. 下载 LIBERO 3D 资产到集中存储目录（~408 MB）
#    <DATA_ROOT> 是数据集统一存放目录，根据实际机器设置
#    国内机器可用 HF 镜像：export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('jadechoghari/libero-assets',
    local_dir='<DATA_ROOT>/libero_assets', max_workers=2)
"

# 3. 创建软链接（data/ 下全是软链接）
ln -sf <DATA_ROOT>/libero_assets data/libero_assets
ln -sf <DATA_ROOT>/libero_object data/libero_object

# 4. 创建 LIBERO 配置文件
LIBERO_PKG=$(python -c "import libero.libero; import os; print(os.path.dirname(os.path.abspath(libero.libero.__file__)))")
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << EOF
benchmark_root: ${LIBERO_PKG}
bddl_files: ${LIBERO_PKG}/bddl_files
init_states: ${LIBERO_PKG}/init_files
datasets: <DATA_ROOT>
assets: <DATA_ROOT>/libero_assets
EOF
```

### 使用示例

```python
import os
os.environ['MUJOCO_GL'] = 'egl'  # headless 渲染

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# 获取任务
BenchmarkCls = get_benchmark('libero_object')
b = BenchmarkCls()
task = b.get_task(0)

# 创建环境
env = OffScreenRenderEnv(
    bddl_file_name=b.get_task_bddl_file_path(0),
    camera_heights=128, camera_widths=128,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

obs = env.reset()
# obs['agentview_image']: (128, 128, 3) — 主摄像头 RGB
# obs['robot0_eye_in_hand_image']: (128, 128, 3) — 手眼摄像头
# obs['robot0_proprio-state']: (39,) — 本体感知
# obs['object-state']: (98,) — 物体状态
```

6GB 显存的 RTX 3050 可以跑仿真（渲染在 CPU/EGL 上），但 WM + 仿真同时加载时需注意显存管理。

## 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| `transformers>=5.9.0` 导致 `ViTEncoder` 反序列化失败 | 已修复 | 锁定 `transformers==5.5.4` |
| HDF5 blosc 压缩插件找不到 | 已修复 | `train_planner.py` 自动设置 `HDF5_PLUGIN_PATH` |
| `pl.callbacks.ModelCheckpoint` 导入失败 (lightning 2.6.x) | 已修复 | 改用显式 `import lightning.pytorch.callbacks` |