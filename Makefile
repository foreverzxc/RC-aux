# RC-aux: WM ablation experiments + Planner training
# Usage:
#   make setup                       — create checkpoints/ and data/ symlinks
#   make all                         — run all 3 WM ablations sequentially
#   make ablation_baseline           — single ablation
#   make train-planner               — train PlannerDecoder
#   make clean                       — remove caches and training outputs

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python

# The autodl platform sets OMP_NUM_THREADS=0 which triggers
# "libgomp: Invalid value for environment variable OMP_NUM_THREADS"
# on every Python invocation. Un-export it so the warning is silent.
unexport OMP_NUM_THREADS

DATA ?= libero_goal
WANDB ?= false

# ---- Symlink setup (paths configurable per machine) ----
CHECKPOINT_DIR ?= /root/autodl-tmp/rcaux-checkpoints
DATA_DIR ?= /root/autodl-fs

# ---- WM training (ablation) ----
TRAIN_SCRIPT := train.py
ABLATION_TARGETS := ablation_baseline ablation_dir ablation_dir_ss

.PHONY: all setup train-planner $(ABLATION_TARGETS) clean

all: $(ABLATION_TARGETS)

# ---- Symlink setup ----
setup:
	@echo "=== Setting up symlinks ==="
	@[ -L checkpoints ] || [ -d checkpoints ] || (rm -rf checkpoints && ln -s $(CHECKPOINT_DIR) checkpoints)
	@[ -L data ] || [ -d data ] || (rm -rf data && ln -s $(DATA_DIR) data)
	@mkdir -p $(CHECKPOINT_DIR)
	@echo "  checkpoints/ → $(CHECKPOINT_DIR)"
	@echo "  data/        → $(DATA_DIR)"
	@echo "=== Setup done ==="

# ---- WM ablation (uses train.py) ----
$(ABLATION_TARGETS):
	@echo "=== Running $@ (data=$(DATA)) ==="
	$(PYTHON) $(TRAIN_SCRIPT) --config-name $@ data=$(DATA) wandb.enabled=$(WANDB)

# ---- Planner training (uses train_planner.py) ----
PLANNER_CKPT ?= checkpoints/rcaux_libero_goal.ckpt
PLANNER_EPOCHS ?= 200
PLANNER_MAX_SAMPLES ?= 50

train-planner:
	@echo "=== Training PlannerDecoder ==="
	@echo "  ckpt:    $(PLANNER_CKPT)"
	@echo "  data:    $(DATA)"
	@echo "  epochs:  $(PLANNER_EPOCHS)"
	@echo "  samples: $(PLANNER_MAX_SAMPLES)"
	$(PYTHON) train_planner.py --config-name planner_ft data=$(DATA) \
		planner.ckpt_path=$(PLANNER_CKPT) \
		trainer.max_epochs=$(PLANNER_EPOCHS) \
		wandb.enabled=$(WANDB) \
		max_samples=$(PLANNER_MAX_SAMPLES)

# ---- Compare planner rollout: baseline vs dir ----
compare-planners:
	@echo "=== Comparing planners: baseline vs dir ==="
	$(PYTHON) scripts/viz/compare_planners.py \
		--wm-bl checkpoints/baseline/ablation_baseline_epoch_10_object.ckpt \
		--planner-bl checkpoints/planner_baseline/last.ckpt \
		--wm-dir checkpoints/dir/ablation_dir_epoch_10_object.ckpt \
		--planner-dir checkpoints/planner_dir/last.ckpt \
		--num-samples 200

# ---- Cleanup ----
OUTPUT_DIR ?= /root/autodl-tmp/rcaux-outputs

clean:
	rm -rf ~/.cache/stable_worldmodel/*-ablation_*
	rm -rf $(OUTPUT_DIR)/*
	rm -f checkpoints/last*.ckpt
	@echo "=== Clean done ==="
