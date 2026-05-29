# Ablation experiments for RC-aux
# Usage:
#   make all              — run all 3 ablations sequentially
#   make ablation_baseline
#   make ablation_dir
#   make ablation_dir_ss
#   make all DATA=pusht   — override dataset (pusht, ogb, dmc, tworoom)

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
TRAIN_SCRIPT := train.py

DATA ?= ogb
WANDB ?= false

ABLATION_TARGETS := ablation_baseline ablation_dir ablation_dir_ss

.PHONY: all $(ABLATION_TARGETS) clean

all: $(ABLATION_TARGETS)

$(ABLATION_TARGETS):
	@echo "=== Running $@ (data=$(DATA)) ==="
	$(PYTHON) $(TRAIN_SCRIPT) --config-name $@ data=$(DATA) wandb.enabled=$(WANDB)

clean:
	rm -rf $(HOME)/.cache/stable_worldmodel/*-ablation_*
	rm -rf $(HOME)/autodl-tmp/rcaux-outputs/*ablation_*
