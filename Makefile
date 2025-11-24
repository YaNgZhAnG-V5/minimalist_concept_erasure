# Makefile will include make test make clean make build make run
include .env

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf *.egg-info
	rm -rf ./logs/*
	rm -rf ./configs/param_tuning/*

precomit:
	$(PYTHON) -m pip install pre-commit && pre-commit install

format:
	pre-commit run --all-files

sync:
	git pull
	git pull origin main

test:
	pytest -s

d=d
c=c
visual:
	$(PYTHON) ./scripts/train.py --save_dir $(RESULTS_DIR) -c ${c} -d ${d}


cfg=cfg
accvisual:
	CUDA_VISIBLE_DEVICES="1" accelerate launch ./scripts/train.py --save_dir $(RESULTS_DIR) --cfg $(CONFIG_DIR)/${cfg}.yaml


gen:
	$(PYTHON) ./scripts/hyperparameter_tuning.py --task gen

run:
	$(PYTHON) ./scripts/hyperparameter_tuning.py --task run
