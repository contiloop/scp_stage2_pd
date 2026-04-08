.PHONY: setup set preprocess train train-resume eval eval-benchmarks eval-benchmarks-base eval-benchmarks-both show-config

PYTHON ?= python3
config ?= config
limit ?= 400

setup:
	$(PYTHON) -m pip install -e . --no-deps -q
	$(PYTHON) -m pip install -U huggingface_hub -q
	$(PYTHON) -m pip install "transformers>=5.2.0,<=5.3.0" "trl>=0.15.0" --no-deps -q
	$(PYTHON) -m pip install "hydra-core>=1.3.2" "omegaconf>=2.3.0" -q
	$(PYTHON) -m pip install --upgrade unsloth unsloth-zoo --no-deps -q
	$(PYTHON) -c "import causal_conv1d" 2>/dev/null || $(PYTHON) -m pip install causal-conv1d -q
	$(PYTHON) -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null || $(PYTHON) -m pip install flash-linear-attention -q
	@$(PYTHON) -c "import torch; print('  flash_sdp:', torch.backends.cuda.flash_sdp_enabled())"
	$(PYTHON) -m pip install lm-eval -q 2>/dev/null || true

set: setup

preprocess:
	$(PYTHON) -m src.preprocess --config-path ../configs --config-name $(config)

train:
	$(PYTHON) -m src.train --config-path ../configs --config-name $(config)

train-resume:
	$(PYTHON) -m src.train --config-path ../configs --config-name $(config) training.resume_from_checkpoint=auto

eval:
	$(PYTHON) -m src.evaluate --config-path configs --config-name $(config) --limit $(limit)

eval-benchmarks:
	$(PYTHON) -m src.evaluate --config-path configs --config-name $(config) --benchmarks_only --bench_target cpt --limit $(limit)

eval-benchmarks-base:
	$(PYTHON) -m src.evaluate --config-path configs --config-name $(config) --benchmarks_only --bench_target base --limit $(limit)

eval-benchmarks-both:
	$(PYTHON) -m src.evaluate --config-path configs --config-name $(config) --benchmarks_only --bench_target both --limit $(limit)

show-config:
	$(PYTHON) -m src.train --config-path ../configs --config-name $(config) --cfg job
