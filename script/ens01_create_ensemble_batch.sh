#!/usr/bin/bash

python3 ensemble_configs_creator.py --dry-run --low 3 --high 7 --configs-pattern "../configs/configs_baseline_*.yaml" --output-dir ../ensemble_configs/ --run-configs-filepath ../infer_configs/batch_ensemble_infer.yaml --max-model-weight 0.5
