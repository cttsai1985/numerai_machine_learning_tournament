#!/usr/bin/bash

python3 ensemble_configs_creator.py --dry-run --low 7 --high 11 --configs-pattern "../configs/configs_baseline_*.yaml" --output-dir ../ensemble_configs/ --run-configs-filepath ../infer_configs/batch_ensemble_infer.yaml --max-model-weight 0.3
python3 ensemble_configs_creator.py --dry-run --low 5 --high 9 --configs-pattern "../configs/configs_baseline_*.yaml" --output-dir ../ensemble_configs/ --run-configs-filepath ../infer_configs/batch_ensemble_infer.yamlensemble_infer.yaml --max-model-weight 0.3
