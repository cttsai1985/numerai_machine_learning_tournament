#!/usr/bin/bash

python3 download_new_datasets.py --refresh 

python3 analyze_data_file_columns.py

python3 feature_target_corr.py --configs ../configs/configs_baseline_lgbm_gbdt.yaml
python3 feature_target_corr.py --configs ../configs/configs_neutralized_lgbm_gbdt.yaml
