#!/usr/bin/bash

python3 run.py --configs ../infer_configs/infer_baseline_cttsai.yaml
python3 run.py --configs ../infer_configs/infer_focus_fnc.yaml
python3 run.py --configs ../infer_configs/infer_focus_risk.yaml
python3 run.py --configs ../infer_configs/infer_focus_sharpe.yaml
python3 run.py --configs ../infer_configs/infer_focus_best.yaml


python3 run.py --configs ../infer_configs/infer_reference.yaml
python3 run.py --configs ../infer_configs/infer_reference_arthur.yaml
python3 run.py --configs ../infer_configs/infer_reference_william.yaml
python3 run.py --configs ../infer_configs/infer_reference_example.yaml

