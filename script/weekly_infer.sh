#!/usr/bin/bash

python3 download_new_datasets.py --refresh --update

#
python3 run.py --configs ../infer_configs/infer_baseline_cttsai_lgbm.yaml
python3 run.py --configs ../infer_configs/infer_baseline_cttsai_catb.yaml

# buggy
# python3 run.py --configs ../infer_configs/infer_baseline_cttsai_xgb.yaml 
# python3 run.py --configs ../infer_configs/infer_baseline_cttsai.yaml
# python3 run.py --configs ../infer_configs/infer_baseline_cttsai_rest.yaml

# focus
python3 run.py --configs ../infer_configs/infer_focus_fnc.yaml
python3 run.py --configs ../infer_configs/infer_focus_risk.yaml
python3 run.py --configs ../infer_configs/infer_focus_sharpe.yaml
python3 run.py --configs ../infer_configs/infer_focus_best.yaml

# mix
python3 run.py --configs ../infer_configs/infer_mix_light_clf.yaml
python3 run.py --configs ../infer_configs/infer_mix_cat_ranker.yaml
python3 run.py --configs ../infer_configs/infer_mix_cat_multiclf_alt.yaml

# reference
python3 run.py --configs ../infer_configs/infer_reference.yaml
python3 run.py --configs ../infer_configs/infer_reference_arthur.yaml
python3 run.py --configs ../infer_configs/infer_reference_ben.yaml
python3 run.py --configs ../infer_configs/infer_reference_thomas.yaml
python3 run.py --configs ../infer_configs/infer_reference_william.yaml
python3 run.py --configs ../infer_configs/infer_reference_example.yaml

