#!/usr/bin/bash

python3 download_new_datasets.py --refresh --update

RUN_UPDATE=true
# RUN_UPDATE=false
RUN_BASE_LGBM=true
# RUN_BASE_LGBM=false
RUN_BASE_CATB=true
# RUN_BASE_CATB=false
RUN_ENS_FOCUS=true
# RUN_ENS_FOCUS=false
RUN_BASE_REF=true
# RUN_BASE_REF=false


if $RUN_UPDATE
then
  python download_new_datasets.py --refresh --update
fi


if $RUN_BASE_LGBM
then
  echo "run base lgbm set"
  # base models
  python run.py --configs ../infer_configs/infer_baseline_cttsai_lgbm_fair.yaml
  python run.py --configs ../infer_configs/infer_baseline_cttsai_lgbm_huber.yaml
  python run.py --configs ../infer_configs/infer_baseline_cttsai_lgbm_multiclf.yaml
  python run.py --configs ../infer_configs/infer_baseline_cttsai_lgbm_xentropy.yaml
fi


if $RUN_BASE_CATB
then
  echo "run base catb set"
  # base models
  python run.py --configs ../infer_configs/infer_baseline_cttsai_catb_multiclf.yaml
  python run.py --configs ../infer_configs/infer_baseline_cttsai_catb_pairlogitp.yaml
fi


if $RUN_ENS_FOCUS
then
  echo "run focus set"
  # focus
  python run.py --configs ../infer_configs/infer_focus_best.yaml
  
  python run.py --configs ../infer_configs/infer_top_jerome.yaml
  python run.py --configs ../infer_configs/infer_top_arthur.yaml
  python run.py --configs ../infer_configs/infer_top_william.yaml

  python run.py --configs ../infer_configs/infer_mix_alt_xentropy.yaml
  python run.py --configs ../infer_configs/infer_mix_alt_huber.yaml
  python run.py --configs ../infer_configs/infer_mix_alt_fair.yaml

  # mix
  python run.py --configs ../infer_configs/infer_mix_light_multiclf.yaml
  python run.py --configs ../infer_configs/infer_mix_cat_multiclf.yaml
  python run.py --configs ../infer_configs/infer_mix_cat_ranker.yaml
fi


if $RUN_BASE_REF
then
  echo "run reference target set"
  # reference
  python run.py --configs ../infer_configs/infer_reference.yaml
  python run.py --configs ../infer_configs/infer_reference_arthur.yaml
  python run.py --configs ../infer_configs/infer_reference_ben.yaml
  python run.py --configs ../infer_configs/infer_reference_jerome.yaml
  python run.py --configs ../infer_configs/infer_reference_thomas.yaml
  python run.py --configs ../infer_configs/infer_reference_william.yaml
  python run.py --configs ../infer_configs/infer_reference_example.yaml
fi

