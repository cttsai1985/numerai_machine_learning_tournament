#!/usr/bin/bash

python3 housekeep.py --destination-output ../input/old_base/ --destination-configs ../configs/not_active_base/ --configs-pattern "../configs/configs_baseline_catb*.yaml" --output-pattern "catb_*" --output-file-lookup validation_predictions.csv

python3 housekeep.py --destination-output ../input/old_base/ --destination-configs ../configs/not_active_base/ --configs-pattern "../configs/configs_baseline_xgb*.yaml" --output-pattern "xgb_*" --output-file-lookup validation_predictions.csv

python3 housekeep.py --destination-output ../input/old_base/ --destination-configs ../configs/not_active_base/ --configs-pattern "../configs/configs_baseline_lgbm*.yaml" --output-pattern "lgbm_*" --output-file-lookup validation_predictions.csv

python3 housekeep.py --destination-output ../input/old_adhoc/ --destination-configs ../configs/not_active_adhoc/ --configs-pattern "../ensemble_configs/*.yaml" --output-pattern "ensemble_*" --output-file-lookup validation_predictions.csv

python3 housekeep.py --destination-output ../input/old_adhoc/ --destination-configs ../configs/not_active_adhoc/ --configs-pattern "../neutralize_configs/*.yaml" --output-pattern "neutralize_*" --output-file-lookup validation_predictions.csv
