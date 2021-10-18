#!/usr/bin/bash

python3 housekeep.py --destination-output ../input/old_adhoc/ --destination-configs ../configs/not_active_adhoc/ --configs-pattern "../ensemble_configs/*.yaml" --output-pattern "ensemble_*" --output-file-lookup validation_predictions.csv

python3 housekeep.py --destination-output ../input/old_adhoc/ --destination-configs ../configs/not_active_adhoc/ --configs-pattern "../neutralize_configs/*.yaml" --output-pattern "neutralize_*" --output-file-lookup validation_predictions.csv
