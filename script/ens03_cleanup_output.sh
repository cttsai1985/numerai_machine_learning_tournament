#!/usr/bin/bash

python3 housekeep.py --dry-run --destination-output ../input/numerai_tournament_resource/old_ens/ --destination-configs ../configs/not_active_ens/ --configs-pattern "../ensemble_configs/*.yaml" --output-pattern "ensemble_sel_*" --output-file-lookup validation_predictions.csv

# python3 housekeep.py --dry-run --destination-output ../input/numerai_tournament_resource/old_ens/ --destination-configs ../configs/not_active_ens/ --configs-pattern "../ensemble_configs/*.yaml" --output-pattern "ens*" --output-file-lookup  validation_model_diagnostics.csv
