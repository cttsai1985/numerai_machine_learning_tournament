import os
from typing import List, Tuple

# root path
root_resource_path: str = os.environ.get(
    "rootResourcePath", os.path.join("..", "input", "numerai_tournament_resource"))
default_data_dir: str = os.path.join(root_resource_path, "latest_tournament_datasets")
default_meta_data_dir: str = os.path.join(root_resource_path, "metadata")

numerai_data_filename_pairs: List[Tuple[str]] = [
    ("training", "numerai_training_data.parquet",),
    ("validation", "numerai_validation_data.parquet",),
    ("tournament", "numerai_tournament_data.parquet",),
    ("live", "numerai_live_data.parquet",),
]

numerai_example_filename_pairs: List[Tuple[str]] = [
    ("validation", "example_validation_predictions.parquet"),
    ("tournament", "example_predictions.parquet"),
]

# feature/target columns
default_feature_collection_filename: str = "features_numerai.json"
default_target_collection_filename: str = "targets.json"
round_identifier_template: str = "numerai_tournament_round_{num_round:04d}"

# feature evaluation
feature_corr_filename_template: str = "numerai_{eval_type}_corr_{target}.parquet"

# feature neutral feature
feature_neutral_filename_template: str = "numerai_{eval_type}_feature_neutral.parquet"
feature_neutral_tb_filename_template: str = "numerai_{eval_type}_feature_neutral_tb{tb_num:04d}.parquet"
feature_neutral_target_filename_template: str = "numerai_{eval_type}_feature_neutral_{target}.parquet"

# feature analysis
feature_analysis_filename_template: str = "numerai_feature_analysis_{eval_type}_{target}.parquet"
feature_analysis_tb_filename_template: str = "numerai_feature_analysis_{eval_type}_tb{tb_num:04d}_{target}.parquet"

# evaluation y with features
feature_exposure_filename_template: str = "{eval_type}_feature_exposure.parquet"
feature_exposure_tb_filename_template: str = "{eval_type}_tb{tb_num:04d}_feature_exposure.parquet"

# evaluation y with example predictions
example_analytics_filename_template: str = "{eval_type}_example_analytics.parquet"
example_analytics_tb_filename_template: str = "{eval_type}_tb{tb_num:04d}_example_analytics.parquet"

# score file split
score_split_filename_template: str = "{eval_type}_score_split.parquet"
score_tb_split_filename_template: str = "{eval_type}_tb{tb_num:04d}_score_split.parquet"

# score file split and target
score_target_split_filename_template: str = "{eval_type}_score_split_{target}.parquet"
score_target_all_filename_template: str = "{eval_type}_score_all_{target}.parquet"

# score file all
score_all_filename_template: str = "{eval_type}_score_all.parquet"
score_tb_all_filename_template: str = "{eval_type}_tb{tb_num:04d}_score_all.parquet"


# prediction files
predictions_parquet_filename_template: str = "{eval_type}_predictions.parquet"
predictions_csv_filename_template: str = "{eval_type}_predictions.csv"

numerai_data_filename_template: str = "numerai_{eval_type}_data.parquet"

# example predictions
example_validation_predictions_parquet_filename: str = "example_validation_predictions.parquet"
example_predictions_parquet_filename: str = "example_predictions.parquet"

#
model_diagnostics_filename_template: str = "{eval_type}_model_diagnostics.{filename_extension}"
model_diagnostics_tb_filename_template: str = "{eval_type}_tb{tb_num:04d}_model_diagnostics.{filename_extension}"
prediction_diagnostics_filename_template: str = "{eval_type}_prediction_diagnostics.{filename_extension}"
