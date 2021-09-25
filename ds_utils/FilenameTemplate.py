from typing import List, Tuple

# feature evaluation
feature_exposure_filename_template: str = "{eval_type}_feature_exposure.parquet"
example_analytics_filename_template: str = "{eval_type}_example_analytics.parquet"

# score file
score_split_filename_template: str = "{eval_type}_score_split.parquet"
score_all_filename_template: str = "{eval_type}_score_all.parquet"
score_target_split_filename_template: str = "{eval_type}_score_split_{target}.parquet"
score_target_all_filename_template: str = "{eval_type}_score_all_{target}.parquet"

# prediction files
predictions_parquet_filename_template: str = "{eval_type}_predictions.parquet"
predictions_csv_filename_template: str = "{eval_type}_predictions.csv"

numerai_data_filename_template: str = "numerai_{eval_type}_data.parquet"

# example predictions
example_validation_predictions_parquet_filename: str = "example_validation_predictions.parquet"
example_predictions_parquet_filename: str = "example_predictions.parquet"

#
model_diagnostics_filename_template: str = "{eval_type}_model_diagnostics.{filename_extension}"

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