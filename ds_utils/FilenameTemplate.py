feature_exposure_filename_template: str = "{eval_type}_feature_exposure.parquet"

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
