## README

This is a monolith with full modeling lifecycle participating numerai tournament. It contains E2E pipelines for:
- training scikit-learn compatible gradient boosting models (such as XGBoost, LightGBM and CatBoost)
- validate models using industry standard metrics such as Sharpe, Corr, Stdev, MDD.
- using optuna for hyperparameters tuning
- custermize objective, metrics and postprocessing such as neutralization
- weekly inference to meet the tournament requeiremt
