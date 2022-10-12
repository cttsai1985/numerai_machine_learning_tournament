## README

This is a monolith with full modeling lifecycle participating numerai tournament. It contains E2E pipelines for:
- training scikit-learn compatible gradient boosting models (such as XGBoost, LightGBM and CatBoost) and ensemble.
- validate models using industry standard metrics such as Sharpe, Corr, Stdev, MDD.
- using optuna for hyperparameters tuning
- custermize objective, metrics and postprocessing such as neutralization
- weekly inference to meet the tournament requeiremt

### Model Performance
This codebase was used for several models:
ct_balance was on the weekly top 5%/15% (20+ times) performance within 1 yr time frame: https://numer.ai/ct_balance

### Info for NumerAI tournament
https://docs.numer.ai/tournament/learn
