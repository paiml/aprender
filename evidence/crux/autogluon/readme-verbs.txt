# AutoGluon 1.6.3 README verbs, ranked by fold position (../autogluon/README.md, 2026-09-16)
# fold 1 — the only code block in the README:
pip install autogluon
TabularPredictor(label="class").fit("train.csv", presets="best")
predictor.predict("test.csv")
# fold 2 — docs/tutorials/tabular/tabular-quick-start.ipynb call order:
TabularPredictor(label).fit(train_data, time_limit=...)
predictor.predict(test_data)
predictor.evaluate(test_data)
predictor.leaderboard(test_data)
predictor.feature_importance(test_data)
# fold 3 — timeseries quick start:
TimeSeriesPredictor(prediction_length=48, eval_metric="WQL").fit(TimeSeriesDataFrame, presets="medium_quality")
predictor.predict(train_data)
predictor.leaderboard(test_data)
# tabular presets (tabular/src/autogluon/tabular/configs/presets_configs.py):
extreme_quality best_quality high_quality good_quality medium_quality optimize_for_deployment ignore_text ignore_text_ngrams interpretable noncommercial tabarena
# tabular model families (tabular/src/autogluon/tabular/models/):
catboost ebm fastainn imodels knn lgb lr mitra nori realmlp rf tabdpt tabicl tabm tabpfnmix tabpfnv2 tabprep tabular_nn xgboost xt (+ automm/image/text wrappers)
# timeseries models (timeseries/src/autogluon/timeseries/models/):
Naive SeasonalNaive Average SeasonalAverage NPTS Zero | AutoARIMA ARIMA AutoETS ETS AutoCES DynamicOptimizedTheta Theta Croston ADIDA IMAPA | DeepAR SimpleFeedForward TemporalFusionTransformer DLinear PatchTST WaveNet TiDE | Chronos Chronos2 Toto Toto2 | PerStepTabular RecursiveTabular DirectTabular | ensembles: greedy selection, per-item greedy, weighted, array-based
# timeseries metrics (timeseries/src/autogluon/timeseries/metrics/):
MQL WQL SQL RMSE MSE MAE MAEB BIAS WAPE WAPEB SMAPE MAPE MASE RMSSE RMSLE WCD
