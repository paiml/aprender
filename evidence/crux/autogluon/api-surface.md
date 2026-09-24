# AutoGluon 1.6.3 — user-facing surface (evidence for CRUX Category O)

Surveyed 2026-09-16 from `../autogluon` @ 77946149 (`VERSION` = 1.6.3). Paths are repo-relative to the autogluon checkout.

## TabularPredictor (`tabular/src/autogluon/tabular/predictor/predictor.py`)

Public methods (65): fit, fit_extra, fit_pseudolabel, predict, predict_proba, predict_from_proba, evaluate, evaluate_predictions, leaderboard, learning_curves, model_failures, predict_multi, predict_proba_multi, fit_summary, transform_features, transform_labels, feature_importance, compile, persist, unpersist, refit_full, model_best, set_model_best, model_refit_map, info, model_info, model_hyperparameters, fit_weighted_ensemble, calibrate_decision_threshold, set_decision_threshold, predict_oof, predict_proba_oof, save_space, delete_models, disk_usage, model_names, distill, plot_ensemble_model, save, load, load_log, clone, clone_for_deployment, simulation_artifact, confusion_matrix, plus properties (problem_type, eval_metric, decision_threshold, feature_metadata, class_labels, positive_class, quantile_levels).

`fit` kwargs that carry a story: presets, time_limit, hyperparameters, num_bag_folds, num_bag_sets, num_stack_levels, auto_stack, dynamic_stacking, fit_weighted_ensemble, refit_full, set_best_to_refit_full, save_bag_folds, keep_only_best, holdout_frac, use_bag_holdout, infer_limit, infer_limit_batch_size, calibrate_decision_threshold, learning_curves, memory_limit, num_cpus, num_gpus, fit_strategy, feature_generator, excluded_model_types, included_model_types, raise_on_no_models_fitted, callbacks, core_kwargs/aux_kwargs (1.6).

Presets (`tabular/src/autogluon/tabular/configs/presets_configs.py`): extreme_quality (zeroshot portfolio, 8 bag folds, foundation models), best_quality (auto_stack + dynamic_stacking), high_quality (+ refit_full, no bag folds saved), good_quality (light portfolio), medium_quality (no bagging), optimize_for_deployment (keep_only_best + save_space), ignore_text, ignore_text_ngrams, interpretable, noncommercial, tabarena. Portfolios: `configs/zeroshot/zeroshot_portfolio_{2023,2025,cpu_2025_12_18,gpu_2025_12_18,commercial_2026_08_05,noncommercial_2026_08_05}.py`.

Model families (`tabular/src/autogluon/tabular/models/`): catboost, ebm, fastainn, imodels, knn, lgb, lr, mitra, nori, realmlp, rf, tabdpt, tabicl, tabm, tabpfnmix, tabpfnv2, tabprep, tabular_nn, xgboost, xt (+ automm, image_prediction, text_prediction wrappers). The 2026 commercial portfolio names CAT, GBM, XGB, MITRA, TABICL, TABM.

Feature pipeline (`features/src/autogluon/features/generators/`): auto_ml_pipeline (enable_numeric/categorical/datetime/text_special/text_ngram/raw_text/vision features), astype, binned, category, cat_int, datetime, drop_duplicates, drop_unique, fillna, frequency, groupby, isnan, label_encoder, one_hot_encoder, oof_target_encoder, text_ngram, text_special, memory_minimize, skrub, rsfc, selection.

## TimeSeriesPredictor (`timeseries/src/autogluon/timeseries/predictor.py`)

Constructor: target, known_covariates_names, prediction_length, freq, eval_metric, eval_metric_seasonal_period, horizon_weight, quantile_levels, cache_predictions (deprecated 1.6), log_to_file.
fit: train_data (TimeSeriesDataFrame with item_id/timestamp index + static_features), tuning_data, time_limit, presets, hyperparameters, hyperparameter_tune_kwargs, excluded_model_types, ensemble_hyperparameters, num_val_windows ("auto" since 1.5), val_step_size, refit_every_n_windows ("auto"), refit_full, enable_ensemble, skip_model_selection, random_seed.
Methods: predict, backtest_predictions, backtest_targets (1.5), evaluate, feature_importance, leaderboard, fit_summary, refit_full, persist/unpersist, export_model (1.6: standalone checkpoint), update (1.6 experimental: ensemble re-selection), make_future_data_frame, plot.
Metrics (`timeseries/.../metrics/point.py`, `quantile.py`): MQL WQL SQL RMSE MSE MAE MAEB BIAS WAPE WAPEB SMAPE MAPE MASE RMSSE RMSLE WCD (MAEB/WAPEB/BIAS/MQL new in 1.6).
Models: local (Naive, SeasonalNaive, Average, SeasonalAverage, NPTS, Zero; statsforecast AutoARIMA/ARIMA/AutoETS/ETS/AutoCES/Theta/DynamicOptimizedTheta/Croston/ADIDA/IMAPA), gluonts (DeepAR, SimpleFeedForward, TFT, DLinear, PatchTST, WaveNet, TiDE), pretrained (Chronos, Chronos2 with LoRA/full fine-tune, Toto, Toto2), tabular (per-step, recursive/direct via mlforecast), ensembles (greedy selection, per-item greedy, weighted, array-based, multi-layer since 1.5).

## MultiModalPredictor (`multimodal/src/autogluon/multimodal/predictor.py`) — CUT on epic #3370

Problem types: classification, regression, few_shot_classification, object_detection, ner / named_entity_recognition, image/text/image_text similarity (matching), semantic_segmentation, zero_shot_image_classification, document classification. Methods: fit, predict, predict_proba, evaluate, extract_embedding, export_onnx, optimize_for_inference, dump_model, list_supported_models.

## Release headlines used for demand scoring

- 1.4 (2025): extreme preset; TabPFNv2, TabICL, TabM, RealMLP; Mitra; MLZero (AutoGluon Assistant).
- 1.5: Chronos-2 with zero-shot + fine-tuning; item-level and multi-layer forecast ensembles; `num_val_windows="auto"`, `backtest_predictions`; RealTabPFN-2/2.5, TabDPT, TabPrep-LightGBM, EBM; new CPU/GPU portfolios; TabArena SOTA.
- 1.6: Nori, TabPFN-3, TabDPT-Turbo, TabPFN-2.6, TabICLv2; Toto-2; MAEB/WAPEB/BIAS/MQL metrics; `TimeSeriesPredictor.export_model`; `update()`; calibrated CPU/GPU memory estimates; GPU-aware parallel bagging; feature-importance memory cut 25x; params immutable after construction (deprecation).
- Docs index (`docs/tutorials/{tabular,timeseries,multimodal,cloud_fit_deploy}/index.md`): tabular quick start → essentials → in-depth → feature engineering → foundational models → multimodal → FAQ; timeseries quick start → in-depth → Chronos → ensembles → metrics → model zoo → FAQ.

## What aprender has today (eb262f8eb), for the S column

- `crates/aprender-core/src/automl/`: AutoTuner, SearchSpace, TPE, GridSearch, RandomSearch, DESearch, ActiveLearningSearch, TimeBudget, EarlyStopping, ProgressCallback — tunes ONE chosen estimator.
- `model_selection/`: KFold, StratifiedKFold, cross_validate, cross_val_score, grid_search, randomized_search, train_test_split.
- `preprocessing/`: LabelEncoder, OneHotEncoder, OrdinalEncoder, Standard/MinMax/MaxAbs/Robust scalers, Normalizer, PolynomialFeatures, PCA, TSNE — all applied by hand, no type inference.
- `tree/`: DecisionTree{Classifier,Regressor}, RandomForest{Classifier,Regressor}, GradientBoostingClassifier (no regressor).
- `calibration.rs`: PlattScaling, IsotonicRegression, TemperatureScaling, ECE/MCE/Brier. No decision-threshold search.
- `interpret/`, `explainable/`: ShapExplainer, LIME, PermutationImportance, IntegratedGradients, CounterfactualExplainer — per estimator.
- `time_series/`: `ARIMA` only (fit/forecast/order), single f32 series. `metrics/`: no MASE/RMSSE/WQL.
- `ensemble/`: MixtureOfExperts + SoftmaxGating. `stack/`: deployment health, not model stacking.
- `apr train` = causal-LM pre-training; `apr finetune --task classify` = text classification. No `apr automl`, no `apr forecast`.
