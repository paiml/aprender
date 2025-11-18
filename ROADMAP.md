# Aprender Roadmap

Next Generation Machine Learning in Pure Rust

## Version Status

| Version | Status | Description |
|---------|--------|-------------|
| v0.1.0  | ✅ **Released** | Foundation - Linear Regression + K-Means |
| v0.2.0  | ✅ **Released** | Decision Trees, Random Forests, Cross-Validation, Serialization |
| v0.3.0  | Planned | Regularization & Optimization |
| v0.4.0  | Planned | Classification |
| v0.5.0  | Planned | Neural Networks |
| v1.0.0  | Planned | Production Hardening |

---

## v0.2.0: Tree Models & Cross-Validation (Current)

**Target**: Decision Trees, Random Forests, Model Selection, Model Persistence

### Completed

- [x] **Tree Module**: Complete decision tree implementation
  - DecisionTreeClassifier with GINI impurity
  - Configurable max_depth
  - Recursive tree building
  - Integration tests with Iris dataset
- [x] **Random Forest**: Bootstrap aggregating ensemble
  - RandomForestClassifier with configurable n_estimators
  - Bootstrap sampling with replacement
  - Majority voting for predictions
  - Reproducible with random_state
- [x] **Cross-Validation**: Model evaluation utilities
  - train_test_split() with reproducible random seeds
  - KFold cross-validator with optional shuffling
  - cross_validate() with statistics (mean, std, min, max)
  - CrossValidationResult struct
- [x] **Model Serialization**: Save/load models to disk
  - Serde + bincode integration
  - Works with all models (LinearRegression, KMeans, DecisionTree, RandomForest)
  - Example demonstrating persistence
- [x] **Examples**: New comprehensive examples
  - decision_tree_iris.rs - Decision tree classification
  - random_forest_iris.rs - Ensemble classification
  - cross_validation.rs - Model evaluation workflow
  - model_persistence.rs - Save/load demonstration
- [x] **Documentation**: EXTREME TDD Book
  - 90+ chapter structure on GitHub Pages
  - Complete case study: Cross-Validation
  - RED-GREEN-REFACTOR methodology
  - Live at https://paiml.github.io/aprender/
- [x] **Bug Fixes**
  - Clear error messages for underdetermined systems

### Quality Metrics Achieved

- TDG Score: 93.3/100 (A grade)
- Total Tests: 184 passing (+64 from v0.1.0)
- Property Tests: 22
- Doc Tests: 16
- Test Coverage: ~97%
- Max Cyclomatic Complexity: ≤10
- Zero clippy warnings
- Zero SATD comments

### Released ✅

- [x] Published to crates.io (2024-11-18)
- [x] GitHub Release created with release notes
- [x] EXTREME TDD Book deployed
- [x] CI/CD pipeline passing

---

## v0.1.0: Foundation

**Target**: Linear Regression + K-Means (2 algorithms, viable from day one)

### Completed

- [x] Project scaffolding (Cargo workspace, quality gates)
- [x] Core primitives: Vector, Matrix (with Cholesky solver)
- [x] DataFrame (named column container with `to_matrix()`)
- [x] Linear Regression (OLS via normal equations)
- [x] K-Means clustering (k-means++ initialization + Lloyd's algorithm)
- [x] Metrics: R², MSE, RMSE, MAE, inertia, silhouette_score
- [x] Estimator/UnsupervisedEstimator/Transformer traits
- [x] Property-based tests (19 proptest properties)
- [x] Unit tests (120 tests)
- [x] Doc tests (13 tests)
- [x] Examples: boston_housing, iris_clustering, dataframe_basics
- [x] Benchmarks: linear_regression, kmeans

### Quality Metrics Achieved

- TDG Score: 95.6/100 (A+ grade)
- Repository Score: 95.0/100 (A+)
- Test Coverage: 97.72%
- Mutation Score: 85.3%
- Max Cyclomatic Complexity: 5
- Zero clippy warnings
- Zero SATD comments
- Total Tests: 149 (127 unit + 22 property)

### Released ✅

- [x] Published to crates.io (2024-11-18)
- [x] GitHub Release created with artifacts
- [x] Complete rustdoc coverage
- [x] CI/CD pipeline operational

---

## v0.3.0: Regularization & Optimization

- [x] Ridge regression (L2 regularization)
- [x] Lasso regression (L1 via coordinate descent)
- [ ] Elastic Net (L1 + L2)
- [ ] SGD optimizer (mini-batch gradient descent)
- [ ] Adam optimizer (adaptive learning rates)
- [ ] Loss functions: MSE, MAE, Huber
- [ ] Stratified cross-validation
- [ ] Grid search (hyperparameter tuning)
- [x] StandardScaler transformer
- [x] MinMaxScaler transformer

---

## v0.4.0: Classification

- [ ] Logistic regression (binary classification)
- [ ] Softmax regression (multi-class)
- [ ] Support Vector Machines (SVM)
- [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [ ] Metrics: accuracy, precision, recall, F1, ROC-AUC
- [ ] Confusion matrix utilities

---

## v0.5.0: Advanced Tree Methods

- [ ] Gradient Boosting (XGBoost-style)
- [ ] Feature importance scores
- [ ] Out-of-bag error estimation
- [ ] Decision tree regression
- [ ] Random Forest regression

---

## v0.6.0: Neural Networks

- [ ] Autodiff integration (feature-gated)
- [ ] Dense layers (fully connected)
- [ ] Activation functions (ReLU, Leaky ReLU, ELU, SELU)
- [ ] Optimizers: SGD, Adam, AdaGrad, RMSprop
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Sequential model API

---

## v0.7.0: Advanced Statistics

- [ ] Generalized Linear Models (GLM)
- [ ] Statistical tests: t-test, chi-square, ANOVA
- [ ] Covariance/correlation matrices
- [ ] Principal Component Analysis (PCA)
- [ ] Independent Component Analysis (ICA)

---

## v0.8.0: Time Series

- [ ] ARIMA models
- [ ] Exponential smoothing (Holt-Winters)
- [ ] Seasonal decomposition
- [ ] Forecasting metrics: MAPE, SMAPE

---

## v1.0.0: Production Hardening

- [ ] GPU benchmarks and optimization
- [ ] WASM examples (in-browser training)
- [ ] Model serialization versioning
- [ ] Complete EXTREME TDD Book content
- [ ] Performance whitepaper
- [ ] Production deployment examples

---

## Quality Targets

All releases must meet:

- **TDG Score**: A+ (95.0+/100)
- **Test Coverage**: 95%+ line coverage
- **Mutation Score**: 85%+ (cargo-mutants)
- **Complexity**: ≤10 cyclomatic per function
- **Documentation**: 100% rustdoc coverage

---

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `cargo test --all`
- No clippy warnings: `cargo clippy --all-targets`
- Code is formatted: `cargo fmt`

Priorities:
1. Bug fixes and test coverage improvements
2. Documentation and examples
3. Performance optimizations
4. New algorithms (must include tests and benchmarks)
