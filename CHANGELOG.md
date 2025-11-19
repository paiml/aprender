# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2025-11-19

### Changed

#### Dependencies
- **Upgraded trueno to v0.4.0** (from v0.2.2)
  - AVX-512 backend support (11-12x speedup for compute-bound operations on supported CPUs)
  - New vector operations: `norm_l2()`, `norm_l1()`, `norm_linf()`, `scale()`, `abs()`, `clamp()`, `lerp()`, `fma()`
  - Neural network activation functions: `relu()`, `sigmoid()`, `gelu()`, `swish()`, `tanh()`, `exp()`
  - Refactored multi-backend dispatch with macros (reduces ~1000 lines of code)
  - 100% functional equivalence maintained (all 827 trueno tests passing)
  - Critical bugfix: Missing `abs()` implementation in trueno v0.2.2 (Issue trueno#2)

### Quality
- All 541 tests passing with trueno v0.4.0
- Zero clippy warnings
- Release build verified

### Notes
This is a dependency upgrade release that brings performance improvements and new capabilities from trueno's AVX-512 backend. No API changes to aprender itself - fully backward compatible with v0.4.0.

## [0.4.0] - 2025-11-19

### 🎉 **MAJOR MILESTONE: TOP 10 ML ALGORITHMS - 100% COMPLETE!**

This release completes all 10 of the most popular machine learning algorithms used in industry, achieving full coverage of the Analytics Vidhya 2025 TOP 10 list.

### Added

#### K-Nearest Neighbors (kNN) - Issue #23

- **KNearestNeighbors** classifier with lazy learning
  - Distance metrics: Euclidean, Manhattan, Minkowski(p)
  - Weighted and uniform voting strategies
  - `predict()` and `predict_proba()` methods
  - Builder pattern: `with_metric()`, `with_weights()`
  - 17 comprehensive tests
  - Example: `examples/knn_iris.rs` (90% accuracy)
  - Theory: `book/src/ml-fundamentals/knn.md`
  - Case study: `book/src/examples/knn-iris.md`

#### Gaussian Naive Bayes - Issue #25

- **GaussianNB** probabilistic classifier
  - Bayes' theorem with Gaussian likelihood
  - Log probabilities for numerical stability
  - Variance smoothing parameter (default 1e-9)
  - Class priors computed from training data
  - 16 comprehensive tests
  - Example: `examples/naive_bayes_iris.rs` (100% accuracy - outperforms kNN!)
  - Theory: `book/src/ml-fundamentals/naive-bayes.md`
  - Case study: `book/src/examples/naive-bayes-iris.md`

#### Linear Support Vector Machine (SVM) - Issue #24

- **LinearSVM** maximum-margin classifier
  - Subgradient descent with hinge loss
  - C parameter for regularization control
  - Learning rate decay for convergence
  - `decision_function()` returns margin-based scores
  - Builder pattern: `with_c()`, `with_learning_rate()`, `with_max_iter()`, `with_tolerance()`
  - 14 comprehensive tests
  - Example: `examples/svm_iris.rs` (100% accuracy on binary classification)
  - Theory: `book/src/ml-fundamentals/svm.md`
  - Case study: `book/src/examples/svm-iris.md`

#### Gradient Boosting Machine (GBM) - Issue #26

- **GradientBoostingClassifier** sequential ensemble
  - Gradient descent in function space
  - Fits trees to negative gradients (residuals)
  - Hyperparameters: `n_estimators`, `learning_rate`, `max_depth`
  - Uses DecisionTreeClassifier as weak learners
  - Log-odds initialization, sigmoid probability conversion
  - Early stopping when tree fitting fails
  - 13 comprehensive tests
  - Example: `examples/gbm_iris.rs` (demonstrates hyperparameter effects)
  - Case study: `book/src/examples/gbm-iris.md`

#### Principal Component Analysis (PCA)

- **PCA** dimensionality reduction via eigendecomposition
  - Computes principal components from covariance matrix
  - `explained_variance_ratio()` for variance analysis
  - `transform()` projects data to lower dimensions
  - Builder pattern: `with_n_components()`
  - 13 comprehensive tests
  - Example: `examples/pca_iris.rs` (4D → 2D visualization)
  - Theory: `book/src/ml-fundamentals/pca.md`
  - Case study: `book/src/examples/pca-iris.md`

### Documentation

- Updated `SUMMARY.md` with all new theory and case study chapters
- Updated `tree/mod.rs` documentation to mention ensemble methods
- Updated `classification/mod.rs` to include kNN, Naive Bayes, and Linear SVM

### Test Coverage

- **Total tests**: 541 (up from 515)
- **New tests**: 26 (13 GBM + 13 other algorithms)
- **All tests pass**: ✅
- **Zero clippy warnings**: ✅
- **Code formatting**: ✅ rustfmt compliant

### Quality Assurance

- All examples run successfully
- Comprehensive error handling (untrained models, dimension mismatches, empty data)
- Builder patterns for ergonomic API
- Probabilistic predictions where applicable (`predict_proba`)

### TOP 10 Algorithms - Complete List

1. ✅ **Linear Regression** (v0.1.0)
2. ✅ **Logistic Regression** (v0.2.0)
3. ✅ **Decision Tree** (v0.2.0)
4. ✅ **Random Forest** (v0.2.0)
5. ✅ **K-Means** (v0.1.0)
6. ✅ **PCA** (v0.4.0) - NEW
7. ✅ **K-Nearest Neighbors** (v0.4.0) - NEW
8. ✅ **Naive Bayes** (v0.4.0) - NEW
9. ✅ **Support Vector Machine** (v0.4.0) - NEW
10. ✅ **Gradient Boosting** (v0.4.0) - NEW

**All industry-standard ML algorithms are now available in aprender!**

## [0.3.1] - 2025-11-19

### Added

#### SafeTensors Model Serialization - Complete Coverage (Issue #8)

**All 7 remaining models now support SafeTensors format**:

- **Ridge** (linear_model)
  - `Ridge::save_safetensors()` / `Ridge::load_safetensors()`
  - Serializes: coefficients, intercept, alpha hyperparameter
  - 11 comprehensive tests (roundtrip, metadata, multiple cycles, R² preservation)

- **Lasso** (linear_model)
  - `Lasso::save_safetensors()` / `Lasso::load_safetensors()`
  - Serializes: coefficients, intercept, alpha, max_iter, tol
  - 12 comprehensive tests including sparsity preservation
  - Validates L1 regularization produces zero coefficients

- **ElasticNet** (linear_model)
  - `ElasticNet::save_safetensors()` / `ElasticNet::load_safetensors()`
  - Serializes: coefficients, intercept, alpha, l1_ratio, max_iter, tol
  - 12 comprehensive tests including L1/L2 mix validation
  - Tests l1_ratio extremes (0.0=Ridge, 0.5=balanced, 1.0=Lasso)

- **DecisionTreeClassifier** (tree)
  - `DecisionTreeClassifier::save_safetensors()` / `DecisionTreeClassifier::load_safetensors()`
  - Serializes: Tree structure flattened to 6 parallel arrays via pre-order traversal
  - Arrays: node_features, node_thresholds, node_classes, node_samples, node_left_child, node_right_child
  - 11 comprehensive tests including deep trees (10+ levels), single leaf edge case
  - Preserves exact tree structure and decision boundaries

- **RandomForestClassifier** (tree)
  - `RandomForestClassifier::save_safetensors()` / `RandomForestClassifier::load_safetensors()`
  - Serializes: Multiple trees with index prefixes (tree_0_, tree_1_, etc.)
  - Each tree: 7 tensors (6 structure arrays + max_depth)
  - Hyperparameters: n_estimators, max_depth, random_state
  - 12 comprehensive tests including large ensembles (20 trees)
  - Preserves voting behavior through exact tree reconstruction

- **KMeans** (cluster)
  - `KMeans::save_safetensors()` / `KMeans::load_safetensors()`
  - Serializes: Centroids matrix (k × d), hyperparameters (n_clusters, max_iter, tol, random_state)
  - Metadata: inertia (within-cluster sum of squares), n_iter
  - 13 comprehensive tests including high-dimensional data (5 features)
  - Preserves exact centroid positions for reproducible cluster assignments

- **StandardScaler** (preprocessing)
  - `StandardScaler::save_safetensors()` / `StandardScaler::load_safetensors()`
  - Serializes: Mean vector, std vector, with_mean flag, with_std flag
  - 14 comprehensive tests including inverse transform preservation
  - Tests all configurations (center only, scale only, both, neither/identity)
  - Preserves exact scaling parameters for reproducible transformations

**Key Technical Achievements**:
- Tree serialization via pre-order traversal (eliminates recursion in storage)
- Shared helper functions (flatten_tree_node, reconstruct_tree_node) for code reuse
- Ensemble serialization with index prefixes for multiple models
- Matrix serialization with shape metadata for multi-dimensional data
- Boolean flags encoded as floats (1.0/0.0) for SafeTensors compatibility

**Test Coverage**:
- Total: +85 SafeTensors tests across 7 models
- All tests passing (100% success rate)
- Property tests: idempotency, preservation of scores/predictions/inertia
- Edge cases: unfitted models, corrupted files, nonexistent files

**Cross-Platform Compatibility**:
- Compatible with HuggingFace ecosystem
- Compatible with PyTorch, TensorFlow via SafeTensors
- Compatible with realizar inference engine
- Enables Rust → Python, Python → Rust model deployment
- Eliminates pickle security vulnerabilities

## [0.3.0] - 2025-11-19

### Added

#### Model Serialization

- **SafeTensors Format Support - LogisticRegression** (Issue #6)
  - `LogisticRegression::save_safetensors()` - Export binary classification models to SafeTensors format
  - `LogisticRegression::load_safetensors()` - Load models from SafeTensors format
  - Compatible with HuggingFace ecosystem, Ollama, PyTorch, TensorFlow
  - Compatible with realizar inference engine
  - Deterministic serialization (sorted keys for reproducibility)
  - 5 comprehensive tests (unfitted model, roundtrip, corrupted file, missing file, probability preservation)
  - Full documentation with rustdoc examples
  - Serializes coefficients + intercept tensors
  - Probability predictions preserved exactly after save/load roundtrip

- **SafeTensors Format Support - LinearRegression** (Issue #5)
  - `LinearRegression::save_safetensors()` - Export models to SafeTensors format
  - `LinearRegression::load_safetensors()` - Load models from SafeTensors format
  - Compatible with HuggingFace ecosystem, Ollama, PyTorch, TensorFlow
  - Compatible with realizar inference engine
  - Deterministic serialization (sorted keys for reproducibility)
  - Comprehensive error handling (missing files, corrupted headers)
  - 8-byte header + JSON metadata + F32 tensor data (little-endian)
  - 7 integration tests covering roundtrip, validation, and error cases
  - Full documentation with usage examples

### Changed

- Dependencies: Added `serde_json = "1.0"` for SafeTensors metadata parsing
- Test count: +12 SafeTensors tests (5 LogisticRegression + 7 LinearRegression, total: 417 lib tests)

## [0.2.0] - 2024-11-18

### Added

#### Decision Tree & Random Forest

- **DecisionTreeClassifier** - GINI-based decision tree classifier
  - Configurable `max_depth` parameter
  - Recursive tree building algorithm
  - Support for multi-class classification
  - Implements `Estimator` trait
- **RandomForestClassifier** - Bootstrap aggregating ensemble
  - Configurable `n_estimators` (number of trees)
  - Bootstrap sampling with replacement
  - Majority voting for predictions
  - Reproducible results with `random_state`
  - Builder pattern: `with_max_depth()`, `with_random_state()`

#### Cross-Validation & Model Selection

- **train_test_split()** - Random train/test splitting
  - Configurable test_size (0.0 to 1.0)
  - Optional random_state for reproducibility
  - Shuffles data before splitting
- **KFold** - K-fold cross-validator
  - Configurable number of splits
  - Optional shuffling with `with_shuffle()`
  - Reproducible with `with_random_state()`
  - Handles uneven splits (distributes remainder across first folds)
- **cross_validate()** - Automated cross-validation
  - Works with any `Estimator` implementation
  - Returns `CrossValidationResult` with statistics
  - Methods: `mean()`, `std()`, `min()`, `max()`

#### Model Persistence

- **Model Serialization** - Save/load models to disk
  - Serde + bincode binary serialization
  - Works with all models: LinearRegression, KMeans, DecisionTree, RandomForest
  - Simple `save()` and `load()` API
  - Example: `examples/model_persistence.rs`

#### Examples

- `decision_tree_iris.rs` - Decision tree classification demo
- `random_forest_iris.rs` - Random Forest ensemble demo (20 trees, 100% accuracy)
- `cross_validation.rs` - Complete CV workflow (train/test split, KFold, automated CV)
- `model_persistence.rs` - Model save/load demonstration

#### Documentation

- **EXTREME TDD Book** - Comprehensive methodology guide
  - 90+ chapter structure deployed to GitHub Pages
  - Live at: https://paiml.github.io/aprender/
  - Complete case study: Cross-Validation implementation
  - RED-GREEN-REFACTOR cycle documentation
  - Toyota Way principles (Kaizen, Jidoka, PDCA)
  - Anti-hallucination enforcement (all examples test-backed)

### Changed

- **Dependencies**:
  - Added `rand = "0.8"` for random sampling
  - **Upgraded to trueno v0.2.2** - SIMD-accelerated tensor operations
    - Replaces internal Vector/Matrix with optimized trueno implementation
    - SIMD abs() performance improvements
    - All 184 tests passing with trueno backend
- Total test count: 184 (+64 from v0.1.0)
- Property tests: 22 (+3)
- Doc tests: 16 (+3)

### Fixed

- **LinearRegression**: Clear error message for underdetermined systems (Issue #4)
  - Now returns "Cannot solve: system is underdetermined (more features than samples)"
  - Previously threw cryptic Cholesky decomposition errors

## [0.1.0] - 2024-11-18

### Added

#### Core Primitives
- `Vector<f32>` - 1D numerical array with operations:
  - Statistical: `sum`, `mean`, `variance`, `argmin`, `argmax`
  - Algebraic: `dot`, `norm`, `add`, `sub`, `mul`
- `Matrix<f32>` - 2D numerical array with operations:
  - Linear algebra: `matmul`, `matvec`, `transpose`
  - Solvers: `cholesky_solve` for normal equations
- `DataFrame` - Named column container:
  - Column access: `column()`, `select()`
  - Row access: `row()`
  - Conversion: `to_matrix()`
  - Statistics: `describe()`

#### Machine Learning Models
- `LinearRegression` - Ordinary Least Squares via normal equations
  - Implements `Estimator` trait (`fit`, `predict`, `score`)
  - Returns coefficients and intercept
  - R² score for model evaluation
- `KMeans` - K-means++ initialization with Lloyd's algorithm
  - Implements `UnsupervisedEstimator` trait
  - Configurable: `with_max_iter()`, `with_tol()`, `with_random_state()`
  - Returns labels, centroids, inertia, iteration count

#### Metrics
- Regression: `r_squared`, `mse`, `rmse`, `mae`
- Clustering: `silhouette_score`, `inertia`

#### Traits
- `Estimator<X, Y>` - Supervised learning interface
- `UnsupervisedEstimator<X>` - Unsupervised learning interface
- `Transformer<X>` - Data transformation interface

#### Testing
- 120 unit tests covering all modules
- 19 property-based tests (proptest)
- 13 documentation tests
- Edge case coverage for numerical stability

#### Examples
- `boston_housing.rs` - Linear regression demo
- `iris_clustering.rs` - K-Means clustering demo
- `dataframe_basics.rs` - DataFrame operations demo

#### Benchmarks
- `linear_regression.rs` - Fit/predict performance
- `kmeans.rs` - Clustering performance

#### Documentation
- Complete rustdoc for public API
- README with quick start examples
- ROADMAP with version planning
- CHANGELOG (this file)

### Quality Metrics

- **TDG Score**: 95.6/100 (A+ grade)
- **Repository Score**: 95.0/100 (A+)
- **Test Coverage**: 97.72%
- **Mutation Score**: 85.3%
- **Max Cyclomatic Complexity**: 5 (target ≤10)
- **Max Cognitive Complexity**: 8 (target ≤15)
- **Clippy**: Zero warnings
- **SATD**: Zero TODO/FIXME comments

### Technical Details

- Pure Rust implementation (no external ML dependencies)
- f32 precision for all numerical operations
- Cholesky decomposition for solving normal equations
- K-means++ for intelligent centroid initialization

---

## Release Notes

### v0.1.0

First release of Aprender, providing a minimal viable foundation for machine learning in Rust. This release focuses on two core algorithms (Linear Regression and K-Means) implemented with comprehensive testing following EXTREME TDD methodology.

**Highlights**:
- Production-ready OLS linear regression
- Efficient K-means clustering with k-means++ initialization
- Clean, sklearn-inspired API via traits
- Extensive test coverage (120+ tests)
- High quality score (TDG 94.1/100)

**Known Limitations**:
- f32 only (no f64 support yet)
- No GPU acceleration (planned for v1.0)
- No model serialization (planned for v1.0)
- No train/test split utility (planned for v0.2)

## Release Notes

### v0.2.0

Major feature release adding tree-based models, ensemble methods, cross-validation, and model persistence.

**Highlights**:
- Decision Tree and Random Forest classifiers
- Complete cross-validation utilities (train/test split, KFold, automated CV)
- Model serialization for all models
- EXTREME TDD Book with comprehensive methodology guide
- 64 new tests (+54% increase)

**Breaking Changes**: None (backward compatible)

**Migration Guide**: No migration needed. All v0.1.0 APIs remain unchanged.

---

[Unreleased]: https://github.com/paiml/aprender/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/paiml/aprender/releases/tag/v0.2.0
[0.1.0]: https://github.com/paiml/aprender/releases/tag/v0.1.0
