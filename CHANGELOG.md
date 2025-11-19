# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
