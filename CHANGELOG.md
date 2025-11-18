# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive edge case tests for LinearRegression (9 tests)
  - Single sample, negative/large/small values, constant target, extrapolation
- Comprehensive edge case tests for KMeans (10 tests)
  - Identical points, 1D/high-dim data, exact k samples, tolerance/iterations
- DataFrame example (`examples/dataframe_basics.rs`)

### Changed

- Total test count increased to 120 unit tests + 19 property tests + 13 doctests

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

[Unreleased]: https://github.com/paiml/aprender/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/aprender/releases/tag/v0.1.0
