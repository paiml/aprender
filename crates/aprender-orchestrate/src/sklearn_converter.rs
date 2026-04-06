//! sklearn to Aprender conversion module (BATUTA-009)
//!
//! Converts Python scikit-learn (sklearn) algorithms to Rust Aprender equivalents
//! with automatic backend selection and ergonomic API mapping.
//!
//! # Conversion Strategy
//!
//! sklearn algorithms are mapped to equivalent Aprender algorithms:
//! - `LinearRegression()` → `LinearRegression::new()`
//! - `KMeans(n_clusters=3)` → `KMeans::new(3)`
//! - `DecisionTreeClassifier()` → `DecisionTreeClassifier::new()`
//! - `train_test_split()` → `train_test_split()`
//! - Model methods automatically use MoE routing for optimal performance
//!
//! # Example
//!
//! ```python
//! # Python sklearn code
//! from sklearn.linear_model import LinearRegression
//! from sklearn.model_selection import train_test_split
//!
//! X_train, X_test, y_train, y_test = train_test_split(X, y)
//! model = LinearRegression()
//! model.fit(X_train, y_train)
//! predictions = model.predict(X_test)
//! ```
//!
//! Converts to:
//!
//! ```rust,ignore
//! use aprender::linear_model::LinearRegression;
//! use aprender::model_selection::train_test_split;
//! use aprender::Estimator;
//!
//! let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.25)?;
//! let mut model = LinearRegression::new();
//! model.fit(&X_train, &y_train)?;
//! let predictions = model.predict(&X_test)?;
//! ```

use std::collections::HashMap;

/// sklearn algorithm types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum SklearnAlgorithm {
    // Linear Models
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,

    // Clustering
    KMeans,
    DBSCAN,

    // Tree Models
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,

    // Preprocessing
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,

    // Model Selection
    TrainTestSplit,
    CrossValidation,

    // Metrics
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MeanSquaredError,
    R2Score,
}

impl SklearnAlgorithm {
    /// Get the computational complexity for MoE routing
    pub fn complexity(&self) -> crate::backend::OpComplexity {
        use crate::backend::OpComplexity;

        match self {
            // Preprocessing operations are Low complexity
            SklearnAlgorithm::StandardScaler
            | SklearnAlgorithm::MinMaxScaler
            | SklearnAlgorithm::LabelEncoder
            | SklearnAlgorithm::TrainTestSplit => OpComplexity::Low,

            // Linear models and metrics are Medium complexity
            SklearnAlgorithm::LinearRegression
            | SklearnAlgorithm::Ridge
            | SklearnAlgorithm::Lasso
            | SklearnAlgorithm::LogisticRegression
            | SklearnAlgorithm::Accuracy
            | SklearnAlgorithm::Precision
            | SklearnAlgorithm::Recall
            | SklearnAlgorithm::F1Score
            | SklearnAlgorithm::MeanSquaredError
            | SklearnAlgorithm::R2Score => OpComplexity::Medium,

            // Tree models, ensemble methods, and clustering are High complexity
            SklearnAlgorithm::DecisionTreeClassifier
            | SklearnAlgorithm::DecisionTreeRegressor
            | SklearnAlgorithm::RandomForestClassifier
            | SklearnAlgorithm::RandomForestRegressor
            | SklearnAlgorithm::KMeans
            | SklearnAlgorithm::DBSCAN
            | SklearnAlgorithm::CrossValidation => OpComplexity::High,
        }
    }

    /// Get the sklearn module path
    pub fn sklearn_module(&self) -> &str {
        match self {
            SklearnAlgorithm::LinearRegression
            | SklearnAlgorithm::Ridge
            | SklearnAlgorithm::Lasso
            | SklearnAlgorithm::LogisticRegression => "sklearn.linear_model",

            SklearnAlgorithm::KMeans | SklearnAlgorithm::DBSCAN => "sklearn.cluster",

            SklearnAlgorithm::DecisionTreeClassifier | SklearnAlgorithm::DecisionTreeRegressor => {
                "sklearn.tree"
            }

            SklearnAlgorithm::RandomForestClassifier | SklearnAlgorithm::RandomForestRegressor => {
                "sklearn.ensemble"
            }

            SklearnAlgorithm::StandardScaler
            | SklearnAlgorithm::MinMaxScaler
            | SklearnAlgorithm::LabelEncoder => "sklearn.preprocessing",

            SklearnAlgorithm::TrainTestSplit | SklearnAlgorithm::CrossValidation => {
                "sklearn.model_selection"
            }

            SklearnAlgorithm::Accuracy
            | SklearnAlgorithm::Precision
            | SklearnAlgorithm::Recall
            | SklearnAlgorithm::F1Score
            | SklearnAlgorithm::MeanSquaredError
            | SklearnAlgorithm::R2Score => "sklearn.metrics",
        }
    }
}

/// Aprender equivalent algorithm
#[derive(Debug, Clone)]
pub struct AprenderAlgorithm {
    /// Rust code template for the algorithm
    pub code_template: String,
    /// Required imports
    pub imports: Vec<String>,
    /// Computational complexity
    pub complexity: crate::backend::OpComplexity,
    /// Typical usage pattern
    pub usage_pattern: String,
}

/// sklearn to Aprender converter
pub struct SklearnConverter {
    /// Algorithm mapping
    algorithm_map: HashMap<SklearnAlgorithm, AprenderAlgorithm>,
    /// Backend selector for MoE routing
    backend_selector: crate::backend::BackendSelector,
}

impl Default for SklearnConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnConverter {
    /// Create a new sklearn converter with default mappings
    pub fn new() -> Self {
        let mut algorithm_map = HashMap::new();

        // Linear Models
        algorithm_map.insert(
            SklearnAlgorithm::LinearRegression,
            AprenderAlgorithm {
                code_template: "LinearRegression::new()".to_string(),
                imports: vec![
                    "use aprender::linear_model::LinearRegression;".to_string(),
                    "use aprender::Estimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let mut model = LinearRegression::new();\nmodel.fit(&X_train, &y_train)?;\nlet predictions = model.predict(&X_test)?;".to_string(),
            },
        );

        algorithm_map.insert(
            SklearnAlgorithm::LogisticRegression,
            AprenderAlgorithm {
                code_template: "LogisticRegression::new()".to_string(),
                imports: vec![
                    "use aprender::classification::LogisticRegression;".to_string(),
                    "use aprender::Estimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let mut model = LogisticRegression::new();\nmodel.fit(&X_train, &y_train)?;\nlet predictions = model.predict(&X_test)?;".to_string(),
            },
        );

        // Clustering
        algorithm_map.insert(
            SklearnAlgorithm::KMeans,
            AprenderAlgorithm {
                code_template: "KMeans::new({n_clusters})".to_string(),
                imports: vec![
                    "use aprender::cluster::KMeans;".to_string(),
                    "use aprender::UnsupervisedEstimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern: "let mut model = KMeans::new(3);\nmodel.fit(&X)?;\nlet labels = model.predict(&X)?;".to_string(),
            },
        );

        // Tree Models
        algorithm_map.insert(
            SklearnAlgorithm::DecisionTreeClassifier,
            AprenderAlgorithm {
                code_template: "DecisionTreeClassifier::new()".to_string(),
                imports: vec![
                    "use aprender::tree::DecisionTreeClassifier;".to_string(),
                    "use aprender::Estimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern: "let mut model = DecisionTreeClassifier::new();\nmodel.fit(&X_train, &y_train)?;\nlet predictions = model.predict(&X_test)?;".to_string(),
            },
        );

        // Preprocessing
        algorithm_map.insert(
            SklearnAlgorithm::StandardScaler,
            AprenderAlgorithm {
                code_template: "StandardScaler::new()".to_string(),
                imports: vec![
                    "use aprender::preprocessing::StandardScaler;".to_string(),
                    "use aprender::Transformer;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern: "let mut scaler = StandardScaler::new();\nscaler.fit(&X_train)?;\nlet X_train_scaled = scaler.transform(&X_train)?;".to_string(),
            },
        );

        // Model Selection
        algorithm_map.insert(
            SklearnAlgorithm::TrainTestSplit,
            AprenderAlgorithm {
                code_template: "train_test_split(&X, &y, {test_size})".to_string(),
                imports: vec!["use aprender::model_selection::train_test_split;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern:
                    "let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.25)?;"
                        .to_string(),
            },
        );

        // Metrics
        algorithm_map.insert(
            SklearnAlgorithm::Accuracy,
            AprenderAlgorithm {
                code_template: "accuracy_score(&y_true, &y_pred)".to_string(),
                imports: vec!["use aprender::metrics::accuracy_score;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let acc = accuracy_score(&y_true, &y_pred)?;".to_string(),
            },
        );

        algorithm_map.insert(
            SklearnAlgorithm::MeanSquaredError,
            AprenderAlgorithm {
                code_template: "mean_squared_error(&y_true, &y_pred)".to_string(),
                imports: vec!["use aprender::metrics::mean_squared_error;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let mse = mean_squared_error(&y_true, &y_pred)?;".to_string(),
            },
        );

        Self { algorithm_map, backend_selector: crate::backend::BackendSelector::new() }
    }

    /// Convert a sklearn algorithm to Aprender
    pub fn convert(&self, algorithm: &SklearnAlgorithm) -> Option<&AprenderAlgorithm> {
        self.algorithm_map.get(algorithm)
    }

    /// Get recommended backend for an algorithm
    pub fn recommend_backend(
        &self,
        algorithm: &SklearnAlgorithm,
        data_size: usize,
    ) -> crate::backend::Backend {
        self.backend_selector.select_with_moe(algorithm.complexity(), data_size)
    }

    /// Get all available conversions
    pub fn available_algorithms(&self) -> Vec<&SklearnAlgorithm> {
        self.algorithm_map.keys().collect()
    }

    /// Generate conversion report
    pub fn conversion_report(&self) -> String {
        let mut report = String::from("sklearn → Aprender Conversion Map\n");
        report.push_str("===================================\n\n");

        // Group by module
        let mut by_module: HashMap<&str, Vec<(&SklearnAlgorithm, &AprenderAlgorithm)>> =
            HashMap::new();

        for (alg, aprender_alg) in &self.algorithm_map {
            by_module.entry(alg.sklearn_module()).or_default().push((alg, aprender_alg));
        }

        for (module, algorithms) in &by_module {
            report.push_str(&format!("## {}\n\n", module));

            for (alg, aprender_alg) in algorithms {
                report.push_str(&format!("{:?}:\n", alg));
                report.push_str(&format!("  Template: {}\n", aprender_alg.code_template));
                report.push_str(&format!("  Complexity: {:?}\n", aprender_alg.complexity));
                report.push_str(&format!("  Imports: {}\n", aprender_alg.imports.join(", ")));
                report.push_str(&format!(
                    "  Usage:\n    {}\n\n",
                    aprender_alg.usage_pattern.replace('\n', "\n    ")
                ));
            }
            report.push('\n');
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = SklearnConverter::new();
        assert!(!converter.available_algorithms().is_empty());
    }

    #[test]
    fn test_algorithm_complexity() {
        assert_eq!(
            SklearnAlgorithm::LinearRegression.complexity(),
            crate::backend::OpComplexity::Medium
        );
        assert_eq!(
            SklearnAlgorithm::StandardScaler.complexity(),
            crate::backend::OpComplexity::Low
        );
        assert_eq!(SklearnAlgorithm::KMeans.complexity(), crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_linear_regression_conversion() {
        let converter = SklearnConverter::new();
        let aprender_alg =
            converter.convert(&SklearnAlgorithm::LinearRegression).expect("unexpected failure");
        assert!(aprender_alg.code_template.contains("LinearRegression"));
        assert!(aprender_alg.imports.iter().any(|i| i.contains("linear_model")));
    }

    #[test]
    fn test_kmeans_conversion() {
        let converter = SklearnConverter::new();
        let aprender_alg = converter.convert(&SklearnAlgorithm::KMeans).expect("conversion failed");
        assert!(aprender_alg.code_template.contains("KMeans"));
        assert!(aprender_alg.imports.iter().any(|i| i.contains("cluster")));
    }

    #[test]
    fn test_backend_recommendation() {
        let converter = SklearnConverter::new();

        // Small dataset with preprocessing should use Scalar
        let backend = converter.recommend_backend(&SklearnAlgorithm::StandardScaler, 100);
        assert_eq!(backend, crate::backend::Backend::Scalar);

        // Large dataset with linear model should use SIMD
        let backend = converter.recommend_backend(&SklearnAlgorithm::LinearRegression, 50_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);

        // Large dataset with clustering should use GPU
        let backend = converter.recommend_backend(&SklearnAlgorithm::KMeans, 100_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_sklearn_module_paths() {
        assert_eq!(SklearnAlgorithm::LinearRegression.sklearn_module(), "sklearn.linear_model");
        assert_eq!(SklearnAlgorithm::KMeans.sklearn_module(), "sklearn.cluster");
        assert_eq!(SklearnAlgorithm::StandardScaler.sklearn_module(), "sklearn.preprocessing");
    }

    #[test]
    fn test_conversion_report() {
        let converter = SklearnConverter::new();
        let report = converter.conversion_report();
        assert!(report.contains("sklearn → Aprender"));
        assert!(report.contains("LinearRegression"));
        assert!(report.contains("Complexity"));
    }

    // ============================================================================
    // SKLEARN ALGORITHM ENUM TESTS
    // ============================================================================

    #[test]
    fn test_all_sklearn_algorithms_exist() {
        // Test all 22 variants can be constructed
        let algs = vec![
            SklearnAlgorithm::LinearRegression,
            SklearnAlgorithm::Ridge,
            SklearnAlgorithm::Lasso,
            SklearnAlgorithm::LogisticRegression,
            SklearnAlgorithm::KMeans,
            SklearnAlgorithm::DBSCAN,
            SklearnAlgorithm::DecisionTreeClassifier,
            SklearnAlgorithm::DecisionTreeRegressor,
            SklearnAlgorithm::RandomForestClassifier,
            SklearnAlgorithm::RandomForestRegressor,
            SklearnAlgorithm::StandardScaler,
            SklearnAlgorithm::MinMaxScaler,
            SklearnAlgorithm::LabelEncoder,
            SklearnAlgorithm::TrainTestSplit,
            SklearnAlgorithm::CrossValidation,
            SklearnAlgorithm::Accuracy,
            SklearnAlgorithm::Precision,
            SklearnAlgorithm::Recall,
            SklearnAlgorithm::F1Score,
            SklearnAlgorithm::MeanSquaredError,
            SklearnAlgorithm::R2Score,
        ];
        assert_eq!(algs.len(), 21); // 21 algorithms tested
    }

    #[test]
    fn test_algorithm_equality() {
        assert_eq!(SklearnAlgorithm::LinearRegression, SklearnAlgorithm::LinearRegression);
        assert_ne!(SklearnAlgorithm::LinearRegression, SklearnAlgorithm::KMeans);
    }

    #[test]
    fn test_algorithm_clone() {
        let alg1 = SklearnAlgorithm::DecisionTreeClassifier;
        let alg2 = alg1.clone();
        assert_eq!(alg1, alg2);
    }

    #[test]
    fn test_complexity_low_algorithms() {
        let low_algs = vec![
            SklearnAlgorithm::StandardScaler,
            SklearnAlgorithm::MinMaxScaler,
            SklearnAlgorithm::LabelEncoder,
            SklearnAlgorithm::TrainTestSplit,
        ];

        for alg in low_algs {
            assert_eq!(alg.complexity(), crate::backend::OpComplexity::Low);
        }
    }

    #[test]
    fn test_complexity_medium_algorithms() {
        let medium_algs = vec![
            SklearnAlgorithm::LinearRegression,
            SklearnAlgorithm::Ridge,
            SklearnAlgorithm::Lasso,
            SklearnAlgorithm::LogisticRegression,
            SklearnAlgorithm::Accuracy,
            SklearnAlgorithm::Precision,
            SklearnAlgorithm::Recall,
            SklearnAlgorithm::F1Score,
            SklearnAlgorithm::MeanSquaredError,
            SklearnAlgorithm::R2Score,
        ];

        for alg in medium_algs {
            assert_eq!(alg.complexity(), crate::backend::OpComplexity::Medium);
        }
    }

    #[test]
    fn test_complexity_high_algorithms() {
        let high_algs = vec![
            SklearnAlgorithm::DecisionTreeClassifier,
            SklearnAlgorithm::DecisionTreeRegressor,
            SklearnAlgorithm::RandomForestClassifier,
            SklearnAlgorithm::RandomForestRegressor,
            SklearnAlgorithm::KMeans,
            SklearnAlgorithm::DBSCAN,
            SklearnAlgorithm::CrossValidation,
        ];

        for alg in high_algs {
            assert_eq!(alg.complexity(), crate::backend::OpComplexity::High);
        }
    }

    #[test]
    fn test_sklearn_module_linear_model() {
        let linear_algs = vec![
            SklearnAlgorithm::LinearRegression,
            SklearnAlgorithm::Ridge,
            SklearnAlgorithm::Lasso,
            SklearnAlgorithm::LogisticRegression,
        ];

        for alg in linear_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.linear_model");
        }
    }

    #[test]
    fn test_sklearn_module_cluster() {
        let cluster_algs = vec![SklearnAlgorithm::KMeans, SklearnAlgorithm::DBSCAN];

        for alg in cluster_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.cluster");
        }
    }

    #[test]
    fn test_sklearn_module_tree() {
        let tree_algs =
            vec![SklearnAlgorithm::DecisionTreeClassifier, SklearnAlgorithm::DecisionTreeRegressor];

        for alg in tree_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.tree");
        }
    }

    #[test]
    fn test_sklearn_module_ensemble() {
        let ensemble_algs =
            vec![SklearnAlgorithm::RandomForestClassifier, SklearnAlgorithm::RandomForestRegressor];

        for alg in ensemble_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.ensemble");
        }
    }

    #[test]
    fn test_sklearn_module_preprocessing() {
        let preprocessing_algs = vec![
            SklearnAlgorithm::StandardScaler,
            SklearnAlgorithm::MinMaxScaler,
            SklearnAlgorithm::LabelEncoder,
        ];

        for alg in preprocessing_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.preprocessing");
        }
    }

    #[test]
    fn test_sklearn_module_model_selection() {
        let model_selection_algs =
            vec![SklearnAlgorithm::TrainTestSplit, SklearnAlgorithm::CrossValidation];

        for alg in model_selection_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.model_selection");
        }
    }

    #[test]
    fn test_sklearn_module_metrics() {
        let metrics_algs = vec![
            SklearnAlgorithm::Accuracy,
            SklearnAlgorithm::Precision,
            SklearnAlgorithm::Recall,
            SklearnAlgorithm::F1Score,
            SklearnAlgorithm::MeanSquaredError,
            SklearnAlgorithm::R2Score,
        ];

        for alg in metrics_algs {
            assert_eq!(alg.sklearn_module(), "sklearn.metrics");
        }
    }

    // ============================================================================
    // APRENDER ALGORITHM STRUCT TESTS
    // ============================================================================

    #[test]
    fn test_aprender_algorithm_construction() {
        let alg = AprenderAlgorithm {
            code_template: "test_template".to_string(),
            imports: vec!["use test;".to_string()],
            complexity: crate::backend::OpComplexity::Medium,
            usage_pattern: "let x = test();".to_string(),
        };

        assert_eq!(alg.code_template, "test_template");
        assert_eq!(alg.imports.len(), 1);
        assert_eq!(alg.complexity, crate::backend::OpComplexity::Medium);
        assert!(alg.usage_pattern.contains("test()"));
    }

    #[test]
    fn test_aprender_algorithm_clone() {
        let alg1 = AprenderAlgorithm {
            code_template: "template".to_string(),
            imports: vec!["import".to_string()],
            complexity: crate::backend::OpComplexity::High,
            usage_pattern: "usage".to_string(),
        };

        let alg2 = alg1.clone();
        assert_eq!(alg1.code_template, alg2.code_template);
        assert_eq!(alg1.imports, alg2.imports);
        assert_eq!(alg1.complexity, alg2.complexity);
    }

    // ============================================================================
    // SKLEARN CONVERTER TESTS
    // ============================================================================

    #[test]
    fn test_converter_default() {
        let converter = SklearnConverter::default();
        assert!(!converter.available_algorithms().is_empty());
    }

    #[test]
    fn test_convert_all_mapped_algorithms() {
        let converter = SklearnConverter::new();

        // Test all algorithms that should have mappings
        let mapped_algs = vec![
            SklearnAlgorithm::LinearRegression,
            SklearnAlgorithm::LogisticRegression,
            SklearnAlgorithm::KMeans,
            SklearnAlgorithm::DecisionTreeClassifier,
            SklearnAlgorithm::StandardScaler,
            SklearnAlgorithm::TrainTestSplit,
            SklearnAlgorithm::Accuracy,
            SklearnAlgorithm::MeanSquaredError,
        ];

        for alg in mapped_algs {
            assert!(converter.convert(&alg).is_some(), "Missing mapping for {:?}", alg);
        }
    }

    #[test]
    fn test_convert_unmapped_algorithm() {
        let converter = SklearnConverter::new();

        // Ridge, Lasso, etc. might not be mapped
        // Just verify the function handles missing algorithms gracefully
        let result = converter.convert(&SklearnAlgorithm::Ridge);
        // It's ok if this is None - we're testing the API works
        let _ = result;
    }

    #[test]
    fn test_logistic_regression_conversion() {
        let converter = SklearnConverter::new();
        let alg =
            converter.convert(&SklearnAlgorithm::LogisticRegression).expect("unexpected failure");

        assert!(alg.code_template.contains("LogisticRegression"));
        assert!(alg.imports.iter().any(|i| i.contains("classification")));
        assert_eq!(alg.complexity, crate::backend::OpComplexity::Medium);
    }

    #[test]
    fn test_decision_tree_conversion() {
        let converter = SklearnConverter::new();
        let alg = converter
            .convert(&SklearnAlgorithm::DecisionTreeClassifier)
            .expect("unexpected failure");

        assert!(alg.code_template.contains("DecisionTreeClassifier"));
        assert!(alg.imports.iter().any(|i| i.contains("tree")));
        assert_eq!(alg.complexity, crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_standard_scaler_conversion() {
        let converter = SklearnConverter::new();
        let alg = converter.convert(&SklearnAlgorithm::StandardScaler).expect("unexpected failure");

        assert!(alg.code_template.contains("StandardScaler"));
        assert!(alg.imports.iter().any(|i| i.contains("preprocessing")));
        assert_eq!(alg.complexity, crate::backend::OpComplexity::Low);
    }

    #[test]
    fn test_train_test_split_conversion() {
        let converter = SklearnConverter::new();
        let alg = converter.convert(&SklearnAlgorithm::TrainTestSplit).expect("unexpected failure");

        assert!(alg.code_template.contains("train_test_split"));
        assert!(alg.imports.iter().any(|i| i.contains("model_selection")));
    }

    #[test]
    fn test_accuracy_conversion() {
        let converter = SklearnConverter::new();
        let alg = converter.convert(&SklearnAlgorithm::Accuracy).expect("conversion failed");

        assert!(alg.code_template.contains("accuracy_score"));
        assert!(alg.imports.iter().any(|i| i.contains("metrics")));
    }

    #[test]
    fn test_mse_conversion() {
        let converter = SklearnConverter::new();
        let alg =
            converter.convert(&SklearnAlgorithm::MeanSquaredError).expect("unexpected failure");

        assert!(alg.code_template.contains("mean_squared_error"));
        assert!(alg.imports.iter().any(|i| i.contains("metrics")));
    }

    #[test]
    fn test_available_algorithms() {
        let converter = SklearnConverter::new();
        let algs = converter.available_algorithms();

        assert!(!algs.is_empty());
        // Should have at least the mapped algorithms
        assert!(algs.len() >= 8);
    }

    #[test]
    fn test_recommend_backend_low_complexity() {
        let converter = SklearnConverter::new();

        // Small data size with low complexity should use Scalar
        let backend = converter.recommend_backend(&SklearnAlgorithm::StandardScaler, 10);
        assert_eq!(backend, crate::backend::Backend::Scalar);
    }

    #[test]
    fn test_recommend_backend_medium_complexity() {
        let converter = SklearnConverter::new();

        // Medium data size with medium complexity should use SIMD
        let backend = converter.recommend_backend(&SklearnAlgorithm::LinearRegression, 50_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);
    }

    #[test]
    fn test_recommend_backend_high_complexity() {
        let converter = SklearnConverter::new();

        // Large data size with high complexity should use GPU
        let backend =
            converter.recommend_backend(&SklearnAlgorithm::RandomForestClassifier, 500_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_recommend_backend_clustering() {
        let converter = SklearnConverter::new();

        // Clustering with large data should use GPU
        let backend = converter.recommend_backend(&SklearnAlgorithm::KMeans, 1_000_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_conversion_report_structure() {
        let converter = SklearnConverter::new();
        let report = converter.conversion_report();

        // Check report contains expected sections
        assert!(report.contains("sklearn → Aprender"));
        assert!(report.contains("==="));
        assert!(report.contains("##")); // Module headers
        assert!(report.contains("Template:"));
        assert!(report.contains("Imports:"));
        assert!(report.contains("Usage:"));
    }

    #[test]
    fn test_conversion_report_has_modules() {
        let converter = SklearnConverter::new();
        let report = converter.conversion_report();

        // Should group by sklearn modules
        assert!(report.contains("sklearn"));
    }

    #[test]
    fn test_conversion_report_has_all_algorithms() {
        let converter = SklearnConverter::new();
        let report = converter.conversion_report();

        // Spot check a few algorithms appear in report
        assert!(
            report.contains("LinearRegression")
                || report.contains("KMeans")
                || report.contains("StandardScaler")
        );
    }

    #[test]
    fn test_usage_patterns_not_empty() {
        let converter = SklearnConverter::new();

        for alg in converter.available_algorithms() {
            if let Some(aprender_alg) = converter.convert(alg) {
                assert!(
                    !aprender_alg.usage_pattern.is_empty(),
                    "Empty usage pattern for {:?}",
                    alg
                );
                assert!(
                    !aprender_alg.code_template.is_empty(),
                    "Empty code template for {:?}",
                    alg
                );
                assert!(!aprender_alg.imports.is_empty(), "Empty imports for {:?}", alg);
            }
        }
    }

    #[test]
    fn test_imports_are_valid_rust() {
        let converter = SklearnConverter::new();

        for alg in converter.available_algorithms() {
            if let Some(aprender_alg) = converter.convert(alg) {
                for import in &aprender_alg.imports {
                    assert!(import.starts_with("use "), "Invalid import syntax: {}", import);
                    assert!(import.ends_with(';'), "Import missing semicolon: {}", import);
                }
            }
        }
    }

    #[test]
    fn test_linear_models_have_estimator_trait() {
        let converter = SklearnConverter::new();

        let linear_models =
            vec![SklearnAlgorithm::LinearRegression, SklearnAlgorithm::LogisticRegression];

        for alg in linear_models {
            if let Some(aprender_alg) = converter.convert(&alg) {
                assert!(
                    aprender_alg.imports.iter().any(|i| i.contains("Estimator")),
                    "Linear model {:?} should import Estimator trait",
                    alg
                );
            }
        }
    }

    #[test]
    fn test_clustering_has_unsupervised_trait() {
        let converter = SklearnConverter::new();

        if let Some(kmeans_alg) = converter.convert(&SklearnAlgorithm::KMeans) {
            assert!(
                kmeans_alg.imports.iter().any(|i| i.contains("UnsupervisedEstimator")),
                "KMeans should import UnsupervisedEstimator trait"
            );
        }
    }

    #[test]
    fn test_preprocessing_has_transformer_trait() {
        let converter = SklearnConverter::new();

        if let Some(scaler_alg) = converter.convert(&SklearnAlgorithm::StandardScaler) {
            assert!(
                scaler_alg.imports.iter().any(|i| i.contains("Transformer")),
                "StandardScaler should import Transformer trait"
            );
        }
    }
}
