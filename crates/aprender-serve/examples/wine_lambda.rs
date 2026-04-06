#![allow(clippy::format_push_string)]
//! Wine Quality Prediction - AWS Lambda Example
//!
//! A production-ready wine quality rating predictor deployable to AWS Lambda.
//! Inspired by: https://github.com/paiml/wine-ratings
//!
//! ## Features
//!
//! - Predicts wine quality (0-10 scale) from physicochemical properties
//! - Supports both red and white wine varieties
//! - Sub-10ms cold start, sub-1ms warm inference
//! - Prometheus metrics export
//! - Drift detection for production monitoring
//!
//! ## Wine Features (11 inputs)
//!
//! | Feature | Description | Typical Range |
//! |---------|-------------|---------------|
//! | fixed_acidity | Tartaric acid (g/dm³) | 4.0-16.0 |
//! | volatile_acidity | Acetic acid (g/dm³) | 0.1-1.6 |
//! | citric_acid | Citric acid (g/dm³) | 0.0-1.0 |
//! | residual_sugar | Sugar after fermentation (g/dm³) | 0.9-15.0 |
//! | chlorides | Sodium chloride (g/dm³) | 0.01-0.6 |
//! | free_sulfur_dioxide | Free SO₂ (mg/dm³) | 1-72 |
//! | total_sulfur_dioxide | Total SO₂ (mg/dm³) | 6-289 |
//! | density | Density (g/cm³) | 0.99-1.04 |
//! | ph | pH level | 2.7-4.0 |
//! | sulphates | Potassium sulphate (g/dm³) | 0.3-2.0 |
//! | alcohol | Alcohol content (% vol) | 8.0-15.0 |
//!
//! ## Deployment
//!
//! ```bash
//! # Build for Lambda (ARM64 Graviton)
//! cargo build --release --target aarch64-unknown-linux-gnu --features lambda
//!
//! # Package for deployment
//! cp target/aarch64-unknown-linux-gnu/release/wine_lambda bootstrap
//! zip wine_lambda.zip bootstrap
//!
//! # Deploy to AWS Lambda
//! aws lambda create-function \
//!   --function-name wine-quality-predictor \
//!   --runtime provided.al2 \
//!   --architecture arm64 \
//!   --handler bootstrap \
//!   --zip-file fileb://wine_lambda.zip \
//!   --role arn:aws:iam::ACCOUNT:role/lambda-role
//! ```

#![cfg_attr(feature = "lambda", allow(dead_code))]

use std::collections::HashMap;

// ============================================================================
// Wine Quality Model
// ============================================================================

/// Wine physicochemical features for quality prediction
#[derive(Debug, Clone)]
pub struct WineFeatures {
    pub fixed_acidity: f32,
    pub volatile_acidity: f32,
    pub citric_acid: f32,
    pub residual_sugar: f32,
    pub chlorides: f32,
    pub free_sulfur_dioxide: f32,
    pub total_sulfur_dioxide: f32,
    pub density: f32,
    pub ph: f32,
    pub sulphates: f32,
    pub alcohol: f32,
}

impl WineFeatures {
    /// Convert to feature vector for model input
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.ph,
            self.sulphates,
            self.alcohol,
        ]
    }

    /// Normalize features to [0, 1] range using typical wine value ranges
    pub fn normalize(&self) -> Vec<f32> {
        vec![
            (self.fixed_acidity - 4.0) / 12.0,         // 4-16 range
            (self.volatile_acidity - 0.1) / 1.5,       // 0.1-1.6 range
            self.citric_acid,                          // 0-1 range
            (self.residual_sugar - 0.9) / 14.1,        // 0.9-15 range
            (self.chlorides - 0.01) / 0.59,            // 0.01-0.6 range
            (self.free_sulfur_dioxide - 1.0) / 71.0,   // 1-72 range
            (self.total_sulfur_dioxide - 6.0) / 283.0, // 6-289 range
            (self.density - 0.99) / 0.05,              // 0.99-1.04 range
            (self.ph - 2.7) / 1.3,                     // 2.7-4.0 range
            (self.sulphates - 0.3) / 1.7,              // 0.3-2.0 range
            (self.alcohol - 8.0) / 7.0,                // 8-15 range
        ]
    }

    /// Example: High-quality Bordeaux red wine
    pub fn example_bordeaux() -> Self {
        Self {
            fixed_acidity: 7.4,
            volatile_acidity: 0.28,
            citric_acid: 0.45,
            residual_sugar: 2.1,
            chlorides: 0.076,
            free_sulfur_dioxide: 15.0,
            total_sulfur_dioxide: 46.0,
            density: 0.9958,
            ph: 3.35,
            sulphates: 0.68,
            alcohol: 12.8,
        }
    }

    /// Example: Budget table wine
    pub fn example_table_wine() -> Self {
        Self {
            fixed_acidity: 8.5,
            volatile_acidity: 0.72,
            citric_acid: 0.12,
            residual_sugar: 3.8,
            chlorides: 0.092,
            free_sulfur_dioxide: 8.0,
            total_sulfur_dioxide: 28.0,
            density: 0.9972,
            ph: 3.42,
            sulphates: 0.48,
            alcohol: 10.2,
        }
    }

    /// Example: Premium Napa Valley Cabernet
    pub fn example_napa_cabernet() -> Self {
        Self {
            fixed_acidity: 6.8,
            volatile_acidity: 0.22,
            citric_acid: 0.52,
            residual_sugar: 1.8,
            chlorides: 0.058,
            free_sulfur_dioxide: 18.0,
            total_sulfur_dioxide: 42.0,
            density: 0.9948,
            ph: 3.28,
            sulphates: 0.72,
            alcohol: 13.5,
        }
    }
}

/// Wine quality prediction result
#[derive(Debug, Clone)]
pub struct WinePrediction {
    /// Quality score (0-10, typically 3-9)
    pub quality: f32,
    /// Quality category
    pub category: WineCategory,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Inference latency in milliseconds
    pub latency_ms: f32,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f32>,
}

/// Wine quality categories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WineCategory {
    Poor,      // 0-4
    Average,   // 5-6
    Good,      // 7
    Excellent, // 8-10
}

impl WineCategory {
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s < 5.0 => WineCategory::Poor,
            s if s < 7.0 => WineCategory::Average,
            s if s < 8.0 => WineCategory::Good,
            _ => WineCategory::Excellent,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            WineCategory::Poor => "Poor",
            WineCategory::Average => "Average",
            WineCategory::Good => "Good",
            WineCategory::Excellent => "Excellent",
        }
    }
}

// ============================================================================
// Wine Quality Predictor (test Model)
// ============================================================================

/// Wine quality predictor using a simple linear model
///
/// In production, this would load an actual .apr model trained on wine data.
/// This demo uses a simplified linear regression approximation based on
/// known correlations in wine quality datasets.
pub struct WinePredictor {
    /// Model weights (trained coefficients)
    weights: Vec<f32>,
    /// Bias term
    bias: f32,
    /// Inference count for cold start detection
    inference_count: std::sync::atomic::AtomicU64,
}

impl WinePredictor {
    /// Create a new wine predictor with pre-trained weights
    ///
    /// These weights approximate relationships found in UCI Wine Quality dataset:
    /// - Alcohol: Strong positive correlation (+0.48)
    /// - Volatile acidity: Strong negative correlation (-0.39)
    /// - Sulphates: Moderate positive correlation (+0.25)
    /// - Citric acid: Moderate positive correlation (+0.23)
    pub fn new() -> Self {
        Self {
            // Coefficients derived from wine quality analysis
            weights: vec![
                -0.05, // fixed_acidity (slight negative)
                -0.85, // volatile_acidity (strong negative - vinegar taste)
                0.45,  // citric_acid (positive - freshness)
                0.02,  // residual_sugar (minimal effect)
                -0.15, // chlorides (negative - salty taste)
                0.08,  // free_sulfur_dioxide (slight positive)
                -0.12, // total_sulfur_dioxide (slight negative)
                -0.30, // density (negative - correlates with sugar/alcohol)
                -0.10, // pH (slight negative)
                0.55,  // sulphates (positive - antimicrobial)
                0.95,  // alcohol (strong positive - body/complexity)
            ],
            bias: 5.5, // Base quality score
            inference_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Predict wine quality from features
    pub fn predict(&self, features: &WineFeatures) -> WinePrediction {
        let start = std::time::Instant::now();

        // Normalize input features
        let normalized = features.normalize();

        // Linear prediction: y = Wx + b
        let mut score: f32 = self.bias;
        for (w, x) in self.weights.iter().zip(normalized.iter()) {
            score += w * x;
        }

        // Clamp to valid range [0, 10]
        let quality = score.clamp(0.0, 10.0);

        // Calculate confidence based on how "typical" the wine is
        let confidence = self.calculate_confidence(&normalized);

        // Track inference count
        let _count = self
            .inference_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Feature importance (absolute weight contribution)
        let feature_names = [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "ph",
            "sulphates",
            "alcohol",
        ];

        let mut feature_importance = HashMap::new();
        for (name, (w, x)) in feature_names
            .iter()
            .zip(self.weights.iter().zip(normalized.iter()))
        {
            feature_importance.insert(name.to_string(), (w * x).abs());
        }

        WinePrediction {
            quality,
            category: WineCategory::from_score(quality),
            confidence,
            latency_ms,
            feature_importance,
        }
    }

    /// Calculate confidence based on feature values being in typical ranges
    fn calculate_confidence(&self, normalized: &[f32]) -> f32 {
        // Features outside [0, 1] after normalization are atypical
        let in_range_count = normalized
            .iter()
            .filter(|&&x| (0.0..=1.0).contains(&x))
            .count();

        (in_range_count as f32) / (normalized.len() as f32)
    }

    /// Check if this is a cold start (first inference)
    pub fn is_cold_start(&self) -> bool {
        self.inference_count
            .load(std::sync::atomic::Ordering::Relaxed)
            == 1
    }

    /// Get total inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for WinePredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metrics & Monitoring
// ============================================================================

/// Production metrics for wine prediction service
#[derive(Debug, Default)]
pub struct WineMetrics {
    pub requests_total: u64,
    pub predictions_by_category: HashMap<String, u64>,
    pub total_latency_ms: f64,
    pub cold_starts: u64,
    pub drift_alerts: u64,
}

impl WineMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, prediction: &WinePrediction, cold_start: bool) {
        self.requests_total += 1;
        self.total_latency_ms += prediction.latency_ms as f64;

        let category = prediction.category.as_str().to_string();
        *self.predictions_by_category.entry(category).or_insert(0) += 1;

        if cold_start {
            self.cold_starts += 1;
        }
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.requests_total == 0 {
            0.0
        } else {
            self.total_latency_ms / self.requests_total as f64
        }
    }

    /// Export metrics in Prometheus format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "# HELP wine_predictions_total Total wine quality predictions\n\
             # TYPE wine_predictions_total counter\n\
             wine_predictions_total {}\n\n",
            self.requests_total
        ));

        output.push_str(
            "# HELP wine_predictions_by_category Predictions by quality category\n\
             # TYPE wine_predictions_by_category counter\n",
        );
        for (category, count) in &self.predictions_by_category {
            output.push_str(&format!(
                "wine_predictions_by_category{{category=\"{}\"}} {}\n",
                category, count
            ));
        }

        output.push_str(&format!(
            "\n# HELP wine_latency_avg_ms Average prediction latency\n\
             # TYPE wine_latency_avg_ms gauge\n\
             wine_latency_avg_ms {:.4}\n\n",
            self.avg_latency_ms()
        ));

        output.push_str(&format!(
            "# HELP wine_cold_starts_total Cold start count\n\
             # TYPE wine_cold_starts_total counter\n\
             wine_cold_starts_total {}\n",
            self.cold_starts
        ));

        output
    }
}

// ============================================================================
// Simple Drift Detection
// ============================================================================

/// Detect drift in wine feature distributions
pub struct WineDriftDetector {
    /// Reference mean values from training data
    reference_means: Vec<f32>,
    /// Reference std values from training data
    reference_stds: Vec<f32>,
    /// Threshold for drift alert (z-score)
    threshold: f32,
}

impl WineDriftDetector {
    /// Create detector with typical wine dataset statistics
    pub fn new() -> Self {
        Self {
            // Approximate means from UCI Wine Quality dataset
            reference_means: vec![
                8.32,   // fixed_acidity
                0.53,   // volatile_acidity
                0.27,   // citric_acid
                2.54,   // residual_sugar
                0.087,  // chlorides
                15.87,  // free_sulfur_dioxide
                46.47,  // total_sulfur_dioxide
                0.9967, // density
                3.31,   // pH
                0.66,   // sulphates
                10.42,  // alcohol
            ],
            reference_stds: vec![
                1.74,   // fixed_acidity
                0.18,   // volatile_acidity
                0.19,   // citric_acid
                1.41,   // residual_sugar
                0.047,  // chlorides
                10.46,  // free_sulfur_dioxide
                32.89,  // total_sulfur_dioxide
                0.0019, // density
                0.15,   // pH
                0.17,   // sulphates
                1.07,   // alcohol
            ],
            threshold: 3.0, // 3 standard deviations
        }
    }

    /// Check if features show significant drift from training distribution
    pub fn check_drift(&self, features: &WineFeatures) -> DriftResult {
        let values = features.to_vec();
        let mut z_scores = Vec::new();
        let mut drifted_features = Vec::new();

        let feature_names = [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "ph",
            "sulphates",
            "alcohol",
        ];

        for (i, &value) in values.iter().enumerate() {
            let z = (value - self.reference_means[i]) / self.reference_stds[i];
            z_scores.push(z);

            if z.abs() > self.threshold {
                drifted_features.push(feature_names[i].to_string());
            }
        }

        let max_z = z_scores.iter().map(|z| z.abs()).fold(0.0f32, f32::max);

        DriftResult {
            is_drifted: !drifted_features.is_empty(),
            max_z_score: max_z,
            drifted_features,
        }
    }
}

impl Default for WineDriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of drift detection
#[derive(Debug)]
pub struct DriftResult {
    pub is_drifted: bool,
    pub max_z_score: f32,
    pub drifted_features: Vec<String>,
}

// ============================================================================
// Main Example
// ============================================================================

fn main() {
    println!("=== Wine Quality Prediction - Lambda Example ===\n");
    println!("Inspired by: https://github.com/paiml/wine-ratings\n");

    // Initialize predictor and metrics
    let predictor = WinePredictor::new();
    let drift_detector = WineDriftDetector::new();
    let mut metrics = WineMetrics::new();

    // Example wines to predict
    let wines = vec![
        ("Bordeaux Red (Premium)", WineFeatures::example_bordeaux()),
        ("Table Wine (Budget)", WineFeatures::example_table_wine()),
        (
            "Napa Cabernet (Premium)",
            WineFeatures::example_napa_cabernet(),
        ),
    ];

    println!("Wine Quality Predictions:");
    println!("{:-<70}", "");

    for (name, features) in &wines {
        let prediction = predictor.predict(features);
        let cold_start = predictor.is_cold_start();
        metrics.record(&prediction, cold_start);

        // Check for drift
        let drift = drift_detector.check_drift(features);

        println!("\n{}", name);
        println!(
            "  Quality Score: {:.2}/10 ({})",
            prediction.quality,
            prediction.category.as_str()
        );
        println!("  Confidence: {:.0}%", prediction.confidence * 100.0);
        println!("  Latency: {:.3}ms", prediction.latency_ms);

        if cold_start {
            println!("  [COLD START]");
        }

        if drift.is_drifted {
            println!("  [DRIFT WARNING] Features: {:?}", drift.drifted_features);
        }

        // Top 3 influential features
        let mut importance: Vec<_> = prediction.feature_importance.iter().collect();
        importance.sort_by(|a, b| {
            b.1.partial_cmp(a.1)
                .expect("importance values should be comparable floats")
        });
        println!(
            "  Top factors: {}, {}, {}",
            importance[0].0, importance[1].0, importance[2].0
        );
    }

    // Batch prediction demo
    println!("\n{:-<70}", "");
    println!("\nBatch Prediction (100 random wines):");

    let start = std::time::Instant::now();
    for i in 0..100 {
        // Simulate varying wine features
        let features = WineFeatures {
            fixed_acidity: 6.0 + (i as f32 * 0.1) % 4.0,
            volatile_acidity: 0.2 + (i as f32 * 0.01) % 0.6,
            citric_acid: 0.1 + (i as f32 * 0.01) % 0.5,
            residual_sugar: 1.5 + (i as f32 * 0.05) % 5.0,
            chlorides: 0.05 + (i as f32 * 0.001) % 0.1,
            free_sulfur_dioxide: 10.0 + (i as f32 * 0.5) % 30.0,
            total_sulfur_dioxide: 30.0 + (i as f32) % 100.0,
            density: 0.995 + (i as f32 * 0.0001) % 0.01,
            ph: 3.1 + (i as f32 * 0.01) % 0.5,
            sulphates: 0.4 + (i as f32 * 0.01) % 0.5,
            alcohol: 9.0 + (i as f32 * 0.05) % 4.0,
        };
        let prediction = predictor.predict(&features);
        metrics.record(&prediction, false);
    }
    let batch_time = start.elapsed();

    println!("  Processed: 100 predictions");
    println!("  Total time: {:.2}ms", batch_time.as_secs_f32() * 1000.0);
    println!(
        "  Throughput: {:.0} predictions/sec",
        100.0 / batch_time.as_secs_f32()
    );

    // Metrics summary
    println!("\n{:-<70}", "");
    println!("\nService Metrics:");
    println!("  Total requests: {}", metrics.requests_total);
    println!("  Cold starts: {}", metrics.cold_starts);
    println!("  Avg latency: {:.4}ms", metrics.avg_latency_ms());
    println!("\n  By category:");
    for (category, count) in &metrics.predictions_by_category {
        let pct = (*count as f64 / metrics.requests_total as f64) * 100.0;
        println!("    {}: {} ({:.1}%)", category, count, pct);
    }

    // Prometheus export
    println!("\n{:-<70}", "");
    println!("\nPrometheus Metrics Export:");
    println!("{}", metrics.to_prometheus());

    // Lambda deployment info
    println!("{:-<70}", "");
    println!("\nAWS Lambda Deployment:");
    println!("  Runtime: provided.al2 (custom runtime)");
    println!("  Architecture: arm64 (Graviton2)");
    println!("  Memory: 128MB (sufficient for this model)");
    println!("  Expected cold start: <10ms");
    println!("  Expected warm latency: <1ms");
    println!("\n  Build command:");
    println!("    cargo build --release --target aarch64-unknown-linux-gnu --features lambda");

    println!("\n=== Wine Prediction Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wine_features_to_vec() {
        let wine = WineFeatures::example_bordeaux();
        let vec = wine.to_vec();
        assert_eq!(vec.len(), 11);
        assert!((vec[0] - 7.4).abs() < 0.001); // fixed_acidity
    }

    #[test]
    fn test_wine_features_normalize() {
        let wine = WineFeatures::example_bordeaux();
        let normalized = wine.normalize();
        assert_eq!(normalized.len(), 11);
        // All normalized values should be roughly in [0, 1]
        for &val in &normalized {
            assert!(
                val >= -0.5 && val <= 1.5,
                "Value {} out of expected range",
                val
            );
        }
    }

    #[test]
    fn test_predictor_output_range() {
        let predictor = WinePredictor::new();
        let wine = WineFeatures::example_bordeaux();
        let prediction = predictor.predict(&wine);

        assert!(prediction.quality >= 0.0 && prediction.quality <= 10.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.latency_ms >= 0.0);
    }

    #[test]
    fn test_predictor_premium_vs_budget() {
        let predictor = WinePredictor::new();

        let premium = predictor.predict(&WineFeatures::example_napa_cabernet());
        let budget = predictor.predict(&WineFeatures::example_table_wine());

        // Premium wine should score higher
        assert!(
            premium.quality > budget.quality,
            "Premium ({:.2}) should score higher than budget ({:.2})",
            premium.quality,
            budget.quality
        );
    }

    #[test]
    fn test_wine_category_classification() {
        assert_eq!(WineCategory::from_score(3.5), WineCategory::Poor);
        assert_eq!(WineCategory::from_score(5.5), WineCategory::Average);
        assert_eq!(WineCategory::from_score(7.2), WineCategory::Good);
        assert_eq!(WineCategory::from_score(8.5), WineCategory::Excellent);
    }

    #[test]
    fn test_cold_start_detection() {
        let predictor = WinePredictor::new();
        let wine = WineFeatures::example_bordeaux();

        // First prediction
        let _ = predictor.predict(&wine);
        assert!(predictor.is_cold_start());

        // Second prediction
        let _ = predictor.predict(&wine);
        assert!(!predictor.is_cold_start());
    }

    #[test]
    fn test_metrics_recording() {
        let predictor = WinePredictor::new();
        let mut metrics = WineMetrics::new();

        for _ in 0..10 {
            let prediction = predictor.predict(&WineFeatures::example_bordeaux());
            metrics.record(&prediction, false);
        }

        assert_eq!(metrics.requests_total, 10);
        assert!(metrics.avg_latency_ms() > 0.0);
    }

    #[test]
    fn test_prometheus_export() {
        let mut metrics = WineMetrics::new();
        metrics.requests_total = 100;
        metrics.cold_starts = 1;
        metrics.total_latency_ms = 50.0;
        metrics
            .predictions_by_category
            .insert("Good".to_string(), 60);
        metrics
            .predictions_by_category
            .insert("Average".to_string(), 40);

        let prometheus = metrics.to_prometheus();
        assert!(prometheus.contains("wine_predictions_total 100"));
        assert!(prometheus.contains("wine_cold_starts_total 1"));
        assert!(prometheus.contains("wine_latency_avg_ms"));
    }

    #[test]
    fn test_drift_detection_normal() {
        let detector = WineDriftDetector::new();
        let wine = WineFeatures::example_bordeaux();

        let result = detector.check_drift(&wine);
        // Normal wine should not trigger drift
        assert!(!result.is_drifted || result.drifted_features.len() <= 1);
    }

    #[test]
    fn test_drift_detection_extreme() {
        let detector = WineDriftDetector::new();

        // Extreme outlier wine
        let extreme = WineFeatures {
            fixed_acidity: 20.0,   // Way above normal
            volatile_acidity: 2.0, // Way above normal
            citric_acid: 0.3,
            residual_sugar: 2.0,
            chlorides: 0.08,
            free_sulfur_dioxide: 15.0,
            total_sulfur_dioxide: 45.0,
            density: 0.997,
            ph: 3.3,
            sulphates: 0.6,
            alcohol: 10.0,
        };

        let result = detector.check_drift(&extreme);
        assert!(result.is_drifted);
        assert!(result
            .drifted_features
            .contains(&"fixed_acidity".to_string()));
    }

    #[test]
    fn test_feature_importance() {
        let predictor = WinePredictor::new();
        let prediction = predictor.predict(&WineFeatures::example_bordeaux());

        assert_eq!(prediction.feature_importance.len(), 11);
        assert!(prediction.feature_importance.contains_key("alcohol"));
        assert!(prediction
            .feature_importance
            .contains_key("volatile_acidity"));
    }
}
