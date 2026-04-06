//! Statistical Analysis Module (PROBAR-SPEC-006 Section I)
//!
//! Implements variance decomposition, Apdex scoring, knee detection,
//! and quantile regression for load test analysis.
//!
//! Based on research:
//! - [C8] VProfiler variance trees
//! - [C9] Treadmill tail latency attribution
//! - [C12] Tail at Scale methodology

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::use_self)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::useless_format)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// I.2 Variance Tree (following [C8] VProfiler methodology)
// =============================================================================

/// Hierarchical variance decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceTree {
    /// Total variance across all components
    pub total_variance: f64,
    /// Root-level components
    pub components: Vec<VarianceComponent>,
}

impl VarianceTree {
    /// Create a new variance tree
    pub fn new() -> Self {
        Self {
            total_variance: 0.0,
            components: Vec::new(),
        }
    }

    /// Add a component
    pub fn add_component(&mut self, component: VarianceComponent) {
        self.total_variance += component.variance;
        self.components.push(component);
    }

    /// Recalculate percentages based on total variance
    pub fn recalculate_percentages(&mut self) {
        if self.total_variance > 0.0 {
            for comp in &mut self.components {
                comp.percentage = (comp.variance / self.total_variance) * 100.0;
                comp.recalculate_percentages(self.total_variance);
            }
        }
    }

    /// Build from latency samples with component attribution
    pub fn from_samples(samples: &[LatencySample]) -> Self {
        let mut tree = Self::new();

        // Group by component
        let mut component_samples: HashMap<String, Vec<f64>> = HashMap::new();
        for sample in samples {
            for (component, latency) in &sample.components {
                component_samples
                    .entry(component.clone())
                    .or_default()
                    .push(*latency);
            }
        }

        // Calculate variance for each component
        for (name, values) in component_samples {
            let variance = calculate_variance(&values);
            tree.add_component(VarianceComponent {
                name,
                variance,
                percentage: 0.0,
                children: Vec::new(),
            });
        }

        tree.recalculate_percentages();
        tree
    }
}

impl Default for VarianceTree {
    fn default() -> Self {
        Self::new()
    }
}

/// A component in the variance tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceComponent {
    /// Component name (e.g., "Network I/O", "WASM Execution")
    pub name: String,
    /// Variance contribution in ms²
    pub variance: f64,
    /// Percentage of total variance
    pub percentage: f64,
    /// Child components
    pub children: Vec<VarianceComponent>,
}

impl VarianceComponent {
    /// Create a new component
    pub fn new(name: &str, variance: f64) -> Self {
        Self {
            name: name.to_string(),
            variance,
            percentage: 0.0,
            children: Vec::new(),
        }
    }

    /// Add a child component
    pub fn add_child(&mut self, child: VarianceComponent) {
        self.children.push(child);
    }

    /// Recalculate child percentages
    fn recalculate_percentages(&mut self, total: f64) {
        for child in &mut self.children {
            child.percentage = (child.variance / total) * 100.0;
            child.recalculate_percentages(total);
        }
    }
}

/// Latency sample with component breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySample {
    /// Total latency in ms
    pub total_ms: f64,
    /// Breakdown by component
    pub components: HashMap<String, f64>,
    /// Timestamp
    pub timestamp_ms: u64,
}

// =============================================================================
// I.2 Apdex Score
// =============================================================================

/// Apdex (Application Performance Index) calculator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApdexCalculator {
    /// Satisfied threshold in ms (requests below this are "satisfied")
    pub satisfied_threshold_ms: u64,
    /// Tolerating threshold in ms (requests below this are "tolerating")
    pub tolerating_threshold_ms: u64,
    /// Count of satisfied requests
    satisfied_count: u64,
    /// Count of tolerating requests
    tolerating_count: u64,
    /// Count of frustrated requests
    frustrated_count: u64,
}

impl ApdexCalculator {
    /// Create a new Apdex calculator
    /// Default: T = 100ms (satisfied), 4T = 400ms (tolerating)
    pub fn new(satisfied_ms: u64, tolerating_ms: u64) -> Self {
        Self {
            satisfied_threshold_ms: satisfied_ms,
            tolerating_threshold_ms: tolerating_ms,
            satisfied_count: 0,
            tolerating_count: 0,
            frustrated_count: 0,
        }
    }

    /// Record a latency sample
    pub fn record(&mut self, latency_ms: u64) {
        if latency_ms <= self.satisfied_threshold_ms {
            self.satisfied_count += 1;
        } else if latency_ms <= self.tolerating_threshold_ms {
            self.tolerating_count += 1;
        } else {
            self.frustrated_count += 1;
        }
    }

    /// Calculate Apdex score (0.0 to 1.0)
    pub fn score(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            return 1.0; // No data = perfect
        }
        (self.satisfied_count as f64 + self.tolerating_count as f64 / 2.0) / total as f64
    }

    /// Get total count
    pub fn total_count(&self) -> u64 {
        self.satisfied_count + self.tolerating_count + self.frustrated_count
    }

    /// Get satisfied count
    pub fn satisfied(&self) -> u64 {
        self.satisfied_count
    }

    /// Get tolerating count
    pub fn tolerating(&self) -> u64 {
        self.tolerating_count
    }

    /// Get frustrated count
    pub fn frustrated(&self) -> u64 {
        self.frustrated_count
    }

    /// Get rating based on score
    pub fn rating(&self) -> ApdexRating {
        let score = self.score();
        if score >= 0.94 {
            ApdexRating::Excellent
        } else if score >= 0.85 {
            ApdexRating::Good
        } else if score >= 0.70 {
            ApdexRating::Fair
        } else if score >= 0.50 {
            ApdexRating::Poor
        } else {
            ApdexRating::Unacceptable
        }
    }

    /// Reset calculator
    pub fn reset(&mut self) {
        self.satisfied_count = 0;
        self.tolerating_count = 0;
        self.frustrated_count = 0;
    }
}

impl Default for ApdexCalculator {
    fn default() -> Self {
        Self::new(100, 400) // T=100ms, 4T=400ms
    }
}

/// Apdex rating levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ApdexRating {
    /// 0.94 - 1.00
    Excellent,
    /// 0.85 - 0.93
    Good,
    /// 0.70 - 0.84
    Fair,
    /// 0.50 - 0.69
    Poor,
    /// 0.00 - 0.49
    Unacceptable,
}

impl ApdexRating {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Unacceptable => "Unacceptable",
        }
    }
}

// =============================================================================
// I.2 Throughput Knee Detection (following [C12] Tail at Scale)
// =============================================================================

/// Knee detector for finding throughput inflection point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KneeDetector {
    /// Data points (load, latency)
    points: Vec<(f64, f64)>,
    /// Detected knee point (load, latency)
    pub knee_point: Option<(f64, f64)>,
    /// Recommended capacity (80% of knee)
    pub recommended_capacity: Option<f64>,
}

impl KneeDetector {
    /// Create a new knee detector
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            knee_point: None,
            recommended_capacity: None,
        }
    }

    /// Add a data point (load level, latency)
    pub fn add_point(&mut self, load: f64, latency: f64) {
        self.points.push((load, latency));
    }

    /// Detect the knee point using second derivative
    pub fn detect(&mut self) {
        if self.points.len() < 3 {
            return;
        }

        // Sort by load
        self.points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate second derivative (approximation)
        let mut max_curvature = 0.0;
        let mut knee_idx = 0;

        for i in 1..self.points.len() - 1 {
            let (x0, y0) = self.points[i - 1];
            let (x1, y1) = self.points[i];
            let (x2, y2) = self.points[i + 1];

            // First derivatives
            let dy1 = (y1 - y0) / (x1 - x0);
            let dy2 = (y2 - y1) / (x2 - x1);

            // Second derivative (curvature approximation)
            let d2y = (dy2 - dy1) / ((x2 - x0) / 2.0);

            if d2y > max_curvature {
                max_curvature = d2y;
                knee_idx = i;
            }
        }

        if max_curvature > 0.0 {
            self.knee_point = Some(self.points[knee_idx]);
            self.recommended_capacity = Some(self.points[knee_idx].0 * 0.8);
        }
    }

    /// Get points
    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }
}

impl Default for KneeDetector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// I.2 Tail Latency Attribution (following [C9] Treadmill)
// =============================================================================

/// Tail latency attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailAttribution {
    /// Percentile (e.g., 99)
    pub percentile: u8,
    /// Latency value at this percentile
    pub latency_ms: u64,
    /// Primary cause
    pub primary_cause: String,
    /// Contributing factors with weights
    pub contributing_factors: Vec<(String, f64)>,
}

impl TailAttribution {
    /// Create a new attribution
    pub fn new(percentile: u8, latency_ms: u64, primary_cause: &str) -> Self {
        Self {
            percentile,
            latency_ms,
            primary_cause: primary_cause.to_string(),
            contributing_factors: Vec::new(),
        }
    }

    /// Add a contributing factor
    pub fn add_factor(&mut self, factor: &str, weight: f64) {
        self.contributing_factors.push((factor.to_string(), weight));
    }
}

/// Quantile regression results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRegression {
    /// Quantiles analyzed
    pub quantiles: Vec<f64>,
    /// Attributions for each quantile
    pub attributions: Vec<TailAttribution>,
}

impl QuantileRegression {
    /// Create new quantile regression
    pub fn new() -> Self {
        Self {
            quantiles: vec![0.5, 0.9, 0.95, 0.99, 0.999],
            attributions: Vec::new(),
        }
    }

    /// Add an attribution
    pub fn add_attribution(&mut self, attr: TailAttribution) {
        self.attributions.push(attr);
    }
}

impl Default for QuantileRegression {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// I.3 Statistical Analysis Report
// =============================================================================

/// Complete statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Scenario name
    pub scenario_name: String,
    /// Variance tree
    pub variance_tree: VarianceTree,
    /// Apdex calculator
    pub apdex: ApdexCalculator,
    /// Knee detector
    pub knee_detector: KneeDetector,
    /// Quantile regression
    pub quantile_regression: QuantileRegression,
    /// Coefficient of variation (σ/μ)
    pub coefficient_of_variation: f64,
}

impl StatisticalAnalysis {
    /// Create new analysis
    pub fn new(scenario_name: &str) -> Self {
        Self {
            scenario_name: scenario_name.to_string(),
            variance_tree: VarianceTree::new(),
            apdex: ApdexCalculator::default(),
            knee_detector: KneeDetector::new(),
            quantile_regression: QuantileRegression::new(),
            coefficient_of_variation: 0.0,
        }
    }
}

// =============================================================================
// Rendering
// =============================================================================

/// Render statistical analysis as TUI
pub fn render_statistical_report(analysis: &StatisticalAnalysis) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "STATISTICAL ANALYSIS: {}\n",
        analysis.scenario_name
    ));
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    // Variance decomposition
    out.push_str("VARIANCE DECOMPOSITION\n");
    out.push_str("┌───────────────────────────────────────────────────────────────┐\n");
    out.push_str(&format!(
        "│ Total Latency Variance: {:.0} ms²                              │\n",
        analysis.variance_tree.total_variance
    ));
    out.push_str("│                                                               │\n");

    for comp in &analysis.variance_tree.components {
        let bar_len = (comp.percentage / 5.0) as usize;
        let bar: String = "█".repeat(bar_len.min(20));
        out.push_str(&format!(
            "│ ├── {:<12}: {:>6.0} ms² ({:>4.1}%)  {:20} │\n",
            truncate(&comp.name, 12),
            comp.variance,
            comp.percentage,
            bar
        ));
    }
    out.push_str("└───────────────────────────────────────────────────────────────┘\n\n");

    // Apdex
    out.push_str("APDEX SCORE\n");
    out.push_str("┌───────────────────────────────────────────────────────────────┐\n");
    out.push_str(&format!(
        "│ Target: {}ms (Satisfied), {}ms (Tolerating)               │\n",
        analysis.apdex.satisfied_threshold_ms, analysis.apdex.tolerating_threshold_ms
    ));
    out.push_str(&format!(
        "│                                                               │\n"
    ));
    out.push_str(&format!(
        "│ Satisfied:  {:>6} requests ({:>4.1}%)                        │\n",
        analysis.apdex.satisfied(),
        (analysis.apdex.satisfied() as f64 / analysis.apdex.total_count().max(1) as f64) * 100.0
    ));
    out.push_str(&format!(
        "│ Tolerating: {:>6} requests ({:>4.1}%)                        │\n",
        analysis.apdex.tolerating(),
        (analysis.apdex.tolerating() as f64 / analysis.apdex.total_count().max(1) as f64) * 100.0
    ));
    out.push_str(&format!(
        "│ Frustrated: {:>6} requests ({:>4.1}%)                        │\n",
        analysis.apdex.frustrated(),
        (analysis.apdex.frustrated() as f64 / analysis.apdex.total_count().max(1) as f64) * 100.0
    ));
    out.push_str(&format!(
        "│                                                               │\n"
    ));
    out.push_str(&format!(
        "│ Apdex Score: {:.2} ({})                                     │\n",
        analysis.apdex.score(),
        analysis.apdex.rating().as_str()
    ));
    out.push_str("└───────────────────────────────────────────────────────────────┘\n\n");

    // Knee detection
    if let Some((load, latency)) = analysis.knee_detector.knee_point {
        out.push_str("THROUGHPUT KNEE DETECTION\n");
        out.push_str("┌───────────────────────────────────────────────────────────────┐\n");
        out.push_str(&format!(
            "│ Knee detected at: {:.0} concurrent users                      │\n",
            load
        ));
        out.push_str(&format!(
            "│ Latency at knee: {:.0}ms                                       │\n",
            latency
        ));
        if let Some(rec) = analysis.knee_detector.recommended_capacity {
            out.push_str(&format!(
                "│ Recommended capacity: {:.0} users (80% of knee)               │\n",
                rec
            ));
        }
        out.push_str("└───────────────────────────────────────────────────────────────┘\n\n");
    }

    out
}

/// Render as JSON
pub fn render_statistical_json(analysis: &StatisticalAnalysis) -> String {
    serde_json::to_string_pretty(analysis).unwrap_or_else(|_| "{}".to_string())
}

// =============================================================================
// Helper functions
// =============================================================================

/// Calculate variance of a slice
fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance
}

/// Truncate string
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_component() {
        let comp = VarianceComponent::new("Network", 1000.0);
        assert_eq!(comp.name, "Network");
        assert_eq!(comp.variance, 1000.0);
    }

    #[test]
    fn test_variance_tree() {
        let mut tree = VarianceTree::new();
        tree.add_component(VarianceComponent::new("Network", 600.0));
        tree.add_component(VarianceComponent::new("WASM", 400.0));
        tree.recalculate_percentages();

        assert_eq!(tree.total_variance, 1000.0);
        assert_eq!(tree.components[0].percentage, 60.0);
        assert_eq!(tree.components[1].percentage, 40.0);
    }

    #[test]
    fn test_apdex_calculator() {
        let mut apdex = ApdexCalculator::new(100, 400);

        // 8 satisfied, 2 tolerating
        for _ in 0..8 {
            apdex.record(50);
        }
        for _ in 0..2 {
            apdex.record(200);
        }

        assert_eq!(apdex.satisfied(), 8);
        assert_eq!(apdex.tolerating(), 2);
        assert_eq!(apdex.frustrated(), 0);
        // Score = (8 + 2/2) / 10 = 0.9
        assert!((apdex.score() - 0.9).abs() < 0.001);
        assert_eq!(apdex.rating(), ApdexRating::Good);
    }

    #[test]
    fn test_apdex_frustrated() {
        let mut apdex = ApdexCalculator::new(100, 400);
        apdex.record(500); // Frustrated
        apdex.record(1000); // Frustrated

        assert_eq!(apdex.frustrated(), 2);
        assert_eq!(apdex.score(), 0.0);
        assert_eq!(apdex.rating(), ApdexRating::Unacceptable);
    }

    #[test]
    fn test_apdex_rating() {
        assert_eq!(ApdexRating::Excellent.as_str(), "Excellent");
        assert_eq!(ApdexRating::Poor.as_str(), "Poor");
    }

    #[test]
    fn test_knee_detector() {
        let mut detector = KneeDetector::new();

        // Simulate linear then exponential growth
        detector.add_point(10.0, 50.0);
        detector.add_point(20.0, 55.0);
        detector.add_point(30.0, 60.0);
        detector.add_point(40.0, 70.0);
        detector.add_point(50.0, 100.0);
        detector.add_point(60.0, 200.0);
        detector.add_point(70.0, 400.0);

        detector.detect();

        assert!(detector.knee_point.is_some());
        assert!(detector.recommended_capacity.is_some());
    }

    #[test]
    fn test_tail_attribution() {
        let mut attr = TailAttribution::new(99, 456, "Network congestion");
        attr.add_factor("DNS", 0.15);
        attr.add_factor("TLS handshake", 0.25);

        assert_eq!(attr.percentile, 99);
        assert_eq!(attr.contributing_factors.len(), 2);
    }

    #[test]
    fn test_quantile_regression() {
        let mut qr = QuantileRegression::new();
        qr.add_attribution(TailAttribution::new(50, 78, "Typical case"));
        qr.add_attribution(TailAttribution::new(99, 456, "Network"));

        assert_eq!(qr.attributions.len(), 2);
    }

    #[test]
    fn test_calculate_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let variance = calculate_variance(&values);
        // Mean = 5, Variance = 4
        assert!((variance - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_statistical_analysis() {
        let analysis = StatisticalAnalysis::new("Test Scenario");
        assert_eq!(analysis.scenario_name, "Test Scenario");
    }

    #[test]
    fn test_render_statistical_report() {
        let mut analysis = StatisticalAnalysis::new("WASM Boot");
        analysis
            .variance_tree
            .add_component(VarianceComponent::new("Network", 500.0));
        analysis.variance_tree.recalculate_percentages();
        analysis.apdex.record(50);
        analysis.apdex.record(100);

        let report = render_statistical_report(&analysis);
        assert!(report.contains("WASM Boot"));
        assert!(report.contains("VARIANCE"));
        assert!(report.contains("APDEX"));
    }

    #[test]
    fn test_render_statistical_json() {
        let analysis = StatisticalAnalysis::new("JSON Test");
        let json = render_statistical_json(&analysis);
        assert!(json.contains("JSON Test"));
    }

    #[test]
    fn test_variance_tree_default() {
        let tree = VarianceTree::default();
        assert_eq!(tree.total_variance, 0.0);
        assert!(tree.components.is_empty());
    }

    #[test]
    fn test_variance_tree_from_samples() {
        let mut samples = Vec::new();

        // Create sample 1
        let mut components1 = HashMap::new();
        components1.insert("network".to_string(), 50.0);
        components1.insert("compute".to_string(), 20.0);
        samples.push(LatencySample {
            total_ms: 70.0,
            components: components1,
            timestamp_ms: 1000,
        });

        // Create sample 2
        let mut components2 = HashMap::new();
        components2.insert("network".to_string(), 60.0);
        components2.insert("compute".to_string(), 25.0);
        samples.push(LatencySample {
            total_ms: 85.0,
            components: components2,
            timestamp_ms: 2000,
        });

        // Create sample 3
        let mut components3 = HashMap::new();
        components3.insert("network".to_string(), 55.0);
        components3.insert("compute".to_string(), 22.0);
        samples.push(LatencySample {
            total_ms: 77.0,
            components: components3,
            timestamp_ms: 3000,
        });

        let tree = VarianceTree::from_samples(&samples);

        // Should have 2 components (network and compute)
        assert_eq!(tree.components.len(), 2);
        assert!(tree.total_variance > 0.0);

        // Each component should have a percentage
        for comp in &tree.components {
            assert!(comp.percentage >= 0.0);
            assert!(comp.percentage <= 100.0);
        }
    }

    #[test]
    fn test_variance_tree_recalculate_empty() {
        let mut tree = VarianceTree::new();
        // Should not panic with empty tree
        tree.recalculate_percentages();
        assert_eq!(tree.total_variance, 0.0);
    }

    #[test]
    fn test_variance_tree_recalculate_with_children() {
        let mut tree = VarianceTree::new();

        // Add parent component with children
        let mut parent = VarianceComponent::new("parent", 100.0);
        parent.add_child(VarianceComponent::new("child1", 60.0));
        parent.add_child(VarianceComponent::new("child2", 40.0));

        tree.add_component(parent);
        tree.recalculate_percentages();

        // Parent should be 100% of tree
        assert!((tree.components[0].percentage - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_sample() {
        let mut components = HashMap::new();
        components.insert("dns".to_string(), 10.0);
        components.insert("connect".to_string(), 20.0);

        let sample = LatencySample {
            total_ms: 30.0,
            components,
            timestamp_ms: 1234567890,
        };

        assert_eq!(sample.total_ms, 30.0);
        assert_eq!(sample.components.len(), 2);
    }

    #[test]
    fn test_apdex_reset() {
        let mut apdex = ApdexCalculator::new(100, 400);
        apdex.record(50);
        apdex.record(200);
        apdex.record(500);

        assert_eq!(apdex.total_count(), 3);

        apdex.reset();
        assert_eq!(apdex.total_count(), 0);
        assert_eq!(apdex.satisfied(), 0);
        assert_eq!(apdex.tolerating(), 0);
        assert_eq!(apdex.frustrated(), 0);
    }

    #[test]
    fn test_apdex_default() {
        let apdex = ApdexCalculator::default();
        assert_eq!(apdex.satisfied_threshold_ms, 100);
        assert_eq!(apdex.tolerating_threshold_ms, 400);
    }

    #[test]
    fn test_apdex_rating_fair() {
        let mut apdex = ApdexCalculator::new(100, 400);
        // Need score between 0.70 and 0.84
        // 7 satisfied + 1 tolerating + 2 frustrated = (7 + 0.5) / 10 = 0.75
        for _ in 0..7 {
            apdex.record(50); // satisfied
        }
        apdex.record(200); // tolerating
        apdex.record(500); // frustrated
        apdex.record(600); // frustrated

        assert_eq!(apdex.rating(), ApdexRating::Fair);
    }

    #[test]
    fn test_apdex_rating_excellent() {
        let mut apdex = ApdexCalculator::new(100, 400);
        // Need score >= 0.94
        for _ in 0..10 {
            apdex.record(50); // all satisfied
        }
        assert_eq!(apdex.rating(), ApdexRating::Excellent);
    }

    #[test]
    fn test_apdex_empty_score() {
        let apdex = ApdexCalculator::new(100, 400);
        // Empty = perfect score
        assert_eq!(apdex.score(), 1.0);
    }

    #[test]
    fn test_apdex_rating_all_variants() {
        assert_eq!(ApdexRating::Excellent.as_str(), "Excellent");
        assert_eq!(ApdexRating::Good.as_str(), "Good");
        assert_eq!(ApdexRating::Fair.as_str(), "Fair");
        assert_eq!(ApdexRating::Poor.as_str(), "Poor");
        assert_eq!(ApdexRating::Unacceptable.as_str(), "Unacceptable");
    }

    #[test]
    fn test_knee_detector_insufficient_points() {
        let mut detector = KneeDetector::new();
        detector.add_point(10.0, 50.0);
        detector.add_point(20.0, 60.0);
        // Only 2 points, need at least 3
        detector.detect();
        assert!(detector.knee_point.is_none());
    }

    #[test]
    fn test_knee_detector_default() {
        let detector = KneeDetector::default();
        assert!(detector.knee_point.is_none());
        assert!(detector.recommended_capacity.is_none());
    }

    #[test]
    fn test_calculate_variance_empty() {
        let empty: Vec<f64> = vec![];
        assert_eq!(calculate_variance(&empty), 0.0);
    }

    #[test]
    fn test_calculate_variance_single() {
        let single = vec![5.0];
        assert_eq!(calculate_variance(&single), 0.0);
    }

    #[test]
    fn test_quantile_regression_default() {
        let qr = QuantileRegression::default();
        assert!(qr.attributions.is_empty());
    }

    #[test]
    fn test_variance_component_with_child_percentages() {
        let mut parent = VarianceComponent::new("parent", 100.0);
        let mut child1 = VarianceComponent::new("child1", 60.0);
        child1.percentage = 60.0;
        let mut child2 = VarianceComponent::new("child2", 40.0);
        child2.percentage = 40.0;
        parent.add_child(child1);
        parent.add_child(child2);

        // Recalculate based on total_variance=200.0
        parent.recalculate_percentages(200.0);

        // Children percentages should be recalculated relative to 200
        assert_eq!(parent.children[0].percentage, 30.0); // 60/200 * 100
        assert_eq!(parent.children[1].percentage, 20.0); // 40/200 * 100
    }

    #[test]
    fn test_statistical_analysis_all_fields() {
        let mut analysis = StatisticalAnalysis::new("Full Test");

        // Add variance data
        analysis
            .variance_tree
            .add_component(VarianceComponent::new("A", 100.0));
        analysis.variance_tree.recalculate_percentages();

        // Add apdex data
        analysis.apdex.record(50);
        analysis.apdex.record(200);

        // Add knee detection
        analysis.knee_detector.add_point(10.0, 50.0);
        analysis.knee_detector.add_point(20.0, 55.0);
        analysis.knee_detector.add_point(30.0, 80.0);
        analysis.knee_detector.detect();

        // Add quantile regression
        analysis
            .quantile_regression
            .add_attribution(TailAttribution::new(95, 200, "Network latency"));

        // Render should include all sections
        let report = render_statistical_report(&analysis);
        assert!(report.contains("Full Test"));
        assert!(report.contains("VARIANCE"));
        assert!(report.contains("APDEX"));
    }

    #[test]
    fn test_tail_attribution_multiple_factors() {
        let mut attr = TailAttribution::new(99, 500, "High tail latency");
        attr.add_factor("GC pause", 0.45);
        attr.add_factor("Network", 0.30);
        attr.add_factor("Disk I/O", 0.25);

        assert_eq!(attr.contributing_factors.len(), 3);
        assert_eq!(attr.contributing_factors[0].0, "GC pause");
        assert!((attr.contributing_factors[0].1 - 0.45).abs() < 0.01);
    }
}
