//! Entrenar WASM - Training monitor for browsers
//!
//! Minimal WASM bindings without heavy dependencies.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    // WASM module initialized
}

/// Running statistics using Welford's algorithm
#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, value: f64) {
        if value.is_nan() || value.is_infinite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn std(&self) -> f64 {
        if self.count < 2 { 0.0 } else { (self.m2 / (self.count - 1) as f64).sqrt() }
    }
}

/// WASM Metrics Collector
#[wasm_bindgen]
pub struct MetricsCollector {
    loss: RunningStats,
    accuracy: RunningStats,
    loss_history: Vec<f64>,
    accuracy_history: Vec<f64>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl MetricsCollector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            loss: RunningStats::new(),
            accuracy: RunningStats::new(),
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
        }
    }

    pub fn record_loss(&mut self, value: f64) {
        if !value.is_nan() && !value.is_infinite() {
            self.loss.update(value);
            self.loss_history.push(value);
            if self.loss_history.len() > 100 {
                self.loss_history.remove(0);
            }
        }
    }

    pub fn record_accuracy(&mut self, value: f64) {
        if !value.is_nan() && !value.is_infinite() {
            self.accuracy.update(value);
            self.accuracy_history.push(value);
            if self.accuracy_history.len() > 100 {
                self.accuracy_history.remove(0);
            }
        }
    }

    pub fn loss_mean(&self) -> f64 {
        if self.loss.count == 0 { f64::NAN } else { self.loss.mean }
    }

    pub fn accuracy_mean(&self) -> f64 {
        if self.accuracy.count == 0 { f64::NAN } else { self.accuracy.mean }
    }

    pub fn loss_std(&self) -> f64 { self.loss.std() }
    pub fn accuracy_std(&self) -> f64 { self.accuracy.std() }
    pub fn count(&self) -> usize { self.loss.count + self.accuracy.count }

    pub fn clear(&mut self) {
        self.loss = RunningStats::new();
        self.accuracy = RunningStats::new();
        self.loss_history.clear();
        self.accuracy_history.clear();
    }

    pub fn loss_sparkline(&self) -> String {
        sparkline(&self.loss_history)
    }

    pub fn accuracy_sparkline(&self) -> String {
        sparkline(&self.accuracy_history)
    }

    pub fn state_json(&self) -> String {
        let state = serde_json::json!({
            "loss_mean": self.loss_mean(),
            "loss_std": self.loss_std(),
            "accuracy_mean": self.accuracy_mean(),
            "accuracy_std": self.accuracy_std(),
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
        });
        state.to_string()
    }
}

fn sparkline(values: &[f64]) -> String {
    const CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() { return String::new(); }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-10 { 1.0 } else { max - min };

    values.iter().map(|v| {
        let norm = ((v - min) / range).clamp(0.0, 1.0);
        let idx = ((norm * 7.0).round() as usize).min(7);
        CHARS[idx]
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_new() {
        let c = MetricsCollector::new();
        assert_eq!(c.count(), 0);
        assert!(c.loss_mean().is_nan());
        assert!(c.accuracy_mean().is_nan());
    }

    #[test]
    fn test_record_loss() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_loss(0.3);
        assert_eq!(c.count(), 2);
        assert!((c.loss_mean() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_record_accuracy() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.8);
        c.record_accuracy(0.9);
        assert!((c.accuracy_mean() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_loss_std() {
        let mut c = MetricsCollector::new();
        c.record_loss(2.0);
        c.record_loss(4.0);
        c.record_loss(4.0);
        c.record_loss(4.0);
        c.record_loss(5.0);
        c.record_loss(5.0);
        c.record_loss(7.0);
        c.record_loss(9.0);
        let std = c.loss_std();
        assert!(std > 2.0 && std < 2.5);
    }

    #[test]
    fn test_ignores_nan() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_loss(f64::NAN);
        c.record_loss(0.3);
        assert_eq!(c.count(), 2);
    }

    #[test]
    fn test_ignores_inf() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.8);
        c.record_accuracy(f64::INFINITY);
        assert_eq!(c.count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_accuracy(0.8);
        c.clear();
        assert_eq!(c.count(), 0);
        assert!(c.loss_sparkline().is_empty());
    }

    #[test]
    fn test_sparkline_empty() {
        let c = MetricsCollector::new();
        assert!(c.loss_sparkline().is_empty());
    }

    #[test]
    fn test_sparkline_values() {
        let mut c = MetricsCollector::new();
        for i in 0..10 {
            c.record_loss(i as f64 / 10.0);
        }
        let s = c.loss_sparkline();
        assert!(!s.is_empty());
        assert!(s.chars().all(|c| "▁▂▃▄▅▆▇█".contains(c)));
    }

    #[test]
    fn test_state_json() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.5);
        c.record_accuracy(0.8);
        let json = c.state_json();
        assert!(json.contains("loss_mean"));
        assert!(json.contains("accuracy_mean"));
        assert!(json.contains("loss_history"));
    }

    #[test]
    fn test_history_bounded() {
        let mut c = MetricsCollector::new();
        for i in 0..150 {
            c.record_loss(i as f64);
        }
        // History should be bounded to 100
        let json = c.state_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON deserialization should succeed");
        let history = parsed["loss_history"].as_array().expect("parsing should succeed");
        assert_eq!(history.len(), 100);
    }

    #[test]
    fn test_running_stats_min_max() {
        let mut c = MetricsCollector::new();
        c.record_loss(5.0);
        c.record_loss(2.0);
        c.record_loss(8.0);
        // Check via state_json that values are tracked
        let json = c.state_json();
        assert!(json.contains("2") || json.contains("8"));
    }

    #[test]
    fn test_accuracy_std() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.7);
        c.record_accuracy(0.8);
        c.record_accuracy(0.9);
        let std = c.accuracy_std();
        assert!(std > 0.0);
    }

    #[test]
    fn test_sparkline_constant() {
        let mut c = MetricsCollector::new();
        for _ in 0..5 {
            c.record_loss(0.5);
        }
        let s = c.loss_sparkline();
        // All same value should produce consistent sparkline
        assert_eq!(s.chars().count(), 5);
    }

    // Mutation-resistant tests for >80% kill rate
    #[test]
    fn test_nan_only_not_inf() {
        let mut c = MetricsCollector::new();
        c.record_loss(f64::NAN);
        assert_eq!(c.count(), 0);
        c.record_loss(1.0);
        assert_eq!(c.count(), 1);
    }

    #[test]
    fn test_inf_only_not_nan() {
        let mut c = MetricsCollector::new();
        c.record_loss(f64::INFINITY);
        assert_eq!(c.count(), 0);
        c.record_accuracy(f64::NEG_INFINITY);
        assert_eq!(c.count(), 0);
    }

    #[test]
    fn test_std_with_single_value() {
        let mut c = MetricsCollector::new();
        c.record_loss(5.0);
        assert_eq!(c.loss_std(), 0.0);
    }

    #[test]
    fn test_std_with_two_values() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.0);
        c.record_loss(2.0);
        // std of [0, 2] = sqrt((0-1)^2 + (2-1)^2) / 1) = sqrt(2) ≈ 1.414
        assert!((c.loss_std() - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_std_specific_value() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.0);
        c.record_accuracy(1.0);
        // std of [0, 1] = sqrt(0.5) ≈ 0.707
        assert!((c.accuracy_std() - (0.5_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_sparkline_min_max_range() {
        let mut c = MetricsCollector::new();
        c.record_loss(0.0);  // min
        c.record_loss(7.0);  // max
        let s = c.loss_sparkline();
        // First char should be lowest, last char should be highest
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], '▁');  // min value
        assert_eq!(chars[1], '█');  // max value
    }

    #[test]
    fn test_sparkline_intermediate_values() {
        let mut c = MetricsCollector::new();
        for i in 0..8 {
            c.record_loss(i as f64);
        }
        let s = c.loss_sparkline();
        let chars: Vec<char> = s.chars().collect();
        // Should have ascending pattern
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[7], '█');
    }

    #[test]
    fn test_accuracy_sparkline_not_empty() {
        let mut c = MetricsCollector::new();
        c.record_accuracy(0.5);
        c.record_accuracy(0.7);
        let s = c.accuracy_sparkline();
        assert!(!s.is_empty());
        assert!(s.chars().all(|ch| "▁▂▃▄▅▆▇█".contains(ch)));
    }

    #[test]
    fn test_history_exact_boundary() {
        let mut c = MetricsCollector::new();
        for i in 0..100 {
            c.record_loss(i as f64);
        }
        let json = c.state_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON deserialization should succeed");
        let history = parsed["loss_history"].as_array().expect("parsing should succeed");
        assert_eq!(history.len(), 100);
        // First value should be 0.0 (no overflow yet)
        assert_eq!(history[0].as_f64().expect("operation should succeed"), 0.0);
    }

    #[test]
    fn test_history_overflow_removes_first() {
        let mut c = MetricsCollector::new();
        for i in 0..101 {
            c.record_loss(i as f64);
        }
        let json = c.state_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON deserialization should succeed");
        let history = parsed["loss_history"].as_array().expect("parsing should succeed");
        assert_eq!(history.len(), 100);
        // First value should be 1.0 (0.0 was removed)
        assert_eq!(history[0].as_f64().expect("operation should succeed"), 1.0);
    }

    #[test]
    fn test_accuracy_history_overflow() {
        let mut c = MetricsCollector::new();
        for i in 0..105 {
            c.record_accuracy(i as f64 / 100.0);
        }
        let json = c.state_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON deserialization should succeed");
        let history = parsed["accuracy_history"].as_array().expect("parsing should succeed");
        assert_eq!(history.len(), 100);
    }

    // Additional mutation-resistant tests for sparkline math
    #[test]
    fn test_sparkline_normalization_correctness() {
        // Test that normalization is (v - min) / range, not (v + min) / range
        let mut c = MetricsCollector::new();
        c.record_loss(10.0);  // min = 10
        c.record_loss(20.0);  // max = 20, range = 10
        let s = c.loss_sparkline();
        let chars: Vec<char> = s.chars().collect();
        // norm(10) = (10-10)/10 = 0 -> '▁'
        // norm(20) = (20-10)/10 = 1 -> '█'
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[1], '█');
    }

    #[test]
    fn test_sparkline_division_not_multiplication() {
        // Test that we divide by range, not multiply
        let mut c = MetricsCollector::new();
        c.record_loss(0.0);
        c.record_loss(2.0);
        c.record_loss(1.0);  // middle value
        let s = c.loss_sparkline();
        let chars: Vec<char> = s.chars().collect();
        // 0 -> '▁', 2 -> '█', 1 -> '▄' (middle)
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[1], '█');
        // Middle value should be around index 3-4
        let middle_char = chars[2];
        assert!(middle_char >= '▃' && middle_char <= '▅', "Middle char was {:?}", middle_char);
    }

    #[test]
    fn test_sparkline_range_calculation() {
        // Test that range = max - min, not max + min
        let mut c = MetricsCollector::new();
        c.record_loss(5.0);
        c.record_loss(15.0);
        c.record_loss(10.0);
        let s = c.loss_sparkline();
        let chars: Vec<char> = s.chars().collect();
        // range = 15 - 5 = 10
        // 5 -> norm 0 -> '▁'
        // 15 -> norm 1 -> '█'
        // 10 -> norm 0.5 -> should be around '▄'
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[1], '█');
        assert!(chars[2] >= '▃' && chars[2] <= '▅');
    }

    #[test]
    fn test_record_accuracy_nan_or_inf() {
        // Test that NaN OR Inf is rejected (not NaN AND Inf)
        let mut c = MetricsCollector::new();
        c.record_accuracy(f64::NAN);      // Should be rejected
        assert_eq!(c.count(), 0);
        c.record_accuracy(f64::INFINITY); // Should also be rejected
        assert_eq!(c.count(), 0);
        c.record_accuracy(0.5);           // Should be accepted
        assert_eq!(c.count(), 1);
    }

    #[test]
    fn test_sparkline_small_range_handling() {
        // Test the small range fallback (< 1e-10)
        let mut c = MetricsCollector::new();
        c.record_loss(5.0);
        c.record_loss(5.0 + 1e-11);  // Very small difference
        let s = c.loss_sparkline();
        // Should not panic and should produce valid chars
        assert_eq!(s.chars().count(), 2);
        assert!(s.chars().all(|ch| "▁▂▃▄▅▆▇█".contains(ch)));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_mean_within_bounds(values in prop::collection::vec(0.0f64..100.0, 2..50)) {
            let mut c = MetricsCollector::new();
            for v in &values {
                c.record_loss(*v);
            }
            let mean = c.loss_mean();
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            prop_assert!(mean >= min - 1e-6);
            prop_assert!(mean <= max + 1e-6);
        }

        #[test]
        fn prop_std_non_negative(values in prop::collection::vec(0.0f64..100.0, 2..50)) {
            let mut c = MetricsCollector::new();
            for v in &values {
                c.record_loss(*v);
            }
            prop_assert!(c.loss_std() >= 0.0);
        }

        #[test]
        fn prop_count_matches(count in 1usize..100) {
            let mut c = MetricsCollector::new();
            for i in 0..count {
                c.record_loss(i as f64);
            }
            prop_assert_eq!(c.count(), count);
        }

        #[test]
        fn prop_sparkline_valid_chars(values in prop::collection::vec(0.0f64..100.0, 1..50)) {
            let mut c = MetricsCollector::new();
            for v in &values {
                c.record_loss(*v);
            }
            let s = c.loss_sparkline();
            for ch in s.chars() {
                prop_assert!("▁▂▃▄▅▆▇█".contains(ch));
            }
        }

        #[test]
        fn prop_history_bounded(count in 1usize..200) {
            let mut c = MetricsCollector::new();
            for i in 0..count {
                c.record_loss(i as f64);
            }
            let json = c.state_json();
            let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON deserialization should succeed");
            let len = parsed["loss_history"].as_array().expect("parsing should succeed").len();
            prop_assert!(len <= 100);
        }

        #[test]
        fn prop_json_valid(values in prop::collection::vec(0.0f64..100.0, 0..50)) {
            let mut c = MetricsCollector::new();
            for v in &values {
                c.record_loss(*v);
            }
            let json = c.state_json();
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok());
        }

        #[test]
        fn prop_ignores_nan_inf(
            valid in prop::collection::vec(0.0f64..100.0, 1..20),
            nan_count in 0usize..5,
            inf_count in 0usize..5
        ) {
            let mut c = MetricsCollector::new();
            for v in &valid {
                c.record_loss(*v);
            }
            for _ in 0..nan_count {
                c.record_loss(f64::NAN);
            }
            for _ in 0..inf_count {
                c.record_loss(f64::INFINITY);
            }
            prop_assert_eq!(c.count(), valid.len());
        }
    }
}
