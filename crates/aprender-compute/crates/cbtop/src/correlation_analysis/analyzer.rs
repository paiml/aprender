//! Correlation analyzer for multi-metric performance analysis.

use std::collections::VecDeque;

use super::{
    CorrelationResult, EventSample, EventType, InterferenceResult, IsolationAction,
    IsolationRecommendation, PerformanceSample, SystemSnapshot,
};

/// Correlation analyzer
#[derive(Debug)]
pub struct CorrelationAnalyzer {
    /// Performance samples
    perf_samples: VecDeque<PerformanceSample>,
    /// Event samples by type
    event_samples: VecDeque<EventSample>,
    /// System snapshots
    snapshots: VecDeque<SystemSnapshot>,
    /// Maximum samples to keep
    max_samples: usize,
    /// CV spike threshold (%)
    spike_threshold: f64,
    /// Correlation window (seconds)
    window_sec: f64,
}

impl Default for CorrelationAnalyzer {
    fn default() -> Self {
        Self {
            perf_samples: VecDeque::new(),
            event_samples: VecDeque::new(),
            snapshots: VecDeque::new(),
            max_samples: 1000,
            spike_threshold: 15.0,
            window_sec: 60.0,
        }
    }
}

impl CorrelationAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum samples
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self
    }

    /// Set CV spike threshold
    pub fn with_spike_threshold(mut self, percent: f64) -> Self {
        self.spike_threshold = percent;
        self
    }

    /// Set correlation window
    pub fn with_window(mut self, seconds: f64) -> Self {
        self.window_sec = seconds;
        self
    }

    /// Add performance sample
    pub fn add_perf_sample(&mut self, sample: PerformanceSample) {
        if self.perf_samples.len() >= self.max_samples {
            self.perf_samples.pop_front();
        }
        self.perf_samples.push_back(sample);
    }

    /// Add event sample
    pub fn add_event_sample(&mut self, sample: EventSample) {
        if self.event_samples.len() >= self.max_samples {
            self.event_samples.pop_front();
        }
        self.event_samples.push_back(sample);
    }

    /// Add system snapshot
    pub fn add_snapshot(&mut self, snapshot: SystemSnapshot) {
        if self.snapshots.len() >= self.max_samples / 10 {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Get performance sample count
    pub fn perf_sample_count(&self) -> usize {
        self.perf_samples.len()
    }

    /// Get event sample count
    pub fn event_sample_count(&self) -> usize {
        self.event_samples.len()
    }

    /// Calculate correlation between CV and event type
    pub fn correlate_events(&self, event_type: EventType) -> Option<CorrelationResult> {
        let events: Vec<_> =
            self.event_samples.iter().filter(|e| e.event_type == event_type).collect();

        if events.len() < 5 || self.perf_samples.len() < 5 {
            return None;
        }

        // Simple Pearson correlation at lag 0
        let pearson_r = self.compute_correlation(&events);

        let is_significant = pearson_r.abs() > 0.3 && events.len() >= 10;

        Some(CorrelationResult {
            event_type,
            pearson_r,
            sample_count: events.len().min(self.perf_samples.len()),
            is_significant,
            optimal_lag: 0.0,
        })
    }

    /// Compute Pearson correlation
    fn compute_correlation(&self, events: &[&EventSample]) -> f64 {
        // Match events to closest perf samples by timestamp
        let mut paired: Vec<(f64, f64)> = Vec::new();

        for event in events {
            if let Some(perf) = self.find_closest_perf(event.timestamp) {
                paired.push((event.value, perf.cv_percent));
            }
        }

        if paired.len() < 3 {
            return 0.0;
        }

        let n = paired.len() as f64;
        let sum_x: f64 = paired.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = paired.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = paired.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = paired.iter().map(|(x, _)| x * x).sum();
        let sum_yy: f64 = paired.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Find closest performance sample
    fn find_closest_perf(&self, timestamp: f64) -> Option<&PerformanceSample> {
        self.perf_samples
            .iter()
            .min_by(|a, b| {
                (a.timestamp - timestamp)
                    .abs()
                    .partial_cmp(&(b.timestamp - timestamp).abs())
                    .expect("values should be comparable")
            })
            .filter(|p| (p.timestamp - timestamp).abs() < self.window_sec)
    }

    /// Detect interference source
    pub fn detect_interference(&self) -> Option<InterferenceResult> {
        let event_types = [
            EventType::Interrupt,
            EventType::DiskIo,
            EventType::Network,
            EventType::ContextSwitch,
            EventType::PageFault,
            EventType::ProcessCpu,
        ];

        let mut correlations: Vec<(EventType, f64)> = Vec::new();

        for event_type in event_types {
            if let Some(result) = self.correlate_events(event_type) {
                if result.pearson_r.abs() > 0.1 {
                    correlations.push((event_type, result.pearson_r));
                }
            }
        }

        if correlations.is_empty() {
            return None;
        }

        correlations.sort_by(|a, b| {
            b.1.abs().partial_cmp(&a.1.abs()).expect("values should be comparable")
        });

        let (primary_source, correlation) = correlations[0];
        let secondary_sources: Vec<_> = correlations.into_iter().skip(1).collect();

        let confidence = if correlation.abs() > 0.5 && self.perf_samples.len() > 20 {
            0.9
        } else if correlation.abs() > 0.3 {
            0.7
        } else {
            0.5
        };

        Some(InterferenceResult { primary_source, correlation, confidence, secondary_sources })
    }

    /// Recommend isolation strategy
    pub fn recommend_isolation(&self) -> IsolationRecommendation {
        if let Some(interference) = self.detect_interference() {
            let (action, expected_improvement) = match interference.primary_source {
                EventType::Interrupt => (IsolationAction::CpuPin, 20.0),
                EventType::ContextSwitch => (IsolationAction::RealtimePriority, 15.0),
                EventType::ProcessCpu => (IsolationAction::CpuPin, 25.0),
                EventType::DiskIo => (IsolationAction::MemoryIsolation, 10.0),
                EventType::Network => (IsolationAction::NetworkIsolation, 15.0),
                EventType::PageFault => (IsolationAction::MemoryIsolation, 20.0),
            };

            let reason = format!(
                "{} correlation ({:.2}) with {} detected",
                interference.category().name(),
                interference.correlation,
                interference.primary_source.name()
            );

            IsolationRecommendation {
                action,
                reason,
                expected_improvement: expected_improvement * interference.correlation.abs(),
                confidence: interference.confidence,
            }
        } else {
            IsolationRecommendation {
                action: IsolationAction::None,
                reason: "No significant interference detected".to_string(),
                expected_improvement: 0.0,
                confidence: 0.9,
            }
        }
    }

    /// Capture current system state
    pub fn capture_system_state(&mut self, timestamp: f64) -> SystemSnapshot {
        // In real implementation, would read from /proc
        // For now, return empty snapshot
        let snapshot = SystemSnapshot::new(timestamp);
        self.add_snapshot(snapshot.clone());
        snapshot
    }

    /// Get CV spikes
    pub fn get_spikes(&self) -> Vec<&PerformanceSample> {
        self.perf_samples.iter().filter(|s| s.is_spike(self.spike_threshold)).collect()
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.perf_samples.clear();
        self.event_samples.clear();
        self.snapshots.clear();
    }
}
