use crate::http_client::*;
// =========================================================================

/// Performance trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Regressing,
}

/// IMP-153a: Performance history entry
#[derive(Debug, Clone)]
pub struct PerformanceEntry {
    pub timestamp: String,
    pub throughput_tps: f64,
    pub milestone: String,
    pub gap_vs_target_percent: f64,
}

/// IMP-153a: Performance history tracking
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    pub entries: Vec<PerformanceEntry>,
    pub target_tps: f64,
    pub trend: PerformanceTrend,
    pub avg_improvement_per_entry: f64,
}

impl PerformanceHistory {
    pub fn new(target_tps: f64) -> Self {
        Self {
            entries: Vec::new(),
            target_tps,
            trend: PerformanceTrend::Stable,
            avg_improvement_per_entry: 0.0,
        }
    }

    pub fn add_entry(&mut self, throughput_tps: f64, milestone: &str) {
        let gap = if self.target_tps > 0.0 {
            ((throughput_tps - self.target_tps) / self.target_tps) * 100.0
        } else {
            0.0
        };
        self.entries.push(PerformanceEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            throughput_tps,
            milestone: milestone.to_string(),
            gap_vs_target_percent: gap,
        });
        self.recalculate_trend();
    }

    fn recalculate_trend(&mut self) {
        if self.entries.len() < 2 {
            self.trend = PerformanceTrend::Stable;
            self.avg_improvement_per_entry = 0.0;
            return;
        }

        // Calculate improvements between consecutive entries
        let mut improvements = Vec::new();
        for i in 1..self.entries.len() {
            let prev = self.entries[i - 1].throughput_tps;
            let curr = self.entries[i].throughput_tps;
            if prev > 0.0 {
                improvements.push((curr - prev) / prev * 100.0);
            }
        }

        if improvements.is_empty() {
            self.trend = PerformanceTrend::Stable;
            self.avg_improvement_per_entry = 0.0;
            return;
        }

        let avg: f64 = improvements.iter().sum::<f64>() / improvements.len() as f64;
        self.avg_improvement_per_entry = avg;

        // Determine trend: >2% avg improvement = improving, <-2% = regressing
        self.trend = if avg > 2.0 {
            PerformanceTrend::Improving
        } else if avg < -2.0 {
            PerformanceTrend::Regressing
        } else {
            PerformanceTrend::Stable
        };
    }

    pub fn latest_throughput(&self) -> Option<f64> {
        self.entries.last().map(|e| e.throughput_tps)
    }

    pub fn entries_count(&self) -> usize {
        self.entries.len()
    }
}

/// IMP-153a: Test performance history tracking
#[test]
fn test_imp_153a_performance_history() {
    let mut history = PerformanceHistory::new(256.0); // Target: llama.cpp baseline

    // Add progression entries
    history.add_entry(80.0, "Baseline");
    history.add_entry(100.0, "P1-25%");
    history.add_entry(120.0, "P1");
    history.add_entry(160.0, "P1-P2");
    history.add_entry(200.0, "P2");

    assert_eq!(
        history.entries_count(),
        5,
        "IMP-153a: Should have 5 entries"
    );
    assert_eq!(
        history.latest_throughput(),
        Some(200.0),
        "IMP-153a: Latest should be 200"
    );
    assert_eq!(
        history.trend,
        PerformanceTrend::Improving,
        "IMP-153a: Trend should be improving"
    );
    assert!(
        history.avg_improvement_per_entry > 20.0,
        "IMP-153a: Should show >20% avg improvement per entry"
    );

    println!("\nIMP-153a: Performance History:");
    for (i, entry) in history.entries.iter().enumerate() {
        println!(
            "  Entry {}: {} tok/s ({}) gap={:+.1}%",
            i + 1,
            entry.throughput_tps,
            entry.milestone,
            entry.gap_vs_target_percent
        );
    }
    println!("  Trend: {:?}", history.trend);
    println!(
        "  Avg improvement: {:.1}%/entry",
        history.avg_improvement_per_entry
    );
}

/// IMP-153b: Test trend detection
#[test]
fn test_imp_153b_trend_detection() {
    // Scenario 1: Improving trend
    let mut improving = PerformanceHistory::new(256.0);
    improving.add_entry(80.0, "Start");
    improving.add_entry(100.0, "Mid");
    improving.add_entry(130.0, "End");
    assert_eq!(
        improving.trend,
        PerformanceTrend::Improving,
        "IMP-153b: 80→100→130 should be improving"
    );

    // Scenario 2: Regressing trend
    let mut regressing = PerformanceHistory::new(256.0);
    regressing.add_entry(120.0, "Start");
    regressing.add_entry(110.0, "Mid");
    regressing.add_entry(95.0, "End");
    assert_eq!(
        regressing.trend,
        PerformanceTrend::Regressing,
        "IMP-153b: 120→110→95 should be regressing"
    );

    // Scenario 3: Stable trend (within ±2%)
    let mut stable = PerformanceHistory::new(256.0);
    stable.add_entry(100.0, "Start");
    stable.add_entry(101.0, "Mid");
    stable.add_entry(100.5, "End");
    assert_eq!(
        stable.trend,
        PerformanceTrend::Stable,
        "IMP-153b: 100→101→100.5 should be stable"
    );

    // Scenario 4: Single entry = stable
    let mut single = PerformanceHistory::new(256.0);
    single.add_entry(100.0, "Only");
    assert_eq!(
        single.trend,
        PerformanceTrend::Stable,
        "IMP-153b: Single entry should be stable"
    );

    println!("\nIMP-153b: Trend Detection:");
    println!(
        "  Improving: 80→100→130, avg={:.1}%",
        improving.avg_improvement_per_entry
    );
    println!(
        "  Regressing: 120→110→95, avg={:.1}%",
        regressing.avg_improvement_per_entry
    );
    println!(
        "  Stable: 100→101→100.5, avg={:.1}%",
        stable.avg_improvement_per_entry
    );
}

/// IMP-153c: Milestone progress summary
#[derive(Debug, Clone)]
pub struct MilestoneProgress {
    pub current_tps: f64,
    pub p1_target: f64,
    pub p2_target: f64,
    pub parity_target: f64,
    pub p1_achieved: bool,
    pub p2_achieved: bool,
    pub parity_achieved: bool,
    pub next_milestone: String,
    pub gap_to_next: f64,
}

impl MilestoneProgress {
    pub fn new(current_tps: f64) -> Self {
        let p1_target: f64 = 120.0; // Per spec: 1.5x baseline
        let p2_target: f64 = 200.0; // Per spec: 2.5x baseline
        let parity_target: f64 = 230.0; // Per spec: within 10% of 256

        let p1_achieved = current_tps >= p1_target;
        let p2_achieved = current_tps >= p2_target;
        let parity_achieved = current_tps >= parity_target;

        let (next_milestone, gap_to_next) = if !p1_achieved {
            ("P1".to_string(), p1_target - current_tps)
        } else if !p2_achieved {
            ("P2".to_string(), p2_target - current_tps)
        } else if !parity_achieved {
            ("Parity".to_string(), parity_target - current_tps)
        } else {
            ("Complete".to_string(), 0.0)
        };

        Self {
            current_tps,
            p1_target,
            p2_target,
            parity_target,
            p1_achieved,
            p2_achieved,
            parity_achieved,
            next_milestone,
            gap_to_next,
        }
    }
}

/// IMP-153c: Test milestone progress tracking
#[test]
fn test_imp_153c_milestone_progress() {
    // Scenario 1: Before P1
    let before_p1 = MilestoneProgress::new(80.0);
    assert!(!before_p1.p1_achieved, "IMP-153c: 80 should not achieve P1");
    assert_eq!(before_p1.next_milestone, "P1");
    assert!(
        (before_p1.gap_to_next - 40.0).abs() < 0.1,
        "IMP-153c: 40 tok/s to P1"
    );

    // Scenario 2: Between P1 and P2
    let between = MilestoneProgress::new(150.0);
    assert!(between.p1_achieved, "IMP-153c: 150 should achieve P1");
    assert!(!between.p2_achieved, "IMP-153c: 150 should not achieve P2");
    assert_eq!(between.next_milestone, "P2");
    assert!(
        (between.gap_to_next - 50.0).abs() < 0.1,
        "IMP-153c: 50 tok/s to P2"
    );

    // Scenario 3: Between P2 and Parity
    let near_parity = MilestoneProgress::new(210.0);
    assert!(near_parity.p2_achieved, "IMP-153c: 210 should achieve P2");
    assert!(
        !near_parity.parity_achieved,
        "IMP-153c: 210 should not achieve Parity"
    );
    assert_eq!(near_parity.next_milestone, "Parity");

    // Scenario 4: Parity achieved
    let at_parity = MilestoneProgress::new(240.0);
    assert!(
        at_parity.parity_achieved,
        "IMP-153c: 240 should achieve Parity"
    );
    assert_eq!(at_parity.next_milestone, "Complete");

    println!("\nIMP-153c: Milestone Progress:");
    println!(
        "  80 tok/s: P1={} P2={} Parity={} Next={} Gap={:.0}",
        before_p1.p1_achieved,
        before_p1.p2_achieved,
        before_p1.parity_achieved,
        before_p1.next_milestone,
        before_p1.gap_to_next
    );
    println!(
        "  150 tok/s: P1={} P2={} Parity={} Next={} Gap={:.0}",
        between.p1_achieved,
        between.p2_achieved,
        between.parity_achieved,
        between.next_milestone,
        between.gap_to_next
    );
    println!(
        "  210 tok/s: P1={} P2={} Parity={} Next={} Gap={:.0}",
        near_parity.p1_achieved,
        near_parity.p2_achieved,
        near_parity.parity_achieved,
        near_parity.next_milestone,
        near_parity.gap_to_next
    );
    println!(
        "  240 tok/s: P1={} P2={} Parity={} Next={}",
        at_parity.p1_achieved,
        at_parity.p2_achieved,
        at_parity.parity_achieved,
        at_parity.next_milestone
    );
}

/// IMP-153d: Gap trend tracking
#[derive(Debug, Clone)]
pub struct GapTrend {
    pub initial_gap_percent: f64,
    pub current_gap_percent: f64,
    pub gap_closed_percent: f64,
    pub estimated_entries_to_parity: usize,
}

impl GapTrend {
    pub fn new(
        initial_tps: f64,
        current_tps: f64,
        target_tps: f64,
        avg_improvement_percent: f64,
    ) -> Self {
        let initial_gap = if target_tps > 0.0 {
            ((target_tps - initial_tps) / target_tps) * 100.0
        } else {
            0.0
        };
        let current_gap = if target_tps > 0.0 {
            ((target_tps - current_tps) / target_tps) * 100.0
        } else {
            0.0
        };
        let gap_closed = initial_gap - current_gap;

        // Estimate entries to reach parity (within 10% of target)
        let parity_gap: f64 = 10.0;
        let remaining_gap = current_gap - parity_gap;
        let estimated_entries = if remaining_gap <= 0.0 || avg_improvement_percent <= 0.0 {
            0
        } else {
            // Rough estimate: remaining_gap / avg_improvement_percent
            // This is simplified; real calculation would consider compound growth
            ((remaining_gap / avg_improvement_percent) * 1.5).ceil() as usize
        };

        Self {
            initial_gap_percent: initial_gap,
            current_gap_percent: current_gap,
            gap_closed_percent: gap_closed,
            estimated_entries_to_parity: estimated_entries,
        }
    }
}

/// IMP-153d: Test gap trend tracking
#[test]
fn test_imp_153d_gap_trend() {
    // Scenario: Started at 80 tok/s, now at 120 tok/s, targeting 256 tok/s
    // With 25% avg improvement per entry
    let trend = GapTrend::new(80.0, 120.0, 256.0, 25.0);

    // Initial gap: (256-80)/256 = 68.75%
    // Current gap: (256-120)/256 = 53.125%
    // Gap closed: 68.75 - 53.125 = 15.625%
    assert!(
        (trend.initial_gap_percent - 68.75).abs() < 0.1,
        "IMP-153d: Initial gap should be ~68.75%"
    );
    assert!(
        (trend.current_gap_percent - 53.125).abs() < 0.1,
        "IMP-153d: Current gap should be ~53.125%"
    );
    assert!(
        (trend.gap_closed_percent - 15.625).abs() < 0.1,
        "IMP-153d: Gap closed should be ~15.625%"
    );
    assert!(
        trend.estimated_entries_to_parity > 0,
        "IMP-153d: Should estimate entries needed"
    );

    // Already at parity
    let at_parity = GapTrend::new(80.0, 240.0, 256.0, 25.0);
    assert_eq!(
        at_parity.estimated_entries_to_parity, 0,
        "IMP-153d: At parity should be 0 entries"
    );

    println!("\nIMP-153d: Gap Trend:");
    println!(
        "  Initial: {:.1}% gap (80 vs 256)",
        trend.initial_gap_percent
    );
    println!(
        "  Current: {:.1}% gap (120 vs 256)",
        trend.current_gap_percent
    );
    println!("  Closed: {:.1}%", trend.gap_closed_percent);
    println!(
        "  Est. entries to parity: {}",
        trend.estimated_entries_to_parity
    );
    println!(
        "  At parity (240 vs 256): {} entries",
        at_parity.estimated_entries_to_parity
    );
}

// =========================================================================
// IMP-154: Automated Performance Gate Validation (EXTREME TDD)
// Per spec §10.1: CI/CD integration for performance regression prevention
// =========================================================================

/// Gate status for performance checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
}

include!("higher_performance_gate.rs");
include!("imp_154d.rs");
include!("imp_156b.rs");
include!("imp_157d.rs");
