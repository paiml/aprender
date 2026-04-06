
/// IMP-154a: Individual performance gate
#[derive(Debug, Clone)]
pub struct PerformanceGate {
    pub name: String,
    pub measured_value: f64,
    pub pass_threshold: f64,
    pub warn_threshold: f64,
    pub unit: String,
    pub status: GateStatus,
    pub message: String,
}

impl PerformanceGate {
    /// Create a gate where higher values are better (e.g., throughput)
    pub fn higher_is_better(
        name: &str,
        measured: f64,
        pass_threshold: f64,
        warn_threshold: f64,
        unit: &str,
    ) -> Self {
        let status = if measured >= pass_threshold {
            GateStatus::Pass
        } else if measured >= warn_threshold {
            GateStatus::Warn
        } else {
            GateStatus::Fail
        };
        let message = match status {
            GateStatus::Pass => format!(
                "{:.1}{} >= {:.1}{} (PASS)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Warn => format!(
                "{:.1}{} < {:.1}{} (WARN)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Fail => format!(
                "{:.1}{} < {:.1}{} (FAIL)",
                measured, unit, warn_threshold, unit
            ),
        };
        Self {
            name: name.to_string(),
            measured_value: measured,
            pass_threshold,
            warn_threshold,
            unit: unit.to_string(),
            status,
            message,
        }
    }

    /// Create a gate where lower values are better (e.g., latency)
    pub fn lower_is_better(
        name: &str,
        measured: f64,
        pass_threshold: f64,
        warn_threshold: f64,
        unit: &str,
    ) -> Self {
        let status = if measured <= pass_threshold {
            GateStatus::Pass
        } else if measured <= warn_threshold {
            GateStatus::Warn
        } else {
            GateStatus::Fail
        };
        let message = match status {
            GateStatus::Pass => format!(
                "{:.1}{} <= {:.1}{} (PASS)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Warn => format!(
                "{:.1}{} > {:.1}{} (WARN)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Fail => format!(
                "{:.1}{} > {:.1}{} (FAIL)",
                measured, unit, warn_threshold, unit
            ),
        };
        Self {
            name: name.to_string(),
            measured_value: measured,
            pass_threshold,
            warn_threshold,
            unit: unit.to_string(),
            status,
            message,
        }
    }
}

/// IMP-154a: Test individual performance gate
#[test]
fn test_imp_154a_performance_gate() {
    // Throughput gate: Pass if >= 120 tok/s, Warn if >= 100, Fail otherwise
    let pass_gate = PerformanceGate::higher_is_better("Throughput", 130.0, 120.0, 100.0, " tok/s");
    assert_eq!(
        pass_gate.status,
        GateStatus::Pass,
        "IMP-154a: 130 should pass 120 threshold"
    );

    let warn_gate = PerformanceGate::higher_is_better("Throughput", 110.0, 120.0, 100.0, " tok/s");
    assert_eq!(
        warn_gate.status,
        GateStatus::Warn,
        "IMP-154a: 110 should warn (100-120)"
    );

    let fail_gate = PerformanceGate::higher_is_better("Throughput", 90.0, 120.0, 100.0, " tok/s");
    assert_eq!(
        fail_gate.status,
        GateStatus::Fail,
        "IMP-154a: 90 should fail (<100)"
    );

    // Latency gate: Pass if <= 50ms, Warn if <= 100ms, Fail otherwise
    let latency_pass = PerformanceGate::lower_is_better("P50 Latency", 40.0, 50.0, 100.0, "ms");
    assert_eq!(
        latency_pass.status,
        GateStatus::Pass,
        "IMP-154a: 40ms should pass"
    );

    let latency_warn = PerformanceGate::lower_is_better("P50 Latency", 70.0, 50.0, 100.0, "ms");
    assert_eq!(
        latency_warn.status,
        GateStatus::Warn,
        "IMP-154a: 70ms should warn"
    );

    let latency_fail = PerformanceGate::lower_is_better("P50 Latency", 150.0, 50.0, 100.0, "ms");
    assert_eq!(
        latency_fail.status,
        GateStatus::Fail,
        "IMP-154a: 150ms should fail"
    );

    println!("\nIMP-154a: Performance Gates:");
    println!("  {} - {}", pass_gate.name, pass_gate.message);
    println!("  {} - {}", warn_gate.name, warn_gate.message);
    println!("  {} - {}", fail_gate.name, fail_gate.message);
    println!("  {} - {}", latency_pass.name, latency_pass.message);
    println!("  {} - {}", latency_warn.name, latency_warn.message);
    println!("  {} - {}", latency_fail.name, latency_fail.message);
}

/// IMP-154b: Composite gate that aggregates multiple checks
#[derive(Debug, Clone)]
pub struct CompositeGate {
    pub gates: Vec<PerformanceGate>,
    pub overall_status: GateStatus,
    pub pass_count: usize,
    pub warn_count: usize,
    pub fail_count: usize,
}

impl CompositeGate {
    pub fn new(gates: Vec<PerformanceGate>) -> Self {
        let pass_count = gates
            .iter()
            .filter(|g| g.status == GateStatus::Pass)
            .count();
        let warn_count = gates
            .iter()
            .filter(|g| g.status == GateStatus::Warn)
            .count();
        let fail_count = gates
            .iter()
            .filter(|g| g.status == GateStatus::Fail)
            .count();

        // Overall: Fail if any fail, Warn if any warn, Pass otherwise
        let overall_status = if fail_count > 0 {
            GateStatus::Fail
        } else if warn_count > 0 {
            GateStatus::Warn
        } else {
            GateStatus::Pass
        };

        Self {
            gates,
            overall_status,
            pass_count,
            warn_count,
            fail_count,
        }
    }

    pub fn all_passed(&self) -> bool {
        self.overall_status == GateStatus::Pass
    }

    pub fn should_block_merge(&self) -> bool {
        self.overall_status == GateStatus::Fail
    }
}

/// IMP-154b: Test composite gate
#[test]
fn test_imp_154b_composite_gate() {
    // Scenario 1: All pass
    let all_pass = CompositeGate::new(vec![
        PerformanceGate::higher_is_better("Throughput", 130.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("Latency", 40.0, 50.0, 100.0, "ms"),
    ]);
    assert!(all_pass.all_passed(), "IMP-154b: All gates should pass");
    assert!(!all_pass.should_block_merge(), "IMP-154b: Should not block");
    assert_eq!(all_pass.pass_count, 2);

    // Scenario 2: One warn
    let one_warn = CompositeGate::new(vec![
        PerformanceGate::higher_is_better("Throughput", 110.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("Latency", 40.0, 50.0, 100.0, "ms"),
    ]);
    assert_eq!(
        one_warn.overall_status,
        GateStatus::Warn,
        "IMP-154b: Should be warn"
    );
    assert!(
        !one_warn.should_block_merge(),
        "IMP-154b: Warn should not block"
    );

    // Scenario 3: One fail
    let one_fail = CompositeGate::new(vec![
        PerformanceGate::higher_is_better("Throughput", 90.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("Latency", 40.0, 50.0, 100.0, "ms"),
    ]);
    assert_eq!(
        one_fail.overall_status,
        GateStatus::Fail,
        "IMP-154b: Should be fail"
    );
    assert!(one_fail.should_block_merge(), "IMP-154b: Fail should block");

    println!("\nIMP-154b: Composite Gates:");
    println!(
        "  All pass: {:?} (block={})",
        all_pass.overall_status,
        all_pass.should_block_merge()
    );
    println!(
        "  One warn: {:?} (block={})",
        one_warn.overall_status,
        one_warn.should_block_merge()
    );
    println!(
        "  One fail: {:?} (block={})",
        one_fail.overall_status,
        one_fail.should_block_merge()
    );
}

/// IMP-154c: Standard gate configuration for performance parity
pub struct ParityGateConfig {
    pub p1_throughput_pass: f64,
    pub p1_throughput_warn: f64,
    pub parity_throughput_pass: f64,
    pub parity_throughput_warn: f64,
    pub regression_threshold_percent: f64,
}

impl Default for ParityGateConfig {
    fn default() -> Self {
        Self {
            p1_throughput_pass: 120.0,     // P1 milestone: 1.5x baseline
            p1_throughput_warn: 100.0,     // 25% of P1 progress
            parity_throughput_pass: 230.0, // Within 10% of 256
            parity_throughput_warn: 200.0, // P2 milestone
            regression_threshold_percent: 5.0,
        }
    }
}

/// IMP-154c: Test parity gate configuration
#[test]
fn test_imp_154c_parity_gate_config() {
    let config = ParityGateConfig::default();

    // Create gates based on config
    let current_tps: f64 = 150.0;
    let baseline_tps: f64 = 145.0;

    // P1 gate
    let p1_gate = PerformanceGate::higher_is_better(
        "P1 Throughput",
        current_tps,
        config.p1_throughput_pass,
        config.p1_throughput_warn,
        " tok/s",
    );
    assert_eq!(
        p1_gate.status,
        GateStatus::Pass,
        "IMP-154c: 150 should pass P1"
    );

    // Parity gate (150 < 200 warn threshold = Fail)
    let parity_gate = PerformanceGate::higher_is_better(
        "Parity Throughput",
        current_tps,
        config.parity_throughput_pass,
        config.parity_throughput_warn,
        " tok/s",
    );
    assert_eq!(
        parity_gate.status,
        GateStatus::Fail,
        "IMP-154c: 150 < 200 should fail parity"
    );

    // Regression gate (higher is better = no regression)
    let regression_percent = ((current_tps - baseline_tps) / baseline_tps) * 100.0;
    let regression_gate = PerformanceGate::higher_is_better(
        "Regression Check",
        regression_percent,
        -config.regression_threshold_percent, // Pass if >= -5%
        -10.0,                                // Warn if >= -10%
        "%",
    );
    assert_eq!(
        regression_gate.status,
        GateStatus::Pass,
        "IMP-154c: +3.4% should pass regression"
    );

    let composite = CompositeGate::new(vec![p1_gate, parity_gate, regression_gate]);
    assert_eq!(
        composite.overall_status,
        GateStatus::Fail,
        "IMP-154c: Should fail (parity gate failed)"
    );

    println!("\nIMP-154c: Parity Gate Config:");
    println!(
        "  Current: {:.0} tok/s, Baseline: {:.0} tok/s",
        current_tps, baseline_tps
    );
    for gate in &composite.gates {
        println!("  {} - {}", gate.name, gate.message);
    }
    println!("  Overall: {:?}", composite.overall_status);
}

/// IMP-154d: Gate report for CI output
#[derive(Debug, Clone)]
pub struct GateReport {
    pub title: String,
    pub composite: CompositeGate,
    pub summary: String,
    pub exit_code: i32,
}

impl GateReport {
    pub fn new(title: &str, composite: CompositeGate) -> Self {
        let summary = format!(
            "{}: {} PASS, {} WARN, {} FAIL -> {:?}",
            title,
            composite.pass_count,
            composite.warn_count,
            composite.fail_count,
            composite.overall_status
        );
        let exit_code = match composite.overall_status {
            GateStatus::Pass => 0,
            GateStatus::Warn => 0, // Warn doesn't fail CI
            GateStatus::Fail => 1,
        };
        Self {
            title: title.to_string(),
            composite,
            summary,
            exit_code,
        }
    }

    pub fn format_for_ci(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("## {}\n\n", self.title));
        output.push_str("| Gate | Status | Details |\n");
        output.push_str("|------|--------|--------|\n");
        for gate in &self.composite.gates {
            let status_emoji = match gate.status {
                GateStatus::Pass => "✅",
                GateStatus::Warn => "⚠️",
                GateStatus::Fail => "❌",
            };
            output.push_str(&format!(
                "| {} | {} | {} |\n",
                gate.name, status_emoji, gate.message
            ));
        }
        output.push_str(&format!("\n**Result**: {}\n", self.summary));
        output
    }
}
