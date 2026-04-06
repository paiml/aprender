//! Types, enums, constants, and scorecard for the Ironman falsification suite.

use std::collections::HashMap;

/// Ironman quality gate result
#[derive(Debug, Clone, PartialEq)]
pub enum GateResult {
    /// Gate passed with optional details
    Pass(String),
    /// Gate failed with reason
    Fail(String),
    /// Gate skipped (e.g., tool not installed)
    Skip(String),
    /// Gate result pending (async check)
    Pending,
}

impl GateResult {
    /// Check if the gate passed
    pub fn passed(&self) -> bool {
        matches!(self, GateResult::Pass(_))
    }

    /// Check if the gate failed
    pub fn failed(&self) -> bool {
        matches!(self, GateResult::Fail(_))
    }

    /// Get score contribution (0 if failed/skipped, weight if passed)
    pub fn score(&self, weight: u32) -> u32 {
        match self {
            GateResult::Pass(_) => weight,
            GateResult::Fail(_) | GateResult::Skip(_) | GateResult::Pending => 0,
        }
    }
}

/// Quality gate definition
#[derive(Debug, Clone)]
pub struct QualityGate {
    /// Falsification ID (F901-F920)
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// Tool used for validation
    pub tool: &'static str,
    /// Target threshold
    pub target: &'static str,
    /// Maximum points for this gate
    pub weight: u32,
    /// Category for grouping
    pub category: GateCategory,
}

/// Gate category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateCategory {
    /// Mutation and fuzzing resilience
    Resilience,
    /// Memory safety (Miri, sanitizers)
    Safety,
    /// Code quality (complexity, docs, deps)
    Quality,
    /// Performance (size, startup, latency)
    Performance,
    /// Usability (a11y, i18n)
    Usability,
}

impl GateCategory {
    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            GateCategory::Resilience => "Resilience",
            GateCategory::Safety => "Safety",
            GateCategory::Quality => "Quality",
            GateCategory::Performance => "Performance",
            GateCategory::Usability => "Usability",
        }
    }
}

/// All Ironman quality gates per section 34
pub const IRONMAN_GATES: &[QualityGate] = &[
    QualityGate {
        id: "F901",
        name: "Mutation Resilience",
        tool: "cargo mutants",
        target: ">90%",
        weight: 15,
        category: GateCategory::Resilience,
    },
    QualityGate {
        id: "F902",
        name: "Fuzzing Coverage",
        tool: "cargo fuzz",
        target: ">90%",
        weight: 10,
        category: GateCategory::Resilience,
    },
    QualityGate {
        id: "F903",
        name: "Miri UB-Free",
        tool: "cargo miri test",
        target: "0 UB",
        weight: 15,
        category: GateCategory::Safety,
    },
    QualityGate {
        id: "F904",
        name: "Loom Concurrency",
        tool: "loom",
        target: "0 races",
        weight: 5,
        category: GateCategory::Safety,
    },
    QualityGate {
        id: "F905",
        name: "ThreadSanitizer",
        tool: "-Zsanitizer=thread",
        target: "0 races",
        weight: 5,
        category: GateCategory::Safety,
    },
    QualityGate {
        id: "F906",
        name: "AddressSanitizer",
        tool: "-Zsanitizer=address",
        target: "0 errors",
        weight: 5,
        category: GateCategory::Safety,
    },
    QualityGate {
        id: "F907",
        name: "LeakSanitizer",
        tool: "-Zsanitizer=leak",
        target: "0 leaks",
        weight: 5,
        category: GateCategory::Safety,
    },
    QualityGate {
        id: "F908",
        name: "Panic Freedom",
        tool: "fuzz inputs",
        target: "0 panics",
        weight: 5,
        category: GateCategory::Resilience,
    },
    QualityGate {
        id: "F909",
        name: "Unsafe Audit",
        tool: "cargo geiger",
        target: "0 forbid",
        weight: 10,
        category: GateCategory::Quality,
    },
    QualityGate {
        id: "F910",
        name: "Dependency Audit",
        tool: "cargo audit",
        target: "0 vulns",
        weight: 10,
        category: GateCategory::Quality,
    },
    QualityGate {
        id: "F911",
        name: "Dead Code",
        tool: "cargo udeps",
        target: "0 unused",
        weight: 5,
        category: GateCategory::Quality,
    },
    QualityGate {
        id: "F912",
        name: "Cognitive Complexity",
        tool: "clippy",
        target: "<15/fn",
        weight: 10,
        category: GateCategory::Quality,
    },
    QualityGate {
        id: "F913",
        name: "Documentation",
        tool: "rustdoc",
        target: "100% pub",
        weight: 5,
        category: GateCategory::Quality,
    },
    QualityGate {
        id: "F914",
        name: "License Compliance",
        tool: "cargo deny",
        target: "approved",
        weight: 5,
        category: GateCategory::Quality,
    },
    QualityGate {
        id: "F915",
        name: "Binary Size",
        tool: "strip",
        target: "<8MB",
        weight: 5,
        category: GateCategory::Performance,
    },
    QualityGate {
        id: "F916",
        name: "Startup Time",
        tool: "cold start",
        target: "<20ms",
        weight: 10,
        category: GateCategory::Performance,
    },
    QualityGate {
        id: "F917",
        name: "Frame Latency",
        tool: "P99 render",
        target: "<8ms",
        weight: 10,
        category: GateCategory::Performance,
    },
    QualityGate {
        id: "F918",
        name: "Battery Impact",
        tool: "powertop",
        target: "<1W idle",
        weight: 5,
        category: GateCategory::Performance,
    },
    QualityGate {
        id: "F919",
        name: "Accessibility",
        tool: "screen reader",
        target: "readable",
        weight: 5,
        category: GateCategory::Usability,
    },
    QualityGate {
        id: "F920",
        name: "Internationalization",
        tool: "non-ASCII",
        target: "no crash",
        weight: 5,
        category: GateCategory::Usability,
    },
];

/// Ironman validation scorecard
#[derive(Debug, Clone)]
pub struct IronmanScorecard {
    /// Results for each gate
    pub results: HashMap<&'static str, GateResult>,
    /// Total score achieved
    pub total_score: u32,
    /// Maximum possible score
    pub max_score: u32,
    /// Pass threshold (default 90%)
    pub pass_threshold: f64,
    /// Timestamp of validation
    pub timestamp: std::time::SystemTime,
}

impl IronmanScorecard {
    /// Create a new empty scorecard
    pub fn new() -> Self {
        let max_score = IRONMAN_GATES.iter().map(|g| g.weight).sum();
        Self {
            results: HashMap::new(),
            total_score: 0,
            max_score,
            pass_threshold: 0.90,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Record a gate result
    pub fn record(&mut self, gate_id: &'static str, result: GateResult) {
        if let Some(gate) = IRONMAN_GATES.iter().find(|g| g.id == gate_id) {
            let score = result.score(gate.weight);
            self.total_score += score;
            self.results.insert(gate_id, result);
        }
    }

    /// Get percentage score
    pub fn percentage(&self) -> f64 {
        if self.max_score == 0 {
            return 0.0;
        }
        (self.total_score as f64 / self.max_score as f64) * 100.0
    }

    /// Check if overall validation passed
    pub fn passed(&self) -> bool {
        self.percentage() >= self.pass_threshold * 100.0
    }

    /// Get score by category
    pub fn category_score(&self, category: GateCategory) -> (u32, u32) {
        let mut achieved = 0u32;
        let mut max = 0u32;

        for gate in IRONMAN_GATES.iter().filter(|g| g.category == category) {
            max += gate.weight;
            if let Some(result) = self.results.get(gate.id) {
                achieved += result.score(gate.weight);
            }
        }

        (achieved, max)
    }

    /// Get failed gates
    pub fn failed_gates(&self) -> Vec<&QualityGate> {
        IRONMAN_GATES
            .iter()
            .filter(|g| self.results.get(g.id).map_or(false, |r| r.failed()))
            .collect()
    }

    /// Get skipped gates
    pub fn skipped_gates(&self) -> Vec<&QualityGate> {
        IRONMAN_GATES
            .iter()
            .filter(|g| self.results.get(g.id).map_or(true, |r| matches!(r, GateResult::Skip(_))))
            .collect()
    }
}

impl Default for IronmanScorecard {
    fn default() -> Self {
        Self::new()
    }
}
