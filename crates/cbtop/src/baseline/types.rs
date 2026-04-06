//! Baseline types: server baselines, GPU classes, grades, and health indicators.

use std::fmt;

/// Industry server baseline data from Satna (2026) benchmarks.
///
/// Citation: [21] Satna, R. (2026). "LLM Inference Benchmarking Framework."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServerBaseline {
    /// Server name (vLLM, TGI, Triton)
    pub name: &'static str,
    /// Peak tokens per second
    pub peak_tok_per_sec: u32,
    /// P95 latency in milliseconds
    pub p95_latency_ms: u32,
    /// SM utilization percentage
    pub sm_utilization: u8,
    /// Memory overhead percentage
    pub memory_overhead: u8,
    /// Reference GPU
    pub gpu: &'static str,
}

/// Industry baselines from Satna (2026) on A10 GPU.
pub const VLLM_BASELINE: ServerBaseline = ServerBaseline {
    name: "vLLM",
    peak_tok_per_sec: 412,
    p95_latency_ms: 1715,
    sm_utilization: 99,
    memory_overhead: 42,
    gpu: "A10",
};

pub const TGI_BASELINE: ServerBaseline = ServerBaseline {
    name: "TGI",
    peak_tok_per_sec: 408,
    p95_latency_ms: 1704,
    sm_utilization: 98,
    memory_overhead: 44,
    gpu: "A10",
};

pub const TRITON_BASELINE: ServerBaseline = ServerBaseline {
    name: "Triton",
    peak_tok_per_sec: 385,
    p95_latency_ms: 2007,
    sm_utilization: 97,
    memory_overhead: 45,
    gpu: "A10",
};

/// All industry baselines.
pub const INDUSTRY_BASELINES: [ServerBaseline; 3] = [VLLM_BASELINE, TGI_BASELINE, TRITON_BASELINE];

/// GPU class with expected throughput ranges.
///
/// From cbtop spec §21.7 "Expected Throughput by GPU Class".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuClass {
    /// NVIDIA A10 (24GB) - Data center inference GPU
    A10,
    /// NVIDIA A100 (40GB or 80GB) - Data center training/inference GPU
    A100,
    /// NVIDIA H100 (80GB) - Hopper architecture flagship
    H100,
    /// NVIDIA RTX 4090 (24GB) - Consumer flagship
    Rtx4090,
    /// NVIDIA RTX 3090 (24GB) - Previous gen consumer flagship
    Rtx3090,
    /// Unknown GPU class
    Unknown,
}

/// Per-variant specification for a GPU class.
struct GpuSpec {
    /// Display label (e.g. "A10 (24GB)")
    label: &'static str,
    /// Expected throughput range (min, max) in tok/s
    throughput: (u32, u32),
    /// VRAM size in GB
    vram_gb: u32,
}

/// Single source of truth for all GPU class specifications.
///
/// Adding a new GPU class only requires adding one entry here (plus the enum variant
/// and a `from_name` match arm), eliminating the previous 4 separate match blocks.
const fn gpu_spec(class: &GpuClass) -> GpuSpec {
    match class {
        GpuClass::A10 => GpuSpec { label: "A10 (24GB)", throughput: (350, 450), vram_gb: 24 },
        GpuClass::A100 => GpuSpec {
            label: "A100 (40/80GB)",
            throughput: (800, 1200),
            vram_gb: 80, // Using 80GB variant
        },
        GpuClass::H100 => GpuSpec { label: "H100 (80GB)", throughput: (1800, 2400), vram_gb: 80 },
        GpuClass::Rtx4090 => {
            GpuSpec { label: "RTX 4090 (24GB)", throughput: (300, 400), vram_gb: 24 }
        }
        GpuClass::Rtx3090 => {
            GpuSpec { label: "RTX 3090 (24GB)", throughput: (200, 300), vram_gb: 24 }
        }
        GpuClass::Unknown => GpuSpec {
            label: "Unknown GPU",
            throughput: (100, 500), // Conservative estimate
            vram_gb: 8,
        },
    }
}

impl GpuClass {
    /// Expected throughput range (min, max) in tok/s.
    ///
    /// From cbtop spec §21.7.
    pub fn expected_throughput(&self) -> (u32, u32) {
        gpu_spec(self).throughput
    }

    /// VRAM size in GB.
    pub fn vram_gb(&self) -> u32 {
        gpu_spec(self).vram_gb
    }

    /// Detect GPU class from GPU name string.
    ///
    /// Parses common GPU name formats from nvidia-smi, NVML, etc.
    pub fn from_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        if name_lower.contains("h100") {
            GpuClass::H100
        } else if name_lower.contains("a100") {
            GpuClass::A100
        } else if name_lower.contains("a10") && !name_lower.contains("a100") {
            GpuClass::A10
        } else if name_lower.contains("4090") {
            GpuClass::Rtx4090
        } else if name_lower.contains("3090") {
            GpuClass::Rtx3090
        } else {
            GpuClass::Unknown
        }
    }
}

impl fmt::Display for GpuClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", gpu_spec(self).label)
    }
}

/// Throughput grade (A/B/C/D/F) based on baseline comparison.
///
/// From cbtop spec F983: "Throughput grade calculated".
/// Ordering: F < D < C < B < A (A is best).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThroughputGrade {
    /// < 40% of vLLM baseline (worst)
    F,
    /// >= 40% of vLLM baseline
    D,
    /// >= 60% of vLLM baseline
    C,
    /// >= 80% of vLLM baseline
    B,
    /// >= 100% of vLLM baseline (best)
    A,
}

/// Per-grade specification for throughput grading.
struct GradeSpec {
    /// Minimum percentage threshold to earn this grade
    threshold: f64,
    /// Short letter label
    label: &'static str,
    /// Human-readable description
    description: &'static str,
}

/// Ordered from highest to lowest grade for `from_percentage` lookup.
const GRADE_SPECS: [(ThroughputGrade, GradeSpec); 5] = [
    (
        ThroughputGrade::A,
        GradeSpec {
            threshold: 100.0,
            label: "A",
            description: "Excellent - meets or exceeds baseline",
        },
    ),
    (
        ThroughputGrade::B,
        GradeSpec { threshold: 80.0, label: "B", description: "Good - 80%+ of baseline" },
    ),
    (
        ThroughputGrade::C,
        GradeSpec { threshold: 60.0, label: "C", description: "Fair - 60%+ of baseline" },
    ),
    (
        ThroughputGrade::D,
        GradeSpec { threshold: 40.0, label: "D", description: "Poor - 40%+ of baseline" },
    ),
    (
        ThroughputGrade::F,
        GradeSpec { threshold: 0.0, label: "F", description: "Failing - below 40% of baseline" },
    ),
];

/// Look up the spec for a given grade variant.
fn grade_spec(grade: &ThroughputGrade) -> &'static GradeSpec {
    &GRADE_SPECS.iter().find(|(g, _)| g == grade).expect("all variants present in GRADE_SPECS").1
}

impl ThroughputGrade {
    /// Calculate grade from actual throughput vs baseline.
    pub fn from_percentage(percentage: f64) -> Self {
        GRADE_SPECS
            .iter()
            .find(|(_, spec)| percentage >= spec.threshold)
            .map(|(grade, _)| *grade)
            .unwrap_or(ThroughputGrade::F)
    }

    /// Get threshold percentage for this grade.
    pub fn threshold(&self) -> f64 {
        grade_spec(self).threshold
    }
}

impl fmt::Display for ThroughputGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spec = grade_spec(self);
        write!(f, "{} ({})", spec.label, spec.description)
    }
}

/// SM utilization health indicator.
///
/// From cbtop spec §21.7: SM utilization thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmHealth {
    /// SM utilization > 95% - at risk of saturation
    Saturated,
    /// SM utilization 80-95% - optimal range
    Optimal,
    /// SM utilization 50-80% - room for improvement
    Moderate,
    /// SM utilization < 50% - critical underutilization
    Critical,
}

/// Per-variant specification for SM health status.
struct SmHealthSpec {
    /// Minimum SM utilization to qualify (exclusive for Saturated, inclusive otherwise)
    min_util: u8,
    /// Whether the threshold comparison is strict greater-than
    exclusive: bool,
    /// Display label (e.g. "OPTIMAL (80-95%)")
    label: &'static str,
}

/// Ordered from highest to lowest threshold for `from_utilization` lookup.
const SM_HEALTH_SPECS: [(SmHealth, SmHealthSpec); 4] = [
    (
        SmHealth::Saturated,
        SmHealthSpec { min_util: 95, exclusive: true, label: "SATURATED (>95%)" },
    ),
    (SmHealth::Optimal, SmHealthSpec { min_util: 80, exclusive: false, label: "OPTIMAL (80-95%)" }),
    (
        SmHealth::Moderate,
        SmHealthSpec { min_util: 50, exclusive: false, label: "MODERATE (50-80%)" },
    ),
    (SmHealth::Critical, SmHealthSpec { min_util: 0, exclusive: false, label: "CRITICAL (<50%)" }),
];

/// Look up the spec for a given SM health variant.
fn sm_health_spec(health: &SmHealth) -> &'static SmHealthSpec {
    &SM_HEALTH_SPECS
        .iter()
        .find(|(h, _)| h == health)
        .expect("all variants present in SM_HEALTH_SPECS")
        .1
}

impl SmHealth {
    /// Calculate SM health from utilization percentage.
    pub fn from_utilization(sm_util: u8) -> Self {
        SM_HEALTH_SPECS
            .iter()
            .find(
                |(_, spec)| {
                    if spec.exclusive {
                        sm_util > spec.min_util
                    } else {
                        sm_util >= spec.min_util
                    }
                },
            )
            .map(|(health, _)| *health)
            .unwrap_or(SmHealth::Critical)
    }

    /// Is this health status acceptable for production?
    pub fn is_acceptable(&self) -> bool {
        matches!(self, SmHealth::Optimal | SmHealth::Saturated)
    }
}

impl fmt::Display for SmHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", sm_health_spec(self).label)
    }
}

/// Comparison with a single baseline server.
#[derive(Debug, Clone)]
pub struct SingleComparison {
    /// Baseline server
    pub baseline: ServerBaseline,
    /// Percentage achieved (actual / baseline * 100)
    pub percentage: f64,
    /// Delta in tok/s (actual - baseline)
    pub delta_tok_per_sec: i32,
}
