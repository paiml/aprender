//! Efficiency & Cost Tracking Module (ENT-008 through ENT-012)
//!
//! Provides compute device abstraction, energy/cost metrics, model paradigm
//! classification, platform efficiency tracking, and cost-performance benchmarking.
//!
//! # Components
//!
//! - [`device`] - Compute device abstraction (CPU, GPU, TPU, Apple Silicon)
//! - [`paradigm`] - Model paradigm classification (ML, DL, FineTuning, etc.)
//! - [`metrics`] - Energy and cost metrics tracking
//! - [`platform`] - Platform efficiency (server vs edge)
//! - [`benchmark`] - Cost-performance benchmarking with Pareto analysis

pub mod benchmark;
pub mod device;
pub mod metrics;
pub mod paradigm;
pub mod platform;

pub use benchmark::{BenchmarkEntry, BenchmarkStatistics, CostPerformanceBenchmark};
pub use device::{AppleSiliconInfo, ComputeDevice, CpuInfo, GpuInfo, SimdCapability, TpuInfo};
pub use metrics::{pricing, CostMetrics, EfficiencyMetrics, EnergyMetrics};
pub use paradigm::{FineTuneMethod, ModelParadigm};
pub use platform::{
    BudgetViolation, EdgeEfficiency, PlatformEfficiency, ServerEfficiency, WasmBudget,
};
