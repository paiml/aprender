//! Analyzer bricks (Layer 2)
//!
//! Business logic and derived metrics.

mod bottleneck;
mod efficiency;
mod thermal;
mod throughput;

pub use bottleneck::BottleneckAnalyzerBrick;
pub use efficiency::EfficiencyAnalyzerBrick;
pub use thermal::ThermalAnalyzerBrick;
pub use throughput::ThroughputAnalyzerBrick;
