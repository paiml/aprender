//! Roofline Model Analyzer (PMAT-022)
//!
//! Implements Williams Roofline Model per Citation [70] for visual bottleneck
//! analysis. Determines if workload is compute-bound or memory-bound based
//! on operational intensity.
//!
//! # Roofline Model Components
//!
//! | Component | Formula | Unit |
//! |-----------|---------|------|
//! | Operational Intensity (OI) | FLOP / Bytes | FLOP/Byte |
//! | Peak Compute | Theoretical GFLOPS | GFLOP/s |
//! | Peak Memory BW | Memory bandwidth | GB/s |
//! | Ridge Point | Peak Compute / Peak BW | FLOP/Byte |
//!
//! # Citations
//!
//! - [Williams et al. 2009] "Roofline: An Insightful Visual Performance Model" CACM 52(4)
//! - [Ofenbeck et al. 2014] "Applying the Roofline Model" IEEE ISPASS

/// Bottleneck classification based on operational intensity vs ridge point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    /// OI < Ridge Point - workload is limited by memory bandwidth
    MemoryBound,
    /// OI > Ridge Point - workload is limited by compute throughput
    ComputeBound,
    /// OI ≈ Ridge Point (within 10%) - both equally important
    Balanced,
}

impl BottleneckType {
    /// Get optimization recommendation
    pub fn recommendation(&self) -> &'static str {
        match self {
            BottleneckType::MemoryBound => {
                "Improve memory access patterns: coalescing, prefetching, cache blocking"
            }
            BottleneckType::ComputeBound => {
                "Improve compute efficiency: SIMD, kernel fusion, algorithm optimization"
            }
            BottleneckType::Balanced => {
                "Both memory and compute matter equally; profile to find specific bottleneck"
            }
        }
    }

    /// Get short name
    pub fn name(&self) -> &'static str {
        match self {
            BottleneckType::MemoryBound => "memory-bound",
            BottleneckType::ComputeBound => "compute-bound",
            BottleneckType::Balanced => "balanced",
        }
    }
}

/// Hardware profile for roofline analysis
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Device name
    pub name: String,
    /// Peak compute throughput in GFLOPS (FP32)
    pub peak_gflops: f64,
    /// Peak memory bandwidth in GB/s
    pub peak_bandwidth_gbps: f64,
    /// Ridge point (peak_gflops / peak_bandwidth_gbps)
    ridge_point: f64,
}

impl HardwareProfile {
    /// Create a new hardware profile
    pub fn new(name: &str, peak_gflops: f64, peak_bandwidth_gbps: f64) -> Self {
        let ridge_point =
            if peak_bandwidth_gbps > 0.0 { peak_gflops / peak_bandwidth_gbps } else { 0.0 };
        Self { name: name.to_string(), peak_gflops, peak_bandwidth_gbps, ridge_point }
    }

    /// Get the ridge point (transition from memory-bound to compute-bound)
    pub fn ridge_point(&self) -> f64 {
        self.ridge_point
    }

    /// Calculate theoretical peak performance for a given OI
    pub fn theoretical_peak_at_oi(&self, operational_intensity: f64) -> f64 {
        // Roofline: min(peak_compute, peak_bandwidth * OI)
        let memory_bound_peak = self.peak_bandwidth_gbps * operational_intensity;
        self.peak_gflops.min(memory_bound_peak)
    }

    /// Classify bottleneck based on operational intensity
    pub fn classify_bottleneck(&self, operational_intensity: f64) -> BottleneckType {
        let ratio = operational_intensity / self.ridge_point;

        if ratio < 0.9 {
            BottleneckType::MemoryBound
        } else if ratio > 1.1 {
            BottleneckType::ComputeBound
        } else {
            BottleneckType::Balanced
        }
    }
}

/// Pre-defined hardware profiles
pub mod profiles {
    use super::HardwareProfile;

    /// NVIDIA A100 SXM 40GB/80GB
    pub fn a100_sxm() -> HardwareProfile {
        HardwareProfile::new("NVIDIA A100 SXM", 19_500.0, 2_039.0)
    }

    /// NVIDIA H100 SXM 80GB
    pub fn h100_sxm() -> HardwareProfile {
        HardwareProfile::new("NVIDIA H100 SXM", 51_200.0, 3_350.0)
    }

    /// NVIDIA RTX 4090
    pub fn rtx_4090() -> HardwareProfile {
        HardwareProfile::new("NVIDIA RTX 4090", 82_580.0, 1_008.0)
    }

    /// NVIDIA RTX 3090
    pub fn rtx_3090() -> HardwareProfile {
        HardwareProfile::new("NVIDIA RTX 3090", 35_580.0, 936.0)
    }

    /// AMD Instinct MI250X
    pub fn mi250x() -> HardwareProfile {
        HardwareProfile::new("AMD Instinct MI250X", 47_872.0, 3_277.0)
    }

    /// Intel Xeon with AVX-512 (per core)
    pub fn avx512_per_core() -> HardwareProfile {
        HardwareProfile::new("AVX-512 (per core)", 128.0, 50.0)
    }

    /// Apple M2 Ultra GPU
    pub fn m2_ultra_gpu() -> HardwareProfile {
        HardwareProfile::new("Apple M2 Ultra GPU", 27_200.0, 800.0)
    }

    /// All predefined profiles
    pub fn all() -> Vec<HardwareProfile> {
        vec![
            a100_sxm(),
            h100_sxm(),
            rtx_4090(),
            rtx_3090(),
            mi250x(),
            avx512_per_core(),
            m2_ultra_gpu(),
        ]
    }
}

/// Workload metrics for roofline analysis
#[derive(Debug, Clone)]
pub struct WorkloadMetrics {
    /// Workload name
    pub name: String,
    /// Total floating-point operations
    pub total_flops: f64,
    /// Total bytes transferred (read + write)
    pub total_bytes: f64,
    /// Measured performance in GFLOPS
    pub measured_gflops: f64,
    /// Execution time in seconds
    pub execution_time_s: f64,
}

impl WorkloadMetrics {
    /// Create new workload metrics
    pub fn new(name: &str, total_flops: f64, total_bytes: f64, execution_time_s: f64) -> Self {
        let measured_gflops =
            if execution_time_s > 0.0 { total_flops / execution_time_s / 1e9 } else { 0.0 };
        Self { name: name.to_string(), total_flops, total_bytes, measured_gflops, execution_time_s }
    }

    /// Calculate operational intensity (FLOP/Byte)
    pub fn operational_intensity(&self) -> f64 {
        if self.total_bytes > 0.0 {
            self.total_flops / self.total_bytes
        } else {
            0.0
        }
    }
}

/// Roofline analysis result
#[derive(Debug, Clone)]
pub struct RooflineAnalysis {
    /// Hardware profile used
    pub hardware: HardwareProfile,
    /// Workload metrics
    pub workload: WorkloadMetrics,
    /// Operational intensity
    pub operational_intensity: f64,
    /// Theoretical peak at this OI
    pub theoretical_peak: f64,
    /// Attained performance (measured / theoretical)
    pub attained_efficiency: f64,
    /// Bottleneck classification
    pub bottleneck: BottleneckType,
}

impl RooflineAnalysis {
    /// Perform roofline analysis
    pub fn analyze(hardware: &HardwareProfile, workload: &WorkloadMetrics) -> Self {
        let operational_intensity = workload.operational_intensity();
        let theoretical_peak = hardware.theoretical_peak_at_oi(operational_intensity);
        let attained_efficiency = if theoretical_peak > 0.0 {
            (workload.measured_gflops / theoretical_peak) * 100.0
        } else {
            0.0
        };
        let bottleneck = hardware.classify_bottleneck(operational_intensity);

        Self {
            hardware: hardware.clone(),
            workload: workload.clone(),
            operational_intensity,
            theoretical_peak,
            attained_efficiency,
            bottleneck,
        }
    }

    /// Get actionable recommendation
    pub fn recommendation(&self) -> String {
        let base = self.bottleneck.recommendation();
        format!(
            "{} (OI={:.2} FLOP/Byte, Ridge={:.2}, Efficiency={:.1}%)",
            base,
            self.operational_intensity,
            self.hardware.ridge_point(),
            self.attained_efficiency
        )
    }
}

/// Roofline visualization data point
#[derive(Debug, Clone)]
pub struct RooflinePlotPoint {
    /// Log2 of operational intensity (x-axis)
    pub log_oi: f64,
    /// Log2 of performance in GFLOPS (y-axis)
    pub log_perf: f64,
    /// Original OI
    pub oi: f64,
    /// Original performance
    pub perf: f64,
    /// Label
    pub label: String,
}

impl RooflinePlotPoint {
    /// Create a plot point
    pub fn new(label: &str, oi: f64, perf: f64) -> Self {
        Self { log_oi: oi.log2(), log_perf: perf.log2(), oi, perf, label: label.to_string() }
    }
}

/// Roofline plot data for visualization
#[derive(Debug, Clone)]
pub struct RooflinePlot {
    /// Hardware profile
    pub hardware: HardwareProfile,
    /// Memory-bound line points (OI from 0.1 to ridge)
    pub memory_bound_line: Vec<RooflinePlotPoint>,
    /// Compute-bound line points (OI from ridge to 100)
    pub compute_bound_line: Vec<RooflinePlotPoint>,
    /// Workload points
    pub workload_points: Vec<RooflinePlotPoint>,
    /// Ridge point marker
    pub ridge_point: RooflinePlotPoint,
}

impl RooflinePlot {
    /// Generate roofline plot data
    pub fn generate(hardware: &HardwareProfile, workloads: &[WorkloadMetrics]) -> Self {
        let ridge = hardware.ridge_point();

        // Memory-bound line (slope = bandwidth)
        let memory_bound_line: Vec<RooflinePlotPoint> = (0..=20)
            .map(|i| {
                let oi = 0.1 * (ridge / 0.1).powf(i as f64 / 20.0);
                let perf = hardware.peak_bandwidth_gbps * oi;
                RooflinePlotPoint::new("memory-bound", oi, perf)
            })
            .collect();

        // Compute-bound line (flat at peak)
        let compute_bound_line: Vec<RooflinePlotPoint> = (0..=10)
            .map(|i| {
                let oi = ridge * (100.0 / ridge).powf(i as f64 / 10.0);
                RooflinePlotPoint::new("compute-bound", oi, hardware.peak_gflops)
            })
            .collect();

        // Workload points
        let workload_points: Vec<RooflinePlotPoint> = workloads
            .iter()
            .map(|w| RooflinePlotPoint::new(&w.name, w.operational_intensity(), w.measured_gflops))
            .collect();

        // Ridge point
        let ridge_point = RooflinePlotPoint::new("ridge", ridge, hardware.peak_gflops);

        Self {
            hardware: hardware.clone(),
            memory_bound_line,
            compute_bound_line,
            workload_points,
            ridge_point,
        }
    }
}

/// Batch analysis of multiple workloads
#[derive(Debug)]
pub struct BatchRooflineAnalysis {
    /// Hardware profile
    pub hardware: HardwareProfile,
    /// Individual analyses
    pub analyses: Vec<RooflineAnalysis>,
}

impl BatchRooflineAnalysis {
    /// Analyze multiple workloads
    pub fn analyze(hardware: &HardwareProfile, workloads: &[WorkloadMetrics]) -> Self {
        let analyses = workloads.iter().map(|w| RooflineAnalysis::analyze(hardware, w)).collect();
        Self { hardware: hardware.clone(), analyses }
    }

    /// Get summary statistics
    pub fn summary(&self) -> BatchSummary {
        let memory_bound =
            self.analyses.iter().filter(|a| a.bottleneck == BottleneckType::MemoryBound).count();
        let compute_bound =
            self.analyses.iter().filter(|a| a.bottleneck == BottleneckType::ComputeBound).count();
        let balanced =
            self.analyses.iter().filter(|a| a.bottleneck == BottleneckType::Balanced).count();
        let avg_efficiency = if self.analyses.is_empty() {
            0.0
        } else {
            self.analyses.iter().map(|a| a.attained_efficiency).sum::<f64>()
                / self.analyses.len() as f64
        };

        BatchSummary {
            total: self.analyses.len(),
            memory_bound,
            compute_bound,
            balanced,
            avg_efficiency,
        }
    }
}

/// Summary of batch analysis
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// Total workloads analyzed
    pub total: usize,
    /// Number of memory-bound workloads
    pub memory_bound: usize,
    /// Number of compute-bound workloads
    pub compute_bound: usize,
    /// Number of balanced workloads
    pub balanced: usize,
    /// Average attained efficiency
    pub avg_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_point_calculation() {
        let profile = HardwareProfile::new("Test", 1000.0, 100.0);
        assert!((profile.ridge_point() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_bottleneck_classification_memory_bound() {
        let profile = HardwareProfile::new("Test", 1000.0, 100.0);
        // OI = 5 < Ridge = 10 → memory-bound
        assert_eq!(profile.classify_bottleneck(5.0), BottleneckType::MemoryBound);
    }

    #[test]
    fn test_bottleneck_classification_compute_bound() {
        let profile = HardwareProfile::new("Test", 1000.0, 100.0);
        // OI = 20 > Ridge = 10 → compute-bound
        assert_eq!(profile.classify_bottleneck(20.0), BottleneckType::ComputeBound);
    }

    #[test]
    fn test_bottleneck_classification_balanced() {
        let profile = HardwareProfile::new("Test", 1000.0, 100.0);
        // OI = 10 ≈ Ridge = 10 → balanced
        assert_eq!(profile.classify_bottleneck(10.0), BottleneckType::Balanced);
    }

    #[test]
    fn test_operational_intensity() {
        let workload = WorkloadMetrics::new("test", 1000.0, 100.0, 1.0);
        assert!((workload.operational_intensity() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_a100_profile() {
        let a100 = profiles::a100_sxm();
        assert!((a100.peak_gflops - 19500.0).abs() < 1.0);
        assert!((a100.peak_bandwidth_gbps - 2039.0).abs() < 1.0);
        // Ridge point ≈ 9.56
        assert!((a100.ridge_point() - 9.56).abs() < 0.1);
    }

    #[test]
    fn test_h100_profile() {
        let h100 = profiles::h100_sxm();
        // Ridge point ≈ 15.28
        assert!((h100.ridge_point() - 15.28).abs() < 0.1);
    }

    #[test]
    fn test_rtx_4090_profile() {
        let rtx4090 = profiles::rtx_4090();
        // Ridge point ≈ 81.9
        assert!((rtx4090.ridge_point() - 81.9).abs() < 0.5);
    }

    #[test]
    fn test_roofline_analysis() {
        let hardware = HardwareProfile::new("Test", 1000.0, 100.0);
        let workload = WorkloadMetrics::new("matmul", 1e9, 1e8, 0.01); // OI=10, 100 GFLOPS

        let analysis = RooflineAnalysis::analyze(&hardware, &workload);

        assert!((analysis.operational_intensity - 10.0).abs() < 0.01);
        assert_eq!(analysis.bottleneck, BottleneckType::Balanced);
    }
}
