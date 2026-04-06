//! wgpu/WGSL shader analyzer
//!
//! Analyzes WGSL compute shaders for workgroup configuration and potential issues.

use crate::analyzer::{
    AnalysisReport, Analyzer, MemoryPattern, MudaType, MudaWarning, RegisterUsage, RooflineMetric,
};
use crate::error::Result;
use regex::Regex;

/// Workgroup size configuration
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkgroupSize {
    /// X dimension
    pub x: u32,
    /// Y dimension
    pub y: u32,
    /// Z dimension
    pub z: u32,
}

impl WorkgroupSize {
    /// Total threads per workgroup
    #[must_use]
    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Check if workgroup size is optimal for GPU (multiple of 32 for warp efficiency)
    #[must_use]
    pub fn is_warp_aligned(&self) -> bool {
        self.total().is_multiple_of(32)
    }
}

/// WGSL shader statistics
#[derive(Debug, Clone, Default)]
pub struct WgslStats {
    /// Workgroup size from `@workgroup_size` attribute
    pub workgroup_size: WorkgroupSize,
    /// Number of storage buffer bindings
    pub storage_buffers: u32,
    /// Number of uniform buffer bindings
    pub uniform_buffers: u32,
    /// Number of texture bindings
    pub textures: u32,
    /// Number of arithmetic operations
    pub arithmetic_ops: u32,
    /// Number of memory operations
    pub memory_ops: u32,
}

/// WGSL/wgpu compute shader analyzer
pub struct WgpuAnalyzer {
    /// Minimum workgroup size for efficiency warning
    pub min_workgroup_size: u32,
    /// Maximum workgroup size before occupancy warning
    pub max_workgroup_size: u32,
}

impl Default for WgpuAnalyzer {
    fn default() -> Self {
        Self { min_workgroup_size: 64, max_workgroup_size: 1024 }
    }
}

impl WgpuAnalyzer {
    /// Create a new WGSL analyzer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse workgroup size from WGSL
    fn parse_workgroup_size(&self, wgsl: &str) -> WorkgroupSize {
        // Match @workgroup_size(x), @workgroup_size(x, y), or @workgroup_size(x, y, z)
        let pattern =
            Regex::new(r"@workgroup_size\s*\(\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*\)")
                .expect("valid regex pattern");

        if let Some(caps) = pattern.captures(wgsl) {
            let x = caps.get(1).map_or(1, |m| m.as_str().parse().unwrap_or(1));
            let y = caps.get(2).map_or(1, |m| m.as_str().parse().unwrap_or(1));
            let z = caps.get(3).map_or(1, |m| m.as_str().parse().unwrap_or(1));
            WorkgroupSize { x, y, z }
        } else {
            WorkgroupSize { x: 1, y: 1, z: 1 }
        }
    }

    /// Count bindings in WGSL
    fn count_bindings(&self, wgsl: &str) -> (u32, u32, u32) {
        let storage_pattern = Regex::new(r"var<storage").expect("valid regex pattern");
        let uniform_pattern = Regex::new(r"var<uniform>").expect("valid regex pattern");
        let texture_pattern = Regex::new(r"texture_\w+<").expect("valid regex pattern");

        let storage = storage_pattern.find_iter(wgsl).count() as u32;
        let uniform = uniform_pattern.find_iter(wgsl).count() as u32;
        let textures = texture_pattern.find_iter(wgsl).count() as u32;

        (storage, uniform, textures)
    }

    /// Count operations in WGSL
    fn count_operations(&self, wgsl: &str) -> (u32, u32) {
        // Arithmetic: +, -, *, /, dot, cross, etc.
        let arith_pattern =
            Regex::new(r"(\+|-|\*|/|dot|cross|normalize|length|sqrt|pow|exp|log|sin|cos|tan)")
                .expect("valid regex pattern");
        // Memory: load, store, array access
        let mem_pattern = Regex::new(r"(\[[\w\s+\-*/]+\]|textureLoad|textureSample|textureStore)")
            .expect("valid regex pattern");

        let arith = arith_pattern.find_iter(wgsl).count() as u32;
        let mem = mem_pattern.find_iter(wgsl).count() as u32;

        (arith, mem)
    }

    /// Analyze WGSL shader
    fn analyze_wgsl(&self, wgsl: &str) -> WgslStats {
        let workgroup_size = self.parse_workgroup_size(wgsl);
        let (storage_buffers, uniform_buffers, textures) = self.count_bindings(wgsl);
        let (arithmetic_ops, memory_ops) = self.count_operations(wgsl);

        WgslStats {
            workgroup_size,
            storage_buffers,
            uniform_buffers,
            textures,
            arithmetic_ops,
            memory_ops,
        }
    }

    /// Detect potential issues in WGSL
    fn detect_wgsl_muda(&self, stats: &WgslStats) -> Vec<MudaWarning> {
        let mut warnings = Vec::new();

        // Check workgroup size
        let total = stats.workgroup_size.total();

        if total < self.min_workgroup_size {
            warnings.push(MudaWarning {
                muda_type: MudaType::Waiting,
                description: format!(
                    "Small workgroup size: {} threads (minimum recommended: {})",
                    total, self.min_workgroup_size
                ),
                impact: "Low GPU occupancy, potential for underutilization".to_string(),
                line: None,
                suggestion: Some(format!(
                    "Consider increasing workgroup size to at least {}",
                    self.min_workgroup_size
                )),
            });
        }

        if total > self.max_workgroup_size {
            warnings.push(MudaWarning {
                muda_type: MudaType::Overprocessing,
                description: format!(
                    "Large workgroup size: {} threads (maximum recommended: {})",
                    total, self.max_workgroup_size
                ),
                impact: "May cause register pressure and reduce occupancy".to_string(),
                line: None,
                suggestion: Some(format!(
                    "Consider reducing workgroup size to at most {}",
                    self.max_workgroup_size
                )),
            });
        }

        if !stats.workgroup_size.is_warp_aligned() && total > 1 {
            warnings.push(MudaWarning {
                muda_type: MudaType::Waiting,
                description: format!(
                    "Workgroup size {} is not a multiple of 32 (warp size)",
                    total
                ),
                impact: "Partial warp execution wastes GPU cycles".to_string(),
                line: None,
                suggestion: Some("Align workgroup size to a multiple of 32".to_string()),
            });
        }

        warnings
    }
}

impl Analyzer for WgpuAnalyzer {
    fn target_name(&self) -> &str {
        "WGSL (wgpu)"
    }

    fn analyze(&self, wgsl: &str) -> Result<AnalysisReport> {
        let stats = self.analyze_wgsl(wgsl);
        let warnings = self.detect_muda(wgsl);

        // Estimate instruction count from operations
        let instruction_count = stats.arithmetic_ops + stats.memory_ops;

        // Estimate "occupancy" based on workgroup efficiency
        let total_threads = stats.workgroup_size.total();
        let occupancy = if total_threads >= self.min_workgroup_size {
            (total_threads as f32 / self.max_workgroup_size as f32).min(1.0)
        } else {
            total_threads as f32 / self.min_workgroup_size as f32
        };

        Ok(AnalysisReport {
            name: "wgsl_analysis".to_string(),
            target: self.target_name().to_string(),
            registers: RegisterUsage::default(), // WGSL doesn't expose register info
            memory: MemoryPattern {
                global_loads: stats.memory_ops,
                global_stores: 0,
                shared_loads: 0,
                shared_stores: 0,
                coalesced_ratio: 1.0, // Assume coalesced by default
            },
            roofline: self.estimate_roofline(&AnalysisReport::default()),
            warnings,
            instruction_count,
            estimated_occupancy: occupancy,
        })
    }

    fn detect_muda(&self, wgsl: &str) -> Vec<MudaWarning> {
        let stats = self.analyze_wgsl(wgsl);
        self.detect_wgsl_muda(&stats)
    }

    fn estimate_roofline(&self, _analysis: &AnalysisReport) -> RooflineMetric {
        RooflineMetric {
            arithmetic_intensity: 1.0,
            theoretical_peak_gflops: 500.0, // Placeholder for wgpu
            memory_bound: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workgroup_size_1d() {
        let wgsl = "@workgroup_size(64)\nfn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let size = analyzer.parse_workgroup_size(wgsl);

        assert_eq!(size.x, 64);
        assert_eq!(size.y, 1);
        assert_eq!(size.z, 1);
        assert_eq!(size.total(), 64);
    }

    #[test]
    fn test_parse_workgroup_size_2d() {
        let wgsl = "@workgroup_size(8, 8)\nfn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let size = analyzer.parse_workgroup_size(wgsl);

        assert_eq!(size.x, 8);
        assert_eq!(size.y, 8);
        assert_eq!(size.z, 1);
        assert_eq!(size.total(), 64);
    }

    #[test]
    fn test_parse_workgroup_size_3d() {
        let wgsl = "@workgroup_size(4, 4, 4)\nfn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let size = analyzer.parse_workgroup_size(wgsl);

        assert_eq!(size.x, 4);
        assert_eq!(size.y, 4);
        assert_eq!(size.z, 4);
        assert_eq!(size.total(), 64);
    }

    #[test]
    fn test_parse_workgroup_size_missing() {
        let wgsl = "fn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let size = analyzer.parse_workgroup_size(wgsl);

        assert_eq!(size.total(), 1);
    }

    #[test]
    fn test_warp_aligned() {
        assert!(WorkgroupSize { x: 64, y: 1, z: 1 }.is_warp_aligned());
        assert!(WorkgroupSize { x: 8, y: 8, z: 1 }.is_warp_aligned());
        assert!(WorkgroupSize { x: 256, y: 1, z: 1 }.is_warp_aligned());
        assert!(!WorkgroupSize { x: 33, y: 1, z: 1 }.is_warp_aligned());
        assert!(!WorkgroupSize { x: 7, y: 7, z: 1 }.is_warp_aligned());
    }

    #[test]
    fn test_count_bindings() {
        let wgsl = r"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @group(0) @binding(2) var<uniform> params: Params;
        ";
        let analyzer = WgpuAnalyzer::new();
        let (storage, uniform, textures) = analyzer.count_bindings(wgsl);

        assert_eq!(storage, 2);
        assert_eq!(uniform, 1);
        assert_eq!(textures, 0);
    }

    #[test]
    fn test_detect_small_workgroup() {
        let wgsl = "@workgroup_size(8)\nfn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let warnings = analyzer.detect_muda(wgsl);

        assert!(!warnings.is_empty(), "Should warn on small workgroup");
        assert!(warnings.iter().any(|w| w.description.contains("Small workgroup")));
    }

    #[test]
    fn test_detect_non_warp_aligned() {
        let wgsl = "@workgroup_size(33)\nfn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let warnings = analyzer.detect_muda(wgsl);

        assert!(warnings.iter().any(|w| w.description.contains("not a multiple of 32")));
    }

    #[test]
    fn test_optimal_workgroup_no_warnings() {
        let wgsl = "@workgroup_size(256)\nfn main() {}";
        let analyzer = WgpuAnalyzer::new();
        let warnings = analyzer.detect_muda(wgsl);

        // 256 is warp-aligned and within bounds
        assert!(warnings.is_empty(), "Optimal workgroup should have no warnings");
    }

    /// F067: Detects workgroup size
    #[test]
    fn f067_detect_workgroup_size() {
        let wgsl = r"
            @compute @workgroup_size(64, 4, 1)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                // compute work
            }
        ";

        let analyzer = WgpuAnalyzer::new();
        let report = analyzer.analyze(wgsl).unwrap();

        // Verify analysis completed
        assert_eq!(report.target, "WGSL (wgpu)");

        // Verify workgroup was parsed (64*4*1 = 256 threads)
        let stats = analyzer.analyze_wgsl(wgsl);
        assert_eq!(stats.workgroup_size.x, 64);
        assert_eq!(stats.workgroup_size.y, 4);
        assert_eq!(stats.workgroup_size.z, 1);
        assert_eq!(stats.workgroup_size.total(), 256);
    }

    #[test]
    fn test_analyze_full_wgsl() {
        let wgsl = r"
            struct Params {
                size: u32,
            }

            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<uniform> params: Params;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x;
                if idx < params.size {
                    result[idx] = a[idx] + b[idx];
                }
            }
        ";

        let analyzer = WgpuAnalyzer::new();
        let report = analyzer.analyze(wgsl).unwrap();

        assert_eq!(report.target, "WGSL (wgpu)");
        assert!(report.warnings.is_empty(), "Valid WGSL should have no warnings");
    }
}
