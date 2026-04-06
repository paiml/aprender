//! SIMD vectorization analyzer
//!
//! Analyzes x86 assembly for SIMD instruction usage and vectorization patterns.

use crate::analyzer::{
    AnalysisReport, Analyzer, MemoryPattern, MudaType, MudaWarning, RegisterUsage, RooflineMetric,
};
use crate::error::Result;
use regex::Regex;

/// Supported SIMD architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdArch {
    /// SSE2 (128-bit)
    Sse2,
    /// AVX/AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
}

impl SimdArch {
    /// Vector width in bits
    #[must_use]
    pub fn width_bits(&self) -> u32 {
        match self {
            Self::Sse2 | Self::Neon => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
        }
    }

    /// Maximum f32 elements per vector
    #[must_use]
    pub fn f32_lanes(&self) -> u32 {
        self.width_bits() / 32
    }
}

/// SIMD instruction counts
#[derive(Debug, Clone, Default)]
pub struct SimdInstructionCounts {
    /// Scalar instructions
    pub scalar: u32,
    /// SSE/SSE2 instructions (128-bit)
    pub sse: u32,
    /// AVX/AVX2 instructions (256-bit)
    pub avx: u32,
    /// AVX-512 instructions (512-bit)
    pub avx512: u32,
}

impl SimdInstructionCounts {
    /// Calculate vectorization ratio (0.0-1.0)
    #[must_use]
    pub fn vectorization_ratio(&self) -> f32 {
        let total = self.scalar + self.sse + self.avx + self.avx512;
        if total == 0 {
            return 0.0;
        }
        let vectorized = self.sse + self.avx + self.avx512;
        vectorized as f32 / total as f32
    }
}

/// SIMD code analyzer
pub struct SimdAnalyzer {
    /// Target architecture for analysis
    pub target_arch: SimdArch,
    /// Warn if vectorization ratio below threshold
    pub vectorization_threshold: f32,
}

impl Default for SimdAnalyzer {
    fn default() -> Self {
        Self { target_arch: SimdArch::Avx2, vectorization_threshold: 0.5 }
    }
}

impl SimdAnalyzer {
    /// Create a new SIMD analyzer for the given architecture
    #[must_use]
    pub fn new(arch: SimdArch) -> Self {
        Self { target_arch: arch, ..Default::default() }
    }

    /// Count SIMD instructions in assembly
    fn count_instructions(&self, asm: &str) -> SimdInstructionCounts {
        let mut counts = SimdInstructionCounts::default();

        // AVX-512 patterns (zmm registers, 512-bit ops)
        let avx512_pattern =
            Regex::new(r"(?i)(v\w+.*zmm|vp\w+.*zmm)").expect("valid regex pattern");
        counts.avx512 = avx512_pattern.find_iter(asm).count() as u32;

        // AVX/AVX2 patterns (ymm registers, 256-bit ops, v-prefix)
        let avx_pattern = Regex::new(
            r"(?i)(v\w+.*ymm|vp\w+.*ymm|vmovaps|vmovups|vmulps|vaddps|vsubps|vdivps|vfmadd|vfmsub)",
        )
        .expect("valid regex pattern");
        counts.avx = avx_pattern.find_iter(asm).count() as u32;

        // SSE patterns (xmm registers without v-prefix)
        // Note: Rust regex doesn't support look-behind, so we match and filter
        let sse_pattern = Regex::new(r"(?i)\b(movaps|movups|mulps|addps|subps|divps)\b.*xmm")
            .expect("valid regex pattern");
        counts.sse = sse_pattern.find_iter(asm).count() as u32;

        // Scalar floating-point (ss = scalar single-precision)
        let scalar_pattern =
            Regex::new(r"(?i)\b(movss|mulss|addss|subss|divss|cvtsi2ss|cvtss2si)\b")
                .expect("valid regex pattern");
        counts.scalar = scalar_pattern.find_iter(asm).count() as u32;

        counts
    }

    /// Detect scalar fallback code (Muda of Overprocessing)
    fn detect_scalar_fallback(&self, counts: &SimdInstructionCounts) -> Option<MudaWarning> {
        let ratio = counts.vectorization_ratio();
        if ratio < self.vectorization_threshold && counts.scalar > 0 {
            Some(MudaWarning {
                muda_type: MudaType::Overprocessing,
                description: format!(
                    "Low vectorization: {:.1}% (threshold: {:.1}%)",
                    ratio * 100.0,
                    self.vectorization_threshold * 100.0
                ),
                impact: format!(
                    "Potential {:.1}x speedup from better vectorization",
                    self.target_arch.f32_lanes()
                ),
                line: None,
                suggestion: Some("Check for alignment issues or loop trip count".to_string()),
            })
        } else {
            None
        }
    }
}

impl Analyzer for SimdAnalyzer {
    fn target_name(&self) -> &str {
        match self.target_arch {
            SimdArch::Sse2 => "x86 ASM (SSE2)",
            SimdArch::Avx2 => "x86 ASM (AVX2)",
            SimdArch::Avx512 => "x86 ASM (AVX-512)",
            SimdArch::Neon => "ARM ASM (NEON)",
        }
    }

    fn analyze(&self, asm: &str) -> Result<AnalysisReport> {
        let counts = self.count_instructions(asm);
        let warnings = self.detect_muda(asm);

        let total_instructions = counts.scalar + counts.sse + counts.avx + counts.avx512;
        let vectorization = counts.vectorization_ratio();

        Ok(AnalysisReport {
            name: "simd_analysis".to_string(),
            target: self.target_name().to_string(),
            registers: RegisterUsage::default(),
            memory: MemoryPattern::default(),
            roofline: self.estimate_roofline(&AnalysisReport::default()),
            warnings,
            instruction_count: total_instructions,
            estimated_occupancy: vectorization, // Repurpose as vectorization ratio
        })
    }

    fn detect_muda(&self, asm: &str) -> Vec<MudaWarning> {
        let mut warnings = Vec::new();
        let counts = self.count_instructions(asm);

        if let Some(w) = self.detect_scalar_fallback(&counts) {
            warnings.push(w);
        }

        warnings
    }

    fn estimate_roofline(&self, _analysis: &AnalysisReport) -> RooflineMetric {
        // SIMD typically memory-bound for large data
        RooflineMetric {
            arithmetic_intensity: 1.0,
            theoretical_peak_gflops: 1000.0, // Placeholder
            memory_bound: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_arch_width() {
        assert_eq!(SimdArch::Sse2.width_bits(), 128);
        assert_eq!(SimdArch::Avx2.width_bits(), 256);
        assert_eq!(SimdArch::Avx512.width_bits(), 512);
    }

    #[test]
    fn test_simd_arch_lanes() {
        assert_eq!(SimdArch::Sse2.f32_lanes(), 4);
        assert_eq!(SimdArch::Avx2.f32_lanes(), 8);
        assert_eq!(SimdArch::Avx512.f32_lanes(), 16);
    }

    #[test]
    fn test_count_avx_instructions() {
        let asm = r"
            vmovaps ymm0, [rdi]
            vmovaps ymm1, [rsi]
            vaddps ymm2, ymm0, ymm1
            vmovaps [rdx], ymm2
        ";

        let analyzer = SimdAnalyzer::new(SimdArch::Avx2);
        let counts = analyzer.count_instructions(asm);

        assert!(counts.avx > 0, "Should detect AVX instructions");
    }

    #[test]
    fn test_count_sse_instructions() {
        let asm = r"
            movaps xmm0, [rdi]
            movaps xmm1, [rsi]
            addps xmm0, xmm1
            movaps [rdx], xmm0
        ";

        let analyzer = SimdAnalyzer::new(SimdArch::Sse2);
        let counts = analyzer.count_instructions(asm);

        assert!(counts.sse > 0, "Should detect SSE instructions");
    }

    #[test]
    fn test_vectorization_ratio() {
        let counts = SimdInstructionCounts { scalar: 2, sse: 0, avx: 8, avx512: 0 };

        let ratio = counts.vectorization_ratio();
        assert!((ratio - 0.8).abs() < 0.01, "Expected 80% vectorization");
    }

    #[test]
    fn test_vectorization_ratio_zero() {
        let counts = SimdInstructionCounts::default();
        assert_eq!(counts.vectorization_ratio(), 0.0);
    }

    #[test]
    fn test_detect_scalar_fallback() {
        let asm = r"
            movss xmm0, [rdi]
            mulss xmm0, xmm1
            addss xmm0, xmm2
        ";

        let analyzer = SimdAnalyzer::new(SimdArch::Avx2);
        let warnings = analyzer.detect_muda(asm);

        assert!(!warnings.is_empty(), "Should warn on scalar code");
    }

    /// F051: Detects AVX2 instructions
    #[test]
    fn f051_detect_avx2_instructions() {
        let asm = "vmulps ymm0, ymm1, ymm2";
        let analyzer = SimdAnalyzer::new(SimdArch::Avx2);
        let counts = analyzer.count_instructions(asm);

        assert!(counts.avx > 0, "Should detect vmulps");
    }

    /// F055: Calculates vectorization ratio
    #[test]
    fn f055_vectorization_ratio_positive() {
        let asm = r"
            vmovaps ymm0, [rdi]
            vaddps ymm0, ymm0, ymm1
        ";

        let analyzer = SimdAnalyzer::new(SimdArch::Avx2);
        let report = analyzer.analyze(asm).unwrap();

        // estimated_occupancy repurposed as vectorization ratio
        assert!(report.estimated_occupancy > 0.0, "Vectorization ratio should be > 0%");
    }
}
