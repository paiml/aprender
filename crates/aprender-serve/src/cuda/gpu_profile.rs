//! Auto-detected GPU kernel configuration.
//!
//! Replaces per-machine env var tuning (`DP4A_Q4K`, `HW_DP4A_Q4K`, `MWV_Q6K`, etc.)
//! with automatic detection based on `compute_capability()`.
//!
//! Env vars still work as overrides for experimentation, but the defaults are
//! now correct for each GPU — no forjar config drift.

use trueno_gpu::driver::CudaContext;

/// Kernel variant for Q4K GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4kVariant {
    /// Legacy single-warp (32 threads), no DP4A. Fallback for sm < 7.5.
    Legacy,
    /// Wide: 128 threads per output row.
    Wide,
    /// Vectorized: 32 threads with vectorized loads.
    Vectorized,
    /// Multi-warp DP4A: 32 threads/super-block with shfl broadcast.
    MwvDp4a,
    /// Half-warp DP4A: 16 threads/super-block, direct scale loads. Best on sm_75+.
    HwDp4a,
    /// Multi-warp vectorized (no DP4A).
    Mwv,
}

/// Kernel variant for Q6K GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q6kVariant {
    /// Original single-warp Q6K (fallback).
    Legacy,
    /// Multi-warp vectorized Q6K (GH-118).
    Mwv,
    /// DP4A Q6K with Q8 pre-quantization.
    Dp4a,
    /// Half-warp DP4A Q6K: 16 threads/SB, direct scale loads (PMAT-030).
    HwDp4a,
}

/// Auto-detected GPU profile for kernel dispatch.
///
/// Computed once at executor init from `compute_capability()`.
/// All kernel dispatch reads from this instead of env vars.
#[derive(Debug, Clone)]
pub struct GpuProfile {
    /// Q4K GEMV kernel variant (auto-detected: HwDp4a on sm_75+).
    pub q4k: Q4kVariant,
    /// Q6K GEMV kernel variant (auto-detected: Dp4a on sm_75+).
    pub q6k: Q6kVariant,
    /// Multi-warp GEMV warp count (default: 3, override: MWV_WARPS env).
    pub mwv_warps: u32,
    /// Whether batched prefill is enabled (default: true, override: BATCHED_PREFILL=0).
    pub batched_prefill: bool,
    /// Whether to use cuBLAS HGEMM for decode (M=1) on high-BW GPUs.
    /// Auto-detected: true on sm_75+ with >=32 SMs (desktop/server class).
    /// Override: HGEMM_DECODE=1/0 or CUBLAS_GEMM_THRESHOLD=1.
    pub hgemm_decode: bool,
    /// Whether to use fused gate+up+SwiGLU kernel (PMAT-034).
    /// Saves 11% instructions + eliminates SwiGLU kernel + 4 buffer passes.
    /// Auto-detected: true when q4k=HwDp4a. Override: FUSED_GATE_UP=0/1.
    pub fused_gate_up: bool,
    /// PMAT-067: Use FP8 E4M3 weights for prefill GEMM (1 B/elem vs FP16's 2 B/elem).
    /// Auto-detected: true on sm_89+ (Ada Lovelace FP8 tensor cores).
    /// Override: FP8_PREFILL=0 to disable, FP8_PREFILL=1 to force.
    /// cuBLASLt FP8 GEMM halves weight bandwidth — TTFT improvement ~1.25x.
    pub fp8_prefill: bool,
    /// PMAT-090: Use FP8 cuBLASLt GEMM for batched decode (M>=2).
    /// DP4A Q4K GEMV is compute-bound at M>1 (DP4A ceiling = 306 tok/s at M=4).
    /// FP8 (1 B/elem) reads 1.78× more than Q4K (0.5625) but tensor cores keep
    /// it memory-bound — expected: ITL ~15→~8ms, aggregate ~257→~380 tok/s.
    /// Auto-detected: true on sm_89+ when fp8_prefill is enabled.
    /// Override: FP8_DECODE=0 to disable, FP8_DECODE=1 to force.
    pub fp8_decode: bool,
    /// PMAT-091: Use column-interleaved Q4K WMMA GEMM for batched decode (M>=2).
    /// W4A16: INT4 storage (Q4K, 0.5625 B/elem) + FP16 tensor core compute.
    /// Interleaved layout fixes 864-byte cross-column stride → perfect 128B coalescing.
    /// At 70% WMMA efficiency: est. +34% c=4 aggregate over FP8.
    /// Override: W4A16_INTERLEAVED=0 to disable, W4A16_INTERLEAVED=1 to force.
    pub w4a16_interleaved: bool,
    /// SM version for logging (e.g., "sm_89").
    pub sm_target: String,
    /// Numeric compute capability (major*10 + minor, e.g. 89 for sm_89).
    /// Used for numeric comparisons instead of string lexicographic (avoids sm_100 bug).
    pub cc: u32,
}

impl GpuProfile {
    /// Detect optimal kernel configuration from GPU hardware.
    ///
    /// Priority: env var override > auto-detect from compute capability.
    /// This means `HW_DP4A_Q4K=1` still works for experimentation,
    /// but production deployments need zero env vars.
    pub fn detect(context: &CudaContext) -> Self {
        contract_pre_target_parity!();
        let (major, minor) = context.compute_capability().unwrap_or((7, 0));
        // GH-480: PTX source `.target` must use a version that PTX 8.0 supports (max sm_90).
        // The CUDA JIT compiler (`CU_JIT_TARGET` in module.rs) receives the REAL
        // compute capability (e.g. 121 for Blackwell) so it compiles natively.
        // PTX `.target` = minimum ISA needed; JIT target = actual device.
        let (ptx_major, ptx_minor) = if major > 9 || (major == 9 && minor > 0) {
            (9, 0) // sm_90 is max target PTX 8.0 supports
        } else {
            (major, minor)
        };
        let sm_target = format!("sm_{ptx_major}{ptx_minor}");
        let has_dp4a = major > 7 || (major == 7 && minor >= 5); // sm_75+ (Turing)
        let num_sms = context.multiprocessor_count().unwrap_or(8) as u32;

        let q4k = Self::detect_q4k(has_dp4a);
        let q6k = Self::detect_q6k(has_dp4a);
        let mwv_warps = Self::detect_mwv_warps();
        let batched_prefill = Self::detect_batched_prefill();
        let hgemm_decode = Self::detect_hgemm_decode(has_dp4a, num_sms);
        let fused_gate_up = Self::detect_fused_gate_up(&q4k);

        let cc = major as u32 * 10 + minor as u32;
        let fp8_prefill = Self::detect_fp8_prefill(cc);
        let fp8_decode = Self::detect_fp8_decode(fp8_prefill, cc);
        let w4a16_interleaved = Self::detect_w4a16_interleaved(cc);

        let profile = Self {
            q4k,
            q6k,
            mwv_warps,
            batched_prefill,
            hgemm_decode,
            fused_gate_up,
            fp8_prefill,
            fp8_decode,
            w4a16_interleaved,
            sm_target,
            cc,
        };

        // GH-611: Suppressed — was noisy in non-verbose mode

        profile
    }

    /// Q4K variant: env var override, else HwDp4a on sm_75+, else Mwv.
    fn detect_q4k(has_dp4a: bool) -> Q4kVariant {
        // Env var overrides (for experimentation only)
        if std::env::var("WIDE_Q4K_DISABLE").is_ok() {
            return Q4kVariant::Legacy;
        }
        if std::env::var("WIDE_Q4K").is_ok() {
            return Q4kVariant::Wide;
        }
        if std::env::var("VECTORIZED_Q4K").is_ok() {
            return Q4kVariant::Vectorized;
        }
        if std::env::var("HW_DP4A_Q4K").is_ok() {
            return Q4kVariant::HwDp4a;
        }
        if std::env::var("DP4A_Q4K").is_ok() {
            return Q4kVariant::MwvDp4a;
        }
        // PMAT-096: Force FP32 MWV variant (no Q8 quantization overhead)
        if std::env::var("MWV_Q4K").is_ok() {
            return Q4kVariant::Mwv;
        }

        // Auto-detect: HW DP4A on any GPU with DP4A support (sm_75+)
        if has_dp4a {
            Q4kVariant::HwDp4a
        } else {
            Q4kVariant::Mwv
        }
    }

    /// Q6K variant: env var override, else HwDp4a on sm_75+, else Mwv.
    fn detect_q6k(has_dp4a: bool) -> Q6kVariant {
        if std::env::var("HW_DP4A_Q6K").is_ok() {
            return Q6kVariant::HwDp4a;
        }
        if std::env::var("DP4A_Q6K").is_ok() {
            return Q6kVariant::Dp4a;
        }
        if std::env::var("MWV_Q6K").is_ok() {
            return Q6kVariant::Mwv;
        }

        if has_dp4a {
            Q6kVariant::HwDp4a
        } else {
            Q6kVariant::Mwv
        }
    }

    /// MWV warp count: env var override, else 3.
    /// PMAT-089: 4 warps FALSIFIED (-2% decode due to register pressure). 3 is optimal.
    fn detect_mwv_warps() -> u32 {
        std::env::var("MWV_WARPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
    }

    /// Batched prefill: env var override, else always on.
    fn detect_batched_prefill() -> bool {
        // BATCHED_PREFILL=0 disables; any other value or absent = enabled
        std::env::var("BATCHED_PREFILL")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    /// Fused gate+up+SwiGLU: enabled when HW DP4A Q4K is active.
    /// Saves 11% instructions + eliminates SwiGLU kernel + 4 intermediate buffer passes.
    fn detect_fused_gate_up(q4k: &Q4kVariant) -> bool {
        if let Ok(v) = std::env::var("FUSED_GATE_UP") {
            return v != "0";
        }
        // Auto-enable when using HW DP4A Q4K (the fused kernel is HW DP4A only)
        *q4k == Q4kVariant::HwDp4a
    }

    /// PMAT-053b: FP8 prefill — default ON for sm_89+ (Ada/Hopper), OFF for Blackwell.
    ///
    /// FP8 E4M3 weights are 1 B/elem vs FP16's 2 B/elem — halves weight bandwidth.
    /// Per-tensor absmax scaling recovers dynamic range (TTFT 46.4→35.5ms, 1.31x).
    /// Override: FP8_PREFILL=0 to disable, FP8_PREFILL=1 to force on older GPUs.
    ///
    /// GH-542: Blackwell (sm_100+, cc >= 100) FP8 warmup crashes context.
    /// But the FP8 cuBLASLt GEMM itself works on sm_121 (PMAT-410 verified).
    /// Enable FP8 prefill on all cc >= 89; warmup_fp8_cache separately guards
    /// against the warmup crash (cc < 100 check in attention.rs).
    fn detect_fp8_prefill(cc: u32) -> bool {
        contract_pre_fp8_architecture_guard!();
        match std::env::var("FP8_PREFILL").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => cc >= 89,
        }
    }

    /// PMAT-090: FP8 batched decode — cuBLASLt FP8 GEMM replaces DP4A Q4K GEMV at M>=2.
    ///
    /// DP4A GEMV is compute-bound at M>1: 4 independent DP4A accumulation chains
    /// saturate INT32 units. DP4A ceiling = 306 tok/s at M=4 (theoretical).
    /// FP8 cuBLASLt reads 1.78× more BW (1 B/elem vs Q4K 0.5625) but stays
    /// memory-bound via tensor cores. Expected: ~1.5× c=4 aggregate improvement.
    ///
    /// PMAT-410: FP8 decode follows fp8_prefill. No separate cc guard needed.
    fn detect_fp8_decode(fp8_prefill: bool, _cc: u32) -> bool {
        match std::env::var("FP8_DECODE").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => fp8_prefill,
        }
    }

    /// PMAT-091: W4A16 interleaved WMMA for batched decode.
    /// Requires sm_70+ for WMMA tensor cores. Default OFF (experimental).
    fn detect_w4a16_interleaved(cc: u32) -> bool {
        match std::env::var("W4A16_INTERLEAVED").as_deref() {
            Ok("0") => false,
            Ok("1") => cc >= 70,
            _ => false, // Default OFF until benchmarked
        }
    }

    /// HGEMM decode: use cuBLAS HGEMM (cached FP16 weights) for M=1 decode.
    ///
    /// PMAT-037 RESULT: cuBLAS HGEMM for M=1 is SLOWER than Q4K GEMV on both
    /// 4090 (109 vs 193 tok/s) and Jetson Orin. FP16 reads 3.56x more data
    /// and cuBLAS launch overhead dominates at M=1. Disabled by default.
    fn detect_hgemm_decode(_has_dp4a: bool, _num_sms: u32) -> bool {
        // Env var override (for experimentation)
        if let Ok(v) = std::env::var("HGEMM_DECODE") {
            return v == "1";
        }
        // PMAT-037 RESULT: cuBLAS HGEMM for M=1 is SLOWER than Q4K GEMV (109 vs 200 tok/s).
        // FP16 reads 3.56x more data, and cuBLAS overhead dominates at M=1.
        // Keep disabled by default — only useful for M>=4 prefill (batched path).
        false
    }
}
