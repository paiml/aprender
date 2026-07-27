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

        // Real device compute capability (e.g. 121 for GB10 Blackwell). Uses the
        // true major/minor, NOT the PTX-clamped target — PMAT-806 needs the real
        // value to gate the Blackwell fp32-MWV-Q4K default.
        let cc = major as u32 * 10 + minor as u32;

        let q4k = Self::detect_q4k(has_dp4a, cc);
        let q6k = Self::detect_q6k(has_dp4a);
        let mwv_warps = Self::detect_mwv_warps();
        let batched_prefill = Self::detect_batched_prefill();
        let hgemm_decode = Self::detect_hgemm_decode(has_dp4a, num_sms);
        let fused_gate_up = Self::detect_fused_gate_up(&q4k);

        let fp8_prefill = Self::detect_fp8_prefill(cc);
        let fp8_decode = Self::detect_fp8_decode(fp8_prefill, cc);
        let w4a16_interleaved = Self::detect_w4a16_interleaved(cc);

        // GH-611: Suppressed — was noisy in non-verbose mode

        Self {
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
        }
    }

    /// Q4K variant: env var override, else HwDp4a on sm_75+, else Mwv.
    ///
    /// PMAT-806 (Blackwell massive-activation parity): on Blackwell (cc≥120,
    /// e.g. GB10 sm_121) the HwDp4a path's INT8 Q8_1 *activation* quantization
    /// mis-estimates massive-activation channels by ~15% (latent until a deep
    /// FFN cancels the outlier → catastrophic cancellation → CPU/GPU cosine
    /// craters; on Qwen2.5-coder-1.5B Q4_K_M the load-time parity gate FAILED →
    /// silent CPU/wgpu fallback). The fp32 MWV variant does NOT quantize
    /// activations, so it is immune. On-device sweep (gx10 GB10, 2026-06-16):
    /// HwDp4a gate cosine 0.9817 (FAILs deeper models) → MWV_Q4K 0.9939 (PASS,
    /// argmax matches). Defaulting Blackwell Q4K to fp32 MWV restores CPU/GPU
    /// parity and unblocks GPU serving of quantized models there.
    ///
    /// FALSIFY-Q4K-ADA-PARITY-001 (2026-07-27) EXTENDS that default to ALL GPUs.
    /// PMAT-806's remaining assumption — "discrete GPUs (RTX 4090 sm_89, etc.)
    /// keep the fast HwDp4a path, their DP4A activation quant is reliable for
    /// these models" — was measured FALSE on an RTX 4090 running the very model
    /// it named: HwDp4a cosine 0.9186 (F2 REJECT) vs MWV 0.9937 (ACCEPT) on
    /// qwen2.5-coder-1.5B Q4_K_M. Compute capability was never the discriminator;
    /// the INT8 Q8_1 activation quant is. HwDp4a is now opt-in via HW_DP4A_Q4K.
    /// Contract: contracts/apr-cpu-vs-gpu-output-parity-v1.yaml (FALSIFY-CPU-GPU-008).
    fn detect_q4k(has_dp4a: bool, cc: u32) -> Q4kVariant {
        // Env var overrides (for experimentation only) take precedence over the
        // Blackwell default, so HW_DP4A_Q4K=1 still forces DP4A for A/B testing.
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

        Self::auto_q4k(has_dp4a, cc)
    }

    /// PMAT-806: Pure auto-detection mapping (no env vars) for the Q4K variant.
    ///
    /// ALL GPUs → fp32 MWV. HwDp4a is opt-in only (`HW_DP4A_Q4K=1`).
    ///
    /// FALSIFY-Q4K-ADA-PARITY-001. PMAT-806 defaulted Blackwell (cc≥120) to MWV
    /// because HwDp4a's INT8 Q8_1 *activation* quantization mis-estimates
    /// massive-activation channels. It left discrete GPUs on HwDp4a, asserting
    /// "their DP4A activation quant is reliable for these models".
    ///
    /// MEASURED FALSE on 2026-07-27, RTX 4090 (sm_89, cc=89, driver 570.207),
    /// on qwen2.5-coder-1.5B Q4_K_M — the very model that claim named:
    ///
    ///   HwDp4a (old default): [F2-VALIDATION] GPU diverges from CPU at real
    ///       position 1 (argmax 198 != 40, cosine 0.9186) — REJECTED
    ///   MWV (this change):    all 42 real positions match, min cosine 0.9937
    ///       — ACCEPTED
    ///
    /// 0.9186 is below the F2 floor (0.95) and below even the 0.9817 that
    /// PMAT-806 recorded as failing on Blackwell. So the degradation is not
    /// Blackwell-specific; cc≥120 was never the discriminator, the DP4A
    /// activation quant itself is.
    ///
    /// The cost of getting this wrong is NOT "slightly worse tokens" — the F2
    /// gate is fail-closed, so a rejected CUDA path falls to wgpu, which fails
    /// its own parity gate (cosine 0.884 < 0.99), and then to CPU: ~20 tok/s
    /// instead of ~400 on the most common discrete GPU there is. HwDp4a's speed
    /// advantage is worth nothing when the gate refuses to let it serve a token.
    ///
    /// Kept as a pure fn so the policy is unit-testable without a device.
    #[must_use]
    pub(crate) fn auto_q4k(has_dp4a: bool, cc: u32) -> Q4kVariant {
        // `has_dp4a`/`cc` retained: the signature is the policy's seam, and a
        // future per-arch carve-out belongs here rather than at the call site.
        let _ = (has_dp4a, cc);
        Q4kVariant::Mwv
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

#[cfg(test)]
mod pmat806_q4k_variant_tests {
    use super::{GpuProfile, Q4kVariant};

    /// PMAT-806: Blackwell (cc≥120, e.g. GB10 sm_121=121) MUST default Q4K to
    /// fp32 MWV — INT8 DP4A activation quant mis-estimates massive-activation
    /// channels and FAILs the CPU/GPU parity gate on quantized models.
    #[test]
    fn blackwell_defaults_to_fp32_mwv() {
        assert_eq!(
            GpuProfile::auto_q4k(true, 121),
            Q4kVariant::Mwv,
            "GB10 sm_121"
        );
        assert_eq!(
            GpuProfile::auto_q4k(true, 120),
            Q4kVariant::Mwv,
            "cc==120 boundary"
        );
    }

    /// FALSIFY-Q4K-ADA-PARITY-001 — SUPERSEDES the previous
    /// `discrete_dp4a_gpus_keep_hwdp4a` assertion.
    ///
    /// That test encoded PMAT-806's claim that discrete DP4A GPUs "keep the fast
    /// HwDp4a path — their DP4A activation quant is reliable for these models".
    /// That claim was MEASURED FALSE on 2026-07-27, on an RTX 4090 (sm_89,
    /// driver 570.207), running qwen2.5-coder-1.5B Q4_K_M — the exact model it
    /// named:
    ///
    ///   HwDp4a: GPU diverges from CPU at real position 1
    ///           (argmax 198 != 40, cosine 0.9186) -> F2 REJECTS
    ///   MWV:    all 42 real positions match, min cosine 0.9937 -> ACCEPTED
    ///
    /// 0.9186 is under the F2 floor (0.95) and under the 0.9817 PMAT-806 itself
    /// recorded as failing on Blackwell — so compute capability was never the
    /// real discriminator; the INT8 Q8_1 activation quant is.
    ///
    /// Consequence of the old default: the fail-closed F2 gate rejects CUDA, the
    /// run falls to wgpu (which fails its own 0.99 parity gate), then to CPU —
    /// ~20 tok/s instead of ~400 on the most common discrete GPU there is.
    #[test]
    fn discrete_dp4a_gpus_default_to_mwv_not_hwdp4a() {
        for (cc, name) in [
            (89u32, "RTX 4090 sm_89"),
            (80, "A100 sm_80"),
            (75, "Turing sm_75"),
        ] {
            assert_eq!(
                GpuProfile::auto_q4k(true, cc),
                Q4kVariant::Mwv,
                "FALSIFY-Q4K-ADA-PARITY-001: {name} must default to fp32 MWV. HwDp4a \
                 measured cosine 0.9186 vs CPU on sm_89 (F2 floor 0.95), so the gate \
                 rejects it and decode silently degrades to CPU."
            );
        }
    }

    /// No GPU may default to the degraded path, at any compute capability.
    /// Guards against a future carve-out reintroducing HwDp4a as a default;
    /// it stays reachable only via explicit `HW_DP4A_Q4K=1` opt-in.
    #[test]
    fn no_compute_capability_defaults_to_hwdp4a() {
        for cc in [0u32, 60, 70, 75, 80, 86, 89, 90, 100, 119, 120, 121, 130] {
            for has_dp4a in [false, true] {
                assert_ne!(
                    GpuProfile::auto_q4k(has_dp4a, cc),
                    Q4kVariant::HwDp4a,
                    "cc={cc} has_dp4a={has_dp4a} must not DEFAULT to the degraded \
                     HwDp4a path (opt-in via HW_DP4A_Q4K only)"
                );
            }
        }
    }

    /// Non-DP4A GPUs (sm<7.5) keep MWV (pre-existing behavior, unchanged).
    #[test]
    fn non_dp4a_gpus_use_mwv() {
        assert_eq!(GpuProfile::auto_q4k(false, 70), Q4kVariant::Mwv);
        assert_eq!(GpuProfile::auto_q4k(false, 60), Q4kVariant::Mwv);
    }
}
