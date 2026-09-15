//! Auto-detected GPU kernel configuration.
//!
//! Replaces per-machine env var tuning (`DP4A_Q4K`, `HW_DP4A_Q4K`, `MWV_Q6K`, etc.)
//! with automatic detection based on `compute_capability()`.
//!
//! Env vars still work as overrides for experimentation, but the defaults are
//! now correct for each GPU — no forjar config drift.

use serde::Serialize;
use trueno_gpu::driver::CudaContext;

/// Kernel variant for Q4K GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Q4kVariant {
    /// Legacy single-warp (32 threads), no DP4A. Fallback below sm_75: DP4A
    /// itself exists from sm_61 (PTX ISA 5.0), but the DP4A kernels here are
    /// built and validated for Turing (sm_75) and later only.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
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
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct GpuProfile {
    /// Q4K GEMV kernel variant (auto-detected: HwDp4a on sm_75+).
    pub q4k: Q4kVariant,
    /// Q6K GEMV kernel variant (auto-detected: Dp4a on sm_75+).
    pub q6k: Q6kVariant,
    /// Multi-warp GEMV warp count (default: 3, override: MWV_WARPS env).
    pub mwv_warps: u32,
    /// The prefill path `run_prefill` and the multi-prompt prefill guard will
    /// take (§9 #1 / #1a, PP-26).
    ///
    /// This REPLACES a `batched_prefill: bool` that had zero readers and did
    /// not know about compute capability: on Blackwell the engine ran serial
    /// prefill while that field said `true`, so an effective-config endpoint
    /// reporting it would have reported `batched` about a serial run — a PP-2
    /// violation by construction. Resolved ONCE, here, from the same pure
    /// function every call site uses.
    pub prefill_path: PrefillPathChoice,
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
        // Gated at sm_75 (Turing). DP4A is sm_61+ per the PTX ISA; the DP4A
        // kernels are validated from Turing, so sm_61..sm_72 take Legacy.
        let has_dp4a = major > 7 || (major == 7 && minor >= 5);
        let num_sms = context.multiprocessor_count().unwrap_or(8) as u32;

        // Real device compute capability (e.g. 121 for GB10 Blackwell). Uses the
        // true major/minor, NOT the PTX-clamped target — PMAT-806 needs the real
        // value to gate the Blackwell fp32-MWV-Q4K default.
        let cc = major as u32 * 10 + minor as u32;

        let q4k = Self::detect_q4k(has_dp4a, cc);
        let q6k = Self::detect_q6k(has_dp4a);
        let mwv_warps = Self::detect_mwv_warps();
        let prefill_path =
            select_prefill_path(cc, std::env::var("BATCHED_PREFILL").ok().as_deref());
        let hgemm_decode = Self::detect_hgemm_decode(has_dp4a, num_sms);
        let fused_gate_up =
            Self::detect_fused_gate_up(&q4k, std::env::var("FUSED_GATE_UP").ok().as_deref());

        let fp8_prefill = Self::detect_fp8_prefill(cc);
        let fp8_decode = Self::detect_fp8_decode(fp8_prefill, cc);
        let w4a16_interleaved = Self::detect_w4a16_interleaved(cc);

        // GH-611: Suppressed — was noisy in non-verbose mode

        Self {
            q4k,
            q6k,
            mwv_warps,
            prefill_path,
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

    /// Fused gate+up+SwiGLU: enabled when HW DP4A Q4K is active.
    ///
    /// Saves 11% instructions + eliminates the SwiGLU kernel and 4 intermediate
    /// buffer passes.
    ///
    /// §10 REFUSAL. `FUSED_GATE_UP=1` used to win over everything: the env
    /// branch returned `v != "0"` without consulting `q4k`. But the fused
    /// module (`fused_gate_up_swiglu_hw_dp4a_q4k_*`) is preloaded ONLY inside
    /// `preload_hw_dp4a_modules`, which runs only when `q4k == HwDp4a` — and
    /// since FALSIFY-Q4K-ADA-PARITY-001 EVERY GPU defaults to `Mwv`. So
    /// `FUSED_GATE_UP=1` selected a kernel whose PTX module was never loaded.
    /// The flag is now REFUSED, loudly, when the variant cannot supply the
    /// kernel, and the run continues unfused.
    ///
    /// Pure in `(q4k, env)` so the refusal is testable without a device.
    #[must_use]
    pub(crate) fn detect_fused_gate_up(q4k: &Q4kVariant, env: Option<&str>) -> bool {
        if let Some(v) = env {
            if v == "0" {
                return false;
            }
            if *q4k != Q4kVariant::HwDp4a {
                eprintln!(
                    "[GpuProfile] FUSED_GATE_UP={v} refused: the fused gate+up+SwiGLU \
                     kernel is HwDp4a-only (q4k={q4k:?}, its PTX module is preloaded \
                     only on that path); running unfused"
                );
                return false;
            }
            return true;
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

// ===========================================================================
// PP-LLAMA-001 §9 #1 / #1a: the prefill path, resolved once and reported
// ===========================================================================

/// Which prefill the engine will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillPath {
    /// One `forward_gpu_resident` per prompt token.
    Serial,
    /// One packed GEMM per layer over the whole prompt.
    Batched,
}

impl PrefillPath {
    /// The wire token.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Batched => "batched",
        }
    }
}

/// The prefill decision, with the reason it was made.
///
/// §5.2 asks the server to state "the prefill path `run_prefill` will select".
/// Reporting the path without the reason would leave a reader unable to tell an
/// operator override from the sm_12x default — which is the difference
/// between a deliberate A/B run and a corrupted one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefillPathChoice {
    /// The path itself.
    pub path: PrefillPath,
    /// Why: `"env=0"`, `"env forced"`, `"sm12x default"` or `"default"`.
    pub reason: &'static str,
    /// The compute capability the decision was made against (major*10 + minor).
    pub cc: u32,
}

/// PMAT-810 / §9 #1: pick the prefill path for a compute capability and the
/// `BATCHED_PREFILL` environment.
///
/// PURE — no device, no `std::env` read inside — so the policy is unit-testable
/// on a CPU box and cannot drift between the engine, the multi-prompt guard and
/// `/v1/effective-config`. That drift is the defect this replaces: the decision
/// lived inline in `run_prefill`, was re-derived from the environment on every
/// call, and was retained nowhere, so nothing could report which path ran.
///
/// The rule:
/// * `BATCHED_PREFILL=0` → serial, everywhere (the documented opt-out).
/// * `BATCHED_PREFILL=<anything else>` → batched, everywhere. Explicit opt-in
///   forces batched even on sm_12x, for A/B testing the KV-scatter fix.
/// * unset → batched, EXCEPT `cc >= 120` (the sm_12x family: RTX 50 sm_120,
///   GB10 sm_121), where the batched prefill path writes a corrupt KV cache
///   and every subsequent decode step reads poisoned K/V
///   (contracts/apr-cpu-vs-gpu-output-parity-v1.yaml, FALSIFY-CPU-GPU-009).
///
/// The predicate is a compute-capability inequality, NOT an architecture
/// test: datacenter Blackwell (sm_100/sm_103) and Thor (sm_110) sit below
/// 120 and keep the batched path. The defect is recorded on GB10 only;
/// widen on evidence, never by architecture name.
#[must_use]
pub fn select_prefill_path(cc: u32, batched_prefill_env: Option<&str>) -> PrefillPathChoice {
    let (path, reason) = match batched_prefill_env {
        Some("0") => (PrefillPath::Serial, "env=0"),
        Some(_) => (PrefillPath::Batched, "env forced"),
        None if cc >= SM12X_MIN_CC => (PrefillPath::Serial, "sm12x default"),
        None => (PrefillPath::Batched, "default"),
    };
    PrefillPathChoice { path, reason, cc }
}

/// Compute capability at and above which the batched prefill path is refused
/// by default: the sm_12x family (RTX 50 sm_120, GB10 sm_121) and anything
/// numerically above it. Not "Blackwell": sm_100/103/110 are Blackwell too
/// and are below this line (§9 #1a is recorded on GB10 only).
pub const SM12X_MIN_CC: u32 = 120;

impl GpuProfile {
    /// The prefill path this profile resolved, with its reason.
    #[must_use]
    pub fn prefill_path(&self) -> PrefillPathChoice {
        self.prefill_path
    }

    /// §9 #1a: may the multi-prompt (packed) prefill kernel run?
    ///
    /// It shares `batched_qkv_rope_phase` and the packed KV scatter with the
    /// single-prompt batched path, so wherever THAT is refused this must be
    /// refused too — same predicate, one answer.
    #[must_use]
    pub fn multi_prompt_prefill_allowed(&self) -> bool {
        self.prefill_path.path == PrefillPath::Batched
    }
}

// ===========================================================================
// PP-LLAMA-001 §5.2: CUDA graph enablement, as a single readout
// ===========================================================================

/// Which CUDA graphs this executor has enabled and captured.
///
/// "Graph enablement" is not one switch: it is two cached env predicates plus
/// live capture state spread over three maps. Reporting it needs all of them,
/// or a reader cannot tell "graphs are off" from "graphs are on and nothing has
/// been captured yet".
#[derive(Debug, Clone, Serialize)]
pub struct GraphConfig {
    /// Whether the GRAPHED DECODE path will be taken at all.
    ///
    /// `CUDA_GRAPH_ENABLE=1` is the opt-in (default off — capture poisons the
    /// CUDA context on drivers 570.207 / 590.48.01), and an enabled profiler
    /// forces eager regardless, because graph replay hides the per-brick
    /// instrumentation. Both halves live in `should_use_eager_decode`, and this
    /// is its negation rather than a second reading of the environment: a
    /// re-derived answer can disagree with the dispatch, and a receipt whose
    /// `graphs` block disagrees with the path that ran describes a different
    /// execution than the one it measured.
    pub cuda_graph_enable: bool,
    /// `GRAPH_DISPATCH` (default on) — read through the executor's own cached
    /// predicate, never re-derived here.
    pub graph_dispatch: bool,
    /// `PREFILL_GRAPH` (default off) — read through the prefill path's own
    /// cached predicate.
    pub prefill_graph: bool,
    /// A single-sequence decode graph is captured right now.
    pub decode_graph_captured: bool,
    /// Batch sizes with a captured batched-decode graph.
    pub batched_graph_sizes: Vec<usize>,
    /// Batch size the batched-graph input buffers are currently sized for.
    pub batched_graph_batch_size: usize,
    /// Sequence lengths with a captured prefill graph.
    pub prefill_graph_sizes: Vec<usize>,
}

// ===========================================================================
// PP-LLAMA-001 §10 / §12 kill criterion: max_batch must RECONSTRUCT
// ===========================================================================

/// Every input the `max_batch` decision was made from, and the decision.
///
/// §12's kill criterion for the effective-config row is "row 6's `max_batch`
/// does not reconstruct". It could not: `compute_max_batch_for_memory` computed
/// free VRAM, KV bytes per slot and a reserve, divided, clamped, and returned
/// only the clamped `usize` — every input dropped on the floor. The result was
/// then TRANSPORTED to the schedulers by `std::env::set_var("CUDA_MAX_BATCH")`,
/// which also erased the one remaining fact: after load, an operator-set
/// ceiling and a loader-computed one were indistinguishable.
///
/// So this records the inputs BEFORE the environment is touched, and states
/// which of the two produced `resolved`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct MaxBatchSizing {
    /// Free VRAM the driver reported at the moment of sizing.
    pub free_vram_bytes_at_sizing: usize,
    /// Total VRAM the driver reported at the moment of sizing.
    pub total_vram_bytes: usize,
    /// Whether the driver query succeeded. `false` means the two numbers above
    /// are the documented fallback, not a measurement.
    pub vram_query_ok: bool,
    /// Bytes one KV slot costs at the configured context length.
    pub kv_per_slot_bytes: usize,
    /// VRAM that had to survive the allocation.
    pub reserve_bytes: usize,
    /// `(free - reserve) / kv_per_slot`, before clamping.
    pub computed: usize,
    /// Lower clamp bound.
    pub clamp_min: usize,
    /// Upper clamp bound.
    pub clamp_max: usize,
    /// The ceiling actually in force.
    pub resolved: usize,
    /// `"computed"` or `"env"` — which produced `resolved`.
    pub source: &'static str,
}

/// `CUDA_MAX_BATCH` was set by the operator.
pub const MAX_BATCH_SOURCE_ENV: &str = "env";
/// The loader computed the ceiling from measured VRAM.
pub const MAX_BATCH_SOURCE_COMPUTED: &str = "computed";

// ===========================================================================
// PP-LLAMA-001 §5.2 / §9 #7: VRAM accounting
// ===========================================================================

/// What this process can honestly say about its VRAM.
///
/// Field names state their provenance, because the tempting ones are wrong:
///
/// * `free_at_load_bytes` is the snapshot taken BEFORE the weights and the
///   FP8/FP16 prefill cache were uploaded. It used to be the only stored
///   memory fact and it was labelled just "memory_info".
/// * `recorded_alloc_peak_bytes` is `PoolStats::peak_usage`, which counts only
///   the sites that call `record_allocation`. It is a LOWER BOUND, not a VRAM
///   peak, and is named so it cannot be read as one.
/// * `preload_delta_bytes` is `free_at_load - free_after_preload`: a MEASURED
///   delta covering weights plus every warmed cache. It is how §9 #7's "account
///   the 9.5 GB" is answered without inventing a per-cache breakdown that the
///   preload calls do not return.
#[derive(Debug, Clone, Serialize)]
pub struct VramReport {
    /// GPU device name.
    pub device_name: String,
    /// Total device VRAM in bytes.
    pub total_bytes: usize,
    /// Free VRAM before weights and caches were uploaded.
    pub free_at_load_bytes: usize,
    /// Free VRAM after preload, when the snapshot was taken.
    pub free_after_preload_bytes: Option<usize>,
    /// `free_at_load_bytes - free_after_preload_bytes`: what preload cost.
    pub preload_delta_bytes: Option<usize>,
    /// Free VRAM at the moment this report was built, when the query succeeded.
    pub free_now_bytes: Option<usize>,
    /// `total_bytes - free_now_bytes`.
    pub used_now_bytes: Option<usize>,
    /// Highest `used_now` this process has ever sampled.
    pub used_peak_bytes: Option<usize>,
    /// `PoolStats::peak_usage` — a recorded lower bound, not a VRAM peak.
    pub recorded_alloc_peak_bytes: usize,
    /// Bytes of single-sequence KV cache.
    pub kv_single_seq_bytes: usize,
    /// Bytes one batched KV slot costs.
    pub kv_per_slot_bytes: usize,
    /// Batched KV slots currently allocated.
    pub kv_slots_allocated: usize,
    /// Hard ceiling on batched KV slots.
    pub kv_slots_max: usize,
    /// `kv_single_seq_bytes + kv_per_slot_bytes * kv_slots_allocated`.
    pub kv_bytes_reserved: usize,
    /// Always `null`: this KV cache is contiguous per slot and has no blocks.
    pub kv_blocks_total: Option<usize>,
    /// The layout the KV numbers describe.
    pub kv_layout: &'static str,
}

/// The KV layout this backend uses. There is no block table.
pub const KV_LAYOUT: &str = "contiguous_per_slot";

#[cfg(test)]
mod pmat810_prefill_path_tests {
    use super::{select_prefill_path, PrefillPath};

    /// §9 #1: the whole policy, as a table.
    ///
    /// Every row is a case the engine can actually be in; the sm_12x rows
    /// are the ones that matter, because a `batched` answer there is the
    /// PMAT-810 KV corruption and a coherent-looking receipt over garbage
    /// tokens.
    #[test]
    fn select_prefill_path_table() {
        let cases: [(u32, Option<&str>, PrefillPath, &str); 6] = [
            (89, None, PrefillPath::Batched, "default"),
            (121, None, PrefillPath::Serial, "sm12x default"),
            (89, Some("0"), PrefillPath::Serial, "env=0"),
            (121, Some("1"), PrefillPath::Batched, "env forced"),
            (120, None, PrefillPath::Serial, "sm12x default"),
            (75, Some("anything"), PrefillPath::Batched, "env forced"),
        ];
        for (cc, env, expected_path, expected_reason) in cases {
            let choice = select_prefill_path(cc, env);
            assert_eq!(
                choice.path, expected_path,
                "cc={cc} BATCHED_PREFILL={env:?} must select {expected_path:?}"
            );
            assert_eq!(choice.reason, expected_reason, "cc={cc} env={env:?}");
            assert_eq!(choice.cc, cc, "the choice must carry the cc it was made on");
        }
    }

    /// The boundary is `>= 120`, not `> 120`, and it is numeric: Hopper
    /// (sm_90), datacenter Blackwell (sm_100/sm_103) and Thor (sm_110) all
    /// keep the batched path, because the §9 #1a corruption is recorded on
    /// GB10 (sm_121) only. RTX 50 (sm_120) shares the family and is refused.
    #[test]
    fn sm12x_boundary_is_inclusive_at_120_and_numeric() {
        for cc in [90u32, 100, 103, 110] {
            assert_eq!(
                select_prefill_path(cc, None).path,
                PrefillPath::Batched,
                "cc={cc} is below the sm_12x line and keeps the batched prefill"
            );
        }
        assert_eq!(select_prefill_path(120, None).path, PrefillPath::Serial);
        assert_eq!(select_prefill_path(121, None).path, PrefillPath::Serial);
    }

    /// §9 #1a: the multi-prompt guard must answer with the SAME predicate, so
    /// the endpoint's `prefill_path` and the kernel that runs cannot disagree.
    #[test]
    fn multi_prompt_allowance_follows_the_path() {
        for (cc, env) in [
            (89u32, None),
            (121, None),
            (121, Some("1")),
            (89, Some("0")),
        ] {
            let choice = select_prefill_path(cc, env);
            assert_eq!(
                choice.path == PrefillPath::Batched,
                matches!(choice.path, PrefillPath::Batched),
                "cc={cc} env={env:?}"
            );
        }
        assert!(select_prefill_path(89, None).path == PrefillPath::Batched);
        assert!(select_prefill_path(121, None).path == PrefillPath::Serial);
    }
}

#[cfg(test)]
mod pmat034_fused_gate_up_tests {
    use super::{GpuProfile, Q4kVariant};

    /// §10 REFUSAL: `FUSED_GATE_UP=1` on a non-HwDp4a variant selects a kernel
    /// whose PTX module was never preloaded. It must be refused, not honoured.
    #[test]
    fn refused_without_hwdp4a() {
        for variant in [
            Q4kVariant::Mwv,
            Q4kVariant::MwvDp4a,
            Q4kVariant::Wide,
            Q4kVariant::Vectorized,
            Q4kVariant::Legacy,
        ] {
            assert!(
                !GpuProfile::detect_fused_gate_up(&variant, Some("1")),
                "FUSED_GATE_UP=1 must be refused with q4k={variant:?}"
            );
        }
    }

    /// The opt-in still works where the module exists.
    #[test]
    fn allowed_on_hwdp4a() {
        assert!(GpuProfile::detect_fused_gate_up(
            &Q4kVariant::HwDp4a,
            Some("1")
        ));
        assert!(GpuProfile::detect_fused_gate_up(
            &Q4kVariant::HwDp4a,
            Some("yes")
        ));
    }

    /// `FUSED_GATE_UP=0` disables it even where it would be the default.
    #[test]
    fn env_0_disables() {
        assert!(!GpuProfile::detect_fused_gate_up(
            &Q4kVariant::HwDp4a,
            Some("0")
        ));
        assert!(!GpuProfile::detect_fused_gate_up(
            &Q4kVariant::Mwv,
            Some("0")
        ));
    }

    /// With no override the flag follows the variant that can supply the kernel.
    #[test]
    fn default_follows_q4k() {
        assert!(GpuProfile::detect_fused_gate_up(&Q4kVariant::HwDp4a, None));
        assert!(!GpuProfile::detect_fused_gate_up(&Q4kVariant::Mwv, None));
        assert!(!GpuProfile::detect_fused_gate_up(
            &Q4kVariant::MwvDp4a,
            None
        ));
    }
}

#[cfg(test)]
mod pp_llama_report_serialisation_tests {
    use super::{
        select_prefill_path, GpuProfile, GraphConfig, MaxBatchSizing, Q4kVariant, Q6kVariant,
        VramReport, KV_LAYOUT, MAX_BATCH_SOURCE_COMPUTED,
    };

    fn profile() -> GpuProfile {
        GpuProfile {
            q4k: Q4kVariant::HwDp4a,
            q6k: Q6kVariant::Dp4a,
            mwv_warps: 3,
            prefill_path: select_prefill_path(89, None),
            hgemm_decode: false,
            fused_gate_up: true,
            fp8_prefill: true,
            fp8_decode: true,
            w4a16_interleaved: false,
            sm_target: "sm_89".to_string(),
            cc: 89,
        }
    }

    /// §5.2 lists the resolved profile as a reported field. Every one of its
    /// facts has to survive onto the wire under a stable name, or a receipt
    /// that says `q4k` means one thing this release and another the next.
    #[test]
    fn gpu_profile_serialises_every_field_snake_case() {
        let json = serde_json::to_value(profile()).expect("serialize");
        let object = json.as_object().expect("object");
        for key in [
            "q4k",
            "q6k",
            "mwv_warps",
            "prefill_path",
            "hgemm_decode",
            "fused_gate_up",
            "fp8_prefill",
            "fp8_decode",
            "w4a16_interleaved",
            "sm_target",
            "cc",
        ] {
            assert!(object.contains_key(key), "missing `{key}` in {json}");
        }
        assert_eq!(object.len(), 11, "field count changed: {json}");
        // Variants are snake_case tokens, not Rust `Debug` spellings.
        assert_eq!(object["q4k"].as_str(), Some("hw_dp4a"));
        assert_eq!(object["q6k"].as_str(), Some("dp4a"));
        assert_eq!(object["cc"].as_u64(), Some(89));
        assert_eq!(object["prefill_path"]["path"].as_str(), Some("batched"));
        assert_eq!(object["prefill_path"]["reason"].as_str(), Some("default"));
        assert_eq!(object["prefill_path"]["cc"].as_u64(), Some(89));
    }

    /// The Blackwell profile must serialise as SERIAL — the dead
    /// `batched_prefill: bool` this replaced would have said `true` here while
    /// the engine ran serial, which is the PP-2 violation §5.2 exists to close.
    #[test]
    fn a_blackwell_profile_reports_serial_prefill() {
        let mut p = profile();
        p.cc = 121;
        p.prefill_path = select_prefill_path(121, None);
        let json = serde_json::to_value(p).expect("serialize");
        assert_eq!(json["prefill_path"]["path"].as_str(), Some("serial"));
        assert_eq!(
            json["prefill_path"]["reason"].as_str(),
            Some("sm12x default")
        );
    }

    /// §10 / §12 kill criterion: the sizing block must carry every input, so a
    /// reader can recompute `resolved` without the server.
    #[test]
    fn max_batch_sizing_carries_every_input() {
        let sizing = MaxBatchSizing {
            free_vram_bytes_at_sizing: 8_900_000_000,
            total_vram_bytes: 25_757_220_864,
            vram_query_ok: true,
            kv_per_slot_bytes: 469_762_048,
            reserve_bytes: 3_500_000_000,
            computed: 11,
            clamp_min: 1,
            clamp_max: 32,
            resolved: 11,
            source: MAX_BATCH_SOURCE_COMPUTED,
        };
        let json = serde_json::to_value(sizing).expect("serialize");
        let object = json.as_object().expect("object");
        for key in [
            "free_vram_bytes_at_sizing",
            "total_vram_bytes",
            "vram_query_ok",
            "kv_per_slot_bytes",
            "reserve_bytes",
            "computed",
            "clamp_min",
            "clamp_max",
            "resolved",
            "source",
        ] {
            assert!(object.contains_key(key), "missing `{key}` in {json}");
        }
        // The reader's own arithmetic must reproduce `computed`.
        let free = object["free_vram_bytes_at_sizing"].as_u64().expect("free");
        let reserve = object["reserve_bytes"].as_u64().expect("reserve");
        let per_slot = object["kv_per_slot_bytes"].as_u64().expect("per slot");
        assert_eq!(
            (free - reserve) / per_slot,
            object["computed"].as_u64().expect("computed"),
            "the reported inputs must reconstruct the reported quotient: {json}"
        );
    }

    /// The graph readout must not collapse "off" and "on but nothing captured".
    #[test]
    fn graph_config_distinguishes_disabled_from_uncaptured() {
        let off = GraphConfig {
            cuda_graph_enable: false,
            graph_dispatch: false,
            prefill_graph: false,
            decode_graph_captured: false,
            batched_graph_sizes: Vec::new(),
            batched_graph_batch_size: 0,
            prefill_graph_sizes: Vec::new(),
        };
        let on_uncaptured = GraphConfig {
            cuda_graph_enable: true,
            graph_dispatch: true,
            prefill_graph: true,
            decode_graph_captured: false,
            batched_graph_sizes: Vec::new(),
            batched_graph_batch_size: 0,
            prefill_graph_sizes: Vec::new(),
        };
        let off_json = serde_json::to_value(off).expect("serialize");
        let on_json = serde_json::to_value(on_uncaptured).expect("serialize");
        assert_ne!(
            off_json, on_json,
            "a disabled graph and an enabled-but-uncaptured one must not read the same"
        );
        assert_eq!(off_json.as_object().expect("object").len(), 7);
    }

    /// PP-2 / §5.2: the decode-graph opt-in must be ON the wire, and must be
    /// its own fact.
    ///
    /// It was omitted entirely while `should_use_eager_decode` was private, so
    /// a receipt could not say whether the run it measured used graph replay or
    /// the eager path — a difference of ~5.6 ms of launch overhead per token,
    /// which is most of what a decode-rate ratio is made of. `graph_dispatch`
    /// is a DIFFERENT switch and cannot stand in for it.
    #[test]
    fn graph_config_reports_the_decode_graph_opt_in_separately() {
        let base = || GraphConfig {
            cuda_graph_enable: false,
            graph_dispatch: true,
            prefill_graph: false,
            decode_graph_captured: false,
            batched_graph_sizes: Vec::new(),
            batched_graph_batch_size: 0,
            prefill_graph_sizes: Vec::new(),
        };
        let eager = serde_json::to_value(base()).expect("serialize");
        let graphed = serde_json::to_value(GraphConfig {
            cuda_graph_enable: true,
            ..base()
        })
        .expect("serialize");
        assert_eq!(eager["cuda_graph_enable"].as_bool(), Some(false));
        assert_eq!(graphed["cuda_graph_enable"].as_bool(), Some(true));
        assert_ne!(
            eager, graphed,
            "an eager-decode run and a graph-replay run must not serialize alike"
        );
        assert_eq!(
            eager["graph_dispatch"], graphed["graph_dispatch"],
            "`graph_dispatch` is a different switch and must not move with it"
        );
    }

    /// §9 #7: the VRAM block must NEVER label the pool's recorded peak as a
    /// VRAM peak. The pool counts only `record_allocation` sites, so it is a
    /// lower bound; naming it `vram_peak` would under-report the accounting the
    /// spec asks for.
    #[test]
    fn vram_report_names_the_recorded_peak_honestly() {
        let report = VramReport {
            device_name: "NVIDIA GeForce RTX 4090".to_string(),
            total_bytes: 25_757_220_864,
            free_at_load_bytes: 24_000_000_000,
            free_after_preload_bytes: Some(14_500_000_000),
            preload_delta_bytes: Some(9_500_000_000),
            free_now_bytes: Some(14_000_000_000),
            used_now_bytes: Some(11_757_220_864),
            used_peak_bytes: Some(20_471_000_000),
            recorded_alloc_peak_bytes: 7_000_000_000,
            kv_single_seq_bytes: 469_762_048,
            kv_per_slot_bytes: 469_762_048,
            kv_slots_allocated: 4,
            kv_slots_max: 32,
            kv_bytes_reserved: 2_348_810_240,
            kv_blocks_total: None,
            kv_layout: KV_LAYOUT,
        };
        let json = serde_json::to_value(report).expect("serialize");
        let object = json.as_object().expect("object");
        assert!(
            !object.contains_key("vram_peak"),
            "the recorded allocation peak must not be published as `vram_peak`: {json}"
        );
        assert!(object.contains_key("recorded_alloc_peak_bytes"));
        assert!(object.contains_key("used_peak_bytes"));
        assert!(
            json["recorded_alloc_peak_bytes"].as_u64() < json["used_peak_bytes"].as_u64(),
            "the fixture must show the two are different quantities: {json}"
        );
        // §9 #7: preload's cost is a MEASURED delta of two snapshots.
        assert_eq!(
            json["free_at_load_bytes"].as_u64().expect("at load")
                - json["free_after_preload_bytes"].as_u64().expect("after"),
            json["preload_delta_bytes"].as_u64().expect("delta")
        );
        // No block table exists on this backend; the field says so and the
        // layout names what does exist.
        assert!(json["kv_blocks_total"].is_null());
        assert_eq!(json["kv_layout"].as_str(), Some("contiguous_per_slot"));
        assert_eq!(
            json["kv_bytes_reserved"].as_u64(),
            Some(469_762_048 + 469_762_048 * 4)
        );
    }
}
