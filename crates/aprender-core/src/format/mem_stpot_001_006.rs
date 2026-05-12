// Bundles two sister contracts in one verdict module:
//
//   `memory-safety-v1` (FALSIFY-MEM-001..003)
//   `streaming-tpot-v1` (FALSIFY-STPOT-001..003)
//
// Both are 3-gate sister contracts whose verdicts share no internal
// structure but appear in the same coverage band.
//
// MEM-001: allocated_bytes(t) == product(t.shape) * dtype_size(t.dtype)
// MEM-002: out-of-bounds index returns Err(IndexOutOfBounds), not panic
// MEM-003: Tensor::zeros bytes are 0u8 across the buffer
// STPOT-001: TPOT P50 > 0 (server actually streamed)
// STPOT-002: TTFT P50 < latency P50 * MAX_TTFT_FRACTION (default 0.5)
// STPOT-003: concat(streaming_tokens) == non_streaming_response.content

/// Module name was disambiguated (FALSIFY-ST-* collides with
/// safetensors-format-safety) — we use STPOT for the streaming gates
/// and MEM for the memory-safety gates.
pub const AC_MEM_DTYPE_F32_SIZE: usize = 4;
pub const AC_MEM_DTYPE_F16_SIZE: usize = 2;
pub const AC_MEM_DTYPE_U8_SIZE: usize = 1;

/// STPOT-002 default: TTFT must be a small fraction of total latency
/// to prove streaming is active. Per contract `TTFT/latency < 0.95`
/// is the relaxed gate; `< 0.50` is the per-prediction stricter check.
pub const AC_STPOT_MAX_TTFT_FRACTION: f32 = 0.50;
/// Per spec: `TTFT/latency < 0.95` is the strictly-streaming bound.
pub const AC_STPOT_PROOF_OF_STREAMING_BOUND: f32 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemStpotVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy)]
pub enum AccessOutcome {
    Ok,
    ErrIndexOutOfBounds,
    Panic,
}

// ----------------------------------------------------------------
// MEM-001
// ----------------------------------------------------------------

/// MEM-001: allocated bytes match `product(shape) * dtype_size`.
///
/// Pass iff `observed_bytes == product(shape) * dtype_size` AND
/// shape is non-empty AND dtype_size > 0.
#[must_use]
pub fn verdict_from_allocation_bounds(
    shape: &[usize],
    dtype_size: usize,
    observed_bytes: usize,
) -> MemStpotVerdict {
    if shape.is_empty() || dtype_size == 0 {
        return MemStpotVerdict::Fail;
    }
    let mut prod: usize = 1;
    for &d in shape {
        if d == 0 {
            return MemStpotVerdict::Fail;
        }
        let Some(p) = prod.checked_mul(d) else {
            return MemStpotVerdict::Fail;
        };
        prod = p;
    }
    let Some(expected) = prod.checked_mul(dtype_size) else {
        return MemStpotVerdict::Fail;
    };
    if observed_bytes == expected {
        MemStpotVerdict::Pass
    } else {
        MemStpotVerdict::Fail
    }
}

// ----------------------------------------------------------------
// MEM-002
// ----------------------------------------------------------------

/// MEM-002: OOB index access returns `Err(IndexOutOfBounds)`, not Ok and not Panic.
#[must_use]
pub fn verdict_from_oob_returns_err(outcome: AccessOutcome) -> MemStpotVerdict {
    match outcome {
        AccessOutcome::ErrIndexOutOfBounds => MemStpotVerdict::Pass,
        AccessOutcome::Ok | AccessOutcome::Panic => MemStpotVerdict::Fail,
    }
}

// ----------------------------------------------------------------
// MEM-003
// ----------------------------------------------------------------

/// MEM-003: every byte of `Tensor::zeros(...)` buffer is 0u8.
#[must_use]
pub fn verdict_from_zero_init(buffer: &[u8]) -> MemStpotVerdict {
    if buffer.is_empty() {
        return MemStpotVerdict::Fail;
    }
    if buffer.iter().all(|&b| b == 0) {
        MemStpotVerdict::Pass
    } else {
        MemStpotVerdict::Fail
    }
}

// ----------------------------------------------------------------
// STPOT-001
// ----------------------------------------------------------------

/// STPOT-001: TPOT P50 > 0 — server actually streamed.
#[must_use]
pub fn verdict_from_tpot_positive(tpot_p50_ms: f32) -> MemStpotVerdict {
    if !tpot_p50_ms.is_finite() {
        return MemStpotVerdict::Fail;
    }
    if tpot_p50_ms > 0.0 {
        MemStpotVerdict::Pass
    } else {
        MemStpotVerdict::Fail
    }
}

// ----------------------------------------------------------------
// STPOT-002
// ----------------------------------------------------------------

/// STPOT-002: TTFT separable from total latency.
///
/// Pass iff:
/// - `latency_p50_ms > 0`
/// - `ttft_p50_ms / latency_p50_ms < AC_STPOT_MAX_TTFT_FRACTION` (default 0.5)
///
/// Per contract this confirms streaming is active for 128-token gen.
#[must_use]
pub fn verdict_from_ttft_separable(ttft_p50_ms: f32, latency_p50_ms: f32) -> MemStpotVerdict {
    if !ttft_p50_ms.is_finite() || !latency_p50_ms.is_finite() {
        return MemStpotVerdict::Fail;
    }
    if latency_p50_ms <= 0.0 || ttft_p50_ms < 0.0 {
        return MemStpotVerdict::Fail;
    }
    let ratio = ttft_p50_ms / latency_p50_ms;
    if ratio < AC_STPOT_MAX_TTFT_FRACTION {
        MemStpotVerdict::Pass
    } else {
        MemStpotVerdict::Fail
    }
}

// ----------------------------------------------------------------
// STPOT-003
// ----------------------------------------------------------------

/// STPOT-003: concat(streaming_tokens) == non_streaming_response.
#[must_use]
pub fn verdict_from_content_parity(
    streaming_tokens: &[&str],
    non_streaming_text: &str,
) -> MemStpotVerdict {
    if streaming_tokens.is_empty() && non_streaming_text.is_empty() {
        return MemStpotVerdict::Fail;
    }
    let concat: String = streaming_tokens.iter().copied().collect();
    if concat == non_streaming_text {
        MemStpotVerdict::Pass
    } else {
        MemStpotVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_dtype_sizes() {
        assert_eq!(AC_MEM_DTYPE_F32_SIZE, 4);
        assert_eq!(AC_MEM_DTYPE_F16_SIZE, 2);
        assert_eq!(AC_MEM_DTYPE_U8_SIZE, 1);
    }

    #[test]
    fn provenance_stpot_thresholds() {
        assert_eq!(AC_STPOT_MAX_TTFT_FRACTION, 0.50);
        assert_eq!(AC_STPOT_PROOF_OF_STREAMING_BOUND, 0.95);
    }

    // -----------------------------------------------------------------
    // Section 2: MEM-001 allocation bounds.
    // -----------------------------------------------------------------
    #[test]
    fn fmem001_pass_2d_f32() {
        // shape [4,4] F32 → 64 bytes
        let v = verdict_from_allocation_bounds(&[4, 4], AC_MEM_DTYPE_F32_SIZE, 64);
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fmem001_pass_3d_f16() {
        // shape [2,3,4] F16 → 48 bytes
        let v = verdict_from_allocation_bounds(&[2, 3, 4], AC_MEM_DTYPE_F16_SIZE, 48);
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fmem001_fail_under_allocated() {
        let v = verdict_from_allocation_bounds(&[4, 4], AC_MEM_DTYPE_F32_SIZE, 32);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fmem001_fail_over_allocated() {
        let v = verdict_from_allocation_bounds(&[4, 4], AC_MEM_DTYPE_F32_SIZE, 128);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fmem001_fail_zero_dim() {
        let v = verdict_from_allocation_bounds(&[4, 0, 4], AC_MEM_DTYPE_F32_SIZE, 0);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fmem001_fail_empty_shape() {
        let v = verdict_from_allocation_bounds(&[], AC_MEM_DTYPE_F32_SIZE, 4);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: MEM-002 OOB returns Err.
    // -----------------------------------------------------------------
    #[test]
    fn fmem002_pass_returns_index_out_of_bounds() {
        let v = verdict_from_oob_returns_err(AccessOutcome::ErrIndexOutOfBounds);
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fmem002_fail_panics() {
        let v = verdict_from_oob_returns_err(AccessOutcome::Panic);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fmem002_fail_silently_ok() {
        // The most dangerous regression — silent success on OOB
        // (UB territory; e.g., reading uninitialized memory).
        let v = verdict_from_oob_returns_err(AccessOutcome::Ok);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: MEM-003 zero-init guarantee.
    // -----------------------------------------------------------------
    #[test]
    fn fmem003_pass_all_zeros() {
        let buf = vec![0u8; 64];
        let v = verdict_from_zero_init(&buf);
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fmem003_fail_one_nonzero() {
        let mut buf = vec![0u8; 64];
        buf[33] = 1;
        let v = verdict_from_zero_init(&buf);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fmem003_fail_all_nonzero() {
        let buf = vec![0xFF_u8; 64];
        let v = verdict_from_zero_init(&buf);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fmem003_fail_empty_buffer() {
        let v = verdict_from_zero_init(&[]);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: STPOT-001..003.
    // -----------------------------------------------------------------
    #[test]
    fn fstpot001_pass_5ms_per_token() {
        let v = verdict_from_tpot_positive(5.0);
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fstpot001_fail_zero_tpot() {
        // The exact regression class: server didn't stream.
        let v = verdict_from_tpot_positive(0.0);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot001_fail_negative() {
        let v = verdict_from_tpot_positive(-1.0);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot001_fail_nan() {
        let v = verdict_from_tpot_positive(f32::NAN);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot002_pass_canonical_streaming() {
        // ttft 50ms / latency 1500ms = 0.033 < 0.5
        let v = verdict_from_ttft_separable(50.0, 1500.0);
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fstpot002_fail_non_streaming() {
        // Server collected the full response then returned: TTFT == latency
        let v = verdict_from_ttft_separable(1500.0, 1500.0);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot002_fail_at_threshold() {
        // ratio == 0.5 is Fail (strict <)
        let v = verdict_from_ttft_separable(750.0, 1500.0);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot002_fail_zero_latency() {
        let v = verdict_from_ttft_separable(0.0, 0.0);
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot003_pass_concat_matches() {
        let stream = ["Hello", ", ", "world", "!"];
        let v = verdict_from_content_parity(&stream, "Hello, world!");
        assert_eq!(v, MemStpotVerdict::Pass);
    }

    #[test]
    fn fstpot003_fail_drift() {
        let stream = ["Hello", " world"];
        let v = verdict_from_content_parity(&stream, "Hello world!");
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    #[test]
    fn fstpot003_fail_both_empty() {
        let v = verdict_from_content_parity(&[], "");
        assert_eq!(v, MemStpotVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_shape_byte_product() {
        for &(shape, ds) in &[
            ([1usize, 1, 1].as_slice(), 4),
            (&[2, 2], 4),
            (&[10, 20, 30], 2),
            (&[1024, 1024], 4),
        ] {
            let prod: usize = shape.iter().product();
            let bytes = prod * ds;
            let v = verdict_from_allocation_bounds(shape, ds, bytes);
            assert_eq!(v, MemStpotVerdict::Pass, "shape={shape:?} ds={ds}");
        }
    }

    #[test]
    fn mutation_survey_002_ttft_ratio_band() {
        // Sweep ratios 0.1..0.6 in 0.05 steps
        for ratio_x100 in [10_u32, 25, 49, 50, 51, 75] {
            let ratio = ratio_x100 as f32 / 100.0;
            let latency = 1000.0;
            let ttft = latency * ratio;
            let v = verdict_from_ttft_separable(ttft, latency);
            let expected = if ratio < AC_STPOT_MAX_TTFT_FRACTION {
                MemStpotVerdict::Pass
            } else {
                MemStpotVerdict::Fail
            };
            assert_eq!(v, expected, "ratio={ratio}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_6() {
        let v1 = verdict_from_allocation_bounds(&[4, 4], 4, 64);
        let v2 = verdict_from_oob_returns_err(AccessOutcome::ErrIndexOutOfBounds);
        let v3 = verdict_from_zero_init(&[0u8; 64]);
        let v4 = verdict_from_tpot_positive(5.0);
        let v5 = verdict_from_ttft_separable(50.0, 1500.0);
        let v6 = verdict_from_content_parity(&["Hello", ", ", "world", "!"], "Hello, world!");
        assert_eq!(v1, MemStpotVerdict::Pass);
        assert_eq!(v2, MemStpotVerdict::Pass);
        assert_eq!(v3, MemStpotVerdict::Pass);
        assert_eq!(v4, MemStpotVerdict::Pass);
        assert_eq!(v5, MemStpotVerdict::Pass);
        assert_eq!(v6, MemStpotVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // Regression class:
        //   1: shape says [4,4] but allocator gave 32 bytes (rounded)
        //   2: OOB silently returned Ok → UB
        //   3: zeros() left uninitialized (random byte sneaks through)
        //   4: TPOT 0.0 (server didn't actually stream)
        //   5: TTFT == latency (full buffering, no streaming)
        //   6: streaming dropped a token vs non-streaming
        let v1 = verdict_from_allocation_bounds(&[4, 4], 4, 32);
        let v2 = verdict_from_oob_returns_err(AccessOutcome::Ok);
        let mut buf = vec![0u8; 64];
        buf[7] = 0xCD;
        let v3 = verdict_from_zero_init(&buf);
        let v4 = verdict_from_tpot_positive(0.0);
        let v5 = verdict_from_ttft_separable(1500.0, 1500.0);
        let v6 = verdict_from_content_parity(&["Hello", " world"], "Hello, world!");
        assert_eq!(v1, MemStpotVerdict::Fail);
        assert_eq!(v2, MemStpotVerdict::Fail);
        assert_eq!(v3, MemStpotVerdict::Fail);
        assert_eq!(v4, MemStpotVerdict::Fail);
        assert_eq!(v5, MemStpotVerdict::Fail);
        assert_eq!(v6, MemStpotVerdict::Fail);
    }
}
