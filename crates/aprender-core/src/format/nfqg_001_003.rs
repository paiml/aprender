// SHIP-TWO-001 — `nf4-fused-qkv-gemm-v1` algorithm-level PARTIAL
// discharge for FALSIFY-NF4-QKV-001..003 (closes 3/3 sweep).
//
// Contract: `contracts/nf4-fused-qkv-gemm-v1.yaml`.
// Spec: Fused NF4 Q/K/V GEMM for GQA attention — entrenar QLoRA
// training kernel (PMAT-478; replicates QWEN-009 dual-output pattern
// for K+V projection).

// ===========================================================================
// NF4-QKV-001 — Fused K+V matches separate K, V projections within 1e-4
// ===========================================================================

pub const AC_NFQG_001_TOLERANCE: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfqg001Verdict { Pass, Fail }

/// Pass iff `fused_kv` (concatenated [k_out; v_out]) matches the
/// concatenation of `separate_k` and `separate_v` element-wise within
/// tolerance.
#[must_use]
pub fn verdict_from_fused_kv_equivalence(
    fused_kv: &[f32],
    separate_k: &[f32],
    separate_v: &[f32],
) -> Nfqg001Verdict {
    if fused_kv.is_empty() || separate_k.is_empty() || separate_v.is_empty() {
        return Nfqg001Verdict::Fail;
    }
    if fused_kv.len() != separate_k.len() + separate_v.len() {
        return Nfqg001Verdict::Fail;
    }
    let kv_split = separate_k.len();
    for (i, &f) in fused_kv.iter().enumerate() {
        if !f.is_finite() { return Nfqg001Verdict::Fail; }
        let s = if i < kv_split { separate_k[i] } else { separate_v[i - kv_split] };
        if !s.is_finite() { return Nfqg001Verdict::Fail; }
        if (f - s).abs() > AC_NFQG_001_TOLERANCE { return Nfqg001Verdict::Fail; }
    }
    Nfqg001Verdict::Pass
}

// ===========================================================================
// NF4-QKV-002 — DRAM read reduction (3 → 2) AND throughput ≥ 10% gain
// ===========================================================================

pub const AC_NFQG_002_FUSED_DRAM_READS: u64 = 2;
pub const AC_NFQG_002_SEPARATE_DRAM_READS: u64 = 3;
pub const AC_NFQG_002_MIN_GAIN: f32 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfqg002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_dram_and_throughput(
    fused_dram_reads: u64,
    separate_dram_reads: u64,
    fused_tps: f32,
    separate_tps: f32,
) -> Nfqg002Verdict {
    if fused_dram_reads != AC_NFQG_002_FUSED_DRAM_READS { return Nfqg002Verdict::Fail; }
    if separate_dram_reads != AC_NFQG_002_SEPARATE_DRAM_READS { return Nfqg002Verdict::Fail; }
    if !fused_tps.is_finite() || !separate_tps.is_finite() { return Nfqg002Verdict::Fail; }
    if fused_tps <= 0.0 || separate_tps <= 0.0 { return Nfqg002Verdict::Fail; }
    let gain = (fused_tps / separate_tps) - 1.0;
    if !gain.is_finite() { return Nfqg002Verdict::Fail; }
    if gain < AC_NFQG_002_MIN_GAIN { return Nfqg002Verdict::Fail; }
    Nfqg002Verdict::Pass
}

// ===========================================================================
// NF4-QKV-003 — K dim == V dim precondition (fused path rejects asymmetric)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nfqg003Verdict { Pass, Fail }

/// Pass iff `kv_dim_k == kv_dim_v AND both > 0`. The fused K+V path
/// requires shared output dim per the GQA contract.
#[must_use]
pub const fn verdict_from_kv_dim_match(kv_dim_k: u64, kv_dim_v: u64) -> Nfqg003Verdict {
    if kv_dim_k == 0 || kv_dim_v == 0 { return Nfqg003Verdict::Fail; }
    if kv_dim_k != kv_dim_v { return Nfqg003Verdict::Fail; }
    Nfqg003Verdict::Pass
}

// ===========================================================================
// Bandwidth helper (per contract formula): savings = M * K * 4 bytes
// ===========================================================================

#[must_use]
pub const fn qkv_bandwidth_savings_bytes(m: u64, k: u64) -> u64 {
    m.saturating_mul(k).saturating_mul(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    // NF4-QKV-001 (fused K+V equivalence)
    #[test] fn nfqg_001_pass_canonical() {
        // fused_kv = [k0, k1, v0, v1]; separate_k = [k0, k1]; separate_v = [v0, v1].
        let separate_k = vec![1.0_f32, 2.0];
        let separate_v = vec![3.0_f32, 4.0];
        let fused_kv = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            verdict_from_fused_kv_equivalence(&fused_kv, &separate_k, &separate_v),
            Nfqg001Verdict::Pass
        );
    }
    #[test] fn nfqg_001_pass_within_tolerance() {
        let k = vec![1.0_f32];
        let v = vec![2.0_f32];
        let fused = vec![1.0_f32 + 5e-5, 2.0_f32 - 5e-5]; // < 1e-4
        assert_eq!(
            verdict_from_fused_kv_equivalence(&fused, &k, &v),
            Nfqg001Verdict::Pass
        );
    }
    #[test] fn nfqg_001_fail_above_tolerance() {
        let k = vec![1.0_f32];
        let v = vec![2.0_f32];
        let fused = vec![1.001_f32, 2.0_f32];
        assert_eq!(
            verdict_from_fused_kv_equivalence(&fused, &k, &v),
            Nfqg001Verdict::Fail
        );
    }
    #[test] fn nfqg_001_fail_length_mismatch() {
        let k = vec![1.0_f32];
        let v = vec![2.0_f32];
        let fused = vec![1.0_f32, 2.0, 3.0]; // wrong size
        assert_eq!(
            verdict_from_fused_kv_equivalence(&fused, &k, &v),
            Nfqg001Verdict::Fail
        );
    }
    #[test] fn nfqg_001_fail_v_corrupted() {
        // Fused matches K cleanly but V slot drifts.
        let k = vec![1.0_f32];
        let v = vec![2.0_f32];
        let fused = vec![1.0_f32, 99.0]; // V slot corrupted
        assert_eq!(
            verdict_from_fused_kv_equivalence(&fused, &k, &v),
            Nfqg001Verdict::Fail
        );
    }
    #[test] fn nfqg_001_fail_nan() {
        let k = vec![f32::NAN];
        let v = vec![2.0_f32];
        let fused = vec![f32::NAN, 2.0];
        assert_eq!(
            verdict_from_fused_kv_equivalence(&fused, &k, &v),
            Nfqg001Verdict::Fail
        );
    }

    // NF4-QKV-002 (DRAM + throughput)
    #[test] fn nfqg_002_pass_canonical() {
        // 2 fused reads vs 3 separate reads, 15% throughput gain.
        assert_eq!(
            verdict_from_dram_and_throughput(2, 3, 1150.0, 1000.0),
            Nfqg002Verdict::Pass
        );
    }
    #[test] fn nfqg_002_fail_below_10_percent() {
        // 9% gain — below the 10% threshold.
        assert_eq!(
            verdict_from_dram_and_throughput(2, 3, 1090.0, 1000.0),
            Nfqg002Verdict::Fail
        );
    }
    #[test] fn nfqg_002_fail_wrong_fused_reads() {
        // Fused path read A from DRAM 3x — not actually fused.
        assert_eq!(
            verdict_from_dram_and_throughput(3, 3, 1150.0, 1000.0),
            Nfqg002Verdict::Fail
        );
    }
    #[test] fn nfqg_002_fail_wrong_separate_reads() {
        // The "separate" baseline reads only 2x — measurement is wrong.
        assert_eq!(
            verdict_from_dram_and_throughput(2, 2, 1150.0, 1000.0),
            Nfqg002Verdict::Fail
        );
    }
    #[test] fn nfqg_002_fail_zero_separate_tps() {
        assert_eq!(
            verdict_from_dram_and_throughput(2, 3, 1150.0, 0.0),
            Nfqg002Verdict::Fail
        );
    }
    #[test] fn nfqg_002_fail_nan() {
        assert_eq!(
            verdict_from_dram_and_throughput(2, 3, f32::NAN, 1000.0),
            Nfqg002Verdict::Fail
        );
    }
    #[test] fn nfqg_002_fail_regression() {
        assert_eq!(
            verdict_from_dram_and_throughput(2, 3, 800.0, 1000.0),
            Nfqg002Verdict::Fail
        );
    }

    // NF4-QKV-003 (kv_dim match)
    #[test] fn nfqg_003_pass_qwen_canonical() {
        // Qwen 1.5B kv_dim = 256 (both K and V).
        assert_eq!(verdict_from_kv_dim_match(256, 256), Nfqg003Verdict::Pass);
    }
    #[test] fn nfqg_003_pass_full_attention() {
        // Non-GQA: K and V match each other (and Q).
        assert_eq!(verdict_from_kv_dim_match(1536, 1536), Nfqg003Verdict::Pass);
    }
    #[test] fn nfqg_003_fail_asymmetric() {
        // The contract's stated falsifier: "Attempt fused K+V with
        // kv_dim_k=256 and kv_dim_v=128. Expect error, not silent
        // corruption." The verdict must Fail this case.
        assert_eq!(verdict_from_kv_dim_match(256, 128), Nfqg003Verdict::Fail);
    }
    #[test] fn nfqg_003_fail_zero_k() {
        assert_eq!(verdict_from_kv_dim_match(0, 256), Nfqg003Verdict::Fail);
    }
    #[test] fn nfqg_003_fail_zero_v() {
        assert_eq!(verdict_from_kv_dim_match(256, 0), Nfqg003Verdict::Fail);
    }
    #[test] fn nfqg_003_fail_swapped() {
        // Off-by-one or transposition on a single side.
        assert_eq!(verdict_from_kv_dim_match(257, 256), Nfqg003Verdict::Fail);
    }

    // Bandwidth helper sanity (per contract: savings = M * K * 4 bytes)
    #[test] fn qkv_bandwidth_qwen_1_5b_batch_4() {
        // M=2048 (4 batch * 512 seq), K=1536: 2048 * 1536 * 4 = 12,582,912 bytes ≈ 12.6 MB.
        assert_eq!(qkv_bandwidth_savings_bytes(2048, 1536), 12_582_912);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_NFQG_001_TOLERANCE - 1e-4).abs() < 1e-9);
        assert_eq!(AC_NFQG_002_FUSED_DRAM_READS, 2);
        assert_eq!(AC_NFQG_002_SEPARATE_DRAM_READS, 3);
        assert!((AC_NFQG_002_MIN_GAIN - 0.10).abs() < 1e-9);
    }
}
