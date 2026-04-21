//! CRUX-C-16 — LoRA hotswap classifiers.
//!
//! Pure, deterministic algorithm-level gates discharging FALSIFY-CRUX-C-16-001..004
//! at PARTIAL_ALGORITHM_LEVEL. Full discharge is blocked on `apr serve --enable-lora`
//! + `/v1/lora/{load,unload}` HTTP endpoints wired to a real adapter-hotswap runtime.
//!
//! Design rule: no silent passes — every ill-formed input maps to a distinct
//! Outcome variant so defect classes cannot collapse into each other.

/// Contract-pinned rank window. Adapters outside are rejected as "likely malformed".
pub const LORA_RANK_MIN: u32 = 1;
pub const LORA_RANK_MAX: u32 = 512;

/// Contract-pinned P99 load-latency budget (seconds).
pub const LORA_LOAD_LATENCY_P99_S: f64 = 2.0;

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 1: hotswap token parity vs offline-merged baseline
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotswapParityOutcome {
    Ok,
    EmptinessMismatch { merged_empty: bool, hotswap_empty: bool },
    LengthMismatch { merged_len: usize, hotswap_len: usize },
    TokenDivergence { at_index: usize, merged_token: u32, hotswap_token: u32 },
}

pub fn classify_hotswap_parity(
    merged_tokens: &[u32],
    hotswap_tokens: &[u32],
) -> HotswapParityOutcome {
    let merged_empty = merged_tokens.is_empty();
    let hotswap_empty = hotswap_tokens.is_empty();
    if merged_empty != hotswap_empty {
        return HotswapParityOutcome::EmptinessMismatch { merged_empty, hotswap_empty };
    }
    if merged_tokens.len() != hotswap_tokens.len() {
        return HotswapParityOutcome::LengthMismatch {
            merged_len: merged_tokens.len(),
            hotswap_len: hotswap_tokens.len(),
        };
    }
    for (i, (m, h)) in merged_tokens.iter().zip(hotswap_tokens.iter()).enumerate() {
        if m != h {
            return HotswapParityOutcome::TokenDivergence {
                at_index: i,
                merged_token: *m,
                hotswap_token: *h,
            };
        }
    }
    HotswapParityOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 2: load-latency P99 budget
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LoadLatencyOutcome {
    Ok { p99_seconds: f64 },
    InvalidInput { reason: &'static str },
    Exceeded { p99_seconds: f64, budget_seconds: f64 },
}

/// Compute P99 with nearest-rank (ceiling) method: P99 = sorted[ceil(0.99 * N) - 1].
/// Deterministic; no randomness; no interpolation.
pub fn classify_load_latency(samples_seconds: &[f64], budget_seconds: f64) -> LoadLatencyOutcome {
    if !budget_seconds.is_finite() || budget_seconds <= 0.0 {
        return LoadLatencyOutcome::InvalidInput { reason: "budget_seconds must be > 0" };
    }
    if samples_seconds.is_empty() {
        return LoadLatencyOutcome::InvalidInput { reason: "samples_seconds is empty" };
    }
    if samples_seconds.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return LoadLatencyOutcome::InvalidInput {
            reason: "sample contains NaN/inf or negative",
        };
    }

    let mut sorted: Vec<f64> = samples_seconds.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite after guard"));
    let n = sorted.len();
    // Nearest-rank P99: index = ceil(0.99 * N) - 1, clamped to [0, N-1].
    let rank_f = 0.99_f64 * (n as f64);
    let rank_ceil = rank_f.ceil() as usize;
    let idx = rank_ceil.saturating_sub(1).min(n - 1);
    let p99 = sorted[idx];

    if p99 > budget_seconds {
        return LoadLatencyOutcome::Exceeded {
            p99_seconds: p99,
            budget_seconds,
        };
    }
    LoadLatencyOutcome::Ok { p99_seconds: p99 }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 3: adapter↔base compatibility (sha256, target modules, rank window)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCompatOutcome {
    Ok,
    EmptyBaseSha256,
    EmptyAdapterBaseSha256,
    BaseSha256Mismatch,
    EmptyTargetModules,
    UnknownTargetModules { unknown: Vec<String> },
    RankTooSmall { rank: u32 },
    RankTooLarge { rank: u32 },
}

pub fn classify_adapter_compat(
    base_sha256: &str,
    adapter_base_sha256: &str,
    base_module_names: &[&str],
    adapter_target_modules: &[&str],
    adapter_rank: u32,
) -> AdapterCompatOutcome {
    if base_sha256.is_empty() {
        return AdapterCompatOutcome::EmptyBaseSha256;
    }
    if adapter_base_sha256.is_empty() {
        return AdapterCompatOutcome::EmptyAdapterBaseSha256;
    }
    // sha256 hex is case-insensitive — follow same rule used by C-09 spec-dec.
    if !base_sha256.eq_ignore_ascii_case(adapter_base_sha256) {
        return AdapterCompatOutcome::BaseSha256Mismatch;
    }
    if adapter_target_modules.is_empty() {
        return AdapterCompatOutcome::EmptyTargetModules;
    }
    let unknown: Vec<String> = adapter_target_modules
        .iter()
        .filter(|m| !base_module_names.contains(m))
        .map(|m| (*m).to_string())
        .collect();
    if !unknown.is_empty() {
        return AdapterCompatOutcome::UnknownTargetModules { unknown };
    }
    if adapter_rank < LORA_RANK_MIN {
        return AdapterCompatOutcome::RankTooSmall { rank: adapter_rank };
    }
    if adapter_rank > LORA_RANK_MAX {
        return AdapterCompatOutcome::RankTooLarge { rank: adapter_rank };
    }
    AdapterCompatOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 4: unload restores base model (pristine-state check)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnloadRestoreOutcome {
    Ok,
    EmptinessMismatch { fresh_empty: bool, after_unload_empty: bool },
    LengthMismatch { fresh_len: usize, after_unload_len: usize },
    TokenDivergence { at_index: usize, fresh_token: u32, after_unload_token: u32 },
}

pub fn classify_unload_restore(
    fresh_tokens: &[u32],
    after_unload_tokens: &[u32],
) -> UnloadRestoreOutcome {
    let fresh_empty = fresh_tokens.is_empty();
    let after_unload_empty = after_unload_tokens.is_empty();
    if fresh_empty != after_unload_empty {
        return UnloadRestoreOutcome::EmptinessMismatch {
            fresh_empty,
            after_unload_empty,
        };
    }
    if fresh_tokens.len() != after_unload_tokens.len() {
        return UnloadRestoreOutcome::LengthMismatch {
            fresh_len: fresh_tokens.len(),
            after_unload_len: after_unload_tokens.len(),
        };
    }
    for (i, (f, a)) in fresh_tokens.iter().zip(after_unload_tokens.iter()).enumerate() {
        if f != a {
            return UnloadRestoreOutcome::TokenDivergence {
                at_index: i,
                fresh_token: *f,
                after_unload_token: *a,
            };
        }
    }
    UnloadRestoreOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — pure, deterministic.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hotswap parity ─────────────────────────────────────────────────────

    #[test]
    fn hotswap_parity_ok_on_identical() {
        assert_eq!(
            classify_hotswap_parity(&[10, 20, 30], &[10, 20, 30]),
            HotswapParityOutcome::Ok
        );
    }

    #[test]
    fn hotswap_parity_ok_on_two_empty() {
        assert_eq!(classify_hotswap_parity(&[], &[]), HotswapParityOutcome::Ok);
    }

    #[test]
    fn hotswap_parity_rejects_emptiness_mismatch() {
        assert_eq!(
            classify_hotswap_parity(&[], &[1]),
            HotswapParityOutcome::EmptinessMismatch {
                merged_empty: true,
                hotswap_empty: false,
            }
        );
    }

    #[test]
    fn hotswap_parity_rejects_length_mismatch() {
        assert_eq!(
            classify_hotswap_parity(&[1, 2], &[1, 2, 3]),
            HotswapParityOutcome::LengthMismatch {
                merged_len: 2,
                hotswap_len: 3,
            }
        );
    }

    #[test]
    fn hotswap_parity_rejects_first_divergence() {
        assert_eq!(
            classify_hotswap_parity(&[1, 2, 3, 4], &[1, 9, 3, 9]),
            HotswapParityOutcome::TokenDivergence {
                at_index: 1,
                merged_token: 2,
                hotswap_token: 9,
            }
        );
    }

    #[test]
    fn hotswap_parity_is_deterministic() {
        for _ in 0..5 {
            assert_eq!(
                classify_hotswap_parity(&[7, 8, 9], &[7, 8, 9]),
                HotswapParityOutcome::Ok
            );
        }
    }

    // ── load latency ───────────────────────────────────────────────────────

    #[test]
    fn load_latency_ok_under_budget() {
        let samples = vec![0.1_f64; 100];
        match classify_load_latency(&samples, 2.0) {
            LoadLatencyOutcome::Ok { p99_seconds } => {
                assert!((p99_seconds - 0.1).abs() < 1e-9);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_ok_single_sample() {
        // P99 of a single sample is that sample.
        match classify_load_latency(&[1.5], 2.0) {
            LoadLatencyOutcome::Ok { p99_seconds } => assert_eq!(p99_seconds, 1.5),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_rejects_exceeded_budget() {
        // Nearest-rank P99 with 10 samples: ceil(0.99 * 10) = 10, idx = 9 = max.
        // 9 fast + 1 slow → P99 == slow.
        let mut samples = vec![0.1_f64; 9];
        samples.push(2.5);
        match classify_load_latency(&samples, 2.0) {
            LoadLatencyOutcome::Exceeded { p99_seconds, budget_seconds } => {
                assert_eq!(p99_seconds, 2.5);
                assert_eq!(budget_seconds, 2.0);
            }
            other => panic!("expected Exceeded, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_nearest_rank_p99_is_99th_smallest_of_100() {
        // 100 samples: ceil(0.99 * 100) = 99, idx = 98 = sorted[98].
        // 99 fast + 1 slow → P99 == fast (99% of samples are ≤ 0.1).
        let mut samples = vec![0.1_f64; 99];
        samples.push(2.5);
        match classify_load_latency(&samples, 2.0) {
            LoadLatencyOutcome::Ok { p99_seconds } => assert_eq!(p99_seconds, 0.1),
            other => panic!("expected Ok (99th of 100 is 0.1), got {other:?}"),
        }
    }

    #[test]
    fn load_latency_rejects_zero_budget() {
        match classify_load_latency(&[0.1], 0.0) {
            LoadLatencyOutcome::InvalidInput { reason } => assert!(reason.contains("budget")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_rejects_empty_samples() {
        match classify_load_latency(&[], 2.0) {
            LoadLatencyOutcome::InvalidInput { reason } => assert!(reason.contains("empty")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_rejects_nan_sample() {
        match classify_load_latency(&[0.1, f64::NAN], 2.0) {
            LoadLatencyOutcome::InvalidInput { reason } => assert!(reason.contains("NaN")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_rejects_negative_sample() {
        match classify_load_latency(&[-0.1, 0.2], 2.0) {
            LoadLatencyOutcome::InvalidInput { reason } => assert!(reason.contains("negative")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn load_latency_is_deterministic() {
        let samples = [1.0, 0.1, 0.2, 0.5, 0.3];
        for _ in 0..5 {
            match classify_load_latency(&samples, 2.0) {
                LoadLatencyOutcome::Ok { .. } => {}
                other => panic!("expected Ok, got {other:?}"),
            }
        }
    }

    // ── adapter compat ─────────────────────────────────────────────────────

    #[test]
    fn adapter_compat_ok_on_matching_everything() {
        assert_eq!(
            classify_adapter_compat(
                "abc123",
                "abc123",
                &["q_proj", "k_proj", "v_proj"],
                &["q_proj", "v_proj"],
                64,
            ),
            AdapterCompatOutcome::Ok
        );
    }

    #[test]
    fn adapter_compat_ok_case_insensitive_sha() {
        assert_eq!(
            classify_adapter_compat(
                "ABC123",
                "abc123",
                &["q_proj"],
                &["q_proj"],
                16,
            ),
            AdapterCompatOutcome::Ok
        );
    }

    #[test]
    fn adapter_compat_rejects_empty_base_sha() {
        assert_eq!(
            classify_adapter_compat("", "abc", &["q_proj"], &["q_proj"], 16),
            AdapterCompatOutcome::EmptyBaseSha256
        );
    }

    #[test]
    fn adapter_compat_rejects_empty_adapter_sha() {
        assert_eq!(
            classify_adapter_compat("abc", "", &["q_proj"], &["q_proj"], 16),
            AdapterCompatOutcome::EmptyAdapterBaseSha256
        );
    }

    #[test]
    fn adapter_compat_rejects_sha_mismatch() {
        assert_eq!(
            classify_adapter_compat("abc", "def", &["q_proj"], &["q_proj"], 16),
            AdapterCompatOutcome::BaseSha256Mismatch
        );
    }

    #[test]
    fn adapter_compat_rejects_empty_target_modules() {
        assert_eq!(
            classify_adapter_compat("abc", "abc", &["q_proj"], &[], 16),
            AdapterCompatOutcome::EmptyTargetModules
        );
    }

    #[test]
    fn adapter_compat_rejects_unknown_target_modules() {
        match classify_adapter_compat(
            "abc",
            "abc",
            &["q_proj", "k_proj"],
            &["q_proj", "x_proj"],
            16,
        ) {
            AdapterCompatOutcome::UnknownTargetModules { unknown } => {
                assert_eq!(unknown, vec!["x_proj".to_string()]);
            }
            other => panic!("expected UnknownTargetModules, got {other:?}"),
        }
    }

    #[test]
    fn adapter_compat_rejects_rank_too_small() {
        assert_eq!(
            classify_adapter_compat("abc", "abc", &["q_proj"], &["q_proj"], 0),
            AdapterCompatOutcome::RankTooSmall { rank: 0 }
        );
    }

    #[test]
    fn adapter_compat_rejects_rank_too_large() {
        assert_eq!(
            classify_adapter_compat("abc", "abc", &["q_proj"], &["q_proj"], 513),
            AdapterCompatOutcome::RankTooLarge { rank: 513 }
        );
    }

    #[test]
    fn adapter_compat_ok_at_rank_boundaries() {
        assert_eq!(
            classify_adapter_compat("abc", "abc", &["q_proj"], &["q_proj"], LORA_RANK_MIN),
            AdapterCompatOutcome::Ok
        );
        assert_eq!(
            classify_adapter_compat("abc", "abc", &["q_proj"], &["q_proj"], LORA_RANK_MAX),
            AdapterCompatOutcome::Ok
        );
    }

    #[test]
    fn adapter_compat_is_deterministic() {
        for _ in 0..5 {
            assert_eq!(
                classify_adapter_compat(
                    "abc",
                    "abc",
                    &["q_proj", "v_proj"],
                    &["q_proj"],
                    32,
                ),
                AdapterCompatOutcome::Ok
            );
        }
    }

    // ── unload restore ─────────────────────────────────────────────────────

    #[test]
    fn unload_restore_ok_on_identical() {
        assert_eq!(
            classify_unload_restore(&[1, 2, 3], &[1, 2, 3]),
            UnloadRestoreOutcome::Ok
        );
    }

    #[test]
    fn unload_restore_ok_on_two_empty() {
        assert_eq!(
            classify_unload_restore(&[], &[]),
            UnloadRestoreOutcome::Ok
        );
    }

    #[test]
    fn unload_restore_rejects_emptiness_mismatch() {
        assert_eq!(
            classify_unload_restore(&[1, 2], &[]),
            UnloadRestoreOutcome::EmptinessMismatch {
                fresh_empty: false,
                after_unload_empty: true,
            }
        );
    }

    #[test]
    fn unload_restore_rejects_length_mismatch() {
        assert_eq!(
            classify_unload_restore(&[1, 2, 3], &[1, 2]),
            UnloadRestoreOutcome::LengthMismatch {
                fresh_len: 3,
                after_unload_len: 2,
            }
        );
    }

    #[test]
    fn unload_restore_rejects_first_divergence() {
        assert_eq!(
            classify_unload_restore(&[1, 2, 3], &[1, 9, 3]),
            UnloadRestoreOutcome::TokenDivergence {
                at_index: 1,
                fresh_token: 2,
                after_unload_token: 9,
            }
        );
    }

    #[test]
    fn unload_restore_is_deterministic() {
        for _ in 0..5 {
            assert_eq!(
                classify_unload_restore(&[1, 2, 3], &[1, 2, 3]),
                UnloadRestoreOutcome::Ok
            );
        }
    }

    // ── constants ──────────────────────────────────────────────────────────

    #[test]
    fn lora_constants_are_canonical() {
        assert_eq!(LORA_RANK_MIN, 1);
        assert_eq!(LORA_RANK_MAX, 512);
        assert!((LORA_LOAD_LATENCY_P99_S - 2.0).abs() < 1e-9);
    }
}
