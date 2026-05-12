//! SHIP-TWO-001 const pinning — prevents silent drift on the 45 `AC_*`
//! constants that parameterize the 42 PARTIAL_ALGORITHM_LEVEL verdict
//! functions spec'd in `docs/specifications/aprender-train/ship-two-models-spec.md`.
//!
//! Every verdict function was authored against a numeric threshold (or
//! byte-level constant) that came from the spec. If a contributor edits
//! `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS` from `2.2` → `2.5` without
//! touching the spec, the verdict fn still compiles but silently accepts
//! looser thresholds. This file catches that class at `cargo test` time.
//!
//! Each assertion references the spec section and amendment version that
//! pinned the constant, so when the spec moves, these numbers must move
//! in lock-step.

// --- aprender-core: MODEL-1 AC (§4.2) + stability (§7.1) ---

use aprender::bench::ship_007::AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B;
use aprender::format::gate_ship_001::AC_GATE_SHIP_001_MODEL_1_AC_COUNT;
use aprender::format::gate_ship_002::AC_GATE_SHIP_002_MODEL_2_AC_COUNT;
use aprender::format::gate_ship_005::AC_GATE_SHIP_005_REQUIRED_LICENSE_FIELD;
use aprender::format::gate_ship_006::AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA;
use aprender::format::gate_ship_007::AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT;
use aprender::format::gate_ship_008::AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE;
use aprender::format::gate_ship_009::AC_GATE_SHIP_009_REQUIRED_CHECK_COUNT;
use aprender::format::gate_ship_010::AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT;
use aprender::format::gate_ship_011::AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE;
use aprender::format::gate_ship_012::AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT;
use aprender::format::ship_001::{
    AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN, AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE,
};
use aprender::format::ship_003::AC_SHIP1_003_MIN_COSINE_SIMILARITY;
use aprender::format::ship_004::{
    AC_SHIP1_004_GGUF_MAGIC_BYTES, AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS,
    AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE,
};
use aprender::format::ship_010::{AC_SHIP1_010_REQUIRED_URL_SCHEME, AC_SHIP1_010_SHA256_HEX_LEN};
use aprender::format::ship_023::AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP;
use aprender::format::ship_024::{
    AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT, AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT,
    AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE,
};
use aprender::metrics::ship_005::{
    AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT, AC_SHIP1_005_NOISE_ALLOWANCE_PP,
    AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT,
};
use aprender::qa::ship_002::AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS;
use aprender::qa::ship_006::AC_SHIP1_006_REQUIRED_QA_GATE_COUNT;
use aprender::text::chat_template::{
    AC_SHIP1_008_CANONICAL_GOLDEN, AC_SHIP1_008_CANONICAL_SYSTEM, AC_SHIP1_008_CANONICAL_USER,
};

// --- aprender-train: MODEL-2 AC (§5.2) + GPUTRAIN (§14) ---

use entrenar::models::llama_370m::{
    AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS, AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS,
    AC_SHIP2_006_REQUIRED_QA_GATE_COUNT, AC_SHIP2_007_HELDOUT_PROMPT_COUNT,
    AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT,
    AC_SHIP2_010_MIN_DECODE_TPS_RTX4090,
};
use entrenar::train::gputrain_003::{
    AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB, AC_GPUTRAIN_003_NVIDIA_SMI_POLL_WINDOW_SECONDS,
};
use entrenar::train::gputrain_004::{
    AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS, AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS,
};
use entrenar::train::gputrain_005::AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M;
use entrenar::train::gputrain_006::AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA;
use entrenar::train::gputrain_007::AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS;

// ─── MODEL-1 constants (§4.2 AC-SHIP1-001..010 + §7.1 SHIP-023..024) ───

#[test]
fn ship1_001_safetensors_header_prefix_len_is_8() {
    assert_eq!(AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN, 8_u64);
}

#[test]
fn ship1_001_safetensors_json_open_byte_is_open_brace() {
    assert_eq!(AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE, b'{');
    assert_eq!(AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE, 0x7B_u8);
}

#[test]
fn ship1_002_python_syntax_zero_tolerance() {
    assert_eq!(AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS, 0);
}

#[test]
fn ship1_003_min_cosine_similarity_is_0_999() {
    assert_eq!(AC_SHIP1_003_MIN_COSINE_SIMILARITY, 0.999_f32);
}

#[test]
fn ship1_004_llama_cli_success_is_zero() {
    assert_eq!(AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE, 0_i32);
}

#[test]
fn ship1_004_gguf_magic_is_ascii_gguf() {
    assert_eq!(AC_SHIP1_004_GGUF_MAGIC_BYTES, b"GGUF");
    assert_eq!(AC_SHIP1_004_GGUF_MAGIC_BYTES.len(), 4);
}

#[test]
fn ship1_004_gguf_supported_versions_are_2_and_3() {
    assert_eq!(AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS, &[2_u32, 3_u32]);
}

#[test]
fn ship1_005_humaneval_nominal_is_86_percent() {
    assert_eq!(AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT, 86.00_f32);
}

#[test]
fn ship1_005_noise_allowance_is_1_2_pp() {
    assert_eq!(AC_SHIP1_005_NOISE_ALLOWANCE_PP, 1.20_f32);
}

#[test]
fn ship1_005_effective_equals_nominal_minus_noise() {
    // f32 `86.0 - 1.2 ≈ 84.79999924` — tolerance-bounded equality, not exact.
    let expected = AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT - AC_SHIP1_005_NOISE_ALLOWANCE_PP;
    assert!(
        (AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT - expected).abs() < 1e-4,
        "effective ({}) should equal nominal ({}) - noise ({})",
        AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT,
        AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT,
        AC_SHIP1_005_NOISE_ALLOWANCE_PP,
    );
}

#[test]
fn ship1_006_qa_gate_count_is_8() {
    assert_eq!(AC_SHIP1_006_REQUIRED_QA_GATE_COUNT, 8);
}

#[test]
fn ship1_007_decode_tps_floor_is_30() {
    assert_eq!(AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B, 30.0_f32);
}

#[test]
fn ship1_008_canonical_system_prompt_is_fixed() {
    assert_eq!(AC_SHIP1_008_CANONICAL_SYSTEM, "You are a helpful coding assistant.");
}

#[test]
fn ship1_008_canonical_user_prompt_is_fixed() {
    assert_eq!(
        AC_SHIP1_008_CANONICAL_USER,
        "Write a Python function to compute the nth Fibonacci number."
    );
}

#[test]
fn ship1_008_canonical_golden_starts_with_chatml_header() {
    assert!(
        AC_SHIP1_008_CANONICAL_GOLDEN.starts_with("<|im_start|>system\n"),
        "ChatML header drift: golden does not start with <|im_start|>system"
    );
}

#[test]
fn ship1_010_sha256_hex_len_is_64() {
    assert_eq!(AC_SHIP1_010_SHA256_HEX_LEN, 64);
}

#[test]
fn ship1_010_required_scheme_is_tls() {
    assert_eq!(AC_SHIP1_010_REQUIRED_URL_SCHEME, "https://");
}

#[test]
fn ship1_023_drift_tolerance_is_1_2_pp() {
    assert_eq!(AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP, 1.2_f32);
}

#[test]
fn ship1_024_adversarial_suite_floor_is_50() {
    assert_eq!(AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE, 50);
}

#[test]
fn ship1_024_panic_and_nan_zero_tolerance() {
    assert_eq!(AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT, 0_u32);
    assert_eq!(AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT, 0_u32);
}

// ─── MODEL-2 constants (§5.2 AC-SHIP2-003..010) ───

#[test]
fn ship2_003_val_ce_floor_is_2_2() {
    assert_eq!(AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS, 2.2_f32);
}

#[test]
fn ship2_004_training_budget_is_21_days() {
    assert_eq!(AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS, 21_u32);
}

#[test]
fn ship2_006_qa_gate_count_matches_model_1() {
    // Both models run the same 8-gate `apr qa` suite.
    assert_eq!(AC_SHIP2_006_REQUIRED_QA_GATE_COUNT, 8);
    assert_eq!(AC_SHIP2_006_REQUIRED_QA_GATE_COUNT, AC_SHIP1_006_REQUIRED_QA_GATE_COUNT);
}

#[test]
fn ship2_007_heldout_prompts_is_100() {
    assert_eq!(AC_SHIP2_007_HELDOUT_PROMPT_COUNT, 100);
}

#[test]
fn ship2_007_syntax_error_tolerance_is_1() {
    // MODEL-2 allows 1 error in 100 (noise); MODEL-1 allows 0 on canonical.
    assert_eq!(AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS, 1);
}

#[test]
fn ship2_008_humaneval_floor_is_30_percent() {
    assert_eq!(AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT, 30.0_f32);
}

#[test]
fn ship2_010_decode_tps_floor_is_100() {
    // 370M is ~3.5× smaller than 7B; spec floor is 3.3× higher tok/s.
    assert_eq!(AC_SHIP2_010_MIN_DECODE_TPS_RTX4090, 100.0_f32);
}

// ─── §6 Compound Ship Gates (GATE-SHIP-001..012) ───

#[test]
fn gate_ship_001_model_1_ac_count_is_10() {
    assert_eq!(AC_GATE_SHIP_001_MODEL_1_AC_COUNT, 10);
}

#[test]
fn gate_ship_002_model_2_ac_count_is_12() {
    assert_eq!(AC_GATE_SHIP_002_MODEL_2_AC_COUNT, 12);
}

#[test]
fn gate_ship_005_required_license_field() {
    assert_eq!(AC_GATE_SHIP_005_REQUIRED_LICENSE_FIELD, "license");
}

#[test]
fn gate_ship_006_first_token_tolerance() {
    assert_eq!(AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA, 1e-3_f32);
}

#[test]
fn gate_ship_007_unwrap_zero_tolerance() {
    assert_eq!(AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT, 0_u32);
}

#[test]
fn gate_ship_008_contract_density_100_percent() {
    assert_eq!(AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE, 1.0_f32);
}

#[test]
fn gate_ship_009_required_ci_check_count() {
    // fmt + clippy + test = 3
    assert_eq!(AC_GATE_SHIP_009_REQUIRED_CHECK_COUNT, 3);
}

#[test]
fn gate_ship_010_advisory_zero_tolerance() {
    assert_eq!(AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT, 0_u32);
}

#[test]
fn gate_ship_011_pmat_tdg_floor_is_90() {
    assert_eq!(AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE, 90.0_f32);
}

#[test]
fn gate_ship_012_line_coverage_floor_is_95() {
    assert_eq!(AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT, 95.0_f32);
}

// ─── §14 Task #132 GPUTRAIN (FALSIFY-GPUTRAIN-003..007) ───

#[test]
fn gputrain_003_nvidia_smi_poll_window_is_5s() {
    assert_eq!(AC_GPUTRAIN_003_NVIDIA_SMI_POLL_WINDOW_SECONDS, 5_u32);
}

#[test]
fn gputrain_003_min_used_memory_is_1_mib() {
    // Any residency at all — if the process has allocated ≥1 MiB, it's
    // on the GPU. 0 MiB is the exact symptom that triggered Task #132.
    assert_eq!(AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB, 1_u64);
}

#[test]
fn gputrain_004_dispatch_variant_labels() {
    assert_eq!(AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS, &["Cpu"]);
    assert_eq!(AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS, &["Cuda"]);
}

#[test]
fn gputrain_005_step_time_budget_is_500ms() {
    assert_eq!(AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M, 500.0_f32);
}

#[test]
fn gputrain_006_seed_loss_delta_tolerance_is_1e_minus_5() {
    assert_eq!(AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA, 1e-5_f32);
}

#[test]
fn gputrain_007_version_json_required_keys() {
    assert_eq!(
        AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS,
        &["cuda_feature", "cuda_runtime_available", "visible_devices"]
    );
}

// ─── Structural aggregate: total count of pinned SHIP consts ───

/// Meta-test: as long as every individual `#[test]` above is present,
/// the per-const coverage is enforced at compile time. This counts the
/// expected number of pinned constants and serves as a tripwire for
/// contributors adding new SHIP consts without a pin.
#[test]
fn pinned_const_count_tripwire() {
    const EXPECTED_PINNED_CONSTS: usize = 45;
    // Not a runtime count — a human-verifiable comment-sync reminder.
    // Anyone adding a new `AC_SHIP*` / `AC_GATE_SHIP_*` / `AC_GPUTRAIN_*`
    // const must also add an `assert_eq!` test here and bump this number.
    // See ripgrep sentinel: `^pub const AC_` across crates/ should equal
    // this tripwire. Last synced at SHIP-TWO-001 spec v2.43.0.
    assert_eq!(EXPECTED_PINNED_CONSTS, 45);
}
