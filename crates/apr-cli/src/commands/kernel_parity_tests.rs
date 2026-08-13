//! Tests for the `apr kernel parity` producer (aprender#2377 finding 3).
//!
//! The load-bearing test is `round_trip_*`: the producer's own output is fed to
//! `apr attn-parity-lint` and the lint must ACCEPT it — at the SHIPPED default
//! tolerances, not at tolerances chosen to make it pass. The negative half
//! corrupts the body and requires the lint to reject it, so the round trip
//! cannot pass vacuously.

use super::*;
use crate::commands::attn_parity_lint;

fn dims() -> ParityDims {
    ParityDims {
        seq_len: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        seed: 7,
    }
}

// ── ROUND TRIP ───────────────────────────────────────────────────────────

#[cfg(feature = "inference")]
#[test]
fn round_trip_producer_output_is_accepted_by_attn_parity_lint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let obs = dir.path().join("parity.json");

    run(
        KernelImpl::Tiled,
        KernelRef::Naive,
        dims(),
        true,
        Some(&obs),
        false,
    )
    .expect("the tiled kernel must produce a measurement");

    // One body, both gates — parity numerics AND provenance — at the shipped
    // defaults (5e-3 / 0.9999).
    attn_parity_lint::run(
        Some(&obs),
        Some(&obs),
        None,
        attn_parity_lint::ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
        attn_parity_lint::ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
        false,
    )
    .expect("attn-parity-lint must accept the producer's own observation");
}

#[cfg(feature = "inference")]
#[test]
fn round_trip_cannot_pass_vacuously_when_the_body_is_corrupted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let obs = dir.path().join("parity.json");
    run(
        KernelImpl::Tiled,
        KernelRef::Naive,
        dims(),
        true,
        Some(&obs),
        false,
    )
    .expect("producer");
    let good: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&obs).expect("read")).expect("parse");

    for (label, mutate) in [
        (
            "max_abs_diff past the FA2 bound",
            Box::new(|v: &mut serde_json::Value| v["max_abs_diff"] = serde_json::json!(0.5))
                as Box<dyn Fn(&mut serde_json::Value)>,
        ),
        (
            "cosine below the floor",
            Box::new(|v: &mut serde_json::Value| v["cosine_sim"] = serde_json::json!(0.9)),
        ),
        (
            "provenance claiming flash2 with no pinned sha",
            Box::new(|v: &mut serde_json::Value| {
                v["attn_impl"] = serde_json::json!("flash2");
                v["kernel_source"] = serde_json::Value::Null;
            }),
        ),
        (
            "fallback reason blanked out",
            Box::new(|v: &mut serde_json::Value| v["fallback"] = serde_json::json!("")),
        ),
    ] {
        let mut bad = good.clone();
        mutate(&mut bad);
        let path = dir.path().join("bad.json");
        std::fs::write(&path, serde_json::to_string(&bad).expect("ser")).expect("write");
        let err = attn_parity_lint::run(
            Some(&path),
            Some(&path),
            None,
            attn_parity_lint::ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
            attn_parity_lint::ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
            false,
        )
        .expect_err(&format!("lint must reject: {label}"));
        assert!(
            matches!(err, CliError::ValidationFailed(_)),
            "{label}: expected a validation refusal, got {err:?}"
        );
    }
}

/// The head-dim refusal is itself an observation: the error JSON it writes must
/// be accepted by `attn-parity-lint --head-dim-error-file`.
#[test]
fn round_trip_head_dim_refusal_is_accepted_by_the_head_dim_gate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err_json = dir.path().join("head-dim.json");
    let mut d = dims();
    d.head_dim = 96;

    let err = run(
        KernelImpl::Flash2,
        KernelRef::Naive,
        d,
        true,
        Some(&err_json),
        false,
    )
    .expect_err("head_dim 96 must be refused, not slow-pathed");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
    assert!(err.exit_code_value() != 0, "a refusal must not exit 0");

    attn_parity_lint::run(
        None,
        None,
        Some(&err_json),
        attn_parity_lint::ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
        attn_parity_lint::ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
        false,
    )
    .expect("the head-dim gate must accept the producer's own error body");
}

/// The supported set must render as a SET, not `[64, 128]` — which reads as a
/// closed interval and would make head_dim 96 look supported by the very
/// message refusing it.
#[test]
fn the_head_dim_refusal_names_a_set_not_an_interval() {
    let mut d = dims();
    d.head_dim = 96;
    let err = run(KernelImpl::Flash2, KernelRef::Naive, d, false, None, false)
        .expect_err("96 must be refused");
    let msg = err.to_string();
    assert!(msg.contains("{64, 128}"), "got: {msg}");
    assert!(
        !msg.contains("[64, 128]"),
        "interval notation would imply 96 is in range: {msg}"
    );
}

#[test]
fn head_dim_gate_rejects_an_error_body_that_is_not_about_head_dim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("other.json");
    std::fs::write(&path, r#"{"error":"out of memory"}"#).expect("write");
    let err = attn_parity_lint::run(
        None,
        None,
        Some(&path),
        attn_parity_lint::ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
        attn_parity_lint::ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
        false,
    )
    .expect_err("an unrelated error must not discharge the head-dim gate");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
}

// ── honest refusals ──────────────────────────────────────────────────────

#[test]
fn flash2_is_refused_rather_than_answered_by_the_tiled_kernel() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("flash2.json");
    let err = run(
        KernelImpl::Flash2,
        KernelRef::Naive,
        dims(),
        true,
        Some(&out),
        false,
    )
    .expect_err("a kernel this binary does not embed must not report a measurement");
    assert!(
        matches!(err, CliError::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );

    let body: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read")).expect("parse");
    assert!(
        body.get("max_abs_diff").is_none(),
        "a refusal must not carry a parity number: {body}"
    );
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|s| s.contains("flash2-kernel-unavailable")),
        "the refusal must name what is missing: {body}"
    );
}

#[test]
fn flash2_at_a_supported_head_dim_still_refuses_without_the_kernel() {
    for head_dim in FLASH2_SUPPORTED_HEAD_DIMS {
        let mut d = dims();
        d.head_dim = head_dim;
        let err = run(KernelImpl::Flash2, KernelRef::Naive, d, false, None, false)
            .expect_err("head_dim being supported does not conjure the kernel");
        assert!(matches!(err, CliError::NotImplemented(_)), "{err:?}");
    }
}

#[test]
fn zero_head_dim_is_refused_with_a_head_dim_message() {
    let mut d = dims();
    d.head_dim = 0;
    let err = run(KernelImpl::Tiled, KernelRef::Naive, d, false, None, false)
        .expect_err("head_dim 0 is not a kernel configuration");
    assert!(err.to_string().contains("head-dim"), "got: {err}");
}

#[test]
fn gqa_group_mismatch_is_refused() {
    let mut d = dims();
    d.num_heads = 5;
    d.num_kv_heads = 2;
    let err = run(KernelImpl::Tiled, KernelRef::Naive, d, false, None, false)
        .expect_err("5 query heads cannot be split into 2 whole KV groups");
    assert!(err.to_string().contains("whole groups"), "got: {err}");
}

#[test]
fn refuses_to_clobber_an_existing_output_without_force() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("existing.json");
    std::fs::write(&out, "precious").expect("write");
    let err = run(
        KernelImpl::Tiled,
        KernelRef::Naive,
        dims(),
        true,
        Some(&out),
        false,
    )
    .expect_err("an existing output must not be overwritten silently");
    assert!(err.to_string().contains("--force"), "got: {err}");
}

// ── the measurement itself ───────────────────────────────────────────────

#[cfg(feature = "inference")]
#[test]
fn tiled_and_naive_agree_far_inside_the_fa2_bound() {
    use realizar::brick::FlashAttentionBrick;
    let d = dims();
    let (q, k, v) = draw_qkv(&d);
    let tiled = FlashAttentionBrick::new(d.num_heads, d.num_kv_heads, d.head_dim)
        .forward(&q, &k, &v, d.seq_len)
        .expect("tiled forward");
    let naive = naive_attention(&q, &k, &v, &d);
    let mad = max_abs_diff(&tiled, &naive);
    assert!(
        mad < 1e-5,
        "two f32 implementations of the same attention must agree to ~1e-6; got {mad:e}"
    );
}

/// The comparison must be able to FAIL — otherwise it proves nothing about the
/// kernel. Perturbing one output by 0.5 has to blow past the 5e-3 bound.
#[test]
fn the_parity_metrics_are_not_vacuous() {
    let a = vec![0.25f32, -0.5, 0.75, 1.0];
    let mut b = a.clone();
    b[2] += 0.5;
    assert!(
        max_abs_diff(&a, &b) > 5e-3,
        "max_abs_diff must see the perturbation"
    );
    let cos = cosine_sim(&a, &b).expect("non-zero norms");
    assert!(cos < 0.9999, "cosine must see the perturbation, got {cos}");
    assert_eq!(max_abs_diff(&a, &a), 0.0);
}

#[test]
fn cosine_of_a_zero_vector_is_undefined_not_one() {
    assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 1.0]), None);
}

#[test]
fn the_same_seed_draws_the_same_inputs() {
    let (q1, k1, v1) = draw_qkv(&dims());
    let (q2, k2, v2) = draw_qkv(&dims());
    assert_eq!(q1, q2);
    assert_eq!(k1, k2);
    assert_eq!(v1, v2);

    let mut other = dims();
    other.seed = 8;
    let (q3, _, _) = draw_qkv(&other);
    assert_ne!(q1, q3, "a different seed must draw different inputs");
}

#[test]
fn drawn_values_stay_inside_the_unit_interval() {
    let (q, k, v) = draw_qkv(&dims());
    for (name, xs) in [("q", &q), ("k", &k), ("v", &v)] {
        assert!(
            xs.iter().all(|x| (-1.0..1.0).contains(x)),
            "{name} escaped [-1, 1)"
        );
    }
}

/// A single-head, single-position case where the answer is known by hand:
/// attention over one key returns that value exactly.
#[test]
fn naive_attention_over_one_position_returns_that_value() {
    let d = ParityDims {
        seq_len: 1,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        seed: 0,
    };
    let out = naive_attention(&[1.0, 0.0], &[0.5, 0.5], &[3.0, -4.0], &d);
    assert!((out[0] - 3.0).abs() < 1e-6, "got {out:?}");
    assert!((out[1] - -4.0).abs() < 1e-6, "got {out:?}");
}
