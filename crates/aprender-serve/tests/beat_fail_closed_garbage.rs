//! BEAT-FAIL-CLOSED — Pillar-4 correctness beat (PMAT-744).
//!
//! The mission's HEADLINE Pillar-4 beat is not "faster than Ollama" (apr concedes
//! raw CPU decode) — it is: **apr provably never ships garbage; the incumbents
//! provably do.** A GGUF/SafeTensors that *parses* (valid magic, shapes, dtypes)
//! but is *semantically* corrupt — weights that are all-zero, NaN, Inf, or
//! effectively empty — is silently LOADED and run by llama.cpp / Ollama, emitting
//! garbage tokens with exit 0 and no warning. apr's Poka-Yoke validation
//! (PMAT-234/235, F-DATA-QUALITY-001..004) REJECTS such a tensor at load time.
//!
//! This beat is the CI-gated, falsifiable form of that guarantee: apr must reject
//! EVERY semantically-broken tensor class (fail-closed) AND accept a healthy one
//! (no false positives). A regression that lets any broken class through hard-fails
//! the gate. The head-to-head incumbent measurement (llama.cpp/Ollama accept the
//! same artifacts) is recorded in docs/BEATS.md / the contract evidence — it needs
//! model files + runtimes, so it is not wired into per-PR CI, but the apr-side
//! invariant proven here is what makes the claim falsifiable.
//!
//! Contract: contracts/apr-fail-closed-garbage-beat-v1.yaml.

use realizar::safetensors::validation::{validate_embedding, validate_weight};

const OUT: usize = 64;
const IN: usize = 64;
const N: usize = OUT * IN;

/// A healthy, dense, finite, varied weight matrix — must PASS (no false positive).
fn healthy_weight() -> Vec<f32> {
    (0..N)
        .map(|i| {
            // deterministic, dense, varied, mean≈0, L2≫0, no exact zeros
            let x = ((i % 17) as f32) - 8.0;
            if x == 0.0 {
                0.5
            } else {
                x * 0.01
            }
        })
        .collect()
}

/// Each entry: (class name, broken weight data, expected_out, expected_in).
/// Every one MUST be rejected by `validate_weight` (fail-closed).
fn broken_weight_classes() -> Vec<(&'static str, Vec<f32>, usize, usize)> {
    let mut all_zero = vec![0.0_f32; N];
    // keep a couple non-zero so it's not caught only by L2 — density gate must fire
    all_zero[0] = 1.0;
    all_zero[1] = -1.0;

    // 90% zeros — wrong-offset signature, density gate (>80%)
    let mut mostly_zero = vec![0.0_f32; N];
    for i in 0..(N / 10) {
        mostly_zero[i] = 0.3;
    }

    let mut with_nan = healthy_weight();
    with_nan[123] = f32::NAN;

    let mut with_inf = healthy_weight();
    with_inf[200] = f32::INFINITY;

    // effectively empty — sub-threshold magnitudes → L2 ~ 0
    let near_zero_l2 = vec![1e-9_f32; N];

    // structurally wrong: parses as data but wrong element count for its role
    let wrong_shape = healthy_weight()[..N - 10].to_vec();

    vec![
        ("all_zero_weight", all_zero, OUT, IN),
        ("mostly_zero_weight_>80pct", mostly_zero, OUT, IN),
        ("nan_weight", with_nan, OUT, IN),
        ("inf_weight", with_inf, OUT, IN),
        ("near_zero_l2_weight", near_zero_l2, OUT, IN),
        ("shape_mismatch_weight", wrong_shape, OUT, IN),
    ]
}

#[test]
fn beat_apr_rejects_all_broken_weight_classes_fail_closed() {
    let mut rejected = 0;
    let classes = broken_weight_classes();
    let total = classes.len();
    for (name, data, out_dim, in_dim) in classes {
        let r = validate_weight(name, &data, out_dim, in_dim);
        assert!(
            !r.passed,
            "FAIL-CLOSED VIOLATION: apr ACCEPTED broken weight class `{name}` \
             (this is the garbage llama.cpp/Ollama ship silently). stats={:?}",
            r.stats
        );
        rejected += 1;
    }
    assert_eq!(
        rejected, total,
        "apr must reject ALL {total} broken weight classes (fail-closed); rejected {rejected}"
    );
    println!(
        "BEAT-FAIL-CLOSED weights: apr rejected {rejected}/{total} broken classes (incumbents: 0)"
    );
}

#[test]
fn beat_apr_rejects_broken_embeddings_fail_closed() {
    // Embedding-specific gates: density (>50%), NaN/Inf, constant, dead-token spot-check.
    let vocab = 96;
    let hidden = 32;
    let n = vocab * hidden;

    let healthy: Vec<f32> = (0..n)
        .map(|i| (((i % 13) as f32) - 6.0) * 0.02 + 0.01)
        .collect();

    // dead token at 50% of vocab (spot-check gate) — rest healthy
    let mut dead_token = healthy.clone();
    let tok = vocab * 50 / 100;
    for v in dead_token.iter_mut().skip(tok * hidden).take(hidden) {
        *v = 0.0;
    }

    let constant = vec![0.7_f32; n]; // all identical → distribution gate
    let mut emb_nan = healthy.clone();
    emb_nan[10] = f32::NAN;

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("all_zero_embedding", vec![0.0; n]),
        ("constant_embedding", constant),
        ("nan_embedding", emb_nan),
        ("dead_token_embedding", dead_token),
    ];
    let total = cases.len();
    let mut rejected = 0;
    for (name, data) in cases {
        let r = validate_embedding(name, &data, vocab, hidden);
        assert!(
            !r.passed,
            "FAIL-CLOSED VIOLATION: apr ACCEPTED broken embedding `{name}`"
        );
        rejected += 1;
    }
    assert_eq!(
        rejected, total,
        "apr must reject ALL {total} broken embedding classes"
    );
    println!("BEAT-FAIL-CLOSED embeddings: apr rejected {rejected}/{total} broken classes");
}

#[test]
fn beat_apr_accepts_healthy_weight_no_false_positive() {
    // The dual obligation: fail-closed must NOT mean "reject everything".
    let r = validate_weight("healthy", &healthy_weight(), OUT, IN);
    assert!(
        r.passed,
        "FALSE POSITIVE: apr rejected a healthy weight — fail-closed must not block valid models. failures={:?}",
        r.failures
    );
    let emb: Vec<f32> = (0..(96 * 32))
        .map(|i| (((i % 13) as f32) - 6.0) * 0.02 + 0.01)
        .collect();
    let re = validate_embedding("healthy_emb", &emb, 96, 32);
    assert!(
        re.passed,
        "FALSE POSITIVE on healthy embedding: {:?}",
        re.failures
    );
}
