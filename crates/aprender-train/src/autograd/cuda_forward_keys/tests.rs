//! `pre_warm_keys ⊇ runtime_keys` — the property the Blackwell cascade violated
//! seven times, asserted on a CPU (YOGA-NIGHTLY-001 R-2).
//!
//! No GPU, no CUDA toolkit, no model, no second machine. It runs on intel, on
//! every PR, in microseconds. The counterpart on hardware is
//! `falsify_cuda_prewarm_covers_runtime_no_jit_001` (R-3), which counts JIT
//! compiles after a real pass; that one needs a GPU and catches the defect after
//! the fact. This one catches it before the merge.
//!
//! Every case below is a real model shape from the fleet's corpus. Two of them
//! are GQA (`num_kv_heads < num_heads`), and one has `q_dim != hidden`, because
//! both of those asymmetries produced their own pre-warm misses in the record.

use super::*;

/// Qwen2.5-0.5B — the Phase-3 smoke model.
fn qwen2_0_5b() -> ModelKeyShape {
    ModelKeyShape {
        hidden: 896,
        intermediate: 4864,
        num_heads: 14,
        num_kv_heads: 2,
        head_dim: 64,
        max_seq_len: 256,
    }
}

/// Qwen2.5-Coder-1.5B — the corpus model the fleet actually trains and serves.
fn qwen2_1_5b() -> ModelKeyShape {
    ModelKeyShape {
        hidden: 1536,
        intermediate: 8960,
        num_heads: 12,
        num_kv_heads: 2,
        head_dim: 128,
        max_seq_len: 256,
    }
}

/// Qwen3-4B — `q_dim` (4096) is NOT `hidden` (2560). Pre-warming the NF4 Q
/// projection under `hidden` here is a miss, and that is a real recorded one.
fn qwen3_4b() -> ModelKeyShape {
    ModelKeyShape {
        hidden: 2560,
        intermediate: 9728,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: 256,
    }
}

/// Llama-ish MHA: `num_kv_heads == num_heads`, so every GQA branch collapses.
fn llama_mha() -> ModelKeyShape {
    ModelKeyShape {
        hidden: 2048,
        intermediate: 5632,
        num_heads: 32,
        num_kv_heads: 32,
        head_dim: 64,
        max_seq_len: 256,
    }
}

fn shapes() -> Vec<(&'static str, ModelKeyShape)> {
    vec![
        ("qwen2.5-0.5b", qwen2_0_5b()),
        ("qwen2.5-coder-1.5b", qwen2_1_5b()),
        ("qwen3-4b", qwen3_4b()),
        ("llama-mha", llama_mha()),
    ]
}

fn spec(shape: ModelKeyShape, has_cublas: bool) -> PreWarmSpec {
    PreWarmSpec { shape, has_cublas, rope_seq_lens: vec![1, shape.max_seq_len] }
}

/// THE PROPERTY. Every key a forward pass asks for must already be in the cache.
///
/// A single missing key is a JIT compile mid-forward: silent and slow on sm_89,
/// a poisoned stream on sm_121. The assertion prints the DENOMINATOR — how many
/// runtime keys were checked — because "0 missing" over 0 keys is this fleet's
/// signature defect and would pass forever if `forward_runtime_keys` ever
/// returned nothing.
#[test]
fn prewarm_covers_every_runtime_key() {
    let mut checked = 0usize;
    let mut cases = 0usize;
    for (name, shape) in shapes() {
        for has_cublas in [true, false] {
            for eps in [QWEN2_RMS_EPS, LLAMA_RMS_EPS] {
                let warmed = prewarm_keys(&spec(shape, has_cublas));
                let wanted = forward_runtime_keys(shape, shape.max_seq_len, eps);
                assert!(
                    !wanted.is_empty(),
                    "{name}: runtime key set is EMPTY — the property is vacuous"
                );
                let missing: Vec<&String> = wanted.difference(&warmed).collect();
                assert!(
                    missing.is_empty(),
                    "{name} (cublas={has_cublas}, eps={eps:e}): {} runtime key(s) NOT pre-warmed \
                     — each one JIT-compiles mid-forward: {missing:?}",
                    missing.len()
                );
                checked += wanted.len();
                cases += 1;
            }
        }
    }
    // DERIVED, not invented: 4 shapes x 2 cuBLAS states x 2 epsilons.
    assert_eq!(cases, 4 * 2 * 2, "the case table did not run its own product");
    // THE DENOMINATOR, and its floor is derived rather than typed: every shape
    // contributes at least the 2 norms, 1 RoPE, 6 attention, 2 FFN and 4 NF4
    // projections = 15 runtime keys. Measured on this table: 284. A future
    // refactor that made `forward_runtime_keys` return almost nothing would keep
    // the property trivially true and this is what refuses that.
    assert!(
        checked >= cases * 15,
        "only {checked} runtime keys checked across {cases} cases — under 15 per case, \
         the property is close to vacuous"
    );
}

/// THE PROPERTY IS FALSIFIABLE, and this is what proves it.
///
/// PMAT-698j is reintroduced exactly as it was: `warm!` hardcoding one literal
/// key, so every kernel collides on one cache entry. If the assertion above
/// cannot see that, it is not testing anything.
#[test]
fn prewarm_defect_pmat_698j_hardcoded_key_is_caught() {
    let shape = qwen2_1_5b();
    let wanted = forward_runtime_keys(shape, shape.max_seq_len, QWEN2_RMS_EPS);
    // The defect: one key, for everything.
    let mut defective = std::collections::BTreeSet::new();
    defective.insert(fixed::SILU_FORWARD.to_string());
    let missing: Vec<&String> = wanted.difference(&defective).collect();
    assert!(
        !missing.is_empty(),
        "the hardcoded-key defect went UNDETECTED — the property cannot fail, so it \
         proves nothing about the seven PRs it exists to have prevented"
    );
    // EXACT, not a guessed floor: a cache holding ONE key can cover at most one
    // of the keys the runtime asks for, so everything else must miss. That is
    // the defect stated as arithmetic.
    assert!(
        missing.len() >= wanted.len() - 1,
        "a one-entry cache left {} of {} runtime keys covered; at most 1 is possible",
        wanted.len() - missing.len(),
        wanted.len()
    );
}

/// PMAT-698k: the pre-warm key omitted the `_eps{bits}` suffix the runtime key
/// carries. The two strings must be DIFFERENT, or the suffix is decorative.
#[test]
fn prewarm_defect_pmat_698k_missing_eps_suffix_is_caught() {
    let with_eps = batched_rmsnorm_fwd(1536, QWEN2_RMS_EPS);
    let without = "batched_rmsnorm_fwd_1536".to_string();
    assert_ne!(with_eps, without, "the eps suffix is not part of the key");
    let wanted = forward_runtime_keys(qwen2_1_5b(), 256, QWEN2_RMS_EPS);
    let mut defective = prewarm_keys(&spec(qwen2_1_5b(), true));
    defective.remove(&with_eps);
    defective.insert(without);
    assert!(
        wanted.difference(&defective).count() > 0,
        "dropping the eps suffix from the pre-warm key was not detected"
    );
}

/// PMAT-698n: pre-warmed at Llama's epsilon while running Qwen2's. The keys must
/// differ, and pre-warming only one must fail the other.
#[test]
fn prewarm_defect_pmat_698n_wrong_epsilon_is_caught() {
    assert_ne!(
        batched_rmsnorm_fwd(896, QWEN2_RMS_EPS),
        batched_rmsnorm_fwd(896, LLAMA_RMS_EPS),
        "1e-6 and 1e-5 produce the same key — the bit-pattern suffix is broken"
    );
    let shape = qwen2_0_5b();
    let mut one_eps_only = prewarm_keys(&spec(shape, true));
    one_eps_only.remove(&batched_rmsnorm_fwd(shape.hidden, QWEN2_RMS_EPS));
    one_eps_only.remove(&batched_fused_residual_rmsnorm(shape.hidden, QWEN2_RMS_EPS));
    let wanted = forward_runtime_keys(shape, 256, QWEN2_RMS_EPS);
    assert_eq!(
        wanted.difference(&one_eps_only).count(),
        2,
        "warming only the Llama epsilon left the Qwen2 norms uncovered and nothing noticed"
    );
}

/// PMAT-698p: RoPE pre-warmed at `seq_len = 1` while the corpus phase ran at
/// 256. The pre-warm spec must carry BOTH, and dropping the corpus length must
/// be visible.
#[test]
fn prewarm_defect_pmat_698p_rope_seq_len_is_caught() {
    let shape = qwen2_1_5b();
    let seq_only_1 = PreWarmSpec { shape, has_cublas: true, rope_seq_lens: vec![1] };
    let warmed = prewarm_keys(&seq_only_1);
    let wanted = forward_runtime_keys(shape, 256, QWEN2_RMS_EPS);
    let missing: Vec<&String> = wanted.difference(&warmed).collect();
    assert!(
        missing.iter().any(|k| k.starts_with("batched_rope_neox_fwd_")),
        "warming RoPE only at seq_len=1 while running at 256 was not detected: {missing:?}"
    );
}

/// FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001: the fused kernel had no pre-warm
/// entry at all.
#[test]
fn prewarm_defect_missing_fused_residual_rmsnorm_is_caught() {
    let shape = qwen2_1_5b();
    let mut warmed = prewarm_keys(&spec(shape, true));
    for eps in [QWEN2_RMS_EPS, LLAMA_RMS_EPS] {
        warmed.remove(&batched_fused_residual_rmsnorm(shape.hidden, eps));
    }
    let wanted = forward_runtime_keys(shape, 256, QWEN2_RMS_EPS);
    assert!(
        wanted.difference(&warmed).any(|k| k.starts_with("batched_fused_residual_rmsnorm_")),
        "a kernel with NO pre-warm entry was not detected as missing"
    );
}

/// cuBLAS presence changes what is pre-warmed (PMAT-700), and must not change
/// whether the property holds. If skipping the four PTX GEMMs broke coverage,
/// that skip would be a VRAM optimisation that reintroduced the cascade.
#[test]
fn cublas_skip_does_not_break_coverage() {
    for (name, shape) in shapes() {
        let with = prewarm_keys(&spec(shape, true));
        let without = prewarm_keys(&spec(shape, false));
        assert!(with.is_subset(&without), "{name}: the cuBLAS-present set is not a subset");
        let wanted = forward_runtime_keys(shape, shape.max_seq_len, QWEN2_RMS_EPS);
        assert_eq!(
            wanted.difference(&with).count(),
            0,
            "{name}: cublas=true leaves runtime keys uncovered"
        );
    }
}

/// A key constructor is a FUNCTION: same inputs, same string, every time. A key
/// that varied would miss its own cache entry.
#[test]
fn keys_are_deterministic_and_distinct() {
    assert_eq!(nf4_gemm_forward(1536, 8960), nf4_gemm_forward(1536, 8960));
    assert_ne!(
        nf4_gemm_forward(1536, 8960),
        nf4_gemm_forward(8960, 1536),
        "k and n are not ordered"
    );
    assert_ne!(
        batched_4d_gemm(1, 12, 256, 256, 128),
        batched_4d_gemm(1, 12, 256, 128, 256),
        "the Q@K^T and attn@V GEMMs collide"
    );
    assert_ne!(nf4_gemm_forward(1536, 8960), nf4_gemm_transpose(1536, 8960));
}
