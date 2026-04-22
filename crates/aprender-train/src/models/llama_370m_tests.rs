//! Unit tests for `llama_370m` (extracted from `llama_370m.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "llama_370m_tests.rs"] mod tests;` in the parent.

use super::*;

/// INV-ARCH-370M-002/003/004/005/006/008 — byte-equality with contract.
#[test]
fn config_matches_contract_values() {
    // §architecture
    assert_eq!(Llama370MConfig::HIDDEN_DIM, 1024);
    assert_eq!(Llama370MConfig::NUM_LAYERS, 24);
    assert_eq!(Llama370MConfig::NUM_HEADS, 16);
    assert_eq!(Llama370MConfig::NUM_KV_HEADS, 4);
    assert_eq!(Llama370MConfig::HEAD_DIM, 64);
    assert_eq!(Llama370MConfig::INTERMEDIATE_DIM, 2816);
    assert_eq!(Llama370MConfig::VOCAB_SIZE, 50_257);
    assert_eq!(Llama370MConfig::MAX_POSITION_EMBEDDINGS, 4096);
    assert!((Llama370MConfig::ROPE_THETA - 10_000.0_f32).abs() < 1e-6);
    assert!((Llama370MConfig::RMS_NORM_EPS - 1.0e-5_f32).abs() < 1e-9);

    // §constraints
    assert!(Llama370MConfig::TIED_EMBEDDINGS);
    assert!(!Llama370MConfig::HAS_BIAS);

    // Derived: INV-ARCH-370M-002 & 003
    assert_eq!(Llama370MConfig::NUM_HEADS * Llama370MConfig::HEAD_DIM, Llama370MConfig::HIDDEN_DIM,);
    assert_eq!(Llama370MConfig::NUM_HEADS % Llama370MConfig::NUM_KV_HEADS, 0);
}

/// GATE-ARCH-370M-011 / INV-ARCH-370M-006 — pure vocab-parity helper
/// MUST reject any mismatch between tokenizer vocab_size and model
/// vocab_size, and MUST accept equal values. The real-compute MODEL-2
/// dispatch at commit 29607ed33 surfaced this when a tokenizer at
/// vocab=50_257 was paired with a model pinned at VOCAB_SIZE=50_000;
/// the N-09 OOB escape masked the mismatch → garbage gradients.
/// Task #131 bumped VOCAB_SIZE to 50_257 (Option A); the counter-example
/// value below now exercises the opposite drift (a tokenizer one token
/// short of contract) so the helper is still exercised on real mismatch.
#[test]
fn falsify_gate_arch_370m_011_helper_rejects_mismatch() {
    assert!(assert_tokenizer_vocab_matches_model(
        Llama370MConfig::VOCAB_SIZE,
        Llama370MConfig::VOCAB_SIZE,
    )
    .is_ok());

    let mismatch = Llama370MConfig::VOCAB_SIZE - 1;
    let err = assert_tokenizer_vocab_matches_model(mismatch, Llama370MConfig::VOCAB_SIZE)
        .expect_err("mismatch must return Err");
    assert!(
        err.contains("GATE-ARCH-370M-011")
            && err.contains(&mismatch.to_string())
            && err.contains(&Llama370MConfig::VOCAB_SIZE.to_string()),
        "error must name the gate and both vocab sizes for forensics, got: {err}",
    );

    assert!(assert_tokenizer_vocab_matches_model(0, 1).is_err());
    assert!(assert_tokenizer_vocab_matches_model(
        Llama370MConfig::VOCAB_SIZE + 1,
        Llama370MConfig::VOCAB_SIZE
    )
    .is_err());
}

/// INV-ARCH-370M-001 — estimated param count within [366M, 374M].
///
/// Recomputes the canonical transformer param formula and asserts the
/// answer lies in the ±1% band the contract permits for the final
/// trained artifact.
#[test]
fn estimated_param_count_within_contract_band() {
    let p = estimated_param_count();
    let stored = estimated_stored_param_count();

    // Sanity printout for debugging drift.
    eprintln!("albor-370m nominal param count = {p} ({} M)", p / 1_000_000,);
    eprintln!("albor-370m stored  param count = {stored} ({} M, lm_head tied)", stored / 1_000_000,);

    // INV-ARCH-370M-001 — nominal ±1% band.
    assert!(
        p >= Llama370MConfig::PARAMETERS_MIN,
        "nominal param count {p} below INV-ARCH-370M-001 floor (366M)",
    );
    assert!(
        p <= Llama370MConfig::PARAMETERS_MAX,
        "nominal param count {p} above INV-ARCH-370M-001 ceiling (374M)",
    );

    // Tighter ±5% sanity band around the 370M nominal figure, per
    // this scaffold's unit-test requirements.
    let nominal = Llama370MConfig::PARAMETERS_NOMINAL as f64;
    let pct = (p as f64 - nominal).abs() / nominal;
    assert!(pct < 0.05, "nominal param count {p} differs from 370M by {:.2}% (> 5%)", pct * 100.0,);

    // Tying must reduce storage by exactly one vocab*hidden matrix.
    assert_eq!(
        p - stored,
        Llama370MConfig::VOCAB_SIZE * Llama370MConfig::HIDDEN_DIM,
        "tying accounting mismatch",
    );
}

/// Sanity: the compile-time `validate()` matches the runtime check.
#[test]
fn validate_is_a_noop_at_runtime() {
    // If `validate()` compiled, it's already been proven to not panic
    // (the `const _: () = ...;` at module scope forced evaluation at
    // compile time). Calling it again at runtime is a free
    // defence-in-depth assertion.
    Llama370MConfig::validate();
}

/// Shape newtypes are zero-sized and usable in generic contexts.
#[test]
fn shape_newtypes_compile_and_roundtrip() {
    type Hidden = HiddenDim<{ Llama370MConfig::HIDDEN_DIM }>;
    type Heads = NumHeads<{ Llama370MConfig::NUM_HEADS }>;
    type KvHeads = NumKvHeads<{ Llama370MConfig::NUM_KV_HEADS }>;
    type Head = HeadDim<{ Llama370MConfig::HEAD_DIM }>;
    type Inter = IntermediateDim<{ Llama370MConfig::INTERMEDIATE_DIM }>;
    type Layers = NumLayers<{ Llama370MConfig::NUM_LAYERS }>;
    type Vocab = VocabSize<{ Llama370MConfig::VOCAB_SIZE }>;

    assert_eq!(Hidden::VALUE, 1024);
    assert_eq!(Heads::VALUE, 16);
    assert_eq!(KvHeads::VALUE, 4);
    assert_eq!(Head::VALUE, 64);
    assert_eq!(Inter::VALUE, 2816);
    assert_eq!(Layers::VALUE, 24);
    assert_eq!(Vocab::VALUE, 50_257);

    // Zero-sized: all shape newtypes cost nothing at runtime.
    assert_eq!(std::mem::size_of::<Hidden>(), 0);
    assert_eq!(std::mem::size_of::<Heads>(), 0);
}

// ========================================================================
// C-LLAMA-370M-SOVEREIGN / AC-SHIP2-001 / FALSIFY-SHIP-011
// ========================================================================

/// The sovereign contract YAML embedded at compile time so the test
/// binary has a byte-frozen copy — any edit to the file is caught
/// by the next test run, not discovered post-publish.
const SOVEREIGN_CONTRACT_YAML: &str =
    include_str!("../../../../contracts/model-families/llama-370m-sovereign-v1.yaml");

/// GATE-ARCH-370M-001 / INV-ARCH-370M-002..008: every architectural
/// constant declared in `contracts/model-families/llama-370m-sovereign-v1.yaml`
/// matches the Rust scaffold `Llama370MConfig::*` const byte-equally.
///
/// Discharges FALSIFY-SHIP-011 (AC-SHIP2-001): architecture registered
/// in a llama-family contract entry whose dimensions validate against
/// `contracts/model-families/_schema.yaml` AND match the compile-time
/// Rust config that the training loop will actually consume. Binds the
/// YAML contract and the Rust scaffold: if either drifts without the
/// other, this test fails — catching the MODEL-1 QLoRA class of
/// recipe/artifact drift at `cargo test` time, before a single step
/// of pretraining compute runs.
#[test]
fn falsify_ship_011_rust_scaffold_matches_yaml_contract() {
    let doc: serde_yaml::Value = serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML)
        .expect("llama-370m-sovereign-v1.yaml must parse as YAML");

    // Contract identity — must be the right contract.
    assert_eq!(
        doc["contract_id"].as_str(),
        Some("C-LLAMA-370M-SOVEREIGN"),
        "wrong contract loaded — check include_str! path",
    );
    assert_eq!(doc["family"].as_str(), Some("llama"));
    assert_eq!(doc["size_variant"].as_str(), Some("370m"));

    // Architectural dimensions (INV-ARCH-370M-002, -003, -005, -006).
    let arch = &doc["architecture"];
    assert_eq!(
        arch["hidden_dim"].as_u64().map(|v| v as usize),
        Some(Llama370MConfig::HIDDEN_DIM),
        "YAML architecture.hidden_dim drifted from Rust const",
    );
    assert_eq!(arch["num_layers"].as_u64().map(|v| v as usize), Some(Llama370MConfig::NUM_LAYERS),);
    assert_eq!(arch["num_heads"].as_u64().map(|v| v as usize), Some(Llama370MConfig::NUM_HEADS),);
    assert_eq!(
        arch["num_kv_heads"].as_u64().map(|v| v as usize),
        Some(Llama370MConfig::NUM_KV_HEADS),
    );
    assert_eq!(arch["head_dim"].as_u64().map(|v| v as usize), Some(Llama370MConfig::HEAD_DIM),);
    assert_eq!(
        arch["intermediate_dim"].as_u64().map(|v| v as usize),
        Some(Llama370MConfig::INTERMEDIATE_DIM),
    );
    assert_eq!(arch["vocab_size"].as_u64().map(|v| v as usize), Some(Llama370MConfig::VOCAB_SIZE),);
    assert_eq!(
        arch["max_position_embeddings"].as_u64().map(|v| v as usize),
        Some(Llama370MConfig::MAX_POSITION_EMBEDDINGS),
    );
    let rope_theta = arch["rope_theta"].as_f64().expect("rope_theta must be a float");
    assert!(
        (rope_theta - f64::from(Llama370MConfig::ROPE_THETA)).abs() < 1e-6,
        "YAML rope_theta {rope_theta} != Rust const {}",
        Llama370MConfig::ROPE_THETA,
    );

    // Constraints (INV-ARCH-370M-004, -008).
    let constraints = &doc["constraints"];
    assert_eq!(constraints["tied_embeddings"].as_bool(), Some(Llama370MConfig::TIED_EMBEDDINGS),);
    assert_eq!(constraints["has_bias"].as_bool(), Some(Llama370MConfig::HAS_BIAS),);
    assert_eq!(constraints["attention_type"].as_str(), Some("gqa"));
    assert_eq!(constraints["activation"].as_str(), Some("silu"));
    assert_eq!(constraints["norm_type"].as_str(), Some("rmsnorm"));
    assert_eq!(constraints["positional_encoding"].as_str(), Some("rope"));
    assert_eq!(constraints["mlp_type"].as_str(), Some("swiglu"));
}

/// GATE-ARCH-370M-001 (gate status): once FALSIFY-SHIP-011 is
/// discharged, the sovereign contract MUST declare status ACTIVE —
/// a PROPOSED gate cannot be a ship-blocker.
#[test]
fn falsify_ship_011_sovereign_contract_is_active() {
    let doc: serde_yaml::Value =
        serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
    assert_eq!(
        doc["status"].as_str(),
        Some("ACTIVE"),
        "C-LLAMA-370M-SOVEREIGN must be ACTIVE once FALSIFY-SHIP-011 \
             discharges — PROPOSED contracts cannot gate a ship",
    );
}

// ========================================================================
// GATE-ARCH-370M-004 / AC-SHIP2-009 / FALSIFY-SHIP-019
// ========================================================================

/// Enumerate every APR tensor name the 370M architecture produces.
///
/// Returns `(name, expected_shape)` pairs. Ordering mirrors the
/// canonical GGUF/APR dump order: embedding → per-layer tensors
/// (24 layers × 9 tensors) → final norm. `lm_head.weight` shares
/// storage with `model.embed_tokens.weight` per INV-ARCH-370M-004
/// (tied), but the layout contract records it as a separate entry
/// because the kernel path needs a named row-major [vocab, hidden]
/// reference at decode time.
fn enumerate_370m_apr_tensors() -> Vec<(String, Vec<usize>)> {
    let h = Llama370MConfig::HIDDEN_DIM;
    let v = Llama370MConfig::VOCAB_SIZE;
    let i = Llama370MConfig::INTERMEDIATE_DIM;
    let nh = Llama370MConfig::NUM_HEADS;
    let nkv = Llama370MConfig::NUM_KV_HEADS;
    let hd = Llama370MConfig::HEAD_DIM;
    let layers = Llama370MConfig::NUM_LAYERS;

    let mut out: Vec<(String, Vec<usize>)> = Vec::with_capacity(3 + 9 * layers);
    out.push(("model.embed_tokens.weight".into(), vec![v, h]));
    out.push(("lm_head.weight".into(), vec![v, h]));
    for n in 0..layers {
        out.push((format!("model.layers.{n}.self_attn.q_proj.weight"), vec![nh * hd, h]));
        out.push((format!("model.layers.{n}.self_attn.k_proj.weight"), vec![nkv * hd, h]));
        out.push((format!("model.layers.{n}.self_attn.v_proj.weight"), vec![nkv * hd, h]));
        out.push((format!("model.layers.{n}.self_attn.o_proj.weight"), vec![h, nh * hd]));
        out.push((format!("model.layers.{n}.mlp.gate_proj.weight"), vec![i, h]));
        out.push((format!("model.layers.{n}.mlp.up_proj.weight"), vec![i, h]));
        out.push((format!("model.layers.{n}.mlp.down_proj.weight"), vec![h, i]));
        out.push((format!("model.layers.{n}.input_layernorm.weight"), vec![h]));
        out.push((format!("model.layers.{n}.post_attention_layernorm.weight"), vec![h]));
    }
    out.push(("model.norm.weight".into(), vec![h]));
    out
}

/// FALSIFY-SHIP-019 (AC-SHIP2-009) — algorithm-level PARTIAL proof
/// that every APR tensor the 370M architecture produces is covered
/// by `aprender::format::layout_contract` (the authoritative
/// row-major validator reused by every GGUF↔APR export site, per
/// spec §9 Risk #2 mitigation).
///
/// This test proves three things without needing a trained model:
///   1. **Coverage:** every 370M tensor name normalises to a
///      contract entry — no unknown-tensor silent-skip gap.
///   2. **Row-major ordering:** every 2D tensor's enumerated shape
///      is `[out_dim, in_dim]` (the row-major APR layout mandated
///      by INV-ARCH-370M-009 and by LAYOUT-001). Specifically
///      `lm_head.weight` is `[vocab, hidden]`, never reversed —
///      GH-202 root cause.
///   3. **Critical-tensor enforcement:** `validate_apr_shape` on
///      `lm_head.weight` accepts `[vocab, hidden]` AND rejects
///      `[hidden, vocab]`, proving the validator actively catches
///      the GH-202 class of layout bug.
///
/// **Discharge:** `evidence_discharged_by` on GATE-ARCH-370M-004;
/// full discharge blocks on real trained 370M artifact (need the
/// GGUF export path to actually invoke `validate_apr_shape` on
/// real tensor bytes, which requires a trained `.apr`).
#[test]
fn falsify_ship_019_layout_contract_covers_every_370m_tensor() {
    use aprender::format::layout_contract::LayoutContract;
    let contract = LayoutContract::new();
    let tensors = enumerate_370m_apr_tensors();

    // Invariant 1: the enumerator produces exactly the expected number
    // of APR entries for a 24-layer 370M Llama (1 embedding + 1 lm_head
    // + 9 per-layer + 1 final norm).
    assert_eq!(
        tensors.len(),
        3 + 9 * Llama370MConfig::NUM_LAYERS,
        "370M enumerator produced wrong tensor count — scaffold drift",
    );

    // Invariant 2: coverage — every enumerated name resolves to a
    // TensorContract entry. Pattern-normalisation collapses
    // `model.layers.<n>.*` to `model.layers.{n}.*`.
    for (name, _) in &tensors {
        assert!(
            contract.get_apr_contract(name).is_some(),
            "370M tensor `{name}` has no layout_contract entry — \
                 LAYOUT-001 coverage gap (every tensor in this model must \
                 pattern-match a TensorContract or GGUF export layout will \
                 silently skip it)",
        );
    }

    // Invariant 3: row-major ordering — every 2D tensor enumerated
    // above has shape `[out_dim, in_dim]`. The ordering is the whole
    // point of LAYOUT-001 (see layout_contract.rs §Key Principles).
    // Spot-check the pinned invariants rather than re-parsing the
    // formula strings.
    let lm =
        tensors.iter().find(|(n, _)| n == "lm_head.weight").expect("lm_head must be enumerated");
    assert_eq!(
        lm.1,
        vec![Llama370MConfig::VOCAB_SIZE, Llama370MConfig::HIDDEN_DIM],
        "lm_head.weight must be row-major [vocab, hidden] — GH-202 \
             root cause; reversed `[hidden, vocab]` produces [PAD] garbage",
    );
    let embed = tensors
        .iter()
        .find(|(n, _)| n == "model.embed_tokens.weight")
        .expect("embed_tokens must be enumerated");
    assert_eq!(
        embed.1,
        vec![Llama370MConfig::VOCAB_SIZE, Llama370MConfig::HIDDEN_DIM],
        "embed_tokens.weight must be row-major [vocab, hidden]",
    );
    // GQA: K/V projections are 4× smaller on the out_dim axis vs Q/O.
    let k0 = tensors
        .iter()
        .find(|(n, _)| n == "model.layers.0.self_attn.k_proj.weight")
        .expect("k_proj layer 0 must be enumerated");
    assert_eq!(
        k0.1,
        vec![
            Llama370MConfig::NUM_KV_HEADS * Llama370MConfig::HEAD_DIM,
            Llama370MConfig::HIDDEN_DIM,
        ],
        "k_proj must be row-major [kv_heads*head_dim, hidden] — GQA",
    );
    let q0 = tensors
        .iter()
        .find(|(n, _)| n == "model.layers.0.self_attn.q_proj.weight")
        .expect("q_proj layer 0 must be enumerated");
    assert_eq!(
        q0.1,
        vec![Llama370MConfig::NUM_HEADS * Llama370MConfig::HEAD_DIM, Llama370MConfig::HIDDEN_DIM,],
        "q_proj must be row-major [heads*head_dim, hidden]",
    );

    // Invariant 4: `validate_apr_shape` actively enforces the critical
    // tensor. Correct shape passes, reversed shape fails — the
    // validator must catch the GH-202 class of bug, not just
    // silently accept.
    contract
        .validate_apr_shape(
            "lm_head.weight",
            &[Llama370MConfig::VOCAB_SIZE, Llama370MConfig::HIDDEN_DIM],
            Llama370MConfig::VOCAB_SIZE,
            Llama370MConfig::HIDDEN_DIM,
        )
        .expect("correct [vocab, hidden] lm_head must validate");
    let bad = contract.validate_apr_shape(
        "lm_head.weight",
        &[Llama370MConfig::HIDDEN_DIM, Llama370MConfig::VOCAB_SIZE],
        Llama370MConfig::VOCAB_SIZE,
        Llama370MConfig::HIDDEN_DIM,
    );
    assert!(
        bad.is_err(),
        "reversed [hidden, vocab] lm_head MUST be rejected by the \
             layout contract — this is GH-202 regression protection",
    );
}

/// GATE-ARCH-370M-004 wiring check: once FALSIFY-SHIP-019 has an
/// algorithm-level PARTIAL discharge, the sovereign contract YAML
/// MUST record `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
/// `evidence_discharged_by` + `full_discharge_blocks_on` on
/// GATE-ARCH-370M-004. Any edit that drops those fields fails this
/// test before the artifact ships.
#[test]
fn falsify_ship_019_gate_arch_370m_004_has_partial_discharge_marker() {
    let doc: serde_yaml::Value =
        serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
    let gates = doc["gates"].as_sequence().expect("gates must be a sequence in sovereign contract");
    let gate = gates
        .iter()
        .find(|g| g["id"].as_str() == Some("GATE-ARCH-370M-004"))
        .expect("GATE-ARCH-370M-004 must exist in sovereign contract");

    assert_eq!(
        gate["falsification_id"].as_str(),
        Some("FALSIFY-SHIP-019"),
        "GATE-ARCH-370M-004 must bind FALSIFY-SHIP-019",
    );
    assert_eq!(
        gate["binds_to"].as_str(),
        Some("AC-SHIP2-009"),
        "GATE-ARCH-370M-004 must bind AC-SHIP2-009",
    );
    assert_eq!(
        gate["discharge_status"].as_str(),
        Some("PARTIAL_ALGORITHM_LEVEL"),
        "GATE-ARCH-370M-004 must advertise PARTIAL_ALGORITHM_LEVEL \
             (full discharge blocks on real trained 370M .apr)",
    );
    let evidence = gate["evidence_discharged_by"]
        .as_sequence()
        .expect("GATE-ARCH-370M-004 must have evidence_discharged_by");
    assert!(
        !evidence.is_empty(),
        "GATE-ARCH-370M-004 evidence_discharged_by must list \
             at least one test function or artifact",
    );
    assert!(
        gate["full_discharge_blocks_on"].as_str().is_some(),
        "PARTIAL gate must document full_discharge_blocks_on",
    );
    assert_eq!(
        gate["ship_blocking"].as_bool(),
        Some(true),
        "GATE-ARCH-370M-004 must advertise ship_blocking:true — the \
             gate's `verdict:pass` alone is insufficient green while \
             discharge_status == PARTIAL_ALGORITHM_LEVEL",
    );
}
