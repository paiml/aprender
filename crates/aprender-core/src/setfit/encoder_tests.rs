//! `BertSentenceEncoder` conformance tests (plan 01-06).
//!
//! Everything here runs against the REAL 2-layer / hidden-64 slice weights, not
//! synthetic values: 01-03's spike already proved graph flow on synthetic
//! weights at hidden 16, and repeating that would add no evidence. What is new
//! here is the real import, the real remap, the real tokenizer boundary and the
//! real HF parameter names.
//!
//! Test names all start `encoder_` (or `mha_seeded_dropout_` for the attention
//! hook) so the plan's single positional filter selects exactly this file.

use std::path::PathBuf;

use super::*;

use crate::autograd::{self, OpError};
use crate::setfit::import::{SliceConfig, VocabRemap};
use crate::setfit::tokenizer::MiniLmTokenizer;

// ---------------------------------------------------------------------------
// Fixture plumbing
// ---------------------------------------------------------------------------

fn fixtures_dir() -> PathBuf {
    if let Ok(p) = std::env::var("APRENDER_SETFIT_FIXTURES") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixtures_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn encoder_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/setfit/encoder.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Source assertions — these hold in RED as well as GREEN, on purpose: they
// describe the SHAPE of the implementation, not its behaviour.
// ---------------------------------------------------------------------------

#[test]
fn encoder_from_import_is_sealed_to_pub_crate() {
    let src = encoder_source();
    assert!(
        src.contains("pub(crate) fn from_import("),
        "D-08: from_import must be declared pub(crate)"
    );
    assert!(
        !src.contains("pub fn from_import("),
        "D-08 seal broken: a bare `pub fn from_import(` exists in setfit/encoder.rs"
    );
}

#[test]
fn encoder_does_not_import_the_asserting_bert_embeddings_path() {
    let src = encoder_source();
    // models/bert/embeddings.rs asserts on over-length input and then slices
    // unchecked. D-01 keeps that path out of this encoder entirely.
    assert!(
        !src.contains("models::bert::embeddings"),
        "encoder.rs reaches into the asserting BERT embeddings path"
    );
}

#[test]
fn encoder_dropout_probability_agrees_with_the_enc01_pin() {
    use crate::setfit::import::{PINNED_ATTENTION_DROPOUT_PROB, PINNED_HIDDEN_DROPOUT_PROB};
    // Compared at f32, the precision the model actually computes in: 0.1f32 and
    // 0.1f64 are different numbers, so an f64 comparison would reject the pin's
    // own value (the same narrowing rule 01-05 applied to layer_norm_eps).
    #[allow(clippy::cast_possible_truncation)]
    let hidden = PINNED_HIDDEN_DROPOUT_PROB as f32;
    #[allow(clippy::cast_possible_truncation)]
    let attention = PINNED_ATTENTION_DROPOUT_PROB as f32;
    assert_eq!(super::DROPOUT_P, hidden, "hidden_dropout_prob");
    assert_eq!(super::DROPOUT_P, attention, "attention_probs_dropout_prob");
}

#[test]
fn encoder_defines_no_competing_op_error_conversion() {
    let src = encoder_source();
    // W5: SetFitError::Op + its From impl are 01-05's. A second conversion here
    // would give two ways for the same op failure to reach a caller.
    assert!(
        !src.contains("impl From<OpError>"),
        "encoder.rs defines a bespoke OpError conversion; use `?` on the 01-05 impl"
    );
}

// ---------------------------------------------------------------------------
// Slice-backed tests
// ---------------------------------------------------------------------------

#[cfg(feature = "conformance-fixtures")]
mod slice {
    use super::*;

    /// Seed used by every encoder built here unless a test varies it.
    const SEED: u64 = 0x0106_5E7F_1701;

    fn slice_config() -> SliceConfig {
        SliceConfig::from_json_bytes(&read_fixture("slice_config.json")).expect("slice_config.json")
    }

    fn slice_remap(vocab: usize) -> VocabRemap {
        VocabRemap::from_json_bytes(&read_fixture("vocab_remap.json"), vocab)
            .expect("vocab_remap.json")
    }

    fn slice_import() -> MiniLmImport {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        MiniLmImport::open_slice_fixture(&fixtures_dir().join("slice_model.apr"), &cfg, &remap)
            .expect("the frozen slice fixture must open")
    }

    fn encoder() -> BertSentenceEncoder {
        BertSentenceEncoder::from_import(&slice_import(), SEED).expect("encoder must build")
    }

    fn tokenizer() -> MiniLmTokenizer {
        MiniLmTokenizer::from_bytes(&read_fixture("tokenizer.json")).expect("tokenizer must build")
    }

    /// The frozen `mixed_length_pair` case: 5 valid tokens in row 0, 20 in row 1.
    fn mixed_batch() -> SentenceBatch {
        tokenizer()
            .encode_batch(&[
                "Short text.",
                "This sentence is deliberately longer so the batch holds two different \
                 lengths and padding is exercised.",
            ])
            .expect("tokenize")
    }

    fn single_batch() -> SentenceBatch {
        tokenizer()
            .encode_batch(&["A quick brown fox jumps over the lazy dog."])
            .expect("tokenize")
    }

    fn parameter_order() -> Vec<String> {
        let v: serde_json::Value =
            serde_json::from_slice(&read_fixture("gradients.json")).expect("gradients.json");
        v["parameter_order"]
            .as_array()
            .expect("parameter_order is an ordered ARRAY, not an object")
            .iter()
            .map(|s| s.as_str().expect("name").to_string())
            .collect()
    }

    fn analytically_zero() -> Vec<String> {
        let v: serde_json::Value =
            serde_json::from_slice(&read_fixture("gradients.json")).expect("gradients.json");
        v["analytically_zero"]
            .as_array()
            .expect("analytically_zero array")
            .iter()
            .map(|e| e["name"].as_str().expect("name").to_string())
            .collect()
    }

    fn l2(v: &[f32]) -> f64 {
        v.iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt()
    }

    // -----------------------------------------------------------------------
    // Naming (D-18)
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_named_parameters_match_the_frozen_parameter_order_exactly() {
        let enc = encoder();
        let got: Vec<String> = enc.named_parameters().into_iter().map(|(n, _)| n).collect();
        // Vec<String> equality: same names, same COUNT, same ORDER. Compared
        // against an ordered array rather than JSON object keys, which carry no
        // ordering guarantee.
        assert_eq!(
            got,
            parameter_order(),
            "HF dotted names must equal torch named_parameters() verbatim and in order"
        );
    }

    #[test]
    fn encoder_named_parameters_exclude_the_pooler() {
        let enc = encoder();
        for (name, _) in enc.named_parameters() {
            assert!(
                !name.starts_with("pooler"),
                "pooler.* must not be registered: {name}"
            );
        }
    }

    #[test]
    fn encoder_named_parameters_agree_with_positional_in_arity_and_order() {
        let enc = encoder();
        let positional = enc.parameters();
        let named = enc.named_parameters();
        assert_eq!(named.len(), positional.len(), "named/positional arity");
        for (i, ((name, nt), pt)) in named.iter().zip(positional.iter()).enumerate() {
            assert_eq!(
                nt.id(),
                pt.id(),
                "slot {i} (`{name}`) refers to a different tensor in the two traversals"
            );
        }
        let mut unique: Vec<&String> = named.iter().map(|(n, _)| n).collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), named.len(), "duplicate parameter name");
    }

    #[test]
    fn encoder_named_parameters_mut_mirrors_named_parameters() {
        let mut enc = encoder();
        let names: Vec<String> = enc.named_parameters().into_iter().map(|(n, _)| n).collect();
        let mut_names: Vec<String> = enc
            .named_parameters_mut()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, mut_names);
    }

    // -----------------------------------------------------------------------
    // Sequence bound
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_max_seq_is_the_min_of_the_sentence_bound_and_the_position_table() {
        let enc = encoder();
        // The slice has only 64 position rows, so a hardcoded `<= 256` would
        // admit an out-of-range position gather.
        assert_eq!(enc.max_seq(), 64, "min(256, max_position_embeddings=64)");
        assert!(
            enc.max_seq() < crate::setfit::tokenizer::MAX_SEQUENCE_LENGTH,
            "this test is only meaningful while the two bounds differ"
        );
    }

    // -----------------------------------------------------------------------
    // Forward shape and graph connectivity
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_forward_tokens_returns_a_graph_connected_batch() {
        autograd::clear_graph();
        let enc = encoder();
        let batch = mixed_batch();
        let out = enc.forward_tokens(&batch).expect("forward");
        assert_eq!(out.shape(), &[batch.batch(), batch.seq(), 64]);
        assert!(
            out.requires_grad_enabled(),
            "the encoder weights require grad, so the output must too"
        );
    }

    #[test]
    fn encoder_forward_tokens_per_layer_returns_embeddings_plus_one_output_per_layer() {
        autograd::clear_graph();
        let enc = encoder();
        let batch = mixed_batch();
        let (embeddings_out, layer_outputs) =
            enc.forward_tokens_per_layer(&batch).expect("per-layer");

        let want = &[batch.batch(), batch.seq(), 64][..];
        assert_eq!(embeddings_out.shape(), want);
        // The fixture records exactly one layer_outputs entry per encoder layer.
        let fixture: serde_json::Value =
            serde_json::from_slice(&read_fixture("forward_per_layer.json")).expect("fixture");
        let fixture_layers = fixture["cases"][0]["layer_outputs"]
            .as_array()
            .expect("layer_outputs")
            .len();
        assert_eq!(layer_outputs.len(), fixture_layers, "layer count");
        assert_eq!(layer_outputs.len(), 2, "the slice has 2 encoder layers");

        for (i, t) in layer_outputs.iter().enumerate() {
            assert_eq!(t.shape(), want, "layer {i} shape");
            assert!(
                t.requires_grad_enabled(),
                "layer {i} output is detached from the graph"
            );
        }
        assert!(embeddings_out.requires_grad_enabled());
    }

    #[test]
    fn encoder_forward_tokens_is_bitwise_identical_to_the_last_per_layer_output() {
        // EVAL MODE is mandatory here: these are two separate calls, so in train
        // mode the seeded dropout RNG advances between them and the comparison
        // would fail for a reason unrelated to divergence. Weakening this test
        // to a tolerance would destroy the one structural proof that both public
        // entry points route through the single `forward_layers`.
        autograd::clear_graph();
        let mut enc = encoder();
        enc.set_training(false);
        let batch = mixed_batch();

        let direct = enc.forward_tokens(&batch).expect("forward");
        let (_, per_layer) = enc.forward_tokens_per_layer(&batch).expect("per-layer");
        let last = per_layer.last().expect("at least one layer");

        assert_eq!(direct.shape(), last.shape());
        for (i, (a, b)) in direct.data().iter().zip(last.data().iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: forward_tokens gave {a}, layer_outputs.last() gave {b} — \
                 the two entry points are running DIFFERENT forward implementations"
            );
        }
    }

    #[test]
    fn encoder_has_exactly_one_layer_loop() {
        let src = encoder_source();
        let needle = "for layer in &self.layers {";
        assert_eq!(
            src.matches(needle).count(),
            1,
            "the compute layer loop must appear exactly once, inside forward_layers"
        );
        let loop_at = src.find(needle).expect("the loop");
        let impl_at = src
            .find("fn forward_layers(")
            .expect("forward_layers must exist");
        assert!(
            loop_at > impl_at,
            "the single layer loop must live inside forward_layers"
        );
    }

    #[test]
    fn encoder_forward_tokens_per_layer_is_public_and_conformance_gated() {
        let src = encoder_source();
        assert_eq!(
            src.matches("pub fn forward_tokens_per_layer").count(),
            1,
            "01-08 is out-of-crate and reaches this through SetFitMiniLm::encoder()"
        );
        let at = src
            .find("pub fn forward_tokens_per_layer")
            .expect("declaration");
        let head = &src[..at];
        assert!(
            head.rfind("#[cfg(feature = \"conformance-fixtures\")]")
                .is_some_and(|g| head[g..].matches("fn ").count() == 0),
            "forward_tokens_per_layer must be immediately preceded by the \
             conformance-fixtures gate"
        );
    }

    #[test]
    fn encoder_uses_the_exact_erf_gelu() {
        let src = encoder_source();
        assert!(src.contains("gelu_exact"), "the FFN must call gelu_exact");
        assert!(
            !src.contains(".gelu()"),
            "the tanh gelu is a DIFFERENT function (4.73e-4 apart) and is rejected by ENC-01"
        );
    }

    // -----------------------------------------------------------------------
    // Boundary rejection matrix (T-1-11, T-1-21)
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_rejects_a_batch_from_a_foreign_tokenizer_through_both_entry_points() {
        let enc = encoder();
        let mut batch = mixed_batch();
        batch.tokenizer_sha256 = "0".repeat(64);

        let err = enc.forward_tokens(&batch).expect_err("must reject");
        assert!(
            matches!(err, SetFitError::TokenizerHashMismatch { .. }),
            "got {err:?}"
        );
        // Through the per-layer entry point too: validation must live in the
        // SHARED path, not be duplicated into whichever caller remembered it.
        let err = enc
            .forward_tokens_per_layer(&batch)
            .expect_err("must reject");
        assert!(
            matches!(err, SetFitError::TokenizerHashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn encoder_rejects_a_canonical_id_outside_the_slice_closure() {
        let enc = encoder();
        let mut batch = mixed_batch();
        // 30522 is one past the pinned BERT vocabulary, so it is in no closure.
        batch.input_ids[3] = 30_522;
        let err = enc.forward_tokens(&batch).expect_err("must reject");
        assert_eq!(
            err,
            SetFitError::VocabOutOfSlice {
                canonical_id: 30_522
            },
            "a zero row would be indistinguishable from a legitimate embedding"
        );
    }

    #[test]
    fn encoder_rejects_a_mask_length_mismatch() {
        let enc = encoder();
        let mut batch = mixed_batch();
        batch.attention_mask.pop();
        let err = enc.forward_tokens(&batch).expect_err("must reject");
        match err {
            SetFitError::BatchInvalid { reason } => {
                assert!(reason.contains("attention_mask"), "got {reason}");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn encoder_rejects_an_all_padding_row() {
        let enc = encoder();
        let mut batch = mixed_batch();
        let seq = batch.seq();
        for v in &mut batch.attention_mask[..seq] {
            *v = 0;
        }
        let err = enc.forward_tokens(&batch).expect_err("must reject");
        assert_eq!(err, SetFitError::Op(OpError::AllPaddingRow { row: 0 }));
    }

    #[test]
    fn encoder_rejects_a_sequence_over_max_seq_even_when_it_is_under_256() {
        let enc = encoder();
        // ~100 real tokens: over the slice's 64 position rows, under the
        // tokenizer's 256 bound, so only the min() form rejects it.
        let long = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            .repeat(10)
            .trim_end()
            .to_string();
        let batch = tokenizer()
            .encode_batch(&[long.as_str()])
            .expect("tokenize");
        assert!(
            batch.seq() > enc.max_seq() && batch.seq() <= 256,
            "the probe must sit strictly between max_seq ({}) and 256, got {}",
            enc.max_seq(),
            batch.seq()
        );
        let err = enc.forward_tokens(&batch).expect_err("must reject");
        assert_eq!(
            err,
            SetFitError::OversizeInput {
                len: batch.seq(),
                max: 64
            }
        );
    }

    #[test]
    fn encoder_accepts_a_single_sentence_batch() {
        // The rejection matrix above would be satisfied by an encoder that
        // refuses everything. This is the other side of the line.
        autograd::clear_graph();
        let enc = encoder();
        let batch = single_batch();
        let out = enc.forward_tokens(&batch).expect("must accept");
        assert_eq!(out.shape(), &[1, batch.seq(), 64]);
    }

    // -----------------------------------------------------------------------
    // Remap
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_remaps_canonical_ids_internally_without_mutating_the_batch() {
        autograd::clear_graph();
        let enc = encoder();
        let batch = mixed_batch();
        let before = batch.input_ids().to_vec();
        // Canonical ids run far above the 97-row slice table (14108 appears in
        // the frozen mixed_length_pair case), so a forward that succeeds proves
        // the remap ran; without it embedding_gather returns OutOfVocabulary.
        assert!(
            before.iter().any(|id| *id as usize >= 97),
            "the fixture batch must carry ids above the slice vocabulary"
        );
        enc.forward_tokens(&batch).expect("forward");
        assert_eq!(batch.input_ids(), &before[..], "the batch was mutated");
    }

    // -----------------------------------------------------------------------
    // ENC-04 gradient flow on REAL slice weights
    // -----------------------------------------------------------------------

    /// One scalar loss over the whole mixed-length batch, plus the per-name
    /// gradients it produced.
    fn mixed_batch_gradients() -> Vec<(String, Vec<f32>)> {
        autograd::clear_graph();
        let mut enc = encoder();
        // Eval mode: dropout inert, so the measurement is deterministic and
        // reproduces the mode the fixtures were generated in (D-16).
        enc.set_training(false);
        let batch = mixed_batch();
        let tokens = enc.forward_tokens(&batch).expect("forward");

        // A weighted sum rather than a plain sum: a plain sum can cancel
        // structurally and hand back a zero gradient for reasons unrelated to
        // graph connectivity.
        let n = tokens.numel();
        let sel: Vec<f32> = (0..n).map(|i| 0.31 + 0.017 * (i % 13) as f32).collect();
        let loss = tokens
            .mul(&crate::autograd::Tensor::new(&sel, tokens.shape()))
            .sum();
        assert!(loss.item().is_finite(), "loss is {}", loss.item());
        loss.backward();

        enc.named_parameters()
            .into_iter()
            .map(|(name, t)| {
                let g = autograd::get_grad(t.id()).unwrap_or_else(|| {
                    panic!("`{name}` received NO gradient — the graph is severed upstream of it")
                });
                assert_eq!(g.numel(), t.numel(), "`{name}` gradient arity");
                (name, g.data().to_vec())
            })
            .collect()
    }

    #[test]
    fn encoder_backward_gives_a_finite_gradient_to_every_named_parameter() {
        let grads = mixed_batch_gradients();
        assert_eq!(grads.len(), 37, "the slice has 37 registered tensors");
        for (name, g) in &grads {
            if let Some(pos) = g.iter().position(|v| !v.is_finite()) {
                panic!(
                    "`{name}`: non-finite gradient at element {pos} ({})",
                    g[pos]
                );
            }
        }
    }

    #[test]
    fn encoder_backward_gives_a_non_zero_aggregate_gradient_to_every_enc04_component() {
        let grads = mixed_batch_gradients();
        // Per COMPONENT, not per tensor: the key biases are analytically zero,
        // so "every tensor has a non-zero gradient" is unsatisfiable against a
        // correct implementation (01-03 T-1-16).
        let component = |name: &str| -> String {
            if name.starts_with("embeddings") {
                return "embeddings".to_string();
            }
            let layer = name.split('.').nth(2).unwrap_or("?").to_string();
            if name.contains(".attention.self.") || name.contains(".attention.output.dense") {
                format!("layer{layer}.attention")
            } else if name.contains("LayerNorm") {
                format!("layer{layer}.norm")
            } else {
                format!("layer{layer}.ffn")
            }
        };

        let mut components: Vec<String> = grads.iter().map(|(n, _)| component(n)).collect();
        components.sort();
        components.dedup();
        assert_eq!(
            components.len(),
            1 + 2 * 3,
            "expected embeddings + {{attention, ffn, norm}} x 2 layers, got {components:?}"
        );

        for c in &components {
            let acc: f64 = grads
                .iter()
                .filter(|(n, _)| component(n) == *c)
                .flat_map(|(_, g)| g.iter())
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum();
            assert!(
                acc.sqrt() > 1e-9,
                "component `{c}` aggregate gradient L2 is {:e} — gradient is not reaching it",
                acc.sqrt()
            );
        }
    }

    #[test]
    fn encoder_key_biases_are_near_zero_while_the_other_biases_carry_real_gradient() {
        let grads = mixed_batch_gradients();
        let zero_names = analytically_zero();
        assert_eq!(zero_names.len(), 2, "one key bias per layer");

        let mut checked = 0;
        for (name, g) in &grads {
            if !zero_names.contains(name) {
                continue;
            }
            checked += 1;
            for (i, v) in g.iter().enumerate() {
                assert!(
                    v.abs() <= 1e-5,
                    "`{name}`[{i}] = {v:e}: the key bias is analytically zero by softmax \
                     shift invariance. A value this large means the constant shift is NOT \
                     cancelling — suspect the mask, the softmax, or the head-axis broadcast."
                );
            }
        }
        assert_eq!(checked, 2);

        // SECOND SIDE. Near-zero is only evidence if the other biases are not:
        // a backward returning zeros for every bias would sail through above.
        let l2_of = |suffix: &str| -> f64 {
            grads
                .iter()
                .filter(|(n, _)| n.ends_with(suffix))
                .map(|(_, g)| l2(g))
                .sum()
        };
        let k = l2_of("attention.self.key.bias");
        let q = l2_of("attention.self.query.bias");
        let v = l2_of("attention.self.value.bias");
        assert!(
            q > 1e-4 && v > 1e-4,
            "query ({q:e}) and value ({v:e}) biases must carry REAL gradient, else the \
             key-bias assertion is vacuous"
        );
        assert!(
            q > k * 1e3,
            "the key bias ({k:e}) must be orders below the query bias ({q:e})"
        );
    }

    // -----------------------------------------------------------------------
    // Task 2: encode() pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_encode_produces_unit_norm_rows() {
        autograd::clear_graph();
        let enc = encoder();
        let batch = mixed_batch();
        let e = enc.encode(&batch).expect("encode");
        assert_eq!(e.shape(), &[batch.batch(), 64]);
        for row in 0..batch.batch() {
            let n = l2(&e.data()[row * 64..(row + 1) * 64]);
            assert!(
                (n - 1.0).abs() < 4.0 * f64::from(f32::EPSILON),
                "row {row} has L2 norm {n}, not 1.0"
            );
        }
    }

    #[test]
    fn encoder_encode_is_graph_connected() {
        autograd::clear_graph();
        let enc = encoder();
        let e = enc.encode(&mixed_batch()).expect("encode");
        assert!(
            e.requires_grad_enabled(),
            "pooling or normalization severed the graph"
        );
    }

    #[test]
    fn encoder_encode_adds_no_third_forward_path() {
        let src = encoder_source();
        let at = src.find("pub fn encode(").expect("encode");
        let body_end = src[at..].find("\n    }").expect("end of encode") + at;
        let body = &src[at..body_end];
        assert!(
            body.contains("self.forward_tokens("),
            "encode must reach the layer stack through forward_tokens"
        );
        assert!(
            !body.contains("self.layers"),
            "encode must not iterate the layers itself"
        );
    }

    #[test]
    fn encoder_encode_is_bitwise_deterministic_in_eval_mode() {
        autograd::clear_graph();
        let mut enc = encoder();
        enc.set_training(false);
        let batch = mixed_batch();

        let a = autograd::no_grad(|| enc.encode(&batch)).expect("encode");
        let b = autograd::no_grad(|| enc.encode(&batch)).expect("encode");
        for (i, (x, y)) in a.data().iter().zip(b.data().iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "element {i}: eval-mode encode is not deterministic ({x} vs {y})"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Task 2: dropout policy and the ENC-05 mode contract
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_mode_dropout_sites_are_one_plus_three_per_layer_and_unique() {
        let enc = encoder();
        let sites = enc.dropout_sites();
        // 1 embeddings + {attention probs, attention output, FFN output} per
        // layer. The slice has 2 layers, so 7. Read off the ACTIVE modules, not
        // re-derived from a name list.
        assert_eq!(
            sites.len(),
            1 + 3 * enc.layers.len(),
            "expected 1 + 3*{} sites, got {sites:?}",
            enc.layers.len()
        );
        assert_eq!(sites.len(), 7, "the slice has 2 encoder layers");
        let mut unique = sites.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            sites.len(),
            "duplicate site name in {sites:?}"
        );
        // HF-faithful names, so a reader can find the site in the reference.
        assert_eq!(sites[0], "embeddings.dropout");
        assert!(sites.contains(&"encoder.layer.1.attention.self.dropout".to_string()));
        assert!(sites.contains(&"encoder.layer.1.attention.output.dropout".to_string()));
        assert!(sites.contains(&"encoder.layer.1.output.dropout".to_string()));
    }

    #[test]
    fn encoder_mode_every_site_has_its_own_stream() {
        let enc = encoder();
        // Derived from the DOTTED NAME, so no two sites share a mask even though
        // they share a root seed. Sharing would make the whole policy one draw.
        //
        // MIGRATED (03-02): 01-06 read the per-site u64 SEED off the attention
        // module. That API is gone; the stream identity now IS the derived Philox
        // key, so this reads the key. Strictly stronger — a seed collision and a
        // key collision are the same defect, but the key is what the mask
        // actually depends on, whereas the seed was one derivation removed.
        assert!(
            enc.layers
                .iter()
                .all(|l| l.attention.has_attention_dropout_masks()),
            "every layer must have an attention-probs mask source"
        );
        let mut lanes: Vec<[u32; 2]> = enc
            .layers
            .iter()
            .map(|l| l.attention_probs_dropout.key().lanes())
            .collect();
        assert_eq!(lanes.len(), enc.layers.len(), "every layer must be keyed");
        lanes.sort_unstable();
        lanes.dedup();
        assert_eq!(lanes.len(), enc.layers.len(), "two layers share a stream");
    }

    /// Read the training flag off every dropout-bearing module DIRECTLY.
    ///
    /// Not a behavioural proxy: this is the assertion that fails if
    /// `set_training` stops recursing into any one site.
    fn site_modes(enc: &BertSentenceEncoder) -> Vec<(String, bool)> {
        let mut out = vec![(
            "embeddings.dropout".to_string(),
            enc.embeddings_dropout.training(),
        )];
        for (i, layer) in enc.layers.iter().enumerate() {
            out.push((
                format!("encoder.layer.{i}.attention.self.dropout"),
                layer.attention.training(),
            ));
            out.push((
                format!("encoder.layer.{i}.attention.output.dropout"),
                layer.attention_output_dropout.training(),
            ));
            out.push((
                format!("encoder.layer.{i}.output.dropout"),
                layer.output_dropout.training(),
            ));
        }
        out
    }

    #[test]
    fn encoder_mode_set_training_flips_every_dropout_site_recursively() {
        let mut enc = encoder();

        enc.set_training(true);
        assert!(enc.training());
        for (site, on) in site_modes(&enc) {
            assert!(on, "`{site}` did not follow set_training(true)");
        }

        enc.set_training(false);
        assert!(!enc.training());
        for (site, on) in site_modes(&enc) {
            assert!(!on, "`{site}` did not follow set_training(false)");
        }
    }

    #[test]
    fn encoder_mode_train_and_eval_spellings_also_propagate() {
        // The crate convention is that train()/eval() are leaf-local and
        // set_training is the channel. On a module whose whole point is dropout
        // placement, an eval() that left dropout active would make every
        // "inference" run stochastic with no error, so both spellings route
        // through the one channel here.
        let mut enc = encoder();
        enc.train();
        for (site, on) in site_modes(&enc) {
            assert!(on, "`{site}` did not follow train()");
        }
        enc.eval();
        for (site, on) in site_modes(&enc) {
            assert!(!on, "`{site}` did not follow eval()");
        }
    }

    #[test]
    fn encoder_mode_every_site_is_actually_applied_in_the_forward() {
        // `dropout_sites()` proves each site EXISTS and is active. It cannot
        // prove the forward pass ever calls it — a site constructed, mode-flipped
        // and then never applied would satisfy both the count test and the
        // recursion test. This turns exactly one site on at a time against an
        // otherwise eval-mode encoder: if that site is not on the forward path,
        // the output does not move.
        autograd::clear_graph();
        let batch = mixed_batch();
        let run = |enc: &BertSentenceEncoder| -> Vec<f32> {
            autograd::no_grad(|| enc.forward_tokens(&batch))
                .expect("forward")
                .data()
                .to_vec()
        };

        let names = encoder().dropout_sites();
        assert_eq!(names.len(), 7);
        for (index, name) in names.iter().enumerate() {
            let mut enc = encoder();
            enc.set_training(false);
            let base = run(&enc);

            match index {
                0 => enc.embeddings_dropout.set_training(true),
                _ => {
                    let layer = (index - 1) / 3;
                    match (index - 1) % 3 {
                        0 => enc.layers[layer].attention.set_training(true),
                        1 => enc.layers[layer]
                            .attention_output_dropout
                            .set_training(true),
                        _ => enc.layers[layer].output_dropout.set_training(true),
                    }
                }
            }

            let moved = run(&enc);
            assert!(
                base.iter()
                    .zip(moved.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "turning `{name}` on changed nothing — the site is constructed and \
                 mode-aware but never applied in forward_layers"
            );
        }
    }

    #[test]
    fn encoder_mode_parameters_are_byte_identical_across_train_eval_train() {
        let mut enc = encoder();
        // ENC-05: reuses 01-02's shared helper rather than a local copy.
        let before = crate::nn::tests_named_module::snapshot_named(&enc);
        enc.set_training(true);
        let train = crate::nn::tests_named_module::snapshot_named(&enc);
        enc.set_training(false);
        let eval = crate::nn::tests_named_module::snapshot_named(&enc);
        enc.set_training(true);
        let again = crate::nn::tests_named_module::snapshot_named(&enc);

        assert_eq!(before, train, "train() mutated a registered parameter");
        assert_eq!(train, eval, "eval() mutated a registered parameter");
        assert_eq!(eval, again, "train() mutated a registered parameter");
        assert_eq!(again.len(), 37);
    }

    #[test]
    fn encoder_mode_no_seed_or_rng_state_is_registered_as_a_parameter() {
        // Pitfall 7: naming RNG state would put non-learnable values into
        // optimizer and freeze partitions AND break the byte-identity proof
        // above, since RNG state legitimately changes across a forward pass.
        let enc = encoder();
        for (name, _) in enc.named_parameters() {
            assert!(
                !name.contains("seed") && !name.contains("rng") && !name.contains("dropout"),
                "module state leaked into named_parameters: {name}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Task 2: seeded determinism (the A5 discharge)
    // -----------------------------------------------------------------------

    fn train_mode_tokens(seed: u64) -> Vec<f32> {
        autograd::clear_graph();
        let mut enc =
            BertSentenceEncoder::from_import(&slice_import(), seed).expect("encoder must build");
        enc.set_training(true);
        autograd::no_grad(|| enc.forward_tokens(&mixed_batch()))
            .expect("forward")
            .data()
            .to_vec()
    }

    #[test]
    fn encoder_mode_same_root_seed_gives_bitwise_identical_train_mode_output() {
        let a = train_mode_tokens(SEED);
        let b = train_mode_tokens(SEED);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "element {i}: two encoders with root seed {SEED:#x} disagree ({x} vs {y}) — \
                 at least one dropout site is drawing from the ambient RNG"
            );
        }
    }

    #[test]
    fn encoder_mode_a_different_root_seed_gives_a_different_train_mode_output() {
        // The other side of the line: identical outputs would also be produced
        // by an encoder whose dropout never fires at all.
        let a = train_mode_tokens(SEED);
        let b = train_mode_tokens(SEED ^ 0xdead_beef);
        assert!(
            a.iter()
                .zip(b.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "changing the root seed changed nothing — dropout is inert in train mode"
        );
    }

    #[test]
    fn encoder_mode_train_mode_differs_from_eval_mode() {
        // Without this, every determinism assertion above is satisfied by an
        // encoder that simply never drops anything.
        autograd::clear_graph();
        let mut enc = encoder();
        enc.set_training(false);
        let batch = mixed_batch();
        let evaluated = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();
        enc.set_training(true);
        let trained = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();
        assert!(
            evaluated
                .iter()
                .zip(trained.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "train and eval produced the same states — no dropout site is active"
        );
    }

    #[test]
    fn encoder_mode_the_seeded_stream_advances_across_forward_passes() {
        // MIGRATED (03-02, D-15). 01-06's seeded sites advanced an INTERNAL
        // per-call counter, so two consecutive passes differed by construction.
        // Under D-15 the coordinate is the caller-supplied forward ordinal
        // `2*step + branch`, so "the stream advances" now means: advancing the
        // ordinal changes the mask. A site that ignored the ordinal would replay
        // one fixed mask every step — reproducible, and no longer dropout.
        autograd::clear_graph();
        let mut enc = encoder();
        enc.set_training(true);
        let batch = mixed_batch();
        let first = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();
        enc.set_forward_ordinal(1).expect("ordinal 1 fits u32");
        let second = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();
        assert!(
            first
                .iter()
                .zip(second.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "advancing the forward ordinal changed nothing — the ordinal is not \
             reaching the dropout sites, so the same mask is replayed every step"
        );

        // The half 01-06 could not state: at the SAME ordinal a repeat pass is
        // BITWISE identical. That is TRN-06's two-clean-runs guarantee, and an
        // internal counter made it structurally impossible.
        enc.set_forward_ordinal(0).expect("ordinal 0 fits u32");
        let replayed = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();
        for (i, (a, b)) in first.iter().zip(replayed.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: returning to ordinal 0 did not replay the mask — a \
                 dropout site is carrying hidden state"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 03-02 Task 2: the two ENCODER-LEVEL gates on the counter-based masks
    //
    // `dropout_rng`'s own tests prove the derivation. These two prove it reaches
    // a REAL forward pass over the slice weights — a mask source can be perfect
    // and still be wired to nothing, which is exactly the failure `dropout_sites`
    // alone cannot see.
    // -----------------------------------------------------------------------

    #[test]
    fn encoder_dropout_rng_same_forward_ordinal_replays_bitwise() {
        autograd::clear_graph();
        let mut enc = encoder();
        enc.set_training(true);
        let batch = mixed_batch();

        let ordinal = crate::setfit::dropout_rng::forward_ordinal(21, 0).expect("2*21 fits u32");
        enc.set_forward_ordinal(u64::from(ordinal))
            .expect("ordinal fits");
        let first = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();

        // Move away and come back: a site holding hidden state would not land on
        // the same mask, and "twice in a row is the same" would not catch it.
        enc.set_forward_ordinal(999).expect("ordinal fits");
        let _ = autograd::no_grad(|| enc.forward_tokens(&batch)).expect("forward");
        enc.set_forward_ordinal(u64::from(ordinal))
            .expect("ordinal fits");
        let again = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();

        assert_eq!(first.len(), again.len());
        for (i, (a, b)) in first.iter().zip(again.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: a real train-mode forward at forward ordinal {ordinal} \
                 did not replay bitwise ({a} vs {b}) — TRN-06's two-clean-runs \
                 guarantee is not reachable from here"
            );
        }
    }

    #[test]
    fn encoder_dropout_rng_the_two_branches_of_one_step_differ() {
        // D-15 at the level that matters. `pair_cosine_mse(za, zb, labels)` takes
        // TWO [B,H] matrices, so ONE training step runs TWO encoder forwards. If
        // both branches drew the same masks the pair objective would see an
        // artificial correlation between its two halves — and every determinism
        // test in this file would still be green.
        autograd::clear_graph();
        let mut enc = encoder();
        enc.set_training(true);
        let batch = mixed_batch();
        let step = 21_u64;

        let a_ordinal = crate::setfit::dropout_rng::forward_ordinal(step, 0).expect("fits");
        let b_ordinal = crate::setfit::dropout_rng::forward_ordinal(step, 1).expect("fits");
        assert_eq!(u64::from(a_ordinal) + 1, u64::from(b_ordinal));

        enc.set_forward_ordinal(u64::from(a_ordinal)).expect("fits");
        let branch_a = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();
        enc.set_forward_ordinal(u64::from(b_ordinal)).expect("fits");
        let branch_b = autograd::no_grad(|| enc.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();

        let differing = branch_a
            .iter()
            .zip(branch_b.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differing * 2 > branch_a.len(),
            "only {differing} of {} activations differ between the A and B branches \
             of step {step} — the two siamese forwards are sharing a dropout \
             stream, so the pair objective's halves are correlated",
            branch_a.len()
        );
    }

    #[test]
    fn encoder_mode_eval_passes_do_not_consume_the_dropout_stream() {
        // Inference between training steps must not shift the stream, or a
        // reproducible run would depend on how many times it was evaluated.
        autograd::clear_graph();
        let batch = mixed_batch();

        let mut a =
            BertSentenceEncoder::from_import(&slice_import(), SEED).expect("encoder must build");
        a.set_training(true);
        let baseline = autograd::no_grad(|| a.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();

        let mut b =
            BertSentenceEncoder::from_import(&slice_import(), SEED).expect("encoder must build");
        b.set_training(false);
        let _ = autograd::no_grad(|| b.forward_tokens(&batch)).expect("forward");
        let _ = autograd::no_grad(|| b.forward_tokens(&batch)).expect("forward");
        b.set_training(true);
        let after_eval = autograd::no_grad(|| b.forward_tokens(&batch))
            .expect("forward")
            .data()
            .to_vec();

        for (i, (x, y)) in baseline.iter().zip(after_eval.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "element {i}: two eval passes shifted the training dropout stream"
            );
        }
    }

    #[test]
    fn encoder_encode_dropout_drop_rate_sits_in_the_expected_band() {
        // Statistics test, Rust-side only (D-16): p = 0.1 means ~10% of a large
        // tensor's elements are zeroed. Measured over 100k elements so the band
        // is not sampling noise.
        //
        // MIGRATED (03-02): measured on the SITE THE ENCODER ACTUALLY USES.
        // 01-06 probed `nn::Dropout::with_seed(0.1, site_seed(..))`, which after
        // this plan is no longer on the SetFit path at all — the measurement
        // would have been of a module the encoder never calls.
        let n = 100_000;
        let x = crate::autograd::Tensor::from_vec(vec![1.0f32; n], &[n]);
        let d = super::super::dropout_rng::SiteDropout::new(SEED, "embeddings.dropout", 0.1)
            .expect("0.1 is a valid dropout rate");
        let y = d.forward(&x);
        let dropped = y.data().iter().filter(|v| **v == 0.0).count();
        #[allow(clippy::cast_precision_loss)]
        let rate = dropped as f64 / n as f64;
        assert!(
            (0.08..=0.12).contains(&rate),
            "empirical drop rate {rate} is outside [0.08, 0.12] for p = 0.1"
        );
    }
}
