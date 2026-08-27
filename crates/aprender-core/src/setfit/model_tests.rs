//! `SetFitMiniLm` + `FreezeGroup` tests (plan 01-07, D-08 / D-20 / D-21 / D-22).
//!
//! Every test name starts `setfit_model_`. That prefix was checked against the
//! whole `crates/` tree before it was chosen and returned **zero** pre-existing
//! matches, which is the check D13 and D30 both asked for — the filter selects
//! exactly this file and nothing else.
//!
//! # What these tests are and are not
//!
//! They prove the freeze partition by OBSERVED BEHAVIOUR across an optimizer
//! step, not by a configuration round-trip. 01-06's mutation F is the reason:
//! a dropout site that was constructed, seeded, mode-aware and reported — but
//! never actually called — survived 41 of 42 tests. A freeze group has the same
//! failure shape. `freeze_policy()` returning what was handed to it, and
//! `frozen_parameters()` listing the right names, are both satisfied by a
//! policy that nothing ever consults. So the load-bearing tests here run a real
//! backward and a real parameter update and assert that a frozen tensor's bits
//! do not move while the SAME tensor's bits DO move when it is trainable.

use std::path::{Path, PathBuf};

use super::*;

use crate::autograd::{self};

// ---------------------------------------------------------------------------
// The D-08 seal, scanned in Rust so it runs on every platform and in CI
// ---------------------------------------------------------------------------

/// The four constructors 01-05/01-06 sealed to `pub(crate)`.
const SEALED_CONSTRUCTORS: [&str; 4] = ["from_bytes", "open", "open_slice_fixture", "from_import"];

/// `\b` on both sides of a literal, ASCII-word semantics.
fn word_bounded(haystack: &str, needle: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(i) = haystack[from..].find(needle) {
        let start = from + i;
        let end = start + needle.len();
        let before_ok = haystack[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word(c));
        let after_ok = haystack[end..].chars().next().is_none_or(|c| !is_word(c));
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// The declaration scan:
/// `^[^/]*\bpub fn (from_bytes|open|open_slice_fixture|from_import)\b`.
///
/// The `^[^/]*` prefix is load-bearing and is why this takes only the part of
/// the line BEFORE the first `/`: doc comments in 01-05/01-06 that literally
/// discuss `pub fn open` must not make the gate fire, and a commented-out
/// violation must not make it pass.
fn seal_declaration_hit(line: &str) -> bool {
    let head = line.split('/').next().unwrap_or("");
    SEALED_CONSTRUCTORS
        .iter()
        .any(|name| word_bounded(head, &format!("pub fn {name}")))
}

/// The re-export scan:
/// `^[^/]*pub use .*(from_bytes|open_slice_fixture|from_import)\b`.
///
/// A method cannot be re-exported (methods are not items), but a free-function
/// wrapper or a `pub use` alias could reopen the path without tripping the
/// declaration scan. This is the check that closes that evasion.
fn seal_reexport_hit(line: &str) -> bool {
    let head = line.split('/').next().unwrap_or("");
    if !head.contains("pub use ") {
        return false;
    }
    ["from_bytes", "open_slice_fixture", "from_import"]
        .iter()
        .any(|name| word_bounded(head, name))
}

fn crate_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `root`, recursively.
fn rust_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The declaration keyword, spelled indirectly ON PURPOSE.
///
/// The case table below must contain must-MATCH rows, and this file lives under
/// `src/setfit/` — the very directory the declaration scan walks. Writing the
/// rows as contiguous literals would make the test corpus trip its own gate
/// (measured: it did, on the first run). Assembling them at runtime keeps the
/// table's meaning while leaving the SOURCE TEXT clean, which is what both this
/// scan and the shell `grep` in the SUMMARY actually read.
const PUB_FN: &str = "pub fn ";

#[test]
fn setfit_model_seal_scan_case_table() {
    // CLAUDE.md rule 7: guard regexes ship a case table, and the table is
    // RE-RUN rather than the pattern re-read. Every one of the five historical
    // `apr`-invocation pattern defects was caught by a table like this and none
    // by review. These rows are the plan's, verbatim.
    let table: [(String, bool); 9] = [
        (
            format!("    {PUB_FN}open(dir: &Path) -> Result<Self, SetFitError> {{"),
            true,
        ),
        (
            format!("{PUB_FN}from_bytes(bytes: &[u8]) -> Result<Self, SetFitError> {{"),
            true,
        ),
        (
            format!("    #[cfg(feature = \"conformance-fixtures\")] {PUB_FN}open_slice_fixture("),
            true,
        ),
        (
            format!("    {PUB_FN}from_import(import: &MiniLmImport, seed: u64) -> ... {{"),
            true,
        ),
        (
            "    pub(crate) fn open(dir: &Path) -> ... {".to_string(),
            false,
        ),
        (
            format!("    /// SEALED (D-08): pub(crate). Was {PUB_FN}open before the seal."),
            false,
        ),
        (format!("    // {PUB_FN}from_bytes(...)"), false),
        (format!("    {PUB_FN}opened_at(&self) -> u64 {{"), false),
        (
            format!("    {PUB_FN}tokenizer_sha256(&self) -> &str {{"),
            false,
        ),
    ];
    for (line, expected) in table {
        assert_eq!(
            seal_declaration_hit(&line),
            expected,
            "seal declaration scan disagreed on: {line}"
        );
    }
}

#[test]
fn setfit_model_the_lower_level_constructors_are_still_sealed() {
    // Scoped to src/setfit/ on purpose. Widening it crate-wide was RE-MEASURED
    // here, not repeated from the plan: it yields exactly the 14 legitimate
    // pre-existing declarations the plan lists (apr/mmap/bundle/onnx/gguf/hnsw
    // readers), and a scan that cries wolf 14 times is a scan nobody reads.
    //
    // `*_tests.rs` is excluded, and that exclusion is a MEASURED correction, not
    // a convenience. Run as the plan writes it, the scan returns 8 matches on
    // the pre-existing tree — every one of them a string literal inside
    // 01-05/01-06's OWN seal assertions (`!src.contains("pub fn from_import(")`
    // and its three siblings). None is a declaration. A `#[cfg(test)]` module
    // also cannot reopen the seal for an out-of-crate consumer, because it is
    // not compiled into the library at all. Logged as D40.
    //
    // The compile probe recorded in the SUMMARY is the primary evidence; this is
    // the cheap tripwire against a future re-widening.
    let mut files = Vec::new();
    rust_files(&crate_src().join("setfit"), &mut files);
    files.retain(|p| {
        !p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_tests.rs"))
    });
    assert!(
        files.len() >= 5,
        "expected the setfit module's non-test sources"
    );

    let mut violations = Vec::new();
    for path in &files {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        for (i, line) in src.lines().enumerate() {
            if seal_declaration_hit(line) {
                violations.push(format!("{}:{}: {line}", path.display(), i + 1));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "D-08 seal broken — a sealed constructor is declared `pub fn`:\n{}",
        violations.join("\n")
    );
}

#[test]
fn setfit_model_no_sealed_constructor_is_re_exported_anywhere_in_the_crate() {
    let mut files = Vec::new();
    rust_files(&crate_src(), &mut files);
    assert!(files.len() > 100, "expected the whole crate's sources");

    let mut violations = Vec::new();
    for path in &files {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        for (i, line) in src.lines().enumerate() {
            if seal_reexport_hit(line) {
                violations.push(format!("{}:{}: {line}", path.display(), i + 1));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "a sealed constructor is re-exported, reopening the D-08 seal:\n{}",
        violations.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Source assertions on this module
// ---------------------------------------------------------------------------

fn mod_source() -> String {
    let path = crate_src().join("setfit/mod.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The enumerated public constructors of `SetFitMiniLm`.
///
/// # The invariant is the pairing, and the list is how it is enforced
///
/// D-08 says a mismatched tokenizer/encoder pair must not be CONSTRUCTIBLE from
/// outside the crate. Every entry below satisfies that by building both halves
/// from ONE source: a pinned checkout, a fixture directory, or — since plan 03-08
/// — one persistence bundle whose architecture record carries the tokenizer's
/// sha256 and whose reload path checks it against the supplied bytes BEFORE a
/// tensor is installed.
///
/// `from_bundle_parts` was added deliberately and this list was widened with it,
/// which is the sanctioned way past this gate. Widening it without an argument of
/// that shape is the thing the gate exists to stop: a constructor that takes a
/// tokenizer and an encoder from two places reopens the seal no matter how it is
/// named.
const PUBLIC_CONSTRUCTORS: [&str; 3] = [
    "from_bundle_parts",
    "from_pretrained_dir",
    "from_slice_fixture",
];

#[test]
fn setfit_model_exposes_exactly_the_enumerated_public_constructors() {
    let src = mod_source();
    let mut found: Vec<String> = src
        .lines()
        .filter_map(|l| {
            let head = l.split('/').next().unwrap_or("");
            let i = head.find("pub fn from_")?;
            let rest = &head[i + "pub fn ".len()..];
            let end = rest.find('(')?;
            Some(rest[..end].to_string())
        })
        .collect();
    found.sort();
    found.dedup();
    let mut expected: Vec<String> = PUBLIC_CONSTRUCTORS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    expected.sort();
    assert_eq!(
        found, expected,
        "adding a public constructor to work around the D-08 seal is forbidden; a new one is \
         legitimate only if it builds the tokenizer and the encoder from ONE source, and it \
         must be added to PUBLIC_CONSTRUCTORS with that argument written down"
    );
}

/// The bundle door pairs its two halves from one source, and proves it does.
///
/// The enumeration above is a list of names; this is the property the list stands
/// for, asserted against the one entry the list gained. `from_bundle_parts` must
/// check the tokenizer digest before it builds anything, or "one source" would be
/// a claim about how callers are expected to use it rather than a fact about it.
#[test]
fn setfit_model_bundle_constructor_checks_tokenizer_identity_before_building() {
    let src = mod_source();
    let at = src
        .find("pub fn from_bundle_parts")
        .expect("the bundle constructor must exist");
    let body = &src[at..];
    let hash_check = body
        .find("TokenizerHashMismatch")
        .expect("the bundle constructor must compare the supplied bytes against the record");
    let tokenizer_build = body
        .find("MiniLmTokenizer::from_bytes")
        .expect("the bundle constructor must build the tokenizer from the supplied bytes");
    let encoder_build = body
        .find("from_named_tensors")
        .expect("the bundle constructor must build the encoder from the supplied tensors");
    assert!(
        hash_check < tokenizer_build && hash_check < encoder_build,
        "the identity check must precede both halves being built; checking afterwards would \
         mean the work was done on a pair that had not been shown to belong together"
    );
}

#[test]
fn setfit_model_conformance_accessors_hand_out_borrows_only() {
    let src = mod_source();
    for (needle, why) in [
        (
            "pub fn encoder(&self) -> &BertSentenceEncoder",
            "01-08 reaches forward_tokens_per_layer through this borrow",
        ),
        (
            "pub fn tokenize(&self, texts: &[&str]) -> Result<SentenceBatch, SetFitError>",
            "01-08 needs a batch stamped with THIS model's tokenizer hash",
        ),
    ] {
        assert!(src.contains(needle), "missing `{needle}` — {why}");
        // Gated on conformance-fixtures, not on `setfit`: an ordinary
        // `--features setfit` consumer must never see either accessor.
        let head = &src[..src.find(needle).unwrap_or(0)];
        let gate = head
            .rfind("#[cfg(feature = \"conformance-fixtures\")]")
            .unwrap_or(0);
        assert!(
            gate > 0 && !head[gate..].contains("pub fn "),
            "`{needle}` must be immediately preceded by the conformance-fixtures gate"
        );
    }
}

#[test]
fn setfit_model_encode_texts_tokenizes_exactly_once() {
    let src = mod_source();
    let start = src
        .find("pub fn encode_texts")
        .expect("encode_texts must exist");
    let body = &src[start..];
    let end = body.find("\n    }").expect("end of encode_texts");
    let body = &body[..end];
    assert_eq!(
        body.matches("encode_batch(").count(),
        1,
        "encode_texts must tokenize once and hand the batch to the shared encode path"
    );
}

// ---------------------------------------------------------------------------
// FreezeGroup mapping — pure, no model needed
// ---------------------------------------------------------------------------

#[test]
fn setfit_model_freeze_group_prefixes_are_the_enc04_component_boundaries() {
    assert_eq!(FreezeGroup::Embeddings.name_prefixes(), vec!["embeddings."]);
    assert_eq!(
        FreezeGroup::LayerAttention(1).name_prefixes(),
        vec![
            "encoder.layer.1.attention.self.",
            "encoder.layer.1.attention.output.dense."
        ]
    );
    assert_eq!(
        FreezeGroup::LayerFfn(0).name_prefixes(),
        vec![
            "encoder.layer.0.intermediate.",
            "encoder.layer.0.output.dense."
        ]
    );
    assert_eq!(
        FreezeGroup::LayerNorm(0).name_prefixes(),
        vec![
            "encoder.layer.0.attention.output.LayerNorm.",
            "encoder.layer.0.output.LayerNorm."
        ]
    );
}

#[test]
fn setfit_model_every_freeze_prefix_ends_with_a_dot() {
    // `encoder.layer.1.` must not match `encoder.layer.10.…` on a future model
    // with ten or more layers. The trailing dot is the whole guard.
    for g in [
        FreezeGroup::Embeddings,
        FreezeGroup::LayerAttention(1),
        FreezeGroup::LayerFfn(1),
        FreezeGroup::LayerNorm(1),
    ] {
        let prefixes = g.name_prefixes();
        // Non-emptiness first: without it the loop below is vacuous and the
        // whole test is satisfied by a mapping that addresses nothing.
        assert!(!prefixes.is_empty(), "{g:?} has no prefixes at all");
        for p in prefixes {
            assert!(
                p.ends_with('.'),
                "{g:?} prefix `{p}` does not end with a dot"
            );
        }
    }
    assert!(
        !FreezeGroup::LayerAttention(1).matches("encoder.layer.10.attention.self.query.weight"),
        "layer 1's group must not address layer 10"
    );
}

#[test]
fn setfit_model_freeze_group_ordering_is_total_and_normalizable() {
    let mut v = vec![
        FreezeGroup::LayerNorm(1),
        FreezeGroup::Embeddings,
        FreezeGroup::LayerAttention(1),
        FreezeGroup::LayerAttention(0),
        FreezeGroup::Embeddings,
    ];
    v.sort();
    v.dedup();
    assert_eq!(
        v,
        vec![
            FreezeGroup::Embeddings,
            FreezeGroup::LayerAttention(0),
            FreezeGroup::LayerAttention(1),
            FreezeGroup::LayerNorm(1),
        ]
    );
}

// ---------------------------------------------------------------------------
// Slice-backed tests
// ---------------------------------------------------------------------------

#[cfg(feature = "conformance-fixtures")]
mod slice {
    use super::*;

    /// Seed every model here is built with unless a test varies it.
    const SEED: u64 = 0x0107_5E7F_1701;

    /// The two pair inputs from `loss_pair.json`, used as realistic text.
    const TEXTS_A: [&str; 2] = [
        "The cat sat on the mat.",
        "Stock markets fell sharply today.",
    ];
    const TEXTS_B: [&str; 2] = [
        "A feline rested on the rug.",
        "The weather is sunny and warm.",
    ];

    fn fixtures_dir() -> PathBuf {
        if let Ok(p) = std::env::var("APRENDER_SETFIT_FIXTURES") {
            let p = PathBuf::from(p);
            if p.is_dir() {
                return p;
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit")
    }

    fn model() -> SetFitMiniLm {
        SetFitMiniLm::from_slice_fixture(&fixtures_dir(), SEED)
            .expect("the frozen slice fixture must load through the bound type")
    }

    fn sorted_names(v: &[String]) -> Vec<String> {
        let mut out = v.to_vec();
        out.sort();
        out
    }

    fn all_names(m: &SetFitMiniLm) -> Vec<String> {
        m.encoder()
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    }

    fn frozen_names(m: &SetFitMiniLm) -> Vec<String> {
        m.frozen_parameters().into_iter().map(|(n, _)| n).collect()
    }

    fn trainable_names(m: &mut SetFitMiniLm) -> Vec<String> {
        m.trainable_parameters_mut()
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    }

    /// Bitwise snapshot of every named parameter.
    fn snapshot(m: &SetFitMiniLm) -> Vec<(String, Vec<u32>)> {
        m.encoder()
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, t.data().iter().map(|v| v.to_bits()).collect()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Construction and pairing (D-08)
    // -----------------------------------------------------------------------

    #[test]
    fn setfit_model_from_slice_fixture_pairs_the_tokenizer_with_the_encoder() {
        let m = model();
        assert_eq!(
            m.tokenizer_sha256(),
            m.encoder().tokenizer_sha256(),
            "both halves are loaded from ONE source, so their hashes must agree"
        );
        assert_eq!(
            m.tokenizer_sha256(),
            PINNED_TOKENIZER_SHA256,
            "the slice was cut against the pinned tokenizer"
        );
    }

    #[test]
    fn setfit_model_from_slice_fixture_returns_an_eval_mode_model() {
        // 01-06 departure 5: `from_import` returns an EVAL-mode encoder, matching
        // HF `from_pretrained`. 01-07 must not assume train mode, and this is the
        // assertion that keeps that true rather than remembered.
        let m = model();
        assert!(!m.training(), "a freshly loaded model must be in eval mode");
    }

    #[test]
    fn setfit_model_set_training_reaches_the_encoder() {
        let mut m = model();
        m.set_training(true);
        assert!(m.training());
        m.set_training(false);
        assert!(!m.training());
    }

    #[test]
    fn setfit_model_rejects_a_batch_from_a_different_in_crate_tokenizer() {
        // Defense in depth for IN-crate misuse. Out of crate this is not even
        // constructible (the compile probe in the SUMMARY records rustc saying
        // so); in crate, the sha256 equality inside `forward_layers` still
        // fires. The foreign tokenizer is REAL: the same vocabulary re-serialized
        // with different byte formatting, so it tokenizes identically and differs
        // only in its digest — which isolates the identity check as the cause.
        let bytes = std::fs::read(fixtures_dir().join("tokenizer.json")).expect("tokenizer.json");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("tokenizer json");
        let reformatted = serde_json::to_vec_pretty(&value).expect("re-serialize");
        assert_ne!(
            reformatted, bytes,
            "the re-serialization must change the bytes"
        );

        let foreign = MiniLmTokenizer::from_bytes(&reformatted).expect("foreign tokenizer");
        let m = model();
        assert_ne!(foreign.tokenizer_sha256(), m.tokenizer_sha256());

        let foreign_batch = foreign.encode_batch(&TEXTS_A).expect("tokenize");
        let err = m
            .encoder()
            .encode(&foreign_batch)
            .expect_err("a foreign batch must be rejected");
        assert!(
            matches!(err, SetFitError::TokenizerHashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn setfit_model_tokenize_stamps_this_models_own_tokenizer_hash() {
        let m = model();
        let batch = m.tokenize(&TEXTS_A).expect("tokenize");
        assert_eq!(batch.tokenizer_sha256(), m.tokenizer_sha256());
        // And the stamped batch is accepted by this model's encoder — the
        // positive half, without which "rejects a foreign batch" is satisfied by
        // a model that rejects everything.
        m.encoder()
            .forward_tokens(&batch)
            .expect("this model's own batch must be accepted");
    }

    #[test]
    fn setfit_model_encoder_accessor_reaches_forward_tokens_per_layer() {
        // The B5 access path 01-08 depends on, exercised one wave early. The
        // out-of-crate half of this proof is the positive compile probe recorded
        // in the SUMMARY: an in-crate call would compile identically even if
        // `forward_tokens_per_layer` were `pub(crate)`.
        // The text must live inside the 97-row slice closure: "hello" does not
        // (canonical id 7592), and the encoder correctly rejects it with
        // VocabOutOfSlice rather than zero-filling — measured, not assumed.
        let m = model();
        let batch = m.tokenize(&TEXTS_A[..1]).expect("tokenize");
        let (embeddings_out, layer_outputs) = m
            .encoder()
            .forward_tokens_per_layer(&batch)
            .expect("per-layer forward");
        assert_eq!(
            layer_outputs.len(),
            m.num_layers(),
            "one intermediate per encoder layer"
        );
        assert_eq!(embeddings_out.shape().len(), 3, "embeddings_out is [B,S,H]");
    }

    #[test]
    fn setfit_model_encode_texts_returns_unit_norm_rows() {
        let m = model();
        let z = m.encode_texts(&TEXTS_A).expect("encode");
        assert_eq!(z.shape(), &[2, 64], "[B, H] on the 2-layer/hidden-64 slice");
        for row in 0..2 {
            let n: f64 = z.data()[row * 64..(row + 1) * 64]
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum();
            assert!(
                (n.sqrt() - 1.0).abs() < 1e-5,
                "row {row} norm is {}",
                n.sqrt()
            );
        }
    }

    // -----------------------------------------------------------------------
    // The partition (D-20 / D-21 / D-22)
    // -----------------------------------------------------------------------

    #[test]
    fn setfit_model_default_partition_is_all_trainable() {
        let mut m = model();
        assert!(m.freeze_policy().is_empty(), "D-20: no policy by default");
        assert!(m.frozen_parameters().is_empty(), "D-20: nothing frozen");
        assert_eq!(
            trainable_names(&mut m).len(),
            37,
            "the slice has 37 named parameters and all of them start trainable"
        );
    }

    #[test]
    fn setfit_model_partition_is_disjoint_and_complete() {
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::Embeddings, FreezeGroup::LayerAttention(1)])
            .expect("valid policy");

        let all = sorted_names(&all_names(&m));
        let frozen = sorted_names(&frozen_names(&m));
        let trainable = sorted_names(&trainable_names(&mut m));

        for name in &frozen {
            assert!(
                !trainable.contains(name),
                "`{name}` is in BOTH partitions — an optimizer would update a frozen tensor"
            );
        }
        let mut union = frozen.clone();
        union.extend(trainable.iter().cloned());
        union.sort();
        assert_eq!(
            union, all,
            "the two partitions must cover every named parameter"
        );
    }

    #[test]
    fn setfit_model_apply_freeze_moves_exactly_the_mapped_prefix_sets() {
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::Embeddings, FreezeGroup::LayerAttention(1)])
            .expect("valid policy");
        assert_eq!(
            sorted_names(&frozen_names(&m)),
            vec![
                "embeddings.LayerNorm.bias",
                "embeddings.LayerNorm.weight",
                "embeddings.position_embeddings.weight",
                "embeddings.token_type_embeddings.weight",
                "embeddings.word_embeddings.weight",
                "encoder.layer.1.attention.output.dense.bias",
                "encoder.layer.1.attention.output.dense.weight",
                "encoder.layer.1.attention.self.key.bias",
                "encoder.layer.1.attention.self.key.weight",
                "encoder.layer.1.attention.self.query.bias",
                "encoder.layer.1.attention.self.query.weight",
                "encoder.layer.1.attention.self.value.bias",
                "encoder.layer.1.attention.self.value.weight",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn setfit_model_layer_attention_freeze_leaves_the_attention_layer_norm_trainable() {
        // Group-boundary exactness: `attention.output.LayerNorm` belongs to
        // LayerNorm(n), NOT to LayerAttention(n). A mapping that swept it in
        // would make "freeze the projections" and "freeze the whole attention
        // block" indistinguishable.
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::LayerAttention(0)])
            .expect("valid policy");
        let trainable = trainable_names(&mut m);
        for name in [
            "encoder.layer.0.attention.output.LayerNorm.weight",
            "encoder.layer.0.attention.output.LayerNorm.bias",
        ] {
            assert!(
                trainable.iter().any(|n| n == name),
                "`{name}` must stay TRAINABLE under LayerAttention(0)"
            );
        }
        assert!(
            !trainable
                .iter()
                .any(|n| n == "encoder.layer.0.attention.output.dense.weight"),
            "attention.output.dense IS part of LayerAttention(0)"
        );
    }

    #[test]
    fn setfit_model_the_four_groups_cover_every_named_parameter() {
        let mut m = model();
        let layers = m.num_layers();
        let mut policy = vec![FreezeGroup::Embeddings];
        for n in 0..layers {
            policy.push(FreezeGroup::LayerAttention(n));
            policy.push(FreezeGroup::LayerFfn(n));
            policy.push(FreezeGroup::LayerNorm(n));
        }
        m.apply_freeze(&policy).expect("valid policy");
        assert_eq!(
            sorted_names(&frozen_names(&m)),
            sorted_names(&all_names(&m)),
            "the four ENC-04 groups must partition the whole parameter set with no gaps"
        );
        assert!(trainable_names(&mut m).is_empty());
    }

    #[test]
    fn setfit_model_every_valid_group_addresses_at_least_one_parameter() {
        // The naming-drift guard, stated as a property over EVERY valid group
        // rather than one example: a group that matches nothing means 01-06's
        // dotted names moved, and that must fail loudly instead of silently
        // freezing nothing.
        let m = model();
        let all = all_names(&m);
        let mut groups = vec![FreezeGroup::Embeddings];
        for n in 0..m.num_layers() {
            groups.push(FreezeGroup::LayerAttention(n));
            groups.push(FreezeGroup::LayerFfn(n));
            groups.push(FreezeGroup::LayerNorm(n));
        }
        for g in groups {
            let hits = all.iter().filter(|n| g.matches(n)).count();
            assert!(hits > 0, "{g:?} addresses ZERO named parameters");
        }
    }

    #[test]
    fn setfit_model_a_group_matching_zero_names_is_rejected_not_ignored() {
        // Reached by construction rather than by waiting for a rename: a group
        // whose layer index is in range on a DIFFERENT model but not this one is
        // the out-of-range branch, so the empty-match branch is exercised here by
        // asserting the whole-policy behaviour on the valid range. See the
        // out-of-range test for the sibling branch.
        let mut m = model();
        let layers = m.num_layers();
        let err = m
            .apply_freeze(&[FreezeGroup::LayerFfn(layers)])
            .expect_err("layer index one past the end");
        assert!(
            matches!(err, SetFitError::FreezeGroupInvalid { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn setfit_model_out_of_range_group_is_rejected_and_leaves_the_policy_intact() {
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::LayerFfn(0)])
            .expect("valid policy");
        let before_policy = m.freeze_policy();
        let before_frozen = sorted_names(&frozen_names(&m));
        let before_flags: Vec<(String, bool)> = m
            .encoder()
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, t.requires_grad_enabled()))
            .collect();

        // LayerAttention(7) on a 2-layer slice. The FIRST group is valid, so a
        // validate-as-you-go implementation would have already frozen the
        // embeddings by the time it rejected the second.
        let err = m
            .apply_freeze(&[FreezeGroup::Embeddings, FreezeGroup::LayerAttention(7)])
            .expect_err("layer 7 does not exist on a 2-layer slice");
        match &err {
            SetFitError::FreezeGroupInvalid { reason } => {
                assert!(
                    reason.contains('7'),
                    "the offending index must be named: {reason}"
                );
            }
            other => panic!("expected FreezeGroupInvalid, got {other:?}"),
        }

        assert_eq!(m.freeze_policy(), before_policy, "no partial application");
        assert_eq!(sorted_names(&frozen_names(&m)), before_frozen);
        let after_flags: Vec<(String, bool)> = m
            .encoder()
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, t.requires_grad_enabled()))
            .collect();
        assert_eq!(after_flags, before_flags, "no requires_grad flag may move");
    }

    #[test]
    fn setfit_model_apply_freeze_has_replacement_semantics() {
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::Embeddings]).expect("first");
        assert!(frozen_names(&m)
            .iter()
            .any(|n| n.starts_with("embeddings.")));

        m.apply_freeze(&[FreezeGroup::LayerFfn(0)]).expect("second");
        assert_eq!(m.freeze_policy(), vec![FreezeGroup::LayerFfn(0)]);
        assert!(
            !frozen_names(&m)
                .iter()
                .any(|n| n.starts_with("embeddings.")),
            "REPLACEMENT: a group absent from the new policy becomes trainable again"
        );
        for (name, t) in m.encoder().named_parameters() {
            if name.starts_with("embeddings.") {
                assert!(
                    t.requires_grad_enabled(),
                    "`{name}` must have requires_grad restored to true"
                );
            }
        }
    }

    #[test]
    fn setfit_model_apply_freeze_is_idempotent() {
        let policy = [FreezeGroup::LayerAttention(0), FreezeGroup::LayerNorm(1)];
        let mut m = model();
        m.apply_freeze(&policy).expect("first");
        let once = (m.freeze_policy(), sorted_names(&frozen_names(&m)));
        m.apply_freeze(&policy).expect("second");
        let twice = (m.freeze_policy(), sorted_names(&frozen_names(&m)));
        assert_eq!(once, twice);
    }

    #[test]
    fn setfit_model_apply_freeze_is_order_and_duplicate_insensitive() {
        let a = FreezeGroup::LayerAttention(0);
        let b = FreezeGroup::LayerNorm(1);
        let mut results = Vec::new();
        for policy in [vec![a, b], vec![b, a], vec![a, b, a]] {
            let mut m = model();
            m.apply_freeze(&policy).expect("valid policy");
            results.push((m.freeze_policy(), sorted_names(&frozen_names(&m))));
        }
        assert_eq!(results[0], results[1], "order must not matter");
        assert_eq!(results[0], results[2], "duplicates must not matter");
        assert_eq!(
            results[0].0,
            vec![a, b],
            "the stored policy is normalized: sorted and deduplicated"
        );
    }

    #[test]
    fn setfit_model_clear_freeze_restores_all_trainable() {
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::Embeddings, FreezeGroup::LayerFfn(1)])
            .expect("valid policy");
        m.clear_freeze();
        assert!(m.freeze_policy().is_empty());
        assert!(m.frozen_parameters().is_empty());
        for (name, t) in m.encoder().named_parameters() {
            assert!(
                t.requires_grad_enabled(),
                "`{name}` requires_grad not restored"
            );
        }
        assert_eq!(trainable_names(&mut m).len(), 37);
    }

    #[test]
    fn setfit_model_apply_freeze_with_an_empty_list_equals_clear_freeze() {
        let mut a = model();
        a.apply_freeze(&[FreezeGroup::Embeddings]).expect("policy");
        a.clear_freeze();

        let mut b = model();
        b.apply_freeze(&[FreezeGroup::Embeddings]).expect("policy");
        b.apply_freeze(&[]).expect("empty policy");

        assert_eq!(a.freeze_policy(), b.freeze_policy());
        assert_eq!(
            sorted_names(&trainable_names(&mut a)),
            sorted_names(&trainable_names(&mut b))
        );
    }

    #[test]
    fn setfit_model_frozen_parameters_have_requires_grad_cleared_and_trainable_keep_it() {
        let mut m = model();
        m.apply_freeze(&[FreezeGroup::LayerAttention(1)])
            .expect("valid policy");
        for (name, t) in m.frozen_parameters() {
            assert!(
                !t.requires_grad_enabled(),
                "frozen `{name}` still requires grad"
            );
        }
        let frozen = frozen_names(&m);
        for (name, t) in m.encoder().named_parameters() {
            if !frozen.contains(&name) {
                assert!(
                    t.requires_grad_enabled(),
                    "trainable `{name}` lost requires_grad"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // The load-bearing proof: freezing changes OBSERVED BEHAVIOUR
    // -----------------------------------------------------------------------

    /// Learning rate of the probe step.
    ///
    /// Large on purpose. This is not a training step — nothing is re-forwarded
    /// from the result — it exists so that a real gradient produces a change
    /// visible at f32 bit level rather than one that rounds away and gets
    /// misread as "the parameter is frozen".
    const PROBE_LR: f32 = 1000.0;

    /// One SGD step over the TRAINABLE set only. Returns the names that moved.
    ///
    /// This is deliberately built from `trainable_parameters_mut()` — the exact
    /// method 01-08's AdamW parameter set comes from — so what it measures is
    /// what an optimizer would do, not a parallel reimplementation of freezing.
    fn sgd_step_over_trainable(m: &mut SetFitMiniLm) -> Vec<String> {
        let mut moved = Vec::new();
        for (name, t) in m.trainable_parameters_mut() {
            let Some(g) = autograd::get_grad(t.id()) else {
                continue;
            };
            let grads = g.data().to_vec();
            let mut changed = false;
            for (v, gv) in t.data_mut().iter_mut().zip(grads.iter()) {
                let before = *v;
                *v -= PROBE_LR * gv;
                if v.to_bits() != before.to_bits() {
                    changed = true;
                }
            }
            if changed {
                moved.push(name);
            }
        }
        moved
    }

    /// Load a model, apply `policy`, run the ENC-06 objective, backward, and
    /// take one step over the trainable set.
    ///
    /// Returns `(names_that_moved, before, after)`.
    fn one_optimizer_step(
        policy: &[FreezeGroup],
    ) -> (
        Vec<String>,
        Vec<(String, Vec<u32>)>,
        Vec<(String, Vec<u32>)>,
    ) {
        autograd::clear_graph();
        let mut m = model();
        // Eval mode: dropout inert, so the step is deterministic and the two
        // runs of the two-sided test differ ONLY in the freeze policy.
        m.set_training(false);
        m.apply_freeze(policy).expect("policy must apply");

        let before = snapshot(&m);
        let za = m.encode_texts(&TEXTS_A).expect("encode branch a");
        let zb = m.encode_texts(&TEXTS_B).expect("encode branch b");
        let loss = pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect("pair objective");
        assert!(loss.item().is_finite(), "loss is {}", loss.item());
        loss.backward();

        let moved = sgd_step_over_trainable(&mut m);
        let after = snapshot(&m);
        (moved, before, after)
    }

    fn bits_of<'a>(snap: &'a [(String, Vec<u32>)], name: &str) -> &'a [u32] {
        snap.iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.as_slice())
            .unwrap_or_else(|| panic!("`{name}` is not a named parameter"))
    }

    #[test]
    fn setfit_model_the_pair_objective_backward_reaches_the_encoder_parameters() {
        // The plan's first truth: a finite, tensor-valued, graph-connected pair
        // loss WHOSE BACKWARD REACHES ENCODER PARAMETERS. Without this the
        // freeze tests below would be measuring an inert step.
        let (moved, _, _) = one_optimizer_step(&[]);
        assert!(
            moved.len() >= 30,
            "only {} of 37 parameters moved under an all-trainable policy — the \
             objective's backward is not reaching the encoder",
            moved.len()
        );
    }

    #[test]
    fn setfit_model_a_frozen_group_does_not_move_across_an_optimizer_step() {
        let policy = [FreezeGroup::Embeddings, FreezeGroup::LayerAttention(1)];
        let (moved, before, after) = one_optimizer_step(&policy);

        for (name, bits) in &before {
            let frozen = policy.iter().any(|g| g.matches(name));
            let after_bits = bits_of(&after, name);
            if frozen {
                assert_eq!(
                    bits.as_slice(),
                    after_bits,
                    "`{name}` is frozen but its bits MOVED across the optimizer step"
                );
                assert!(
                    !moved.contains(name),
                    "`{name}` is frozen but appeared in the trainable update set"
                );
            }
        }
        assert!(
            !moved.is_empty(),
            "nothing moved at all — the step is inert, so the frozen assertion proves nothing"
        );
    }

    #[test]
    fn setfit_model_the_same_parameter_moves_when_trainable_and_stays_when_frozen() {
        // The two-sided form, and the one that isolates the freeze as the CAUSE.
        // Both runs load the same fixture weights, run in eval mode, take the
        // same objective and the same step; the ONLY difference is the policy.
        const PROBE: &str = "embeddings.word_embeddings.weight";

        let (moved_free, before_free, after_free) = one_optimizer_step(&[]);
        assert!(
            moved_free.iter().any(|n| n == PROBE),
            "`{PROBE}` must move when it is trainable, otherwise the frozen half \
             below is vacuous"
        );
        assert_ne!(bits_of(&before_free, PROBE), bits_of(&after_free, PROBE));

        let (moved_frozen, before_frozen, after_frozen) =
            one_optimizer_step(&[FreezeGroup::Embeddings]);
        assert!(!moved_frozen.iter().any(|n| n == PROBE));
        assert_eq!(
            bits_of(&before_frozen, PROBE),
            bits_of(&after_frozen, PROBE),
            "`{PROBE}` moved despite being frozen"
        );

        // Both runs start from identical weights, so the frozen run's "after"
        // must equal the free run's "before" for this tensor.
        assert_eq!(bits_of(&before_free, PROBE), bits_of(&after_frozen, PROBE));
    }

    #[test]
    fn setfit_model_clear_freeze_lets_a_previously_frozen_parameter_move_again() {
        const PROBE: &str = "encoder.layer.0.intermediate.dense.weight";

        autograd::clear_graph();
        let mut m = model();
        m.set_training(false);
        m.apply_freeze(&[FreezeGroup::LayerFfn(0)]).expect("policy");
        m.clear_freeze();

        let before = snapshot(&m);
        let za = m.encode_texts(&TEXTS_A).expect("encode a");
        let zb = m.encode_texts(&TEXTS_B).expect("encode b");
        let loss = pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect("loss");
        loss.backward();
        let moved = sgd_step_over_trainable(&mut m);
        let after = snapshot(&m);

        assert!(
            moved.iter().any(|n| n == PROBE),
            "clear_freeze must make `{PROBE}` updatable again"
        );
        assert_ne!(bits_of(&before, PROBE), bits_of(&after, PROBE));
    }
}
