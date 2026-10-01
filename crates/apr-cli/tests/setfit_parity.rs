//! The three-surface SetFit parity gate (Phase 4 D-13 / D-14).
//!
//! # What this file proves, and what it deliberately does not
//!
//! It feeds ONE ordered input set through THREE surfaces and compares them
//! pairwise:
//!
//! | leg | reached by |
//! | --- | ---------- |
//! | library | `aprender::setfit::VerifiedSetFitModel::classify` |
//! | CLI | a SPAWNED `env!("CARGO_BIN_EXE_apr")` running `predict --input … --json` |
//! | HTTP | `realizar::api::create_router_with_config` driven by `tower` `oneshot` |
//!
//! **It does NOT prove that training produces an artifact all three agree on.**
//! Before Phase 5's 05-03 calibration edit (commit `a63bb130b`), no user-reachable
//! path produced a `setfit-apr-v1` at all: `CALIBRATED_REGIMES` admitted exactly
//! one encoder (the phase-3 MiniLM slice) whose 97-row vocabulary closure cannot
//! compute the `probe_unicode` probe (finding F-10, measured on three independent
//! routes by plan 04-12). That is closed — 05-07's
//! `setfit_cli_production_chain_completes_after_the_calibration_edit` trains,
//! inspects, evaluates and predicts from one real artifact — but closing it did
//! not change what THIS file claims, and the artifact it compares over is
//! deliberately still a SYNTHETIC fixture: producing a real one needs the 86.7 MB
//! production checkout, which is an offline prerequisite rather than something a
//! parity suite may require. The fixture is written through core's public
//! `write_setfit_apr` and loaded through core's public `load_setfit_apr` — the
//! same two doors production uses. That is honest for a PARITY claim, which is
//! about three READERS agreeing on one artifact, and it would NOT be honest for
//! an end-to-end claim, which this file does not make.
//!
//! # Why the input set is one document and not one text per line (review M2)
//!
//! The probe set contains `probe_whitespace`, a text with an embedded newline
//! and a tab. A line-delimited CLI input format would have re-split that text
//! into two CLI texts while the HTTP leg still posted one — so the three legs
//! would have received DIFFERENT ordered input sets and the gate would have
//! reported agreement it had never tested. All three legs here carry the
//! identical `ClassifyRequestDocument`, and the CLI leg is handed the SAME
//! serialized bytes the HTTP leg posts. `parity_every_leg_returns_one_result_per_text`
//! is the standing witness for that.
//!
//! # Latency is compared for neither equality nor positivity
//!
//! `latency_ms` is a MEASUREMENT. Core's `PartialEq for ClassifyResponse`
//! already excludes it, and this comparator checks only `is_finite() && >= 0.0`:
//! a fast operation under a coarse timer legitimately reports `0.0`, so a
//! `> 0.0` assertion is a flake with a schedule.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use aprender::setfit::artifact::{PROBE_TRUNCATION_REPEAT_COUNT, PROBE_TRUNCATION_REPEAT_UNIT};
use aprender::setfit::{
    artifact_sha256_hex, load_setfit_apr, write_setfit_apr, ClassifyRequestDocument,
    ClassifyResponse, EncoderArchitecture, SetFitArtifactView, VerifiedSetFitModel,
    PROBE_LOGITS_ABS_TOLERANCE, PROBE_PROBABILITIES_ABS_TOLERANCE,
};

// ---------------------------------------------------------------------------
// The tiny fixture artifact
// ---------------------------------------------------------------------------

mod fixture {
    //! A `setfit-apr-v1` artifact small enough to build at test start.
    //!
    //! **A DUPLICATE of `aprender-core`'s `setfit::artifact::fixture` (the
    //! `fixture_view_full_pin_shape()` recipe), by way of `aprender-serve`'s own
    //! copy in `api/setfit_handlers.rs`.** Core's is `#[cfg(test)]` and plan
    //! 03-10's acceptance criteria reject a `#[doc(hidden)]` test-support door on
    //! the shipped surface, so it is unreachable from an integration test by
    //! design rather than by oversight.
    //!
    //! What is duplicated is the SHAPE, and every part is built through core's
    //! PUBLIC API: `SetFitArtifactView` and `EncoderArchitecture` have public
    //! fields, `write_setfit_apr` / `load_setfit_apr` are the two production
    //! doors, and `artifact_sha256_hex` is the crate's one hashing path. If the
    //! artifact schema changes this builder stops producing a loadable artifact
    //! and these tests go red, which is the correct coupling.
    //!
    //! It is the FULL PRODUCTION NULLABILITY SHAPE — `vocab_remap: None`,
    //! `evidence.epsilon_used: null` — so the parity legs run on the shape a
    //! pinned MiniLM serializes, not a nullability-free simplification.
    //!
    //! The ~90 MB real model is NOT used and nothing large is committed: the
    //! whole artifact is a few hundred kilobytes and lives in a tempdir.

    use super::{
        artifact_sha256_hex, load_setfit_apr, write_setfit_apr, BTreeMap, EncoderArchitecture,
        SetFitArtifactView, VerifiedSetFitModel,
    };
    use aprender::setfit::{
        L2_EPS, MAX_SEQUENCE_LENGTH, NORMALIZATION_POLICY, PADDING_MODE, PINNED_ACTIVATION,
        PINNED_REVISION, POOLING_POLICY,
    };
    use serde_json::json;

    const FIXTURE_HIDDEN: usize = 8;
    const FIXTURE_HEADS: usize = 2;
    const FIXTURE_LAYERS: usize = 2;
    const FIXTURE_INTERMEDIATE: usize = 16;
    const FIXTURE_TYPE_VOCAB: usize = 2;
    pub const FIXTURE_LABELS: [&str; 3] = ["against", "favor", "neutral"];

    /// A tiny WordPiece vocabulary. Everything outside it becomes `[UNK]`, so
    /// every id the tokenizer can emit is `< TINY_VOCAB.len()` — which is what
    /// lets the fixture carry `vocab_remap: None` (the PRODUCTION shape) with a
    /// 48-row embedding table instead of the pin's 30522.
    const TINY_VOCAB: [&str; 48] = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "ok",
        "few",
        "shot",
        "classification",
        "with",
        "contrastive",
        "pairs",
        "line",
        "one",
        "two",
        "tabbed",
        "spaced",
        "stance",
        "detection",
        "i",
        "firmly",
        "support",
        "this",
        "position",
        "el",
        "zorro",
        "cafe",
        "naive",
        "pi",
        "##s",
        "##ed",
        ".",
        ",",
        "!",
        "#",
        "@",
        ":",
        "/",
        "-",
        "=",
    ];

    /// A valid, self-contained `tokenizer.json` in the pinned file's exact shape
    /// (BertNormalizer + BertPreTokenizer + TemplateProcessing + WordPiece) with
    /// [`TINY_VOCAB`] substituted for the 30522-entry pin.
    ///
    /// Self-contained ON PURPOSE: reading the committed fixture file would honour
    /// the `APRENDER_SETFIT_FIXTURES` override, and a test whose input depends on
    /// an environment variable is not a test.
    fn tiny_tokenizer_json() -> Vec<u8> {
        use std::fmt::Write as _;
        let mut s = String::new();
        s.push_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":["#);
        for (id, content) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            .iter()
            .enumerate()
        {
            if id > 0 {
                s.push(',');
            }
            let _ = write!(
                s,
                r#"{{"id":{id},"special":true,"content":"{content}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false}}"#
            );
        }
        s.push_str(
            r###"],"normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}},"decoder":{"type":"WordPiece","prefix":"##","cleanup":true},"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"###,
        );
        for (id, token) in TINY_VOCAB.iter().enumerate() {
            if id > 0 {
                s.push(',');
            }
            let _ = write!(s, r#""{token}":{id}"#);
        }
        s.push_str("}}}");
        s.into_bytes()
    }

    /// A deterministic, platform-independent filler.
    ///
    /// Every produced value is `k / 65536 - 0.5` for an integer `k`, so it is
    /// EXACTLY representable in `f32` on every target: the fixture's own bytes
    /// cannot be a source of cross-platform drift, and neither can the goldens
    /// frozen from it.
    struct Filler(u64);

    impl Filler {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let quantum = f32::from(u16::try_from((self.0 >> 40) & 0xFFFF).unwrap_or(0)) / 65536.0;
            quantum - 0.5
        }

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn fixture_architecture() -> EncoderArchitecture {
        EncoderArchitecture {
            hidden: FIXTURE_HIDDEN,
            heads: FIXTURE_HEADS,
            head_dim: FIXTURE_HIDDEN / FIXTURE_HEADS,
            num_layers: FIXTURE_LAYERS,
            intermediate: FIXTURE_INTERMEDIATE,
            vocab: TINY_VOCAB.len(),
            positions: MAX_SEQUENCE_LENGTH,
            type_vocab_size: FIXTURE_TYPE_VOCAB,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            hidden_act: PINNED_ACTIVATION.to_string(),
            source_revision: PINNED_REVISION.to_string(),
            tokenizer_sha256: artifact_sha256_hex(&tiny_tokenizer_json()),
            // `None` is the PRODUCTION shape (a pinned MiniLM serializes no
            // remap), and it is what makes the writer's nullability walk
            // non-vacuous on this fixture.
            vocab_remap: None,
        }
    }

    fn put(
        t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
        f: &mut Filler,
        name: String,
        shape: Vec<usize>,
    ) {
        let n = shape.iter().product();
        t.insert(name, (shape, f.vec(n)));
    }

    fn fixture_tensors(arch: &EncoderArchitecture) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let h = arch.hidden;
        let im = arch.intermediate;
        let mut f = Filler::new(0x0409_0001);
        let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();

        put(
            &mut t,
            &mut f,
            "embeddings.word_embeddings.weight".to_string(),
            vec![arch.vocab, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.position_embeddings.weight".to_string(),
            vec![arch.positions, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.token_type_embeddings.weight".to_string(),
            vec![arch.type_vocab_size, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.LayerNorm.weight".to_string(),
            vec![h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.LayerNorm.bias".to_string(),
            vec![h],
        );

        for n in 0..arch.num_layers {
            let p = format!("encoder.layer.{n}");
            for leaf in ["query", "key", "value"] {
                put(
                    &mut t,
                    &mut f,
                    format!("{p}.attention.self.{leaf}.weight"),
                    vec![h, h],
                );
                put(
                    &mut t,
                    &mut f,
                    format!("{p}.attention.self.{leaf}.bias"),
                    vec![h],
                );
            }
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.dense.weight"),
                vec![h, h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.dense.bias"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.LayerNorm.weight"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.LayerNorm.bias"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.intermediate.dense.weight"),
                vec![im, h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.intermediate.dense.bias"),
                vec![im],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.output.dense.weight"),
                vec![h, im],
            );
            put(&mut t, &mut f, format!("{p}.output.dense.bias"), vec![h]);
            put(
                &mut t,
                &mut f,
                format!("{p}.output.LayerNorm.weight"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.output.LayerNorm.bias"),
                vec![h],
            );
        }
        t
    }

    /// The full-pin-shape view — the `fixture_view_full_pin_shape()` recipe.
    ///
    /// `disallowed_methods` is allowed for the `serde_json::json!` expansions
    /// below and for NOTHING ELSE: the macro's own generated code calls
    /// `Result::unwrap` on an infallible `Value` construction. There is no
    /// `unwrap` written in this file (the source assertion
    /// `parity_harness_writes_no_unwrap` pins that), so the allow cannot be
    /// silently widened into hand-written fallibility.
    #[allow(clippy::disallowed_methods)]
    fn fixture_view_full_pin_shape() -> SetFitArtifactView {
        let architecture = fixture_architecture();
        let tensors = fixture_tensors(&architecture);
        let mut head = Filler::new(0x0409_0002);
        let k = FIXTURE_LABELS.len();
        let max_len = u32::try_from(MAX_SEQUENCE_LENGTH).expect("256 fits in u32");
        SetFitArtifactView {
            bundle_schema_version: 1,
            format_id: "setfit-apr-v1-fixture".to_string(),
            tokenizer_bytes: tiny_tokenizer_json(),
            head_weights: head.vec(k * architecture.hidden),
            head_intercepts: head.vec(k),
            head_n_features: architecture.hidden,
            architecture,
            tensors,
            pooling: POOLING_POLICY.to_string(),
            normalization: NORMALIZATION_POLICY.to_string(),
            l2_epsilon: L2_EPS,
            truncation_max_sequence_length: max_len,
            padding_mode: PADDING_MODE.to_string(),
            max_length: max_len,
            root_seed: 0x0409_0000_0000_0002,
            ordered_labels: FIXTURE_LABELS.iter().map(|s| (*s).to_string()).collect(),
            requested_config: json!({
                "max_length": 256,
                "pair_config": { "budget": null, "hard_cap": null, "strategy": "all_pairs" },
                "requested_device": "cpu",
                "seed": 7
            }),
            resolved_config: json!({ "resolved_device": "cpu" }),
            evidence: json!({
                "epsilon_used": null,
                "table_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "per_class": {
                    "against": { "support": 8, "mean_margin": 0.1 },
                    "favor": { "support": 8, "mean_margin": 0.5 },
                    "neutral": { "support": 8, "mean_margin": 0.25 }
                }
            }),
            provenance: json!({
                "dataset_fingerprint": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "validation_split_fingerprint": "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",
                "selection_semantic_hash": "ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",
                "selection_ledger_hash": "llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll",
                "selection_root_seed": 11,
                "shots_per_class": 8
            }),
        }
    }

    /// The artifact BYTES.
    pub fn fixture_bytes() -> Vec<u8> {
        write_setfit_apr(&fixture_view_full_pin_shape()).expect("the tiny fixture is writable")
    }

    /// Load a model from bytes through the ONE production door.
    ///
    /// `load_setfit_apr` runs the whole ladder including probe replay, so this
    /// helper cannot mint a `VerifiedSetFitModel` that skipped a rung — and the
    /// fact that it succeeds is itself evidence the duplicated shape is still a
    /// valid artifact.
    pub fn load(bytes: &[u8]) -> VerifiedSetFitModel {
        load_setfit_apr(bytes).expect("the tiny fixture passes every load rung")
    }
}

// ---------------------------------------------------------------------------
// The ONE committed input set
// ---------------------------------------------------------------------------

/// All six contract-resident probe strings plus two extras.
///
/// The extras are the EMPTY string (an entry a `filter(|s| !s.is_empty())`
/// anywhere in a surface would silently drop, shifting every later index) and a
/// 4-byte-UTF-8 emoji run (the pangram and unicode probes only reach 3-byte
/// sequences). `probe_whitespace` carries both an embedded newline and a tab —
/// the review-M2 witness.
fn parity_texts() -> Vec<String> {
    vec![
        // probe_minimal
        "ok".to_string(),
        // probe_ascii_pangram
        "the quick brown fox jumps over the lazy dog".to_string(),
        // probe_unicode
        "El rapido zorro marron salta sobre el perro perezoso — naive cafe, pi = 3.14159"
            .to_string(),
        // probe_truncation_boundary, built from the contract's own constants
        PROBE_TRUNCATION_REPEAT_UNIT.repeat(PROBE_TRUNCATION_REPEAT_COUNT),
        // probe_social
        "Stance detection: I firmly support this position!!! #debate @user123 https://example.com"
            .to_string(),
        // probe_whitespace — an embedded newline AND a tab AND a run of spaces
        "line one\nline two\ttabbed   spaced".to_string(),
        // extra: the empty string
        String::new(),
        // extra: 4-byte UTF-8
        "🦊🌮 café naïve π ✅".to_string(),
    ]
}

/// The batch document every pairwise test runs on.
fn batch_document() -> ClassifyRequestDocument {
    ClassifyRequestDocument::new(parity_texts())
}

/// A single-text document — the arity a batch cannot witness.
fn single_document() -> ClassifyRequestDocument {
    ClassifyRequestDocument::new(["line one\nline two\ttabbed   spaced"])
}

/// The batch document with per-class logits requested.
fn logits_document() -> ClassifyRequestDocument {
    batch_document().with_logits()
}

// ---------------------------------------------------------------------------
// The harness: one artifact, one serialized document, three legs
// ---------------------------------------------------------------------------

/// One fixture artifact on disk plus the model the LIBRARY leg reads through.
struct Harness {
    _dir: tempfile::TempDir,
    apr_path: std::path::PathBuf,
    bytes: Vec<u8>,
    model: VerifiedSetFitModel,
}

impl Harness {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("a tempdir is creatable");
        let apr_path = dir.path().join("parity-fixture.apr");
        let bytes = fixture::fixture_bytes();
        std::fs::write(&apr_path, &bytes).expect("the fixture artifact is writable");
        let model = fixture::load(&bytes);
        Self {
            _dir: dir,
            apr_path,
            bytes,
            model,
        }
    }

    /// The hash of the bytes on disk, derived independently of any surface.
    fn artifact_sha256(&self) -> String {
        artifact_sha256_hex(&self.bytes)
    }

    /// Serialize the document ONCE. These exact bytes are what the CLI leg reads
    /// from a file and what the HTTP leg posts as a body — so no leg can be
    /// handed a differently-shaped input than another (review M2).
    fn serialize(document: &ClassifyRequestDocument) -> String {
        serde_json::to_string(document).expect("the request document serializes")
    }

    fn write_document(&self, document: &ClassifyRequestDocument) -> std::path::PathBuf {
        let path = self._dir.path().join(format!(
            "request-{}-{}.json",
            document.texts.len(),
            usize::from(document.include_logits)
        ));
        std::fs::write(&path, Self::serialize(document)).expect("the request document is writable");
        path
    }

    // -- leg (a): the library --------------------------------------------------

    fn library(&self, document: &ClassifyRequestDocument) -> ClassifyResponse {
        self.model
            .classify(document)
            .expect("the library leg classifies the committed document")
    }

    // -- leg (b): the spawned CLI ---------------------------------------------

    /// Spawn `apr predict <apr> --input <json> --json` and parse stdout INTO
    /// core's envelope.
    ///
    /// The binary is resolved by `CARGO_BIN_EXE_apr` and NEVER by a PATH lookup:
    /// four `apr` binaries have coexisted on this dev box and a bare `apr` once
    /// resolved to a 26-day-old copy (CLAUDE.md discipline 3 and 8, T-04-29).
    ///
    /// The status is read from `output.status.success()` — never through a pipe
    /// (CLAUDE.md Verification rule 1).
    ///
    /// Parsing INTO `ClassifyResponse` rather than into a `serde_json::Value` is
    /// deliberate: any CLI re-keying, dropped field or malformed row becomes a
    /// deserialization failure here, and core's validated `Deserialize` also
    /// proves the CLI emitted a well-formed envelope rather than merely valid
    /// JSON.
    fn cli(&self, document: &ClassifyRequestDocument) -> ClassifyResponse {
        let doc_path = self.write_document(document);
        cli_predict(&self.apr_path, &doc_path)
    }

    // -- leg (c): in-process HTTP ---------------------------------------------

    /// POST the SAME serialized document to the REAL router.
    ///
    /// The model in the router's slot is an INDEPENDENT `load_setfit_apr` of the
    /// same bytes, so the artifact-hash equality this gate asserts is a claim
    /// about two loads rather than about one shared object.
    fn http(&self, document: &ClassifyRequestDocument) -> ClassifyResponse {
        let body = Self::serialize(document);
        let model = fixture::load(&self.bytes);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a current-thread runtime is buildable");
        runtime.block_on(async move {
            use realizar::api::{create_router_with_config, AppState, RouterConfig};
            use tower::util::ServiceExt;

            let state = AppState::default().with_setfit_model(std::sync::Arc::new(model));
            let app = create_router_with_config(state, RouterConfig::default());
            let response = app
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri("/v1/classify")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(body))
                        .expect("the request is well formed"),
                )
                .await
                .expect("the router answers");
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("the body is readable");
            assert_eq!(
                status,
                axum::http::StatusCode::OK,
                "the HTTP leg must answer 200; body = {}",
                String::from_utf8_lossy(&bytes)
            );
            serde_json::from_slice::<ClassifyResponse>(&bytes)
                .expect("the HTTP body deserializes INTO core's envelope")
        })
    }
}

/// The spawned-CLI call, factored out so the smoke test and the parity legs
/// resolve the binary through exactly one place.
fn cli_predict(apr_path: &Path, doc_path: &Path) -> ClassifyResponse {
    let output = Command::new(env!("CARGO_BIN_EXE_apr"))
        .arg("predict")
        .arg(apr_path)
        .arg("--input")
        .arg(doc_path)
        .arg("--json")
        .output()
        .expect("the apr binary is spawnable");
    assert!(
        output.status.success(),
        "apr predict failed: status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice::<ClassifyResponse>(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "apr predict --json did not emit core's envelope ({error}); stdout was:\n{}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

// ---------------------------------------------------------------------------
// The comparator — the thing this whole plan is a gate around
// ---------------------------------------------------------------------------

/// The NaN-VISIBLE comparison the contract mandates.
///
/// `matches!(delta.partial_cmp(&bound), Some(Less | Equal))`. A bare
/// `delta <= bound` is a contract violation: it happens to reject NaN in this
/// direction, but refactors into `!(delta > bound)`, which ACCEPTS NaN silently.
fn within(delta: f64, bound: f64) -> bool {
    matches!(
        delta.partial_cmp(&bound),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

/// Every way two surfaces can disagree, named.
///
/// A RETURNED value rather than a panic, so the in-band negative can assert
/// rejection by `matches!` instead of by log inspection.
#[derive(Debug, Clone, PartialEq)]
enum ParityMismatch {
    SchemaVersion {
        left: u32,
        right: u32,
    },
    ArtifactSha256 {
        left: String,
        right: String,
    },
    Backend {
        left: String,
        right: String,
    },
    ResultCount {
        left: usize,
        right: usize,
    },
    Label {
        index: usize,
        left: String,
        right: String,
    },
    ProbabilityArity {
        index: usize,
        left: usize,
        right: usize,
    },
    Probability {
        index: usize,
        class: usize,
        left: f64,
        right: f64,
    },
    LogitsPresence {
        index: usize,
        left: bool,
        right: bool,
    },
    Logit {
        index: usize,
        class: usize,
        left: f64,
        right: f64,
    },
    Margin {
        index: usize,
        left: f64,
        right: f64,
    },
    TokenCount {
        index: usize,
        left: u32,
        right: u32,
    },
    Truncated {
        index: usize,
        left: bool,
        right: bool,
    },
    /// `latency_ms` is checked for FINITENESS and non-negativity only — never for
    /// equality and never for strict positivity.
    LatencyNotAMeasurement {
        side: &'static str,
        value: f64,
    },
}

/// Compare two envelopes at the contract's parity tolerances.
///
/// Exact: labels, `artifact_sha256`, `schema_version`, `backend`, `token_count`,
/// `truncated`, result arity, logits presence.
/// Tolerance `1.0e-5` (`parity_probabilities_abs` / `parity_logits_abs`, the same
/// numbers core exports as `PROBE_*_ABS_TOLERANCE` — read from ONE place):
/// probabilities, logits, margins.
/// Neither: `latency_ms`.
#[allow(clippy::too_many_lines)]
fn compare_parity(left: &ClassifyResponse, right: &ClassifyResponse) -> Result<(), ParityMismatch> {
    for (side, response) in [("left", left), ("right", right)] {
        let latency = response.latency_ms();
        if !latency.is_finite() || latency < 0.0 {
            return Err(ParityMismatch::LatencyNotAMeasurement {
                side,
                value: latency,
            });
        }
    }

    if left.schema_version() != right.schema_version() {
        return Err(ParityMismatch::SchemaVersion {
            left: left.schema_version(),
            right: right.schema_version(),
        });
    }
    if left.artifact_sha256() != right.artifact_sha256() {
        return Err(ParityMismatch::ArtifactSha256 {
            left: left.artifact_sha256().to_string(),
            right: right.artifact_sha256().to_string(),
        });
    }
    if left.backend() != right.backend() {
        return Err(ParityMismatch::Backend {
            left: left.backend().to_string(),
            right: right.backend().to_string(),
        });
    }
    if left.results().len() != right.results().len() {
        return Err(ParityMismatch::ResultCount {
            left: left.results().len(),
            right: right.results().len(),
        });
    }

    for (index, (a, b)) in left
        .results()
        .iter()
        .zip(right.results().iter())
        .enumerate()
    {
        if a.label() != b.label() {
            return Err(ParityMismatch::Label {
                index,
                left: a.label().to_string(),
                right: b.label().to_string(),
            });
        }
        if a.probabilities().len() != b.probabilities().len() {
            return Err(ParityMismatch::ProbabilityArity {
                index,
                left: a.probabilities().len(),
                right: b.probabilities().len(),
            });
        }
        for (class, (p, q)) in a
            .probabilities()
            .iter()
            .zip(b.probabilities().iter())
            .enumerate()
        {
            if !within((p - q).abs(), PROBE_PROBABILITIES_ABS_TOLERANCE) {
                return Err(ParityMismatch::Probability {
                    index,
                    class,
                    left: *p,
                    right: *q,
                });
            }
        }
        match (a.logits(), b.logits()) {
            (Some(x), Some(y)) => {
                if x.len() != y.len() {
                    return Err(ParityMismatch::ProbabilityArity {
                        index,
                        left: x.len(),
                        right: y.len(),
                    });
                }
                for (class, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                    if !within((p - q).abs(), PROBE_LOGITS_ABS_TOLERANCE) {
                        return Err(ParityMismatch::Logit {
                            index,
                            class,
                            left: *p,
                            right: *q,
                        });
                    }
                }
            }
            (None, None) => {}
            (x, y) => {
                return Err(ParityMismatch::LogitsPresence {
                    index,
                    left: x.is_some(),
                    right: y.is_some(),
                })
            }
        }
        if !within(
            (a.margin() - b.margin()).abs(),
            PROBE_PROBABILITIES_ABS_TOLERANCE,
        ) {
            return Err(ParityMismatch::Margin {
                index,
                left: a.margin(),
                right: b.margin(),
            });
        }
        if a.token_count() != b.token_count() {
            return Err(ParityMismatch::TokenCount {
                index,
                left: a.token_count(),
                right: b.token_count(),
            });
        }
        if a.truncated() != b.truncated() {
            return Err(ParityMismatch::Truncated {
                index,
                left: a.truncated(),
                right: b.truncated(),
            });
        }
    }
    Ok(())
}

fn assert_parity(what: &str, left: &ClassifyResponse, right: &ClassifyResponse) {
    if let Err(mismatch) = compare_parity(left, right) {
        panic!("{what}: the surfaces disagree — {mismatch:?}");
    }
}

/// `backend` is an execution-derived IDENTITY (D-12), so it is compared for
/// equality across legs and asserted to carry no CPU-capability token: an
/// availability detection describes the host, never the run that happened.
fn assert_backend_is_a_capability_free_identity(response: &ClassifyResponse) {
    let backend = response.backend();
    assert_eq!(
        backend.split(':').count(),
        3,
        "the backend identity's grammar is three colon-separated segments; got {backend:?}"
    );
    let lowered = backend.to_ascii_lowercase();
    for forbidden in ["avx", "sse", "neon", "simd", "fma"] {
        assert!(
            !lowered.contains(forbidden),
            "backend {backend:?} names a CPU capability ({forbidden}); a capability describes the \
             HOST, not the dispatch that ran (D-12, review B6)"
        );
    }
}

// ---------------------------------------------------------------------------
// The pairwise tests
// ---------------------------------------------------------------------------

#[test]
fn parity_library_and_cli_agree_on_the_batch_document() {
    let harness = Harness::new();
    let document = batch_document();
    let library = harness.library(&document);
    let cli = harness.cli(&document);
    assert_eq!(
        library.artifact_sha256(),
        harness.artifact_sha256(),
        "the library leg reports the hash of the bytes on disk"
    );
    assert_eq!(
        cli.artifact_sha256(),
        harness.artifact_sha256(),
        "the CLI leg reports the hash of the bytes on disk"
    );
    assert_backend_is_a_capability_free_identity(&library);
    assert_parity("library vs CLI (batch)", &library, &cli);
}

#[test]
fn parity_library_and_http_agree_on_the_batch_document() {
    let harness = Harness::new();
    let document = batch_document();
    let library = harness.library(&document);
    let http = harness.http(&document);
    assert_eq!(http.artifact_sha256(), harness.artifact_sha256());
    assert_backend_is_a_capability_free_identity(&http);
    assert_parity("library vs HTTP (batch)", &library, &http);
}

#[test]
fn parity_cli_and_http_agree_on_the_batch_document() {
    let harness = Harness::new();
    let document = batch_document();
    let cli = harness.cli(&document);
    let http = harness.http(&document);
    assert_eq!(cli.artifact_sha256(), harness.artifact_sha256());
    assert_eq!(http.artifact_sha256(), harness.artifact_sha256());
    assert_parity("CLI vs HTTP (batch)", &cli, &http);
    // Core's own `PartialEq` excludes `latency_ms` (F-14(3)), so the plain
    // comparison is the intended one and is asserted alongside the comparator.
    assert_eq!(cli, http, "core's envelope equality holds across surfaces");
}

#[test]
fn parity_all_three_agree_on_a_single_text_document() {
    let harness = Harness::new();
    let document = single_document();
    let library = harness.library(&document);
    let cli = harness.cli(&document);
    let http = harness.http(&document);
    assert_parity("single: library vs CLI", &library, &cli);
    assert_parity("single: library vs HTTP", &library, &http);
    assert_parity("single: CLI vs HTTP", &cli, &http);
    for (leg, response) in [("library", &library), ("cli", &cli), ("http", &http)] {
        assert_eq!(
            response.results().len(),
            1,
            "{leg} returned {} results for a one-text document",
            response.results().len()
        );
    }
}

#[test]
fn parity_all_three_agree_with_logits_requested() {
    let harness = Harness::new();
    let document = logits_document();
    let library = harness.library(&document);
    let cli = harness.cli(&document);
    let http = harness.http(&document);
    for (leg, response) in [("library", &library), ("cli", &cli), ("http", &http)] {
        for (index, result) in response.results().iter().enumerate() {
            let logits = result
                .logits()
                .unwrap_or_else(|| panic!("{leg} result {index} carries no logits"));
            assert_eq!(
                logits.len(),
                result.probabilities().len(),
                "{leg} result {index}: logit arity must equal probability arity"
            );
        }
    }
    assert_parity("logits: library vs CLI", &library, &cli);
    assert_parity("logits: library vs HTTP", &library, &http);
    assert_parity("logits: CLI vs HTTP", &cli, &http);
}

/// The input-fidelity witness (review M2, T-04-53).
///
/// A line-delimited CLI format would have re-split `probe_whitespace` into two
/// texts, so the CLI leg would have returned MORE results than the document had
/// texts while every value it did return still matched. This is the assertion
/// that would have caught it.
#[test]
fn parity_every_leg_returns_one_result_per_text() {
    let harness = Harness::new();
    let document = batch_document();
    let expected = document.texts.len();
    assert!(
        document.texts.iter().any(|t| t.contains('\n')),
        "the committed input set must contain a newline-bearing text or this test is vacuous"
    );
    assert!(
        document.texts.iter().any(|t| t.contains('\t')),
        "the committed input set must contain a tab-bearing text or this test is vacuous"
    );
    assert!(
        document.texts.iter().any(String::is_empty),
        "the committed input set must contain the empty string or this test is vacuous"
    );
    assert!(
        document.texts.iter().any(|t| !t.is_ascii()),
        "the committed input set must contain non-ASCII or this test is vacuous"
    );

    for (leg, response) in [
        ("library", harness.library(&document)),
        ("cli", harness.cli(&document)),
        ("http", harness.http(&document)),
    ] {
        assert_eq!(
            response.results().len(),
            expected,
            "{leg} returned {} results for a {expected}-text document — a surface that \
             re-split or dropped a text",
            response.results().len()
        );
    }
}

/// The truncation-boundary probe must actually truncate, on every leg.
///
/// Without this, `truncated`/`token_count` parity could hold vacuously by both
/// legs reporting `false`/small on every text.
#[test]
fn parity_the_truncation_boundary_text_truncates_on_every_leg() {
    let harness = Harness::new();
    let document = batch_document();
    let boundary = document
        .texts
        .iter()
        .position(|t| t.len() > 1000)
        .expect("the truncation-boundary probe is in the committed input set");
    for (leg, response) in [
        ("library", harness.library(&document)),
        ("cli", harness.cli(&document)),
        ("http", harness.http(&document)),
    ] {
        let result = &response.results()[boundary];
        assert!(
            result.truncated(),
            "{leg}: the truncation-boundary probe must report truncated == true"
        );
        assert!(
            result.token_count() > 0,
            "{leg}: a truncated text consumed a positive number of positions"
        );
    }
}

// ---------------------------------------------------------------------------
// Source assertions over this file
// ---------------------------------------------------------------------------

/// This file's own source.
fn harness_source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/setfit_parity.rs"
    ))
    .expect("the harness can read its own source")
}

/// CODE lines only.
///
/// A guard that scans documentation fails on its own explanation of the rule it
/// enforces — orchestrator finding F-05, observed twice in plan 04-07. The
/// module header above deliberately spells `latency_ms` and describes the
/// line-delimited format this gate forbids; neither is behaviour.
///
/// Deliberately implemented with `split('\n')` rather than the obvious
/// iterator, so the forbidden-needle scan below can be honest about the fact
/// that the harness never splits text into lines.
fn code_lines(source: &str) -> Vec<&str> {
    source
        .split('\n')
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .collect()
}

/// A needle assembled at RUNTIME from fragments.
///
/// A literal needle would appear in the file being scanned — this file — so
/// every `contains` would be satisfied by its own source and the measurement
/// would be zero-information. Assembling means a hit is a real hit.
fn needle(parts: &[&str]) -> String {
    parts.concat()
}

fn count_occurrences(haystack: &[&str], needle: &str) -> usize {
    haystack
        .iter()
        .map(|line| line.matches(needle).count())
        .sum()
}

#[test]
fn parity_harness_resolves_the_binary_through_cargo_and_never_through_path() {
    let source = harness_source();
    let code = code_lines(&source);
    assert!(
        code.len() > 400,
        "the code-line filter ate the file ({} lines survived) — a vacuous scan",
        code.len()
    );

    let cargo_bin = needle(&["CARGO_BIN_", "EXE_apr"]);
    assert!(
        count_occurrences(&code, &cargo_bin) >= 1,
        "the binary must be resolved through cargo's own env var"
    );

    // A PATH lookup is the shadowed-artifact defect (CLAUDE.md discipline 3/8,
    // T-04-29): four `apr` binaries have coexisted on this box.
    for forbidden in [
        needle(&["Command::new(\"a", "pr\")"]),
        needle(&["Command::new(\"a", "pr\".to_string())"]),
    ] {
        assert_eq!(
            count_occurrences(&code, &forbidden),
            0,
            "the harness must never resolve `apr` through PATH ({forbidden})"
        );
    }
}

#[test]
fn parity_harness_never_splits_an_input_text_into_lines() {
    let source = harness_source();
    let code = code_lines(&source);

    // Review M2: a line-delimited CLI format would have re-split
    // `probe_whitespace`, so the three legs would have compared different
    // ordered input sets while reporting agreement.
    let lines_call = needle(&[".li", "nes()"]);
    assert_eq!(
        count_occurrences(&code, &lines_call),
        0,
        "no code line may split text on newlines"
    );

    // The positive half: the CLI leg is handed a JSON document, not texts.
    let input_flag = needle(&["\"--in", "put\""]);
    assert!(
        count_occurrences(&code, &input_flag) >= 1,
        "the CLI leg must pass the request document with the --input flag"
    );
}

#[test]
fn parity_harness_makes_no_assertion_about_latency_being_positive() {
    let source = harness_source();
    let code = code_lines(&source);

    // The review's LOW finding: a fast operation under a coarse timer
    // legitimately reports 0.0, so `> 0` is a flake with a schedule.
    for forbidden in [
        needle(&["latency_ms() >", " 0"]),
        needle(&["latency_ms() >", "= 0.0 &&"]),
        needle(&["latency >", " 0.0"]),
        needle(&["assert!(latency", " >"]),
    ] {
        assert_eq!(
            count_occurrences(&code, &forbidden),
            0,
            "no positivity assertion on latency_ms may exist ({forbidden})"
        );
    }

    // And the finiteness check that IS legitimate must be present, or the
    // absence above would be satisfied by not looking at latency at all.
    let finite = needle(&["latency.is_", "finite()"]);
    assert!(
        count_occurrences(&code, &finite) >= 1,
        "latency is still checked for finiteness"
    );
}

#[test]
fn parity_harness_writes_no_unwrap() {
    let source = harness_source();
    let code = code_lines(&source);
    let bare_unwrap = needle(&[".unw", "rap()"]);
    assert_eq!(
        count_occurrences(&code, &bare_unwrap),
        0,
        "unwrap() is banned crate-wide (.clippy.toml disallowed-methods); the one \
         `allow(clippy::disallowed_methods)` in this file covers `serde_json::json!`'s \
         macro expansion and nothing hand-written"
    );
}

#[test]
fn parity_harness_builds_its_fixture_from_the_full_pin_shape_recipe() {
    let source = harness_source();
    let code = code_lines(&source);
    let recipe = needle(&["fixture_view_full_", "pin_shape"]);
    assert!(
        count_occurrences(&code, &recipe) >= 2,
        "the fixture must be built through the core `fixture_view_full_pin_shape()` recipe \
         (definition + call), so the parity legs run on the production nullability shape"
    );
    // The two production doors, not a hand-rolled container writer.
    for door in [
        needle(&["write_setfit_", "apr("]),
        needle(&["load_setfit_", "apr("]),
    ] {
        assert!(
            count_occurrences(&code, &door) >= 1,
            "the fixture goes through core's own door ({door})"
        );
    }
}

// ---------------------------------------------------------------------------
// Frozen goldens + the SHA-256 manifest (Ph1 D-13)
// ---------------------------------------------------------------------------
//
// The live pairwise comparison proves the three surfaces AGREE. It cannot prove
// they still agree on the SAME ANSWER they agreed on yesterday: three surfaces
// that all changed together are still in perfect agreement. The goldens are the
// "nobody moved" pin, and the manifest is what makes editing them a visible act
// rather than a silent one.

/// One frozen result row. NO latency: a measurement cannot be frozen.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GoldenResult {
    label: String,
    probabilities: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logits: Option<Vec<f64>>,
    margin: f64,
    token_count: u32,
    truncated: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GoldenDocument {
    id: String,
    text_count: usize,
    include_logits: bool,
    results: Vec<GoldenResult>,
}

/// The frozen library-leg answers on the committed request documents.
///
/// `backend` is recorded as an EXACT string because it is an identity (D-12).
/// `artifact_sha256` is recorded because a moved fixture is exactly what this
/// pin exists to make loud.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Goldens {
    note: String,
    schema_version: u32,
    artifact_sha256: String,
    backend: String,
    documents: Vec<GoldenDocument>,
}

fn goldens_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/setfit_parity"
    ))
}

fn goldens_path() -> std::path::PathBuf {
    goldens_dir().join("goldens.json")
}

fn manifest_path() -> std::path::PathBuf {
    goldens_dir().join("goldens.sha256")
}

/// The three committed documents, in the order the goldens record them.
fn golden_documents() -> Vec<(&'static str, ClassifyRequestDocument)> {
    vec![
        ("batch", batch_document()),
        ("single", single_document()),
        ("batch_with_logits", logits_document()),
    ]
}

fn goldens_from_library(harness: &Harness) -> Goldens {
    let mut documents = Vec::new();
    let mut backend = String::new();
    let mut schema_version = 0;
    for (id, document) in golden_documents() {
        let response = harness.library(&document);
        backend = response.backend().to_string();
        schema_version = response.schema_version();
        documents.push(GoldenDocument {
            id: id.to_string(),
            text_count: document.texts.len(),
            include_logits: document.include_logits,
            results: response
                .results()
                .iter()
                .map(|r| GoldenResult {
                    label: r.label().to_string(),
                    probabilities: r.probabilities().to_vec(),
                    logits: r.logits().map(<[f64]>::to_vec),
                    margin: r.margin(),
                    token_count: r.token_count(),
                    truncated: r.truncated(),
                })
                .collect(),
        });
    }
    Goldens {
        note: "Frozen library-leg answers for the 04-09 three-surface parity gate. \
               Blessing is deliberate: set APR_BLESS_SETFIT_PARITY_GOLDENS=1 and the \
               goldens.json + goldens.sha256 pair is rewritten together. Reviewing the \
               resulting git diff is the control; the manifest is what makes an edit to \
               goldens.json ALONE fail loudly."
            .to_string(),
        schema_version,
        artifact_sha256: harness.artifact_sha256(),
        backend,
        documents,
    }
}

/// A manifest line is `<64 hex>  <filename>` — the `sha256sum` shape.
fn manifest_expectation(manifest: &str) -> Result<(String, String), String> {
    let line = manifest
        .split('\n')
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .ok_or_else(|| "the manifest carries no entry".to_string())?;
    let mut parts = line.split_whitespace();
    let hash = parts
        .next()
        .ok_or_else(|| format!("malformed manifest line: {line:?}"))?;
    let name = parts
        .next()
        .ok_or_else(|| format!("manifest line names no file: {line:?}"))?;
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("manifest hash is not 64 hex chars: {hash:?}"));
    }
    Ok((hash.to_string(), name.to_string()))
}

/// The manifest check, as a FUNCTION so the flipped-byte negative can call it on
/// mutated bytes without touching the file on disk.
fn check_manifest(goldens_bytes: &[u8], manifest: &str) -> Result<(), String> {
    let (expected, name) = manifest_expectation(manifest)?;
    if name != "goldens.json" {
        return Err(format!("the manifest names {name:?}, not goldens.json"));
    }
    let observed = artifact_sha256_hex(goldens_bytes);
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "goldens.json does not match its manifest: expected {expected}, observed {observed}"
        ))
    }
}

fn bless_requested() -> bool {
    std::env::var("APR_BLESS_SETFIT_PARITY_GOLDENS").is_ok_and(|v| v == "1")
}

/// Compare a document's frozen rows against live ones.
///
/// The ONE row comparison, used by the library golden test AND by the spawned
/// smoke test — so "the served answers are the frozen answers" is the same claim
/// in both places rather than two claims that can drift apart.
///
/// Exact: label, probability arity, logits presence and arity, `token_count`,
/// `truncated`. Tolerance: probabilities, logits, margins.
fn assert_golden_rows(id: &str, frozen: &[GoldenResult], live: &[GoldenResult]) {
    assert_eq!(frozen.len(), live.len(), "{id}: result count changed");

    // NON-VACUITY. The fixture's head is random, and it happens to produce a
    // CONSTANT argmax across every probe — so a labels-only comparison here
    // would be nearly information-free (a reversed row order still passes it;
    // measured). The rows differ in probability and token facts by orders of
    // magnitude more than the tolerance, and that is what this comparison is
    // actually resting on. Asserted, so the day a fixture change flattens the
    // rows this test says so instead of quietly proving nothing.
    if frozen.len() > 1 {
        assert!(
            frozen.iter().any(|r| !within(
                (r.probabilities[0] - frozen[0].probabilities[0]).abs(),
                PROBE_PROBABILITIES_ABS_TOLERANCE
            )),
            "{id}: every frozen row carries the same leading probability — this comparison \
             cannot distinguish a row permutation and is therefore vacuous"
        );
        assert!(
            frozen
                .iter()
                .any(|r| r.token_count != frozen[0].token_count),
            "{id}: every frozen row carries the same token_count — vacuous"
        );
    }

    for (index, (a, b)) in frozen.iter().zip(live.iter()).enumerate() {
        assert_eq!(
            a.label, b.label,
            "{id} result {index}: label moved (labels are compared EXACTLY)"
        );
        assert_eq!(
            a.probabilities.len(),
            b.probabilities.len(),
            "{id} result {index}: probability arity moved"
        );
        for (class, (p, q)) in a
            .probabilities
            .iter()
            .zip(b.probabilities.iter())
            .enumerate()
        {
            assert!(
                within((p - q).abs(), PROBE_PROBABILITIES_ABS_TOLERANCE),
                "{id} result {index} class {class}: probability moved {p} -> {q}"
            );
        }
        match (a.logits.as_ref(), b.logits.as_ref()) {
            (Some(x), Some(y)) => {
                assert_eq!(x.len(), y.len(), "{id} result {index}: logit arity moved");
                for (class, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                    assert!(
                        within((p - q).abs(), PROBE_LOGITS_ABS_TOLERANCE),
                        "{id} result {index} class {class}: logit moved {p} -> {q}"
                    );
                }
            }
            (None, None) => {}
            (x, y) => panic!(
                "{id} result {index}: logits presence moved ({} -> {})",
                x.is_some(),
                y.is_some()
            ),
        }
        assert!(
            within(
                (a.margin - b.margin).abs(),
                PROBE_PROBABILITIES_ABS_TOLERANCE
            ),
            "{id} result {index}: margin moved {} -> {}",
            a.margin,
            b.margin
        );
        assert_eq!(
            a.token_count, b.token_count,
            "{id} result {index}: token_count moved"
        );
        assert_eq!(
            a.truncated, b.truncated,
            "{id} result {index}: truncated moved"
        );
    }
}

#[test]
fn golden_library_answers_are_unchanged() {
    let harness = Harness::new();
    let live = goldens_from_library(&harness);

    if bless_requested() {
        let rendered = serde_json::to_string_pretty(&live).expect("the goldens serialize") + "\n";
        std::fs::create_dir_all(goldens_dir()).expect("the goldens directory is creatable");
        std::fs::write(goldens_path(), rendered.as_bytes()).expect("goldens.json is writable");
        let manifest = format!(
            "# SHA-256 manifest for the 04-09 parity goldens (Ph1 D-13).\n\
             # Regenerate together with goldens.json:\n\
             #   APR_BLESS_SETFIT_PARITY_GOLDENS=1 cargo test -p apr-cli \\\n\
             #     --features setfit,inference --test setfit_parity golden\n\
             {}  goldens.json\n",
            artifact_sha256_hex(rendered.as_bytes())
        );
        std::fs::write(manifest_path(), manifest).expect("goldens.sha256 is writable");
        return;
    }

    let frozen_bytes = std::fs::read(goldens_path()).expect(
        "tests/fixtures/setfit_parity/goldens.json is committed; bless it with \
         APR_BLESS_SETFIT_PARITY_GOLDENS=1 if it is genuinely absent",
    );
    let frozen: Goldens =
        serde_json::from_slice(&frozen_bytes).expect("the committed goldens deserialize");

    assert_eq!(
        frozen.schema_version, live.schema_version,
        "the envelope's schema version moved"
    );
    assert_eq!(
        frozen.backend, live.backend,
        "the backend IDENTITY moved — this is an exact comparison because a backend is an \
         identity, not a measurement (D-12)"
    );
    assert_eq!(
        frozen.artifact_sha256, live.artifact_sha256,
        "the FIXTURE ARTIFACT's bytes moved. If every value below still matches at tolerance, \
         the model did not change and only the container did — re-bless deliberately and put \
         the reason in the commit message"
    );
    assert_eq!(
        frozen.documents.len(),
        live.documents.len(),
        "the committed document set changed"
    );

    for (frozen_doc, live_doc) in frozen.documents.iter().zip(live.documents.iter()) {
        assert_eq!(frozen_doc.id, live_doc.id, "document order changed");
        assert_eq!(
            frozen_doc.text_count, live_doc.text_count,
            "{}: the committed input set changed length",
            frozen_doc.id
        );
        assert_eq!(
            frozen_doc.include_logits, live_doc.include_logits,
            "{}: include_logits changed",
            frozen_doc.id
        );
        assert_eq!(
            frozen_doc.results.len(),
            live_doc.results.len(),
            "{}: result count changed",
            frozen_doc.id
        );
        assert_golden_rows(&frozen_doc.id, &frozen_doc.results, &live_doc.results);
    }
}

#[test]
fn golden_file_matches_its_committed_manifest() {
    let bytes = std::fs::read(goldens_path()).expect("goldens.json is committed");
    let manifest = std::fs::read_to_string(manifest_path()).expect("goldens.sha256 is committed");
    check_manifest(&bytes, &manifest).expect("the committed goldens match their manifest");
}

/// NEGATIVE 2: a flipped byte in a golden must fail the manifest check.
///
/// A manifest that has only ever been observed passing is theatre (CLAUDE.md
/// Verification Discipline 5). This runs the mutation in memory in EVERY cargo
/// test invocation, so the check is proven able to fire without anyone having to
/// remember to edit a file.
#[test]
fn golden_a_single_flipped_byte_fails_the_manifest() {
    let bytes = std::fs::read(goldens_path()).expect("goldens.json is committed");
    let manifest = std::fs::read_to_string(manifest_path()).expect("goldens.sha256 is committed");
    check_manifest(&bytes, &manifest).expect("the un-mutated bytes pass — non-vacuity");

    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    let verdict = check_manifest(&tampered, &manifest);
    assert!(
        verdict.is_err(),
        "one flipped byte must fail the manifest; got {verdict:?}"
    );
    let message = verdict.unwrap_err();
    assert!(
        message.contains("does not match its manifest"),
        "the failure must name the mismatch: {message}"
    );

    // And a manifest whose hash line is edited must fail too — otherwise the
    // check could be satisfied by rewriting the cheaper of the two files.
    let forged = manifest.replace(
        &manifest_expectation(&manifest)
            .expect("the committed manifest parses")
            .0,
        &"0".repeat(64),
    );
    assert!(
        check_manifest(&bytes, &forged).is_err(),
        "an edited manifest hash must fail against untouched goldens"
    );
}

// ---------------------------------------------------------------------------
// The in-band negative — the gate proven able to fail, in every cargo test run
// ---------------------------------------------------------------------------
//
// House discipline chain: Ph1 D-24 / Ph2 D-25 / Ph3 D-08 / this plan. "A gate
// that can go vacuous is not a gate": a comparator that has only ever been
// handed equal values is not evidence that it can tell unequal ones apart.

/// Rebuild `response` with ONE probability perturbed by 10x the parity
/// tolerance, through the envelope's PUBLIC validating constructors.
///
/// **No backdoor is needed and none is shipped.** The perturbed value is finite
/// and the perturbation is MASS-PRESERVING (+delta on class 0, -delta on class
/// 1), so `ClassifyResult::new`'s finiteness and probability-mass checks both
/// accept it. That is the point: the gate under test here is the COMPARATOR, not
/// the constructor, so the negative must be a value the type system and the
/// constructor consider entirely legitimate.
fn skewed_response(response: &ClassifyResponse) -> ClassifyResponse {
    let skew = 10.0 * PROBE_PROBABILITIES_ABS_TOLERANCE;
    let mut results = Vec::new();
    for (index, result) in response.results().iter().enumerate() {
        let mut probabilities = result.probabilities().to_vec();
        assert!(
            probabilities.len() >= 2,
            "a mass-preserving skew needs at least two classes"
        );
        if index == 0 {
            probabilities[0] += skew;
            probabilities[1] -= skew;
        }
        results.push(
            aprender::setfit::ClassifyResult::new(
                result.label().to_string(),
                probabilities,
                result.logits().map(<[f64]>::to_vec),
                result.margin(),
                result.token_count(),
                result.truncated(),
            )
            .expect("a finite, mass-preserving perturbation is a legal ClassifyResult"),
        );
    }
    ClassifyResponse::new(
        response.artifact_sha256().to_string(),
        response.backend().to_string(),
        response.latency_ms(),
        results,
    )
    .expect("a legal envelope")
}

/// NEGATIVE 1: the comparator REJECTS a skewed response, and says which value.
#[test]
fn golden_negative_the_comparator_rejects_a_skewed_probability() {
    let harness = Harness::new();
    let document = batch_document();
    let good = harness.library(&document);

    // Non-vacuity: the comparator accepts the unskewed value first, so a
    // comparator that rejected everything could not pass this test.
    compare_parity(&good, &good).expect("a response agrees with itself");

    let skewed = skewed_response(&good);
    let verdict = compare_parity(&good, &skewed);
    assert!(
        matches!(
            verdict,
            Err(ParityMismatch::Probability {
                index: 0,
                class: 0,
                ..
            })
        ),
        "a 10x-tolerance probability skew must be REJECTED and named; got {verdict:?}"
    );

    // The same rejection through the assertion wrapper the pairwise tests use —
    // captured as a panic, never by log inspection. The hook is silenced only
    // for the expected panic, then restored.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(|| assert_parity("negative", &good, &skewed));
    std::panic::set_hook(previous);
    assert!(
        caught.is_err(),
        "assert_parity must panic on a skewed response"
    );
}

/// NEGATIVE 1b: a skew SMALLER than the tolerance is accepted.
///
/// Without this the rejection above would be consistent with a comparator that
/// rejects any two distinct floats, which is a different (and useless) gate.
#[test]
fn golden_negative_a_sub_tolerance_perturbation_is_still_parity() {
    let harness = Harness::new();
    let good = harness.library(&batch_document());
    let tiny = PROBE_PROBABILITIES_ABS_TOLERANCE / 10.0;
    let mut results = Vec::new();
    for (index, result) in good.results().iter().enumerate() {
        let mut probabilities = result.probabilities().to_vec();
        if index == 0 {
            probabilities[0] += tiny;
            probabilities[1] -= tiny;
        }
        results.push(
            aprender::setfit::ClassifyResult::new(
                result.label().to_string(),
                probabilities,
                result.logits().map(<[f64]>::to_vec),
                result.margin(),
                result.token_count(),
                result.truncated(),
            )
            .expect("a legal result"),
        );
    }
    let nudged = ClassifyResponse::new(
        good.artifact_sha256().to_string(),
        good.backend().to_string(),
        good.latency_ms(),
        results,
    )
    .expect("a legal envelope");
    compare_parity(&good, &nudged)
        .expect("a perturbation an order of magnitude BELOW the bound is still parity");
}

/// NEGATIVE 3: the envelope's validated `Deserialize` is live at the surface
/// boundary (review M1).
///
/// Both legs that cross a process or socket boundary parse INTO
/// `ClassifyResponse`, so this is what makes "the CLI/HTTP body deserialized"
/// mean more than "it was syntactically JSON".
#[test]
fn golden_negative_a_malformed_envelope_does_not_deserialize() {
    // (a) a `null` probability — serde's own type check.
    let with_null = r#"{
        "schema_version": 1,
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "backend": "cpu:setfit-core:autograd-trueno-matmul",
        "latency_ms": 1.0,
        "results": [{
            "label": "favor",
            "probabilities": [null, 0.5, 0.5],
            "margin": 0.0,
            "token_count": 4,
            "truncated": false
        }]
    }"#;
    let verdict = serde_json::from_str::<ClassifyResponse>(with_null);
    assert!(
        verdict.is_err(),
        "a null probability must not deserialize into the envelope; got {verdict:?}"
    );

    // (b) a probability row that does not sum to 1 — the VALIDATING TryFrom,
    //     which a plain serde type check would have waved through.
    let bad_mass = r#"{
        "schema_version": 1,
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "backend": "cpu:setfit-core:autograd-trueno-matmul",
        "latency_ms": 1.0,
        "results": [{
            "label": "favor",
            "probabilities": [0.5, 0.5, 0.5],
            "margin": 0.0,
            "token_count": 4,
            "truncated": false
        }]
    }"#;
    let verdict = serde_json::from_str::<ClassifyResponse>(bad_mass);
    assert!(
        verdict.is_err(),
        "a probability row summing to 1.5 must not deserialize; got {verdict:?}"
    );

    // Non-vacuity: the same shape with a legal row DOES deserialize, so the two
    // rejections above are about the values and not about the schema.
    let legal = r#"{
        "schema_version": 1,
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "backend": "cpu:setfit-core:autograd-trueno-matmul",
        "latency_ms": 1.0,
        "results": [{
            "label": "favor",
            "probabilities": [0.25, 0.5, 0.25],
            "margin": 0.25,
            "token_count": 4,
            "truncated": false
        }]
    }"#;
    serde_json::from_str::<ClassifyResponse>(legal)
        .expect("a well-formed envelope deserializes — non-vacuity for the two rejections above");
}

/// The skewed helper must use the PUBLIC constructors — no test-only door, no
/// `pub(crate)` backdoor, no field mutation.
#[test]
fn golden_negative_uses_the_public_validating_constructors() {
    let source = harness_source();
    let code = code_lines(&source);
    for constructor in [
        needle(&["ClassifyResponse::", "new("]),
        needle(&["ClassifyResult::", "new("]),
    ] {
        assert!(
            count_occurrences(&code, &constructor) >= 1,
            "the negative must be built through {constructor}"
        );
    }
    // There is no such door in core, and this asserts none was invented here.
    for backdoor in [
        needle(&["for_", "tests("]),
        needle(&["_unchec", "ked("]),
        needle(&["skip_valid", "ation"]),
    ] {
        assert_eq!(
            count_occurrences(&code, &backdoor),
            0,
            "no validation backdoor may appear in this harness ({backdoor})"
        );
    }
}

// ---------------------------------------------------------------------------
// The ONE tier3 spawned-serve smoke test (D-14)
// ---------------------------------------------------------------------------
//
// D-14's shape, and why it is exactly one: an all-spawned suite is the flaky-gate
// failure mode (every test racing a socket, a process and a scheduler), while an
// in-process-only suite never proves the INSTALLED binary serves anything. So the
// deterministic HTTP leg above runs in every `cargo test` and this one — the only
// test in the repository that starts a real `apr serve` — is `#[ignore]`d and run
// by the tier3 target plan 04-10 wires.
//
// No HTTP client dependency is added: the round trip is written directly on
// `std::net::TcpStream` with `Connection: close`. T-04-SC — zero new packages
// this phase — is a constraint on this file too.

/// A bounded, drained pipe reader.
///
/// A chatty child that fills a 64 KiB pipe buffer BLOCKS on write, and a parent
/// that only reads after `wait()` deadlocks forever. Both streams are therefore
/// drained on their own threads from the moment the child exists, into a buffer
/// capped so a runaway child cannot exhaust the test runner's memory either.
struct DrainedOutput {
    text: std::sync::Arc<std::sync::Mutex<String>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

const MAX_CAPTURED_CHILD_BYTES: usize = 64 * 1024;

impl DrainedOutput {
    fn spawn<R: std::io::Read + Send + 'static>(mut stream: R) -> Self {
        let text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink = std::sync::Arc::clone(&text);
        let handle = std::thread::spawn(move || {
            let mut buffer = [0u8; 4096];
            loop {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if let Ok(mut guard) = sink.lock() {
                            if guard.len() < MAX_CAPTURED_CHILD_BYTES {
                                guard.push_str(&String::from_utf8_lossy(&buffer[..n]));
                            }
                        }
                    }
                }
            }
        });
        Self {
            text,
            handle: Some(handle),
        }
    }

    fn snapshot(&self) -> String {
        self.text
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "<capture thread poisoned>".to_string())
    }

    fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Kills and REAPS the child on every exit path, including a panic.
///
/// The exit status is read from the reaped `ExitStatus` — never inferred from a
/// pipe (CLAUDE.md Verification rule 1).
struct ServeChild {
    child: Option<std::process::Child>,
    stdout: DrainedOutput,
    stderr: DrainedOutput,
    port: u16,
}

impl ServeChild {
    fn tail(&self) -> String {
        format!(
            "--- child stdout ---\n{}\n--- child stderr ---\n{}",
            self.stdout.snapshot(),
            self.stderr.snapshot()
        )
    }

    /// Has the child already exited? Read WITHOUT blocking, so a child that
    /// died on a bind error is detected instead of waited out.
    fn exited(&mut self) -> Option<std::process::ExitStatus> {
        self.child
            .as_mut()
            .and_then(|c| c.try_wait().ok().flatten())
    }
}

impl Drop for ServeChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.stdout.join();
        self.stderr.join();
    }
}

#[derive(Debug)]
struct SmokeFailure {
    message: String,
    /// A port that was reserved and then taken by someone else in the release
    /// window is not a defect in anything this plan owns — retry it.
    address_in_use: bool,
}

/// One raw HTTP/1.1 round trip on loopback.
///
/// Returns `(status, body)`. `Connection: close` makes read-to-EOF the framing,
/// and the chunked case is de-framed explicitly rather than assumed away.
fn http_round_trip(port: u16, request: &str) -> Result<(u16, String), String> {
    use std::io::{Read as _, Write as _};

    let mut stream =
        std::net::TcpStream::connect(("127.0.0.1", port)).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .map_err(|e| format!("set_read_timeout: {e}"))?;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().map_err(|e| format!("flush: {e}"))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("read: {e}"))?;
    let text = String::from_utf8_lossy(&raw).into_owned();

    let split = text
        .find("\r\n\r\n")
        .ok_or_else(|| format!("no header terminator in response: {text:?}"))?;
    let (head, rest) = text.split_at(split);
    let body_raw = &rest[4..];

    let status_line = head
        .split("\r\n")
        .next()
        .ok_or_else(|| "empty response".to_string())?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| format!("malformed status line: {status_line:?}"))?
        .parse()
        .map_err(|e| format!("unparseable status in {status_line:?}: {e}"))?;

    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(body_raw)?
    } else {
        body_raw.to_string()
    };
    Ok((status, body))
}

fn dechunk(body: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = body;
    loop {
        let end = rest
            .find("\r\n")
            .ok_or_else(|| "truncated chunk header".to_string())?;
        let size = usize::from_str_radix(rest[..end].trim(), 16)
            .map_err(|e| format!("bad chunk size {:?}: {e}", &rest[..end]))?;
        rest = &rest[end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if rest.len() < size {
            return Err("truncated chunk body".to_string());
        }
        out.push_str(&rest[..size]);
        rest = &rest[size + 2..];
    }
}

fn get_request(port: u16, path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
}

fn post_json_request(port: u16, path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Reserve a concrete port by binding an ephemeral one and releasing it.
///
/// Review finding: `127.0.0.1:0` cannot be used by the PARENT unless the chosen
/// port is communicated back, and a fixed high port is collision-prone. So the
/// parent picks, releases, and hands the number to the child — and the
/// reserve-then-release race window is handled by RETRYING the whole sequence
/// rather than pretended away.
fn reserve_port() -> Result<u16, String> {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| format!("reserve bind: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("local_addr: {e}"))?
        .port();
    drop(listener);
    Ok(port)
}

struct SmokeReport {
    port: u16,
    readiness_ms: u128,
    polls: u32,
    rows: Vec<GoldenResult>,
}

/// Startup -> readiness -> one classify round trip, against the INSTALLED binary.
fn spawned_serve_round_trip(harness: &Harness) -> Result<SmokeReport, SmokeFailure> {
    let fail = |message: String| SmokeFailure {
        message,
        address_in_use: false,
    };

    let port = reserve_port().map_err(fail)?;

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_apr"))
        .arg("serve")
        .arg("run")
        .arg(&harness.apr_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| fail(format!("spawning apr serve: {e}")))?;

    let stdout = DrainedOutput::spawn(
        child
            .stdout
            .take()
            .ok_or_else(|| fail("no stdout pipe".to_string()))?,
    );
    let stderr = DrainedOutput::spawn(
        child
            .stderr
            .take()
            .ok_or_else(|| fail("no stderr pipe".to_string()))?,
    );
    let mut guard = ServeChild {
        child: Some(child),
        stdout,
        stderr,
        port,
    };

    // (1) READINESS, bounded: 50 polls x 100 ms.
    let started = std::time::Instant::now();
    let mut ready_body = String::new();
    let mut polls = 0u32;
    for attempt in 1..=50u32 {
        polls = attempt;
        if let Some(status) = guard.exited() {
            let tail = guard.tail();
            let address_in_use = tail.to_ascii_lowercase().contains("address already in use")
                || tail.contains("Bind:");
            return Err(SmokeFailure {
                message: format!(
                    "apr serve exited before becoming ready (status {status:?}, port {port})\n{tail}"
                ),
                address_in_use,
            });
        }
        if let Ok((200, body)) =
            http_round_trip(guard.port, &get_request(guard.port, "/health/ready"))
        {
            ready_body = body;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let readiness_ms = started.elapsed().as_millis();
    if ready_body.is_empty() {
        return Err(fail(format!(
            "apr serve never answered 200 on /health/ready within 5s (port {port})\n{}",
            guard.tail()
        )));
    }

    // (2) READINESS REPORTS THE ARTIFACT (OPS-05).
    let ready: serde_json::Value = serde_json::from_str(&ready_body)
        .map_err(|e| fail(format!("readiness body is not JSON ({e}): {ready_body:?}")))?;
    let reported = ready
        .get("classifier_artifact_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            fail(format!(
                "readiness carries no classifier hash: {ready_body}"
            ))
        })?;
    if reported != harness.artifact_sha256() {
        return Err(fail(format!(
            "readiness reports {reported}, the fixture file hashes to {}",
            harness.artifact_sha256()
        )));
    }
    if ready
        .get("classifier_verified")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return Err(fail(format!(
            "readiness does not report a verified classifier: {ready_body}"
        )));
    }

    // (3) ONE CLASSIFY ROUND TRIP, on the SAME committed document.
    let document = batch_document();
    let body = Harness::serialize(&document);
    let (status, response_body) = http_round_trip(
        guard.port,
        &post_json_request(guard.port, "/v1/classify", &body),
    )
    .map_err(|e| fail(format!("classify round trip: {e}\n{}", guard.tail())))?;
    if status != 200 {
        return Err(fail(format!(
            "POST /v1/classify answered {status}: {response_body}\n{}",
            guard.tail()
        )));
    }
    let response: ClassifyResponse = serde_json::from_str(&response_body).map_err(|e| {
        fail(format!(
            "the spawned server's body is not the envelope ({e}): {response_body}"
        ))
    })?;
    if response.artifact_sha256() != harness.artifact_sha256() {
        return Err(fail(format!(
            "the served response names artifact {}, the fixture is {}",
            response.artifact_sha256(),
            harness.artifact_sha256()
        )));
    }

    let rows: Vec<GoldenResult> = response
        .results()
        .iter()
        .map(|r| GoldenResult {
            label: r.label().to_string(),
            probabilities: r.probabilities().to_vec(),
            logits: r.logits().map(<[f64]>::to_vec),
            margin: r.margin(),
            token_count: r.token_count(),
            truncated: r.truncated(),
        })
        .collect();

    // (4) The child is killed and REAPED by `guard`'s Drop on the way out.
    Ok(SmokeReport {
        port,
        readiness_ms,
        polls,
        rows,
    })
}

/// The rows the goldens froze for the batch document.
fn golden_batch_rows() -> Vec<GoldenResult> {
    let bytes = std::fs::read(goldens_path()).expect("goldens.json is committed");
    let goldens: Goldens = serde_json::from_slice(&bytes).expect("the goldens deserialize");
    goldens
        .documents
        .into_iter()
        .find(|d| d.id == "batch")
        .expect("the goldens record the batch document")
        .results
}

/// The ONE spawned-serve test. Run by the tier3 target (plan 04-10) as
/// `cargo test -p apr-cli --features setfit,inference --test setfit_parity -- \
///  --ignored spawned_serve_smoke`.
#[test]
#[ignore = "tier3: spawns a real `apr serve` and binds a loopback port"]
fn spawned_serve_smoke() {
    let harness = Harness::new();
    let expected = golden_batch_rows();

    let mut attempts = Vec::new();
    for attempt in 1..=3u32 {
        match spawned_serve_round_trip(&harness) {
            Ok(report) => {
                // The FULL frozen rows, not just the labels: the fixture's random
                // head produces a constant argmax, so a labels-only comparison
                // here would pass a reversed row order (measured, then fixed).
                assert_golden_rows("spawned batch", &expected, &report.rows);
                println!(
                    "spawned_serve_smoke: attempt {attempt}, port {}, ready after {} ms \
                     ({} poll(s)), {} rows matched the frozen goldens",
                    report.port,
                    report.readiness_ms,
                    report.polls,
                    report.rows.len()
                );
                return;
            }
            Err(failure) if failure.address_in_use && attempt < 3 => {
                attempts.push(format!("attempt {attempt}: {}", failure.message));
            }
            Err(failure) => {
                attempts.push(format!("attempt {attempt}: {}", failure.message));
                panic!("spawned serve smoke failed:\n{}", attempts.join("\n"));
            }
        }
    }
    panic!(
        "spawned serve smoke lost the reserved port three times:\n{}",
        attempts.join("\n")
    );
}

#[test]
fn parity_harness_spawns_exactly_one_serve_process() {
    let source = harness_source();
    let code = code_lines(&source);

    // D-14: exactly one spawned-serve site. More would be the flaky-gate shape
    // this decision exists to prevent.
    let serve_arg = needle(&["arg(\"se", "rve\")"]);
    assert_eq!(
        count_occurrences(&code, &serve_arg),
        1,
        "exactly one code site starts a server process"
    );

    // The port must be REAL and handed over, never a literal and never 0.
    let ephemeral = needle(&["TcpListener::bind(\"127.0.0.1", ":0\")"]);
    assert_eq!(
        count_occurrences(&code, &ephemeral),
        1,
        "the parent reserves the port by binding an ephemeral one"
    );
    let port_flag = needle(&["\"--po", "rt\""]);
    assert_eq!(
        count_occurrences(&code, &port_flag),
        1,
        "the reserved port is handed to the child explicitly"
    );
    for literal in [
        needle(&["arg(\"80", "80\")"]),
        needle(&["arg(\"1", "1434\")"]),
        needle(&["port(\"", "0\")"]),
    ] {
        assert_eq!(
            count_occurrences(&code, &literal),
            0,
            "no fixed port literal may be passed to the child ({literal})"
        );
    }

    // Both pipes drained on threads, and the cleanup guard implements Drop.
    assert_eq!(
        count_occurrences(&code, &needle(&["DrainedOutput::sp", "awn("])),
        2,
        "exactly two call sites — child stdout and child stderr are BOTH drained on \
         their own threads, so a chatty child cannot fill a pipe and deadlock"
    );
    assert_eq!(
        count_occurrences(&code, &needle(&["std::thread::sp", "awn("])),
        1,
        "the drain thread is the one place a thread is spawned"
    );
    assert_eq!(
        count_occurrences(&code, &needle(&["impl Drop for Serve", "Child"])),
        1,
        "the child is reaped by a Drop guard, so no orphan survives a panic"
    );
}
