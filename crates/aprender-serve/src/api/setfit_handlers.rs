//! `POST /v1/classify` — the SetFit classification surface (D-09, OPS-05).
//!
//! # This module is TRANSPORT. All of it.
//!
//! CLAUDE.md's realizar-first table carries one deliberate exception, and this is
//! it: SetFit classification is `aprender-core`'s, because core's fixture-verified
//! graph path is the only conformance-proven implementation there is. What lives
//! here is the route, the `AppState` slot and the readiness fields. What does NOT
//! live here — and must never — is a tokenizer, a pooling rule, a head, or a
//! request or response schema.
//!
//! Concretely: the extractor is core's [`ClassifyRequestDocument`] and the success
//! body is core's [`ClassifyResponse`]. The ONLY struct this module serializes
//! that it did not get from core is the pre-existing transport
//! [`ErrorResponse`](super::ErrorResponse), which every other handler in this
//! directory already uses. A serve-local request or response type would make "the
//! Rust API, the CLI and HTTP agree" a claim about three independently-maintained
//! structs; routing core's own types through the wire makes it true by
//! construction (classify.rs's module header states the same rule from the other
//! side).
//!
//! # Why the bounds are checked here AND in core
//!
//! [`MAX_BATCH_TEXTS`] is re-checked by `VerifiedSetFitModel::classify` before it
//! tokenizes anything. That is not a reason to omit it here — it is the reason
//! the check here uses core's CONSTANT rather than a literal `256`. Two numbers
//! would be two bounds; one constant checked twice is defense in depth (T-04-23).

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use aprender::setfit::classify::{
    ClassifyError, ClassifyRequestDocument, ClassifyResponse, MAX_BATCH_TEXTS,
    MAX_REQUEST_BODY_BYTES,
};
use aprender::setfit::VerifiedSetFitModel;

use super::{AppState, ErrorResponse};

// ---------------------------------------------------------------------------
// The AppState slot
// ---------------------------------------------------------------------------

impl AppState {
    /// Install a verified SetFit classifier.
    ///
    /// A consuming builder in the shape of `with_verbose`/`with_trace`, so a
    /// state may hold a classifier alongside anything else it already holds.
    /// Composed with [`AppState::default`] it spells the classifier-only server
    /// `apr serve` builds for a SetFit APR.
    ///
    /// There is no `set_` variant and no way to install an unverified model: the
    /// parameter type is the witness, and it is not constructible outside
    /// `aprender-core` (APR-04).
    #[must_use]
    pub fn with_setfit_model(mut self, model: Arc<VerifiedSetFitModel>) -> Self {
        self.setfit_model = Some(model);
        self
    }

    /// The resident classifier, if any.
    #[must_use]
    pub fn setfit_model(&self) -> Option<&Arc<VerifiedSetFitModel>> {
        self.setfit_model.as_ref()
    }
}

// ---------------------------------------------------------------------------
// The route's body bound
// ---------------------------------------------------------------------------

/// The contract's `max_request_body_bytes`, as axum's layer wants it.
///
/// A function and not a `const` because the contract states the bound in `u64`
/// and the layer takes `usize`; `unwrap_or(usize::MAX)` is unreachable on every
/// target this crate builds for (a 16-bit target could not host axum), and
/// saturating UP rather than panicking is the right failure direction for a
/// transport bound whose real enforcement is core's.
#[must_use]
pub(crate) fn classify_body_limit_bytes() -> usize {
    usize::try_from(MAX_REQUEST_BODY_BYTES).unwrap_or(usize::MAX)
}

// ---------------------------------------------------------------------------
// The handler
// ---------------------------------------------------------------------------

/// A typed transport refusal.
fn refuse(code: StatusCode, message: String) -> (StatusCode, Json<ErrorResponse>) {
    (code, Json(ErrorResponse { error: message }))
}

/// Map a [`ClassifyError`] onto a status.
///
/// The split is by WHOSE fault it is, not by convenience:
///
/// * the request's shape — 400, the client can fix it by sending something else;
/// * everything else — 500, because a verified model failing to encode, or an
///   envelope constructor refusing a value the model produced, is this server's
///   problem and a client retrying the same body will not help.
///
/// The wildcard is mandatory (`ClassifyError` is `#[non_exhaustive]`) and lands
/// on 500 deliberately: a variant this build has never heard of is an internal
/// condition, and guessing 400 would tell a client to change a body that is fine.
fn classify_error_response(error: &ClassifyError) -> (StatusCode, Json<ErrorResponse>) {
    let code = match error {
        ClassifyError::EmptyInput
        | ClassifyError::BatchTooLarge { .. }
        | ClassifyError::UnsupportedSchemaVersion { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    refuse(code, error.to_string())
}

/// `POST /v1/classify` — classify an ordered batch of texts.
///
/// The route exists whenever the `setfit` feature is compiled in, so a server
/// with no classifier resident answers **503**, never 404 (review M4). See the
/// installation site in `router.rs` for why that distinction is load-bearing.
///
/// Order of business, and the order matters:
///
/// 1. the slot — a request that cannot be served at all is refused before its
///    contents are judged, so a client is never told its body is wrong when the
///    real answer is "this server has no classifier";
/// 2. the request bounds, using core's constants;
/// 3. core's one classify path.
///
/// The response is core's envelope, serialized by core's own `Serialize`. This
/// function neither builds nor rewrites it, which is what makes the CLI's JSON
/// and this body the same bytes for the same model and input.
pub(crate) async fn setfit_classify_handler(
    State(state): State<AppState>,
    Json(request): Json<ClassifyRequestDocument>,
) -> Result<Json<ClassifyResponse>, (StatusCode, Json<ErrorResponse>)> {
    let model = state.setfit_model.as_ref().ok_or_else(|| {
        refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "no SetFit model is loaded: this server was started without a setfit-apr-v1 \
             artifact, so /v1/classify exists but cannot be served"
                .to_string(),
        )
    })?;

    if request.texts.is_empty() {
        return Err(refuse(
            StatusCode::BAD_REQUEST,
            ClassifyError::EmptyInput.to_string(),
        ));
    }
    if request.texts.len() > MAX_BATCH_TEXTS {
        return Err(refuse(
            StatusCode::BAD_REQUEST,
            ClassifyError::BatchTooLarge {
                max: MAX_BATCH_TEXTS,
                got: request.texts.len(),
            }
            .to_string(),
        ));
    }

    model
        .classify(&request)
        .map(Json)
        .map_err(|e| classify_error_response(&e))
}

// ---------------------------------------------------------------------------
// The tiny fixture
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "setfit"))]
mod fixture {
    //! A `setfit-apr-v1` artifact small enough to build in a unit test.
    //!
    //! **This is a DUPLICATE of `aprender-core`'s
    //! `setfit::artifact::fixture` (artifact.rs:3004-3409), and duplication was
    //! the only option.** That module is `#[cfg(test)]`, and its own header
    //! records the decision that it stays that way — plan 03-10's acceptance
    //! criteria reject a `#[doc(hidden)]` test-support door on the shipped
    //! surface. So it is unreachable from this crate by design, not by oversight.
    //!
    //! What is duplicated is the fixture's SHAPE, and every part of it is built
    //! through core's PUBLIC API: `SetFitArtifactView` and `EncoderArchitecture`
    //! have public fields, `write_setfit_apr` and `load_setfit_apr` are the same
    //! two doors production uses, and `artifact_sha256_hex` is the crate's one
    //! hashing path. Nothing here reimplements a rule — if the artifact schema
    //! changes, this builder stops producing a loadable artifact and these tests
    //! go red, which is the correct coupling.
    //!
    //! No golden hash is pinned. Every identity assertion in the suite below
    //! compares two values MEASURED in the same run (the loader's hash against
    //! `artifact_sha256_hex` of the bytes it was handed), so a drift in this
    //! copy cannot make an assertion vacuously true.

    use std::collections::BTreeMap;

    use aprender::setfit::{
        artifact_sha256_hex, load_setfit_apr, write_setfit_apr, EncoderArchitecture,
        SetFitArtifactView, VerifiedSetFitModel, L2_EPS, MAX_SEQUENCE_LENGTH, NORMALIZATION_POLICY,
        PADDING_MODE, PINNED_ACTIVATION, PINNED_REVISION, POOLING_POLICY,
    };
    use serde_json::json;

    /// Reduced dimensions. `positions` is NOT reduced: the tokenizer truncates at
    /// [`MAX_SEQUENCE_LENGTH`], and the writer's truncation-boundary probe
    /// produces a row of that length, which an encoder with fewer position rows
    /// refuses before a probe can be recorded.
    const FIXTURE_HIDDEN: usize = 8;
    const FIXTURE_HEADS: usize = 2;
    const FIXTURE_LAYERS: usize = 2;
    const FIXTURE_INTERMEDIATE: usize = 16;
    const FIXTURE_TYPE_VOCAB: usize = 2;
    pub(super) const FIXTURE_LABELS: [&str; 3] = ["against", "favor", "neutral"];

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
        let mut s = String::new();
        s.push_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":["#);
        for (id, content) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            .iter()
            .enumerate()
        {
            if id > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                r#"{{"id":{id},"special":true,"content":"{content}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false}}"#
            ));
        }
        s.push_str(
            r###"],"normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}},"decoder":{"type":"WordPiece","prefix":"##","cleanup":true},"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"###,
        );
        for (id, token) in TINY_VOCAB.iter().enumerate() {
            if id > 0 {
                s.push(',');
            }
            s.push_str(&format!(r#""{token}":{id}"#));
        }
        s.push_str("}}}");
        s.into_bytes()
    }

    /// A deterministic, platform-independent filler.
    ///
    /// Every produced value is `k / 65536 - 0.5` for an integer `k`, so it is
    /// EXACTLY representable in `f32` on every target: the fixture's own bytes
    /// cannot be a source of cross-platform drift.
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

    fn fixture_tensors(arch: &EncoderArchitecture) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let h = arch.hidden;
        let im = arch.intermediate;
        let mut f = Filler::new(0x0408_0001);
        let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let put = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   f: &mut Filler,
                   name: String,
                   shape: Vec<usize>| {
            let n = shape.iter().product();
            t.insert(name, (shape, f.vec(n)));
        };

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

    fn fixture_view() -> SetFitArtifactView {
        let architecture = fixture_architecture();
        let tensors = fixture_tensors(&architecture);
        let mut head = Filler::new(0x0408_0002);
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
            root_seed: 0x0408_0000_0000_0002,
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

    /// The artifact BYTES and the model loaded FROM THEM.
    ///
    /// Both are returned so the suite can compare the loader's minted hash with
    /// the hash of the bytes it was handed, rather than with a constant. The
    /// model comes out of `load_setfit_apr` — the one production door — so this
    /// helper cannot mint a `VerifiedSetFitModel` that skipped a rung, and the
    /// fact that it succeeds is itself evidence the fixture is a valid artifact.
    pub(super) fn fixture_model() -> (Vec<u8>, VerifiedSetFitModel) {
        let bytes = write_setfit_apr(&fixture_view()).expect("the tiny fixture is writable");
        let model = load_setfit_apr(&bytes).expect("the tiny fixture passes every load rung");
        (bytes, model)
    }
}

// ---------------------------------------------------------------------------
// The in-process HTTP suite (D-14's deterministic leg)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "setfit"))]
mod tests {
    //! Every test here drives the REAL router — the same
    //! `create_router_with_config` `apr serve` mounts — through
    //! `tower::util::ServiceExt::oneshot`. No port is bound and no process is
    //! spawned, so these run in every `cargo test`; the ONE spawned smoke test
    //! is 04-09's (tier3).
    //!
    //! Every assertion is an EXACT status code. The multi-status tolerance the
    //! older suites in `api/tests/` use (`status == OK || status == NOT_FOUND
    //! || ...`) is deliberately not copied: it cannot distinguish the behaviour
    //! under test from four other behaviours, which is the whole of what review
    //! finding M4 was about.

    use super::fixture::{fixture_model, FIXTURE_LABELS};
    use super::*;

    use aprender::setfit::artifact_sha256_hex;
    use aprender::setfit::classify::{ClassifyResponse, MAX_BATCH_TEXTS, MAX_REQUEST_BODY_BYTES};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;

    use crate::api::{create_router_with_config, RouterConfig};

    /// The router with a verified classifier resident, plus the bytes it came
    /// from so a test can re-derive the identity independently.
    fn app_with_model() -> (axum::Router, Vec<u8>, String) {
        let (bytes, model) = fixture_model();
        let hash = model.artifact_sha256().to_string();
        let state = AppState::default().with_setfit_model(Arc::new(model));
        (
            create_router_with_config(state, RouterConfig::default()),
            bytes,
            hash,
        )
    }

    /// The router with NO classifier — the state `apr serve` has before a model
    /// is installed, and the one review finding M4 is about.
    fn app_without_model() -> axum::Router {
        create_router_with_config(AppState::default(), RouterConfig::default())
    }

    async fn post_classify(app: axum::Router, body: String) -> (StatusCode, Vec<u8>) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/classify")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("the request is well formed"),
            )
            .await
            .expect("the router answers");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body is readable");
        (status, bytes.to_vec())
    }

    async fn get_ready(app: axum::Router) -> (StatusCode, serde_json::Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health/ready")
                    .body(Body::empty())
                    .expect("the request is well formed"),
            )
            .await
            .expect("the router answers");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body is readable");
        let value = serde_json::from_slice(&bytes).expect("readiness answers JSON");
        (status, value)
    }

    #[tokio::test]
    async fn setfit_classify_returns_core_envelope_for_a_mixed_batch() {
        let (app, bytes, hash) = app_with_model();
        let body = serde_json::json!({
            "texts": [
                "ok",
                "the quick brown fox jumps over the lazy dog",
                "i firmly support this position ."
            ]
        })
        .to_string();

        let (status, raw) = post_classify(app, body).await;
        assert_eq!(status, StatusCode::OK, "a valid batch must be served");

        // THE TYPE-LEVEL PARITY WITNESS. Deserializing INTO core's envelope means
        // the wire form passed core's own validating `TryFrom` — schema version,
        // probability mass, finiteness and label arity all re-checked on the way
        // in. A `serde_json::Value` comparison, or string matching, would assert
        // only that some JSON came back.
        let response: ClassifyResponse =
            serde_json::from_slice(&raw).expect("the body IS a core ClassifyResponse");

        assert_eq!(response.results().len(), 3, "three texts, three results");
        for result in response.results() {
            assert_eq!(
                result.probabilities().len(),
                FIXTURE_LABELS.len(),
                "every row carries the FULL probability vector, not just the winner"
            );
            assert!(
                FIXTURE_LABELS.contains(&result.label()),
                "the winning label must come from the head's own ordered labels; got {}",
                result.label()
            );
        }

        // Identity: two values measured in this run, never a pinned constant.
        assert_eq!(
            response.artifact_sha256(),
            hash,
            "the response must carry the loaded model's artifact hash"
        );
        assert_eq!(
            response.artifact_sha256(),
            artifact_sha256_hex(&bytes),
            "and that hash must be the digest of the bytes the model was loaded from"
        );

        // D-12 / review B6: the backend names the kernel entry point that RAN.
        // A CPU-capability token here would be a claim about the HOST, which is
        // consistent with a scalar execution of this very batch.
        let backend = response.backend();
        assert!(!backend.is_empty(), "the backend identity must be reported");
        for forbidden in ["avx", "neon", "sse", "simd"] {
            assert!(
                !backend.to_ascii_lowercase().contains(forbidden),
                "the backend identity must not report a SIMD capability token; got {backend}"
            );
        }
    }

    #[tokio::test]
    async fn setfit_classify_treats_an_embedded_newline_as_one_text() {
        // The M2 witness AT THE HTTP BOUNDARY. A line-delimited request format
        // cannot represent this input: the CLI would see two texts where HTTP
        // sees one, and the two surfaces would compare different ordered sets
        // while appearing to agree on everything they did compare. Because both
        // carry the SAME `ClassifyRequestDocument`, the newline is just a byte.
        let (app, _bytes, _hash) = app_with_model();
        let body = serde_json::json!({ "texts": ["line one\nline two"] }).to_string();

        let (status, raw) = post_classify(app, body).await;
        assert_eq!(status, StatusCode::OK);
        let response: ClassifyResponse =
            serde_json::from_slice(&raw).expect("the body IS a core ClassifyResponse");
        assert_eq!(
            response.results().len(),
            1,
            "a text containing a newline is ONE text, not two"
        );
    }

    #[tokio::test]
    async fn setfit_classify_refuses_an_empty_batch() {
        let (app, _bytes, _hash) = app_with_model();
        let body = serde_json::json!({ "texts": [] }).to_string();

        let (status, raw) = post_classify(app, body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "zero texts is the client's"
        );
        let rendered = String::from_utf8_lossy(&raw);
        assert!(
            rendered.contains("zero texts"),
            "the refusal must be core's typed message; got: {rendered}"
        );
    }

    #[tokio::test]
    async fn setfit_classify_refuses_a_batch_one_over_the_contract_bound() {
        let (app, _bytes, _hash) = app_with_model();
        let texts: Vec<String> = (0..=MAX_BATCH_TEXTS).map(|_| "ok".to_string()).collect();
        assert_eq!(texts.len(), 257, "one past the contract's 256");
        let body = serde_json::json!({ "texts": texts }).to_string();

        let (status, raw) = post_classify(app, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let rendered = String::from_utf8_lossy(&raw);
        assert!(
            rendered.contains("257") && rendered.contains(&MAX_BATCH_TEXTS.to_string()),
            "the refusal must name what was asked for AND the bound; got: {rendered}"
        );
    }

    /// The bound ITSELF — the only input that can observe the check at `:152`.
    ///
    /// [`MAX_BATCH_TEXTS`] is enforced here AND again inside
    /// `VerifiedSetFitModel::classify` (module header, "Why the bounds are
    /// checked here AND in core", T-04-23). That redundancy is deliberate, and
    /// it has a consequence nobody had measured: a REFUSAL at
    /// `MAX_BATCH_TEXTS + 1` cannot distinguish the two implementations.
    /// Measured under a hand-applied `>` → `==`, a 257-text batch still answers
    /// `400` with the byte-identical body
    /// `{"error":"the request carried 257 texts; the bound is 256"}` — because
    /// the mutated transport check falls through, core's re-check produces the
    /// same `BatchTooLarge { max: 256, got: 257 }`, and `classify_error_response`
    /// maps it to the same status. So the neighbouring
    /// `setfit_classify_refuses_a_batch_one_over_the_contract_bound`, whose name
    /// asserts boundary coverage, provably CANNOT kill that mutant — and did
    /// not, for three commits.
    ///
    /// Only an ACCEPTANCE at exactly the bound separates them: `>` admits 256,
    /// while both `==` and `>=` refuse it. Measured: 255→200, 256→200, 257→400
    /// under `>`; 255→200, 256→**400**, 257→400 under both mutants.
    ///
    /// Kills, by their `cargo mutants --list` names — re-run either with
    /// `cargo mutants -F 'replace > with (==|>=) in setfit_classify_handler'`:
    /// * `setfit_handlers.rs:152:28: replace > with == in setfit_classify_handler`
    /// * `setfit_handlers.rs:152:28: replace > with >= in setfit_classify_handler`
    #[tokio::test]
    async fn setfit_classify_admits_a_batch_at_exactly_the_contract_bound() {
        let (app, _bytes, _hash) = app_with_model();
        // Read off the exported constant, never a literal 256, so a contract
        // change moves this test WITH the bound instead of leaving it asserting
        // a number the contract no longer states.
        let texts: Vec<String> = (0..MAX_BATCH_TEXTS).map(|_| "ok".to_string()).collect();
        assert_eq!(
            texts.len(),
            MAX_BATCH_TEXTS,
            "the batch must sit exactly ON the bound"
        );
        let body = serde_json::json!({ "texts": texts }).to_string();

        // The OTHER bound must not be what this measures. `MAX_REQUEST_BODY_BYTES`
        // also applies to this request; short texts keep the body far under it,
        // and asserting so means a future change to either constant cannot
        // silently turn this into a body-limit test that passes for the wrong
        // reason.
        assert!(
            body.len() < classify_body_limit_bytes() / 2,
            "the at-the-bound body must sit far under the body limit, else this \
             test measures the wrong bound; body={} limit={}",
            body.len(),
            classify_body_limit_bytes()
        );

        let (status, raw) = post_classify(app, body).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "exactly MAX_BATCH_TEXTS is LEGAL — the bound is `>`, not `>=`"
        );
        let response: ClassifyResponse =
            serde_json::from_slice(&raw).expect("the body IS a core ClassifyResponse");
        assert_eq!(
            response.results().len(),
            MAX_BATCH_TEXTS,
            "every text in a batch ON the bound must be classified"
        );
    }

    #[tokio::test]
    async fn setfit_classify_refuses_a_body_over_the_contract_limit() {
        let (app, _bytes, _hash) = app_with_model();
        // One text comfortably past `max_request_body_bytes`. The batch bound
        // would NOT catch this (it is a single text), which is the point: the
        // body limit bounds the PARSE and the batch bound the WORK, and neither
        // subsumes the other.
        let oversized = "a".repeat(usize::try_from(MAX_REQUEST_BODY_BYTES).unwrap_or(usize::MAX));
        let body = serde_json::json!({ "texts": [oversized] }).to_string();
        assert!(
            u64::try_from(body.len()).unwrap_or(u64::MAX) > MAX_REQUEST_BODY_BYTES,
            "the fixture body must actually exceed the limit under test"
        );

        let (status, _raw) = post_classify(app, body).await;
        assert!(
            status.is_client_error(),
            "an oversized body must be refused 4xx before any compute; got {status}"
        );
        assert_eq!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "and the specific answer is 413, which names the reason"
        );
    }

    #[tokio::test]
    async fn setfit_classify_with_no_model_is_503_and_is_not_404() {
        // REVIEW FINDING M4, pinned. The previous design installed the route only
        // when the slot was populated, which makes this request a 404 — and a 404
        // is indistinguishable, to a client, from a build with no classify
        // surface at all. The route exists; the model does not; 503 says exactly
        // that.
        let body = serde_json::json!({ "texts": ["ok"] }).to_string();
        let (status, raw) = post_classify(app_without_model(), body).await;

        assert_ne!(
            status,
            StatusCode::NOT_FOUND,
            "the route MUST exist whenever the feature is compiled in (review M4)"
        );
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "a missing model is 503, the same answer readiness gives"
        );
        let rendered = String::from_utf8_lossy(&raw);
        assert!(
            rendered.contains("no SetFit model is loaded"),
            "the refusal must name what is missing; got: {rendered}"
        );
    }

    #[tokio::test]
    async fn setfit_readiness_is_503_with_no_model_and_reports_no_classifier() {
        let (status, body) = get_ready(app_without_model()).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "nothing resident means not ready (OPS-05)"
        );
        assert_eq!(body["model_loaded"], serde_json::Value::Bool(false));
        assert!(
            body.get("classifier_artifact_sha256").is_none(),
            "with no classifier the key is ABSENT, so a non-classifier build's \
             health body is byte-identical to what it was; got: {body}"
        );
        assert!(body.get("classifier_verified").is_none());
    }

    #[tokio::test]
    async fn setfit_readiness_is_200_and_reports_the_exact_artifact_hash() {
        let (app, bytes, hash) = app_with_model();
        let (status, body) = get_ready(app).await;

        assert_eq!(
            status,
            StatusCode::OK,
            "a resident verified classifier IS a loaded model"
        );
        assert_eq!(body["model_loaded"], serde_json::Value::Bool(true));
        // The EXACT value, not merely the field's presence: a readiness probe
        // that reports some hash is worth nothing to a deployment that wants to
        // know WHICH artifact answered (T-04-25).
        assert_eq!(
            body["classifier_artifact_sha256"],
            serde_json::Value::String(hash.clone()),
            "readiness must report the loaded model's hash"
        );
        assert_eq!(
            body["classifier_artifact_sha256"],
            serde_json::Value::String(artifact_sha256_hex(&bytes)),
            "which is the digest of the artifact bytes themselves"
        );
        assert_eq!(body["classifier_verified"], serde_json::Value::Bool(true));
    }

    #[tokio::test]
    async fn setfit_classify_and_readiness_agree_on_the_artifact() {
        // The cross-surface claim OPS-05 actually makes: the hash a client reads
        // off a RESPONSE is the hash an operator reads off READINESS. Asserting
        // each against the model separately would leave the two free to drift
        // through different accessors; this compares the two WIRE values.
        let (app, _bytes, _hash) = app_with_model();
        let (ready_status, ready_body) = get_ready(app).await;
        assert_eq!(ready_status, StatusCode::OK);

        let (app, _bytes, _hash) = app_with_model();
        let body = serde_json::json!({ "texts": ["stance detection"] }).to_string();
        let (status, raw) = post_classify(app, body).await;
        assert_eq!(status, StatusCode::OK);
        let response: ClassifyResponse =
            serde_json::from_slice(&raw).expect("the body IS a core ClassifyResponse");

        assert_eq!(
            ready_body["classifier_artifact_sha256"],
            serde_json::Value::String(response.artifact_sha256().to_string()),
            "readiness and the classify response must name the SAME artifact"
        );
    }

    #[tokio::test]
    async fn setfit_classify_is_the_only_surface_added_and_predict_is_untouched() {
        // The wave-6 ownership claim, made executable: installing /v1/classify
        // must not have changed /v1/predict's behaviour on a state with no APR
        // model. If a future edit routed classify through the predict handler or
        // moved the OpenAI group, this is what goes red.
        let response = app_without_model()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"features":[1.0,2.0]}"#))
                    .expect("the request is well formed"),
            )
            .await
            .expect("the router answers");
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "/v1/predict keeps its own slot-empty 503; it is not this plan's route"
        );
    }
}
