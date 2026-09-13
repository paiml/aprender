//! ENC-01 tests: the pinned import contract and its rejection matrix.
//!
//! Every rejection below is proven by exercising the rejection path against a
//! checkout that differs from the pin in exactly ONE respect, and asserting the
//! offending field's own name appears in the typed error. A test that merely
//! observed `is_err()` would pass just as happily if the loader rejected the
//! artifact for an unrelated reason.
//!
//! The base artifacts are the REAL pinned bytes, not paraphrases: their sha256
//! digests are asserted against `tests/fixtures/setfit/upstream_manifest.json`
//! by `import_pin_embedded_artifacts_match_the_frozen_upstream_digests`, so a
//! typo in this file fails loudly instead of quietly weakening every mutation
//! that derives from it.

use super::*;

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// The pinned artifacts, verbatim
// ---------------------------------------------------------------------------

/// `config.json` at [`PINNED_REVISION`], byte for byte.
///
/// Note the six fields this crate does NOT model — `_name_or_path`,
/// `gradient_checkpointing`, `initializer_range`, `transformers_version`,
/// `use_cache` and (modelled but not pinned to a dimension) `model_type`. They
/// are exactly why `deny_unknown_fields` is absent from the parser.
const PINNED_CONFIG_JSON: &str = r#"{
  "_name_or_path": "nreimers/MiniLM-L6-H384-uncased",
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 384,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 6,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.8.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
"#;

/// `modules.json` at [`PINNED_REVISION`], byte for byte (no trailing newline).
const PINNED_MODULES_JSON: &str = r#"[
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Normalize",
    "type": "sentence_transformers.models.Normalize"
  }
]"#;

/// `1_Pooling/config.json` at [`PINNED_REVISION`], byte for byte.
const PINNED_POOLING_JSON: &str = r#"{
  "word_embedding_dimension": 384,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}"#;

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

fn upstream_manifest() -> serde_json::Value {
    serde_json::from_slice(&read_fixture("upstream_manifest.json")).expect("parse manifest")
}

/// A synthetic checkout, mutated in exactly one respect.
struct Checkout {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
}

struct CheckoutSpec {
    config: serde_json::Value,
    modules: String,
    pooling: String,
    tokenizer: Vec<u8>,
    max_seq_length: Option<usize>,
}

impl CheckoutSpec {
    /// Every file exactly as pinned.
    fn pinned() -> Self {
        Self {
            config: serde_json::from_str(PINNED_CONFIG_JSON).expect("parse pinned config"),
            modules: PINNED_MODULES_JSON.to_string(),
            pooling: PINNED_POOLING_JSON.to_string(),
            tokenizer: read_fixture("tokenizer.json"),
            max_seq_length: Some(PINNED_MAX_SEQ_LENGTH),
        }
    }

    /// Replace one `config.json` field, leaving every other byte alone.
    fn with_config_field(mut self, field: &str, value: serde_json::Value) -> Self {
        let obj = self
            .config
            .as_object_mut()
            .expect("pinned config is an object");
        assert!(
            obj.contains_key(field),
            "mutation target `{field}` is not in the pinned config — the test would \
             be adding a field rather than changing one"
        );
        obj.insert(field.to_string(), value);
        self
    }

    fn write(self) -> Checkout {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let dir = tmp.path().to_path_buf();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&self.config).expect("serialize config"),
        )
        .expect("write config.json");
        std::fs::write(dir.join("modules.json"), self.modules.as_bytes()).expect("write modules");
        std::fs::create_dir_all(dir.join("1_Pooling")).expect("mkdir 1_Pooling");
        std::fs::write(dir.join("1_Pooling/config.json"), self.pooling.as_bytes())
            .expect("write pooling");
        std::fs::write(dir.join("tokenizer.json"), &self.tokenizer).expect("write tokenizer");
        if let Some(len) = self.max_seq_length {
            std::fs::write(
                dir.join("sentence_bert_config.json"),
                format!("{{\"max_seq_length\": {len}, \"do_lower_case\": false}}"),
            )
            .expect("write sentence_bert_config");
        }
        Checkout { _tmp: tmp, dir }
    }
}

/// Open a checkout that differs from the pin in exactly one respect and assert
/// the error names `field`.
fn assert_rejected_naming(spec: CheckoutSpec, field: &str) -> SetFitError {
    let checkout = spec.write();
    let err = MiniLmImport::open(&checkout.dir)
        .err()
        .unwrap_or_else(|| panic!("mutation of `{field}` was ACCEPTED by the pin"));
    assert!(
        err.to_string().contains(field),
        "error does not name the offending field `{field}`: {err}"
    );
    err
}

// ---------------------------------------------------------------------------
// The pin agrees with the frozen artifact set
// ---------------------------------------------------------------------------

#[test]
fn import_pin_revision_agrees_with_the_frozen_upstream_manifest() {
    let manifest = upstream_manifest();
    assert_eq!(
        manifest["revision"].as_str().expect("manifest revision"),
        PINNED_REVISION,
        "PINNED_REVISION drifted from the revision 01-04 froze"
    );
    assert_eq!(
        manifest["files"]["tokenizer.json"]
            .as_str()
            .expect("manifest tokenizer digest"),
        PINNED_TOKENIZER_SHA256,
        "PINNED_TOKENIZER_SHA256 drifted from the manifest"
    );
    // And the tokenizer bytes actually committed hash to it.
    assert_eq!(
        sha256_hex(&read_fixture("tokenizer.json")),
        PINNED_TOKENIZER_SHA256
    );
}

#[test]
fn import_pin_embedded_artifacts_match_the_frozen_upstream_digests() {
    let manifest = upstream_manifest();
    for (name, text) in [
        ("config.json", PINNED_CONFIG_JSON),
        ("modules.json", PINNED_MODULES_JSON),
        ("1_Pooling/config.json", PINNED_POOLING_JSON),
    ] {
        assert_eq!(
            sha256_hex(text.as_bytes()),
            manifest["files"][name].as_str().expect("manifest entry"),
            "embedded `{name}` is not byte-identical to the pinned artifact"
        );
    }
}

#[test]
fn import_pin_agrees_with_the_bert_minilm_l6_preset() {
    // The pin is expressed against BertConfig::minilm_l6(), not against numbers
    // retyped here; this asserts the two never diverge.
    let preset = BertConfig::minilm_l6();
    let cfg: serde_json::Value =
        serde_json::from_str(PINNED_CONFIG_JSON).expect("parse pinned config");
    assert_eq!(cfg["hidden_size"], preset.hidden_dim);
    assert_eq!(cfg["num_hidden_layers"], preset.num_layers);
    assert_eq!(cfg["num_attention_heads"], preset.num_heads);
    assert_eq!(cfg["intermediate_size"], preset.intermediate_dim);
    assert_eq!(cfg["vocab_size"], preset.vocab_size);
    assert_eq!(
        cfg["max_position_embeddings"],
        preset.max_position_embeddings
    );
    assert_eq!(cfg["type_vocab_size"], preset.type_vocab_size);
    assert_eq!(cfg["pad_token_id"], preset.pad_token_id);
}

// ---------------------------------------------------------------------------
// Acceptance: unmodelled metadata must not be a rejection reason
// ---------------------------------------------------------------------------

#[test]
fn import_pin_accepts_the_real_config_with_its_unmodelled_extra_fields() {
    // The pinned config carries six fields this crate does not model. A
    // `deny_unknown_fields` parser would reject the correct artifact here.
    let checkout = CheckoutSpec::pinned().write();
    let err = MiniLmImport::open(&checkout.dir)
        .err()
        .expect("no weights file exists in this checkout, so open must still fail");
    // The failure must be about the ABSENT WEIGHTS, proving config validation,
    // module validation and the tokenizer hash check all passed on the real
    // pinned bytes. Any config-shaped error here would be the gate failing on
    // the correct artifact.
    let text = err.to_string();
    assert!(
        matches!(err, SetFitError::ImportIo { .. }),
        "expected the missing-weights I/O error, got {text}"
    );
    assert!(
        text.contains(".apr"),
        "expected the error to name the missing weights file: {text}"
    );
    for modelled in [
        "hidden_size",
        "hidden_act",
        "architectures",
        "max_seq_length",
    ] {
        assert!(
            !text.contains(modelled),
            "config validation rejected the PINNED config on `{modelled}`: {text}"
        );
    }
}

#[test]
fn import_pin_accepts_a_config_carrying_a_brand_new_unknown_field() {
    // Forward compatibility: a future transformers release adding a field must
    // not break the gate.
    let mut spec = CheckoutSpec::pinned();
    spec.config
        .as_object_mut()
        .expect("object")
        .insert("some_future_field".to_string(), serde_json::json!(true));
    let checkout = spec.write();
    let err = MiniLmImport::open(&checkout.dir).err().expect("no weights");
    assert!(
        matches!(err, SetFitError::ImportIo { .. }),
        "an unknown field became a rejection reason: {err}"
    );
}

/// End-to-end open of the REAL 86.7 MB pinned checkout, when one has been
/// materialised (D-10, `scripts/setfit_fixtures/fetch_full_weights.py`).
///
/// Every other pin test proves a REJECTION or stops at the missing-weights
/// boundary. This is the only test that drives `open()` all the way through
/// 37 real tensor reads and the finiteness scan, so without it "the pin
/// accepts the pinned model" would be an inference rather than a measurement.
/// It skips loudly rather than failing when the checkout is absent, because the
/// artifact is deliberately not vendored.
#[test]
fn import_pin_opens_the_real_pinned_checkout_when_materialised() {
    let Some(dir) = real_checkout_dir() else {
        eprintln!(
            "SKIP import_pin_opens_the_real_pinned_checkout_when_materialised: \
             no checkout at $APRENDER_MINILM_DIR (run scripts/setfit_fixtures/fetch_full_weights.py)"
        );
        return;
    };
    let import = MiniLmImport::open(&dir).expect("the real pinned checkout must open");
    let preset = BertConfig::minilm_l6();
    assert_eq!(import.dims().hidden, preset.hidden_dim);
    assert_eq!(import.dims().layers, preset.num_layers);
    assert_eq!(import.dims().heads, preset.num_heads);
    assert_eq!(import.dims().vocab, preset.vocab_size);
    assert_eq!(import.revision(), PINNED_REVISION);
    assert_eq!(import.tokenizer_sha256(), PINNED_TOKENIZER_SHA256);
    assert!(
        import.vocab_remap().is_none(),
        "the full-pin path must NOT carry a slice remap"
    );
}

/// `$APRENDER_MINILM_DIR`, or the default the D-10 script writes to.
fn real_checkout_dir() -> Option<PathBuf> {
    let dir = std::env::var("APRENDER_MINILM_DIR").map_or_else(
        |_| {
            dirs_home()
                .map(|h| h.join(".cache/aprender/minilm-l6-v2-1110a243"))
                .unwrap_or_default()
        },
        PathBuf::from,
    );
    (dir.join("config.json").exists() && dir.join("full_model.apr").exists()).then_some(dir)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// The mutation matrix — one field per test
// ---------------------------------------------------------------------------

#[test]
fn import_pin_rejects_mutated_hidden_size() {
    let err = assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("hidden_size", serde_json::json!(512)),
        "hidden_size",
    );
    assert!(matches!(err, SetFitError::ImportConfigMismatch { .. }));
}

#[test]
fn import_pin_rejects_mutated_num_hidden_layers() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("num_hidden_layers", serde_json::json!(12)),
        "num_hidden_layers",
    );
}

#[test]
fn import_pin_rejects_mutated_num_attention_heads() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("num_attention_heads", serde_json::json!(6)),
        "num_attention_heads",
    );
}

#[test]
fn import_pin_rejects_mutated_intermediate_size() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("intermediate_size", serde_json::json!(3072)),
        "intermediate_size",
    );
}

#[test]
fn import_pin_rejects_mutated_vocab_size() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("vocab_size", serde_json::json!(30523)),
        "vocab_size",
    );
}

#[test]
fn import_pin_rejects_mutated_max_position_embeddings() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("max_position_embeddings", serde_json::json!(256)),
        "max_position_embeddings",
    );
}

#[test]
fn import_pin_rejects_mutated_layer_norm_eps() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("layer_norm_eps", serde_json::json!(1e-5)),
        "layer_norm_eps",
    );
}

#[test]
fn import_pin_rejects_mutated_type_vocab_size() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("type_vocab_size", serde_json::json!(1)),
        "type_vocab_size",
    );
}

#[test]
fn import_pin_rejects_mutated_pad_token_id() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("pad_token_id", serde_json::json!(1)),
        "pad_token_id",
    );
}

#[test]
fn import_pin_rejects_hidden_act_gelu_new() {
    // "gelu_new" selects the TANH approximation. Measured gap to the pinned erf
    // form is 4.734993e-04 — two orders above the frozen activation tolerance —
    // so silently accepting it would break every ENC-05 comparison downstream.
    let err = assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("hidden_act", serde_json::json!("gelu_new")),
        "hidden_act",
    );
    assert!(
        matches!(err, SetFitError::UnsupportedActivation { .. }),
        "expected UnsupportedActivation, got {err}"
    );
    assert!(err.to_string().contains("gelu_new"), "{err}");
}

#[test]
fn import_pin_rejects_hidden_act_gelu_pytorch_tanh() {
    let err = assert_rejected_naming(
        CheckoutSpec::pinned()
            .with_config_field("hidden_act", serde_json::json!("gelu_pytorch_tanh")),
        "hidden_act",
    );
    assert!(matches!(err, SetFitError::UnsupportedActivation { .. }));
}

#[test]
fn import_pin_rejects_hidden_act_relu() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("hidden_act", serde_json::json!("relu")),
        "hidden_act",
    );
}

#[test]
fn import_pin_rejects_mutated_hidden_dropout_prob() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("hidden_dropout_prob", serde_json::json!(0.2)),
        "hidden_dropout_prob",
    );
}

#[test]
fn import_pin_rejects_mutated_attention_probs_dropout_prob() {
    assert_rejected_naming(
        CheckoutSpec::pinned()
            .with_config_field("attention_probs_dropout_prob", serde_json::json!(0.0)),
        "attention_probs_dropout_prob",
    );
}

#[test]
fn import_pin_rejects_relative_position_embeddings() {
    assert_rejected_naming(
        CheckoutSpec::pinned()
            .with_config_field("position_embedding_type", serde_json::json!("relative_key")),
        "position_embedding_type",
    );
}

#[test]
fn import_pin_rejects_a_non_bert_model_architecture() {
    let err = assert_rejected_naming(
        CheckoutSpec::pinned()
            .with_config_field("architectures", serde_json::json!(["BertForMaskedLM"])),
        "architectures",
    );
    assert!(
        matches!(err, SetFitError::UnsupportedArchitecture { .. }),
        "expected UnsupportedArchitecture, got {err}"
    );
}

#[test]
fn import_pin_rejects_a_non_bert_model_type() {
    assert_rejected_naming(
        CheckoutSpec::pinned().with_config_field("model_type", serde_json::json!("roberta")),
        "model_type",
    );
}

#[test]
fn import_pin_rejects_cls_pooling() {
    let mut spec = CheckoutSpec::pinned();
    spec.pooling = PINNED_POOLING_JSON
        .replace(
            "\"pooling_mode_cls_token\": false",
            "\"pooling_mode_cls_token\": true",
        )
        .replace(
            "\"pooling_mode_mean_tokens\": true",
            "\"pooling_mode_mean_tokens\": false",
        );
    let err = assert_rejected_naming(spec, "pooling_mode_mean_tokens");
    assert!(
        matches!(err, SetFitError::UnsupportedPooling { .. }),
        "expected UnsupportedPooling, got {err}"
    );
}

#[test]
fn import_pin_rejects_max_token_pooling() {
    let mut spec = CheckoutSpec::pinned();
    spec.pooling = PINNED_POOLING_JSON.replace(
        "\"pooling_mode_max_tokens\": false",
        "\"pooling_mode_max_tokens\": true",
    );
    assert_rejected_naming(spec, "pooling_mode_max_tokens");
}

#[test]
fn import_pin_rejects_a_module_stack_without_normalize() {
    // The normalize flag lives in modules.json as the trailing Normalize module.
    // Dropping it changes the embedding the model produces, so it is a
    // behavior-affecting mutation like any config field.
    let mut spec = CheckoutSpec::pinned();
    let mods: Vec<serde_json::Value> =
        serde_json::from_str(PINNED_MODULES_JSON).expect("parse modules");
    let trimmed: Vec<serde_json::Value> = mods.into_iter().take(2).collect();
    spec.modules = serde_json::to_string(&trimmed).expect("serialize modules");
    assert_rejected_naming(spec, "Normalize");
}

#[test]
fn import_pin_rejects_a_mutated_sentence_transformer_max_seq_length() {
    assert_rejected_naming(
        CheckoutSpec {
            max_seq_length: Some(128),
            ..CheckoutSpec::pinned()
        },
        "max_seq_length",
    );
}

#[test]
fn import_pin_rejects_tokenizer_bytes_with_one_flipped_byte() {
    let mut spec = CheckoutSpec::pinned();
    // Flip a byte deep inside the vocabulary, not in the leading brace, so the
    // file still parses as JSON and only the DIGEST distinguishes it.
    let mid = spec.tokenizer.len() / 2;
    spec.tokenizer[mid] ^= 0x01;
    let checkout = spec.write();
    let err = MiniLmImport::open(&checkout.dir)
        .err()
        .expect("must reject");
    assert!(
        matches!(err, SetFitError::TokenizerHashMismatch { .. }),
        "expected TokenizerHashMismatch, got {err}"
    );
    assert!(
        err.to_string().contains(PINNED_TOKENIZER_SHA256),
        "error should name the expected digest: {err}"
    );
}

#[test]
fn import_pin_rejects_slice_dimensions() {
    // The bypass is NOT reachable from the public pin path: handing open() the
    // slice's own dimensions must fail, or the slice constructor would be
    // pointless and PF-011 would be back.
    let spec = CheckoutSpec::pinned()
        .with_config_field("hidden_size", serde_json::json!(64))
        .with_config_field("num_hidden_layers", serde_json::json!(2))
        .with_config_field("num_attention_heads", serde_json::json!(2))
        .with_config_field("intermediate_size", serde_json::json!(256))
        .with_config_field("vocab_size", serde_json::json!(97))
        .with_config_field("max_position_embeddings", serde_json::json!(64));
    assert_rejected_naming(spec, "hidden_size");
}

#[test]
fn import_pin_rejects_a_missing_config_file() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let err = MiniLmImport::open(tmp.path()).err().expect("must reject");
    assert!(matches!(err, SetFitError::ImportIo { .. }), "{err}");
    assert!(err.to_string().contains("config.json"), "{err}");
}

#[test]
fn import_pin_rejects_an_unparseable_config_file() {
    let checkout = CheckoutSpec::pinned().write();
    std::fs::write(checkout.dir.join("config.json"), b"{ not json").expect("write");
    let err = MiniLmImport::open(&checkout.dir)
        .err()
        .expect("must reject");
    assert!(matches!(err, SetFitError::ImportIo { .. }), "{err}");
}

// ---------------------------------------------------------------------------
// D-08 seal, asserted against the source
// ---------------------------------------------------------------------------

#[test]
fn import_pin_constructors_are_sealed() {
    let src = include_str!("import.rs");
    assert!(
        src.contains("pub(crate) fn open("),
        "D-08: MiniLmImport::open must be pub(crate)"
    );
    assert!(
        !src.contains("pub fn open("),
        "D-08 seal broken: a bare `pub fn open(` exists in import.rs"
    );
    assert!(
        src.contains("pub(crate) fn open_slice_fixture("),
        "D-08: open_slice_fixture must be pub(crate)"
    );
    assert!(
        !src.contains("pub fn open_slice_fixture("),
        "D-08 seal broken: a bare `pub fn open_slice_fixture(` exists"
    );
    // Comment-safe: the module docs deliberately DISCUSS `deny_unknown_fields`,
    // so a bare `contains` would flag the explanation of why it is absent. Only
    // non-comment lines count.
    let offending: Vec<&str> = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .filter(|l| l.contains("deny_unknown_fields"))
        .collect();
    assert!(
        offending.is_empty(),
        "deny_unknown_fields would reject the pinned config itself: {offending:?}"
    );
}

// ---------------------------------------------------------------------------
// Slice fixtures (conformance-fixtures only)
// ---------------------------------------------------------------------------

#[cfg(feature = "conformance-fixtures")]
mod slice {
    use super::*;

    use crate::format::v2::{AprV2Metadata, AprV2Writer};

    fn slice_config() -> SliceConfig {
        SliceConfig::from_json_bytes(&read_fixture("slice_config.json")).expect("slice_config.json")
    }

    fn slice_remap(vocab: usize) -> VocabRemap {
        VocabRemap::from_json_bytes(&read_fixture("vocab_remap.json"), vocab)
            .expect("vocab_remap.json")
    }

    fn slice_apr_path() -> PathBuf {
        fixtures_dir().join("slice_model.apr")
    }

    /// Write bytes to a temp `.apr` and hand back the path plus its guard.
    fn temp_apr(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("slice.apr");
        std::fs::write(&path, bytes).expect("write apr");
        (tmp, path)
    }

    /// Build a slice-shaped APR, optionally corrupting one tensor.
    fn build_slice_apr(cfg: &SliceConfig, corrupt: Corruption) -> Vec<u8> {
        let h = cfg.hidden;
        let im = cfg.intermediate;
        let mut w = AprV2Writer::new(AprV2Metadata::default());
        let mut put = |name: &str, shape: Vec<usize>| {
            if let Corruption::Drop(target) = corrupt {
                if name == target {
                    return;
                }
            }
            let mut shape = shape;
            if let Corruption::Reshape(target) = corrupt {
                if name == target {
                    shape[0] += 1;
                }
            }
            let numel: usize = shape.iter().product();
            let mut data = vec![0.05f32; numel];
            if let Corruption::NonFinite(target) = corrupt {
                if name == target {
                    data[numel / 2] = f32::NAN;
                }
            }
            w.add_f32_tensor(name, shape, &data);
        };

        put("embeddings.word_embeddings.weight", vec![cfg.vocab, h]);
        put(
            "embeddings.position_embeddings.weight",
            vec![cfg.positions, h],
        );
        put(
            "embeddings.token_type_embeddings.weight",
            vec![cfg.type_vocab_size, h],
        );
        put("embeddings.LayerNorm.weight", vec![h]);
        put("embeddings.LayerNorm.bias", vec![h]);
        for idx in 0..cfg.num_layers {
            let p = format!("encoder.layer.{idx}");
            for proj in ["query", "key", "value"] {
                put(&format!("{p}.attention.self.{proj}.weight"), vec![h, h]);
                put(&format!("{p}.attention.self.{proj}.bias"), vec![h]);
            }
            put(&format!("{p}.attention.output.dense.weight"), vec![h, h]);
            put(&format!("{p}.attention.output.dense.bias"), vec![h]);
            put(&format!("{p}.attention.output.LayerNorm.weight"), vec![h]);
            put(&format!("{p}.attention.output.LayerNorm.bias"), vec![h]);
            put(&format!("{p}.intermediate.dense.weight"), vec![im, h]);
            put(&format!("{p}.intermediate.dense.bias"), vec![im]);
            put(&format!("{p}.output.dense.weight"), vec![h, im]);
            put(&format!("{p}.output.dense.bias"), vec![h]);
            put(&format!("{p}.output.LayerNorm.weight"), vec![h]);
            put(&format!("{p}.output.LayerNorm.bias"), vec![h]);
        }
        w.write().expect("write apr bytes")
    }

    #[derive(Clone, Copy)]
    enum Corruption {
        None,
        Drop(&'static str),
        Reshape(&'static str),
        NonFinite(&'static str),
    }

    #[test]
    fn import_slice_loads_the_frozen_apr_and_exposes_the_remap() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        let import = MiniLmImport::open_slice_fixture(&slice_apr_path(), &cfg, &remap)
            .expect("frozen slice must load");
        assert_eq!(import.dims().hidden, 64);
        assert_eq!(import.dims().layers, 2);
        assert_eq!(import.dims().heads, 2);
        assert_eq!(import.dims().intermediate, 256);
        assert_eq!(import.dims().vocab, 97);
        assert_eq!(import.dims().max_positions, 64);
        assert!(
            import.vocab_remap().is_some(),
            "a slice import must carry its remap"
        );
        assert_eq!(import.revision(), PINNED_REVISION);
        assert_eq!(import.tokenizer_sha256(), PINNED_TOKENIZER_SHA256);
    }

    #[test]
    fn import_slice_config_agrees_with_the_frozen_pin() {
        let cfg = slice_config();
        assert_eq!(cfg.source_revision, PINNED_REVISION);
        assert_eq!(cfg.tokenizer_sha256, PINNED_TOKENIZER_SHA256);
        assert_eq!(cfg.hidden, cfg.heads * cfg.head_dim);
    }

    #[test]
    fn import_slice_rejects_a_foreign_source_revision() {
        let mut cfg = slice_config();
        cfg.source_revision = "0000000000000000000000000000000000000000".to_string();
        let remap = slice_remap(cfg.vocab);
        let err = MiniLmImport::open_slice_fixture(&slice_apr_path(), &cfg, &remap)
            .err()
            .expect("a slice from another revision must be rejected");
        assert!(err.to_string().contains("source_revision"), "{err}");
    }

    #[test]
    fn import_slice_rejects_a_non_pinned_activation() {
        let mut cfg = slice_config();
        cfg.hidden_act = "gelu_new".to_string();
        let remap = slice_remap(cfg.vocab);
        let err = MiniLmImport::open_slice_fixture(&slice_apr_path(), &cfg, &remap)
            .err()
            .expect("the slice bypass must NOT bypass the activation check");
        assert!(
            matches!(err, SetFitError::UnsupportedActivation { .. }),
            "{err}"
        );
    }

    #[test]
    fn import_slice_rejects_inconsistent_head_geometry() {
        let mut cfg = slice_config();
        cfg.heads = 3; // 3 * 32 != 64
        let remap = slice_remap(cfg.vocab);
        let err = MiniLmImport::open_slice_fixture(&slice_apr_path(), &cfg, &remap)
            .err()
            .expect("hidden must equal heads * head_dim");
        assert!(err.to_string().contains("heads"), "{err}");
    }

    #[test]
    fn import_slice_rejects_a_missing_tensor() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        let bytes = build_slice_apr(
            &cfg,
            Corruption::Drop("encoder.layer.1.output.dense.weight"),
        );
        let (_tmp, path) = temp_apr(&bytes);
        let err = MiniLmImport::open_slice_fixture(&path, &cfg, &remap)
            .err()
            .expect("a missing tensor must be rejected");
        assert!(
            matches!(err, SetFitError::ImportTensor(_)),
            "expected ImportTensor, got {err}"
        );
        assert!(
            err.to_string()
                .contains("encoder.layer.1.output.dense.weight"),
            "{err}"
        );
    }

    #[test]
    fn import_slice_rejects_a_shape_inconsistent_tensor() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        let bytes = build_slice_apr(
            &cfg,
            Corruption::Reshape("embeddings.word_embeddings.weight"),
        );
        let (_tmp, path) = temp_apr(&bytes);
        let err = MiniLmImport::open_slice_fixture(&path, &cfg, &remap)
            .err()
            .expect("a wrong-shaped tensor must be rejected");
        assert!(
            matches!(err, SetFitError::ImportTensor(_)),
            "expected ImportTensor, got {err}"
        );
        assert!(err.to_string().contains("word_embeddings"), "{err}");
    }

    #[test]
    fn import_slice_rejects_a_non_finite_tensor() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        let bytes = build_slice_apr(
            &cfg,
            Corruption::NonFinite("encoder.layer.0.output.dense.bias"),
        );
        let (_tmp, path) = temp_apr(&bytes);
        let err = MiniLmImport::open_slice_fixture(&path, &cfg, &remap)
            .err()
            .expect("a NaN weight must be rejected before use");
        assert!(
            matches!(err, SetFitError::NonFiniteTensor { .. }),
            "expected NonFiniteTensor, got {err}"
        );
        assert!(
            err.to_string()
                .contains("encoder.layer.0.output.dense.bias"),
            "{err}"
        );
    }

    #[test]
    fn import_slice_accepts_a_synthetic_but_well_formed_apr() {
        // Proves the corruption tests above fail for their stated reason and not
        // because the synthetic builder is simply unreadable.
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        let bytes = build_slice_apr(&cfg, Corruption::None);
        let (_tmp, path) = temp_apr(&bytes);
        MiniLmImport::open_slice_fixture(&path, &cfg, &remap).expect("clean synthetic apr loads");
    }

    #[test]
    fn import_slice_rejects_a_missing_apr_file() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        let err =
            MiniLmImport::open_slice_fixture(Path::new("/nonexistent/slice.apr"), &cfg, &remap)
                .err()
                .expect("a missing file must be rejected");
        assert!(matches!(err, SetFitError::ImportIo { .. }), "{err}");
    }

    // -- remap validation -------------------------------------------------

    #[test]
    fn import_slice_remap_round_trips_every_row() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        assert_eq!(remap.slice_vocab(), cfg.vocab);
        for row in 0..cfg.vocab {
            let row = u32::try_from(row).expect("row fits u32");
            let canonical = remap.to_canonical(row).expect("row is mapped");
            assert_eq!(remap.to_slice_row(canonical).expect("inverse"), row);
        }
    }

    #[test]
    fn import_slice_remap_rejects_a_canonical_id_outside_the_closure() {
        let cfg = slice_config();
        let remap = slice_remap(cfg.vocab);
        // 30521 is a real MiniLM id that the 97-row slice does not retain.
        let err = remap
            .to_slice_row(30_521)
            .err()
            .expect("an out-of-closure id must not silently map");
        assert!(
            matches!(
                err,
                SetFitError::VocabOutOfSlice {
                    canonical_id: 30_521
                }
            ),
            "expected VocabOutOfSlice, got {err}"
        );
    }

    #[test]
    fn import_slice_remap_rejects_an_out_of_range_slice_row() {
        let json = br#"{"orig_to_slice": {"0": 0, "7": 99}, "slice_to_orig": [0, 7]}"#;
        let err = VocabRemap::from_json_bytes(json, 2)
            .err()
            .expect("slice row 99 is outside a 2-row table");
        assert!(matches!(err, SetFitError::RemapInvalid { .. }), "{err}");
        assert!(err.to_string().contains("99"), "{err}");
    }

    #[test]
    fn import_slice_remap_rejects_inverse_inconsistency() {
        // Both directions are individually in range, but they disagree.
        let json = br#"{"orig_to_slice": {"0": 0, "7": 1}, "slice_to_orig": [0, 9]}"#;
        let err = VocabRemap::from_json_bytes(json, 2)
            .err()
            .expect("a non-invertible remap must be rejected");
        assert!(matches!(err, SetFitError::RemapInvalid { .. }), "{err}");
        // Naming the direction that disagrees is what makes this actionable —
        // and is what stops the test passing against a blanket rejection.
        assert!(err.to_string().contains("slice_to_orig"), "{err}");
    }

    #[test]
    fn import_slice_remap_rejects_wrong_arity() {
        let json = br#"{"orig_to_slice": {"0": 0, "7": 1}, "slice_to_orig": [0, 7]}"#;
        let err = VocabRemap::from_json_bytes(json, 3)
            .err()
            .expect("slice_to_orig length must equal the slice vocab");
        assert!(matches!(err, SetFitError::RemapInvalid { .. }), "{err}");
        let text = err.to_string();
        assert!(text.contains('3') && text.contains('2'), "{text}");
    }

    #[test]
    fn import_slice_rejects_a_remap_that_disagrees_with_the_slice_vocab() {
        let cfg = slice_config();
        let err = VocabRemap::from_json_bytes(&read_fixture("vocab_remap.json"), cfg.vocab + 1)
            .err()
            .expect("a remap sized for another slice must be rejected");
        assert!(matches!(err, SetFitError::RemapInvalid { .. }), "{err}");
        assert!(
            err.to_string().contains(&(cfg.vocab + 1).to_string()),
            "the error must name the size it was asked for: {err}"
        );
    }
}
