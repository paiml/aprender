//! Falsifiers for qwen35-format-roundtrip-v1 QFR-003: an `.apr` imported from
//! an HF directory exports the SOURCE `config.json`, not one inferred from
//! tensor shapes. The inferred config cannot tell Qwen3.5 from Qwen2, so
//! re-importing an exported Qwen3.5 changed its architecture.

#[allow(unused_imports)]
use super::super::*;
use crate::format::test_factory::build_pygmy_safetensors;
use std::fs;

/// A source config whose identity the tensors cannot reveal: the class,
/// family and a Qwen3.5-only key.
fn source_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "intermediate_size": 8,
        "linear_num_value_heads": 16,
        "layer_types": ["linear_attention"],
        "tie_word_embeddings": false
    })
}

fn import_hf_dir(dir: &Path, config: Option<&serde_json::Value>) -> PathBuf {
    let st = dir.join("model.safetensors");
    fs::write(&st, build_pygmy_safetensors()).expect("write safetensors");
    if let Some(cfg) = config {
        fs::write(dir.join("config.json"), cfg.to_string()).expect("write config.json");
    }
    let apr = dir.join("model.apr");
    let options = ImportOptions {
        allow_no_config: config.is_none(),
        ..ImportOptions::default()
    };
    apr_import(st.to_str().expect("utf8"), &apr, options).expect("import");
    apr
}

fn export_config(apr: &Path, out_dir: &Path, format: ExportFormat) -> serde_json::Value {
    fs::create_dir_all(out_dir).expect("out dir");
    let (output, config_path) = match format {
        ExportFormat::Mlx => (out_dir.join("mlx"), out_dir.join("mlx").join("config.json")),
        _ => (out_dir.join("model.safetensors"), out_dir.join("config.json")),
    };
    let options = ExportOptions {
        format,
        include_config: true,
        include_tokenizer: false,
        ..Default::default()
    };
    apr_export(apr, &output, options).expect("export");
    let text = fs::read_to_string(config_path).expect("exported config.json");
    serde_json::from_str(&text).expect("exported config.json is JSON")
}

/// FALSIFY-QFR-003: import → export(safetensors) → config.json is the source
/// config, and re-importing the export keeps the source architecture.
/// Reverting either half (import not storing `hf_config`, or export
/// inferring) turns this RED: the inferred config has no Qwen3.5 class.
#[test]
fn falsify_qfr_003_export_writes_source_config_back() {
    let src = tempfile::tempdir().expect("tempdir");
    let apr = import_hf_dir(src.path(), Some(&source_config()));

    let out = tempfile::tempdir().expect("tempdir");
    let exported = export_config(&apr, out.path(), ExportFormat::SafeTensors);
    assert_eq!(exported, source_config(), "exported config.json must be the source");

    // Round trip: the bare export re-imports (no config.json copied, and
    // allow_no_config off) and keeps the source architecture.
    let back = out.path().join("back.apr");
    apr_import(
        out.path().join("model.safetensors").to_str().expect("utf8"),
        &back,
        ImportOptions::default(),
    )
    .expect("the bare export re-imports");
    let bytes = fs::read(&back).expect("read re-imported apr");
    let reader = crate::format::v2::AprV2Reader::from_bytes(&bytes).expect("parse apr");
    let meta = reader.metadata();
    assert_eq!(meta.hf_architecture.as_deref(), Some("Qwen3_5ForCausalLM"));
    assert_eq!(meta.architecture.as_deref(), Some("qwen3_5"));
}

/// FALSIFY-QFR-003 (MLX): the MLX directory gets the source config too.
#[test]
fn falsify_qfr_003_mlx_export_writes_source_config_back() {
    let src = tempfile::tempdir().expect("tempdir");
    let apr = import_hf_dir(src.path(), Some(&source_config()));
    let out = tempfile::tempdir().expect("tempdir");
    assert_eq!(export_config(&apr, out.path(), ExportFormat::Mlx), source_config());
}

/// With no source config stored, export still writes an inferred one; the
/// fallback is not removed.
#[test]
fn qfr_003_no_stored_config_falls_back_to_inferred() {
    let src = tempfile::tempdir().expect("tempdir");
    let apr = import_hf_dir(src.path(), None);
    let out = tempfile::tempdir().expect("tempdir");
    let exported = export_config(&apr, out.path(), ExportFormat::SafeTensors);
    assert!(exported.get("hidden_size").is_some(), "{exported}");
    assert!(exported.get("linear_num_value_heads").is_none(), "{exported}");
}
