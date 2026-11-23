//! #7169 S1 (d): inference on `qwen4_exp` refuses by name, never produces output.
use realizar::safetensors_infer::SafetensorsToAprConverter;

#[test]
fn convert_refuses_qwen4_exp_naming_every_unsupported_op() {
    let dir = tempfile::tempdir().expect("tmp");
    // Minimal valid safetensors file: 8-byte LE header length + "{}".
    let mut bytes = 2u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    std::fs::write(dir.path().join("model.safetensors"), bytes).expect("write");
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen4_exp/config.json"
        ),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let err = SafetensorsToAprConverter::convert(&dir.path().join("model.safetensors"))
        .expect_err("must refuse");
    let msg = err.to_string();
    for n in [
        "qwen4_exp",
        "sparse-attention indexer",
        "ngram embedding",
        "ple",
        "hc residual",
        "sigmoid output gate",
        "mtp",
        "refusal, not a fallback",
    ] {
        assert!(msg.contains(n), "{n} missing from: {msg}");
    }
}

/// The config/index fixture tests (moved from src/qwen4_exp.rs: a src include_str! of a /tests/
/// fixture breaks `cargo test` from the published tarball, because the package excludes /tests/).
mod config_fixture {
    use realizar::qwen4_exp::*;
    use std::collections::BTreeMap;

    const CONFIG: &str = include_str!("fixtures/qwen4_exp/config.json");
    const INDEX: &str = include_str!("fixtures/qwen4_exp/model.safetensors.index.json");

    #[test]
    fn config_fixture_parses_with_every_new_key() {
        let c = Qwen4ExpConfig::parse_config_json(CONFIG).expect("parse");
        assert_eq!(c.hidden_size, Some(2560));
        assert_eq!(c.num_hidden_layers, Some(48));
        assert_eq!(c.num_experts, Some(512));
        assert_eq!(c.num_experts_per_tok, Some(10));
        assert_eq!(c.full_attention_interval, Some(4));
        assert_eq!(c.partial_rotary_factor, Some(0.25));
        assert_eq!(c.layer_types.as_ref().map(Vec::len), Some(48));
        assert_eq!(c.indexer_budget, Some(2048));
        assert_eq!(c.indexer_compress_ratio, Some(4));
        assert_eq!(c.indexer_head_dim, Some(128));
        assert_eq!(c.indexer_kv_heads, Some(1));
        assert_eq!(c.indexer_n_heads, Some(4));
        assert_eq!(c.ngram_size, Some(3));
        assert_eq!(c.ngram_vocab_size_base, Some(20_000_000));
        assert_eq!(c.heads_per_ngram, Some(8));
        assert_eq!(c.split_ngram_parts, Some(128));
        assert_eq!(c.make_ngram_vocab_size_divisible_by, Some(128));
        assert_eq!(c.ple_layer_ids, Some(vec![2]));
        assert_eq!(c.ple_embed_dim, Some(2560));
        assert_eq!(c.ple_conv_kernel_size, Some(4));
        assert_eq!(c.hc_count, Some(4));
        assert_eq!(c.hc_lowrank, Some(320));
        assert_eq!(c.output_gate_type.as_deref(), Some("sigmoid"));
        assert_eq!(c.mtp_num_hidden_layers, Some(1));
        assert_eq!(c.mtp.as_ref().and_then(|m| m.hybrid), Some(true));
        let rp = c.rope_parameters.as_ref().expect("rope");
        assert_eq!(rp.mrope_section, Some(vec![11, 11, 10]));
        assert_eq!(rp.mrope_interleaved, Some(true));
        assert_eq!(c.vision_config.as_ref().and_then(|v| v.depth), Some(27));
    }

    #[test]
    fn recognised_as_qwen4_exp_not_a_fallback_family() {
        let c = Qwen4ExpConfig::parse_config_json(CONFIG).expect("parse");
        assert_eq!(c.architectures, vec![ARCHITECTURE.to_string()]);
        assert!(is_qwen4_exp(Some("qwen4_exp"), &[]));
        assert!(is_qwen4_exp(Some("qwen4_exp_text"), &[]));
        assert!(is_qwen4_exp(None, &c.architectures));
        for other in ["qwen2", "qwen3", "qwen3_5", "llama"] {
            assert!(!is_qwen4_exp(Some(other), &[]), "{other}");
        }
        let qwen35 = r#"{"model_type":"qwen3_5","hidden_size":8}"#;
        assert!(Qwen4ExpConfig::parse_config_json(qwen35).is_err());
    }

    #[test]
    fn every_index_tensor_is_classified() {
        let v: serde_json::Value = serde_json::from_str(INDEX).expect("index");
        let map = v["weight_map"].as_object().expect("weight_map");
        let mut fam: BTreeMap<TensorFamily, usize> = BTreeMap::new();
        let mut unknown = Vec::new();
        for name in map.keys() {
            match classify_tensor(name) {
                Some(f) => *fam.entry(f).or_default() += 1,
                None => unknown.push(name.clone()),
            }
        }
        assert!(unknown.is_empty(), "unclassified: {unknown:?}");
        assert_eq!(fam.len(), 10, "all ten families present: {fam:?}");
    }

    #[test]
    fn config_lists_all_six_unsupported_ops_by_name() {
        let c = Qwen4ExpConfig::parse_config_json(CONFIG).expect("parse");
        let ops = c.unsupported_ops();
        assert_eq!(ops.len(), 6);
        let msg = refusal_reason(&ops);
        for n in [
            "sparse-attention indexer",
            "ngram embedding",
            "ple",
            "hc residual",
            "sigmoid output gate",
            "mtp",
        ] {
            assert!(msg.contains(n), "{n} missing from: {msg}");
        }
        assert!(msg.contains("refusal, not a fallback"));
    }

    #[test]
    fn check_config_refuses_qwen4_exp_by_name_and_passes_others() {
        use realizar::SafetensorsConfig;
        let cfg: SafetensorsConfig = serde_json::from_str(CONFIG).expect("cfg");
        let err = check_config(&cfg).expect_err("must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("qwen4_exp") && msg.contains("refusal, not a fallback"),
            "{msg}"
        );
        assert!(msg.contains("no tensor data was read"), "{msg}");
        assert!(!msg.contains("nothing was loaded"), "{msg}");
        let other: SafetensorsConfig =
            serde_json::from_str(r#"{"model_type":"qwen3","hidden_size":8}"#).expect("cfg");
        assert!(check_config(&other).is_ok());
    }
}
