//! #7330: `apr inspect --no-weights` names every tensor family from config + index only.

use super::*;
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen38_2_4t")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn every_tensor_in_the_2_4t_index_is_named() {
    let r = report(
        &fixture("config.json"),
        &fixture("model.safetensors.index.json"),
    )
    .expect("report");
    assert!(r.unnamed.is_empty(), "unnamed tensors: {:?}", r.unnamed);
    assert_eq!(r.model_type, "qwen3_5_moe_text");
    assert_eq!(r.num_hidden_layers, Some(92));
    assert_eq!(r.num_experts, Some(512));
    let names: Vec<&str> = r.families.iter().map(|f| f.family.as_str()).collect();
    for want in [
        "language.embed_tokens",
        "language.final_norm",
        "head.lm_head",
        "language.linear_attn",
        "language.self_attn",
        "language.mlp.gate",
        "language.mlp.experts",
        "language.mlp.shared_expert",
        "language.mlp.shared_expert_gate",
        "language.input_layernorm",
        "language.post_attention_layernorm",
        "mtp.fc_hidden",
        "mtp.mlp.experts",
        "vision.blocks",
        "vision.merger",
    ] {
        assert!(
            names.contains(&want),
            "family {want} missing from {names:?}"
        );
    }
    let total: usize = r.families.iter().map(|f| f.count).sum();
    assert_eq!(
        total, r.tensor_count,
        "every tensor is counted in exactly one family"
    );
}

#[test]
fn an_unknown_tensor_is_listed_as_unnamed_and_fails_the_run() {
    let r = report(&fixture("config.json"), &fixture("unnamed.index.json")).expect("report");
    assert_eq!(r.unnamed.len(), 1, "{:?}", r.unnamed);
    assert!(r.unnamed[0].ends_with("mystery_block.w"));
    assert!(render_text(&r).contains("UNNAMED"));
}

#[test]
fn json_carries_families_and_an_empty_unnamed_list() {
    let r = report(
        &fixture("config.json"),
        &fixture("model.safetensors.index.json"),
    )
    .expect("report");
    let v = render_json(&r);
    assert_eq!(v["model_type"], "qwen3_5_moe_text");
    assert_eq!(v["unnamed"].as_array().expect("array").len(), 0);
    assert!(v["families"].as_array().expect("array").len() >= 15);
    assert_eq!(v["weights_read"], false);
}

#[test]
fn a_real_published_index_has_zero_unnamed() {
    // Verbatim Qwen3.8-Flash-Next index (1658 tensors, hyper-connections, PLE,
    // indexer, MTP, vision): the naming rules are not fitted to the synthetic fixture.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-serve/tests/fixtures/qwen4_exp");
    let cfg = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let idx = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("index");
    let r = report(&cfg, &idx).expect("report");
    assert_eq!(r.tensor_count, 1658);
    assert!(r.unnamed.is_empty(), "unnamed: {:?}", r.unnamed);
}

#[test]
fn malformed_inputs_are_errors_not_empty_reports() {
    assert!(report("not json", "{}").is_err());
    assert!(report("{}", "{\"weight_map\": 3}").is_err());
    assert!(
        report("{}", "{}").is_err(),
        "no weight_map is not an empty model"
    );
}

#[test]
fn run_no_weights_reads_a_directory_and_refuses_one_without_an_index() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_2_4t");
    assert!(run_no_weights(&dir, true).is_ok());
    let empty = std::env::temp_dir().join(format!("apr-7330-{}", std::process::id()));
    std::fs::create_dir_all(&empty).expect("mkdir");
    assert!(run_no_weights(&empty, true).is_err());
    let _ = std::fs::remove_dir_all(&empty);
}

#[test]
fn the_cli_flag_dispatches_to_the_no_weights_path_on_a_directory() {
    use clap::Parser;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38_2_4t");
    let args = vec![
        "apr".to_string(),
        "inspect".to_string(),
        dir.to_string_lossy().into_owned(),
        "--no-weights".to_string(),
        "--json".to_string(),
    ];
    // clap's parser for the full subcommand set needs a bigger stack than a test thread.
    let res = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let cli = crate::Cli::try_parse_from(args).expect("parse");
            crate::execute_command(&cli)
        })
        .expect("spawn")
        .join()
        .expect("join");
    res.expect("a directory with config + index inspects with no weights");
}
