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
/// The synthetic fixture (layers 0-3 of the 2.4T index, hand-written; see the fixture
/// README) has no unnamed tensor. Not a check of the published index (PR #7356 round 3,
/// lane 2 finding 3).
fn every_tensor_in_the_synthetic_2_4t_fixture_is_named() {
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
    // Round 3 (lane 2 finding 6): the name says the run fails, so assert it here too.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), fixture("config.json")).expect("config");
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        fixture("unnamed.index.json"),
    )
    .expect("index");
    let err = run_no_weights(dir.path(), false).expect_err("the run fails");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err}");
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

/// PR #7356 lane-2 finding 3: the run itself, not only the report, fails on an unnamed
/// tensor (the old test above never called `run_no_weights`).
#[test]
fn run_no_weights_fails_on_a_directory_with_an_unnamed_tensor() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), fixture("config.json")).expect("config");
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        fixture("unnamed.index.json"),
    )
    .expect("index");
    let err = run_no_weights(dir.path(), true).expect_err("an unnamed tensor fails the run");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err}");
    assert!(
        err.to_string().contains("1 tensor(s) have no family name"),
        "{err}"
    );
}

/// PR #7356 lane-2 finding 2: `--no-weights` with a weight-reading flag is a parse error
/// (contracts/apr-inspect-flags-v1.yaml: never a silent ignore), not a run that drops it.
#[test]
fn no_weights_conflicts_with_every_weight_reading_flag() {
    use clap::Parser;
    for flag in ["--weights", "--vocab", "--filters", "--quality"] {
        let args = vec![
            "apr".to_string(),
            "inspect".to_string(),
            "some/dir".to_string(),
            flag.to_string(),
            "--no-weights".to_string(),
        ];
        let res = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                crate::Cli::try_parse_from(args)
                    .map(|_| ())
                    .map_err(|e| e.kind())
            })
            .expect("spawn")
            .join()
            .expect("join");
        assert_eq!(
            res,
            Err(clap::error::ErrorKind::ArgumentConflict),
            "{flag} --no-weights"
        );
    }
}

#[test]
fn a_real_published_index_has_zero_unnamed() {
    // Verbatim Qwen3.8-Flash-Next index (1658 tensors, hyper-connections, PLE,
    // indexer, MTP, vision): the naming rules are not fitted to the synthetic fixture.
    // From the published apr-cli tarball the sibling crate is absent, so there is nothing to
    // read (PR #7356 lane-2 finding 4); inside the workspace its manifest exists and the
    // fixture must be there (a missing fixture still fails, it is not skipped).
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .to_path_buf();
    if !crates.join("aprender-serve/Cargo.toml").exists() {
        eprintln!("skip: aprender-serve is not beside apr-cli (packaged build)");
        return;
    }
    let dir = crates.join("aprender-serve/tests/fixtures/qwen4_exp");
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
