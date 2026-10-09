//! EXT-28 (aprender#4410): the C2 speed arms.

use super::*;
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;

const BENCH_JSON: &str = r#"[
  {"n_prompt": 512, "n_gen": 0, "avg_ts": 900.0, "samples_ts": [880.0, 900.0, 910.0, 905.0, 899.0]},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 101.0, "samples_ts": [99.0, 104.0, 101.0, 100.0, 120.0]}
]"#;

const OLLAMA_STATS: &str = "\
total duration:       2.1s
prompt eval count:    12 token(s)
prompt eval rate:     480.00 tokens/s
eval count:           128 token(s)
eval rate:            61.50 tokens/s
prompt eval rate:     470.00 tokens/s
eval rate:            60.00 tokens/s
";

#[test]
fn median_needs_min_samples_and_positive_speeds() {
    assert!(median(&[1.0; MIN_SAMPLES - 1]).is_err());
    assert_eq!(median(&[5.0, 1.0, 3.0, 2.0, 4.0]), Ok(3.0));
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), Ok(3.5));
    assert!(median(&[1.0, 2.0, 0.0, 4.0, 5.0]).is_err());
    assert!(median(&[1.0, 2.0, f64::NAN, 4.0, 5.0]).is_err());
}

#[test]
fn llama_bench_decode_samples_only() {
    let s = parse_llama_bench(BENCH_JSON).expect("parse");
    assert_eq!(s, vec![99.0, 104.0, 101.0, 100.0, 120.0]);
    // The median ignores the one fast outlier; the mean would not.
    assert_eq!(median(&s), Ok(101.0));
    // No decode test, or two, is refused rather than guessed.
    assert!(parse_llama_bench(r#"[{"n_prompt":512,"n_gen":0,"samples_ts":[1]}]"#).is_err());
    let two = r#"[{"n_prompt":0,"n_gen":64,"samples_ts":[1]},{"n_prompt":0,"n_gen":128,"samples_ts":[2]}]"#;
    assert!(parse_llama_bench(two).is_err());
}

#[test]
fn ollama_reads_eval_rate_never_prompt_eval_rate() {
    assert_eq!(parse_ollama_verbose(OLLAMA_STATS), Ok(vec![61.5, 60.0]));
    assert!(parse_ollama_verbose("eval rate: fast tokens/s").is_err());
}

#[test]
fn llama_cpp_pin_is_enforced() {
    assert!(check_llama_pin("version: 6500 (d1d3c339)").is_ok());
    assert!(check_llama_pin("version: 6512 (0badc0de)").is_err());
    assert!(check_llama_pin("").is_err());
}

#[test]
fn mistralrs_is_recorded_not_dropped() {
    let ArmOutcome::NotRun { arm, reason } = mistralrs_outcome() else {
        panic!("mistral.rs must be NotRun until measured");
    };
    assert_eq!(arm, "mistral.rs");
    assert!(reason.contains(MISTRALRS_TAG) && reason.starts_with("Refused"));
}

#[test]
fn ollama_arm_is_digest_pinned_and_offline() {
    let dir = TempDir::new().expect("tmp");
    let spec = ollama_arm("Say hi", dir.path());
    let image = spec.image.clone().expect("container arm");
    let argv =
        super::super::comparator::container_argv(&image, &spec.command, &spec.env, dir.path())
            .expect("pinned image");
    assert!(argv.contains(&"--network=none".to_string()));
    assert!(image.contains("@sha256:"));
    let models = dir.path().join("ollama-models").display().to_string();
    assert!(spec
        .env
        .iter()
        .any(|(k, v)| k == "OLLAMA_MODELS" && *v == models));
}

fn fake_llama(bin: &Path, version: &str, json: &str) {
    std::fs::create_dir_all(bin).expect("bin");
    let write = |name: &str, body: String| {
        let p = bin.join(name);
        std::fs::write(&p, body).expect("write");
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    };
    write("llama-cli", format!("#!/bin/sh\necho '{version}' >&2\n"));
    write(
        "llama-bench",
        format!("#!/bin/sh\ncat <<'EOF'\n{json}\nEOF\n"),
    );
}

#[test]
fn llama_cpp_arm_end_to_end_hashes_what_it_reads() {
    let dir = TempDir::new().expect("tmp");
    let bin = dir.path().join("bin");
    fake_llama(&bin, "version: 6500 (d1d3c339)", BENCH_JSON);
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"gguf").expect("gguf");
    let spec = llama_cpp_arm(&bin, &gguf, dir.path());
    let ArmOutcome::Measured(m) = measure_llama_cpp(&spec, dir.path()).expect("measure") else {
        panic!("measured");
    };
    assert_eq!(m.arm, "llama.cpp");
    assert_eq!(m.decode_tok_s, 101.0);
    let bytes = std::fs::read(&spec.artifact).expect("artifact");
    let want = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes));
    assert_eq!(m.comparator.artifact_sha256, want);
}

#[test]
fn llama_cpp_arm_off_pin_is_refused() {
    let dir = TempDir::new().expect("tmp");
    let bin = dir.path().join("bin");
    fake_llama(&bin, "version: 6512 (0badc0de)", BENCH_JSON);
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"gguf").expect("gguf");
    let spec = llama_cpp_arm(&bin, &gguf, dir.path());
    let err = measure_llama_cpp(&spec, dir.path()).expect_err("off pin");
    assert!(err.contains("not at the pin"), "{err}");
}

#[test]
fn llama_cpp_arm_with_too_few_samples_is_refused() {
    let dir = TempDir::new().expect("tmp");
    let bin = dir.path().join("bin");
    let short = r#"[{"n_prompt":0,"n_gen":128,"samples_ts":[100.0,101.0]}]"#;
    fake_llama(&bin, "version: 6500 (d1d3c339)", short);
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"gguf").expect("gguf");
    let spec = llama_cpp_arm(&bin, &gguf, dir.path());
    assert!(measure_llama_cpp(&spec, dir.path()).is_err());
}
