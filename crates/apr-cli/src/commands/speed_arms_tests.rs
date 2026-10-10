//! EXT-28 (aprender#4410): the C2 speed arms.

use super::*;
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;

/// An `apr test llm bench` report, trimmed to the fields the arm reads.
const REPORT_JSON: &str = r#"{"runs": [
  {"total_requests": 9, "successful": 9, "failed": 0, "decode_tok_per_sec": 99.0, "prefill_tok_per_sec": 900.0},
  {"total_requests": 9, "successful": 9, "failed": 0, "decode_tok_per_sec": 104.0, "prefill_tok_per_sec": 905.0},
  {"total_requests": 9, "successful": 9, "failed": 0, "decode_tok_per_sec": 101.0, "prefill_tok_per_sec": 899.0},
  {"total_requests": 9, "successful": 9, "failed": 0, "decode_tok_per_sec": 100.0, "prefill_tok_per_sec": 910.0},
  {"total_requests": 9, "successful": 9, "failed": 0, "decode_tok_per_sec": 120.0, "prefill_tok_per_sec": 880.0}
], "regressions": []}"#;

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
fn llm_bench_reads_each_runs_decode_rate_never_prefill() {
    let s = parse_llm_bench(REPORT_JSON).expect("parse");
    assert_eq!(s, vec![99.0, 104.0, 101.0, 100.0, 120.0]);
    // The median ignores the one fast outlier; the mean would not.
    assert_eq!(median(&s), Ok(101.0));
    // No runs list, a failed request, no successful one, or no decode rate is
    // refused rather than guessed.
    assert!(parse_llm_bench(r#"[{"decode_tok_per_sec":1.0}]"#).is_err());
    let run = |s: &str| format!(r#"{{"runs":[{s}]}}"#);
    let good = r#"{"successful":3,"failed":0,"decode_tok_per_sec":7.5}"#;
    assert_eq!(parse_llm_bench(&run(good)), Ok(vec![7.5]));
    for bad in [
        r#"{"successful":3,"failed":1,"decode_tok_per_sec":7.5}"#,
        r#"{"successful":0,"failed":0,"decode_tok_per_sec":7.5}"#,
        r#"{"failed":0,"decode_tok_per_sec":7.5}"#,
        r#"{"successful":3,"decode_tok_per_sec":7.5}"#,
        r#"{"successful":3,"failed":0}"#,
    ] {
        assert!(parse_llm_bench(&run(bad)).is_err(), "{bad}");
    }
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

/// A fake host: `llama-server` prints `version`, and a fake `apr` records its argv
/// beside itself (`apr.argv`) and writes `report` to the path after `--output`,
/// as the real client does.
fn fake_host(root: &Path, version: &str, report: &str) -> (PathBuf, PathBuf) {
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).expect("bin");
    let write = |p: PathBuf, body: String| {
        std::fs::write(&p, body).expect("write");
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        p
    };
    write(
        bin.join("llama-server"),
        format!("#!/bin/sh\necho '{version}' >&2\n"),
    );
    let apr = write(
        root.join("apr"),
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$0.argv\"\n\
             while [ $# -gt 0 ]; do [ \"$1\" = --output ] && o=\"$2\"; shift; done\n\
             cat > \"$o\" <<'EOF'\n{report}\nEOF\n"
        ),
    );
    (apr, bin)
}

/// Stands in for the declared flags (scripts/llama_pin.toml).
fn declared_flags() -> Vec<String> {
    ["-ngl", "999", "-c", "4096"].map(String::from).to_vec()
}

#[test]
fn llama_cpp_arm_is_our_client_driving_llama_server() {
    let dir = TempDir::new().expect("tmp");
    let (apr, bin) = (dir.path().join("apr"), dir.path().join("bin"));
    let gguf = dir.path().join("m.gguf");
    let spec = llama_cpp_arm(&apr, &bin, &gguf, &declared_flags(), 18093, dir.path());
    let c = &spec.command;
    // The process the arm executes is our client, never a llama.cpp binary.
    let client = [
        apr.display().to_string(),
        "test".into(),
        "llm".into(),
        "bench".into(),
    ];
    assert_eq!(c[..4], client);
    let after = |flag: &str| {
        let i = c.iter().position(|a| a == flag).expect("flag present");
        c[i + 1].clone()
    };
    assert_eq!(
        after("--url"),
        format!("http://{}:18093", Ipv4Addr::LOCALHOST)
    );
    // The server the client starts is llama-server, with the declared flags verbatim.
    assert_eq!(
        after("--start"),
        format!(
            "exec '{}' -m '{}' --port 18093 '-ngl' '999' '-c' '4096'",
            bin.join("llama-server").display(),
            gguf.display()
        )
    );
    assert_eq!(after("--runs"), MIN_SAMPLES.to_string());
    assert_eq!(after("--profile"), PROFILE);
    assert_eq!(after("--concurrency"), "1");
    assert_eq!(after("--output"), spec.artifact.display().to_string());
    assert!(c.iter().any(|a| a == "--stream"));
}

#[test]
fn llama_cpp_arm_end_to_end_hashes_what_it_reads() {
    let dir = TempDir::new().expect("tmp");
    let (apr, bin) = fake_host(dir.path(), "version: 6500 (d1d3c339)", REPORT_JSON);
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"gguf").expect("gguf");
    let spec = llama_cpp_arm(&apr, &bin, &gguf, &declared_flags(), 18093, dir.path());
    let ArmOutcome::Measured(m) = measure_llama_cpp(&spec, dir.path()).expect("measure") else {
        panic!("measured");
    };
    assert_eq!(m.arm, "llama.cpp");
    assert_eq!(m.decode_tok_s, 101.0);
    let bytes = std::fs::read(&spec.artifact).expect("artifact");
    let want = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes));
    assert_eq!(m.comparator.artifact_sha256, want);
    // The client was handed the server command whole, as one argument.
    let argv = std::fs::read_to_string(dir.path().join("apr.argv")).expect("argv");
    assert!(
        argv.lines()
            .any(|l| l.starts_with("exec ") && l.ends_with("'-c' '4096'")),
        "{argv}"
    );
}

#[test]
fn llama_cpp_arm_off_pin_is_refused() {
    let dir = TempDir::new().expect("tmp");
    let (apr, bin) = fake_host(dir.path(), "version: 6512 (0badc0de)", REPORT_JSON);
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"gguf").expect("gguf");
    let spec = llama_cpp_arm(&apr, &bin, &gguf, &declared_flags(), 18093, dir.path());
    let err = measure_llama_cpp(&spec, dir.path()).expect_err("off pin");
    assert!(err.contains("not at the pin"), "{err}");
}

#[test]
fn llama_cpp_arm_with_too_few_samples_is_refused() {
    let dir = TempDir::new().expect("tmp");
    let short = r#"{"runs": [
      {"successful": 9, "failed": 0, "decode_tok_per_sec": 100.0},
      {"successful": 9, "failed": 0, "decode_tok_per_sec": 101.0}
    ]}"#;
    let (apr, bin) = fake_host(dir.path(), "version: 6500 (d1d3c339)", short);
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"gguf").expect("gguf");
    let spec = llama_cpp_arm(&apr, &bin, &gguf, &declared_flags(), 18093, dir.path());
    let err = measure_llama_cpp(&spec, dir.path()).expect_err("2 samples");
    assert!(err.contains("at least"), "{err}");
}
