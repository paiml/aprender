//! EXT-28 (aprender#4410, collapsed into #4401): the C2 speed arms (EXT-001 §10).
//!
//! Each competitor arm is an EXT-26 comparator arm whose artifact is the arm's own
//! speed output, so the block hashes exactly the bytes the number is read from:
//!
//! - `llama.cpp` — the existing oracle: `llama-server` built at [`LLAMA_CPP_COMMIT`]
//!   on the host, started and timed by OUR client, `apr test llm bench`, as
//!   scripts/parity_host_receipt.sh does (PERF-019, APR-PERF-GATE-001 §4.4.8). Its
//!   version line must name that commit, or the arm is refused.
//! - `ollama` — `NotRun` until our client times it (#5033): the pinned image's
//!   only timing so far is Ollama's own client (`ollama run --verbose`), which is
//!   not PP-25's one client (see [`ollama_outcome`]).
//! - `mistral.rs` — `NotRun`: upstream claims `qwen35` at v0.9.4 but nobody has
//!   measured it (EXT-24 bind receipt). It is recorded, never silently dropped.
//!
//! An arm's decode speed is the median of at least [`MIN_SAMPLES`] samples: one
//! run on a shared host has a CV near 19% (EXT-04, yoga, n=10). The output has the
//! field names of the EXT-19 ledger's `ArmSpeed`. No ratio is computed here; that
//! happens only inside the ledger (T28).

use super::comparator::{run_arm, ArmSpec, ComparatorBlock};
use serde::Serialize;
use serde_json::Value;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};

/// llama.cpp pin (contracts/pp-llama-001-comparator-pin-v1.yaml).
pub(crate) const LLAMA_CPP_COMMIT: &str = "d1d3c3396aa13a5f239109a822666c4870490ad5";
/// The Ollama image, by digest (EXT-24 bind receipt).
pub(crate) const OLLAMA_IMAGE: &str =
    "ollama/ollama:0.34.4@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551";
/// The Ollama model the arm runs; its manifest sha256 is pinned in the EXT-24 receipt.
pub(crate) const OLLAMA_MODEL: &str = "qwen3.5:4b";
/// mistral.rs pin, recorded with its refusal.
pub(crate) const MISTRALRS_TAG: &str = "v0.9.4";
/// Fewest samples an arm's median may be taken over.
pub(crate) const MIN_SAMPLES: usize = 5;
/// The client's prompt profile for the llama.cpp arm: 128 prompt and 128 generated
/// tokens per request.
pub(crate) const PROFILE: &str = "medium";

/// One arm on one cell: measured, or not run with the reason.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ArmOutcome {
    Measured(ArmMeasurement),
    NotRun { arm: String, reason: String },
}

/// Field-for-field the ledger's `ArmSpeed`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ArmMeasurement {
    pub arm: String,
    pub decode_tok_s: f64,
    pub comparator: ComparatorBlock,
}

fn sh(script: String) -> Vec<String> {
    vec!["sh".into(), "-c".into(), script]
}

/// Single-quote `s` for `sh -c`.
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn base_env() -> Vec<(String, String)> {
    vec![("PATH".into(), "/usr/local/bin:/usr/bin:/bin".into())]
}

/// The llama.cpp arm: `llama-server` in `bin_dir` on `port`, started and measured
/// by `apr test llm bench` (`apr`), which writes its report to the artifact.
///
/// One client for every server: llama.cpp's own bench binary times a different
/// quantity and is banned on the comparator path
/// (scripts/check_comparator_one_client.sh). `server_flags` come from the
/// declaration (scripts/llama_pin.toml through `llama_comparator_server_flags`),
/// never from a copy here.
pub(crate) fn llama_cpp_arm(
    apr: &Path,
    bin_dir: &Path,
    gguf: &Path,
    server_flags: &[String],
    port: u16,
    workdir: &Path,
) -> ArmSpec {
    let server = quote(&bin_dir.join("llama-server").display().to_string());
    let flags: String = server_flags
        .iter()
        .map(|f| format!(" {}", quote(f)))
        .collect();
    // `exec`, so the client's teardown signals the server, not a shell around it.
    let start = format!(
        "exec {server} -m {} --port {port}{flags}",
        quote(&gguf.display().to_string())
    );
    let artifact = workdir.join("llama.cpp-bench.json");
    let command = [
        apr.display().to_string(),
        "test".into(),
        "llm".into(),
        "bench".into(),
        "--url".into(),
        format!("http://{}:{port}", Ipv4Addr::LOCALHOST),
        "--start".into(),
        start,
        "--health-timeout".into(),
        "300".into(),
        "--profile".into(),
        PROFILE.into(),
        "--concurrency".into(),
        "1".into(),
        "--stream".into(),
        "--warmup".into(),
        "10".into(),
        "--duration".into(),
        "30".into(),
        "--cooldown".into(),
        "5".into(),
        "--runs".into(),
        MIN_SAMPLES.to_string(),
        "--runtime-name".into(),
        "llama.cpp".into(),
        "--output".into(),
        artifact.display().to_string(),
    ];
    ArmSpec {
        name: "llama.cpp".into(),
        command: command.into(),
        version_command: sh(format!("{server} --version 2>&1")),
        env: base_env(),
        artifact,
        image: None,
        workdir: workdir.to_path_buf(),
    }
}

/// The Ollama arm is recorded, not run, until our client times it (#5033).
///
/// `ollama run --verbose` reports Ollama's own client timing itself (`eval
/// rate`), not PERF-019's one client (PP-25), so it is no sample.
pub(crate) fn ollama_outcome() -> ArmOutcome {
    let image = OLLAMA_IMAGE.split('@').next().unwrap_or(OLLAMA_IMAGE);
    ArmOutcome::NotRun {
        arm: "ollama".into(),
        reason: format!(
            "Refused{{one-client}}: {image} with {OLLAMA_MODEL} is not yet timed by our \
             client, and its own `eval rate` is not PP-25's one client (#5033)"
        ),
    }
}

/// The mistral.rs arm is recorded, not run, until its `qwen35` load is measured.
pub(crate) fn mistralrs_outcome() -> ArmOutcome {
    ArmOutcome::NotRun {
        arm: "mistral.rs".into(),
        reason: format!(
            "Refused{{unmeasured}}: qwen35 load claimed upstream at {MISTRALRS_TAG} \
             (examples/{{python,server}}/qwen3_5.py), never measured (EXT-24)"
        ),
    }
}

/// Median of at least [`MIN_SAMPLES`] finite positive samples.
pub(crate) fn median(samples: &[f64]) -> Result<f64, String> {
    if samples.len() < MIN_SAMPLES {
        return Err(format!(
            "{} samples; the median needs at least {MIN_SAMPLES}",
            samples.len()
        ));
    }
    if let Some(bad) = samples.iter().find(|x| !x.is_finite() || **x <= 0.0) {
        return Err(format!("sample {bad} is not a positive speed"));
    }
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Ok(if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    })
}

/// Decode samples from an `apr test llm bench` report: each run's
/// `decode_tok_per_sec` (1000 / ITL p50, so time to first token is excluded). A
/// run with a failed request, or with no successful one, is refused: the client
/// refuses it too, but the number is read from this file, so the file is checked.
pub(crate) fn parse_llm_bench(json: &str) -> Result<Vec<f64>, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("bench report: {e}"))?;
    let runs = v["runs"].as_array().ok_or("bench report: no `runs` list")?;
    runs.iter()
        .enumerate()
        .map(|(i, r)| {
            let n = i + 1;
            let (ok, failed) = (r["successful"].as_u64(), r["failed"].as_u64());
            if !matches!((ok, failed), (Some(s), Some(0)) if s > 0) {
                return Err(format!(
                    "bench report: run {n} has successful {ok:?}, failed {failed:?}"
                ));
            }
            r["decode_tok_per_sec"]
                .as_f64()
                .ok_or_else(|| format!("bench report: run {n} has no decode_tok_per_sec"))
        })
        .collect()
}

/// The version line must name the pinned commit (llama.cpp prints its first 8
/// hex digits: `version: 6500 (d1d3c339)`).
pub(crate) fn check_llama_pin(version: &str) -> Result<(), String> {
    if version.contains(&LLAMA_CPP_COMMIT[..8]) {
        Ok(())
    } else {
        Err(format!(
            "llama.cpp arm is not at the pin {}: version `{version}`",
            &LLAMA_CPP_COMMIT[..8]
        ))
    }
}

fn measured(spec: &ArmSpec, block: ComparatorBlock, samples: &[f64]) -> Result<ArmOutcome, String> {
    Ok(ArmOutcome::Measured(ArmMeasurement {
        arm: spec.name.clone(),
        decode_tok_s: median(samples).map_err(|e| format!("{}: {e}", spec.name))?,
        comparator: block,
    }))
}

fn read(path: &PathBuf) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Run the llama.cpp arm and read its decode speed from the hashed artifact.
pub(crate) fn measure_llama_cpp(spec: &ArmSpec, log_dir: &Path) -> Result<ArmOutcome, String> {
    let block = run_arm(spec, log_dir)?;
    check_llama_pin(&block.version)?;
    let samples = parse_llm_bench(&read(&spec.artifact)?)?;
    measured(spec, block, &samples)
}

#[cfg(test)]
#[path = "speed_arms_tests.rs"]
mod tests;
