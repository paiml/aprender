//! EXT-28 (aprender#4410, collapsed into #4401): the C2 speed arms (EXT-001 §10).
//!
//! Each competitor arm is an EXT-26 comparator arm whose artifact is the arm's own
//! speed output, so the block hashes exactly the bytes the number is read from:
//!
//! - `llama.cpp` — the existing oracle, `llama-bench` built at [`LLAMA_CPP_COMMIT`]
//!   on the host. Its version line must name that commit, or the arm is refused.
//! - `ollama` — the pinned image, run by digest with the network denied; the model
//!   store lives inside the workdir, pre-populated with the pinned manifest.
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
/// Tokens generated per sample.
pub(crate) const N_GEN: u32 = 128;

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

/// The llama.cpp arm: `llama-bench` in `bin_dir`, decode only, JSON to the artifact.
pub(crate) fn llama_cpp_arm(bin_dir: &Path, gguf: &Path, workdir: &Path) -> ArmSpec {
    let bench = quote(&bin_dir.join("llama-bench").display().to_string());
    let cli = quote(&bin_dir.join("llama-cli").display().to_string());
    let artifact = workdir.join("llama.cpp-bench.json");
    ArmSpec {
        name: "llama.cpp".into(),
        command: sh(format!(
            "{bench} -m {} -p 0 -n {N_GEN} -r {MIN_SAMPLES} -o json > {}",
            quote(&gguf.display().to_string()),
            quote(&artifact.display().to_string())
        )),
        version_command: sh(format!("{cli} --version 2>&1")),
        env: base_env(),
        artifact,
        image: None,
        workdir: workdir.to_path_buf(),
    }
}

/// The Ollama arm: the pinned image, offline, `MIN_SAMPLES` verbose runs whose
/// stats go to the artifact. `workdir/ollama-models` must hold the pinned model.
pub(crate) fn ollama_arm(prompt: &str, workdir: &Path) -> ArmSpec {
    let artifact = workdir.join("ollama-verbose.txt");
    let a = quote(&artifact.display().to_string());
    let script = format!(
        "ollama serve >/dev/null 2>&1 & \
         i=0; until ollama list >/dev/null 2>&1; do i=$((i+1)); [ $i -gt 60 ] && exit 3; sleep 1; done; \
         : > {a}; n=0; while [ $n -lt {MIN_SAMPLES} ]; do \
         ollama run {OLLAMA_MODEL} --verbose {} 2>>{a} >/dev/null || exit 4; n=$((n+1)); done",
        quote(prompt)
    );
    let mut env = base_env();
    env.push((
        "OLLAMA_MODELS".into(),
        workdir.join("ollama-models").display().to_string(),
    ));
    ArmSpec {
        name: "ollama".into(),
        command: sh(script),
        version_command: vec!["ollama".into(), "--version".into()],
        env,
        artifact,
        image: Some(OLLAMA_IMAGE.into()),
        workdir: workdir.to_path_buf(),
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

/// Decode samples from `llama-bench -o json`: the one test with `n_prompt == 0`
/// and `n_gen > 0`, its `samples_ts`.
pub(crate) fn parse_llama_bench(json: &str) -> Result<Vec<f64>, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("llama-bench json: {e}"))?;
    let tests: Vec<&Value> = v
        .as_array()
        .ok_or("llama-bench json: not a list")?
        .iter()
        .filter(|t| t["n_prompt"].as_u64() == Some(0) && t["n_gen"].as_u64().unwrap_or(0) > 0)
        .collect();
    let [t] = tests.as_slice() else {
        return Err(format!(
            "llama-bench json: {} decode tests, want exactly 1",
            tests.len()
        ));
    };
    t["samples_ts"]
        .as_array()
        .ok_or("llama-bench json: no samples_ts")?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| format!("samples_ts: {x} is not a number"))
        })
        .collect()
}

/// Decode samples from `ollama run --verbose` stats: each `eval rate:` line, never
/// a `prompt eval rate:` line.
pub(crate) fn parse_ollama_verbose(text: &str) -> Result<Vec<f64>, String> {
    text.lines()
        .map(str::trim)
        .filter_map(|l| l.strip_prefix("eval rate:"))
        .map(|rest| {
            let num = rest.trim().trim_end_matches("tokens/s").trim();
            num.parse::<f64>()
                .map_err(|e| format!("eval rate `{}`: {e}", rest.trim()))
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
    let samples = parse_llama_bench(&read(&spec.artifact)?)?;
    measured(spec, block, &samples)
}

/// Run the Ollama arm and read its decode speed from the hashed artifact.
pub(crate) fn measure_ollama(spec: &ArmSpec, log_dir: &Path) -> Result<ArmOutcome, String> {
    let block = run_arm(spec, log_dir)?;
    let samples = parse_ollama_verbose(&read(&spec.artifact)?)?;
    measured(spec, block, &samples)
}

#[cfg(test)]
#[path = "speed_arms_tests.rs"]
mod tests;
