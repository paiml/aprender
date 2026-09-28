// ── #4551 APR-OBS-001 §2.7: `apr bench --emit raw-samples-v1` ───────────────
//
// The subject EMITS; it never opens a ledger. The recorder (`apr-perf-recorder`)
// validates this document against `contracts/apr-raw-samples-v1.yaml`, derives
// every statistic, checks identity, and appends the ledger row. So this block
// carries only what the subject itself measured or knows, and no summary: a
// derived number written here would be one the recorder cannot re-derive.

/// The only `--emit` format `apr bench` knows.
pub(crate) const RAW_SAMPLES_V1: &str = "raw-samples-v1";

/// What the environment contributes to the document. Split out so the
/// builder is pure and its tests do not depend on the host or the clock.
struct EmitEnv {
    ts: String,
    host: Option<String>,
    build_identity: Option<String>,
    binary_sha256: Option<String>,
    model_sha256: Option<String>,
}

impl EmitEnv {
    fn capture(model: &Path) -> Self {
        let exe = std::env::current_exe().ok();
        Self {
            ts: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            host: host_name(),
            build_identity: git_sha(env!("APR_GIT_SHA")),
            binary_sha256: exe.as_deref().and_then(file_sha256),
            model_sha256: file_sha256(model),
        }
    }
}

/// The host's name, or `None` when it cannot be read. Never a placeholder:
/// the recorder refuses a row whose host it cannot match.
fn host_name() -> Option<String> {
    let from_proc = std::fs::read_to_string("/proc/sys/kernel/hostname").ok();
    let from_cmd = || {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
    };
    from_proc
        .or_else(from_cmd)
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
}

/// The commit the binary was built from, or `None` when the build stamped a
/// non-commit fallback (`v{version}+no-git`, see `aprender-build-sha`). The
/// contract wants a commit, so a fallback is dropped rather than passed on.
fn git_sha(stamp: &str) -> Option<String> {
    let s = stamp.trim();
    ((7..=40).contains(&s.len()) && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then(|| s.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Build the `raw-samples-v1` document, or refuse.
///
/// Refuses when the path that ran did not record per-iteration samples, and
/// when the samples do not line up with the timed iterations. Both would
/// otherwise be filled with invented numbers.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn raw_samples_v1(
    model: &Path,
    config: &BenchConfig,
    result: &BenchResult,
    env: &EmitEnv,
) -> Result<serde_json::Value> {
    let Some(raw) = result.raw.as_ref() else {
        return Err(CliError::ValidationFailed(format!(
            "--emit {RAW_SAMPLES_V1}: this model's bench path does not record \
             per-iteration tokens and TTFT, so there are no raw samples to emit \
             (supported: Qwen3.5 hybrid GGUF, the qwen35 session path)"
        )));
    };
    if raw.samples.is_empty() || raw.samples.len() != result.iteration_times.len() {
        return Err(CliError::ValidationFailed(format!(
            "--emit {RAW_SAMPLES_V1}: {} samples for {} timed iterations",
            raw.samples.len(),
            result.iteration_times.len()
        )));
    }
    let samples: Vec<serde_json::Value> = raw
        .samples
        .iter()
        .enumerate()
        .map(|(index, s)| {
            serde_json::json!({
                "index": index,
                "wall_ms": s.wall.as_secs_f64() * 1000.0,
                "ttft_ms": s.ttft.as_secs_f64() * 1000.0,
                "completion_tokens": s.completion_tokens,
            })
        })
        .collect();
    let model_id = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.display().to_string());
    let request_id = sha256_hex(
        format!(
            "{}|{}|{}|{}",
            env.ts,
            env.host.as_deref().unwrap_or(""),
            std::process::id(),
            model_id
        )
        .as_bytes(),
    );
    Ok(serde_json::json!({
        "schema": RAW_SAMPLES_V1,
        "ts": env.ts,
        "host": env.host,
        "apr_version": env!("CARGO_PKG_VERSION"),
        "build_identity": env.build_identity,
        "binary_sha256": env.binary_sha256,
        "model_id": model_id,
        "model_sha256": env.model_sha256,
        "backend": compute_class(),
        "gpu_proof": {"source": "qwen35_session.on_gpu", "on_gpu": raw.on_gpu},
        "request_id": &request_id[..32],
        "workload": {
            "prompt_sha256": sha256_hex(config.prompt.as_bytes()),
            "max_tokens": config.max_tokens,
            "warmup": config.warmup,
            "iterations": config.iterations,
        },
        "samples": samples,
    }))
}

#[cfg(test)]
mod raw_samples_v1_tests {
    use super::*;

    fn env() -> EmitEnv {
        EmitEnv {
            ts: "2026-09-27T15:00:00.000Z".to_string(),
            host: Some("lambda-vector".to_string()),
            build_identity: Some("2138b1c79a".to_string()),
            binary_sha256: Some("b".repeat(64)),
            model_sha256: Some("m".repeat(64)),
        }
    }

    fn config(iterations: usize) -> BenchConfig {
        BenchConfig {
            warmup: 1,
            iterations,
            max_tokens: 128,
            prompt: "What is 2+2?".to_string(),
            quiet: true,
        }
    }

    fn result(times_ms: &[u64], raw: Option<RawSamples>) -> BenchResult {
        let iteration_times: Vec<Duration> =
            times_ms.iter().map(|&m| Duration::from_millis(m)).collect();
        BenchResult {
            total_tokens: 0,
            total_time: iteration_times.iter().sum(),
            tokens_per_second: 0.0,
            time_to_first_token: Duration::ZERO,
            iteration_times,
            mean_time: Duration::ZERO,
            median_time: Duration::ZERO,
            std_dev: Duration::ZERO,
            passed: true,
            raw,
        }
    }

    fn sample(wall_ms: u64, tokens: usize, ttft_ms: u64) -> RawSample {
        RawSample {
            wall: Duration::from_millis(wall_ms),
            completion_tokens: tokens,
            ttft: Duration::from_millis(ttft_ms),
        }
    }

    /// Every sample survives, in order, with its own tokens and TTFT.
    #[test]
    fn emits_one_sample_per_timed_iteration() {
        let raw = RawSamples {
            on_gpu: true,
            samples: vec![sample(1000, 128, 40), sample(1100, 127, 45)],
        };
        let doc = raw_samples_v1(
            Path::new("/m/Qwen3.5-4B-Q4_K_M.gguf"),
            &config(2),
            &result(&[1000, 1100], Some(raw)),
            &env(),
        )
        .expect("a measured run emits");
        assert_eq!(doc["schema"], RAW_SAMPLES_V1);
        assert_eq!(doc["model_id"], "Qwen3.5-4B-Q4_K_M.gguf");
        assert_eq!(doc["build_identity"], "2138b1c79a");
        assert_eq!(doc["gpu_proof"]["on_gpu"], true);
        assert_eq!(doc["workload"]["max_tokens"], 128);
        let s = doc["samples"].as_array().expect("samples array");
        assert_eq!(s.len(), 2);
        assert_eq!(s[1]["index"], 1);
        assert_eq!(s[1]["completion_tokens"], 127);
        assert_eq!(s[1]["wall_ms"], 1100.0);
        assert_eq!(s[1]["ttft_ms"], 45.0);
        assert_eq!(doc["request_id"].as_str().map(str::len), Some(32));
    }

    /// No summary statistic is emitted: the recorder derives them all.
    #[test]
    fn emits_no_derived_statistics() {
        let raw = RawSamples {
            on_gpu: false,
            samples: vec![sample(10, 1, 1)],
        };
        let doc = raw_samples_v1(Path::new("m.gguf"), &config(1), &result(&[10], Some(raw)), &env())
            .expect("emits");
        for key in ["tokens_per_second", "mean_time_ms", "median_time_ms", "passed"] {
            assert!(doc.get(key).is_none(), "{key} is recorder-derived");
        }
    }

    /// A build stamped without a commit emits `null`, which the contract refuses,
    /// never the fallback string.
    #[test]
    fn a_non_commit_build_stamp_is_not_passed_on() {
        assert_eq!(git_sha("2138b1c79a").as_deref(), Some("2138b1c79a"));
        for stamp in ["v0.71.0+no-git", "unknown", "", "abc", "2138B1C79A"] {
            assert_eq!(git_sha(stamp), None, "{stamp}");
        }
    }

    /// A path that did not measure per-iteration tokens refuses; it does not
    /// emit an empty or invented sample list.
    #[test]
    fn refuses_a_path_without_samples() {
        let err = raw_samples_v1(Path::new("m.gguf"), &config(2), &result(&[1, 2], None), &env())
            .expect_err("no samples → refuse");
        assert!(err.to_string().contains("does not record"), "{err}");
    }

    /// Samples that do not line up with the timed iterations refuse.
    #[test]
    fn refuses_samples_that_do_not_match_the_iterations() {
        let raw = RawSamples {
            on_gpu: true,
            samples: vec![sample(1, 1, 1)],
        };
        let err = raw_samples_v1(Path::new("m.gguf"), &config(2), &result(&[1, 2], Some(raw)), &env())
            .expect_err("1 sample for 2 iterations → refuse");
        assert!(err.to_string().contains("1 samples for 2"), "{err}");
    }
}
