//! `apr test llm` — GH-876 Milestone 2.
//!
//! A CLI surface over `aprender-test-lib`'s llm module, which is where this
//! project's inference-benchmark logic lives and has lived. Nothing in `apr`
//! could reach it: apr-cli carried only a DEV-dependency on the crate, without
//! its `llm` feature, and `commands/serve_loadtest.rs` — the single file that
//! imported it — was not listed in `commands/mod.rs`, so it never compiled.
//! Unreachable code that measures the thing you are trying to measure is worse
//! than absent code, because the measurement gets hand-rolled instead.
//!
//! It was, on 2026-08-24. Two errors followed that this harness makes
//! structurally hard:
//!
//!   · an END-TO-END rate (11.8 tok/s, including ~9.7 s of model load) quoted
//!     against a historical DECODE reference of ~40 tok/s — two different
//!     quantities compared as though they were one;
//!   · a differential built by subtracting a `latency` that itself INCLUDES
//!     the variable load time, yielding 244 tok/s where the true marginal
//!     decode rate is ~107.
//!
//! `LoadTestResult` already carries the right quantity for both:
//! `decode_tok_per_sec` = 1000 / `itl_p50_ms`, the inter-token rate, which
//! excludes time-to-first-token by construction and so cannot absorb a model
//! load. Reading a field beats deriving a number.
use crate::error::{CliError, Result};
use apr_test::llm::{
    benchmark::{Benchmark, BenchmarkConfig, BenchmarkReport},
    client::ChatRequest,
    load_profile, load_prompts_from_file,
    loadtest::LoadTestResult,
    PromptProfile,
};
use std::path::Path;
use std::time::Duration;

/// Arguments for one benchmark invocation.
///
/// A struct rather than a 16-argument function: the clippy pedantic lint that
/// would fire here is pointing at something real, since positional arguments of
/// the same type (five `u64` durations) are exactly where a caller silently
/// transposes warmup and cooldown.
pub struct BenchArgs<'a> {
    /// Endpoint under measurement.
    pub url: &'a str,
    /// Model name sent in the request body.
    pub model: &'a str,
    /// Command that starts the runtime, if the harness owns its lifecycle.
    pub start: Option<&'a str>,
    /// Seconds to wait for readiness.
    pub health_timeout: u64,
    /// Warm-up seconds, discarded.
    pub warmup: u64,
    /// Measured seconds per run.
    pub duration: u64,
    /// Concurrent request streams.
    pub concurrency: usize,
    /// Number of measured runs.
    pub runs: usize,
    /// Cooldown seconds between runs.
    pub cooldown: u64,
    /// Label recorded in the report.
    pub runtime_name: &'a str,
    /// Prior report or run to compare against.
    pub baseline: Option<&'a Path>,
    /// Fractional regression that fails the run.
    pub fail_on_regression: Option<f64>,
    /// Where to write the JSON report.
    pub output: Option<&'a Path>,
    /// Streaming responses, needed for TTFT and TPOT.
    pub stream: bool,
    /// Named prompt profile.
    pub profile: &'a str,
    /// Prompt file, overriding the profile.
    pub prompts: Option<&'a Path>,
}

/// Run the benchmark lifecycle and report.
pub async fn run_bench(args: BenchArgs<'_>) -> Result<()> {
    let prompts = resolve_prompts(args.profile, args.prompts)?;
    let workload = describe_workload(args.profile, args.prompts, prompts.len());
    let baseline = load_baseline(args.baseline)?;

    let config = BenchmarkConfig {
        url: args.url.to_string(),
        model: args.model.to_string(),
        start_command: args.start.map(str::to_string),
        health_timeout: Duration::from_secs(args.health_timeout),
        warmup: Duration::from_secs(args.warmup),
        duration: Duration::from_secs(args.duration),
        concurrency: args.concurrency,
        runs: args.runs,
        cooldown: Duration::from_secs(args.cooldown),
        prompts,
        runtime_name: args.runtime_name.to_string(),
        baseline,
        fail_on_regression: args.fail_on_regression,
        stream: args.stream,
        trace_level: None,
        num_layers: None,
    };

    println!("runtime  {}", args.runtime_name);
    println!("endpoint {}", args.url);
    println!("workload {workload}");
    println!(
        "protocol {} run(s) x {}s, warmup {}s, cooldown {}s, concurrency {}",
        args.runs, args.duration, args.warmup, args.cooldown, args.concurrency
    );

    let mut benchmark = Benchmark::new(config);
    let report = benchmark
        .run()
        .await
        .map_err(|e| CliError::InferenceFailed(e.to_string()))?;

    print_report(&report);

    if let Some(path) = args.output {
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::InvalidFormat(format!("serialising report: {e}")))?;
        std::fs::write(path, json)?;
        println!("\nreport written to {}", path.display());
    }

    // A RUN THAT GENERATED NO TOKENS IS A FAILED RUN, NOT A FAST ONE.
    //
    // `successful` counts HTTP 200. A server can answer 200 with an empty
    // completion, and then every derived rate is zero while the request count
    // and the throughput look spectacular. Observed here on 2026-08-24 while
    // testing PREFILL_GRAPH=1: 727 "successful" requests in 15s — 40x the
    // normal rate — every one of them carrying zero tokens, reported as
    // `decode 0.0 tok/s` beside a passing run. The same shape as every
    // cannot-fail gate this protocol exists to catch, sitting in the
    // measurement tool itself.
    let empty: Vec<usize> = report
        .runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.successful > 0 && r.avg_tok_per_req <= 0.0)
        .map(|(i, _)| i + 1)
        .collect();
    if !empty.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "run(s) {} completed {} request(s) that generated ZERO tokens. A 200 \
             with an empty completion is not a measurement — every rate derived \
             from it is zero while the request count looks excellent.",
            empty
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            report.runs.iter().map(|r| r.successful).sum::<u64>()
        )));
    }

    // A benchmark that detects a regression past its declared threshold and
    // then exits 0 is a gate that cannot fail.
    let failed: Vec<&str> = report
        .regressions
        .iter()
        .filter(|r| r.exceeds_threshold)
        .map(|r| r.metric.as_str())
        .collect();
    if failed.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "regression past threshold in: {}",
            failed.join(", ")
        )))
    }
}

/// A file overrides the profile; an unknown profile name is rejected rather
/// than quietly falling back to a default, since a silent substitution changes
/// the workload the report then claims to have run.
fn resolve_prompts(profile: &str, file: Option<&Path>) -> Result<Vec<ChatRequest>> {
    if let Some(p) = file {
        return load_prompts_from_file(p)
            .map_err(|e| CliError::InvalidFormat(format!("prompt file {}: {e}", p.display())));
    }
    let parsed = PromptProfile::from_name(profile).ok_or_else(|| {
        CliError::InvalidInput(format!(
            "unknown prompt profile {profile:?}; expected micro, short, medium or long"
        ))
    })?;
    Ok(load_profile(parsed))
}

/// One line naming the workload, so the report is self-describing.
fn describe_workload(profile: &str, file: Option<&Path>, count: usize) -> String {
    match file {
        Some(p) => format!("{} prompt(s) from {}", count, p.display()),
        None => format!("profile {profile} ({count} prompt(s))"),
    }
}

/// Accept either a full report or a bare run as the baseline.
fn load_baseline(path: Option<&Path>) -> Result<Option<LoadTestResult>> {
    let Some(p) = path else { return Ok(None) };
    let content = std::fs::read_to_string(p)?;
    if let Ok(report) = serde_json::from_str::<BenchmarkReport>(&content) {
        return Ok(report.runs.into_iter().next());
    }
    let single: LoadTestResult = serde_json::from_str(&content)
        .map_err(|e| CliError::InvalidFormat(format!("baseline {}: {e}", p.display())))?;
    Ok(Some(single))
}

fn print_report(report: &BenchmarkReport) {
    for (i, run) in report.runs.iter().enumerate() {
        println!("\n--- run {}/{} ---", i + 1, report.runs.len());
        println!(
            "  requests     {} ok / {} failed",
            run.successful, run.failed
        );
        println!("  ttft   p50   {:.1} ms", run.ttft_p50_ms);
        println!("  itl    p50   {:.2} ms", run.itl_p50_ms);
        // The headline number. Excludes TTFT, so a model load cannot inflate
        // or deflate it — unlike an end-to-end tokens/wall-clock rate.
        println!("  decode       {:.1} tok/s", run.decode_tok_per_sec);
        println!("  prefill      {:.1} tok/s", run.prefill_tok_per_sec);
        println!("  throughput   {:.2} req/s", run.throughput_rps);
        println!(
            "  end-to-end   {:.1} tok/s  (INCLUDES prefill; not a decode rate)",
            run.tokens_per_sec
        );
    }

    let a = &report.aggregate;
    println!("\n--- across {} run(s) ---", report.runs.len());
    print_stat("throughput (req/s)", &a.throughput_rps);
    print_stat("latency p50 (ms) ", &a.latency_p50);
    print_stat("ttft p50 (ms)    ", &a.ttft_p50);
    print_stat("tpot p50 (ms)    ", &a.tpot_p50);
    print_stat("tokens/s (e2e)   ", &a.tokens_per_sec);

    if !report.regressions.is_empty() {
        println!("\n--- vs baseline ---");
        for r in &report.regressions {
            let verdict = if r.exceeds_threshold { "FAIL" } else { "ok  " };
            println!(
                "  {verdict} {:<18} {:.2} -> {:.2}  ({:+.1}%)",
                r.metric, r.baseline_value, r.current_value, r.change_pct
            );
        }
    }
}

/// Print a metric with its spread. A single number with no interval invites
/// the reader to treat run-to-run noise as a result.
fn print_stat(label: &str, s: &apr_test::llm::benchmark::StatSummary) {
    println!(
        "  {label}  mean {:>9.2}  sd {:>8.2}  95% CI [{:.2}, {:.2}]  n={}",
        s.mean,
        s.stddev,
        s.ci_95_lower,
        s.ci_95_upper,
        s.values.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_profile_resolves_to_prompts() {
        for name in ["micro", "short", "medium", "long"] {
            let got = resolve_prompts(name, None)
                .unwrap_or_else(|e| panic!("profile {name} should resolve: {e}"));
            assert!(!got.is_empty(), "profile {name} yielded no prompts");
        }
    }

    #[test]
    fn an_unknown_profile_is_rejected_not_defaulted() {
        // A silent fallback would let the report name a workload it did not run.
        let err = resolve_prompts("gigantic", None).expect_err("must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("gigantic"),
            "error should quote the input: {msg}"
        );
        assert!(
            msg.contains("medium"),
            "error should list the options: {msg}"
        );
    }

    #[test]
    fn profile_case_does_not_change_the_workload() {
        let lower = resolve_prompts("medium", None).expect("lower");
        let upper = resolve_prompts("MEDIUM", None).expect("upper");
        assert_eq!(lower.len(), upper.len());
    }

    #[test]
    fn a_prompt_file_overrides_the_profile() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("prompts.yaml");
        // Two prompts, so the count cannot coincide with a one-prompt profile.
        std::fs::write(
            &path,
            "prompts:\n  - role: user\n    content: \"hi\"\n    max_tokens: 4\n  - role: user\n    content: \"there\"\n    max_tokens: 4\n",
        )
        .expect("write");
        let from_file = resolve_prompts("long", Some(&path)).expect("file should load");
        let from_profile = resolve_prompts("long", None).expect("profile");
        assert_eq!(from_file.len(), 2, "the file defines the workload");
        assert_ne!(
            from_file.len(),
            from_profile.len(),
            "the file must not agree with the profile by accident, or the test proves nothing"
        );
    }

    #[test]
    fn a_malformed_prompt_file_fails_rather_than_falling_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "prompts: []\n").expect("write");
        // An empty workload would otherwise benchmark nothing and report a rate.
        resolve_prompts("medium", Some(&path)).expect_err("an empty prompt set must fail");
    }

    #[test]
    fn a_missing_baseline_is_none_and_a_bad_one_is_an_error() {
        assert!(load_baseline(None).expect("none is fine").is_none());
        let dir = tempfile::tempdir().expect("tempdir");
        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, "{\"not\": \"a result\"}").expect("write");
        load_baseline(Some(&bad)).expect_err("an unparseable baseline must fail loudly");
    }

    #[test]
    fn the_workload_line_names_its_source() {
        assert!(describe_workload("medium", None, 3).contains("profile medium"));
        let p = Path::new("/tmp/x.json");
        assert!(describe_workload("medium", Some(p), 7).contains("x.json"));
    }
}
