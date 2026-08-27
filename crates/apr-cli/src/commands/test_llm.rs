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

    // VALIDITY, THEN THE RECEIPT — in that order, inside `emit_report`, which
    // exists so the order is a property of one function a test can execute.
    emit_report(&report, args.output)?;

    // A benchmark that detects a regression past its declared threshold and
    // then exits 0 is a gate that cannot fail.
    //
    // This one runs AFTER the write on purpose: a regression is a policy
    // verdict on numbers `emit_report` has already certified as a real
    // measurement, and the receipt describing the regressed run is the
    // evidence. Suppressing it would delete the finding.
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

/// Everything that decides whether this run produced a MEASUREMENT AT ALL.
///
/// `Ok(())` does not mean the run passed its gates. It means the numbers in the
/// report describe the thing that was asked for, so that a gate applied to them
/// is a gate applied to a measurement.
fn check_measurement_validity(report: &BenchmarkReport) -> Result<()> {
    // MEASUREMENT VALIDITY BEFORE MEASUREMENT — adopted from SGLang, which
    // asserts `res["completed"] == num_prompts` before it reads a throughput at
    // all (test_bench_serving.py). A request that never completed contributes
    // no sample, so a mean over the survivors silently EXCLUDES the failure and
    // reports the remainder as the result.
    //
    // That is not hypothetical here. `apr serve run --gpu --batch` hangs on four
    // concurrent chat requests; the benchmark reported `0.5 tok/s aggregate`
    // rather than an error, and a reader would call that slow rather than
    // broken (#2696).
    let failures: Vec<String> = report
        .runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.failed > 0)
        .map(|(i, r)| format!("run {} had {} failed request(s)", i + 1, r.failed))
        .collect();
    if !failures.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "{} — a throughput averaged over the requests that survived is not a \
             measurement of this runtime, it is a measurement of its survivors",
            failures.join("; ")
        )));
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

    Ok(())
}

/// Write the receipt — and only for a run that earned one.
///
/// WRITE-AFTER-VALIDATE, and the ordering is the whole point. Until #2706 the
/// write came FIRST and the validity assertions ran after it, so a run that
/// exited non-zero still left a receipt-shaped JSON file on disk. Every
/// consumer of that file then read it as a measurement:
///
///   · `scripts/parity_host_receipt.sh:104` and `:120` invoke this command with
///     `--output` and end the line `|| true`, discarding the exit status by
///     construction;
///   · `scripts/lib/parity_block.py:56` and `:78` decide whether a band and a
///     lane exist by `os.path.exists` on those paths, and `_samples` (`:27-31`)
///     then reads `runs[]` without ever consulting a verdict.
///
/// So the artifact WAS the interface, and a failed benchmark could be fed into
/// a published parity ratio — the fabricated-measurement class this epic exists
/// to remove, sitting inside the epic's own instrument.
///
/// ABSENCE rather than a `"valid": false` marker, deliberately. A marker is
/// only better if consumers read it, and neither consumer above opens the file
/// to look for one; the flag would change nothing until every consumer changed
/// too, which is a gate that cannot fail wearing a new field name. Absence is
/// already load-bearing in the code that exists: a missing band makes
/// `_band_from` return `None` while `_lane_from` still declares all four, and
/// `bench_receipt.py:484-487` rejects that block with "an unmeasured band is
/// not a passing band"; a missing lane side makes `_lane_from` refuse "half a
/// comparison" (`parity_block.py:79`). The failure surfaces AT THE GATE instead
/// of being laundered into a ratio.
///
/// A STALE receipt is removed for the same reason. Leaving the previous run's
/// file at the path this run was told to write means the next consumer reads a
/// measurement of a DIFFERENT run and cannot tell — a shadowed artifact is
/// worse than a missing one.
fn emit_report(report: &BenchmarkReport, output: Option<&Path>) -> Result<()> {
    let verdict = check_measurement_validity(report);
    let Some(path) = output else { return verdict };

    if let Err(invalid) = verdict {
        return Err(match discard_stale_receipt(path) {
            Ok(true) => CliError::ValidationFailed(format!(
                "{invalid}\n  discarded {} — it would have described a run that \
                 produced no measurement",
                path.display()
            )),
            Ok(false) => invalid,
            Err(e) => CliError::ValidationFailed(format!(
                "{invalid}\n  AND the stale receipt at {} could not be removed \
                 ({e}), so it may still be read as a measurement of this run",
                path.display()
            )),
        });
    }

    let json = serde_json::to_string_pretty(report)
        .map_err(|e| CliError::InvalidFormat(format!("serialising report: {e}")))?;
    std::fs::write(path, json)?;
    println!("\nreport written to {}", path.display());
    Ok(())
}

/// Remove a receipt this run is not entitled to write. `Ok(true)` means one was
/// there; a path that was already clear is not an error.
fn discard_stale_receipt(path: &Path) -> std::io::Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
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

    // === #2706 — THE RECEIPT RULE ==========================================
    //
    // `--output` used to be written BEFORE the validity assertions ran, so a
    // benchmark that exited non-zero still left a receipt-shaped JSON file on
    // disk. The consumers of that path do not read the exit status
    // (`scripts/parity_host_receipt.sh:104`, `:120` end in `|| true`) and do
    // not read a verdict out of the file (`scripts/lib/parity_block.py:56`,
    // `:78` gate on `os.path.exists`), so the artifact was the interface.
    //
    // MUTATION for the four tests below that pass an output path: in
    // `emit_report`, hoist the write above `check_measurement_validity` — the
    // pre-#2706 ordering. Verified 2026-08-27: the three no-artifact tests go
    // RED (rc=101) and `a_valid_run_still_writes_the_receipt_its_consumer_reads`
    // stays GREEN. That last one is the discrimination case — a change that
    // simply stopped writing receipts would satisfy the other three.

    /// A `BenchmarkReport` in the exact shape `--output` writes and
    /// `parity_block.py::_samples` reads back.
    fn report_json(failed: u64, avg_tok_per_req: f64) -> String {
        let stat =
            r#"{"mean":1.0,"stddev":0.1,"ci_95_lower":0.9,"ci_95_upper":1.1,"values":[1.0]}"#;
        format!(
            r#"{{"runs":[{{"total_requests":10,"successful":{successful},"failed":{failed},
            "throughput_rps":2.0,"latency_p50_ms":100.0,"latency_p95_ms":120.0,
            "latency_p99_ms":130.0,"ttft_p50_ms":30.0,"tokens_per_sec":50.0,
            "avg_tok_per_req":{avg_tok_per_req},"itl_p50_ms":10.0,
            "decode_tok_per_sec":100.0,"prefill_tok_per_sec":400.0,
            "timestamp":"2026-08-27T00:00:00Z","runtime_name":"apr-cpu-c1",
            "elapsed_secs":30.0,"concurrency":1}}],
            "aggregate":{{"throughput_rps":{stat},"latency_p50":{stat},
            "tokens_per_sec":{stat},"ttft_p50":{stat},"tpot_p50":{stat}}},
            "regressions":[]}}"#,
            successful = 10 - failed,
        )
    }

    fn report(failed: u64, avg_tok_per_req: f64) -> BenchmarkReport {
        serde_json::from_str(&report_json(failed, avg_tok_per_req))
            .expect("fixture must deserialise as a BenchmarkReport")
    }

    #[test]
    fn a_run_with_failed_requests_leaves_no_consumable_artifact() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("apr-cpu-c1.json");
        let err = emit_report(&report(3, 42.0), Some(&path)).expect_err("must reject");
        assert!(
            err.to_string().contains("survivors"),
            "the error must say why: {err}"
        );
        assert!(
            !path.exists(),
            "a failed benchmark wrote {} — every consumer of that path gates on \
             existence, not on the exit status",
            path.display()
        );
    }

    #[test]
    fn a_zero_token_run_leaves_no_consumable_artifact() {
        // The second invalid class, so the guard is not proved by one input.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("apr-cpu-c4.json");
        let err = emit_report(&report(0, 0.0), Some(&path)).expect_err("must reject");
        assert!(
            err.to_string().contains("ZERO tokens"),
            "the error must say why: {err}"
        );
        assert!(
            !path.exists(),
            "a zero-token run wrote {}, which reads as 0.0 tok/s rather than as broken",
            path.display()
        );
    }

    #[test]
    fn an_invalid_run_discards_the_previous_run_s_receipt() {
        // A shadowed artifact is worse than a missing one: the stale file would
        // be read as a measurement of THIS run.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("apr-cpu-c8.json");
        std::fs::write(&path, r#"{"runs":[{"decode_tok_per_sec":999.0}]}"#).expect("seed");
        let err = emit_report(&report(1, 42.0), Some(&path)).expect_err("must reject");
        assert!(
            err.to_string().contains("discarded"),
            "removing a stale receipt must be reported, not silent: {err}"
        );
        assert!(
            !path.exists(),
            "the previous run's receipt survived at {} and now describes a run \
             that never happened",
            path.display()
        );
    }

    #[test]
    fn a_valid_run_still_writes_the_receipt_its_consumer_reads() {
        // DISCRIMINATION. Not writing anything ever would satisfy the three
        // tests above; this one fails unless a good run still produces the
        // `runs[]` array `parity_block.py::_samples` indexes by `decode_tok_per_sec`.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("apr-cpu-c16.json");
        emit_report(&report(0, 42.0), Some(&path)).expect("a valid run earns its receipt");
        let text = std::fs::read_to_string(&path).expect("receipt must exist");
        let back: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert!(
            back["runs"][0]["decode_tok_per_sec"].is_number(),
            "the receipt must carry the field its consumer samples: {text}"
        );
    }

    #[test]
    fn without_an_output_path_the_verdict_is_still_the_return_value() {
        // The validity check is not a side effect of writing a file.
        emit_report(&report(0, 42.0), None).expect("a valid run is Ok");
        emit_report(&report(2, 42.0), None).expect_err("an invalid run is Err");
    }

    #[test]
    fn the_workload_line_names_its_source() {
        assert!(describe_workload("medium", None, 3).contains("profile medium"));
        let p = Path::new("/tmp/x.json");
        assert!(describe_workload("medium", Some(p), 7).contains("x.json"));
    }
}
