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

// ===========================================================================
// PERF-025 — `apr test llm bench --band`: the §4.4 protocol, reachable.
//
// PERF-024 landed the conformant measurement protocol (`run_band`, the §4.4.2
// termination rule, `drain_ms`, the §4.4.4 interval, §4.4.5 retention) and
// NOTHING CALLED IT outside its own tests. `apr test llm bench` still ran
// `LoadTest::run`, whose termination rule is "stop after `duration` seconds" —
// no minimum sample count, no warmup-then-quiesce, no drain accounting, no
// tokenization block. So the repo simultaneously contained a conformant
// protocol and could not produce a single conformant measurement.
//
// The other half of the same defect: `scripts/perf_gate.sh`'s real mode
// (`--host/--phase/--workload/--receipt`) had no caller, because nothing could
// write the receipt it reads. Both halves close here.
//
// THIS IS NOT A SECOND HARNESS. It is a mode of the existing `apr test llm
// bench` subcommand, driving the same `LlmClient` that `LoadTest` drives and
// that `scripts/parity_host_receipt.sh` points at both `apr serve` and
// `llama-server` (§4.4.8: one client, both servers, or the ratio is refused).
// PERF-009's one-entrypoint rule is preserved by construction.
//
// THERE IS DELIBERATELY NO FLAG THAT SHRINKS THE WINDOW. `BandConfig::relaxed`
// exists for unit tests and is not reachable from the CLI. A `--band-duration`
// would be the single easiest way to turn this gate back into one that cannot
// fail, and the 60 s floor is the whole difference between a load test and a
// measurement. `--replicates` is the one knob, it defaults to the spec's N=3,
// and anything below N=3 is written into the receipt as a protocol violation
// rather than being silently accepted.
use crate::LlmSubcommand;
use apr_test::llm::band::run_band;
use apr_test::llm::client::LlmClient;
use apr_test::perf_gate::protocol::{BandConfig, Tokenization};
use apr_test::perf_gate::receipt::{
    sha256_file, write_receipt, Provenance, ReceiptMeta, Replicate,
};
use apr_test::perf_gate::REPLICATES;

/// Arguments for one §4.4-conformant band run.
pub struct BandArgs<'a> {
    /// Endpoint under measurement.
    pub url: &'a str,
    /// Model name sent in the request body.
    pub model: &'a str,
    /// Concurrency levels, comma-separated, e.g. `1,4,8,16`.
    pub bands: &'a str,
    /// §4.4.2 replicates per cell.
    pub replicates: usize,
    /// Directory receiving `receipt.json` and the gzipped JSONL samples.
    pub receipt: &'a Path,
    /// §4.3 workload identifier.
    pub workload: &'a str,
    /// Join key: which machine.
    pub host: &'a str,
    /// Join key: which accelerator.
    pub accelerator: &'a str,
    /// Join key: which quantization.
    pub quantization: &'a str,
    /// The dispatch path the SERVER took.
    pub compute_class: &'a str,
    /// The SERVER's build features, if known.
    pub server_features: &'a [String],
    /// §4.4.6 counting method.
    pub tokenization: &'a str,
    /// §4.4.6 digest, required for `client_tokenizer`.
    pub tokenizer_sha256: Option<&'a str>,
    /// §4.4.6.
    pub counts_special_tokens: bool,
    /// §4.4.6.
    pub counts_prompt_echo: bool,
    /// Commit under measurement.
    pub commit: Option<&'a str>,
    /// Streaming responses. Required for TTFT, ITL and `decode_tok_s`.
    pub stream: bool,
    /// Named prompt profile.
    pub profile: &'a str,
    /// Prompt file, overriding the profile.
    pub prompts: Option<&'a Path>,
}

/// `1,4,8,16` into concurrency levels.
///
/// A zero or unparseable level is rejected where it was typed. `BandConfig`
/// clamps `0` up to `1`, which would silently measure a different band than the
/// operator asked for and label it with the number they typed.
fn parse_bands(spec: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for raw in spec.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let c: usize = token.parse().map_err(|_| {
            CliError::InvalidInput(format!("--bands: {token:?} is not a concurrency level"))
        })?;
        if c == 0 {
            return Err(CliError::InvalidInput(
                "--bands: 0 is not a concurrency level; a band with no workers measures nothing"
                    .to_string(),
            ));
        }
        out.push(c);
    }
    if out.is_empty() {
        return Err(CliError::InvalidInput(
            "--bands named no concurrency levels".to_string(),
        ));
    }
    Ok(out)
}

/// §4.4.6 — `method` has no default, so an unknown value is refused rather
/// than falling back to one of the two.
fn build_tokenization(args: &BandArgs<'_>) -> Result<Tokenization> {
    match args.tokenization {
        "server_usage" => Ok(Tokenization::server_usage(
            args.counts_special_tokens,
            args.counts_prompt_echo,
        )),
        "client_tokenizer" => {
            let sha = args.tokenizer_sha256.ok_or_else(|| {
                CliError::InvalidInput(
                    "--tokenization client_tokenizer requires --tokenizer-sha256".to_string(),
                )
            })?;
            Tokenization::client_tokenizer(sha, args.counts_special_tokens, args.counts_prompt_echo)
                .map_err(CliError::InvalidInput)
        }
        other => Err(CliError::InvalidInput(format!(
            "--tokenization {other:?}: expected server_usage or client_tokenizer (§4.4.6 gives \
             `method` no default)"
        ))),
    }
}

/// Which binary is measuring, and what it hashes to.
///
/// `current_exe` rather than a `$PATH` lookup or a hardcoded path: four `apr`
/// binaries have coexisted on the dev box, and a bare `apr` once resolved to a
/// 26-day-old copy. The binary that writes the receipt is the binary the
/// receipt names, by construction.
fn build_provenance(args: &BandArgs<'_>) -> Result<Provenance> {
    let exe = std::env::current_exe()
        .map_err(|e| CliError::InvalidInput(format!("cannot resolve current_exe: {e}")))?;
    let binary_sha256 = sha256_file(&exe)
        .map_err(|e| CliError::InvalidInput(format!("cannot hash {}: {e}", exe.display())))?;
    let prov = Provenance {
        binary_path: exe.display().to_string(),
        binary_sha256,
        resolution: "current_exe".to_string(),
        compute_class: args.compute_class.to_string(),
        host: args.host.to_string(),
        accelerator: args.accelerator.to_string(),
        model: args.model.to_string(),
        quantization: args.quantization.to_string(),
        feature_set: if args.server_features.is_empty() {
            None
        } else {
            Some(args.server_features.to_vec())
        },
    };
    prov.validate().map_err(CliError::InvalidInput)?;
    Ok(prov)
}

/// Run every replicate of one band.
async fn run_cell_replicates(
    client: &LlmClient,
    prompts: &[ChatRequest],
    concurrency: usize,
    args: &BandArgs<'_>,
    tokenization: &Tokenization,
) -> Result<Vec<Replicate>> {
    let band = BandConfig::conformant(concurrency);
    let mut out = Vec::with_capacity(args.replicates);
    for k in 0..args.replicates {
        println!("  c={concurrency} replicate {}/{}", k + 1, args.replicates);
        let run = run_band(client, prompts, &band, tokenization.clone(), args.stream)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("band c={concurrency}: {e}")))?;
        println!(
            "    agg {:.2} tok/s  decode {:.2} tok/s  requested {}  completed {}  \
             timeouts {}  drain {:.1} ms  peak_in_flight {}",
            run.metrics.agg_tok_s,
            run.metrics.decode_tok_s,
            run.metrics.requested,
            run.metrics.completed,
            run.metrics.timeouts,
            run.window.drain_ms,
            run.window.client_peak_in_flight
        );
        out.push(run.into_replicate());
    }
    Ok(out)
}

/// Run the §4.4 protocol over every requested band and write the receipt.
///
/// # Errors
/// When the endpoint is unreachable, any band fails, provenance or the §4.4.6
/// block does not validate, the receipt cannot be written, or the finished
/// receipt could not pass `perf_gate.sh`'s Arm C.
pub async fn run_bands(args: BandArgs<'_>) -> Result<()> {
    let levels = parse_bands(args.bands)?;
    let tokenization = build_tokenization(&args)?;
    let provenance = build_provenance(&args)?;
    let prompts = resolve_prompts(args.profile, args.prompts)?;
    let prompts: Vec<ChatRequest> = prompts
        .into_iter()
        .map(|mut p| {
            p.model = args.model.to_string();
            p
        })
        .collect();

    let client = LlmClient::new(args.url, args.model);
    client.health_check().await.map_err(|e| {
        CliError::InferenceFailed(format!("endpoint {} is not ready: {e}", args.url))
    })?;

    println!("protocol APR-PERF-GATE-001 v2.2 §4.4 (closed-loop, conformant)");
    println!("endpoint {}", args.url);
    println!(
        "workload {}",
        describe_workload(args.profile, args.prompts, prompts.len())
    );
    println!(
        "bands    {levels:?} x {} replicate(s); warmup 2c, quiesce 5s, \
         min max(30, 8c) samples AND min 60s wall-clock, last bound wins",
        args.replicates
    );
    if !args.stream {
        println!(
            "NOTE     --stream is off: §4.4.3 ttft_ms, itl_ms and decode_tok_s are UNDEFINED \
             without per-token arrival times and will read 0"
        );
    }

    let mut cells = Vec::with_capacity(levels.len());
    for c in levels {
        cells.push((
            c,
            run_cell_replicates(&client, &prompts, c, &args, &tokenization).await?,
        ));
    }

    let meta = ReceiptMeta {
        workload: args.workload.to_string(),
        provenance,
        tokenization,
        replicates: args.replicates,
        commit: args.commit.map(str::to_string),
    };
    let (receipt, written) = write_receipt(args.receipt, &meta, &cells)
        .map_err(|e| CliError::InvalidFormat(format!("writing receipt: {e}")))?;

    println!(
        "\nreceipt  {} ({} bytes)",
        written.receipt.display(),
        written.bytes
    );
    for path in &written.sample_files {
        println!("samples  {}", path.display());
    }
    println!(
        "conformant {} ({} protocol violation(s))",
        receipt.conformant,
        receipt.protocol_violations.len()
    );
    for v in &receipt.protocol_violations {
        println!("  ! {v}");
    }
    println!(
        "\nnext     scripts/perf_gate.sh --host {} --phase merge --workload {} --receipt {}",
        args.host,
        args.workload,
        written.receipt.display()
    );

    // A run whose own counts cannot pass Arm C must not exit 0. Arm C is
    // re-run by the gate; checking it here means the operator learns it now
    // rather than after a CI round-trip, and it makes `rc = 0` mean something.
    if receipt.arm_c_would_pass() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "receipt cannot pass perf_gate.sh Arm C: requested={} completed={} timeouts={} — a \
             throughput averaged over the requests that survived is not a measurement of this \
             runtime, it is a measurement of its survivors",
            receipt.requested, receipt.completed, receipt.timeouts
        )))
    }
}

/// Route `apr test llm <SUB>` (GH-876 Milestone 2; PERF-025 band mode).
///
/// The arm lives HERE rather than inline in `dispatch_analysis.rs`'s match.
/// That file's router was already at cognitive 24 against a threshold of 25, so
/// adding the band branch inline tipped it over and the pre-commit gate refused
/// the commit. Moving the whole arm out puts the routing next to the two
/// functions it routes to and leaves the router simpler than it was.
///
/// # Errors
/// Propagates whichever mode ran.
pub fn dispatch(command: &LlmSubcommand) -> Result<()> {
    match command {
        LlmSubcommand::Bench {
            url,
            model,
            start,
            health_timeout,
            warmup,
            duration,
            concurrency,
            runs,
            cooldown,
            runtime_name,
            baseline,
            fail_on_regression,
            output,
            stream,
            profile,
            prompts,
            band,
            receipt,
            bands,
            replicates,
            workload,
            host,
            accelerator,
            quantization,
            compute_class,
            server_features,
            tokenization,
            tokenizer_sha256,
            counts_special_tokens,
            counts_prompt_echo,
            commit,
        } => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| CliError::InferenceFailed(format!("tokio runtime: {e}")))?;
            // TWO MODES, ONE ENTRYPOINT. `--band` selects the §4.4-conformant
            // protocol (PERF-024's `run_band`, previously called by nothing
            // outside its own tests). Without it the pre-existing
            // `LoadTest::run` lifecycle is unchanged, because a great deal of
            // tooling reads its `LoadTestResult` and changing its termination
            // rule underneath those readers would silently change every number
            // they have ever recorded.
            if *band {
                let receipt = receipt.as_deref().ok_or_else(|| {
                    // Unreachable: clap's `requires = "receipt"` enforces it.
                    // Stated rather than unwrapped — a receipt-less band run
                    // would measure for minutes and discard the measurement.
                    CliError::InvalidInput("--band requires --receipt <DIR>".to_string())
                })?;
                return rt.block_on(run_bands(BandArgs {
                    url,
                    model,
                    bands,
                    replicates: *replicates,
                    receipt,
                    workload,
                    host,
                    accelerator,
                    quantization,
                    compute_class,
                    server_features,
                    tokenization,
                    tokenizer_sha256: tokenizer_sha256.as_deref(),
                    counts_special_tokens: *counts_special_tokens,
                    counts_prompt_echo: *counts_prompt_echo,
                    commit: commit.as_deref(),
                    stream: *stream,
                    profile,
                    prompts: prompts.as_deref(),
                }));
            }
            rt.block_on(run_bench(BenchArgs {
                url,
                model,
                start: start.as_deref(),
                health_timeout: *health_timeout,
                warmup: *warmup,
                duration: *duration,
                concurrency: *concurrency,
                runs: *runs,
                cooldown: *cooldown,
                runtime_name,
                baseline: baseline.as_deref(),
                fail_on_regression: *fail_on_regression,
                output: output.as_deref(),
                stream: *stream,
                profile,
                prompts: prompts.as_deref(),
            }))
        }
    }
}

/// §4.4.2's `N`, re-exported so the CLI default and the spec constant cannot
/// drift apart.
#[must_use]
pub const fn default_replicates() -> usize {
    REPLICATES
}

#[cfg(test)]
mod band_tests {
    use super::*;

    fn args<'a>(bands: &'a str, tokenization: &'a str) -> BandArgs<'a> {
        BandArgs {
            url: "http://127.0.0.1:8080",
            model: "m",
            bands,
            replicates: 3,
            receipt: Path::new("/tmp/receipt"),
            workload: "W1",
            host: "lambda",
            accelerator: "rtx-4090",
            quantization: "Q4_K_M",
            compute_class: "cpu",
            server_features: &[],
            tokenization,
            tokenizer_sha256: None,
            counts_special_tokens: true,
            counts_prompt_echo: false,
            commit: None,
            stream: true,
            profile: "medium",
            prompts: None,
        }
    }

    #[test]
    fn bands_parse_in_the_order_given() {
        assert_eq!(parse_bands("1,4,8,16").expect("valid"), vec![1, 4, 8, 16]);
        assert_eq!(parse_bands(" 2 , 3 ").expect("spaces"), vec![2, 3]);
    }

    /// `BandConfig::conformant` clamps 0 up to 1, so a zero that reaches it
    /// measures c=1 while the receipt says the operator asked for 0.
    #[test]
    fn a_zero_band_is_rejected_rather_than_clamped() {
        let err = parse_bands("1,0,4").expect_err("must reject");
        assert!(err.to_string().contains("measures nothing"), "{err}");
    }

    #[test]
    fn an_unparseable_band_is_rejected() {
        assert!(parse_bands("1,four").is_err());
        assert!(parse_bands("").is_err());
        assert!(parse_bands(",,").is_err());
    }

    /// §4.4.6 gives `method` no default; an unknown value must not fall back.
    #[test]
    fn an_unknown_tokenization_method_is_refused() {
        let err = build_tokenization(&args("1", "guess")).expect_err("must reject");
        let msg = err.to_string();
        assert!(msg.contains("server_usage"), "{msg}");
        assert!(msg.contains("client_tokenizer"), "{msg}");
    }

    #[test]
    fn client_tokenizer_without_a_digest_is_refused() {
        let err =
            build_tokenization(&args("1", "client_tokenizer")).expect_err("digest is required");
        assert!(err.to_string().contains("tokenizer-sha256"), "{err}");
    }

    #[test]
    fn server_usage_needs_no_digest() {
        assert!(build_tokenization(&args("1", "server_usage")).is_ok());
    }

    /// The provenance the CLI builds must be one `bench_receipt.py` accepts:
    /// a real 64-hex digest of the binary that is actually running.
    #[test]
    fn provenance_hashes_the_running_binary() {
        let prov = build_provenance(&args("1", "server_usage")).expect("current_exe must hash");
        assert_eq!(prov.binary_sha256.len(), 64);
        assert_eq!(prov.resolution, "current_exe");
        assert!(prov.validate().is_ok());
    }

    #[test]
    fn an_unknown_compute_class_is_refused_where_it_was_typed() {
        let mut a = args("1", "server_usage");
        a.compute_class = "tpu";
        assert!(build_provenance(&a).is_err());
    }

    /// The CLI default must not drift from §4.4.2's N.
    #[test]
    fn the_replicate_default_is_the_spec_constant() {
        assert_eq!(default_replicates(), 3);
    }
}
