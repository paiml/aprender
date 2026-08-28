//! `apr test llm bench --band` — PERF-025: the §4.4 protocol, reachable.
//!
//! # The defect this closes
//!
//! PERF-024 landed the conformant measurement protocol — `llm::band::run_band`,
//! the §4.4.2 warmup/quiesce/termination rule, `perf_gate::window`'s drain
//! accounting, the §4.4.4 bootstrap — and PERF-026 landed the receipt producer
//! (`perf_gate::drain::BandInput`, `perf_gate::receipt::ReceiptInput`).
//! **Nothing called either of them.** On `50d2bc2bb`,
//! `git grep "ReceiptInput\|BandInput\|DerivedBand\|perf_gate::"` outside the
//! module returned rc=1: zero callers. `apr test llm bench` still ran
//! `LoadTest::run`, whose termination rule is "stop after `--duration`
//! seconds" — no minimum sample count, no warmup-then-quiesce, no drain
//! accounting, no §4.4.6 tokenization block.
//!
//! The other half of the same defect: `scripts/perf_gate.sh`'s real mode
//! (`--host/--phase/--workload/--receipt`) was invoked only as `--selftest`
//! (`ci.yml:1011`), because nothing in the repo could write the receipt its
//! real mode reads. So the repo simultaneously held a conformant protocol, a
//! conformant receipt producer and a gate — and could not produce one
//! conformant measurement. Both halves close here.
//!
//! # This is a MODE, not a second harness
//!
//! `scripts/check_no_competing_harnesses.sh` names `apr test llm bench` the
//! canonical entrypoint, at `BASELINE=0`, shrink-only. This extends that
//! subcommand and drives the same `LlmClient` the legacy mode drives, so
//! PERF-009's one-entrypoint rule holds by construction rather than by promise.
//!
//! # There is deliberately no flag that shrinks the window
//!
//! `BandConfig::relaxed` exists for unit tests and is unreachable from the CLI.
//! The `max(30, 8c)` sample floor and the 60 s wall-clock floor are the entire
//! difference between a load test and a measurement; a `--band-duration` would
//! be the shortest path back to a gate that cannot fail. `--replicates` is the
//! only knob, it defaults to §4.4.2's `N = 3`, and going below that is written
//! into the receipt directory as a stated violation rather than silently taken.
//!
//! # One receipt per replicate, and why
//!
//! `ReceiptInput.bands` holds **one entry per concurrency**, and
//! `perf_gate.sh`'s Arm A indexes it with `{b["concurrency"]: b for b in
//! bands}` — a Python dict comprehension. Emitting `N` bands that all say
//! `concurrency: 1` would therefore be silently reduced to whichever one landed
//! last: a receipt that looks like three measurements and is judged as one.
//! Collapsing the replicates into a "representative" band instead would mean
//! picking a median of three, which is exactly the estimator that reported
//! 2.91x for a cell whose real answer was 1.21x (#2567).
//!
//! So each replicate gets its own complete, independently judgeable receipt,
//! `receipt.r1.json` … `receipt.rN.json`. The cell's verdict is the conjunction
//! over them, which is a decision for the runner, not something this producer
//! should pre-collapse.

use crate::error::{CliError, Result};
use apr_test::llm::band::{run_band, BandRun};
use apr_test::llm::client::{ChatRequest, LlmClient};
use apr_test::perf_gate::protocol::BandConfig;
use apr_test::perf_gate::{
    sha256_file, write_samples_gz, BandInput, ComparatorStatus, ComputeClass, Provenance,
    ReceiptInput, RequestOutcome, TokenizationBlock, Workload, REPLICATES,
};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use super::test_llm::{describe_workload, resolve_prompts};

/// Arguments for one §4.4-conformant band sweep.
///
/// A struct rather than a 20-argument function, for the reason `BenchArgs` is
/// one: positional arguments of the same type are where a caller transposes
/// `accelerator` and `quantization` and the receipt records a lie.
pub struct BandArgs<'a> {
    /// Endpoint under measurement.
    pub url: &'a str,
    /// Model name sent in the request body.
    pub model: &'a str,
    /// Concurrency levels, comma-separated, e.g. `1,4,8,16`.
    pub bands: &'a str,
    /// §4.4.2 replicates per cell.
    pub replicates: usize,
    /// Directory receiving the receipts and the gzipped JSONL samples.
    pub receipt: &'a Path,
    /// §4.3 workload identifier.
    pub workload: &'a str,
    /// Join key: which machine measured.
    pub host: &'a str,
    /// Join key: which accelerator served the request.
    pub accelerator: &'a str,
    /// Join key: which quantization the served model uses.
    pub quantization: &'a str,
    /// The dispatch path the SERVER took.
    pub compute_class: &'a str,
    /// The SERVER's build features.
    pub server_features: &'a [String],
    /// §4.4.6 counting method.
    pub tokenization: &'a str,
    /// §4.4.6 digest, required for `client_tokenizer`.
    pub tokenizer_sha256: Option<&'a str>,
    /// §4.4.6 — do the counts include special tokens?
    pub counts_special_tokens: bool,
    /// §4.4.6 — do the counts include the echoed prompt?
    pub counts_prompt_echo: bool,
    /// Commit under measurement.
    pub commit: Option<&'a str>,
    /// Streaming responses. Required for TTFT, ITL and `decode_tok_s`.
    pub stream: bool,
    /// Named prompt profile.
    pub profile: &'a str,
    /// Prompt file, overriding the profile.
    pub prompts: Option<&'a Path>,
    /// Who owes the comparator measurement this producer refuses to invent.
    pub comparator_owner: &'a str,
}

/// `1,4,8,16` into concurrency levels.
///
/// A zero or unparseable level is refused where it was typed.
/// `BandConfig::conformant` clamps `0` up to `1`, which would measure a
/// different band than the operator asked for and then label it with the number
/// they typed.
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

/// §4.4.6 — `method` has no default, so an unknown value is refused rather than
/// falling back to one of the two.
fn build_tokenization(args: &BandArgs<'_>) -> Result<TokenizationBlock> {
    let block = match args.tokenization {
        "server_usage" => TokenizationBlock::ServerUsage {
            counts_special_tokens: args.counts_special_tokens,
            counts_prompt_echo: args.counts_prompt_echo,
        },
        "client_tokenizer" => {
            let sha = args.tokenizer_sha256.ok_or_else(|| {
                CliError::InvalidInput(
                    "--tokenization client_tokenizer requires --tokenizer-sha256".to_string(),
                )
            })?;
            TokenizationBlock::ClientTokenizer {
                tokenizer_sha256: sha.to_string(),
                counts_special_tokens: args.counts_special_tokens,
                counts_prompt_echo: args.counts_prompt_echo,
            }
        }
        other => {
            return Err(CliError::InvalidInput(format!(
                "--tokenization {other:?}: expected server_usage or client_tokenizer (§4.4.6 \
                 gives `method` no default)"
            )))
        }
    };
    block.validate().map_err(CliError::InvalidInput)?;
    Ok(block)
}

/// Which binary is measuring, and what it hashes to.
///
/// `current_exe` rather than a `$PATH` lookup or a hardcoded path: four `apr`
/// binaries have coexisted on the dev box and a bare `apr` once resolved to a
/// 26-day-old copy. The binary that writes the receipt is the binary the
/// receipt names, by construction rather than by discipline.
///
/// The features recorded are the **server's**, never this client's own
/// `cfg!(feature = ...)`. `bench_receipt.py` uses `feature_set` to refuse a
/// `compute_class` the build cannot reach; pointing that check at the measuring
/// binary instead of the measured one would make it read green while checking
/// nothing.
fn build_provenance(args: &BandArgs<'_>) -> Result<Provenance> {
    let exe = std::env::current_exe()
        .map_err(|e| CliError::InvalidInput(format!("cannot resolve current_exe: {e}")))?;
    let binary_sha256 = sha256_file(&exe)
        .map_err(|e| CliError::InvalidInput(format!("cannot hash {}: {e}", exe.display())))?;
    let compute_class = ComputeClass::from_str(args.compute_class)
        .map_err(|e| CliError::InvalidInput(format!("--compute-class: {e}")))?;
    let prov = Provenance {
        binary_path: exe.display().to_string(),
        binary_sha256,
        resolution: "current_exe".to_string(),
        compute_class,
        host: args.host.to_string(),
        accelerator: args.accelerator.to_string(),
        model: args.model.to_string(),
        quantization: args.quantization.to_string(),
        feature_set: args.server_features.to_vec(),
    };
    prov.validate().map_err(CliError::InvalidInput)?;
    Ok(prov)
}

/// §4.7.1 — this cell's comparator posture.
///
/// There is no CLI path to a measured ratio, and that is the point. A ratio
/// needs a baseline object that itself passes every receipt rule (I-3) and a
/// comparator lane driven by the same client binary (I-15). This producer has
/// neither, so it emits `UNMEASURED` with an owner and Arm B reports and skips.
/// An `agg_ratio` synthesised here is precisely the fabrication the epic exists
/// to remove.
fn comparator_status(args: &BandArgs<'_>) -> ComparatorStatus {
    ComparatorStatus::Unmeasured {
        owner: args.comparator_owner.to_string(),
        reason: "APR-PERF-GATE-001 I-3/I-15: a comparator ratio needs a baseline receipt and a \
                 comparator lane driven by this same client binary. `apr test llm bench --band` \
                 measures one lane, so it declares the ratio unmeasured rather than deriving one."
            .to_string(),
    }
}

/// One `run_band` result as the receipt producer's input.
///
/// `RequestSample` records offsets from the band origin in **seconds**;
/// `RequestOutcome` records them in **milliseconds**. The conversion is here,
/// once, rather than at each field's use.
///
/// `ttft_ms` is a duration from issue to first token, while `token_times_ms`
/// are absolute offsets from the band origin — the two differ by `issued_ms`,
/// and `RequestOutcome::itl_gaps_ms` differences the latter, so mixing the
/// conventions would leave the gaps correct and TTFT wrong by the request's
/// start offset.
fn band_input(run: &BandRun, comparator: ComparatorStatus) -> BandInput {
    let requests = run
        .samples
        .iter()
        .map(|s| RequestOutcome {
            issued_ms: s.start_s * 1000.0,
            settled_ms: s.end_s * 1000.0,
            outcome: s.outcome,
            generated_tokens: s.generated_tokens,
            ttft_ms: s.token_times_s.first().map(|t| (t - s.start_s) * 1000.0),
            token_times_ms: s.token_times_s.iter().map(|t| t * 1000.0).collect(),
        })
        .collect();
    BandInput {
        concurrency: u32::try_from(run.config.concurrency).unwrap_or(u32::MAX),
        window_ms: run.window.window_ms,
        requests,
        comparator,
    }
}

/// Print one replicate's headline numbers as it finishes.
fn report_run(concurrency: usize, k: usize, replicates: usize, run: &BandRun) {
    println!("  c={concurrency} replicate {}/{replicates}", k + 1);
    println!(
        "    agg {:.2} tok/s  decode {:.2} tok/s  requested {}  completed {}  timeouts {}  \
         drain {:.1} ms  peak_in_flight {}",
        run.metrics.agg_tok_s,
        run.metrics.decode_tok_s,
        run.metrics.requested,
        run.metrics.completed,
        run.metrics.timeouts,
        run.window.drain_ms,
        run.window.client_peak_in_flight
    );
    for v in &run.protocol_violations {
        println!("    ! {v}");
    }
}

/// Write one replicate's receipt and its per-band sample files.
fn write_replicate(
    args: &BandArgs<'_>,
    tokenization: &TokenizationBlock,
    provenance: &Provenance,
    replicate: usize,
    runs: &[(usize, BandRun)],
) -> Result<PathBuf> {
    for (c, run) in runs {
        let path = args
            .receipt
            .join(format!("samples.c{c}.r{}.jsonl.gz", replicate + 1));
        let file = write_samples_gz(&path, &run.samples)
            .map_err(|e| CliError::InvalidFormat(format!("writing {}: {e}", path.display())))?;
        println!(
            "samples  {} ({} rows, {} bytes)",
            file.path.display(),
            file.rows,
            file.bytes
        );
    }

    let workload = Workload::from_str(args.workload)
        .map_err(|e| CliError::InvalidInput(format!("--workload: {e}")))?;
    let input = ReceiptInput {
        provenance: provenance.clone(),
        tokenization: tokenization.clone(),
        workload,
        commit: args.commit.unwrap_or("UNPINNED").to_string(),
        bands: runs
            .iter()
            .map(|(_, run)| band_input(run, comparator_status(args)))
            .collect(),
        kv: None,
    };
    let rendered = input
        .render_string()
        .map_err(|e| CliError::ValidationFailed(format!("receipt r{}: {e}", replicate + 1)))?;

    let path = args
        .receipt
        .join(format!("receipt.r{}.json", replicate + 1));
    std::fs::write(&path, rendered.as_bytes())
        .map_err(|e| CliError::InvalidFormat(format!("writing {}: {e}", path.display())))?;
    println!("receipt  {} ({} bytes)", path.display(), rendered.len());
    Ok(path)
}

/// Run the §4.4 protocol over every requested band and write the receipts.
///
/// # Errors
/// When the endpoint is unreachable, any band fails, provenance or the §4.4.6
/// block does not validate, a receipt cannot be rendered (which is where a
/// zero-token band or an I-14 violation surfaces), or a receipt cannot be
/// written.
pub async fn run_bands(args: BandArgs<'_>) -> Result<()> {
    let levels = parse_bands(args.bands)?;
    if args.replicates == 0 {
        return Err(CliError::InvalidInput(
            "--replicates 0 measures nothing".to_string(),
        ));
    }
    // Everything that can be refused without spending a measurement is refused
    // before the first request. A 14-minute sweep that fails at receipt-write
    // time because `--host` was blank has thrown away the measurement.
    let tokenization = build_tokenization(&args)?;
    let provenance = build_provenance(&args)?;
    let workload = Workload::from_str(args.workload)
        .map_err(|e| CliError::InvalidInput(format!("--workload: {e}")))?;
    std::fs::create_dir_all(args.receipt).map_err(|e| {
        CliError::InvalidFormat(format!("creating {}: {e}", args.receipt.display()))
    })?;

    let prompts: Vec<ChatRequest> = resolve_prompts(args.profile, args.prompts)?
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
    println!("workload {}", workload.wire_token());
    println!(
        "prompts  {}",
        describe_workload(args.profile, args.prompts, prompts.len())
    );
    println!(
        "bands    {levels:?} x {} replicate(s); warmup 2c, quiesce 5s, window closes when BOTH \
         max(30, 8c) samples AND 60s wall-clock are met",
        args.replicates
    );
    if args.replicates < REPLICATES {
        println!(
            "!        --replicates {} is below §4.4.2's N={REPLICATES}; the cell is \
             under-replicated and its bootstrap CI is correspondingly weak",
            args.replicates
        );
    }
    if !args.stream {
        println!(
            "NOTE     --stream is off: §4.4.3 ttft_ms, itl_ms and decode_tok_s are UNDEFINED \
             without per-token arrival times and are omitted from the receipt"
        );
    }

    let mut written = Vec::new();
    for k in 0..args.replicates {
        let mut runs = Vec::with_capacity(levels.len());
        for &c in &levels {
            let band = BandConfig::conformant(c);
            let run = run_band(&client, &prompts, &band, tokenization.clone(), args.stream)
                .await
                .map_err(|e| {
                    CliError::InferenceFailed(format!("band c={c} replicate {}: {e}", k + 1))
                })?;
            report_run(c, k, args.replicates, &run);
            runs.push((c, run));
        }
        written.push(write_replicate(
            &args,
            &tokenization,
            &provenance,
            k,
            &runs,
        )?);
    }

    println!("\nnext");
    for path in &written {
        println!(
            "  scripts/perf_gate.sh --host {} --phase merge --workload {} --receipt {}",
            args.host,
            workload.wire_token(),
            path.display()
        );
    }
    Ok(())
}

/// §4.4.2's `N`, re-exported so the CLI default and the spec constant cannot
/// drift apart.
#[must_use]
pub const fn default_replicates() -> usize {
    REPLICATES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args<'a>(bands: &'a str, tokenization: &'a str) -> BandArgs<'a> {
        BandArgs {
            url: "http://127.0.0.1:8080",
            model: "qwen2.5-coder-1.5b-instruct",
            bands,
            replicates: 3,
            receipt: Path::new("/tmp/perf-025-receipt"),
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
            comparator_owner: "perf-gate",
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

    /// The provenance the CLI builds must be one `bench_receipt.py` accepts: a
    /// real 64-hex digest of the binary that is actually running.
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
        let err = build_provenance(&a).expect_err("must reject");
        assert!(err.to_string().contains("compute-class"), "{err}");
    }

    /// I-2: declaring `cuda` without the server having been built with it is a
    /// claim about a path the build cannot take.
    #[test]
    fn cuda_without_the_server_feature_is_refused() {
        let mut a = args("1", "server_usage");
        a.compute_class = "cuda";
        assert!(build_provenance(&a).is_err());

        let features = vec!["cuda".to_string()];
        let mut ok = args("1", "server_usage");
        ok.compute_class = "cuda";
        ok.server_features = &features;
        assert!(build_provenance(&ok).is_ok());
    }

    /// An empty join key must be refused before the sweep, not after it.
    #[test]
    fn a_blank_join_key_is_refused() {
        let mut a = args("1", "server_usage");
        a.host = "";
        let err = build_provenance(&a).expect_err("host is required");
        assert!(err.to_string().contains("host"), "{err}");
    }

    /// The producer must never be able to emit a comparator ratio.
    #[test]
    fn the_comparator_is_always_unmeasured_with_an_owner() {
        let status = comparator_status(&args("1", "server_usage"));
        assert_eq!(status.wire_token(), "UNMEASURED");
        match status {
            ComparatorStatus::Unmeasured { owner, .. } => assert_eq!(owner, "perf-gate"),
            ComparatorStatus::NotApplicable { .. } => panic!("must not be permanent"),
        }
    }

    /// The CLI default must not drift from §4.4.2's N.
    #[test]
    fn the_replicate_default_is_the_spec_constant() {
        assert_eq!(default_replicates(), 3);
    }

    #[test]
    fn an_unknown_workload_is_refused() {
        assert!(Workload::from_str("W3").is_err());
    }
}
