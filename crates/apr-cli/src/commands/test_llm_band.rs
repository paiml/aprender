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
//! only knob and it defaults to §4.4.2's `N = 3`.
//!
//! Going below `N` is written **into the receipt** as `replicates.below_spec`
//! and a `stated_violations` entry that `scripts/perf_gate.sh` reads. Until
//! PERF-048 this comment said "written into the receipt directory", the CLI's
//! own `--replicates` help said "stated on stdout", and the code did stdout:
//! `grep -ic replicate receipt.r1.json` returned `0`, so an `--replicates 1`
//! run was byte-indistinguishable from one replicate of a spec `N = 3` cell and
//! reported itself conformant (#2755). Two doc comments disagreeing with each
//! other and with the code is how that survived review.
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
use apr_test::perf_gate::protocol::{min_sampled_requests, BandConfig};
use apr_test::perf_gate::{
    sha256_file, write_samples_gz, BandInput, ComparatorStatus, ComputeClass, Provenance,
    ReceiptInput, Replicates, RequestOutcome, TokenizationBlock, TokenizationObservation, Workload,
    WorkloadCorpus, REPLICATES,
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
        // `BandInput.concurrency` is a `u32`. Refusing the excess here keeps
        // that conversion infallible, so the receipt cannot end up labelled
        // with a concurrency the operator did not ask for -- the same
        // silent-substitution defect as the clamped zero above.
        if u32::try_from(c).is_err() {
            return Err(CliError::InvalidInput(format!(
                "--bands: {c} exceeds the receipt's u32 concurrency field"
            )));
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

/// The prompt texts the harness will cycle, in issue order.
///
/// Every message's content, joined — not just the last user turn. Two corpora
/// differing only in their system prompt are two different workloads, and a
/// digest that could not tell them apart would be a label of the same kind
/// `--workload` already was.
fn prompt_texts(prompts: &[ChatRequest]) -> Vec<String> {
    prompts
        .iter()
        .map(|p| {
            p.messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect()
}

/// §4.3 — bind `--workload` to the prompts actually sent, and refuse the label
/// when the sent set cannot bear it (#2756).
///
/// The refusal happens **here**, before the health check and before the first
/// request, so a corpus that cannot carry the label costs nothing rather than
/// invalidating a 14-minute sweep after the fact.
///
/// Unlike the §4.4.6 downgrade, this refuses rather than states: a conformant
/// corpus is committed in-tree, so refusing points at the fix instead of
/// removing the measurement.
fn build_corpus(
    args: &BandArgs<'_>,
    workload: Workload,
    prompts: &[ChatRequest],
    levels: &[usize],
) -> Result<WorkloadCorpus> {
    let source = describe_workload(args.profile, args.prompts, prompts.len());
    let corpus = WorkloadCorpus::from_prompt_texts(&prompt_texts(prompts), source);
    let narrowest = levels
        .iter()
        .copied()
        .map(min_sampled_requests)
        .min()
        .unwrap_or_else(|| min_sampled_requests(1));
    if let Some(refusal) = corpus.label_refusal(workload, narrowest) {
        return Err(CliError::InvalidInput(refusal));
    }
    Ok(corpus)
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
        // Infallible: `parse_bands` refused anything a `u32` cannot hold, so
        // there is no clamp here to silently relabel the band.
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

/// Everything one replicate's receipt needs beyond the runs themselves.
///
/// A struct rather than seven positional arguments: this is the function that
/// decides what the receipt ASSERTS, and a transposed pair here is exactly the
/// class of defect PERF-048 exists to close.
struct ReplicateContext<'a> {
    tokenization: &'a TokenizationBlock,
    provenance: &'a Provenance,
    workload: Workload,
    corpus: &'a WorkloadCorpus,
    replicates: Replicates,
    /// Departures from §4.4 the runs stated, carried into the receipt instead
    /// of being printed and lost.
    stated_violations: Vec<String>,
}

/// §4.4.6 — one replicate's observation is the sum of its bands'.
///
/// Summed rather than taken from a "representative" band: a receipt covers
/// every band of one replicate, and a mixture in which one band's server
/// reported usage and another's did not is the fallback class, not the server
/// class. Summing is what makes that visible.
fn observe_replicate(runs: &[(usize, BandRun)]) -> TokenizationObservation {
    runs.iter().fold(
        TokenizationObservation {
            responses_with_server_usage: 0,
            responses_counted_by_client_tokenizer: 0,
            responses_counted: 0,
        },
        |acc, (_, run)| acc.merged(run.tokenization_observed),
    )
}

/// Write one replicate's receipt and its per-band sample files.
fn write_replicate(
    args: &BandArgs<'_>,
    ctx: &ReplicateContext<'_>,
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

    let mut stated_violations = ctx.stated_violations.clone();
    for (_, run) in runs {
        // §4.4 departures the run recorded. `BandRun::protocol_violations` was
        // computed on every run and read only by `report_run`'s println, so a
        // shrunken warmup or a SUSPECT drain left the receipt reading exactly
        // like a clean one — the same shape as #2755's replicate count.
        for v in &run.protocol_violations {
            if !stated_violations.contains(v) {
                stated_violations.push(v.clone());
            }
        }
    }
    let input = ReceiptInput {
        provenance: ctx.provenance.clone(),
        tokenization: ctx.tokenization.clone(),
        tokenization_observed: observe_replicate(runs),
        workload: ctx.workload,
        workload_corpus: ctx.corpus.clone(),
        replicates: Replicates {
            index: replicate + 1,
            ..ctx.replicates
        },
        stated_violations,
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

    // §4.3 — refuse a label the sent prompts cannot carry BEFORE the endpoint
    // is touched. `--workload W1 --profile short` (one prompt, sent 30 times)
    // used to produce a receipt saying `"workload": "W1"` (#2756).
    let corpus = build_corpus(&args, workload, &prompts, &levels)?;
    let widest = levels
        .iter()
        .copied()
        .map(min_sampled_requests)
        .max()
        .unwrap_or_else(|| min_sampled_requests(1));
    let mut stated_violations = Vec::new();
    if let Some(v) = corpus.repetition_violation(workload, widest) {
        stated_violations.push(v);
    }
    let replicates = Replicates {
        index: 1,
        effective: args.replicates,
        required: REPLICATES,
    };

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
    println!(
        "corpus   {} prompt(s), {} distinct, sha256 {}",
        corpus.prompts, corpus.distinct_prompts, corpus.sha256
    );
    // Every one of these goes into the receipt as well. Printing is for the
    // operator watching; the receipt is for the operator reading it a month
    // later, and #2755 is what happens when only the first exists.
    for v in replicates
        .violation()
        .into_iter()
        .chain(stated_violations.clone())
    {
        println!("!        {v}");
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
            &ReplicateContext {
                tokenization: &tokenization,
                provenance: &provenance,
                workload,
                corpus: &corpus,
                replicates,
                stated_violations: stated_violations.clone(),
            },
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

    /// A level a `u32` cannot hold must be refused where it was typed rather
    /// than clamped into a receipt that names a band nobody ran.
    #[test]
    fn a_band_wider_than_the_receipt_field_is_rejected() {
        let too_wide = u64::from(u32::MAX) + 1;
        let err = parse_bands(&format!("1,{too_wide}")).expect_err("must reject");
        assert!(err.to_string().contains("u32"), "{err}");
        assert!(parse_bands(&format!("{}", u32::MAX)).is_ok());
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

    // ----------------------------------------------------------- PERF-048 ---

    use apr_test::llm::client::{ChatMessage, Role};

    fn corpus_of(n: usize) -> Vec<ChatRequest> {
        (0..n)
            .map(|i| ChatRequest {
                model: "m".to_string(),
                messages: vec![ChatMessage {
                    role: Role::User,
                    content: format!("// w1-{i:04} body"),
                }],
                temperature: Some(0.0),
                max_tokens: Some(128),
                stream: Some(true),
                seed: Some(0),
                ignore_eos: Some(true),
            })
            .collect()
    }

    /// **THE PERF-045 INVOCATION**: `--workload W1 --profile short`, one prompt
    /// sent 30 times, recorded as W1 (#2756). It must be refused where it was
    /// typed — before the endpoint is touched, so no measurement is thrown
    /// away.
    ///
    /// RED for: dropping the `label_refusal` call from `build_corpus`.
    #[test]
    fn a_one_prompt_corpus_cannot_be_labelled_w1() {
        let a = args("1,4,8,16", "server_usage");
        let err = build_corpus(&a, Workload::W1, &corpus_of(1), &[1, 4, 8, 16])
            .expect_err("one prompt is not W1");
        let msg = err.to_string();
        assert!(msg.contains("prefix caching"), "{msg}");
        assert!(msg.contains("prompts-w1.jsonl"), "{msg}");
    }

    /// THE DISCRIMINATION CASE: the committed corpus's size is accepted, and
    /// the digest it produces is the one that goes in the receipt.
    #[test]
    fn a_committed_sized_corpus_is_accepted_and_digested() {
        let a = args("1,4,8,16", "server_usage");
        let c = build_corpus(&a, Workload::W1, &corpus_of(256), &[1, 4, 8, 16])
            .expect("256 distinct prompts carry the label");
        assert_eq!(c.prompts, 256);
        assert_eq!(c.distinct_prompts, 256);
        assert_eq!(c.sha256.len(), 64);
        // ...and it does not repeat inside the widest declared band.
        assert!(c.repetition_violation(Workload::W1, 128).is_none());
    }

    /// The floor tracks the NARROWEST band that will run, not a constant: a
    /// c=16-only sweep needs 128 distinct prompts, not 30.
    #[test]
    fn the_refusal_floor_follows_the_narrowest_band() {
        let a = args("16", "server_usage");
        assert!(build_corpus(&a, Workload::W1, &corpus_of(30), &[16]).is_err());
        assert!(build_corpus(&a, Workload::W1, &corpus_of(128), &[16]).is_ok());
        // The same 30-prompt set is fine when c=1 is the narrowest band.
        let a1 = args("1", "server_usage");
        assert!(build_corpus(&a1, Workload::W1, &corpus_of(30), &[1]).is_ok());
    }

    /// The digest covers every message, so two corpora differing only in a
    /// system turn are two corpora.
    #[test]
    fn the_digest_covers_every_message_not_just_the_last_turn() {
        let plain = corpus_of(30);
        let mut with_system = corpus_of(30);
        with_system[0].messages.insert(
            0,
            ChatMessage {
                role: Role::System,
                content: "You are terse.".to_string(),
            },
        );
        let a = args("1", "server_usage");
        let x = build_corpus(&a, Workload::W1, &plain, &[1]).expect("ok");
        let y = build_corpus(&a, Workload::W1, &with_system, &[1]).expect("ok");
        assert_ne!(x.sha256, y.sha256);
    }

    /// §4.4.6 — the replicate's observation is the SUM over its bands, so one
    /// band whose server went quiet downgrades the whole receipt rather than
    /// being outvoted.
    #[test]
    fn a_replicates_observation_sums_its_bands() {
        let merged = TokenizationObservation {
            responses_with_server_usage: 30,
            responses_counted_by_client_tokenizer: 0,
            responses_counted: 30,
        }
        .merged(TokenizationObservation {
            responses_with_server_usage: 0,
            responses_counted_by_client_tokenizer: 0,
            responses_counted: 32,
        });
        assert_eq!(merged.responses_counted, 62);
        assert!(!merged.every_response_carried_server_usage());
    }

    /// §4.4.2 — the effective N reaches the receipt whatever it is, and the
    /// index is per-replicate.
    #[test]
    fn the_effective_replicate_count_is_recorded() {
        let below = Replicates {
            index: 1,
            effective: 1,
            required: REPLICATES,
        };
        assert!(below.below_spec());
        assert!(below.violation().is_some());
        let spec = Replicates {
            index: 3,
            effective: REPLICATES,
            required: REPLICATES,
        };
        assert!(!spec.below_spec());
        assert!(spec.violation().is_none());
    }
}
