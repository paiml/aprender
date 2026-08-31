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
use apr_test::llm::{assert_prompt_tokens_in_band, PromptTokenBand};
use apr_test::perf_gate::protocol::{BandConfig, Outcome};
use apr_test::perf_gate::{
    sha256_file, write_samples_gz, BandInput, ClientTokenizer, ComparatorStatus, ComputeClass,
    Provenance, ReceiptInput, RequestOutcome, RequestSample, TokenAccounting, TokenizationBlock,
    Workload, REPLICATES,
};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use super::test_llm::{describe_workload, resolve_corpus};

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
    /// The model's `tokenizer.json`, required for `client_tokenizer`. The
    /// receipt's digest is COMPUTED from this file.
    pub tokenizer: Option<&'a Path>,
    /// §4.4.6 digest, ASSERTED against the digest of `tokenizer`.
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
///
/// # `client_tokenizer` now opens a file
///
/// It used to take `--tokenizer-sha256` and put whatever 64 hex characters the
/// operator typed straight into the receipt. Nothing computed that digest,
/// nothing counted a token client-side, and `perf_gate.sh:79` checks only that
/// `tokenization.method` is a non-empty string — so a `client_tokenizer` receipt
/// was a declaration with no referent, while the counts inside it were still the
/// server's own `usage` block. Measured, that block differs between `apr serve`
/// (513) and `llama-server` (534) on identical W1 prompts whose wire text is
/// 505 tokens — the corpus has since been retuned to 512, which moves all three
/// numbers together and leaves the disagreement exactly where it was.
///
/// The digest is now computed from `--tokenizer`, and `--tokenizer-sha256` is
/// demoted to an assertion about it.
fn build_accounting(args: &BandArgs<'_>) -> Result<TokenAccounting> {
    match args.tokenization {
        "server_usage" => {
            if let Some(path) = args.tokenizer {
                return Err(CliError::InvalidInput(format!(
                    "--tokenizer {} was supplied with --tokenization server_usage: the counts \
                     would come from the server anyway, so the tokenizer would be recorded and \
                     never used. Declare client_tokenizer or drop --tokenizer.",
                    path.display()
                )));
            }
            Ok(TokenAccounting::server_usage(
                args.counts_special_tokens,
                args.counts_prompt_echo,
            ))
        }
        "client_tokenizer" => {
            let path = args.tokenizer.ok_or_else(|| {
                CliError::InvalidInput(
                    "--tokenization client_tokenizer requires --tokenizer <PATH>: §4.4.6's \
                     digest is computed from the tokenizer this run opens and counts with. A \
                     digest cannot be declared."
                        .to_string(),
                )
            })?;
            let counter = ClientTokenizer::from_file(path)
                .map_err(|e| CliError::InvalidInput(e.to_string()))?;
            if let Some(expected) = args.tokenizer_sha256 {
                counter
                    .assert_digest(expected)
                    .map_err(|e| CliError::InvalidInput(e.to_string()))?;
            }
            // The two §4.4.6 booleans are FACTS of this counter, not
            // declarations about it: it encodes with `add_special_tokens =
            // false` and counts the prompt and the completion from separate
            // strings. Accepting `--counts-special-tokens` here would write a
            // receipt field that contradicts the code that produced the numbers.
            if args.counts_special_tokens || args.counts_prompt_echo {
                return Err(CliError::InvalidInput(
                    "--counts-special-tokens / --counts-prompt-echo are declarations about a \
                     SERVER's counting semantics and cannot be set under --tokenization \
                     client_tokenizer: the client counter counts neither, and the receipt \
                     records what it actually did."
                        .to_string(),
                ));
            }
            Ok(TokenAccounting::client_tokenizer(counter))
        }
        other => Err(CliError::InvalidInput(format!(
            "--tokenization {other:?}: expected server_usage or client_tokenizer (§4.4.6 \
             gives `method` no default)"
        ))),
    }
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
    let accounting = build_accounting(&args)?;
    let provenance = build_provenance(&args)?;
    let workload = Workload::from_str(args.workload)
        .map_err(|e| CliError::InvalidInput(format!("--workload: {e}")))?;
    std::fs::create_dir_all(args.receipt).map_err(|e| {
        CliError::InvalidFormat(format!("creating {}: {e}", args.receipt.display()))
    })?;

    let corpus = resolve_corpus(args.profile, args.prompts)?;
    let prompt_band = corpus.band;
    let prompts: Vec<ChatRequest> = corpus
        .requests
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
    match accounting.counter() {
        Some(counter) => println!(
            "tokens   §4.4.6 client_tokenizer — {} ({} bytes)\n         sha256 {}",
            counter.origin(),
            counter.source_len(),
            counter.tokenizer_sha256()
        ),
        None => println!(
            "tokens   §4.4.6 server_usage — the counts are the SERVER's own `usage` block. \
             Two servers' usage blocks are two implementations' opinions; --tokenization \
             client_tokenizer --tokenizer <PATH> counts both lanes with one tokenizer."
        ),
    }

    let mut written = Vec::new();
    for k in 0..args.replicates {
        let mut runs = Vec::with_capacity(levels.len());
        for &c in &levels {
            let band = BandConfig::conformant(c);
            let run = run_band(&client, &prompts, &band, &accounting, args.stream)
                .await
                .map_err(|e| {
                    CliError::InferenceFailed(format!("band c={c} replicate {}: {e}", k + 1))
                })?;
            check_prompt_band(prompt_band, prompts.len(), &run.samples).map_err(|e| {
                CliError::InvalidInput(format!("band c={c} replicate {}: {e}", k + 1))
            })?;
            report_run(c, k, args.replicates, &run);
            runs.push((c, run));
        }
        written.push(write_replicate(
            &args,
            accounting.block(),
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

/// Assert §4.3.1's prompt-length band against what the server actually counted.
///
/// # The defect this closes (PERF-056, #2778)
///
/// `prompts-w1.jsonl`'s `_meta.token_count_note` says, in the file, that
/// "the 512 +/-8 of 4.3.1 is asserted by the harness against the model's own
/// tokenizer at measurement time". No line of code did that. The generator
/// says so too and cannot do it (it runs no tokenizer); the loader says the
/// harness does it; the harness did nothing. A workload 200 tokens off its
/// declared shape would have been measured, receipted and ratcheted as W1.
///
/// This runs after each band because there is nowhere earlier it CAN run: the
/// counts are the server's, reported per request. Under I-9 the run is spent
/// either way — so the point is not to save the replicate, it is to stop the
/// receipt being written as if the workload had been W1.
///
/// `RequestSample::index` is the monotone ISSUE index; the corpus is consumed
/// modulo its length (`band.rs`'s `prompts[slot.0 % prompts.len()]`), so the
/// prompt index is recovered the same way. Only `Completed` samples are read:
/// a failed or timed-out request carries `prompt_tokens: 0` by construction,
/// and counting those would red every band that saw one timeout.
fn check_prompt_band(
    prompt_band: Option<PromptTokenBand>,
    corpus_len: usize,
    samples: &[RequestSample],
) -> std::result::Result<(), String> {
    let Some(prompt_band) = prompt_band else {
        return Ok(());
    };
    if corpus_len == 0 {
        return Ok(());
    }
    let observed: Vec<(usize, u32)> = samples
        .iter()
        .filter(|s| s.outcome == Outcome::Completed)
        .map(|s| (s.index % corpus_len, s.prompt_tokens))
        .collect();
    assert_prompt_tokens_in_band(prompt_band, &observed)
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

    // ---------------------------------------------------------------------
    // §4.3.1's PROMPT-LENGTH BAND, at the point of measurement (PERF-056,
    // #2778). `prompts-w1.jsonl` promises in its own `_meta` that "the 512
    // +/-8 of 4.3.1 is asserted by the harness against the model's own
    // tokenizer at measurement time". This is the harness. Until PERF-056 it
    // did nothing of the kind, and a W1 receipt could be written over a
    // workload of any shape.
    //
    //  server-reported prompt_tokens over the band | must  | why
    //  --------------------------------------------|-------|-----------------
    //  every completed request at 512              | OK    |
    //  504 / 520 at the edges       [BOUNDARY]     | OK    | the band is a band
    //  one request at 521                          | ERR   | names the prompt
    //  a FAILED request's structural 0  [BOUNDARY] | OK    | not a workload defect
    //  no band declared (a --profile) [BOUNDARY]   | OK    | nothing claimed it
    // ---------------------------------------------------------------------

    const BAND_512: PromptTokenBand = PromptTokenBand {
        target: 512,
        tolerance: 8,
    };

    fn sample(index: usize, prompt_tokens: u32, outcome: Outcome) -> RequestSample {
        RequestSample {
            index,
            worker: 0,
            start_s: 0.0,
            end_s: 1.0,
            token_times_s: Vec::new(),
            generated_tokens: 128,
            prompt_tokens,
            outcome,
            in_flight_at_start: 1,
            drained: false,
        }
    }

    #[test]
    fn prompt_band_in_band_run_is_accepted() {
        let s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, 512, Outcome::Completed))
            .collect();
        check_prompt_band(Some(BAND_512), 256, &s).expect("512 is dead centre");
    }

    #[test]
    fn prompt_band_edges_stay_green() {
        // DISCRIMINATION. 504 and 520 are conformant W1. A run reddened here
        // is a gate that cannot pass, which is the mirror of the defect above.
        let s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, if i % 2 == 0 { 504 } else { 520 }, Outcome::Completed))
            .collect();
        check_prompt_band(Some(BAND_512), 256, &s).expect("both edges are INSIDE 512 +/- 8");
    }

    #[test]
    fn prompt_band_one_out_of_band_request_fails_the_band_naming_the_prompt() {
        let mut s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, 512, Outcome::Completed))
            .collect();
        // Issue index 269 over a 256-prompt corpus is prompt 13 -- the modulo
        // the worker loop uses, so the message points at the record an
        // operator would actually edit.
        s[7] = sample(269, 521, Outcome::Completed);
        let err = check_prompt_band(Some(BAND_512), 256, &s)
            .expect_err("521 is one token past the high edge");
        assert!(
            err.contains("prompt 13"),
            "must name the CORPUS prompt: {err}"
        );
        assert!(err.contains("521"), "must give the actual length: {err}");
        // REVERT -> GREEN.
        s[7] = sample(269, 512, Outcome::Completed);
        check_prompt_band(Some(BAND_512), 256, &s).expect("reverted run is in band");
    }

    #[test]
    fn prompt_band_ignores_non_completed_samples() {
        // DISCRIMINATION, and the one that would otherwise red every band that
        // saw a single timeout: a failed or abandoned request carries
        // `prompt_tokens: 0` BY CONSTRUCTION (band.rs's `sample_from`), not
        // because the workload was wrong.
        let mut s: Vec<RequestSample> = (0..32)
            .map(|i| sample(i, 512, Outcome::Completed))
            .collect();
        s.push(sample(99, 0, Outcome::Failed));
        s.push(sample(100, 0, Outcome::Timeout));
        s.push(sample(101, 0, Outcome::AbandonedAtDrain));
        check_prompt_band(Some(BAND_512), 256, &s)
            .expect("a failed request's structural zero is not an out-of-band prompt");
    }

    #[test]
    fn prompt_band_absent_declares_nothing_and_asserts_nothing() {
        // A built-in `--profile` corpus declares no band. Inventing 512 +/- 8
        // for it would be the fabricated-threshold defect this epic is named
        // after, and would red every non-W1 run.
        let s = vec![sample(0, 37, Outcome::Completed)];
        check_prompt_band(None, 256, &s).expect("no claim, no rule");
    }

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
            tokenizer: None,
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
        let err = build_accounting(&args("1", "guess")).expect_err("must reject");
        let msg = err.to_string();
        assert!(msg.contains("server_usage"), "{msg}");
        assert!(msg.contains("client_tokenizer"), "{msg}");
    }

    /// The in-tree MiniLM `tokenizer.json`, borrowed from SetFit's fixtures.
    fn a_real_tokenizer() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-core/tests/fixtures/setfit/tokenizer.json")
    }

    /// A digest can no longer be typed; it has to be computed from a file.
    #[test]
    fn client_tokenizer_without_a_tokenizer_file_is_refused() {
        let err = build_accounting(&args("1", "client_tokenizer")).expect_err("a file is required");
        let msg = err.to_string();
        assert!(msg.contains("--tokenizer <PATH>"), "{msg}");
        assert!(msg.contains("cannot be declared"), "{msg}");
    }

    /// The §4.4.6 digest in the receipt is the sha256 of the bytes that were
    /// opened, computed here rather than accepted from the command line.
    #[test]
    fn client_tokenizer_computes_the_digest_from_the_file_it_opens() {
        let path = a_real_tokenizer();
        let mut a = args("1", "client_tokenizer");
        a.tokenizer = Some(&path);
        a.counts_special_tokens = false;
        let accounting = build_accounting(&a).expect("the fixture must load");

        let expected =
            apr_test::perf_gate::sha256_file(&path).expect("independent hash of the same file");
        match accounting.block() {
            TokenizationBlock::ClientTokenizer {
                tokenizer_sha256,
                counts_special_tokens,
                counts_prompt_echo,
            } => {
                assert_eq!(tokenizer_sha256, &expected);
                assert!(!counts_special_tokens);
                assert!(!counts_prompt_echo);
            }
            other => panic!("expected client_tokenizer, got {other:?}"),
        }
        assert!(accounting.counter().is_some());
    }

    /// `--tokenizer-sha256` asserts. It refuses a run; it supplies nothing.
    #[test]
    fn an_asserted_digest_that_does_not_match_the_file_refuses_the_run() {
        let path = a_real_tokenizer();
        let wrong = "a".repeat(64);
        let mut a = args("1", "client_tokenizer");
        a.tokenizer = Some(&path);
        a.tokenizer_sha256 = Some(&wrong);
        a.counts_special_tokens = false;
        let err = build_accounting(&a).expect_err("a wrong assertion must refuse the run");
        let msg = err.to_string();
        assert!(msg.contains("does not match"), "{msg}");
        assert!(msg.contains("COMPUTED"), "{msg}");
    }

    #[test]
    fn an_asserted_digest_that_matches_the_file_is_accepted() {
        let path = a_real_tokenizer();
        let right = apr_test::perf_gate::sha256_file(&path).expect("hash");
        let mut a = args("1", "client_tokenizer");
        a.tokenizer = Some(&path);
        a.tokenizer_sha256 = Some(&right);
        a.counts_special_tokens = false;
        assert!(build_accounting(&a).is_ok());
    }

    #[test]
    fn a_tokenizer_path_that_is_not_there_is_refused_where_it_was_typed() {
        let missing = PathBuf::from("/nonexistent/perf-gate/tokenizer.json");
        let mut a = args("1", "client_tokenizer");
        a.tokenizer = Some(&missing);
        a.counts_special_tokens = false;
        let err = build_accounting(&a).expect_err("must refuse");
        assert!(err.to_string().contains("cannot read tokenizer"), "{err}");
    }

    /// The §4.4.6 booleans describe a SERVER's semantics. Under
    /// `client_tokenizer` they are facts of the counter, and the counter counts
    /// neither special tokens nor an echoed prompt.
    #[test]
    fn declaring_special_token_counting_under_client_tokenizer_is_refused() {
        let path = a_real_tokenizer();
        let mut a = args("1", "client_tokenizer");
        a.tokenizer = Some(&path);
        a.counts_special_tokens = true;
        let err = build_accounting(&a).expect_err("a contradicted receipt field");
        let msg = err.to_string();
        assert!(msg.contains("--counts-special-tokens"), "{msg}");
        assert!(msg.contains("records what it actually did"), "{msg}");
    }

    /// A tokenizer that will never be consulted must not be recorded as if it
    /// had been.
    #[test]
    fn a_tokenizer_under_server_usage_is_refused() {
        let path = a_real_tokenizer();
        let mut a = args("1", "server_usage");
        a.tokenizer = Some(&path);
        let err = build_accounting(&a).expect_err("must refuse");
        assert!(err.to_string().contains("never used"), "{err}");
    }

    #[test]
    fn server_usage_needs_no_tokenizer() {
        assert!(build_accounting(&args("1", "server_usage")).is_ok());
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
