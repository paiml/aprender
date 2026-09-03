//! `apr tokenize plan/apply` — BPE vocabulary training pipeline.
//!
//! Plan validates the corpus and estimates training time.
//! Apply trains a BPE tokenizer and writes vocab.json + merges.txt.

use colored::Colorize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

/// Run `apr tokenize plan` — validate inputs and estimate training.
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run_plan(
    data: &Path,
    vocab_size: usize,
    algorithm: &str,
    output_dir: &Path,
    format: &str,
    json_output: bool,
) -> Result<()> {
    contract_pre_tokenizer_training_correctness!();
    validate_algorithm(algorithm)?;
    validate_vocab_size(vocab_size)?;

    if !data.exists() {
        return Err(CliError::FileNotFound(data.to_path_buf()));
    }

    let corpus_stats = analyze_corpus(data)?;

    let plan = TokenizePlan {
        algorithm: algorithm.to_string(),
        vocab_size,
        corpus_path: data.display().to_string(),
        corpus_lines: corpus_stats.lines,
        corpus_bytes: corpus_stats.bytes,
        unique_chars: corpus_stats.unique_chars,
        output_dir: output_dir.display().to_string(),
        estimated_minutes: estimate_training_time(corpus_stats.bytes, vocab_size),
        verdict: plan_verdict(&corpus_stats, vocab_size),
    };

    let effective_format = if json_output { "json" } else { format };
    match effective_format {
        "json" => {
            let json = serde_json::to_string_pretty(&plan)
                .map_err(|e| CliError::InvalidFormat(e.to_string()))?;
            println!("{json}");
        }
        "yaml" => {
            return Err(CliError::ValidationFailed(
                "YAML output not supported. Use --format json or --format text.".to_string(),
            ));
        }
        _ => print_plan_text(&plan),
    }

    if plan.verdict == "blocked" {
        return Err(CliError::ValidationFailed(
            "Plan is blocked — resolve failures before applying".to_string(),
        ));
    }

    contract_post_tokenizer_training_correctness!(&());
    Ok(())
}

/// Run `apr tokenize apply` — train tokenizer and write output.
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run_apply(
    data: &Path,
    vocab_size: usize,
    algorithm: &str,
    output_dir: &Path,
    max_lines: usize,
    json_output: bool,
) -> Result<()> {
    validate_algorithm(algorithm)?;
    validate_vocab_size(vocab_size)?;

    if !data.exists() {
        return Err(CliError::FileNotFound(data.to_path_buf()));
    }

    // Read corpus
    let corpus_text = read_corpus(data, max_lines)?;
    let corpus_refs: Vec<&str> = corpus_text.iter().map(String::as_str).collect();

    if corpus_refs.is_empty() {
        return Err(CliError::ValidationFailed(
            "Corpus is empty — no text to train on".to_string(),
        ));
    }

    if !json_output {
        print_apply_header(data, vocab_size, algorithm, output_dir, corpus_refs.len());
    }

    // Train
    let start = Instant::now();
    let tokenizer = train_tokenizer(&corpus_refs, vocab_size, algorithm)?;
    let elapsed = start.elapsed();

    // Write output
    std::fs::create_dir_all(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

    let actual_vocab_size = tokenizer.vocab_size();
    write_vocab_json(output_dir, &tokenizer)?;
    write_merges_txt(output_dir, &tokenizer)?;

    let result = TokenizeResult {
        algorithm: algorithm.to_string(),
        vocab_size: actual_vocab_size,
        corpus_lines: corpus_refs.len(),
        training_seconds: elapsed.as_secs_f64(),
        output_dir: output_dir.display().to_string(),
    };

    if json_output {
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?;
        println!("{json}");
    } else {
        print_apply_result(&result);
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn validate_algorithm(algorithm: &str) -> Result<()> {
    match algorithm {
        "bpe" | "wordpiece" | "unigram" => Ok(()),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown algorithm: {algorithm}. Supported: bpe, wordpiece, unigram"
        ))),
    }
}

fn validate_vocab_size(vocab_size: usize) -> Result<()> {
    if vocab_size < 10 {
        return Err(CliError::ValidationFailed(format!(
            "vocab_size must be at least 10, got {vocab_size}"
        )));
    }
    if vocab_size > 1_000_000 {
        return Err(CliError::ValidationFailed(format!(
            "vocab_size {vocab_size} is unreasonably large (max 1M)"
        )));
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct TokenizePlan {
    algorithm: String,
    vocab_size: usize,
    corpus_path: String,
    corpus_lines: usize,
    corpus_bytes: u64,
    unique_chars: usize,
    output_dir: String,
    estimated_minutes: f64,
    verdict: String,
}

#[derive(serde::Serialize)]
struct TokenizeResult {
    algorithm: String,
    vocab_size: usize,
    corpus_lines: usize,
    training_seconds: f64,
    output_dir: String,
}

struct CorpusStats {
    lines: usize,
    bytes: u64,
    unique_chars: usize,
}

fn analyze_corpus(path: &Path) -> Result<CorpusStats> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?;
    let bytes = metadata.len();

    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read {}: {e}", path.display())))?;

    let lines = content.lines().count();
    let unique_chars: std::collections::HashSet<char> = content.chars().collect();

    Ok(CorpusStats {
        lines,
        bytes,
        unique_chars: unique_chars.len(),
    })
}

fn estimate_training_time(bytes: u64, vocab_size: usize) -> f64 {
    // Rough estimate: ~1 MB/sec for BPE training, scales with vocab_size
    let mb = bytes as f64 / (1024.0 * 1024.0);
    let vocab_factor = (vocab_size as f64 / 32000.0).max(1.0);
    (mb * vocab_factor) / 60.0
}

fn plan_verdict(stats: &CorpusStats, vocab_size: usize) -> String {
    if stats.lines == 0 {
        return "blocked".to_string();
    }
    if vocab_size > stats.unique_chars * 100 {
        return "warning".to_string();
    }
    "ready".to_string()
}

fn read_corpus(path: &Path, max_lines: usize) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot read corpus {}: {e}", path.display()))
    })?;

    let lines: Vec<String> = if max_lines > 0 {
        content.lines().take(max_lines).map(String::from).collect()
    } else {
        content.lines().map(String::from).collect()
    };

    Ok(lines)
}

/// Wrapper around aprender's tokenizer training.
struct TrainedTokenizer {
    vocab: std::collections::HashMap<String, u32>,
    merges: Vec<(String, String)>,
}

impl TrainedTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Task #103: Train a BPE tokenizer via aprender-train's `BPETokenizer`, which
/// honors `--min-frequency` (config.min_frequency pair-pruning) and
/// `--normalization nfc` (INV-TOK-003) — two knobs the legacy
/// `aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)` path
/// silently ignored.
///
/// Requires the `training` feature (default-on for `cargo install aprender`).
/// Without it the `entrenar` dep isn't linked, so fall back to the legacy
/// path for minimal builds — but the user's `--min-frequency` choice is lost
/// in that configuration (pre-existing behavior preserved).
#[cfg(feature = "training")]
fn train_bpe_via_entrenar(
    corpus: &[&str],
    vocab_size: usize,
    min_frequency: usize,
    normalization: &str,
) -> Result<TrainedTokenizer> {
    use entrenar::tokenizer::{BPETokenizer, Normalization, Tokenizer, TokenizerConfig};

    let norm = match normalization {
        "nfc" => Normalization::NFC,
        "none" => Normalization::None,
        other => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown normalization: {other}. Supported: none, nfc"
            )));
        }
    };

    let config = TokenizerConfig::bpe()
        .with_vocab_size(vocab_size)
        .with_min_frequency(min_frequency)
        .with_normalization(norm);
    let mut tokenizer = BPETokenizer::new(config);
    tokenizer
        .train(corpus)
        .map_err(|e| CliError::ValidationFailed(format!("BPE training failed: {e}")))?;

    Ok(TrainedTokenizer {
        vocab: tokenizer.vocab().clone(),
        merges: tokenizer.merges().to_vec(),
    })
}

/// Fallback path when built without the `training` feature. Calls the legacy
/// `aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)` surface,
/// which ignores `min_frequency` and `normalization` (pre-task-#103 behavior).
#[cfg(not(feature = "training"))]
fn train_bpe_via_entrenar(
    corpus: &[&str],
    vocab_size: usize,
    _min_frequency: usize,
    _normalization: &str,
) -> Result<TrainedTokenizer> {
    let tokenizer = aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)
        .map_err(|e| CliError::ValidationFailed(format!("BPE training failed: {e}")))?;
    Ok(TrainedTokenizer {
        vocab: tokenizer.vocab().clone(),
        merges: tokenizer.merges().to_vec(),
    })
}

fn train_tokenizer(
    corpus: &[&str],
    vocab_size: usize,
    algorithm: &str,
) -> Result<TrainedTokenizer> {
    match algorithm {
        "bpe" => {
            let tokenizer = aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)
                .map_err(|e| CliError::ValidationFailed(format!("BPE training failed: {e}")))?;
            Ok(TrainedTokenizer {
                vocab: tokenizer.vocab().clone(),
                merges: tokenizer.merges().to_vec(),
            })
        }
        "wordpiece" => {
            let tokenizer = aprender::text::tokenize::WordPieceTokenizer::train(corpus, vocab_size)
                .map_err(|e| {
                    CliError::ValidationFailed(format!("WordPiece training failed: {e}"))
                })?;
            // WordPiece has vocab but no merges
            Ok(TrainedTokenizer {
                vocab: tokenizer.vocab().clone(),
                merges: Vec::new(),
            })
        }
        "unigram" => {
            let tokenizer = aprender::text::tokenize::UnigramTokenizer::train(corpus, vocab_size)
                .map_err(|e| {
                CliError::ValidationFailed(format!("Unigram training failed: {e}"))
            })?;
            // Unigram has vocab (as id map) but no merges
            Ok(TrainedTokenizer {
                vocab: tokenizer.vocab_ids(),
                merges: Vec::new(),
            })
        }
        _ => unreachable!("algorithm validated above"),
    }
}

fn write_vocab_json(output_dir: &Path, tokenizer: &TrainedTokenizer) -> Result<()> {
    let vocab_path = output_dir.join("vocab.json");
    // Sort by ID for deterministic output
    let mut entries: Vec<(&String, &u32)> = tokenizer.vocab.iter().collect();
    entries.sort_by_key(|(_, id)| *id);
    let ordered: serde_json::Map<String, serde_json::Value> = entries
        .into_iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::Number((*v).into())))
        .collect();
    let json = serde_json::to_string_pretty(&ordered)
        .map_err(|e| CliError::InvalidFormat(e.to_string()))?;
    std::fs::write(&vocab_path, json).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot write {}: {e}", vocab_path.display()))
    })?;
    Ok(())
}

fn write_merges_txt(output_dir: &Path, tokenizer: &TrainedTokenizer) -> Result<()> {
    let merges_path = output_dir.join("merges.txt");
    let mut content = String::from("#version: 0.2\n");
    for (left, right) in &tokenizer.merges {
        content.push_str(left);
        content.push(' ');
        content.push_str(right);
        content.push('\n');
    }
    std::fs::write(&merges_path, content).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot write {}: {e}", merges_path.display()))
    })?;
    Ok(())
}

// ─── Output formatting ──────────────────────────────────────────────────────

fn print_plan_text(plan: &TokenizePlan) {
    output::header("apr tokenize plan — Tokenizer Training Pre-flight");
    println!();
    output::section("Configuration");
    output::kv("  Algorithm", &plan.algorithm);
    output::kv("  Vocab size", format_number(plan.vocab_size));
    output::kv("  Corpus", &plan.corpus_path);
    output::kv("  Output", &plan.output_dir);
    println!();
    output::section("Corpus Analysis");
    output::kv("  Lines", format_number(plan.corpus_lines));
    output::kv("  Size", format_bytes(plan.corpus_bytes));
    output::kv("  Unique chars", format_number(plan.unique_chars));
    println!();
    output::section("Estimates");
    output::kv("  Training time", format_duration(plan.estimated_minutes));
    println!();

    let verdict_display = match plan.verdict.as_str() {
        "ready" => format!("{}", "READY".green().bold()),
        "warning" => format!("{}", "WARNING".yellow().bold()),
        "blocked" => format!("{}", "BLOCKED".red().bold()),
        _ => plan.verdict.clone(),
    };
    output::kv("  Verdict", verdict_display);
    println!();
}

fn print_apply_header(
    data: &Path,
    vocab_size: usize,
    algorithm: &str,
    output_dir: &Path,
    corpus_lines: usize,
) {
    output::header("apr tokenize apply — Training Tokenizer");
    println!();
    output::kv("  Algorithm", algorithm);
    output::kv("  Vocab size", format_number(vocab_size));
    output::kv("  Corpus", data.display().to_string());
    output::kv("  Lines", format_number(corpus_lines));
    output::kv("  Output", output_dir.display().to_string());
    println!();
}

fn print_apply_result(result: &TokenizeResult) {
    output::section("Result");
    println!("  {} Tokenizer trained successfully", "OK".green().bold());
    output::kv("  Final vocab size", format_number(result.vocab_size));
    output::kv(
        "  Training time",
        format!("{:.1}s", result.training_seconds),
    );
    output::kv("  vocab.json", format!("{}/vocab.json", result.output_dir));
    output::kv("  merges.txt", format!("{}/merges.txt", result.output_dir));
    println!();
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_duration(minutes: f64) -> String {
    if minutes < 1.0 {
        format!("{:.0} sec", minutes * 60.0)
    } else if minutes < 60.0 {
        format!("{:.1} min", minutes)
    } else {
        format!("{:.1} hours", minutes / 60.0)
    }
}

// ─── `apr tokenize train` — BPE for MODEL-2 (contracts/tokenizer-bpe-v1.yaml) ──

#[derive(serde::Serialize)]
struct TokenizeTrainResult {
    algorithm: String,
    vocab_size: usize,
    corpus_lines: usize,
    corpus_files: usize,
    min_frequency: usize,
    normalization: String,
    training_seconds: f64,
    output_dir: String,
}

/// Run `apr tokenize train` — train BPE from a JSONL corpus with NFC normalization.
pub(crate) fn run_train(
    corpus: &Path,
    vocab_size: usize,
    min_frequency: usize,
    output_dir: &Path,
    normalization: &str,
    json_output: bool,
) -> Result<()> {
    validate_vocab_size(vocab_size)?;
    validate_normalization(normalization)?;

    if !corpus.exists() {
        return Err(CliError::FileNotFound(corpus.to_path_buf()));
    }

    let files = collect_jsonl_files(corpus)?;
    if files.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "No .jsonl files found under {}",
            corpus.display()
        )));
    }

    // Task #103: aprender-train's BPETokenizer applies the configured
    // normalization internally in `preprocess()`, so we pass raw `content`
    // strings here and let the trainer honor `--normalization`. Applying NFC
    // in the CLI reader would be redundant (NFC is idempotent) and would
    // hide bugs in the trainer's own normalization plumbing.
    let mut lines: Vec<String> = Vec::new();
    for file in &files {
        read_jsonl_content(file, &mut lines)?;
    }

    if lines.is_empty() {
        return Err(CliError::ValidationFailed(
            "Corpus contained no `content` fields — nothing to train on".to_string(),
        ));
    }

    if !json_output {
        print_train_header(corpus, vocab_size, output_dir, files.len(), lines.len());
    }

    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let start = Instant::now();
    // Task #103: route `apr tokenize train` through aprender-train's BPE so
    // `--min-frequency` and `--normalization nfc` are actually honored
    // (contracts/tokenizer-bpe-v1.yaml INV-TOK-002, INV-TOK-003). The
    // aprender-core `BpeTokenizer::train(corpus, vocab_size)` legacy path had
    // no `min_frequency` knob and no NFC plumbing.
    let trained = train_bpe_via_entrenar(&refs, vocab_size, min_frequency, normalization)?;
    let elapsed = start.elapsed();

    std::fs::create_dir_all(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

    write_vocab_json(output_dir, &trained)?;
    write_merges_txt(output_dir, &trained)?;

    let result = TokenizeTrainResult {
        algorithm: "bpe".to_string(),
        vocab_size: trained.vocab_size(),
        corpus_lines: lines.len(),
        corpus_files: files.len(),
        min_frequency,
        normalization: normalization.to_string(),
        training_seconds: elapsed.as_secs_f64(),
        output_dir: output_dir.display().to_string(),
    };

    if json_output {
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?;
        println!("{json}");
    } else {
        print_train_result(&result);
    }

    Ok(())
}

fn validate_normalization(norm: &str) -> Result<()> {
    match norm {
        "none" | "nfc" => Ok(()),
        other => Err(CliError::ValidationFailed(format!(
            "Unknown normalization: {other}. Supported: none, nfc"
        ))),
    }
}

/// Per-doc-chunk batch size for the parallel encode path. Chosen so each
/// rayon spawn amortizes thread setup against ~10K BPE encodes (~13K tok/s
/// per worker → ~1s of work per chunk on the SHIP-TWO-001 760M-char Python
/// corpus). Smaller chunks waste rayon overhead; larger chunks delay shard
/// flush feedback (issue #1547).
#[cfg(feature = "training")]
const ENCODE_CHUNK_SIZE: usize = 10_000;

/// Operator-facing progress emission knobs for `apr tokenize encode-corpus`
/// (issue #1547, contract apr-tokenize-parallel-bpe-v1.yaml v1.2.0).
///
/// Default emits a `[progress]` line on stderr every 1000 docs OR every
/// 60 seconds, whichever comes first. `quiet=true` suppresses everything
/// except the final summary line. Tunable bounds let CI pin smaller
/// budgets without re-deriving the full `ProgressEmitter` state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProgressConfig {
    /// When `true`, suppresses all per-doc and final progress lines on
    /// stderr (the JSON manifest and stdout summary still emit). Used
    /// by CI/log-scraping callers that prefer total silence.
    pub quiet: bool,
    /// Emit a progress line at most every N docs (default 1000).
    pub interval_docs: u64,
    /// Emit a progress line at most every S seconds (default 60).
    pub interval_seconds: u64,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            quiet: false,
            interval_docs: 1000,
            interval_seconds: 60,
        }
    }
}

/// Stateful per-run emitter. Tracks last emission tick (docs + wall) so
/// the OR-cadence can fire correctly: emit when EITHER N docs have been
/// seen since last tick OR S seconds have elapsed since last tick. Final
/// emission shows total wall + tokens + per-doc rate at completion.
///
/// Format: `[progress] doc=N/T tokens=K rate=X.X docs/s eta=YYYY-MM-DDTHH:MM:SSZ`.
/// When total `T` is unknown, omits the `/T` and `eta=` fragments.
#[cfg(feature = "training")]
pub(crate) struct ProgressEmitter {
    cfg: ProgressConfig,
    start: Instant,
    last_emit_docs: u64,
    last_emit_time: Instant,
    total_docs_hint: Option<u64>,
}

#[cfg(feature = "training")]
impl ProgressEmitter {
    pub(crate) fn new(cfg: ProgressConfig, total_docs_hint: Option<u64>) -> Self {
        let now = Instant::now();
        Self {
            cfg,
            start: now,
            last_emit_docs: 0,
            last_emit_time: now,
            total_docs_hint,
        }
    }

    /// Decide whether `(docs_seen, tokens_seen)` warrants an emit. Returns
    /// `true` when EITHER `docs_seen - last_emit_docs >= interval_docs`
    /// OR `wall_since_last >= interval_seconds`. Pure-function caller can
    /// mock without IO.
    pub(crate) fn should_emit(&self, docs_seen: u64, now: Instant) -> bool {
        if self.cfg.quiet {
            return false;
        }
        let docs_due = docs_seen.saturating_sub(self.last_emit_docs) >= self.cfg.interval_docs;
        let time_due = now.saturating_duration_since(self.last_emit_time).as_secs()
            >= self.cfg.interval_seconds;
        docs_due || time_due
    }

    /// Reset the per-tick clocks (call right after an emit) so the next
    /// firing requires a fresh `interval_docs`/`interval_seconds` budget.
    pub(crate) fn mark_emitted(&mut self, docs_seen: u64, now: Instant) {
        self.last_emit_docs = docs_seen;
        self.last_emit_time = now;
    }

    /// Format the per-tick progress line. Public for test inspection so
    /// the format invariants (issue #1547 §AC1) can be pinned without
    /// scraping stderr in unit tests.
    pub(crate) fn format_line(&self, docs_seen: u64, tokens_seen: u64, now: Instant) -> String {
        let elapsed = now.saturating_duration_since(self.start).as_secs_f64();
        // Avoid division by zero on the very first emit (elapsed≈0).
        let rate = if elapsed > 0.0 {
            docs_seen as f64 / elapsed
        } else {
            0.0
        };
        match self.total_docs_hint {
            Some(total) if total > 0 => {
                let remaining = total.saturating_sub(docs_seen);
                let eta_secs = if rate > 0.0 {
                    (remaining as f64 / rate).round() as i64
                } else {
                    0
                };
                let eta = format_eta_iso8601_utc(eta_secs);
                format!(
                    "[progress] doc={docs_seen}/{total} tokens={tokens_seen} \
                     rate={rate:.1} docs/s eta={eta}"
                )
            }
            _ => format!("[progress] doc={docs_seen} tokens={tokens_seen} rate={rate:.1} docs/s"),
        }
    }

    /// Emit a per-tick line to stderr, then update the tick clocks. No-op
    /// when `quiet=true`.
    pub(crate) fn emit_tick(&mut self, docs_seen: u64, tokens_seen: u64, now: Instant) {
        if self.cfg.quiet {
            return;
        }
        eprintln!("{}", self.format_line(docs_seen, tokens_seen, now));
        self.mark_emitted(docs_seen, now);
    }

    /// Emit the final completion line. Format follows AC4 — total wall +
    /// tokens + per-doc rate. Always shows total docs (no /T fragment
    /// needed since we know the actual count at this point).
    pub(crate) fn emit_final(&self, total_docs: u64, total_tokens: u64) {
        if self.cfg.quiet {
            return;
        }
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            total_docs as f64 / elapsed
        } else {
            0.0
        };
        eprintln!(
            "[progress] done docs={total_docs} tokens={total_tokens} \
             elapsed={elapsed:.1}s rate={rate:.1} docs/s"
        );
    }
}

/// Operator pre-flight knobs for `apr tokenize encode-corpus
/// --estimate-only` (issue #1547, contract apr-tokenize-parallel-bpe-v1.yaml
/// v1.3.0). When `enabled = true`, the encode loop reads the first
/// `sample_docs` documents, encodes them under the configured tokenizer,
/// observes (tokens, wall-time-per-doc), and extrapolates against the
/// total document count of the corpus. NO shards or manifest are written.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EstimateConfig {
    /// When `true`, run the pre-flight extrapolation path and return
    /// without writing shards or manifest.
    pub enabled: bool,
    /// Number of documents to sample. Default 1000 — large enough to
    /// average out per-doc rate noise, small enough to keep the
    /// pre-flight wall under a few seconds on a fast tokenizer.
    pub sample_docs: u64,
}

impl Default for EstimateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_docs: 1000,
        }
    }
}

/// Pure-function extrapolation kernel. Given a sample (sample_size docs
/// took sample_wall seconds and produced sample_tokens), extrapolate
/// (estimated_total_tokens, estimated_shards, estimated_wall_seconds)
/// against `total_docs` and the operator-configured shard size /
/// worker count. AC4 formula:
///
/// ```text
/// estimated_wall = (sample_wall / sample_size) × total_docs / num_workers
/// ```
///
/// Pure-function so unit tests can pin the math on a tiny synthetic
/// fixture without involving the BPE tokenizer or filesystem.
pub(crate) fn extrapolate_estimate(
    sample_size: u64,
    sample_tokens: u64,
    sample_wall_seconds: f64,
    total_docs: u64,
    shard_tokens: u64,
    num_workers: u64,
) -> (u64, u64, f64) {
    if sample_size == 0 {
        return (0, 0, 0.0);
    }
    // Per-doc averages from the sample.
    let tokens_per_doc = sample_tokens as f64 / sample_size as f64;
    let wall_per_doc = sample_wall_seconds / sample_size as f64;

    // Extrapolate to the full corpus.
    let estimated_total_tokens = (tokens_per_doc * total_docs as f64).round() as u64;
    let estimated_shards = if shard_tokens == 0 {
        0
    } else {
        // ceil(estimated_total_tokens / shard_tokens)
        estimated_total_tokens.div_ceil(shard_tokens)
    };
    let workers = num_workers.max(1);
    let estimated_wall = wall_per_doc * total_docs as f64 / workers as f64;

    (estimated_total_tokens, estimated_shards, estimated_wall)
}

/// Format a forward-looking ETA (in seconds from now) as an ISO-8601 UTC
/// timestamp without external chrono dependency. Computes SystemTime::now()
/// + offset → seconds since epoch → breaks into Y/M/D/H/M/S using the
/// civil-from-days algorithm (Howard Hinnant's date library, public domain).
///
/// Output: `2026-05-05T18:33:07Z`. Pure function of (now, offset_secs).
fn format_eta_iso8601_utc(offset_secs: i64) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let target = now_epoch.saturating_add(offset_secs);
    let (days, seconds_of_day) = (target.div_euclid(86_400), target.rem_euclid(86_400));
    let h = seconds_of_day / 3600;
    let m = (seconds_of_day % 3600) / 60;
    let s = seconds_of_day % 60;
    // Howard Hinnant civil_from_days (proleptic Gregorian, days since 1970-01-01).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m_civ = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_civ = if m_civ <= 2 { y + 1 } else { y };
    format!("{y_civ:04}-{m_civ:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Count the total number of input documents fast — for JSONL we count
/// non-empty lines (≈ `wc -l`); for parquet we sum file-level row counts
/// from the metadata footers (no row-group decode). This is purely for
/// the `--estimate-only` extrapolation; the result is advisory and the
/// real encode path doesn't depend on it.
///
/// Errors propagate as `CliError::ValidationFailed` for unreadable
/// input — same surface as the real encode path so the operator gets a
/// consistent error class regardless of whether they ran with
/// `--estimate-only` or for real.
#[cfg(feature = "training")]
fn count_corpus_docs_fast(files: &[std::path::PathBuf], format: CorpusFormat) -> Result<u64> {
    use std::io::BufRead;

    let mut total: u64 = 0;
    for file in files {
        match format {
            CorpusFormat::Jsonl => {
                let f = std::fs::File::open(file).map_err(|e| {
                    CliError::ValidationFailed(format!("Cannot open {}: {e}", file.display()))
                })?;
                let reader = std::io::BufReader::new(f);
                // Count non-empty lines. Matches `iter_corpus_texts`'s
                // skip-empty behavior exactly so the extrapolation total
                // doesn't drift from what the real encode would see.
                for line in reader.lines() {
                    let line = line.map_err(|e| {
                        CliError::ValidationFailed(format!("Read error in {}: {e}", file.display()))
                    })?;
                    if !line.trim().is_empty() {
                        total += 1;
                    }
                }
            }
            CorpusFormat::Parquet => {
                use parquet::file::reader::{FileReader, SerializedFileReader};
                let f = std::fs::File::open(file).map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "Cannot open parquet {}: {e}",
                        file.display()
                    ))
                })?;
                let reader = SerializedFileReader::new(f).map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "Cannot read parquet metadata {}: {e}",
                        file.display()
                    ))
                })?;
                let metadata = reader.metadata();
                for rg in metadata.row_groups() {
                    total += u64::try_from(rg.num_rows()).unwrap_or(0);
                }
            }
        }
    }
    Ok(total)
}

/// Pre-flight extrapolation path for `apr tokenize encode-corpus
/// --estimate-only`. Reads the first `sample_size` docs from the canonical
/// source iterator, encodes them under `tokenizer`, observes (tokens,
/// wall-time-per-doc), then extrapolates against the full corpus document
/// count. Emits operator-facing `[estimate]` lines to stderr. Writes
/// NOTHING to disk (no shards, no manifest) — AC1+AC2.
///
/// The returned tuple is unused at the dispatch boundary; the caller's
/// signature is `Result<()>` so the operator's exit code matches a
/// successful real encode (zero-on-success). The body is split out so
/// the math is unit-testable via `extrapolate_estimate` (no IO).
#[cfg(feature = "training")]
fn run_estimate_only_path<F>(
    mut encode: F,
    files: &[std::path::PathBuf],
    corpus_format: CorpusFormat,
    content_field: &str,
    sample_size: u64,
    shard_tokens: usize,
    num_workers: usize,
) -> Result<()>
where
    F: FnMut(&str) -> std::result::Result<Vec<u32>, String>,
{
    if sample_size == 0 {
        return Err(CliError::ValidationFailed(
            "--estimate-sample-docs must be >= 1 (got 0)".to_string(),
        ));
    }

    // 1. Count total docs in the corpus (fast — JSONL = wc -l style;
    //    parquet = sum of row-group metadata footers).
    let total_docs = count_corpus_docs_fast(files, corpus_format)?;
    if total_docs == 0 {
        return Err(CliError::ValidationFailed(format!(
            "Corpus contains zero documents — nothing to estimate. \
             Files inspected: {}",
            files.len()
        )));
    }

    // 2. Pull at most `sample_size` documents (or `total_docs` if smaller)
    //    and encode them. We measure both sample-tokens and sample-wall.
    let take_n = sample_size.min(total_docs);
    let mut source = iter_corpus_texts(files, corpus_format, content_field);
    let sample_start = Instant::now();
    let mut sample_tokens: u64 = 0;
    let mut sample_count: u64 = 0;
    while sample_count < take_n {
        let triple = match source.next() {
            Some(Ok(t)) => t,
            Some(Err(e)) => return Err(e),
            None => break,
        };
        let (file_display, locator, text) = triple;
        let ids = encode(&text).map_err(|e| {
            CliError::ValidationFailed(format!("Encoding failed at {file_display} {locator}: {e}"))
        })?;
        sample_tokens += u64::try_from(ids.len()).unwrap_or(0);
        sample_count += 1;
    }
    let sample_wall = sample_start.elapsed().as_secs_f64();

    // 3. Extrapolate via the pure-function kernel — easy to unit-test
    //    in isolation.
    let (estimated_total_tokens, estimated_shards, estimated_wall) = extrapolate_estimate(
        sample_count,
        sample_tokens,
        sample_wall,
        total_docs,
        u64::try_from(shard_tokens).unwrap_or(0),
        u64::try_from(num_workers).unwrap_or(1),
    );

    // 4. Emit the operator-facing `[estimate]` block. We use stderr to
    //    keep stdout clean for any operator that pipes the JSON
    //    manifest from a real encode through to a downstream tool.
    eprintln!("[estimate] input_docs={total_docs}");
    eprintln!(
        "[estimate] sample_size={sample_count} sample_tokens={sample_tokens} \
         sample_wall={sample_wall:.3}s"
    );
    eprintln!("[estimate] estimated_total_tokens={estimated_total_tokens}");
    eprintln!("[estimate] estimated_shards={estimated_shards} (at shard_tokens={shard_tokens})");
    eprintln!(
        "[estimate] estimated_wall={estimated_wall:.0} seconds (at --num-workers={num_workers})"
    );

    Ok(())
}

/// Resolve the effective rayon worker count for `apr tokenize encode-corpus`.
///
/// `None` (default) → `std::thread::available_parallelism()`, falling back
/// to 1 if the OS cannot answer (e.g. cgroup-restricted CI). Explicit `Some(0)`
/// is rejected as a config error rather than silently coerced. `Some(1)`
/// triggers the legacy single-threaded byte-identical path.
#[cfg(feature = "training")]
fn resolve_num_workers(num_workers: Option<usize>) -> Result<usize> {
    match num_workers {
        Some(0) => Err(CliError::ValidationFailed(
            "--num-workers must be >= 1 (got 0)".to_string(),
        )),
        Some(n) => Ok(n),
        None => Ok(std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)),
    }
}

/// Two-format tokenizer dispatch for `apr tokenize encode-corpus`.
///
/// `Hex` — aprender-train's `BPETokenizer` (hex-byte format). Used
/// when vocab.json contains canonical `00..ff` hex tokens (i.e.,
/// trained by `apr tokenize train`).
///
/// `ByteLevel` — aprender-core's `BpeTokenizer` (GPT-2 byte-level
/// + Ġ-prefix). Used when vocab.json is HuggingFace-format (i.e.,
/// extracted by `apr tokenize import-hf` from Qwen2/Llama2/Mistral
/// tokenizer.json). This path closes the SHIP-TWO §60 silent-`<unk>`
/// defect class — pre-this-PR, the hex loader fail-fasted with
/// FALSIFY-BPE-FORMAT-MISMATCH-001 (PR #1585) instead of routing
/// through the byte-level encoder.
#[cfg(feature = "training")]
enum EncodeTokenizer {
    Hex(entrenar::tokenizer::BPETokenizer),
    ByteLevel(aprender::text::bpe::BpeTokenizer),
}

#[cfg(feature = "training")]
impl EncodeTokenizer {
    fn vocab_size(&self) -> usize {
        match self {
            Self::Hex(t) => entrenar::tokenizer::Tokenizer::vocab_size(t),
            Self::ByteLevel(t) => t.vocab_size(),
        }
    }
    fn token_to_id(&self, name: &str) -> Option<u32> {
        match self {
            Self::Hex(t) => entrenar::tokenizer::Tokenizer::token_to_id(t, name),
            Self::ByteLevel(t) => t.token_to_id(name),
        }
    }
    fn encode(&self, text: &str) -> std::result::Result<Vec<u32>, String> {
        match self {
            Self::Hex(t) => {
                entrenar::tokenizer::Tokenizer::encode(t, text).map_err(|e| format!("{e}"))
            }
            Self::ByteLevel(t) => Ok(t.encode(text)),
        }
    }
}

/// Validate every `apr tokenize encode-corpus` argument that can be
/// rejected before the corpus is walked. Preserves the pre-decomposition
/// rejection ORDER exactly — normalization, then eos_policy, then
/// shard_tokens, then `--num-workers`, then each `--corpus` path, then
/// vocab.json, then merges.txt — because the operator-visible error for a
/// call that is wrong in two ways is the first one in that list.
///
/// Returns the resolved worker count plus the two tokenizer file paths so
/// the caller does not re-derive them.
#[cfg(feature = "training")]
fn validate_encode_corpus_args(
    corpus: &[std::path::PathBuf],
    tokenizer_dir: &Path,
    shard_tokens: usize,
    normalization: &str,
    eos_policy: &str,
    num_workers: Option<usize>,
) -> Result<(usize, PathBuf, PathBuf)> {
    validate_normalization(normalization)?;
    match eos_policy {
        "none" | "between" | "after" => {}
        other => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown eos_policy: {other}. Supported: none, between, after"
            )));
        }
    }
    if shard_tokens == 0 {
        return Err(CliError::ValidationFailed(
            "shard_tokens must be > 0".to_string(),
        ));
    }
    let workers = resolve_num_workers(num_workers)?;
    // SPEC §83 P2-C: validate each --corpus path exists.
    for path in corpus {
        if !path.exists() {
            return Err(CliError::FileNotFound(path.clone()));
        }
    }
    let vocab_path = tokenizer_dir.join("vocab.json");
    let merges_path = tokenizer_dir.join("merges.txt");
    if !vocab_path.exists() {
        return Err(CliError::FileNotFound(vocab_path));
    }
    if !merges_path.exists() {
        return Err(CliError::FileNotFound(merges_path));
    }
    Ok((workers, vocab_path, merges_path))
}

/// Render a tokenizer file path as `&str`, rejecting non-UTF-8 bytes with
/// the same message the inline block used (`what` is the file's basename).
#[cfg(feature = "training")]
fn tokenizer_path_str(path: &Path, what: &str) -> Result<String> {
    path.to_str()
        .map(std::string::ToString::to_string)
        .ok_or_else(|| CliError::ValidationFailed(format!("{what} path has non-utf8 bytes")))
}

/// Normalization dispatch for encode-corpus. `validate_normalization` has
/// already rejected anything outside {`nfc`, `none`}.
#[cfg(feature = "training")]
fn encode_normalization(normalization: &str) -> entrenar::tokenizer::Normalization {
    match normalization {
        "nfc" => entrenar::tokenizer::Normalization::NFC,
        "none" => entrenar::tokenizer::Normalization::None,
        _ => unreachable!("validated above"),
    }
}

/// Detect the vocab.json flavour by counting canonical hex-byte tokens
/// `"00".."ff"`. A legitimate hex-byte vocab from `apr tokenize train`
/// always has all 256; a GPT-2 byte-level vocab (from
/// `apr tokenize import-hf` of Qwen/Llama2/Mistral) has < 200.
///
/// This detection is independent of `BPETokenizer::from_vocab_merges`'s
/// behavior — it works whether or not the upstream fail-fast (PR #1585)
/// has merged. Discovery: PR #1596's "try hex first, fall through on
/// FALSIFY-001" strategy depended on PR #1585's fail-fast, which was not
/// yet on main. With #1585 absent, the hex loader silently succeeded on
/// Qwen-format vocabs and produced 99% `<unk>`. Upfront detection
/// eliminates the dependency.
#[cfg(feature = "training")]
fn detect_hex_byte_vocab(vocab_json: &str) -> Result<bool> {
    const MIN_HEX_BYTES: usize = 200;
    let vocab: std::collections::HashMap<String, u32> = serde_json::from_str(vocab_json)
        .map_err(|e| CliError::ValidationFailed(format!("vocab.json is not valid JSON: {e}")))?;
    let hex_byte_count = (0u8..=255)
        .map(|b| format!("{b:02x}"))
        .filter(|hex| vocab.contains_key(hex))
        .count();
    Ok(hex_byte_count >= MIN_HEX_BYTES)
}

/// Load the GPT-2 byte-level tokenizer (from `apr tokenize import-hf` or a
/// direct tokenizer.json). Prefers a sibling tokenizer.json when present
/// (canonical HF format with added_tokens registered); falls back to
/// vocab.json + merges.txt.
///
/// NOTE on naming: `load_from_files` takes JSON STRING and MERGES STRING
/// as CONTENT (not file paths). We read the contents and pass them.
///
/// Verified: `aprender::text::bpe::load_from_json` on a real Qwen
/// tokenizer.json produces 0% unk (test
/// `falsify_bpe_qwen_encode_python_does_not_unk_99pct` in
/// `crates/aprender-core/src/text/bpe/tests_encode_decode.rs`).
#[cfg(feature = "training")]
fn load_byte_level_tokenizer(
    tokenizer_dir: &Path,
    vocab_json: &str,
    merges_path_str: &str,
) -> Result<aprender::text::bpe::BpeTokenizer> {
    let tokenizer_json_path = tokenizer_dir.join("tokenizer.json");
    if tokenizer_json_path.exists() {
        let json = std::fs::read_to_string(&tokenizer_json_path).map_err(|e| {
            CliError::ValidationFailed(format!(
                "byte-level loader: cannot read {}: {e}",
                tokenizer_json_path.display()
            ))
        })?;
        aprender::text::bpe::load_from_json(&json).map_err(|byte_err| {
            CliError::ValidationFailed(format!("byte-level loader (tokenizer.json): {byte_err}"))
        })
    } else {
        let merges_txt = std::fs::read_to_string(merges_path_str).map_err(|e| {
            CliError::ValidationFailed(format!(
                "byte-level loader: cannot read merges.txt {merges_path_str}: {e}"
            ))
        })?;
        aprender::text::bpe::load_from_files(vocab_json, &merges_txt).map_err(|byte_err| {
            CliError::ValidationFailed(format!(
                "byte-level loader (vocab.json+merges.txt): {byte_err}"
            ))
        })
    }
}

/// Two-format dispatch (PMAT-CODE-TOKENIZE-BPE-UPSTREAM-001 follow-up to
/// PR #1596's initial dispatch). Detects the vocab flavour UPFRONT, then
/// builds either the hex-byte `BPETokenizer` or the byte-level
/// `BpeTokenizer`.
#[cfg(feature = "training")]
fn load_encode_tokenizer(
    tokenizer_dir: &Path,
    vocab_path: &Path,
    merges_path: &Path,
    normalization: &str,
) -> Result<EncodeTokenizer> {
    let config = entrenar::tokenizer::TokenizerConfig::bpe()
        .with_normalization(encode_normalization(normalization));
    let vocab_path_str = tokenizer_path_str(vocab_path, "vocab.json")?;
    let merges_path_str = tokenizer_path_str(merges_path, "merges.txt")?;

    let vocab_json_for_detect = std::fs::read_to_string(&vocab_path_str).map_err(|e| {
        CliError::ValidationFailed(format!("cannot read vocab.json {vocab_path_str}: {e}"))
    })?;

    if detect_hex_byte_vocab(&vocab_json_for_detect)? {
        // Hex-byte format (legacy `apr tokenize train` output).
        entrenar::tokenizer::BPETokenizer::from_vocab_merges(
            &vocab_path_str,
            &merges_path_str,
            config,
        )
        .map(EncodeTokenizer::Hex)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot load tokenizer: {e}")))
    } else {
        load_byte_level_tokenizer(tokenizer_dir, &vocab_json_for_detect, &merges_path_str)
            .map(EncodeTokenizer::ByteLevel)
    }
}

/// SPEC §83 P2-C: collect across all `--corpus` paths into a tagged list,
/// then derive the two legacy views the rest of the run needs.
///
/// When a single `--corpus` is passed (back-compat path), behaviour is
/// byte-identical to pre-P2-C since the tagged list collapses to the
/// legacy `(files, format)` shape on the legacy iterator wrapper. When the
/// sources are mixed-format, the reported format is Parquet — its
/// per-row-group streaming semantics are what the `--estimate-only` path
/// wants; the actual encode loop dispatches per-file via
/// `iter_corpus_texts_tagged` and never consults this value.
#[cfg(feature = "training")]
struct CorpusLayout {
    /// Every input file paired with the format it was detected as, in
    /// `--corpus` argument order. Drives the real encode loop.
    tagged_files: Vec<(std::path::PathBuf, CorpusFormat)>,
    /// The same files without their tags — the legacy `files` view used by
    /// the `--estimate-only` path and by the manifest's `input_files`.
    files: Vec<std::path::PathBuf>,
    /// The single detected format, or Parquet when the sources are mixed.
    format: CorpusFormat,
}

#[cfg(feature = "training")]
fn resolve_corpus_layout(corpus: &[std::path::PathBuf]) -> Result<CorpusLayout> {
    let tagged_files = collect_corpus_files_multi(corpus)?;
    let files: Vec<std::path::PathBuf> = tagged_files.iter().map(|(p, _)| p.clone()).collect();
    let unique_formats: std::collections::HashSet<CorpusFormat> =
        tagged_files.iter().map(|(_, f)| *f).collect();
    let format = if unique_formats.len() == 1 {
        *unique_formats.iter().next().expect("non-empty")
    } else {
        CorpusFormat::Parquet
    };
    Ok(CorpusLayout {
        tagged_files,
        files,
        format,
    })
}

/// Sequential shard-writing state for `apr tokenize encode-corpus`.
///
/// Both the single-threaded and the chunked-parallel encode paths drive
/// this same struct through `push_doc`, so the output bytes for a document
/// are determined purely by (doc_index, tokenizer, eos_policy) and never by
/// the worker count — the `parallel_correctness` invariant of
/// `contracts/apr-tokenize-parallel-bpe-v1.yaml`.
#[cfg(feature = "training")]
struct ShardWriter<'a> {
    output_dir: &'a Path,
    shard_tokens: usize,
    eos_policy: &'a str,
    eos_id: Option<u32>,
    vocab_size: usize,
    writer: std::io::BufWriter<std::fs::File>,
    shard_idx: usize,
    tokens_in_shard: usize,
    total_tokens: u64,
    total_docs: u64,
    eos_count: u64,
    doc_iter_count: u64,
}

#[cfg(feature = "training")]
impl<'a> ShardWriter<'a> {
    /// Open shard 0 and zero every counter.
    fn new(
        output_dir: &'a Path,
        shard_tokens: usize,
        eos_policy: &'a str,
        eos_id: Option<u32>,
        vocab_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            output_dir,
            shard_tokens,
            eos_policy,
            eos_id,
            vocab_size,
            writer: open_shard(output_dir, 0)?,
            shard_idx: 0,
            tokens_in_shard: 0,
            total_tokens: 0,
            total_docs: 0,
            eos_count: 0,
            doc_iter_count: 0,
        })
    }

    /// Append one little-endian u32 to the open shard, updating the shard
    /// and run token counters.
    fn write_token(&mut self, id: u32) -> Result<()> {
        use std::io::Write as _;
        self.writer
            .write_all(&id.to_le_bytes())
            .map_err(|e| CliError::ValidationFailed(format!("Shard write failed: {e}")))?;
        self.tokens_in_shard += 1;
        self.total_tokens += 1;
        Ok(())
    }

    /// Append the EOS token when the tokenizer defines one. A tokenizer
    /// with no recognisable EOS id writes nothing — the pre-decomposition
    /// `if let Some(eos)` had no else arm either.
    fn write_eos(&mut self) -> Result<()> {
        if let Some(eos) = self.eos_id {
            self.write_token(eos)?;
            self.eos_count += 1;
        }
        Ok(())
    }

    /// Emit one document's encoded ids under the configured EOS policy,
    /// fail-fasting on any id outside the vocabulary (INV-PRETOK-001).
    fn emit_doc(&mut self, file_display: &str, locator: &str, ids: &[u32]) -> Result<()> {
        if self.eos_policy == "between" && self.doc_iter_count > 0 {
            self.write_eos()?;
        }

        for id in ids {
            if (*id as usize) >= self.vocab_size {
                let vocab_size = self.vocab_size;
                return Err(CliError::ValidationFailed(format!(
                    "Token id {id} >= vocab_size {vocab_size} at {file_display} {locator} \
                     (INV-PRETOK-001 violation)"
                )));
            }
            self.write_token(*id)?;
        }

        if self.eos_policy == "after" {
            self.write_eos()?;
        }

        self.doc_iter_count += 1;
        Ok(())
    }

    /// Flush and roll to the next shard once the current one has reached
    /// its token budget.
    fn rotate_if_full(&mut self) -> Result<()> {
        if self.tokens_in_shard >= self.shard_tokens {
            self.flush()?;
            self.shard_idx += 1;
            self.tokens_in_shard = 0;
            self.writer = open_shard(self.output_dir, self.shard_idx)?;
        }
        Ok(())
    }

    /// The whole per-document step, shared by both encode paths: emit the
    /// ids, count the doc, rotate the shard when full, then let the
    /// emitter decide whether this doc warrants a `[progress]` line.
    ///
    /// The OR-cadence is: emit when either N docs OR S seconds have
    /// accumulated since the last tick (issue #1547 / contract v1.2.0).
    /// `quiet=true` short-circuits inside `emit_tick`/`should_emit`.
    fn push_doc(
        &mut self,
        emitter: &mut ProgressEmitter,
        file_display: &str,
        locator: &str,
        ids: &[u32],
    ) -> Result<()> {
        self.emit_doc(file_display, locator, ids)?;
        self.total_docs += 1;
        self.rotate_if_full()?;
        let now = Instant::now();
        if emitter.should_emit(self.total_docs, now) {
            emitter.emit_tick(self.total_docs, self.total_tokens, now);
        }
        Ok(())
    }

    /// Flush the open shard's buffer to disk.
    fn flush(&mut self) -> Result<()> {
        use std::io::Write as _;
        self.writer
            .flush()
            .map_err(|e| CliError::ValidationFailed(format!("Shard flush failed: {e}")))
    }

    /// Number of shard files written so far (shards are 0-indexed).
    fn shard_count(&self) -> usize {
        self.shard_idx + 1
    }
}

/// Legacy single-threaded encode path. MUST stay byte-identical to
/// pre-#1547 output for any operator job that pinned `--num-workers 1`.
#[cfg(feature = "training")]
fn encode_corpus_single_threaded(
    source: &mut dyn Iterator<Item = Result<(String, String, String)>>,
    tokenizer: &EncodeTokenizer,
    shards: &mut ShardWriter<'_>,
    emitter: &mut ProgressEmitter,
) -> Result<()> {
    for triple in source {
        let (file_display, locator, text) = triple?;
        let ids = tokenizer.encode(&text).map_err(|e| {
            CliError::ValidationFailed(format!("Encoding failed at {file_display} {locator}: {e}"))
        })?;
        shards.push_doc(emitter, &file_display, &locator, &ids)?;
    }
    Ok(())
}

/// Pull the next chunk of documents from the canonical source iterator.
/// Errors here are JSON-parse / IO errors — surfaced before the chunk is
/// dispatched so the failing locator is precise.
#[cfg(feature = "training")]
fn next_encode_chunk(
    source: &mut dyn Iterator<Item = Result<(String, String, String)>>,
) -> Result<Vec<(String, String, String)>> {
    let mut chunk: Vec<(String, String, String)> = Vec::with_capacity(ENCODE_CHUNK_SIZE);
    for _ in 0..ENCODE_CHUNK_SIZE {
        match source.next() {
            Some(Ok(triple)) => chunk.push(triple),
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    Ok(chunk)
}

/// Parallel encode of one chunk inside the caller's rayon pool.
/// `par_iter` over a slice preserves index order in the collected `Vec`,
/// so the sequential write phase sees docs in original input order.
#[cfg(feature = "training")]
fn encode_chunk_parallel(
    pool: &rayon::ThreadPool,
    tokenizer: &EncodeTokenizer,
    chunk: &[(String, String, String)],
) -> Vec<Result<(String, String, Vec<u32>)>> {
    use rayon::prelude::*;
    pool.install(|| {
        chunk
            .par_iter()
            .map(|(file_display, locator, text)| {
                tokenizer
                    .encode(text)
                    .map(|ids| (file_display.clone(), locator.clone(), ids))
                    .map_err(|e| {
                        CliError::ValidationFailed(format!(
                            "Encoding failed at {file_display} {locator}: {e}"
                        ))
                    })
            })
            .collect()
    })
}

/// Chunked parallel encode path. Read CHUNK docs sequentially, encode them
/// in parallel via rayon (preserving chunk-local order via index), then
/// drain into the shard writer sequentially. This bounds memory at roughly
/// `CHUNK * avg_doc_token_count * 4` bytes and bounds rayon spawn overhead
/// at `total_docs / CHUNK` dispatches.
#[cfg(feature = "training")]
fn encode_corpus_parallel(
    source: &mut dyn Iterator<Item = Result<(String, String, String)>>,
    tokenizer: &EncodeTokenizer,
    shards: &mut ShardWriter<'_>,
    emitter: &mut ProgressEmitter,
    workers: usize,
) -> Result<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| {
            CliError::ValidationFailed(format!("Cannot build rayon pool ({workers}): {e}"))
        })?;

    loop {
        let chunk = next_encode_chunk(source)?;
        if chunk.is_empty() {
            break;
        }
        for result in encode_chunk_parallel(&pool, tokenizer, &chunk) {
            let (file_display, locator, ids) = result?;
            shards.push_doc(emitter, &file_display, &locator, &ids)?;
        }
    }
    Ok(())
}

/// The fixed inputs of one `encode-corpus` run that the manifest and the
/// operator summary need after the encode loop finishes. Grouped so the
/// report step takes three arguments instead of fifteen.
#[cfg(feature = "training")]
struct EncodeRun<'a> {
    corpus: &'a [std::path::PathBuf],
    files: &'a [std::path::PathBuf],
    corpus_format: CorpusFormat,
    tokenizer_dir: &'a Path,
    output_dir: &'a Path,
    content_field: &'a str,
    normalization: &'a str,
    eos_policy: &'a str,
    workers: usize,
    json_output: bool,
}

/// Build the `pretokenize-bin-v1` manifest for a finished run.
#[cfg(feature = "training")]
fn build_encode_manifest(
    run: &EncodeRun<'_>,
    shards: &ShardWriter<'_>,
    elapsed: std::time::Duration,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "pretokenize-bin-v1",
        "tokenizer_dir": run.tokenizer_dir.display().to_string(),
        "vocab_size": shards.vocab_size,
        "eos_policy": run.eos_policy,
        "eos_token_id": shards.eos_id,
        "eos_token_count": shards.eos_count,
        "shard_count": shards.shard_count(),
        "total_tokens": shards.total_tokens,
        "total_documents": shards.total_docs,
        "content_field": run.content_field,
        "normalization": run.normalization,
        "input_format": match run.corpus_format {
            CorpusFormat::Jsonl => "jsonl",
            CorpusFormat::Parquet => "parquet",
        },
        "input_files": run.files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        // SPEC §83 P2-C: `corpus_roots` tracks the distinct `--corpus`
        // arguments passed to this run. Discharges INV-MERGE-001 (≥ 2
        // sources) of contracts/corpus-merge-v3-v1.yaml at the manifest
        // level. Per-shard provenance tagging is a v1.1 follow-up.
        "corpus_roots": run.corpus.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "num_workers": run.workers,
        "elapsed_seconds": elapsed.as_secs_f64(),
    })
}

/// Write `manifest.json` and emit the operator-facing summary — the JSON
/// manifest on stdout under `--json`, the human table otherwise.
#[cfg(feature = "training")]
fn report_encode_result(
    run: &EncodeRun<'_>,
    shards: &ShardWriter<'_>,
    elapsed: std::time::Duration,
) -> Result<()> {
    let manifest = build_encode_manifest(run, shards, elapsed);
    let manifest_path = run.output_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?,
    )
    .map_err(|e| CliError::ValidationFailed(format!("Cannot write manifest: {e}")))?;

    if run.json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest)
                .map_err(|e| CliError::InvalidFormat(e.to_string()))?
        );
    } else {
        output::header("apr tokenize encode-corpus — Pretokenization Result");
        output::kv("  Shards", format_number(shards.shard_count()));
        output::kv(
            "  Total tokens",
            format_number(shards.total_tokens as usize),
        );
        output::kv(
            "  Total documents",
            format_number(shards.total_docs as usize),
        );
        output::kv("  Vocab size", format_number(shards.vocab_size));
        output::kv("  Workers", run.workers.to_string());
        output::kv("  Elapsed", format!("{:.1}s", elapsed.as_secs_f64()));
        output::kv("  Manifest", manifest_path.display().to_string());
    }

    Ok(())
}

/// Run `apr tokenize encode-corpus` — pretokenize a JSONL corpus into `.bin`
/// shards per contracts/pretokenize-bin-v1.yaml. Emits flat little-endian u32
/// streams (the exact format ShardBatchIter expects at MODEL-2 pretrain time).
///
/// Requires the `training` feature so `entrenar::tokenizer::BPETokenizer`
/// is linked; without it, encode-corpus is unavailable (matching `run_train`).
///
/// `num_workers` controls per-document BPE encoding parallelism (issue #1547).
/// `Some(1)` runs the byte-identical single-threaded legacy path; `None` uses
/// `available_parallelism`; `Some(N)` for N > 1 uses chunked rayon while
/// preserving original document order per `parallel_correctness` invariant
/// in `contracts/apr-tokenize-parallel-bpe-v1.yaml`.
#[cfg(feature = "training")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_encode_corpus(
    corpus: &[std::path::PathBuf],
    tokenizer_dir: &Path,
    output_dir: &Path,
    shard_tokens: usize,
    content_field: &str,
    normalization: &str,
    eos_policy: &str,
    num_workers: Option<usize>,
    progress: ProgressConfig,
    estimate: EstimateConfig,
    json_output: bool,
) -> Result<()> {
    let (workers, vocab_path, merges_path) = validate_encode_corpus_args(
        corpus,
        tokenizer_dir,
        shard_tokens,
        normalization,
        eos_policy,
        num_workers,
    )?;

    let tokenizer = load_encode_tokenizer(tokenizer_dir, &vocab_path, &merges_path, normalization)?;
    let vocab_size = tokenizer.vocab_size();
    let eos_id = ["</s>", "<|endoftext|>", "<eos>", "<|eos|>"]
        .iter()
        .find_map(|name| tokenizer.token_to_id(name));

    let layout = resolve_corpus_layout(corpus)?;

    // Pre-flight: --estimate-only path (issue #1547 / contract v1.3.0).
    // Sample the first `sample_docs` docs, encode them, observe (tokens,
    // wall-time-per-doc), extrapolate to the full corpus, emit the
    // `[estimate]` lines, and return without writing any output. This
    // is intentionally placed BEFORE `create_dir_all` so a dry-run never
    // touches the output directory at all (AC1).
    if estimate.enabled {
        let workers = resolve_num_workers(num_workers)?;
        return run_estimate_only_path(
            |text| tokenizer.encode(text),
            &layout.files,
            layout.format,
            content_field,
            estimate.sample_docs,
            shard_tokens,
            workers,
        );
    }

    std::fs::create_dir_all(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

    let start = Instant::now();
    let mut shards = ShardWriter::new(output_dir, shard_tokens, eos_policy, eos_id, vocab_size)?;

    // Per-doc progress emitter (issue #1547 / contract v1.2.0). The
    // total-docs hint is currently `None` — counting the input would
    // double-walk the corpus (the parquet adapter especially is not
    // free). The `format_line` path tolerates `None` and emits without
    // the `/T` and `eta=` fragments, matching AC1.
    let mut emitter = ProgressEmitter::new(progress, None);

    // Source iterator: yields (file_display, locator, text) triples in
    // canonical input order. Both the single-threaded and chunked-parallel
    // paths consume from this same iterator — the only difference is whether
    // chunks are encoded in lock-step or via rayon. SPEC §83 P2-C: use the
    // tagged dispatch so mixed-format multi-source --corpus calls (e.g.
    // codeparrot JSONL + the-stack-v2 parquet) encode correctly per file.
    let tagged_iter: Box<dyn Iterator<Item = (std::path::PathBuf, CorpusFormat)>> =
        Box::new(layout.tagged_files.iter().cloned());
    let mut source = iter_corpus_texts_tagged(tagged_iter, content_field);

    if workers <= 1 {
        encode_corpus_single_threaded(&mut source, &tokenizer, &mut shards, &mut emitter)?;
    } else {
        encode_corpus_parallel(&mut source, &tokenizer, &mut shards, &mut emitter, workers)?;
    }

    shards.flush()?;
    let elapsed = start.elapsed();

    // Final progress emission (issue #1547 §AC4 / contract v1.2.0). Suppressed
    // when `quiet=true`. The stdout/JSON summary below is always emitted —
    // this is the operator-facing stderr line.
    emitter.emit_final(shards.total_docs, shards.total_tokens);

    report_encode_result(
        &EncodeRun {
            corpus,
            files: &layout.files,
            corpus_format: layout.format,
            tokenizer_dir,
            output_dir,
            content_field,
            normalization,
            eos_policy,
            workers,
            json_output,
        },
        &shards,
        elapsed,
    )
}

#[cfg(feature = "training")]
fn open_shard(output_dir: &Path, shard_idx: usize) -> Result<std::io::BufWriter<std::fs::File>> {
    let path = output_dir.join(format!("shard-{shard_idx:05}.bin"));
    let file = std::fs::File::create(&path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot create shard {}: {e}", path.display()))
    })?;
    Ok(std::io::BufWriter::new(file))
}

// Task #103: removed `build_normalizer` — aprender-train's BPETokenizer now
// applies normalization internally via `TokenizerConfig::with_normalization`
// (commit b0e0a280b). The local NFC pass threaded by task #90 is obsolete.

fn collect_jsonl_files(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let meta = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?;

    if meta.is_file() {
        if is_jsonl(path) {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(CliError::ValidationFailed(format!(
            "Corpus file {} is not a .jsonl file",
            path.display()
        )));
    }

    let mut out = Vec::new();
    let entries = std::fs::read_dir(path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot read directory {}: {e}", path.display()))
    })?;
    for entry in entries {
        let entry =
            entry.map_err(|e| CliError::ValidationFailed(format!("Directory entry error: {e}")))?;
        let p = entry.path();
        if p.is_file() && is_jsonl(&p) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn is_jsonl(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("jsonl")
}

/// Issue #1410: Stack v1.2 / codeparrot ship as parquet, not JSONL. The
/// `apr tokenize encode-corpus` corpus argument now accepts either format.
/// Detection is by extension (in directory mode, parquet wins if both
/// extensions are present — the JSONL adapter is the legacy path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CorpusFormat {
    Jsonl,
    Parquet,
}

/// Detect corpus format and collect files. Mirrors `collect_jsonl_files`
/// for both formats, returning the chosen format alongside the file list.
///
/// File mode: extension decides format; non-`.jsonl`/`.parquet` files
/// error out.
///
/// Directory mode: parquet shards are preferred when both are present
/// (the new path); fall back to JSONL otherwise. Empty directories error.
fn collect_corpus_files(path: &Path) -> Result<(Vec<std::path::PathBuf>, CorpusFormat)> {
    let meta = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?;

    if meta.is_file() {
        if super::tokenize_parquet::is_parquet(path) {
            return Ok((vec![path.to_path_buf()], CorpusFormat::Parquet));
        }
        if is_jsonl(path) {
            return Ok((vec![path.to_path_buf()], CorpusFormat::Jsonl));
        }
        return Err(CliError::ValidationFailed(format!(
            "Corpus file {} is not a .jsonl or .parquet file",
            path.display()
        )));
    }

    let parquet = super::tokenize_parquet::collect_parquet_files(path).unwrap_or_default();
    if !parquet.is_empty() {
        return Ok((parquet, CorpusFormat::Parquet));
    }

    let jsonl = collect_jsonl_files(path)?;
    if jsonl.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "No .jsonl or .parquet files found under {}",
            path.display()
        )));
    }
    Ok((jsonl, CorpusFormat::Jsonl))
}

/// SPEC §83 P2-C multi-source corpus assembly. Calls `collect_corpus_files`
/// per source path and concatenates the results into a single tagged list
/// `[(file, format), ...]` so the downstream tokenize loop can dispatch
/// per-file regardless of mixed formats.
///
/// Contract: contracts/corpus-merge-v3-v1.yaml (INV-MERGE-001 ≥ 2 sources).
fn collect_corpus_files_multi(
    corpora: &[std::path::PathBuf],
) -> Result<Vec<(std::path::PathBuf, CorpusFormat)>> {
    if corpora.is_empty() {
        return Err(CliError::ValidationFailed(
            "At least one --corpus path is required".to_string(),
        ));
    }
    let mut tagged: Vec<(std::path::PathBuf, CorpusFormat)> = Vec::new();
    for path in corpora {
        let (files, fmt) = collect_corpus_files(path)?;
        for f in files {
            tagged.push((f, fmt));
        }
    }
    Ok(tagged)
}

/// Unified text iterator: yields `(file_display, locator, text)` triples
/// regardless of source format. `locator` is human-readable ("line 5" /
/// "row 12", 1-indexed) so error messages stay consistent across formats.
///
/// Streaming for both branches: parquet reads one row group at a time;
/// JSONL reads each file's content into memory one file at a time
/// (matches the legacy behavior — Stack v1.2 shards are ~200 MB, CSN-Python
/// shards are ~50 MB, both well under typical RAM).
#[cfg(feature = "training")]
fn iter_corpus_texts<'a>(
    files: &'a [std::path::PathBuf],
    format: CorpusFormat,
    content_field: &'a str,
) -> Box<dyn Iterator<Item = Result<(String, String, String)>> + 'a> {
    iter_corpus_texts_tagged(
        Box::new(files.iter().map(move |f| (f.clone(), format))),
        content_field,
    )
}

/// SPEC §83 P2-C multi-source variant: yields texts from a tagged list of
/// `(file, format)` pairs, dispatching per-file. Used by `run_encode_corpus`
/// when `--corpus` is repeated; the legacy single-format `iter_corpus_texts`
/// wraps this with a uniform format tag.
#[cfg(feature = "training")]
fn iter_corpus_texts_tagged<'a>(
    files: Box<dyn Iterator<Item = (std::path::PathBuf, CorpusFormat)> + 'a>,
    content_field: &'a str,
) -> Box<dyn Iterator<Item = Result<(String, String, String)>> + 'a> {
    Box::new(files.flat_map(move |(file, format)| {
        let single = vec![file];
        let inner: Box<dyn Iterator<Item = Result<(String, String, String)>>> = match format {
            CorpusFormat::Parquet => iter_corpus_texts_parquet(&single, content_field),
            CorpusFormat::Jsonl => iter_corpus_texts_jsonl(&single, content_field),
        };
        inner.collect::<Vec<_>>().into_iter()
    }))
}

/// Pull the parquet-format body out of the original `iter_corpus_texts`.
#[cfg(feature = "training")]
fn iter_corpus_texts_parquet<'a>(
    files: &'a [std::path::PathBuf],
    content_field: &'a str,
) -> Box<dyn Iterator<Item = Result<(String, String, String)>> + 'a> {
    iter_corpus_texts_old(files, CorpusFormat::Parquet, content_field)
}

/// Pull the JSONL-format body out of the original `iter_corpus_texts`.
#[cfg(feature = "training")]
fn iter_corpus_texts_jsonl<'a>(
    files: &'a [std::path::PathBuf],
    content_field: &'a str,
) -> Box<dyn Iterator<Item = Result<(String, String, String)>> + 'a> {
    iter_corpus_texts_old(files, CorpusFormat::Jsonl, content_field)
}

/// Original single-format iterator body (preserved as the per-format
/// implementation; `iter_corpus_texts` and `iter_corpus_texts_tagged`
/// both flow through here).
#[cfg(feature = "training")]
fn iter_corpus_texts_old<'a>(
    files: &'a [std::path::PathBuf],
    format: CorpusFormat,
    content_field: &'a str,
) -> Box<dyn Iterator<Item = Result<(String, String, String)>> + 'a> {
    match format {
        CorpusFormat::Parquet => Box::new(files.iter().flat_map(move |file| {
            let file_display = file.display().to_string();
            match super::tokenize_parquet::iter_parquet_content(file, content_field) {
                Ok(it) => {
                    let fd = file_display;
                    let inner: Box<dyn Iterator<Item = Result<(String, String, String)>>> =
                        Box::new(it.enumerate().map(move |(idx, r)| {
                            r.map(|t| (fd.clone(), format!("row {}", idx + 1), t))
                        }));
                    inner
                }
                Err(e) => {
                    let inner: Box<dyn Iterator<Item = Result<(String, String, String)>>> =
                        Box::new(std::iter::once(Err(e)));
                    inner
                }
            }
        })),
        CorpusFormat::Jsonl => Box::new(files.iter().flat_map(move |file| {
            let file_display = file.display().to_string();
            match std::fs::read_to_string(file) {
                Ok(content) => {
                    let fd = file_display;
                    let triples: Vec<Result<(String, String, String)>> = content
                        .lines()
                        .enumerate()
                        .filter_map(|(idx, line)| {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                return None;
                            }
                            match serde_json::from_str::<serde_json::Value>(trimmed) {
                                Ok(v) => v.get(content_field).and_then(|x| x.as_str()).map(|s| {
                                    Ok((fd.clone(), format!("line {}", idx + 1), s.to_string()))
                                }),
                                Err(e) => Some(Err(CliError::ValidationFailed(format!(
                                    "Invalid JSON in {fd} line {}: {e}",
                                    idx + 1
                                )))),
                            }
                        })
                        .collect();
                    let inner: Box<dyn Iterator<Item = Result<(String, String, String)>>> =
                        Box::new(triples.into_iter());
                    inner
                }
                Err(e) => {
                    let msg = format!("Cannot read {file_display}: {e}");
                    let inner: Box<dyn Iterator<Item = Result<(String, String, String)>>> =
                        Box::new(std::iter::once(Err(CliError::ValidationFailed(msg))));
                    inner
                }
            }
        })),
    }
}

fn read_jsonl_content(path: &Path, out: &mut Vec<String>) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read {}: {e}", path.display())))?;
    for (line_idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Invalid JSON in {} line {}: {e}",
                path.display(),
                line_idx + 1
            ))
        })?;
        if let Some(text) = value.get("content").and_then(|v| v.as_str()) {
            // Raw content — aprender-train's `BPETokenizer::preprocess` applies
            // the user-selected normalization during `train`.
            out.push(text.to_string());
        }
    }
    Ok(())
}

fn print_train_header(
    corpus: &Path,
    vocab_size: usize,
    output_dir: &Path,
    files: usize,
    lines: usize,
) {
    output::header("apr tokenize train — Training BPE Tokenizer");
    println!();
    output::kv("  Algorithm", "bpe");
    output::kv("  Vocab size", format_number(vocab_size));
    output::kv("  Corpus", corpus.display().to_string());
    output::kv("  Files", format_number(files));
    output::kv("  Lines", format_number(lines));
    output::kv("  Output", output_dir.display().to_string());
    println!();
}

fn print_train_result(result: &TokenizeTrainResult) {
    output::section("Result");
    println!(
        "  {} BPE tokenizer trained successfully",
        "OK".green().bold()
    );
    output::kv("  Final vocab size", format_number(result.vocab_size));
    output::kv("  Normalization", &result.normalization);
    output::kv(
        "  Training time",
        format!("{:.1}s", result.training_seconds),
    );
    output::kv("  vocab.json", format!("{}/vocab.json", result.output_dir));
    output::kv("  merges.txt", format!("{}/merges.txt", result.output_dir));
    println!();
}

// ─── apr tokenize import-hf ─────────────────────────────────────────────────
// Per `contracts/apr-cli-tokenize-import-hf-v1.yaml` (§50.4 step 5g.0).
//
// Extracts vocab.json + merges.txt from a HuggingFace tokenizer.json so the
// downstream `apr pretrain --tokenizer <DIR>` polymorphic preflight (per
// apr-pretrain-arch-polymorphic-v1) consumes it without modification. The
// canonical use case is fine-tuning from public Qwen2.5/Llama2/Mistral
// checkpoints which distribute as a single tokenizer.json.

/// Run `apr tokenize import-hf` — convert HF tokenizer.json into aprender's
/// vocab.json + merges.txt + manifest.json layout.
///
/// Implements `contracts/apr-cli-tokenize-import-hf-v1.yaml` §extraction_signature.
/// Falsifies FALSIFY-TOK-IMPORT-HF-001..005.
pub(crate) fn run_import_hf(
    input: &Path,
    output: &Path,
    include_added_tokens: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(CliError::FileNotFound(input.to_path_buf()));
    }

    let raw = std::fs::read_to_string(input).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] cannot read {}: {e}",
            input.display()
        ))
    })?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] {} is not valid JSON: {e}",
            input.display()
        ))
    })?;

    // Per FALSIFY-TOK-IMPORT-HF-005: only BPE inputs are accepted.
    let model_type = parsed
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "[apr-cli-tokenize-import-hf-v1] {} has no model.type field; \
                 not a recognizable HF tokenizer.json",
                input.display()
            ))
        })?;
    if model_type != "BPE" {
        return Err(CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] FALSIFY-TOK-IMPORT-HF-005: \
             model.type = '{model_type}' but only 'BPE' is supported. \
             {} cannot be imported with this subcommand. \
             Aprender's BPE loader requires GPT-2-style vocab.json + merges.txt; \
             Unigram and WordPiece use different state machines and need separate paths.",
            input.display()
        )));
    }

    // Extract model.vocab — token-string → integer-id map.
    let vocab_obj = parsed
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "[apr-cli-tokenize-import-hf-v1] {} has no model.vocab object",
                input.display()
            ))
        })?;
    let bpe_vocab_count = vocab_obj.len();

    // Extract model.merges — array of "a b" strings (or [a, b] tuples in older formats).
    let merges_arr = parsed
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "[apr-cli-tokenize-import-hf-v1] {} has no model.merges array",
                input.display()
            ))
        })?;
    let merges_count = merges_arr.len();

    let added_tokens_arr = parsed
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let added_tokens_count = added_tokens_arr.len();

    // Build the output vocab map. Default: BPE state machine only. With
    // --include-added-tokens, also include each added_token's content → id.
    let mut effective_vocab: serde_json::Map<String, serde_json::Value> = vocab_obj.clone();
    if include_added_tokens {
        for tok in &added_tokens_arr {
            if let (Some(content), Some(id)) = (
                tok.get("content").and_then(serde_json::Value::as_str),
                tok.get("id").and_then(serde_json::Value::as_u64),
            ) {
                effective_vocab.insert(
                    content.to_string(),
                    serde_json::Value::Number(serde_json::Number::from(id)),
                );
            }
        }
    }

    // Create output dir.
    std::fs::create_dir_all(output).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] cannot create output dir {}: {e}",
            output.display()
        ))
    })?;

    // Write vocab.json.
    let vocab_path = output.join("vocab.json");
    let vocab_json = serde_json::to_string_pretty(&effective_vocab)
        .map_err(|e| CliError::InvalidFormat(e.to_string()))?;
    std::fs::write(&vocab_path, vocab_json).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] cannot write {}: {e}",
            vocab_path.display()
        ))
    })?;

    // Write merges.txt — one merge per line in original order.
    let merges_path = output.join("merges.txt");
    let mut merges_body = String::from("#version: 0.2\n");
    for (idx, m) in merges_arr.iter().enumerate() {
        // Two formats are common: (a) "a b" string, (b) ["a", "b"] tuple.
        let line = match m {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) if parts.len() == 2 => {
                let a = parts[0].as_str().unwrap_or("");
                let b = parts[1].as_str().unwrap_or("");
                format!("{a} {b}")
            }
            _ => {
                return Err(CliError::ValidationFailed(format!(
                    "[apr-cli-tokenize-import-hf-v1] merges[{idx}] is neither a string \
                     nor a [a, b] tuple in {}",
                    input.display()
                )));
            }
        };
        merges_body.push_str(&line);
        merges_body.push('\n');
    }
    std::fs::write(&merges_path, merges_body).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] cannot write {}: {e}",
            merges_path.display()
        ))
    })?;

    // Write manifest.json with provenance.
    let manifest = serde_json::json!({
        "schema": "apr-cli-tokenize-import-hf-v1",
        "source": input.display().to_string(),
        "source_sha256": sha256_file(input)?,
        "model_type": "BPE",
        "bpe_vocab_count": bpe_vocab_count,
        "merges_count": merges_count,
        "added_tokens_count": added_tokens_count,
        "include_added_tokens": include_added_tokens,
        "effective_vocab_count": effective_vocab.len(),
        "extraction_timestamp_utc": chrono::Utc::now().to_rfc3339(),
    });
    let manifest_path = output.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?,
    )
    .map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] cannot write {}: {e}",
            manifest_path.display()
        ))
    })?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest)
                .map_err(|e| CliError::InvalidFormat(e.to_string()))?
        );
    } else {
        output::header("apr tokenize import-hf — HF BPE → aprender extraction");
        println!();
        output::kv("  Source", input.display().to_string());
        output::kv("  BPE vocab", format_number(bpe_vocab_count));
        output::kv("  Merges", format_number(merges_count));
        output::kv("  Added tokens", format_number(added_tokens_count));
        output::kv("  Effective vocab", format_number(effective_vocab.len()));
        output::kv("  Output dir", output.display().to_string());
        println!();
        println!("{}", "Wrote:".green().bold());
        output::kv("  vocab.json", format!("{}/vocab.json", output.display()));
        output::kv("  merges.txt", format!("{}/merges.txt", output.display()));
        output::kv(
            "  manifest.json",
            format!("{}/manifest.json", output.display()),
        );
    }

    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-cli-tokenize-import-hf-v1] cannot read {} for sha256: {e}",
            path.display()
        ))
    })?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(feature = "training")]
fn collect_shard_paths(output_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let entries = std::fs::read_dir(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] cannot read output dir {}: {e}",
            output_dir.display()
        ))
    })?;
    let mut shards: Vec<std::path::PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("shard-") && n.ends_with(".bin"))
        })
        .collect();
    shards.sort();
    Ok(shards)
}

#[cfg(feature = "training")]
fn read_vocab_size_from_tokenizer(tokenizer_dir: &Path) -> Result<usize> {
    let vocab_path = tokenizer_dir.join("vocab.json");
    let raw = std::fs::read_to_string(&vocab_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] cannot read {}: {e}",
            vocab_path.display()
        ))
    })?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] {} is not valid JSON: {e}",
            vocab_path.display()
        ))
    })?;
    let obj = parsed.as_object().ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] {} is not a JSON object",
            vocab_path.display()
        ))
    })?;
    Ok(obj.len())
}

/// Reconstruct manifest.json from existing shard-NNNN.bin files.
///
/// Falsifiers per `contracts/apr-tokenize-repair-manifest-v1.yaml`:
/// - FALSIFY-REPAIR-MANIFEST-001: shard_count == count(shard-*.bin in output_dir)
/// - FALSIFY-REPAIR-MANIFEST-002: total_tokens == Σ file_size(shard) / 4
/// - FALSIFY-REPAIR-MANIFEST-003: schema == "pretokenize-bin-v1"
/// - FALSIFY-REPAIR-MANIFEST-004: repair == true ∧ valid_rfc3339(repaired_at)
/// - FALSIFY-REPAIR-MANIFEST-005: ShardBatchIter::new succeeds after repair
/// - FALSIFY-REPAIR-MANIFEST-006: idempotent modulo repaired_at
#[cfg(feature = "training")]
pub(crate) fn run_repair_manifest(
    output_dir: &Path,
    tokenizer_dir: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    if !output_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] output dir {} does not exist or is not a directory",
            output_dir.display()
        )));
    }

    let shards = collect_shard_paths(output_dir)?;
    if shards.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] no shard-*.bin files in {} — nothing to repair",
            output_dir.display()
        )));
    }

    let mut total_bytes: u64 = 0;
    for shard in &shards {
        let meta = std::fs::metadata(shard).map_err(|e| {
            CliError::ValidationFailed(format!(
                "[apr-tokenize-repair-manifest-v1] cannot stat {}: {e}",
                shard.display()
            ))
        })?;
        let len = meta.len();
        if !len.is_multiple_of(4) {
            return Err(CliError::ValidationFailed(format!(
                "[apr-tokenize-repair-manifest-v1] {} byte length {} is not a multiple of 4 \
                 (shards are little-endian u32 streams; corrupt or non-shard file)",
                shard.display(),
                len
            )));
        }
        total_bytes += len;
    }
    let total_tokens: u64 = total_bytes / 4;
    let shard_count = shards.len();

    let vocab_size = match tokenizer_dir {
        Some(dir) => Some(read_vocab_size_from_tokenizer(dir)?),
        None => None,
    };

    let manifest = serde_json::json!({
        "schema": "pretokenize-bin-v1",
        "shard_count": shard_count,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "tokenizer_dir": tokenizer_dir.map(|p| p.display().to_string()),
        "repair": true,
        "repaired_at": chrono::Utc::now().to_rfc3339(),
        "source": "repair-manifest",
        "note": "Reconstructed from existing shard-*.bin file sizes; original \
                 encoder process exited before writing manifest.json. \
                 ShardBatchIter consumes this directory regardless — the \
                 manifest is provenance, not load-bearing.",
    });

    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?,
    )
    .map_err(|e| {
        CliError::ValidationFailed(format!(
            "[apr-tokenize-repair-manifest-v1] cannot write {}: {e}",
            manifest_path.display()
        ))
    })?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest)
                .map_err(|e| CliError::InvalidFormat(e.to_string()))?
        );
    } else {
        output::header("apr tokenize repair-manifest — Provenance Recovery");
        output::kv("  Shards", format_number(shard_count));
        output::kv("  Total tokens", format_number(total_tokens as usize));
        output::kv("  Total bytes", format_number(total_bytes as usize));
        if let Some(v) = vocab_size {
            output::kv("  Vocab size", format_number(v));
        } else {
            output::kv("  Vocab size", "(unknown — pass --tokenizer)".to_string());
        }
        output::kv("  Manifest", manifest_path.display().to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_corpus_file(dir: &Path, name: &str, lines: &[&str]) -> std::path::PathBuf {
        let p = dir.join(name);
        let body = lines.join("\n");
        std::fs::write(&p, body).expect("write corpus");
        p
    }

    #[test]
    fn run_train_happy_path_jsonl_file() {
        let tmp = TempDir::new().expect("tempdir");
        let corpus = write_corpus_file(
            tmp.path(),
            "corpus.jsonl",
            &[
                r#"{"content": "hello world hello"}"#,
                r#"{"content": "hello there world"}"#,
            ],
        );
        let out = tmp.path().join("tok");

        run_train(&corpus, 300, 1, &out, "nfc", true).expect("train");

        assert!(out.join("vocab.json").exists());
        assert!(out.join("merges.txt").exists());
        let vocab = std::fs::read_to_string(out.join("vocab.json")).expect("read vocab");
        assert!(
            vocab.contains("\"<unk>\""),
            "vocab.json missing <unk>: {}",
            vocab
        );
        let merges = std::fs::read_to_string(out.join("merges.txt")).expect("read merges");
        assert!(
            merges.starts_with("#version: 0.2"),
            "merges.txt missing header: {}",
            merges
        );
    }

    #[test]
    fn run_train_directory_corpus_walks_jsonl() {
        let tmp = TempDir::new().expect("tempdir");
        let corpus_dir = tmp.path().join("corpus");
        std::fs::create_dir_all(&corpus_dir).expect("mkdir");
        write_corpus_file(
            &corpus_dir,
            "a.jsonl",
            &[r#"{"content": "alpha beta alpha"}"#],
        );
        write_corpus_file(
            &corpus_dir,
            "b.jsonl",
            &[r#"{"content": "gamma delta gamma"}"#],
        );
        // Non-jsonl file must be ignored.
        std::fs::write(corpus_dir.join("notes.txt"), "ignore me").expect("write ignored");

        let out = tmp.path().join("tok");
        run_train(&corpus_dir, 300, 2, &out, "nfc", true).expect("train");

        assert!(out.join("vocab.json").exists());
        assert!(out.join("merges.txt").exists());
    }

    // Task #103: `--min-frequency` must actually prune low-frequency pairs.
    // Corpus has "abc" 5× (pairs 61-62 and 62-63 appear 5×) and "xyz" 1×
    // (pairs 78-79 and 79-7a appear once). With `--min-frequency 2`, only the
    // frequent pairs get merged; the singleton "xyz" pairs are left as base
    // bytes. This is the whole point of the knob — proves the CLI now honors
    // it after switching to `entrenar::tokenizer::BPETokenizer`.
    #[cfg(feature = "training")]
    #[test]
    fn run_train_honors_min_frequency_pruning() {
        let tmp = TempDir::new().expect("tempdir");
        let lines: Vec<String> = std::iter::repeat_n(r#"{"content": "abc"}"#.to_string(), 5)
            .chain(std::iter::once(r#"{"content": "xyz"}"#.to_string()))
            .collect();
        let body = lines.join("\n");
        let corpus = tmp.path().join("corpus.jsonl");
        std::fs::write(&corpus, body).expect("write corpus");
        let out = tmp.path().join("tok");

        // `vocab_size=300` leaves room for merges beyond the 256 base bytes +
        // 5 special tokens, so only `min_frequency` gates which pairs merge.
        run_train(&corpus, 300, 2, &out, "nfc", true).expect("train");

        let merges = std::fs::read_to_string(out.join("merges.txt")).expect("read merges.txt");
        // "abc" pair bytes: 61 (a), 62 (b), 63 (c). Hex representation in merges.
        assert!(
            merges.contains("61 62") || merges.contains("62 63"),
            "Expected a merge from the frequent 'abc' pair, got: {}",
            merges
        );
        // "xyz" byte pairs MUST NOT appear as merges under min_frequency=2.
        assert!(
            !merges.contains("78 79"),
            "min_frequency=2 failed to prune singleton 'xy' pair: {}",
            merges
        );
        assert!(
            !merges.contains("79 7a"),
            "min_frequency=2 failed to prune singleton 'yz' pair: {}",
            merges
        );

        // Belt-and-suspenders: confirm no merged token whose hex spells "xyz"
        // made it into the vocabulary.
        let vocab = std::fs::read_to_string(out.join("vocab.json")).expect("read vocab");
        assert!(
            !vocab.contains("\"78797a\""),
            "min_frequency=2 failed to prune merged 'xyz' token from vocab: {}",
            vocab
        );
    }

    #[test]
    fn run_train_rejects_unknown_normalization() {
        let tmp = TempDir::new().expect("tempdir");
        let corpus = write_corpus_file(tmp.path(), "corpus.jsonl", &[r#"{"content": "x y"}"#]);
        let err = run_train(&corpus, 300, 1, tmp.path(), "nfkd", true)
            .expect_err("should reject unsupported normalization");
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("nfkd")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ─── apr-cli-tokenize-import-hf-v1 falsifier tests (§50.4 step 5g.0) ───
    // FALSIFY-TOK-IMPORT-HF-002..005. (FALSIFY-001 is the dispatch surface
    // test in tests/cli_commands.rs.)

    /// Build a minimal HF tokenizer.json file in the BPE format with `n_vocab`
    /// vocab entries and `n_merges` merges. Used by the falsifier tests below.
    fn write_minimal_bpe_tokenizer_json(dir: &Path, n_vocab: usize, n_merges: usize) -> PathBuf {
        let mut vocab = serde_json::Map::new();
        for i in 0..n_vocab {
            vocab.insert(format!("tok{i}"), serde_json::Value::Number(i.into()));
        }
        let merges: Vec<serde_json::Value> = (0..n_merges)
            .map(|i| serde_json::Value::String(format!("a{i} b{i}")))
            .collect();
        let added_tokens = vec![serde_json::json!({
            "id": n_vocab,
            "content": "<|endoftext|>",
            "special": true,
        })];
        let tok = serde_json::json!({
            "version": "1.0",
            "added_tokens": added_tokens,
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
            },
        });
        let path = dir.join("tokenizer.json");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&tok).expect("serialize tok"),
        )
        .expect("write tok");
        path
    }

    /// FALSIFY-TOK-IMPORT-HF-002: BPE input produces non-empty vocab.json + merges.txt.
    #[test]
    fn import_hf_qwen_bpe_writes_vocab_and_merges() {
        let tmp = TempDir::new().expect("tempdir");
        let input = write_minimal_bpe_tokenizer_json(tmp.path(), 1000, 800);
        let output = tmp.path().join("extracted");

        run_import_hf(&input, &output, false, true).expect("import-hf should succeed on BPE input");

        let vocab_path = output.join("vocab.json");
        assert!(vocab_path.exists(), "vocab.json must exist");
        let vocab_str = std::fs::read_to_string(&vocab_path).expect("read vocab.json");
        let vocab_obj: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&vocab_str).expect("parse vocab.json");
        assert_eq!(
            vocab_obj.len(),
            1000,
            "FALSIFY-TOK-IMPORT-HF-002: vocab.json must have 1000 entries (default mode), got {}",
            vocab_obj.len()
        );

        let merges_path = output.join("merges.txt");
        assert!(merges_path.exists(), "merges.txt must exist");
        let merges_str = std::fs::read_to_string(&merges_path).expect("read merges.txt");
        let merge_lines = merges_str.lines().filter(|l| !l.starts_with('#')).count();
        assert_eq!(
            merge_lines, 800,
            "FALSIFY-TOK-IMPORT-HF-002: merges.txt must have 800 merge lines, got {merge_lines}"
        );
    }

    /// FALSIFY-TOK-IMPORT-HF-003: vocab.json entry count == |tokenizer.json:model.vocab|.
    #[test]
    fn import_hf_vocab_count_matches_input() {
        let tmp = TempDir::new().expect("tempdir");
        let input = write_minimal_bpe_tokenizer_json(tmp.path(), 12345, 100);
        let output = tmp.path().join("extracted");

        run_import_hf(&input, &output, false, true).expect("import-hf");

        let vocab_obj: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(output.join("vocab.json")).unwrap())
                .unwrap();
        assert_eq!(
            vocab_obj.len(),
            12345,
            "FALSIFY-TOK-IMPORT-HF-003: vocab count must match input model.vocab"
        );
    }

    /// FALSIFY-TOK-IMPORT-HF-004: merges.txt has one merge per line, in original order.
    #[test]
    fn import_hf_merges_format_and_order() {
        let tmp = TempDir::new().expect("tempdir");
        let input = write_minimal_bpe_tokenizer_json(tmp.path(), 10, 5);
        let output = tmp.path().join("extracted");

        run_import_hf(&input, &output, false, true).expect("import-hf");

        let body = std::fs::read_to_string(output.join("merges.txt")).expect("read merges");
        let lines: Vec<&str> = body.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(lines.len(), 5);
        // The minimal-tokenizer fixture writes "a0 b0", "a1 b1", ... in order.
        for (i, line) in lines.iter().enumerate() {
            assert_eq!(
                line.trim(),
                format!("a{i} b{i}"),
                "FALSIFY-TOK-IMPORT-HF-004: merge[{i}] order or format mismatch"
            );
        }
    }

    /// FALSIFY-TOK-IMPORT-HF-005: non-BPE input fails fast.
    #[test]
    fn import_hf_unigram_input_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let input = tmp.path().join("tokenizer.json");
        let unigram = serde_json::json!({
            "model": { "type": "Unigram", "vocab": [] },
        });
        std::fs::write(&input, serde_json::to_string_pretty(&unigram).unwrap()).unwrap();
        let output = tmp.path().join("extracted");

        let err = run_import_hf(&input, &output, false, true)
            .expect_err("FALSIFY-TOK-IMPORT-HF-005: Unigram input MUST fail-fast");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("FALSIFY-TOK-IMPORT-HF-005"),
                    "error must cite falsifier id (auditability): {msg}"
                );
                assert!(
                    msg.contains("Unigram"),
                    "error must name the actual model type: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// Sanity: --include-added-tokens incorporates added_tokens into vocab.json.
    /// Pins the §extraction_signature precondition that include_added_tokens is
    /// a non-default path (default keeps BPE state machine pure).
    #[test]
    fn import_hf_include_added_tokens_appends_specials() {
        let tmp = TempDir::new().expect("tempdir");
        let input = write_minimal_bpe_tokenizer_json(tmp.path(), 100, 50);

        // Default: no added tokens in vocab.json.
        let out_default = tmp.path().join("default");
        run_import_hf(&input, &out_default, false, true).expect("default import");
        let v_default: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(out_default.join("vocab.json")).unwrap())
                .unwrap();
        assert_eq!(v_default.len(), 100);
        assert!(
            !v_default.contains_key("<|endoftext|>"),
            "default mode must NOT include added_tokens"
        );

        // With flag: added tokens included.
        let out_full = tmp.path().join("full");
        run_import_hf(&input, &out_full, true, true).expect("full import");
        let v_full: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(out_full.join("vocab.json")).unwrap())
                .unwrap();
        assert_eq!(v_full.len(), 101);
        assert!(
            v_full.contains_key("<|endoftext|>"),
            "include-added-tokens mode must include the special"
        );
    }

    // ─── apr tokenize encode-corpus --num-workers (issue #1547) ─────────
    // SHIP-TWO-001 5g.1 unblock: per-document BPE encoding is independent
    // across rows, so chunked rayon with sequential write-phase preserves
    // both byte-identity (for `--num-workers 1`) and shard order (for any N).
    // These tests pin those properties.
    //
    // Note: FALSIFY-APR-TOK-PAR-004 (`--num-workers` advertised in
    // `apr tokenize encode-corpus --help`) is verified by the integration
    // test at `tests/falsification_apr_tok_par_004.rs` — running clap's
    // CommandFactory in-process recurses through the full Cli tree, which
    // overflows the default test stack on this binary's command surface.
    // The integration test invokes the compiled `apr` binary instead, which
    // matches the operator-facing surface 1:1 and avoids the stack issue.

    #[cfg(feature = "training")]
    #[test]
    fn encode_corpus_resolve_workers_default_is_available_parallelism() {
        // FALSIFY-#1547-DEFAULT: `None` → std::thread::available_parallelism().
        // Allow the OS-cgroup edge case where parallelism is reported as 1
        // (covered by the unwrap_or(1) fallback). The contract is "not zero
        // and matches the platform reading", not a hardcoded core count.
        let resolved = resolve_num_workers(None).expect("default must resolve");
        let expected = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        assert_eq!(
            resolved, expected,
            "default must equal available_parallelism (or 1 fallback)"
        );
        assert!(resolved >= 1, "resolved worker count must be >= 1");
    }

    #[cfg(feature = "training")]
    #[test]
    fn encode_corpus_resolve_workers_explicit_value_passes_through() {
        // Explicit Some(N) for N >= 1 returns N verbatim — operators can
        // pin worker count for memory/sharing reasons without surprise.
        assert_eq!(resolve_num_workers(Some(1)).expect("Some(1)"), 1);
        assert_eq!(resolve_num_workers(Some(4)).expect("Some(4)"), 4);
        assert_eq!(resolve_num_workers(Some(64)).expect("Some(64)"), 64);
    }

    #[cfg(feature = "training")]
    #[test]
    fn encode_corpus_resolve_workers_rejects_zero() {
        // Some(0) is a config error, not a silent fallback. Keeps the
        // failure-mode loud — a user who typed `--num-workers 0` almost
        // certainly meant something else, and silently coercing to 1 (or
        // available_parallelism) hides the typo.
        let err = resolve_num_workers(Some(0)).expect_err("zero must error");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("--num-workers"), "error must name flag: {msg}");
                assert!(msg.contains(">= 1"), "error must state bound: {msg}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// Issue #1547 byte-identity AC: `--num-workers 1` MUST produce the
    /// same shard bytes as the pre-PR single-threaded path on a small
    /// fixture, AND `--num-workers N` (N > 1) MUST produce the same shard
    /// bytes as `--num-workers 1`. The latter is the operative invariant
    /// for the SHIP-TWO-001 5g.1 retraining: parallelism MUST NOT alter
    /// the token stream the trainer ingests.
    ///
    /// Per `contracts/apr-tokenize-parallel-bpe-v1.yaml` §parallel_correctness:
    ///   ENC(jsonl_full, tok) ≡ concat(ENC(chunk_0), ..., ENC(chunk_N-1))
    /// when chunks preserve input order and BPE has no cross-row state.
    ///
    /// 10 distinct documents (avoids any "chunk-of-1 happens to be ordered"
    /// degenerate case) and a non-trivial vocab to exercise real merges.
    #[cfg(feature = "training")]
    #[test]
    fn encode_corpus_num_workers_1_matches_num_workers_n_byte_for_byte() {
        let tmp = TempDir::new().expect("tempdir");

        // Build a tokenizer in-tree so the test stays hermetic.
        let train_corpus = write_corpus_file(
            tmp.path(),
            "train.jsonl",
            &[
                r#"{"content": "hello world the quick brown fox"}"#,
                r#"{"content": "the lazy dog jumped over the fence"}"#,
                r#"{"content": "rust is a systems programming language"}"#,
            ],
        );
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");

        // 10-doc encode corpus — diverse content so chunk-internal order
        // matters (any reordering shows up as token-stream divergence).
        let encode_lines: Vec<String> = (0..10)
            .map(|i| {
                format!(
                    r#"{{"content": "doc {i} alpha beta gamma the quick brown fox jumps {i}"}}"#
                )
            })
            .collect();
        let encode_refs: Vec<&str> = encode_lines.iter().map(String::as_str).collect();
        let corpus = write_corpus_file(tmp.path(), "encode.jsonl", &encode_refs);

        // Single-threaded reference output.
        let out_1 = tmp.path().join("out_1");
        run_encode_corpus(
            std::slice::from_ref(&corpus),
            &tok_dir,
            &out_1,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig::default(),
            true,
        )
        .expect("encode --num-workers 1");

        // Parallel output (4 workers — > 1 forces the rayon path).
        let out_n = tmp.path().join("out_n");
        run_encode_corpus(
            std::slice::from_ref(&corpus),
            &tok_dir,
            &out_n,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(4),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig::default(),
            true,
        )
        .expect("encode --num-workers 4");

        // Compare every shard byte-for-byte. The manifest gains a
        // `num_workers` field (additive metadata) so we deliberately only
        // diff the .bin shards — they encode the load-bearing invariant.
        let shards_1: Vec<_> = std::fs::read_dir(&out_1)
            .expect("read out_1")
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().and_then(std::ffi::OsStr::to_str) == Some("bin"))
            .collect();
        let shards_n: Vec<_> = std::fs::read_dir(&out_n)
            .expect("read out_n")
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().and_then(std::ffi::OsStr::to_str) == Some("bin"))
            .collect();
        assert_eq!(
            shards_1.len(),
            shards_n.len(),
            "shard count must match across worker counts"
        );
        assert!(!shards_1.is_empty(), "test must produce at least one shard");

        let mut names_1: Vec<String> = shards_1
            .iter()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        let mut names_n: Vec<String> = shards_n
            .iter()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names_1.sort();
        names_n.sort();
        assert_eq!(
            names_1, names_n,
            "shard filenames must match (deterministic naming)"
        );

        for name in &names_1 {
            let bytes_1 = std::fs::read(out_1.join(name)).expect("read single-threaded shard");
            let bytes_n = std::fs::read(out_n.join(name)).expect("read parallel shard");
            assert_eq!(
                bytes_1, bytes_n,
                "FALSIFY-#1547-PARITY: shard {name} must be byte-identical \
                 between --num-workers 1 and --num-workers 4"
            );
        }
    }

    // ─── apr tokenize encode-corpus --quiet / --progress-* (issue #1547,
    //      contract apr-tokenize-parallel-bpe-v1.yaml v1.2.0) ─────────────
    // Per-doc progress emission obeys an OR-cadence: emit when EITHER N
    // docs OR S seconds have elapsed since the last tick. The unit tests
    // below pin the predicate (`should_emit`) directly so we don't have
    // to scrape stderr — that's the operator-facing wire format and is
    // already covered by `format_line` plus a dedicated stdout-shape
    // assertion. `quiet` pins suppression at every layer.

    #[cfg(feature = "training")]
    #[test]
    fn progress_emit_every_n_docs_when_under_seconds_window() {
        // FALSIFY-APR-TOK-PAR-007: doc-tick branch fires when N docs have
        // accumulated since last emit, even though the wall-clock window
        // hasn't elapsed yet. Use a 1000-doc / 60s default config and
        // simulate a fast-running encode where 1500 docs land in ~0s.
        let cfg = ProgressConfig {
            quiet: false,
            interval_docs: 1000,
            interval_seconds: 60,
        };
        let emitter = ProgressEmitter::new(cfg, None);
        let now = emitter.start; // simulated "no time has passed"

        // Below threshold: must NOT trigger.
        assert!(
            !emitter.should_emit(999, now),
            "999 docs (< 1000 threshold) must not trigger doc-tick emission"
        );
        // At threshold: MUST trigger (first emit on the boundary).
        assert!(
            emitter.should_emit(1000, now),
            "1000 docs (== threshold) must trigger doc-tick emission"
        );
        // Far above: still fires.
        assert!(
            emitter.should_emit(5000, now),
            "5000 docs must trigger doc-tick emission"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn progress_emit_every_n_seconds_when_under_docs_window() {
        // FALSIFY-APR-TOK-PAR-008: time-tick branch fires when S seconds
        // have elapsed even though doc count is below the doc threshold.
        // Set interval_docs huge so only the time path can trigger.
        let cfg = ProgressConfig {
            quiet: false,
            interval_docs: 1_000_000,
            interval_seconds: 1, // 1s threshold so we can simulate easily
        };
        let emitter = ProgressEmitter::new(cfg, None);

        // Just-now: must NOT trigger.
        let now0 = emitter.start;
        assert!(
            !emitter.should_emit(10, now0),
            "0s elapsed must not trigger time-tick emission"
        );

        // Past 1s: MUST trigger even at 10 docs (well below the 1M doc
        // threshold). This pins the OR-cadence: time alone is enough.
        let now1 = emitter.start + std::time::Duration::from_secs(1);
        assert!(
            emitter.should_emit(10, now1),
            "1s elapsed must trigger time-tick emission even with only 10 docs"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn progress_quiet_flag_suppresses_emission() {
        // FALSIFY-APR-TOK-PAR-009: --quiet must suppress emission at the
        // predicate level. Even at the boundary of both bounds, quiet=true
        // returns false so no stderr line is ever generated. This is the
        // operative invariant for CI callers that scrape logs.
        let cfg = ProgressConfig {
            quiet: true,
            interval_docs: 1,
            interval_seconds: 1,
        };
        let emitter = ProgressEmitter::new(cfg, None);

        // Even at the trigger boundary for both sub-conditions, quiet
        // wins.
        let now = emitter.start + std::time::Duration::from_secs(120);
        assert!(
            !emitter.should_emit(10_000, now),
            "quiet=true must suppress emission regardless of doc/time window"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn progress_format_line_no_total_omits_eta_fragment() {
        // AC1: when the total-docs hint is absent, the per-tick line must
        // emit `doc=N rate=X.X docs/s` without the `/T` and `eta=` parts.
        let cfg = ProgressConfig::default();
        let emitter = ProgressEmitter::new(cfg, None);
        let now = emitter.start + std::time::Duration::from_secs(10);
        let line = emitter.format_line(2000, 50_000, now);
        assert!(line.starts_with("[progress] "), "expected prefix: {line}");
        assert!(line.contains("doc=2000"), "doc count missing: {line}");
        assert!(
            !line.contains("doc=2000/"),
            "must not include /T fragment when total unknown: {line}"
        );
        assert!(
            !line.contains("eta="),
            "must not include eta= when total unknown: {line}"
        );
        assert!(
            line.contains("tokens=50000"),
            "tokens count missing: {line}"
        );
        assert!(
            line.contains("rate=") && line.contains("docs/s"),
            "rate fragment missing: {line}"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn progress_format_line_with_total_includes_eta_fragment() {
        // AC1: when total is known, line must include `doc=N/T` and an
        // `eta=YYYY-MM-DDTHH:MM:SSZ` ISO-8601 UTC timestamp.
        let cfg = ProgressConfig::default();
        let emitter = ProgressEmitter::new(cfg, Some(10_000));
        let now = emitter.start + std::time::Duration::from_secs(5);
        let line = emitter.format_line(1000, 25_000, now);
        assert!(
            line.contains("doc=1000/10000"),
            "doc/total fragment missing: {line}"
        );
        assert!(line.contains("eta="), "eta fragment missing: {line}");
        // ISO-8601 anchor: must end in `Z` (UTC).
        let eta_idx = line.find("eta=").expect("eta= present");
        let after_eta = &line[eta_idx + 4..];
        assert!(
            after_eta.contains('T') && after_eta.trim_end().ends_with('Z'),
            "eta must be ISO-8601 UTC (`...T...Z`): {line}"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn progress_mark_emitted_resets_both_clocks() {
        // After an emit, both the doc tick and the time tick must reset
        // so the NEXT emit requires another full interval. This is what
        // makes the OR-cadence correct: a doc-triggered emit also resets
        // the time clock.
        let cfg = ProgressConfig {
            quiet: false,
            interval_docs: 1000,
            interval_seconds: 60,
        };
        let mut emitter = ProgressEmitter::new(cfg, None);
        let t0 = emitter.start;

        // First emission at 1000 docs / 0s.
        assert!(emitter.should_emit(1000, t0));
        emitter.mark_emitted(1000, t0);

        // Immediately after: 1500 docs is only +500 since last tick → no.
        assert!(
            !emitter.should_emit(1500, t0),
            "1500 - 1000 = 500 < 1000 threshold; must not re-emit"
        );

        // 2000 docs → +1000 since last tick → yes.
        assert!(
            emitter.should_emit(2000, t0),
            "2000 - 1000 = 1000 >= threshold; must re-emit"
        );
    }

    // ─── apr tokenize encode-corpus --estimate-only (issue #1547,
    //      contract apr-tokenize-parallel-bpe-v1.yaml v1.3.0) ─────────────
    // Pre-flight extrapolation. AC1: no shards; AC2: no manifest;
    // AC3: emits [estimate] lines on stderr; AC4: extrapolation formula
    // respects --num-workers.

    #[cfg(feature = "training")]
    #[test]
    fn estimate_only_extrapolation_formula_correct() {
        // FALSIFY-APR-TOK-PAR-011: pure-function extrapolation kernel.
        // Sample: 1000 docs took 1.0s and produced 50_000 tokens → 50
        // tokens/doc, 1ms wall/doc. Total: 100_000 docs → 5_000_000
        // tokens. With shard_tokens=1_000_000 → 5 shards. With 4
        // workers → 1ms × 100_000 / 4 = 25 seconds wall.
        let (tokens, shards, wall) = extrapolate_estimate(
            1000,      // sample_size
            50_000,    // sample_tokens
            1.0,       // sample_wall_seconds
            100_000,   // total_docs
            1_000_000, // shard_tokens
            4,         // num_workers
        );
        assert_eq!(tokens, 5_000_000, "estimated_total_tokens math");
        assert_eq!(shards, 5, "estimated_shards must be ceil(total/shard)");
        assert!(
            (wall - 25.0).abs() < 0.01,
            "estimated_wall = wall_per_doc × total_docs / num_workers; got {wall}"
        );

        // Edge: 0 sample_size → all zeros (no extrapolation possible).
        assert_eq!(extrapolate_estimate(0, 0, 0.0, 100, 1000, 4), (0, 0, 0.0));

        // Edge: 0 num_workers must clamp to 1 (avoid divide-by-zero).
        let (_, _, wall_zero_workers) =
            extrapolate_estimate(1000, 50_000, 1.0, 100_000, 1_000_000, 0);
        assert!(
            (wall_zero_workers - 100.0).abs() < 0.01,
            "0 workers must clamp to 1; got {wall_zero_workers}"
        );

        // Edge: shard_tokens=0 → 0 estimated_shards (avoid div-by-zero
        // and misleading numbers — operator should re-run with the
        // real --shard-tokens to get a real estimate).
        let (_, shards_zero, _) = extrapolate_estimate(1000, 50_000, 1.0, 100_000, 0, 4);
        assert_eq!(
            shards_zero, 0,
            "shard_tokens=0 must yield 0 estimated_shards"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn estimate_only_no_shards_written() {
        // FALSIFY-APR-TOK-PAR-012: --estimate-only must NOT produce any
        // .bin shard files in the (would-be) output directory. Since
        // create_dir_all is gated behind the estimate short-circuit,
        // the directory itself must also not exist after the call.
        let tmp = TempDir::new().expect("tempdir");

        // Build a minimal tokenizer in-tree (same pattern as the byte-
        // identity test; keeps the suite hermetic).
        let train_corpus = write_corpus_file(
            tmp.path(),
            "train.jsonl",
            &[
                r#"{"content": "alpha beta gamma delta"}"#,
                r#"{"content": "epsilon zeta eta theta"}"#,
            ],
        );
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");

        // 50-doc encode corpus. We'll sample 10 of them in --estimate-only
        // and the remaining 40 must NEVER be encoded (no shards written).
        let encode_lines: Vec<String> = (0..50)
            .map(|i| format!(r#"{{"content": "doc {i} alpha beta gamma {i}"}}"#))
            .collect();
        let encode_refs: Vec<&str> = encode_lines.iter().map(String::as_str).collect();
        let corpus = write_corpus_file(tmp.path(), "encode.jsonl", &encode_refs);

        // Run with --estimate-only. The output dir is the path that
        // would have been created — we'll assert nothing was placed
        // there.
        let out = tmp.path().join("out_estimate");
        run_encode_corpus(
            std::slice::from_ref(&corpus),
            &tok_dir,
            &out,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig {
                enabled: true,
                sample_docs: 10,
            },
            true,
        )
        .expect("estimate-only must succeed without writing shards");

        // AC1: no .bin files (the dir may not even exist).
        if out.exists() {
            let bins: Vec<_> = std::fs::read_dir(&out)
                .expect("read estimate out dir")
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().and_then(std::ffi::OsStr::to_str) == Some("bin"))
                .collect();
            assert!(
                bins.is_empty(),
                "FALSIFY-APR-TOK-PAR-012: --estimate-only produced {} \
                 shard(s); estimate is supposed to write nothing",
                bins.len()
            );
        }
    }

    #[cfg(feature = "training")]
    #[test]
    fn estimate_only_no_manifest_written() {
        // FALSIFY-APR-TOK-PAR-013: --estimate-only must NOT produce a
        // manifest.json in the output directory.
        let tmp = TempDir::new().expect("tempdir");

        let train_corpus = write_corpus_file(
            tmp.path(),
            "train.jsonl",
            &[r#"{"content": "alpha beta gamma"}"#],
        );
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");

        let encode_lines: Vec<String> = (0..20)
            .map(|i| format!(r#"{{"content": "doc {i}"}}"#))
            .collect();
        let encode_refs: Vec<&str> = encode_lines.iter().map(String::as_str).collect();
        let corpus = write_corpus_file(tmp.path(), "encode.jsonl", &encode_refs);

        let out = tmp.path().join("out_no_manifest");
        run_encode_corpus(
            std::slice::from_ref(&corpus),
            &tok_dir,
            &out,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig {
                enabled: true,
                sample_docs: 5,
            },
            true,
        )
        .expect("estimate-only must succeed");

        // AC2: manifest.json absence — either dir doesn't exist, or
        // exists but doesn't contain manifest.json.
        let manifest = out.join("manifest.json");
        assert!(
            !manifest.exists(),
            "FALSIFY-APR-TOK-PAR-013: --estimate-only produced manifest.json at {}",
            manifest.display()
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn estimate_only_emits_estimate_lines_to_stderr() {
        // FALSIFY-APR-TOK-PAR-014: wiring check — --estimate-only must
        // run the estimate kernel and exit Ok(()). The actual stderr
        // line format is verified by the AC4 formula test above; here
        // we only confirm the path returns success on a small fixture.
        let tmp = TempDir::new().expect("tempdir");

        let train_corpus = write_corpus_file(
            tmp.path(),
            "train.jsonl",
            &[r#"{"content": "alpha beta gamma delta epsilon"}"#],
        );
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");

        let encode_lines: Vec<String> = (0..15)
            .map(|i| format!(r#"{{"content": "doc {i} content"}}"#))
            .collect();
        let encode_refs: Vec<&str> = encode_lines.iter().map(String::as_str).collect();
        let corpus = write_corpus_file(tmp.path(), "encode.jsonl", &encode_refs);

        let out = tmp.path().join("out_lines");
        let result = run_encode_corpus(
            std::slice::from_ref(&corpus),
            &tok_dir,
            &out,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig {
                enabled: true,
                sample_docs: 5,
            },
            true,
        );
        assert!(
            result.is_ok(),
            "FALSIFY-APR-TOK-PAR-014: --estimate-only path must return Ok(()) \
             on a valid corpus + tokenizer; got {result:?}"
        );
    }

    #[cfg(feature = "training")]
    #[test]
    fn estimate_only_rejects_zero_sample_size() {
        // Belt-and-suspenders: --estimate-sample-docs must be >= 1.
        // The clap default is 1000 so this only triggers when an
        // operator passes `--estimate-sample-docs 0` deliberately.
        let tmp = TempDir::new().expect("tempdir");

        let train_corpus =
            write_corpus_file(tmp.path(), "train.jsonl", &[r#"{"content": "alpha"}"#]);
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");

        let corpus = write_corpus_file(tmp.path(), "encode.jsonl", &[r#"{"content": "doc"}"#]);

        let out = tmp.path().join("out_zero_sample");
        let err = run_encode_corpus(
            std::slice::from_ref(&corpus),
            &tok_dir,
            &out,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig {
                enabled: true,
                sample_docs: 0, // operator typo → must error
            },
            true,
        )
        .expect_err("sample_docs=0 must error");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("--estimate-sample-docs"),
                    "error must name flag: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// SPEC §83 P2-C: when `--corpus` is repeated with multiple paths,
    /// `run_encode_corpus` MUST collect documents from every source and
    /// write them as a single contiguous shard stream. Verifies the
    /// multi-source extension that unblocks PMAT-681 corpus widening.
    #[test]
    fn encode_corpus_accepts_multiple_corpus_paths() {
        let tmp = TempDir::new().expect("tempdir");
        // Train a tiny tokenizer on a generic vocab.
        let train_corpus = write_corpus_file(
            tmp.path(),
            "train.jsonl",
            &[r#"{"content": "alpha beta gamma delta epsilon"}"#],
        );
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");

        // Two distinct JSONL corpora — different content per source so
        // the merged output can be inspected for both contributions.
        let corpus_a = write_corpus_file(
            tmp.path(),
            "src_a.jsonl",
            &[
                r#"{"content": "alpha alpha alpha"}"#,
                r#"{"content": "alpha beta"}"#,
            ],
        );
        let corpus_b = write_corpus_file(
            tmp.path(),
            "src_b.jsonl",
            &[r#"{"content": "gamma delta epsilon"}"#],
        );
        let out = tmp.path().join("merged_out");

        let result = run_encode_corpus(
            &[corpus_a.clone(), corpus_b.clone()],
            &tok_dir,
            &out,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig::default(),
            true,
        );
        assert!(
            result.is_ok(),
            "multi-corpus encode must succeed: {result:?}"
        );

        // Manifest reports total_docs that sums BOTH sources (2 + 1 = 3).
        let manifest_path = out.join("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path).expect("manifest written");
        let manifest: serde_json::Value =
            serde_json::from_slice(&manifest_bytes).expect("manifest is valid JSON");
        let total_documents = manifest
            .get("total_documents")
            .and_then(|v| v.as_u64())
            .expect("manifest has total_documents");
        assert_eq!(
            total_documents, 3,
            "merged corpus must contain 3 docs (2 from src_a + 1 from src_b), got {total_documents}"
        );
        // Manifest also tracks input_files — must list both sources.
        let input_files = manifest
            .get("input_files")
            .and_then(|v| v.as_array())
            .expect("manifest has input_files");
        assert!(
            input_files.len() >= 2,
            "input_files must list at least 2 source files, got {}",
            input_files.len(),
        );
        // SPEC §83 P2-C: manifest must list distinct --corpus roots
        // (INV-MERGE-001 of corpus-merge-v3-v1).
        let corpus_roots = manifest
            .get("corpus_roots")
            .and_then(|v| v.as_array())
            .expect("manifest has corpus_roots");
        assert_eq!(
            corpus_roots.len(),
            2,
            "corpus_roots must list exactly 2 sources (src_a + src_b), got {}",
            corpus_roots.len(),
        );
        let root_strs: Vec<&str> = corpus_roots.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            root_strs.iter().any(|s| s.contains("src_a.jsonl")),
            "corpus_roots must reference src_a, got {root_strs:?}",
        );
        assert!(
            root_strs.iter().any(|s| s.contains("src_b.jsonl")),
            "corpus_roots must reference src_b, got {root_strs:?}",
        );
    }

    /// SPEC §83 P2-C: empty `--corpus` slice must error fast (not silently
    /// produce an empty output corpus).
    #[test]
    fn encode_corpus_rejects_empty_corpus_list() {
        let tmp = TempDir::new().expect("tempdir");
        let train_corpus =
            write_corpus_file(tmp.path(), "train.jsonl", &[r#"{"content": "alpha"}"#]);
        let tok_dir = tmp.path().join("tok");
        run_train(&train_corpus, 400, 1, &tok_dir, "nfc", true).expect("train tokenizer");
        let out = tmp.path().join("out_empty");

        let err = run_encode_corpus(
            &[],
            &tok_dir,
            &out,
            10_000_000,
            "content",
            "nfc",
            "between",
            Some(1),
            ProgressConfig {
                quiet: true,
                ..ProgressConfig::default()
            },
            EstimateConfig::default(),
            true,
        )
        .expect_err("empty --corpus slice must error");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("--corpus") || msg.contains("required"),
                    "error must mention missing corpus: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-125 B4: pure validator / formatter coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_algorithm_supported() {
        assert!(validate_algorithm("bpe").is_ok());
        assert!(validate_algorithm("wordpiece").is_ok());
        assert!(validate_algorithm("unigram").is_ok());
    }

    #[test]
    fn test_validate_algorithm_rejects_unknown() {
        let err = validate_algorithm("sentencepiece").unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("sentencepiece"));
                assert!(msg.contains("bpe"));
            }
            other => panic!("unexpected: {other:?}"),
        }
        assert!(validate_algorithm("").is_err());
        assert!(validate_algorithm("BPE").is_err()); // case-sensitive
    }

    #[test]
    fn test_validate_vocab_size_bounds() {
        assert!(validate_vocab_size(10).is_ok());
        assert!(validate_vocab_size(32_000).is_ok());
        assert!(validate_vocab_size(1_000_000).is_ok());
    }

    #[test]
    fn test_validate_vocab_size_too_small() {
        let err = validate_vocab_size(9).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("at least 10")),
            other => panic!("unexpected: {other:?}"),
        }
        assert!(validate_vocab_size(0).is_err());
    }

    #[test]
    fn test_validate_vocab_size_too_large() {
        let err = validate_vocab_size(1_000_001).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("unreasonably large")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_validate_normalization_ok_and_err() {
        assert!(validate_normalization("none").is_ok());
        assert!(validate_normalization("nfc").is_ok());
        let err = validate_normalization("nfkc").unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("nfkc"));
                assert!(msg.contains("none, nfc"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_estimate_training_time_scales_with_vocab() {
        let one_mb = 1024 * 1024;
        let base = estimate_training_time(one_mb, 16_000);
        assert!((base - (1.0 / 60.0)).abs() < 1e-6);
        let big = estimate_training_time(one_mb, 64_000);
        assert!((big - base * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_training_time_zero_bytes() {
        assert_eq!(estimate_training_time(0, 32_000), 0.0);
    }

    #[test]
    fn test_plan_verdict_blocked_empty() {
        let stats = CorpusStats {
            lines: 0,
            bytes: 0,
            unique_chars: 0,
        };
        assert_eq!(plan_verdict(&stats, 32_000), "blocked");
    }

    #[test]
    fn test_plan_verdict_warning_vocab_too_large_for_chars() {
        let stats = CorpusStats {
            lines: 5,
            bytes: 100,
            unique_chars: 10,
        };
        assert_eq!(plan_verdict(&stats, 5_000), "warning"); // 5000 > 1000
    }

    #[test]
    fn test_plan_verdict_ready() {
        let stats = CorpusStats {
            lines: 100,
            bytes: 10_000,
            unique_chars: 80,
        };
        assert_eq!(plan_verdict(&stats, 5_000), "ready"); // 5000 <= 8000
    }

    #[test]
    fn test_format_number_thresholds() {
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_000), "1.0K");
        assert_eq!(format_number(1_500), "1.5K");
        assert_eq!(format_number(999_999), "1000.0K");
        assert_eq!(format_number(1_000_000), "1.0M");
        assert_eq!(format_number(2_500_000), "2.5M");
        assert_eq!(format_number(0), "0");
    }

    #[test]
    fn test_format_bytes_tokenize_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
        assert_eq!(format_bytes(5 * 1_073_741_824), "5.0 GB");
    }

    #[test]
    fn test_format_duration_buckets() {
        assert_eq!(format_duration(0.5), "30 sec");
        assert_eq!(format_duration(1.0), "1.0 min");
        assert_eq!(format_duration(45.0), "45.0 min");
        assert_eq!(format_duration(90.0), "1.5 hours");
        assert_eq!(format_duration(120.0), "2.0 hours");
    }

    #[test]
    fn test_resolve_num_workers_explicit() {
        assert_eq!(resolve_num_workers(Some(4)).unwrap(), 4);
        assert_eq!(resolve_num_workers(Some(1)).unwrap(), 1);
    }

    #[test]
    fn test_resolve_num_workers_zero_is_error() {
        let err = resolve_num_workers(Some(0)).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains(">= 1")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_resolve_num_workers_none_uses_parallelism() {
        let n = resolve_num_workers(None).unwrap();
        assert!(n >= 1);
    }

    #[test]
    fn test_is_jsonl_extension() {
        assert!(is_jsonl(Path::new("corpus.jsonl")));
        assert!(is_jsonl(Path::new("/a/b/data.jsonl")));
        assert!(!is_jsonl(Path::new("corpus.json")));
        assert!(!is_jsonl(Path::new("corpus.parquet")));
        assert!(!is_jsonl(Path::new("corpus")));
        assert!(!is_jsonl(Path::new("corpus.JSONL"))); // case-sensitive
    }

    #[test]
    fn test_format_eta_iso8601_utc_shape_and_determinism() {
        let s = format_eta_iso8601_utc(0);
        assert_eq!(s.len(), 20); // YYYY-MM-DDTHH:MM:SSZ
        assert!(s.ends_with('Z'));
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[13..14], ":");
        let later = format_eta_iso8601_utc(3600);
        assert!(later >= s);
    }

    #[test]
    fn test_format_eta_iso8601_utc_known_epoch_offset() {
        let a = format_eta_iso8601_utc(0);
        let b = format_eta_iso8601_utc(86_400);
        assert_ne!(&a[0..10], &b[0..10]); // date advances by a day
        assert_eq!(&a[11..19], &b[11..19]); // same HH:MM:SS
    }
}
