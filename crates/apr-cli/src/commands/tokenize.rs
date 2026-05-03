//! `apr tokenize plan/apply` — BPE vocabulary training pipeline.
//!
//! Plan validates the corpus and estimates training time.
//! Apply trains a BPE tokenizer and writes vocab.json + merges.txt.

use colored::Colorize;
use std::path::Path;
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

/// Run `apr tokenize encode-corpus` — pretokenize a JSONL corpus into `.bin`
/// shards per contracts/pretokenize-bin-v1.yaml. Emits flat little-endian u32
/// streams (the exact format ShardBatchIter expects at MODEL-2 pretrain time).
///
/// Requires the `training` feature so `entrenar::tokenizer::BPETokenizer`
/// is linked; without it, encode-corpus is unavailable (matching `run_train`).
#[cfg(feature = "training")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_encode_corpus(
    corpus: &Path,
    tokenizer_dir: &Path,
    output_dir: &Path,
    shard_tokens: usize,
    content_field: &str,
    normalization: &str,
    eos_policy: &str,
    json_output: bool,
) -> Result<()> {
    use entrenar::tokenizer::{BPETokenizer, Normalization, Tokenizer, TokenizerConfig};
    use std::io::Write as IoWrite;

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
    if !corpus.exists() {
        return Err(CliError::FileNotFound(corpus.to_path_buf()));
    }
    let vocab_path = tokenizer_dir.join("vocab.json");
    let merges_path = tokenizer_dir.join("merges.txt");
    if !vocab_path.exists() {
        return Err(CliError::FileNotFound(vocab_path));
    }
    if !merges_path.exists() {
        return Err(CliError::FileNotFound(merges_path));
    }

    let norm = match normalization {
        "nfc" => Normalization::NFC,
        "none" => Normalization::None,
        _ => unreachable!("validated above"),
    };
    let config = TokenizerConfig::bpe().with_normalization(norm);
    let tokenizer = BPETokenizer::from_vocab_merges(
        vocab_path.to_str().ok_or_else(|| {
            CliError::ValidationFailed("vocab.json path has non-utf8 bytes".to_string())
        })?,
        merges_path.to_str().ok_or_else(|| {
            CliError::ValidationFailed("merges.txt path has non-utf8 bytes".to_string())
        })?,
        config,
    )
    .map_err(|e| CliError::ValidationFailed(format!("Cannot load tokenizer: {e}")))?;
    let vocab_size = tokenizer.vocab_size();
    let eos_id = ["</s>", "<|endoftext|>", "<eos>", "<|eos|>"]
        .iter()
        .find_map(|name| tokenizer.token_to_id(name));

    let (files, corpus_format) = collect_corpus_files(corpus)?;

    std::fs::create_dir_all(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

    let start = Instant::now();
    let mut shard_idx: usize = 0;
    let mut tokens_in_shard: usize = 0;
    let mut total_tokens: u64 = 0;
    let mut total_docs: u64 = 0;
    let mut eos_count: u64 = 0;
    let mut writer = open_shard(output_dir, shard_idx)?;
    let mut doc_iter_count: u64 = 0;

    for triple in iter_corpus_texts(&files, corpus_format, content_field) {
        let (file_display, locator, text) = triple?;
        let ids = tokenizer.encode(&text).map_err(|e| {
            CliError::ValidationFailed(format!("Encoding failed at {file_display} {locator}: {e}"))
        })?;

        if eos_policy == "between" && doc_iter_count > 0 {
            if let Some(eos) = eos_id {
                writer
                    .write_all(&eos.to_le_bytes())
                    .map_err(|e| CliError::ValidationFailed(format!("Shard write failed: {e}")))?;
                tokens_in_shard += 1;
                total_tokens += 1;
                eos_count += 1;
            }
        }

        for id in &ids {
            if (*id as usize) >= vocab_size {
                return Err(CliError::ValidationFailed(format!(
                    "Token id {id} >= vocab_size {vocab_size} at {file_display} {locator} \
                     (INV-PRETOK-001 violation)"
                )));
            }
            writer
                .write_all(&id.to_le_bytes())
                .map_err(|e| CliError::ValidationFailed(format!("Shard write failed: {e}")))?;
            tokens_in_shard += 1;
            total_tokens += 1;
        }

        if eos_policy == "after" {
            if let Some(eos) = eos_id {
                writer
                    .write_all(&eos.to_le_bytes())
                    .map_err(|e| CliError::ValidationFailed(format!("Shard write failed: {e}")))?;
                tokens_in_shard += 1;
                total_tokens += 1;
                eos_count += 1;
            }
        }

        doc_iter_count += 1;
        total_docs += 1;

        if tokens_in_shard >= shard_tokens {
            writer
                .flush()
                .map_err(|e| CliError::ValidationFailed(format!("Shard flush failed: {e}")))?;
            shard_idx += 1;
            tokens_in_shard = 0;
            writer = open_shard(output_dir, shard_idx)?;
        }
    }
    writer
        .flush()
        .map_err(|e| CliError::ValidationFailed(format!("Shard flush failed: {e}")))?;
    let shard_count = shard_idx + 1;
    let elapsed = start.elapsed();

    let manifest = serde_json::json!({
        "schema": "pretokenize-bin-v1",
        "tokenizer_dir": tokenizer_dir.display().to_string(),
        "vocab_size": vocab_size,
        "eos_policy": eos_policy,
        "eos_token_id": eos_id,
        "eos_token_count": eos_count,
        "shard_count": shard_count,
        "total_tokens": total_tokens,
        "total_documents": total_docs,
        "content_field": content_field,
        "normalization": normalization,
        "input_format": match corpus_format {
            CorpusFormat::Jsonl => "jsonl",
            CorpusFormat::Parquet => "parquet",
        },
        "input_files": files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "elapsed_seconds": elapsed.as_secs_f64(),
    });
    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?,
    )
    .map_err(|e| CliError::ValidationFailed(format!("Cannot write manifest: {e}")))?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest)
                .map_err(|e| CliError::InvalidFormat(e.to_string()))?
        );
    } else {
        output::header("apr tokenize encode-corpus — Pretokenization Result");
        output::kv("  Shards", format_number(shard_count));
        output::kv("  Total tokens", format_number(total_tokens as usize));
        output::kv("  Total documents", format_number(total_docs as usize));
        output::kv("  Vocab size", format_number(vocab_size));
        output::kv("  Elapsed", format!("{:.1}s", elapsed.as_secs_f64()));
        output::kv("  Manifest", manifest_path.display().to_string());
    }

    Ok(())
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}
