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

    let normalizer = build_normalizer(normalization);
    let mut lines: Vec<String> = Vec::new();
    for file in &files {
        read_jsonl_content(file, &normalizer, &mut lines)?;
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
    let tokenizer = aprender::text::tokenize::BpeTokenizer::train(&refs, vocab_size)
        .map_err(|e| CliError::ValidationFailed(format!("BPE training failed: {e}")))?;
    let elapsed = start.elapsed();

    std::fs::create_dir_all(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

    let trained = TrainedTokenizer {
        vocab: tokenizer.vocab().clone(),
        merges: tokenizer.merges().to_vec(),
    };
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

/// Build the normalizer closure. Threaded locally per task #90 since the
/// upstream TokenizerConfig doesn't yet carry a normalization field (task #89).
fn build_normalizer(normalization: &str) -> Box<dyn Fn(&str) -> String> {
    match normalization {
        "nfc" => Box::new(|s: &str| {
            use unicode_normalization::UnicodeNormalization;
            s.nfc().collect::<String>()
        }),
        _ => Box::new(|s: &str| s.to_string()),
    }
}

fn collect_jsonl_files(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let meta = std::fs::metadata(path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display()))
    })?;

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

fn read_jsonl_content(
    path: &Path,
    normalizer: &dyn Fn(&str) -> String,
    out: &mut Vec<String>,
) -> Result<()> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot read {}: {e}", path.display()))
    })?;
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
            out.push(normalizer(text));
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
    println!("  {} BPE tokenizer trained successfully", "OK".green().bold());
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

    #[test]
    fn run_train_rejects_unknown_normalization() {
        let tmp = TempDir::new().expect("tempdir");
        let corpus = write_corpus_file(
            tmp.path(),
            "corpus.jsonl",
            &[r#"{"content": "x y"}"#],
        );
        let err = run_train(&corpus, 300, 1, tmp.path(), "nfkd", true)
            .expect_err("should reject unsupported normalization");
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("nfkd")),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
