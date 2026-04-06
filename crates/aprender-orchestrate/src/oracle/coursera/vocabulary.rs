//! Vocabulary extraction from transcripts
//!
//! Detects technical terms via: uppercase mid-word, acronyms, known suffixes,
//! and frequency >= 2. Organizes by category with context extraction.

use std::collections::HashMap;

use super::transcript::format_timestamp;
use super::types::{ConceptCategory, TranscriptInput, VocabularyEntry};

/// Known technical suffixes that indicate domain terms
const TECH_SUFFIXES: &[&str] =
    &["ization", "isation", "ment", "tion", "sion", "ness", "ity", "ence", "ance"];

/// Known acronyms / uppercase terms (case-insensitive lookup set)
const KNOWN_ACRONYMS: &[&str] = &[
    "API",
    "GPU",
    "CPU",
    "TPU",
    "ML",
    "AI",
    "NLP",
    "CNN",
    "RNN",
    "GAN",
    "LLM",
    "BERT",
    "GPT",
    "LSTM",
    "GRU",
    "RLHF",
    "RAG",
    "SIMD",
    "AVX",
    "NEON",
    "REST",
    "HTTP",
    "HTTPS",
    "JSON",
    "YAML",
    "TOML",
    "SQL",
    "CI",
    "CD",
    "MLOps",
    "DevOps",
    "AWS",
    "GCP",
    "CLI",
    "SDK",
    "TDD",
    "BDD",
    "OOP",
    "WASM",
    "CUDA",
    "MCP",
    "SSE",
    "TLS",
    "TCP",
    "UDP",
    "DNS",
    "SSH",
    "GGUF",
    "LoRA",
    "QLoRA",
    "GPTQ",
    "AWQ",
    "KV",
    "LZ4",
    "ZSTD",
    "Docker",
    "Kubernetes",
    "K8s",
    "ECS",
    "S3",
    "EC2",
    "Lambda",
    "PyTorch",
    "TensorFlow",
    "NumPy",
    "SciPy",
    "Pandas",
    "Sklearn",
    "HuggingFace",
    "SafeTensors",
    "Parquet",
    "Arrow",
    "Kafka",
    "NCCL",
    "MPI",
    "RPC",
    "gRPC",
    "OAuth",
    "JWT",
    "RBAC",
];

/// Process a single transcript into the shared term accumulator.
fn accumulate_transcript(
    transcript: &TranscriptInput,
    term_data: &mut HashMap<String, TermAccumulator>,
) {
    let sentences = split_sentences(&transcript.text);

    for (i, sentence) in sentences.iter().enumerate() {
        let words = extract_candidate_terms(sentence);

        for word in &words {
            let normalized = normalize_term(word);
            if normalized.len() < 2 || is_stop_word(&normalized) {
                continue;
            }

            let entry = term_data.entry(normalized.clone()).or_insert_with(|| {
                let timestamp = find_timestamp_for_sentence(transcript, i, &sentences);
                TermAccumulator {
                    original_form: word.clone(),
                    first_occurrence: timestamp,
                    frequency: 0,
                    contexts: Vec::new(),
                    source: transcript.source_path.clone(),
                }
            });

            entry.frequency += 1;
            if entry.contexts.len() < 3 {
                entry.contexts.push(sentence.trim().to_string());
            }
        }
    }
}

/// Extract vocabulary from multiple transcripts.
pub fn extract_vocabulary(transcripts: &[TranscriptInput]) -> Vec<VocabularyEntry> {
    let mut term_data: HashMap<String, TermAccumulator> = HashMap::new();

    for transcript in transcripts {
        accumulate_transcript(transcript, &mut term_data);
    }

    let mut entries: Vec<VocabularyEntry> = term_data
        .into_iter()
        .filter(|(term, acc)| acc.frequency >= 2 || is_known_acronym(term))
        .map(|(term, acc)| {
            let category = categorize_term(&term);
            let definition = derive_definition(&acc.contexts, &term);
            VocabularyEntry {
                term: acc.original_form,
                definition,
                first_occurrence: acc.first_occurrence,
                frequency: acc.frequency,
                category,
            }
        })
        .collect();

    entries.sort_by(|a, b| b.frequency.cmp(&a.frequency));
    entries
}

/// Render vocabulary entries as Markdown.
pub fn render_vocabulary_markdown(entries: &[VocabularyEntry]) -> String {
    let mut md = String::new();
    md.push_str("# Course Vocabulary\n\n");

    if entries.is_empty() {
        md.push_str("No vocabulary terms extracted.\n");
        return md;
    }

    // Group by category
    let mut by_category: HashMap<&str, Vec<&VocabularyEntry>> = HashMap::new();
    for entry in entries {
        by_category.entry(entry.category.as_str()).or_default().push(entry);
    }

    // Sort categories for deterministic output
    let mut categories: Vec<&&str> = by_category.keys().collect();
    categories.sort();

    for cat in categories {
        let cat_entries = &by_category[*cat];
        md.push_str(&format!("## {}\n\n", cat));
        md.push_str("| Term | Definition | Frequency | First Seen |\n");
        md.push_str("|------|-----------|-----------|------------|\n");

        for entry in cat_entries {
            md.push_str(&format!(
                "| **{}** | {} | {} | {} |\n",
                entry.term, entry.definition, entry.frequency, entry.first_occurrence,
            ));
        }
        md.push('\n');
    }

    md
}

// ============================================================================
// Internal helpers
// ============================================================================

struct TermAccumulator {
    original_form: String,
    first_occurrence: String,
    frequency: usize,
    contexts: Vec<String>,
    source: String,
}

fn split_sentences(text: &str) -> Vec<String> {
    // Split on sentence-ending punctuation followed by whitespace or EOL
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

fn extract_candidate_terms(sentence: &str) -> Vec<String> {
    let mut terms = Vec::new();

    for word in sentence.split_whitespace() {
        let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');

        if cleaned.is_empty() {
            continue;
        }

        // Check if it looks technical
        if is_technical_word(cleaned) {
            terms.push(cleaned.to_string());
        }
    }

    // Also extract hyphenated compounds
    for window in sentence.split_whitespace().collect::<Vec<_>>().windows(3) {
        if window.len() == 3 && window[1] == "-" {
            let compound = format!("{}-{}", window[0], window[2]);
            let cleaned = compound.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
            if cleaned.len() > 3 {
                terms.push(cleaned.to_string());
            }
        }
    }

    terms
}

fn is_technical_word(word: &str) -> bool {
    // All caps (acronym): ML, API, GPU
    if word.len() >= 2 && word.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return true;
    }

    // CamelCase or mixed case mid-word: MLOps, DevOps, PyTorch
    let has_mid_upper = word.chars().enumerate().any(|(i, c)| {
        i > 0 && c.is_uppercase() && word.chars().nth(i - 1).is_some_and(|p| p.is_lowercase())
    });
    if has_mid_upper {
        return true;
    }

    // Contains hyphen (compound term): cross-validation, pre-training
    if word.contains('-') && word.len() > 5 {
        return true;
    }

    // Known acronym match
    if is_known_acronym(word) {
        return true;
    }

    // Technical suffix
    let lower = word.to_lowercase();
    if TECH_SUFFIXES.iter().any(|s| lower.ends_with(s)) && word.len() > 6 {
        return true;
    }

    false
}

fn is_known_acronym(word: &str) -> bool {
    let lower = word.to_lowercase();
    KNOWN_ACRONYMS.iter().any(|a| a.to_lowercase() == lower)
}

fn normalize_term(word: &str) -> String {
    // Preserve original casing for acronyms, lowercase for others
    if word.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        word.to_string()
    } else {
        word.to_lowercase()
    }
}

fn is_stop_word(word: &str) -> bool {
    const STOP: &[&str] = &[
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "or",
        "and",
        "but",
        "if",
        "not",
        "no",
        "so",
        "up",
        "out",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "we",
        "you",
        "they",
        "he",
        "she",
        "my",
        "your",
        "our",
        "us",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "about",
        "which",
        "what",
        "when",
        "where",
        "how",
        "who",
        "whom",
        "why",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "same",
        "different",
        "then",
        "there",
        "here",
        "new",
        "old",
        "many",
        "much",
        "own",
        "only",
        "well",
    ];
    STOP.contains(&word.to_lowercase().as_str())
}

fn find_timestamp_for_sentence(
    transcript: &TranscriptInput,
    sentence_idx: usize,
    sentences: &[String],
) -> String {
    if transcript.segments.is_empty() {
        return format!("sentence {}", sentence_idx + 1);
    }

    // Approximate: find which segment contains this sentence
    let target_sentence = &sentences[sentence_idx];
    for seg in &transcript.segments {
        if seg.text.contains(target_sentence.split_whitespace().next().unwrap_or(""))
            || target_sentence.contains(seg.text.split_whitespace().next().unwrap_or(""))
        {
            return format_timestamp(seg.start);
        }
    }

    // Fallback: estimate based on sentence position
    if let Some(last_seg) = transcript.segments.last() {
        let ratio = sentence_idx as f64 / sentences.len().max(1) as f64;
        let estimated_time = ratio * last_seg.end;
        return format_timestamp(estimated_time);
    }

    format!("sentence {}", sentence_idx + 1)
}

fn categorize_term(term: &str) -> ConceptCategory {
    let lower = term.to_lowercase();

    // Tool patterns
    if KNOWN_ACRONYMS.iter().any(|a| {
        let al = a.to_lowercase();
        al == lower
            && matches!(
                al.as_str(),
                "docker"
                    | "kubernetes"
                    | "k8s"
                    | "pytorch"
                    | "tensorflow"
                    | "numpy"
                    | "scipy"
                    | "pandas"
                    | "sklearn"
                    | "kafka"
                    | "huggingface"
                    | "mlflow"
            )
    }) {
        return ConceptCategory::Tool;
    }

    // Algorithm patterns
    let algo_keywords = [
        "sort",
        "search",
        "gradient",
        "descent",
        "backprop",
        "boosting",
        "regression",
        "classification",
        "clustering",
        "optimization",
        "attention",
        "convolution",
        "pooling",
        "softmax",
        "normalization",
    ];
    if algo_keywords.iter().any(|k| lower.contains(k)) {
        return ConceptCategory::Algorithm;
    }

    // Data structure patterns
    let ds_keywords =
        ["tree", "graph", "array", "tensor", "matrix", "vector", "queue", "stack", "hash", "cache"];
    if ds_keywords.iter().any(|k| lower.contains(k)) {
        return ConceptCategory::DataStructure;
    }

    // Metric patterns
    let metric_keywords = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "loss",
        "score",
        "metric",
        "perplexity",
        "bleu",
        "rouge",
        "latency",
        "throughput",
    ];
    if metric_keywords.iter().any(|k| lower.contains(k)) {
        return ConceptCategory::Metric;
    }

    // Pattern keywords
    let pattern_keywords = [
        "pattern",
        "pipeline",
        "workflow",
        "architecture",
        "design",
        "ops",
        "devops",
        "mlops",
        "ci/cd",
        "microservice",
    ];
    if pattern_keywords.iter().any(|k| lower.contains(k)) {
        return ConceptCategory::Pattern;
    }

    ConceptCategory::General
}

/// Try to extract a definition from a context string using "X is ..." or "X refers to ..." patterns.
fn try_extract_definition(ctx: &str, lower_term: &str) -> Option<String> {
    let lower_ctx = ctx.to_lowercase();

    if let Some(pos) = lower_ctx.find(&format!("{} is ", lower_term)) {
        let start = pos + lower_term.len() + 4;
        if let Some(def) = ctx.get(start..) {
            let end = def.find('.').unwrap_or(def.len()).min(120);
            return Some(capitalize_first(safe_truncate_bytes(def, end).trim()));
        }
    }

    if let Some(pos) = lower_ctx.find(&format!("{} refers to ", lower_term)) {
        let start = pos + lower_term.len() + 11;
        if let Some(def) = ctx.get(start..) {
            let end = def.find('.').unwrap_or(def.len()).min(120);
            return Some(capitalize_first(safe_truncate_bytes(def, end).trim()));
        }
    }

    None
}

fn derive_definition(contexts: &[String], term: &str) -> String {
    let lower_term = term.to_lowercase();

    for ctx in contexts {
        if let Some(def) = try_extract_definition(ctx, &lower_term) {
            return def;
        }
    }

    if let Some(first) = contexts.first() {
        return if first.len() > 100 {
            format!("{}...", safe_truncate_bytes(first, 100))
        } else {
            first.clone()
        };
    }

    format!("Technical term: {term}")
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

/// Truncate a string at the nearest char boundary at or before `max_bytes`.
fn safe_truncate_bytes(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::coursera::types::TranscriptSegment;

    fn make_transcript(text: &str) -> TranscriptInput {
        TranscriptInput {
            text: text.to_string(),
            language: "en".to_string(),
            segments: vec![],
            source_path: "test.txt".to_string(),
        }
    }

    #[test]
    fn test_extract_vocabulary_basic() {
        let t = make_transcript(
            "MLOps combines ML and DevOps. MLOps is the practice of deploying ML models. \
             DevOps principles apply to ML workflows. API endpoints serve predictions. \
             The API handles inference requests.",
        );
        let entries = extract_vocabulary(&[t]);
        assert!(!entries.is_empty());

        let mlops = entries.iter().find(|e| e.term.to_lowercase() == "mlops");
        assert!(mlops.is_some(), "Should find MLOps");
        assert!(mlops.expect("unexpected failure").frequency >= 2);
    }

    #[test]
    fn test_extract_vocabulary_empty() {
        let entries = extract_vocabulary(&[]);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_extract_vocabulary_no_technical_terms() {
        let t = make_transcript("The cat sat on the mat. It was a good day.");
        let entries = extract_vocabulary(&[t]);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_render_vocabulary_markdown() {
        let entries = vec![VocabularyEntry {
            term: "MLOps".to_string(),
            definition: "Machine Learning Operations".to_string(),
            first_occurrence: "0:05".to_string(),
            frequency: 5,
            category: ConceptCategory::Pattern,
        }];
        let md = render_vocabulary_markdown(&entries);
        assert!(md.contains("# Course Vocabulary"));
        assert!(md.contains("MLOps"));
        assert!(md.contains("Machine Learning Operations"));
        assert!(md.contains("Pattern"));
    }

    #[test]
    fn test_render_vocabulary_markdown_empty() {
        let md = render_vocabulary_markdown(&[]);
        assert!(md.contains("No vocabulary terms extracted"));
    }

    #[test]
    fn test_is_technical_word() {
        assert!(is_technical_word("API"));
        assert!(is_technical_word("MLOps"));
        assert!(is_technical_word("DevOps"));
        assert!(is_technical_word("pre-training"));
        assert!(!is_technical_word("the"));
        assert!(!is_technical_word("good"));
    }

    #[test]
    fn test_categorize_term() {
        assert_eq!(categorize_term("gradient descent"), ConceptCategory::Algorithm);
        assert_eq!(categorize_term("tensor"), ConceptCategory::DataStructure);
        assert_eq!(categorize_term("accuracy"), ConceptCategory::Metric);
        assert_eq!(categorize_term("pipeline"), ConceptCategory::Pattern);
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? Fine!");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_vocabulary_with_segments() {
        let t = TranscriptInput {
            text: "GPU acceleration is important. GPU kernels run SIMD operations. \
                   GPU computing enables parallel workloads."
                .to_string(),
            language: "en".to_string(),
            segments: vec![
                TranscriptSegment {
                    start: 0.0,
                    end: 5.0,
                    text: "GPU acceleration is important.".to_string(),
                },
                TranscriptSegment {
                    start: 5.0,
                    end: 10.0,
                    text: "GPU kernels run SIMD operations.".to_string(),
                },
            ],
            source_path: "lesson.json".to_string(),
        };
        let entries = extract_vocabulary(&[t]);
        let gpu = entries.iter().find(|e| e.term == "GPU");
        assert!(gpu.is_some());
    }

    #[test]
    fn test_derive_definition_pattern() {
        let contexts =
            vec!["MLOps is the practice of deploying ML models in production.".to_string()];
        let def = derive_definition(&contexts, "mlops");
        assert!(def.contains("practice"), "Got: {def}");
    }

    #[test]
    fn test_normalize_term() {
        assert_eq!(normalize_term("API"), "API");
        assert_eq!(normalize_term("DevOps"), "devops");
    }

    #[test]
    fn test_derive_definition_refers_to() {
        // Triggers "refers to" pattern (lines 569-574)
        let contexts =
            vec!["MLOps refers to the practice of deploying ML models in production.".to_string()];
        let def = derive_definition(&contexts, "mlops");
        assert!(def.contains("practice") || def.contains("deploying"), "Got: {def}");
    }

    #[test]
    fn test_derive_definition_fallback_long_context() {
        // Triggers long context truncation fallback (line 580-581)
        let long = format!("{}. More text follows.", "A".repeat(120));
        let contexts = vec![long];
        let def = derive_definition(&contexts, "nonexistentterm");
        assert!(def.contains("..."), "Got: {def}");
    }

    #[test]
    fn test_derive_definition_no_contexts() {
        // Triggers empty contexts fallback (line 588)
        let def = derive_definition(&[], "SomeTerm");
        assert_eq!(def, "Technical term: SomeTerm");
    }

    #[test]
    fn test_categorize_term_tool() {
        // Triggers Tool match (lines 461-481)
        assert_eq!(categorize_term("Docker"), ConceptCategory::Tool);
        assert_eq!(categorize_term("Kubernetes"), ConceptCategory::Tool);
        assert_eq!(categorize_term("PyTorch"), ConceptCategory::Tool);
    }

    #[test]
    fn test_hyphenated_compound_extraction() {
        // Triggers hyphenated compound extraction (lines 251-258)
        let terms = extract_candidate_terms("The cross - validation technique is used");
        assert!(
            terms.iter().any(|t| t.contains("cross") && t.contains("validation")),
            "Terms: {:?}",
            terms
        );
    }

    #[test]
    fn test_find_timestamp_estimated_fallback() {
        // Triggers estimated timestamp fallback (lines 448-452)
        let t = TranscriptInput {
            text: "First sentence. Second sentence. Third sentence.".to_string(),
            language: "en".to_string(),
            segments: vec![
                TranscriptSegment {
                    start: 0.0,
                    end: 10.0,
                    text: "Unrelated segment text here".to_string(),
                },
                TranscriptSegment {
                    start: 10.0,
                    end: 30.0,
                    text: "Another unrelated segment".to_string(),
                },
            ],
            source_path: "test.json".to_string(),
        };
        let sentences = split_sentences(&t.text);
        // Third sentence won't match any segment, triggering estimated fallback
        let ts = find_timestamp_for_sentence(&t, 2, &sentences);
        // Should produce a time-based estimate, not "sentence 3"
        assert!(ts.contains(':'), "Expected timestamp, got: {ts}");
    }

    #[test]
    fn test_capitalize_first_empty() {
        assert_eq!(capitalize_first(""), "");
    }

    #[test]
    fn test_safe_truncate_bytes_multibyte() {
        // Triggers char boundary adjustment loop (lines 604-607)
        let s = "café résumé";
        let truncated = safe_truncate_bytes(s, 4);
        // 'é' is 2 bytes, position 4 should be safe but let's test the boundary
        assert!(!truncated.is_empty());
        assert!(s.is_char_boundary(truncated.len()));
    }

    #[test]
    fn test_stop_word_filtering() {
        // Triggers continue on stop words (line 113)
        assert!(is_stop_word("the"));
        assert!(is_stop_word("and"));
        assert!(!is_stop_word("kubernetes"));
    }

    #[test]
    fn test_extract_candidate_terms_empty_word() {
        // Triggers empty word skip (line 240-242)
        let terms = extract_candidate_terms("... --- *** plain text");
        // None of these should produce technical terms
        assert!(!terms.iter().any(|t| t == "..." || t == "---" || t == "***"),);
    }

    #[test]
    fn test_derive_definition_short_context() {
        // Triggers short context fallback (line 583 - context.len() <= 100)
        let contexts = vec!["A brief context.".to_string()];
        let def = derive_definition(&contexts, "xyzterm");
        assert_eq!(def, "A brief context.");
    }
}
