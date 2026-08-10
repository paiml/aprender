//! `apr rerank` — BERT cross-encoder relevance scoring (GH-326 Phase 3).
//!
//! Loads a cross-encoder from an APR v2 file via
//! `aprender_core::models::bert::CrossEncoder::load_from_reader` (Phase 1)
//! and scores a single `(input_ids, token_type_ids)` pair. Tokenisation is
//! NOT applied here — callers pass pre-tokenised u32 arrays. A tokenizer-
//! aware mode is Phase 3b follow-up scope.
//!
//! Wires the per-CRUX-Sovereign-Stack flow:
//!
//!   $ apr import hf://cross-encoder/ms-marco-MiniLM-L-6-v2 -o rerank.apr
//!   $ apr tokenize encode "..." --format ids -o ids.json
//!   $ apr rerank rerank.apr --input-ids 101,2024,102,3456,102 \
//!         --token-type-ids 0,0,0,1,1
//!   → 0.8347 (relevance probability ∈ [0, 1])
//!
//! Phase 3b will add `apr rerank --query "..." --passage "..."` with the
//! tokeniser pre-fixed in by the loaded checkpoint.

use crate::error::{CliError, Result};
use aprender::format::v2::AprV2Reader;
use aprender::models::bert::{BertConfig, CrossEncoder};
use std::collections::HashMap;
use std::path::Path;

/// Parse a comma-delimited list of `u32` IDs.
fn parse_id_list(s: &str, flag: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<u32>().map_err(|e| {
                CliError::ValidationFailed(format!("--{flag}: invalid u32 token {t:?}: {e}"))
            })
        })
        .collect()
}

/// Load a WordPiece `vocab.txt`: one token per line, line index = id (GH-326 Phase 3b).
///
/// Tokens with trailing whitespace are trimmed; empty lines become an empty
/// string token (which is harmless because real BERT vocabs never contain
/// empty strings). The returned map is suitable for
/// `WordPieceTokenizer::from_vocab`.
fn load_vocab_txt(path: &Path) -> Result<HashMap<String, u32>> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to read vocab {}: {e}", path.display()))
    })?;
    let mut map = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        map.insert(line.trim_end().to_string(), i as u32);
    }
    Ok(map)
}

/// Load a HuggingFace Tokenizers-format `tokenizer.json`, extracting the
/// WordPiece `model.vocab` map AND merging in the `added_tokens` array
/// (where BERT's special tokens `[CLS]`/`[SEP]`/`[UNK]` actually live —
/// they are NOT inside `model.vocab` per the Tokenizers convention).
fn load_tokenizer_json(path: &Path) -> Result<HashMap<String, u32>> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to read tokenizer.json {}: {e}",
            path.display()
        ))
    })?;
    let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        CliError::ValidationFailed(format!(
            "tokenizer.json {} is not valid JSON: {e}",
            path.display()
        ))
    })?;

    let mut map: HashMap<String, u32> = HashMap::new();

    // First merge `added_tokens` (special tokens live here).
    if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
        for entry in added {
            let content = entry
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    CliError::ValidationFailed(format!(
                        "tokenizer.json {}: added_tokens entry missing `content`",
                        path.display()
                    ))
                })?;
            let id = entry.get("id").and_then(|v| v.as_u64()).ok_or_else(|| {
                CliError::ValidationFailed(format!(
                    "tokenizer.json {}: added_tokens entry missing `id`",
                    path.display()
                ))
            })?;
            map.insert(content.to_string(), id as u32);
        }
    }

    // Then merge `model.vocab` (the bulk WordPiece vocabulary).
    let vocab_obj = root
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "tokenizer.json {}: missing or non-object `model.vocab` \
                 (only WordPiece-style tokenizer.json is supported)",
                path.display()
            ))
        })?;
    for (token, id_val) in vocab_obj {
        let id = id_val.as_u64().ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "tokenizer.json {}: model.vocab entry {token:?} has non-integer id",
                path.display()
            ))
        })?;
        map.insert(token.to_string(), id as u32);
    }

    Ok(map)
}

/// Dispatch vocab loading based on file extension — `.json` → HF Tokenizers
/// format; otherwise treat as legacy line-per-token `vocab.txt`.
fn load_vocab(path: &Path) -> Result<HashMap<String, u32>> {
    let is_json = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    if is_json {
        load_tokenizer_json(path)
    } else {
        load_vocab_txt(path)
    }
}

/// Tokenise `[CLS] query [SEP] passage [SEP]` using a WordPiece vocab
/// (GH-326 Phase 3b). Returns `(input_ids, token_type_ids)` ready to feed
/// `CrossEncoder::forward`.
///
/// `token_type_ids` is 0 for the `[CLS]`+query+first `[SEP]` segment and 1
/// for the passage+second `[SEP]` segment, matching the HuggingFace BERT
/// cross-encoder convention.
fn tokenize_query_passage(
    query: &str,
    passage: &str,
    vocab_path: &Path,
) -> Result<(Vec<u32>, Vec<u32>)> {
    use aprender::text::tokenize::WordPieceTokenizer;

    let vocab = load_vocab(vocab_path)?;
    let cls_id = *vocab.get("[CLS]").ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "vocab {} missing required token [CLS]",
            vocab_path.display()
        ))
    })?;
    let sep_id = *vocab.get("[SEP]").ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "vocab {} missing required token [SEP]",
            vocab_path.display()
        ))
    })?;
    // [UNK] presence is verified by WordPieceTokenizer::from_vocab at encode
    // time; we surface a clearer error here if the user passes the wrong file.
    if !vocab.contains_key("[UNK]") {
        return Err(CliError::ValidationFailed(format!(
            "vocab {} missing required token [UNK]",
            vocab_path.display()
        )));
    }

    let tokenizer = WordPieceTokenizer::from_vocab(vocab);
    let q_ids = tokenizer
        .encode(query)
        .map_err(|e| CliError::ValidationFailed(format!("query tokenisation failed: {e:?}")))?;
    let p_ids = tokenizer
        .encode(passage)
        .map_err(|e| CliError::ValidationFailed(format!("passage tokenisation failed: {e:?}")))?;

    let mut input_ids = Vec::with_capacity(1 + q_ids.len() + 1 + p_ids.len() + 1);
    input_ids.push(cls_id);
    input_ids.extend(&q_ids);
    input_ids.push(sep_id);
    input_ids.extend(&p_ids);
    input_ids.push(sep_id);

    let mut token_type_ids = Vec::with_capacity(input_ids.len());
    token_type_ids.extend(std::iter::repeat_n(0u32, 1 + q_ids.len() + 1));
    token_type_ids.extend(std::iter::repeat_n(1u32, p_ids.len() + 1));

    Ok((input_ids, token_type_ids))
}

/// Entry point for `apr rerank` — loads the model, scores the pair, prints
/// the relevance probability (or raw logit) as text or JSON.
///
/// Two input modes:
/// - **ID mode (Phase 3)**: caller supplies `--input-ids` + `--token-type-ids`
///   as comma-delimited u32 lists.
/// - **Text mode (Phase 3b)**: caller supplies `--query` + `--passage` +
///   `--vocab` and the helper tokenises in-process using
///   `aprender::text::tokenize::WordPieceTokenizer`.
/// - **Batch mode (Phase 5)**: caller supplies `--query` + repeated
///   `--passages` + `--vocab`. Each passage is scored independently
///   against the same query. `--sort` orders output by descending
///   score; `--top-k N` limits to the top N (implies `--sort`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: &Path,
    input_ids_str: Option<&str>,
    token_type_ids_str: Option<&str>,
    query: Option<&str>,
    passage: Option<&str>,
    passages: &[String],
    sort: bool,
    top_k: usize,
    vocab: Option<&Path>,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    num_labels: usize,
    with_pooler: bool,
    raw_logit: bool,
    json: bool,
) -> Result<()> {
    let config = BertConfig {
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
    };

    // `CrossEncoder::new` builds `MultiHeadAttention`, which *asserts*
    // `hidden_dim % num_heads == 0`. `--hidden-dim` / `--num-heads` are
    // user-facing override flags, so reject bad combinations here instead of
    // aborting the process with a library panic. Checked before the model is
    // read so a bad flag costs no I/O.
    config
        .validate()
        .map_err(|e| CliError::ValidationFailed(format!("invalid BERT config: {e}")))?;

    // Single-pair mode: resolve and check the (input_ids, token_type_ids)
    // pair BEFORE the model is read, so a bad id also costs no I/O.
    let single_pair = if passages.is_empty() {
        let (input_ids, token_type_ids) =
            match (input_ids_str, token_type_ids_str, query, passage, vocab) {
                (Some(id_str), Some(tt_str), None, None, None) => {
                    let input_ids = parse_id_list(id_str, "input-ids")?;
                    let token_type_ids = parse_id_list(tt_str, "token-type-ids")?;
                    (input_ids, token_type_ids)
                }
                (None, None, Some(q), Some(p), Some(vp)) => tokenize_query_passage(q, p, vp)?,
                _ => {
                    return Err(CliError::ValidationFailed(
                        "apr rerank requires EITHER \
                     (--input-ids AND --token-type-ids) \
                     OR (--query AND --passage AND --vocab) \
                     OR (--query AND --passages... AND --vocab)"
                            .to_string(),
                    ));
                }
            };

        if input_ids.is_empty() {
            return Err(CliError::ValidationFailed(
                "--input-ids must be non-empty".to_string(),
            ));
        }
        if input_ids.len() != token_type_ids.len() {
            return Err(CliError::ValidationFailed(format!(
                "--input-ids ({}) and --token-type-ids ({}) must have the same length",
                input_ids.len(),
                token_type_ids.len()
            )));
        }
        // `BertEmbeddings::forward` slices the embedding tables unchecked, so
        // an id past the end of a table aborts the process. Reject it here.
        config
            .validate_ids(&input_ids, &token_type_ids)
            .map_err(|e| CliError::ValidationFailed(format!("invalid tokens: {e}")))?;
        Some((input_ids, token_type_ids))
    } else {
        None
    };

    // Load model once — reused across all passages in batch mode.
    let model_bytes = std::fs::read(model).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to read {}: {e}", model.display()))
    })?;
    let reader = AprV2Reader::from_bytes(&model_bytes).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to parse APR v2 at {}: {e:?}",
            model.display()
        ))
    })?;

    let mut cross_encoder = CrossEncoder::new(&config, num_labels, with_pooler);
    cross_encoder
        .load_from_reader(&reader, &config)
        .map_err(|e| CliError::ValidationFailed(format!("BERT weight loading failed: {e}")))?;

    // No single pair resolved ⟹ Phase 5 batch mode: `--query` + repeated
    // `--passages` + `--vocab`, each passage tokenised and checked in turn.
    let Some((input_ids, token_type_ids)) = single_pair else {
        return run_batch(
            &cross_encoder,
            &config,
            model,
            query,
            passages,
            vocab,
            sort,
            top_k,
            raw_logit,
            json,
        );
    };

    let logit_tensor = cross_encoder.forward(&input_ids, &token_type_ids);
    let logits: &[f32] = logit_tensor.data();

    if json {
        #[allow(clippy::disallowed_methods)]
        {
            let payload = if raw_logit {
                serde_json::json!({
                    "model": model.display().to_string(),
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "logits": logits,
                })
            } else {
                let probs: Vec<f32> = logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
                serde_json::json!({
                    "model": model.display().to_string(),
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "scores": probs,
                })
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&payload).unwrap_or_default()
            );
        }
        return Ok(());
    }

    // Single-pair text-mode output (continues below).
    if raw_logit {
        for (i, &l) in logits.iter().enumerate() {
            println!("logit[{i}] = {l:.6}");
        }
    } else {
        for (i, &l) in logits.iter().enumerate() {
            let score = 1.0 / (1.0 + (-l).exp());
            println!("score[{i}] = {score:.6}");
        }
    }
    Ok(())
}

/// Phase 5 — batch ranking. Score each passage in `passages` against
/// `query` using the loaded `cross_encoder`. Emits one `score[i]` line
/// per passage (or a JSON array). With `--sort` or `--top-k`, the JSON
/// array is sorted by descending score and optionally truncated.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    cross_encoder: &CrossEncoder,
    config: &BertConfig,
    model: &Path,
    query: Option<&str>,
    passages: &[String],
    vocab: Option<&Path>,
    sort: bool,
    top_k: usize,
    raw_logit: bool,
    json: bool,
) -> Result<()> {
    let (Some(query), Some(vocab)) = (query, vocab) else {
        return Err(CliError::ValidationFailed(
            "--passages requires both --query and --vocab".to_string(),
        ));
    };

    // Score every (query, passage) pair. Indices preserved so we can
    // include the original passage text in the JSON output even after
    // sort.
    let mut scored: Vec<(usize, f32, f32)> = Vec::with_capacity(passages.len());
    for (i, p) in passages.iter().enumerate() {
        let (input_ids, token_type_ids) = tokenize_query_passage(query, p, vocab)?;
        config
            .validate_ids(&input_ids, &token_type_ids)
            .map_err(|e| {
                CliError::ValidationFailed(format!("invalid tokens for passage {i}: {e}"))
            })?;
        let logit_tensor = cross_encoder.forward(&input_ids, &token_type_ids);
        let logit = logit_tensor.data()[0];
        let score = 1.0 / (1.0 + (-logit).exp());
        scored.push((i, logit, score));
    }

    // Sort descending by score if requested OR if --top-k is set.
    let do_sort = sort || top_k > 0;
    if do_sort {
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    }
    let limit = if top_k > 0 {
        top_k.min(scored.len())
    } else {
        scored.len()
    };
    let scored = &scored[..limit];

    if json {
        #[allow(clippy::disallowed_methods)]
        {
            let array: Vec<serde_json::Value> = scored
                .iter()
                .map(|&(i, logit, score)| {
                    serde_json::json!({
                        "index": i,
                        "passage": passages[i],
                        "logit": logit,
                        "score": score,
                    })
                })
                .collect();
            let payload = serde_json::json!({
                "model": model.display().to_string(),
                "query": query,
                "num_passages": passages.len(),
                "returned": scored.len(),
                "sorted": do_sort,
                "results": array,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&payload).unwrap_or_default()
            );
        }
        return Ok(());
    }

    if raw_logit {
        for (i, logit, _score) in scored {
            println!("logit[{i}] = {logit:.6}");
        }
    } else {
        for (i, _logit, score) in scored {
            println!("score[{i}] = {score:.6}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Invoke `run` in `--input-ids` mode with MiniLM-L-6 defaults, allowing
    /// individual overrides. The model path is deliberately bogus: every
    /// argument-validation failure must be reported *before* the model is
    /// read, so a returned "Failed to read" means the guard did not fire.
    fn run_ids_mode(
        ids: &str,
        token_type_ids: &str,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        run(
            Path::new("/nonexistent/rerank-guard.apr"),
            Some(ids),
            Some(token_type_ids),
            None,
            None,
            &[],
            false,
            0,
            None,
            hidden_dim,
            6,
            num_heads,
            1536,
            30522,
            512,
            2,
            1,
            true,
            false,
            false,
        )
    }

    fn validation_message(res: Result<()>) -> String {
        match res.expect_err("must be rejected") {
            CliError::ValidationFailed(msg) => msg,
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    /// Regression (dogfood 0.63.0): `apr rerank … --hidden-dim 999` aborted
    /// with `thread 'main' panicked at nn/transformer/mod.rs:124: embed_dim
    /// (999) must be divisible by num_heads (12)` and exit 101. It must now
    /// return a validation error naming both flag values.
    #[test]
    fn rerank_rejects_hidden_dim_not_divisible_by_num_heads() {
        let msg = validation_message(run_ids_mode("101,2024,102", "0,0,0", 999, 12));
        assert!(msg.contains("999") && msg.contains("12"), "{msg}");
        assert!(msg.contains("divisible"), "{msg}");
        assert!(
            !msg.contains("Failed to read"),
            "flags must be checked before the model is read: {msg}"
        );
    }

    /// Same defect reached through `--num-heads`: 384 % 7 != 0.
    #[test]
    fn rerank_rejects_num_heads_not_dividing_hidden_dim() {
        let msg = validation_message(run_ids_mode("101,2024,102", "0,0,0", 384, 7));
        assert!(msg.contains("384") && msg.contains("7"), "{msg}");
    }

    /// Regression (dogfood 0.63.0): `--input-ids 101,999999,102` aborted with
    /// `range start index 383999616 out of range for slice of length 11720448`
    /// from the unchecked embedding-table slice, exit 101.
    #[test]
    fn rerank_rejects_input_id_beyond_vocab_size() {
        let msg = validation_message(run_ids_mode("101,999999,102", "0,0,0", 384, 12));
        assert!(msg.contains("999999") && msg.contains("30522"), "{msg}");
        assert!(msg.contains("input_ids[1]"), "{msg}");
    }

    /// Same panic class via `--token-type-ids` (table has 2 rows).
    #[test]
    fn rerank_rejects_token_type_id_beyond_type_vocab_size() {
        let msg = validation_message(run_ids_mode("101,2024,102", "0,7,0", 384, 12));
        assert!(msg.contains("token_type_ids[1]"), "{msg}");
        assert!(msg.contains('7'), "{msg}");
    }

    /// Regression: a 600-token pair aborted with `sequence length 600 exceeds
    /// max_position_embeddings 512`.
    #[test]
    fn rerank_rejects_sequence_longer_than_position_table() {
        let ids = vec!["101"; 600].join(",");
        let tt = vec!["0"; 600].join(",");
        let msg = validation_message(run_ids_mode(&ids, &tt, 384, 12));
        assert!(msg.contains("600") && msg.contains("512"), "{msg}");
    }

    /// The guard must not reject legitimate input: with in-range ids and a
    /// valid config, `run` gets past argument validation and fails on the
    /// (deliberately missing) model file instead.
    #[test]
    fn rerank_accepts_in_range_ids_and_proceeds_to_model_load() {
        let msg = validation_message(run_ids_mode("101,2024,102,3456,102", "0,0,0,1,1", 384, 12));
        assert!(
            msg.contains("Failed to read"),
            "valid arguments must reach the model load, got: {msg}"
        );
    }

    #[test]
    fn parse_id_list_accepts_commas_and_spaces() {
        assert_eq!(
            parse_id_list("1,2,3", "input-ids").unwrap(),
            vec![1u32, 2, 3]
        );
        assert_eq!(
            parse_id_list(" 101, 2024, 102 ", "input-ids").unwrap(),
            vec![101u32, 2024, 102]
        );
    }

    #[test]
    fn parse_id_list_rejects_invalid_token() {
        let err = parse_id_list("1,xx,3", "input-ids").expect_err("xx must reject");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("input-ids"));
                assert!(msg.contains("xx"));
            }
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn parse_id_list_skips_empty_tokens_from_trailing_comma() {
        assert_eq!(
            parse_id_list("1,2,3,", "input-ids").unwrap(),
            vec![1u32, 2, 3]
        );
    }

    /// Phase 3b — `load_vocab_txt` round-trip: each line index maps to the
    /// trimmed line content.
    #[test]
    fn load_vocab_txt_assigns_line_index_as_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n##world\n").unwrap();
        let map = load_vocab_txt(&path).expect("load");
        assert_eq!(map.get("[PAD]").copied(), Some(0));
        assert_eq!(map.get("[UNK]").copied(), Some(1));
        assert_eq!(map.get("[CLS]").copied(), Some(2));
        assert_eq!(map.get("[SEP]").copied(), Some(3));
        assert_eq!(map.get("hello").copied(), Some(4));
        assert_eq!(map.get("##world").copied(), Some(5));
    }

    /// Phase 3b — `tokenize_query_passage` emits `[CLS] q [SEP] p [SEP]`
    /// with `token_type_ids` 0 for query side, 1 for passage side.
    #[test]
    fn tokenize_query_passage_builds_correct_segment_pair() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vocab.txt");
        // Minimal vocab: specials + words our test prompt uses (lower-case
        // because WordPieceTokenizer lower-cases input by default).
        std::fs::write(
            &path,
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\nfoo\nbar\n",
        )
        .unwrap();
        let (input_ids, token_type_ids) =
            tokenize_query_passage("hello world", "foo bar", &path).expect("tokenize");

        // [CLS]=2, hello=4, world=5, [SEP]=3, foo=6, bar=7, [SEP]=3.
        assert_eq!(input_ids, vec![2u32, 4, 5, 3, 6, 7, 3]);
        // token_type_ids: 0 for [CLS]+query+first [SEP] (4 tokens), 1 for
        // passage+second [SEP] (3 tokens).
        assert_eq!(token_type_ids, vec![0u32, 0, 0, 0, 1, 1, 1]);
    }

    /// Phase 3b — `tokenize_query_passage` rejects vocabs missing required
    /// special tokens with a clear error.
    #[test]
    fn tokenize_query_passage_rejects_missing_cls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vocab.txt");
        // No [CLS] in this vocab — should error.
        std::fs::write(&path, "[PAD]\n[UNK]\n[SEP]\nhello\n").unwrap();
        let err =
            tokenize_query_passage("hello", "world", &path).expect_err("missing [CLS] must reject");
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("[CLS]"), "{msg}"),
            _ => panic!("expected ValidationFailed"),
        }
    }

    /// Phase 3b — `tokenize_query_passage` rejects vocabs missing `[SEP]`.
    #[test]
    fn tokenize_query_passage_rejects_missing_sep() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, "[PAD]\n[UNK]\n[CLS]\nhello\n").unwrap();
        let err =
            tokenize_query_passage("hello", "world", &path).expect_err("missing [SEP] must reject");
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("[SEP]"), "{msg}"),
            _ => panic!("expected ValidationFailed"),
        }
    }
}
