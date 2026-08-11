//! `apr embed` — BERT sentence-embedding bi-encoder (GH-326 Phase 6).
//!
//! First-stage dense-retrieval companion to `apr rerank`. Loads an
//! encoder-only `BertModel` (e.g.
//! `sentence-transformers/all-MiniLM-L6-v2`), tokenises text with
//! WordPiece, runs the encoder forward, then pools hidden states to
//! produce a single sentence embedding per input text.
//!
//! Two pooling strategies:
//! - `--pool cls`: take the `[CLS]` hidden state (BERT convention)
//! - `--pool mean`: mean over the non-padding token positions
//!   (sentence-transformers convention; default)
//!
//! Optionally L2-normalises the result so cosine similarity ≡ dot
//! product (sentence-transformers convention; default ON).
//!
//! ## Usage
//!
//! ```bash
//! apr pull sentence-transformers/all-MiniLM-L6-v2
//! apr import .../*.safetensors --arch bert -o embed.apr --allow-no-config
//! apr embed embed.apr \
//!     --text "what is the capital of France?" \
//!     --text "Paris is the capital of France." \
//!     --vocab .../tokenizer.json \
//!     --json
//! ```
//!
//! Output is one `[hidden_dim]` f32 vector per `--text` input.

use crate::error::{CliError, Result};
use aprender::autograd::Tensor;
use aprender::format::v2::AprV2Reader;
use aprender::models::bert::{BertConfig, BertEmbeddings, BertEncoder};
use std::collections::HashMap;
use std::path::Path;

// Re-use Phase 3b's vocab loader. Inlined here as a thin wrapper to avoid a
// public re-export from the `rerank` module.
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
    if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
        for entry in added {
            if let (Some(c), Some(i)) = (
                entry.get("content").and_then(|v| v.as_str()),
                entry.get("id").and_then(|v| v.as_u64()),
            ) {
                map.insert(c.to_string(), i as u32);
            }
        }
    }
    let vocab_obj = root
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "tokenizer.json {}: missing or non-object `model.vocab`",
                path.display()
            ))
        })?;
    for (token, id_val) in vocab_obj {
        if let Some(id) = id_val.as_u64() {
            map.insert(token.to_string(), id as u32);
        }
    }
    Ok(map)
}

/// Tokenise a single text and produce `(input_ids, token_type_ids)`.
///
/// Wraps text as `[CLS] text [SEP]` — the standard single-segment
/// encoding for sentence-embedding bi-encoders. `token_type_ids` is
/// all 0s (single segment).
fn tokenize_single(text: &str, vocab_path: &Path) -> Result<(Vec<u32>, Vec<u32>)> {
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
    if !vocab.contains_key("[UNK]") {
        return Err(CliError::ValidationFailed(format!(
            "vocab {} missing required token [UNK]",
            vocab_path.display()
        )));
    }

    let tokenizer = WordPieceTokenizer::from_vocab(vocab);
    let body_ids = tokenizer
        .encode(text)
        .map_err(|e| CliError::ValidationFailed(format!("tokenisation failed: {e:?}")))?;

    let mut input_ids = Vec::with_capacity(body_ids.len() + 2);
    input_ids.push(cls_id);
    input_ids.extend(&body_ids);
    input_ids.push(sep_id);
    let token_type_ids = vec![0u32; input_ids.len()];
    Ok((input_ids, token_type_ids))
}

/// Apply pooling to `[1, seq_len, hidden_dim]` encoder output → `[hidden_dim]`.
fn pool(hidden: &Tensor, seq_len: usize, hidden_dim: usize, mode: &str) -> Result<Vec<f32>> {
    let data = hidden.data();
    let need = seq_len * hidden_dim;
    if data.len() < need {
        return Err(CliError::ValidationFailed(format!(
            "encoder output {} smaller than seq_len*hidden_dim {}",
            data.len(),
            need
        )));
    }
    match mode {
        "cls" => Ok(data[..hidden_dim].to_vec()),
        "mean" => {
            // Mean across the seq_len token positions. We don't mask padding
            // because Phase 6 doesn't pad — each input is encoded standalone
            // with its own seq_len.
            let mut acc = vec![0.0f32; hidden_dim];
            for t in 0..seq_len {
                let row = &data[t * hidden_dim..(t + 1) * hidden_dim];
                for i in 0..hidden_dim {
                    acc[i] += row[i];
                }
            }
            let denom = seq_len as f32;
            for v in &mut acc {
                *v /= denom;
            }
            Ok(acc)
        }
        other => Err(CliError::ValidationFailed(format!(
            "--pool must be `cls` or `mean`, got {other:?}"
        ))),
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Phase 7 (#326) — read texts from a file, one per line. Blank lines
/// and lines starting with `#` are skipped. Result is appended to the
/// caller's `--text` collection.
fn load_text_file(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to read --text-file {}: {e}",
            path.display()
        ))
    })?;
    let mut out = Vec::new();
    for raw in content.lines() {
        let line = raw.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(line.to_string());
    }
    Ok(out)
}

/// Entry point. Loads the encoder once + processes all texts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: &Path,
    texts: &[String],
    text_file: Option<&Path>,
    vocab: &Path,
    pool_mode: &str,
    normalize: bool,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    json: bool,
) -> Result<()> {
    // Concat `--text` (in CLI order) with `--text-file` rows. Phase 7
    // supports both modes simultaneously so callers can mix a query
    // (`--text "$Q"`) with a passage corpus (`--text-file docs.txt`).
    let mut all_texts: Vec<String> = texts.to_vec();
    if let Some(path) = text_file {
        all_texts.extend(load_text_file(path)?);
    }
    let texts = all_texts.as_slice();

    if texts.is_empty() {
        return Err(CliError::ValidationFailed(
            "apr embed requires at least one --text or --text-file row".to_string(),
        ));
    }

    let model_bytes = std::fs::read(model).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to read {}: {e}", model.display()))
    })?;
    let reader = AprV2Reader::from_bytes(&model_bytes).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to parse APR v2 at {}: {e:?}",
            model.display()
        ))
    })?;

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

    let mut embeddings = BertEmbeddings::new(&config);
    embeddings
        .load_from_reader(&reader, &config)
        .map_err(|e| CliError::ValidationFailed(format!("BERT embeddings load failed: {e}")))?;
    let mut encoder = BertEncoder::new(&config);
    encoder
        .load_from_reader(&reader, &config)
        .map_err(|e| CliError::ValidationFailed(format!("BERT encoder load failed: {e}")))?;

    let mut results: Vec<(String, Vec<f32>)> = Vec::with_capacity(texts.len());
    for text in texts {
        let (input_ids, token_type_ids) = tokenize_single(text, vocab)?;
        let seq_len = input_ids.len();
        let emb_tensor = embeddings.forward(&input_ids, &token_type_ids);
        let hidden = encoder.forward(&emb_tensor, None);
        let mut pooled = pool(&hidden, seq_len, hidden_dim, pool_mode)?;
        if normalize {
            l2_normalize(&mut pooled);
        }
        results.push((text.clone(), pooled));
    }

    if json {
        #[allow(clippy::disallowed_methods)]
        {
            let array: Vec<serde_json::Value> = results
                .iter()
                .map(|(t, v)| {
                    serde_json::json!({
                        "text": t,
                        "embedding": v,
                        "dim": v.len(),
                    })
                })
                .collect();
            let payload = serde_json::json!({
                "model": model.display().to_string(),
                "pool": pool_mode,
                "normalize": normalize,
                "results": array,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&payload).unwrap_or_default()
            );
        }
        return Ok(());
    }

    // Plain text output: one line per text with the first few values + dim.
    for (text, v) in &results {
        let preview: Vec<String> = v.iter().take(4).map(|x| format!("{x:+.4}")).collect();
        println!(
            "text={text:?} dim={} preview=[{}, …]",
            v.len(),
            preview.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::autograd::Tensor;

    #[test]
    fn pool_cls_returns_first_token() {
        let hidden_dim = 4;
        let seq_len = 3;
        let data: Vec<f32> = (0..(seq_len * hidden_dim)).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, &[1, seq_len, hidden_dim]);
        let out = pool(&t, seq_len, hidden_dim, "cls").unwrap();
        // CLS is the first token: data[0..4] = [0,1,2,3].
        assert_eq!(out, vec![0.0f32, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn pool_mean_averages_all_tokens() {
        let hidden_dim = 2;
        let seq_len = 3;
        // tokens: [1,2], [3,4], [5,6]. Mean: [3, 4].
        let t = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, seq_len, hidden_dim],
        );
        let out = pool(&t, seq_len, hidden_dim, "mean").unwrap();
        assert_eq!(out, vec![3.0f32, 4.0]);
    }

    #[test]
    fn pool_rejects_unknown_mode() {
        let t = Tensor::from_vec(vec![0.0f32; 4], &[1, 2, 2]);
        let err = pool(&t, 2, 2, "max").expect_err("max not yet supported");
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("`cls` or `mean`"), "{msg}"),
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0]; // norm = 5
        l2_normalize(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_handles_zero_vector() {
        let mut v = vec![0.0f32; 4];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0f32; 4]);
    }

    /// Phase 7 — `load_text_file` reads one text per line, skipping
    /// blank lines and `#`-prefixed comments. Trailing whitespace is
    /// trimmed.
    #[test]
    fn load_text_file_reads_one_per_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("docs.txt");
        std::fs::write(
            &path,
            "first document\n\
             second document\n\
             # this is a comment\n\
             \n\
             third document   \n",
        )
        .unwrap();
        let texts = load_text_file(&path).expect("load");
        assert_eq!(
            texts,
            vec![
                "first document".to_string(),
                "second document".to_string(),
                "third document".to_string(),
            ]
        );
    }

    /// Phase 7 — empty file yields empty Vec without erroring.
    #[test]
    fn load_text_file_empty_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "").unwrap();
        let texts = load_text_file(&path).expect("load");
        assert!(texts.is_empty());
    }

    /// Phase 7 — file with only comments + blank lines yields empty Vec.
    #[test]
    fn load_text_file_only_comments_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("comments.txt");
        std::fs::write(
            &path,
            "# header comment\n\
             \n\
             # another\n\
             \n",
        )
        .unwrap();
        let texts = load_text_file(&path).expect("load");
        assert!(texts.is_empty());
    }

    /// Phase 7 — missing file produces a structured error mentioning
    /// the path.
    #[test]
    fn load_text_file_missing_path_errors_with_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("does-not-exist.txt");
        let err = load_text_file(&path).expect_err("missing path must error");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("does-not-exist.txt"), "{msg}")
            }
            _ => panic!("expected ValidationFailed"),
        }
    }

    // ── `apr embed --normalize` CLI surface ──────────────────────────
    //
    // `--help` documents "Pass `--normalize false` to keep raw
    // magnitudes". With a bare `bool` field clap derives a SetTrue
    // switch, so `default_value_t = true` pins the value ON and the
    // documented `false` is rejected by the parser (exit 2). These
    // tests assert the PARSED VALUE for each documented spelling, not
    // merely that parsing succeeds.

    /// Parse an exact `apr embed …` argv and return the `normalize`
    /// field. `Err` carries clap's rendered error (what the user sees).
    ///
    /// Parsed on a 16 MiB thread: the flattened `Commands` enum is large
    /// enough to overflow the default test stack (same reason as the
    /// `apr pretrain` CLI tests).
    fn parse_embed_normalize_argv(argv: &[&str]) -> std::result::Result<bool, String> {
        let argv: Vec<String> = argv.iter().map(|s| (*s).to_string()).collect();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                match crate::Cli::try_parse_from(&argv) {
                    Ok(cli) => match *cli.command {
                        crate::Commands::Extended(crate::ExtendedCommands::Embed {
                            normalize,
                            ..
                        }) => Ok(normalize),
                        other => panic!("expected ExtendedCommands::Embed, got {other:?}"),
                    },
                    Err(e) => Err(e.to_string()),
                }
            })
            .expect("spawn parse thread")
            .join()
            .expect("parse thread must not panic")
    }

    /// `apr embed <MODEL> --vocab … --text … <extra>` — the documented
    /// argument order, with `extra` appended.
    fn parse_embed_normalize(extra: &[&str]) -> std::result::Result<bool, String> {
        let mut argv: Vec<&str> = vec![
            "apr",
            "embed",
            "/tmp/_embed_norm/model.apr",
            "--vocab",
            "/tmp/_embed_norm/vocab.txt",
            "--text",
            "hello",
        ];
        argv.extend_from_slice(extra);
        parse_embed_normalize_argv(&argv)
    }

    #[test]
    fn embed_normalize_defaults_to_on_when_omitted() {
        assert_eq!(
            parse_embed_normalize(&[]),
            Ok(true),
            "omitting --normalize must keep the sentence-transformers default (L2-normalise)"
        );
    }

    #[test]
    fn embed_normalize_bare_flag_is_on() {
        assert_eq!(
            parse_embed_normalize(&["--normalize"]),
            Ok(true),
            "a bare --normalize must still mean ON"
        );
    }

    #[test]
    fn embed_normalize_space_separated_false_turns_it_off() {
        // `--help` literally says: "Pass `--normalize false` to keep raw
        // magnitudes". Before the fix clap rejected this with
        // "unexpected argument 'false' found" (exit 2).
        assert_eq!(
            parse_embed_normalize(&["--normalize", "false"]),
            Ok(false),
            "`--normalize false` is documented in --help and MUST reach the handler as false"
        );
    }

    #[test]
    fn embed_normalize_equals_false_turns_it_off() {
        // Before the fix: "unexpected value 'false' for '--normalize'
        // found; no more were expected" (exit 2).
        assert_eq!(
            parse_embed_normalize(&["--normalize=false"]),
            Ok(false),
            "`--normalize=false` MUST reach the handler as false"
        );
    }

    #[test]
    fn embed_normalize_equals_true_is_on() {
        assert_eq!(parse_embed_normalize(&["--normalize=true"]), Ok(true));
    }

    #[test]
    fn embed_normalize_rejects_non_boolean_value() {
        // The optional value must still be type-checked — `--normalize
        // maybe` is a user error, not a silent default.
        let err = parse_embed_normalize(&["--normalize", "maybe"])
            .expect_err("non-boolean value must be rejected");
        assert!(err.contains("maybe"), "{err}");
    }

    /// The one cost of an optional-valued flag: `--normalize` directly
    /// followed by the MODEL positional is ambiguous, and clap resolves
    /// it by treating the path as the flag's value. That is inherent to
    /// `num_args(0..=1)` — the alternative (`require_equals`) would break
    /// the `--normalize false` spelling that `--help` documents.
    ///
    /// It is a legible error listing the accepted values, not a silent
    /// misparse, and `apr embed <MODEL> --normalize` (the documented
    /// order) is unaffected. Asserted here so the trade-off is a
    /// deliberate, reviewed behaviour rather than a latent surprise.
    #[test]
    fn embed_normalize_before_positional_errors_legibly() {
        let err = parse_embed_normalize_argv(&[
            "apr",
            "embed",
            "--normalize",
            "/tmp/_embed_norm/model.apr",
            "--vocab",
            "/tmp/_embed_norm/vocab.txt",
            "--text",
            "hello",
        ])
        .expect_err("`--normalize <MODEL>` consumes the path as the flag value");
        assert!(
            err.contains("model.apr") && err.contains("true") && err.contains("false"),
            "the error must name the offending token and list the accepted values, got: {err}"
        );
    }
}
