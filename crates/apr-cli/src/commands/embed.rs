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

/// Entry point. Loads the encoder once + processes all texts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: &Path,
    texts: &[String],
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
    if texts.is_empty() {
        return Err(CliError::ValidationFailed(
            "apr embed requires at least one --text".to_string(),
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
}
