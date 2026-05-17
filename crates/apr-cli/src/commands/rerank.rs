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

/// Entry point for `apr rerank` — loads the model, scores the pair, prints
/// the relevance probability (or raw logit) as text or JSON.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: &Path,
    input_ids_str: &str,
    token_type_ids_str: &str,
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
    let input_ids = parse_id_list(input_ids_str, "input-ids")?;
    let token_type_ids = parse_id_list(token_type_ids_str, "token-type-ids")?;

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

    let mut cross_encoder = CrossEncoder::new(&config, num_labels, with_pooler);
    cross_encoder
        .load_from_reader(&reader, &config)
        .map_err(|e| CliError::ValidationFailed(format!("BERT weight loading failed: {e}")))?;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
