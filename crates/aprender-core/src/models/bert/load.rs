//! BERT weight loading from APR v2 files (GH-326 Phase 1).
//!
//! Loads pre-trained BERT cross-encoder weights from `.apr` files produced
//! by `apr import hf://...`. The tensor name convention matches
//! HuggingFace BERT:
//!
//! ```text
//! Embeddings:
//!   bert.embeddings.word_embeddings.weight
//!   bert.embeddings.position_embeddings.weight
//!   bert.embeddings.token_type_embeddings.weight
//!   bert.embeddings.LayerNorm.{weight,bias}
//!
//! Per layer N:
//!   bert.encoder.layer.{N}.attention.self.{query,key,value}.{weight,bias}
//!   bert.encoder.layer.{N}.attention.output.dense.{weight,bias}
//!   bert.encoder.layer.{N}.attention.output.LayerNorm.{weight,bias}
//!   bert.encoder.layer.{N}.intermediate.dense.{weight,bias}
//!   bert.encoder.layer.{N}.output.dense.{weight,bias}
//!   bert.encoder.layer.{N}.output.LayerNorm.{weight,bias}
//!
//! Optional cross-encoder head:
//!   bert.pooler.dense.{weight,bias}
//!   classifier.{weight,bias}
//! ```
//!
//! `Architecture::Bert.bert_map_name` (tensor_expectation.rs) currently passes
//! these names through verbatim, so the APR file uses the exact HF names.

use crate::autograd::Tensor;
use crate::format::v2::AprV2Reader;
use crate::models::bert::{BertConfig, BertEmbeddings, BertEncoder, BertLayer, CrossEncoder};

/// Error returned when a required tensor is missing or has the wrong shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BertLoadError {
    /// Tensor name that triggered the error.
    pub tensor: String,
    /// One-line description of what went wrong.
    pub reason: String,
}

impl std::fmt::Display for BertLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BertLoadError({}: {})", self.tensor, self.reason)
    }
}

impl std::error::Error for BertLoadError {}

/// Read a tensor by name from the reader, dequantising to f32 if needed.
///
/// Returns `BertLoadError` if the tensor is missing or the dtype path fails.
fn read_tensor(
    reader: &AprV2Reader,
    name: &str,
    expected_shape: &[usize],
) -> Result<Tensor, BertLoadError> {
    let entry = reader.get_tensor(name).ok_or_else(|| BertLoadError {
        tensor: name.to_string(),
        reason: "tensor not present in APR file".to_string(),
    })?;

    let data = reader
        .get_tensor_as_f32(name)
        .ok_or_else(|| BertLoadError {
            tensor: name.to_string(),
            reason: format!("get_tensor_as_f32 failed for dtype {:?}", entry.dtype),
        })?;

    let expected_numel: usize = expected_shape.iter().product();
    if data.len() != expected_numel {
        return Err(BertLoadError {
            tensor: name.to_string(),
            reason: format!(
                "element count mismatch: got {}, expected {} (shape {:?})",
                data.len(),
                expected_numel,
                expected_shape
            ),
        });
    }

    Ok(Tensor::from_vec(data, expected_shape))
}

/// Load weights into a previously-constructed `BertEmbeddings`.
///
/// Reads:
/// - `bert.embeddings.word_embeddings.weight`     `[vocab_size, hidden_dim]`
/// - `bert.embeddings.position_embeddings.weight` `[max_pos, hidden_dim]`
/// - `bert.embeddings.token_type_embeddings.weight` `[type_vocab, hidden_dim]`
/// - `bert.embeddings.LayerNorm.{weight,bias}`    `[hidden_dim]`
///
/// Mutates the existing zero-init tensors in `embeddings`. The
/// `BertEmbeddings` fields are crate-private (`pub(crate)`) so this helper
/// lives next to them.
pub(crate) fn load_embeddings_from_reader(
    embeddings: &mut BertEmbeddings,
    reader: &AprV2Reader,
    config: &BertConfig,
) -> Result<(), BertLoadError> {
    let h = config.hidden_dim;
    embeddings.word_embeddings = read_tensor(
        reader,
        "bert.embeddings.word_embeddings.weight",
        &[config.vocab_size, h],
    )?;
    embeddings.position_embeddings = read_tensor(
        reader,
        "bert.embeddings.position_embeddings.weight",
        &[config.max_position_embeddings, h],
    )?;
    embeddings.token_type_embeddings = read_tensor(
        reader,
        "bert.embeddings.token_type_embeddings.weight",
        &[config.type_vocab_size, h],
    )?;
    embeddings.layer_norm.set_weight(read_tensor(
        reader,
        "bert.embeddings.LayerNorm.weight",
        &[h],
    )?);
    embeddings
        .layer_norm
        .set_bias(read_tensor(reader, "bert.embeddings.LayerNorm.bias", &[h])?);
    Ok(())
}

/// Load weights into a previously-constructed `BertLayer` at index `idx`.
///
/// Reads the 6 weight/bias pairs that make up one BERT encoder block:
/// Q/K/V/O projections in attention, attention LayerNorm, intermediate +
/// output dense, output LayerNorm.
pub(crate) fn load_layer_from_reader(
    layer: &mut BertLayer,
    reader: &AprV2Reader,
    idx: usize,
    config: &BertConfig,
) -> Result<(), BertLoadError> {
    let h = config.hidden_dim;
    let im = config.intermediate_dim;
    let prefix = format!("bert.encoder.layer.{idx}");

    // Q/K/V projections — square `[h, h]` weight + `[h]` bias.
    for (proj, name) in [("query", "q"), ("key", "k"), ("value", "v")] {
        let w_name = format!("{prefix}.attention.self.{proj}.weight");
        let b_name = format!("{prefix}.attention.self.{proj}.bias");
        let weight = read_tensor(reader, &w_name, &[h, h])?;
        let bias = read_tensor(reader, &b_name, &[h])?;
        let proj_linear = match name {
            "q" => layer.attention_mut().q_proj_mut(),
            "k" => layer.attention_mut().k_proj_mut(),
            "v" => layer.attention_mut().v_proj_mut(),
            _ => unreachable!("only q/k/v iterated"),
        };
        proj_linear.set_weight(weight);
        proj_linear.set_bias(bias);
    }

    // Attention output projection.
    let attn_out_w = read_tensor(
        reader,
        &format!("{prefix}.attention.output.dense.weight"),
        &[h, h],
    )?;
    let attn_out_b = read_tensor(
        reader,
        &format!("{prefix}.attention.output.dense.bias"),
        &[h],
    )?;
    layer.attention_mut().out_proj_mut().set_weight(attn_out_w);
    layer.attention_mut().out_proj_mut().set_bias(attn_out_b);

    // Attention post-residual LayerNorm.
    layer.attention_norm_mut().set_weight(read_tensor(
        reader,
        &format!("{prefix}.attention.output.LayerNorm.weight"),
        &[h],
    )?);
    layer.attention_norm_mut().set_bias(read_tensor(
        reader,
        &format!("{prefix}.attention.output.LayerNorm.bias"),
        &[h],
    )?);

    // FFN expand projection (h → im).
    let intermediate_w = read_tensor(
        reader,
        &format!("{prefix}.intermediate.dense.weight"),
        &[im, h],
    )?;
    let intermediate_b = read_tensor(reader, &format!("{prefix}.intermediate.dense.bias"), &[im])?;
    layer.intermediate_mut().set_weight(intermediate_w);
    layer.intermediate_mut().set_bias(intermediate_b);

    // FFN contract projection (im → h).
    let output_w = read_tensor(reader, &format!("{prefix}.output.dense.weight"), &[h, im])?;
    let output_b = read_tensor(reader, &format!("{prefix}.output.dense.bias"), &[h])?;
    layer.output_dense_mut().set_weight(output_w);
    layer.output_dense_mut().set_bias(output_b);

    // FFN post-residual LayerNorm.
    layer.output_norm_mut().set_weight(read_tensor(
        reader,
        &format!("{prefix}.output.LayerNorm.weight"),
        &[h],
    )?);
    layer.output_norm_mut().set_bias(read_tensor(
        reader,
        &format!("{prefix}.output.LayerNorm.bias"),
        &[h],
    )?);

    Ok(())
}

/// Load weights into a previously-constructed `BertEncoder`.
///
/// Iterates `config.num_layers` and loads each via `load_layer_from_reader`.
/// Stops at the first missing/mismatched tensor with `BertLoadError`.
pub(crate) fn load_encoder_from_reader(
    encoder: &mut BertEncoder,
    reader: &AprV2Reader,
    config: &BertConfig,
) -> Result<(), BertLoadError> {
    let num_layers = config.num_layers;
    if encoder.num_layers() != num_layers {
        return Err(BertLoadError {
            tensor: "<encoder>".to_string(),
            reason: format!(
                "encoder has {} layers but config says {num_layers}",
                encoder.num_layers()
            ),
        });
    }
    for idx in 0..num_layers {
        load_layer_from_reader(encoder.layer_mut(idx), reader, idx, config)?;
    }
    Ok(())
}

/// Load weights into a previously-constructed `CrossEncoder`.
///
/// In addition to the embedding + encoder tables, loads:
/// - `bert.pooler.dense.{weight,bias}` if the encoder has a pooler
/// - `classifier.{weight,bias}` for the relevance head
///
/// Different cross-encoder checkpoints persist the head under different
/// names (`classifier`, `score`, `rank_head`). This loader tries
/// `classifier` first and falls back to other common names with a clear
/// error if none match.
pub(crate) fn load_cross_encoder_from_reader(
    model: &mut CrossEncoder,
    reader: &AprV2Reader,
    config: &BertConfig,
) -> Result<(), BertLoadError> {
    let h = config.hidden_dim;
    load_embeddings_from_reader(model.embeddings_mut(), reader, config)?;
    load_encoder_from_reader(model.encoder_mut(), reader, config)?;

    if let Some(pooler) = model.pooler_mut() {
        pooler.set_weight(read_tensor(reader, "bert.pooler.dense.weight", &[h, h])?);
        pooler.set_bias(read_tensor(reader, "bert.pooler.dense.bias", &[h])?);
    }

    // Cross-encoder head — try common names. Shape is [num_labels, h] for
    // weight + [num_labels] for bias.
    let num_labels = model.num_labels();
    let mut tried: Vec<String> = Vec::new();
    for prefix in ["classifier", "score", "rank_head"] {
        let w_name = format!("{prefix}.weight");
        let b_name = format!("{prefix}.bias");
        if reader.get_tensor(&w_name).is_some() {
            let w = read_tensor(reader, &w_name, &[num_labels, h])?;
            let b = read_tensor(reader, &b_name, &[num_labels])?;
            model.classifier_mut().set_weight(w);
            model.classifier_mut().set_bias(b);
            return Ok(());
        }
        tried.push(prefix.to_string());
    }
    Err(BertLoadError {
        tensor: "<classifier head>".to_string(),
        reason: format!(
            "no classifier tensor found; tried prefixes {tried:?} \
             (expected one of `classifier.weight`, `score.weight`, `rank_head.weight`)"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};

    #[test]
    fn bert_load_error_display_includes_tensor_and_reason() {
        let err = BertLoadError {
            tensor: "bert.embeddings.word_embeddings.weight".to_string(),
            reason: "tensor not present in APR file".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("bert.embeddings.word_embeddings.weight"));
        assert!(display.contains("tensor not present"));
    }

    /// Tiny shape-compatible BERT config so the round-trip test can build a
    /// synthetic APR file in <1ms without allocating BERT-base-sized
    /// (109M-param) tensors.
    fn tiny_config() -> BertConfig {
        BertConfig {
            vocab_size: 32,
            hidden_dim: 8,
            num_layers: 2,
            num_heads: 2,
            intermediate_dim: 16,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    /// Build a synthetic APR v2 byte blob containing all tensors a
    /// `CrossEncoder` loader expects, sized per `tiny_config`.
    fn build_stub_bert_apr(config: &BertConfig, with_pooler: bool, num_labels: usize) -> Vec<u8> {
        let h = config.hidden_dim;
        let im = config.intermediate_dim;
        let mut w = AprV2Writer::new(AprV2Metadata::default());

        // Embeddings.
        w.add_f32_tensor(
            "bert.embeddings.word_embeddings.weight",
            vec![config.vocab_size, h],
            &vec![0.1f32; config.vocab_size * h],
        );
        w.add_f32_tensor(
            "bert.embeddings.position_embeddings.weight",
            vec![config.max_position_embeddings, h],
            &vec![0.01f32; config.max_position_embeddings * h],
        );
        w.add_f32_tensor(
            "bert.embeddings.token_type_embeddings.weight",
            vec![config.type_vocab_size, h],
            &vec![0.001f32; config.type_vocab_size * h],
        );
        w.add_f32_tensor(
            "bert.embeddings.LayerNorm.weight",
            vec![h],
            &vec![1.0f32; h],
        );
        w.add_f32_tensor("bert.embeddings.LayerNorm.bias", vec![h], &vec![0.0f32; h]);

        // Per layer.
        for idx in 0..config.num_layers {
            let p = format!("bert.encoder.layer.{idx}");
            // Q/K/V/O — [h,h] weight + [h] bias.
            for proj in ["query", "key", "value"] {
                w.add_f32_tensor(
                    &format!("{p}.attention.self.{proj}.weight"),
                    vec![h, h],
                    &vec![0.0f32; h * h],
                );
                w.add_f32_tensor(
                    &format!("{p}.attention.self.{proj}.bias"),
                    vec![h],
                    &vec![0.0f32; h],
                );
            }
            w.add_f32_tensor(
                &format!("{p}.attention.output.dense.weight"),
                vec![h, h],
                &vec![0.0f32; h * h],
            );
            w.add_f32_tensor(
                &format!("{p}.attention.output.dense.bias"),
                vec![h],
                &vec![0.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.attention.output.LayerNorm.weight"),
                vec![h],
                &vec![1.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.attention.output.LayerNorm.bias"),
                vec![h],
                &vec![0.0f32; h],
            );
            // FFN.
            w.add_f32_tensor(
                &format!("{p}.intermediate.dense.weight"),
                vec![im, h],
                &vec![0.0f32; im * h],
            );
            w.add_f32_tensor(
                &format!("{p}.intermediate.dense.bias"),
                vec![im],
                &vec![0.0f32; im],
            );
            w.add_f32_tensor(
                &format!("{p}.output.dense.weight"),
                vec![h, im],
                &vec![0.0f32; h * im],
            );
            w.add_f32_tensor(&format!("{p}.output.dense.bias"), vec![h], &vec![0.0f32; h]);
            w.add_f32_tensor(
                &format!("{p}.output.LayerNorm.weight"),
                vec![h],
                &vec![1.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.output.LayerNorm.bias"),
                vec![h],
                &vec![0.0f32; h],
            );
        }

        // Pooler + classifier head.
        if with_pooler {
            w.add_f32_tensor("bert.pooler.dense.weight", vec![h, h], &vec![0.0f32; h * h]);
            w.add_f32_tensor("bert.pooler.dense.bias", vec![h], &vec![0.0f32; h]);
        }
        w.add_f32_tensor(
            "classifier.weight",
            vec![num_labels, h],
            &vec![0.0f32; num_labels * h],
        );
        w.add_f32_tensor(
            "classifier.bias",
            vec![num_labels],
            &vec![0.0f32; num_labels],
        );

        w.write().expect("AprV2Writer must produce bytes")
    }

    /// **FALSIFY-BERT-326-PHASE1-LOAD** — `CrossEncoder::load_from_reader`
    /// loads all weights from a synthetic APR file without panic or
    /// `BertLoadError`. Validates the full tensor-name expectation set:
    /// embeddings (3 tables + LN), N layers × (Q/K/V/O proj + 2 LayerNorms +
    /// intermediate + output), pooler + classifier.
    #[test]
    fn falsify_bert_326_phase1_load_full_cross_encoder() {
        let config = tiny_config();
        let bytes = build_stub_bert_apr(&config, true, 1);
        let reader = AprV2Reader::from_bytes(&bytes).expect("AprV2Reader parse");

        let mut model = CrossEncoder::new(&config, 1, true);
        model
            .load_from_reader(&reader, &config)
            .expect("CrossEncoder::load_from_reader must succeed for full BERT-named APR");

        // Sanity: the loaded model can forward on a minimal input without
        // panicking. Output shape is `[1, num_labels]`.
        let input_ids = vec![1u32, 2, 3];
        let token_type_ids = vec![0u32, 0, 0];
        let out = model.forward(&input_ids, &token_type_ids);
        assert_eq!(out.shape(), &[1, 1]);
    }

    /// Loader reports `BertLoadError` on a missing classifier head with a
    /// message that names all the prefixes it tried (helps users diagnose
    /// custom-named heads in their checkpoints).
    #[test]
    fn falsify_bert_326_phase1_missing_classifier_returns_structured_error() {
        let config = tiny_config();
        // Skip writing classifier.* — everything else present.
        let h = config.hidden_dim;
        let im = config.intermediate_dim;
        let mut w = AprV2Writer::new(AprV2Metadata::default());
        w.add_f32_tensor(
            "bert.embeddings.word_embeddings.weight",
            vec![config.vocab_size, h],
            &vec![0.0f32; config.vocab_size * h],
        );
        w.add_f32_tensor(
            "bert.embeddings.position_embeddings.weight",
            vec![config.max_position_embeddings, h],
            &vec![0.0f32; config.max_position_embeddings * h],
        );
        w.add_f32_tensor(
            "bert.embeddings.token_type_embeddings.weight",
            vec![config.type_vocab_size, h],
            &vec![0.0f32; config.type_vocab_size * h],
        );
        w.add_f32_tensor(
            "bert.embeddings.LayerNorm.weight",
            vec![h],
            &vec![1.0f32; h],
        );
        w.add_f32_tensor("bert.embeddings.LayerNorm.bias", vec![h], &vec![0.0f32; h]);
        for idx in 0..config.num_layers {
            let p = format!("bert.encoder.layer.{idx}");
            for proj in ["query", "key", "value"] {
                w.add_f32_tensor(
                    &format!("{p}.attention.self.{proj}.weight"),
                    vec![h, h],
                    &vec![0.0f32; h * h],
                );
                w.add_f32_tensor(
                    &format!("{p}.attention.self.{proj}.bias"),
                    vec![h],
                    &vec![0.0f32; h],
                );
            }
            w.add_f32_tensor(
                &format!("{p}.attention.output.dense.weight"),
                vec![h, h],
                &vec![0.0f32; h * h],
            );
            w.add_f32_tensor(
                &format!("{p}.attention.output.dense.bias"),
                vec![h],
                &vec![0.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.attention.output.LayerNorm.weight"),
                vec![h],
                &vec![1.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.attention.output.LayerNorm.bias"),
                vec![h],
                &vec![0.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.intermediate.dense.weight"),
                vec![im, h],
                &vec![0.0f32; im * h],
            );
            w.add_f32_tensor(
                &format!("{p}.intermediate.dense.bias"),
                vec![im],
                &vec![0.0f32; im],
            );
            w.add_f32_tensor(
                &format!("{p}.output.dense.weight"),
                vec![h, im],
                &vec![0.0f32; h * im],
            );
            w.add_f32_tensor(&format!("{p}.output.dense.bias"), vec![h], &vec![0.0f32; h]);
            w.add_f32_tensor(
                &format!("{p}.output.LayerNorm.weight"),
                vec![h],
                &vec![1.0f32; h],
            );
            w.add_f32_tensor(
                &format!("{p}.output.LayerNorm.bias"),
                vec![h],
                &vec![0.0f32; h],
            );
        }
        // No pooler, no classifier — should error on the head.
        let bytes = w.write().expect("AprV2Writer must produce bytes");
        let reader = AprV2Reader::from_bytes(&bytes).expect("AprV2Reader parse");

        let mut model = CrossEncoder::new(&config, 1, false);
        let err = model
            .load_from_reader(&reader, &config)
            .expect_err("loader must report missing classifier");
        assert!(
            err.reason.contains("classifier"),
            "error reason must reference classifier tried prefixes: {err:?}"
        );
    }

    /// Loader reports `BertLoadError` on a shape mismatch with a message
    /// that names the tensor + the expected vs actual element count.
    #[test]
    fn falsify_bert_326_phase1_shape_mismatch_returns_structured_error() {
        let config = tiny_config();
        let h = config.hidden_dim;
        let mut w = AprV2Writer::new(AprV2Metadata::default());
        // Wrong shape: vocab_size=99 instead of config.vocab_size=32.
        w.add_f32_tensor(
            "bert.embeddings.word_embeddings.weight",
            vec![99, h],
            &vec![0.0f32; 99 * h],
        );
        let bytes = w.write().expect("AprV2Writer must produce bytes");
        let reader = AprV2Reader::from_bytes(&bytes).expect("AprV2Reader parse");

        let mut emb = BertEmbeddings::new(&config);
        let err = load_embeddings_from_reader(&mut emb, &reader, &config)
            .expect_err("loader must reject shape mismatch");
        assert!(err.reason.contains("element count mismatch"), "{err:?}");
        assert!(err.tensor.contains("word_embeddings"), "{err:?}");
    }
}
