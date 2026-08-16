//! PMAT-333: WGPU adapter — dequantize OwnedQuantizedModel → WgslForwardPass weights.
//!
//! Converts Q4K/Q6K quantized weights to F32 and uploads to trueno's
//! WgslForwardPass for inference on AMD/Intel/Apple GPUs via Vulkan/Metal/WebGPU.

use crate::dev_trace::dev_trace_enabled;
use crate::error::{RealizarError, Result};
use crate::gguf::OwnedQuantizedModel;
use crate::quantize::{dequantize_q4_k, dequantize_q5_k, dequantize_q6_k};

/// The line the user sees when the wgpu path starts dequantizing weights.
///
/// Dequantizing a 1.5B Q4K model to F32 takes seconds and allocates gigabytes,
/// so the user is told it is happening — but in English. The model geometry
/// (`hidden=`, `heads=`, `intermediate=`) means nothing without the source and
/// is held back for `APR_DEV_TRACE`.
#[must_use]
fn dequant_start_message(
    num_layers: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate: usize,
    dev_trace: bool,
) -> String {
    if dev_trace {
        format!(
            "Preparing GPU weights: dequantizing {num_layers} layers to F32 \
             (hidden={hidden}, heads={num_heads}/{num_kv_heads}, intermediate={intermediate})"
        )
    } else {
        format!("Preparing GPU weights: dequantizing {num_layers} layers to F32")
    }
}

/// The line the user sees when dequantization finishes.
///
/// The F32 footprint is genuinely useful — it is why the machine just spent
/// several GB of RAM — so it stays unconditional.
#[must_use]
fn dequant_done_message(weight_count: usize, total_bytes: usize) -> String {
    format!(
        "GPU weights ready: {weight_count} tensors, {:.1} MB F32",
        total_bytes as f64 / 1e6
    )
}

/// PMAT-333: Dequantize all model weights to F32 for WGPU upload.
///
/// Returns a map of weight name → (F32 data, rows, cols) ready for
/// `WgslForwardPass::upload_weight()`.
#[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "dequant_correctness")]
pub fn dequant_model_weights(
    model: &OwnedQuantizedModel,
) -> Result<Vec<(String, Vec<f32>, usize, usize)>> {
    dequant_model_weights_except(model, &std::collections::HashSet::new())
}

/// Dequantize every weight EXCEPT those named in `skip`.
///
/// #2378 finding 8. `try_wgpu_generate` uploads Q4_K projection weights as raw
/// Q4_K bytes, then called `dequant_model_weights` for the rest -- but that
/// materialized an F32 `Vec` for EVERY tensor including the Q4_K ones, and only
/// the *upload* was skipped. The dequantization and the allocation had already
/// happened; the result was built and thrown away.
///
/// That is the remaining half of the memory problem `batch_wgpu.rs` documents
/// ("called TWICE (28 GB each)", 56 GB -> 28 GB). Calling it once was the first
/// half; not dequantizing what was never going to be uploaded is this one.
///
/// `dequant_model_weights` delegates here with an empty set, so the two other
/// call sites keep their exact previous behaviour.
///
/// # Errors
/// Propagates any tensor that cannot be dequantized.
pub fn dequant_model_weights_except<S: std::hash::BuildHasher>(
    model: &OwnedQuantizedModel,
    skip: &std::collections::HashSet<String, S>,
) -> Result<Vec<(String, Vec<f32>, usize, usize)>> {
    let config = &model.config;
    let hidden = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim();
    let intermediate = config.intermediate_dim;
    let num_layers = model.layers().len();

    let mut weights = Vec::new();

    // A MACRO, not a helper fn, and that is the whole point: macro arguments
    // expand inside the `if`, so `dequant_tensor_public(..)?` is never evaluated
    // for a skipped tensor. A function would evaluate its arguments first and
    // dequantize exactly what we are trying not to dequantize.
    macro_rules! push_w {
        ($name:expr, $data:expr, $rows:expr, $cols:expr $(,)?) => {{
            let n: String = $name;
            if !skip.contains(&n) {
                weights.push((n, $data, $rows, $cols));
            }
        }};
    }

    eprintln!(
        "{}",
        dequant_start_message(
            num_layers,
            hidden,
            num_heads,
            num_kv_heads,
            intermediate,
            dev_trace_enabled(),
        )
    );

    for (i, layer) in model.layers().iter().enumerate() {
        let prefix = format!("layer.{i}");

        // Norm weights (already F32)
        push_w!(
            format!("{prefix}.attn_norm"),
            layer.attn_norm_weight.clone(),
            1,
            hidden,
        );

        if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            push_w!(format!("{prefix}.ffn_norm"), ffn_norm.clone(), 1, hidden);
        }

        // QKV weights — dequantize from quantized format
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        match &layer.qkv_weight {
            crate::gguf::OwnedQKVWeights::Fused(tensor) => {
                let f32_data = dequant_tensor_public(tensor)?;
                let total_out = q_dim + 2 * kv_dim;
                // Split fused QKV into separate Q, K, V
                let q_data = f32_data[..q_dim * hidden].to_vec();
                let k_data = f32_data[q_dim * hidden..(q_dim + kv_dim) * hidden].to_vec();
                let v_data = f32_data[(q_dim + kv_dim) * hidden..total_out * hidden].to_vec();
                push_w!(format!("{prefix}.q_proj"), q_data, q_dim, hidden);
                push_w!(format!("{prefix}.k_proj"), k_data, kv_dim, hidden);
                push_w!(format!("{prefix}.v_proj"), v_data, kv_dim, hidden);
            },
            crate::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                push_w!(
                    format!("{prefix}.q_proj"),
                    dequant_tensor_public(q)?,
                    q_dim,
                    hidden,
                );
                push_w!(
                    format!("{prefix}.k_proj"),
                    dequant_tensor_public(k)?,
                    kv_dim,
                    hidden,
                );
                push_w!(
                    format!("{prefix}.v_proj"),
                    dequant_tensor_public(v)?,
                    kv_dim,
                    hidden,
                );
            },
        }

        // PMAT-342: QKV biases (required for Qwen2)
        if let Some(ref bias) = layer.qkv_bias {
            // Fused QKV bias: split into q_bias, k_bias, v_bias
            if bias.len() >= q_dim + 2 * kv_dim {
                push_w!(format!("{prefix}.q_bias"), bias[..q_dim].to_vec(), 1, q_dim);
                push_w!(
                    format!("{prefix}.k_bias"),
                    bias[q_dim..q_dim + kv_dim].to_vec(),
                    1,
                    kv_dim,
                );
                push_w!(
                    format!("{prefix}.v_bias"),
                    bias[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec(),
                    1,
                    kv_dim,
                );
            }
        }

        // O projection
        push_w!(
            format!("{prefix}.o_proj"),
            dequant_tensor_public(&layer.attn_output_weight)?,
            hidden,
            q_dim,
        );

        // FFN weights
        if let Some(ref gate) = layer.ffn_gate_weight {
            push_w!(
                format!("{prefix}.gate_proj"),
                dequant_tensor_public(gate)?,
                intermediate,
                hidden,
            );
        }
        push_w!(
            format!("{prefix}.up_proj"),
            dequant_tensor_public(&layer.ffn_up_weight)?,
            intermediate,
            hidden,
        );
        push_w!(
            format!("{prefix}.down_proj"),
            dequant_tensor_public(&layer.ffn_down_weight)?,
            hidden,
            intermediate,
        );

        if (i + 1) % 7 == 0 || i == num_layers - 1 {
            eprintln!("  Dequantized layer {}/{}", i + 1, num_layers);
        }
    }

    // LM head
    push_w!(
        "lm_head".to_string(),
        dequant_tensor_public(model.lm_head_weight())?,
        config.vocab_size,
        hidden,
    );

    // PMAT-345: Weight layout analysis.
    // GGUF stores [ne0, ne1] with data layout data[i0 + i1*ne0].
    // For a weight W with GGUF dims [in_dim, out_dim]:
    //   data[in + out*in_dim] → this IS row-major [out_dim, in_dim]
    // The dequant_tensor produces data in this same order.
    // Our (rows=out_dim, cols=in_dim) labels match the data layout.
    // WGSL GEMV: w[row * K + col] = data[out * in_dim + in] ← CORRECT
    // NO TRANSPOSE NEEDED — GGUF layout is already row-major for [out, in].
    //
    // Previous transpose was WRONG — it double-transposed, causing garbled output.

    let total_bytes: usize = weights.iter().map(|(_, d, _, _)| d.len() * 4).sum();
    eprintln!("{}", dequant_done_message(weights.len(), total_bytes));

    Ok(weights)
}

/// PMAT-364: Extract raw Q4K weight bytes for fused dequant+GEMV on GPU.
/// Returns (name, raw_bytes, rows, cols) for Q4K tensors only. Other types skipped.
pub fn raw_q4k_weights(model: &OwnedQuantizedModel) -> Vec<(String, Vec<u8>, usize, usize)> {
    const GGUF_TYPE_Q4_K: u32 = 12;
    let config = &model.config;
    let hidden = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim();
    let intermediate = config.intermediate_dim;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let mut raw = Vec::new();

    for (i, layer) in model.layers().iter().enumerate() {
        let prefix = format!("layer.{i}");
        // Only extract Q4K projection weights (skip norms, biases)
        let projections: Vec<(&str, &crate::gguf::OwnedQuantizedTensor, usize, usize)> = vec![
            ("o_proj", &layer.attn_output_weight, hidden, q_dim),
            ("up_proj", &layer.ffn_up_weight, intermediate, hidden),
            ("down_proj", &layer.ffn_down_weight, hidden, intermediate),
        ];
        if let Some(ref gate) = layer.ffn_gate_weight {
            raw.push((
                format!("{prefix}.gate_proj"),
                gate.data.clone(),
                intermediate,
                hidden,
            ));
        }
        for (name, tensor, rows, cols) in projections {
            if tensor.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.{name}"), tensor.data.clone(), rows, cols));
            }
        }
        // QKV: handle separate weights
        if let crate::gguf::OwnedQKVWeights::Separate { q, k, v } = &layer.qkv_weight {
            if q.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.q_proj"), q.data.clone(), q_dim, hidden));
            }
            if k.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.k_proj"), k.data.clone(), kv_dim, hidden));
            }
            if v.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.v_proj"), v.data.clone(), kv_dim, hidden));
            }
        }
    }
    raw
}

/// Dequantize a single OwnedQuantizedTensor to F32
/// GH-560: Public per-tensor dequantization for streaming wgpu weight upload.
pub fn dequant_tensor_public(tensor: &crate::gguf::OwnedQuantizedTensor) -> Result<Vec<f32>> {
    const GGUF_TYPE_Q4_K: u32 = 12;
    const GGUF_TYPE_Q6_K: u32 = 14;
    const GGUF_TYPE_Q5_K: u32 = 13;
    const GGUF_TYPE_F32: u32 = 0;
    const GGUF_TYPE_F16: u32 = 1;
    use crate::gguf::{APR_TYPE_Q4, APR_TYPE_Q8};

    match tensor.qtype {
        GGUF_TYPE_Q4_K => dequantize_q4_k(&tensor.data),
        GGUF_TYPE_Q6_K => dequantize_q6_k(&tensor.data),
        GGUF_TYPE_Q5_K => dequantize_q5_k(&tensor.data),
        GGUF_TYPE_F32 => Ok(tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GGUF_TYPE_F16 => Ok(tensor
            .data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        // GH-478: Native APR q4/q8 — per-tensor scratch dequant
        APR_TYPE_Q4 => Ok(crate::apr::dequant::dequantize_apr_q4(
            &tensor.data,
            tensor.in_dim * tensor.out_dim,
        )),
        APR_TYPE_Q8 => Ok(crate::apr::dequant::dequantize_apr_q8(
            &tensor.data,
            tensor.in_dim * tensor.out_dim,
        )),
        other => Err(RealizarError::FormatError {
            reason: format!("Unsupported quantization type {} for WGPU dequant", other),
        }),
    }
}

#[cfg(test)]
mod ticket_free_output_tests {
    use super::{dequant_done_message, dequant_start_message};

    /// A bracketed token such as `[PMAT-333]` or
    /// `[apr-cpu-vs-gpu-output-parity-v1]` is an internal ticket, not something
    /// a user can act on.
    fn has_ticket_tag(line: &str) -> bool {
        line.contains("PMAT-")
            || line.contains("GH-")
            || line.contains("PAR-")
            || line.contains('[')
    }

    /// The defect: every `apr run` that took the wgpu path opened with
    /// `[PMAT-333] Dequantizing 28 layers (hidden=1536, heads=12/2,
    /// intermediate=8960)`. The user is told what is happening, in English,
    /// with no ticket number and no model geometry.
    #[test]
    fn dequant_start_line_is_english_and_ticket_free() {
        let line = dequant_start_message(28, 1536, 12, 2, 8960, false);
        assert!(
            !has_ticket_tag(&line),
            "start line still addresses the user in ticket numbers: {line}"
        );
        assert_eq!(line, "Preparing GPU weights: dequantizing 28 layers to F32");
        for geometry in ["hidden=", "heads=", "intermediate="] {
            assert!(
                !line.contains(geometry),
                "developer geometry {geometry:?} leaked into default output: {line}"
            );
        }
    }

    /// Gating must not delete the diagnostic: the geometry is still available,
    /// it just has to be asked for.
    #[test]
    fn dequant_start_line_keeps_geometry_under_dev_trace() {
        let line = dequant_start_message(28, 1536, 12, 2, 8960, true);
        assert!(line.contains("hidden=1536"), "{line}");
        assert!(line.contains("heads=12/2"), "{line}");
        assert!(line.contains("intermediate=8960"), "{line}");
        assert!(!has_ticket_tag(&line.replace(['(', ')'], "")), "{line}");
    }

    /// The defect: `[PMAT-333] Dequantized 337 weights, 6174.9 MB F32`. The
    /// footprint is worth keeping; the ticket number is not.
    #[test]
    fn dequant_done_line_is_english_and_ticket_free() {
        let line = dequant_done_message(337, 6_174_900_000);
        assert!(
            !has_ticket_tag(&line),
            "done line still addresses the user in ticket numbers: {line}"
        );
        assert_eq!(line, "GPU weights ready: 337 tensors, 6174.9 MB F32");
    }
}

#[cfg(test)]
mod dequant_skip_2378 {
    //! FALSIFY-2378-8: a tensor uploaded as raw Q4_K must not also be
    //! dequantized to F32 and thrown away.
    //!
    //! `try_wgpu_generate` uploads Q4_K projection weights as raw Q4_K bytes,
    //! then called `dequant_model_weights` and skipped the UPLOAD for those
    //! names. The skip was too late: an F32 `Vec` had already been materialized
    //! for each and immediately dropped. On a Q4_K model that is the bulk of
    //! the weights, and it is the remaining half of the memory problem
    //! `batch_wgpu.rs` documents ("called TWICE (28 GB each)").
    //!
    //! Host-side, so this needs no GPU -- which is the point. The defect was
    //! filed as GPU-path and is measurable without one.

    use super::*;
    use crate::gguf::test_helpers::create_test_model_with_config;
    use crate::gguf::GGUFConfig;

    /// hidden_dim must be a multiple of QK_K for the fixture to produce real
    /// Q4_K tensors, which is the whole precondition of this test.
    fn q4k_config() -> GGUFConfig {
        GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 256,
            intermediate_dim: 512,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            query_pre_attn_scalar: None,
            bos_token_id: None,
            eos_token_id: None,
        }
    }

    fn f32_elements(w: &[(String, Vec<f32>, usize, usize)]) -> usize {
        w.iter().map(|(_, d, _, _)| d.len()).sum()
    }

    #[test]
    fn a_skipped_tensor_is_never_dequantized() {
        // THE discriminating test. An earlier version of this compared f32
        // element COUNTS between the filtered and unfiltered results -- and a
        // mutation that evaluated the dequant eagerly and merely skipped the
        // push passed it, because the OUTPUT is identical either way. It
        // measured the result, not the work.
        //
        // This makes the work observable instead: corrupt a Q4_K tensor so that
        // dequantizing it FAILS. If the filter is lazy the corrupt tensor is
        // never touched and the call succeeds; if it is eager the call errors.
        let mut model = create_test_model_with_config(&q4k_config());

        let raw: std::collections::HashSet<String> = raw_q4k_weights(&model)
            .into_iter()
            .map(|(n, _, _, _)| n)
            .collect();
        assert!(
            !raw.is_empty(),
            "the fixture has no Q4_K weights, so the skip is untested"
        );

        // Corrupt the up-projection of layer 0 and confirm it is in the skip set.
        model.layers[0].ffn_up_weight.data.truncate(3);
        let corrupted: Vec<&String> = raw.iter().filter(|n| n.contains("up_proj")).collect();
        assert!(
            !corrupted.is_empty(),
            "the corrupted tensor is not among the raw-Q4K uploads, so this test \
             is aimed at the wrong tensor: {raw:?}"
        );

        // CONTROL: unfiltered must FAIL. Without this, "filtered succeeded"
        // could just mean the corruption was harmless.
        assert!(
            dequant_model_weights(&model).is_err(),
            "the corrupted tensor dequantized fine, so this test cannot \
             distinguish lazy from eager"
        );

        // THE CLAIM: skipping it means never dequantizing it.
        assert!(
            dequant_model_weights_except(&model, &raw).is_ok(),
            "a tensor uploaded as raw Q4_K was dequantized anyway -- the skip \
             happens after the work rather than instead of it"
        );
    }

    #[test]
    fn an_empty_skip_set_is_the_old_behaviour() {
        // dequant_model_weights delegates with an empty set; the other two call
        // sites rely on that being byte-identical to what they had before.
        let model = create_test_model_with_config(&q4k_config());
        let a = dequant_model_weights(&model).expect("a");
        let b = dequant_model_weights_except(&model, &std::collections::HashSet::new()).expect("b");
        assert_eq!(a.len(), b.len());
        assert_eq!(f32_elements(&a), f32_elements(&b));
    }
}
