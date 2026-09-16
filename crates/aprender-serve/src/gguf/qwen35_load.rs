#![allow(missing_docs)]
use crate::error::Result;
use crate::gguf::quantized::QuantizedTensorRef;
use crate::gguf::types::GGUFValue;
use crate::gguf::GGUFModel;
use crate::gguf::QuantizedGGUFTransformer;

#[derive(Clone, Debug)]
pub struct Qwen35DeltaNetLayer {
    pub attn_norm: QuantizedTensorRef,
    pub attn_qkv: QuantizedTensorRef,
    pub attn_gate: QuantizedTensorRef,
    pub ssm_alpha: QuantizedTensorRef,
    pub ssm_beta: QuantizedTensorRef,
    pub ssm_a: QuantizedTensorRef,
    pub ssm_dt_bias: QuantizedTensorRef,
    pub ssm_conv1d_weight: QuantizedTensorRef,
    pub ssm_norm_weight: QuantizedTensorRef,
    pub ssm_out: QuantizedTensorRef,
    pub post_attention_norm: QuantizedTensorRef,
    pub ffn_gate: QuantizedTensorRef,
    pub ffn_up: QuantizedTensorRef,
    pub ffn_down: QuantizedTensorRef,
}

#[derive(Clone, Debug)]
pub struct Qwen35AttentionLayer {
    pub attn_norm: QuantizedTensorRef,
    pub attn_q: QuantizedTensorRef,
    pub attn_k: QuantizedTensorRef,
    pub attn_v: QuantizedTensorRef,
    pub attn_q_norm: QuantizedTensorRef,
    pub attn_k_norm: QuantizedTensorRef,
    pub attn_output: QuantizedTensorRef,
    pub post_attention_norm: QuantizedTensorRef,
    pub ffn_gate: QuantizedTensorRef,
    pub ffn_up: QuantizedTensorRef,
    pub ffn_down: QuantizedTensorRef,
}

#[derive(Clone, Debug)]
pub enum Qwen35Layer {
    DeltaNet(Qwen35DeltaNetLayer),
    Attention(Qwen35AttentionLayer),
}

fn as_u32(v: &GGUFValue) -> Option<u32> {
    match v {
        GGUFValue::UInt32(x) => Some(*x),
        _ => None,
    }
}

pub fn load_qwen35_layers(model: &GGUFModel, data: &[u8]) -> Result<Vec<Qwen35Layer>> {
    let mut layers = Vec::new();
    let num_layers = model
        .metadata
        .get("block_count")
        .and_then(as_u32)
        .unwrap_or(24) as usize;

    let interval = model
        .metadata
        .get("full_attention_interval")
        .and_then(as_u32)
        .unwrap_or(4) as usize;

    for i in 0..num_layers {
        if (i + 1) % interval == 0 {
            // Full attention
            layers.push(Qwen35Layer::Attention(Qwen35AttentionLayer {
                attn_norm: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_norm.weight", i),
                )?,
                attn_q: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_q.weight", i),
                )?,
                attn_k: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_k.weight", i),
                )?,
                attn_v: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_v.weight", i),
                )?,
                attn_q_norm: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_q_norm.weight", i),
                )?,
                attn_k_norm: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_k_norm.weight", i),
                )?,
                attn_output: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_output.weight", i),
                )?,
                post_attention_norm: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.post_attention_norm.weight", i),
                )?,
                ffn_gate: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ffn_gate.weight", i),
                )?,
                ffn_up: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ffn_up.weight", i),
                )?,
                ffn_down: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ffn_down.weight", i),
                )?,
            }));
        } else {
            // DeltaNet
            layers.push(Qwen35Layer::DeltaNet(Qwen35DeltaNetLayer {
                attn_norm: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_norm.weight", i),
                )?,
                attn_qkv: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_qkv.weight", i),
                )?,
                attn_gate: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.attn_gate.weight", i),
                )?,
                ssm_alpha: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_alpha.weight", i),
                )?,
                ssm_beta: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_beta.weight", i),
                )?,
                ssm_a: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_a", i),
                )?,
                ssm_dt_bias: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_dt.bias", i),
                )?,
                ssm_conv1d_weight: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_conv1d.weight", i),
                )?,
                ssm_norm_weight: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_norm.weight", i),
                )?,
                ssm_out: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ssm_out.weight", i),
                )?,
                post_attention_norm: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.post_attention_norm.weight", i),
                )?,
                ffn_gate: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ffn_gate.weight", i),
                )?,
                ffn_up: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ffn_up.weight", i),
                )?,
                ffn_down: QuantizedGGUFTransformer::get_tensor_ref(
                    model,
                    data,
                    &format!("blk.{}.ffn_down.weight", i),
                )?,
            }));
        }
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen35_load_fixture() {
        let fixture_path = std::path::PathBuf::from("/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf");
        if !fixture_path.exists() {
            println!("SKIP: {} absent", fixture_path.display());
            return;
        }

        let data = std::fs::read(&fixture_path).expect("failed to read fixture file");
        let model = GGUFModel::from_bytes(&data).expect("failed to load fixture");
        let layers = load_qwen35_layers(&model, &data).expect("failed to extract layers");

        assert_eq!(layers.len(), 24);

        let mut num_full_attn = 0;
        let mut num_delta_net = 0;

        for (i, layer) in layers.iter().enumerate() {
            match layer {
                Qwen35Layer::Attention(attn) => {
                    num_full_attn += 1;
                    assert_eq!((i + 1) % 4, 0);
                    // Check some shapes
                    assert_eq!(attn.attn_q.num_elements, 1024 * 4096);
                    assert_eq!(attn.attn_k.num_elements, 1024 * 512);
                    assert_eq!(attn.attn_v.num_elements, 1024 * 512);
                },
                Qwen35Layer::DeltaNet(delta) => {
                    num_delta_net += 1;
                    assert_ne!((i + 1) % 4, 0);
                    // Check some shapes
                    assert_eq!(delta.attn_qkv.num_elements, 1024 * 6144);
                    assert_eq!(delta.ssm_alpha.num_elements, 1024 * 16);
                    assert_eq!(delta.ssm_conv1d_weight.num_elements, 4 * 6144);
                },
            }
        }

        assert_eq!(num_delta_net, 18);
        assert_eq!(num_full_attn, 6);
    }
}
