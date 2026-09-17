//! Regression fixtures for user-found GGUF load defects (PMAT-3429, #3429).
//!
//! Each defect below was first reported against a multi-GB download. These
//! builders reproduce the file SHAPE that triggered it in a few kilobytes, and
//! `regression_fixture_bytes_are_pinned` pins every builder's bytes by sha256
//! so a fixture cannot drift silently.
//!
//! | issue | fixture | test | revert that turns it RED |
//! |-------|---------|------|--------------------------|
//! | #2535 | `build_qwen3_moe_gguf(Q4_K)` | `regression_2535_…` | `validate_quantized_tensors` checks the MoE dense-FFN placeholders |
//! | #3341 | `build_qwen3_moe_gguf(Q4_0)` | `moe_load_contract_refuses_q4_0_experts` | the load-time expert-qtype contract is skipped |
//! | #3091 | `mixed_qtype_iq2_xxs_base()` | `regression_3091_…` | none yet: a pin of today's refusal, inverted by #3432 |
//!
//! #1749 (bench MoE route) lives in apr-cli, and #1789 (MoE serve route) in `api/`.

use crate::gguf::test_factory::{
    create_f32_embedding_data, create_f32_norm_weights, create_q4_0_data, create_q4_k_data,
    create_q4_k_data_2d, create_q6_k_data, GGUFBuilder,
};
use crate::gguf::types::{GgmlQuantType, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};

const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 256;
const VOCAB: usize = 32;
const NUM_EXPERTS: usize = 2;
const HEADS: usize = 4;

/// Byte payload for a stacked expert tensor of `n` elements at `qtype`.
pub(crate) fn expert_bytes(qtype: u32, n: usize) -> Vec<u8> {
    match qtype {
        GGUF_TYPE_Q4_K => create_q4_k_data(n),
        GGUF_TYPE_Q6_K => create_q6_k_data(n),
        GGUF_TYPE_Q4_0 => create_q4_0_data(n),
        other => panic!("fixture has no byte layout for qtype {other}"),
    }
}

pub(crate) fn add_expert_tensor(
    b: GGUFBuilder,
    name: &str,
    dims: &[u64],
    qtype: u32,
) -> GGUFBuilder {
    let n: usize = dims.iter().map(|&d| d as usize).product();
    let bytes = expert_bytes(qtype, n);
    match qtype {
        GGUF_TYPE_Q4_K => b.add_q4_k_tensor(name, dims, &bytes),
        GGUF_TYPE_Q6_K => b.add_q6_k_tensor(name, dims, &bytes),
        GGUF_TYPE_Q4_0 => b.add_q4_0_tensor(name, dims, &bytes),
        other => panic!("fixture cannot add qtype {other}"),
    }
}

pub(crate) fn build_qwen3_moe_gguf(expert_qtype: u32) -> Vec<u8> {
    let arch = "qwen3moe";
    let expert_dims = [NUM_EXPERTS as u64, INTERMEDIATE as u64, HIDDEN as u64];
    let router_data = vec![0.0f32; NUM_EXPERTS * HIDDEN];

    let b = GGUFBuilder::new()
        .architecture(arch)
        .hidden_dim(arch, HIDDEN as u32)
        .num_layers(arch, 1)
        .num_heads(arch, HEADS as u32)
        .num_kv_heads(arch, HEADS as u32)
        .context_length(arch, 256)
        .rope_freq_base(arch, 10000.0)
        .rms_epsilon(arch, 1e-6)
        .ffn_hidden_dim(arch, INTERMEDIATE as u32)
        .add_u32("qwen3moe.expert_count", NUM_EXPERTS as u32)
        .add_u32("qwen3moe.expert_used_count", 1)
        .add_f32_tensor(
            "token_embd.weight",
            &[VOCAB as u64, HIDDEN as u64],
            &create_f32_embedding_data(VOCAB, HIDDEN),
        )
        .add_f32_tensor(
            "blk.0.attn_norm.weight",
            &[HIDDEN as u64],
            &create_f32_norm_weights(HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_f32_tensor(
            "blk.0.ffn_norm.weight",
            &[HIDDEN as u64],
            &create_f32_norm_weights(HIDDEN),
        )
        .add_f32_tensor(
            "blk.0.ffn_gate_inp.weight",
            &[NUM_EXPERTS as u64, HIDDEN as u64],
            &router_data,
        );

    let b = add_expert_tensor(b, "blk.0.ffn_gate_exps.weight", &expert_dims, expert_qtype);
    let b = add_expert_tensor(b, "blk.0.ffn_up_exps.weight", &expert_dims, expert_qtype);
    let b = add_expert_tensor(b, "blk.0.ffn_down_exps.weight", &expert_dims, expert_qtype);

    b.add_f32_tensor(
        "output_norm.weight",
        &[HIDDEN as u64],
        &create_f32_norm_weights(HIDDEN),
    )
    .build()
}

pub(crate) fn mixed_qtype_iq2_xxs_base() -> Vec<u8> {
    let arch = "llama";
    // IQ2_XXS: 66 bytes per 256-element block, sized from the tensor it is
    // attached to (`ffn_down`, HIDDEN x INTERMEDIATE). A short payload would
    // still be refused today, but for the wrong reason once #3432 adds the arm.
    let iq2_xxs_payload_len = (HIDDEN * INTERMEDIATE).div_ceil(256) * 66;
    let iq2_xxs_payload = vec![0u8; iq2_xxs_payload_len];

    let b = GGUFBuilder::new()
        .architecture(arch)
        .hidden_dim(arch, HIDDEN as u32)
        .num_layers(arch, 1)
        .num_heads(arch, HEADS as u32)
        .num_kv_heads(arch, HEADS as u32)
        .context_length(arch, 256)
        .rope_freq_base(arch, 10000.0)
        .rms_epsilon(arch, 1e-6)
        .ffn_hidden_dim(arch, INTERMEDIATE as u32)
        .add_q4_k_tensor(
            "token_embd.weight",
            &[VOCAB as u64, HIDDEN as u64],
            &create_q4_k_data_2d(VOCAB, HIDDEN),
        )
        .add_f32_tensor(
            "blk.0.attn_norm.weight",
            &[HIDDEN as u64],
            &create_f32_norm_weights(HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[HIDDEN as u64, HIDDEN as u64],
            &create_q4_k_data_2d(HIDDEN, HIDDEN),
        )
        .add_f32_tensor(
            "blk.0.ffn_norm.weight",
            &[HIDDEN as u64],
            &create_f32_norm_weights(HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[INTERMEDIATE as u64, HIDDEN as u64],
            &create_q4_k_data_2d(INTERMEDIATE, HIDDEN),
        )
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[INTERMEDIATE as u64, HIDDEN as u64],
            &create_q4_k_data_2d(INTERMEDIATE, HIDDEN),
        )
        .add_raw_tensor(
            "blk.0.ffn_down.weight",
            &[HIDDEN as u64, INTERMEDIATE as u64],
            GgmlQuantType::IQ2XXS as u32,
            &iq2_xxs_payload,
        )
        .add_f32_tensor(
            "output_norm.weight",
            &[HIDDEN as u64],
            &create_f32_norm_weights(HIDDEN),
        );

    b.build()
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel, QuantizedGGUFTransformer};
    use sha2::{Digest, Sha256};
    use std::io::Write;

    #[test]
    fn regression_fixture_bytes_are_pinned() {
        let q4_k_moe = build_qwen3_moe_gguf(GGUF_TYPE_Q4_K);
        let q4_0_moe = build_qwen3_moe_gguf(GGUF_TYPE_Q4_0);
        let iq2_xxs_base = mixed_qtype_iq2_xxs_base();

        let hash_q4_k_moe = format!("{:x}", Sha256::digest(&q4_k_moe));
        let hash_q4_0_moe = format!("{:x}", Sha256::digest(&q4_0_moe));
        let hash_iq2_xxs_base = format!("{:x}", Sha256::digest(&iq2_xxs_base));

        const PIN: &str =
            "fixture bytes changed — a fixture change must be deliberate: update the pin in the same commit";
        assert_eq!(
            hash_q4_k_moe, "32686a821fb10525fc9167d178d7a3877ef2e78119c846b6f8eac0f04745c8fd",
            "q4_k MoE: {PIN}"
        );
        assert_eq!(
            hash_q4_0_moe, "2679ae0ecf2738f7accb0be3695ebb17f958f95012fb53dd5ab0866549fdb8c1",
            "q4_0 MoE: {PIN}"
        );
        assert_eq!(
            hash_iq2_xxs_base, "f4b71031b357af294fc6776a0f116d09b8ae577016c6cc46023a84c7e0525aae",
            "iq2_xxs base: {PIN}"
        );
    }

    /// #2535: a COMPLETE MoE file was refused as "truncated/corrupt … file is
    /// incomplete" because `validate_quantized_tensors` checked the dense-FFN
    /// placeholder slots a MoE model deliberately leaves empty.
    #[test]
    fn regression_2535_complete_moe_passes_validate_quantized_tensors() {
        let bytes = build_qwen3_moe_gguf(GGUF_TYPE_Q4_K);
        let mut temp_file = tempfile::NamedTempFile::new().expect("create temp GGUF");
        temp_file.write_all(&bytes).expect("write temp GGUF");
        let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("map fixture GGUF");

        if let Err(e) = OwnedQuantizedModel::from_mapped(&mapped) {
            panic!("#2535 regression: a complete MoE GGUF was refused at load: {e}");
        }
    }

    /// CHARACTERIZATION PIN of the #3091 reopen, not a statement of intent: a GGUF
    /// carrying an IQ2_XXS (ggml type 16) tensor is refused today because
    /// `tensor_byte_size` has no arm for it. #3432 (one refusal + byte-size table)
    /// must INVERT this test to assert the load succeeds; that inversion is the
    /// observed RED→GREEN. Until then, a silent change of the refusal fails here.
    #[test]
    fn regression_3091_iq2_xxs_in_shared_base_refused_until_3432() {
        let bytes = mixed_qtype_iq2_xxs_base();
        let model = GGUFModel::from_bytes(&bytes).expect("fixture GGUF must parse");

        match QuantizedGGUFTransformer::from_gguf(&model, &bytes) {
            Ok(_) => panic!(
                "#3091: the IQ2_XXS file now loads — if #3432 landed, invert this pin to assert success"
            ),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("tensor_byte_size")
                        && msg.contains("Unsupported quantization type: 16"),
                    "#3091 (#3432): the refusal changed shape; got: {msg}"
                );
            },
        }
    }
}
