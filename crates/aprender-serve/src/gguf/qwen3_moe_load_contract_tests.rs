// #3341 — load-time contract for Qwen3-MoE expert tensors.
//
// `apr run Qwen3-Coder-30B-A3B-Instruct-Q4_0.gguf --no-gpu` used to LOAD
// cleanly and then die at the first token with
// `Operation 'moe_expert_matvec' not supported: MoE expert tensor qtype 2
// not supported` — no tensor name, no file name, after the whole model had
// been mapped. The loader already knew every expert tensor's qtype; nobody
// asked it. These tests pin the refusal at load, and pin the const that
// keeps the loader's accepted set and the forward dispatcher's arms from
// drifting apart.
//
// Named `moe_load_contract_*` so `cargo nextest run -p aprender-serve --lib
// moe_load_contract` selects exactly this file.

#[cfg(test)]
mod moe_load_contract_tests {
    use super::super::test_factory::{
        create_f32_embedding_data, create_f32_norm_weights, create_q4_0_data, create_q4_k_data,
        create_q4_k_data_2d, create_q6_k_data, GGUFBuilder,
    };
    use super::*;
    use crate::gguf::types::{GGUF_TYPE_Q4_0, GGUF_TYPE_Q5_K, GGUF_TYPE_Q8_0};
    use crate::gguf::{GGUFModel, QuantizedGGUFTransformer};

    const HIDDEN: usize = 256;
    const INTERMEDIATE: usize = 256;
    const VOCAB: usize = 32;
    const NUM_EXPERTS: usize = 2;
    const HEADS: usize = 4;

    /// Byte payload for a stacked expert tensor of `n` elements at `qtype`.
    ///
    /// Sizes mirror `QuantizedGGUFTransformer::tensor_byte_size`'s non-2D
    /// branch, so the fixture's declared dims and its bytes agree and
    /// `resolve_qtype`'s truncation fallback never fires (which would
    /// silently rewrite the qtype under test).
    fn expert_bytes(qtype: u32, n: usize) -> Vec<u8> {
        match qtype {
            GGUF_TYPE_Q4_K => create_q4_k_data(n),
            GGUF_TYPE_Q6_K => create_q6_k_data(n),
            GGUF_TYPE_Q4_0 => create_q4_0_data(n),
            other => panic!("fixture has no byte layout for qtype {other}"),
        }
    }

    fn add_expert_tensor(b: GGUFBuilder, name: &str, dims: &[u64], qtype: u32) -> GGUFBuilder {
        let n: usize = dims.iter().map(|&d| d as usize).product();
        let bytes = expert_bytes(qtype, n);
        match qtype {
            GGUF_TYPE_Q4_K => b.add_q4_k_tensor(name, dims, &bytes),
            GGUF_TYPE_Q6_K => b.add_q6_k_tensor(name, dims, &bytes),
            GGUF_TYPE_Q4_0 => b.add_q4_0_tensor(name, dims, &bytes),
            other => panic!("fixture cannot add qtype {other}"),
        }
    }

    /// A minimal single-layer `qwen3moe` GGUF whose three expert tensors all
    /// carry `expert_qtype`. Everything else (attention, norms, embedding,
    /// F32 router) is exactly what the real Qwen3-Coder file carries, so the
    /// ONLY variable between the accept and refuse cases is the expert qtype.
    fn build_qwen3_moe_gguf(expert_qtype: u32) -> Vec<u8> {
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
            // Router is F32 in every shipped Qwen3-MoE file; the forward
            // reinterprets these bytes as `&[f32]`.
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

    fn dummy_ref(qtype: u32, byte_size: usize) -> QuantizedTensorRef {
        QuantizedTensorRef {
            offset: 0,
            byte_size,
            num_elements: byte_size,
            qtype,
        }
    }

    fn layer_with(router: QuantizedTensorRef, experts: QuantizedTensorRef) -> Qwen3MoeQuantizedLayer {
        Qwen3MoeQuantizedLayer {
            router,
            gate_exps: experts.clone(),
            up_exps: experts.clone(),
            down_exps: experts,
        }
    }

    /// THE defect (#3341): a Q4_0-expert MoE GGUF must be refused at LOAD,
    /// with the offending tensor named — not at the first token, nameless.
    #[test]
    fn moe_load_contract_refuses_q4_0_experts() {
        let bytes = build_qwen3_moe_gguf(GGUF_TYPE_Q4_0);
        let model = GGUFModel::from_bytes(&bytes).expect("fixture GGUF must parse");

        // `expect_err` is unavailable: the Ok type is not Debug.
        let err = match QuantizedGGUFTransformer::from_gguf_for_moe(&model, &bytes) {
            Ok(_) => panic!(
                "a Q4_0-expert MoE GGUF loaded successfully — this is #3341: the failure \
                 is deferred to the first token, where no tensor is named"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();

        assert!(
            msg.contains("blk.0.ffn_gate_exps.weight"),
            "refusal must NAME the tensor (that is the whole point of #3341), got: {msg}"
        );
        assert!(
            msg.contains("qtype"),
            "refusal must say which qtype, got: {msg}"
        );
        assert!(
            msg.contains("Q4_0(2)"),
            "refusal must name the qtype the file actually carries, got: {msg}"
        );
        assert!(
            msg.contains("#3341"),
            "refusal must cite the ticket so the next reader finds the ruling, got: {msg}"
        );
    }

    /// Anti-vacuity partner of the test above: the SAME fixture with Q4_K
    /// experts must still load. Without this, a check that refuses every MoE
    /// file would pass the refusal test.
    #[test]
    fn moe_load_contract_accepts_q4_k_experts() {
        let bytes = build_qwen3_moe_gguf(GGUF_TYPE_Q4_K);
        let model = GGUFModel::from_bytes(&bytes).expect("fixture GGUF must parse");

        let transformer = QuantizedGGUFTransformer::from_gguf_for_moe(&model, &bytes)
            .expect("Q4_K experts are supported — load must still succeed");
        let moe = transformer.moe_layers[0]
            .as_ref()
            .expect("layer 0 must carry MoE descriptors");
        assert_eq!(moe.gate_exps.qtype, GGUF_TYPE_Q4_K);
        assert!(moe.gate_exps.byte_size > 0);
    }

    /// Q6_K experts (the down-projection quant in real Q4_K_M files) also load.
    #[test]
    fn moe_load_contract_accepts_q6_k_experts() {
        let bytes = build_qwen3_moe_gguf(GGUF_TYPE_Q6_K);
        let model = GGUFModel::from_bytes(&bytes).expect("fixture GGUF must parse");
        let transformer = QuantizedGGUFTransformer::from_gguf_for_moe(&model, &bytes)
            .expect("Q6_K experts are supported — load must still succeed");
        assert_eq!(
            transformer.moe_layers[0]
                .as_ref()
                .expect("layer 0 MoE descriptors")
                .down_exps
                .qtype,
            GGUF_TYPE_Q6_K
        );
    }

    /// The drift gate: `SUPPORTED_EXPERT_QTYPES` is the ONE source of truth.
    ///
    /// - every qtype IN the const must reach a kernel (no
    ///   `moe_expert_matvec` refusal), so a const entry without a match arm
    ///   fails here;
    /// - every qtype OUTSIDE it must be refused, so a match arm added
    ///   without a const entry fails here too.
    #[test]
    fn moe_load_contract_supported_qtypes_match_matvec_dispatch() {
        let in_dim = 256usize;
        let out_dim = 1usize;
        let activations = vec![0.0f32; in_dim];

        assert!(
            !SUPPORTED_EXPERT_QTYPES.is_empty(),
            "an empty support set would make every assertion below vacuous"
        );

        for &qtype in SUPPORTED_EXPERT_QTYPES {
            let weights = expert_bytes(qtype, in_dim * out_dim);
            let got = matvec_for_qtype(qtype, &weights, &activations, in_dim, out_dim);
            if let Err(e) = &got {
                assert!(
                    !matches!(e, RealizarError::UnsupportedOperation { operation, .. }
                        if operation == "moe_expert_matvec"),
                    "qtype {qtype} is in SUPPORTED_EXPERT_QTYPES but matvec_for_qtype \
                     refuses it — the const and the dispatcher have drifted: {e}"
                );
            }
        }

        // 0..=30 covers every ggml type id this crate's reader knows.
        for qtype in 0u32..=30 {
            if SUPPORTED_EXPERT_QTYPES.contains(&qtype) {
                continue;
            }
            let err = matvec_for_qtype(qtype, &[], &activations, in_dim, out_dim).expect_err(
                &format!("qtype {qtype} is outside SUPPORTED_EXPERT_QTYPES so matvec_for_qtype \
                          must refuse it — a kernel arm was added without updating the const"),
            );
            assert!(
                matches!(&err, RealizarError::UnsupportedOperation { operation, .. }
                    if operation == "moe_expert_matvec"),
                "qtype {qtype} must be refused by the moe_expert_matvec guard, got: {err}"
            );
        }
    }

    /// The loader's accepted set IS the dispatcher's set — asserted against
    /// the real load path, not just the kernel: Q5_K and Q8_0 are readable
    /// tensor types (`tensor_byte_size` sizes them) that the MoE forward has
    /// no kernel for, so the load must refuse them.
    #[test]
    fn moe_load_contract_refuses_qtypes_outside_the_supported_set() {
        for qtype in [GGUF_TYPE_Q5_K, GGUF_TYPE_Q8_0, GGUF_TYPE_Q4_0] {
            assert!(
                !SUPPORTED_EXPERT_QTYPES.contains(&qtype),
                "test premise: qtype {qtype} must be outside the supported set"
            );
            let layer = layer_with(dummy_ref(GGUF_TYPE_F32, 2048), dummy_ref(qtype, 4096));
            let err = validate_moe_layer_tensors(7, &layer)
                .expect_err("unsupported expert qtype must be refused at load");
            let msg = err.to_string();
            assert!(
                msg.contains("blk.7.ffn_gate_exps.weight"),
                "refusal must name the tensor of the layer it was given, got: {msg}"
            );
        }
    }

    /// An expert tensor the forward will read but which carries no bytes is
    /// just as unusable as a wrong qtype — and just as silent until decode.
    #[test]
    fn moe_load_contract_refuses_empty_expert_tensor() {
        let layer = layer_with(dummy_ref(GGUF_TYPE_F32, 2048), dummy_ref(GGUF_TYPE_Q4_K, 0));
        let err = validate_moe_layer_tensors(3, &layer)
            .expect_err("a 0-byte expert tensor must be refused at load");
        let msg = err.to_string();
        assert!(msg.contains("blk.3.ffn_gate_exps.weight"), "got: {msg}");
        assert!(msg.contains("0 bytes"), "got: {msg}");
    }

    /// The router is read as raw F32 by `moe_ffn_forward_layer`; a quantized
    /// router is a first-token failure unless the load refuses it.
    #[test]
    fn moe_load_contract_refuses_quantized_router() {
        let layer = layer_with(dummy_ref(GGUF_TYPE_Q4_K, 2048), dummy_ref(GGUF_TYPE_Q4_K, 4096));
        let err = validate_moe_layer_tensors(0, &layer)
            .expect_err("a non-F32 router must be refused at load");
        let msg = err.to_string();
        assert!(msg.contains("blk.0.ffn_gate_inp.weight"), "got: {msg}");
        assert!(msg.contains("qtype"), "got: {msg}");
    }

    /// An empty router is refused too (it would slice 0 bytes and read past
    /// the end in `moe_ffn_forward_layer`'s byte-size check).
    #[test]
    fn moe_load_contract_refuses_empty_router() {
        let layer = layer_with(dummy_ref(GGUF_TYPE_F32, 0), dummy_ref(GGUF_TYPE_Q4_K, 4096));
        let err = validate_moe_layer_tensors(0, &layer)
            .expect_err("a 0-byte router must be refused at load");
        assert!(
            err.to_string().contains("blk.0.ffn_gate_inp.weight"),
            "got: {err}"
        );
    }

    /// A fully supported layer passes — the positive control for every
    /// refusal above.
    #[test]
    fn moe_load_contract_accepts_a_supported_layer() {
        let layer = Qwen3MoeQuantizedLayer {
            router: dummy_ref(GGUF_TYPE_F32, 2048),
            gate_exps: dummy_ref(GGUF_TYPE_Q4_K, 4096),
            up_exps: dummy_ref(GGUF_TYPE_Q4_K, 4096),
            down_exps: dummy_ref(GGUF_TYPE_Q6_K, 4096),
        };
        validate_moe_layer_tensors(0, &layer)
            .expect("Q4_K gate/up + Q6_K down + F32 router is exactly Qwen3-Coder Q4_K_M");
    }
}
