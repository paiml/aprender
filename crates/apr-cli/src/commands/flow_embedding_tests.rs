// APR-EMBED-001 EG-1: `apr flow` draws an embedding model by kind (flow_embedding.rs).

#[cfg(test)]
mod flow_embedding_tests {
    use super::*;
    use aprender::format::rosetta::{FormatType, InspectionReport, TensorInfo};
    use std::collections::BTreeMap;

    fn tensor(name: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F32".to_string(),
            shape: shape.to_vec(),
            size_bytes: 0,
            stats: None,
        }
    }

    /// A 4-layer embedding report: layers 0..=2 local (attn_q 512 → 1024), layer 3 global (→ 2048).
    fn embedding_report(arch: &str, q_out: [usize; 4]) -> InspectionReport {
        let mut metadata = BTreeMap::new();
        for (k, v) in [
            ("general.architecture".to_string(), arch.to_string()),
            (format!("{arch}.attention.causal"), "false".to_string()),
            (format!("{arch}.embedding_length"), "512".to_string()),
            (format!("{arch}.block_count"), "4".to_string()),
            (
                format!("{arch}.attention.sliding_window"),
                "1024".to_string(),
            ),
            (format!("{arch}.pooling_type"), "1".to_string()),
            (format!("{arch}.embedding_length_out"), "768".to_string()),
        ] {
            metadata.insert(k, v);
        }
        let mut tensors = vec![
            tensor("token_embd.weight", &[512, 262_144]),
            tensor("output.weight", &[512, 768]),
            tensor("blk.0.ffn_gate.weight", &[512, 2048]),
        ];
        for (l, w) in q_out.iter().enumerate() {
            tensors.push(tensor(&format!("blk.{l}.attn_q.weight"), &[512, *w]));
        }
        InspectionReport {
            format: FormatType::Gguf,
            file_size: 0,
            metadata,
            tensors,
            total_params: 0,
            quantization: None,
            architecture: Some(arch.to_string()),
        }
    }

    const LOCAL_LOCAL_LOCAL_GLOBAL: [usize; 4] = [1024, 1024, 1024, 2048];

    #[test]
    fn embedding_flow_reads_kind_layers_pooling_and_width_from_the_file_eg1() {
        let flow = embedding_flow(&embedding_report(
            "gemma-embedding2",
            LOCAL_LOCAL_LOCAL_GLOBAL,
        ))
        .expect("admitted")
        .expect("an embedding model");
        assert_eq!(flow.hidden, 512);
        assert_eq!(global_layers(&flow), vec![3]);
        assert_eq!(flow.sliding_window, Some(1024));
        assert!(flow.gated_ffn);
        assert_eq!(flow.pooling, "mean");
        assert_eq!(flow.output_dim, Some(768));
    }

    /// I-9: the kinds follow the shapes, never the layer index.
    #[test]
    fn swapping_two_layers_attn_q_shapes_swaps_their_kinds_eg1() {
        let flow = embedding_flow(&embedding_report(
            "gemma-embedding2",
            [2048, 1024, 1024, 1024],
        ))
        .expect("admitted")
        .expect("an embedding model");
        assert_eq!(global_layers(&flow), vec![0]);
    }

    /// The `gemma` prefix match is not widened, and a generative file keeps the generative flow.
    #[test]
    fn generative_and_other_gemma_architectures_are_not_drawn_as_embedding_eg1() {
        for arch in ["qwen2", "gemma3", "gemma-embedding"] {
            let got = embedding_flow(&embedding_report(arch, LOCAL_LOCAL_LOCAL_GLOBAL))
                .expect("not refused");
            assert!(got.is_none(), "{arch}");
        }
        let mut no_arch = embedding_report("gemma-embedding2", LOCAL_LOCAL_LOCAL_GLOBAL);
        no_arch.metadata.remove("general.architecture");
        assert!(embedding_flow(&no_arch).expect("not refused").is_none());
    }

    #[test]
    fn embedding_architecture_with_causal_true_is_refused_eg1() {
        let mut report = embedding_report("gemma-embedding2", LOCAL_LOCAL_LOCAL_GLOBAL);
        report.metadata.insert(
            "gemma-embedding2.attention.causal".to_string(),
            "true".to_string(),
        );
        assert!(embedding_flow(&report).is_err());
    }

    #[test]
    fn single_attn_q_width_or_missing_layer_is_refused_eg1() {
        assert!(embedding_flow(&embedding_report("gemma-embedding2", [1024; 4])).is_err());
        let mut report = embedding_report("gemma-embedding2", LOCAL_LOCAL_LOCAL_GLOBAL);
        report.tensors.retain(|t| t.name != "blk.2.attn_q.weight");
        assert!(embedding_flow(&report).is_err());
    }

    /// The EG-0 probe's silent wrong answer: an LM head and a next-token output.
    #[test]
    fn embedding_flow_lines_draw_pool_projection_and_l2_never_an_lm_head_eg1() {
        let flow = embedding_flow(&embedding_report(
            "gemma-embedding2",
            LOCAL_LOCAL_LOCAL_GLOBAL,
        ))
        .expect("admitted")
        .expect("an embedding model");
        let text = embedding_flow_lines(&flow).join("\n");
        for want in [
            "bidirectional",
            "local × 3, window 1024; global × 1: 3",
            "gated GELU FFN",
            "Pool: mean",
            "[batch, 512] → [batch, 768]",
            "L2 normalise",
            "OUTPUT  embedding [batch, 768]",
        ] {
            assert!(text.contains(want), "missing {want:?} in\n{text}");
        }
        for never in ["LM Head", "logits", "next_token", "decoder"] {
            assert!(!text.contains(never), "{never:?} in\n{text}");
        }
    }

    #[test]
    fn boxed_pads_every_row_to_the_widest_eg1() {
        let rows = boxed("ab", &["abcd".to_string()]);
        let widths: Vec<usize> = rows.iter().map(|r| r.chars().count()).collect();
        assert_eq!(widths, vec![8, 8, 8, 8], "{rows:?}");
    }

    #[test]
    fn embedding_flow_json_names_the_kind_and_global_layers_eg1() {
        let report = embedding_report("gemma-embedding2", LOCAL_LOCAL_LOCAL_GLOBAL);
        let flow = embedding_flow(&report)
            .expect("admitted")
            .expect("an embedding model");
        let json = embedding_flow_json(Path::new("/m/e.gguf"), &report, &flow);
        assert_eq!(json["kind"], "embedding");
        assert_eq!(json["file"], "e.gguf");
        assert_eq!(json["global_layers"], serde_json::json!([3]));
        assert_eq!(json["pooling"], "mean");
        assert_eq!(json["output_dim"], 768);
    }

    #[test]
    fn only_component_full_draws_an_embedding_model_eg1() {
        let report = embedding_report("gemma-embedding2", LOCAL_LOCAL_LOCAL_GLOBAL);
        let flow = embedding_flow(&report)
            .expect("admitted")
            .expect("an embedding model");
        for component in [
            FlowComponent::Encoder,
            FlowComponent::Decoder,
            FlowComponent::CrossAttention,
            FlowComponent::SelfAttention,
            FlowComponent::Ffn,
        ] {
            let err = run_embedding_flow(Path::new("e.gguf"), &report, &flow, component, true)
                .expect_err("refused");
            assert!(err.to_string().contains("--component full"), "{err}");
        }
        run_embedding_flow(
            Path::new("e.gguf"),
            &report,
            &flow,
            FlowComponent::Full,
            true,
        )
        .expect("full is drawn");
    }

    #[test]
    fn pooling_names_follow_llama_pooling_type_eg1() {
        let name = |v: &str| pooling_name(Some(&v.to_string()));
        assert_eq!(pooling_name(None), "unspecified");
        assert_eq!(name("0"), "none");
        assert_eq!(name("1"), "mean");
        assert_eq!(name("2"), "cls");
        assert_eq!(name("3"), "last");
        assert_eq!(name("4"), "rank");
        assert_eq!(name("9"), "pooling_type 9");
    }
}
