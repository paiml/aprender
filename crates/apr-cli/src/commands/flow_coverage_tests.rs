
// ════════════════════════════════════════════════════════════════════
// Coverage tests for print_cross_attention_flow, print_self_attention_flow,
// print_ffn_flow, and compute_stats (PMAT coverage gap)
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod flow_print_cross_tests {
    use super::*;

    // ── compute_stats ──────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_empty() {
        let (min, max, mean, std) = compute_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_element() {
        let (min, max, mean, std) = compute_stats(&[42.0]);
        assert_eq!(min, 42.0);
        assert_eq!(max, 42.0);
        assert_eq!(mean, 42.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_uniform() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_known_values() {
        // [1, 2, 3, 4, 5]: mean = 3.0, var = 2.0, std = sqrt(2.0)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert!((mean - 3.0).abs() < 1e-5);
        assert!((std - 2.0_f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, -3.0);
        assert_eq!(max, 3.0);
        assert!((mean - 0.0).abs() < 1e-5);
        // var = (9+1+0+1+9)/5 = 4.0, std = 2.0
        assert!((std - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_stats_large_range() {
        let data = vec![-1000.0, 0.0, 1000.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -1000.0);
        assert_eq!(max, 1000.0);
        assert!((mean - 0.0).abs() < 1e-3);
    }

    // ── print_cross_attention_flow ─────────────────────────────────────

    #[test]
    fn test_print_cross_attention_flow_no_cross_attn_layers() {
        // No cross-attention layers in the tensor names
        let tensor_names = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.k_proj.weight".to_string(),
        ];
        // Should print "No cross-attention layers found" without panicking
        print_cross_attention_flow(None, &tensor_names, None, false);
    }

    #[test]
    fn test_print_cross_attention_flow_empty_names() {
        print_cross_attention_flow(None, &[], None, false);
    }

    #[test]
    fn test_print_cross_attention_flow_with_cross_attn() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.0.encoder_attn.k_proj.weight".to_string(),
            "decoder.layers.0.encoder_attn.v_proj.weight".to_string(),
            "decoder.layers.0.encoder_attn.out_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
        ];
        // Should render cross-attention diagram for 2 layers without panicking
        print_cross_attention_flow(None, &tensor_names, None, false);
    }

    #[test]
    fn test_print_cross_attention_flow_with_layer_filter() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.2.encoder_attn.q_proj.weight".to_string(),
        ];
        // Filter to layer 1 only
        print_cross_attention_flow(None, &tensor_names, Some("layers.1"), false);
    }

    #[test]
    fn test_print_cross_attention_flow_verbose_no_reader() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
        ];
        // Verbose with no reader — should skip weight stats gracefully
        print_cross_attention_flow(None, &tensor_names, None, true);
    }

    #[test]
    fn test_print_cross_attention_flow_cross_attn_variant() {
        // Test with "cross_attn" naming convention (alternative to "encoder_attn")
        let tensor_names = vec![
            "model.layers.0.cross_attn.q_proj.weight".to_string(),
            "model.layers.0.cross_attn.k_proj.weight".to_string(),
        ];
        print_cross_attention_flow(None, &tensor_names, None, false);
    }

    // ── print_self_attention_flow ──────────────────────────────────────

    #[test]
    fn test_print_self_attention_flow_no_panic() {
        print_self_attention_flow(None, &[], None, false);
    }

    #[test]
    fn test_print_self_attention_flow_with_names() {
        let names = vec!["model.layers.0.self_attn.q_proj.weight".to_string()];
        print_self_attention_flow(None, &names, None, true);
    }

    // ── print_ffn_flow ─────────────────────────────────────────────────

    #[test]
    fn test_print_ffn_flow_no_panic() {
        print_ffn_flow(None, &[], None, false);
    }

    #[test]
    fn test_print_ffn_flow_with_names() {
        let names = vec!["model.layers.0.mlp.fc1.weight".to_string()];
        print_ffn_flow(None, &names, None, true);
    }
}
