
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_creation() {
        let contract = LayoutContract::new();

        // Verify all expected tensors are present
        assert!(contract.get_gguf_contract("output.weight").is_some());
        assert!(contract.get_gguf_contract("token_embd.weight").is_some());
        assert!(contract.get_gguf_contract("output_norm.weight").is_some());
    }

    #[test]
    fn test_f_layout_contract_001_all_2d_transposed() {
        // F-LAYOUT-CONTRACT-001: All 2D weights are transposed
        let contract = LayoutContract::new();

        let transpose_tensors = contract.transpose_tensors();
        assert!(
            !transpose_tensors.is_empty(),
            "Should have transpose tensors"
        );

        for tensor in transpose_tensors {
            assert!(
                tensor.should_transpose,
                "Tensor {} should be transposed",
                tensor.gguf_name
            );
            // 2D tensors have different GGUF and APR shapes
            assert_ne!(
                tensor.gguf_shape_formula, tensor.apr_shape_formula,
                "Tensor {} should have different GGUF/APR shapes",
                tensor.gguf_name
            );
        }
    }

    #[test]
    fn test_f_layout_contract_002_lm_head_shape() {
        // F-LAYOUT-CONTRACT-002: lm_head shape matches kernel expectation
        let contract = LayoutContract::new();

        let lm_head = contract
            .get_gguf_contract("output.weight")
            .expect("lm_head contract should exist");

        assert!(lm_head.is_critical, "lm_head should be critical");
        assert_eq!(lm_head.apr_shape_formula, "[vocab, hidden]");
        assert_eq!(lm_head.kernel_out_dim, "vocab_size");
        assert_eq!(lm_head.kernel_in_dim, "hidden_dim");

        // Validate actual shape
        let result = contract.validate_apr_shape("lm_head.weight", &[151936, 896], 151936, 896);
        assert!(result.is_ok(), "Valid lm_head shape should pass");

        // Invalid shape should fail
        let result = contract.validate_apr_shape("lm_head.weight", &[896, 151936], 151936, 896);
        assert!(result.is_err(), "Swapped lm_head shape should fail");
    }

    #[test]
    fn test_f_layout_contract_003_1d_unchanged() {
        // F-LAYOUT-CONTRACT-003: 1D tensors unchanged
        let contract = LayoutContract::new();

        let non_transpose = contract.non_transpose_tensors();
        assert!(!non_transpose.is_empty(), "Should have 1D tensors");

        for tensor in non_transpose {
            assert!(
                !tensor.should_transpose,
                "1D tensor {} should NOT be transposed",
                tensor.gguf_name
            );
            // 1D tensors have same GGUF and APR shapes
            assert_eq!(
                tensor.gguf_shape_formula, tensor.apr_shape_formula,
                "1D tensor {} should have same GGUF/APR shapes",
                tensor.gguf_name
            );
        }
    }

    #[test]
    fn test_f_layout_contract_004_byte_size() {
        // F-LAYOUT-CONTRACT-004: Byte size matches kernel expectation

        // Q6K: 210 bytes per super-block, 256 elements per super-block
        // For lm_head [vocab=151936, hidden=896]:
        // Expected = vocab * ceil(hidden/256) * 210 = 151936 * 4 * 210 = 127,626,240
        let expected = LayoutContract::calculate_q6k_bytes(151936, 896);
        assert_eq!(expected, 127_626_240, "Q6K byte calculation should match");

        // Q4K: 144 bytes per super-block
        let expected_q4k = LayoutContract::calculate_q4k_bytes(151936, 896);
        assert_eq!(
            expected_q4k,
            151936 * 4 * 144,
            "Q4K byte calculation should match"
        );
    }

    #[test]
    fn test_pattern_matching() {
        let contract = LayoutContract::new();

        // Test GGUF pattern matching
        assert!(contract.get_gguf_contract("blk.0.attn_q.weight").is_some());
        assert!(contract.get_gguf_contract("blk.15.attn_q.weight").is_some());
        assert!(contract.get_gguf_contract("blk.99.attn_k.weight").is_some());

        // Test APR pattern matching
        assert!(contract
            .get_apr_contract("model.layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(contract
            .get_apr_contract("model.layers.27.mlp.gate_proj.weight")
            .is_some());
    }

    #[test]
    fn test_normalize_layer_pattern() {
        assert_eq!(
            normalize_layer_pattern("blk.0.attn_q.weight"),
            "blk.{n}.attn_q.weight"
        );
        assert_eq!(
            normalize_layer_pattern("blk.15.ffn_gate.weight"),
            "blk.{n}.ffn_gate.weight"
        );
        assert_eq!(
            normalize_layer_pattern("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.{n}.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_layer_pattern("output.weight"),
            "output.weight" // No layer number
        );
    }

    #[test]
    fn test_critical_tensors() {
        let contract = LayoutContract::new();
        let critical = contract.critical_tensors();

        // Only lm_head should be critical
        assert_eq!(critical.len(), 1, "Only lm_head should be critical");
        assert_eq!(critical[0].gguf_name, "output.weight");
    }

    #[test]
    fn test_should_transpose() {
        let contract = LayoutContract::new();

        // 2D tensors should transpose
        assert!(contract.should_transpose_gguf("output.weight"));
        assert!(contract.should_transpose_gguf("token_embd.weight"));
        assert!(contract.should_transpose_gguf("blk.0.attn_q.weight"));
        assert!(contract.should_transpose_gguf("blk.5.ffn_down.weight"));

        // 1D tensors should NOT transpose
        assert!(!contract.should_transpose_gguf("output_norm.weight"));
        assert!(!contract.should_transpose_gguf("blk.0.attn_norm.weight"));
        assert!(!contract.should_transpose_gguf("blk.3.ffn_norm.weight"));
    }

    #[test]
    fn test_global_contract() {
        // Test the global lazy static instance
        assert!(contract().get_gguf_contract("output.weight").is_some());
        assert!(contract().should_transpose_gguf("output.weight"));
        assert!(!contract().should_transpose_gguf("output_norm.weight"));
    }

    // ========================================================================
    // MANDATORY ENFORCEMENT TESTS (GH-208)
    // These tests FAIL if contract enforcement is bypassed
    // ========================================================================

    #[test]
    fn test_enforce_import_contract_embedding() {
        // GGUF embedding: [hidden=1536, vocab=151936]
        // APR embedding: [vocab=151936, hidden=1536] (shape reversed, NO data transpose)
        let (apr_shape, needs_transpose) =
            enforce_import_contract("token_embd.weight", &[1536, 151936], 151936, 1536);

        assert_eq!(
            apr_shape,
            vec![151936, 1536],
            "Embedding shape must be [vocab, hidden]"
        );
        assert!(!needs_transpose, "GGUF→APR NEVER needs data transpose");
    }

    #[test]
    fn test_enforce_import_contract_lm_head() {
        // GGUF lm_head: [hidden=1536, vocab=151936]
        // APR lm_head: [vocab=151936, hidden=1536]
        let (apr_shape, needs_transpose) =
            enforce_import_contract("output.weight", &[1536, 151936], 151936, 1536);

        assert_eq!(
            apr_shape,
            vec![151936, 1536],
            "LM head shape must be [vocab, hidden]"
        );
        assert!(!needs_transpose, "GGUF→APR NEVER needs data transpose");
    }

    #[test]
    fn test_enforce_import_contract_1d_unchanged() {
        // 1D tensors: shape unchanged, no transpose
        let (apr_shape, needs_transpose) =
            enforce_import_contract("output_norm.weight", &[1536], 151936, 1536);

        assert_eq!(apr_shape, vec![1536], "1D tensor shape unchanged");
        assert!(!needs_transpose, "1D tensors never need transpose");
    }

    #[test]
    fn test_enforce_embedding_contract_valid() {
        // Valid embedding: vocab * hidden elements
        enforce_embedding_contract(151936 * 1536, 151936, 1536);
        // Should not panic
    }

    #[test]
    #[should_panic(expected = "CONTRACT VIOLATION")]
    fn test_enforce_embedding_contract_invalid() {
        // Invalid embedding: wrong number of elements
        enforce_embedding_contract(1000, 151936, 1536);
        // Should panic
    }

    #[test]
    fn test_enforce_matmul_contract_valid() {
        // Valid lm_head: [vocab=151936, hidden=1536]
        enforce_matmul_contract("lm_head.weight", &[151936, 1536], 151936, 1536);
        // Should not panic
    }

    #[test]
    #[should_panic(expected = "CONTRACT VIOLATION")]
    fn test_enforce_matmul_contract_wrong_shape() {
        // Wrong lm_head: [hidden=1536, vocab=151936] (not transposed)
        enforce_matmul_contract("lm_head.weight", &[1536, 151936], 151936, 1536);
        // Should panic - this was the GH-202 root cause
    }

    #[test]
    fn test_enforce_load_contract_critical_tensor() {
        // Valid critical tensor
        let result = enforce_load_contract("lm_head.weight", &[151936, 1536], 151936, 1536);
        assert!(result.is_ok(), "Valid lm_head shape should pass");

        // Invalid critical tensor
        let result = enforce_load_contract("lm_head.weight", &[1536, 151936], 151936, 1536);
        assert!(result.is_err(), "Invalid lm_head shape MUST fail");
    }

    #[test]
    fn test_data_transpose_never_needed() {
        // This test codifies the GH-208 lesson:
        // GGUF data layout data[i0 + i1*ne0] for shape [ne0, ne1] IS row-major [ne1, ne0]
        // Therefore data transpose is NEVER needed during GGUF→APR import.

        let all_tensors = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ];

        for tensor in all_tensors {
            let (_, needs_transpose) = enforce_import_contract(tensor, &[100, 200], 200, 100);
            assert!(
                !needs_transpose,
                "GGUF tensor '{}' should NEVER need data transpose. \
                 If this fails, you're repeating the GH-208 bug.",
                tensor
            );
        }
    }

    // ============================================================================
    // FALSIFY: format-parity-v1.yaml — proptest properties
    // ============================================================================

    /// FALSIFY-FP-001: Transpose involution — swap(swap(shape)) == shape
    #[test]
    fn test_falsify_fp001_transpose_involution() {
        // For any 2D shape, applying enforce_import_contract twice must return original shape
        let shapes: &[(usize, usize)] = &[
            (896, 151936),
            (151936, 896),
            (4864, 896),
            (896, 4864),
            (1, 1),
            (128, 64),
            (65536, 256),
        ];

        for &(a, b) in shapes {
            let gguf_shape = vec![a, b];
            // First pass: GGUF → APR
            let (apr_shape, _) = enforce_import_contract("output.weight", &gguf_shape, b, a);
            assert_eq!(apr_shape, vec![b, a], "FALSIFY-FP-001: first transpose failed for [{a}, {b}]");

            // Second pass: applying the same swap again must recover original
            let recovered = vec![apr_shape[1], apr_shape[0]];
            assert_eq!(recovered, gguf_shape,
                "FALSIFY-FP-001: transpose involution violated for [{a}, {b}] — swap(swap(s)) != s");
        }
    }

    /// FALSIFY-FP-002: Element count preserved — product(gguf) == product(apr)
    #[test]
    fn test_falsify_fp002_element_count_preserved() {
        let test_cases: &[(&str, &[usize])] = &[
            ("output.weight", &[896, 151936]),
            ("token_embd.weight", &[896, 151936]),
            ("blk.0.attn_q.weight", &[896, 896]),
            ("blk.0.ffn_gate.weight", &[4864, 896]),
            ("output_norm.weight", &[896]),        // 1D — no transpose
            ("blk.0.attn_norm.weight", &[896]),    // 1D — no transpose
        ];

        for &(name, gguf_shape) in test_cases {
            let gguf_elements: usize = gguf_shape.iter().product();
            let (apr_shape, _) = enforce_import_contract(name, gguf_shape, 151936, 896);
            let apr_elements: usize = apr_shape.iter().product();
            assert_eq!(gguf_elements, apr_elements,
                "FALSIFY-FP-002: element count changed for '{name}': GGUF {gguf_elements} != APR {apr_elements}");
        }
    }

    /// FALSIFY-FP-003: 1D identity — 1D shapes pass through unchanged
    #[test]
    fn test_falsify_fp003_1d_identity() {
        let one_d_tensors = &[
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ];

        for &name in one_d_tensors {
            for &dim in &[896_usize, 1536, 4096, 1] {
                let input = vec![dim];
                let (output, needs_transpose) = enforce_import_contract(name, &input, 151936, dim);
                assert_eq!(output, input,
                    "FALSIFY-FP-003: 1D tensor '{name}' [{dim}] was modified during import");
                assert!(!needs_transpose,
                    "FALSIFY-FP-003: 1D tensor '{name}' [{dim}] should never need data transpose");
            }
        }
    }

    /// FALSIFY-FP-004: Data transpose is NEVER needed (GH-208 regression guard)
    #[test]
    fn test_falsify_fp004_data_transpose_never_needed() {
        // For ANY tensor, enforce_import_contract must return needs_data_transpose=false
        // This is the core GH-208 lesson: GGUF data is already row-major when shape is swapped
        let all_dims: &[(usize, usize)] = &[
            (896, 896), (4864, 896), (896, 4864),
            (151936, 896), (896, 151936),
            (128, 64), (1, 65536),
        ];

        let tensor_names = &[
            "output.weight", "token_embd.weight",
            "blk.0.attn_q.weight", "blk.0.attn_k.weight",
            "blk.0.attn_v.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        ];

        for &name in tensor_names {
            for &(a, b) in all_dims {
                let (_, needs_transpose) = enforce_import_contract(name, &[a, b], b, a);
                assert!(!needs_transpose,
                    "FALSIFY-FP-004: '{name}' [{a}, {b}] claims data transpose needed — GH-208 regression!");
            }
        }
    }
}
