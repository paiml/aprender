
#[cfg(test)]
mod tests {
    use super::*;

    /// Shared generator factory -- single seed for all deterministic tests
    fn gen() -> SyntheticWeightGenerator {
        SyntheticWeightGenerator::new(42)
    }

    // ------------------------------------------------------------------
    // Determinism
    // ------------------------------------------------------------------

    #[test]
    fn test_generator_deterministic() {
        let gen1 = SyntheticWeightGenerator::new(42);
        let gen2 = SyntheticWeightGenerator::new(42);

        let w1 = gen1.generate_f32(&[10, 10]);
        let w2 = gen2.generate_f32(&[10, 10]);

        assert_eq!(w1, w2, "Same seed should produce same weights");
    }

    #[test]
    fn test_generator_different_seeds() {
        let gen1 = SyntheticWeightGenerator::new(42);
        let gen2 = SyntheticWeightGenerator::new(43);

        let w1 = gen1.generate_f32(&[10, 10]);
        let w2 = gen2.generate_f32(&[10, 10]);

        assert_ne!(w1, w2, "Different seeds should produce different weights");
    }

    // ------------------------------------------------------------------
    // Quantized block sizes -- table-driven
    // Consolidates: test_q4_0_block_size, test_q4_0_multiple_blocks,
    //               test_q8_0_block_size (3 tests -> 1 loop)
    // ------------------------------------------------------------------

    #[test]
    fn test_quant_block_sizes() {
        let g = gen();

        // (label, generate_fn, num_elements, expected_bytes)
        let cases: &[(&str, fn(&SyntheticWeightGenerator, usize) -> Vec<u8>, usize, usize)] = &[
            // Q4_0: 18 bytes per 32-element block
            ("Q4_0 1-block", SyntheticWeightGenerator::generate_q4_0, 32, 18),
            ("Q4_0 2-blocks", SyntheticWeightGenerator::generate_q4_0, 64, 36),
            // Q8_0: 34 bytes per 32-element block
            ("Q8_0 1-block", SyntheticWeightGenerator::generate_q8_0, 32, 34),
        ];

        for &(label, generate_fn, num_elements, expected) in cases {
            let data = generate_fn(&g, num_elements);
            assert_eq!(data.len(), expected, "{label}: expected {expected} bytes for {num_elements} elements");
        }
    }

    // ------------------------------------------------------------------
    // Quant dispatch -- table-driven
    // Consolidates 6 sub-assertions into a single loop
    // ------------------------------------------------------------------

    #[test]
    fn test_quant_dispatch() {
        let g = gen();

        // (quant_type, num_elements, expected_bytes)
        let cases: &[(QuantType, usize, usize)] = &[
            (QuantType::F32, 100, 400),   // 4 bytes per element
            (QuantType::F16, 100, 200),   // 2 bytes per element
            (QuantType::BF16, 100, 200),  // 2 bytes per element
            (QuantType::Q4_K, 256, 144),  // 144 bytes per 256-element super-block
            (QuantType::Q5_K, 32, 22),    // uses Q5_0: 22 bytes per 32-element block
            (QuantType::Q6_K, 32, 34),    // uses Q8_0: 34 bytes per 32-element block
        ];

        for &(quant, num_elements, expected) in cases {
            let data = g.generate_quant(num_elements, quant);
            assert_eq!(data.len(), expected, "{quant:?}: expected {expected} bytes for {num_elements} elements");
        }
    }

    // ------------------------------------------------------------------
    // Model weights
    // ------------------------------------------------------------------

    #[test]
    fn test_model_weights_generation() {
        let config = ModelConfig::tiny();
        let weights = gen().generate_model_weights(&config, QuantType::Q4_0);

        assert_eq!(weights.layer_weights.len(), config.num_layers);
        assert_eq!(weights.output_norm.len(), config.hidden_dim);
        assert!(weights.total_bytes() > 0);
    }

    #[test]
    fn test_model_weights_metrics() {
        let config = ModelConfig::tiny();
        let weights = gen().generate_model_weights(&config, QuantType::F32);

        assert!(weights.total_bytes() > 0);
        assert_eq!(weights.param_count(), config.param_count());
    }

    // ------------------------------------------------------------------
    // Token generators
    // ------------------------------------------------------------------

    #[test]
    fn test_token_generator() {
        let tg = TokenGenerator::new(42, 256);
        let tokens = tg.generate(10);

        assert_eq!(tokens.len(), 10);
        assert!(tokens.iter().all(|&t| t > 0 && t < 256));
    }

    #[test]
    fn test_token_generator_deterministic() {
        let gen1 = TokenGenerator::new(42, 256);
        let gen2 = TokenGenerator::new(42, 256);

        assert_eq!(gen1.generate(10), gen2.generate(10));
    }

    #[test]
    fn test_token_generator_distribution() {
        let vocab_size = 1000;
        let tg = TokenGenerator::new(42, vocab_size);
        let common = vec![1, 2, 3];
        let tokens = tg.generate_with_distribution(100, &common);
        assert_eq!(tokens.len(), 100);
        assert!(tokens.iter().all(|&t| t < vocab_size as u32));
    }

    #[test]
    fn test_token_generator_distribution_empty_common() {
        let tg = TokenGenerator::new(42, 100);
        let tokens = tg.generate_with_distribution(10, &[]);
        assert_eq!(tokens.len(), 10);
    }

    // ------------------------------------------------------------------
    // Scalar generation helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_f16_generation() {
        let weights = gen().generate_f16(&[100]);

        assert_eq!(weights.len(), 100);
        assert!(weights.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_generate_f32_scaled() {
        let scale = 10.0;
        let weights = gen().generate_f32_scaled(&[100], scale);
        assert_eq!(weights.len(), 100);
        for &w in &weights {
            assert!(w >= -scale && w <= scale);
        }
    }

    #[test]
    fn test_f32_empty_shape() {
        let weights = gen().generate_f32(&[]);
        assert_eq!(weights.len(), 1); // Product of empty is 1
    }
}
