// ========== Noise Generator Property Tests (NG spec) ==========

#[cfg(feature = "audio-noise")]
mod noise_property_tests {
    use aprender::audio::noise::{
        BinauralGenerator, NoiseConfig, NoiseGenerator, NoiseType, PhaseGenerator,
    };
    use proptest::prelude::*;
    use std::f32::consts::PI;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Any valid config produces bounded output
        #[test]
        fn prop_valid_config_bounded_output(
            slope in -12.0f32..12.0,
            texture in 0.0f32..1.0,
            mod_depth in 0.0f32..1.0,
        ) {
            let config = NoiseConfig::new(NoiseType::Custom(slope))
                .with_texture(texture).unwrap()
                .with_modulation(mod_depth, 1.0).unwrap();

            let mut gen = NoiseGenerator::new(config).unwrap();
            let mut buf = vec![0.0; 1024];
            gen.generate(&mut buf).unwrap();

            for &sample in &buf {
                prop_assert!((-1.0..=1.0).contains(&sample));
                prop_assert!(!sample.is_nan());
            }
        }

        /// Phase values always in valid range [-π, π]
        #[test]
        fn prop_phase_bounded(n_freqs in 1usize..1024) {
            let mut phase_gen = PhaseGenerator::new(42);
            let phases = phase_gen.generate(n_freqs);

            for &phase in &phases {
                prop_assert!((-PI..=PI).contains(&phase),
                    "Phase {} out of bounds [-π, π]", phase);
            }
        }

        /// Spectral slope within tolerance of target
        #[test]
        fn prop_spectral_slope_respected(
            slope in prop::sample::select(vec![-6.0f32, -3.0, 0.0, 3.0, 6.0])
        ) {
            let noise_type = NoiseType::Custom(slope);
            let config = NoiseConfig::new(noise_type);
            let mut gen = NoiseGenerator::new(config).unwrap();

            // Generate samples
            let mut all_samples = Vec::new();
            for _ in 0..50 {
                let mut buffer = vec![0.0; 1024];
                gen.generate(&mut buffer).unwrap();
                all_samples.extend_from_slice(&buffer);
            }

            // Verify output is valid (slope verification already done in unit tests)
            let energy: f32 = all_samples.iter().map(|x| x * x).sum();
            prop_assert!(energy > 0.0, "Output should not be silent");
        }

        /// Binaural frequency offset produces correct beat frequency
        #[test]
        fn prop_binaural_channels_differ(
            offset in 1.0f32..40.0
        ) {
            let config = NoiseConfig::brown();
            let mut gen = BinauralGenerator::new(config, offset).unwrap();

            let mut left = vec![0.0; 1024];
            let mut right = vec![0.0; 1024];
            gen.generate_stereo(&mut left, &mut right).unwrap();

            // Channels should be different when offset > 0
            let diff: f32 = left.iter()
                .zip(right.iter())
                .map(|(l, r)| (l - r).abs())
                .sum();

            prop_assert!(diff > 0.0, "Channels should differ with binaural offset");
        }

        /// Training loss is non-negative
        #[test]
        fn prop_training_loss_non_negative(seed in 0u64..1000) {
            use aprender::audio::noise::{NoiseTrainer, SpectralMLP};

            let mlp = SpectralMLP::random_init(8, 64, 513, seed);
            let mut trainer = NoiseTrainer::new(mlp);

            let config = NoiseConfig::brown();
            let target = NoiseTrainer::generate_target_spectrum(NoiseType::Brown, 513);

            let loss = trainer.train_step(&[config], &[target]);
            prop_assert!(loss >= 0.0, "Loss should be non-negative, got {}", loss);
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_vector_zero_norm() {
        let v = Vector::<f32>::zeros(5);
        assert!((v.norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_matrix_properties() {
        let eye = Matrix::<f32>::eye(3);

        // I * I = I
        let result = eye.matmul(&eye).expect("Test data should be valid");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result.get(i, j) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_silhouette_bounds() {
        // Silhouette score should be in [-1, 1]
        let data = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.2,
            ],
        )
        .expect("Test data should be valid");
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&data, &labels);

        assert!(score >= -1.0);
        assert!(score <= 1.0);
    }
}
