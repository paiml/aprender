
    // ==========================================================
    // EXTREME TDD: Tests written first
    // ==========================================================

    #[test]
    fn test_layer_type_all() {
        let types = LayerType::all();
        assert!(types.len() >= 5, "Should have multiple layer types");
        assert!(types.contains(&LayerType::Dense));
        assert!(types.contains(&LayerType::Conv2d));
    }

    #[test]
    fn test_layer_config_builder() {
        let config = LayerConfig::new(LayerType::Dense)
            .with_units(128)
            .with_activation("relu")
            .with_dropout_rate(0.5);

        assert_eq!(config.layer_type, LayerType::Dense);
        assert_eq!(config.units, Some(128));
        assert_eq!(config.activation, Some("relu".to_string()));
        assert_eq!(config.dropout_rate, Some(0.5));
        assert!(config.active);
    }

    #[test]
    fn test_nas_search_space_builder() {
        let space = NasSearchSpace::new()
            .with_max_layers(8)
            .with_min_layers(2)
            .with_units_range(64, 512)
            .with_activation_choices(&["relu", "gelu"]);

        assert_eq!(space.max_layers, 8);
        assert_eq!(space.min_layers, 2);
        assert_eq!(space.units_range, (64, 512));
        assert_eq!(space.activations.len(), 2);
    }

    #[test]
    fn test_nas_genome_random() {
        let space = NasSearchSpace::new().with_max_layers(5).with_min_layers(2);

        let genome = NasGenome::random(&space, 42);

        assert!(genome.len() >= 2);
        assert!(genome.len() <= 5);
        assert!(!genome.is_empty());
    }

    #[test]
    fn test_nas_genome_deterministic() {
        let space = NasSearchSpace::new();

        let genome1 = NasGenome::random(&space, 42);
        let genome2 = NasGenome::random(&space, 42);

        assert_eq!(genome1.len(), genome2.len());
    }

    #[test]
    fn test_nas_genome_encode_decode() {
        let space = NasSearchSpace::new().with_max_layers(3).with_min_layers(3);

        let original = NasGenome::random(&space, 42);
        let encoded = original.encode(&space);
        let decoded = NasGenome::decode(&encoded, &space, original.len());

        assert_eq!(decoded.len(), original.len());
    }

    #[test]
    fn test_mutation_add_layer() {
        let space = NasSearchSpace::new().with_max_layers(10);
        let mut genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);
        let original_len = genome.len();

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::AddLayer, &space, &mut rng);

        assert_eq!(genome.len(), original_len + 1);
    }

    #[test]
    fn test_mutation_remove_layer() {
        let space = NasSearchSpace::new().with_min_layers(1);
        let mut genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense),
            LayerConfig::new(LayerType::Dropout),
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::RemoveLayer, &space, &mut rng);

        assert_eq!(genome.len(), 1);
    }

    #[test]
    fn test_mutation_respects_bounds() {
        let space = NasSearchSpace::new().with_max_layers(2).with_min_layers(2);

        let mut genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense),
            LayerConfig::new(LayerType::Dense),
        ]);

        let mut rng = StdRng::seed_from_u64(42);

        // Try to add - should not exceed max
        mutate_genome(&mut genome, NasMutation::AddLayer, &space, &mut rng);
        assert!(genome.len() <= 2);

        // Try to remove - should not go below min
        mutate_genome(&mut genome, NasMutation::RemoveLayer, &space, &mut rng);
        assert!(genome.len() >= 2);
    }

    #[test]
    fn test_crossover_genomes() {
        let parent1 = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense).with_units(64),
            LayerConfig::new(LayerType::Dense).with_units(128),
        ]);
        let parent2 = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Conv2d).with_units(32),
            LayerConfig::new(LayerType::Dropout),
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        let child = crossover_genomes(&parent1, &parent2, &mut rng);

        assert!(!child.is_empty());
    }

    #[test]
    fn test_architecture_complexity() {
        let simple =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(64)]);

        let complex = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Conv2d).with_units(128),
            LayerConfig::new(LayerType::Lstm).with_units(256),
        ]);

        let simple_complexity = architecture_complexity(&simple);
        let complex_complexity = architecture_complexity(&complex);

        assert!(complex_complexity > simple_complexity);
    }

    #[test]
    fn test_inactive_layers_not_counted() {
        let mut layer = LayerConfig::new(LayerType::Dense).with_units(128);
        layer.active = false;

        let genome = NasGenome::from_layers(vec![layer]);

        assert_eq!(genome.active_layers(), 0);
        assert!(architecture_complexity(&genome) < 0.01);
    }

    #[test]
    fn test_fitness_tracking() {
        let mut genome = NasGenome::new();
        assert!(genome.fitness().is_none());

        genome.set_fitness(0.95);
        assert_eq!(genome.fitness(), Some(0.95));
    }

    #[test]
    fn test_toggle_active_mutation() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);

        assert!(genome.layers()[0].active);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ToggleActive, &space, &mut rng);

        // Should have toggled
        assert!(!genome.layers()[0].active);
    }

    #[test]
    fn test_empty_genome_handling() {
        let genome = NasGenome::new();
        assert!(genome.is_empty());
        assert_eq!(genome.len(), 0);
        assert_eq!(genome.active_layers(), 0);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_layer_config_with_kernel_size() {
        let config = LayerConfig::new(LayerType::Conv2d).with_kernel_size(5);
        assert_eq!(config.kernel_size, Some(5));
    }

    #[test]
    fn test_layer_config_with_dropout_rate_clamping() {
        let config = LayerConfig::new(LayerType::Dropout).with_dropout_rate(1.5);
        assert_eq!(config.dropout_rate, Some(1.0)); // Clamped to max

        let config2 = LayerConfig::new(LayerType::Dropout).with_dropout_rate(-0.5);
        assert_eq!(config2.dropout_rate, Some(0.0)); // Clamped to min
    }

    #[test]
    fn test_search_space_with_filters_range() {
        let space = NasSearchSpace::new().with_filters_range(32, 256);
        assert_eq!(space.filters_range, (32, 256));
    }

    #[test]
    fn test_search_space_with_kernel_sizes() {
        let space = NasSearchSpace::new().with_kernel_sizes(vec![3, 5, 7, 9]);
        assert_eq!(space.kernel_sizes, vec![3, 5, 7, 9]);
    }

    #[test]
    fn test_search_space_with_dropout_range() {
        let space = NasSearchSpace::new().with_dropout_range(0.2, 0.6);
        assert_eq!(space.dropout_range, (0.2, 0.6));
    }

    #[test]
    fn test_search_space_dropout_range_clamping() {
        let space = NasSearchSpace::new().with_dropout_range(-0.5, 1.5);
        assert_eq!(space.dropout_range.0, 0.0);
        assert_eq!(space.dropout_range.1, 1.0);
    }

    #[test]
    fn test_search_space_with_empty_layer_types() {
        let space = NasSearchSpace::new().with_layer_types(vec![]);
        // Should not change when passed empty
        assert!(!space.layer_types.is_empty());
    }

    #[test]
    fn test_search_space_with_empty_kernel_sizes() {
        let space = NasSearchSpace::new().with_kernel_sizes(vec![]);
        // Should not change when passed empty
        assert!(!space.kernel_sizes.is_empty());
    }

    #[test]
    fn test_search_space_with_empty_activations() {
        let space = NasSearchSpace::new().with_activation_choices(&[]);
        // Should not change when passed empty
        assert!(!space.activations.is_empty());
    }

    #[test]
    fn test_nas_genome_random_with_conv2d() {
        let space = NasSearchSpace::new()
            .with_layer_types(vec![LayerType::Conv2d])
            .with_filters_range(16, 64)
            .with_kernel_sizes(vec![3, 5])
            .with_min_layers(3)
            .with_max_layers(3);

        let genome = NasGenome::random(&space, 123);
        assert_eq!(genome.len(), 3);

        for layer in genome.layers() {
            assert_eq!(layer.layer_type, LayerType::Conv2d);
            assert!(layer.units.is_some());
            assert!(layer.kernel_size.is_some());
        }
    }

    #[test]
    fn test_nas_genome_random_with_lstm() {
        let space = NasSearchSpace::new()
            .with_layer_types(vec![LayerType::Lstm])
            .with_min_layers(2)
            .with_max_layers(2);

        let genome = NasGenome::random(&space, 456);

        for layer in genome.layers() {
            assert_eq!(layer.layer_type, LayerType::Lstm);
            assert!(layer.units.is_some());
        }
    }

    #[test]
    fn test_nas_genome_random_with_batchnorm() {
        let space = NasSearchSpace::new()
            .with_layer_types(vec![LayerType::BatchNorm])
            .with_min_layers(1)
            .with_max_layers(1);

        let genome = NasGenome::random(&space, 789);
        assert_eq!(genome.layers()[0].layer_type, LayerType::BatchNorm);
    }

    #[test]
    fn test_nas_genome_encode_without_activation() {
        let space = NasSearchSpace::new();
        let genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dropout), // No activation
        ]);

        let encoded = genome.encode(&space);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_nas_genome_decode_short_encoding() {
        let space = NasSearchSpace::new();
        // Very short encoding - should handle gracefully
        let encoded = vec![0.5, 0.5, 0.5, 0.5, 1.0];

        let decoded = NasGenome::decode(&encoded, &space, 2);
        // Only one layer can be decoded from 5 values (stride=5)
        assert_eq!(decoded.len(), 1);
    }

    #[test]
    fn test_nas_genome_default() {
        let genome = NasGenome::default();
        assert!(genome.is_empty());
    }

    #[test]
    fn test_mutation_change_type() {
        let space = NasSearchSpace::new().with_layer_types(vec![
            LayerType::Dense,
            LayerType::Conv2d,
            LayerType::Dropout,
        ]);
        let mut genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ChangeType, &space, &mut rng);

        // Type might have changed
        assert!(!genome.is_empty());
    }

    #[test]
    fn test_mutation_modify_params() {
        let space = NasSearchSpace::new().with_units_range(32, 256);
        let mut genome =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(128)]);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ModifyParams, &space, &mut rng);

        // Units might have changed
        let units = genome.layers()[0].units.unwrap();
        assert!(units >= 32 && units <= 256);
    }

    #[test]
    fn test_mutation_modify_params_without_units() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dropout), // No units
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        // Should not panic
        mutate_genome(&mut genome, NasMutation::ModifyParams, &space, &mut rng);
    }

    #[test]
    fn test_mutation_change_type_empty_genome() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        // Should not panic on empty genome
        mutate_genome(&mut genome, NasMutation::ChangeType, &space, &mut rng);
        assert!(genome.is_empty());
    }

    #[test]
    fn test_mutation_toggle_empty_genome() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ToggleActive, &space, &mut rng);
        assert!(genome.is_empty());
    }

    #[test]
    fn test_mutation_modify_params_empty_genome() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ModifyParams, &space, &mut rng);
        assert!(genome.is_empty());
    }

    #[test]
    fn test_crossover_with_empty_parent1() {
        let parent1 = NasGenome::new();
        let parent2 =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(64)]);

        let mut rng = StdRng::seed_from_u64(42);
        let child = crossover_genomes(&parent1, &parent2, &mut rng);

        assert!(!child.is_empty());
    }

    #[test]
    fn test_crossover_with_empty_parent2() {
        let parent1 =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(64)]);
        let parent2 = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        let child = crossover_genomes(&parent1, &parent2, &mut rng);

        assert!(!child.is_empty());
    }

    #[test]
    fn test_layer_type_debug_clone_hash() {
        use std::collections::HashSet;

        let lt = LayerType::Dense;
        let cloned = lt;
        assert_eq!(lt, cloned);

        let debug = format!("{:?}", lt);
        assert!(debug.contains("Dense"));

        let mut set = HashSet::new();
        set.insert(LayerType::Dense);
        set.insert(LayerType::Conv2d);
        assert_eq!(set.len(), 2);
    }
