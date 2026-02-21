
    #[test]
    fn test_layer_config_debug_clone_eq() {
        let config = LayerConfig::new(LayerType::Dense).with_units(64);
        let cloned = config.clone();
        assert_eq!(config, cloned);

        let debug = format!("{:?}", config);
        assert!(debug.contains("Dense"));
    }

    #[test]
    fn test_search_space_debug_clone() {
        let space = NasSearchSpace::new();
        let cloned = space.clone();
        assert_eq!(space.max_layers, cloned.max_layers);

        let debug = format!("{:?}", space);
        assert!(debug.contains("max_layers"));
    }

    #[test]
    fn test_nas_genome_debug_clone() {
        let genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);
        let cloned = genome.clone();
        assert_eq!(genome.len(), cloned.len());

        let debug = format!("{:?}", genome);
        assert!(debug.contains("layers"));
    }

    #[test]
    fn test_nas_mutation_debug_clone() {
        let mutation = NasMutation::AddLayer;
        let cloned = mutation;
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("AddLayer"));

        // Test all variants
        let mutations = [
            NasMutation::AddLayer,
            NasMutation::RemoveLayer,
            NasMutation::ChangeType,
            NasMutation::ModifyParams,
            NasMutation::ToggleActive,
        ];
        for m in &mutations {
            let d = format!("{:?}", m);
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_architecture_complexity_all_layer_types() {
        let layers = vec![
            LayerConfig::new(LayerType::Dense).with_units(64),
            LayerConfig::new(LayerType::Conv2d).with_units(64),
            LayerConfig::new(LayerType::MaxPool2d),
            LayerConfig::new(LayerType::AvgPool2d),
            LayerConfig::new(LayerType::BatchNorm),
            LayerConfig::new(LayerType::Dropout),
            LayerConfig::new(LayerType::Skip),
            LayerConfig::new(LayerType::Lstm).with_units(64),
            LayerConfig::new(LayerType::Attention).with_units(64),
        ];
        let genome = NasGenome::from_layers(layers);
        let complexity = architecture_complexity(&genome);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_layer_config_active_default() {
        let config = LayerConfig::new(LayerType::Dense);
        assert!(config.active);
    }

    #[test]
    fn test_encode_layer_type_not_found() {
        // Create space with only one layer type
        let space = NasSearchSpace::new().with_layer_types(vec![LayerType::Dense]);

        // Create genome with a different layer type
        let genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Conv2d)]);

        // Should not panic, returns 0 for unknown type
        let encoded = genome.encode(&space);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_activation_not_found() {
        let space = NasSearchSpace::new().with_activation_choices(&["relu", "tanh"]);

        let genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense).with_activation("unknown_activation")
        ]);

        // Should not panic, returns 0 for unknown activation
        let encoded = genome.encode(&space);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_units_range_with_same_min_max() {
        let space = NasSearchSpace::new().with_units_range(64, 64);
        // min+1 should make max at least 65
        assert!(space.units_range.1 >= space.units_range.0);
    }

    #[test]
    fn test_filters_range_with_same_min_max() {
        let space = NasSearchSpace::new().with_filters_range(64, 64);
        // min+1 should make max at least 65
        assert!(space.filters_range.1 >= space.filters_range.0);
    }
