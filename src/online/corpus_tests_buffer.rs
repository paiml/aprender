
    #[test]
    fn test_corpus_buffer_basic() {
        let mut buffer = CorpusBuffer::new(100);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_corpus_buffer_deduplication() {
        let mut buffer = CorpusBuffer::new(100);

        // Add same sample twice
        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        let added = buffer.add_raw(vec![1.0, 2.0], vec![3.0]);

        assert!(!added, "Duplicate should not be added");
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_corpus_buffer_no_deduplication() {
        let config = CorpusBufferConfig {
            max_size: 100,
            deduplicate: false,
            ..Default::default()
        };
        let mut buffer = CorpusBuffer::with_config(config);

        // Add same sample twice
        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        let added = buffer.add_raw(vec![1.0, 2.0], vec![3.0]);

        assert!(added, "Duplicate should be added when dedup disabled");
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_corpus_buffer_fifo_eviction() {
        let config = CorpusBufferConfig {
            max_size: 3,
            policy: EvictionPolicy::FIFO,
            deduplicate: false,
            ..Default::default()
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);
        buffer.add_raw(vec![3.0], vec![3.0]);
        buffer.add_raw(vec![4.0], vec![4.0]);

        assert_eq!(buffer.len(), 3);
        // First sample should be evicted
        assert_eq!(buffer.samples()[0].features[0], 2.0);
    }

    #[test]
    fn test_corpus_buffer_importance_weighted_eviction() {
        let config = CorpusBufferConfig {
            max_size: 3,
            policy: EvictionPolicy::ImportanceWeighted,
            deduplicate: false,
            ..Default::default()
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add(Sample::with_weight(vec![1.0], vec![1.0], 0.5));
        buffer.add(Sample::with_weight(vec![2.0], vec![2.0], 0.8));
        buffer.add(Sample::with_weight(vec![3.0], vec![3.0], 0.3));
        buffer.add(Sample::with_weight(vec![4.0], vec![4.0], 1.0));

        assert_eq!(buffer.len(), 3);

        // Check that lowest weight (0.3) was evicted
        let weights: Vec<f64> = buffer.samples().iter().map(|s| s.weight).collect();
        assert!(!weights.contains(&0.3));
    }

    #[test]
    fn test_corpus_buffer_reservoir_sampling() {
        let config = CorpusBufferConfig {
            max_size: 10,
            policy: EvictionPolicy::Reservoir,
            deduplicate: false,
            seed: Some(42),
        };
        let mut buffer = CorpusBuffer::with_config(config);

        // Add many samples
        for i in 0..100 {
            buffer.add_raw(vec![i as f64], vec![(i * 2) as f64]);
        }

        assert_eq!(buffer.len(), 10);
    }

    #[test]
    fn test_corpus_buffer_to_dataset() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add_raw(vec![1.0, 2.0], vec![3.0]);
        buffer.add_raw(vec![4.0, 5.0], vec![6.0]);

        let (features, targets, n_samples, n_features) = buffer.to_dataset();

        assert_eq!(n_samples, 2);
        assert_eq!(n_features, 2);
        assert_eq!(features.len(), 4);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_corpus_buffer_empty_dataset() {
        let buffer = CorpusBuffer::new(100);
        let (features, targets, n_samples, n_features) = buffer.to_dataset();

        assert!(features.is_empty());
        assert!(targets.is_empty());
        assert_eq!(n_samples, 0);
        assert_eq!(n_features, 0);
    }

    #[test]
    fn test_corpus_buffer_clear() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_corpus_buffer_weights() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add(Sample::with_weight(vec![1.0], vec![1.0], 0.5));
        buffer.add(Sample::with_weight(vec![2.0], vec![2.0], 1.5));

        let weights = buffer.weights();
        assert_eq!(weights, vec![0.5, 1.5]);
    }

    #[test]
    fn test_corpus_buffer_update_weight() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add_raw(vec![1.0], vec![1.0]);

        buffer.update_weight(0, 2.0).unwrap();
        assert_eq!(buffer.samples()[0].weight, 2.0);

        // Invalid index
        assert!(buffer.update_weight(10, 1.0).is_err());
    }

    #[test]
    fn test_corpus_buffer_samples_by_source() {
        let mut buffer = CorpusBuffer::new(100);

        buffer.add(Sample::with_source(
            vec![1.0],
            vec![1.0],
            SampleSource::Synthetic,
        ));
        buffer.add(Sample::with_source(
            vec![2.0],
            vec![2.0],
            SampleSource::Production,
        ));
        buffer.add(Sample::with_source(
            vec![3.0],
            vec![3.0],
            SampleSource::Synthetic,
        ));

        let synthetic = buffer.samples_by_source(&SampleSource::Synthetic);
        assert_eq!(synthetic.len(), 2);

        let production = buffer.samples_by_source(&SampleSource::Production);
        assert_eq!(production.len(), 1);
    }

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(vec![1.0, 2.0], vec![3.0]);
        assert_eq!(sample.features, vec![1.0, 2.0]);
        assert_eq!(sample.target, vec![3.0]);
        assert_eq!(sample.weight, 1.0);
        assert!(sample.timestamp.is_none());
    }

    #[test]
    fn test_corpus_source() {
        let samples = vec![Sample::new(vec![1.0], vec![1.0])];
        let source = CorpusSource::new("test", samples)
            .with_weight(2.0)
            .with_priority(5);

        assert_eq!(source.name, "test");
        assert_eq!(source.weight, 2.0);
        assert_eq!(source.priority, 5);
    }

    #[test]
    fn test_corpus_merger_basic() {
        let samples1 = vec![
            Sample::new(vec![1.0], vec![1.0]),
            Sample::new(vec![2.0], vec![2.0]),
        ];
        let samples2 = vec![
            Sample::new(vec![3.0], vec![3.0]),
            Sample::new(vec![4.0], vec![4.0]),
        ];

        let mut merger = CorpusMerger::new();
        merger.add_source(CorpusSource::new("source1", samples1));
        merger.add_source(CorpusSource::new("source2", samples2));

        let (buffer, provenance) = merger.merge().unwrap();

        assert_eq!(buffer.len(), 4);
        assert_eq!(provenance.sources.len(), 2);
    }

    #[test]
    fn test_corpus_merger_with_weights() {
        let samples = vec![
            Sample::new(vec![1.0], vec![1.0]),
            Sample::new(vec![2.0], vec![2.0]),
        ];

        let mut merger = CorpusMerger::new().deduplicate(false); // Disable dedup for this test
        merger.add_source(CorpusSource::new("weighted", samples).with_weight(2.0));

        let (buffer, _) = merger.merge().unwrap();

        // Weight 2.0 should double the samples (4 with repeats)
        assert!(
            buffer.len() >= 4,
            "Expected at least 4 samples, got {}",
            buffer.len()
        );
    }

    #[test]
    fn test_corpus_merger_deduplication() {
        let samples1 = vec![Sample::new(vec![1.0], vec![1.0])];
        let samples2 = vec![Sample::new(vec![1.0], vec![1.0])]; // Duplicate

        let mut merger = CorpusMerger::new();
        merger.add_source(CorpusSource::new("source1", samples1).with_priority(1));
        merger.add_source(CorpusSource::new("source2", samples2).with_priority(0));

        let (buffer, provenance) = merger.merge().unwrap();

        assert_eq!(buffer.len(), 1);
        assert_eq!(provenance.duplicates_removed, 1);
    }

    #[test]
    fn test_corpus_merger_no_deduplication() {
        let samples1 = vec![Sample::new(vec![1.0], vec![1.0])];
        let samples2 = vec![Sample::new(vec![1.0], vec![1.0])];

        let mut merger = CorpusMerger::new().deduplicate(false);
        merger.add_source(CorpusSource::new("source1", samples1));
        merger.add_source(CorpusSource::new("source2", samples2));

        let (buffer, provenance) = merger.merge().unwrap();

        assert_eq!(buffer.len(), 2);
        assert_eq!(provenance.duplicates_removed, 0);
    }

    #[test]
    fn test_corpus_merger_shuffle() {
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample::new(vec![i as f64], vec![i as f64]))
            .collect();

        let mut merger = CorpusMerger::new().shuffle_seed(42);
        merger.add_source(CorpusSource::new("ordered", samples));

        let (buffer, _) = merger.merge().unwrap();

        // Check that order is different (with high probability)
        let features: Vec<f64> = buffer.samples().iter().map(|s| s.features[0]).collect();
        let ordered: Vec<f64> = (0..10).map(|i| i as f64).collect();

        assert_ne!(features, ordered);
    }

    #[test]
    fn test_corpus_provenance() {
        let mut provenance = CorpusProvenance::new();

        provenance.add_source("test1", 100, 200);
        provenance.add_source("test2", 50, 50);
        provenance.set_final_size(250);

        assert_eq!(provenance.sources.len(), 2);
        assert_eq!(provenance.final_size, 250);
    }

    #[test]
    fn test_eviction_policy_default() {
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::Reservoir);
    }

    #[test]
    fn test_sample_source_default() {
        assert_eq!(SampleSource::default(), SampleSource::Production);
    }

    #[test]
    fn test_corpus_buffer_config_default() {
        let config = CorpusBufferConfig::default();
        assert_eq!(config.max_size, 10_000);
        assert!(config.deduplicate);
        assert_eq!(config.policy, EvictionPolicy::Reservoir);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_eviction_policy_eq() {
        assert_eq!(EvictionPolicy::FIFO, EvictionPolicy::FIFO);
        assert_ne!(EvictionPolicy::FIFO, EvictionPolicy::Reservoir);
    }

    #[test]
    fn test_eviction_policy_debug() {
        let policy = EvictionPolicy::DiversitySampling;
        let debug = format!("{:?}", policy);
        assert!(debug.contains("DiversitySampling"));
    }

    #[test]
    fn test_eviction_policy_clone() {
        let policy = EvictionPolicy::ImportanceWeighted;
        let cloned = policy;
        assert_eq!(policy, cloned);
    }

    #[test]
    fn test_sample_source_external() {
        let source = SampleSource::External("dataset.csv".to_string());
        let debug = format!("{:?}", source);
        assert!(debug.contains("External"));
        assert!(debug.contains("dataset.csv"));
    }

    #[test]
    fn test_sample_source_eq() {
        assert_eq!(SampleSource::Synthetic, SampleSource::Synthetic);
        assert_ne!(SampleSource::Synthetic, SampleSource::HandCrafted);
        assert_eq!(
            SampleSource::External("a".to_string()),
            SampleSource::External("a".to_string())
        );
        assert_ne!(
            SampleSource::External("a".to_string()),
            SampleSource::External("b".to_string())
        );
    }

    #[test]
    fn test_sample_source_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SampleSource::Synthetic);
        set.insert(SampleSource::Production);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_sample_debug() {
        let sample = Sample::new(vec![1.0], vec![2.0]);
        let debug = format!("{:?}", sample);
        assert!(debug.contains("Sample"));
    }

    #[test]
    fn test_sample_clone() {
        let original = Sample::with_weight(vec![1.0, 2.0], vec![3.0], 0.5);
        let cloned = original.clone();
        assert_eq!(original.features, cloned.features);
        assert_eq!(original.target, cloned.target);
        assert_eq!(original.weight, cloned.weight);
    }

    #[test]
    fn test_corpus_buffer_config_debug() {
        let config = CorpusBufferConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("CorpusBufferConfig"));
    }

    #[test]
    fn test_corpus_buffer_config_clone() {
        let original = CorpusBufferConfig::default();
        let cloned = original.clone();
        assert_eq!(original.max_size, cloned.max_size);
    }

    #[test]
    fn test_corpus_buffer_debug() {
        let buffer = CorpusBuffer::new(10);
        let debug = format!("{:?}", buffer);
        assert!(debug.contains("CorpusBuffer"));
    }

    #[test]
    fn test_corpus_source_debug() {
        let source = CorpusSource::new("test", vec![]);
        let debug = format!("{:?}", source);
        assert!(debug.contains("CorpusSource"));
    }

    #[test]
    fn test_corpus_source_clone() {
        let original = CorpusSource::new("test", vec![])
            .with_weight(2.0)
            .with_priority(3);
        let cloned = original.clone();
        assert_eq!(original.name, cloned.name);
        assert_eq!(original.weight, cloned.weight);
        assert_eq!(original.priority, cloned.priority);
    }

    #[test]
    fn test_corpus_provenance_debug() {
        let prov = CorpusProvenance::new();
        let debug = format!("{:?}", prov);
        assert!(debug.contains("CorpusProvenance"));
    }
