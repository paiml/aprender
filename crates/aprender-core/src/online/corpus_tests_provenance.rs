
    #[test]
    fn test_corpus_provenance_clone() {
        let mut original = CorpusProvenance::new();
        original.add_source("test", 10, 20);
        original.set_final_size(20);
        let cloned = original.clone();
        assert_eq!(original.final_size, cloned.final_size);
    }

    #[test]
    fn test_corpus_merger_debug() {
        let merger = CorpusMerger::new();
        let debug = format!("{:?}", merger);
        assert!(debug.contains("CorpusMerger"));
    }

    #[test]
    fn test_corpus_merger_default() {
        let merger = CorpusMerger::default();
        assert!(merger.deduplicate);
    }

    #[test]
    fn test_corpus_provenance_default() {
        let prov = CorpusProvenance::default();
        assert_eq!(prov.final_size, 0);
        assert!(prov.sources.is_empty());
    }

    #[test]
    fn test_corpus_buffer_diversity_sampling() {
        let config = CorpusBufferConfig {
            max_size: 3,
            policy: EvictionPolicy::DiversitySampling,
            deduplicate: false,
            seed: Some(42),
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add_raw(vec![0.0], vec![0.0]);
        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);
        // Add a sample that's diverse from existing
        buffer.add_raw(vec![100.0], vec![100.0]);

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_corpus_buffer_is_full() {
        let mut buffer = CorpusBuffer::new(2);
        assert!(!buffer.is_full());

        buffer.add_raw(vec![1.0], vec![1.0]);
        assert!(!buffer.is_full());

        buffer.add_raw(vec![2.0], vec![2.0]);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_sample_with_timestamp() {
        let mut sample = Sample::new(vec![1.0], vec![2.0]);
        sample.timestamp = Some(12345);
        assert_eq!(sample.timestamp, Some(12345));
    }

    #[test]
    fn test_corpus_buffer_fifo_with_dedup() {
        let config = CorpusBufferConfig {
            max_size: 2,
            policy: EvictionPolicy::FIFO,
            deduplicate: true,
            seed: None,
        };
        let mut buffer = CorpusBuffer::with_config(config);

        buffer.add_raw(vec![1.0], vec![1.0]);
        buffer.add_raw(vec![2.0], vec![2.0]);
        buffer.add_raw(vec![3.0], vec![3.0]); // Should evict first

        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_corpus_merger_subsample() {
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample::new(vec![i as f64], vec![i as f64]))
            .collect();

        let mut merger = CorpusMerger::new().deduplicate(false);
        merger.add_source(CorpusSource::new("subsampled", samples).with_weight(0.5));

        let (buffer, _) = merger.merge().unwrap();

        // Weight 0.5 should halve the samples
        assert!(buffer.len() <= 5);
    }

    #[test]
    fn test_sample_sources_all_variants() {
        let sources = vec![
            SampleSource::Synthetic,
            SampleSource::HandCrafted,
            SampleSource::Examples,
            SampleSource::Production,
            SampleSource::External("test".to_string()),
        ];

        for source in sources {
            let debug = format!("{:?}", source);
            assert!(!debug.is_empty());
        }
    }
