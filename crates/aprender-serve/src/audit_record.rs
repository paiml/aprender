
// ============================================================================
// EXTREME TDD TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // -------------------------------------------------------------------------
    // AuditRecord Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_record_new() {
        let request_id = Uuid::new_v4();
        let record = AuditRecord::new(request_id, "abc123", "LogisticRegression");

        assert_eq!(record.request_id, request_id.to_string());
        assert_eq!(record.model_hash, "abc123");
        assert_eq!(record.model_type, "LogisticRegression");
        assert!(record.quality_nan_check);
        assert!(record.quality_confidence_check);
    }

    #[test]
    fn test_audit_record_builder_pattern() {
        let request_id = Uuid::new_v4();
        let record = AuditRecord::new(request_id, "hash123", "RandomForest")
            .with_client_hash("client_hash_456")
            .with_model_version("1.0.0")
            .with_input_dims(vec![784])
            .with_input_hash("input_hash_789")
            .with_prediction(serde_json::json!({"class": 1}))
            .with_confidence(0.95)
            .with_latency(Duration::from_millis(50))
            .with_warning("Low confidence")
            .with_quality_checks(true, false);

        assert_eq!(record.client_id_hash, Some("client_hash_456".to_string()));
        assert_eq!(record.model_version, "1.0.0");
        assert_eq!(record.input_dims, vec![784]);
        assert_eq!(record.input_hash, "input_hash_789");
        assert_eq!(record.prediction, serde_json::json!({"class": 1}));
        assert_eq!(record.confidence, Some(0.95));
        assert!((record.latency_ms - 50.0).abs() < 0.1);
        assert_eq!(record.warnings, vec!["Low confidence"]);
        assert!(record.quality_nan_check);
        assert!(!record.quality_confidence_check);
    }

    #[test]
    fn test_audit_record_serialization() {
        let request_id = Uuid::new_v4();
        let record =
            AuditRecord::new(request_id, "hash", "KNN").with_prediction(serde_json::json!(5));

        let json = serde_json::to_string(&record).expect("test");
        let deserialized: AuditRecord = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.model_type, "KNN");
        assert_eq!(deserialized.prediction, serde_json::json!(5));
    }

    #[test]
    fn test_audit_record_with_distillation_teacher() {
        let record = AuditRecord::new(Uuid::new_v4(), "student", "MLP")
            .with_distillation_teacher("teacher_hash_abc");

        assert_eq!(
            record.distillation_teacher_hash,
            Some("teacher_hash_abc".to_string())
        );
    }

    // -------------------------------------------------------------------------
    // AuditOptions Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_options_apr() {
        let options = AuditOptions::apr(true, Some(0.8));

        assert!(options.explain);
        assert_eq!(options.confidence_threshold, Some(0.8));
        assert!(options.max_tokens.is_none());
        assert!(options.temperature.is_none());
    }

    #[test]
    fn test_audit_options_llm() {
        let options = AuditOptions::llm(256, 0.7);

        assert!(!options.explain);
        assert!(options.confidence_threshold.is_none());
        assert_eq!(options.max_tokens, Some(256));
        assert_eq!(options.temperature, Some(0.7));
    }

    // -------------------------------------------------------------------------
    // LatencyBreakdown Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_latency_breakdown_new() {
        let breakdown = LatencyBreakdown::new(1.5, 10.0, 0.5);

        assert!((breakdown.preprocessing_ms - 1.5).abs() < 0.001);
        assert!((breakdown.inference_ms - 10.0).abs() < 0.001);
        assert!((breakdown.postprocessing_ms - 0.5).abs() < 0.001);
        assert!(breakdown.explanation_ms.is_none());
    }

    #[test]
    fn test_latency_breakdown_with_explanation() {
        let breakdown = LatencyBreakdown::new(1.0, 5.0, 1.0).with_explanation(3.0);

        assert_eq!(breakdown.explanation_ms, Some(3.0));
    }

    #[test]
    fn test_latency_breakdown_total() {
        let breakdown = LatencyBreakdown::new(1.0, 5.0, 2.0);
        assert!((breakdown.total_ms() - 8.0).abs() < 0.001);

        let with_explain = breakdown.with_explanation(2.0);
        assert!((with_explain.total_ms() - 10.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // ProvenanceChain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_provenance_chain_new() {
        let chain = ProvenanceChain::new();

        assert!(chain.training_data_hash.is_none());
        assert!(chain.training_code_sha.is_none());
        assert!(chain.distillation_chain.is_empty());
        assert!(chain.signatures.is_empty());
    }

    #[test]
    fn test_provenance_chain_builder() {
        let chain = ProvenanceChain::new()
            .with_training_data("data_hash_123")
            .with_code_sha("abc123def");

        assert_eq!(chain.training_data_hash, Some("data_hash_123".to_string()));
        assert_eq!(chain.training_code_sha, Some("abc123def".to_string()));
    }

    #[test]
    fn test_provenance_chain_add_distillation() {
        let mut chain = ProvenanceChain::new();
        chain.add_distillation(DistillationStep {
            teacher_hash: "teacher123".to_string(),
            method: "Standard".to_string(),
            temperature: 4.0,
            alpha: 0.5,
            final_loss: 0.01,
            timestamp: Utc::now(),
        });

        assert_eq!(chain.distillation_chain.len(), 1);
        assert_eq!(chain.distillation_chain[0].teacher_hash, "teacher123");
    }

    #[test]
    fn test_provenance_chain_add_signature() {
        let mut chain = ProvenanceChain::new();
        chain.add_signature(SignatureRecord {
            signer: "paiml-release".to_string(),
            algorithm: "Ed25519".to_string(),
            timestamp: Utc::now(),
            signature: "base64sig==".to_string(),
        });

        assert_eq!(chain.signatures.len(), 1);
        assert_eq!(chain.signatures[0].signer, "paiml-release");
    }

    // -------------------------------------------------------------------------
    // InMemoryAuditSink Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_in_memory_sink_write_batch() {
        let sink = InMemoryAuditSink::new();
        let records = vec![
            AuditRecord::new(Uuid::new_v4(), "h1", "LR"),
            AuditRecord::new(Uuid::new_v4(), "h2", "RF"),
        ];

        sink.write_batch(&records).expect("test");

        assert_eq!(sink.count(), 2);
        let stored = sink.records();
        assert_eq!(stored[0].model_type, "LR");
        assert_eq!(stored[1].model_type, "RF");
    }

    #[test]
    fn test_in_memory_sink_flush() {
        let sink = InMemoryAuditSink::new();
        assert!(sink.flush().is_ok());
    }

    // -------------------------------------------------------------------------
    // AuditLogger Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_logger_in_memory() {
        let (logger, sink) = AuditLogger::in_memory();

        let request_id = logger.log_request("LogisticRegression", &[784]);
        logger.log_response(
            request_id,
            serde_json::json!({"class": 1}),
            Duration::from_millis(5),
            Some(0.95),
        );

        assert_eq!(logger.buffer_size(), 1);
        logger.flush().expect("test");
        assert_eq!(logger.buffer_size(), 0);
        assert_eq!(sink.count(), 1);
    }

    #[test]
    fn test_audit_logger_with_model_hash() {
        let (logger, _sink) = AuditLogger::in_memory();
        let logger = logger.with_model_hash("sha256_abc123");

        assert_eq!(logger.model_hash(), "sha256_abc123");
    }

    #[test]
    fn test_audit_logger_total_logged() {
        let (logger, _sink) = AuditLogger::in_memory();

        for _ in 0..5 {
            let id = logger.log_request("Test", &[10]);
            logger.log_response(id, serde_json::json!(0), Duration::from_millis(1), None);
        }

        logger.flush().expect("test");
        assert_eq!(logger.total_logged(), 5);
    }

    #[test]
    fn test_audit_logger_auto_flush() {
        let sink = std::sync::Arc::new(InMemoryAuditSink::new());
        let logger =
            AuditLogger::new(Box::new(InMemorySinkWrapper(sink.clone()))).with_buffer_threshold(3);

        // Log 3 requests - should auto-flush
        for _ in 0..3 {
            let id = logger.log_request("Test", &[5]);
            logger.log_response(id, serde_json::json!(1), Duration::from_millis(1), None);
        }

        // Buffer should be empty after auto-flush
        assert_eq!(logger.buffer_size(), 0);
        assert_eq!(sink.count(), 3);
    }

    #[test]
    fn test_audit_logger_load_timestamp() {
        let timestamp = Utc::now();
        let (logger, _) = AuditLogger::in_memory();
        let logger = logger.with_load_timestamp(timestamp);

        assert_eq!(logger.load_timestamp(), timestamp);
    }

    // -------------------------------------------------------------------------
    // AuditError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_error_display() {
        let io_err = AuditError::IoError("disk full".to_string());
        assert!(io_err.to_string().contains("disk full"));

        let ser_err = AuditError::SerializationError("invalid json".to_string());
        assert!(ser_err.to_string().contains("invalid json"));

        let not_found = AuditError::NotFound("abc-123".to_string());
        assert!(not_found.to_string().contains("abc-123"));
    }

    // -------------------------------------------------------------------------
    // Serialization Round-trip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_latency_breakdown_serialization() {
        let breakdown = LatencyBreakdown::new(1.0, 5.0, 2.0).with_explanation(1.5);
        let json = serde_json::to_string(&breakdown).expect("test");
        let restored: LatencyBreakdown = serde_json::from_str(&json).expect("test");

        assert!((restored.inference_ms - 5.0).abs() < 0.001);
        assert_eq!(restored.explanation_ms, Some(1.5));
    }

    #[test]
    fn test_provenance_chain_serialization() {
        let chain = ProvenanceChain::new()
            .with_training_data("data123")
            .with_code_sha("commit456");

        let json = serde_json::to_string(&chain).expect("test");
        let restored: ProvenanceChain = serde_json::from_str(&json).expect("test");

        assert_eq!(restored.training_data_hash, Some("data123".to_string()));
        assert_eq!(restored.training_code_sha, Some("commit456".to_string()));
    }

    #[test]
    fn test_quantization_provenance_serialization() {
        let quant = QuantizationProvenance {
            original_hash: "original_model_hash".to_string(),
            method: "Q4_K_M".to_string(),
            bits: 4,
            calibration_hash: Some("calibration_data_hash".to_string()),
        };

        let json = serde_json::to_string(&quant).expect("test");
        let restored: QuantizationProvenance = serde_json::from_str(&json).expect("test");

        assert_eq!(restored.method, "Q4_K_M");
        assert_eq!(restored.bits, 4);
    }

    // -------------------------------------------------------------------------
    // JsonFileAuditSink Tests (95% coverage push)
    // -------------------------------------------------------------------------

    #[test]
    fn test_json_file_audit_sink_new() {
        // Just verify sink can be created without panic
        let _sink = JsonFileAuditSink::new("/tmp/test_audit.jsonl");
    }

    #[test]
    fn test_json_file_audit_sink_write_batch() {
        use std::fs;
        let path = std::env::temp_dir().join("audit_test_write_batch.jsonl");
        let sink = JsonFileAuditSink::new(&path);

        let records = vec![
            AuditRecord::new(Uuid::new_v4(), "hash1", "LR")
                .with_prediction(serde_json::json!({"class": 0})),
            AuditRecord::new(Uuid::new_v4(), "hash2", "RF")
                .with_prediction(serde_json::json!({"class": 1})),
        ];

        sink.write_batch(&records).expect("write_batch");
        sink.flush().expect("flush");

        // Verify file was written
        let content = fs::read_to_string(&path).expect("read file");
        assert!(content.contains("hash1"));
        assert!(content.contains("hash2"));
        assert!(content.lines().count() == 2);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_json_file_audit_sink_append() {
        use std::fs;
        let path = std::env::temp_dir().join("audit_test_append.jsonl");
        // Clean up stale data from previous runs
        let _ = fs::remove_file(&path);
        let sink = JsonFileAuditSink::new(&path);

        // Write first batch
        let records1 = vec![AuditRecord::new(Uuid::new_v4(), "h1", "T1")];
        sink.write_batch(&records1).expect("write 1");

        // Write second batch (should append)
        let records2 = vec![AuditRecord::new(Uuid::new_v4(), "h2", "T2")];
        sink.write_batch(&records2).expect("write 2");

        let content = fs::read_to_string(&path).expect("read");
        assert_eq!(content.lines().count(), 2);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_json_file_audit_sink_flush_noop() {
        let sink = JsonFileAuditSink::new("/tmp/nonexistent_audit.jsonl");
        // Flush does nothing but returns Ok
        assert!(sink.flush().is_ok());
    }

    #[test]
    fn test_audit_logger_with_file_sink() {
        use std::fs;
        let path = std::env::temp_dir().join("audit_test_logger_file.jsonl");
        let sink = JsonFileAuditSink::new(&path);
        let logger = AuditLogger::new(Box::new(sink));

        let id = logger.log_request("FileTest", &[100, 200]);
        logger.log_response(
            id,
            serde_json::json!(42),
            Duration::from_millis(10),
            Some(0.99),
        );
        logger.flush().expect("flush");

        let content = fs::read_to_string(&path).expect("read");
        assert!(content.contains("FileTest"));
        assert!(content.contains("100"));
        assert!(content.contains("200"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_audit_record_with_explanation_summary() {
        let record = AuditRecord::new(Uuid::new_v4(), "h", "T");
        // Verify explanation_summary is None by default
        assert!(record.explanation_summary.is_none());
    }

    #[test]
    fn test_audit_options_default_values() {
        let apr_opts = AuditOptions::apr(false, None);
        assert!(!apr_opts.explain);
        assert!(apr_opts.confidence_threshold.is_none());

        let llm_opts = AuditOptions::llm(100, 1.0);
        assert_eq!(llm_opts.max_tokens, Some(100));
        assert_eq!(llm_opts.temperature, Some(1.0));
    }
}
