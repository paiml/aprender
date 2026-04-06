#[cfg(test)]
mod tests {
    use crate::inference_trace::*;

    #[test]
    fn test_trace_config_enabled_steps() {
        let config = TraceConfig::enabled();
        // All steps should be traced when enabled
        assert!(config.should_trace(TraceStep::Tokenize));
        assert!(config.should_trace(TraceStep::Embed));
        assert!(config.should_trace(TraceStep::TransformerBlock));
        assert!(config.should_trace(TraceStep::LmHead));
        assert!(config.should_trace(TraceStep::Sample));
        assert!(config.should_trace(TraceStep::Decode));
    }

    #[test]
    #[allow(deprecated)]
    fn test_trace_step_legacy_names() {
        // legacy_name returns uppercase
        assert_eq!(TraceStep::Tokenize.legacy_name(), "ENCODE");
        assert_eq!(TraceStep::Embed.legacy_name(), "EMBED");
        assert_eq!(TraceStep::TransformerBlock.legacy_name(), "TRANSFORMER");
        assert_eq!(TraceStep::LmHead.legacy_name(), "LM_HEAD");
        assert_eq!(TraceStep::Sample.legacy_name(), "SAMPLE");
        assert_eq!(TraceStep::Decode.legacy_name(), "DECODE");
    }

    #[test]
    fn test_record_execution_failed() {
        let config = TraceConfig::enabled();
        let mut tracer = InferenceTracer::new(config);

        tracer.record_execution_failed("Token overflow", "ID 50001 > vocab 50000");

        assert_eq!(tracer.error_count(), 1);
        let events = tracer.events();
        assert!(!events.is_empty());

        // Should have ExecutionFailed event type
        let has_failed = events
            .iter()
            .any(|e| e.event_type == AwsEventType::ExecutionFailed);
        assert!(has_failed);
    }
include!("tests_trace_step_tensor.rs");
include!("tests_cov_format.rs");
}
