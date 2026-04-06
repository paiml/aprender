
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_capture_from_json_invalid() {
        let result = RequestCapture::from_json("{\"input\":}");
        assert!(result.is_err());
    }

    #[test]
    fn test_request_capture_clone() {
        let capture = RequestCapture::new()
            .with_input("test")
            .with_params("k", "v");
        let cloned = capture.clone();
        assert_eq!(cloned.input(), capture.input());
    }

    // ==================== StateDump Tests ====================

    #[test]
    fn test_state_dump_new() {
        let dump = StateDump::new();
        assert_eq!(dump.error(), "");
        assert_eq!(dump.stack_trace(), "");
        assert!(dump.state().is_empty());
    }

    #[test]
    fn test_state_dump_default() {
        let dump = StateDump::default();
        assert_eq!(dump.error(), "");
    }

    #[test]
    fn test_state_dump_with_error() {
        let dump = StateDump::new().with_error("OutOfMemory");
        assert_eq!(dump.error(), "OutOfMemory");
    }

    #[test]
    fn test_state_dump_with_stack_trace() {
        let dump = StateDump::new().with_stack_trace("at func1\nat func2");
        assert_eq!(dump.stack_trace(), "at func1\nat func2");
    }

    #[test]
    fn test_state_dump_with_state() {
        let dump = StateDump::new()
            .with_state("batch_size", "32")
            .with_state("seq_len", "256");

        assert_eq!(dump.state().get("batch_size"), Some(&"32".to_string()));
        assert_eq!(dump.state().get("seq_len"), Some(&"256".to_string()));
    }

    #[test]
    fn test_state_dump_to_json() {
        let dump = StateDump::new()
            .with_error("Error")
            .with_stack_trace("trace")
            .with_state("key", "value");
        let json = dump.to_json();

        assert!(json.contains("\"error\":\"Error\""));
        assert!(json.contains("\"stack_trace\":\"trace\""));
        assert!(json.contains("\"key\":\"value\""));
    }

    #[test]
    fn test_state_dump_to_json_newline_escape() {
        let dump = StateDump::new().with_stack_trace("line1\nline2");
        let json = dump.to_json();
        assert!(json.contains("\\n"));
    }

    #[test]
    fn test_state_dump_clone() {
        let dump = StateDump::new().with_error("err").with_stack_trace("trace");
        let cloned = dump.clone();
        assert_eq!(cloned.error(), dump.error());
        assert_eq!(cloned.stack_trace(), dump.stack_trace());
    }

    // ==================== DiagnosticsSummary Tests ====================

    #[test]
    fn test_diagnostics_summary_fields() {
        let summary = DiagnosticsSummary { request_count: 42 };
        assert_eq!(summary.request_count, 42);
    }

    #[test]
    fn test_diagnostics_summary_clone() {
        let summary = DiagnosticsSummary { request_count: 10 };
        let cloned = summary.clone();
        assert_eq!(cloned.request_count, summary.request_count);
    }
include!("diagnostics_log_level.rs");
}
