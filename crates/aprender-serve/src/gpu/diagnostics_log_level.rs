
    // ==================== LogLevel Tests ====================

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Warn.as_str(), "WARN");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
    }

    #[test]
    fn test_log_level_clone_copy() {
        let level = LogLevel::Info;
        let cloned = level;
        let copied: LogLevel = level;
        assert_eq!(level, cloned);
        assert_eq!(level, copied);
    }

    #[test]
    fn test_log_level_eq() {
        assert_eq!(LogLevel::Error, LogLevel::Error);
        assert_ne!(LogLevel::Error, LogLevel::Warn);
    }

    // ==================== LogEntry Tests ====================

    #[test]
    fn test_log_entry_new() {
        let entry = LogEntry::new(LogLevel::Info, "Test message");
        assert_eq!(entry.level(), LogLevel::Info);
        assert!(entry.timestamp() > 0);
        assert!(entry.correlation_id().is_none());
    }

    #[test]
    fn test_log_entry_with_correlation_id() {
        let entry = LogEntry::new(LogLevel::Debug, "msg").with_correlation_id("req-123");
        assert_eq!(entry.correlation_id(), Some("req-123"));
    }

    #[test]
    fn test_log_entry_with_field() {
        let entry = LogEntry::new(LogLevel::Warn, "warning")
            .with_field("key1", "value1")
            .with_field("key2", "value2");
        let json = entry.to_json();
        assert!(json.contains("\"key1\":\"value1\""));
        assert!(json.contains("\"key2\":\"value2\""));
    }

    #[test]
    fn test_log_entry_to_json_basic() {
        let entry = LogEntry::new(LogLevel::Error, "error message");
        let json = entry.to_json();
        assert!(json.contains("\"level\":\"ERROR\""));
        assert!(json.contains("\"message\":\"error message\""));
        assert!(json.contains("\"timestamp\":"));
    }

    #[test]
    fn test_log_entry_to_json_with_correlation() {
        let entry = LogEntry::new(LogLevel::Info, "msg").with_correlation_id("corr-456");
        let json = entry.to_json();
        assert!(json.contains("\"correlation_id\":\"corr-456\""));
    }

    #[test]
    fn test_log_entry_clone() {
        let entry = LogEntry::new(LogLevel::Debug, "test")
            .with_correlation_id("id")
            .with_field("k", "v");
        let cloned = entry.clone();
        assert_eq!(cloned.level(), entry.level());
        assert_eq!(cloned.correlation_id(), entry.correlation_id());
    }

    // ==================== LogConfig Tests ====================

    #[test]
    fn test_log_config_new() {
        let config = LogConfig::new();
        assert!(!config.json_format);
    }

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert!(!config.json_format);
    }

    #[test]
    fn test_log_config_with_level() {
        let config = LogConfig::new().with_level(LogLevel::Debug);
        assert_eq!(config.default_level, LogLevel::Debug);
    }

    #[test]
    fn test_log_config_with_json_format() {
        let config = LogConfig::new().with_json_format(true);
        assert!(config.json_format);
    }

    #[test]
    fn test_log_config_with_module_level() {
        let config = LogConfig::new()
            .with_module_level("gpu", LogLevel::Trace)
            .with_module_level("inference", LogLevel::Error);
        assert_eq!(config.module_levels.get("gpu"), Some(&LogLevel::Trace));
        assert_eq!(
            config.module_levels.get("inference"),
            Some(&LogLevel::Error)
        );
    }

    #[test]
    fn test_log_config_clone() {
        let config = LogConfig::new()
            .with_level(LogLevel::Warn)
            .with_json_format(true);
        let cloned = config.clone();
        assert_eq!(cloned.default_level, config.default_level);
        assert_eq!(cloned.json_format, config.json_format);
    }

    // ==================== Logger Tests ====================

    #[test]
    fn test_logger_new() {
        let config = LogConfig::new().with_level(LogLevel::Info);
        let logger = Logger::new(config);
        assert!(logger.is_enabled(LogLevel::Error, "any_module"));
    }

    #[test]
    fn test_logger_is_enabled_default_level() {
        let config = LogConfig::new().with_level(LogLevel::Info);
        let logger = Logger::new(config);

        assert!(!logger.is_enabled(LogLevel::Trace, "module"));
        assert!(!logger.is_enabled(LogLevel::Debug, "module"));
        assert!(logger.is_enabled(LogLevel::Info, "module"));
        assert!(logger.is_enabled(LogLevel::Warn, "module"));
        assert!(logger.is_enabled(LogLevel::Error, "module"));
    }

    #[test]
    fn test_logger_is_enabled_module_level() {
        let config = LogConfig::new()
            .with_level(LogLevel::Info)
            .with_module_level("gpu", LogLevel::Trace);
        let logger = Logger::new(config);

        // GPU module should allow Trace
        assert!(logger.is_enabled(LogLevel::Trace, "gpu"));
        // Other modules use default Info level
        assert!(!logger.is_enabled(LogLevel::Debug, "other"));
    }

    // ==================== PhaseTimer Tests ====================

    #[test]
    fn test_phase_timer_new() {
        let timer = PhaseTimer::new();
        let breakdown = timer.breakdown();
        assert!(breakdown.is_empty());
    }

    #[test]
    fn test_phase_timer_default() {
        let timer = PhaseTimer::default();
        assert!(timer.breakdown().is_empty());
    }

    #[test]
    fn test_phase_timer_start_end_phase() {
        let timer = PhaseTimer::new();
        timer.start_phase("attention");
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.end_phase("attention");

        let breakdown = timer.breakdown();
        assert!(breakdown.contains_key("attention"));
        assert!(breakdown["attention"] > 0);
    }

    #[test]
    fn test_phase_timer_multiple_phases() {
        let timer = PhaseTimer::new();
        timer.start_phase("phase1");
        timer.end_phase("phase1");
        timer.start_phase("phase2");
        timer.end_phase("phase2");

        let breakdown = timer.breakdown();
        assert_eq!(breakdown.len(), 2);
    }

    #[test]
    fn test_phase_timer_end_without_start() {
        let timer = PhaseTimer::new();
        timer.end_phase("nonexistent");
        // Should not panic, just ignore
        assert!(timer.breakdown().is_empty());
    }

    // ==================== MemoryReport Tests ====================

    #[test]
    fn test_memory_report_fields() {
        let report = MemoryReport {
            peak_bytes: 1024,
            current_bytes: 512,
            allocation_count: 10,
        };
        assert_eq!(report.peak_bytes, 1024);
        assert_eq!(report.current_bytes, 512);
        assert_eq!(report.allocation_count, 10);
    }

    #[test]
    fn test_memory_report_clone() {
        let report = MemoryReport {
            peak_bytes: 2048,
            current_bytes: 1024,
            allocation_count: 5,
        };
        let cloned = report.clone();
        assert_eq!(cloned.peak_bytes, report.peak_bytes);
    }

    // ==================== MemoryTracker Tests ====================

    #[test]
    fn test_memory_tracker_new() {
        let tracker = MemoryTracker::new();
        let report = tracker.report();
        assert_eq!(report.current_bytes, 0);
        assert_eq!(report.peak_bytes, 0);
        assert_eq!(report.allocation_count, 0);
    }

    #[test]
    fn test_memory_tracker_default() {
        let tracker = MemoryTracker::default();
        let report = tracker.report();
        assert_eq!(report.current_bytes, 0);
    }

    #[test]
    fn test_memory_tracker_record_allocation() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("buffer1", 1024);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 1024);
        assert_eq!(report.peak_bytes, 1024);
        assert_eq!(report.allocation_count, 1);
    }

    #[test]
    fn test_memory_tracker_multiple_allocations() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("a", 100);
        tracker.record_allocation("b", 200);
        tracker.record_allocation("c", 300);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 600);
        assert_eq!(report.peak_bytes, 600);
        assert_eq!(report.allocation_count, 3);
    }

    #[test]
    fn test_memory_tracker_record_deallocation() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("buf", 1000);
        tracker.record_deallocation("buf", 400);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 600);
        assert_eq!(report.peak_bytes, 1000); // Peak unchanged
    }

    #[test]
    fn test_memory_tracker_peak_tracking() {
        let tracker = MemoryTracker::new();
        tracker.record_allocation("a", 500);
        tracker.record_allocation("b", 500);
        // Peak is now 1000
        tracker.record_deallocation("a", 500);
        tracker.record_allocation("c", 200);

        let report = tracker.report();
        assert_eq!(report.current_bytes, 700);
        assert_eq!(report.peak_bytes, 1000); // Peak was 1000
    }

    // ==================== DiagnosticsCollector Tests ====================

    #[test]
    fn test_diagnostics_collector_new() {
        let collector = DiagnosticsCollector::new();
        let summary = collector.summary();
        assert_eq!(summary.request_count, 0);
    }

    #[test]
    fn test_diagnostics_collector_default() {
        let collector = DiagnosticsCollector::default();
        assert_eq!(collector.summary().request_count, 0);
    }

    #[test]
    fn test_diagnostics_collector_record_request_timing() {
        let collector = DiagnosticsCollector::new();
        let mut timing = HashMap::new();
        timing.insert("attention".to_string(), 100);
        timing.insert("ffn".to_string(), 50);

        collector.record_request_timing("req-1", timing);

        let summary = collector.summary();
        assert_eq!(summary.request_count, 1);
    }

    #[test]
    fn test_diagnostics_collector_multiple_requests() {
        let collector = DiagnosticsCollector::new();

        collector.record_request_timing("r1", HashMap::new());
        collector.record_request_timing("r2", HashMap::new());
        collector.record_request_timing("r3", HashMap::new());

        assert_eq!(collector.summary().request_count, 3);
    }

    #[test]
    fn test_diagnostics_collector_record_memory_snapshot() {
        let collector = DiagnosticsCollector::new();
        let report = MemoryReport {
            peak_bytes: 1024,
            current_bytes: 512,
            allocation_count: 2,
        };

        collector.record_memory_snapshot(report);
        // No assertion needed - just verify no panic
    }

    // ==================== DebugMode Tests ====================

    #[test]
    fn test_debug_mode_new() {
        let debug = DebugMode::new();
        assert!(!debug.is_enabled());
    }

    #[test]
    fn test_debug_mode_default() {
        let debug = DebugMode::default();
        assert!(!debug.is_enabled());
    }

    #[test]
    fn test_debug_mode_enable() {
        let debug = DebugMode::new();
        assert!(!debug.is_enabled());
        debug.enable();
        assert!(debug.is_enabled());
    }

    #[test]
    fn test_debug_mode_disable() {
        let debug = DebugMode::new();
        debug.enable();
        assert!(debug.is_enabled());
        debug.disable();
        assert!(!debug.is_enabled());
    }

    // ==================== RequestCapture Tests ====================

    #[test]
    fn test_request_capture_new() {
        let capture = RequestCapture::new();
        assert_eq!(capture.input(), "");
        assert!(capture.params().is_empty());
    }

    #[test]
    fn test_request_capture_default() {
        let capture = RequestCapture::default();
        assert_eq!(capture.input(), "");
    }

    #[test]
    fn test_request_capture_with_input() {
        let capture = RequestCapture::new().with_input("Hello, world!");
        assert_eq!(capture.input(), "Hello, world!");
    }

    #[test]
    fn test_request_capture_with_params() {
        let capture = RequestCapture::new()
            .with_params("temp", "0.7")
            .with_params("max_tokens", "100");

        assert_eq!(capture.params().get("temp"), Some(&"0.7".to_string()));
        assert_eq!(capture.params().get("max_tokens"), Some(&"100".to_string()));
    }

    #[test]
    fn test_request_capture_to_json() {
        let capture = RequestCapture::new()
            .with_input("prompt")
            .with_params("temp", "0.5");
        let json = capture.to_json();

        assert!(json.contains("\"input\":\"prompt\""));
        assert!(json.contains("\"temp\":\"0.5\""));
    }

    #[test]
    fn test_request_capture_from_json() {
        let json = "{\"input\":\"test prompt\",\"params\":{}}";
        let capture = RequestCapture::from_json(json).expect("capture");
        assert_eq!(capture.input(), "test prompt");
    }

    #[test]
    fn test_request_capture_from_json_error() {
        let result = RequestCapture::from_json("{\"no_input\":\"value\"}");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Missing input");
    }
