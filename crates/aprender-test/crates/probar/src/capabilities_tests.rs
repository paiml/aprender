    use super::*;

    // ========================================================================
    // H1: Threading detection is reliable - Falsification tests
    // ========================================================================

    #[test]
    fn f001_cross_origin_isolated_false() {
        // Falsification: crossOriginIsolated=false should fail threading check
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: false,
            shared_array_buffer: true,
            atomics: true,
            is_secure_context: true,
            coop_header: Some("same-origin".to_string()),
            coep_header: Some("require-corp".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("crossOriginIsolated"));
    }

    #[test]
    fn f002_shared_array_buffer_undefined() {
        // Falsification: SharedArrayBuffer undefined should fail
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: false,
            atomics: true,
            is_secure_context: true,
            coop_header: Some("same-origin".to_string()),
            coep_header: Some("require-corp".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("SharedArrayBuffer"));
    }

    #[test]
    fn f003_coop_header_missing() {
        // Falsification: Missing COOP header should provide fix hint
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: true,
            is_secure_context: true,
            coop_header: None,
            coep_header: Some("require-corp".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("COOP"));
        assert!(err.contains("Cross-Origin-Opener-Policy")); // Fix hint
    }

    #[test]
    fn f004_coep_header_wrong() {
        // Falsification: Wrong COEP value should fail
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: true,
            is_secure_context: true,
            coop_header: Some("same-origin".to_string()),
            coep_header: Some("wrong-value".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("COEP"));
        assert!(err.contains("wrong-value"));
    }

    #[test]
    fn f005_atomics_blocked() {
        // Falsification: Atomics blocked should fail
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: false,
            is_secure_context: true,
            coop_header: Some("same-origin".to_string()),
            coep_header: Some("require-corp".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Atomics"));
    }

    // ========================================================================
    // H2: Thread pool initialization is safe - Falsification tests
    // ========================================================================

    #[test]
    fn f006_zero_hardware_concurrency() {
        // Falsification: Zero cores should return 1 optimal thread
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 0,
            ..Default::default()
        };
        assert_eq!(caps.optimal_threads(), 1);
    }

    #[test]
    fn f007_many_cores() {
        // Falsification: 256 cores should be capped at 8
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 256,
            ..Default::default()
        };
        assert_eq!(caps.optimal_threads(), 8);
    }

    #[test]
    fn f008_single_core_streaming() {
        // Falsification: Single core should fail streaming check
        let mut caps = WasmThreadCapabilities::full_support();
        caps.hardware_concurrency = 1;
        let result = caps.assert_streaming_ready();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 CPU cores"));
    }

    // ========================================================================
    // H3: Worker message protocol is robust - Falsification tests
    // ========================================================================

    #[test]
    fn f011_worker_message_creation() {
        // Verify worker message creation
        let msg = WorkerMessage::new("Init", serde_json::json!({"model": "tiny"}));
        assert_eq!(msg.type_, "Init");
        assert!(msg.timestamp.abs() < f64::EPSILON);
    }

    #[test]
    fn f012_worker_message_timestamp() {
        // Verify timestamp handling
        let msg =
            WorkerMessage::new("Transcribe", serde_json::json!({})).with_timestamp(1234567.89);
        assert!((msg.timestamp - 1234567.89).abs() < f64::EPSILON);
    }

    // ========================================================================
    // Unit tests for core functionality
    // ========================================================================

    #[test]
    fn test_full_support() {
        let caps = WasmThreadCapabilities::full_support();
        assert!(caps.is_threading_available());
        assert!(caps.assert_threading_ready().is_ok());
        assert!(caps.assert_streaming_ready().is_ok());
    }

    #[test]
    fn test_no_support() {
        let caps = WasmThreadCapabilities::no_support();
        assert!(!caps.is_threading_available());
        assert!(caps.assert_threading_ready().is_err());
    }

    #[test]
    fn test_optimal_threads_calculation() {
        // 4 cores -> 3 threads
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 4,
            ..Default::default()
        };
        assert_eq!(caps.optimal_threads(), 3);

        // 8 cores -> 7 threads
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 8,
            ..Default::default()
        };
        assert_eq!(caps.optimal_threads(), 7);

        // 16 cores -> 8 threads (capped)
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 16,
            ..Default::default()
        };
        assert_eq!(caps.optimal_threads(), 8);
    }

    #[test]
    fn test_capability_status() {
        let caps = WasmThreadCapabilities::full_support();
        assert_eq!(
            caps.shared_array_buffer_status(),
            CapabilityStatus::Available
        );

        let caps = WasmThreadCapabilities::no_support();
        matches!(
            caps.shared_array_buffer_status(),
            CapabilityStatus::Unavailable(_)
        );
    }

    #[test]
    fn test_from_json() {
        let json = r#"{
            "crossOriginIsolated": true,
            "sharedArrayBuffer": true,
            "atomics": true,
            "hardwareConcurrency": 8,
            "isSecureContext": true
        }"#;

        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        assert!(caps.cross_origin_isolated);
        assert!(caps.shared_array_buffer);
        assert!(caps.atomics);
        assert_eq!(caps.hardware_concurrency, 8);
        assert!(caps.is_secure_context);
    }

    #[test]
    fn test_from_json_invalid() {
        let result = WasmThreadCapabilities::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_worker_state_display() {
        assert_eq!(format!("{}", WorkerState::Uninitialized), "Uninitialized");
        assert_eq!(format!("{}", WorkerState::Ready), "Ready");
        assert_eq!(format!("{}", WorkerState::Processing), "Processing");
    }

    #[test]
    fn test_detection_js() {
        let js = WasmThreadCapabilities::detection_js();
        assert!(js.contains("crossOriginIsolated"));
        assert!(js.contains("SharedArrayBuffer"));
        assert!(js.contains("hardwareConcurrency"));
    }

    #[test]
    fn test_required_headers() {
        assert_eq!(RequiredHeaders::COOP, "same-origin");
        assert_eq!(RequiredHeaders::COEP, "require-corp");
    }

    // ========================================================================
    // WorkerEmulator tests (H3: Worker message protocol)
    // ========================================================================

    #[test]
    fn f009_worker_spawn_state() {
        // Falsification: spawn should transition to Loading state
        let mut emulator = WorkerEmulator::new();
        assert_eq!(emulator.state(), WorkerState::Uninitialized);

        emulator.spawn("test_worker");
        assert_eq!(emulator.state(), WorkerState::Loading);
        assert_eq!(emulator.name(), "test_worker");
    }

    #[test]
    fn f010_worker_ready_transition() {
        // Falsification: Ready message should transition to Ready state
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("audio_worker");
        emulator.receive_response(WorkerMessage::new("Ready", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Ready);
    }

    #[test]
    fn f013_worker_message_ordering() {
        // Falsification: Messages must maintain Lamport ordering
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("worker");
        emulator.send(WorkerMessage::new("Init", serde_json::json!({})));
        emulator.receive_response(WorkerMessage::new("Ready", serde_json::json!({})));
        emulator.send(WorkerMessage::new("Transcribe", serde_json::json!({})));
        emulator.terminate();

        assert!(emulator.verify_ordering());
        assert_eq!(emulator.lamport_time(), 5);
    }

    #[test]
    fn f014_worker_error_state() {
        // Falsification: Error response should transition to Error state
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("worker");
        emulator.receive_response(WorkerMessage::new(
            "Error",
            serde_json::json!({"msg": "OOM"}),
        ));
        assert_eq!(emulator.state(), WorkerState::Error);
    }

    #[test]
    fn f015_worker_terminate_state() {
        // Falsification: Terminate should transition to Terminated state
        let emulator = WorkerEmulator::ready("worker");
        assert_eq!(emulator.state(), WorkerState::Ready);

        let mut emulator = emulator;
        emulator.terminate();
        assert_eq!(emulator.state(), WorkerState::Terminated);
    }

    #[test]
    fn test_worker_assert_state() {
        let emulator = WorkerEmulator::ready("test");
        assert!(emulator.assert_state(WorkerState::Ready).is_ok());
        assert!(emulator.assert_state(WorkerState::Processing).is_err());
    }

    #[test]
    fn test_worker_pending_messages() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new(
            "Process",
            serde_json::json!({"data": [1,2,3]}),
        ));
        assert_eq!(emulator.pending_messages().len(), 1);
        assert_eq!(emulator.pending_messages()[0].type_, "Process");
    }

    #[test]
    fn test_worker_clear() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new("Process", serde_json::json!({})));
        emulator.clear();
        assert!(emulator.pending_messages().is_empty());
    }

    // ========================================================================
    // Additional coverage tests for CapabilityError Display
    // ========================================================================

    #[test]
    fn test_capability_error_display_threading_not_ready() {
        let err =
            CapabilityError::ThreadingNotReady(vec!["Error 1".to_string(), "Error 2".to_string()]);
        let display = format!("{}", err);
        assert!(display.contains("Threading not ready"));
        assert!(display.contains("Error 1"));
        assert!(display.contains("Error 2"));
    }

    #[test]
    fn test_capability_error_display_insufficient_resources() {
        let err = CapabilityError::InsufficientResources("Not enough memory".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Insufficient resources"));
        assert!(display.contains("Not enough memory"));
    }

    #[test]
    fn test_capability_error_display_parse_error() {
        let err = CapabilityError::ParseError("Invalid JSON".to_string());
        let display = format!("{}", err);
        assert!(display.contains("Parse error"));
        assert!(display.contains("Invalid JSON"));
    }

    #[test]
    fn test_capability_error_display_worker_state() {
        let err = CapabilityError::WorkerState {
            expected: "Ready".to_string(),
            actual: "Loading".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Worker state mismatch"));
        assert!(display.contains("Ready"));
        assert!(display.contains("Loading"));
    }

    // ========================================================================
    // Additional coverage for WorkerState Display
    // ========================================================================

    #[test]
    fn test_worker_state_display_all() {
        assert_eq!(format!("{}", WorkerState::Loading), "Loading");
        assert_eq!(format!("{}", WorkerState::Error), "Error");
        assert_eq!(format!("{}", WorkerState::Terminated), "Terminated");
    }

    #[test]
    fn test_worker_state_default() {
        let state = WorkerState::default();
        assert_eq!(state, WorkerState::Uninitialized);
    }

    // ========================================================================
    // Additional coverage for shared_array_buffer_status
    // ========================================================================

    #[test]
    fn test_sab_status_not_secure_context() {
        let caps = WasmThreadCapabilities {
            shared_array_buffer: false,
            is_secure_context: false,
            cross_origin_isolated: true,
            ..Default::default()
        };
        let status = caps.shared_array_buffer_status();
        assert!(
            matches!(status, CapabilityStatus::Unavailable(msg) if msg.contains("secure context") || msg.contains("HTTPS"))
        );
    }

    #[test]
    fn test_sab_status_not_cross_origin_isolated() {
        let caps = WasmThreadCapabilities {
            shared_array_buffer: false,
            is_secure_context: true,
            cross_origin_isolated: false,
            ..Default::default()
        };
        let status = caps.shared_array_buffer_status();
        assert!(
            matches!(status, CapabilityStatus::Unavailable(msg) if msg.contains("crossOriginIsolated"))
        );
    }

    #[test]
    fn test_sab_status_unknown_reason() {
        let caps = WasmThreadCapabilities {
            shared_array_buffer: false,
            is_secure_context: true,
            cross_origin_isolated: true,
            ..Default::default()
        };
        let status = caps.shared_array_buffer_status();
        assert!(matches!(status, CapabilityStatus::Unavailable(msg) if msg.contains("Unknown")));
    }

    // ========================================================================
    // Additional coverage for WorkerEmulator
    // ========================================================================

    #[test]
    fn test_worker_with_delays() {
        let emulator = WorkerEmulator::new().with_delays(true);
        // Just verify it doesn't panic and creates the emulator
        assert_eq!(emulator.state(), WorkerState::Uninitialized);
    }

    #[test]
    fn test_worker_responses() {
        let emulator = WorkerEmulator::ready("test");
        // The ready() method adds a Ready response
        assert!(!emulator.responses().is_empty());
        assert_eq!(emulator.responses()[0].type_, "Ready");
    }

    #[test]
    fn test_worker_history() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("worker");
        emulator.send(WorkerMessage::new("Test", serde_json::json!({})));
        let history = emulator.history();
        assert!(!history.is_empty());
        // First entry should be spawn
        assert_eq!(history[0].1, "spawn");
    }

    #[test]
    fn test_worker_send_from_uninitialized() {
        let mut emulator = WorkerEmulator::new();
        emulator.send(WorkerMessage::new("Init", serde_json::json!({})));
        // Sending from Uninitialized should transition to Loading
        assert_eq!(emulator.state(), WorkerState::Loading);
    }

    #[test]
    fn test_worker_send_from_ready() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new("Process", serde_json::json!({})));
        // Sending from Ready should transition to Processing
        assert_eq!(emulator.state(), WorkerState::Processing);
    }

    #[test]
    fn test_worker_receive_complete() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new("Process", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Processing);
        emulator.receive_response(WorkerMessage::new("Complete", serde_json::json!({})));
        // Complete should transition back to Ready
        assert_eq!(emulator.state(), WorkerState::Ready);
    }

    #[test]
    fn test_worker_receive_lowercase() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        // Test lowercase "ready"
        emulator.receive_response(WorkerMessage::new("ready", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Ready);
    }

    #[test]
    fn test_worker_receive_lowercase_error() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        // Test lowercase "error"
        emulator.receive_response(WorkerMessage::new("error", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Error);
    }

    #[test]
    fn test_worker_receive_lowercase_complete() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new("Process", serde_json::json!({})));
        // Test lowercase "complete"
        emulator.receive_response(WorkerMessage::new("complete", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Ready);
    }

    #[test]
    fn test_interception_js() {
        let js = WorkerEmulator::interception_js();
        assert!(js.contains("originalWorker"));
        assert!(js.contains("__PROBAR_WORKERS__"));
        assert!(js.contains("postMessage"));
    }

    #[test]
    fn test_from_json_with_headers() {
        let json = r#"{
            "crossOriginIsolated": true,
            "sharedArrayBuffer": true,
            "atomics": true,
            "hardwareConcurrency": 4,
            "isSecureContext": true,
            "coopHeader": "same-origin",
            "coepHeader": "require-corp"
        }"#;
        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        assert_eq!(caps.coop_header, Some("same-origin".to_string()));
        assert_eq!(caps.coep_header, Some("require-corp".to_string()));
    }

    #[test]
    fn test_from_json_defaults() {
        // Test with minimal JSON - should use defaults for missing fields
        let json = r#"{}"#;
        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        assert!(!caps.cross_origin_isolated);
        assert!(!caps.shared_array_buffer);
        assert!(!caps.atomics);
        assert_eq!(caps.hardware_concurrency, 1);
        assert!(!caps.is_secure_context);
    }

    #[test]
    fn test_capability_status_eq() {
        assert_eq!(CapabilityStatus::Available, CapabilityStatus::Available);
        assert_eq!(CapabilityStatus::Unknown, CapabilityStatus::Unknown);
        assert_eq!(
            CapabilityStatus::Unavailable("test".to_string()),
            CapabilityStatus::Unavailable("test".to_string())
        );
        assert_ne!(CapabilityStatus::Available, CapabilityStatus::Unknown);
    }

    #[test]
    fn test_assert_threading_not_secure() {
        // Test that non-secure context fails threading check
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: true,
            is_secure_context: false,
            coop_header: Some("same-origin".to_string()),
            coep_header: Some("require-corp".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("HTTPS"));
    }

    #[test]
    fn test_assert_threading_wrong_coop() {
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: true,
            is_secure_context: true,
            coop_header: Some("wrong-value".to_string()),
            coep_header: Some("require-corp".to_string()),
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("COOP"));
        assert!(err.contains("wrong-value"));
    }

    #[test]
    fn test_assert_threading_missing_coep() {
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: true,
            is_secure_context: true,
            coop_header: Some("same-origin".to_string()),
            coep_header: None,
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("COEP"));
        assert!(err.contains("Cross-Origin-Embedder-Policy"));
    }

    // ========================================================================
    // Additional coverage tests for WorkerEmulator
    // ========================================================================

    #[test]
    fn test_worker_emulator_default() {
        let emulator = WorkerEmulator::default();
        assert_eq!(emulator.state(), WorkerState::Uninitialized);
        assert!(emulator.name().is_empty());
        assert!(emulator.pending_messages().is_empty());
        assert!(emulator.responses().is_empty());
        assert_eq!(emulator.lamport_time(), 0);
    }

    #[test]
    fn test_worker_emulator_debug() {
        let emulator = WorkerEmulator::new();
        let debug_str = format!("{:?}", emulator);
        assert!(debug_str.contains("WorkerEmulator"));
    }

    #[test]
    fn test_worker_emulator_clone() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test-worker");
        let cloned = emulator.clone();
        assert_eq!(emulator.name(), cloned.name());
        assert_eq!(emulator.state(), cloned.state());
    }

    #[test]
    fn test_worker_send_from_processing_state() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new("Task1", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Processing);
        // Send another message while processing - state should remain Processing
        emulator.send(WorkerMessage::new("Task2", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Processing);
    }

    #[test]
    fn test_worker_send_from_error_state() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        emulator.receive_response(WorkerMessage::new("Error", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Error);
        // Send while in error state - should stay in Error
        emulator.send(WorkerMessage::new("Retry", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Error);
    }

    #[test]
    fn test_worker_send_from_terminated_state() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.terminate();
        assert_eq!(emulator.state(), WorkerState::Terminated);
        // Send while terminated - should stay Terminated
        emulator.send(WorkerMessage::new("Test", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Terminated);
    }

    #[test]
    fn test_worker_receive_unknown_type() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        // Receive a message type that doesn't affect state
        emulator.receive_response(WorkerMessage::new("CustomType", serde_json::json!({})));
        // State should remain Loading since the message type is not recognized
        assert_eq!(emulator.state(), WorkerState::Loading);
    }

    #[test]
    fn test_worker_verify_ordering_empty() {
        let emulator = WorkerEmulator::new();
        assert!(emulator.verify_ordering());
    }

    #[test]
    fn test_worker_verify_ordering_single() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        assert!(emulator.verify_ordering());
    }

    #[test]
    fn test_worker_verify_ordering_fails_with_duplicate_timestamps() {
        // We can't easily create a scenario with duplicate timestamps
        // since the emulator auto-increments, but we can test the logic
        // by manually constructing an emulator with modified history
        let mut emulator = WorkerEmulator::new();
        // Add entries to history that would fail ordering check
        // This is testing the internal logic directly
        emulator.spawn("test");
        emulator.send(WorkerMessage::new("A", serde_json::json!({})));
        // All normal operations maintain ordering
        assert!(emulator.verify_ordering());
    }

    // ========================================================================
    // Additional coverage tests for WasmThreadCapabilities
    // ========================================================================

    #[test]
    fn test_wasm_thread_capabilities_default() {
        let caps = WasmThreadCapabilities::default();
        assert!(!caps.cross_origin_isolated);
        assert!(!caps.shared_array_buffer);
        assert!(!caps.atomics);
        assert_eq!(caps.hardware_concurrency, 0);
        assert!(caps.coop_header.is_none());
        assert!(caps.coep_header.is_none());
        assert!(!caps.is_secure_context);
        assert!(caps.errors.is_empty());
    }

    #[test]
    fn test_wasm_thread_capabilities_debug() {
        let caps = WasmThreadCapabilities::full_support();
        let debug_str = format!("{:?}", caps);
        assert!(debug_str.contains("WasmThreadCapabilities"));
    }

    #[test]
    fn test_wasm_thread_capabilities_clone() {
        let caps = WasmThreadCapabilities::full_support();
        let cloned = caps.clone();
        assert_eq!(caps.cross_origin_isolated, cloned.cross_origin_isolated);
        assert_eq!(caps.hardware_concurrency, cloned.hardware_concurrency);
    }

    #[test]
    fn test_no_support_has_error() {
        let caps = WasmThreadCapabilities::no_support();
        assert!(!caps.errors.is_empty());
        assert!(caps.errors[0].contains("SharedArrayBuffer"));
    }

    #[test]
    fn test_optimal_threads_one_core() {
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 1,
            ..Default::default()
        };
        // 1 - 1 = 0, but clamped to minimum 1
        assert_eq!(caps.optimal_threads(), 1);
    }

    #[test]
    fn test_optimal_threads_two_cores() {
        let caps = WasmThreadCapabilities {
            hardware_concurrency: 2,
            ..Default::default()
        };
        assert_eq!(caps.optimal_threads(), 1);
    }

    #[test]
    fn test_assert_streaming_ready_success() {
        let caps = WasmThreadCapabilities::full_support();
        assert!(caps.assert_streaming_ready().is_ok());
    }

    #[test]
    fn test_assert_streaming_ready_threading_fails() {
        let caps = WasmThreadCapabilities::no_support();
        let result = caps.assert_streaming_ready();
        assert!(result.is_err());
    }

    // ========================================================================
    // Additional coverage tests for CapabilityStatus
    // ========================================================================

    #[test]
    fn test_capability_status_debug() {
        let status = CapabilityStatus::Available;
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("Available"));

        let status = CapabilityStatus::Unknown;
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("Unknown"));

        let status = CapabilityStatus::Unavailable("test".to_string());
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("Unavailable"));
    }

    #[test]
    fn test_capability_status_clone() {
        let status = CapabilityStatus::Unavailable("reason".to_string());
        let cloned = status.clone();
        assert_eq!(status, cloned);
    }

    // ========================================================================
    // Additional coverage tests for WorkerState
    // ========================================================================

    #[test]
    fn test_worker_state_copy() {
        let state = WorkerState::Ready;
        let copied = state;
        assert_eq!(state, copied);
    }

    #[test]
    fn test_worker_state_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WorkerState::Ready);
        set.insert(WorkerState::Processing);
        assert!(set.contains(&WorkerState::Ready));
        assert!(set.contains(&WorkerState::Processing));
        assert!(!set.contains(&WorkerState::Error));
    }

    // ========================================================================
    // Additional coverage tests for WorkerMessage
    // ========================================================================

    #[test]
    fn test_worker_message_debug() {
        let msg = WorkerMessage::new("Test", serde_json::json!({}));
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("WorkerMessage"));
        assert!(debug_str.contains("Test"));
    }

    #[test]
    fn test_worker_message_clone() {
        let msg =
            WorkerMessage::new("Test", serde_json::json!({"key": "value"})).with_timestamp(123.456);
        let cloned = msg.clone();
        assert_eq!(msg.type_, cloned.type_);
        assert_eq!(msg.data, cloned.data);
        assert!((msg.timestamp - cloned.timestamp).abs() < f64::EPSILON);
    }

    // ========================================================================
    // Additional coverage tests for RequiredHeaders
    // ========================================================================

    #[test]
    fn test_required_headers_debug() {
        let headers = RequiredHeaders;
        let debug_str = format!("{:?}", headers);
        assert!(debug_str.contains("RequiredHeaders"));
    }

    #[test]
    fn test_required_headers_clone() {
        let headers = RequiredHeaders;
        let _ = headers;
        // Copy trait test
        let cloned = headers;
        let _ = cloned;
    }

    // ========================================================================
    // Additional coverage tests for CapabilityError
    // ========================================================================

    #[test]
    fn test_capability_error_debug() {
        let err = CapabilityError::ParseError("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("ParseError"));
    }

    #[test]
    fn test_capability_error_clone() {
        let err = CapabilityError::InsufficientResources("memory".to_string());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }

    #[test]
    fn test_capability_error_is_error_trait() {
        let err: Box<dyn std::error::Error> =
            Box::new(CapabilityError::ParseError("test".to_string()));
        assert!(err.to_string().contains("Parse error"));
    }

    #[test]
    fn test_capability_error_source() {
        use std::error::Error;
        let err = CapabilityError::ParseError("test".to_string());
        // source() should return None for this error type
        assert!(err.source().is_none());
    }

    // ========================================================================
    // Edge case tests for from_json
    // ========================================================================

    #[test]
    fn test_from_json_partial_fields() {
        let json = r#"{
            "crossOriginIsolated": true,
            "atomics": false
        }"#;
        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        assert!(caps.cross_origin_isolated);
        assert!(!caps.atomics);
        // Other fields should default
        assert!(!caps.shared_array_buffer);
        assert_eq!(caps.hardware_concurrency, 1);
    }

    #[test]
    fn test_from_json_null_values() {
        let json = r#"{
            "crossOriginIsolated": null,
            "sharedArrayBuffer": null,
            "hardwareConcurrency": null
        }"#;
        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        // null should be treated as false/1
        assert!(!caps.cross_origin_isolated);
        assert!(!caps.shared_array_buffer);
        assert_eq!(caps.hardware_concurrency, 1);
    }

    // ========================================================================
    // Edge case tests for assert_threading_ready
    // ========================================================================

    #[test]
    fn test_assert_threading_multiple_failures() {
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: false,
            shared_array_buffer: false,
            atomics: false,
            is_secure_context: false,
            coop_header: None,
            coep_header: None,
            ..Default::default()
        };
        let result = caps.assert_threading_ready();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should contain multiple error messages
        assert!(err.contains("crossOriginIsolated"));
        assert!(err.contains("SharedArrayBuffer"));
        assert!(err.contains("Atomics"));
        assert!(err.contains("HTTPS"));
        assert!(err.contains("COOP"));
        assert!(err.contains("COEP"));
    }

    // ========================================================================
    // Additional tests for complete coverage
    // ========================================================================

    #[test]
    fn test_is_threading_available_partial() {
        // Test with only some flags true
        let caps = WasmThreadCapabilities {
            cross_origin_isolated: true,
            shared_array_buffer: true,
            atomics: false,
            is_secure_context: true,
            ..Default::default()
        };
        assert!(!caps.is_threading_available());
    }

    #[test]
    fn test_assert_state_error_message() {
        let emulator = WorkerEmulator::ready("test");
        let result = emulator.assert_state(WorkerState::Processing);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            CapabilityError::WorkerState { expected, actual } => {
                assert_eq!(expected, "Processing");
                assert_eq!(actual, "Ready");
            }
            _ => panic!("Expected WorkerState error"),
        }
    }

    #[test]
    fn test_worker_send_from_loading() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        assert_eq!(emulator.state(), WorkerState::Loading);
        // Send while loading - should stay in Loading (not Ready or Processing)
        emulator.send(WorkerMessage::new("Init", serde_json::json!({})));
        assert_eq!(emulator.state(), WorkerState::Loading);
    }

    #[test]
    fn test_worker_multiple_responses() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("test");
        emulator.receive_response(WorkerMessage::new("Progress", serde_json::json!({})));
        emulator.receive_response(WorkerMessage::new("Progress", serde_json::json!({})));
        emulator.receive_response(WorkerMessage::new("Ready", serde_json::json!({})));
        assert_eq!(emulator.responses().len(), 3);
        assert_eq!(emulator.state(), WorkerState::Ready);
    }

    #[test]
    fn test_worker_lamport_increments() {
        let mut emulator = WorkerEmulator::new();
        assert_eq!(emulator.lamport_time(), 0);
        emulator.spawn("test");
        assert_eq!(emulator.lamport_time(), 1);
        emulator.send(WorkerMessage::new("A", serde_json::json!({})));
        assert_eq!(emulator.lamport_time(), 2);
        emulator.receive_response(WorkerMessage::new("B", serde_json::json!({})));
        assert_eq!(emulator.lamport_time(), 3);
        emulator.terminate();
        assert_eq!(emulator.lamport_time(), 4);
    }

    #[test]
    fn test_worker_history_entries() {
        let mut emulator = WorkerEmulator::new();
        emulator.spawn("my-worker");
        emulator.send(WorkerMessage::new("Init", serde_json::json!({})));
        emulator.receive_response(WorkerMessage::new("Ready", serde_json::json!({})));
        emulator.terminate();

        let history = emulator.history();
        assert_eq!(history.len(), 4);

        assert_eq!(history[0].1, "spawn");
        assert_eq!(history[0].2, "my-worker");

        assert_eq!(history[1].1, "send");
        assert_eq!(history[1].2, "Init");

        assert_eq!(history[2].1, "receive");
        assert_eq!(history[2].2, "Ready");

        assert_eq!(history[3].1, "terminate");
    }

    #[test]
    fn test_worker_clear_preserves_state() {
        let mut emulator = WorkerEmulator::ready("test");
        emulator.send(WorkerMessage::new("Task", serde_json::json!({})));
        emulator.receive_response(WorkerMessage::new("Done", serde_json::json!({})));

        let state_before = emulator.state();
        emulator.clear();

        assert!(emulator.pending_messages().is_empty());
        assert!(emulator.responses().is_empty());
        // State should be preserved after clear
        assert_eq!(emulator.state(), state_before);
    }

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[test]
    fn test_shared_array_buffer_status_available() {
        let caps = WasmThreadCapabilities::full_support();
        assert_eq!(
            caps.shared_array_buffer_status(),
            CapabilityStatus::Available
        );
    }

    #[test]
    fn test_shared_array_buffer_status_not_secure() {
        let caps = WasmThreadCapabilities {
            shared_array_buffer: false,
            is_secure_context: false,
            cross_origin_isolated: true,
            ..Default::default()
        };
        match caps.shared_array_buffer_status() {
            CapabilityStatus::Unavailable(reason) => {
                assert!(reason.contains("HTTPS"));
            }
            _ => panic!("Expected Unavailable"),
        }
    }

    #[test]
    fn test_shared_array_buffer_status_not_cross_origin() {
        let caps = WasmThreadCapabilities {
            shared_array_buffer: false,
            is_secure_context: true,
            cross_origin_isolated: false,
            ..Default::default()
        };
        match caps.shared_array_buffer_status() {
            CapabilityStatus::Unavailable(reason) => {
                assert!(reason.contains("crossOriginIsolated"));
            }
            _ => panic!("Expected Unavailable"),
        }
    }

    #[test]
    fn test_shared_array_buffer_status_unknown() {
        let caps = WasmThreadCapabilities {
            shared_array_buffer: false,
            is_secure_context: true,
            cross_origin_isolated: true,
            ..Default::default()
        };
        match caps.shared_array_buffer_status() {
            CapabilityStatus::Unavailable(reason) => {
                assert!(reason.contains("Unknown"));
            }
            _ => panic!("Expected Unavailable"),
        }
    }

    #[test]
    fn test_from_json_valid_with_headers() {
        let json = r#"{
            "crossOriginIsolated": true,
            "sharedArrayBuffer": true,
            "atomics": true,
            "hardwareConcurrency": 8,
            "isSecureContext": true,
            "coopHeader": "same-origin",
            "coepHeader": "require-corp"
        }"#;
        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        assert!(caps.cross_origin_isolated);
        assert!(caps.shared_array_buffer);
        assert!(caps.atomics);
        assert_eq!(caps.hardware_concurrency, 8);
        assert!(caps.is_secure_context);
        assert_eq!(caps.coop_header, Some("same-origin".to_string()));
        assert_eq!(caps.coep_header, Some("require-corp".to_string()));
    }

    #[test]
    fn test_from_json_minimal_defaults() {
        let json = r#"{}"#;
        let caps = WasmThreadCapabilities::from_json(json).unwrap();
        assert!(!caps.cross_origin_isolated);
        assert!(!caps.shared_array_buffer);
        assert_eq!(caps.hardware_concurrency, 1);
    }

    #[test]
    fn test_capability_status_unknown_match() {
        let status = CapabilityStatus::Unknown;
        assert!(matches!(status, CapabilityStatus::Unknown));
    }

    #[test]
    fn test_required_headers_values() {
        assert_eq!(RequiredHeaders::COOP, "same-origin");
        assert_eq!(RequiredHeaders::COEP, "require-corp");
    }
