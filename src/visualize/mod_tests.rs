use super::*;

#[test]
fn test_default_config() {
    let config = VisualizeConfig::default();
    assert_eq!(config.tick_rate_ms, 50);
    assert_eq!(config.history_size, 300);
    assert!(config.enable_anomaly);
    assert!(!config.enable_ml);
}

#[test]
fn test_config_clone() {
    let config = VisualizeConfig {
        tick_rate_ms: 100,
        enable_ml: true,
        ml_clusters: 5,
        ..Default::default()
    };
    let cloned = config.clone();
    assert_eq!(cloned.tick_rate_ms, 100);
    assert!(cloned.enable_ml);
    assert_eq!(cloned.ml_clusters, 5);
}

#[test]
fn test_config_all_fields() {
    let config = VisualizeConfig {
        tick_rate_ms: 100,
        enable_anomaly: false,
        enable_ml: true,
        ml_clusters: 5,
        anomaly_threshold: 2.5,
        history_size: 500,
        deterministic: true,
        show_fps: true,
        pid: Some(1234),
        enable_source: true,
        filter: Some("read|write".to_string()),
        otlp_endpoint: Some("http://localhost:4317".to_string()),
        enable_metrics: true,
        enable_alerts: true,
        alert_latency_threshold_us: 5000,
        alert_error_rate_percent: 2.5,
    };

    assert_eq!(config.tick_rate_ms, 100);
    assert!(!config.enable_anomaly);
    assert!(config.enable_ml);
    assert_eq!(config.ml_clusters, 5);
    assert!((config.anomaly_threshold - 2.5).abs() < f32::EPSILON);
    assert_eq!(config.history_size, 500);
    assert!(config.deterministic);
    assert!(config.show_fps);
    assert_eq!(config.pid, Some(1234));
    assert!(config.enable_source);
    assert_eq!(config.filter, Some("read|write".to_string()));
    assert_eq!(config.otlp_endpoint, Some("http://localhost:4317".to_string()));
    assert!(config.enable_metrics);
    assert!(config.enable_alerts);
    assert_eq!(config.alert_latency_threshold_us, 5000);
    assert!((config.alert_error_rate_percent - 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_inject_demo_data() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Initial state
    assert_eq!(app.total_syscalls, 0);

    // Inject demo data for several ticks
    for tick in 0..10 {
        inject_demo_data(&mut app, tick);
    }

    // After injecting, should have syscalls recorded
    assert!(app.total_syscalls > 0);

    // Should have some latency data
    assert!(!app.latency_history.is_empty());
}

#[test]
fn test_inject_demo_data_deterministic() {
    let config = VisualizeConfig::default();
    let mut app1 = app::VisualizeApp::new(config.clone());
    let mut app2 = app::VisualizeApp::new(config);

    // Same tick should produce deterministic results
    inject_demo_data(&mut app1, 42);
    inject_demo_data(&mut app2, 42);

    assert_eq!(app1.total_syscalls, app2.total_syscalls);
}

#[test]
fn test_inject_demo_data_distribution() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Inject enough data to get good distribution
    for tick in 0..100 {
        inject_demo_data(&mut app, tick);
    }

    // Should have substantial syscall count
    assert!(app.total_syscalls > 500);

    // Should have some errors (2% rate)
    assert!(app.total_errors > 0);

    // Should have some anomalies
    assert!(app.anomaly_count > 0);
}

#[test]
fn test_config_debug() {
    let config = VisualizeConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("tick_rate_ms"));
    assert!(debug_str.contains("enable_anomaly"));
}

#[test]
fn test_sprint56_config_defaults() {
    let config = VisualizeConfig::default();

    // Sprint 56 defaults
    assert!(!config.enable_metrics);
    assert!(!config.enable_alerts);
    assert_eq!(config.alert_latency_threshold_us, 10_000);
    assert!((config.alert_error_rate_percent - 5.0).abs() < f32::EPSILON);
}

#[test]
fn test_config_with_sprint56_features() {
    let config = VisualizeConfig {
        enable_metrics: true,
        enable_alerts: true,
        alert_latency_threshold_us: 20_000,
        alert_error_rate_percent: 10.0,
        ..Default::default()
    };

    assert!(config.enable_metrics);
    assert!(config.enable_alerts);
    assert_eq!(config.alert_latency_threshold_us, 20_000);
    assert!((config.alert_error_rate_percent - 10.0).abs() < f32::EPSILON);
}

#[test]
fn test_inject_demo_data_syscall_categories() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Test with various seed values to hit different syscall categories
    for tick in 0..200 {
        inject_demo_data(&mut app, tick);
    }

    // Should have recorded syscalls from various categories
    assert!(app.total_syscalls > 1000);
}

#[test]
fn test_inject_demo_data_anomaly_detection() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Run enough ticks to trigger anomaly detection
    for tick in 0..500 {
        inject_demo_data(&mut app, tick);
    }

    // Should have detected some anomalies (3% chance per syscall)
    assert!(app.anomaly_count > 0);
}

#[test]
fn test_inject_demo_data_error_rate() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Run enough ticks to get reliable error rate
    for tick in 0..1000 {
        inject_demo_data(&mut app, tick);
    }

    // Error rate should be approximately 2%
    let error_rate = app.total_errors as f64 / app.total_syscalls as f64;
    assert!(error_rate > 0.01 && error_rate < 0.05);
}

#[test]
fn test_config_with_pid() {
    let config = VisualizeConfig { pid: Some(12345), ..Default::default() };

    assert_eq!(config.pid, Some(12345));
}

#[test]
fn test_config_with_filter() {
    let config = VisualizeConfig { filter: Some("mmap|munmap".to_string()), ..Default::default() };

    assert_eq!(config.filter, Some("mmap|munmap".to_string()));
}

#[test]
fn test_config_with_otlp() {
    let config = VisualizeConfig {
        otlp_endpoint: Some("http://jaeger:4317".to_string()),
        ..Default::default()
    };

    assert_eq!(config.otlp_endpoint, Some("http://jaeger:4317".to_string()));
}

#[test]
fn test_visualize_config_deterministic_mode() {
    let config = VisualizeConfig { deterministic: true, ..Default::default() };

    assert!(config.deterministic);
}

#[test]
fn test_update_frame_stats() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);
    let mut frame_times: Vec<u64> = Vec::with_capacity(60);

    // Test initial update
    update_frame_stats(&mut app, &mut frame_times, 1000);
    assert_eq!(app.frame_id, 1);
    assert_eq!(app.avg_frame_time_us, 1000);
    assert_eq!(app.max_frame_time_us, 1000);
    assert_eq!(frame_times.len(), 1);

    // Test multiple updates
    update_frame_stats(&mut app, &mut frame_times, 2000);
    assert_eq!(app.frame_id, 2);
    assert_eq!(app.avg_frame_time_us, 1500); // (1000 + 2000) / 2
    assert_eq!(app.max_frame_time_us, 2000);
    assert_eq!(frame_times.len(), 2);

    // Test rolling window (fill beyond 60)
    for i in 0..65 {
        update_frame_stats(&mut app, &mut frame_times, 100 + i as u64);
    }
    assert_eq!(frame_times.len(), 60); // Should cap at 60
}

#[test]
fn test_handle_key_input_quit() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Test 'q' quits
    let key_q = event::KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
    assert!(handle_key_input(&mut app, key_q));

    // Test Ctrl+C quits
    let key_ctrl_c = event::KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert!(handle_key_input(&mut app, key_ctrl_c));
}

#[test]
fn test_handle_key_input_other_keys() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);

    // Test other keys don't quit
    let key_a = event::KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
    assert!(!handle_key_input(&mut app, key_a));

    let key_enter = event::KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert!(!handle_key_input(&mut app, key_enter));

    let key_tab = event::KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
    assert!(!handle_key_input(&mut app, key_tab));
}

#[test]
fn test_create_tracer_handle_demo_mode() {
    let (tx, _rx) = mpsc::channel::<VisualizerEvent>();

    // Test demo mode (no command, no pid)
    let setup = create_tracer_handle(None, None, tx);
    assert!(setup.demo_mode);
    assert!(setup.handle.is_none());
}

#[test]
fn test_terminal_guard_drop() {
    // Test that TerminalGuard can be created and dropped without panic
    {
        let _guard = TerminalGuard;
    }
    // If we get here, drop succeeded
}

#[test]
#[allow(unsafe_code)]
fn test_suppress_stdio_safe() {
    // Test that suppress_stdio doesn't panic when /dev/null isn't available.
    //
    // CRITICAL: suppress_stdio() uses dup2 to redirect stdout/stderr to /dev/null
    // for the ENTIRE process. We MUST save and restore the original fds, otherwise
    // the test harness output for ALL concurrent tests is silently swallowed,
    // causing the CI to appear hung (no output = 30-minute timeout cancel).
    //
    // SAFETY: libc dup/dup2/close are safe on valid fds. We save originals,
    // call the function under test, then restore. Fds are process-global but
    // the save/restore window is brief enough for test purposes.
    unsafe {
        let saved_stdout = libc::dup(libc::STDOUT_FILENO);
        let saved_stderr = libc::dup(libc::STDERR_FILENO);
        assert!(saved_stdout >= 0, "failed to dup stdout");
        assert!(saved_stderr >= 0, "failed to dup stderr");

        suppress_stdio();

        libc::dup2(saved_stdout, libc::STDOUT_FILENO);
        libc::dup2(saved_stderr, libc::STDERR_FILENO);
        libc::close(saved_stdout);
        libc::close(saved_stderr);
    }
}

#[test]
fn test_handle_tracer_finish_none() {
    let config = VisualizeConfig::default();
    let mut app = app::VisualizeApp::new(config);
    let mut handle: Option<thread::JoinHandle<Result<i32>>> = None;

    // Should not panic when handle is None
    handle_tracer_finish(&mut app, &mut handle);
    assert!(!app.trace_complete); // Should not mark complete when handle is None
}

#[test]
fn test_open_tty_fallback() {
    // Test that open_tty returns a file (doesn't panic)
    // In CI environments it will use the fallback /proc/self/fd/1
    let _file = open_tty();
}
