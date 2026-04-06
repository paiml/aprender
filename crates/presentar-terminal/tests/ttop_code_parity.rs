//! Unit tests against ttop's actual code.
//!
//! These tests verify that ptop's implementations match ttop's behavior
//! by directly testing against ttop's exported functions.
//!
//! Reference: SPEC-024 Section 11 - Visual Comparison Findings

/// Tests for ttop theme module parity
mod theme_parity {
    use trueno_viz::monitor::ratatui::style::Color;
    use ttop::theme::{format_bytes, format_uptime, percent_color};

    #[test]
    fn test_format_bytes_matches_ttop() {
        // Test ttop's format_bytes function
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1024), "1.0K");
        assert_eq!(format_bytes(1536), "1.5K"); // 1.5 * 1024
        assert_eq!(format_bytes(1024 * 1024), "1.0M");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0G");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 1024), "1.0T");
    }

    #[test]
    fn test_format_uptime_matches_ttop() {
        // Test ttop's format_uptime function
        assert_eq!(format_uptime(0.0), "0m");
        assert_eq!(format_uptime(60.0), "1m");
        assert_eq!(format_uptime(300.0), "5m");
        assert_eq!(format_uptime(3600.0), "1h 0m");
        assert_eq!(format_uptime(3700.0), "1h 1m");
        assert_eq!(format_uptime(86400.0), "1d 0h");
        assert_eq!(format_uptime(90000.0), "1d 1h");
        // ttop deterministic mode: 5d 3h 47m
        let five_days_3h_47m = 47.0f64.mul_add(60.0, 5.0f64.mul_add(86400.0, 3.0 * 3600.0));
        assert_eq!(format_uptime(five_days_3h_47m), "5d 3h");
    }

    #[test]
    fn test_percent_color_gradient_ttop() {
        // Test ttop's percent_color gradient at key breakpoints
        // Low (0-25): cyan-ish
        let low = percent_color(10.0);
        if let Color::Rgb(_r, g, b) = low {
            assert!(b > 150, "Low percent should have blue tint, got b={b}");
            assert!(g > 150, "Low percent should have green tint, got g={g}");
        }

        // Medium-low (25-50): green-yellow
        let med_low = percent_color(35.0);
        if let Color::Rgb(_r, g, _b) = med_low {
            assert!(g > 200, "Medium-low should have high green, got g={g}");
        }

        // Medium (50-75): yellow-orange
        let medium = percent_color(60.0);
        if let Color::Rgb(r, g, _b) = medium {
            assert!(r == 255, "Medium should have max red");
            assert!(g > 150, "Medium should have some green");
        }

        // High (75-90): orange-red
        let high = percent_color(80.0);
        if let Color::Rgb(r, g, _b) = high {
            assert!(r == 255, "High should have max red");
            assert!(g < 200, "High should reduce green");
        }

        // Critical (90+): bright red
        let critical = percent_color(95.0);
        if let Color::Rgb(r, g, b) = critical {
            assert_eq!(r, 255, "Critical should be max red");
            assert_eq!(
                g, 64,
                "Critical should have ttop's specific red (255, 64, 64)"
            );
            assert_eq!(
                b, 64,
                "Critical should have ttop's specific red (255, 64, 64)"
            );
        }
    }

    #[test]
    fn test_border_colors_match_ttop() {
        use ttop::theme::borders;

        // Verify ttop's exact border colors
        assert_eq!(
            borders::CPU,
            Color::Rgb(100, 200, 255),
            "CPU border: bright cyan"
        );
        assert_eq!(
            borders::MEMORY,
            Color::Rgb(180, 120, 255),
            "Memory border: purple"
        );
        assert_eq!(
            borders::DISK,
            Color::Rgb(100, 180, 255),
            "Disk border: blue"
        );
        assert_eq!(
            borders::NETWORK,
            Color::Rgb(255, 150, 100),
            "Network border: orange"
        );
        assert_eq!(
            borders::PROCESS,
            Color::Rgb(220, 180, 100),
            "Process border: gold"
        );
        assert_eq!(
            borders::GPU,
            Color::Rgb(100, 255, 150),
            "GPU border: bright green"
        );
        assert_eq!(
            borders::BATTERY,
            Color::Rgb(255, 220, 100),
            "Battery border: yellow"
        );
        assert_eq!(
            borders::SENSORS,
            Color::Rgb(255, 100, 150),
            "Sensors border: pink"
        );
        assert_eq!(
            borders::FILES,
            Color::Rgb(180, 140, 100),
            "Files border: warm brown/amber"
        );
    }

    #[test]
    fn test_graph_colors_match_ttop() {
        use ttop::theme::graph;

        // Verify ttop's exact graph colors
        assert_eq!(
            graph::CPU,
            Color::Rgb(100, 200, 255),
            "CPU graph: bright cyan"
        );
        assert_eq!(
            graph::MEMORY,
            Color::Rgb(180, 120, 255),
            "Memory graph: purple"
        );
        assert_eq!(graph::SWAP, Color::Rgb(255, 180, 100), "Swap graph: orange");
        assert_eq!(
            graph::NETWORK_RX,
            Color::Rgb(100, 200, 255),
            "Network RX: cyan"
        );
        assert_eq!(
            graph::NETWORK_TX,
            Color::Rgb(255, 100, 100),
            "Network TX: red"
        );
        assert_eq!(
            graph::GPU,
            Color::Rgb(100, 255, 150),
            "GPU graph: bright green"
        );
        assert_eq!(
            graph::DISK_READ,
            Color::Rgb(100, 180, 255),
            "Disk read: blue"
        );
        assert_eq!(
            graph::DISK_WRITE,
            Color::Rgb(255, 150, 100),
            "Disk write: orange"
        );
    }
}

/// Tests for ttop connections analyzer parity
mod connections_parity {
    use ttop::analyzers::{port_to_icon, port_to_service, ConnState};

    #[test]
    fn test_port_to_service_common_ports() {
        // Test ttop's port_to_service for common ports
        assert_eq!(port_to_service(22), Some("SSH"));
        assert_eq!(port_to_service(80), Some("HTTP"));
        assert_eq!(port_to_service(443), Some("HTTPS"));
        assert_eq!(port_to_service(3306), Some("MySQL"));
        assert_eq!(port_to_service(5432), Some("PgSQL"));
        assert_eq!(port_to_service(6379), Some("Redis"));
        assert_eq!(port_to_service(27017), Some("MongoDB"));
        assert_eq!(port_to_service(9090), Some("Prometheus"));
        assert_eq!(port_to_service(6443), Some("K8s"));
    }

    #[test]
    fn test_port_to_service_email_ports() {
        assert_eq!(port_to_service(25), Some("SMTP"));
        assert_eq!(port_to_service(465), Some("SMTPS"));
        assert_eq!(port_to_service(587), Some("Submit"));
        assert_eq!(port_to_service(993), Some("IMAPS"));
        assert_eq!(port_to_service(995), Some("POP3S"));
    }

    #[test]
    fn test_port_to_service_unknown() {
        assert_eq!(port_to_service(12345), None);
        assert_eq!(port_to_service(0), None);
        assert_eq!(port_to_service(65535), None);
    }

    #[test]
    fn test_port_to_icon_emoji() {
        // ttop uses emoji icons for services
        assert_eq!(port_to_icon(22), "\u{1F510}"); // lock
        assert_eq!(port_to_icon(80), "\u{1F310}"); // globe
        assert_eq!(port_to_icon(443), "\u{1F512}"); // locked
        assert_eq!(port_to_icon(3306), "\u{1F5C4}"); // file cabinet (database)
    }

    #[test]
    fn test_conn_state_as_char() {
        // ttop uses single-char representation for connection states
        assert_eq!(ConnState::Established.as_char(), 'E');
        assert_eq!(ConnState::Listen.as_char(), 'L');
        assert_eq!(ConnState::TimeWait.as_char(), 'T');
        assert_eq!(ConnState::CloseWait.as_char(), 'C');
        assert_eq!(ConnState::SynSent.as_char(), 'S');
        assert_eq!(ConnState::SynRecv.as_char(), 'R');
        assert_eq!(ConnState::FinWait1.as_char(), 'F');
        assert_eq!(ConnState::FinWait2.as_char(), 'F');
        assert_eq!(ConnState::Closing.as_char(), 'X');
        assert_eq!(ConnState::LastAck.as_char(), 'X');
        assert_eq!(ConnState::Close.as_char(), 'X');
        assert_eq!(ConnState::Unknown.as_char(), '?');
    }
}

/// Tests for ttop ring buffer parity
mod ring_buffer_parity {
    use ttop::RingBuffer;

    #[test]
    fn test_ring_buffer_basic_operations() {
        let mut rb: RingBuffer<f64> = RingBuffer::new(5);

        // Push values
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);

        assert_eq!(rb.len(), 3);
        // ttop uses `latest()` not `last()`
        assert_eq!(rb.latest(), Some(&3.0));
    }

    #[test]
    fn test_ring_buffer_wrapping() {
        let mut rb: RingBuffer<f64> = RingBuffer::new(3);

        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);
        rb.push(4.0); // Should wrap, removing 1.0

        assert_eq!(rb.len(), 3);

        // Get all values
        let values: Vec<f64> = rb.iter().copied().collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ring_buffer_mean() {
        let mut rb: RingBuffer<f64> = RingBuffer::new(10);

        rb.push(10.0);
        rb.push(20.0);
        rb.push(30.0);

        let mean = rb.mean();
        assert!(
            (mean - 20.0).abs() < 0.001,
            "Mean should be 20.0, got {mean}"
        );
    }
}

/// Tests for ttop PSI analyzer parity
mod psi_parity {
    use ttop::analyzers::PressureLevel;

    #[test]
    fn test_pressure_level_classification() {
        // ttop's pressure level thresholds
        // These are the thresholds from ttop's psi.rs

        // Low pressure: < 10%
        // Medium: 10-25%
        // High: 25-50%
        // Critical: > 50%

        // We just verify the enum exists and matches our expectations
        let _ = PressureLevel::Low;
        let _ = PressureLevel::Medium;
        let _ = PressureLevel::High;
        let _ = PressureLevel::Critical;
    }
}

/// Tests for ttop sensor health parity
mod sensor_parity {
    use ttop::analyzers::SensorType;

    #[test]
    fn test_sensor_types() {
        // Verify ttop supports these sensor types
        let _ = SensorType::Temperature;
        let _ = SensorType::Fan;
        let _ = SensorType::Voltage;
        let _ = SensorType::Current;
        let _ = SensorType::Power;
    }
}

/// Tests for ttop treemap parity
mod treemap_parity {
    use ttop::analyzers::FileCategory;

    #[test]
    fn test_file_categories() {
        // Verify ttop's file categories for treemap (ttop 0.3.x)
        // These are the actual categories from ttop's treemap.rs
        let _ = FileCategory::Model; // ML models (.gguf, .safetensors)
        let _ = FileCategory::Archive; // .tar, .zip, .zst
        let _ = FileCategory::Build; // target/, node_modules/
        let _ = FileCategory::Media; // video, audio, images
        let _ = FileCategory::Database; // .db, .sqlite
        let _ = FileCategory::Benchmark; // fio artifacts
        let _ = FileCategory::Other;
    }

    #[test]
    fn test_file_category_icons() {
        // Verify ttop's file category icons
        assert_eq!(FileCategory::Model.icon(), '🧠');
        assert_eq!(FileCategory::Archive.icon(), '📦');
        assert_eq!(FileCategory::Build.icon(), '🔨');
        assert_eq!(FileCategory::Media.icon(), '🎬');
        assert_eq!(FileCategory::Database.icon(), '💾');
        assert_eq!(FileCategory::Benchmark.icon(), '⏱');
        assert_eq!(FileCategory::Other.icon(), '📄');
    }

    #[test]
    fn test_file_category_colors() {
        // Verify ttop's file category RGB colors
        assert_eq!(FileCategory::Model.color(), (180, 100, 220)); // purple
        assert_eq!(FileCategory::Archive.color(), (220, 160, 80)); // orange
        assert_eq!(FileCategory::Build.color(), (120, 120, 130)); // gray
        assert_eq!(FileCategory::Media.color(), (100, 180, 220)); // cyan
        assert_eq!(FileCategory::Database.color(), (100, 140, 220)); // blue
        assert_eq!(FileCategory::Benchmark.color(), (80, 80, 90)); // dark gray
        assert_eq!(FileCategory::Other.color(), (160, 160, 160)); // light gray
    }
}

/// Tests for ttop GPU process analyzer parity
mod gpu_parity {
    use ttop::analyzers::GpuProcType;

    #[test]
    fn test_gpu_proc_types() {
        // Verify ttop's GPU process types (ttop 0.3.x)
        let _ = GpuProcType::Compute;
        let _ = GpuProcType::Graphics;
    }

    #[test]
    fn test_gpu_proc_type_display() {
        // ttop uses single-char display for proc types
        assert_eq!(format!("{}", GpuProcType::Compute), "C");
        assert_eq!(format!("{}", GpuProcType::Graphics), "G");
    }
}
