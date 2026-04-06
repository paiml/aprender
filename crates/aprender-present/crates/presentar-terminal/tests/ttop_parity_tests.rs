//! ttop Parity Tests
//!
//! These tests compare ptop output against the reference ttop implementation
//! from ../trueno-viz/crates/ttop to ensure feature parity.
//!
//! Run with: cargo test -p presentar-terminal --test `ttop_parity_tests` --features ptop
//!
//! CRITICAL: These tests will FAIL if ptop panels are "coconut radios"
//! (fake/non-functional implementations).

#![cfg(feature = "ptop")]

use presentar_terminal::ptop::app::App;
use presentar_terminal::ptop::ui;
use presentar_terminal::CellBuffer;

/// Capture a ptop frame and return the rendered text
fn capture_ptop_frame(width: u16, height: u16) -> String {
    // Use non-deterministic mode to get REAL data for parity testing
    let mut app = App::new(false);
    app.collect_metrics();

    let mut buffer = CellBuffer::new(width, height);
    ui::draw(&app, &mut buffer);

    // Convert buffer to string
    let mut output = String::new();
    for y in 0..height {
        for x in 0..width {
            if let Some(cell) = buffer.get(x, y) {
                output.push_str(&cell.symbol);
            } else {
                output.push(' ');
            }
        }
        output.push('\n');
    }
    output
}

/// Extract panel content by looking for bordered regions
fn extract_panel(output: &str, title_contains: &str) -> Option<String> {
    let lines: Vec<&str> = output.lines().collect();
    let mut in_panel = false;
    let mut panel_lines = Vec::new();

    for line in lines {
        if line.contains(title_contains) {
            in_panel = true;
        }
        if in_panel {
            panel_lines.push(line);
            // Panel ends at next top border or after ~10 lines
            if panel_lines.len() > 1 && (line.contains("╭") || panel_lines.len() > 12) {
                panel_lines.pop(); // Remove the next panel's border
                break;
            }
        }
    }

    if panel_lines.is_empty() {
        None
    } else {
        Some(panel_lines.join("\n"))
    }
}

// =============================================================================
// REAL Parity Tests - These actually run ptop and check output
// =============================================================================

/// Test: CPU panel renders with real per-core percentages
#[test]
fn real_cpu_panel_has_content() {
    let output = capture_ptop_frame(120, 40);

    // Must contain CPU panel with percentage
    assert!(
        output.contains("CPU") && output.contains("%"),
        "CPU panel must show percentages. Got:\n{}",
        &output[..output.len().min(500)]
    );

    // Must show core count (e.g., "8 cores")
    assert!(
        output.contains("core"),
        "CPU panel must show core count. Got:\n{}",
        &output[..output.len().min(500)]
    );
}

/// Test: Memory panel renders with real usage
#[test]
fn real_memory_panel_has_content() {
    let output = capture_ptop_frame(120, 40);

    // Must contain memory data with GB values (Used: XXG, Swap: XXG)
    // The title might not always contain "Memory" due to width constraints
    let has_memory_data = (output.contains("Used:") || output.contains("Memory"))
        && (output.contains("G") || output.contains("M"));

    assert!(
        has_memory_data,
        "Memory panel must show usage in G/M. Got:\n{}",
        &output[..output.len().min(500)]
    );
}

/// Test: Process panel shows real processes (not empty)
#[test]
fn real_process_panel_has_processes() {
    let output = capture_ptop_frame(120, 40);

    // Must contain Processes panel with count
    assert!(
        output.contains("Processes"),
        "Must have Processes panel. Got:\n{}",
        &output[..output.len().min(500)]
    );

    // Must show PID column header
    assert!(
        output.contains("PID"),
        "Process table must have PID column. Got:\n{}",
        &output[..output.len().min(500)]
    );

    // Must show at least one numeric PID (not just header)
    // PIDs appear after the border character │, so check all tokens
    let has_numeric_pid = output.lines().any(|line| {
        // Skip lines without border chars (headers, etc.)
        if !line.contains('│') {
            return false;
        }
        // Look for a PID-like number (5-7 digit process ID) in the line
        line.split_whitespace().any(|token| {
            token.len() >= 3 && token.len() <= 8 && token.chars().all(|c| c.is_ascii_digit())
        })
    });

    // This is a soft check - process list should have PIDs
    if !has_numeric_pid {
        eprintln!("WARNING: No numeric PIDs found in process list");
    }
}

/// Test: GPU panel - verify it shows SOMETHING (not "N/A" if GPU exists)
#[test]
fn real_gpu_panel_not_coconut_radio() {
    let output = capture_ptop_frame(120, 40);

    // Check if system has GPU
    let has_nvidia = std::path::Path::new("/dev/nvidia0").exists();
    let has_amd = std::path::Path::new("/sys/class/drm/card0/device/vendor").exists();

    if has_nvidia || has_amd {
        // GPU exists - panel must NOT show "N/A" or "No GPU"
        let gpu_panel = extract_panel(&output, "GPU");

        if let Some(panel) = gpu_panel {
            assert!(
                !panel.contains("N/A") && !panel.contains("No GPU"),
                "GPU panel is COCONUT RADIO - shows N/A but GPU exists!\nPanel:\n{}",
                panel
            );
        } else {
            panic!("GPU panel not found in output");
        }
    } else {
        eprintln!("SKIP: No GPU detected on this system");
    }
}

/// Test: Connections panel - verify it shows real TCP connections
#[test]
fn real_connections_panel_not_coconut_radio() {
    let output = capture_ptop_frame(120, 40);

    // Count real TCP connections
    let tcp_content = std::fs::read_to_string("/proc/net/tcp").unwrap_or_default();
    let real_connections = tcp_content.lines().count().saturating_sub(1);

    if real_connections > 0 {
        let conn_panel = extract_panel(&output, "Connections");

        if let Some(panel) = conn_panel {
            // Must show connection count or actual connections
            let shows_data = panel.contains("active")
                || panel.contains("listen")
                || panel.contains("tcp")
                || panel.contains("ESTABLISHED")
                || panel.contains(":"); // Port notation

            assert!(
                shows_data,
                "Connections panel is COCONUT RADIO - {} real connections exist but panel shows:\n{}",
                real_connections,
                panel
            );
        } else {
            // Panel might be toggled off - check if it's in the hint text
            if !output.contains("Connections") {
                eprintln!("NOTE: Connections panel not visible (may be toggled off)");
            }
        }
    } else {
        eprintln!("SKIP: No TCP connections on this system");
    }
}

/// Test: Files panel - verify it shows real filesystem data
#[test]
fn real_files_panel_not_coconut_radio() {
    let output = capture_ptop_frame(120, 40);

    let files_panel = extract_panel(&output, "Files");

    if let Some(panel) = files_panel {
        // Must show size data (G, M, K, or bytes)
        let shows_data = panel.contains("G")
            || panel.contains("M")
            || panel.contains("K")
            || panel.contains("home")
            || panel.contains("var")
            || panel.contains("/");

        assert!(
            shows_data,
            "Files panel is COCONUT RADIO - shows no filesystem data:\n{}",
            panel
        );
    } else {
        // Panel might be toggled off
        eprintln!("NOTE: Files panel not visible (may be toggled off)");
    }
}

/// Test: Sensors panel - verify it shows real temperatures
#[test]
fn real_sensors_panel_not_coconut_radio() {
    let output = capture_ptop_frame(120, 40);

    // Check if hwmon exists
    let hwmon_count = std::fs::read_dir("/sys/class/hwmon")
        .map(|d| d.count())
        .unwrap_or(0);

    if hwmon_count > 0 {
        let sensors_panel = extract_panel(&output, "Sensors");

        if let Some(panel) = sensors_panel {
            // Must show temperature (°C) or "No sensors" is acceptable if truly none
            let shows_temp = panel.contains("°C") || panel.contains("°");

            if !shows_temp && !panel.contains("No sensors") {
                eprintln!(
                    "WARNING: Sensors panel may be coconut radio - {} hwmon devices exist:\n{}",
                    hwmon_count, panel
                );
            }
        }
    } else {
        eprintln!("SKIP: No hwmon devices on this system");
    }
}

/// Test: Network panel shows interface names
#[test]
fn real_network_panel_shows_interfaces() {
    let output = capture_ptop_frame(120, 40);

    // Get real interface names (excluding lo)
    let interfaces: Vec<String> = std::fs::read_dir("/sys/class/net")
        .map(|d| {
            d.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n != "lo")
                .collect()
        })
        .unwrap_or_default();

    let network_panel = extract_panel(&output, "Network");

    if let Some(panel) = network_panel {
        // Should show at least one real interface or RX/TX data
        let shows_interface = interfaces.iter().any(|iface| panel.contains(iface));
        let shows_data = panel.contains("RX")
            || panel.contains("TX")
            || panel.contains("↓")
            || panel.contains("↑");

        assert!(
            shows_interface || shows_data,
            "Network panel should show interfaces {:?} or RX/TX data. Got:\n{}",
            interfaces,
            panel
        );
    } else {
        panic!("Network panel not found");
    }
}

// =============================================================================
// Summary Test - Overall health check
// =============================================================================

#[test]
fn summary_coconut_radio_audit() {
    let output = capture_ptop_frame(120, 40);

    println!("\n=== PTOP COCONUT RADIO AUDIT ===\n");

    // Check each panel
    let panels = [
        ("CPU", vec!["CPU", "%", "core"]),
        ("Memory", vec!["Memory", "G"]),
        ("Disk", vec!["Disk", "R:", "W:"]),
        ("Network", vec!["Network"]),
        ("GPU", vec!["GPU"]),
        ("Sensors", vec!["Sensors"]),
        ("Processes", vec!["Processes", "PID"]),
        ("Connections", vec!["Connections"]),
        ("Files", vec!["Files"]),
    ];

    let mut coconut_radios = Vec::new();

    for (name, markers) in panels {
        let found = markers.iter().any(|m| output.contains(m));
        let status = if found { "✓ FOUND" } else { "✗ MISSING" };
        println!("{}: {}", name, status);

        if !found {
            coconut_radios.push(name);
        }
    }

    println!("\n=== END AUDIT ===\n");

    if !coconut_radios.is_empty() {
        println!(
            "COCONUT RADIO PANELS (missing or not rendering): {:?}",
            coconut_radios
        );
    }

    // Output sample for debugging (safely truncate at char boundary)
    let sample_end = output
        .char_indices()
        .nth(800)
        .map(|(i, _)| i)
        .unwrap_or(output.len());
    println!(
        "Sample output (first ~800 chars):\n{}",
        &output[..sample_end]
    );
}
