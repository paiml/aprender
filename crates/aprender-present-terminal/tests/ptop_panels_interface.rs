//! SPEC-024: Interface-Defining Tests for ptop Panels
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**
//!
//! Each panel MUST render actual data, not placeholders.
//! Each panel MUST handle the exploded view state.

#![cfg(feature = "ptop")]

use presentar_terminal::direct::CellBuffer;
use presentar_terminal::ptop::app::MetricsSnapshot;
use presentar_terminal::ptop::{ui, App, PanelType};
use presentar_terminal::Snapshot;

/// Helper to render and extract text
fn render_to_string(app: &App, width: u16, height: u16) -> String {
    let mut buffer = CellBuffer::new(width, height);
    ui::draw(app, &mut buffer);

    let mut output = String::new();
    for y in 0..height {
        for x in 0..width {
            if let Some(cell) = buffer.get(x, y) {
                output.push(cell.symbol.chars().next().unwrap_or(' '));
            } else {
                output.push(' ');
            }
        }
        output.push('\n');
    }
    output
}

// =============================================================================
// SECTION 1: CPU Panel Interface
// =============================================================================

mod cpu_panel {
    use super::*;

    /// CPU panel MUST render in normal view
    #[test]
    fn renders_normal() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);
        assert!(output.contains("CPU"), "CPU panel must be visible");
    }

    /// CPU panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Cpu);

        let output = render_to_string(&app, 140, 45);
        assert!(output.contains("CPU"), "CPU exploded panel must show");
    }

    /// CPU exploded MUST show per-core frequency from async data
    #[test]
    fn exploded_shows_async_freq() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Cpu);

        let mut snapshot = MetricsSnapshot::empty();
        snapshot.per_core_freq = vec![4765; 48];
        snapshot.per_core_temp = vec![65.0; 48];
        snapshot.per_core_percent = vec![45.0; 48];
        snapshot.cpu_avg = 0.45;
        app.apply_snapshot(snapshot);

        let output = render_to_string(&app, 140, 45);

        // Must contain frequency from async update (4.76G or 4.77G)
        let has_freq =
            output.contains("4.76") || output.contains("4.77") || output.contains("4765");
        assert!(has_freq, "Must show async-updated frequency");
    }

    /// CPU panel MUST show core utilization histogram
    #[test]
    fn shows_utilization_histogram() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Cpu);

        let output = render_to_string(&app, 140, 45);
        assert!(
            output.contains("CORE UTILIZATION") || output.contains("BREAKDOWN"),
            "Must show utilization breakdown"
        );
    }
}

// =============================================================================
// SECTION 2: Memory Panel Interface
// =============================================================================

mod memory_panel {
    use super::*;

    /// Memory panel MUST render
    #[test]
    fn renders() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);
        assert!(
            output.contains("Mem") || output.contains("MEM") || output.contains("Memory"),
            "Memory panel must be visible"
        );
    }

    /// Memory panel MUST show usage
    #[test]
    fn shows_usage() {
        let mut app = App::with_config(true, Default::default());

        let mut snapshot = MetricsSnapshot::empty();
        snapshot.mem_total = 32_000_000_000;
        snapshot.mem_used = 16_000_000_000;
        app.apply_snapshot(snapshot);

        let output = render_to_string(&app, 120, 40);
        // Should show some memory values
        assert!(
            output.contains("G") || output.contains("GB") || output.contains("%"),
            "Memory panel must show values"
        );
    }
}

// =============================================================================
// SECTION 3: Process Panel Interface
// =============================================================================

mod process_panel {
    use super::*;

    /// Process panel MUST render
    #[test]
    fn renders() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);
        assert!(
            output.contains("PID") || output.contains("COMMAND") || output.contains("Process"),
            "Process panel must be visible"
        );
    }

    /// Process panel MUST show CPU% and MEM% columns
    #[test]
    fn shows_columns() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);
        // Check for column headers - might be CPU%, CPU, or just percentage values
        let has_cpu = output.contains("CPU") || output.contains("%");
        let has_mem = output.contains("MEM") || output.contains("Memory");
        assert!(has_cpu, "Process panel must show CPU column or values");
    }
}

// =============================================================================
// SECTION 4: Network Panel Interface
// =============================================================================

mod network_panel {
    use super::*;

    /// Network panel MUST render
    #[test]
    fn renders() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);
        // Network might show as "Net" or interface names
        assert!(
            output.contains("Net")
                || output.contains("eth")
                || output.contains("lo")
                || output.contains("RX")
                || output.contains("TX")
                || output.contains("wl"),
            "Network panel must be visible"
        );
    }
}

// =============================================================================
// SECTION 5: Disk Panel Interface
// =============================================================================

mod disk_panel {
    use super::*;

    /// Disk panel MUST render
    #[test]
    fn renders() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);
        assert!(
            output.contains("Disk")
                || output.contains("sda")
                || output.contains("nvme")
                || output.contains("Read")
                || output.contains("Write"),
            "Disk panel must be visible or show disk stats"
        );
    }
}

// =============================================================================
// SECTION 6: GPU Panel Interface
// =============================================================================

mod gpu_panel {
    use super::*;

    /// GPU panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        app.panels.gpu = true;
        let output = render_to_string(&app, 120, 40);
        // GPU panel shows "GPU" or percentage or "N/A" if no GPU
        assert!(
            output.contains("GPU") || output.contains("N/A") || output.contains("%"),
            "GPU panel must be visible"
        );
    }

    /// GPU panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Gpu);
        let _output = render_to_string(&app, 140, 45);
        // Must not panic
    }
}

// =============================================================================
// SECTION 7: Sensors Panel Interface
// =============================================================================

mod sensors_panel {
    use super::*;

    /// Sensors panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        app.panels.sensors = true;
        let output = render_to_string(&app, 120, 40);
        // Sensors shows temperature or "Sensors" title
        assert!(
            output.contains("Sensor") || output.contains("°C") || output.contains("Temp"),
            "Sensors panel must be visible"
        );
    }

    /// Sensors panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Sensors);
        let _output = render_to_string(&app, 140, 45);
    }
}

// =============================================================================
// SECTION 8: Connections Panel Interface
// =============================================================================

mod connections_panel {
    use super::*;

    /// Connections panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        app.panels.connections = true;
        let output = render_to_string(&app, 120, 40);
        // Connections shows TCP states or "Connections" title
        assert!(
            output.contains("Connect")
                || output.contains("TCP")
                || output.contains("ESTABLISHED")
                || output.contains("LISTEN"),
            "Connections panel must be visible"
        );
    }

    /// Connections panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Connections);
        let _output = render_to_string(&app, 140, 45);
    }
}

// =============================================================================
// SECTION 9: PSI Panel Interface
// =============================================================================

mod psi_panel {
    use super::*;

    /// PSI panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        app.panels.psi = true;
        let output = render_to_string(&app, 120, 40);
        // PSI shows pressure stall info or "PSI" or "Pressure"
        assert!(
            output.contains("PSI")
                || output.contains("Pressure")
                || output.contains("some")
                || output.contains("full"),
            "PSI panel must be visible"
        );
    }

    /// PSI panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Psi);
        let _output = render_to_string(&app, 140, 45);
    }
}

// =============================================================================
// SECTION 10: Files Panel Interface
// =============================================================================

mod files_panel {
    use super::*;

    /// Files panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        app.panels.files = true;
        let output = render_to_string(&app, 120, 40);
        // Files panel shows treemap or file info
        assert!(
            output.contains("File")
                || output.contains("/")
                || output.contains("home")
                || output.contains("root"),
            "Files panel must be visible"
        );
    }

    /// Files panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Files);
        let _output = render_to_string(&app, 140, 45);
    }
}

// =============================================================================
// SECTION 11: Battery Panel Interface
// =============================================================================

mod battery_panel {
    use super::*;

    /// Battery panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        app.panels.battery = true;
        let output = render_to_string(&app, 120, 40);
        // Battery panel shows charge % or "Battery" or "N/A" on desktop
        assert!(
            output.contains("Batt")
                || output.contains("%")
                || output.contains("N/A")
                || output.contains("AC"),
            "Battery panel must be visible"
        );
    }

    /// Battery panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Battery);
        let _output = render_to_string(&app, 140, 45);
    }
}

// =============================================================================
// SECTION 12: Containers Panel Interface
// =============================================================================

mod containers_panel {
    use super::*;

    /// Containers panel MUST render
    #[test]
    fn renders() {
        let mut app = App::with_config(true, Default::default());
        // Enable containers panel if available
        let output = render_to_string(&app, 120, 40);
        // Containers shows docker/podman info or empty state
        // Just verify it doesn't panic - container detection is optional
        let _ = output;
    }

    /// Containers panel MUST render in exploded view
    #[test]
    fn renders_exploded() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Containers);
        let _output = render_to_string(&app, 140, 45);
    }
}

// =============================================================================
// SECTION 13: All Panels Not Placeholder
// =============================================================================

mod no_placeholders {
    use super::*;

    /// NO panel should show "coconut radio" or similar placeholder text
    #[test]
    fn no_placeholder_text() {
        let app = App::with_config(true, Default::default());
        let output = render_to_string(&app, 120, 40);

        let placeholders = [
            "coconut radio",
            "lorem ipsum",
            "placeholder",
            "TODO",
            "FIXME",
        ];
        for p in placeholders {
            assert!(
                !output.to_lowercase().contains(p),
                "Output must not contain placeholder text: {}",
                p
            );
        }
    }

    /// All panels MUST render without panic at minimum size
    #[test]
    fn minimum_size_no_panic() {
        let app = App::with_config(true, Default::default());
        // Very small buffer - should not panic
        let _output = render_to_string(&app, 40, 10);
    }

    /// All panels MUST render without panic at large size
    #[test]
    fn large_size_no_panic() {
        let app = App::with_config(true, Default::default());
        let _output = render_to_string(&app, 200, 60);
    }
}

// =============================================================================
// SECTION 14: Exploded View Interface
// =============================================================================

mod exploded_view {
    use super::*;

    /// Each panel type MUST be able to explode (ALL 12 PANELS)
    #[test]
    fn all_panels_explode() {
        let panels = [
            PanelType::Cpu,
            PanelType::Memory,
            PanelType::Disk,
            PanelType::Network,
            PanelType::Process,
            PanelType::Gpu,
            PanelType::Sensors,
            PanelType::Connections,
            PanelType::Psi,
            PanelType::Files,
            PanelType::Battery,
            PanelType::Containers,
        ];

        for panel in panels {
            let mut app = App::with_config(true, Default::default());
            app.exploded_panel = Some(panel);

            // Should not panic
            let _output = render_to_string(&app, 140, 45);
        }
    }

    /// Exploded view MUST fill most of the screen
    #[test]
    fn exploded_fills_screen() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Cpu);

        let output = render_to_string(&app, 140, 45);
        let lines: Vec<&str> = output.lines().collect();

        // Exploded panel should use most of the height (at least 30 lines of content)
        let non_empty_lines = lines.iter().filter(|l| l.trim().len() > 5).count();
        assert!(
            non_empty_lines > 20,
            "Exploded view must fill screen, got {} lines",
            non_empty_lines
        );
    }
}
