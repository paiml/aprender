//! Test that DEFINES the interface for CPU exploded panel async updates.
//!
//! SPEC-024 Section 12: "Tests define interface"
//!
//! This test specifies that the CPU exploded panel MUST receive async updates
//! for per-core frequency and temperature data. The implementation MUST satisfy
//! this interface.
//!
//! # The Bug This Test Catches
//!
//! Previously, `draw_cpu_exploded` read from `app.system.cpus()` which was
//! never updated by `apply_snapshot()`. The async architecture moves `System`
//! to the background thread, but `app.system` in the main thread stays stale.
//!
//! # The Fix This Test Enforces
//!
//! 1. `MetricsSnapshot` MUST include `per_core_freq` and `per_core_temp`
//! 2. `App` MUST have `per_core_freq` and `per_core_temp` fields
//! 3. `apply_snapshot()` MUST transfer freq/temp data to App
//! 4. `draw_cpu_exploded` MUST read from App fields, not app.system

#![cfg(feature = "ptop")]

use presentar_terminal::ptop::{app::MetricsSnapshot, App};
use presentar_terminal::Snapshot; // For MetricsSnapshot::empty()

/// **INTERFACE-DEFINING TEST**
///
/// This test WILL FAIL until MetricsSnapshot includes per_core_freq.
/// That's intentional - the test defines what the interface MUST provide.
#[test]
fn test_metrics_snapshot_includes_per_core_freq() {
    // Create a snapshot
    let snapshot = MetricsSnapshot::empty();

    // ASSERTION: MetricsSnapshot MUST have per_core_freq field
    // If this doesn't compile, you need to add the field
    let _freq: &Vec<u64> = &snapshot.per_core_freq;

    // Frequency data should be present (can be zeros for empty snapshot)
    assert!(
        snapshot.per_core_freq.is_empty() || snapshot.per_core_freq.iter().all(|&f| f == 0),
        "Empty snapshot should have empty or zero frequency data"
    );
}

/// **INTERFACE-DEFINING TEST**
///
/// This test WILL FAIL until MetricsSnapshot includes per_core_temp.
#[test]
fn test_metrics_snapshot_includes_per_core_temp() {
    let snapshot = MetricsSnapshot::empty();

    // ASSERTION: MetricsSnapshot MUST have per_core_temp field
    let _temp: &Vec<f32> = &snapshot.per_core_temp;

    assert!(
        snapshot.per_core_temp.is_empty() || snapshot.per_core_temp.iter().all(|&t| t == 0.0),
        "Empty snapshot should have empty or zero temperature data"
    );
}

/// **INTERFACE-DEFINING TEST**
///
/// This test WILL FAIL until App has per_core_freq field.
#[test]
fn test_app_has_per_core_freq_field() {
    let app = App::with_config(true, Default::default()); // deterministic mode

    // ASSERTION: App MUST have per_core_freq field
    let freq: &Vec<u64> = &app.per_core_freq;

    // In deterministic mode with 48 cores, should have 48 entries
    assert_eq!(freq.len(), 48, "Deterministic mode should have 48 cores");
}

/// **INTERFACE-DEFINING TEST**
///
/// This test WILL FAIL until App has per_core_temp field.
#[test]
fn test_app_has_per_core_temp_field() {
    let app = App::with_config(true, Default::default());

    // ASSERTION: App MUST have per_core_temp field
    let temp: &Vec<f32> = &app.per_core_temp;

    assert_eq!(temp.len(), 48, "Deterministic mode should have 48 cores");
}

/// **INTERFACE-DEFINING TEST**
///
/// This test verifies that apply_snapshot transfers freq/temp data.
#[test]
fn test_apply_snapshot_updates_freq_temp() {
    let mut app = App::with_config(true, Default::default());

    // Create a snapshot with specific freq/temp values
    let mut snapshot = MetricsSnapshot::empty();
    snapshot.per_core_freq = vec![4500; 48]; // 4.5 GHz
    snapshot.per_core_temp = vec![65.0; 48]; // 65°C

    // Apply snapshot
    app.apply_snapshot(snapshot);

    // ASSERTION: App fields MUST be updated
    assert_eq!(
        app.per_core_freq[0], 4500,
        "apply_snapshot must transfer frequency data"
    );
    assert!(
        (app.per_core_temp[0] - 65.0).abs() < 0.1,
        "apply_snapshot must transfer temperature data"
    );
}

/// **INTERFACE-DEFINING TEST**
///
/// This test verifies that freq/temp data changes between async updates.
#[test]
fn test_freq_temp_changes_with_async_updates() {
    let mut app = App::with_config(true, Default::default());

    // Initial snapshot
    let mut snapshot1 = MetricsSnapshot::empty();
    snapshot1.per_core_freq = vec![4500; 48];
    snapshot1.per_core_temp = vec![65.0; 48];
    app.apply_snapshot(snapshot1);

    let freq1 = app.per_core_freq[0];
    let temp1 = app.per_core_temp[0];

    // Second snapshot with different values (simulating async update)
    let mut snapshot2 = MetricsSnapshot::empty();
    snapshot2.per_core_freq = vec![4800; 48]; // Boost to 4.8 GHz
    snapshot2.per_core_temp = vec![72.0; 48]; // Warmer
    app.apply_snapshot(snapshot2);

    let freq2 = app.per_core_freq[0];
    let temp2 = app.per_core_temp[0];

    // ASSERTION: Values MUST change between updates
    assert_ne!(freq1, freq2, "Frequency must update with new snapshot");
    assert!(
        (temp1 - temp2).abs() > 0.1,
        "Temperature must update with new snapshot"
    );
}

/// Test that MetricsCollector produces actual frequency data.
#[test]
fn test_collector_produces_frequency_data() {
    use presentar_terminal::ptop::app::MetricsCollector;
    use presentar_terminal::AsyncCollector;

    // Create non-deterministic collector (real data)
    let mut collector = MetricsCollector::new(false);

    // Collect metrics
    let snapshot = collector.collect();

    // On any modern Linux system, we should have freq data
    // (unless running in deterministic mode or on exotic hardware)
    if !snapshot.per_core_freq.is_empty() {
        // At least one core should have non-zero frequency
        let has_freq = snapshot.per_core_freq.iter().any(|&f| f > 0);
        assert!(
            has_freq,
            "Collector should produce non-zero frequency data. Got: {:?}",
            &snapshot.per_core_freq[..snapshot.per_core_freq.len().min(4)]
        );
    }
}

/// Integration test: Verify full render path uses updated data.
///
/// This test ensures that the UI rendering actually uses the async-updated
/// freq/temp data, not stale data from initialization.
#[test]
fn test_render_uses_async_updated_data() {
    use presentar_terminal::direct::CellBuffer;
    use presentar_terminal::ptop::ui;

    let mut app = App::with_config(true, Default::default());
    app.exploded_panel = Some(presentar_terminal::ptop::PanelType::Cpu);

    // Apply snapshot with specific freq value
    let mut snapshot = MetricsSnapshot::empty();
    snapshot.per_core_freq = vec![4765; 48]; // 4.765 GHz - unique value
    snapshot.per_core_temp = vec![71.5; 48];
    snapshot.per_core_percent = vec![45.0; 48];
    snapshot.cpu_avg = 0.45;
    app.apply_snapshot(snapshot);

    // Render
    let mut buffer = CellBuffer::new(140, 45);
    ui::draw(&app, &mut buffer);

    // Convert buffer to string for searching
    let mut output = String::new();
    for y in 0..45 {
        for x in 0..140 {
            if let Some(cell) = buffer.get(x, y) {
                output.push(cell.symbol.chars().next().unwrap_or(' '));
            }
        }
        output.push('\n');
    }

    // ASSERTION: The rendered output must contain the frequency we set
    // The exact format depends on format_freq_mhz(), but 4765 MHz = 4.76G or 4.77G
    let has_freq = output.contains("4.76") || output.contains("4.77") || output.contains("4765");

    assert!(
        has_freq,
        "Rendered output must contain async-updated frequency (4.76G/4.77G/4765)\n\
         This means draw_cpu_exploded is NOT using app.per_core_freq\n\
         First 20 lines of output:\n{}",
        output.lines().take(20).collect::<Vec<_>>().join("\n")
    );
}
