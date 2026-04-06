//! Test display rules capabilities detection
//!
//! Run with: cargo run -p presentar-terminal --features ptop --example `test_display_rules`

use presentar_terminal::widgets::{
    BatteryDisplayRules, DataAvailability, DisplayContext, DisplayRules, DisplayTerminalSize,
    GpuDisplayRules, PsiDisplayRules, SensorsDisplayRules, SystemCapabilities,
};

fn main() {
    println!("=== F-2000: Falsification Protocol - Display Rules ===\n");

    // Detect system capabilities
    let caps = SystemCapabilities::detect();
    println!("SystemCapabilities (detected at startup):");
    println!("  has_nvidia:        {}", caps.has_nvidia);
    println!("  has_amd:           {}", caps.has_amd);
    println!("  has_apple_silicon: {}", caps.has_apple_silicon);
    println!("  has_psi:           {}", caps.has_psi);
    println!("  has_sensors:       {}", caps.has_sensors);
    println!("  has_battery:       {}", caps.has_battery);
    println!("  in_container:      {}", caps.in_container);
    println!();

    // Test with empty data availability (simulates startup)
    let empty_data = DataAvailability::default();
    let ctx = DisplayContext {
        system: &caps,
        terminal: DisplayTerminalSize {
            width: 120,
            height: 40,
        },
        data: empty_data,
    };

    println!("Display Rules Evaluation (empty data):");
    println!("  PSI:     {:?}", PsiDisplayRules.evaluate(&ctx));
    println!("  Sensors: {:?}", SensorsDisplayRules.evaluate(&ctx));
    println!("  GPU:     {:?}", GpuDisplayRules.evaluate(&ctx));
    println!("  Battery: {:?}", BatteryDisplayRules.evaluate(&ctx));
    println!();

    // Test F-2001: Sensor boundary conditions
    println!("=== F-2001: Sensor Boundary Test ===");
    for count in [0, 1, 2, 3, 4, 5] {
        let data = DataAvailability {
            sensors_available: count > 0,
            sensor_count: count,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &caps,
            terminal: DisplayTerminalSize {
                width: 120,
                height: 40,
            },
            data,
        };
        let action = SensorsDisplayRules.evaluate(&ctx);
        println!("  {count} sensors -> {action:?}");
    }
    println!();

    // Test F-2003: GPU with capability but no data
    println!("=== F-2003: Zombie GPU Test ===");
    let caps_with_gpu = SystemCapabilities {
        has_nvidia: true,
        ..caps.clone()
    };

    // GPU capability but no data
    let no_gpu_data = DataAvailability {
        gpu_available: false,
        ..Default::default()
    };
    let ctx = DisplayContext {
        system: &caps_with_gpu,
        terminal: DisplayTerminalSize {
            width: 120,
            height: 40,
        },
        data: no_gpu_data,
    };
    println!(
        "  Capability=true, Data=false -> {:?}",
        GpuDisplayRules.evaluate(&ctx)
    );

    // GPU capability and data
    let with_gpu_data = DataAvailability {
        gpu_available: true,
        ..Default::default()
    };
    let ctx = DisplayContext {
        system: &caps_with_gpu,
        terminal: DisplayTerminalSize {
            width: 120,
            height: 40,
        },
        data: with_gpu_data,
    };
    println!(
        "  Capability=true, Data=true  -> {:?}",
        GpuDisplayRules.evaluate(&ctx)
    );
    println!();

    // Summary
    println!("=== Falsification Summary ===");
    println!(
        "Battery panel on desktop: {:?}",
        BatteryDisplayRules.evaluate(&DisplayContext {
            system: &caps,
            terminal: DisplayTerminalSize {
                width: 120,
                height: 40
            },
            data: DataAvailability::default(),
        })
    );

    if caps.has_battery {
        println!("UNEXPECTED: Battery detected on desktop system");
    } else {
        println!("PASS: Battery correctly detected as absent");
    }
}
