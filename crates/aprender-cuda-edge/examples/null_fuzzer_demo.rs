//! Null Pointer Sentinel Fuzzer Demo
//!
//! Demonstrates how to use NonNullDevicePtr and injection strategies
//! for GPU memory safety testing.
//!
//! Run with: cargo run --example null_fuzzer_demo

use trueno_cuda_edge::null_fuzzer::{
    InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullFuzzerReport, NullSentinelFuzzer,
};

fn main() {
    println!("=== Null Pointer Sentinel Fuzzer Demo ===\n");

    // 1. NonNullDevicePtr Type Safety
    println!("1. NonNullDevicePtr Type Safety");
    println!("   ─────────────────────────────");

    // Valid device pointer
    match NonNullDevicePtr::<f32>::new(0x7f00_0000_0000) {
        Ok(ptr) => println!("   ✓ Valid pointer created: {}", ptr),
        Err(e) => println!("   ✗ Error: {}", e),
    }

    // Null pointer rejection
    match NonNullDevicePtr::<f32>::new(0) {
        Ok(_) => println!("   ✗ Should have rejected null!"),
        Err(e) => println!("   ✓ Null rejected: {}", e),
    }

    println!();

    // 2. Injection Strategies
    println!("2. Injection Strategies");
    println!("   ─────────────────────");

    // Periodic injection
    let periodic = InjectionStrategy::Periodic { interval: 10 };
    println!("   Periodic (interval=10):");
    for i in [0, 5, 10, 15, 20] {
        let inject = periodic.should_inject(i);
        println!("     Call {}: {}", i, if inject { "INJECT" } else { "normal" });
    }

    println!();

    // Probabilistic injection
    let prob = InjectionStrategy::Probabilistic { probability: 0.25 };
    println!("   Probabilistic (25%):");
    let inject_count = (0..100).filter(|&i| prob.should_inject(i)).count();
    println!("     Injections in 100 calls: {}", inject_count);

    println!();

    // 3. Fuzzer State Machine
    println!("3. Fuzzer State Machine");
    println!("   ─────────────────────");

    let config = NullFuzzerConfig {
        strategy: InjectionStrategy::Periodic { interval: 5 },
        total_calls: 20,
        fail_fast: false,
    };

    let mut fuzzer = NullSentinelFuzzer::new(config);

    println!("   Running 20 kernel calls with periodic injection (interval=5):");
    print!("   ");
    for i in 0..20 {
        let inject = fuzzer.next_call();
        print!("{}", if inject { "█" } else { "░" });
        if (i + 1) % 10 == 0 {
            print!(" ");
        }
    }
    println!();
    println!("   (█ = null injected, ░ = normal)");

    println!();

    // 4. Report Generation
    println!("4. Report Analysis");
    println!("   ────────────────");

    let report = NullFuzzerReport { total_calls: 100, injections: 10, caught: 8, crashes: 2 };

    println!("   Total calls:  {}", report.total_calls);
    println!("   Injections:   {}", report.injections);
    println!("   Caught:       {} ({:.0}%)", report.caught, report.catch_rate() * 100.0);
    println!("   Crashes:      {}", report.crashes);

    println!("\n=== Demo Complete ===");
}
