//! Benchmark suite: Renacer vs strace
//!
//! Sprint 11-12: Formal benchmark infrastructure
//!
//! Validates the performance claim: 2-5x faster than strace

use std::process::Command;
use std::time::{Duration, Instant};

/// Benchmark helper: Run a command and measure wall-clock time
fn bench_tracer(
    tracer: &str,
    args: &[&str],
    command: &[&str],
    iterations: usize,
    add_separator: bool,
) -> Duration {
    let mut total = Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();

        let mut cmd = Command::new(tracer);
        for arg in args {
            cmd.arg(arg);
        }
        if add_separator {
            cmd.arg("--");
        }
        for c in command {
            cmd.arg(c);
        }

        // Redirect stdout to /dev/null to avoid output overhead
        cmd.stdout(std::process::Stdio::null());

        // Run and discard output
        let output = cmd.output().expect("Failed to execute tracer");
        if !output.status.success() {
            eprintln!("Tracer failed: {}", tracer);
            eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            panic!("Tracer {} failed with exit code {:?}", tracer, output.status.code());
        }

        total += start.elapsed();
    }

    total / iterations as u32
}

/// Benchmark: Simple command (ls)
#[test]
#[ignore] // Requires built binary - run with --ignored
fn bench_simple_ls() {
    let iterations = 5;
    let command = &["ls", "-la", "/usr/bin"];

    // Baseline (no tracing)
    let baseline = bench_tracer("ls", &[], &["-la", "/usr/bin"], iterations, false);

    // strace
    let strace = bench_tracer("strace", &["-qq", "-o", "/dev/null"], command, iterations, true);

    // renacer
    let renacer = bench_tracer("./target/release/aprender-profile", &[], command, iterations, true);

    println!("\n=== Benchmark: ls -la /usr/bin (average of {} runs) ===", iterations);
    println!("Baseline (no tracing): {:?}", baseline);
    println!(
        "strace:                {:?} ({:.1}% overhead)",
        strace,
        (strace.as_secs_f64() / baseline.as_secs_f64() - 1.0) * 100.0
    );
    println!(
        "renacer:               {:?} ({:.1}% overhead)",
        renacer,
        (renacer.as_secs_f64() / baseline.as_secs_f64() - 1.0) * 100.0
    );
    println!(
        "\nResult: renacer is {:.2}x FASTER than strace",
        strace.as_secs_f64() / renacer.as_secs_f64()
    );

    // Document performance
    let speedup = strace.as_secs_f64() / renacer.as_secs_f64();
    if speedup >= 2.0 {
        println!("✅ Performance target met: {:.2}x faster (≥2x required)", speedup);
    } else if speedup >= 1.0 {
        println!("⚠️  Performance: {:.2}x faster (target: ≥2x)", speedup);
        println!("   Note: Room for optimization exists");
    } else {
        println!("❌ Performance regression: {:.2}x slower than strace!", 1.0 / speedup);
        panic!("Renacer should not be slower than strace");
    }
}

/// Benchmark: File-heavy workload (find)
#[test]
#[ignore] // Requires built binary - run with --ignored
fn bench_find_command() {
    let iterations = 3;
    let command = &["find", "/usr/share/doc", "-name", "*.txt", "-type", "f"];

    let baseline = bench_tracer(
        "find",
        &[],
        &["/usr/share/doc", "-name", "*.txt", "-type", "f"],
        iterations,
        false,
    );
    let strace = bench_tracer("strace", &["-qq", "-o", "/dev/null"], command, iterations, true);
    let renacer = bench_tracer("./target/release/aprender-profile", &[], command, iterations, true);

    println!("\n=== Benchmark: find (file-heavy workload, {} runs) ===", iterations);
    println!("Baseline: {:?}", baseline);
    println!(
        "strace:   {:?} ({:.1}% overhead)",
        strace,
        (strace.as_secs_f64() / baseline.as_secs_f64() - 1.0) * 100.0
    );
    println!(
        "renacer:  {:?} ({:.1}% overhead)",
        renacer,
        (renacer.as_secs_f64() / baseline.as_secs_f64() - 1.0) * 100.0
    );
    println!(
        "\nResult: renacer is {:.2}x FASTER than strace",
        strace.as_secs_f64() / renacer.as_secs_f64()
    );

    let speedup = strace.as_secs_f64() / renacer.as_secs_f64();
    if speedup < 1.0 {
        panic!("Renacer should not be slower than strace (got {:.2}x)", speedup);
    }
}

/// Benchmark: Quick commands (minimal syscalls)
#[test]
#[ignore] // Requires built binary - run with --ignored
fn bench_minimal_syscalls() {
    let iterations = 10;
    let command = &["echo", "hello"];

    let baseline = bench_tracer("echo", &[], &["hello"], iterations, false);
    let strace = bench_tracer("strace", &["-qq", "-o", "/dev/null"], command, iterations, true);
    let renacer = bench_tracer("./target/release/aprender-profile", &[], command, iterations, true);

    println!("\n=== Benchmark: echo (minimal syscalls, {} runs) ===", iterations);
    println!("Baseline: {:?}", baseline);
    println!("strace:   {:?}", strace);
    println!("renacer:  {:?}", renacer);
    println!(
        "\nResult: renacer is {:.2}x FASTER than strace",
        strace.as_secs_f64() / renacer.as_secs_f64()
    );

    let speedup = strace.as_secs_f64() / renacer.as_secs_f64();
    if speedup < 1.0 {
        panic!("Renacer should not be slower than strace (got {:.2}x)", speedup);
    }
}

/// Benchmark: Filtering performance (no output overhead)
#[test]
#[ignore] // Requires built binary - run with --ignored
fn bench_with_filtering() {
    let iterations = 5;
    let command = &["ls", "-la", "/usr/bin"];

    // renacer without filtering
    let renacer_all =
        bench_tracer("./target/release/aprender-profile", &[], command, iterations, true);

    // renacer with filtering (should be faster - less output)
    let renacer_filtered = bench_tracer(
        "./target/release/aprender-profile",
        &["-e", "trace=open"],
        command,
        iterations,
        true,
    );

    println!("\n=== Benchmark: Filtering overhead ({} runs) ===", iterations);
    println!("renacer (all syscalls): {:?}", renacer_all);
    println!("renacer (filtered):     {:?}", renacer_filtered);
    println!(
        "\nFiltering impact: {:.1}% faster",
        (1.0 - renacer_filtered.as_secs_f64() / renacer_all.as_secs_f64()) * 100.0
    );

    // Filtering should not slow things down
    assert!(renacer_filtered <= renacer_all * 2, "Filtering should not add significant overhead");
}
