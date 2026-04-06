//! Direct Backend Benchmark
//!
//! Benchmarks the performance of the direct terminal backend.
//!
//! Run with: cargo run -p presentar-terminal --example `direct_benchmark` --release

use presentar_core::Color;
use presentar_terminal::direct::{CellBuffer, DiffRenderer, Modifiers};
use std::time::Instant;

fn main() {
    println!("=== Direct Backend Benchmark ===\n");
    println!(
        "Running in {} mode\n",
        if cfg!(debug_assertions) {
            "DEBUG"
        } else {
            "RELEASE"
        }
    );

    // Benchmark different terminal sizes
    let sizes = [
        (80, 24, "Standard (80x24)"),
        (120, 40, "Large (120x40)"),
        (200, 50, "Very Large (200x50)"),
    ];

    for (width, height, name) in sizes {
        println!("--- {name} ---");
        benchmark_size(width, height);
        println!();
    }

    // Benchmark differential updates
    println!("--- Differential Update Benchmark ---");
    benchmark_differential();
}

fn benchmark_size(width: u16, height: u16) {
    let mut buffer = CellBuffer::new(width, height);
    let mut renderer = DiffRenderer::new();
    let iterations = 100;

    // Warm up
    for _ in 0..10 {
        buffer.mark_all_dirty();
        let mut output = Vec::with_capacity(32768);
        let _ = renderer.flush(&mut buffer, &mut output);
    }

    // Full redraw benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        buffer.mark_all_dirty();
        let mut output = Vec::with_capacity(32768);
        let _ = renderer.flush(&mut buffer, &mut output);
    }
    let elapsed = start.elapsed();
    let avg = elapsed / iterations;

    let cells = width as usize * height as usize;
    println!(
        "Full redraw:  {:>6.2}ms avg ({} cells, {:.0} cells/ms)",
        avg.as_secs_f64() * 1000.0,
        cells,
        cells as f64 / avg.as_secs_f64() / 1000.0
    );

    // Cell update benchmark (measure allocation overhead)
    let start = Instant::now();
    for i in 0..iterations {
        for y in 0..height {
            for x in 0..width {
                buffer.update(
                    x,
                    y,
                    if (x + y + i as u16) % 2 == 0 {
                        "█"
                    } else {
                        " "
                    },
                    Color::WHITE,
                    Color::BLACK,
                    Modifiers::NONE,
                );
            }
        }
    }
    let elapsed = start.elapsed();
    let total_updates = iterations as usize * cells;
    let updates_per_sec = total_updates as f64 / elapsed.as_secs_f64();

    println!(
        "Cell updates: {:.0}M updates/sec ({} total in {:?})",
        updates_per_sec / 1_000_000.0,
        total_updates,
        elapsed
    );
}

fn benchmark_differential() {
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();
    let iterations = 1000;

    // Initial full render
    buffer.mark_all_dirty();
    let mut output = Vec::with_capacity(8192);
    let _ = renderer.flush(&mut buffer, &mut output);

    // Benchmark partial updates (10% of cells)
    let cells_to_update = 192; // ~10% of 1920

    let start = Instant::now();
    for i in 0..iterations {
        for j in 0..cells_to_update {
            let x = ((i + j) % 80) as u16;
            let y = ((i + j) / 80 % 24) as u16;
            buffer.update(x, y, "X", Color::RED, Color::BLACK, Modifiers::NONE);
        }
        output.clear();
        let _ = renderer.flush(&mut buffer, &mut output);
    }
    let elapsed = start.elapsed();
    let avg = elapsed / iterations;

    println!(
        "10% update:   {:>6.2}μs avg ({} cells changed)",
        avg.as_secs_f64() * 1_000_000.0,
        cells_to_update
    );

    // Benchmark very sparse updates (1% of cells)
    let cells_to_update = 19; // ~1% of 1920

    let start = Instant::now();
    for i in 0..iterations {
        for j in 0..cells_to_update {
            let x = ((i * 7 + j * 13) % 80) as u16;
            let y = ((i * 3 + j * 11) / 80 % 24) as u16;
            buffer.update(x, y, "•", Color::GREEN, Color::BLACK, Modifiers::NONE);
        }
        output.clear();
        let _ = renderer.flush(&mut buffer, &mut output);
    }
    let elapsed = start.elapsed();
    let avg = elapsed / iterations;

    println!(
        "1% update:    {:>6.2}μs avg ({} cells changed)",
        avg.as_secs_f64() * 1_000_000.0,
        cells_to_update
    );

    // Benchmark cursor optimization
    println!();
    println!("Cursor optimization test:");

    // Scattered cells (many cursor moves)
    buffer.clear_dirty();
    for i in 0..10 {
        buffer.update(
            i * 7 % 80,
            i * 3 % 24,
            "S",
            Color::BLUE,
            Color::BLACK,
            Modifiers::NONE,
        );
    }
    output.clear();
    let _ = renderer.flush(&mut buffer, &mut output);
    println!(
        "  Scattered (10 cells): {} cursor moves",
        renderer.cursor_moves()
    );

    // Consecutive cells (minimal cursor moves)
    buffer.clear_dirty();
    for i in 0..10 {
        buffer.update(10 + i, 5, "C", Color::BLUE, Color::BLACK, Modifiers::NONE);
    }
    output.clear();
    let _ = renderer.flush(&mut buffer, &mut output);
    println!(
        "  Consecutive (10 cells): {} cursor move(s)",
        renderer.cursor_moves()
    );
}
