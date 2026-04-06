//! Basic Generator Example
//!
//! Demonstrates how to generate test programs using verificar.
//!
//! Run with: cargo run --example generator_basic

use verificar::generator::{Generator, SamplingStrategy};
use verificar::Language;

fn main() {
    println!("=== Verificar Generator Example ===\n");

    // Create generators for different languages
    let languages = [Language::Python, Language::Bash, Language::C];

    for lang in languages {
        println!("--- Generating {} programs ---", lang);

        let generator = Generator::new(lang);

        // Generate using exhaustive strategy (depth 2 for speed)
        let programs = generator.generate_exhaustive(2);
        println!("  Exhaustive (depth=2): {} programs", programs.len());

        // Show first few programs
        for (i, prog) in programs.iter().take(3).enumerate() {
            let preview = prog.code.replace('\n', "\\n");
            let preview = if preview.len() > 60 {
                format!("{}...", &preview[..57])
            } else {
                preview
            };
            println!("    {}: {}", i + 1, preview);
        }

        // Generate with statistics
        let stats = generator.generate_with_stats(2);
        println!(
            "  Stats: {} total, {} valid, {} invalid",
            stats.total_generated, stats.valid_count, stats.invalid_count
        );

        println!();
    }

    // Demonstrate different sampling strategies
    println!("--- Sampling Strategies ---");

    let generator = Generator::new(Language::Python);

    // Coverage-guided
    let strategy = SamplingStrategy::CoverageGuided {
        coverage_map: None,
        max_depth: 3,
        seed: 42,
    };
    match generator.generate(strategy, 10) {
        Ok(programs) => println!("  CoverageGuided: {} programs", programs.len()),
        Err(e) => println!("  CoverageGuided: error - {}", e),
    }

    // Swarm (random feature subsets per batch)
    let strategy = SamplingStrategy::Swarm {
        features_per_batch: 5,
    };
    match generator.generate(strategy, 10) {
        Ok(programs) => println!("  Swarm: {} programs", programs.len()),
        Err(e) => println!("  Swarm: error - {}", e),
    }

    // Boundary (edge values emphasized)
    let strategy = SamplingStrategy::Boundary {
        boundary_probability: 0.5,
    };
    match generator.generate(strategy, 10) {
        Ok(programs) => println!("  Boundary: {} programs", programs.len()),
        Err(e) => println!("  Boundary: error - {}", e),
    }

    println!("\n=== Example Complete ===");
}
