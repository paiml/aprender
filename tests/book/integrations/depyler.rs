//! Depyler Integration book tests
//!
//! Tests validating examples from the depyler integration chapter.
//!
//! # Depyler Integration Guide
//!
//! Verificar provides test case generation for depyler (Python to Rust transpiler).
//!
//! ## CLI Usage
//!
//! ```bash
//! # Install verificar
//! cargo install verificar
//!
//! # Generate Python test cases for depyler
//! verificar generate --language python --count 100 --max-depth 3
//!
//! # Generate coverage-guided corpus (NAUTILUS-style)
//! verificar corpus --language python --count 1000 --max-depth 4 > corpus.py
//!
//! # Output as JSON for programmatic use
//! verificar generate -l python -c 50 -o json > tests.json
//! ```
//!
//! ## Programmatic Usage
//!
//! ```rust
//! use verificar::generator::{Generator, SamplingStrategy};
//! use verificar::Language;
//!
//! // Create generator for Python (depyler's source language)
//! let generator = Generator::new(Language::Python);
//!
//! // Generate exhaustive test cases up to depth 3
//! let programs = generator.generate_exhaustive(3);
//!
//! // Each program can be fed to depyler for transpilation
//! for prog in &programs {
//!     println!("Python: {}", prog.code);
//!     // depyler::transpile(&prog.code) -> Rust code
//! }
//! ```
//!
//! ## TranspilerOracle Usage
//!
//! ```rust,ignore
//! use verificar::transpiler::{TranspilerOracle, TranspilerVerdict};
//!
//! // Create oracle with your transpiler implementation
//! let oracle = TranspilerOracle::new(depyler);
//!
//! // Verify a single program
//! let result = oracle.verify("print('hello')", "");
//! assert_eq!(result.verdict, TranspilerVerdict::Pass);
//!
//! // Batch verification with stats
//! let (results, stats) = oracle.verify_generated(100, 3);
//! println!("Pass rate: {:.1}%", stats.pass_rate());
//! ```
//!
//! ## Verification Workflow
//!
//! 1. Generate Python programs with verificar
//! 2. Transpile each program with depyler -> Rust
//! 3. Execute both and compare I/O
//! 4. Collect equivalence data for ML training

use verificar::generator::Generator;
use verificar::transpiler::{TranspilerConfig, TranspilerVerdict, VerificationStats};
use verificar::Language;

/// Example: Basic Python generation for depyler
#[test]
fn test_depyler_python_generation() {
    let generator = Generator::new(Language::Python);
    let programs = generator.generate_exhaustive(2);

    assert!(!programs.is_empty(), "Should generate programs");
    for prog in &programs {
        assert_eq!(prog.language, Language::Python);
        assert!(!prog.code.is_empty());
    }
}

/// Example: Coverage-guided generation for comprehensive testing
#[test]
fn test_depyler_coverage_guided() {
    let generator = Generator::new(Language::Python);
    let (programs, stats) = generator.generate_coverage_guided_with_map(10, 2, 42, None);

    assert!(!programs.is_empty(), "Should generate programs");
    assert!(stats.node_types_covered > 0, "Should cover node types");
    assert!(stats.corpus_size > 0, "Should build corpus");
}

/// Example: Statistics for test adequacy
#[test]
fn test_depyler_generation_stats() {
    let generator = Generator::new(Language::Python);
    let stats = generator.generate_with_stats(2);

    // Depyler needs diverse test cases
    assert!(stats.valid_count > 0, "Should generate valid programs");
    assert!(
        stats.pass_rate() >= 50.0,
        "Most programs should be valid Python"
    );
}

/// Example: Transpiler configuration for depyler
#[test]
fn test_depyler_transpiler_config() {
    let config = TranspilerConfig::depyler();

    assert_eq!(config.name, "depyler");
    assert_eq!(config.source, Language::Python);
    assert_eq!(config.target, Language::Rust);
    assert!(config.strict);
}

/// Example: Verification statistics tracking
#[test]
fn test_verification_stats_usage() {
    let stats = VerificationStats {
        total: 100,
        passed: 85,
        transpile_errors: 5,
        source_errors: 2,
        target_errors: 3,
        mismatches: 4,
        timeouts: 1,
    };

    // Pass rate calculation
    assert!((stats.pass_rate() - 85.0).abs() < 0.001);

    // Transpile success rate
    assert!((stats.transpile_rate() - 95.0).abs() < 0.001);
}

/// Example: TranspilerVerdict comparison
#[test]
fn test_transpiler_verdict_types() {
    // Pass indicates I/O equivalence
    assert_eq!(TranspilerVerdict::Pass, TranspilerVerdict::Pass);

    // Different verdicts are distinguishable
    assert_ne!(TranspilerVerdict::Pass, TranspilerVerdict::OutputMismatch);
    assert_ne!(TranspilerVerdict::Pass, TranspilerVerdict::Timeout);
}
