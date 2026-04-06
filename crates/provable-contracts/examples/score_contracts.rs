//! Demonstrates scoring contracts and printing grade distribution.
//!
//! This example shows how to:
//! 1. Score individual contracts across 5 dimensions (spec depth, falsification,
//!    kani, lean, binding)
//! 2. Use custom scoring weights
//! 3. Inspect per-dimension probe breakdowns
//! 4. Compute grade distributions across a set of contracts
//!
//! Run with:
//!   cargo run --example score_contracts
//!
//! Scores the first N contracts found in `contracts/` and prints a report
//! with grade distribution histogram.

use std::collections::BTreeMap;
use std::path::Path;

use provable_contracts::schema::{parse_contract, parse_contract_str};
use provable_contracts::scoring::{
    ContractScore, Grade, ScoringWeights, score_contract, score_contract_weighted,
};

/// Inline contract for self-contained demonstration (no file I/O needed).
const EXAMPLE_CONTRACT: &str = r#"
metadata:
  version: "1.0.0"
  description: "Softmax kernel — numerically stable exponential normalization"
  references:
    - "Bridle (1990) Training Stochastic Model Recognition"
    - "Milakov & Gimelshein (2018) Online normalizer calculation"
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))"
    domain: "x ∈ ℝ^n, n ≥ 1"
    codomain: "σ(x) ∈ (0,1)^n"
    invariants:
      - "Σ σ(x)_i = 1.0 (normalization)"
      - "σ(x)_i > 0 for all i (strict positivity)"
proof_obligations:
  - type: invariant
    property: "normalization"
    formal: "|Σ σ(x)_i - 1.0| < ε"
    tolerance: 1.0e-6
    applies_to: all
  - type: bound
    property: "positivity"
    formal: "σ(x)_i > 0 for all i"
    applies_to: all
kernel_structure:
  phases:
    - name: find_max
      description: "Compute max(x) for numerical stability"
      invariant: "max >= all elements"
    - name: exp_subtract
      description: "Compute exp(x_i - max)"
      invariant: "all values in (0, 1]"
    - name: normalize
      description: "Divide by sum of exponentials"
      invariant: "output sums to 1"
falsification_tests:
  - id: FALSIFY-SM-001
    rule: "normalization"
    prediction: "sum(output) ≈ 1.0"
    if_fails: "Missing max-subtraction trick"
  - id: FALSIFY-SM-002
    rule: "positivity"
    prediction: "output > 0"
    if_fails: "exp underflow"
kani_harnesses:
  - id: KANI-SM-001
    obligation: normalization
    property: "Softmax normalization"
    bound: 16
    strategy: stub_float
  - id: KANI-SM-002
    obligation: positivity
    property: "Softmax positivity"
    bound: 8
    strategy: exhaustive
qa_gate:
  id: F-SM-001
  name: "Softmax Contract"
  checks: ["normalization", "positivity"]
  pass_criteria: "All falsification tests pass"
"#;

fn main() {
    println!("=== provable-contracts: Scoring Example ===\n");

    // --- Part 1: Score the inline example contract ---
    println!("--- Part 1: Score an inline contract ---\n");

    let contract = parse_contract_str(EXAMPLE_CONTRACT).expect("inline contract should parse");

    let score = score_contract(&contract, None, "softmax-kernel-v1");
    print_score_detail(&score);

    // --- Part 2: Score with custom weights ---
    println!("\n--- Part 2: Custom weights (heavy on spec + falsification) ---\n");

    let custom_weights = ScoringWeights {
        spec_depth: 0.40,
        falsification: 0.30,
        kani: 0.15,
        lean: 0.05,
        binding: 0.10,
    };
    let custom_score =
        score_contract_weighted(&contract, None, "softmax-kernel-v1", &custom_weights);
    println!(
        "  Custom weights: spec={:.0}% falsify={:.0}% kani={:.0}% lean={:.0}% bind={:.0}%",
        custom_weights.spec_depth * 100.0,
        custom_weights.falsification * 100.0,
        custom_weights.kani * 100.0,
        custom_weights.lean * 100.0,
        custom_weights.binding * 100.0,
    );
    println!(
        "  Composite: {:.3} (Grade {}) vs default {:.3} (Grade {})",
        custom_score.composite, custom_score.grade, score.composite, score.grade
    );

    // --- Part 3: Probe breakdown ---
    println!("\n--- Part 3: Probe breakdown for each dimension ---\n");
    print_probes(&score);

    // --- Part 4: Grade distribution across contracts on disk ---
    println!("\n--- Part 4: Grade distribution across contract files ---\n");

    let contracts_dir = Path::new("contracts");
    if contracts_dir.is_dir() {
        let scores = score_directory(contracts_dir);
        print_grade_distribution(&scores);
    } else {
        println!("  No contracts/ directory found. Showing distribution from inline example only.");
        let scores = vec![score];
        print_grade_distribution(&scores);
    }
}

/// Score all YAML contracts in a directory.
fn score_directory(dir: &Path) -> Vec<ContractScore> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("readable directory")
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|ext| ext == "yaml")
        })
        .collect();
    entries.sort_by_key(std::fs::DirEntry::path);

    let mut scores = Vec::new();
    for entry in &entries {
        let path = entry.path();
        let Ok(contract) = parse_contract(&path) else {
            continue;
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let s = score_contract(&contract, None, stem);
        scores.push(s);
    }

    // Sort by composite score descending
    scores.sort_by(|a, b| {
        b.composite
            .partial_cmp(&a.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

/// Print detailed score breakdown for a single contract.
fn print_score_detail(score: &ContractScore) {
    println!("  Contract: {}", score.stem);
    println!(
        "  Composite: {:.3} (Grade {})",
        score.composite, score.grade
    );
    println!("  Dimensions:");
    println!("    Spec depth:     {:.3}", score.spec_depth);
    println!("    Falsification:  {:.3}", score.falsification_coverage);
    println!("    Kani coverage:  {:.3}", score.kani_coverage);
    println!("    Lean coverage:  {:.3}", score.lean_coverage);
    println!("    Binding:        {:.3}", score.binding_coverage);
}

/// Print per-dimension probes grouped by dimension.
fn print_probes(score: &ContractScore) {
    let mut by_dimension: BTreeMap<&str, Vec<&provable_contracts::scoring::ScoreProbe>> =
        BTreeMap::new();
    for probe in &score.probes {
        by_dimension
            .entry(&probe.dimension)
            .or_default()
            .push(probe);
    }

    for (dim, probes) in &by_dimension {
        println!("  [{dim}]");
        for p in probes {
            let icon = if p.outcome { "PASS" } else { "FAIL" };
            println!("    [{icon}] {}: {}", p.probe, p.detail);
        }
    }
}

/// Print a grade distribution histogram and top/bottom contracts.
fn print_grade_distribution(scores: &[ContractScore]) {
    if scores.is_empty() {
        println!("  No contracts to score.");
        return;
    }

    // Count grades
    let mut grade_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for label in &["A", "B", "C", "D", "F"] {
        grade_counts.insert(label, 0);
    }
    for s in scores {
        let label = match s.grade {
            Grade::A => "A",
            Grade::B => "B",
            Grade::C => "C",
            Grade::D => "D",
            Grade::F => "F",
        };
        *grade_counts.entry(label).or_insert(0) += 1;
    }

    // Histogram
    let max_count = *grade_counts.values().max().unwrap_or(&1);
    let bar_width = 40;

    println!("  Grade Distribution ({} contracts):\n", scores.len());
    for grade in &["A", "B", "C", "D", "F"] {
        let count = grade_counts.get(grade).copied().unwrap_or(0);
        let bar_len = if max_count > 0 {
            (count * bar_width) / max_count
        } else {
            0
        };
        let bar: String = "#".repeat(bar_len);
        let pct = (count as f64 / scores.len() as f64) * 100.0;
        println!(
            "    {grade} | {bar:<width$} | {count:>3} ({pct:5.1}%)",
            width = bar_width
        );
    }

    // Top 5 and Bottom 5
    if scores.len() >= 2 {
        let top_n = scores.len().min(5);
        println!("\n  Top {top_n} contracts:");
        for s in scores.iter().take(top_n) {
            println!("    {:<40} {:.3} ({})", s.stem, s.composite, s.grade);
        }

        println!(
            "\n  Bottom {} contracts (improvement targets):",
            scores.len().min(5)
        );
        for s in scores.iter().rev().take(scores.len().min(5)) {
            println!("    {:<40} {:.3} ({})", s.stem, s.composite, s.grade);
        }
    }

    // Summary stats
    #[allow(clippy::cast_precision_loss)]
    let mean = scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64;
    let median = {
        let mut sorted: Vec<f64> = scores.iter().map(|s| s.composite).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };

    println!(
        "\n  Summary: {} contracts | mean={:.3} ({}) | median={:.3} ({})",
        scores.len(),
        mean,
        Grade::from_score(mean),
        median,
        Grade::from_score(median),
    );
}
