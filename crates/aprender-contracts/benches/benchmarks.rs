#![allow(clippy::format_collect)]
// Benchmark binary: `.unwrap()` is idiomatic in bench setup.
#![allow(clippy::disallowed_methods)]
//! Criterion benchmarks for provable-contracts.
//!
//! Benchmarks YAML contract parsing, validation, and equation extraction
//! which are the hot paths in the contract pipeline.

#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use provable_contracts::schema::{parse_contract_str, validate_contract};

const MINIMAL_CONTRACT: &str = r#"
metadata:
  version: "1.0.0"
  description: "Minimal contract"
  references:
    - "Test paper (2024)"
equations:
  test_eq:
    formula: "f(x) = x + 1"
proof_obligations: []
falsification_tests: []
"#;

const FULL_CONTRACT: &str = r#"
metadata:
  version: "2.0.0"
  created: "2026-01-15"
  author: "Bench Author"
  description: "Full contract with multiple equations and obligations"
  references:
    - "Vaswani et al. (2017) Attention Is All You Need"
    - "He et al. (2016) Deep Residual Learning"
equations:
  softmax:
    formula: "softmax(x_i) = exp(x_i) / sum(exp(x_j))"
    domain: "x in R^n"
    codomain: "y in (0,1)^n, sum(y) = 1"
  layer_norm:
    formula: "LN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta"
    domain: "x in R^d"
    codomain: "y in R^d"
  cross_entropy:
    formula: "CE(p, q) = -sum(p_i * log(q_i))"
    domain: "p, q in simplex"
    codomain: "R >= 0"
proof_obligations:
  - type: invariant
    property: "sum(softmax(x)) == 1.0"
    tolerance: 1e-6
  - type: bound
    property: "mean(LN(x)) approx 0"
    tolerance: 1e-5
falsification_tests:
  - id: "FT-001"
    rule: "softmax output sums to 1.0"
    prediction: "softmax([1000, 1000, 1000]) sums to 1.0 within 1e-6"
    if_fails: "Overflow in exp() not handled; need log-sum-exp trick"
  - id: "FT-002"
    rule: "cross-entropy handles zero probabilities"
    prediction: "CE([0.5,0.5], [0.0,1.0]) returns Inf or clamps gracefully"
    if_fails: "log(0) produces NaN instead of Inf; add epsilon clamping"
"#;

fn bench_parse_minimal_contract(c: &mut Criterion) {
    c.bench_function("parse_minimal_contract", |b| {
        b.iter(|| parse_contract_str(black_box(MINIMAL_CONTRACT)).unwrap());
    });
}

fn bench_parse_full_contract(c: &mut Criterion) {
    c.bench_function("parse_full_contract", |b| {
        b.iter(|| parse_contract_str(black_box(FULL_CONTRACT)).unwrap());
    });
}

fn bench_validate_contract(c: &mut Criterion) {
    let contract = parse_contract_str(FULL_CONTRACT).unwrap();
    c.bench_function("validate_full_contract", |b| {
        b.iter(|| validate_contract(black_box(&contract)));
    });
}

fn bench_parse_roundtrip(c: &mut Criterion) {
    c.bench_function("contract_yaml_roundtrip", |b| {
        b.iter(|| {
            let contract = parse_contract_str(black_box(FULL_CONTRACT)).unwrap();
            let yaml = serde_yaml::to_string(&contract).unwrap();
            let _: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
        });
    });
}

fn bench_equation_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("equation_scaling");
    for count in [1, 5, 10, 20] {
        let eqs: String = (0..count)
            .map(|i| format!("  eq_{i}:\n    formula: \"f_{i}(x) = x + {i}\"\n"))
            .collect();
        let yaml = format!(
            "metadata:\n  version: \"1.0.0\"\n  description: \"bench\"\n  references:\n    - \"ref\"\nequations:\n{eqs}proof_obligations: []\nfalsification_tests: []\n"
        );
        group.bench_with_input(BenchmarkId::new("equations", count), &yaml, |b, yaml| {
            b.iter(|| parse_contract_str(black_box(yaml)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_minimal_contract,
    bench_parse_full_contract,
    bench_validate_contract,
    bench_parse_roundtrip,
    bench_equation_count_scaling,
);
criterion_main!(benches);
