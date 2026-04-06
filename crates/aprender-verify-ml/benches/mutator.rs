//! Mutator benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use verificar::mutator::{MutationOperator, Mutator};

fn benchmark_aor_mutation(c: &mut Criterion) {
    let mutator = Mutator::with_operators(vec![MutationOperator::Aor]);
    let code = "x = a + b + c + d";

    c.bench_function("aor_mutation", |b| {
        b.iter(|| mutator.mutate(code).expect("mutation should succeed"));
    });
}

fn benchmark_all_operators(c: &mut Criterion) {
    let mutator = Mutator::new();
    let code = "x = a + b\nif x < 10:\n    y = 0";

    c.bench_function("all_operators_mutation", |b| {
        b.iter(|| mutator.mutate(code).expect("mutation should succeed"));
    });
}

criterion_group!(benches, benchmark_aor_mutation, benchmark_all_operators);
criterion_main!(benches);
