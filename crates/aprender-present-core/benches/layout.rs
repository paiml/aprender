//! Benchmark tests for layout operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use presentar_core::{Constraints, Size};

fn bench_constraints_constrain(c: &mut Criterion) {
    let constraints = Constraints::new(0.0, 100.0, 0.0, 100.0);
    let size = Size::new(50.0, 50.0);

    c.bench_function("constraints_constrain", |b| {
        b.iter(|| constraints.constrain(black_box(size)));
    });
}

fn bench_constraints_tight(c: &mut Criterion) {
    let size = Size::new(100.0, 100.0);

    c.bench_function("constraints_tight", |b| {
        b.iter(|| Constraints::tight(black_box(size)));
    });
}

fn bench_size_creation(c: &mut Criterion) {
    c.bench_function("size_new", |b| {
        b.iter(|| Size::new(black_box(100.0), black_box(100.0)));
    });
}

criterion_group!(
    benches,
    bench_constraints_constrain,
    bench_constraints_tight,
    bench_size_creation,
);
criterion_main!(benches);
