use criterion::{criterion_group, criterion_main, Criterion};
fn benchmark_baseline(c: &mut Criterion) {
    c.bench_function("baseline", |b| b.iter(|| std::hint::black_box(42)));
}
criterion_group!(benches, benchmark_baseline);
criterion_main!(benches);
