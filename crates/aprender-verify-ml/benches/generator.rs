//! Generator benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use verificar::generator::{Generator, SamplingStrategy};
use verificar::Language;

fn benchmark_exhaustive_generation(c: &mut Criterion) {
    let generator = Generator::new(Language::Python);

    c.bench_function("exhaustive_depth_3_100_samples", |b| {
        b.iter(|| {
            generator
                .generate(SamplingStrategy::Exhaustive { max_depth: 3 }, 100)
                .expect("generation should succeed")
        });
    });
}

fn benchmark_coverage_guided_generation(c: &mut Criterion) {
    let generator = Generator::new(Language::Python);

    c.bench_function("coverage_guided_100_samples", |b| {
        b.iter(|| {
            generator
                .generate(SamplingStrategy::CoverageGuided, 100)
                .expect("generation should succeed")
        });
    });
}

criterion_group!(
    benches,
    benchmark_exhaustive_generation,
    benchmark_coverage_guided_generation
);
criterion_main!(benches);
