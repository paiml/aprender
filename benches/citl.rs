//! Benchmarks for CITL (Compiler-in-the-Loop Learning) module.
//!
//! Tests SIMD-accelerated similarity operations via trueno.

use aprender::citl::{
    CompilerDiagnostic, DiagnosticSeverity, Difficulty, ErrorCategory, ErrorCode, ErrorEmbedding,
    ErrorEncoder, NeuralEncoderConfig, NeuralErrorEncoder, PatternLibrary, SourceSpan,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Generate random embedding vector for benchmarking.
fn random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut vec = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        // Simple LCG for deterministic "random" values
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        vec.push((state as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    vec
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_cosine_similarity");

    for &dim in &[64, 128, 256, 512, 1024] {
        group.throughput(Throughput::Elements(dim as u64));

        let v1 = random_embedding(dim, 42);
        let v2 = random_embedding(dim, 123);

        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let e1 = ErrorEmbedding::new(v1, code.clone(), 0);
        let e2 = ErrorEmbedding::new(v2, code, 0);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| e1.cosine_similarity(black_box(&e2)));
        });
    }

    group.finish();
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_l2_distance");

    for &dim in &[64, 128, 256, 512, 1024] {
        group.throughput(Throughput::Elements(dim as u64));

        let v1 = random_embedding(dim, 42);
        let v2 = random_embedding(dim, 123);

        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let e1 = ErrorEmbedding::new(v1, code.clone(), 0);
        let e2 = ErrorEmbedding::new(v2, code, 0);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| e1.l2_distance(black_box(&e2)));
        });
    }

    group.finish();
}

fn bench_pattern_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_pattern_search");

    for &n_patterns in &[10, 50, 100, 500] {
        group.throughput(Throughput::Elements(n_patterns as u64));

        // Build pattern library
        let mut lib = PatternLibrary::new();
        for i in 0..n_patterns {
            let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
            let embedding = ErrorEmbedding::new(random_embedding(256, i as u64), code, i as u64);
            let template =
                aprender::citl::FixTemplate::new("$expr.to_string()", "Convert to String");
            lib.add_pattern(embedding, template);
        }

        // Query embedding
        let query_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let query = ErrorEmbedding::new(random_embedding(256, 9999), query_code, 0);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_patterns),
            &n_patterns,
            |b, _| {
                b.iter(|| lib.search(black_box(&query), 5));
            },
        );
    }

    group.finish();
}

fn bench_error_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_error_encoding");

    let encoder = ErrorEncoder::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diagnostic = CompilerDiagnostic::new(
        code,
        DiagnosticSeverity::Error,
        "mismatched types: expected `i32`, found `&str`",
        span,
    );

    // Various source sizes
    for &lines in &[10, 50, 100, 500] {
        let source: String = (0..lines)
            .map(|i| format!("let x{i}: i32 = \"hello\"; // line {i}\n"))
            .collect();

        group.throughput(Throughput::Elements(lines as u64));

        group.bench_with_input(BenchmarkId::from_parameter(lines), &source, |b, src| {
            b.iter(|| encoder.encode(black_box(&diagnostic), black_box(src)));
        });
    }

    group.finish();
}

fn bench_batch_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_batch_similarity");

    // Test batch comparisons (common in pattern matching)
    for &batch_size in &[10, 50, 100, 500] {
        group.throughput(Throughput::Elements(batch_size as u64));

        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let query = ErrorEmbedding::new(random_embedding(256, 42), code.clone(), 0);

        let candidates: Vec<ErrorEmbedding> = (0..batch_size)
            .map(|i| ErrorEmbedding::new(random_embedding(256, i as u64), code.clone(), i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &candidates,
            |b, cands| {
                b.iter(|| {
                    cands
                        .iter()
                        .map(|c| query.cosine_similarity(black_box(c)))
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

fn bench_neural_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_neural_encoder");

    // Create encoder with minimal config for benchmarking
    let encoder = NeuralErrorEncoder::with_config(NeuralEncoderConfig::minimal());

    let error_messages = [
        "E0308: mismatched types, expected i32 found &str",
        "E0382: borrow of moved value: `x`",
        "E0277: the trait bound `Foo: Debug` is not satisfied",
    ];

    let source_contexts = [
        "let x: i32 = \"hello\";",
        "let x = vec![1]; let y = x; let z = x;",
        "println!(\"{:?}\", foo);",
    ];

    // Benchmark single encoding
    group.bench_function("single_encode", |b| {
        b.iter(|| {
            encoder.encode(
                black_box(error_messages[0]),
                black_box(source_contexts[0]),
                black_box("rust"),
            )
        });
    });

    // Benchmark different error types
    for (i, (msg, ctx)) in error_messages
        .iter()
        .zip(source_contexts.iter())
        .enumerate()
    {
        group.bench_with_input(
            BenchmarkId::new("error_type", i),
            &(msg, ctx),
            |b, (msg, ctx)| {
                b.iter(|| encoder.encode(black_box(msg), black_box(ctx), black_box("rust")));
            },
        );
    }

    // Benchmark cross-language encoding
    let languages = ["rust", "python", "typescript"];
    for lang in &languages {
        group.bench_with_input(BenchmarkId::new("language", lang), lang, |b, lang| {
            b.iter(|| {
                encoder.encode(
                    black_box("TypeError: expected int, got str"),
                    black_box("x: int = \"hello\""),
                    black_box(lang),
                )
            });
        });
    }

    group.finish();
}

fn bench_neural_encoder_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("citl_neural_config");

    let configs = [
        ("minimal", NeuralEncoderConfig::minimal()),
        ("small", NeuralEncoderConfig::small()),
    ];

    for (name, config) in &configs {
        let encoder = NeuralErrorEncoder::with_config(config.clone());

        group.bench_with_input(BenchmarkId::from_parameter(name), &encoder, |b, enc| {
            b.iter(|| {
                enc.encode(
                    black_box("E0308: mismatched types"),
                    black_box("let x: i32 = \"hello\";"),
                    black_box("rust"),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_l2_distance,
    bench_pattern_search,
    bench_error_encoding,
    bench_batch_similarity,
    bench_neural_encoder,
    bench_neural_encoder_configs,
);
criterion_main!(benches);
