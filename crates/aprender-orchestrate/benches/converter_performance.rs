//! ML Converter Performance Benchmarks
//!
//! Measures the performance of NumPy, sklearn, and PyTorch converters
//! to validate conversion overhead is minimal.
//!
//! Run with: cargo bench --bench converter_performance

use batuta::numpy_converter::{NumPyConverter, NumPyOp};
use batuta::pytorch_converter::{PyTorchConverter, PyTorchOperation};
use batuta::sklearn_converter::{SklearnAlgorithm, SklearnConverter};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Benchmark NumPy converter operations
fn bench_numpy_converter(c: &mut Criterion) {
    let mut group = c.benchmark_group("numpy_converter");

    let converter = NumPyConverter::new();
    let operations = vec![
        (NumPyOp::Add, "add"),
        (NumPyOp::Dot, "dot"),
        (NumPyOp::Sum, "sum"),
        (NumPyOp::Mean, "mean"),
        (NumPyOp::Multiply, "multiply"),
    ];

    for (op, name) in operations {
        group.bench_with_input(BenchmarkId::new("convert", name), &op, |b, op| {
            b.iter(|| {
                let result = converter.convert(black_box(op));
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("recommend_backend", name), &op, |b, op| {
            b.iter(|| {
                let backend = converter.recommend_backend(black_box(op), black_box(100_000));
                black_box(backend);
            });
        });
    }

    group.finish();
}

/// Benchmark sklearn converter operations
fn bench_sklearn_converter(c: &mut Criterion) {
    let mut group = c.benchmark_group("sklearn_converter");

    let converter = SklearnConverter::new();
    let algorithms = vec![
        (SklearnAlgorithm::LinearRegression, "linear_regression"),
        (SklearnAlgorithm::LogisticRegression, "logistic_regression"),
        (SklearnAlgorithm::KMeans, "kmeans"),
        (SklearnAlgorithm::DecisionTreeClassifier, "decision_tree"),
        (SklearnAlgorithm::StandardScaler, "standard_scaler"),
    ];

    for (alg, name) in algorithms {
        group.bench_with_input(BenchmarkId::new("convert", name), &alg, |b, alg| {
            b.iter(|| {
                let result = converter.convert(black_box(alg));
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("recommend_backend", name), &alg, |b, alg| {
            b.iter(|| {
                let backend = converter.recommend_backend(black_box(alg), black_box(100_000));
                black_box(backend);
            });
        });
    }

    group.finish();
}

/// Benchmark PyTorch converter operations
fn bench_pytorch_converter(c: &mut Criterion) {
    let mut group = c.benchmark_group("pytorch_converter");

    let converter = PyTorchConverter::new();
    let operations = vec![
        (PyTorchOperation::LoadModel, "load_model"),
        (PyTorchOperation::Forward, "forward"),
        (PyTorchOperation::Generate, "generate"),
        (PyTorchOperation::Linear, "linear"),
        (PyTorchOperation::Attention, "attention"),
    ];

    for (op, name) in operations {
        group.bench_with_input(BenchmarkId::new("convert", name), &op, |b, op| {
            b.iter(|| {
                let result = converter.convert(black_box(op));
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("recommend_backend", name), &op, |b, op| {
            b.iter(|| {
                let backend = converter.recommend_backend(black_box(op), black_box(1_000_000));
                black_box(backend);
            });
        });
    }

    group.finish();
}

/// Benchmark conversion report generation
fn bench_conversion_reports(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_reports");

    let numpy_converter = NumPyConverter::new();
    let sklearn_converter = SklearnConverter::new();
    let pytorch_converter = PyTorchConverter::new();

    group.bench_function("numpy_report", |b| {
        b.iter(|| {
            let report = numpy_converter.conversion_report();
            black_box(report);
        });
    });

    group.bench_function("sklearn_report", |b| {
        b.iter(|| {
            let report = sklearn_converter.conversion_report();
            black_box(report);
        });
    });

    group.bench_function("pytorch_report", |b| {
        b.iter(|| {
            let report = pytorch_converter.conversion_report();
            black_box(report);
        });
    });

    group.finish();
}

/// Benchmark available operations listing
fn bench_available_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("available_operations");

    let numpy_converter = NumPyConverter::new();
    let sklearn_converter = SklearnConverter::new();
    let pytorch_converter = PyTorchConverter::new();

    group.bench_function("numpy_available_ops", |b| {
        b.iter(|| {
            let ops = numpy_converter.available_ops();
            black_box(ops);
        });
    });

    group.bench_function("sklearn_available_algorithms", |b| {
        b.iter(|| {
            let algs = sklearn_converter.available_algorithms();
            black_box(algs);
        });
    });

    group.bench_function("pytorch_available_operations", |b| {
        b.iter(|| {
            let ops = pytorch_converter.available_operations();
            black_box(ops);
        });
    });

    group.finish();
}

/// Benchmark converter creation overhead
fn bench_converter_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("converter_creation");

    group.bench_function("numpy_converter_new", |b| {
        b.iter(|| {
            let converter = NumPyConverter::new();
            black_box(converter);
        });
    });

    group.bench_function("sklearn_converter_new", |b| {
        b.iter(|| {
            let converter = SklearnConverter::new();
            black_box(converter);
        });
    });

    group.bench_function("pytorch_converter_new", |b| {
        b.iter(|| {
            let converter = PyTorchConverter::new();
            black_box(converter);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_numpy_converter,
    bench_sklearn_converter,
    bench_pytorch_converter,
    bench_conversion_reports,
    bench_available_operations,
    bench_converter_creation,
);

criterion_main!(benches);
