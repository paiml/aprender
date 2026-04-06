//! Locator Operations Benchmarks
//!
//! Benchmarks for CSS selector parsing, element matching, and locator operations.
//!
//! Run with: `cargo bench --bench locator_ops`

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jugar_probar::prelude::*;

fn bench_selector_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("selector_parsing");

    let selectors = vec![
        ("simple_id", "#my-button"),
        ("simple_class", ".btn-primary"),
        ("tag", "button"),
        ("attribute", "[data-testid=\"submit\"]"),
        ("complex", "div.container > button.btn-primary:first-child"),
        ("nth_child", "li:nth-child(3)"),
        ("multiple_classes", ".btn.btn-lg.btn-primary"),
        ("descendant", "form input[type=\"text\"]"),
    ];

    for (name, selector) in selectors {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &selector,
            |bench, sel| {
                bench.iter(|| {
                    let parsed = Selector::css(black_box(*sel));
                    black_box(parsed);
                });
            },
        );
    }

    group.finish();
}

fn bench_locator_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("locator_creation");

    let test_cases = vec![
        ("css", "css"),
        ("text", "text"),
        ("role", "role"),
        ("label", "label"),
        ("placeholder", "placeholder"),
        ("test_id", "test_id"),
    ];

    for (name, selector_type) in test_cases {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &selector_type,
            |bench, sel_type| {
                bench.iter(|| {
                    let selector = match *sel_type {
                        "css" => Selector::css("#submit-btn"),
                        "text" => Selector::text("Submit"),
                        "role" => Selector::role("button"),
                        "label" => Selector::label("Username"),
                        "placeholder" => Selector::placeholder("Enter your name"),
                        "test_id" => Selector::test_id("login-form"),
                        _ => Selector::css("div"),
                    };
                    let locator = Locator::from_selector(black_box(selector));
                    black_box(locator);
                });
            },
        );
    }

    group.finish();
}

fn bench_locator_chaining(c: &mut Criterion) {
    let mut group = c.benchmark_group("locator_chaining");

    // Benchmark chaining depth using `and`
    let depths = vec![1, 2, 3, 5, 10];

    for depth in depths {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("and_depth_{}", depth)),
            &depth,
            |bench, &d| {
                bench.iter(|| {
                    let mut locator = Locator::from_selector(Selector::css("div"));
                    for i in 0..d {
                        locator = locator.and(Locator::from_selector(Selector::css(format!(
                            ".level-{}",
                            i
                        ))));
                    }
                    black_box(locator);
                });
            },
        );
    }

    group.finish();
}

fn bench_locator_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("locator_filtering");

    let filters = vec![
        ("has_text_short", "OK"),
        ("has_text_medium", "Submit Form"),
        (
            "has_text_long",
            "Click here to submit the form and continue",
        ),
    ];

    for (name, filter_text) in filters {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &filter_text,
            |bench, text| {
                bench.iter(|| {
                    let filter = FilterOptions::new().has_text(black_box(*text));
                    let locator =
                        Locator::from_selector(Selector::css("button")).filter(black_box(filter));
                    black_box(locator);
                });
            },
        );
    }

    group.finish();
}

fn bench_locator_nth_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("locator_nth");

    let operations = vec![
        ("first", 0usize),
        ("nth_5", 5),
        ("nth_10", 10),
        ("nth_100", 100),
    ];

    for (name, index) in operations {
        group.bench_with_input(BenchmarkId::from_parameter(name), &index, |bench, &idx| {
            bench.iter(|| {
                let locator = Locator::from_selector(Selector::css("li")).nth(black_box(idx));
                black_box(locator);
            });
        });
    }

    group.finish();
}

fn bench_selector_to_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("selector_to_query");

    let selectors = vec![
        ("simple", Selector::css("#btn")),
        (
            "complex",
            Selector::css("div.container > button.btn-primary"),
        ),
        ("text", Selector::text("Submit")),
        ("role", Selector::role("button")),
        ("test_id", Selector::test_id("submit-btn")),
    ];

    for (name, selector) in selectors {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &selector,
            |bench, sel| {
                bench.iter(|| {
                    let query = black_box(sel).to_query();
                    black_box(query);
                });
            },
        );
    }

    group.finish();
}

fn bench_locator_or_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("locator_or");

    let counts = vec![2, 5, 10, 20];

    for count in counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_alternatives", count)),
            &count,
            |bench, &n| {
                bench.iter(|| {
                    let mut locator = Locator::from_selector(Selector::css(".option-0"));
                    for i in 1..n {
                        locator = locator.or(Locator::from_selector(Selector::css(format!(
                            ".option-{}",
                            i
                        ))));
                    }
                    black_box(locator);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_selector_parsing,
    bench_locator_creation,
    bench_locator_chaining,
    bench_locator_filtering,
    bench_locator_nth_operations,
    bench_selector_to_query,
    bench_locator_or_operations
);
criterion_main!(benches);
