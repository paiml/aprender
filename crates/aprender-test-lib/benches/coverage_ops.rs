//! Coverage Operations Benchmarks
//!
//! Benchmarks for pixel coverage tracking and GUI coverage.
//!
//! Run with: `cargo bench --bench coverage_ops`

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jugar_probar::pixel_coverage::{
    PixelCoverageTracker, PixelPoint as Point, PixelRegion as Region,
};
use jugar_probar::ux_coverage::{ElementId, InteractionType, UxCoverageTracker};

fn bench_pixel_tracker_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixel_tracker_creation");

    let configs = vec![
        (800, 600, 10, 10, "800x600_10x10"),
        (800, 600, 50, 50, "800x600_50x50"),
        (1920, 1080, 100, 100, "1920x1080_100x100"),
    ];

    for (width, height, cols, rows, name) in configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(width, height, cols, rows),
            |bench, &(w, h, c, r)| {
                bench.iter(|| {
                    let tracker = PixelCoverageTracker::new(
                        black_box(w),
                        black_box(h),
                        black_box(c),
                        black_box(r),
                    );
                    black_box(tracker);
                });
            },
        );
    }

    group.finish();
}

fn bench_pixel_interaction_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixel_interaction_recording");

    let interaction_counts = vec![100, 500, 1000, 5000];

    for count in interaction_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_interactions", count)),
            &count,
            |bench, &n| {
                bench.iter(|| {
                    let mut tracker = PixelCoverageTracker::new(800, 600, 50, 50);
                    for i in 0..n {
                        let x = ((i * 7) % 800) as u32;
                        let y = ((i * 11) % 600) as u32;
                        tracker.record_interaction(Point { x, y });
                    }
                    black_box(tracker);
                });
            },
        );
    }

    group.finish();
}

fn bench_pixel_region_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixel_region_recording");

    let region_counts = vec![10, 50, 100, 200];

    for count in region_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_regions", count)),
            &count,
            |bench, &n| {
                bench.iter(|| {
                    let mut tracker = PixelCoverageTracker::new(800, 600, 50, 50);
                    for i in 0..n {
                        let x = ((i * 37) % 700) as u32;
                        let y = ((i * 41) % 500) as u32;
                        tracker.record_region(Region {
                            x,
                            y,
                            width: 80,
                            height: 60,
                        });
                    }
                    black_box(tracker);
                });
            },
        );
    }

    group.finish();
}

fn bench_pixel_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixel_report_generation");

    let grid_sizes = vec![(10, 10, "10x10"), (50, 50, "50x50"), (100, 100, "100x100")];

    for (cols, rows, name) in grid_sizes {
        let mut tracker = PixelCoverageTracker::new(800, 600, cols, rows);
        // Fill ~75% coverage
        for i in 0..(cols * rows * 3 / 4) {
            let x = (i * 7) % 800;
            let y = (i * 11) % 600;
            tracker.record_interaction(Point { x, y });
        }

        group.bench_with_input(BenchmarkId::from_parameter(name), &tracker, |bench, t| {
            bench.iter(|| {
                let report = black_box(t).generate_report();
                black_box(report);
            });
        });
    }

    group.finish();
}

fn bench_terminal_heatmap_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("terminal_heatmap");

    let grid_sizes = vec![(10, 10, "10x10"), (20, 20, "20x20"), (50, 50, "50x50")];

    for (cols, rows, name) in grid_sizes {
        let mut tracker = PixelCoverageTracker::new(800, 600, cols, rows);
        for i in 0..(cols * rows / 2) {
            let x = (i * 7) % 800;
            let y = (i * 11) % 600;
            tracker.record_interaction(Point { x, y });
        }

        group.bench_with_input(BenchmarkId::from_parameter(name), &tracker, |bench, t| {
            bench.iter(|| {
                let heatmap = black_box(t).terminal_heatmap();
                black_box(heatmap);
            });
        });
    }

    group.finish();
}

fn bench_ux_element_registration(c: &mut Criterion) {
    let mut group = c.benchmark_group("ux_element_registration");

    let element_counts = vec![10, 50, 100, 500];

    for count in element_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_elements", count)),
            &count,
            |bench, &n| {
                bench.iter(|| {
                    let mut tracker = UxCoverageTracker::new();
                    for i in 0..n {
                        tracker.register_button(&format!("btn_{}", i));
                    }
                    black_box(tracker);
                });
            },
        );
    }

    group.finish();
}

fn bench_ux_interaction_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("ux_interaction_recording");

    let element_counts = vec![10, 50, 100];

    for count in element_counts {
        let mut base_tracker = UxCoverageTracker::new();
        for i in 0..count {
            base_tracker.register_button(&format!("btn_{}", i));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_elements", count)),
            &(count, base_tracker),
            |bench, (n, _)| {
                bench.iter(|| {
                    let mut tracker = UxCoverageTracker::new();
                    for i in 0..*n {
                        tracker.register_button(&format!("btn_{}", i));
                    }
                    for i in 0..(*n / 2) {
                        let elem = ElementId::new("button", &format!("btn_{}", i));
                        tracker.record_interaction(&elem, InteractionType::Click);
                    }
                    let report = tracker.generate_report();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

fn bench_ux_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ux_report_generation");

    let element_counts = vec![10, 50, 100, 500];

    for count in element_counts {
        let mut tracker = UxCoverageTracker::new();
        for i in 0..count {
            tracker.register_button(&format!("btn_{}", i));
        }
        for i in 0..(count * 3 / 4) {
            let elem = ElementId::new("button", &format!("btn_{}", i));
            tracker.record_interaction(&elem, InteractionType::Click);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_elements", count)),
            &tracker,
            |bench, t| {
                bench.iter(|| {
                    let report = black_box(t).generate_report();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pixel_tracker_creation,
    bench_pixel_interaction_recording,
    bench_pixel_region_recording,
    bench_pixel_report_generation,
    bench_terminal_heatmap_generation,
    bench_ux_element_registration,
    bench_ux_interaction_recording,
    bench_ux_report_generation
);
criterion_main!(benches);
