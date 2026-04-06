//! Benchmark tests for layout engine operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use presentar_core::widget::{AccessibleRole, LayoutResult};
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Constraints, Event, Rect, Size,
    TypeId, Widget,
};
use presentar_layout::LayoutEngine;
use std::any::Any;
use std::time::Duration;

/// Test widget for benchmarking
struct BenchWidget {
    size: Size,
    children: Vec<Box<dyn Widget>>,
}

impl BenchWidget {
    fn new(width: f32, height: f32) -> Self {
        Self {
            size: Size::new(width, height),
            children: Vec::new(),
        }
    }

    fn with_child(mut self, child: Self) -> Self {
        self.children.push(Box::new(child));
        self
    }

    fn with_n_children(mut self, n: usize, width: f32, height: f32) -> Self {
        for _ in 0..n {
            self.children.push(Box::new(Self::new(width, height)));
        }
        self
    }
}

impl Brick for BenchWidget {
    fn brick_name(&self) -> &'static str {
        "BenchWidget"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![],
            failed: vec![],
            verification_time: Duration::from_micros(1),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for BenchWidget {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(self.size)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, _canvas: &mut dyn Canvas) {}

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

fn bench_layout_single_widget(c: &mut Criterion) {
    let mut engine = LayoutEngine::new();
    let viewport = Size::new(800.0, 600.0);

    c.bench_function("layout_single_widget", |b| {
        b.iter(|| {
            let mut widget = BenchWidget::new(100.0, 50.0);
            engine.compute(black_box(&mut widget), black_box(viewport));
        });
    });
}

fn bench_layout_10_children(c: &mut Criterion) {
    let mut engine = LayoutEngine::new();
    let viewport = Size::new(800.0, 600.0);

    c.bench_function("layout_10_children", |b| {
        b.iter(|| {
            let mut widget = BenchWidget::new(400.0, 300.0).with_n_children(10, 50.0, 50.0);
            engine.compute(black_box(&mut widget), black_box(viewport));
        });
    });
}

fn bench_layout_100_children(c: &mut Criterion) {
    let mut engine = LayoutEngine::new();
    let viewport = Size::new(800.0, 600.0);

    c.bench_function("layout_100_children", |b| {
        b.iter(|| {
            let mut widget = BenchWidget::new(800.0, 600.0).with_n_children(100, 30.0, 30.0);
            engine.compute(black_box(&mut widget), black_box(viewport));
        });
    });
}

fn bench_layout_nested_3_levels(c: &mut Criterion) {
    let mut engine = LayoutEngine::new();
    let viewport = Size::new(800.0, 600.0);

    c.bench_function("layout_nested_3_levels", |b| {
        b.iter(|| {
            let mut widget = BenchWidget::new(400.0, 300.0)
                .with_child(
                    BenchWidget::new(200.0, 150.0).with_child(BenchWidget::new(100.0, 75.0)),
                )
                .with_child(
                    BenchWidget::new(200.0, 150.0).with_child(BenchWidget::new(100.0, 75.0)),
                );
            engine.compute(black_box(&mut widget), black_box(viewport));
        });
    });
}

fn bench_layout_readonly(c: &mut Criterion) {
    let mut engine = LayoutEngine::new();
    let viewport = Size::new(800.0, 600.0);

    c.bench_function("layout_readonly_10_children", |b| {
        b.iter(|| {
            let widget = BenchWidget::new(400.0, 300.0).with_n_children(10, 50.0, 50.0);
            engine.compute_readonly(black_box(&widget), black_box(viewport));
        });
    });
}

fn bench_constraints_operations(c: &mut Criterion) {
    let constraints = Constraints::new(0.0, 800.0, 0.0, 600.0);
    let size = Size::new(500.0, 400.0);

    c.bench_function("constraints_constrain", |b| {
        b.iter(|| constraints.constrain(black_box(size)));
    });

    c.bench_function("constraints_deflate", |b| {
        b.iter(|| constraints.deflate(black_box(20.0), black_box(20.0)));
    });

    c.bench_function("constraints_tight", |b| {
        b.iter(|| Constraints::tight(black_box(size)));
    });

    c.bench_function("constraints_loose", |b| {
        b.iter(|| Constraints::loose(black_box(size)));
    });
}

fn bench_grid_12_column(c: &mut Criterion) {
    use presentar_layout::{compute_grid_layout, GridTemplate};

    c.bench_function("grid_12_column_layout", |b| {
        let template = GridTemplate::twelve_column();
        let child_sizes: Vec<(f32, f32)> = (0..12).map(|_| (100.0, 50.0)).collect();

        b.iter(|| {
            compute_grid_layout(
                black_box(&template),
                black_box(1200.0),
                black_box(600.0),
                black_box(&child_sizes),
            );
        });
    });
}

fn bench_grid_auto_placement(c: &mut Criterion) {
    use presentar_layout::{auto_place_items, GridAutoFlow, GridItem, GridTemplate, TrackSize};

    c.bench_function("grid_auto_place_24_items", |b| {
        let template = GridTemplate {
            columns: vec![TrackSize::Fr(1.0); 4],
            rows: vec![TrackSize::Fr(1.0); 6],
            ..GridTemplate::default()
        };
        let items: Vec<GridItem> = (0..24).map(|_| GridItem::new()).collect();

        b.iter(|| {
            let _ = auto_place_items(
                black_box(&template),
                black_box(&items),
                black_box(GridAutoFlow::RowDense),
            );
        });
    });
}

fn bench_grid_track_sizing(c: &mut Criterion) {
    use presentar_layout::{compute_grid_layout, GridTemplate, TrackSize};

    c.bench_function("grid_mixed_track_sizing", |b| {
        let template = GridTemplate {
            columns: vec![
                TrackSize::Px(100.0),
                TrackSize::Fr(1.0),
                TrackSize::Auto,
                TrackSize::Fr(2.0),
                TrackSize::Px(50.0),
            ],
            rows: vec![TrackSize::Fr(1.0); 3],
            ..GridTemplate::default()
        };
        let child_sizes: Vec<(f32, f32)> = (0..15).map(|_| (80.0, 40.0)).collect();

        b.iter(|| {
            compute_grid_layout(
                black_box(&template),
                black_box(800.0),
                black_box(400.0),
                black_box(&child_sizes),
            );
        });
    });
}

criterion_group!(
    benches,
    bench_layout_single_widget,
    bench_layout_10_children,
    bench_layout_100_children,
    bench_layout_nested_3_levels,
    bench_layout_readonly,
    bench_constraints_operations,
    bench_grid_12_column,
    bench_grid_auto_placement,
    bench_grid_track_sizing,
);
criterion_main!(benches);
