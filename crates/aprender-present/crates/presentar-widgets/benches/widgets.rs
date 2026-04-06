//! Benchmark tests for widget operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use presentar_core::{Constraints, Size, Widget};
use presentar_widgets::{Button, Text};

fn bench_button_creation(c: &mut Criterion) {
    c.bench_function("button_new", |b| {
        b.iter(|| Button::new(black_box("Click me")));
    });
}

fn bench_text_creation(c: &mut Criterion) {
    c.bench_function("text_new", |b| {
        b.iter(|| Text::new(black_box("Hello, World!")));
    });
}

fn bench_button_measure(c: &mut Criterion) {
    let button = Button::new("Click me");
    let constraints = Constraints::new(0.0, 200.0, 0.0, 50.0);

    c.bench_function("button_measure", |b| {
        b.iter(|| button.measure(black_box(constraints)));
    });
}

fn bench_text_measure(c: &mut Criterion) {
    let text = Text::new("Hello, World!");
    let constraints = Constraints::new(0.0, 200.0, 0.0, 50.0);

    c.bench_function("text_measure", |b| {
        b.iter(|| text.measure(black_box(constraints)));
    });
}

fn bench_constraints_constrain(c: &mut Criterion) {
    let constraints = Constraints::new(0.0, 200.0, 0.0, 100.0);
    let size = Size::new(150.0, 75.0);

    c.bench_function("constraints_constrain", |b| {
        b.iter(|| {
            let s = black_box(size);
            constraints.constrain(s)
        });
    });
}

fn bench_list_creation(c: &mut Criterion) {
    use presentar_widgets::{List, ListItem};

    c.bench_function("list_new_with_100_items", |b| {
        b.iter(|| {
            let items: Vec<ListItem> = (0..100)
                .map(|i| ListItem::new(format!("item_{i}")))
                .collect();
            List::new().items(black_box(items))
        });
    });
}

fn bench_list_scroll(c: &mut Criterion) {
    use presentar_widgets::{List, ListItem};

    let items: Vec<ListItem> = (0..1000)
        .map(|i| ListItem::new(format!("item_{i}")))
        .collect();
    let mut list = List::new().items(items);

    c.bench_function("list_scroll_to_item", |b| {
        b.iter(|| {
            list.scroll_to(black_box(500));
        });
    });
}

fn bench_modal_creation(c: &mut Criterion) {
    use presentar_widgets::Modal;

    c.bench_function("modal_new_with_title", |b| {
        b.iter(|| Modal::new().title(black_box("Confirm Action")));
    });
}

fn bench_menu_creation(c: &mut Criterion) {
    use presentar_widgets::{Menu, MenuItem};

    c.bench_function("menu_new_with_10_items", |b| {
        b.iter(|| {
            let items: Vec<MenuItem> = (0..10)
                .map(|i| MenuItem::action(format!("Action {i}"), format!("action_{i}")))
                .collect();
            Menu::new().items(black_box(items))
        });
    });
}

fn bench_reactive_cell(c: &mut Criterion) {
    use presentar_core::binding::ReactiveCell;

    c.bench_function("reactive_cell_get", |b| {
        let cell = ReactiveCell::new(42);
        b.iter(|| black_box(cell.get()));
    });

    c.bench_function("reactive_cell_set", |b| {
        let cell = ReactiveCell::new(0);
        let mut val = 0;
        b.iter(|| {
            val += 1;
            cell.set(black_box(val));
        });
    });
}

criterion_group!(
    benches,
    bench_button_creation,
    bench_text_creation,
    bench_button_measure,
    bench_text_measure,
    bench_constraints_constrain,
    bench_list_creation,
    bench_list_scroll,
    bench_modal_creation,
    bench_menu_creation,
    bench_reactive_cell,
);
criterion_main!(benches);
