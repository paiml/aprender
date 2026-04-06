use super::*;

// ===== ChartType Tests =====

#[test]
fn test_chart_type_default() {
    assert_eq!(ChartType::default(), ChartType::Line);
}

#[test]
fn test_chart_type_variants() {
    let types = [
        ChartType::Line,
        ChartType::Bar,
        ChartType::Scatter,
        ChartType::Area,
        ChartType::Pie,
        ChartType::Histogram,
        ChartType::Heatmap,
        ChartType::BoxPlot,
    ];
    assert_eq!(types.len(), 8);
}

#[test]
fn test_chart_heatmap() {
    let chart = Chart::new().chart_type(ChartType::Heatmap);
    assert_eq!(chart.get_chart_type(), ChartType::Heatmap);
}

#[test]
fn test_chart_boxplot() {
    let chart = Chart::new().chart_type(ChartType::BoxPlot);
    assert_eq!(chart.get_chart_type(), ChartType::BoxPlot);
}

// ===== DataSeries Tests =====

#[test]
fn test_data_series_new() {
    let series = DataSeries::new("Sales");
    assert_eq!(series.name, "Sales");
    assert!(series.points.is_empty());
    assert!(series.show_points);
    assert!(!series.fill);
}

#[test]
fn test_data_series_point() {
    let series = DataSeries::new("Data")
        .point(1.0, 10.0)
        .point(2.0, 20.0)
        .point(3.0, 15.0);
    assert_eq!(series.points.len(), 3);
    assert_eq!(series.points[0], (1.0, 10.0));
}

#[test]
fn test_data_series_points() {
    let data = vec![(1.0, 5.0), (2.0, 10.0), (3.0, 7.0)];
    let series = DataSeries::new("Data").points(data);
    assert_eq!(series.points.len(), 3);
}

#[test]
fn test_data_series_color() {
    let series = DataSeries::new("Data").color(Color::RED);
    assert_eq!(series.color, Color::RED);
}

#[test]
fn test_data_series_line_width() {
    let series = DataSeries::new("Data").line_width(3.0);
    assert_eq!(series.line_width, 3.0);
}

#[test]
fn test_data_series_line_width_min() {
    let series = DataSeries::new("Data").line_width(0.1);
    assert_eq!(series.line_width, 0.5);
}

#[test]
fn test_data_series_point_size() {
    let series = DataSeries::new("Data").point_size(6.0);
    assert_eq!(series.point_size, 6.0);
}

#[test]
fn test_data_series_point_size_min() {
    let series = DataSeries::new("Data").point_size(0.5);
    assert_eq!(series.point_size, 1.0);
}

#[test]
fn test_data_series_show_points() {
    let series = DataSeries::new("Data").show_points(false);
    assert!(!series.show_points);
}

#[test]
fn test_data_series_fill() {
    let series = DataSeries::new("Data").fill(true);
    assert!(series.fill);
}

#[test]
fn test_data_series_x_range() {
    let series = DataSeries::new("Data")
        .point(1.0, 10.0)
        .point(5.0, 20.0)
        .point(3.0, 15.0);
    assert_eq!(series.x_range(), Some((1.0, 5.0)));
}

#[test]
fn test_data_series_x_range_empty() {
    let series = DataSeries::new("Data");
    assert_eq!(series.x_range(), None);
}

#[test]
fn test_data_series_y_range() {
    let series = DataSeries::new("Data")
        .point(1.0, 10.0)
        .point(2.0, 30.0)
        .point(3.0, 5.0);
    assert_eq!(series.y_range(), Some((5.0, 30.0)));
}

#[test]
fn test_data_series_y_range_empty() {
    let series = DataSeries::new("Data");
    assert_eq!(series.y_range(), None);
}

// ===== Axis Tests =====

#[test]
fn test_axis_default() {
    let axis = Axis::default();
    assert!(axis.label.is_none());
    assert!(axis.min.is_none());
    assert!(axis.max.is_none());
    assert_eq!(axis.grid_lines, 5);
    assert!(axis.show_grid);
}

#[test]
fn test_axis_label() {
    let axis = Axis::new().label("Time");
    assert_eq!(axis.label, Some("Time".to_string()));
}

#[test]
fn test_axis_min_max() {
    let axis = Axis::new().min(0.0).max(100.0);
    assert_eq!(axis.min, Some(0.0));
    assert_eq!(axis.max, Some(100.0));
}

#[test]
fn test_axis_range() {
    let axis = Axis::new().range(10.0, 50.0);
    assert_eq!(axis.min, Some(10.0));
    assert_eq!(axis.max, Some(50.0));
}

#[test]
fn test_axis_grid_lines() {
    let axis = Axis::new().grid_lines(10);
    assert_eq!(axis.grid_lines, 10);
}

#[test]
fn test_axis_grid_lines_min() {
    let axis = Axis::new().grid_lines(1);
    assert_eq!(axis.grid_lines, 2);
}

#[test]
fn test_axis_show_grid() {
    let axis = Axis::new().show_grid(false);
    assert!(!axis.show_grid);
}

#[test]
fn test_axis_colors() {
    let axis = Axis::new().color(Color::RED).grid_color(Color::BLUE);
    assert_eq!(axis.color, Color::RED);
    assert_eq!(axis.grid_color, Color::BLUE);
}

// ===== LegendPosition Tests =====

#[test]
fn test_legend_position_default() {
    assert_eq!(LegendPosition::default(), LegendPosition::TopRight);
}

// ===== Chart Construction Tests =====

#[test]
fn test_chart_new() {
    let chart = Chart::new();
    assert_eq!(chart.get_chart_type(), ChartType::Line);
    assert_eq!(chart.series_count(), 0);
    assert!(!chart.has_data());
}

#[test]
fn test_chart_line() {
    let chart = Chart::line();
    assert_eq!(chart.get_chart_type(), ChartType::Line);
}

#[test]
fn test_chart_bar() {
    let chart = Chart::bar();
    assert_eq!(chart.get_chart_type(), ChartType::Bar);
}

#[test]
fn test_chart_scatter() {
    let chart = Chart::scatter();
    assert_eq!(chart.get_chart_type(), ChartType::Scatter);
}

#[test]
fn test_chart_area() {
    let chart = Chart::area();
    assert_eq!(chart.get_chart_type(), ChartType::Area);
}

#[test]
fn test_chart_pie() {
    let chart = Chart::pie();
    assert_eq!(chart.get_chart_type(), ChartType::Pie);
}

#[test]
fn test_chart_builder() {
    let chart = Chart::new()
        .chart_type(ChartType::Bar)
        .series(DataSeries::new("Sales").point(1.0, 100.0))
        .series(DataSeries::new("Expenses").point(1.0, 80.0))
        .title("Revenue")
        .x_axis(Axis::new().label("Month"))
        .y_axis(Axis::new().label("Amount"))
        .legend(LegendPosition::BottomRight)
        .background(Color::WHITE)
        .padding(50.0)
        .width(600.0)
        .height(400.0)
        .accessible_name("Revenue chart")
        .test_id("revenue-chart");

    assert_eq!(chart.get_chart_type(), ChartType::Bar);
    assert_eq!(chart.series_count(), 2);
    assert!(chart.has_data());
    assert_eq!(chart.get_title(), Some("Revenue"));
    assert_eq!(Widget::accessible_name(&chart), Some("Revenue chart"));
    assert_eq!(Widget::test_id(&chart), Some("revenue-chart"));
}

#[test]
fn test_chart_add_series() {
    let series_list = vec![DataSeries::new("A"), DataSeries::new("B")];
    let chart = Chart::new().add_series(series_list);
    assert_eq!(chart.series_count(), 2);
}

// ===== Data Bounds Tests =====

#[test]
fn test_chart_data_bounds() {
    let chart = Chart::new()
        .series(DataSeries::new("S1").point(0.0, 10.0).point(5.0, 20.0))
        .series(DataSeries::new("S2").point(1.0, 5.0).point(4.0, 25.0));

    let bounds = chart.data_bounds().unwrap();
    assert_eq!(bounds.0, 0.0); // x_min
    assert_eq!(bounds.1, 5.0); // x_max
    assert_eq!(bounds.2, 5.0); // y_min
    assert_eq!(bounds.3, 25.0); // y_max
}

#[test]
fn test_chart_data_bounds_with_axis_override() {
    let chart = Chart::new()
        .series(DataSeries::new("S1").point(0.0, 10.0).point(5.0, 20.0))
        .x_axis(Axis::new().min(-5.0).max(10.0))
        .y_axis(Axis::new().min(0.0).max(50.0));

    let bounds = chart.data_bounds().unwrap();
    assert_eq!(bounds.0, -5.0); // x_min (overridden)
    assert_eq!(bounds.1, 10.0); // x_max (overridden)
    assert_eq!(bounds.2, 0.0); // y_min (overridden)
    assert_eq!(bounds.3, 50.0); // y_max (overridden)
}

#[test]
fn test_chart_data_bounds_empty() {
    let chart = Chart::new();
    assert!(chart.data_bounds().is_none());
}

// ===== Dimension Tests =====

#[test]
fn test_chart_padding_min() {
    let chart = Chart::new().padding(-10.0);
    assert_eq!(chart.padding, 0.0);
}

#[test]
fn test_chart_width_min() {
    let chart = Chart::new().width(50.0);
    assert_eq!(chart.width, Some(100.0));
}

#[test]
fn test_chart_height_min() {
    let chart = Chart::new().height(50.0);
    assert_eq!(chart.height, Some(100.0));
}

// ===== Widget Trait Tests =====

#[test]
fn test_chart_type_id() {
    let chart = Chart::new();
    assert_eq!(Widget::type_id(&chart), TypeId::of::<Chart>());
}

#[test]
fn test_chart_measure_default() {
    let chart = Chart::new();
    let size = chart.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
    assert_eq!(size.width, 400.0);
    assert_eq!(size.height, 300.0);
}

#[test]
fn test_chart_measure_custom() {
    let chart = Chart::new().width(600.0).height(400.0);
    let size = chart.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
    assert_eq!(size.width, 600.0);
    assert_eq!(size.height, 400.0);
}

#[test]
fn test_chart_layout() {
    let mut chart = Chart::new();
    let bounds = Rect::new(10.0, 20.0, 500.0, 300.0);
    let result = chart.layout(bounds);
    assert_eq!(result.size, Size::new(500.0, 300.0));
    assert_eq!(chart.bounds, bounds);
}

#[test]
fn test_chart_children() {
    let chart = Chart::new();
    assert!(chart.children().is_empty());
}

#[test]
fn test_chart_is_interactive() {
    let chart = Chart::new();
    assert!(!chart.is_interactive());
}

#[test]
fn test_chart_is_focusable() {
    let chart = Chart::new();
    assert!(!chart.is_focusable());
}

#[test]
fn test_chart_accessible_role() {
    let chart = Chart::new();
    assert_eq!(chart.accessible_role(), AccessibleRole::Image);
}

#[test]
fn test_chart_accessible_name_from_title() {
    let chart = Chart::new().title("Sales Chart");
    assert_eq!(Widget::accessible_name(&chart), Some("Sales Chart"));
}

#[test]
fn test_chart_accessible_name_explicit() {
    let chart = Chart::new()
        .title("Sales Chart")
        .accessible_name("Custom name");
    assert_eq!(Widget::accessible_name(&chart), Some("Custom name"));
}

#[test]
fn test_chart_test_id() {
    let chart = Chart::new().test_id("my-chart");
    assert_eq!(Widget::test_id(&chart), Some("my-chart"));
}

// ===== Plot Area Tests =====

#[test]
fn test_chart_plot_area_no_title() {
    let mut chart = Chart::new().padding(40.0);
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let plot = chart.plot_area();
    assert_eq!(plot.x, 40.0);
    assert_eq!(plot.y, 40.0);
    assert_eq!(plot.width, 320.0);
    assert_eq!(plot.height, 220.0);
}

#[test]
fn test_chart_plot_area_with_title() {
    let mut chart = Chart::new().padding(40.0).title("Test");
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let plot = chart.plot_area();
    assert_eq!(plot.y, 70.0); // 40 + 30 for title
}

// ===== Map Point Tests =====

#[test]
fn test_chart_map_point() {
    let chart = Chart::new();
    let bounds = (0.0, 10.0, 0.0, 100.0);
    let plot = Rect::new(0.0, 0.0, 100.0, 100.0);

    let pt = chart.map_point(5.0, 50.0, &bounds, &plot);
    assert!((pt.x - 50.0).abs() < 0.1);
    assert!((pt.y - 50.0).abs() < 0.1);
}

#[test]
fn test_chart_map_point_origin() {
    let chart = Chart::new();
    let bounds = (0.0, 10.0, 0.0, 100.0);
    let plot = Rect::new(0.0, 0.0, 100.0, 100.0);

    let pt = chart.map_point(0.0, 0.0, &bounds, &plot);
    assert!((pt.x - 0.0).abs() < 0.1);
    assert!((pt.y - 100.0).abs() < 0.1); // Y is flipped
}

// ===== Has Data Tests =====

#[test]
fn test_chart_has_data_empty_series() {
    let chart = Chart::new().series(DataSeries::new("Empty"));
    assert!(!chart.has_data());
}

#[test]
fn test_chart_has_data_with_points() {
    let chart = Chart::new().series(DataSeries::new("Data").point(1.0, 1.0));
    assert!(chart.has_data());
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_data_series_eq() {
    let s1 = DataSeries::new("A").point(1.0, 2.0);
    let s2 = DataSeries::new("A").point(1.0, 2.0);
    assert_eq!(s1, s2);
}

#[test]
fn test_chart_type_eq() {
    assert_eq!(ChartType::Line, ChartType::Line);
    assert_ne!(ChartType::Line, ChartType::Bar);
}

#[test]
fn test_legend_position_all_variants() {
    let positions = [
        LegendPosition::None,
        LegendPosition::TopRight,
        LegendPosition::TopLeft,
        LegendPosition::BottomRight,
        LegendPosition::BottomLeft,
    ];
    assert_eq!(positions.len(), 5);
}

#[test]
fn test_chart_children_mut() {
    let mut chart = Chart::new();
    assert!(chart.children_mut().is_empty());
}

#[test]
fn test_chart_event_returns_none() {
    let mut chart = Chart::new();
    let result = chart.event(&presentar_core::Event::key_down(presentar_core::Key::Down));
    assert!(result.is_none());
}

#[test]
fn test_axis_default_colors() {
    let axis = Axis::default();
    assert_eq!(axis.color.a, 1.0);
    assert_eq!(axis.grid_color.a, 1.0);
}

#[test]
fn test_chart_get_series() {
    let chart = Chart::new()
        .series(DataSeries::new("A"))
        .series(DataSeries::new("B"));
    assert_eq!(chart.get_series().len(), 2);
    assert_eq!(chart.get_series()[0].name, "A");
}

#[test]
fn test_chart_histogram() {
    let chart = Chart::new().chart_type(ChartType::Histogram);
    assert_eq!(chart.get_chart_type(), ChartType::Histogram);
}

#[test]
fn test_chart_data_bounds_single_point() {
    let chart = Chart::new().series(DataSeries::new("S").point(5.0, 10.0));
    let bounds = chart.data_bounds().unwrap();
    assert_eq!(bounds.0, 5.0); // x_min
    assert_eq!(bounds.1, 5.0); // x_max (same as min for single point)
}

#[test]
fn test_chart_legend_none() {
    let chart = Chart::new().legend(LegendPosition::None);
    assert_eq!(chart.legend, LegendPosition::None);
}

#[test]
fn test_chart_legend_top_left() {
    let chart = Chart::new().legend(LegendPosition::TopLeft);
    assert_eq!(chart.legend, LegendPosition::TopLeft);
}

#[test]
fn test_chart_legend_bottom_left() {
    let chart = Chart::new().legend(LegendPosition::BottomLeft);
    assert_eq!(chart.legend, LegendPosition::BottomLeft);
}

#[test]
fn test_chart_test_id_none() {
    let chart = Chart::new();
    assert!(Widget::test_id(&chart).is_none());
}

#[test]
fn test_chart_accessible_name_none() {
    let chart = Chart::new();
    assert!(Widget::accessible_name(&chart).is_none());
}

#[test]
fn test_data_series_default_values() {
    let series = DataSeries::new("Test");
    assert_eq!(series.line_width, 2.0);
    assert_eq!(series.point_size, 4.0);
}

// =========================================================================
// Brick Trait Tests
// =========================================================================

#[test]
fn test_chart_brick_name() {
    let chart = Chart::new();
    assert_eq!(chart.brick_name(), "Chart");
}

#[test]
fn test_chart_brick_assertions() {
    let chart = Chart::new();
    let assertions = chart.assertions();
    assert!(!assertions.is_empty());
    assert!(matches!(assertions[0], BrickAssertion::MaxLatencyMs(16)));
}

#[test]
fn test_chart_brick_budget() {
    let chart = Chart::new();
    let budget = chart.budget();
    // Verify budget has reasonable values
    assert!(budget.layout_ms > 0);
    assert!(budget.paint_ms > 0);
}

#[test]
fn test_chart_brick_verify() {
    let chart = Chart::new();
    let verification = chart.verify();
    assert!(!verification.passed.is_empty());
    assert!(verification.failed.is_empty());
}

#[test]
fn test_chart_brick_to_html() {
    let chart = Chart::new().test_id("my-chart").title("Test Chart");
    let html = chart.to_html();
    assert!(html.contains("brick-chart"));
    assert!(html.contains("my-chart"));
    assert!(html.contains("Test Chart"));
}

#[test]
fn test_chart_brick_to_html_default() {
    let chart = Chart::new();
    let html = chart.to_html();
    assert!(html.contains("data-testid=\"chart\""));
    assert!(html.contains("aria-label=\"Chart\""));
}

#[test]
fn test_chart_brick_to_css() {
    let chart = Chart::new();
    let css = chart.to_css();
    assert!(css.contains(".brick-chart"));
    assert!(css.contains("display: block"));
}

#[test]
fn test_chart_brick_test_id() {
    let chart = Chart::new().test_id("chart-1");
    assert_eq!(Brick::test_id(&chart), Some("chart-1"));
}

#[test]
fn test_chart_brick_test_id_none() {
    let chart = Chart::new();
    assert!(Brick::test_id(&chart).is_none());
}

// =========================================================================
// Chart Type Constructor Tests
// =========================================================================

#[test]
fn test_chart_heatmap_constructor() {
    let chart = Chart::heatmap();
    assert_eq!(chart.get_chart_type(), ChartType::Heatmap);
}

#[test]
fn test_chart_boxplot_constructor() {
    let chart = Chart::boxplot();
    assert_eq!(chart.get_chart_type(), ChartType::BoxPlot);
}

// =========================================================================
// Additional Edge Case Tests
// =========================================================================

#[test]
fn test_chart_data_bounds_with_partial_axis_override() {
    // Only x_min overridden
    let chart = Chart::new()
        .series(DataSeries::new("S1").point(0.0, 10.0).point(5.0, 20.0))
        .x_axis(Axis::new().min(-10.0));

    let bounds = chart.data_bounds().unwrap();
    assert_eq!(bounds.0, -10.0); // x_min overridden
    assert_eq!(bounds.1, 5.0); // x_max from data
}

#[test]
fn test_chart_data_bounds_only_y_axis_override() {
    let chart = Chart::new()
        .series(DataSeries::new("S1").point(0.0, 10.0).point(5.0, 20.0))
        .y_axis(Axis::new().max(100.0));

    let bounds = chart.data_bounds().unwrap();
    assert_eq!(bounds.3, 100.0); // y_max overridden
}

#[test]
fn test_chart_map_point_with_zero_range() {
    let chart = Chart::new();
    // Zero range should be clamped to 1e-10
    let bounds = (5.0, 5.0, 10.0, 10.0); // Same values = zero range
    let plot = Rect::new(0.0, 0.0, 100.0, 100.0);

    // Should not panic, uses 1e-10 as min range
    let pt = chart.map_point(5.0, 10.0, &bounds, &plot);
    assert!(pt.x.is_finite());
    assert!(pt.y.is_finite());
}

#[test]
fn test_chart_measure_constrained() {
    let chart = Chart::new().width(800.0).height(600.0);
    // Constrain to smaller size
    let size = chart.measure(Constraints::tight(Size::new(400.0, 300.0)));
    assert_eq!(size.width, 400.0);
    assert_eq!(size.height, 300.0);
}

#[test]
fn test_data_series_x_range_single_point() {
    let series = DataSeries::new("Data").point(5.0, 10.0);
    let range = series.x_range().unwrap();
    assert_eq!(range.0, 5.0);
    assert_eq!(range.1, 5.0);
}

#[test]
fn test_data_series_y_range_single_point() {
    let series = DataSeries::new("Data").point(5.0, 10.0);
    let range = series.y_range().unwrap();
    assert_eq!(range.0, 10.0);
    assert_eq!(range.1, 10.0);
}

#[test]
fn test_axis_new() {
    let axis = Axis::new();
    assert!(axis.label.is_none());
    assert!(axis.min.is_none());
    assert!(axis.max.is_none());
}

#[test]
fn test_chart_type_clone() {
    let ct = ChartType::Histogram;
    let cloned = ct;
    assert_eq!(cloned, ChartType::Histogram);
}

#[test]
fn test_legend_position_eq() {
    assert_eq!(LegendPosition::TopRight, LegendPosition::TopRight);
    assert_ne!(LegendPosition::TopRight, LegendPosition::TopLeft);
}

#[test]
fn test_chart_with_empty_series() {
    let chart = Chart::new()
        .series(DataSeries::new("Empty1"))
        .series(DataSeries::new("Empty2").point(1.0, 2.0));

    assert!(chart.has_data()); // Second series has data
    assert_eq!(chart.series_count(), 2);
}

#[test]
fn test_chart_multiple_series_data_bounds() {
    let chart = Chart::new()
        .series(DataSeries::new("S1").point(-5.0, 0.0).point(0.0, 10.0))
        .series(DataSeries::new("S2").point(0.0, -10.0).point(10.0, 50.0));

    let bounds = chart.data_bounds().unwrap();
    assert_eq!(bounds.0, -5.0); // min x
    assert_eq!(bounds.1, 10.0); // max x
    assert_eq!(bounds.2, -10.0); // min y
    assert_eq!(bounds.3, 50.0); // max y
}

#[test]
fn test_chart_legend_bottom_right() {
    let chart = Chart::new().legend(LegendPosition::BottomRight);
    assert_eq!(chart.legend, LegendPosition::BottomRight);
}

#[test]
fn test_chart_background_setter() {
    let chart = Chart::new().background(Color::BLACK);
    assert_eq!(chart.background, Color::BLACK);
}

#[test]
fn test_data_series_clone() {
    let series = DataSeries::new("Test").point(1.0, 2.0);
    let cloned = series;
    assert_eq!(cloned.name, "Test");
    assert_eq!(cloned.points.len(), 1);
}

#[test]
fn test_axis_clone() {
    let axis = Axis::new().label("X").min(0.0).max(100.0);
    let cloned = axis;
    assert_eq!(cloned.label, Some("X".to_string()));
    assert_eq!(cloned.min, Some(0.0));
    assert_eq!(cloned.max, Some(100.0));
}

#[test]
fn test_chart_clone() {
    let chart = Chart::new()
        .title("Test")
        .series(DataSeries::new("S1").point(1.0, 2.0));
    let cloned = chart;
    assert_eq!(cloned.get_title(), Some("Test"));
    assert_eq!(cloned.series_count(), 1);
}

#[test]
fn test_chart_type_debug() {
    let ct = ChartType::Pie;
    let debug_str = format!("{ct:?}");
    assert!(debug_str.contains("Pie"));
}

#[test]
fn test_legend_position_debug() {
    let lp = LegendPosition::TopRight;
    let debug_str = format!("{lp:?}");
    assert!(debug_str.contains("TopRight"));
}

#[test]
fn test_data_series_debug() {
    let series = DataSeries::new("Test");
    let debug_str = format!("{series:?}");
    assert!(debug_str.contains("Test"));
}

#[test]
fn test_axis_debug() {
    let axis = Axis::new().label("Time");
    let debug_str = format!("{axis:?}");
    assert!(debug_str.contains("Time"));
}

#[test]
fn test_chart_debug() {
    let chart = Chart::new().title("Debug Test");
    let debug_str = format!("{chart:?}");
    assert!(debug_str.contains("Debug Test"));
}

// =========================================================================
// Paint Method Tests (coverage for rendering code)
// =========================================================================

use presentar_core::RecordingCanvas;

#[test]
fn test_chart_paint_empty() {
    let mut chart = Chart::new();
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Should paint background only (no data)
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_with_title() {
    let mut chart = Chart::new().title("My Chart");
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Should paint background and title
    assert!(canvas.commands().len() >= 2);
}

#[test]
fn test_chart_paint_line_chart() {
    let mut chart = Chart::line().series(
        DataSeries::new("Data")
            .point(0.0, 0.0)
            .point(5.0, 10.0)
            .point(10.0, 5.0),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Should have multiple draw commands for grid, line, and points
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_line_chart_no_points() {
    let mut chart = Chart::line().series(
        DataSeries::new("Data")
            .point(0.0, 0.0)
            .point(5.0, 10.0)
            .show_points(false),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_area_chart() {
    let mut chart = Chart::area().series(
        DataSeries::new("Data")
            .point(0.0, 0.0)
            .point(5.0, 10.0)
            .point(10.0, 5.0)
            .fill(true),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_bar_chart() {
    let mut chart = Chart::bar()
        .series(DataSeries::new("A").point(1.0, 10.0).point(2.0, 20.0))
        .series(DataSeries::new("B").point(1.0, 15.0).point(2.0, 25.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_bar_chart_empty_series() {
    let mut chart = Chart::bar();
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Only background painted (no data)
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_scatter_chart() {
    let mut chart = Chart::scatter().series(
        DataSeries::new("Points")
            .point(1.0, 2.0)
            .point(3.0, 4.0)
            .point(5.0, 6.0),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_pie_chart() {
    let mut chart = Chart::pie()
        .series(DataSeries::new("Slice1").point(0.0, 30.0))
        .series(DataSeries::new("Slice2").point(0.0, 50.0))
        .series(DataSeries::new("Slice3").point(0.0, 20.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 3);
}

#[test]
fn test_chart_paint_pie_chart_zero_total() {
    let mut chart = Chart::pie().series(DataSeries::new("Zero").point(0.0, 0.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Pie with zero total skips rendering segments
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_histogram() {
    let mut chart = Chart::new().chart_type(ChartType::Histogram).series(
        DataSeries::new("Data")
            .point(1.0, 5.0)
            .point(2.0, 10.0)
            .point(3.0, 7.0),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_heatmap() {
    let mut chart = Chart::heatmap()
        .series(
            DataSeries::new("Row1")
                .point(0.0, 10.0)
                .point(1.0, 20.0)
                .point(2.0, 30.0),
        )
        .series(
            DataSeries::new("Row2")
                .point(0.0, 15.0)
                .point(1.0, 25.0)
                .point(2.0, 35.0),
        );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Should paint cells for heatmap
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_heatmap_empty() {
    let mut chart = Chart::heatmap();
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Empty heatmap still paints background
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_boxplot() {
    let mut chart = Chart::boxplot().series(
        DataSeries::new("Stats")
            .point(0.0, 1.0)
            .point(0.0, 2.0)
            .point(0.0, 3.0)
            .point(0.0, 4.0)
            .point(0.0, 5.0)
            .point(0.0, 6.0)
            .point(0.0, 7.0),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Boxplot paints whiskers, box, and median
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_boxplot_insufficient_points() {
    let mut chart = Chart::boxplot().series(
        DataSeries::new("TooFew")
            .point(0.0, 1.0)
            .point(0.0, 2.0)
            .point(0.0, 3.0),
    );
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Should skip boxplot for < 5 points
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_boxplot_empty_series() {
    let mut chart = Chart::boxplot();
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_legend_top_right() {
    let mut chart = Chart::new()
        .legend(LegendPosition::TopRight)
        .series(DataSeries::new("Series A").point(1.0, 2.0))
        .series(DataSeries::new("Series B").point(2.0, 3.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Legend should add fill_rect and stroke_rect commands
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_legend_top_left() {
    let mut chart = Chart::new()
        .legend(LegendPosition::TopLeft)
        .series(DataSeries::new("Data").point(1.0, 2.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_legend_bottom_right() {
    let mut chart = Chart::new()
        .legend(LegendPosition::BottomRight)
        .series(DataSeries::new("Data").point(1.0, 2.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_legend_bottom_left() {
    let mut chart = Chart::new()
        .legend(LegendPosition::BottomLeft)
        .series(DataSeries::new("Data").point(1.0, 2.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_legend_none() {
    let mut chart = Chart::new()
        .legend(LegendPosition::None)
        .series(DataSeries::new("Data").point(1.0, 2.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // No legend commands
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_legend_empty_series() {
    let mut chart = Chart::new().legend(LegendPosition::TopRight);
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // No series = no legend
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_grid_hidden() {
    let mut chart = Chart::new()
        .x_axis(Axis::new().show_grid(false))
        .y_axis(Axis::new().show_grid(false))
        .series(DataSeries::new("Data").point(0.0, 0.0).point(10.0, 10.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Grid hidden but axis labels still drawn
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_line_single_point() {
    let mut chart = Chart::line().series(DataSeries::new("Single").point(5.0, 10.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Single point - line skipped but point drawn
    assert!(!canvas.commands().is_empty());
}

#[test]
fn test_chart_paint_multiple_series_line() {
    let mut chart = Chart::line()
        .series(DataSeries::new("A").point(0.0, 0.0).point(5.0, 10.0))
        .series(DataSeries::new("B").point(0.0, 5.0).point(5.0, 15.0))
        .series(DataSeries::new("C").point(0.0, 10.0).point(5.0, 20.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Multiple lines + points + grid
    assert!(canvas.commands().len() > 10);
}

#[test]
fn test_paint_grid_labels() {
    let mut chart = Chart::new()
        .x_axis(Axis::new().grid_lines(3))
        .y_axis(Axis::new().grid_lines(4))
        .series(DataSeries::new("Data").point(0.0, 0.0).point(10.0, 100.0));
    chart.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Grid lines + labels
    assert!(canvas.commands().len() > 5);
}

#[test]
fn test_chart_paint_with_all_options() {
    let mut chart = Chart::new()
        .chart_type(ChartType::Line)
        .title("Full Chart")
        .series(
            DataSeries::new("Main")
                .point(0.0, 0.0)
                .point(5.0, 50.0)
                .point(10.0, 30.0)
                .color(Color::RED)
                .line_width(3.0)
                .point_size(6.0)
                .show_points(true)
                .fill(true),
        )
        .x_axis(Axis::new().label("X").min(-5.0).max(15.0).grid_lines(4))
        .y_axis(Axis::new().label("Y").min(-10.0).max(60.0).grid_lines(5))
        .legend(LegendPosition::TopRight)
        .background(Color::WHITE)
        .padding(50.0);
    chart.bounds = Rect::new(0.0, 0.0, 500.0, 400.0);
    let mut canvas = RecordingCanvas::new();
    chart.paint(&mut canvas);
    // Should have many commands for all elements
    assert!(canvas.commands().len() > 15);
}
