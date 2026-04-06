use super::*;

#[test]
fn test_rgb_color_creation() {
    let color = Rgb::new(255, 128, 64);
    assert_eq!(color.r, 255);
    assert_eq!(color.g, 128);
    assert_eq!(color.b, 64);
}

#[test]
fn test_rgb_special_colors() {
    assert_eq!(Rgb::NAN_COLOR, Rgb::new(255, 0, 255));
    assert_eq!(Rgb::INF_COLOR, Rgb::new(255, 255, 255));
    assert_eq!(Rgb::NEG_INF_COLOR, Rgb::new(0, 0, 0));
}

#[test]
fn test_color_palette_viridis() {
    let palette = ColorPalette::viridis();
    assert_eq!(palette.colors.len(), 5);

    // Test interpolation at boundaries
    let at_0 = palette.interpolate(0.0);
    let at_1 = palette.interpolate(1.0);

    // Viridis starts dark purple, ends yellow
    assert_eq!(at_0, Rgb::new(68, 1, 84));
    assert_eq!(at_1, Rgb::new(253, 231, 37));
}

#[test]
fn test_color_palette_grayscale() {
    let palette = ColorPalette::grayscale();
    assert_eq!(palette.colors.len(), 3);

    let at_0 = palette.interpolate(0.0);
    let at_1 = palette.interpolate(1.0);

    assert_eq!(at_0, Rgb::new(0, 0, 0));
    assert_eq!(at_1, Rgb::new(255, 255, 255));
}

#[test]
fn test_color_palette_interpolation_midpoint() {
    let palette = ColorPalette::grayscale();
    let at_mid = palette.interpolate(0.5);

    // Should be close to gray
    assert_eq!(at_mid, Rgb::new(128, 128, 128));
}

#[test]
fn test_color_palette_clamping() {
    let palette = ColorPalette::viridis();

    // Values outside [0, 1] should be clamped
    let at_neg = palette.interpolate(-0.5);
    let at_over = palette.interpolate(1.5);

    assert_eq!(at_neg, palette.interpolate(0.0));
    assert_eq!(at_over, palette.interpolate(1.0));
}

#[test]
fn test_visual_regression_config_default() {
    let config = VisualRegressionConfig::default();

    assert_eq!(config.golden_dir, PathBuf::from("golden"));
    assert_eq!(config.output_dir, PathBuf::from("test_output"));
    assert_eq!(config.max_diff_pct, 0.0);
}

#[test]
fn test_visual_regression_config_builder() {
    let config = VisualRegressionConfig::new("my_golden")
        .with_output_dir("my_output")
        .with_max_diff_pct(1.5)
        .with_palette(ColorPalette::grayscale());

    assert_eq!(config.golden_dir, PathBuf::from("my_golden"));
    assert_eq!(config.output_dir, PathBuf::from("my_output"));
    assert_eq!(config.max_diff_pct, 1.5);
}

#[test]
fn test_pixel_diff_result_percentage() {
    let result = PixelDiffResult { different_pixels: 10, total_pixels: 100, max_diff: 50 };

    assert_eq!(result.diff_percentage(), 10.0);
    assert!(!result.matches(5.0));
    assert!(result.matches(10.0));
    assert!(result.matches(15.0));
}

#[test]
fn test_pixel_diff_result_zero_total() {
    let result = PixelDiffResult { different_pixels: 0, total_pixels: 0, max_diff: 0 };

    assert_eq!(result.diff_percentage(), 0.0);
}

#[test]
fn test_pixel_diff_result_pass() {
    let result = PixelDiffResult::pass(100);

    assert_eq!(result.different_pixels, 0);
    assert_eq!(result.total_pixels, 100);
    assert_eq!(result.max_diff, 0);
    assert!(result.matches(0.0));
}

#[test]
fn test_buffer_renderer_default() {
    let renderer = BufferRenderer::default();
    assert!(renderer.range.is_none());
}

#[test]
fn test_buffer_renderer_with_range() {
    let renderer = BufferRenderer::new().with_range(0.0, 10.0);
    assert_eq!(renderer.range, Some((0.0, 10.0)));
}

#[test]
fn test_buffer_renderer_with_palette() {
    let renderer = BufferRenderer::new().with_palette(ColorPalette::grayscale());
    assert_eq!(renderer.palette.colors.len(), 3);
}

#[test]
fn test_buffer_renderer_rgba_output() {
    let renderer = BufferRenderer::new();
    let buffer: Vec<f32> = (0..4).map(|i| i as f32 / 3.0).collect();
    let rgba = renderer.render_to_rgba(&buffer, 2, 2);

    // 4 pixels * 4 bytes = 16 bytes
    assert_eq!(rgba.len(), 16);

    // Check alpha channel is always 255
    for i in (3..16).step_by(4) {
        assert_eq!(rgba[i], 255);
    }
}

#[test]
fn test_buffer_renderer_nan_handling() {
    let renderer = BufferRenderer::new();
    let buffer = vec![0.0, f32::NAN, 1.0, 0.5];
    let rgba = renderer.render_to_rgba(&buffer, 2, 2);

    // Second pixel should be NAN_COLOR (magenta: 255, 0, 255)
    assert_eq!(rgba[4], 255); // R
    assert_eq!(rgba[5], 0); // G
    assert_eq!(rgba[6], 255); // B
    assert_eq!(rgba[7], 255); // A
}

#[test]
fn test_buffer_renderer_inf_handling() {
    let renderer = BufferRenderer::new();
    let buffer = vec![f32::INFINITY, f32::NEG_INFINITY, 0.5, 0.5];
    let rgba = renderer.render_to_rgba(&buffer, 2, 2);

    // First pixel: +INF should be white
    assert_eq!(rgba[0], 255);
    assert_eq!(rgba[1], 255);
    assert_eq!(rgba[2], 255);

    // Second pixel: -INF should be black
    assert_eq!(rgba[4], 0);
    assert_eq!(rgba[5], 0);
    assert_eq!(rgba[6], 0);
}

#[test]
fn test_buffer_renderer_compare_identical() {
    let renderer = BufferRenderer::new();
    let buffer: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    let rgba = renderer.render_to_rgba(&buffer, 4, 4);

    let result = renderer.compare_rgba(&rgba, &rgba, 0);
    assert_eq!(result.different_pixels, 0);
    assert!(result.matches(0.0));
}

#[test]
fn test_buffer_renderer_compare_different() {
    let renderer = BufferRenderer::new();
    let buffer_a: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    let buffer_b: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 / 15.0).collect();

    let rgba_a = renderer.render_to_rgba(&buffer_a, 4, 4);
    let rgba_b = renderer.render_to_rgba(&buffer_b, 4, 4);

    let result = renderer.compare_rgba(&rgba_a, &rgba_b, 0);
    assert!(result.different_pixels > 0);
}

#[test]
fn test_buffer_renderer_compare_with_tolerance() {
    let renderer = BufferRenderer::new();
    let rgba_a = vec![100, 100, 100, 255];
    let rgba_b = vec![105, 102, 98, 255];

    // With tolerance 10, should match
    let result = renderer.compare_rgba(&rgba_a, &rgba_b, 10);
    assert_eq!(result.different_pixels, 0);

    // With tolerance 1, should differ
    let result_strict = renderer.compare_rgba(&rgba_a, &rgba_b, 1);
    assert!(result_strict.different_pixels > 0);
}

#[test]
fn test_golden_baseline_paths() {
    let config = VisualRegressionConfig::new("/test/golden").with_output_dir("/test/output");
    let baseline = GoldenBaseline::new(config);

    assert_eq!(baseline.golden_path("relu_4x4"), PathBuf::from("/test/golden/relu_4x4.golden"));
    assert_eq!(baseline.output_path("relu_4x4"), PathBuf::from("/test/output/relu_4x4.output"));
}

#[test]
fn test_golden_baseline_config_access() {
    let config = VisualRegressionConfig::new("/golden").with_max_diff_pct(2.5);
    let baseline = GoldenBaseline::new(config);

    assert_eq!(baseline.config().max_diff_pct, 2.5);
}
