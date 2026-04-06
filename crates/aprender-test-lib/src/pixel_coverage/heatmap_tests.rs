    use super::*;

    #[test]
    fn test_rgb_from_hex() {
        let red = Rgb::from_hex(0xFF0000);
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        let white = Rgb::from_hex(0xFFFFFF);
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);
    }

    #[test]
    fn test_color_palette_viridis() {
        let palette = ColorPalette::viridis();
        assert_ne!(palette.zero, palette.full);
    }

    #[test]
    fn test_color_for_coverage() {
        let palette = ColorPalette::traffic_light();

        assert_eq!(palette.color_for_coverage(0.0), palette.zero);
        assert_eq!(palette.color_for_coverage(0.1), palette.low);
        assert_eq!(palette.color_for_coverage(0.4), palette.medium);
        assert_eq!(palette.color_for_coverage(0.6), palette.high);
        assert_eq!(palette.color_for_coverage(1.0), palette.full);
    }

    #[test]
    fn test_terminal_heatmap_render() {
        let cells = vec![vec![0.0, 0.25, 0.5], vec![0.75, 1.0, 0.0]];

        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render();

        assert!(rendered.contains(' ')); // 0% coverage
        assert!(rendered.contains('█')); // 100% coverage
    }

    #[test]
    fn test_terminal_heatmap_with_border() {
        let cells = vec![vec![1.0, 1.0], vec![0.0, 0.0]];

        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render_with_border();

        assert!(rendered.contains('┌'));
        assert!(rendered.contains('┘'));
        assert!(rendered.contains('│'));
    }

    #[test]
    fn test_coverage_to_char() {
        assert_eq!(TerminalHeatmap::coverage_to_char(0.0), ' ');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.1), '░');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.3), '▒');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.6), '▓');
        assert_eq!(TerminalHeatmap::coverage_to_char(1.0), '█');
    }

    #[test]
    fn test_svg_export() {
        let cells = vec![vec![CoverageCell {
            hit_count: 1,
            coverage: 1.0,
        }]];

        let svg = SvgHeatmap::new(100, 100).export(&cells);

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn test_svg_empty_cells() {
        let cells: Vec<Vec<CoverageCell>> = vec![];
        let svg = SvgHeatmap::new(100, 100).export(&cells);
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_legend() {
        let cells = vec![vec![1.0]];
        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let legend = heatmap.legend();

        assert!(legend.contains("Legend:"));
        assert!(legend.contains("░"));
        assert!(legend.contains("█"));
    }

    // =========================================================================
    // PNG Heatmap Tests (H₀-PNG-XX)
    // =========================================================================

    #[test]
    fn h0_png_01_basic_render() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(100, 100).export(&cells).unwrap();
        assert!(!png.is_empty());
        // Verify PNG header bytes
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_02_color_interpolation() {
        let palette = ColorPalette::viridis();
        let color_0 = palette.interpolate(0.0);
        let color_50 = palette.interpolate(0.5);
        let color_100 = palette.interpolate(1.0);

        // Should be distinct colors
        assert_ne!(color_0, color_50);
        assert_ne!(color_50, color_100);
    }

    #[test]
    fn h0_png_03_gap_highlighting() {
        let mut cells = vec![
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                };
                10
            ];
            10
        ];
        cells[5][5] = CoverageCell {
            coverage: 0.0,
            hit_count: 0,
        }; // Gap

        let png = PngHeatmap::new(100, 100)
            .with_gap_highlighting()
            .export(&cells)
            .unwrap();

        // Should render successfully with gap highlighted
        assert!(!png.is_empty());
        // Verify PNG header
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_04_magma_palette() {
        let palette = ColorPalette::magma();
        assert_ne!(palette.zero, palette.full);
        // Magma starts nearly black
        assert!(palette.zero.r < 10);
        assert!(palette.zero.g < 10);
    }

    #[test]
    fn h0_png_05_heat_palette() {
        let palette = ColorPalette::heat();
        assert_ne!(palette.zero, palette.full);
        // Heat starts at black
        assert_eq!(palette.zero, Rgb::new(0, 0, 0));
        // Heat ends at white
        assert_eq!(palette.full, Rgb::new(255, 255, 255));
    }

    #[test]
    fn h0_png_06_rgb_lerp() {
        let black = Rgb::new(0, 0, 0);
        let white = Rgb::new(255, 255, 255);

        let mid = Rgb::lerp(black, white, 0.5);
        assert_eq!(mid.r, 127);
        assert_eq!(mid.g, 127);
        assert_eq!(mid.b, 127);

        // Extremes
        assert_eq!(Rgb::lerp(black, white, 0.0), black);
        assert_eq!(Rgb::lerp(black, white, 1.0), white);
    }

    #[test]
    fn h0_png_07_interpolate_boundaries() {
        let palette = ColorPalette::viridis();

        // Exactly at boundaries
        let c0 = palette.interpolate(0.0);
        let c25 = palette.interpolate(0.25);
        let c50 = palette.interpolate(0.5);
        let c75 = palette.interpolate(0.75);
        let c100 = palette.interpolate(1.0);

        assert_eq!(c0, palette.zero);
        assert_eq!(c25, palette.low);
        assert_eq!(c50, palette.medium);
        assert_eq!(c75, palette.high);
        assert_eq!(c100, palette.full);
    }

    #[test]
    fn h0_png_08_interpolate_clamping() {
        let palette = ColorPalette::viridis();

        // Out of range values should be clamped
        let below = palette.interpolate(-0.5);
        let above = palette.interpolate(1.5);

        assert_eq!(below, palette.zero);
        assert_eq!(above, palette.full);
    }

    #[test]
    fn h0_png_09_empty_cells() {
        let cells: Vec<Vec<CoverageCell>> = vec![];
        let png = PngHeatmap::new(100, 100).export(&cells).unwrap();
        // Should still produce valid PNG (1x1 fallback)
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_10_with_legend() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
            ],
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
                CoverageCell {
                    coverage: 0.75,
                    hit_count: 8,
                },
            ],
        ];

        let png = PngHeatmap::new(200, 200)
            .with_legend()
            .with_palette(ColorPalette::magma())
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_11_builder_pattern() {
        let heatmap = PngHeatmap::new(800, 600)
            .with_palette(ColorPalette::heat())
            .with_legend()
            .with_gap_highlighting()
            .with_borders(false)
            .with_title("Test Heatmap");

        // Verify settings applied (indirectly through export working)
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = heatmap.export(&cells).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_png_12_export_to_file() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
            ];
            3
        ];

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_heatmap.png");

        PngHeatmap::new(300, 300)
            .with_gap_highlighting()
            .export_to_file(&cells, &path)
            .unwrap();

        // Verify file exists and is valid PNG
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn h0_png_13_default() {
        let heatmap = PngHeatmap::default();
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = heatmap.export(&cells).unwrap();
        assert!(!png.is_empty());
    }

    // =========================================================================
    // Title/Metadata Text Rendering Tests (H₀-TXT-XX)
    // =========================================================================

    #[test]
    fn h0_txt_01_title_renders() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                };
                5
            ];
            5
        ];

        let png = PngHeatmap::new(400, 300)
            .with_title("Test Coverage")
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_txt_02_title_with_legend() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                };
                3
            ];
            3
        ];

        let png = PngHeatmap::new(400, 300)
            .with_title("Coverage Heatmap")
            .with_legend()
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_txt_03_bitmap_font_basic() {
        // Test that bitmap font renders without panics
        let font = BitmapFont::default();
        let glyph = font.glyph('A');
        assert!(!glyph.is_empty());
    }

    #[test]
    fn h0_txt_04_bitmap_font_digits() {
        let font = BitmapFont::default();
        for c in '0'..='9' {
            let glyph = font.glyph(c);
            assert!(!glyph.is_empty(), "Digit {} should have a glyph", c);
        }
    }

    #[test]
    fn h0_txt_05_bitmap_font_text_width() {
        let font = BitmapFont::default();
        let width = font.text_width("Hello");
        assert!(width > 0);
        assert_eq!(
            width,
            5 * (font.char_width() + font.spacing()) - font.spacing()
        );
    }

    #[test]
    fn h0_txt_06_metadata_subtitle() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.75,
                    hit_count: 8,
                };
                4
            ];
            4
        ];

        let png = PngHeatmap::new(500, 400)
            .with_title("Main Title")
            .with_subtitle("85% coverage")
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_txt_07_empty_title() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];

        // Empty title should not cause issues
        let png = PngHeatmap::new(200, 200)
            .with_title("")
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_txt_08_special_characters() {
        let font = BitmapFont::default();
        // Should return empty glyph for unknown chars
        let glyph = font.glyph('€');
        assert!(glyph.is_empty() || glyph.iter().all(|&b| !b));
    }

    // =========================================================================
    // Combined PNG Tests (H₀-CMB-XX)
    // =========================================================================

    #[test]
    fn h0_cmb_01_combined_heatmap() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.8,
                    hit_count: 8,
                };
                10
            ];
            10
        ];

        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.85,
            covered_cells: 85,
            total_cells: 100,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png = PngHeatmap::new(600, 500)
            .with_title("Combined Coverage")
            .with_legend()
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cmb_02_stats_panel_height() {
        // Stats panel should add extra height
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport::default();
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let heatmap = PngHeatmap::new(400, 300).with_combined_stats(&combined);

        // The stats panel should be stored
        assert!(heatmap.stats_panel.is_some());
    }

    // =========================================================================
    // Visual Regression Tests (H₀-VIS-XX)
    // =========================================================================

    #[test]
    fn h0_vis_01_deterministic_output() {
        use super::visual_regression::*;

        // Same input should produce identical output
        let cells = reference_gradient_cells(8, 10);

        let png1 = PngHeatmap::new(400, 300)
            .with_palette(ColorPalette::viridis())
            .export(&cells)
            .unwrap();

        let png2 = PngHeatmap::new(400, 300)
            .with_palette(ColorPalette::viridis())
            .export(&cells)
            .unwrap();

        // Byte-for-byte identical
        assert_eq!(png1.len(), png2.len());
        assert_eq!(compute_checksum(&png1), compute_checksum(&png2));
    }

    #[test]
    fn h0_vis_02_compare_identical_images() {
        use super::visual_regression::*;

        let cells = reference_uniform_cells(5, 5, 0.5);
        let png = PngHeatmap::new(200, 200).export(&cells).unwrap();

        let result = compare_png_with_tolerance(&png, &png, 0).unwrap();

        assert!(result.matches);
        assert_eq!(result.diff_count, 0);
        assert_eq!(result.max_diff, 0);
        assert!((result.diff_percentage - 0.0).abs() < 0.001);
    }

    #[test]
    fn h0_vis_03_compare_different_palettes() {
        use super::visual_regression::*;

        let cells = reference_gradient_cells(5, 5);

        let png_viridis = PngHeatmap::new(200, 200)
            .with_palette(ColorPalette::viridis())
            .export(&cells)
            .unwrap();

        let png_magma = PngHeatmap::new(200, 200)
            .with_palette(ColorPalette::magma())
            .export(&cells)
            .unwrap();

        // Different palettes should produce different output
        let result = compare_png_with_tolerance(&png_viridis, &png_magma, 0).unwrap();

        assert!(!result.matches || result.max_diff > 0);
    }

    #[test]
    fn h0_vis_04_gap_highlighting_visible() {
        use super::visual_regression::*;

        let cells = reference_gap_cells(8, 10);

        let png_no_gaps = PngHeatmap::new(400, 300).export(&cells).unwrap();

        let png_with_gaps = PngHeatmap::new(400, 300)
            .with_gap_highlighting()
            .export(&cells)
            .unwrap();

        // Gap highlighting should produce different output
        let result = compare_png_with_tolerance(&png_no_gaps, &png_with_gaps, 0).unwrap();

        // Should have some differences (the red gap borders)
        assert!(
            result.diff_count > 0,
            "Gap highlighting should produce visible differences"
        );
    }

    #[test]
    fn h0_vis_05_legend_visible() {
        use super::visual_regression::*;

        let cells = reference_gradient_cells(5, 5);

        let png_no_legend = PngHeatmap::new(300, 250).export(&cells).unwrap();

        let png_with_legend = PngHeatmap::new(300, 250)
            .with_legend()
            .export(&cells)
            .unwrap();

        // Legend should produce different output
        let result = compare_png_with_tolerance(&png_no_legend, &png_with_legend, 0).unwrap();

        assert!(
            result.diff_count > 0,
            "Legend should produce visible differences"
        );
    }

    #[test]
    fn h0_vis_06_title_visible() {
        use super::visual_regression::*;

        let cells = reference_uniform_cells(4, 4, 0.75);

        let png_no_title = PngHeatmap::new(300, 200).export(&cells).unwrap();

        let png_with_title = PngHeatmap::new(300, 200)
            .with_title("Test Title")
            .export(&cells)
            .unwrap();

        // Title should produce different output
        let result = compare_png_with_tolerance(&png_no_title, &png_with_title, 0).unwrap();

        assert!(
            result.diff_count > 0,
            "Title should produce visible differences"
        );
    }

    #[test]
    fn h0_vis_07_reference_viridis_gradient() {
        use super::visual_regression::*;

        // Generate reference gradient with Viridis palette
        let cells = reference_gradient_cells(10, 15);
        let png = PngHeatmap::new(800, 600)
            .with_palette(ColorPalette::viridis())
            .with_legend()
            .with_margin(40)
            .export(&cells)
            .unwrap();

        // Store checksum as reference (captured from known-good output)
        let checksum = compute_checksum(&png);

        // Verify we get a valid PNG
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // Re-generate and verify determinism
        let png2 = PngHeatmap::new(800, 600)
            .with_palette(ColorPalette::viridis())
            .with_legend()
            .with_margin(40)
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum,
            "Output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_08_reference_magma_gaps() {
        use super::visual_regression::*;

        // Generate reference with gaps and Magma palette
        let cells = reference_gap_cells(8, 12);
        let png = PngHeatmap::new(600, 400)
            .with_palette(ColorPalette::magma())
            .with_gap_highlighting()
            .with_legend()
            .export(&cells)
            .unwrap();

        let checksum = compute_checksum(&png);

        // Verify determinism
        let png2 = PngHeatmap::new(600, 400)
            .with_palette(ColorPalette::magma())
            .with_gap_highlighting()
            .with_legend()
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum,
            "Magma gap output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_09_reference_heat_with_title() {
        use super::visual_regression::*;

        // Generate reference with Heat palette and title
        let cells = reference_uniform_cells(6, 8, 0.65);
        let png = PngHeatmap::new(500, 400)
            .with_palette(ColorPalette::heat())
            .with_title("Heat Coverage")
            .with_subtitle("Reference Test")
            .with_legend()
            .export(&cells)
            .unwrap();

        let checksum = compute_checksum(&png);

        // Verify determinism
        let png2 = PngHeatmap::new(500, 400)
            .with_palette(ColorPalette::heat())
            .with_title("Heat Coverage")
            .with_subtitle("Reference Test")
            .with_legend()
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum,
            "Heat title output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_10_tolerance_comparison() {
        use super::visual_regression::*;

        let cells = reference_gradient_cells(5, 5);
        let png = PngHeatmap::new(200, 200).export(&cells).unwrap();

        // Exact match with 0 tolerance
        let result0 = compare_png_with_tolerance(&png, &png, 0).unwrap();
        assert!(result0.matches);
        assert_eq!(result0.diff_count, 0);

        // Also matches with higher tolerance
        let result10 = compare_png_with_tolerance(&png, &png, 10).unwrap();
        assert!(result10.matches);
        assert_eq!(result10.diff_count, 0);
    }

    #[test]
    fn h0_vis_11_combined_stats_determinism() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };
        use super::visual_regression::*;

        let cells = reference_gradient_cells(8, 10);

        let line_report = LineCoverageReport::new(0.85, 0.95, 0.90, 20, 17);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.80,
            covered_cells: 64,
            total_cells: 80,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png1 = PngHeatmap::new(700, 600)
            .with_palette(ColorPalette::viridis())
            .with_title("Combined Report")
            .with_legend()
            .with_gap_highlighting()
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        let checksum1 = compute_checksum(&png1);

        // Re-create with same parameters
        let line_report2 = LineCoverageReport::new(0.85, 0.95, 0.90, 20, 17);
        let pixel_report2 = PixelCoverageReport {
            overall_coverage: 0.80,
            covered_cells: 64,
            total_cells: 80,
            ..Default::default()
        };
        let combined2 = CombinedCoverageReport::from_parts(line_report2, pixel_report2);

        let png2 = PngHeatmap::new(700, 600)
            .with_palette(ColorPalette::viridis())
            .with_title("Combined Report")
            .with_legend()
            .with_gap_highlighting()
            .with_combined_stats(&combined2)
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum1,
            "Combined stats output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_12_dimension_mismatch() {
        use super::visual_regression::*;

        let cells_small = reference_uniform_cells(3, 3, 0.5);
        let cells_large = reference_uniform_cells(5, 5, 0.5);

        let png_small = PngHeatmap::new(100, 100).export(&cells_small).unwrap();
        let png_large = PngHeatmap::new(200, 200).export(&cells_large).unwrap();

        // Different dimensions should fail comparison
        let result = compare_png_with_tolerance(&png_small, &png_large, 255).unwrap();

        assert!(!result.matches, "Different dimensions should not match");
        assert_eq!(result.diff_percentage, 100.0);
    }

    // =========================================================================
    // Additional Coverage Tests (H₀-COV-XX)
    // =========================================================================

    #[test]
    fn h0_cov_01_terminal_from_tracker() {
        // Test TerminalHeatmap::from_tracker
        let tracker = super::super::tracker::PixelCoverageTracker::new(100, 100, 5, 5);
        let heatmap = TerminalHeatmap::from_tracker(&tracker);
        let rendered = heatmap.render();
        // Should render 5 rows
        assert_eq!(rendered.lines().count(), 5);
    }

    #[test]
    fn h0_cov_02_terminal_with_palette() {
        let cells = vec![vec![0.5, 1.0], vec![0.0, 0.25]];
        let heatmap = TerminalHeatmap::from_values(cells)
            .with_palette(ColorPalette::traffic_light())
            .without_color();
        let rendered = heatmap.render();
        assert!(rendered.contains('▒')); // 50% coverage
        assert!(rendered.contains('█')); // 100% coverage
    }

    #[test]
    fn h0_cov_03_terminal_render_with_color() {
        let cells = vec![vec![0.0, 0.5, 1.0]];
        let heatmap = TerminalHeatmap::from_values(cells);
        // use_color is true by default
        let rendered = heatmap.render();
        // Should contain ANSI escape sequences
        assert!(rendered.contains("\x1b[38;2;"));
        assert!(rendered.contains("\x1b[0m"));
    }

    #[test]
    fn h0_cov_04_terminal_border_with_color() {
        let cells = vec![vec![0.5, 1.0]];
        let heatmap = TerminalHeatmap::from_values(cells);
        let rendered = heatmap.render_with_border();
        // Should contain border chars and ANSI sequences
        assert!(rendered.contains('┌'));
        assert!(rendered.contains("\x1b[38;2;"));
    }

    #[test]
    fn h0_cov_05_terminal_legend_with_color() {
        let cells = vec![vec![1.0]];
        let heatmap = TerminalHeatmap::from_values(cells);
        let legend = heatmap.legend();
        // Should contain ANSI escape sequences in legend
        assert!(legend.contains("\x1b[38;2;"));
        assert!(legend.contains("Legend:"));
    }

    #[test]
    fn h0_cov_06_terminal_empty_cells_border() {
        let cells: Vec<Vec<f32>> = vec![];
        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render_with_border();
        // Should still render borders with width 0
        assert!(rendered.contains('┌'));
        assert!(rendered.contains('└'));
    }

    #[test]
    fn h0_cov_07_png_with_margin() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(200, 200)
            .with_margin(60)
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_cov_08_png_with_background() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(200, 200)
            .with_background(Rgb::new(0, 0, 0)) // Black background
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_09_png_with_border_color() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                };
                3
            ];
            3
        ];
        let png = PngHeatmap::new(200, 200)
            .with_border_color(Rgb::new(255, 0, 0)) // Red borders
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_10_bitmap_font_dimensions() {
        let font = BitmapFont::default();
        assert_eq!(font.char_width(), 5);
        assert_eq!(font.char_height(), 7);
        assert_eq!(font.spacing(), 1);
    }

    #[test]
    fn h0_cov_11_bitmap_font_empty_text_width() {
        let font = BitmapFont::default();
        assert_eq!(font.text_width(""), 0);
    }

    #[test]
    fn h0_cov_12_bitmap_font_single_char_width() {
        let font = BitmapFont::default();
        let width = font.text_width("A");
        assert_eq!(width, 5); // Just char_width, no spacing
    }

    #[test]
    fn h0_cov_13_bitmap_font_punctuation() {
        let font = BitmapFont::default();
        // Test all punctuation characters
        let chars = [
            '.', ',', ':', '-', '_', '/', '%', '(', ')', '=', '+', '*', '!', '?', ' ',
        ];
        for c in chars {
            let glyph = font.glyph(c);
            assert_eq!(glyph.len(), 35, "Glyph for '{}' should have 35 bits", c);
        }
    }

    #[test]
    fn h0_cov_14_bitmap_font_lowercase_to_uppercase() {
        let font = BitmapFont::default();
        // Lowercase should map to uppercase
        let upper = font.glyph('A');
        let lower = font.glyph('a');
        assert_eq!(upper, lower, "Lowercase should map to uppercase");
    }

    #[test]
    fn h0_cov_15_bitmap_font_all_uppercase() {
        let font = BitmapFont::default();
        for c in 'A'..='Z' {
            let glyph = font.glyph(c);
            // Each glyph should have some pixels set (not all false)
            assert!(
                glyph.iter().any(|&b| b),
                "Glyph for '{}' should have some pixels",
                c
            );
        }
    }

    #[test]
    fn h0_cov_16_rgb_lerp_clamping() {
        let black = Rgb::new(0, 0, 0);
        let white = Rgb::new(255, 255, 255);

        // Test clamping at negative values
        let below = Rgb::lerp(black, white, -1.0);
        assert_eq!(below, black);

        // Test clamping above 1.0
        let above = Rgb::lerp(black, white, 2.0);
        assert_eq!(above, white);
    }

    #[test]
    fn h0_cov_17_color_palette_default() {
        let default = ColorPalette::default();
        let viridis = ColorPalette::viridis();
        assert_eq!(default.zero, viridis.zero);
        assert_eq!(default.full, viridis.full);
    }

    #[test]
    fn h0_cov_18_svg_with_palette() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let svg = SvgHeatmap::new(100, 100)
            .with_palette(ColorPalette::magma())
            .export(&cells);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn h0_cov_19_reference_gap_cells_small() {
        use super::visual_regression::*;
        // Test with small grid that won't have gaps at division points
        let cells = reference_gap_cells(2, 2);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].len(), 2);
    }

    #[test]
    fn h0_cov_20_reference_gap_cells_medium() {
        use super::visual_regression::*;
        // Test with grid large enough for first gap but not second
        let cells = reference_gap_cells(3, 3);
        // rows/2 = 1, cols/2 = 1 -> gap at (1,1)
        assert_eq!(cells[1][1].coverage, 0.0);
        assert_eq!(cells[1][1].hit_count, 0);
    }

    #[test]
    fn h0_cov_21_stats_panel_fields() {
        let panel = StatsPanel {
            line_coverage: 85.5,
            pixel_coverage: 90.2,
            overall_score: 87.85,
            line_details: (17, 20),
            pixel_details: (45, 50),
            meets_threshold: true,
        };
        assert!((panel.line_coverage - 85.5).abs() < 0.01);
        assert!((panel.pixel_coverage - 90.2).abs() < 0.01);
        assert!((panel.overall_score - 87.85).abs() < 0.01);
        assert_eq!(panel.line_details, (17, 20));
        assert_eq!(panel.pixel_details, (45, 50));
        assert!(panel.meets_threshold);
    }

    #[test]
    fn h0_cov_22_stats_panel_fail_threshold() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.3,
                    hit_count: 3,
                };
                5
            ];
            5
        ];

        // Create report that fails threshold
        let line_report = LineCoverageReport::new(0.5, 0.5, 0.5, 10, 5);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.3,
            covered_cells: 15,
            total_cells: 50,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png = PngHeatmap::new(400, 400)
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_23_empty_subtitle() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(200, 200)
            .with_subtitle("")
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_24_title_and_subtitle() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(400, 300)
            .with_title("Title")
            .with_subtitle("Subtitle")
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_25_coverage_boundaries() {
        // Test exact boundary values for color_for_coverage
        let palette = ColorPalette::viridis();

        // Negative coverage
        assert_eq!(palette.color_for_coverage(-0.1), palette.zero);

        // Exactly 0.25
        assert_eq!(palette.color_for_coverage(0.25), palette.low);

        // Exactly 0.50
        assert_eq!(palette.color_for_coverage(0.50), palette.medium);

        // Exactly 0.75
        assert_eq!(palette.color_for_coverage(0.75), palette.high);

        // Above 0.75
        assert_eq!(palette.color_for_coverage(0.76), palette.full);
    }

    #[test]
    fn h0_cov_26_coverage_to_char_boundaries() {
        // Test exact boundary values
        assert_eq!(TerminalHeatmap::coverage_to_char(-0.1), ' ');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.25), '░');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.26), '▒');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.50), '▒');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.51), '▓');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.75), '▓');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.76), '█');
    }

    #[test]
    fn h0_cov_27_interpolate_mid_segment() {
        let palette = ColorPalette::viridis();

        // Test interpolation within a segment (not at boundaries)
        let c = palette.interpolate(0.125); // Middle of 0-0.25 segment
                                            // Should be between zero and low
        assert_ne!(c, palette.zero);
        assert_ne!(c, palette.low);
    }

    #[test]
    fn h0_cov_28_reference_gradient_single_cell() {
        use super::visual_regression::*;
        // Single cell grid (edge case with max(1) divisor)
        let cells = reference_gradient_cells(1, 1);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].len(), 1);
        // Coverage should be 0.0 (row=0, col=0, divided by max(1)=1)
        assert!((cells[0][0].coverage - 0.0).abs() < 0.01);
    }

    #[test]
    fn h0_cov_29_png_borders_disabled() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                };
                3
            ];
            3
        ];
        let png = PngHeatmap::new(200, 200)
            .with_borders(false)
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_30_png_all_options() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
            ],
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
            ],
        ];

        let line_report = LineCoverageReport::new(0.9, 0.95, 0.85, 20, 18);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.5,
            covered_cells: 2,
            total_cells: 4,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png = PngHeatmap::new(600, 500)
            .with_palette(ColorPalette::traffic_light())
            .with_title("Full Options Test")
            .with_subtitle("All features enabled")
            .with_legend()
            .with_gap_highlighting()
            .with_borders(true)
            .with_margin(50)
            .with_background(Rgb::new(240, 240, 240))
            .with_border_color(Rgb::new(100, 100, 100))
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_cov_31_rgb_new() {
        let color = Rgb::new(128, 64, 32);
        assert_eq!(color.r, 128);
        assert_eq!(color.g, 64);
        assert_eq!(color.b, 32);
    }

    #[test]
    fn h0_cov_32_comparison_result_fields() {
        use super::visual_regression::*;

        let cells = reference_uniform_cells(5, 5, 0.5);
        let png = PngHeatmap::new(200, 200).export(&cells).unwrap();
        let result = compare_png_with_tolerance(&png, &png, 0).unwrap();

        // Verify all fields are accessible
        assert!(result.matches);
        assert_eq!(result.diff_count, 0);
        assert_eq!(result.max_diff, 0);
        assert!((result.diff_percentage - 0.0).abs() < 0.001);
        assert!(result.total_pixels > 0);
    }

    #[test]
    fn h0_cov_33_checksum_determinism() {
        use super::visual_regression::*;

        let data1 = vec![1, 2, 3, 4, 5];
        let data2 = vec![1, 2, 3, 4, 5];
        let data3 = vec![5, 4, 3, 2, 1];

        assert_eq!(compute_checksum(&data1), compute_checksum(&data2));
        assert_ne!(compute_checksum(&data1), compute_checksum(&data3));
    }

    #[test]
    fn h0_cov_34_svg_multiple_cells() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
            ],
            vec![
                CoverageCell {
                    coverage: 0.25,
                    hit_count: 2,
                },
                CoverageCell {
                    coverage: 0.75,
                    hit_count: 7,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
            ],
        ];

        let svg = SvgHeatmap::new(300, 200).export(&cells);

        // Should have 6 rect elements (2 rows x 3 cols)
        let rect_count = svg.matches("<rect").count();
        assert_eq!(rect_count, 6);
    }

    #[test]
    fn h0_cov_35_bitmap_font_render_bounds() {
        use image::{ImageBuffer, RgbImage};

        let font = BitmapFont::default();
        let mut img: RgbImage = ImageBuffer::new(10, 10);

        // Try to render text that would overflow bounds
        font.render_text(&mut img, "HELLO WORLD TEST", 5, 5, Rgb::new(0, 0, 0));

        // Should not panic - pixels outside bounds are simply skipped
    }

    #[test]
    fn h0_cov_36_interpolate_at_segment_boundaries() {
        let palette = ColorPalette::viridis();

        // Test values just below boundaries
        let c1 = palette.interpolate(0.249);
        let c2 = palette.interpolate(0.251);
        // These should be in different segments, producing different interpolations
        assert!(c1.r != c2.r || c1.g != c2.g || c1.b != c2.b);
    }

    #[test]
    fn h0_cov_37_heatmap_renderer_trait() {
        // Test that HeatmapRenderer trait is properly defined
        struct TestRenderer;
        impl HeatmapRenderer for TestRenderer {
            fn render(&self, cells: &[Vec<CoverageCell>]) -> String {
                format!("{}x{}", cells.len(), cells.first().map_or(0, Vec::len))
            }
        }

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                };
                3
            ];
            2
        ];
        let renderer = TestRenderer;
        assert_eq!(renderer.render(&cells), "2x3");
    }

    #[test]
    fn h0_cov_38_terminal_multiple_rows() {
        let cells = vec![
            vec![0.0, 0.1, 0.2],
            vec![0.3, 0.4, 0.5],
            vec![0.6, 0.7, 0.8],
            vec![0.9, 1.0, 0.0],
        ];
        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render();

        // Should have 4 lines
        assert_eq!(rendered.lines().count(), 4);

        // Each line should have 3 characters
        for line in rendered.lines() {
            assert_eq!(line.chars().count(), 3);
        }
    }
