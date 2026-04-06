    use super::*;
    use std::time::SystemTime;

    fn test_screenshot() -> Screenshot {
        Screenshot {
            data: vec![0x89, 0x50, 0x4E, 0x47], // PNG magic bytes
            width: 100,
            height: 100,
            device_pixel_ratio: 1.0,
            timestamp: SystemTime::now(),
        }
    }

    mod svg_config_tests {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = SvgConfig::default();
            assert_eq!(config.viewbox, (800, 600));
            assert!(config.preserve_aspect_ratio);
            assert!(!config.embed_fonts);
            assert_eq!(config.compression, SvgCompression::None);
            assert!(config.include_xml_declaration);
        }

        #[test]
        fn test_new_with_dimensions() {
            let config = SvgConfig::new(1920, 1080);
            assert_eq!(config.viewbox, (1920, 1080));
        }

        #[test]
        fn test_builder_chain() {
            let config = SvgConfig::new(800, 600)
                .with_viewbox(1024, 768)
                .with_preserve_aspect_ratio(false)
                .with_compression(SvgCompression::Minified)
                .with_xml_declaration(false)
                .with_title("Test Screenshot")
                .with_description("A test description");

            assert_eq!(config.viewbox, (1024, 768));
            assert!(!config.preserve_aspect_ratio);
            assert_eq!(config.compression, SvgCompression::Minified);
            assert!(!config.include_xml_declaration);
            assert_eq!(config.title, Some("Test Screenshot".to_string()));
            assert_eq!(config.description, Some("A test description".to_string()));
        }
    }

    mod svg_exporter_tests {
        use super::*;

        #[test]
        fn test_default_exporter() {
            let exporter = SvgExporter::new();
            assert_eq!(exporter.config().viewbox, (800, 600));
        }

        #[test]
        fn test_exporter_with_config() {
            let config = SvgConfig::new(1920, 1080);
            let exporter = SvgExporter::with_config(config);
            assert_eq!(exporter.config().viewbox, (1920, 1080));
        }

        #[test]
        fn test_from_screenshot() {
            let screenshot = test_screenshot();
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));

            let svg = exporter.from_screenshot(&screenshot).unwrap();

            assert!(svg.contains("<?xml version=\"1.0\""));
            assert!(svg.contains("<svg"));
            assert!(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""));
            assert!(svg.contains("<image"));
            assert!(svg.contains("data:image/png;base64,"));
            assert!(svg.contains("</svg>"));
        }

        #[test]
        fn test_from_screenshot_with_annotations() {
            let screenshot = test_screenshot();
            let annotations = vec![Annotation::rectangle(10, 10, 50, 30)];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));

            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("<g id=\"annotations\">"));
            assert!(svg.contains("<rect"));
        }

        #[test]
        fn test_minified_output() {
            let screenshot = test_screenshot();
            let config = SvgConfig::new(100, 100)
                .with_compression(SvgCompression::Minified)
                .with_xml_declaration(false);
            let exporter = SvgExporter::with_config(config);

            let svg = exporter.from_screenshot(&screenshot).unwrap();

            // Minified should have no newlines within
            assert!(!svg.contains("\n  "));
        }

        #[test]
        fn test_with_title_and_description() {
            let screenshot = test_screenshot();
            let config = SvgConfig::new(100, 100)
                .with_title("My Screenshot")
                .with_description("Test description");
            let exporter = SvgExporter::with_config(config);

            let svg = exporter.from_screenshot(&screenshot).unwrap();

            assert!(svg.contains("<title>My Screenshot</title>"));
            assert!(svg.contains("<desc>Test description</desc>"));
        }
    }

    mod svg_shape_tests {
        use super::*;

        #[test]
        fn test_from_shapes() {
            let shapes = vec![
                SvgShape::rect(10.0, 10.0, 100.0, 50.0)
                    .with_fill("blue")
                    .with_stroke("black")
                    .with_stroke_width(2.0),
                SvgShape::circle(150.0, 50.0, 25.0).with_fill("red"),
                SvgShape::line(200.0, 10.0, 300.0, 60.0)
                    .with_stroke("green")
                    .with_stroke_width(3.0),
                SvgShape::text(10.0, 100.0, "Hello SVG").with_fill("black"),
            ];

            let exporter = SvgExporter::with_config(SvgConfig::new(400, 150));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<rect"));
            assert!(svg.contains("fill=\"blue\""));
            assert!(svg.contains("<circle"));
            assert!(svg.contains("<line"));
            assert!(svg.contains("<text"));
            assert!(svg.contains("Hello SVG"));
        }

        #[test]
        fn test_shape_builders() {
            let rect = SvgShape::rect(0.0, 0.0, 100.0, 100.0);
            assert!(matches!(rect, SvgShape::Rect { x: _, y: _, .. }));

            let circle = SvgShape::circle(50.0, 50.0, 25.0);
            assert!(matches!(
                circle,
                SvgShape::Circle {
                    cx: _,
                    cy: _,
                    r: _,
                    ..
                }
            ));

            let line = SvgShape::line(0.0, 0.0, 100.0, 100.0);
            assert!(matches!(
                line,
                SvgShape::Line {
                    x1: _,
                    y1: _,
                    x2: _,
                    y2: _,
                    ..
                }
            ));

            let text = SvgShape::text(10.0, 20.0, "Test");
            assert!(matches!(text, SvgShape::Text { x: _, y: _, .. }));
        }

        #[test]
        fn test_group_shapes() {
            let shapes = vec![SvgShape::Group {
                id: Some("my-group".to_string()),
                children: vec![
                    SvgShape::rect(0.0, 0.0, 50.0, 50.0).with_fill("blue"),
                    SvgShape::circle(25.0, 25.0, 10.0).with_fill("red"),
                ],
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<g id=\"my-group\">"));
            assert!(svg.contains("<rect"));
            assert!(svg.contains("<circle"));
            assert!(svg.contains("</g>"));
        }

        #[test]
        fn test_group_without_id() {
            let shapes = vec![SvgShape::Group {
                id: None,
                children: vec![SvgShape::rect(0.0, 0.0, 50.0, 50.0).with_fill("blue")],
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<g>"));
            assert!(!svg.contains("<g id="));
        }

        #[test]
        fn test_ellipse_shape() {
            let shapes = vec![SvgShape::Ellipse {
                cx: 100.0,
                cy: 50.0,
                rx: 80.0,
                ry: 40.0,
                fill: Some("yellow".to_string()),
                stroke: Some("black".to_string()),
                stroke_width: Some(2.0),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(200, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<ellipse"));
            assert!(svg.contains("cx=\"100\""));
            assert!(svg.contains("rx=\"80\""));
            assert!(svg.contains("ry=\"40\""));
            assert!(svg.contains("fill=\"yellow\""));
        }

        #[test]
        fn test_polyline_shape() {
            let shapes = vec![SvgShape::Polyline {
                points: vec![(10.0, 10.0), (50.0, 90.0), (90.0, 10.0)],
                stroke: Some("purple".to_string()),
                stroke_width: Some(3.0),
                fill: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<polyline"));
            assert!(svg.contains("points=\"10,10 50,90 90,10\""));
            assert!(svg.contains("stroke=\"purple\""));
            assert!(svg.contains("fill=\"none\""));
        }

        #[test]
        fn test_polyline_with_fill() {
            let shapes = vec![SvgShape::Polyline {
                points: vec![(10.0, 10.0), (50.0, 90.0), (90.0, 10.0)],
                stroke: None,
                stroke_width: None,
                fill: Some("orange".to_string()),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"orange\""));
        }

        #[test]
        fn test_polygon_shape() {
            let shapes = vec![SvgShape::Polygon {
                points: vec![(50.0, 10.0), (90.0, 90.0), (10.0, 90.0)],
                fill: Some("green".to_string()),
                stroke: Some("darkgreen".to_string()),
                stroke_width: Some(2.0),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<polygon"));
            assert!(svg.contains("points=\"50,10 90,90 10,90\""));
            assert!(svg.contains("fill=\"green\""));
        }

        #[test]
        fn test_path_shape() {
            let shapes = vec![SvgShape::Path {
                d: "M10 10 L90 90 L10 90 Z".to_string(),
                fill: Some("cyan".to_string()),
                stroke: Some("teal".to_string()),
                stroke_width: Some(1.0),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<path"));
            assert!(svg.contains("d=\"M10 10 L90 90 L10 90 Z\""));
            assert!(svg.contains("fill=\"cyan\""));
        }

        #[test]
        fn test_rect_with_rounded_corners() {
            let shapes = vec![SvgShape::Rect {
                x: 10.0,
                y: 10.0,
                width: 80.0,
                height: 60.0,
                fill: Some("blue".to_string()),
                stroke: None,
                stroke_width: None,
                rx: Some(10.0),
                ry: Some(5.0),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("rx=\"10\""));
            assert!(svg.contains("ry=\"5\""));
        }

        #[test]
        fn test_text_with_font_options() {
            let shapes = vec![SvgShape::Text {
                x: 10.0,
                y: 50.0,
                content: "Hello".to_string(),
                font_size: Some(24.0),
                fill: Some("black".to_string()),
                font_family: Some("Arial".to_string()),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<text"));
            assert!(svg.contains("font-size=\"24\""));
            assert!(svg.contains("font-family=\"Arial\""));
            assert!(svg.contains(">Hello</text>"));
        }

        #[test]
        fn test_shapes_preserve_aspect_ratio_false() {
            let config = SvgConfig::new(100, 100).with_preserve_aspect_ratio(false);
            let exporter = SvgExporter::with_config(config);
            let shapes = vec![SvgShape::rect(0.0, 0.0, 50.0, 50.0)];
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("preserveAspectRatio=\"none\""));
        }

        #[test]
        fn test_shapes_with_title_and_description() {
            let config = SvgConfig::new(100, 100)
                .with_title("My Shapes")
                .with_description("A test shape");
            let exporter = SvgExporter::with_config(config);
            let shapes = vec![SvgShape::rect(0.0, 0.0, 50.0, 50.0)];
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<title>My Shapes</title>"));
            assert!(svg.contains("<desc>A test shape</desc>"));
        }
    }

    mod annotation_tests {
        use super::*;

        #[test]
        fn test_all_annotation_types() {
            let screenshot = test_screenshot();
            let annotations = vec![
                Annotation::rectangle(10, 10, 50, 30),
                Annotation::highlight(60, 10, 50, 30),
                Annotation::circle(120, 25, 15),
                Annotation::arrow(150, 25, 50, 0),
                Annotation::filled_rectangle(10, 60, 50, 30),
            ];

            let exporter = SvgExporter::with_config(SvgConfig::new(200, 200));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            // Check all annotation types are rendered
            assert!(svg.contains("<rect")); // Rectangle, FilledRectangle, and Highlight
            assert!(svg.contains("<circle"));
            assert!(svg.contains("<line")); // Arrow
            assert!(svg.contains("<marker")); // Arrow marker
        }

        #[test]
        fn test_annotation_with_label() {
            let screenshot = test_screenshot();
            let annotations = vec![Annotation::rectangle(10, 20, 50, 30).with_label("Test Label")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("<text"));
            assert!(svg.contains("Test Label"));
        }
    }

    mod helper_tests {
        use super::*;

        #[test]
        fn test_escape_xml() {
            assert_eq!(escape_xml("<>&\"'"), "&lt;&gt;&amp;&quot;&apos;");
            assert_eq!(escape_xml("normal text"), "normal text");
            assert_eq!(
                escape_xml("<script>alert('xss')</script>"),
                "&lt;script&gt;alert(&apos;xss&apos;)&lt;/script&gt;"
            );
        }

        #[test]
        fn test_color_to_svg() {
            assert_eq!(color_to_svg(&[255, 0, 0, 255]), "rgba(255,0,0,1)");
            assert_eq!(
                color_to_svg(&[0, 255, 0, 128]),
                "rgba(0,255,0,0.5019607843137255)"
            );
            assert_eq!(color_to_svg(&[0, 0, 0, 0]), "rgba(0,0,0,0)");
        }

        #[test]
        fn test_base64_encode() {
            assert_eq!(base64_encode(b""), "");
            assert_eq!(base64_encode(b"f"), "Zg==");
            assert_eq!(base64_encode(b"fo"), "Zm8=");
            assert_eq!(base64_encode(b"foo"), "Zm9v");
            assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
            assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
            assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        }

        #[test]
        fn test_base64_encode_binary() {
            let data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            let encoded = base64_encode(&data);
            assert!(!encoded.is_empty());
            // PNG magic bytes in base64
            assert!(encoded.starts_with("iVBORw"));
        }
    }

    mod property_tests {
        use super::*;

        #[test]
        fn prop_viewbox_matches_config() {
            for width in [100, 800, 1920, 4096] {
                for height in [100, 600, 1080, 2160] {
                    let config = SvgConfig::new(width, height);
                    let exporter = SvgExporter::with_config(config);
                    let screenshot = Screenshot {
                        data: vec![0],
                        width,
                        height,
                        device_pixel_ratio: 1.0,
                        timestamp: SystemTime::now(),
                    };
                    let svg = exporter.from_screenshot(&screenshot).unwrap();

                    assert!(svg.contains(&format!("width=\"{width}\"")));
                    assert!(svg.contains(&format!("height=\"{height}\"")));
                    assert!(svg.contains(&format!("viewBox=\"0 0 {width} {height}\"")));
                }
            }
        }

        #[test]
        fn prop_svg_always_valid_xml() {
            let screenshot = test_screenshot();
            let exporter = SvgExporter::new();
            let svg = exporter.from_screenshot(&screenshot).unwrap();

            // Basic XML validity checks
            assert!(svg.starts_with("<?xml") || svg.starts_with("<svg"));
            assert!(svg.contains("</svg>"));
            assert_eq!(svg.matches("<svg").count(), 1);
            assert_eq!(svg.matches("</svg>").count(), 1);
        }
    }

    mod shape_builder_tests {
        use super::*;

        #[test]
        fn test_rect_with_stroke() {
            let shape = SvgShape::rect(0.0, 0.0, 100.0, 50.0)
                .with_stroke("red")
                .with_stroke_width(2.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(200, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"red\""));
            assert!(svg.contains("stroke-width=\"2\""));
        }

        #[test]
        fn test_circle_with_all_properties() {
            let shape = SvgShape::circle(50.0, 50.0, 25.0)
                .with_fill("blue")
                .with_stroke("black")
                .with_stroke_width(3.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"blue\""));
            assert!(svg.contains("stroke=\"black\""));
            assert!(svg.contains("stroke-width=\"3\""));
        }

        #[test]
        fn test_line_with_stroke() {
            let shape = SvgShape::line(0.0, 0.0, 100.0, 100.0)
                .with_stroke("green")
                .with_stroke_width(5.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"green\""));
        }

        #[test]
        fn test_text_with_fill() {
            let shape = SvgShape::text(10.0, 50.0, "Hello World").with_fill("purple");

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(200, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"purple\""));
            assert!(svg.contains("Hello World"));
        }

        #[test]
        fn test_line_ignores_fill() {
            // Fill should be ignored for lines
            let shape = SvgShape::line(0.0, 0.0, 100.0, 100.0).with_fill("red");

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            // Line should not have fill attribute
            assert!(svg.contains("<line"));
        }

        #[test]
        fn test_group_ignores_style_methods() {
            // Group should ignore fill/stroke/stroke_width
            let child = SvgShape::rect(0.0, 0.0, 50.0, 50.0).with_fill("blue");
            let shape = SvgShape::Group {
                id: Some("test".to_string()),
                children: vec![child],
            };

            let modified = shape
                .with_fill("red")
                .with_stroke("green")
                .with_stroke_width(5.0);

            let shapes = vec![modified];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            // Group itself shouldn't have fill, child should have blue
            assert!(svg.contains("fill=\"blue\""));
        }

        #[test]
        fn test_ellipse_with_all_properties() {
            let shape = SvgShape::Ellipse {
                cx: 50.0,
                cy: 50.0,
                rx: 40.0,
                ry: 20.0,
                fill: None,
                stroke: None,
                stroke_width: None,
            };

            let shape = shape
                .with_fill("orange")
                .with_stroke("brown")
                .with_stroke_width(2.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"orange\""));
            assert!(svg.contains("stroke=\"brown\""));
        }

        #[test]
        fn test_polygon_with_fill() {
            let shape = SvgShape::Polygon {
                points: vec![(50.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                fill: None,
                stroke: None,
                stroke_width: None,
            }
            .with_fill("yellow");

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"yellow\""));
        }

        #[test]
        fn test_path_with_stroke() {
            let shape = SvgShape::Path {
                d: "M 0 0 L 100 100".to_string(),
                fill: None,
                stroke: None,
                stroke_width: None,
            }
            .with_stroke("gray")
            .with_stroke_width(4.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"gray\""));
            assert!(svg.contains("stroke-width=\"4\""));
        }

        #[test]
        fn test_polyline_with_stroke() {
            let shape = SvgShape::Polyline {
                points: vec![(0.0, 0.0), (50.0, 100.0), (100.0, 0.0)],
                stroke: None,
                stroke_width: None,
                fill: None,
            }
            .with_stroke("navy")
            .with_fill("none");

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"navy\""));
        }
    }

    mod annotation_rendering_tests {
        use super::*;

        #[test]
        fn test_annotation_with_custom_color() {
            let screenshot = test_screenshot();
            let annotations =
                vec![Annotation::rectangle(10, 10, 50, 50).with_color(0, 255, 0, 255)];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("rgba(0,255,0,1)"));
        }

        #[test]
        fn test_annotation_with_label() {
            let screenshot = test_screenshot();
            let annotations = vec![Annotation::rectangle(10, 10, 50, 50).with_label("Test")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("Test"));
        }
    }

    mod exporter_clone_debug_tests {
        use super::*;

        #[test]
        fn test_svg_config_debug() {
            let config = SvgConfig::new(800, 600);
            let debug = format!("{:?}", config);
            assert!(debug.contains("SvgConfig"));
        }

        #[test]
        fn test_svg_exporter_debug() {
            let exporter = SvgExporter::new();
            let debug = format!("{:?}", exporter);
            assert!(debug.contains("SvgExporter"));
        }

        #[test]
        fn test_svg_shape_debug() {
            let shape = SvgShape::rect(0.0, 0.0, 100.0, 50.0);
            let debug = format!("{:?}", shape);
            assert!(debug.contains("Rect"));
        }
    }

    // Additional tests to increase coverage to 95%+
    mod additional_coverage_tests {
        use super::*;
        use std::path::PathBuf;
        use tempfile::tempdir;

        #[test]
        fn test_svg_exporter_default_trait() {
            // Explicitly test Default trait implementation
            let exporter = SvgExporter::default();
            let config = exporter.config();
            assert_eq!(config.viewbox, (800, 600));
            assert!(config.preserve_aspect_ratio);
        }

        #[test]
        fn test_svg_config_clone() {
            let config = SvgConfig::new(1024, 768)
                .with_title("Test Title")
                .with_description("Test Desc");
            let cloned = config;
            assert_eq!(cloned.viewbox, (1024, 768));
            assert_eq!(cloned.title, Some("Test Title".to_string()));
            assert_eq!(cloned.description, Some("Test Desc".to_string()));
        }

        #[test]
        fn test_svg_exporter_clone() {
            let exporter = SvgExporter::new();
            let cloned = exporter.clone();
            assert_eq!(cloned.config().viewbox, exporter.config().viewbox);
        }

        #[test]
        fn test_svg_shape_clone() {
            let shape = SvgShape::rect(10.0, 20.0, 100.0, 50.0).with_fill("blue");
            let cloned = shape;
            if let SvgShape::Rect { x, y, fill, .. } = cloned {
                assert!((x - 10.0).abs() < f64::EPSILON);
                assert!((y - 20.0).abs() < f64::EPSILON);
                assert_eq!(fill, Some("blue".to_string()));
            } else {
                panic!("Expected Rect shape");
            }
        }

        #[test]
        fn test_svg_compression_copy_eq() {
            let comp1 = SvgCompression::None;
            let comp2 = comp1; // Copy
            assert_eq!(comp1, comp2);

            let comp3 = SvgCompression::Minified;
            assert_ne!(comp1, comp3);
        }

        #[test]
        fn test_svg_compression_default() {
            let default = SvgCompression::default();
            assert_eq!(default, SvgCompression::None);
        }

        #[test]
        fn test_save_to_file() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("test.svg");

            let exporter = SvgExporter::new();
            let shapes = vec![SvgShape::rect(0.0, 0.0, 100.0, 50.0)];
            let svg = exporter.from_shapes(&shapes).unwrap();

            exporter.save(&svg, &path).unwrap();

            let contents = std::fs::read_to_string(&path).unwrap();
            assert!(contents.contains("<svg"));
            assert!(contents.contains("<rect"));
        }

        #[test]
        fn test_save_error_handling() {
            let exporter = SvgExporter::new();
            let svg = "<svg></svg>";

            // Try to save to an invalid path
            let invalid_path = PathBuf::from("/nonexistent_dir_xyz/test.svg");
            let result = exporter.save(svg, &invalid_path);
            assert!(result.is_err());
        }

        #[test]
        fn test_preserve_aspect_ratio_false_screenshot() {
            let screenshot = test_screenshot();
            let config = SvgConfig::new(100, 100).with_preserve_aspect_ratio(false);
            let exporter = SvgExporter::with_config(config);

            let svg = exporter.from_screenshot(&screenshot).unwrap();

            assert!(svg.contains("preserveAspectRatio=\"none\""));
        }

        #[test]
        fn test_minified_shapes_output() {
            let config = SvgConfig::new(100, 100)
                .with_compression(SvgCompression::Minified)
                .with_xml_declaration(false);
            let exporter = SvgExporter::with_config(config);
            let shapes = vec![SvgShape::rect(0.0, 0.0, 50.0, 50.0)];

            let svg = exporter.from_shapes(&shapes).unwrap();

            // Minified should not have indentation
            assert!(!svg.contains("\n  "));
        }

        #[test]
        fn test_shapes_no_xml_declaration() {
            let config = SvgConfig::new(100, 100).with_xml_declaration(false);
            let exporter = SvgExporter::with_config(config);
            let shapes = vec![SvgShape::rect(0.0, 0.0, 50.0, 50.0)];

            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(!svg.contains("<?xml"));
        }

        #[test]
        fn test_title_and_description_xml_escaping() {
            let screenshot = test_screenshot();
            let config = SvgConfig::new(100, 100)
                .with_title("Test <Title> & 'Quotes' \"Escaping\"")
                .with_description("Desc with <xml> & 'special' \"chars\"");
            let exporter = SvgExporter::with_config(config);

            let svg = exporter.from_screenshot(&screenshot).unwrap();

            assert!(svg.contains("&lt;Title&gt;"));
            assert!(svg.contains("&amp;"));
            assert!(svg.contains("&apos;"));
            assert!(svg.contains("&quot;"));
        }

        #[test]
        fn test_empty_shapes_list() {
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let shapes: Vec<SvgShape> = vec![];

            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<svg"));
            assert!(svg.contains("</svg>"));
        }

        #[test]
        fn test_empty_annotations_list() {
            let screenshot = test_screenshot();
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let annotations: Vec<Annotation> = vec![];

            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            // Should not have annotations group when empty
            assert!(!svg.contains("<g id=\"annotations\">"));
        }

        #[test]
        fn test_annotation_circle_with_label() {
            let screenshot = test_screenshot();
            let annotations = vec![Annotation::circle(50, 50, 20).with_label("Circle Label")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("<circle"));
            assert!(svg.contains("Circle Label"));
        }

        #[test]
        fn test_annotation_arrow_with_label() {
            let screenshot = test_screenshot();
            let annotations = vec![Annotation::arrow(10, 10, 50, 50).with_label("Arrow Label")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("<line"));
            assert!(svg.contains("<marker"));
            assert!(svg.contains("Arrow Label"));
        }

        #[test]
        fn test_annotation_highlight_with_label() {
            let screenshot = test_screenshot();
            let annotations =
                vec![Annotation::highlight(10, 10, 50, 30).with_label("Highlight Label")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("fill-opacity=\"0.3\""));
            assert!(svg.contains("Highlight Label"));
        }

        #[test]
        fn test_annotation_filled_rectangle_with_label() {
            let screenshot = test_screenshot();
            let annotations =
                vec![Annotation::filled_rectangle(10, 10, 50, 30).with_label("Filled Label")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("<rect"));
            assert!(svg.contains("Filled Label"));
        }

        #[test]
        fn test_minified_annotations() {
            let screenshot = test_screenshot();
            let config = SvgConfig::new(100, 100).with_compression(SvgCompression::Minified);
            let annotations = vec![Annotation::rectangle(10, 10, 50, 30)];
            let exporter = SvgExporter::with_config(config);

            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            // Minified should have no indented newlines
            assert!(!svg.contains("\n  "));
        }

        #[test]
        fn test_base64_encode_all_chunk_sizes() {
            // Test chunk size 1
            assert_eq!(base64_encode(&[65]), "QQ==");
            // Test chunk size 2
            assert_eq!(base64_encode(&[65, 66]), "QUI=");
            // Test chunk size 3
            assert_eq!(base64_encode(&[65, 66, 67]), "QUJD");
            // Test longer data with all chunk sizes represented
            let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
            let encoded = base64_encode(&data);
            assert!(!encoded.is_empty());
            assert!(!encoded.contains(' '));
        }

        #[test]
        fn test_base64_encode_large_data() {
            let data: Vec<u8> = (0..=255).collect();
            let encoded = base64_encode(&data);
            assert!(!encoded.is_empty());
            // Should be properly padded
            assert!(encoded.len() % 4 == 0);
        }

        #[test]
        fn test_text_ignores_stroke() {
            // Text should ignore stroke and stroke_width
            let shape = SvgShape::text(10.0, 20.0, "Hello")
                .with_stroke("red")
                .with_stroke_width(2.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            // Text element should exist but not have stroke attributes
            assert!(svg.contains("<text"));
            // Verify no stroke attribute in the text element
            let text_line = svg.lines().find(|l| l.contains("<text")).unwrap();
            assert!(!text_line.contains("stroke="));
        }

        #[test]
        fn test_polygon_without_optional_attrs() {
            let shapes = vec![SvgShape::Polygon {
                points: vec![(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)],
                fill: None,
                stroke: None,
                stroke_width: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<polygon"));
            // Should not have fill, stroke, or stroke-width when None
            let polygon_line = svg.lines().find(|l| l.contains("<polygon")).unwrap();
            assert!(!polygon_line.contains("fill="));
            assert!(!polygon_line.contains("stroke="));
        }

        #[test]
        fn test_path_without_optional_attrs() {
            let shapes = vec![SvgShape::Path {
                d: "M0 0 L100 100".to_string(),
                fill: None,
                stroke: None,
                stroke_width: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<path"));
            assert!(svg.contains("d=\"M0 0 L100 100\""));
        }

        #[test]
        fn test_ellipse_without_optional_attrs() {
            let shapes = vec![SvgShape::Ellipse {
                cx: 50.0,
                cy: 50.0,
                rx: 40.0,
                ry: 20.0,
                fill: None,
                stroke: None,
                stroke_width: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<ellipse"));
        }

        #[test]
        fn test_line_without_optional_attrs() {
            let shapes = vec![SvgShape::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 100.0,
                stroke: None,
                stroke_width: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<line"));
        }

        #[test]
        fn test_circle_without_optional_attrs() {
            let shapes = vec![SvgShape::Circle {
                cx: 50.0,
                cy: 50.0,
                r: 25.0,
                fill: None,
                stroke: None,
                stroke_width: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<circle"));
        }

        #[test]
        fn test_rect_without_optional_attrs() {
            let shapes = vec![SvgShape::Rect {
                x: 10.0,
                y: 10.0,
                width: 80.0,
                height: 60.0,
                fill: None,
                stroke: None,
                stroke_width: None,
                rx: None,
                ry: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<rect"));
        }

        #[test]
        fn test_text_without_optional_attrs() {
            let shapes = vec![SvgShape::Text {
                x: 10.0,
                y: 20.0,
                content: "Plain text".to_string(),
                font_size: None,
                fill: None,
                font_family: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<text"));
            assert!(svg.contains(">Plain text</text>"));
        }

        #[test]
        fn test_nested_groups() {
            let inner_group = SvgShape::Group {
                id: Some("inner".to_string()),
                children: vec![SvgShape::circle(25.0, 25.0, 10.0).with_fill("red")],
            };

            let outer_group = SvgShape::Group {
                id: Some("outer".to_string()),
                children: vec![inner_group],
            };

            let shapes = vec![outer_group];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<g id=\"outer\">"));
            assert!(svg.contains("<g id=\"inner\">"));
            assert!(svg.contains("</g>"));
        }

        #[test]
        fn test_group_minified() {
            let config = SvgConfig::new(100, 100).with_compression(SvgCompression::Minified);
            let exporter = SvgExporter::with_config(config);

            let group = SvgShape::Group {
                id: Some("test".to_string()),
                children: vec![SvgShape::rect(0.0, 0.0, 50.0, 50.0)],
            };

            let svg = exporter.from_shapes(&vec![group]).unwrap();

            assert!(svg.contains("<g id=\"test\">"));
        }

        #[test]
        fn test_escape_xml_edge_cases() {
            // Empty string
            assert_eq!(escape_xml(""), "");
            // Only special chars
            assert_eq!(escape_xml("<>&\"'"), "&lt;&gt;&amp;&quot;&apos;");
            // Unicode
            assert_eq!(escape_xml("Hello\u{00A0}World"), "Hello\u{00A0}World");
            // Mixed content
            assert_eq!(
                escape_xml("a < b > c & d \"e\" 'f'"),
                "a &lt; b &gt; c &amp; d &quot;e&quot; &apos;f&apos;"
            );
        }

        #[test]
        fn test_color_to_svg_edge_cases() {
            // Full transparency
            assert_eq!(color_to_svg(&[255, 255, 255, 0]), "rgba(255,255,255,0)");
            // Half transparency
            let half = color_to_svg(&[100, 100, 100, 127]);
            assert!(half.starts_with("rgba(100,100,100,0."));
            // Full opacity
            assert_eq!(color_to_svg(&[0, 0, 0, 255]), "rgba(0,0,0,1)");
        }

        #[test]
        fn test_shapes_with_special_characters() {
            let shapes = vec![SvgShape::text(10.0, 20.0, "Hello <World> & 'Friends'")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("Hello &lt;World&gt; &amp; &apos;Friends&apos;"));
        }

        #[test]
        fn test_polyline_stroke_width() {
            let shapes = vec![SvgShape::Polyline {
                points: vec![(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)],
                stroke: Some("blue".to_string()),
                stroke_width: Some(3.0),
                fill: Some("none".to_string()),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"blue\""));
            assert!(svg.contains("stroke-width=\"3\""));
        }

        #[test]
        fn test_polygon_stroke() {
            let shapes = vec![SvgShape::Polygon {
                points: vec![(50.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                fill: Some("yellow".to_string()),
                stroke: Some("black".to_string()),
                stroke_width: Some(2.0),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"black\""));
            assert!(svg.contains("stroke-width=\"2\""));
        }

        #[test]
        fn test_path_fill() {
            let shapes = vec![SvgShape::Path {
                d: "M10 10 L90 90".to_string(),
                fill: Some("green".to_string()),
                stroke: None,
                stroke_width: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"green\""));
        }

        #[test]
        fn test_with_fill_on_path() {
            let shape = SvgShape::Path {
                d: "M0 0".to_string(),
                fill: None,
                stroke: None,
                stroke_width: None,
            }
            .with_fill("magenta");

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("fill=\"magenta\""));
        }

        #[test]
        fn test_with_stroke_on_polygon() {
            let shape = SvgShape::Polygon {
                points: vec![(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)],
                fill: None,
                stroke: None,
                stroke_width: None,
            }
            .with_stroke("cyan");

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke=\"cyan\""));
        }

        #[test]
        fn test_with_stroke_width_on_polyline() {
            let shape = SvgShape::Polyline {
                points: vec![(0.0, 0.0), (100.0, 100.0)],
                stroke: None,
                stroke_width: None,
                fill: None,
            }
            .with_stroke_width(5.0);

            let shapes = vec![shape];
            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("stroke-width=\"5\""));
        }

        #[test]
        fn test_rect_only_rx() {
            let shapes = vec![SvgShape::Rect {
                x: 10.0,
                y: 10.0,
                width: 80.0,
                height: 60.0,
                fill: None,
                stroke: None,
                stroke_width: None,
                rx: Some(5.0),
                ry: None,
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("rx=\"5\""));
            assert!(!svg.contains("ry="));
        }

        #[test]
        fn test_rect_only_ry() {
            let shapes = vec![SvgShape::Rect {
                x: 10.0,
                y: 10.0,
                width: 80.0,
                height: 60.0,
                fill: None,
                stroke: None,
                stroke_width: None,
                rx: None,
                ry: Some(5.0),
            }];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(!svg.contains("rx="));
            assert!(svg.contains("ry=\"5\""));
        }

        #[test]
        fn test_large_screenshot_data() {
            // Test with larger image data
            let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
            let screenshot = Screenshot {
                data: large_data,
                width: 500,
                height: 500,
                device_pixel_ratio: 2.0,
                timestamp: SystemTime::now(),
            };

            let exporter = SvgExporter::with_config(SvgConfig::new(500, 500));
            let svg = exporter.from_screenshot(&screenshot).unwrap();

            assert!(svg.contains("data:image/png;base64,"));
            assert!(svg.contains("</svg>"));
        }

        #[test]
        fn test_annotation_label_xml_escape() {
            let screenshot = test_screenshot();
            let annotations = vec![
                Annotation::rectangle(10, 10, 50, 30).with_label("Label with <xml> & 'chars'")
            ];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            assert!(svg.contains("&lt;xml&gt;"));
            assert!(svg.contains("&amp;"));
        }

        #[test]
        fn test_multiple_annotations_same_type() {
            let screenshot = test_screenshot();
            let annotations = vec![
                Annotation::rectangle(10, 10, 20, 20),
                Annotation::rectangle(40, 40, 20, 20),
                Annotation::rectangle(70, 70, 20, 20),
            ];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            // Should have 3 rectangles (image uses <image> element, not rect)
            assert_eq!(svg.matches("<rect x=\"").count(), 3);
        }

        #[test]
        fn test_all_shapes_in_single_svg() {
            let shapes = vec![
                SvgShape::rect(10.0, 10.0, 30.0, 20.0),
                SvgShape::circle(50.0, 50.0, 15.0),
                SvgShape::line(0.0, 0.0, 100.0, 100.0),
                SvgShape::text(10.0, 90.0, "Text"),
                SvgShape::Ellipse {
                    cx: 80.0,
                    cy: 20.0,
                    rx: 10.0,
                    ry: 5.0,
                    fill: Some("pink".to_string()),
                    stroke: None,
                    stroke_width: None,
                },
                SvgShape::Polyline {
                    points: vec![(5.0, 5.0), (15.0, 15.0)],
                    stroke: Some("brown".to_string()),
                    stroke_width: None,
                    fill: None,
                },
                SvgShape::Polygon {
                    points: vec![(30.0, 30.0), (40.0, 40.0), (30.0, 40.0)],
                    fill: None,
                    stroke: None,
                    stroke_width: None,
                },
                SvgShape::Path {
                    d: "M50 50 L60 60".to_string(),
                    fill: None,
                    stroke: None,
                    stroke_width: None,
                },
                SvgShape::Group {
                    id: None,
                    children: vec![SvgShape::rect(0.0, 0.0, 10.0, 10.0)],
                },
            ];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter.from_shapes(&shapes).unwrap();

            assert!(svg.contains("<rect"));
            assert!(svg.contains("<circle"));
            assert!(svg.contains("<line"));
            assert!(svg.contains("<text"));
            assert!(svg.contains("<ellipse"));
            assert!(svg.contains("<polyline"));
            assert!(svg.contains("<polygon"));
            assert!(svg.contains("<path"));
            assert!(svg.contains("<g>"));
        }

        #[test]
        fn test_annotation_y_saturating_sub() {
            let screenshot = test_screenshot();
            // Create annotation with y=0 to test saturating_sub
            let annotations = vec![Annotation::rectangle(10, 0, 50, 30).with_label("At top")];

            let exporter = SvgExporter::with_config(SvgConfig::new(100, 100));
            let svg = exporter
                .from_screenshot_with_annotations(&screenshot, &annotations)
                .unwrap();

            // Label y position should be 0 (since 0 - 5 saturates to 0)
            assert!(svg.contains("y=\"0\""));
            assert!(svg.contains("At top"));
        }

        #[test]
        fn test_compression_debug() {
            let comp = SvgCompression::Minified;
            let debug = format!("{:?}", comp);
            assert!(debug.contains("Minified"));
        }
    }
