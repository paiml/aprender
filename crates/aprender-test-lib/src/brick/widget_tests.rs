    use super::*;
    use crate::brick::{BrickAssertion, BrickVerification};

    // ============================================================
    // Test Widget Implementation
    // ============================================================

    /// Test widget implementation
    struct TestWidget {
        text: String,
        size: Size,
        assertions: Vec<BrickAssertion>,
    }

    impl TestWidget {
        fn new(text: &str) -> Self {
            Self {
                text: text.to_string(),
                size: Size::new(100.0, 50.0),
                assertions: vec![
                    BrickAssertion::TextVisible,
                    BrickAssertion::ContrastRatio(4.5),
                ],
            }
        }
    }

    impl Brick for TestWidget {
        fn brick_name(&self) -> &'static str {
            "TestWidget"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &self.assertions
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            let mut passed = Vec::new();
            let mut failed = Vec::new();

            for assertion in &self.assertions {
                match assertion {
                    BrickAssertion::TextVisible => {
                        if !self.text.is_empty() {
                            passed.push(assertion.clone());
                        } else {
                            failed.push((assertion.clone(), "Empty text".into()));
                        }
                    }
                    _ => passed.push(assertion.clone()),
                }
            }

            BrickVerification {
                passed,
                failed,
                verification_time: Duration::from_micros(50),
            }
        }

        fn to_html(&self) -> String {
            format!("<div class=\"widget\">{}</div>", self.text)
        }

        fn to_css(&self) -> String {
            ".widget { display: flex; }".into()
        }
    }

    impl Widget for TestWidget {
        fn measure(&self, constraints: Constraints) -> Size {
            constraints.constrain(self.size)
        }

        fn layout(&mut self, bounds: Rect) -> LayoutResult {
            LayoutResult::success(bounds)
        }

        fn paint(&self, canvas: &mut dyn Canvas) {
            canvas.draw(DrawCommand::Rect {
                bounds: Rect::new(0.0, 0.0, self.size.width, self.size.height),
                color: WidgetColor::WHITE,
                radius: CornerRadius::ZERO,
            });
            canvas.draw(DrawCommand::Text {
                content: self.text.clone(),
                position: WidgetPoint::new(10.0, 25.0),
                style: TextStyle::new(16.0, WidgetColor::BLACK),
            });
        }

        fn event(&mut self, event: &Event) -> Option<Box<dyn Any>> {
            match event {
                Event::Click { .. } => Some(Box::new("clicked")),
                _ => None,
            }
        }
    }

    // ============================================================
    // WidgetPoint tests
    // ============================================================

    #[test]
    fn test_point() {
        let p = WidgetPoint::new(10.0, 20.0);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
        assert_eq!(WidgetPoint::ZERO, WidgetPoint::new(0.0, 0.0));
    }

    #[test]
    fn test_point_default() {
        let p = WidgetPoint::default();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_point_debug_and_clone() {
        let p = WidgetPoint::new(1.0, 2.0);
        let cloned = p;
        assert!(format!("{:?}", cloned).contains("WidgetPoint"));
    }

    #[test]
    fn test_point_equality() {
        let p1 = WidgetPoint::new(10.0, 20.0);
        let p2 = WidgetPoint::new(10.0, 20.0);
        let p3 = WidgetPoint::new(10.0, 30.0);

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    // ============================================================
    // Size tests
    // ============================================================

    #[test]
    fn test_size() {
        let s = Size::new(100.0, 50.0);
        assert!(s.has_area());
        assert!(!Size::ZERO.has_area());
    }

    #[test]
    fn test_size_default() {
        let s = Size::default();
        assert_eq!(s.width, 0.0);
        assert_eq!(s.height, 0.0);
    }

    #[test]
    fn test_size_has_area_edge_cases() {
        assert!(!Size::new(0.0, 100.0).has_area());
        assert!(!Size::new(100.0, 0.0).has_area());
        assert!(!Size::new(-1.0, 100.0).has_area());
        assert!(Size::new(0.001, 0.001).has_area());
    }

    #[test]
    fn test_size_debug_and_clone() {
        let s = Size::new(50.0, 100.0);
        let cloned = s;
        assert!(format!("{:?}", cloned).contains("Size"));
    }

    #[test]
    fn test_size_equality() {
        assert_eq!(Size::new(10.0, 20.0), Size::new(10.0, 20.0));
        assert_ne!(Size::new(10.0, 20.0), Size::new(10.0, 30.0));
    }

    // ============================================================
    // Rect tests
    // ============================================================

    #[test]
    fn test_rect() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(WidgetPoint::new(50.0, 30.0)));
        assert!(!r.contains(WidgetPoint::new(5.0, 30.0)));
        assert_eq!(r.size(), Size::new(100.0, 50.0));
    }

    #[test]
    fn test_rect_default() {
        let r = Rect::default();
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 0.0);
        assert_eq!(r.height, 0.0);
    }

    #[test]
    fn test_rect_from_size() {
        let r = Rect::from_size(Size::new(100.0, 50.0));
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 100.0);
        assert_eq!(r.height, 50.0);
    }

    #[test]
    fn test_rect_origin() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        let origin = r.origin();
        assert_eq!(origin.x, 10.0);
        assert_eq!(origin.y, 20.0);
    }

    #[test]
    fn test_rect_contains_edge_cases() {
        let r = Rect::new(0.0, 0.0, 100.0, 100.0);

        // Inside
        assert!(r.contains(WidgetPoint::new(50.0, 50.0)));

        // On edges (inclusive at start, exclusive at end)
        assert!(r.contains(WidgetPoint::new(0.0, 0.0)));
        assert!(r.contains(WidgetPoint::new(99.9, 99.9)));
        assert!(!r.contains(WidgetPoint::new(100.0, 50.0)));
        assert!(!r.contains(WidgetPoint::new(50.0, 100.0)));

        // Outside
        assert!(!r.contains(WidgetPoint::new(-1.0, 50.0)));
        assert!(!r.contains(WidgetPoint::new(50.0, -1.0)));
    }

    #[test]
    fn test_rect_to_array() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.to_array(), [10.0, 20.0, 100.0, 50.0]);
    }

    #[test]
    fn test_rect_debug_and_clone() {
        let r = Rect::new(1.0, 2.0, 3.0, 4.0);
        let cloned = r;
        assert!(format!("{:?}", cloned).contains("Rect"));
    }

    // ============================================================
    // WidgetColor tests
    // ============================================================

    #[test]
    fn test_color() {
        let c = WidgetColor::from_hex(0xFF0000);
        assert!((c.r - 1.0).abs() < f32::EPSILON);
        assert!(c.g.abs() < f32::EPSILON);
        assert!(c.b.abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_new() {
        let c = WidgetColor::new(0.5, 0.6, 0.7, 0.8);
        assert!((c.r - 0.5).abs() < f32::EPSILON);
        assert!((c.g - 0.6).abs() < f32::EPSILON);
        assert!((c.b - 0.7).abs() < f32::EPSILON);
        assert!((c.a - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_rgb() {
        let c = WidgetColor::rgb(0.1, 0.2, 0.3);
        assert!((c.r - 0.1).abs() < f32::EPSILON);
        assert!((c.g - 0.2).abs() < f32::EPSILON);
        assert!((c.b - 0.3).abs() < f32::EPSILON);
        assert!((c.a - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(WidgetColor::WHITE.r, 1.0);
        assert_eq!(WidgetColor::WHITE.g, 1.0);
        assert_eq!(WidgetColor::WHITE.b, 1.0);
        assert_eq!(WidgetColor::WHITE.a, 1.0);

        assert_eq!(WidgetColor::BLACK.r, 0.0);
        assert_eq!(WidgetColor::BLACK.g, 0.0);
        assert_eq!(WidgetColor::BLACK.b, 0.0);
        assert_eq!(WidgetColor::BLACK.a, 1.0);

        assert_eq!(WidgetColor::TRANSPARENT.a, 0.0);
    }

    #[test]
    fn test_color_to_array() {
        let c = WidgetColor::new(0.1, 0.2, 0.3, 0.4);
        let arr = c.to_array();
        assert!((arr[0] - 0.1).abs() < f32::EPSILON);
        assert!((arr[1] - 0.2).abs() < f32::EPSILON);
        assert!((arr[2] - 0.3).abs() < f32::EPSILON);
        assert!((arr[3] - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_from_hex_all_colors() {
        // Red
        let red = WidgetColor::from_hex(0xFF0000);
        assert!((red.r - 1.0).abs() < f32::EPSILON);

        // Green
        let green = WidgetColor::from_hex(0x00FF00);
        assert!((green.g - 1.0).abs() < f32::EPSILON);

        // Blue
        let blue = WidgetColor::from_hex(0x0000FF);
        assert!((blue.b - 1.0).abs() < f32::EPSILON);

        // Gray
        let gray = WidgetColor::from_hex(0x808080);
        assert!((gray.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_color_default() {
        let c = WidgetColor::default();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert_eq!(c.a, 0.0);
    }

    // ============================================================
    // CornerRadius tests
    // ============================================================

    #[test]
    fn test_corner_radius_uniform() {
        let r = CornerRadius::uniform(10.0);
        assert_eq!(r.top_left, 10.0);
        assert_eq!(r.top_right, 10.0);
        assert_eq!(r.bottom_left, 10.0);
        assert_eq!(r.bottom_right, 10.0);
    }

    #[test]
    fn test_corner_radius_zero() {
        let r = CornerRadius::ZERO;
        assert_eq!(r.top_left, 0.0);
        assert_eq!(r.top_right, 0.0);
        assert_eq!(r.bottom_left, 0.0);
        assert_eq!(r.bottom_right, 0.0);
    }

    #[test]
    fn test_corner_radius_default() {
        let r = CornerRadius::default();
        assert_eq!(r.top_left, 0.0);
    }

    #[test]
    fn test_corner_radius_debug_and_clone() {
        let r = CornerRadius::uniform(5.0);
        let cloned = r;
        assert!(format!("{:?}", cloned).contains("CornerRadius"));
    }

    // ============================================================
    // TextStyle tests
    // ============================================================

    #[test]
    fn test_text_style() {
        let style = TextStyle::new(16.0, WidgetColor::BLACK);
        assert_eq!(style.font_size, 16.0);
        assert_eq!(style.font_family, "sans-serif");
    }

    #[test]
    fn test_text_style_default() {
        let style = TextStyle::default();
        assert!(style.font_family.is_empty());
        assert_eq!(style.font_size, 0.0);
        assert_eq!(style.font_weight, 0);
    }

    #[test]
    fn test_text_style_full() {
        let style = TextStyle::new(24.0, WidgetColor::WHITE);
        assert_eq!(style.font_size, 24.0);
        assert_eq!(style.font_weight, 400);
        assert!((style.line_height - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_style_debug_and_clone() {
        let style = TextStyle::new(12.0, WidgetColor::BLACK);
        let cloned = style;
        assert!(format!("{:?}", cloned).contains("TextStyle"));
    }

    // ============================================================
    // StrokeStyle tests
    // ============================================================

    #[test]
    fn test_stroke_style_defaults() {
        let style = StrokeStyle::default();
        assert!(matches!(style.line_cap, LineCap::Butt));
        assert!(matches!(style.line_join, LineJoin::Miter));
    }

    #[test]
    fn test_stroke_style_debug_and_clone() {
        let style = StrokeStyle::default();
        let cloned = style;
        assert!(format!("{:?}", cloned).contains("StrokeStyle"));
    }

    // ============================================================
    // LineCap and LineJoin tests
    // ============================================================

    #[test]
    fn test_line_cap_variants() {
        let butt = LineCap::Butt;
        let round = LineCap::Round;
        let square = LineCap::Square;

        assert_eq!(butt, LineCap::default());
        assert_ne!(round, square);
    }

    #[test]
    fn test_line_join_variants() {
        let miter = LineJoin::Miter;
        let round = LineJoin::Round;
        let bevel = LineJoin::Bevel;

        assert_eq!(miter, LineJoin::default());
        assert_ne!(round, bevel);
    }

    // ============================================================
    // Transform2D tests
    // ============================================================

    #[test]
    fn test_transform() {
        let t = Transform2D::translate(10.0, 20.0);
        assert_eq!(t.matrix[4], 10.0);
        assert_eq!(t.matrix[5], 20.0);

        let s = Transform2D::scale(2.0, 3.0);
        assert_eq!(s.matrix[0], 2.0);
        assert_eq!(s.matrix[3], 3.0);
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::identity();
        assert_eq!(t.matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transform_default() {
        let t = Transform2D::default();
        assert_eq!(t.matrix, Transform2D::identity().matrix);
    }

    #[test]
    fn test_transform_rotate() {
        use std::f32::consts::PI;

        // 90 degree rotation
        let t = Transform2D::rotate(PI / 2.0);
        assert!((t.matrix[0]).abs() < 0.0001); // cos(90) = 0
        assert!((t.matrix[1] - 1.0).abs() < 0.0001); // sin(90) = 1
    }

    #[test]
    fn test_transform_debug_and_clone() {
        let t = Transform2D::translate(1.0, 2.0);
        let cloned = t;
        assert!(format!("{:?}", cloned).contains("Transform2D"));
    }

    #[test]
    fn test_transform_equality() {
        let t1 = Transform2D::translate(10.0, 20.0);
        let t2 = Transform2D::translate(10.0, 20.0);
        let t3 = Transform2D::scale(2.0, 2.0);

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ============================================================
    // DrawCommand tests
    // ============================================================

    #[test]
    fn test_draw_command_rect() {
        let cmd = DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::uniform(5.0),
        };

        assert!(format!("{:?}", cmd).contains("Rect"));
    }

    #[test]
    fn test_draw_command_circle() {
        let cmd = DrawCommand::Circle {
            center: WidgetPoint::new(50.0, 50.0),
            radius: 25.0,
            color: WidgetColor::BLACK,
        };

        assert!(format!("{:?}", cmd).contains("Circle"));
    }

    #[test]
    fn test_draw_command_text() {
        let cmd = DrawCommand::Text {
            content: "Hello".to_string(),
            position: WidgetPoint::new(10.0, 20.0),
            style: TextStyle::new(16.0, WidgetColor::BLACK),
        };

        assert!(format!("{:?}", cmd).contains("Text"));
    }

    #[test]
    fn test_draw_command_path() {
        let cmd = DrawCommand::Path {
            points: vec![WidgetPoint::new(0.0, 0.0), WidgetPoint::new(100.0, 100.0)],
            style: StrokeStyle::default(),
            closed: false,
        };

        assert!(format!("{:?}", cmd).contains("Path"));
    }

    #[test]
    fn test_draw_command_image() {
        let cmd = DrawCommand::Image {
            data: vec![0, 1, 2, 3],
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
        };

        assert!(format!("{:?}", cmd).contains("Image"));
    }

    #[test]
    fn test_draw_command_group() {
        let cmd = DrawCommand::Group {
            children: vec![DrawCommand::Clear {
                bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
                color: WidgetColor::WHITE,
            }],
            transform: Transform2D::identity(),
        };

        assert!(format!("{:?}", cmd).contains("Group"));
    }

    #[test]
    fn test_draw_command_gradient() {
        let cmd = DrawCommand::Gradient {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            start_color: WidgetColor::WHITE,
            end_color: WidgetColor::BLACK,
            angle: 45.0,
        };

        assert!(format!("{:?}", cmd).contains("Gradient"));
    }

    #[test]
    fn test_draw_command_clear() {
        let cmd = DrawCommand::Clear {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            color: WidgetColor::TRANSPARENT,
        };

        assert!(format!("{:?}", cmd).contains("Clear"));
    }

    #[test]
    fn test_draw_command_clone() {
        let cmd = DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        };

        let _cloned = cmd;
    }

    // ============================================================
    // GpuInstance tests
    // ============================================================

    #[test]
    fn test_gpu_instance_default() {
        let instance = GpuInstance::default();
        assert_eq!(instance.shape_type, 0);
        assert_eq!(instance.corner_radius, 0.0);
    }

    #[test]
    fn test_gpu_instance_debug_and_clone() {
        let instance = GpuInstance {
            bounds: [0.0, 0.0, 100.0, 50.0],
            color: [1.0, 1.0, 1.0, 1.0],
            shape_type: 0,
            corner_radius: 5.0,
            params: [0.0; 4],
        };

        let cloned = instance;
        assert!(format!("{:?}", cloned).contains("GpuInstance"));
    }

    // ============================================================
    // commands_to_gpu_instances tests
    // ============================================================

    #[test]
    fn test_gpu_instances() {
        let commands = vec![
            DrawCommand::Rect {
                bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
                color: WidgetColor::WHITE,
                radius: CornerRadius::uniform(5.0),
            },
            DrawCommand::Circle {
                center: WidgetPoint::new(50.0, 50.0),
                radius: 25.0,
                color: WidgetColor::BLACK,
            },
        ];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].shape_type, 0); // Rect
        assert_eq!(instances[1].shape_type, 1); // Circle
    }

    #[test]
    fn test_gpu_instances_clear() {
        let commands = vec![DrawCommand::Clear {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            color: WidgetColor::WHITE,
        }];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].shape_type, 3); // Clear
    }

    #[test]
    fn test_gpu_instances_gradient() {
        let commands = vec![DrawCommand::Gradient {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            start_color: WidgetColor::WHITE,
            end_color: WidgetColor::BLACK,
            angle: 45.0,
        }];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].shape_type, 4); // Gradient
    }

    #[test]
    fn test_gpu_instances_group_recursive() {
        let commands = vec![DrawCommand::Group {
            children: vec![
                DrawCommand::Rect {
                    bounds: Rect::new(0.0, 0.0, 50.0, 50.0),
                    color: WidgetColor::WHITE,
                    radius: CornerRadius::ZERO,
                },
                DrawCommand::Circle {
                    center: WidgetPoint::new(25.0, 25.0),
                    radius: 10.0,
                    color: WidgetColor::BLACK,
                },
            ],
            transform: Transform2D::identity(),
        }];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 2); // Children are flattened
    }

    #[test]
    fn test_gpu_instances_skips_text_path_image() {
        let commands = vec![
            DrawCommand::Text {
                content: "Hello".to_string(),
                position: WidgetPoint::ZERO,
                style: TextStyle::default(),
            },
            DrawCommand::Path {
                points: vec![],
                style: StrokeStyle::default(),
                closed: false,
            },
            DrawCommand::Image {
                data: vec![],
                bounds: Rect::default(),
            },
        ];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 0); // These need separate render passes
    }

    // ============================================================
    // Constraints tests
    // ============================================================

    #[test]
    fn test_constraints() {
        let constraints = Constraints::loose(Size::new(200.0, 100.0));
        let result = constraints.constrain(Size::new(300.0, 50.0));
        assert_eq!(result.width, 200.0);
        assert_eq!(result.height, 50.0);
    }

    #[test]
    fn test_constraints_unbounded() {
        let c = Constraints::unbounded();
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_width, f32::INFINITY);
        assert_eq!(c.max_height, f32::INFINITY);
    }

    #[test]
    fn test_constraints_tight() {
        let c = Constraints::tight(Size::new(100.0, 50.0));
        assert_eq!(c.min_width, 100.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 50.0);
        assert_eq!(c.max_height, 50.0);
    }

    #[test]
    fn test_constraints_loose() {
        let c = Constraints::loose(Size::new(100.0, 50.0));
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_height, 50.0);
    }

    #[test]
    fn test_constraints_constrain() {
        let c = Constraints {
            min_width: 50.0,
            max_width: 150.0,
            min_height: 25.0,
            max_height: 75.0,
        };

        // Below min
        let result = c.constrain(Size::new(10.0, 10.0));
        assert_eq!(result, Size::new(50.0, 25.0));

        // Above max
        let result = c.constrain(Size::new(200.0, 200.0));
        assert_eq!(result, Size::new(150.0, 75.0));

        // Within range
        let result = c.constrain(Size::new(100.0, 50.0));
        assert_eq!(result, Size::new(100.0, 50.0));
    }

    #[test]
    fn test_constraints_is_satisfied_by() {
        let c = Constraints {
            min_width: 50.0,
            max_width: 150.0,
            min_height: 25.0,
            max_height: 75.0,
        };

        assert!(c.is_satisfied_by(Size::new(100.0, 50.0)));
        assert!(c.is_satisfied_by(Size::new(50.0, 25.0)));
        assert!(c.is_satisfied_by(Size::new(150.0, 75.0)));
        assert!(!c.is_satisfied_by(Size::new(40.0, 50.0)));
        assert!(!c.is_satisfied_by(Size::new(100.0, 80.0)));
    }

    #[test]
    fn test_constraints_default() {
        let c = Constraints::default();
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_width, 0.0);
        assert_eq!(c.max_height, 0.0);
    }

    #[test]
    fn test_constraints_debug_and_clone() {
        let c = Constraints::loose(Size::new(100.0, 100.0));
        let cloned = c;
        assert!(format!("{:?}", cloned).contains("Constraints"));
    }

    // ============================================================
    // LayoutResult tests
    // ============================================================

    #[test]
    fn test_layout_result() {
        let success = LayoutResult::success(Rect::new(0.0, 0.0, 100.0, 50.0));
        assert!(success.success);

        let failure = LayoutResult::failure("Test error");
        assert!(!failure.success);
        assert_eq!(failure.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_layout_result_default() {
        let r = LayoutResult::default();
        assert!(!r.success);
        assert!(r.error.is_none());
    }

    #[test]
    fn test_layout_result_debug_and_clone() {
        let r = LayoutResult::success(Rect::default());
        let cloned = r;
        assert!(format!("{:?}", cloned).contains("LayoutResult"));
    }

    // ============================================================
    // Event tests
    // ============================================================

    #[test]
    fn test_event_click() {
        let event = Event::Click {
            position: WidgetPoint::new(10.0, 20.0),
            button: WidgetMouseButton::Left,
        };

        assert!(format!("{:?}", event).contains("Click"));
    }

    #[test]
    fn test_event_mouse_move() {
        let event = Event::MouseMove {
            position: WidgetPoint::new(50.0, 50.0),
        };

        assert!(format!("{:?}", event).contains("MouseMove"));
    }

    #[test]
    fn test_event_key_press() {
        let event = Event::KeyPress {
            key: "Enter".to_string(),
            modifiers: Modifiers {
                shift: true,
                ctrl: false,
                alt: false,
                meta: false,
            },
        };

        assert!(format!("{:?}", event).contains("KeyPress"));
    }

    #[test]
    fn test_event_focus_blur() {
        let focus = Event::Focus;
        let blur = Event::Blur;

        assert!(format!("{:?}", focus).contains("Focus"));
        assert!(format!("{:?}", blur).contains("Blur"));
    }

    #[test]
    fn test_event_scroll() {
        let event = Event::Scroll {
            delta_x: 10.0,
            delta_y: -20.0,
        };

        assert!(format!("{:?}", event).contains("Scroll"));
    }

    #[test]
    fn test_event_touch() {
        let start = Event::TouchStart {
            position: WidgetPoint::new(100.0, 200.0),
            id: 1,
        };
        let move_ev = Event::TouchMove {
            position: WidgetPoint::new(110.0, 210.0),
            id: 1,
        };
        let end = Event::TouchEnd { id: 1 };

        assert!(format!("{:?}", start).contains("TouchStart"));
        assert!(format!("{:?}", move_ev).contains("TouchMove"));
        assert!(format!("{:?}", end).contains("TouchEnd"));
    }

    #[test]
    fn test_event_clone() {
        let event = Event::Click {
            position: WidgetPoint::ZERO,
            button: WidgetMouseButton::Right,
        };

        let _cloned = event;
    }

    // ============================================================
    // WidgetMouseButton tests
    // ============================================================

    #[test]
    fn test_mouse_button() {
        assert_eq!(WidgetMouseButton::Left, WidgetMouseButton::Left);
        assert_ne!(WidgetMouseButton::Left, WidgetMouseButton::Right);
        assert_ne!(WidgetMouseButton::Right, WidgetMouseButton::Middle);
    }

    #[test]
    fn test_mouse_button_debug_and_clone() {
        let btn = WidgetMouseButton::Middle;
        let cloned = btn;
        assert!(format!("{:?}", cloned).contains("Middle"));
    }

    // ============================================================
    // Modifiers tests
    // ============================================================

    #[test]
    fn test_modifiers_default() {
        let m = Modifiers::default();
        assert!(!m.shift);
        assert!(!m.ctrl);
        assert!(!m.alt);
        assert!(!m.meta);
    }

    #[test]
    fn test_modifiers_debug_and_clone() {
        let m = Modifiers {
            shift: true,
            ctrl: true,
            alt: false,
            meta: true,
        };
        let cloned = m;
        assert!(format!("{:?}", cloned).contains("Modifiers"));
    }

    // ============================================================
    // RecordingCanvas tests
    // ============================================================

    #[test]
    fn test_recording_canvas_new() {
        let canvas = RecordingCanvas::new(Size::new(800.0, 600.0));
        assert_eq!(canvas.size(), Size::new(800.0, 600.0));
        assert!(canvas.commands().is_empty());
    }

    #[test]
    fn test_canvas_clear() {
        let mut canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        canvas.draw(DrawCommand::Clear {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            color: WidgetColor::WHITE,
        });
        assert_eq!(canvas.commands().len(), 1);
        canvas.clear();
        assert_eq!(canvas.commands().len(), 0);
    }

    #[test]
    fn test_canvas_draw_multiple() {
        let mut canvas = RecordingCanvas::new(Size::new(100.0, 100.0));

        canvas.draw(DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 50.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        });
        canvas.draw(DrawCommand::Circle {
            center: WidgetPoint::new(75.0, 75.0),
            radius: 20.0,
            color: WidgetColor::BLACK,
        });

        assert_eq!(canvas.commands().len(), 2);
    }

    #[test]
    fn test_canvas_with_transform() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        let mut transformed = canvas.with_transform(Transform2D::translate(10.0, 20.0));

        transformed.draw(DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 50.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        });

        assert_eq!(transformed.commands().len(), 1);
        // The command should be wrapped in a Group
        if let DrawCommand::Group { transform, .. } = &transformed.commands()[0] {
            assert_eq!(transform.matrix[4], 10.0);
            assert_eq!(transform.matrix[5], 20.0);
        } else {
            panic!("Expected Group command");
        }
    }

    #[test]
    fn test_transformed_canvas_with_nested_transform() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        let transformed = canvas.with_transform(Transform2D::translate(10.0, 10.0));
        let nested = transformed.with_transform(Transform2D::translate(5.0, 5.0));

        assert_eq!(nested.size(), Size::new(100.0, 100.0));
    }

    #[test]
    fn test_transformed_canvas_clear() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        let mut transformed = canvas.with_transform(Transform2D::identity());

        transformed.draw(DrawCommand::Rect {
            bounds: Rect::default(),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        });
        transformed.clear();

        assert_eq!(transformed.commands().len(), 0);
    }

    #[test]
    fn test_recording_canvas_debug() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        assert!(format!("{:?}", canvas).contains("RecordingCanvas"));
    }

    // ============================================================
    // Widget trait tests
    // ============================================================

    #[test]
    fn test_widget_render() {
        let widget = TestWidget::new("Hello");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        widget.render(&mut canvas);

        assert_eq!(canvas.commands().len(), 2);
    }

    #[test]
    fn test_widget_render_invalid() {
        let widget = TestWidget::new(""); // Empty text = invalid
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        widget.render(&mut canvas);

        // Should not paint due to failed verification
        assert_eq!(canvas.commands().len(), 0);
    }

    #[test]
    fn test_widget_render_timed() {
        let widget = TestWidget::new("Hello");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let metrics = widget.render_timed(&mut canvas);

        assert!(metrics.valid);
        assert!(metrics.total_time >= Duration::ZERO);
        assert_eq!(metrics.command_count, 2);
    }

    #[test]
    fn test_widget_render_timed_invalid() {
        let widget = TestWidget::new("");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let metrics = widget.render_timed(&mut canvas);

        assert!(!metrics.valid);
        assert_eq!(metrics.command_count, 0);
        assert_eq!(metrics.paint_time, Duration::ZERO);
    }

    #[test]
    fn test_widget_render_full() {
        let mut widget = TestWidget::new("Hello");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let result = widget.render_full(Rect::new(0.0, 0.0, 100.0, 50.0), &mut canvas);

        assert!(result.success);
        assert_eq!(canvas.commands().len(), 2);
    }

    #[test]
    fn test_widget_render_full_invalid() {
        let mut widget = TestWidget::new("");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let result = widget.render_full(Rect::new(0.0, 0.0, 100.0, 50.0), &mut canvas);

        assert!(!result.success);
        assert_eq!(result.error, Some("Brick verification failed".to_string()));
    }

    #[test]
    fn test_widget_event() {
        let mut widget = TestWidget::new("Hello");

        let result = widget.event(&Event::Click {
            position: WidgetPoint::new(50.0, 25.0),
            button: WidgetMouseButton::Left,
        });

        assert!(result.is_some());
    }

    #[test]
    fn test_widget_event_unhandled() {
        let mut widget = TestWidget::new("Hello");

        let result = widget.event(&Event::Focus);
        assert!(result.is_none());

        let result = widget.event(&Event::Blur);
        assert!(result.is_none());

        let result = widget.event(&Event::Scroll {
            delta_x: 0.0,
            delta_y: 10.0,
        });
        assert!(result.is_none());
    }

    #[test]
    fn test_widget_measure() {
        let widget = TestWidget::new("Hello");
        let size = widget.measure(Constraints::unbounded());
        assert_eq!(size, Size::new(100.0, 50.0));
    }

    #[test]
    fn test_widget_measure_constrained() {
        let widget = TestWidget::new("Hello");
        let size = widget.measure(Constraints::tight(Size::new(50.0, 25.0)));
        assert_eq!(size, Size::new(50.0, 25.0));
    }

    #[test]
    fn test_widget_layout() {
        let mut widget = TestWidget::new("Hello");
        let result = widget.layout(Rect::new(10.0, 20.0, 100.0, 50.0));

        assert!(result.success);
        assert_eq!(result.bounds, Rect::new(10.0, 20.0, 100.0, 50.0));
    }

    #[test]
    fn test_widget_children_default() {
        let widget = TestWidget::new("Hello");
        assert!(widget.children().is_empty());
    }

    #[test]
    fn test_widget_children_mut_default() {
        let mut widget = TestWidget::new("Hello");
        assert!(widget.children_mut().is_empty());
    }

    // ============================================================
    // RenderMetrics tests
    // ============================================================

    #[test]
    fn test_render_metrics_budget() {
        let metrics = RenderMetrics {
            verify_time: Duration::from_millis(1),
            paint_time: Duration::from_millis(5),
            total_time: Duration::from_millis(6),
            valid: true,
            command_count: 10,
        };

        assert!(metrics.within_budget(BrickBudget::uniform(16)));
        assert!(!metrics.within_budget(BrickBudget::uniform(5)));
    }

    #[test]
    fn test_render_metrics_default() {
        let metrics = RenderMetrics::default();
        assert!(!metrics.valid);
        assert_eq!(metrics.command_count, 0);
        assert_eq!(metrics.total_time, Duration::ZERO);
    }

    #[test]
    fn test_render_metrics_debug_and_clone() {
        let metrics = RenderMetrics {
            verify_time: Duration::from_millis(1),
            paint_time: Duration::from_millis(2),
            total_time: Duration::from_millis(3),
            valid: true,
            command_count: 5,
        };

        let cloned = metrics;
        assert!(format!("{:?}", cloned).contains("RenderMetrics"));
    }
