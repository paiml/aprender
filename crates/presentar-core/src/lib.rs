#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::type_complexity)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_wrap)]
//! Core types and traits for Presentar UI framework.
//!
//! This crate provides foundational types used throughout Presentar:
//! - Geometric primitives: [`Point`], [`Size`], [`Rect`]
//! - Color representation: [`Color`] with WCAG contrast calculations
//! - Layout constraints: [`Constraints`]
//! - Events and messages: [`Event`], [`Message`]
//! - Draw commands: [`DrawCommand`] for GPU rendering
//!
//! # Quick Start
//!
//! ```rust
//! use presentar_core::{Color, Size, Constraints, Rect};
//!
//! // Create a size
//! let size = Size::new(100.0, 50.0);
//!
//! // Create constraints
//! let constraints = Constraints::new(0.0, 200.0, 0.0, 100.0);
//! let bounded = constraints.constrain(size);
//!
//! // Create a color
//! let red = Color::RED;
//! assert_eq!(red.r, 1.0);
//! ```
//!
//! # Widget Trait
//!
//! The core [`Widget`] trait defines the measure-layout-paint cycle:
//!
//! ```rust,ignore
//! use presentar_core::{Widget, Constraints, Size, Canvas};
//!
//! struct MyWidget;
//!
//! impl Widget for MyWidget {
//!     fn measure(&self, constraints: &Constraints) -> Size {
//!         constraints.constrain(Size::new(100.0, 50.0))
//!     }
//!
//!     fn layout(&mut self, size: Size) { }
//!
//!     fn paint(&self, canvas: &mut dyn Canvas) { }
//! }
//! ```

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

pub mod accessibility;
pub mod animation;
pub mod binding;
pub mod cache;
mod canvas;
pub mod chart;
pub mod clipboard;
mod color;
mod constraints;
pub mod diff;
pub mod dnd;
pub mod draw;
mod event;
mod geometry;
pub mod gesture;
pub mod history;
pub mod lifecycle;
mod runtime;
pub mod shortcut;
pub mod simd;

// Re-export ComputeBlock primitives for convenient access
pub use simd::{
    batch_mean_f64, batch_min_max_f64, batch_scale_f64, batch_scale_offset_f64, batch_stddev_f64,
    batch_sum_f64, batch_variance_f64, histogram_f64, normalize_f64, normalize_with_range_f64,
    percentile_sorted_f64, weighted_sum_f64,
};
mod state;
pub mod streaming;
pub mod theme;
pub mod validation;
pub mod virtualization;
pub mod widget;

// Brick Architecture types (native - CB-081: no jugar-probar prod dep)
pub mod brick_types;

// Brick Architecture integration (PROBAR-SPEC-009: Brick is mandatory)
pub mod brick_widget;

pub use accessibility::{
    AccessibilityTree, AccessibilityTreeBuilder, AccessibleNode, AccessibleNodeId, CheckedState,
    HitTester, LiveRegion,
};
pub use animation::{
    AnimColor, AnimatedValue, AnimationController, EasedValue, Easing, Interpolate, Keyframe,
    KeyframeTrack, Spring, SpringConfig,
};
pub use cache::{
    CacheBuilder, CacheCallback, CacheConfig, CacheEvent, CacheKey, CacheMetadata, CacheOptions,
    CacheSize, CacheState, CacheStats, DataCache, StringCache,
};
pub use canvas::RecordingCanvas;
pub use chart::{
    ArcGeometry, CatmullRom, CubicBezier, CubicSpline, DataNormalizer, DrawBatch, HistogramBins,
    Interpolator, LinearInterpolator, PathTessellator, Point2D,
};
pub use clipboard::{
    Clipboard, ClipboardData, ClipboardEvent, ClipboardFormat, ClipboardHistory,
    ClipboardOperation, ClipboardResult,
};
pub use color::{Color, ColorParseError};
pub use constraints::Constraints;
pub use diff::{diff_trees, DiffNode, DiffOp, DiffResult, TreeDiffer, WidgetKey};
pub use dnd::{
    DragData, DragDataType, DragDropManager, DragId, DragPayload, DragPhase, DragState, DropEffect,
    DropResult, DropTarget,
};
pub use draw::{
    BoxStyle, DrawCommand, FillRule, LineCap, LineJoin, PathRef, Sampling, Shadow, StrokeStyle,
    TensorRef, Transform2D as DrawTransform,
};
pub use event::{Event, GestureState, Key, MouseButton, PointerId, PointerType, TouchId};
pub use geometry::{CornerRadius, Point, Rect, Size};
pub use gesture::{
    GestureConfig, GestureRecognizer, PointerGestureRecognizer, PointerInfo, RecognizedGesture,
    TouchPoint,
};
pub use history::{
    Checkpoint, CheckpointId, Command as HistoryCommand, CommandHistory, CommandId, CommandResult,
    CompositeCommand, GroupId, HistoryCallback, HistoryConfig, HistoryEvent, SetValueCommand,
};
pub use lifecycle::{
    Effect, EffectManager, HookId, LifecycleEvent, LifecycleManager, LifecyclePhase,
};
pub use runtime::{
    default_executor, AnimatedProperty, AnimationId, AnimationInstance, AnimationState, Animator,
    CommandExecutor, DataRefreshManager, EasingFunction, ExecutionResult, ExecutorConfig,
    FocusDirection, FocusManager, FocusTrap, FrameTimer, MemoryRouter, MemoryStorage, RefreshTask,
    Router, SpringAnimation, Storage, Timer, TransitionConfig, Tween,
};
pub use shortcut::{
    Modifiers, Shortcut, ShortcutBuilder, ShortcutContext, ShortcutId, ShortcutManager,
    ShortcutPriority,
};
pub use state::{Command, CounterMessage, CounterState, State, Store};
pub use streaming::{
    ConnectionState, DataStream, MessageBuffer, RateLimiter, ReconnectConfig, StreamConfig,
    StreamMessage, StreamSubscription,
};
pub use theme::{ColorPalette, ContrastCheck, Radii, Shadows, Spacing, Theme, Typography};
pub use validation::{
    FieldConfig, FieldState, FormValidator, MaxLength, MinLength, Pattern, PatternType, Range,
    Required, ValidateOn, ValidationResult, Validator,
};
pub use virtualization::{
    CellLayout, GridCell, ItemIndex, ItemLayout, ScrollAlign, VirtualGrid, VirtualGridConfig,
    VirtualList, VirtualListConfig, VisibleGridRange, VisibleRange,
};
pub use widget::{
    AccessibleRole, Canvas, FontStyle, FontWeight, LayoutResult, TextStyle, Transform2D, TypeId,
    Widget, WidgetId,
};

// Re-export Brick types (PROBAR-SPEC-009: Brick is mandatory)
pub use widget::{
    Brick, BrickAssertion, BrickBudget, BrickError, BrickPhase, BrickResult, BrickVerification,
    BudgetViolation,
};

// Re-export brick_widget helpers
pub use brick_widget::{BrickWidgetExt, DefaultBrick, SimpleBrick};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ==========================================================================
    // COLOR TESTS - Written FIRST per EXTREME TDD
    // ==========================================================================

    mod color_tests {
        use super::*;
        use proptest::prelude::*;

        #[test]
        fn test_color_new_clamps_values() {
            let c = Color::new(1.5, -0.5, 0.5, 2.0);
            assert_eq!(c.r, 1.0);
            assert_eq!(c.g, 0.0);
            assert_eq!(c.b, 0.5);
            assert_eq!(c.a, 1.0);
        }

        #[test]
        fn test_color_from_rgb() {
            let c = Color::rgb(0.5, 0.5, 0.5);
            assert_eq!(c.r, 0.5);
            assert_eq!(c.g, 0.5);
            assert_eq!(c.b, 0.5);
            assert_eq!(c.a, 1.0);
        }

        #[test]
        fn test_color_from_hex() {
            let c = Color::from_hex("#ff0000").unwrap();
            assert_eq!(c.r, 1.0);
            assert_eq!(c.g, 0.0);
            assert_eq!(c.b, 0.0);

            let c2 = Color::from_hex("#00ff00").unwrap();
            assert_eq!(c2.g, 1.0);

            let c3 = Color::from_hex("0000ff").unwrap();
            assert_eq!(c3.b, 1.0);
        }

        #[test]
        fn test_color_from_hex_with_alpha() {
            let c = Color::from_hex("#ff000080").unwrap();
            assert_eq!(c.r, 1.0);
            assert!((c.a - 0.502).abs() < 0.01); // 128/255 ≈ 0.502
        }

        #[test]
        fn test_color_from_hex_invalid() {
            assert!(Color::from_hex("invalid").is_err());
            assert!(Color::from_hex("#gg0000").is_err());
            assert!(Color::from_hex("#ff").is_err());
        }

        #[test]
        fn test_color_relative_luminance_black() {
            let black = Color::rgb(0.0, 0.0, 0.0);
            assert_eq!(black.relative_luminance(), 0.0);
        }

        #[test]
        fn test_color_relative_luminance_white() {
            let white = Color::rgb(1.0, 1.0, 1.0);
            assert!((white.relative_luminance() - 1.0).abs() < 0.001);
        }

        #[test]
        fn test_color_contrast_ratio_black_white() {
            let black = Color::rgb(0.0, 0.0, 0.0);
            let white = Color::rgb(1.0, 1.0, 1.0);
            let ratio = black.contrast_ratio(&white);
            assert!((ratio - 21.0).abs() < 0.1); // WCAG max contrast is 21:1
        }

        #[test]
        fn test_color_contrast_ratio_wcag_aa() {
            // WCAG AA requires 4.5:1 for normal text
            let dark = Color::rgb(0.0, 0.0, 0.0);
            let light = Color::rgb(0.5, 0.5, 0.5);
            let ratio = dark.contrast_ratio(&light);
            assert!(ratio >= 4.5, "Contrast ratio {ratio} should be >= 4.5");
        }

        #[test]
        fn test_color_contrast_ratio_symmetric() {
            let c1 = Color::rgb(0.2, 0.4, 0.6);
            let c2 = Color::rgb(0.8, 0.6, 0.4);
            assert_eq!(c1.contrast_ratio(&c2), c2.contrast_ratio(&c1));
        }

        #[test]
        fn test_color_to_hex() {
            let c = Color::rgb(1.0, 0.0, 0.0);
            assert_eq!(c.to_hex(), "#ff0000");

            let c2 = Color::new(0.0, 1.0, 0.0, 0.5);
            assert_eq!(c2.to_hex_with_alpha(), "#00ff0080");
        }

        #[test]
        fn test_color_lerp() {
            let black = Color::rgb(0.0, 0.0, 0.0);
            let white = Color::rgb(1.0, 1.0, 1.0);

            let mid = black.lerp(&white, 0.5);
            assert!((mid.r - 0.5).abs() < 0.001);
            assert!((mid.g - 0.5).abs() < 0.001);
            assert!((mid.b - 0.5).abs() < 0.001);
        }

        proptest! {
            #[test]
            fn prop_color_clamps_to_valid_range(r in -1.0f32..2.0, g in -1.0f32..2.0, b in -1.0f32..2.0, a in -1.0f32..2.0) {
                let c = Color::new(r, g, b, a);
                prop_assert!(c.r >= 0.0 && c.r <= 1.0);
                prop_assert!(c.g >= 0.0 && c.g <= 1.0);
                prop_assert!(c.b >= 0.0 && c.b <= 1.0);
                prop_assert!(c.a >= 0.0 && c.a <= 1.0);
            }

            #[test]
            fn prop_contrast_ratio_always_positive(
                r1 in 0.0f32..1.0, g1 in 0.0f32..1.0, b1 in 0.0f32..1.0,
                r2 in 0.0f32..1.0, g2 in 0.0f32..1.0, b2 in 0.0f32..1.0
            ) {
                let c1 = Color::rgb(r1, g1, b1);
                let c2 = Color::rgb(r2, g2, b2);
                prop_assert!(c1.contrast_ratio(&c2) >= 1.0);
            }

            #[test]
            fn prop_lerp_at_zero_returns_self(r in 0.0f32..1.0, g in 0.0f32..1.0, b in 0.0f32..1.0) {
                let c1 = Color::rgb(r, g, b);
                let c2 = Color::rgb(1.0 - r, 1.0 - g, 1.0 - b);
                let result = c1.lerp(&c2, 0.0);
                prop_assert!((result.r - c1.r).abs() < 0.001);
                prop_assert!((result.g - c1.g).abs() < 0.001);
                prop_assert!((result.b - c1.b).abs() < 0.001);
            }

            #[test]
            fn prop_lerp_at_one_returns_other(r in 0.0f32..1.0, g in 0.0f32..1.0, b in 0.0f32..1.0) {
                let c1 = Color::rgb(r, g, b);
                let c2 = Color::rgb(1.0 - r, 1.0 - g, 1.0 - b);
                let result = c1.lerp(&c2, 1.0);
                prop_assert!((result.r - c2.r).abs() < 0.001);
                prop_assert!((result.g - c2.g).abs() < 0.001);
                prop_assert!((result.b - c2.b).abs() < 0.001);
            }
        }
    }

    // ==========================================================================
    // GEOMETRY TESTS - Written FIRST per EXTREME TDD
    // ==========================================================================

    mod geometry_tests {
        use super::*;
        use proptest::prelude::*;

        #[test]
        fn test_point_new() {
            let p = Point::new(10.0, 20.0);
            assert_eq!(p.x, 10.0);
            assert_eq!(p.y, 20.0);
        }

        #[test]
        fn test_point_origin() {
            let p = Point::ORIGIN;
            assert_eq!(p.x, 0.0);
            assert_eq!(p.y, 0.0);
        }

        #[test]
        fn test_point_distance() {
            let p1 = Point::new(0.0, 0.0);
            let p2 = Point::new(3.0, 4.0);
            assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
        }

        #[test]
        fn test_point_add() {
            let p1 = Point::new(1.0, 2.0);
            let p2 = Point::new(3.0, 4.0);
            let sum = p1 + p2;
            assert_eq!(sum.x, 4.0);
            assert_eq!(sum.y, 6.0);
        }

        #[test]
        fn test_point_sub() {
            let p1 = Point::new(5.0, 7.0);
            let p2 = Point::new(2.0, 3.0);
            let diff = p1 - p2;
            assert_eq!(diff.x, 3.0);
            assert_eq!(diff.y, 4.0);
        }

        #[test]
        fn test_size_new() {
            let s = Size::new(100.0, 200.0);
            assert_eq!(s.width, 100.0);
            assert_eq!(s.height, 200.0);
        }

        #[test]
        fn test_size_zero() {
            let s = Size::ZERO;
            assert_eq!(s.width, 0.0);
            assert_eq!(s.height, 0.0);
        }

        #[test]
        fn test_size_area() {
            let s = Size::new(10.0, 20.0);
            assert_eq!(s.area(), 200.0);
        }

        #[test]
        fn test_size_aspect_ratio() {
            let s = Size::new(16.0, 9.0);
            assert!((s.aspect_ratio() - 16.0 / 9.0).abs() < 0.001);
        }

        #[test]
        fn test_size_contains() {
            let s = Size::new(100.0, 100.0);
            let smaller = Size::new(50.0, 50.0);
            let larger = Size::new(150.0, 50.0);
            assert!(s.contains(&smaller));
            assert!(!s.contains(&larger));
        }

        #[test]
        fn test_rect_new() {
            let r = Rect::new(10.0, 20.0, 100.0, 200.0);
            assert_eq!(r.x, 10.0);
            assert_eq!(r.y, 20.0);
            assert_eq!(r.width, 100.0);
            assert_eq!(r.height, 200.0);
        }

        #[test]
        fn test_rect_from_points() {
            let r = Rect::from_points(Point::new(10.0, 20.0), Point::new(110.0, 220.0));
            assert_eq!(r.x, 10.0);
            assert_eq!(r.y, 20.0);
            assert_eq!(r.width, 100.0);
            assert_eq!(r.height, 200.0);
        }

        #[test]
        fn test_rect_from_size() {
            let r = Rect::from_size(Size::new(100.0, 200.0));
            assert_eq!(r.x, 0.0);
            assert_eq!(r.y, 0.0);
            assert_eq!(r.width, 100.0);
            assert_eq!(r.height, 200.0);
        }

        #[test]
        fn test_rect_origin_and_size() {
            let r = Rect::new(10.0, 20.0, 100.0, 200.0);
            assert_eq!(r.origin(), Point::new(10.0, 20.0));
            assert_eq!(r.size(), Size::new(100.0, 200.0));
        }

        #[test]
        fn test_rect_corners() {
            let r = Rect::new(10.0, 20.0, 100.0, 200.0);
            assert_eq!(r.top_left(), Point::new(10.0, 20.0));
            assert_eq!(r.top_right(), Point::new(110.0, 20.0));
            assert_eq!(r.bottom_left(), Point::new(10.0, 220.0));
            assert_eq!(r.bottom_right(), Point::new(110.0, 220.0));
        }

        #[test]
        fn test_rect_center() {
            let r = Rect::new(0.0, 0.0, 100.0, 100.0);
            assert_eq!(r.center(), Point::new(50.0, 50.0));
        }

        #[test]
        fn test_rect_contains_point() {
            let r = Rect::new(10.0, 10.0, 100.0, 100.0);
            assert!(r.contains_point(&Point::new(50.0, 50.0)));
            assert!(r.contains_point(&Point::new(10.0, 10.0))); // Edge inclusive
            assert!(!r.contains_point(&Point::new(5.0, 50.0)));
            assert!(!r.contains_point(&Point::new(111.0, 50.0)));
        }

        #[test]
        fn test_rect_intersects() {
            let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
            let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);
            let r3 = Rect::new(200.0, 200.0, 100.0, 100.0);

            assert!(r1.intersects(&r2));
            assert!(!r1.intersects(&r3));
        }

        #[test]
        fn test_rect_intersection() {
            let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
            let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);

            let inter = r1.intersection(&r2).unwrap();
            assert_eq!(inter.x, 50.0);
            assert_eq!(inter.y, 50.0);
            assert_eq!(inter.width, 50.0);
            assert_eq!(inter.height, 50.0);
        }

        #[test]
        fn test_rect_union() {
            let r1 = Rect::new(0.0, 0.0, 50.0, 50.0);
            let r2 = Rect::new(25.0, 25.0, 50.0, 50.0);

            let union = r1.union(&r2);
            assert_eq!(union.x, 0.0);
            assert_eq!(union.y, 0.0);
            assert_eq!(union.width, 75.0);
            assert_eq!(union.height, 75.0);
        }

        #[test]
        fn test_rect_inset() {
            let r = Rect::new(10.0, 10.0, 100.0, 100.0);
            let inset = r.inset(5.0);
            assert_eq!(inset.x, 15.0);
            assert_eq!(inset.y, 15.0);
            assert_eq!(inset.width, 90.0);
            assert_eq!(inset.height, 90.0);
        }

        #[test]
        fn test_corner_radius() {
            let uniform = CornerRadius::uniform(10.0);
            assert_eq!(uniform.top_left, 10.0);
            assert_eq!(uniform.top_right, 10.0);
            assert_eq!(uniform.bottom_left, 10.0);
            assert_eq!(uniform.bottom_right, 10.0);

            let custom = CornerRadius::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(custom.top_left, 1.0);
            assert_eq!(custom.top_right, 2.0);
            assert_eq!(custom.bottom_right, 3.0);
            assert_eq!(custom.bottom_left, 4.0);
        }

        proptest! {
            #[test]
            fn prop_point_distance_non_negative(x1 in -1000.0f32..1000.0, y1 in -1000.0f32..1000.0, x2 in -1000.0f32..1000.0, y2 in -1000.0f32..1000.0) {
                let p1 = Point::new(x1, y1);
                let p2 = Point::new(x2, y2);
                prop_assert!(p1.distance(&p2) >= 0.0);
            }

            #[test]
            fn prop_point_distance_symmetric(x1 in -1000.0f32..1000.0, y1 in -1000.0f32..1000.0, x2 in -1000.0f32..1000.0, y2 in -1000.0f32..1000.0) {
                let p1 = Point::new(x1, y1);
                let p2 = Point::new(x2, y2);
                prop_assert!((p1.distance(&p2) - p2.distance(&p1)).abs() < 0.001);
            }

            #[test]
            fn prop_rect_area_non_negative(x in -1000.0f32..1000.0, y in -1000.0f32..1000.0, w in 0.0f32..1000.0, h in 0.0f32..1000.0) {
                let r = Rect::new(x, y, w, h);
                prop_assert!(r.area() >= 0.0);
            }

            #[test]
            fn prop_rect_contains_center(x in -1000.0f32..1000.0, y in -1000.0f32..1000.0, w in 1.0f32..1000.0, h in 1.0f32..1000.0) {
                let r = Rect::new(x, y, w, h);
                prop_assert!(r.contains_point(&r.center()));
            }

            #[test]
            fn prop_rect_intersects_self(x in -1000.0f32..1000.0, y in -1000.0f32..1000.0, w in 0.1f32..1000.0, h in 0.1f32..1000.0) {
                let r = Rect::new(x, y, w, h);
                prop_assert!(r.intersects(&r));
            }
        }
    }

    // ==========================================================================
    // CONSTRAINTS TESTS - Written FIRST per EXTREME TDD
    // ==========================================================================

    mod constraints_tests {
        use super::*;

        #[test]
        fn test_constraints_tight() {
            let c = Constraints::tight(Size::new(100.0, 200.0));
            assert_eq!(c.min_width, 100.0);
            assert_eq!(c.max_width, 100.0);
            assert_eq!(c.min_height, 200.0);
            assert_eq!(c.max_height, 200.0);
        }

        #[test]
        fn test_constraints_loose() {
            let c = Constraints::loose(Size::new(100.0, 200.0));
            assert_eq!(c.min_width, 0.0);
            assert_eq!(c.max_width, 100.0);
            assert_eq!(c.min_height, 0.0);
            assert_eq!(c.max_height, 200.0);
        }

        #[test]
        fn test_constraints_unbounded() {
            let c = Constraints::unbounded();
            assert_eq!(c.min_width, 0.0);
            assert_eq!(c.max_width, f32::INFINITY);
            assert_eq!(c.min_height, 0.0);
            assert_eq!(c.max_height, f32::INFINITY);
        }

        #[test]
        fn test_constraints_constrain() {
            let c = Constraints::new(50.0, 150.0, 50.0, 150.0);

            // Within bounds
            assert_eq!(
                c.constrain(Size::new(100.0, 100.0)),
                Size::new(100.0, 100.0)
            );

            // Below min
            assert_eq!(c.constrain(Size::new(10.0, 10.0)), Size::new(50.0, 50.0));

            // Above max
            assert_eq!(
                c.constrain(Size::new(200.0, 200.0)),
                Size::new(150.0, 150.0)
            );
        }

        #[test]
        fn test_constraints_is_tight() {
            let tight = Constraints::tight(Size::new(100.0, 100.0));
            let loose = Constraints::loose(Size::new(100.0, 100.0));

            assert!(tight.is_tight());
            assert!(!loose.is_tight());
        }

        #[test]
        fn test_constraints_has_bounded_width() {
            let bounded = Constraints::new(0.0, 100.0, 0.0, f32::INFINITY);
            let unbounded = Constraints::unbounded();

            assert!(bounded.has_bounded_width());
            assert!(!unbounded.has_bounded_width());
        }
    }

    // ==========================================================================
    // EVENT TESTS - Written FIRST per EXTREME TDD
    // ==========================================================================

    mod event_tests {
        use super::*;

        #[test]
        fn test_event_mouse_move() {
            let e = Event::MouseMove {
                position: Point::new(100.0, 200.0),
            };
            if let Event::MouseMove { position } = e {
                assert_eq!(position.x, 100.0);
                assert_eq!(position.y, 200.0);
            } else {
                panic!("Expected MouseMove event");
            }
        }

        #[test]
        fn test_event_mouse_button() {
            let e = Event::MouseDown {
                position: Point::new(50.0, 50.0),
                button: MouseButton::Left,
            };
            if let Event::MouseDown { button, .. } = e {
                assert_eq!(button, MouseButton::Left);
            } else {
                panic!("Expected MouseDown event");
            }
        }

        #[test]
        fn test_event_key() {
            let e = Event::key_down(Key::Enter);
            if let Event::KeyDown { key, .. } = e {
                assert_eq!(key, Key::Enter);
            } else {
                panic!("Expected KeyDown event");
            }
        }

        #[test]
        fn test_event_scroll() {
            let e = Event::Scroll {
                delta_x: 0.0,
                delta_y: -10.0,
            };
            if let Event::Scroll { delta_y, .. } = e {
                assert_eq!(delta_y, -10.0);
            } else {
                panic!("Expected Scroll event");
            }
        }

        #[test]
        fn test_event_text_input() {
            let e = Event::TextInput {
                text: "hello".to_string(),
            };
            if let Event::TextInput { text } = e {
                assert_eq!(text, "hello");
            } else {
                panic!("Expected TextInput event");
            }
        }
    }
}
