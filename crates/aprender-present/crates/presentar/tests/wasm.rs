//! WASM browser tests - run with `wasm-pack test --headless --chrome`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use presentar_core::draw::DrawCommand;
use presentar_core::{Color, Point, Rect, Size};

// ============================================================================
// DrawCommand JSON Serialization Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_rect_json_roundtrip() {
    let cmd = DrawCommand::filled_rect(Rect::new(10.0, 20.0, 100.0, 50.0), Color::RED);
    let json = serde_json::to_string(&cmd).expect("serialize");
    let parsed: DrawCommand = serde_json::from_str(&json).expect("deserialize");

    match parsed {
        DrawCommand::Rect { bounds, .. } => {
            assert_eq!(bounds.x, 10.0);
            assert_eq!(bounds.width, 100.0);
        }
        _ => panic!("Expected Rect"),
    }
}

#[wasm_bindgen_test]
fn test_circle_json_roundtrip() {
    let cmd = DrawCommand::filled_circle(Point::new(50.0, 50.0), 25.0, Color::BLUE);
    let json = serde_json::to_string(&cmd).expect("serialize");
    let parsed: DrawCommand = serde_json::from_str(&json).expect("deserialize");

    match parsed {
        DrawCommand::Circle { center, radius, .. } => {
            assert_eq!(center.x, 50.0);
            assert_eq!(radius, 25.0);
        }
        _ => panic!("Expected Circle"),
    }
}

#[wasm_bindgen_test]
fn test_text_json_roundtrip() {
    use presentar_core::widget::TextStyle;

    let cmd = DrawCommand::Text {
        content: "Hello WASM".to_string(),
        position: Point::new(10.0, 20.0),
        style: TextStyle::default(),
    };
    let json = serde_json::to_string(&cmd).expect("serialize");
    let parsed: DrawCommand = serde_json::from_str(&json).expect("deserialize");

    match parsed {
        DrawCommand::Text {
            content, position, ..
        } => {
            assert_eq!(content, "Hello WASM");
            assert_eq!(position.x, 10.0);
        }
        _ => panic!("Expected Text"),
    }
}

#[wasm_bindgen_test]
fn test_multiple_commands_json() {
    let commands = vec![
        DrawCommand::filled_rect(Rect::new(0.0, 0.0, 100.0, 100.0), Color::RED),
        DrawCommand::filled_circle(Point::new(50.0, 50.0), 30.0, Color::GREEN),
    ];

    let json = serde_json::to_string(&commands).expect("serialize");
    let parsed: Vec<DrawCommand> = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.len(), 2);
}

// ============================================================================
// Geometry Tests (verify they work in WASM)
// ============================================================================

#[wasm_bindgen_test]
fn test_point_distance_wasm() {
    let p1 = Point::new(0.0, 0.0);
    let p2 = Point::new(3.0, 4.0);
    assert_eq!(p1.distance(&p2), 5.0);
}

#[wasm_bindgen_test]
fn test_rect_contains_wasm() {
    let rect = Rect::new(10.0, 10.0, 100.0, 100.0);
    assert!(rect.contains_point(&Point::new(50.0, 50.0)));
    assert!(!rect.contains_point(&Point::new(5.0, 50.0)));
}

#[wasm_bindgen_test]
fn test_size_area_wasm() {
    let size = Size::new(100.0, 50.0);
    assert_eq!(size.area(), 5000.0);
}

// ============================================================================
// Color Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_color_from_hex_wasm() {
    let color = Color::from_hex("#ff0000").expect("valid hex");
    assert_eq!(color.r, 1.0);
    assert_eq!(color.g, 0.0);
    assert_eq!(color.b, 0.0);
}

#[wasm_bindgen_test]
fn test_color_constants_wasm() {
    assert_eq!(Color::RED.r, 1.0);
    assert_eq!(Color::GREEN.g, 1.0);
    assert_eq!(Color::BLUE.b, 1.0);
}

// ============================================================================
// Widget Measure/Layout Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_widget_measure_in_wasm() {
    use presentar::widgets::Text;
    use presentar_core::{Constraints, Widget};

    let text = Text::new("Test");
    let constraints = Constraints::loose(Size::new(400.0, 300.0));
    let size = text.measure(constraints);

    assert!(size.width > 0.0);
    assert!(size.height > 0.0);
}

#[wasm_bindgen_test]
fn test_column_layout_in_wasm() {
    use presentar::widgets::{Column, Text};
    use presentar_core::{Constraints, Widget};

    let col = Column::new()
        .gap(10.0)
        .child(Text::new("Line 1"))
        .child(Text::new("Line 2"));

    let constraints = Constraints::loose(Size::new(400.0, 300.0));
    let size = col.measure(constraints);

    assert!(size.height > 20.0); // Two lines + gap
}
