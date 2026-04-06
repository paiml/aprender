//! Integration tests for presentar-core.
//!
//! These tests verify the public API works correctly end-to-end.

use presentar_core::{
    Color, Command, Constraints, Point, Rect, Size, State, TextStyle, Transform2D, WidgetId,
};

// =============================================================================
// Color Integration Tests
// =============================================================================

#[test]
fn test_color_roundtrip_hex() {
    let original = Color::rgb(0.5, 0.25, 0.75);
    let hex = original.to_hex();
    let parsed = Color::from_hex(&hex).expect("valid hex");

    // Allow small rounding differences
    assert!((original.r - parsed.r).abs() < 0.01);
    assert!((original.g - parsed.g).abs() < 0.01);
    assert!((original.b - parsed.b).abs() < 0.01);
}

#[test]
fn test_color_wcag_contrast() {
    // Test WCAG AA compliance (4.5:1 for normal text)
    let ratio = Color::BLACK.contrast_ratio(&Color::WHITE);
    assert!(ratio >= 4.5, "Black/white should meet WCAG AA");

    // Test that similar colors have low contrast
    let light_gray = Color::rgb(0.9, 0.9, 0.9);
    let lighter_gray = Color::rgb(0.95, 0.95, 0.95);
    let low_ratio = light_gray.contrast_ratio(&lighter_gray);
    assert!(low_ratio < 4.5, "Similar grays should fail WCAG AA");
}

#[test]
fn test_color_lerp_gradient() {
    let start = Color::RED;
    let end = Color::BLUE;

    // Generate a gradient
    let steps: Vec<Color> = (0..=10)
        .map(|i| start.lerp(&end, i as f32 / 10.0))
        .collect();

    // First and last should match endpoints
    assert_eq!(steps[0], start);
    assert!((steps[10].b - end.b).abs() < 0.01);

    // Middle should be purple-ish
    let mid = &steps[5];
    assert!(mid.r > 0.4 && mid.r < 0.6);
    assert!(mid.b > 0.4 && mid.b < 0.6);
}

// =============================================================================
// Geometry Integration Tests
// =============================================================================

#[test]
fn test_rect_contains_point() {
    let rect = Rect::new(10.0, 10.0, 100.0, 100.0);

    // Inside
    assert!(rect.contains_point(&Point::new(50.0, 50.0)));

    // On edge
    assert!(rect.contains_point(&Point::new(10.0, 10.0)));
    assert!(rect.contains_point(&Point::new(109.0, 109.0)));

    // Outside
    assert!(!rect.contains_point(&Point::new(5.0, 50.0)));
    assert!(!rect.contains_point(&Point::new(50.0, 5.0)));
    assert!(!rect.contains_point(&Point::new(150.0, 50.0)));
}

#[test]
fn test_rect_intersection() {
    let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
    let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);

    let intersection = r1.intersection(&r2).expect("should intersect");
    assert_eq!(intersection.x, 50.0);
    assert_eq!(intersection.y, 50.0);
    assert_eq!(intersection.width, 50.0);
    assert_eq!(intersection.height, 50.0);
}

#[test]
fn test_rect_no_intersection() {
    let r1 = Rect::new(0.0, 0.0, 50.0, 50.0);
    let r2 = Rect::new(100.0, 100.0, 50.0, 50.0);

    assert!(r1.intersection(&r2).is_none());
}

// =============================================================================
// Constraints Integration Tests
// =============================================================================

#[test]
fn test_constraints_layout_flow() {
    // Simulate a layout scenario
    let viewport = Size::new(800.0, 600.0);
    let root_constraints = Constraints::loose(viewport);

    // Child wants fixed size
    let child_desired = Size::new(200.0, 150.0);
    let child_size = root_constraints.constrain(child_desired);
    assert_eq!(child_size, child_desired);

    // Another child wants more than available
    let greedy_desired = Size::new(1000.0, 800.0);
    let greedy_size = root_constraints.constrain(greedy_desired);
    assert_eq!(greedy_size.width, 800.0);
    assert_eq!(greedy_size.height, 600.0);
}

#[test]
fn test_constraints_deflate_for_padding() {
    let container = Constraints::new(0.0, 400.0, 0.0, 300.0);
    let padding = 20.0;

    let content_constraints = container.deflate(padding * 2.0, padding * 2.0);
    assert_eq!(content_constraints.max_width, 360.0);
    assert_eq!(content_constraints.max_height, 260.0);
}

// =============================================================================
// State Integration Tests
// =============================================================================

use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Debug)]
struct TodoItem {
    id: u32,
    text: String,
    completed: bool,
}

#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Debug)]
struct TodoState {
    items: Vec<TodoItem>,
    next_id: u32,
}

#[derive(Debug)]
enum TodoMessage {
    Add(String),
    Toggle(u32),
    Remove(u32),
    ClearCompleted,
}

impl State for TodoState {
    type Message = TodoMessage;

    fn update(&mut self, msg: Self::Message) -> Command<Self::Message> {
        match msg {
            TodoMessage::Add(text) => {
                self.items.push(TodoItem {
                    id: self.next_id,
                    text,
                    completed: false,
                });
                self.next_id += 1;
            }
            TodoMessage::Toggle(id) => {
                if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
                    item.completed = !item.completed;
                }
            }
            TodoMessage::Remove(id) => {
                self.items.retain(|i| i.id != id);
            }
            TodoMessage::ClearCompleted => {
                self.items.retain(|i| !i.completed);
            }
        }
        Command::None
    }
}

#[test]
fn test_todo_state_workflow() {
    let mut state = TodoState::default();

    // Add items
    state.update(TodoMessage::Add("Buy groceries".into()));
    state.update(TodoMessage::Add("Walk the dog".into()));
    state.update(TodoMessage::Add("Write tests".into()));
    assert_eq!(state.items.len(), 3);

    // Toggle completion
    state.update(TodoMessage::Toggle(0));
    assert!(state.items[0].completed);
    assert!(!state.items[1].completed);

    // Remove item
    state.update(TodoMessage::Remove(1));
    assert_eq!(state.items.len(), 2);

    // Clear completed
    state.update(TodoMessage::ClearCompleted);
    assert_eq!(state.items.len(), 1);
    assert_eq!(state.items[0].text, "Write tests");
}

#[test]
fn test_state_serialization() {
    let mut state = TodoState::default();
    state.update(TodoMessage::Add("Test serialization".into()));
    state.update(TodoMessage::Toggle(0));

    // Serialize
    let json = serde_json::to_string(&state).expect("serialize");

    // Deserialize
    let loaded: TodoState = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(state, loaded);
}

// =============================================================================
// Transform Integration Tests
// =============================================================================

#[test]
fn test_transform_composition() {
    let translate = Transform2D::translate(100.0, 50.0);
    let scale = Transform2D::scale(2.0, 2.0);

    // Verify individual transforms
    assert_eq!(translate.matrix[4], 100.0);
    assert_eq!(translate.matrix[5], 50.0);
    assert_eq!(scale.matrix[0], 2.0);
    assert_eq!(scale.matrix[3], 2.0);
}

#[test]
fn test_transform_rotation() {
    use std::f32::consts::PI;

    // 90 degree rotation
    let rot90 = Transform2D::rotate(PI / 2.0);
    assert!((rot90.matrix[0] - 0.0).abs() < 0.001); // cos(90) = 0
    assert!((rot90.matrix[1] - 1.0).abs() < 0.001); // sin(90) = 1
}

// =============================================================================
// Widget ID Tests
// =============================================================================

#[test]
fn test_widget_id_uniqueness() {
    use std::collections::HashSet;

    let ids: HashSet<WidgetId> = (0..1000).map(WidgetId::new).collect();
    assert_eq!(ids.len(), 1000, "All IDs should be unique");
}

// =============================================================================
// TextStyle Tests
// =============================================================================

#[test]
fn test_text_style_customization() {
    use presentar_core::{FontStyle, FontWeight};

    let style = TextStyle {
        size: 24.0,
        color: Color::from_hex("#333333").expect("valid hex"),
        weight: FontWeight::Bold,
        style: FontStyle::Normal,
    };

    assert_eq!(style.size, 24.0);
    assert_eq!(style.weight, FontWeight::Bold);
}
