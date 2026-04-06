//! Canvas implementations for rendering.

use crate::draw::{BoxStyle, DrawCommand, StrokeStyle, Transform2D};
use crate::widget::{Canvas, TextStyle};
use crate::{Color, Point, Rect};

/// A Canvas implementation that records draw operations as `DrawCommand`s.
///
/// This is useful for:
/// - Testing (verify what was painted)
/// - Serialization (send commands to GPU/WASM)
/// - Diffing (compare render outputs)
#[derive(Debug, Default)]
pub struct RecordingCanvas {
    commands: Vec<DrawCommand>,
    clip_stack: Vec<Rect>,
    transform_stack: Vec<Transform2D>,
}

impl RecordingCanvas {
    /// Create a new empty recording canvas.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the recorded draw commands.
    #[must_use]
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Take ownership of the recorded commands, clearing the canvas.
    pub fn take_commands(&mut self) -> Vec<DrawCommand> {
        std::mem::take(&mut self.commands)
    }

    /// Get the number of recorded commands.
    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Check if no commands have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Clear all recorded commands.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.clip_stack.clear();
        self.transform_stack.clear();
    }

    /// Get the current transform (identity if no transforms pushed).
    #[must_use]
    pub fn current_transform(&self) -> Transform2D {
        self.transform_stack
            .last()
            .copied()
            .unwrap_or_else(Transform2D::identity)
    }

    /// Get the current clip bounds (None if no clips pushed).
    #[must_use]
    pub fn current_clip(&self) -> Option<Rect> {
        self.clip_stack.last().copied()
    }

    /// Get the clip stack depth.
    #[must_use]
    pub fn clip_depth(&self) -> usize {
        self.clip_stack.len()
    }

    /// Get the transform stack depth.
    #[must_use]
    pub fn transform_depth(&self) -> usize {
        self.transform_stack.len()
    }

    /// Add a raw draw command.
    pub fn add_command(&mut self, command: DrawCommand) {
        self.commands.push(command);
    }

    /// Draw a filled circle.
    pub fn fill_circle(&mut self, center: Point, radius: f32, color: Color) {
        self.commands
            .push(DrawCommand::filled_circle(center, radius, color));
    }

    /// Draw a line between two points.
    pub fn draw_line(&mut self, from: Point, to: Point, color: Color, width: f32) {
        self.commands.push(DrawCommand::line(
            from,
            to,
            StrokeStyle {
                color,
                width,
                ..Default::default()
            },
        ));
    }

    /// Draw a path (polyline).
    pub fn draw_path(&mut self, points: &[Point], closed: bool, color: Color, width: f32) {
        self.commands.push(DrawCommand::Path {
            points: points.to_vec(),
            closed,
            style: StrokeStyle {
                color,
                width,
                ..Default::default()
            },
        });
    }

    /// Draw a rounded rectangle.
    pub fn fill_rounded_rect(&mut self, rect: Rect, radius: f32, color: Color) {
        self.commands
            .push(DrawCommand::rounded_rect(rect, radius, color));
    }
}

impl Canvas for RecordingCanvas {
    fn fill_rect(&mut self, rect: Rect, color: Color) {
        self.commands.push(DrawCommand::Rect {
            bounds: rect,
            radius: crate::CornerRadius::ZERO,
            style: BoxStyle::fill(color),
        });
    }

    fn stroke_rect(&mut self, rect: Rect, color: Color, width: f32) {
        self.commands.push(DrawCommand::Rect {
            bounds: rect,
            radius: crate::CornerRadius::ZERO,
            style: BoxStyle::stroke(StrokeStyle {
                color,
                width,
                ..Default::default()
            }),
        });
    }

    fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle) {
        self.commands.push(DrawCommand::Text {
            content: text.to_string(),
            position,
            style: style.clone(),
        });
    }

    fn draw_line(&mut self, from: Point, to: Point, color: Color, width: f32) {
        self.commands.push(DrawCommand::Path {
            points: vec![from, to],
            closed: false,
            style: StrokeStyle {
                color,
                width,
                ..Default::default()
            },
        });
    }

    fn fill_circle(&mut self, center: Point, radius: f32, color: Color) {
        self.commands
            .push(DrawCommand::filled_circle(center, radius, color));
    }

    fn stroke_circle(&mut self, center: Point, radius: f32, color: Color, width: f32) {
        self.commands.push(DrawCommand::Circle {
            center,
            radius,
            style: BoxStyle::stroke(StrokeStyle {
                color,
                width,
                ..Default::default()
            }),
        });
    }

    fn fill_arc(
        &mut self,
        center: Point,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        color: Color,
    ) {
        self.commands.push(DrawCommand::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            color,
        });
    }

    fn draw_path(&mut self, points: &[Point], color: Color, width: f32) {
        self.commands.push(DrawCommand::Path {
            points: points.to_vec(),
            closed: false,
            style: StrokeStyle {
                color,
                width,
                ..Default::default()
            },
        });
    }

    fn fill_polygon(&mut self, points: &[Point], color: Color) {
        // For filled polygons, we use a closed path
        // A proper implementation would triangulate the polygon
        // For now, we record the vertices
        self.commands.push(DrawCommand::Path {
            points: points.to_vec(),
            closed: true,
            style: StrokeStyle {
                color,
                width: 0.0, // Fill only
                ..Default::default()
            },
        });
    }

    fn push_clip(&mut self, rect: Rect) {
        self.clip_stack.push(rect);
    }

    fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    fn push_transform(&mut self, transform: crate::widget::Transform2D) {
        // Convert from widget::Transform2D to draw::Transform2D
        let draw_transform = Transform2D {
            matrix: transform.matrix,
        };
        self.transform_stack.push(draw_transform);
    }

    fn pop_transform(&mut self) {
        self.transform_stack.pop();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::widget::FontWeight;

    // =========================================================================
    // RecordingCanvas Creation Tests
    // =========================================================================

    #[test]
    fn test_recording_canvas_new() {
        let canvas = RecordingCanvas::new();
        assert!(canvas.is_empty());
        assert_eq!(canvas.command_count(), 0);
    }

    #[test]
    fn test_recording_canvas_default() {
        let canvas = RecordingCanvas::default();
        assert!(canvas.is_empty());
    }

    // =========================================================================
    // Basic Drawing Tests
    // =========================================================================

    #[test]
    fn test_fill_rect() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_rect(Rect::new(10.0, 20.0, 100.0, 50.0), Color::RED);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.x, 10.0);
                assert_eq!(bounds.y, 20.0);
                assert_eq!(bounds.width, 100.0);
                assert_eq!(bounds.height, 50.0);
                assert_eq!(style.fill, Some(Color::RED));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_stroke_rect() {
        let mut canvas = RecordingCanvas::new();
        canvas.stroke_rect(Rect::new(0.0, 0.0, 50.0, 50.0), Color::BLUE, 2.0);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert!(style.fill.is_none());
                let stroke = style.stroke.as_ref().unwrap();
                assert_eq!(stroke.color, Color::BLUE);
                assert_eq!(stroke.width, 2.0);
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_draw_text() {
        let mut canvas = RecordingCanvas::new();
        let style = TextStyle {
            size: 14.0,
            color: Color::BLACK,
            weight: FontWeight::Bold,
            ..Default::default()
        };
        canvas.draw_text("Hello World", Point::new(10.0, 20.0), &style);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Text {
                content,
                position,
                style: text_style,
            } => {
                assert_eq!(content, "Hello World");
                assert_eq!(position.x, 10.0);
                assert_eq!(position.y, 20.0);
                assert_eq!(text_style.size, 14.0);
                assert_eq!(text_style.weight, FontWeight::Bold);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_fill_circle() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_circle(Point::new(50.0, 50.0), 25.0, Color::GREEN);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                assert_eq!(*center, Point::new(50.0, 50.0));
                assert_eq!(*radius, 25.0);
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Circle command"),
        }
    }

    #[test]
    fn test_draw_line() {
        let mut canvas = RecordingCanvas::new();
        canvas.draw_line(
            Point::new(0.0, 0.0),
            Point::new(100.0, 100.0),
            Color::BLACK,
            1.5,
        );

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Path {
                points,
                closed,
                style,
            } => {
                assert_eq!(points.len(), 2);
                assert_eq!(points[0], Point::new(0.0, 0.0));
                assert_eq!(points[1], Point::new(100.0, 100.0));
                assert!(!closed);
                assert_eq!(style.color, Color::BLACK);
                assert_eq!(style.width, 1.5);
            }
            _ => panic!("Expected Path command"),
        }
    }

    #[test]
    fn test_draw_path() {
        let mut canvas = RecordingCanvas::new();
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(50.0, 100.0),
        ];
        canvas.draw_path(&points, true, Color::BLUE, 2.0);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Path {
                points: p,
                closed,
                style,
            } => {
                assert_eq!(p.len(), 3);
                assert!(*closed);
                assert_eq!(style.color, Color::BLUE);
            }
            _ => panic!("Expected Path command"),
        }
    }

    #[test]
    fn test_fill_rounded_rect() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_rounded_rect(Rect::new(0.0, 0.0, 100.0, 50.0), 8.0, Color::WHITE);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Rect { radius, style, .. } => {
                assert_eq!(radius.top_left, 8.0);
                assert!(radius.is_uniform());
                assert_eq!(style.fill, Some(Color::WHITE));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    // =========================================================================
    // Clip Stack Tests
    // =========================================================================

    #[test]
    fn test_push_pop_clip() {
        let mut canvas = RecordingCanvas::new();
        assert_eq!(canvas.clip_depth(), 0);
        assert!(canvas.current_clip().is_none());

        canvas.push_clip(Rect::new(10.0, 10.0, 100.0, 100.0));
        assert_eq!(canvas.clip_depth(), 1);
        assert_eq!(
            canvas.current_clip(),
            Some(Rect::new(10.0, 10.0, 100.0, 100.0))
        );

        canvas.push_clip(Rect::new(20.0, 20.0, 50.0, 50.0));
        assert_eq!(canvas.clip_depth(), 2);
        assert_eq!(
            canvas.current_clip(),
            Some(Rect::new(20.0, 20.0, 50.0, 50.0))
        );

        canvas.pop_clip();
        assert_eq!(canvas.clip_depth(), 1);
        assert_eq!(
            canvas.current_clip(),
            Some(Rect::new(10.0, 10.0, 100.0, 100.0))
        );

        canvas.pop_clip();
        assert_eq!(canvas.clip_depth(), 0);
        assert!(canvas.current_clip().is_none());
    }

    // =========================================================================
    // Transform Stack Tests
    // =========================================================================

    #[test]
    fn test_push_pop_transform() {
        let mut canvas = RecordingCanvas::new();
        assert_eq!(canvas.transform_depth(), 0);
        assert_eq!(
            canvas.current_transform().matrix,
            Transform2D::identity().matrix
        );

        let t1 = crate::widget::Transform2D::translate(10.0, 20.0);
        canvas.push_transform(t1);
        assert_eq!(canvas.transform_depth(), 1);
        assert_eq!(canvas.current_transform().matrix[4], 10.0);
        assert_eq!(canvas.current_transform().matrix[5], 20.0);

        let t2 = crate::widget::Transform2D::scale(2.0, 2.0);
        canvas.push_transform(t2);
        assert_eq!(canvas.transform_depth(), 2);
        assert_eq!(canvas.current_transform().matrix[0], 2.0);

        canvas.pop_transform();
        assert_eq!(canvas.transform_depth(), 1);
        assert_eq!(canvas.current_transform().matrix[4], 10.0);

        canvas.pop_transform();
        assert_eq!(canvas.transform_depth(), 0);
    }

    // =========================================================================
    // Command Management Tests
    // =========================================================================

    #[test]
    fn test_take_commands() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::RED);
        canvas.fill_rect(Rect::new(20.0, 20.0, 10.0, 10.0), Color::BLUE);

        assert_eq!(canvas.command_count(), 2);

        let commands = canvas.take_commands();
        assert_eq!(commands.len(), 2);
        assert!(canvas.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::RED);
        canvas.push_clip(Rect::new(0.0, 0.0, 100.0, 100.0));
        canvas.push_transform(crate::widget::Transform2D::translate(5.0, 5.0));

        assert!(!canvas.is_empty());
        assert_eq!(canvas.clip_depth(), 1);
        assert_eq!(canvas.transform_depth(), 1);

        canvas.clear();

        assert!(canvas.is_empty());
        assert_eq!(canvas.clip_depth(), 0);
        assert_eq!(canvas.transform_depth(), 0);
    }

    #[test]
    fn test_add_command() {
        let mut canvas = RecordingCanvas::new();
        let cmd = DrawCommand::filled_circle(Point::new(50.0, 50.0), 10.0, Color::RED);
        canvas.add_command(cmd);

        assert_eq!(canvas.command_count(), 1);
    }

    // =========================================================================
    // Multiple Commands Tests
    // =========================================================================

    #[test]
    fn test_multiple_commands_order() {
        let mut canvas = RecordingCanvas::new();

        canvas.fill_rect(Rect::new(0.0, 0.0, 100.0, 100.0), Color::WHITE);
        canvas.stroke_rect(Rect::new(0.0, 0.0, 100.0, 100.0), Color::BLACK, 1.0);
        canvas.draw_text("Hello", Point::new(10.0, 50.0), &TextStyle::default());

        assert_eq!(canvas.command_count(), 3);

        // Verify order
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => assert!(style.fill.is_some()),
            _ => panic!("Expected fill rect first"),
        }
        match &canvas.commands()[1] {
            DrawCommand::Rect { style, .. } => assert!(style.stroke.is_some()),
            _ => panic!("Expected stroke rect second"),
        }
        match &canvas.commands()[2] {
            DrawCommand::Text { .. } => {}
            _ => panic!("Expected text third"),
        }
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_pop_empty_clip_stack() {
        let mut canvas = RecordingCanvas::new();
        canvas.pop_clip(); // Should not panic
        assert_eq!(canvas.clip_depth(), 0);
    }

    #[test]
    fn test_pop_empty_transform_stack() {
        let mut canvas = RecordingCanvas::new();
        canvas.pop_transform(); // Should not panic
        assert_eq!(canvas.transform_depth(), 0);
    }

    #[test]
    fn test_zero_size_rect() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_rect(Rect::new(10.0, 10.0, 0.0, 0.0), Color::RED);
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_empty_text() {
        let mut canvas = RecordingCanvas::new();
        canvas.draw_text("", Point::new(0.0, 0.0), &TextStyle::default());
        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Text { content, .. } => assert!(content.is_empty()),
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_zero_radius_circle() {
        let mut canvas = RecordingCanvas::new();
        canvas.fill_circle(Point::new(50.0, 50.0), 0.0, Color::RED);
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_empty_path() {
        let mut canvas = RecordingCanvas::new();
        canvas.draw_path(&[], false, Color::BLACK, 1.0);
        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Path { points, .. } => assert!(points.is_empty()),
            _ => panic!("Expected Path command"),
        }
    }

    // =========================================================================
    // Canvas Trait Implementation Tests
    // =========================================================================

    #[test]
    fn test_canvas_draw_line() {
        let mut canvas = RecordingCanvas::new();
        Canvas::draw_line(
            &mut canvas,
            Point::new(0.0, 0.0),
            Point::new(100.0, 100.0),
            Color::RED,
            2.0,
        );

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Path { points, style, .. } => {
                assert_eq!(points.len(), 2);
                assert_eq!(style.color, Color::RED);
                assert_eq!(style.width, 2.0);
            }
            _ => panic!("Expected Path command"),
        }
    }

    #[test]
    fn test_canvas_fill_circle() {
        let mut canvas = RecordingCanvas::new();
        Canvas::fill_circle(&mut canvas, Point::new(50.0, 50.0), 25.0, Color::GREEN);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                assert_eq!(*center, Point::new(50.0, 50.0));
                assert_eq!(*radius, 25.0);
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Circle command"),
        }
    }

    #[test]
    fn test_canvas_stroke_circle() {
        let mut canvas = RecordingCanvas::new();
        Canvas::stroke_circle(&mut canvas, Point::new(50.0, 50.0), 20.0, Color::BLUE, 3.0);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Circle { radius, style, .. } => {
                assert_eq!(*radius, 20.0);
                let stroke = style.stroke.as_ref().unwrap();
                assert_eq!(stroke.color, Color::BLUE);
                assert_eq!(stroke.width, 3.0);
            }
            _ => panic!("Expected Circle command"),
        }
    }

    #[test]
    fn test_canvas_fill_arc() {
        let mut canvas = RecordingCanvas::new();
        Canvas::fill_arc(
            &mut canvas,
            Point::new(100.0, 100.0),
            50.0,
            0.0,
            std::f32::consts::PI,
            Color::new(1.0, 0.5, 0.0, 1.0),
        );

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                color,
            } => {
                assert_eq!(*center, Point::new(100.0, 100.0));
                assert_eq!(*radius, 50.0);
                assert_eq!(*start_angle, 0.0);
                assert!((end_angle - std::f32::consts::PI).abs() < 0.001);
                assert_eq!(color.r, 1.0);
            }
            _ => panic!("Expected Arc command"),
        }
    }

    #[test]
    fn test_canvas_draw_path() {
        let mut canvas = RecordingCanvas::new();
        let points = [
            Point::new(0.0, 0.0),
            Point::new(50.0, 100.0),
            Point::new(100.0, 0.0),
        ];
        Canvas::draw_path(&mut canvas, &points, Color::BLACK, 1.5);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Path {
                points: p,
                closed,
                style,
            } => {
                assert_eq!(p.len(), 3);
                assert!(!closed);
                assert_eq!(style.width, 1.5);
            }
            _ => panic!("Expected Path command"),
        }
    }

    #[test]
    fn test_canvas_fill_polygon() {
        let mut canvas = RecordingCanvas::new();
        let points = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(50.0, 100.0),
        ];
        Canvas::fill_polygon(&mut canvas, &points, Color::BLUE);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Path {
                points: p,
                closed,
                style,
            } => {
                assert_eq!(p.len(), 3);
                assert!(*closed);
                assert_eq!(style.color, Color::BLUE);
            }
            _ => panic!("Expected Path command"),
        }
    }
}
