//! Widget trait and related types.
//!
//! This module defines the core `Widget` trait and supporting types for building
//! UI components in Presentar.
//!
//! # Widget Lifecycle (Brick Architecture - PROBAR-SPEC-009)
//!
//! Widgets follow a verify-measure-layout-paint cycle:
//!
//! 1. **Verify**: Check all Brick assertions pass (Popperian falsification)
//! 2. **Measure**: Compute intrinsic size given constraints
//! 3. **Layout**: Position self and children within allocated bounds
//! 4. **Paint**: Generate draw commands for rendering (only if verified)
//!
//! # Brick Integration
//!
//! When the `brick` feature is enabled, all Widgets must implement the `Brick`
//! trait from `jugar_probar`. This enforces the "tests define interface" philosophy:
//!
//! - Assertions are verified before every paint
//! - Budget violations trigger Jidoka (stop-the-line)
//! - Rendering is blocked if any assertion fails
//!
//! # Examples
//!
//! ```
//! use presentar_core::{WidgetId, TypeId, Transform2D};
//!
//! // Create widget identifiers
//! let id = WidgetId::new(42);
//! assert_eq!(id.0, 42);
//!
//! // Get type IDs for widget type comparison
//! let string_type = TypeId::of::<String>();
//! let i32_type = TypeId::of::<i32>();
//! assert_ne!(string_type, i32_type);
//!
//! // Create transforms for rendering
//! let translate = Transform2D::translate(10.0, 20.0);
//! let scale = Transform2D::scale(2.0, 2.0);
//! ```

use crate::constraints::Constraints;
use crate::event::Event;
use crate::geometry::{Rect, Size};
use serde::{Deserialize, Serialize};
use std::any::Any;

// Re-export Brick types from native brick_types (CB-081: no jugar-probar prod dep)
pub use crate::brick_types::{
    Brick, BrickAssertion, BrickBudget, BrickError, BrickPhase, BrickResult, BrickVerification,
    BudgetViolation,
};

/// Unique identifier for a widget instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WidgetId(pub u64);

impl WidgetId {
    /// Create a new widget ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Type identifier for widget types (used for diffing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(std::any::TypeId);

impl TypeId {
    /// Get the type ID for a type.
    #[must_use]
    pub fn of<T: 'static>() -> Self {
        Self(std::any::TypeId::of::<T>())
    }
}

/// Result of laying out a widget.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayoutResult {
    /// Computed size after layout
    pub size: Size,
}

/// Core widget trait that all UI elements implement.
///
/// # Brick Architecture (PROBAR-SPEC-009)
///
/// Widget REQUIRES the `Brick` trait, enforcing the "tests define interface" philosophy:
///
/// - Every Widget has assertions that define its contract
/// - Every Widget has a performance budget
/// - Rendering is blocked if assertions fail (Popperian falsification)
///
/// # Lifecycle
///
/// 1. `verify`: Check Brick assertions (mandatory)
/// 2. `measure`: Compute intrinsic size given constraints
/// 3. `layout`: Position self and children within allocated bounds
/// 4. `paint`: Generate draw commands (only if `can_render()` returns true)
pub trait Widget: Brick + Send + Sync {
    /// Get the type identifier for this widget type.
    fn type_id(&self) -> TypeId;

    /// Compute intrinsic size constraints.
    fn measure(&self, constraints: Constraints) -> Size;

    /// Position children within allocated bounds.
    fn layout(&mut self, bounds: Rect) -> LayoutResult;

    /// Generate draw commands for rendering.
    ///
    /// # Panics
    ///
    /// Panics if called when `can_render()` returns false (Brick verification failed).
    fn paint(&self, canvas: &mut dyn Canvas);

    /// Handle input events.
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>>;

    /// Get child widgets for tree traversal.
    fn children(&self) -> &[Box<dyn Widget>];

    /// Get mutable child widgets.
    fn children_mut(&mut self) -> &mut [Box<dyn Widget>];

    /// Check if this widget is interactive (can receive focus/events).
    fn is_interactive(&self) -> bool {
        false
    }

    /// Check if this widget can receive keyboard focus.
    fn is_focusable(&self) -> bool {
        false
    }

    /// Get the accessible name for screen readers.
    fn accessible_name(&self) -> Option<&str> {
        None
    }

    /// Get the accessible role.
    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }

    /// Get the test ID for this widget (if any).
    fn test_id(&self) -> Option<&str> {
        None
    }

    /// Get the current bounds of this widget.
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
}

// NOTE: Non-Brick Widget trait has been REMOVED (PROBAR-SPEC-009 Phase 6)
// All widgets MUST implement Brick trait. There is no backwards compatibility path.
// This is intentional - "tests define interface" is mandatory, not optional.

/// Canvas trait for paint operations.
///
/// This is a minimal abstraction over the rendering backend.
pub trait Canvas {
    /// Draw a filled rectangle.
    fn fill_rect(&mut self, rect: Rect, color: crate::Color);

    /// Draw a stroked rectangle.
    fn stroke_rect(&mut self, rect: Rect, color: crate::Color, width: f32);

    /// Draw text.
    fn draw_text(&mut self, text: &str, position: crate::Point, style: &TextStyle);

    /// Draw a line between two points.
    fn draw_line(&mut self, from: crate::Point, to: crate::Point, color: crate::Color, width: f32);

    /// Draw a filled circle.
    fn fill_circle(&mut self, center: crate::Point, radius: f32, color: crate::Color);

    /// Draw a stroked circle.
    fn stroke_circle(&mut self, center: crate::Point, radius: f32, color: crate::Color, width: f32);

    /// Draw a filled arc (pie slice).
    fn fill_arc(
        &mut self,
        center: crate::Point,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        color: crate::Color,
    );

    /// Draw a path (polyline).
    fn draw_path(&mut self, points: &[crate::Point], color: crate::Color, width: f32);

    /// Fill a polygon.
    fn fill_polygon(&mut self, points: &[crate::Point], color: crate::Color);

    /// Push a clip region.
    fn push_clip(&mut self, rect: Rect);

    /// Pop the clip region.
    fn pop_clip(&mut self);

    /// Push a transform.
    fn push_transform(&mut self, transform: Transform2D);

    /// Pop the transform.
    fn pop_transform(&mut self);
}

/// Text style for rendering.
///
/// # Examples
///
/// ```
/// use presentar_core::{TextStyle, FontWeight, FontStyle, Color};
///
/// // Use default style
/// let default_style = TextStyle::default();
/// assert_eq!(default_style.size, 16.0);
/// assert_eq!(default_style.weight, FontWeight::Normal);
///
/// // Create custom style
/// let heading_style = TextStyle {
///     size: 24.0,
///     color: Color::from_hex("#1a1a1a").expect("valid hex"),
///     weight: FontWeight::Bold,
///     style: FontStyle::Normal,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextStyle {
    /// Font size in pixels
    pub size: f32,
    /// Text color
    pub color: crate::Color,
    /// Font weight
    pub weight: FontWeight,
    /// Font style
    pub style: FontStyle,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            size: 16.0,
            color: crate::Color::BLACK,
            weight: FontWeight::Normal,
            style: FontStyle::Normal,
        }
    }
}

/// Font weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FontWeight {
    /// Thin (100)
    Thin,
    /// Light (300)
    Light,
    /// Normal (400)
    Normal,
    /// Medium (500)
    Medium,
    /// Semibold (600)
    Semibold,
    /// Bold (700)
    Bold,
    /// Black (900)
    Black,
}

/// Font style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FontStyle {
    /// Normal style
    Normal,
    /// Italic style
    Italic,
}

/// 2D affine transform.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Transform2D {
    /// Matrix elements [a, b, c, d, e, f] for:
    /// | a c e |
    /// | b d f |
    /// | 0 0 1 |
    pub matrix: [f32; 6],
}

impl Transform2D {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };

    /// Create a translation transform.
    #[must_use]
    pub const fn translate(x: f32, y: f32) -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, x, y],
        }
    }

    /// Create a scale transform.
    #[must_use]
    pub const fn scale(sx: f32, sy: f32) -> Self {
        Self {
            matrix: [sx, 0.0, 0.0, sy, 0.0, 0.0],
        }
    }

    /// Create a rotation transform (angle in radians).
    #[must_use]
    pub fn rotate(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            matrix: [cos, sin, -sin, cos, 0.0, 0.0],
        }
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Accessible role for screen readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AccessibleRole {
    /// Generic element
    #[default]
    Generic,
    /// Button
    Button,
    /// Checkbox
    Checkbox,
    /// Text input
    TextInput,
    /// Link
    Link,
    /// Heading
    Heading,
    /// Image
    Image,
    /// List
    List,
    /// List item
    ListItem,
    /// Table
    Table,
    /// Table row
    TableRow,
    /// Table cell
    TableCell,
    /// Menu
    Menu,
    /// Menu item
    MenuItem,
    /// Combo box / dropdown select
    ComboBox,
    /// Slider
    Slider,
    /// Progress bar
    ProgressBar,
    /// Tab
    Tab,
    /// Tab panel
    TabPanel,
    /// Radio group
    RadioGroup,
    /// Radio button
    Radio,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_id() {
        let id = WidgetId::new(42);
        assert_eq!(id.0, 42);
    }

    #[test]
    fn test_widget_id_eq() {
        let id1 = WidgetId::new(1);
        let id2 = WidgetId::new(1);
        let id3 = WidgetId::new(2);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_widget_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WidgetId::new(1));
        set.insert(WidgetId::new(2));
        assert_eq!(set.len(), 2);
        assert!(set.contains(&WidgetId::new(1)));
    }

    #[test]
    fn test_type_id() {
        let id1 = TypeId::of::<u32>();
        let id2 = TypeId::of::<u32>();
        let id3 = TypeId::of::<String>();

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_type_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TypeId::of::<u32>());
        set.insert(TypeId::of::<String>());
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_transform2d_identity() {
        let t = Transform2D::IDENTITY;
        assert_eq!(t.matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transform2d_default() {
        let t = Transform2D::default();
        assert_eq!(t.matrix, Transform2D::IDENTITY.matrix);
    }

    #[test]
    fn test_transform2d_translate() {
        let t = Transform2D::translate(10.0, 20.0);
        assert_eq!(t.matrix[4], 10.0);
        assert_eq!(t.matrix[5], 20.0);
    }

    #[test]
    fn test_transform2d_scale() {
        let t = Transform2D::scale(2.0, 3.0);
        assert_eq!(t.matrix[0], 2.0);
        assert_eq!(t.matrix[3], 3.0);
    }

    #[test]
    fn test_transform2d_rotate() {
        let t = Transform2D::rotate(std::f32::consts::PI / 2.0);
        // 90 degrees rotation: cos = 0, sin = 1
        assert!((t.matrix[0] - 0.0).abs() < 1e-6);
        assert!((t.matrix[1] - 1.0).abs() < 1e-6);
        assert!((t.matrix[2] - (-1.0)).abs() < 1e-6);
        assert!((t.matrix[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_text_style_default() {
        let style = TextStyle::default();
        assert_eq!(style.size, 16.0);
        assert_eq!(style.weight, FontWeight::Normal);
        assert_eq!(style.style, FontStyle::Normal);
        assert_eq!(style.color, crate::Color::BLACK);
    }

    #[test]
    fn test_text_style_eq() {
        let s1 = TextStyle::default();
        let s2 = TextStyle::default();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_text_style_custom() {
        let style = TextStyle {
            size: 24.0,
            color: crate::Color::RED,
            weight: FontWeight::Bold,
            style: FontStyle::Italic,
        };
        assert_eq!(style.size, 24.0);
        assert_eq!(style.weight, FontWeight::Bold);
        assert_eq!(style.style, FontStyle::Italic);
    }

    #[test]
    fn test_font_weight_variants() {
        let weights = [
            FontWeight::Thin,
            FontWeight::Light,
            FontWeight::Normal,
            FontWeight::Medium,
            FontWeight::Semibold,
            FontWeight::Bold,
            FontWeight::Black,
        ];
        assert_eq!(weights.len(), 7);
    }

    #[test]
    fn test_font_style_variants() {
        assert_ne!(FontStyle::Normal, FontStyle::Italic);
    }

    #[test]
    fn test_accessible_role_default() {
        assert_eq!(AccessibleRole::default(), AccessibleRole::Generic);
    }

    #[test]
    fn test_accessible_role_variants() {
        let roles = [
            AccessibleRole::Generic,
            AccessibleRole::Button,
            AccessibleRole::Checkbox,
            AccessibleRole::TextInput,
            AccessibleRole::Link,
            AccessibleRole::Heading,
            AccessibleRole::Image,
            AccessibleRole::List,
            AccessibleRole::ListItem,
            AccessibleRole::Table,
            AccessibleRole::TableRow,
            AccessibleRole::TableCell,
            AccessibleRole::Menu,
            AccessibleRole::MenuItem,
            AccessibleRole::ComboBox,
            AccessibleRole::Slider,
            AccessibleRole::ProgressBar,
            AccessibleRole::Tab,
            AccessibleRole::TabPanel,
            AccessibleRole::RadioGroup,
            AccessibleRole::Radio,
        ];
        assert_eq!(roles.len(), 21);
    }

    #[test]
    fn test_layout_result_default() {
        let result = LayoutResult::default();
        assert_eq!(result.size, Size::new(0.0, 0.0));
    }

    #[test]
    fn test_layout_result_with_size() {
        let result = LayoutResult {
            size: Size::new(100.0, 50.0),
        };
        assert_eq!(result.size.width, 100.0);
        assert_eq!(result.size.height, 50.0);
    }
}
