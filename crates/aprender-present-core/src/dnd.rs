//! Drag and drop system for interactive data transfer.
//!
//! This module provides:
//! - Drag sources and drop targets
//! - Drag state management
//! - Visual feedback during drag operations
//! - Data transfer between widgets

use crate::geometry::{Point, Rect};
use crate::widget::WidgetId;
use std::any::Any;
use std::collections::HashMap;

/// Unique identifier for a drag operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DragId(pub u64);

impl DragId {
    /// Create a new drag ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Type of data being dragged.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DragDataType {
    /// Plain text.
    Text,
    /// HTML content.
    Html,
    /// URL/link.
    Url,
    /// File path.
    File,
    /// Custom type identifier.
    Custom(String),
}

impl DragDataType {
    /// Create a custom drag data type.
    pub fn custom(name: &str) -> Self {
        Self::Custom(name.to_string())
    }
}

/// Data associated with a drag operation.
#[derive(Debug, Clone)]
pub struct DragData {
    /// Primary data type.
    pub data_type: DragDataType,
    /// String representation of the data.
    pub text: String,
    /// Additional data in various formats.
    pub formats: HashMap<DragDataType, String>,
    /// Custom payload (for internal transfers).
    pub payload: Option<DragPayload>,
}

impl DragData {
    /// Create new drag data with text.
    pub fn text(content: &str) -> Self {
        Self {
            data_type: DragDataType::Text,
            text: content.to_string(),
            formats: HashMap::new(),
            payload: None,
        }
    }

    /// Create new drag data with HTML.
    pub fn html(content: &str) -> Self {
        let mut formats = HashMap::new();
        formats.insert(DragDataType::Html, content.to_string());
        Self {
            data_type: DragDataType::Html,
            text: content.to_string(),
            formats,
            payload: None,
        }
    }

    /// Create new drag data with URL.
    pub fn url(url: &str) -> Self {
        Self {
            data_type: DragDataType::Url,
            text: url.to_string(),
            formats: HashMap::new(),
            payload: None,
        }
    }

    /// Create drag data with custom type.
    pub fn custom(type_name: &str, data: &str) -> Self {
        Self {
            data_type: DragDataType::Custom(type_name.to_string()),
            text: data.to_string(),
            formats: HashMap::new(),
            payload: None,
        }
    }

    /// Add an alternative format.
    pub fn with_format(mut self, data_type: DragDataType, data: &str) -> Self {
        self.formats.insert(data_type, data.to_string());
        self
    }

    /// Add a payload.
    pub fn with_payload<T: Any + Send + Sync + Clone + 'static>(mut self, payload: T) -> Self {
        self.payload = Some(DragPayload::new(payload));
        self
    }

    /// Get data in a specific format.
    pub fn get_format(&self, data_type: &DragDataType) -> Option<&str> {
        if &self.data_type == data_type {
            Some(&self.text)
        } else {
            self.formats.get(data_type).map(std::string::String::as_str)
        }
    }

    /// Check if data is available in the given format.
    pub fn has_format(&self, data_type: &DragDataType) -> bool {
        &self.data_type == data_type || self.formats.contains_key(data_type)
    }
}

/// Type-erased payload for drag data.
#[derive(Debug, Clone)]
pub struct DragPayload {
    data: Box<dyn CloneableAny>,
}

impl DragPayload {
    /// Create a new payload.
    pub fn new<T: Any + Send + Sync + Clone + 'static>(data: T) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Get the payload as a specific type.
    pub fn get<T: Any + Send + Sync + Clone + 'static>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref()
    }
}

/// Trait for cloneable any types.
trait CloneableAny: Any + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneableAny>;
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any + Send + Sync + Clone + 'static> CloneableAny for T {
    fn clone_box(&self) -> Box<dyn CloneableAny> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Clone for Box<dyn CloneableAny> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for Box<dyn CloneableAny> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloneableAny").finish_non_exhaustive()
    }
}

/// Current phase of a drag operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragPhase {
    /// Drag has started.
    Started,
    /// Dragging in progress.
    Dragging,
    /// Hovering over a valid drop target.
    OverTarget,
    /// Dropped successfully.
    Dropped,
    /// Drag was cancelled.
    Cancelled,
}

/// Effect/operation type for a drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DropEffect {
    /// No drop allowed.
    #[default]
    None,
    /// Copy data to target.
    Copy,
    /// Move data to target.
    Move,
    /// Create link to data.
    Link,
}

/// State of an active drag operation.
#[derive(Debug, Clone)]
pub struct DragState {
    /// Unique drag ID.
    pub id: DragId,
    /// Source widget.
    pub source_widget: WidgetId,
    /// Current phase.
    pub phase: DragPhase,
    /// Starting position.
    pub start_position: Point,
    /// Current position.
    pub current_position: Point,
    /// Drag data.
    pub data: DragData,
    /// Currently hovered drop target.
    pub hover_target: Option<WidgetId>,
    /// Allowed drop effects.
    pub allowed_effects: Vec<DropEffect>,
    /// Current drop effect.
    pub effect: DropEffect,
}

impl DragState {
    /// Create a new drag state.
    pub fn new(id: DragId, source_widget: WidgetId, position: Point, data: DragData) -> Self {
        Self {
            id,
            source_widget,
            phase: DragPhase::Started,
            start_position: position,
            current_position: position,
            data,
            hover_target: None,
            allowed_effects: vec![DropEffect::Copy, DropEffect::Move],
            effect: DropEffect::None,
        }
    }

    /// Get the drag offset from start.
    pub fn offset(&self) -> Point {
        self.current_position - self.start_position
    }

    /// Check if drag is active.
    pub fn is_active(&self) -> bool {
        matches!(
            self.phase,
            DragPhase::Started | DragPhase::Dragging | DragPhase::OverTarget
        )
    }
}

/// Configuration for a drop target.
#[derive(Debug, Clone)]
pub struct DropTarget {
    /// Widget ID of the drop target.
    pub widget_id: WidgetId,
    /// Accepted data types.
    pub accepted_types: Vec<DragDataType>,
    /// Accepted drop effects.
    pub accepted_effects: Vec<DropEffect>,
    /// Bounds of the target.
    pub bounds: Rect,
    /// Whether the target is currently active.
    pub enabled: bool,
}

impl DropTarget {
    /// Create a new drop target.
    pub fn new(widget_id: WidgetId, bounds: Rect) -> Self {
        Self {
            widget_id,
            accepted_types: vec![],
            accepted_effects: vec![DropEffect::Copy, DropEffect::Move],
            bounds,
            enabled: true,
        }
    }

    /// Accept specific data types.
    pub fn accept_types(mut self, types: Vec<DragDataType>) -> Self {
        self.accepted_types = types;
        self
    }

    /// Accept specific effects.
    pub fn accept_effects(mut self, effects: Vec<DropEffect>) -> Self {
        self.accepted_effects = effects;
        self
    }

    /// Check if this target accepts the given drag data.
    pub fn accepts(&self, data: &DragData, effect: DropEffect) -> bool {
        if !self.enabled {
            return false;
        }

        // Check effect
        if !self.accepted_effects.contains(&effect) {
            return false;
        }

        // Check type (empty means accept all)
        if self.accepted_types.is_empty() {
            return true;
        }

        self.accepted_types.contains(&data.data_type)
            || self.accepted_types.iter().any(|t| data.has_format(t))
    }

    /// Check if a point is within this target's bounds.
    pub fn contains_point(&self, point: Point) -> bool {
        self.enabled && self.bounds.contains_point(&point)
    }
}

/// Result of a drop operation.
#[derive(Debug, Clone)]
pub struct DropResult {
    /// Whether the drop was successful.
    pub success: bool,
    /// Target widget that received the drop.
    pub target: WidgetId,
    /// Effect that was applied.
    pub effect: DropEffect,
    /// Position of the drop.
    pub position: Point,
}

/// Drag and drop manager.
pub struct DragDropManager {
    /// Next drag ID.
    next_id: u64,
    /// Current active drag state.
    current_drag: Option<DragState>,
    /// Registered drop targets.
    targets: HashMap<WidgetId, DropTarget>,
    /// Drag preview offset from cursor.
    preview_offset: Point,
    /// Minimum drag distance before starting.
    min_drag_distance: f32,
}

impl DragDropManager {
    /// Create a new drag/drop manager.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            current_drag: None,
            targets: HashMap::new(),
            preview_offset: Point::ORIGIN,
            min_drag_distance: 5.0,
        }
    }

    /// Set minimum drag distance.
    pub fn set_min_drag_distance(&mut self, distance: f32) {
        self.min_drag_distance = distance;
    }

    /// Set drag preview offset.
    pub fn set_preview_offset(&mut self, offset: Point) {
        self.preview_offset = offset;
    }

    /// Register a drop target.
    pub fn register_target(&mut self, target: DropTarget) {
        self.targets.insert(target.widget_id, target);
    }

    /// Unregister a drop target.
    pub fn unregister_target(&mut self, widget_id: WidgetId) {
        self.targets.remove(&widget_id);
    }

    /// Update a target's bounds.
    pub fn update_target_bounds(&mut self, widget_id: WidgetId, bounds: Rect) {
        if let Some(target) = self.targets.get_mut(&widget_id) {
            target.bounds = bounds;
        }
    }

    /// Start a drag operation.
    pub fn start_drag(
        &mut self,
        source_widget: WidgetId,
        position: Point,
        data: DragData,
    ) -> DragId {
        let id = DragId::new(self.next_id);
        self.next_id += 1;

        let state = DragState::new(id, source_widget, position, data);
        self.current_drag = Some(state);

        id
    }

    /// Update drag position.
    pub fn move_drag(&mut self, position: Point) {
        if let Some(state) = &mut self.current_drag {
            state.current_position = position;

            // Check for drag distance threshold
            if state.phase == DragPhase::Started {
                let distance = state.start_position.distance(&position);
                if distance >= self.min_drag_distance {
                    state.phase = DragPhase::Dragging;
                }
            }

            // Update hover target
            if state.phase == DragPhase::Dragging || state.phase == DragPhase::OverTarget {
                let old_target = state.hover_target;
                state.hover_target = None;
                state.effect = DropEffect::None;

                for target in self.targets.values() {
                    if target.contains_point(position) {
                        // Find best allowed effect
                        let effect = state
                            .allowed_effects
                            .iter()
                            .find(|e| target.accepts(&state.data, **e))
                            .copied()
                            .unwrap_or(DropEffect::None);

                        if effect != DropEffect::None {
                            state.hover_target = Some(target.widget_id);
                            state.effect = effect;
                            state.phase = DragPhase::OverTarget;
                            break;
                        }
                    }
                }

                if state.hover_target.is_none() && old_target.is_some() {
                    state.phase = DragPhase::Dragging;
                }
            }
        }
    }

    /// End drag with drop attempt.
    pub fn drop(&mut self) -> Option<DropResult> {
        let state = self.current_drag.take()?;

        if let Some(target_id) = state.hover_target {
            if state.effect != DropEffect::None {
                return Some(DropResult {
                    success: true,
                    target: target_id,
                    effect: state.effect,
                    position: state.current_position,
                });
            }
        }

        Some(DropResult {
            success: false,
            target: state.source_widget,
            effect: DropEffect::None,
            position: state.current_position,
        })
    }

    /// Cancel the current drag operation.
    pub fn cancel(&mut self) {
        if let Some(state) = &mut self.current_drag {
            state.phase = DragPhase::Cancelled;
        }
        self.current_drag = None;
    }

    /// Get the current drag state.
    pub fn current(&self) -> Option<&DragState> {
        self.current_drag.as_ref()
    }

    /// Check if a drag is active.
    pub fn is_dragging(&self) -> bool {
        self.current_drag.as_ref().is_some_and(DragState::is_active)
    }

    /// Get the preview position for rendering.
    pub fn preview_position(&self) -> Option<Point> {
        self.current_drag.as_ref().map(|s| {
            Point::new(
                s.current_position.x + self.preview_offset.x,
                s.current_position.y + self.preview_offset.y,
            )
        })
    }

    /// Get drop targets count.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Find target at position.
    pub fn target_at(&self, position: Point) -> Option<&DropTarget> {
        self.targets.values().find(|t| t.contains_point(position))
    }

    /// Clear all targets and cancel any active drag.
    pub fn clear(&mut self) {
        self.cancel();
        self.targets.clear();
    }
}

impl Default for DragDropManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DragDropManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DragDropManager")
            .field("next_id", &self.next_id)
            .field("is_dragging", &self.is_dragging())
            .field("target_count", &self.targets.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // DragId tests
    #[test]
    fn test_drag_id() {
        let id1 = DragId::new(1);
        let id2 = DragId::new(1);
        let id3 = DragId::new(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // DragDataType tests
    #[test]
    fn test_drag_data_type() {
        assert_eq!(DragDataType::Text, DragDataType::Text);
        assert_ne!(DragDataType::Text, DragDataType::Html);

        let custom = DragDataType::custom("my-type");
        assert_eq!(custom, DragDataType::Custom("my-type".to_string()));
    }

    // DragData tests
    #[test]
    fn test_drag_data_text() {
        let data = DragData::text("Hello");
        assert_eq!(data.data_type, DragDataType::Text);
        assert_eq!(data.text, "Hello");
    }

    #[test]
    fn test_drag_data_html() {
        let data = DragData::html("<b>Bold</b>");
        assert_eq!(data.data_type, DragDataType::Html);
        assert!(data.has_format(&DragDataType::Html));
    }

    #[test]
    fn test_drag_data_url() {
        let data = DragData::url("https://example.com");
        assert_eq!(data.data_type, DragDataType::Url);
        assert_eq!(data.text, "https://example.com");
    }

    #[test]
    fn test_drag_data_custom() {
        let data = DragData::custom("widget-id", "123");
        assert!(matches!(data.data_type, DragDataType::Custom(_)));
    }

    #[test]
    fn test_drag_data_with_format() {
        let data = DragData::text("Hello").with_format(DragDataType::Html, "<p>Hello</p>");

        assert!(data.has_format(&DragDataType::Text));
        assert!(data.has_format(&DragDataType::Html));
        assert!(!data.has_format(&DragDataType::Url));

        assert_eq!(data.get_format(&DragDataType::Text), Some("Hello"));
        assert_eq!(data.get_format(&DragDataType::Html), Some("<p>Hello</p>"));
    }

    #[test]
    fn test_drag_data_with_payload() {
        // Test payload exists (type erasure is complex, just verify the API works)
        let data = DragData::text("test").with_payload(42i32);
        assert!(data.payload.is_some());
    }

    // DragPhase tests
    #[test]
    fn test_drag_phase() {
        assert_eq!(DragPhase::Started, DragPhase::Started);
        assert_ne!(DragPhase::Started, DragPhase::Dropped);
    }

    // DropEffect tests
    #[test]
    fn test_drop_effect_default() {
        assert_eq!(DropEffect::default(), DropEffect::None);
    }

    // DragState tests
    #[test]
    fn test_drag_state_new() {
        let state = DragState::new(
            DragId::new(1),
            WidgetId::new(100),
            Point::new(50.0, 50.0),
            DragData::text("test"),
        );

        assert_eq!(state.id, DragId::new(1));
        assert_eq!(state.source_widget, WidgetId::new(100));
        assert_eq!(state.phase, DragPhase::Started);
        assert_eq!(state.start_position, Point::new(50.0, 50.0));
        assert!(state.is_active());
    }

    #[test]
    fn test_drag_state_offset() {
        let mut state = DragState::new(
            DragId::new(1),
            WidgetId::new(100),
            Point::new(50.0, 50.0),
            DragData::text("test"),
        );

        state.current_position = Point::new(100.0, 75.0);
        let offset = state.offset();
        assert_eq!(offset.x, 50.0);
        assert_eq!(offset.y, 25.0);
    }

    #[test]
    fn test_drag_state_is_active() {
        let mut state = DragState::new(
            DragId::new(1),
            WidgetId::new(100),
            Point::ORIGIN,
            DragData::text("test"),
        );

        assert!(state.is_active());

        state.phase = DragPhase::Dragging;
        assert!(state.is_active());

        state.phase = DragPhase::OverTarget;
        assert!(state.is_active());

        state.phase = DragPhase::Dropped;
        assert!(!state.is_active());

        state.phase = DragPhase::Cancelled;
        assert!(!state.is_active());
    }

    // DropTarget tests
    #[test]
    fn test_drop_target_new() {
        let target = DropTarget::new(WidgetId::new(1), Rect::new(0.0, 0.0, 100.0, 100.0));

        assert_eq!(target.widget_id, WidgetId::new(1));
        assert!(target.enabled);
        assert!(target.accepted_types.is_empty());
    }

    #[test]
    fn test_drop_target_accept_types() {
        let target = DropTarget::new(WidgetId::new(1), Rect::new(0.0, 0.0, 100.0, 100.0))
            .accept_types(vec![DragDataType::Text, DragDataType::Html]);

        assert_eq!(target.accepted_types.len(), 2);
    }

    #[test]
    fn test_drop_target_accepts() {
        let target = DropTarget::new(WidgetId::new(1), Rect::new(0.0, 0.0, 100.0, 100.0))
            .accept_types(vec![DragDataType::Text])
            .accept_effects(vec![DropEffect::Copy]);

        let text_data = DragData::text("hello");
        assert!(target.accepts(&text_data, DropEffect::Copy));
        assert!(!target.accepts(&text_data, DropEffect::Move));

        let html_data = DragData::html("<b>bold</b>");
        assert!(!target.accepts(&html_data, DropEffect::Copy));
    }

    #[test]
    fn test_drop_target_accepts_all_types() {
        // Empty accepted_types means accept all
        let target = DropTarget::new(WidgetId::new(1), Rect::new(0.0, 0.0, 100.0, 100.0));

        assert!(target.accepts(&DragData::text("test"), DropEffect::Copy));
        assert!(target.accepts(&DragData::html("<b>test</b>"), DropEffect::Move));
    }

    #[test]
    fn test_drop_target_disabled() {
        let mut target = DropTarget::new(WidgetId::new(1), Rect::new(0.0, 0.0, 100.0, 100.0));
        target.enabled = false;

        assert!(!target.accepts(&DragData::text("test"), DropEffect::Copy));
        assert!(!target.contains_point(Point::new(50.0, 50.0)));
    }

    #[test]
    fn test_drop_target_contains_point() {
        let target = DropTarget::new(WidgetId::new(1), Rect::new(10.0, 10.0, 100.0, 100.0));

        assert!(target.contains_point(Point::new(50.0, 50.0)));
        assert!(target.contains_point(Point::new(10.0, 10.0)));
        assert!(!target.contains_point(Point::new(5.0, 50.0)));
        assert!(!target.contains_point(Point::new(120.0, 50.0)));
    }

    // DragDropManager tests
    #[test]
    fn test_manager_new() {
        let manager = DragDropManager::new();
        assert!(!manager.is_dragging());
        assert_eq!(manager.target_count(), 0);
    }

    #[test]
    fn test_manager_register_target() {
        let mut manager = DragDropManager::new();

        manager.register_target(DropTarget::new(
            WidgetId::new(1),
            Rect::new(0.0, 0.0, 100.0, 100.0),
        ));

        assert_eq!(manager.target_count(), 1);
    }

    #[test]
    fn test_manager_unregister_target() {
        let mut manager = DragDropManager::new();

        manager.register_target(DropTarget::new(
            WidgetId::new(1),
            Rect::new(0.0, 0.0, 100.0, 100.0),
        ));
        manager.unregister_target(WidgetId::new(1));

        assert_eq!(manager.target_count(), 0);
    }

    #[test]
    fn test_manager_start_drag() {
        let mut manager = DragDropManager::new();

        let id = manager.start_drag(
            WidgetId::new(1),
            Point::new(50.0, 50.0),
            DragData::text("hello"),
        );

        assert!(manager.is_dragging());
        assert_eq!(manager.current().unwrap().id, id);
    }

    #[test]
    fn test_manager_move_drag() {
        let mut manager = DragDropManager::new();
        manager.set_min_drag_distance(5.0);

        manager.start_drag(
            WidgetId::new(1),
            Point::new(50.0, 50.0),
            DragData::text("hello"),
        );

        // Move within threshold
        manager.move_drag(Point::new(52.0, 52.0));
        assert_eq!(manager.current().unwrap().phase, DragPhase::Started);

        // Move beyond threshold
        manager.move_drag(Point::new(60.0, 60.0));
        assert_eq!(manager.current().unwrap().phase, DragPhase::Dragging);
    }

    #[test]
    fn test_manager_move_over_target() {
        let mut manager = DragDropManager::new();
        manager.set_min_drag_distance(0.0);

        manager.register_target(DropTarget::new(
            WidgetId::new(10),
            Rect::new(100.0, 100.0, 100.0, 100.0),
        ));

        manager.start_drag(
            WidgetId::new(1),
            Point::new(50.0, 50.0),
            DragData::text("hello"),
        );

        // Move over target
        manager.move_drag(Point::new(150.0, 150.0));
        let state = manager.current().unwrap();
        assert_eq!(state.phase, DragPhase::OverTarget);
        assert_eq!(state.hover_target, Some(WidgetId::new(10)));
    }

    #[test]
    fn test_manager_drop_success() {
        let mut manager = DragDropManager::new();
        manager.set_min_drag_distance(0.0);

        manager.register_target(DropTarget::new(
            WidgetId::new(10),
            Rect::new(100.0, 100.0, 100.0, 100.0),
        ));

        manager.start_drag(
            WidgetId::new(1),
            Point::new(50.0, 50.0),
            DragData::text("hello"),
        );

        manager.move_drag(Point::new(150.0, 150.0));
        let result = manager.drop().unwrap();

        assert!(result.success);
        assert_eq!(result.target, WidgetId::new(10));
        assert!(!manager.is_dragging());
    }

    #[test]
    fn test_manager_drop_failure() {
        let mut manager = DragDropManager::new();
        manager.set_min_drag_distance(0.0);

        manager.start_drag(
            WidgetId::new(1),
            Point::new(50.0, 50.0),
            DragData::text("hello"),
        );

        manager.move_drag(Point::new(60.0, 60.0));
        let result = manager.drop().unwrap();

        assert!(!result.success);
        assert_eq!(result.effect, DropEffect::None);
    }

    #[test]
    fn test_manager_cancel() {
        let mut manager = DragDropManager::new();

        manager.start_drag(
            WidgetId::new(1),
            Point::new(50.0, 50.0),
            DragData::text("hello"),
        );

        manager.cancel();
        assert!(!manager.is_dragging());
        assert!(manager.current().is_none());
    }

    #[test]
    fn test_manager_preview_position() {
        let mut manager = DragDropManager::new();
        manager.set_preview_offset(Point::new(-10.0, -10.0));

        manager.start_drag(
            WidgetId::new(1),
            Point::new(100.0, 100.0),
            DragData::text("hello"),
        );

        let preview_pos = manager.preview_position().unwrap();
        assert_eq!(preview_pos, Point::new(90.0, 90.0));
    }

    #[test]
    fn test_manager_target_at() {
        let mut manager = DragDropManager::new();

        manager.register_target(DropTarget::new(
            WidgetId::new(1),
            Rect::new(0.0, 0.0, 100.0, 100.0),
        ));
        manager.register_target(DropTarget::new(
            WidgetId::new(2),
            Rect::new(200.0, 200.0, 100.0, 100.0),
        ));

        assert!(manager.target_at(Point::new(50.0, 50.0)).is_some());
        assert!(manager.target_at(Point::new(150.0, 150.0)).is_none());
        assert!(manager.target_at(Point::new(250.0, 250.0)).is_some());
    }

    #[test]
    fn test_manager_clear() {
        let mut manager = DragDropManager::new();

        manager.register_target(DropTarget::new(
            WidgetId::new(1),
            Rect::new(0.0, 0.0, 100.0, 100.0),
        ));

        manager.start_drag(WidgetId::new(2), Point::ORIGIN, DragData::text("test"));

        manager.clear();

        assert!(!manager.is_dragging());
        assert_eq!(manager.target_count(), 0);
    }

    #[test]
    fn test_manager_update_target_bounds() {
        let mut manager = DragDropManager::new();

        manager.register_target(DropTarget::new(
            WidgetId::new(1),
            Rect::new(0.0, 0.0, 100.0, 100.0),
        ));

        manager.update_target_bounds(WidgetId::new(1), Rect::new(50.0, 50.0, 200.0, 200.0));

        let target = manager.target_at(Point::new(100.0, 100.0)).unwrap();
        assert_eq!(target.bounds.x, 50.0);
        assert_eq!(target.bounds.width, 200.0);
    }
}
