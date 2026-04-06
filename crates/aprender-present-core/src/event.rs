//! Input events for widgets.

use crate::geometry::Point;
use serde::{Deserialize, Serialize};

/// Input event types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// Mouse moved to position
    MouseMove {
        /// New position
        position: Point,
    },
    /// Mouse button pressed
    MouseDown {
        /// Position of click
        position: Point,
        /// Button pressed
        button: MouseButton,
    },
    /// Mouse button released
    MouseUp {
        /// Position of release
        position: Point,
        /// Button released
        button: MouseButton,
    },
    /// Mouse wheel scrolled
    Scroll {
        /// Horizontal scroll delta
        delta_x: f32,
        /// Vertical scroll delta
        delta_y: f32,
    },
    /// Key pressed
    KeyDown {
        /// Key pressed
        key: Key,
        /// Active modifier keys (Ctrl, Alt, Shift, Meta)
        #[serde(default)]
        modifiers: crate::shortcut::Modifiers,
    },
    /// Key released
    KeyUp {
        /// Key released
        key: Key,
        /// Active modifier keys (Ctrl, Alt, Shift, Meta)
        #[serde(default)]
        modifiers: crate::shortcut::Modifiers,
    },
    /// Text input received
    TextInput {
        /// Input text
        text: String,
    },
    /// Widget gained focus
    FocusIn,
    /// Widget lost focus
    FocusOut,
    /// Mouse entered widget bounds
    MouseEnter,
    /// Mouse left widget bounds
    MouseLeave,
    /// Window resized
    Resize {
        /// New width
        width: f32,
        /// New height
        height: f32,
    },
    // Touch events
    /// Touch started
    TouchStart {
        /// Touch identifier
        id: TouchId,
        /// Touch position
        position: Point,
        /// Touch pressure (0.0 to 1.0)
        pressure: f32,
    },
    /// Touch moved
    TouchMove {
        /// Touch identifier
        id: TouchId,
        /// New position
        position: Point,
        /// Touch pressure
        pressure: f32,
    },
    /// Touch ended
    TouchEnd {
        /// Touch identifier
        id: TouchId,
        /// Final position
        position: Point,
    },
    /// Touch cancelled (e.g., palm rejection)
    TouchCancel {
        /// Touch identifier
        id: TouchId,
    },
    // Pointer events (unified mouse/touch/pen)
    /// Pointer down
    PointerDown {
        /// Pointer ID
        pointer_id: PointerId,
        /// Pointer type
        pointer_type: PointerType,
        /// Position
        position: Point,
        /// Pressure
        pressure: f32,
        /// Is primary pointer
        is_primary: bool,
        /// Button (for mouse pointers)
        button: Option<MouseButton>,
    },
    /// Pointer moved
    PointerMove {
        /// Pointer ID
        pointer_id: PointerId,
        /// Pointer type
        pointer_type: PointerType,
        /// Position
        position: Point,
        /// Pressure
        pressure: f32,
        /// Is primary pointer
        is_primary: bool,
    },
    /// Pointer up
    PointerUp {
        /// Pointer ID
        pointer_id: PointerId,
        /// Pointer type
        pointer_type: PointerType,
        /// Position
        position: Point,
        /// Is primary pointer
        is_primary: bool,
        /// Button (for mouse pointers)
        button: Option<MouseButton>,
    },
    /// Pointer cancelled
    PointerCancel {
        /// Pointer ID
        pointer_id: PointerId,
    },
    /// Pointer entered element
    PointerEnter {
        /// Pointer ID
        pointer_id: PointerId,
        /// Pointer type
        pointer_type: PointerType,
    },
    /// Pointer left element
    PointerLeave {
        /// Pointer ID
        pointer_id: PointerId,
        /// Pointer type
        pointer_type: PointerType,
    },
    // Gesture events
    /// Pinch gesture
    GesturePinch {
        /// Scale factor
        scale: f32,
        /// Center point
        center: Point,
        /// Gesture state
        state: GestureState,
    },
    /// Rotate gesture
    GestureRotate {
        /// Rotation angle in radians
        angle: f32,
        /// Center point
        center: Point,
        /// Gesture state
        state: GestureState,
    },
    /// Pan/drag gesture
    GesturePan {
        /// Translation delta
        delta: Point,
        /// Velocity
        velocity: Point,
        /// Gesture state
        state: GestureState,
    },
    /// Long press gesture
    GestureLongPress {
        /// Position
        position: Point,
    },
    /// Tap gesture
    GestureTap {
        /// Position
        position: Point,
        /// Number of taps (1 = single, 2 = double)
        count: u8,
    },
}

/// Touch identifier for multi-touch tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TouchId(pub u32);

/// Pointer identifier for pointer events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct PointerId(pub u32);

/// Type of pointer device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PointerType {
    /// Mouse pointer
    #[default]
    Mouse,
    /// Touch pointer
    Touch,
    /// Pen/stylus pointer
    Pen,
}

/// State of a gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GestureState {
    /// Gesture started
    #[default]
    Started,
    /// Gesture in progress (changed)
    Changed,
    /// Gesture ended
    Ended,
    /// Gesture cancelled
    Cancelled,
}

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MouseButton {
    /// Left mouse button
    Left,
    /// Right mouse button
    Right,
    /// Middle mouse button (wheel click)
    Middle,
    /// Additional button 1
    Button4,
    /// Additional button 2
    Button5,
}

/// Keyboard key identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Key {
    // Letters
    /// A key
    A,
    /// B key
    B,
    /// C key
    C,
    /// D key
    D,
    /// E key
    E,
    /// F key
    F,
    /// G key
    G,
    /// H key
    H,
    /// I key
    I,
    /// J key
    J,
    /// K key
    K,
    /// L key
    L,
    /// M key
    M,
    /// N key
    N,
    /// O key
    O,
    /// P key
    P,
    /// Q key
    Q,
    /// R key
    R,
    /// S key
    S,
    /// T key
    T,
    /// U key
    U,
    /// V key
    V,
    /// W key
    W,
    /// X key
    X,
    /// Y key
    Y,
    /// Z key
    Z,

    // Numbers
    /// 0 key
    Num0,
    /// 1 key
    Num1,
    /// 2 key
    Num2,
    /// 3 key
    Num3,
    /// 4 key
    Num4,
    /// 5 key
    Num5,
    /// 6 key
    Num6,
    /// 7 key
    Num7,
    /// 8 key
    Num8,
    /// 9 key
    Num9,

    // Function keys
    /// F1 key
    F1,
    /// F2 key
    F2,
    /// F3 key
    F3,
    /// F4 key
    F4,
    /// F5 key
    F5,
    /// F6 key
    F6,
    /// F7 key
    F7,
    /// F8 key
    F8,
    /// F9 key
    F9,
    /// F10 key
    F10,
    /// F11 key
    F11,
    /// F12 key
    F12,

    // Control keys
    /// Enter/Return key
    Enter,
    /// Escape key
    Escape,
    /// Backspace key
    Backspace,
    /// Tab key
    Tab,
    /// Space key
    Space,
    /// Delete key
    Delete,
    /// Insert key
    Insert,
    /// Home key
    Home,
    /// End key
    End,
    /// Page Up key
    PageUp,
    /// Page Down key
    PageDown,

    // Arrow keys
    /// Up arrow
    Up,
    /// Down arrow
    Down,
    /// Left arrow
    Left,
    /// Right arrow
    Right,

    // Modifiers
    /// Left Shift
    ShiftLeft,
    /// Right Shift
    ShiftRight,
    /// Left Control
    ControlLeft,
    /// Right Control
    ControlRight,
    /// Left Alt
    AltLeft,
    /// Right Alt
    AltRight,
    /// Left Meta (Windows/Command)
    MetaLeft,
    /// Right Meta (Windows/Command)
    MetaRight,

    // Punctuation
    /// Minus key
    Minus,
    /// Equals key
    Equal,
    /// Left bracket
    BracketLeft,
    /// Right bracket
    BracketRight,
    /// Backslash
    Backslash,
    /// Semicolon
    Semicolon,
    /// Quote/apostrophe
    Quote,
    /// Grave/backtick
    Grave,
    /// Comma
    Comma,
    /// Period
    Period,
    /// Slash
    Slash,
}

impl Event {
    /// Create a key down event with no modifiers.
    #[must_use]
    pub const fn key_down(key: Key) -> Self {
        Self::KeyDown {
            key,
            modifiers: crate::shortcut::Modifiers::NONE,
        }
    }

    /// Create a key up event with no modifiers.
    #[must_use]
    pub const fn key_up(key: Key) -> Self {
        Self::KeyUp {
            key,
            modifiers: crate::shortcut::Modifiers::NONE,
        }
    }

    /// Check if this is a mouse event.
    #[must_use]
    pub const fn is_mouse(&self) -> bool {
        matches!(
            self,
            Self::MouseMove { .. }
                | Self::MouseDown { .. }
                | Self::MouseUp { .. }
                | Self::MouseEnter
                | Self::MouseLeave
        )
    }

    /// Check if this is a keyboard event.
    #[must_use]
    pub const fn is_keyboard(&self) -> bool {
        matches!(
            self,
            Self::KeyDown { .. } | Self::KeyUp { .. } | Self::TextInput { .. }
        )
    }

    /// Check if this is a focus event.
    #[must_use]
    pub const fn is_focus(&self) -> bool {
        matches!(self, Self::FocusIn | Self::FocusOut)
    }

    /// Check if this is a touch event.
    #[must_use]
    pub const fn is_touch(&self) -> bool {
        matches!(
            self,
            Self::TouchStart { .. }
                | Self::TouchMove { .. }
                | Self::TouchEnd { .. }
                | Self::TouchCancel { .. }
        )
    }

    /// Check if this is a pointer event.
    #[must_use]
    pub const fn is_pointer(&self) -> bool {
        matches!(
            self,
            Self::PointerDown { .. }
                | Self::PointerMove { .. }
                | Self::PointerUp { .. }
                | Self::PointerCancel { .. }
                | Self::PointerEnter { .. }
                | Self::PointerLeave { .. }
        )
    }

    /// Check if this is a gesture event.
    #[must_use]
    pub const fn is_gesture(&self) -> bool {
        matches!(
            self,
            Self::GesturePinch { .. }
                | Self::GestureRotate { .. }
                | Self::GesturePan { .. }
                | Self::GestureLongPress { .. }
                | Self::GestureTap { .. }
        )
    }

    /// Get the position if this is a positional event.
    #[must_use]
    pub const fn position(&self) -> Option<Point> {
        match self {
            Self::MouseMove { position }
            | Self::MouseDown { position, .. }
            | Self::MouseUp { position, .. }
            | Self::TouchStart { position, .. }
            | Self::TouchMove { position, .. }
            | Self::TouchEnd { position, .. }
            | Self::PointerDown { position, .. }
            | Self::PointerMove { position, .. }
            | Self::PointerUp { position, .. }
            | Self::GestureLongPress { position }
            | Self::GestureTap { position, .. } => Some(*position),
            Self::GesturePinch { center, .. } | Self::GestureRotate { center, .. } => Some(*center),
            _ => None,
        }
    }

    /// Get the touch ID if this is a touch event.
    #[must_use]
    pub const fn touch_id(&self) -> Option<TouchId> {
        match self {
            Self::TouchStart { id, .. }
            | Self::TouchMove { id, .. }
            | Self::TouchEnd { id, .. }
            | Self::TouchCancel { id } => Some(*id),
            _ => None,
        }
    }

    /// Get the pointer ID if this is a pointer event.
    #[must_use]
    pub const fn pointer_id(&self) -> Option<PointerId> {
        match self {
            Self::PointerDown { pointer_id, .. }
            | Self::PointerMove { pointer_id, .. }
            | Self::PointerUp { pointer_id, .. }
            | Self::PointerCancel { pointer_id }
            | Self::PointerEnter { pointer_id, .. }
            | Self::PointerLeave { pointer_id, .. } => Some(*pointer_id),
            _ => None,
        }
    }

    /// Get the pointer type if this is a pointer event.
    #[must_use]
    pub const fn pointer_type(&self) -> Option<PointerType> {
        match self {
            Self::PointerDown { pointer_type, .. }
            | Self::PointerMove { pointer_type, .. }
            | Self::PointerUp { pointer_type, .. }
            | Self::PointerEnter { pointer_type, .. }
            | Self::PointerLeave { pointer_type, .. } => Some(*pointer_type),
            _ => None,
        }
    }

    /// Get pressure if available (0.0 to 1.0).
    #[must_use]
    pub const fn pressure(&self) -> Option<f32> {
        match self {
            Self::TouchStart { pressure, .. }
            | Self::TouchMove { pressure, .. }
            | Self::PointerDown { pressure, .. }
            | Self::PointerMove { pressure, .. } => Some(*pressure),
            _ => None,
        }
    }

    /// Get gesture state if this is a gesture event.
    #[must_use]
    pub const fn gesture_state(&self) -> Option<GestureState> {
        match self {
            Self::GesturePinch { state, .. }
            | Self::GestureRotate { state, .. }
            | Self::GesturePan { state, .. } => Some(*state),
            _ => None,
        }
    }
}

impl TouchId {
    /// Create a new touch ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl PointerId {
    /// Create a new pointer ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl PointerType {
    /// Check if this is a mouse pointer.
    #[must_use]
    pub const fn is_mouse(&self) -> bool {
        matches!(self, Self::Mouse)
    }

    /// Check if this is a touch pointer.
    #[must_use]
    pub const fn is_touch(&self) -> bool {
        matches!(self, Self::Touch)
    }

    /// Check if this is a pen pointer.
    #[must_use]
    pub const fn is_pen(&self) -> bool {
        matches!(self, Self::Pen)
    }
}

impl GestureState {
    /// Check if gesture is starting.
    #[must_use]
    pub const fn is_start(&self) -> bool {
        matches!(self, Self::Started)
    }

    /// Check if gesture is in progress.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Started | Self::Changed)
    }

    /// Check if gesture has ended.
    #[must_use]
    pub const fn is_end(&self) -> bool {
        matches!(self, Self::Ended | Self::Cancelled)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_event_is_mouse() {
        assert!(Event::MouseMove {
            position: Point::ORIGIN
        }
        .is_mouse());
        assert!(Event::MouseEnter.is_mouse());
        assert!(!Event::key_down(Key::Enter).is_mouse());
    }

    #[test]
    fn test_event_is_keyboard() {
        assert!(Event::key_down(Key::A).is_keyboard());
        assert!(Event::TextInput {
            text: "x".to_string()
        }
        .is_keyboard());
        assert!(!Event::MouseMove {
            position: Point::ORIGIN
        }
        .is_keyboard());
    }

    #[test]
    fn test_event_is_focus() {
        assert!(Event::FocusIn.is_focus());
        assert!(Event::FocusOut.is_focus());
        assert!(!Event::key_down(Key::Tab).is_focus());
    }

    #[test]
    fn test_event_position() {
        let pos = Point::new(100.0, 200.0);
        assert_eq!(Event::MouseMove { position: pos }.position(), Some(pos));
        assert_eq!(
            Event::MouseDown {
                position: pos,
                button: MouseButton::Left
            }
            .position(),
            Some(pos)
        );
        assert_eq!(Event::FocusIn.position(), None);
    }

    #[test]
    fn test_event_mouse_up_position() {
        let pos = Point::new(50.0, 75.0);
        let event = Event::MouseUp {
            position: pos,
            button: MouseButton::Right,
        };
        assert_eq!(event.position(), Some(pos));
        assert!(event.is_mouse());
    }

    #[test]
    fn test_event_scroll() {
        let event = Event::Scroll {
            delta_x: 10.0,
            delta_y: -5.0,
        };
        assert!(!event.is_mouse());
        assert!(!event.is_keyboard());
        assert!(event.position().is_none());
    }

    #[test]
    fn test_event_resize() {
        let event = Event::Resize {
            width: 800.0,
            height: 600.0,
        };
        assert!(!event.is_mouse());
        assert!(!event.is_keyboard());
        assert!(!event.is_focus());
    }

    #[test]
    fn test_event_key_up() {
        let event = Event::key_up(Key::Space);
        assert!(event.is_keyboard());
        assert!(!event.is_mouse());
    }

    #[test]
    fn test_mouse_button_equality() {
        assert_eq!(MouseButton::Left, MouseButton::Left);
        assert_ne!(MouseButton::Left, MouseButton::Right);
    }

    #[test]
    fn test_key_equality() {
        assert_eq!(Key::Enter, Key::Enter);
        assert_ne!(Key::Enter, Key::Space);
    }

    #[test]
    fn test_event_mouse_leave() {
        let event = Event::MouseLeave;
        assert!(event.is_mouse());
        assert!(event.position().is_none());
    }

    // Touch event tests
    #[test]
    fn test_touch_start() {
        let event = Event::TouchStart {
            id: TouchId::new(1),
            position: Point::new(100.0, 200.0),
            pressure: 0.8,
        };
        assert!(event.is_touch());
        assert!(!event.is_mouse());
        assert!(!event.is_pointer());
        assert_eq!(event.touch_id(), Some(TouchId(1)));
        assert_eq!(event.position(), Some(Point::new(100.0, 200.0)));
        assert_eq!(event.pressure(), Some(0.8));
    }

    #[test]
    fn test_touch_move() {
        let event = Event::TouchMove {
            id: TouchId::new(2),
            position: Point::new(150.0, 250.0),
            pressure: 0.5,
        };
        assert!(event.is_touch());
        assert_eq!(event.touch_id(), Some(TouchId(2)));
        assert_eq!(event.position(), Some(Point::new(150.0, 250.0)));
        assert_eq!(event.pressure(), Some(0.5));
    }

    #[test]
    fn test_touch_end() {
        let event = Event::TouchEnd {
            id: TouchId::new(3),
            position: Point::new(200.0, 300.0),
        };
        assert!(event.is_touch());
        assert_eq!(event.touch_id(), Some(TouchId(3)));
        assert_eq!(event.position(), Some(Point::new(200.0, 300.0)));
        assert!(event.pressure().is_none());
    }

    #[test]
    fn test_touch_cancel() {
        let event = Event::TouchCancel {
            id: TouchId::new(4),
        };
        assert!(event.is_touch());
        assert_eq!(event.touch_id(), Some(TouchId(4)));
        assert!(event.position().is_none());
    }

    #[test]
    fn test_touch_id_creation() {
        let id = TouchId::new(42);
        assert_eq!(id.0, 42);
        let default_id = TouchId::default();
        assert_eq!(default_id.0, 0);
    }

    // Pointer event tests
    #[test]
    fn test_pointer_down() {
        let event = Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::new(100.0, 200.0),
            pressure: 0.7,
            is_primary: true,
            button: None,
        };
        assert!(event.is_pointer());
        assert!(!event.is_touch());
        assert!(!event.is_mouse());
        assert_eq!(event.pointer_id(), Some(PointerId(1)));
        assert_eq!(event.pointer_type(), Some(PointerType::Touch));
        assert_eq!(event.position(), Some(Point::new(100.0, 200.0)));
        assert_eq!(event.pressure(), Some(0.7));
    }

    #[test]
    fn test_pointer_down_with_mouse_button() {
        let event = Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Mouse,
            position: Point::new(50.0, 75.0),
            pressure: 0.5,
            is_primary: true,
            button: Some(MouseButton::Left),
        };
        assert!(event.is_pointer());
        assert_eq!(event.pointer_type(), Some(PointerType::Mouse));
    }

    #[test]
    fn test_pointer_move() {
        let event = Event::PointerMove {
            pointer_id: PointerId::new(2),
            pointer_type: PointerType::Pen,
            position: Point::new(150.0, 250.0),
            pressure: 0.9,
            is_primary: false,
        };
        assert!(event.is_pointer());
        assert_eq!(event.pointer_id(), Some(PointerId(2)));
        assert_eq!(event.pointer_type(), Some(PointerType::Pen));
        assert_eq!(event.pressure(), Some(0.9));
    }

    #[test]
    fn test_pointer_up() {
        let event = Event::PointerUp {
            pointer_id: PointerId::new(3),
            pointer_type: PointerType::Mouse,
            position: Point::new(200.0, 300.0),
            is_primary: true,
            button: Some(MouseButton::Right),
        };
        assert!(event.is_pointer());
        assert_eq!(event.pointer_id(), Some(PointerId(3)));
        assert!(event.pressure().is_none());
    }

    #[test]
    fn test_pointer_cancel() {
        let event = Event::PointerCancel {
            pointer_id: PointerId::new(4),
        };
        assert!(event.is_pointer());
        assert_eq!(event.pointer_id(), Some(PointerId(4)));
        assert!(event.pointer_type().is_none());
        assert!(event.position().is_none());
    }

    #[test]
    fn test_pointer_enter() {
        let event = Event::PointerEnter {
            pointer_id: PointerId::new(5),
            pointer_type: PointerType::Mouse,
        };
        assert!(event.is_pointer());
        assert_eq!(event.pointer_id(), Some(PointerId(5)));
        assert_eq!(event.pointer_type(), Some(PointerType::Mouse));
        assert!(event.position().is_none());
    }

    #[test]
    fn test_pointer_leave() {
        let event = Event::PointerLeave {
            pointer_id: PointerId::new(6),
            pointer_type: PointerType::Touch,
        };
        assert!(event.is_pointer());
        assert_eq!(event.pointer_id(), Some(PointerId(6)));
        assert_eq!(event.pointer_type(), Some(PointerType::Touch));
    }

    #[test]
    fn test_pointer_id_creation() {
        let id = PointerId::new(99);
        assert_eq!(id.0, 99);
        let default_id = PointerId::default();
        assert_eq!(default_id.0, 0);
    }

    #[test]
    fn test_pointer_type_helpers() {
        assert!(PointerType::Mouse.is_mouse());
        assert!(!PointerType::Mouse.is_touch());
        assert!(!PointerType::Mouse.is_pen());

        assert!(!PointerType::Touch.is_mouse());
        assert!(PointerType::Touch.is_touch());
        assert!(!PointerType::Touch.is_pen());

        assert!(!PointerType::Pen.is_mouse());
        assert!(!PointerType::Pen.is_touch());
        assert!(PointerType::Pen.is_pen());
    }

    #[test]
    fn test_pointer_type_default() {
        let default = PointerType::default();
        assert_eq!(default, PointerType::Mouse);
    }

    // Gesture event tests
    #[test]
    fn test_gesture_pinch() {
        let event = Event::GesturePinch {
            scale: 1.5,
            center: Point::new(200.0, 200.0),
            state: GestureState::Changed,
        };
        assert!(event.is_gesture());
        assert!(!event.is_touch());
        assert!(!event.is_pointer());
        assert_eq!(event.gesture_state(), Some(GestureState::Changed));
        assert_eq!(event.position(), Some(Point::new(200.0, 200.0)));
    }

    #[test]
    fn test_gesture_rotate() {
        let event = Event::GestureRotate {
            angle: std::f32::consts::PI / 4.0,
            center: Point::new(150.0, 150.0),
            state: GestureState::Started,
        };
        assert!(event.is_gesture());
        assert_eq!(event.gesture_state(), Some(GestureState::Started));
        assert_eq!(event.position(), Some(Point::new(150.0, 150.0)));
    }

    #[test]
    fn test_gesture_pan() {
        let event = Event::GesturePan {
            delta: Point::new(10.0, -5.0),
            velocity: Point::new(100.0, -50.0),
            state: GestureState::Ended,
        };
        assert!(event.is_gesture());
        assert_eq!(event.gesture_state(), Some(GestureState::Ended));
        assert!(event.position().is_none());
    }

    #[test]
    fn test_gesture_long_press() {
        let event = Event::GestureLongPress {
            position: Point::new(100.0, 100.0),
        };
        assert!(event.is_gesture());
        assert_eq!(event.position(), Some(Point::new(100.0, 100.0)));
        assert!(event.gesture_state().is_none());
    }

    #[test]
    fn test_gesture_tap() {
        let single_tap = Event::GestureTap {
            position: Point::new(50.0, 50.0),
            count: 1,
        };
        assert!(single_tap.is_gesture());
        assert_eq!(single_tap.position(), Some(Point::new(50.0, 50.0)));

        let double_tap = Event::GestureTap {
            position: Point::new(50.0, 50.0),
            count: 2,
        };
        assert!(double_tap.is_gesture());
    }

    #[test]
    fn test_gesture_state_helpers() {
        assert!(GestureState::Started.is_start());
        assert!(GestureState::Started.is_active());
        assert!(!GestureState::Started.is_end());

        assert!(!GestureState::Changed.is_start());
        assert!(GestureState::Changed.is_active());
        assert!(!GestureState::Changed.is_end());

        assert!(!GestureState::Ended.is_start());
        assert!(!GestureState::Ended.is_active());
        assert!(GestureState::Ended.is_end());

        assert!(!GestureState::Cancelled.is_start());
        assert!(!GestureState::Cancelled.is_active());
        assert!(GestureState::Cancelled.is_end());
    }

    #[test]
    fn test_gesture_state_default() {
        let default = GestureState::default();
        assert_eq!(default, GestureState::Started);
    }

    // Cross-category tests
    #[test]
    fn test_event_category_exclusivity() {
        let touch = Event::TouchStart {
            id: TouchId::new(1),
            position: Point::ORIGIN,
            pressure: 0.5,
        };
        assert!(touch.is_touch());
        assert!(!touch.is_pointer());
        assert!(!touch.is_gesture());
        assert!(!touch.is_mouse());

        let pointer = Event::PointerDown {
            pointer_id: PointerId::new(1),
            pointer_type: PointerType::Touch,
            position: Point::ORIGIN,
            pressure: 0.5,
            is_primary: true,
            button: None,
        };
        assert!(!pointer.is_touch());
        assert!(pointer.is_pointer());
        assert!(!pointer.is_gesture());
        assert!(!pointer.is_mouse());

        let gesture = Event::GesturePinch {
            scale: 1.0,
            center: Point::ORIGIN,
            state: GestureState::Started,
        };
        assert!(!gesture.is_touch());
        assert!(!gesture.is_pointer());
        assert!(gesture.is_gesture());
        assert!(!gesture.is_mouse());
    }

    #[test]
    fn test_mouse_event_has_no_touch_or_pointer_id() {
        let mouse = Event::MouseDown {
            position: Point::ORIGIN,
            button: MouseButton::Left,
        };
        assert!(mouse.touch_id().is_none());
        assert!(mouse.pointer_id().is_none());
        assert!(mouse.pointer_type().is_none());
        assert!(mouse.pressure().is_none());
        assert!(mouse.gesture_state().is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let events = vec![
            Event::TouchStart {
                id: TouchId::new(1),
                position: Point::new(100.0, 200.0),
                pressure: 0.8,
            },
            Event::PointerDown {
                pointer_id: PointerId::new(2),
                pointer_type: PointerType::Pen,
                position: Point::new(50.0, 75.0),
                pressure: 0.6,
                is_primary: true,
                button: None,
            },
            Event::GesturePinch {
                scale: 2.0,
                center: Point::new(200.0, 200.0),
                state: GestureState::Changed,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let deserialized: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(event, deserialized);
        }
    }

    // ========== Additional edge case tests ==========

    #[test]
    fn test_mouse_button_all_variants() {
        let buttons = [
            MouseButton::Left,
            MouseButton::Right,
            MouseButton::Middle,
            MouseButton::Button4,
            MouseButton::Button5,
        ];
        for button in &buttons {
            let event = Event::MouseDown {
                position: Point::ORIGIN,
                button: *button,
            };
            assert!(event.is_mouse());
        }
    }

    #[test]
    fn test_mouse_button_debug() {
        assert_eq!(format!("{:?}", MouseButton::Left), "Left");
        assert_eq!(format!("{:?}", MouseButton::Middle), "Middle");
    }

    #[test]
    fn test_key_letters() {
        let letters = [
            Key::A,
            Key::B,
            Key::C,
            Key::D,
            Key::E,
            Key::F,
            Key::G,
            Key::H,
            Key::I,
            Key::J,
            Key::K,
            Key::L,
            Key::M,
            Key::N,
            Key::O,
            Key::P,
            Key::Q,
            Key::R,
            Key::S,
            Key::T,
            Key::U,
            Key::V,
            Key::W,
            Key::X,
            Key::Y,
            Key::Z,
        ];
        for key in &letters {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_numbers() {
        let numbers = [
            Key::Num0,
            Key::Num1,
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::Num5,
            Key::Num6,
            Key::Num7,
            Key::Num8,
            Key::Num9,
        ];
        for key in &numbers {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_function_keys() {
        let function_keys = [
            Key::F1,
            Key::F2,
            Key::F3,
            Key::F4,
            Key::F5,
            Key::F6,
            Key::F7,
            Key::F8,
            Key::F9,
            Key::F10,
            Key::F11,
            Key::F12,
        ];
        for key in &function_keys {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_control_keys() {
        let control_keys = [
            Key::Enter,
            Key::Escape,
            Key::Backspace,
            Key::Tab,
            Key::Space,
            Key::Delete,
            Key::Insert,
            Key::Home,
            Key::End,
            Key::PageUp,
            Key::PageDown,
        ];
        for key in &control_keys {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_arrow_keys() {
        let arrows = [Key::Up, Key::Down, Key::Left, Key::Right];
        for key in &arrows {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_modifiers() {
        let modifiers = [
            Key::ShiftLeft,
            Key::ShiftRight,
            Key::ControlLeft,
            Key::ControlRight,
            Key::AltLeft,
            Key::AltRight,
            Key::MetaLeft,
            Key::MetaRight,
        ];
        for key in &modifiers {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_punctuation() {
        let punctuation = [
            Key::Minus,
            Key::Equal,
            Key::BracketLeft,
            Key::BracketRight,
            Key::Backslash,
            Key::Semicolon,
            Key::Quote,
            Key::Grave,
            Key::Comma,
            Key::Period,
            Key::Slash,
        ];
        for key in &punctuation {
            let event = Event::key_down(*key);
            assert!(event.is_keyboard());
        }
    }

    #[test]
    fn test_key_debug() {
        assert_eq!(format!("{:?}", Key::Enter), "Enter");
        assert_eq!(format!("{:?}", Key::F1), "F1");
    }

    #[test]
    fn test_touch_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TouchId::new(1));
        set.insert(TouchId::new(2));
        set.insert(TouchId::new(1)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_pointer_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PointerId::new(1));
        set.insert(PointerId::new(2));
        set.insert(PointerId::new(1)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_pointer_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PointerType::Mouse);
        set.insert(PointerType::Touch);
        set.insert(PointerType::Pen);
        set.insert(PointerType::Mouse); // Duplicate
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_gesture_state_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GestureState::Started);
        set.insert(GestureState::Changed);
        set.insert(GestureState::Ended);
        set.insert(GestureState::Cancelled);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_event_debug() {
        let event = Event::FocusIn;
        let debug = format!("{event:?}");
        assert!(debug.contains("FocusIn"));
    }

    #[test]
    fn test_event_clone() {
        let event = Event::MouseMove {
            position: Point::new(100.0, 200.0),
        };
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    #[test]
    fn test_text_input_event() {
        let event = Event::TextInput {
            text: "Hello, 世界!".to_string(),
        };
        assert!(event.is_keyboard());
        assert!(!event.is_mouse());
        assert!(event.position().is_none());
    }

    #[test]
    fn test_scroll_event_deltas() {
        let event = Event::Scroll {
            delta_x: -10.5,
            delta_y: 20.3,
        };
        assert!(!event.is_mouse());
        assert!(!event.is_touch());
        assert!(!event.is_pointer());
    }

    #[test]
    fn test_resize_event() {
        let event = Event::Resize {
            width: 1920.0,
            height: 1080.0,
        };
        assert!(!event.is_mouse());
        assert!(event.position().is_none());
    }

    #[test]
    fn test_gesture_pan_no_position() {
        let event = Event::GesturePan {
            delta: Point::new(50.0, 30.0),
            velocity: Point::new(200.0, 150.0),
            state: GestureState::Changed,
        };
        assert!(event.is_gesture());
        // GesturePan has delta/velocity, not position
        assert!(event.position().is_none());
    }

    #[test]
    fn test_all_event_serialization() {
        let events = vec![
            Event::MouseMove {
                position: Point::new(1.0, 2.0),
            },
            Event::MouseDown {
                position: Point::new(1.0, 2.0),
                button: MouseButton::Left,
            },
            Event::MouseUp {
                position: Point::new(1.0, 2.0),
                button: MouseButton::Right,
            },
            Event::Scroll {
                delta_x: 1.0,
                delta_y: -1.0,
            },
            Event::key_down(Key::A),
            Event::key_up(Key::B),
            Event::TextInput {
                text: "test".to_string(),
            },
            Event::FocusIn,
            Event::FocusOut,
            Event::MouseEnter,
            Event::MouseLeave,
            Event::Resize {
                width: 800.0,
                height: 600.0,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let deserialized: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(event, deserialized);
        }
    }

    #[test]
    fn test_touch_events_serialization() {
        let events = vec![
            Event::TouchStart {
                id: TouchId(1),
                position: Point::new(10.0, 20.0),
                pressure: 0.5,
            },
            Event::TouchMove {
                id: TouchId(1),
                position: Point::new(15.0, 25.0),
                pressure: 0.6,
            },
            Event::TouchEnd {
                id: TouchId(1),
                position: Point::new(20.0, 30.0),
            },
            Event::TouchCancel { id: TouchId(1) },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let deserialized: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(event, deserialized);
        }
    }

    #[test]
    fn test_gesture_events_serialization() {
        let events = vec![
            Event::GesturePinch {
                scale: 1.5,
                center: Point::new(100.0, 100.0),
                state: GestureState::Started,
            },
            Event::GestureRotate {
                angle: 0.5,
                center: Point::new(100.0, 100.0),
                state: GestureState::Changed,
            },
            Event::GesturePan {
                delta: Point::new(10.0, 5.0),
                velocity: Point::new(50.0, 25.0),
                state: GestureState::Ended,
            },
            Event::GestureLongPress {
                position: Point::new(50.0, 50.0),
            },
            Event::GestureTap {
                position: Point::new(50.0, 50.0),
                count: 2,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let deserialized: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(event, deserialized);
        }
    }

    #[test]
    fn test_pointer_events_serialization() {
        let events = vec![
            Event::PointerDown {
                pointer_id: PointerId(1),
                pointer_type: PointerType::Mouse,
                position: Point::new(10.0, 20.0),
                pressure: 0.5,
                is_primary: true,
                button: Some(MouseButton::Left),
            },
            Event::PointerMove {
                pointer_id: PointerId(1),
                pointer_type: PointerType::Touch,
                position: Point::new(15.0, 25.0),
                pressure: 0.6,
                is_primary: true,
            },
            Event::PointerUp {
                pointer_id: PointerId(1),
                pointer_type: PointerType::Pen,
                position: Point::new(20.0, 30.0),
                is_primary: false,
                button: None,
            },
            Event::PointerCancel {
                pointer_id: PointerId(1),
            },
            Event::PointerEnter {
                pointer_id: PointerId(2),
                pointer_type: PointerType::Mouse,
            },
            Event::PointerLeave {
                pointer_id: PointerId(2),
                pointer_type: PointerType::Touch,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let deserialized: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(event, deserialized);
        }
    }

    #[test]
    fn test_key_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Key::A);
        set.insert(Key::B);
        set.insert(Key::A); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_mouse_button_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MouseButton::Left);
        set.insert(MouseButton::Right);
        set.insert(MouseButton::Left); // Duplicate
        assert_eq!(set.len(), 2);
    }
}
