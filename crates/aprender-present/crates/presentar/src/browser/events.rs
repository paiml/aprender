//! Browser event handling - converts web events to presentar Events.
//!
//! Supports mouse, keyboard, touch, and pointer events.

use presentar_core::{Event, Key, MouseButton as CoreMouseButton, Point};
use wasm_bindgen::prelude::*;
use web_sys::{KeyboardEvent, MouseEvent, PointerEvent, TouchEvent, WheelEvent};

/// Convert a web_sys MouseEvent to a presentar Event.
pub fn mouse_event_to_presentar(event: &MouseEvent, event_type: &str) -> Event {
    let position = Point::new(event.offset_x() as f32, event.offset_y() as f32);
    let button = match event.button() {
        0 => CoreMouseButton::Left,
        1 => CoreMouseButton::Middle,
        2 => CoreMouseButton::Right,
        3 => CoreMouseButton::Button4,
        _ => CoreMouseButton::Button5,
    };

    match event_type {
        "mousedown" => Event::MouseDown { position, button },
        "mouseup" => Event::MouseUp { position, button },
        "mousemove" => Event::MouseMove { position },
        "mouseenter" => Event::MouseEnter,
        "mouseleave" => Event::MouseLeave,
        // Click is mousedown + mouseup, treat as mouseup
        "click" => Event::MouseUp { position, button },
        _ => Event::MouseMove { position },
    }
}

/// Convert a web_sys KeyboardEvent to a presentar Event.
pub fn keyboard_event_to_presentar(event: &KeyboardEvent, event_type: &str) -> Event {
    let key = code_to_key(&event.code());
    let modifiers = presentar_core::Modifiers {
        ctrl: event.ctrl_key(),
        alt: event.alt_key(),
        shift: event.shift_key(),
        meta: event.meta_key(),
    };

    match event_type {
        "keydown" => Event::KeyDown { key, modifiers },
        "keyup" => Event::KeyUp { key, modifiers },
        _ => Event::KeyDown { key, modifiers },
    }
}

/// Static lookup table for JS key code to presentar Key mapping.
static CODE_TO_KEY_TABLE: &[(&str, Key)] = &[
    ("ArrowDown", Key::Down),
    ("ArrowLeft", Key::Left),
    ("ArrowRight", Key::Right),
    ("ArrowUp", Key::Up),
    ("Backspace", Key::Backspace),
    ("ControlLeft", Key::ControlLeft),
    ("ControlRight", Key::ControlRight),
    ("Delete", Key::Delete),
    ("Digit0", Key::Num0),
    ("Digit1", Key::Num1),
    ("Digit2", Key::Num2),
    ("Digit3", Key::Num3),
    ("Digit4", Key::Num4),
    ("Digit5", Key::Num5),
    ("Digit6", Key::Num6),
    ("Digit7", Key::Num7),
    ("Digit8", Key::Num8),
    ("Digit9", Key::Num9),
    ("Enter", Key::Enter),
    ("Escape", Key::Escape),
    ("KeyA", Key::A),
    ("KeyB", Key::B),
    ("KeyC", Key::C),
    ("KeyD", Key::D),
    ("KeyE", Key::E),
    ("KeyF", Key::F),
    ("KeyG", Key::G),
    ("KeyH", Key::H),
    ("KeyI", Key::I),
    ("KeyJ", Key::J),
    ("KeyK", Key::K),
    ("KeyL", Key::L),
    ("KeyM", Key::M),
    ("KeyN", Key::N),
    ("KeyO", Key::O),
    ("KeyP", Key::P),
    ("KeyQ", Key::Q),
    ("KeyR", Key::R),
    ("KeyS", Key::S),
    ("KeyT", Key::T),
    ("KeyU", Key::U),
    ("KeyV", Key::V),
    ("KeyW", Key::W),
    ("KeyX", Key::X),
    ("KeyY", Key::Y),
    ("KeyZ", Key::Z),
    ("ShiftLeft", Key::ShiftLeft),
    ("ShiftRight", Key::ShiftRight),
    ("Space", Key::Space),
    ("Tab", Key::Tab),
];

/// Convert JS key code to presentar Key via sorted table + binary search.
fn code_to_key(code: &str) -> Key {
    CODE_TO_KEY_TABLE
        .binary_search_by_key(&code, |&(k, _)| k)
        .map(|i| CODE_TO_KEY_TABLE[i].1)
        .unwrap_or(Key::Space) // Default fallback
}

/// Create text input event from keyboard event.
pub fn text_input_event(event: &KeyboardEvent) -> Option<Event> {
    let key = event.key();
    // Only single printable characters
    if key.len() == 1 && !event.ctrl_key() && !event.alt_key() && !event.meta_key() {
        Some(Event::TextInput { text: key })
    } else {
        None
    }
}

// =============================================================================
// Touch Events
// =============================================================================

/// Touch point data.
#[derive(Debug, Clone, Copy)]
pub struct TouchPoint {
    /// Touch identifier.
    pub id: i32,
    /// Position of the touch.
    pub position: Point,
    /// Pressure (0.0 to 1.0).
    pub pressure: f32,
}

impl TouchPoint {
    /// Create a new touch point.
    #[must_use]
    pub const fn new(id: i32, position: Point, pressure: f32) -> Self {
        Self {
            id,
            position,
            pressure,
        }
    }
}

/// Convert a web_sys TouchEvent to presentar touch data.
pub fn touch_event_to_points(event: &TouchEvent, canvas_offset: Point) -> Vec<TouchPoint> {
    let touches = event.touches();
    let mut points = Vec::with_capacity(touches.length() as usize);

    for i in 0..touches.length() {
        if let Some(touch) = touches.get(i) {
            let x = touch.client_x() as f32 - canvas_offset.x;
            let y = touch.client_y() as f32 - canvas_offset.y;
            points.push(TouchPoint::new(
                touch.identifier(),
                Point::new(x, y),
                touch.force() as f32,
            ));
        }
    }

    points
}

/// Convert a touch event to a presentar Event (first touch only).
pub fn touch_event_to_presentar(
    event: &TouchEvent,
    event_type: &str,
    canvas_offset: Point,
) -> Event {
    let touches = event.touches();
    if touches.length() == 0 {
        return Event::MouseLeave;
    }

    let touch = touches.get(0).unwrap();
    let position = Point::new(
        touch.client_x() as f32 - canvas_offset.x,
        touch.client_y() as f32 - canvas_offset.y,
    );

    match event_type {
        "touchstart" => Event::MouseDown {
            position,
            button: CoreMouseButton::Left,
        },
        "touchend" | "touchcancel" => Event::MouseUp {
            position,
            button: CoreMouseButton::Left,
        },
        "touchmove" => Event::MouseMove { position },
        _ => Event::MouseMove { position },
    }
}

// =============================================================================
// Pointer Events (unified mouse/touch/pen)
// =============================================================================

/// Pointer type (mouse, touch, pen).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerType {
    /// Mouse pointer.
    Mouse,
    /// Touch pointer.
    Touch,
    /// Pen/stylus pointer.
    Pen,
    /// Unknown pointer type.
    Unknown,
}

impl From<&str> for PointerType {
    fn from(s: &str) -> Self {
        match s {
            "mouse" => Self::Mouse,
            "touch" => Self::Touch,
            "pen" => Self::Pen,
            _ => Self::Unknown,
        }
    }
}

/// Extended pointer data.
#[derive(Debug, Clone, Copy)]
pub struct PointerData {
    /// Pointer ID.
    pub id: i32,
    /// Position.
    pub position: Point,
    /// Pointer type.
    pub pointer_type: PointerType,
    /// Pressure (0.0 to 1.0).
    pub pressure: f32,
    /// Tilt X (-90 to 90 degrees).
    pub tilt_x: f32,
    /// Tilt Y (-90 to 90 degrees).
    pub tilt_y: f32,
    /// Width of contact area.
    pub width: f32,
    /// Height of contact area.
    pub height: f32,
    /// Is primary pointer.
    pub is_primary: bool,
}

impl PointerData {
    /// Create from a PointerEvent.
    pub fn from_event(event: &PointerEvent) -> Self {
        Self {
            id: event.pointer_id(),
            position: Point::new(event.offset_x() as f32, event.offset_y() as f32),
            pointer_type: PointerType::from(event.pointer_type().as_str()),
            pressure: event.pressure() as f32,
            tilt_x: event.tilt_x() as f32,
            tilt_y: event.tilt_y() as f32,
            width: event.width() as f32,
            height: event.height() as f32,
            is_primary: event.is_primary(),
        }
    }
}

/// Convert a web_sys PointerEvent to a presentar Event.
pub fn pointer_event_to_presentar(event: &PointerEvent, event_type: &str) -> Event {
    let position = Point::new(event.offset_x() as f32, event.offset_y() as f32);
    let button = match event.button() {
        0 => CoreMouseButton::Left,
        1 => CoreMouseButton::Middle,
        2 => CoreMouseButton::Right,
        3 => CoreMouseButton::Button4,
        _ => CoreMouseButton::Button5,
    };

    match event_type {
        "pointerdown" => Event::MouseDown { position, button },
        "pointerup" => Event::MouseUp { position, button },
        "pointermove" => Event::MouseMove { position },
        "pointerenter" => Event::MouseEnter,
        "pointerleave" => Event::MouseLeave,
        "pointercancel" => Event::MouseLeave,
        _ => Event::MouseMove { position },
    }
}

// =============================================================================
// Wheel Events
// =============================================================================

/// Scroll delta mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaMode {
    /// Pixels.
    Pixel,
    /// Lines.
    Line,
    /// Pages.
    Page,
}

impl From<u32> for DeltaMode {
    fn from(mode: u32) -> Self {
        match mode {
            0 => Self::Pixel,
            1 => Self::Line,
            2 => Self::Page,
            _ => Self::Pixel,
        }
    }
}

/// Scroll/wheel data.
#[derive(Debug, Clone, Copy)]
pub struct WheelData {
    /// Delta X.
    pub delta_x: f32,
    /// Delta Y.
    pub delta_y: f32,
    /// Delta mode.
    pub mode: DeltaMode,
    /// Cursor position.
    pub position: Point,
}

impl WheelData {
    /// Create from a WheelEvent.
    pub fn from_event(event: &WheelEvent) -> Self {
        Self {
            delta_x: event.delta_x() as f32,
            delta_y: event.delta_y() as f32,
            mode: DeltaMode::from(event.delta_mode()),
            position: Point::new(event.offset_x() as f32, event.offset_y() as f32),
        }
    }

    /// Get normalized delta (always in pixels).
    #[must_use]
    pub fn normalized_delta(&self) -> (f32, f32) {
        let multiplier = match self.mode {
            DeltaMode::Pixel => 1.0,
            DeltaMode::Line => 20.0,  // Approximate line height
            DeltaMode::Page => 400.0, // Approximate page height
        };
        (self.delta_x * multiplier, self.delta_y * multiplier)
    }
}

/// Convert a wheel event to scroll amounts.
pub fn wheel_event_to_scroll(event: &WheelEvent) -> (f32, f32) {
    let data = WheelData::from_event(event);
    data.normalized_delta()
}

/// Closure wrapper for event listeners.
pub struct JsonEventClosure {
    _closure: Closure<dyn FnMut(web_sys::Event)>,
}

impl JsonEventClosure {
    /// Create mouse event closure that calls back with JSON.
    pub fn mouse<F>(mut callback: F) -> Self
    where
        F: FnMut(Event) + 'static,
    {
        let closure = Closure::new(move |e: web_sys::Event| {
            if let Some(mouse_event) = e.dyn_ref::<MouseEvent>() {
                let event_type = e.type_();
                let presentar_event = mouse_event_to_presentar(mouse_event, &event_type);
                callback(presentar_event);
            }
        });
        Self { _closure: closure }
    }

    /// Create keyboard event closure.
    pub fn keyboard<F>(mut callback: F) -> Self
    where
        F: FnMut(Event) + 'static,
    {
        let closure = Closure::new(move |e: web_sys::Event| {
            if let Some(kb_event) = e.dyn_ref::<KeyboardEvent>() {
                let event_type = e.type_();
                let presentar_event = keyboard_event_to_presentar(kb_event, &event_type);
                callback(presentar_event);
            }
        });
        Self { _closure: closure }
    }

    /// Get as JS function.
    pub fn as_function(&self) -> &js_sys::Function {
        self._closure.as_ref().unchecked_ref()
    }

    /// Create touch event closure.
    pub fn touch<F>(mut callback: F, canvas_offset: Point) -> Self
    where
        F: FnMut(Event, Vec<TouchPoint>) + 'static,
    {
        let closure = Closure::new(move |e: web_sys::Event| {
            if let Some(touch_event) = e.dyn_ref::<TouchEvent>() {
                let event_type = e.type_();
                let presentar_event =
                    touch_event_to_presentar(touch_event, &event_type, canvas_offset);
                let points = touch_event_to_points(touch_event, canvas_offset);
                callback(presentar_event, points);
            }
        });
        Self { _closure: closure }
    }

    /// Create pointer event closure.
    pub fn pointer<F>(mut callback: F) -> Self
    where
        F: FnMut(Event, PointerData) + 'static,
    {
        let closure = Closure::new(move |e: web_sys::Event| {
            if let Some(pointer_event) = e.dyn_ref::<PointerEvent>() {
                let event_type = e.type_();
                let presentar_event = pointer_event_to_presentar(pointer_event, &event_type);
                let data = PointerData::from_event(pointer_event);
                callback(presentar_event, data);
            }
        });
        Self { _closure: closure }
    }

    /// Create wheel event closure.
    pub fn wheel<F>(mut callback: F) -> Self
    where
        F: FnMut(f32, f32, Point) + 'static,
    {
        let closure = Closure::new(move |e: web_sys::Event| {
            if let Some(wheel_event) = e.dyn_ref::<WheelEvent>() {
                let data = WheelData::from_event(wheel_event);
                let (dx, dy) = data.normalized_delta();
                callback(dx, dy, data.position);
            }
        });
        Self { _closure: closure }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_type_from_str() {
        assert_eq!(PointerType::from("mouse"), PointerType::Mouse);
        assert_eq!(PointerType::from("touch"), PointerType::Touch);
        assert_eq!(PointerType::from("pen"), PointerType::Pen);
        assert_eq!(PointerType::from("unknown"), PointerType::Unknown);
    }

    #[test]
    fn test_delta_mode_from_u32() {
        assert_eq!(DeltaMode::from(0), DeltaMode::Pixel);
        assert_eq!(DeltaMode::from(1), DeltaMode::Line);
        assert_eq!(DeltaMode::from(2), DeltaMode::Page);
        assert_eq!(DeltaMode::from(99), DeltaMode::Pixel);
    }

    #[test]
    fn test_touch_point_new() {
        let point = TouchPoint::new(1, Point::new(100.0, 200.0), 0.5);
        assert_eq!(point.id, 1);
        assert_eq!(point.position.x, 100.0);
        assert_eq!(point.position.y, 200.0);
        assert_eq!(point.pressure, 0.5);
    }

    #[test]
    fn test_code_to_key_letters() {
        assert!(matches!(code_to_key("KeyA"), Key::A));
        assert!(matches!(code_to_key("KeyZ"), Key::Z));
    }

    #[test]
    fn test_code_to_key_digits() {
        assert!(matches!(code_to_key("Digit0"), Key::Num0));
        assert!(matches!(code_to_key("Digit9"), Key::Num9));
    }

    #[test]
    fn test_code_to_key_special() {
        assert!(matches!(code_to_key("Enter"), Key::Enter));
        assert!(matches!(code_to_key("Escape"), Key::Escape));
        assert!(matches!(code_to_key("Tab"), Key::Tab));
        assert!(matches!(code_to_key("Space"), Key::Space));
        assert!(matches!(code_to_key("ArrowUp"), Key::Up));
        assert!(matches!(code_to_key("ArrowDown"), Key::Down));
    }
}
