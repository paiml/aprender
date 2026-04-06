//! Input handling for terminal applications.

use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers};
use presentar_core::{Event, Key, MouseButton, Point};

/// Key binding configuration.
#[derive(Debug, Clone)]
pub struct KeyBinding {
    /// Key code.
    pub code: KeyCode,
    /// Required modifiers.
    pub modifiers: KeyModifiers,
    /// Action name.
    pub action: String,
}

impl KeyBinding {
    /// Create a new key binding.
    #[must_use]
    pub fn new(code: KeyCode, modifiers: KeyModifiers, action: impl Into<String>) -> Self {
        Self {
            code,
            modifiers,
            action: action.into(),
        }
    }

    /// Create a simple key binding without modifiers.
    #[must_use]
    pub fn simple(code: KeyCode, action: impl Into<String>) -> Self {
        Self::new(code, KeyModifiers::NONE, action)
    }

    /// Check if this binding matches a key event.
    #[must_use]
    pub fn matches(&self, event: &KeyEvent) -> bool {
        event.code == self.code && event.modifiers.contains(self.modifiers)
    }
}

/// Input handler for converting crossterm events to presentar events.
#[derive(Debug, Default)]
pub struct InputHandler {
    bindings: Vec<KeyBinding>,
}

impl InputHandler {
    /// Create a new input handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a key binding.
    pub fn add_binding(&mut self, binding: KeyBinding) {
        self.bindings.push(binding);
    }

    /// Convert a crossterm event to a presentar event.
    #[must_use]
    pub fn convert(&self, event: CrosstermEvent) -> Option<Event> {
        match event {
            CrosstermEvent::Key(key) => self.convert_key(key),
            CrosstermEvent::Mouse(mouse) => Some(self.convert_mouse(mouse)),
            CrosstermEvent::Resize(width, height) => Some(Event::Resize {
                width: f32::from(width),
                height: f32::from(height),
            }),
            CrosstermEvent::FocusGained => Some(Event::FocusIn),
            CrosstermEvent::FocusLost => Some(Event::FocusOut),
            CrosstermEvent::Paste(text) => Some(Event::TextInput { text }),
        }
    }

    fn convert_key(&self, key: KeyEvent) -> Option<Event> {
        let presentar_key = match key.code {
            KeyCode::Char(ch) => char_to_key(ch)?,
            KeyCode::F(n) => fn_key(n)?,
            _ => non_char_key(key.code)?,
        };

        let modifiers = presentar_core::Modifiers {
            ctrl: key.modifiers.contains(KeyModifiers::CONTROL),
            alt: key.modifiers.contains(KeyModifiers::ALT),
            shift: key.modifiers.contains(KeyModifiers::SHIFT),
            meta: key.modifiers.contains(KeyModifiers::SUPER),
        };
        Some(Event::KeyDown {
            key: presentar_key,
            modifiers,
        })
    }

    fn convert_mouse(&self, mouse: crossterm::event::MouseEvent) -> Event {
        use crossterm::event::{MouseButton as CtMouseButton, MouseEventKind};

        let position = Point::new(f32::from(mouse.column), f32::from(mouse.row));

        match mouse.kind {
            MouseEventKind::Down(button) => Event::MouseDown {
                position,
                button: match button {
                    CtMouseButton::Left => MouseButton::Left,
                    CtMouseButton::Right => MouseButton::Right,
                    CtMouseButton::Middle => MouseButton::Middle,
                },
            },
            MouseEventKind::Up(button) => Event::MouseUp {
                position,
                button: match button {
                    CtMouseButton::Left => MouseButton::Left,
                    CtMouseButton::Right => MouseButton::Right,
                    CtMouseButton::Middle => MouseButton::Middle,
                },
            },
            MouseEventKind::Moved | MouseEventKind::Drag(_) => Event::MouseMove { position },
            MouseEventKind::ScrollUp => Event::Scroll {
                delta_x: 0.0,
                delta_y: -1.0,
            },
            MouseEventKind::ScrollDown => Event::Scroll {
                delta_x: 0.0,
                delta_y: 1.0,
            },
            MouseEventKind::ScrollLeft => Event::Scroll {
                delta_x: -1.0,
                delta_y: 0.0,
            },
            MouseEventKind::ScrollRight => Event::Scroll {
                delta_x: 1.0,
                delta_y: 0.0,
            },
        }
    }

    /// Find a matching binding for a key event.
    #[must_use]
    pub fn find_binding(&self, event: &KeyEvent) -> Option<&KeyBinding> {
        self.bindings.iter().find(|b| b.matches(event))
    }
}

// =============================================================================
// Key conversion helpers (extracted from convert_key for CB-200 compliance)
// =============================================================================

/// Sorted table for punctuation/symbol char-to-Key lookups.
static CHAR_PUNCT_TABLE: &[(char, Key)] = &[
    (' ', Key::Space),
    ('\'', Key::Quote),
    (',', Key::Comma),
    ('-', Key::Minus),
    ('.', Key::Period),
    ('/', Key::Slash),
    (';', Key::Semicolon),
    ('=', Key::Equal),
    ('[', Key::BracketLeft),
    ('\\', Key::Backslash),
    (']', Key::BracketRight),
    ('`', Key::Grave),
];

/// Convert a character to a presentar Key.
fn char_to_key(ch: char) -> Option<Key> {
    match ch {
        'a' | 'A' => Some(Key::A),
        'b' | 'B' => Some(Key::B),
        'c' | 'C' => Some(Key::C),
        'd' | 'D' => Some(Key::D),
        'e' | 'E' => Some(Key::E),
        'f' | 'F' => Some(Key::F),
        'g' | 'G' => Some(Key::G),
        'h' | 'H' => Some(Key::H),
        'i' | 'I' => Some(Key::I),
        'j' | 'J' => Some(Key::J),
        'k' | 'K' => Some(Key::K),
        'l' | 'L' => Some(Key::L),
        'm' | 'M' => Some(Key::M),
        'n' | 'N' => Some(Key::N),
        'o' | 'O' => Some(Key::O),
        'p' | 'P' => Some(Key::P),
        'q' | 'Q' => Some(Key::Q),
        'r' | 'R' => Some(Key::R),
        's' | 'S' => Some(Key::S),
        't' | 'T' => Some(Key::T),
        'u' | 'U' => Some(Key::U),
        'v' | 'V' => Some(Key::V),
        'w' | 'W' => Some(Key::W),
        'x' | 'X' => Some(Key::X),
        'y' | 'Y' => Some(Key::Y),
        'z' | 'Z' => Some(Key::Z),
        '0' => Some(Key::Num0),
        '1' => Some(Key::Num1),
        '2' => Some(Key::Num2),
        '3' => Some(Key::Num3),
        '4' => Some(Key::Num4),
        '5' => Some(Key::Num5),
        '6' => Some(Key::Num6),
        '7' => Some(Key::Num7),
        '8' => Some(Key::Num8),
        '9' => Some(Key::Num9),
        _ => CHAR_PUNCT_TABLE
            .binary_search_by_key(&ch, |&(c, _)| c)
            .map(|i| CHAR_PUNCT_TABLE[i].1)
            .ok(),
    }
}

/// Convert function key number to presentar Key.
fn fn_key(n: u8) -> Option<Key> {
    static FN_KEYS: [Key; 12] = [
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
    FN_KEYS.get(n.wrapping_sub(1) as usize).copied()
}

/// Convert non-char KeyCode to presentar Key.
fn non_char_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::Enter => Key::Enter,
        KeyCode::Esc => Key::Escape,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        KeyCode::Delete => Key::Delete,
        KeyCode::Insert => Key::Insert,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        _ => return None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crossterm::event::{MouseButton as CtMouseButton, MouseEvent, MouseEventKind};

    #[test]
    fn test_key_binding_simple() {
        let binding = KeyBinding::simple(KeyCode::Char('q'), "quit");
        assert_eq!(binding.action, "quit");
        assert_eq!(binding.modifiers, KeyModifiers::NONE);
    }

    #[test]
    fn test_key_binding_with_modifiers() {
        let binding = KeyBinding::new(KeyCode::Char('c'), KeyModifiers::CONTROL, "copy");
        let event = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(binding.matches(&event));
    }

    #[test]
    fn test_key_binding_no_match() {
        let binding = KeyBinding::new(KeyCode::Char('c'), KeyModifiers::CONTROL, "copy");
        let event = KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL);
        assert!(!binding.matches(&event));
    }

    #[test]
    fn test_input_handler_add_binding() {
        let mut handler = InputHandler::new();
        handler.add_binding(KeyBinding::simple(KeyCode::Char('q'), "quit"));
        assert_eq!(handler.bindings.len(), 1);
    }

    #[test]
    fn test_input_handler_find_binding() {
        let mut handler = InputHandler::new();
        handler.add_binding(KeyBinding::simple(KeyCode::Char('q'), "quit"));
        handler.add_binding(KeyBinding::new(
            KeyCode::Char('s'),
            KeyModifiers::CONTROL,
            "save",
        ));

        let event = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        let binding = handler.find_binding(&event);
        assert!(binding.is_some());
        assert_eq!(binding.unwrap().action, "quit");

        let event2 = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        assert!(handler.find_binding(&event2).is_none());
    }

    #[test]
    fn test_convert_letter_keys() {
        let handler = InputHandler::new();
        for ch in 'a'..='z' {
            let event = CrosstermEvent::Key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
            let result = handler.convert(event);
            assert!(result.is_some());
        }
        for ch in 'A'..='Z' {
            let event = CrosstermEvent::Key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
            let result = handler.convert(event);
            assert!(result.is_some());
        }
    }

    #[test]
    fn test_convert_number_keys() {
        let handler = InputHandler::new();
        for ch in '0'..='9' {
            let event = CrosstermEvent::Key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
            let result = handler.convert(event);
            assert!(result.is_some());
        }
    }

    #[test]
    fn test_convert_special_keys() {
        let handler = InputHandler::new();
        let special_keys = [
            KeyCode::Enter,
            KeyCode::Esc,
            KeyCode::Backspace,
            KeyCode::Tab,
            KeyCode::Delete,
            KeyCode::Insert,
            KeyCode::Up,
            KeyCode::Down,
            KeyCode::Left,
            KeyCode::Right,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::PageUp,
            KeyCode::PageDown,
        ];
        for key in special_keys {
            let event = CrosstermEvent::Key(KeyEvent::new(key, KeyModifiers::NONE));
            let result = handler.convert(event);
            assert!(result.is_some(), "Failed for {:?}", key);
        }
    }

    #[test]
    fn test_convert_function_keys() {
        let handler = InputHandler::new();
        for n in 1..=12 {
            let event = CrosstermEvent::Key(KeyEvent::new(KeyCode::F(n), KeyModifiers::NONE));
            let result = handler.convert(event);
            assert!(result.is_some(), "Failed for F{}", n);
        }
    }

    #[test]
    fn test_convert_punctuation_keys() {
        let handler = InputHandler::new();
        let punct = [' ', '-', '=', '[', ']', '\\', ';', '\'', '`', ',', '.', '/'];
        for ch in punct {
            let event = CrosstermEvent::Key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
            let result = handler.convert(event);
            assert!(result.is_some(), "Failed for {:?}", ch);
        }
    }

    #[test]
    fn test_convert_unknown_key() {
        let handler = InputHandler::new();
        let event = CrosstermEvent::Key(KeyEvent::new(KeyCode::Char('£'), KeyModifiers::NONE));
        let result = handler.convert(event);
        assert!(result.is_none());
    }

    #[test]
    fn test_convert_mouse_down() {
        let handler = InputHandler::new();
        let mouse = MouseEvent {
            kind: MouseEventKind::Down(CtMouseButton::Left),
            column: 10,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        let event = CrosstermEvent::Mouse(mouse);
        let result = handler.convert(event).unwrap();
        assert!(
            matches!(result, Event::MouseDown { position, button: MouseButton::Left } if position.x == 10.0 && position.y == 5.0)
        );
    }

    #[test]
    fn test_convert_mouse_up() {
        let handler = InputHandler::new();
        let mouse = MouseEvent {
            kind: MouseEventKind::Up(CtMouseButton::Right),
            column: 15,
            row: 8,
            modifiers: KeyModifiers::NONE,
        };
        let event = CrosstermEvent::Mouse(mouse);
        let result = handler.convert(event).unwrap();
        assert!(matches!(
            result,
            Event::MouseUp {
                button: MouseButton::Right,
                ..
            }
        ));
    }

    #[test]
    fn test_convert_mouse_middle() {
        let handler = InputHandler::new();
        let mouse = MouseEvent {
            kind: MouseEventKind::Down(CtMouseButton::Middle),
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let event = CrosstermEvent::Mouse(mouse);
        let result = handler.convert(event).unwrap();
        assert!(matches!(
            result,
            Event::MouseDown {
                button: MouseButton::Middle,
                ..
            }
        ));
    }

    #[test]
    fn test_convert_mouse_move() {
        let handler = InputHandler::new();
        let mouse = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 20,
            row: 10,
            modifiers: KeyModifiers::NONE,
        };
        let event = CrosstermEvent::Mouse(mouse);
        let result = handler.convert(event).unwrap();
        assert!(matches!(result, Event::MouseMove { position } if position.x == 20.0));
    }

    #[test]
    fn test_convert_mouse_drag() {
        let handler = InputHandler::new();
        let mouse = MouseEvent {
            kind: MouseEventKind::Drag(CtMouseButton::Left),
            column: 25,
            row: 12,
            modifiers: KeyModifiers::NONE,
        };
        let event = CrosstermEvent::Mouse(mouse);
        let result = handler.convert(event).unwrap();
        assert!(matches!(result, Event::MouseMove { .. }));
    }

    #[test]
    fn test_convert_scroll_events() {
        let handler = InputHandler::new();

        let scroll_up = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let result = handler.convert(CrosstermEvent::Mouse(scroll_up)).unwrap();
        assert!(matches!(result, Event::Scroll { delta_y, .. } if delta_y < 0.0));

        let scroll_down = MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let result = handler.convert(CrosstermEvent::Mouse(scroll_down)).unwrap();
        assert!(matches!(result, Event::Scroll { delta_y, .. } if delta_y > 0.0));

        let scroll_left = MouseEvent {
            kind: MouseEventKind::ScrollLeft,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let result = handler.convert(CrosstermEvent::Mouse(scroll_left)).unwrap();
        assert!(matches!(result, Event::Scroll { delta_x, .. } if delta_x < 0.0));

        let scroll_right = MouseEvent {
            kind: MouseEventKind::ScrollRight,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let result = handler
            .convert(CrosstermEvent::Mouse(scroll_right))
            .unwrap();
        assert!(matches!(result, Event::Scroll { delta_x, .. } if delta_x > 0.0));
    }

    #[test]
    fn test_convert_resize() {
        let handler = InputHandler::new();
        let event = CrosstermEvent::Resize(120, 40);
        let result = handler.convert(event).unwrap();
        assert!(
            matches!(result, Event::Resize { width, height } if width == 120.0 && height == 40.0)
        );
    }

    #[test]
    fn test_convert_focus_events() {
        let handler = InputHandler::new();

        let result = handler.convert(CrosstermEvent::FocusGained).unwrap();
        assert!(matches!(result, Event::FocusIn));

        let result = handler.convert(CrosstermEvent::FocusLost).unwrap();
        assert!(matches!(result, Event::FocusOut));
    }

    #[test]
    fn test_convert_paste() {
        let handler = InputHandler::new();
        let event = CrosstermEvent::Paste("hello world".to_string());
        let result = handler.convert(event).unwrap();
        assert!(matches!(result, Event::TextInput { text } if text == "hello world"));
    }
}
