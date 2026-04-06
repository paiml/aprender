//! Keyboard input handling with 100% test coverage
//!
//! Probar: Error prevention - Type-safe key actions prevent invalid input

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Actions that can be triggered by keyboard input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Insert a character
    InsertChar(char),
    /// Delete character before cursor (backspace)
    Backspace,
    /// Delete character at cursor
    Delete,
    /// Move cursor left
    CursorLeft,
    /// Move cursor right
    CursorRight,
    /// Move cursor to start
    CursorHome,
    /// Move cursor to end
    CursorEnd,
    /// Evaluate the expression
    Evaluate,
    /// Clear the input
    Clear,
    /// Clear everything including history
    ClearAll,
    /// Recall last expression from history
    RecallLast,
    /// Quit the application
    Quit,
    /// No action (ignored input)
    None,
}

/// Input handler that maps key events to actions
#[derive(Debug, Default)]
pub struct InputHandler;

impl InputHandler {
    /// Creates a new input handler
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Maps a key event to an action
    #[must_use]
    pub fn handle_key(&self, event: KeyEvent) -> KeyAction {
        let KeyEvent {
            code, modifiers, ..
        } = event;

        // Handle Ctrl+key combinations
        if modifiers.contains(KeyModifiers::CONTROL) {
            return match code {
                KeyCode::Char('c' | 'q') => KeyAction::Quit,
                KeyCode::Char('l') => KeyAction::ClearAll,
                KeyCode::Char('a') => KeyAction::CursorHome,
                KeyCode::Char('e') => KeyAction::CursorEnd,
                KeyCode::Char('u') => KeyAction::Clear,
                _ => KeyAction::None,
            };
        }

        // Handle regular keys
        match code {
            KeyCode::Char(c) => KeyAction::InsertChar(c),
            KeyCode::Backspace => KeyAction::Backspace,
            KeyCode::Delete => KeyAction::Delete,
            KeyCode::Left => KeyAction::CursorLeft,
            KeyCode::Right => KeyAction::CursorRight,
            KeyCode::Home => KeyAction::CursorHome,
            KeyCode::End => KeyAction::CursorEnd,
            KeyCode::Enter => KeyAction::Evaluate,
            KeyCode::Esc => KeyAction::Clear,
            KeyCode::Up => KeyAction::RecallLast,
            _ => KeyAction::None,
        }
    }

    /// Returns true if the character is valid for calculator input
    #[must_use]
    pub fn is_valid_char(c: char) -> bool {
        c.is_ascii_digit()
            || c == '.'
            || c == '+'
            || c == '-'
            || c == '*'
            || c == '/'
            || c == '%'
            || c == '^'
            || c == '('
            || c == ')'
            || c == ' '
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn key_event_ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    // ===== Constructor tests =====

    #[test]
    fn test_input_handler_new() {
        let handler = InputHandler::new();
        // Just verify it creates without panic
        let _ = format!("{:?}", handler);
    }

    #[test]
    fn test_input_handler_default() {
        let handler = InputHandler;
        let _ = format!("{:?}", handler);
    }

    // ===== Character input tests =====

    #[test]
    fn test_handle_digit_keys() {
        let handler = InputHandler::new();
        for c in '0'..='9' {
            let event = key_event(KeyCode::Char(c));
            assert_eq!(handler.handle_key(event), KeyAction::InsertChar(c));
        }
    }

    #[test]
    fn test_handle_operator_keys() {
        let handler = InputHandler::new();
        let operators = ['+', '-', '*', '/', '%', '^'];
        for c in operators {
            let event = key_event(KeyCode::Char(c));
            assert_eq!(handler.handle_key(event), KeyAction::InsertChar(c));
        }
    }

    #[test]
    fn test_handle_parentheses() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Char('('))),
            KeyAction::InsertChar('(')
        );
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Char(')'))),
            KeyAction::InsertChar(')')
        );
    }

    #[test]
    fn test_handle_decimal_point() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Char('.'))),
            KeyAction::InsertChar('.')
        );
    }

    #[test]
    fn test_handle_space() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Char(' '))),
            KeyAction::InsertChar(' ')
        );
    }

    // ===== Edit key tests =====

    #[test]
    fn test_handle_backspace() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Backspace)),
            KeyAction::Backspace
        );
    }

    #[test]
    fn test_handle_delete() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Delete)),
            KeyAction::Delete
        );
    }

    // ===== Navigation key tests =====

    #[test]
    fn test_handle_left() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Left)),
            KeyAction::CursorLeft
        );
    }

    #[test]
    fn test_handle_right() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Right)),
            KeyAction::CursorRight
        );
    }

    #[test]
    fn test_handle_home() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Home)),
            KeyAction::CursorHome
        );
    }

    #[test]
    fn test_handle_end() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::End)),
            KeyAction::CursorEnd
        );
    }

    // ===== Action key tests =====

    #[test]
    fn test_handle_enter() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Enter)),
            KeyAction::Evaluate
        );
    }

    #[test]
    fn test_handle_escape() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Esc)),
            KeyAction::Clear
        );
    }

    #[test]
    fn test_handle_up() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::Up)),
            KeyAction::RecallLast
        );
    }

    // ===== Ctrl key tests =====

    #[test]
    fn test_handle_ctrl_c() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('c'))),
            KeyAction::Quit
        );
    }

    #[test]
    fn test_handle_ctrl_q() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('q'))),
            KeyAction::Quit
        );
    }

    #[test]
    fn test_handle_ctrl_l() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('l'))),
            KeyAction::ClearAll
        );
    }

    #[test]
    fn test_handle_ctrl_a() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('a'))),
            KeyAction::CursorHome
        );
    }

    #[test]
    fn test_handle_ctrl_e() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('e'))),
            KeyAction::CursorEnd
        );
    }

    #[test]
    fn test_handle_ctrl_u() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('u'))),
            KeyAction::Clear
        );
    }

    #[test]
    fn test_handle_ctrl_unknown() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event_ctrl(KeyCode::Char('x'))),
            KeyAction::None
        );
    }

    // ===== Unknown key tests =====

    #[test]
    fn test_handle_unknown_key() {
        let handler = InputHandler::new();
        assert_eq!(
            handler.handle_key(key_event(KeyCode::F(1))),
            KeyAction::None
        );
    }

    #[test]
    fn test_handle_tab() {
        let handler = InputHandler::new();
        assert_eq!(handler.handle_key(key_event(KeyCode::Tab)), KeyAction::None);
    }

    // ===== Valid char tests =====

    #[test]
    fn test_is_valid_char_digits() {
        for c in '0'..='9' {
            assert!(InputHandler::is_valid_char(c), "Digit {c} should be valid");
        }
    }

    #[test]
    fn test_is_valid_char_operators() {
        let valid = ['+', '-', '*', '/', '%', '^', '(', ')', '.', ' '];
        for c in valid {
            assert!(InputHandler::is_valid_char(c), "Char '{c}' should be valid");
        }
    }

    #[test]
    fn test_is_valid_char_invalid() {
        let invalid = ['a', 'z', 'A', 'Z', '@', '#', '$', '!', '&', '|'];
        for c in invalid {
            assert!(
                !InputHandler::is_valid_char(c),
                "Char '{c}' should be invalid"
            );
        }
    }

    // ===== KeyAction tests =====

    #[test]
    fn test_key_action_debug() {
        let action = KeyAction::Evaluate;
        assert!(format!("{:?}", action).contains("Evaluate"));
    }

    #[test]
    fn test_key_action_clone() {
        let action = KeyAction::InsertChar('x');
        let cloned = action;
        assert_eq!(action, cloned);
    }

    #[test]
    fn test_key_action_copy() {
        let action = KeyAction::Quit;
        let copied: KeyAction = action;
        assert_eq!(action, copied);
    }
}
