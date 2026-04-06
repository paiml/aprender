//! Input handling for the TUI monitor.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent};

/// Input action resulting from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Quit the application.
    Quit,
    /// Navigate up.
    Up,
    /// Navigate down.
    Down,
    /// Navigate left.
    Left,
    /// Navigate right.
    Right,
    /// Select/confirm.
    Select,
    /// Toggle help.
    Help,
    /// Switch to preset N (0-9).
    Preset(u8),
    /// Filter processes.
    Filter,
    /// Toggle tree view.
    Tree,
    /// Kill selected process.
    Kill,
    /// Refresh immediately.
    Refresh,
    /// No action.
    None,
}

/// Input handler with configurable vim keys.
#[derive(Debug, Clone)]
pub struct InputHandler {
    /// Enable vim-style keys (hjkl).
    pub vim_keys: bool,
}

impl InputHandler {
    /// Creates a new input handler.
    #[must_use]
    pub fn new(vim_keys: bool) -> Self {
        Self { vim_keys }
    }

    /// Handles a key event and returns the corresponding action.
    #[must_use]
    pub fn handle_key(&self, event: KeyEvent) -> Action {
        // Check for Ctrl+C or Ctrl+Q
        if event.modifiers.contains(KeyModifiers::CONTROL) {
            if let KeyCode::Char('c' | 'q') = event.code {
                return Action::Quit;
            }
        }

        match event.code {
            // Quit
            KeyCode::Char('q') | KeyCode::Esc => Action::Quit,

            // Navigation
            KeyCode::Up => Action::Up,
            KeyCode::Down => Action::Down,
            KeyCode::Left => Action::Left,
            KeyCode::Right => Action::Right,

            // Vim keys
            KeyCode::Char('k') if self.vim_keys => Action::Up,
            KeyCode::Char('j') if self.vim_keys => Action::Down,
            KeyCode::Char('h') if self.vim_keys => Action::Left,
            KeyCode::Char('l') if self.vim_keys => Action::Right,

            // Selection
            KeyCode::Enter => Action::Select,

            // Help
            KeyCode::Char('?') | KeyCode::F(1) => Action::Help,

            // Presets
            KeyCode::Char(c @ '0'..='9') => Action::Preset(c.to_digit(10).unwrap_or(0) as u8),

            // Filter
            KeyCode::Char('/' | 'f') => Action::Filter,

            // Tree toggle
            KeyCode::Char('t') => Action::Tree,

            // Kill process
            KeyCode::Char('K') => Action::Kill,

            // Refresh
            KeyCode::Char('r') | KeyCode::F(5) => Action::Refresh,

            _ => Action::None,
        }
    }

    /// Handles a mouse event and returns the corresponding action.
    ///
    /// Note: Mouse handling is intentionally not implemented for this
    /// keyboard-first TUI. Returns `Action::None` for all mouse events.
    #[must_use]
    pub fn handle_mouse(&self, _event: MouseEvent) -> Action {
        // Keyboard-first TUI - mouse events are ignored
        Action::None
    }
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn key_event_ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    #[test]
    fn test_quit_actions() {
        let handler = InputHandler::new(true);

        assert_eq!(handler.handle_key(key_event(KeyCode::Char('q'))), Action::Quit);
        assert_eq!(handler.handle_key(key_event(KeyCode::Esc)), Action::Quit);
        assert_eq!(handler.handle_key(key_event_ctrl(KeyCode::Char('c'))), Action::Quit);
    }

    #[test]
    fn test_navigation() {
        let handler = InputHandler::new(true);

        assert_eq!(handler.handle_key(key_event(KeyCode::Up)), Action::Up);
        assert_eq!(handler.handle_key(key_event(KeyCode::Down)), Action::Down);
        assert_eq!(handler.handle_key(key_event(KeyCode::Left)), Action::Left);
        assert_eq!(handler.handle_key(key_event(KeyCode::Right)), Action::Right);
    }

    #[test]
    fn test_vim_keys_enabled() {
        let handler = InputHandler::new(true);

        assert_eq!(handler.handle_key(key_event(KeyCode::Char('k'))), Action::Up);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('j'))), Action::Down);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('h'))), Action::Left);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('l'))), Action::Right);
    }

    #[test]
    fn test_vim_keys_disabled() {
        let handler = InputHandler::new(false);

        assert_eq!(handler.handle_key(key_event(KeyCode::Char('k'))), Action::None);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('j'))), Action::None);
    }

    #[test]
    fn test_presets() {
        let handler = InputHandler::new(true);

        assert_eq!(handler.handle_key(key_event(KeyCode::Char('0'))), Action::Preset(0));
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('5'))), Action::Preset(5));
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('9'))), Action::Preset(9));
    }

    #[test]
    fn test_help() {
        let handler = InputHandler::new(true);

        assert_eq!(handler.handle_key(key_event(KeyCode::Char('?'))), Action::Help);
        assert_eq!(handler.handle_key(key_event(KeyCode::F(1))), Action::Help);
    }

    #[test]
    fn test_select_action() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event(KeyCode::Enter)), Action::Select);
    }

    #[test]
    fn test_filter_action() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('/'))), Action::Filter);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('f'))), Action::Filter);
    }

    #[test]
    fn test_tree_action() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('t'))), Action::Tree);
    }

    #[test]
    fn test_kill_action() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('K'))), Action::Kill);
    }

    #[test]
    fn test_refresh_action() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event(KeyCode::Char('r'))), Action::Refresh);
        assert_eq!(handler.handle_key(key_event(KeyCode::F(5))), Action::Refresh);
    }

    #[test]
    fn test_ctrl_q_quits() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event_ctrl(KeyCode::Char('q'))), Action::Quit);
    }

    #[test]
    fn test_ctrl_other_key_no_action() {
        let handler = InputHandler::new(true);
        // Ctrl+X should not quit, falls through to match
        assert_eq!(handler.handle_key(key_event_ctrl(KeyCode::Char('x'))), Action::None);
    }

    #[test]
    fn test_handle_mouse_returns_none() {
        use crossterm::event::{MouseButton, MouseEventKind};

        let handler = InputHandler::new(true);
        let mouse_event = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 0,
            row: 0,
            modifiers: KeyModifiers::empty(),
        };
        assert_eq!(handler.handle_mouse(mouse_event), Action::None);
    }

    #[test]
    fn test_default_handler() {
        let handler = InputHandler::default();
        assert!(handler.vim_keys); // Default has vim keys enabled
    }

    #[test]
    fn test_unknown_key_returns_none() {
        let handler = InputHandler::new(true);
        assert_eq!(handler.handle_key(key_event(KeyCode::Tab)), Action::None);
        assert_eq!(handler.handle_key(key_event(KeyCode::Insert)), Action::None);
    }

    #[test]
    fn test_action_clone_and_debug() {
        let action = Action::Quit;
        let cloned = action.clone();
        assert_eq!(action, cloned);

        let debug = format!("{:?}", Action::Preset(5));
        assert!(debug.contains("Preset"));
    }

    #[test]
    fn test_input_handler_debug() {
        let handler = InputHandler::new(false);
        let debug = format!("{handler:?}");
        assert!(debug.contains("InputHandler"));
    }

    #[test]
    fn test_input_handler_clone() {
        let handler = InputHandler::new(true);
        let cloned = handler.clone();
        assert_eq!(handler.vim_keys, cloned.vim_keys);
    }
}
