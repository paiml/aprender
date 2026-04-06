//! Job Flow TUI - Real-time distributed task visualization.
//!
//! This module provides a terminal-based dashboard for monitoring
//! distributed task execution across heterogeneous compute resources.
//!
//! Built with ratatui and tested with 100% probador coverage.

pub mod model;
pub mod render;
pub mod widgets;

pub use model::{
    Alert, App, BackendStatus, BackendType, CompletionRecord, MetricsHistory, NodeState,
    NodeStatus, Selection, TaskQueue,
};
pub use render::draw_ui;

use crate::error::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::time::{Duration, Instant};

/// Action returned from event handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Continue running.
    Continue,
    /// Force refresh.
    Refresh,
    /// Quit the application.
    Quit,
}

/// Handles a key event and returns the action to take.
///
/// This function is separated from I/O for testability.
#[must_use]
pub fn handle_key_event(app: &mut App, key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Char('r') => {
            app.force_refresh = true;
            Action::Refresh
        }
        KeyCode::Up => {
            app.select_prev();
            Action::Continue
        }
        KeyCode::Down => {
            app.select_next();
            Action::Continue
        }
        KeyCode::Tab => {
            app.cycle_focus();
            Action::Continue
        }
        KeyCode::Char('?') => {
            app.toggle_help();
            Action::Continue
        }
        KeyCode::Char('a') => {
            app.toggle_alerts();
            Action::Continue
        }
        _ => Action::Continue,
    }
}

/// Processes a tick and updates app state.
///
/// Returns true if tick was processed, false if not enough time elapsed.
#[must_use]
pub fn process_tick(app: &mut App, last_tick: Instant, tick_rate: Duration) -> bool {
    if last_tick.elapsed() >= tick_rate || app.force_refresh {
        app.tick();
        app.force_refresh = false;
        true
    } else {
        false
    }
}

/// Run the TUI application.
///
/// # Errors
///
/// Returns an error if terminal setup fails or rendering fails.
pub fn run(app: App) -> Result<()> {
    // Setup terminal
    enable_raw_mode().map_err(crate::error::RepartirError::Io)?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .map_err(crate::error::RepartirError::Io)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(crate::error::RepartirError::Io)?;

    // Run event loop
    let res = run_event_loop(&mut terminal, app);

    // Restore terminal
    disable_raw_mode().map_err(crate::error::RepartirError::Io)?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)
        .map_err(crate::error::RepartirError::Io)?;
    terminal.show_cursor().map_err(crate::error::RepartirError::Io)?;

    res
}

/// Event loop for the TUI.
fn run_event_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
) -> Result<()> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw_ui(f, &app)).map_err(crate::error::RepartirError::Io)?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout).map_err(crate::error::RepartirError::Io)? {
            if let Event::Key(key) = event::read().map_err(crate::error::RepartirError::Io)? {
                if handle_key_event(&mut app, key) == Action::Quit {
                    return Ok(());
                }
            }
        }

        if process_tick(&mut app, last_tick, tick_rate) {
            last_tick = Instant::now();
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    unused_must_use,
    clippy::panic
)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    /// Creates a key event for testing.
    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }
    }

    // =========================================================================
    // Action Tests
    // =========================================================================

    #[test]
    fn test_action_equality() {
        assert_eq!(Action::Continue, Action::Continue);
        assert_eq!(Action::Quit, Action::Quit);
        assert_eq!(Action::Refresh, Action::Refresh);
        assert_ne!(Action::Continue, Action::Quit);
    }

    #[test]
    fn test_action_debug() {
        let debug_str = format!("{:?}", Action::Continue);
        assert!(debug_str.contains("Continue"));
    }

    #[test]
    fn test_action_clone() {
        let action = Action::Quit;
        let cloned = action;
        assert_eq!(action, cloned);
    }

    // =========================================================================
    // Key Event Handler Tests
    // =========================================================================

    #[test]
    fn test_handle_key_quit() {
        let mut app = App::new();
        let action = handle_key_event(&mut app, key_event(KeyCode::Char('q')));
        assert_eq!(action, Action::Quit);
    }

    #[test]
    fn test_handle_key_refresh() {
        let mut app = App::new();
        let action = handle_key_event(&mut app, key_event(KeyCode::Char('r')));
        assert_eq!(action, Action::Refresh);
        assert!(app.force_refresh);
    }

    #[test]
    fn test_handle_key_up() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));
        app.add_node(NodeStatus::new("node-2", "127.0.0.1:9001".parse().unwrap()));
        app.selected = Selection::Node(1);

        let action = handle_key_event(&mut app, key_event(KeyCode::Up));
        assert_eq!(action, Action::Continue);
        assert!(matches!(app.selected, Selection::Node(0)));
    }

    #[test]
    fn test_handle_key_down() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));
        app.add_node(NodeStatus::new("node-2", "127.0.0.1:9001".parse().unwrap()));

        let action = handle_key_event(&mut app, key_event(KeyCode::Down));
        assert_eq!(action, Action::Continue);
        assert!(matches!(app.selected, Selection::Node(0)));
    }

    #[test]
    fn test_handle_key_tab() {
        let mut app = App::new();
        assert_eq!(app.focus, model::Focus::Cluster);

        let action = handle_key_event(&mut app, key_event(KeyCode::Tab));
        assert_eq!(action, Action::Continue);
        assert_eq!(app.focus, model::Focus::Queue);
    }

    #[test]
    fn test_handle_key_help() {
        let mut app = App::new();
        assert!(!app.show_help);

        let action = handle_key_event(&mut app, key_event(KeyCode::Char('?')));
        assert_eq!(action, Action::Continue);
        assert!(app.show_help);
    }

    #[test]
    fn test_handle_key_alerts() {
        let mut app = App::new();
        assert!(app.show_alerts);

        let action = handle_key_event(&mut app, key_event(KeyCode::Char('a')));
        assert_eq!(action, Action::Continue);
        assert!(!app.show_alerts);
    }

    #[test]
    fn test_handle_key_unknown() {
        let mut app = App::new();
        let action = handle_key_event(&mut app, key_event(KeyCode::Char('x')));
        assert_eq!(action, Action::Continue);
    }

    #[test]
    fn test_handle_key_enter() {
        let mut app = App::new();
        let action = handle_key_event(&mut app, key_event(KeyCode::Enter));
        assert_eq!(action, Action::Continue);
    }

    #[test]
    fn test_handle_key_esc() {
        let mut app = App::new();
        let action = handle_key_event(&mut app, key_event(KeyCode::Esc));
        assert_eq!(action, Action::Continue);
    }

    // =========================================================================
    // Process Tick Tests
    // =========================================================================

    #[test]
    fn test_process_tick_elapsed() {
        let mut app = App::new();
        let last_tick = Instant::now().checked_sub(Duration::from_millis(200)).unwrap();
        let tick_rate = Duration::from_millis(100);

        let result = process_tick(&mut app, last_tick, tick_rate);
        assert!(result);
        assert_eq!(app.tick_count, 1);
    }

    #[test]
    fn test_process_tick_not_elapsed() {
        let mut app = App::new();
        let last_tick = Instant::now();
        let tick_rate = Duration::from_millis(100);

        let result = process_tick(&mut app, last_tick, tick_rate);
        assert!(!result);
        assert_eq!(app.tick_count, 0);
    }

    #[test]
    fn test_process_tick_force_refresh() {
        let mut app = App::new();
        app.force_refresh = true;
        let last_tick = Instant::now();
        let tick_rate = Duration::from_millis(100);

        let result = process_tick(&mut app, last_tick, tick_rate);
        assert!(result);
        assert!(!app.force_refresh);
        assert_eq!(app.tick_count, 1);
    }

    #[test]
    fn test_process_tick_with_nodes() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));
        let last_tick = Instant::now().checked_sub(Duration::from_millis(200)).unwrap();
        let tick_rate = Duration::from_millis(100);

        let result = process_tick(&mut app, last_tick, tick_rate);
        assert!(result);
        assert_eq!(app.nodes[0].load_history.len(), 1);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_key_sequence() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));
        app.add_node(NodeStatus::new("node-2", "127.0.0.1:9001".parse().unwrap()));

        // Navigate down
        handle_key_event(&mut app, key_event(KeyCode::Down));
        assert!(matches!(app.selected, Selection::Node(0)));

        // Navigate down again
        handle_key_event(&mut app, key_event(KeyCode::Down));
        assert!(matches!(app.selected, Selection::Node(1)));

        // Navigate up
        handle_key_event(&mut app, key_event(KeyCode::Up));
        assert!(matches!(app.selected, Selection::Node(0)));

        // Toggle help
        handle_key_event(&mut app, key_event(KeyCode::Char('?')));
        assert!(app.show_help);

        // Toggle help off
        handle_key_event(&mut app, key_event(KeyCode::Char('?')));
        assert!(!app.show_help);

        // Cycle focus
        handle_key_event(&mut app, key_event(KeyCode::Tab));
        assert_eq!(app.focus, model::Focus::Queue);
    }
}
