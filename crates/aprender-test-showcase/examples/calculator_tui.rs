//! Calculator TUI Example
//!
//! This example demonstrates the calculator TUI interface.
//!
//! Run with: cargo run --example calculator_tui --features tui

use std::io;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use showcase_calculator::tui::{render, CalculatorApp, InputHandler, KeyAction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let result = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {err}");
    }

    Ok(())
}

/// Handle a single key action and return whether to quit
fn handle_action(app: &mut CalculatorApp, action: KeyAction) -> bool {
    match action {
        KeyAction::InsertChar(c) if InputHandler::is_valid_char(c) => app.insert_char(c),
        KeyAction::Backspace => app.delete_char(),
        KeyAction::Delete => app.delete_char_forward(),
        KeyAction::CursorLeft => app.move_cursor_left(),
        KeyAction::CursorRight => app.move_cursor_right(),
        KeyAction::CursorHome => app.move_cursor_start(),
        KeyAction::CursorEnd => app.move_cursor_end(),
        KeyAction::Evaluate => app.evaluate(),
        KeyAction::Clear => app.clear(),
        KeyAction::ClearAll => app.clear_all(),
        KeyAction::RecallLast => app.recall_last(),
        KeyAction::Quit => return true,
        KeyAction::InsertChar(_) | KeyAction::None => {}
    }
    false
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut app = CalculatorApp::new();
    let input_handler = InputHandler::new();

    loop {
        terminal.draw(|f| render(&app, f))?;

        if let Event::Key(key) = event::read()? {
            if handle_action(&mut app, input_handler.handle_key(key)) {
                break;
            }
        }

        if app.should_quit() {
            break;
        }
    }

    Ok(())
}
