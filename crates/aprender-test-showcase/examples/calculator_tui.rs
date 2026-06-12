//! Calculator TUI Example
//!
//! This example demonstrates the calculator TUI interface.
//! Uses presentar-terminal CellBuffer + DiffRenderer for rendering.
//!
//! Run with: cargo run --example calculator_tui --features tui

use std::io::{self, Write};

use crossterm::{
    cursor,
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use presentar_terminal::{CellBuffer, Color, DiffRenderer, Modifiers};
use showcase_calculator::tui::{render_to_buffer, CalculatorApp, InputHandler, KeyAction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        cursor::Hide
    )?;

    // Run app
    let result = run_app(&mut stdout);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        stdout,
        LeaveAlternateScreen,
        DisableMouseCapture,
        cursor::Show,
    )?;

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

fn run_app(stdout: &mut io::Stdout) -> Result<(), Box<dyn std::error::Error>> {
    let mut app = CalculatorApp::new();
    let input_handler = InputHandler::new();

    let (mut width, mut height) = crossterm::terminal::size().unwrap_or((80, 24));
    let mut cell_buf = CellBuffer::new(width, height);
    let mut renderer = DiffRenderer::new();

    loop {
        // Check for terminal resize
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 24));
        if w != width || h != height {
            width = w;
            height = h;
            cell_buf.resize(w, h);
            renderer.reset();
        }

        // Render calculator to text buffer, then transfer to cell buffer
        let text_buf = render_to_buffer(&app, width, height);
        cell_buf.clear();

        // Transfer text content to presentar CellBuffer
        let content = text_buf.to_string_content();
        let mut cx: u16 = 0;
        let mut cy: u16 = 0;
        for ch in content.chars() {
            if cx >= width {
                cx = 0;
                cy += 1;
            }
            if cy >= height {
                break;
            }
            if ch != ' ' {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                cell_buf.update(cx, cy, s, Color::WHITE, Color::TRANSPARENT, Modifiers::NONE);
            }
            cx += 1;
        }

        // Flush to terminal
        renderer.flush(&mut cell_buf, stdout)?;
        stdout.flush()?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press
                && handle_action(&mut app, input_handler.handle_key(key))
            {
                break;
            }
        }

        if app.should_quit() {
            break;
        }
    }

    Ok(())
}
