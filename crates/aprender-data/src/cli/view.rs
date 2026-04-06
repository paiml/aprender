//! TUI view commands for interactive dataset viewing.

use std::{io::Write, path::Path};

use super::basic::load_dataset;
use crate::tui::{DatasetAdapter, DatasetViewer};

/// Interactive TUI viewer for datasets.
pub(crate) fn cmd_view(path: &Path, initial_search: Option<&str>) -> crate::Result<()> {
    use std::io::stdout;

    use crossterm::{cursor, execute, terminal};

    let dataset = load_dataset(path)?;
    let adapter = DatasetAdapter::from_dataset(&dataset)
        .map_err(|e| crate::Error::storage(format!("TUI adapter error: {e}")))?;

    // Get terminal size
    let (width, height) = terminal::size().unwrap_or((80, 24));
    let mut viewer = DatasetViewer::with_dimensions(adapter, width, height.saturating_sub(2));

    // Apply initial search if provided
    if let Some(query) = initial_search {
        viewer.search(query);
    }

    // Enter raw mode for keyboard input
    terminal::enable_raw_mode()
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

    let mut stdout = stdout();

    // Hide cursor
    execute!(stdout, cursor::Hide)
        .map_err(|e| crate::Error::storage(format!("Terminal error: {e}")))?;

    let result = run_tui_loop(&mut viewer, &mut stdout, path);

    // Cleanup: restore terminal — log failures to avoid corrupted terminal state
    if execute!(stdout, cursor::Show).is_err() {
        eprintln!("Warning: failed to restore cursor visibility");
    }
    if terminal::disable_raw_mode().is_err() {
        eprintln!("Warning: failed to disable raw mode — run 'reset' to fix terminal");
    }

    result
}

/// Render the TUI frame (title bar, data lines, status bar).
fn render_frame<W: Write>(
    viewer: &DatasetViewer,
    stdout: &mut W,
    path: &std::path::Path,
) -> crate::Result<()> {
    use crossterm::{
        cursor, execute,
        style::{Attribute, Print, SetAttribute},
        terminal::{self, Clear, ClearType},
    };

    execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0)).map_err(term_err)?;

    let (width, _) = terminal::size().unwrap_or((80, 24));
    let title = format!(
        " {} | {} rows | {}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        viewer.row_count(),
        if viewer.adapter().is_streaming() {
            "Streaming"
        } else {
            "InMemory"
        }
    );
    execute!(
        stdout,
        SetAttribute(Attribute::Reverse),
        Print(format!("{:width$}", title, width = width as usize)),
        SetAttribute(Attribute::Reset),
        Print("\r\n")
    )
    .map_err(term_err)?;

    for line in viewer.render_lines() {
        execute!(stdout, Print(&line), Print("\r\n")).map_err(term_err)?;
    }

    let status = format!(
        " Row {}-{} of {} | {} scroll | PgUp/PgDn page | Home/End | /search | q quit ",
        viewer.scroll_offset() + 1,
        (viewer.scroll_offset() + viewer.visible_row_count() as usize).min(viewer.row_count()),
        viewer.row_count(),
        "\u{2191}\u{2193}"
    );
    execute!(
        stdout,
        SetAttribute(Attribute::Reverse),
        Print(format!("{:width$}", status, width = width as usize)),
        SetAttribute(Attribute::Reset)
    )
    .map_err(term_err)?;

    stdout.flush().map_err(term_err)?;
    Ok(())
}

/// Handle a key event, returning true if the loop should break.
fn handle_key_input<W: Write>(
    viewer: &mut DatasetViewer,
    stdout: &mut W,
    key: crossterm::event::KeyEvent,
) -> crate::Result<bool> {
    use crossterm::event::{KeyCode, KeyModifiers};

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(true),
        KeyCode::Down | KeyCode::Char('j') => viewer.scroll_down(),
        KeyCode::Up | KeyCode::Char('k') => viewer.scroll_up(),
        KeyCode::PageDown | KeyCode::Char(' ') => viewer.page_down(),
        KeyCode::PageUp => viewer.page_up(),
        KeyCode::Home | KeyCode::Char('g') => viewer.home(),
        KeyCode::End | KeyCode::Char('G') => viewer.end(),
        KeyCode::Char('/') => {
            if let Some(query) = prompt_search(stdout)? {
                viewer.search(&query);
            }
        }
        _ => {}
    }
    Ok(false)
}

/// Run the TUI event loop.
pub(crate) fn run_tui_loop<W: Write>(
    viewer: &mut DatasetViewer,
    stdout: &mut W,
    path: &std::path::Path,
) -> crate::Result<()> {
    use crossterm::event::{self, Event};

    loop {
        render_frame(viewer, stdout, path)?;

        if event::poll(std::time::Duration::from_millis(100)).map_err(term_err)? {
            if let Event::Key(key) = event::read().map_err(term_err)? {
                if handle_key_input(viewer, stdout, key)? {
                    break;
                }
            }

            if let Event::Resize(w, h) = event::read().map_err(term_err)? {
                viewer.set_dimensions(w, h.saturating_sub(2));
            }
        }
    }

    Ok(())
}

/// Convert a crossterm error to a crate terminal error
fn term_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::storage(format!("Terminal error: {e}"))
}

/// Prompt for search query.
pub(crate) fn prompt_search<W: Write>(stdout: &mut W) -> crate::Result<Option<String>> {
    use crossterm::{
        cursor,
        event::{self, Event, KeyCode},
        execute,
        style::Print,
        terminal::{self, Clear, ClearType},
    };

    let (_, height) = terminal::size().unwrap_or((80, 24));

    // Move to bottom and show prompt
    execute!(
        stdout,
        cursor::MoveTo(0, height - 1),
        Clear(ClearType::CurrentLine),
        cursor::Show,
        Print("Search: ")
    )
    .map_err(term_err)?;
    stdout.flush().map_err(term_err)?;

    let mut query = String::new();

    loop {
        if let Event::Key(key) = event::read().map_err(term_err)? {
            match key.code {
                KeyCode::Enter => {
                    execute!(stdout, cursor::Hide).map_err(term_err)?;
                    return Ok(if query.is_empty() { None } else { Some(query) });
                }
                KeyCode::Esc => {
                    execute!(stdout, cursor::Hide).map_err(term_err)?;
                    return Ok(None);
                }
                KeyCode::Backspace => {
                    query.pop();
                    execute!(
                        stdout,
                        cursor::MoveTo(8, height - 1),
                        Clear(ClearType::UntilNewLine),
                        Print(&query)
                    )
                    .map_err(term_err)?;
                }
                KeyCode::Char(c) => {
                    query.push(c);
                    execute!(stdout, Print(c)).map_err(term_err)?;
                }
                _ => {}
            }
            stdout.flush().map_err(term_err)?;
        }
    }
}
