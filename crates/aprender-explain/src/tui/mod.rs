//! Interactive TUI mode for trueno-explain
//!
//! Implements Genchi Genbutsu (Go and See) through interactive exploration.
//!
//! Layout:
//! - Left Pane: Source/PTX code with syntax highlighting
//! - Right Pane: Analysis dashboard (registers, memory, warnings, bugs)
//! - Bottom: Status bar with keybindings

mod highlight;
mod widgets;

use crate::analyzer::AnalysisReport;
use crate::ptx::{PtxBugAnalyzer, PtxBugReport};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use presentar_core::{Canvas, Color, Point, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::ColorMode;
use std::io;

use highlight::highlight_ptx_line;
use widgets::render_sidebar;

/// TUI application state
pub struct TuiApp {
    /// The PTX source code to display
    pub ptx_source: String,
    /// Analysis report
    pub report: AnalysisReport,
    /// Bug hunting report (probar-style)
    pub bug_report: PtxBugReport,
    /// Current scroll position in source pane
    pub source_scroll: u16,
    /// Whether sidebar is visible
    pub sidebar_visible: bool,
    /// Should quit
    pub should_quit: bool,
    /// Total lines in source
    source_lines: usize,
}

impl TuiApp {
    /// Create a new TUI application
    #[must_use]
    pub fn new(ptx_source: String, report: AnalysisReport) -> Self {
        let source_lines = ptx_source.lines().count();
        // Perform PTX pattern analysis in strict mode for TUI
        let bug_report = PtxBugAnalyzer::strict().analyze(&ptx_source);
        Self {
            ptx_source,
            report,
            bug_report,
            source_scroll: 0,
            sidebar_visible: true,
            should_quit: false,
            source_lines,
        }
    }

    /// Handle keyboard input
    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char('s') => self.sidebar_visible = !self.sidebar_visible,
            KeyCode::Down | KeyCode::Char('j') => self.scroll_down(),
            KeyCode::Up | KeyCode::Char('k') => self.scroll_up(),
            KeyCode::PageDown => self.page_down(),
            KeyCode::PageUp => self.page_up(),
            KeyCode::Home => self.source_scroll = 0,
            KeyCode::End => self.scroll_to_end(),
            _ => {}
        }
    }

    fn scroll_down(&mut self) {
        if (self.source_scroll as usize) < self.source_lines.saturating_sub(1) {
            self.source_scroll = self.source_scroll.saturating_add(1);
        }
    }

    fn scroll_up(&mut self) {
        self.source_scroll = self.source_scroll.saturating_sub(1);
    }

    fn page_down(&mut self) {
        self.source_scroll =
            self.source_scroll.saturating_add(20).min(self.source_lines.saturating_sub(1) as u16);
    }

    fn page_up(&mut self) {
        self.source_scroll = self.source_scroll.saturating_sub(20);
    }

    fn scroll_to_end(&mut self) {
        self.source_scroll = self.source_lines.saturating_sub(1) as u16;
    }
}

/// Run the TUI application
///
/// # Errors
///
/// Returns `io::Error` if terminal operations fail.
pub fn run_tui(ptx_source: String, report: AnalysisReport) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    execute!(stdout, crossterm::cursor::Hide)?;

    // Create app state
    let mut app = TuiApp::new(ptx_source, report);

    // Create rendering state
    let (width, height) = crossterm::terminal::size()?;
    let mut buffer = CellBuffer::new(width, height);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    // Main loop
    let result = run_app(&mut app, &mut buffer, &mut renderer);

    // Restore terminal
    disable_raw_mode()?;
    execute!(io::stdout(), crossterm::cursor::Show)?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;

    result
}

fn run_app(
    app: &mut TuiApp,
    buffer: &mut CellBuffer,
    renderer: &mut DiffRenderer,
) -> io::Result<()> {
    loop {
        // Resize buffer if terminal size changed
        let (width, height) = crossterm::terminal::size()?;
        if buffer.width() != width || buffer.height() != height {
            *buffer = CellBuffer::new(width, height);
        }

        // Render frame
        {
            let mut canvas = DirectTerminalCanvas::new(buffer);
            ui(&mut canvas, app, width, height);
        }

        // Flush to terminal
        let mut output = Vec::with_capacity(8192);
        renderer.flush(buffer, &mut output).map_err(|e| io::Error::other(e.to_string()))?;
        io::Write::write_all(&mut io::stdout(), &output)?;

        // Handle input
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                app.handle_key(key.code);
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

/// Color constants for the main UI.
const COLOR_CYAN: Color = Color { r: 0.3, g: 1.0, b: 1.0, a: 1.0 };
const COLOR_YELLOW: Color = Color { r: 1.0, g: 1.0, b: 0.3, a: 1.0 };
const COLOR_DIM: Color = Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
const COLOR_LINENUM: Color = Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
const COLOR_BG: Color = Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 };
const COLOR_TEXT: Color = Color { r: 0.8, g: 0.8, b: 0.8, a: 1.0 };
const COLOR_SCROLL_TRACK: Color = Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
const COLOR_SCROLL_THUMB: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };

fn ui(canvas: &mut DirectTerminalCanvas<'_>, app: &TuiApp, width: u16, height: u16) {
    // Clear background
    canvas.fill_rect(
        presentar_core::Rect::new(0.0, 0.0, f32::from(width), f32::from(height)),
        COLOR_BG,
    );

    // Layout: source pane width vs sidebar width
    #[allow(clippy::cast_sign_loss)]
    let (source_width, sidebar_width) = if app.sidebar_visible {
        let sw = (f32::from(width) * 0.4).round() as u16;
        (width.saturating_sub(sw), sw)
    } else {
        (width, 0)
    };

    // Render source pane (top area minus 3-line status bar)
    let source_height = height.saturating_sub(3);
    render_source_pane(canvas, app, 0.0, 0.0, source_width, source_height);

    // Render sidebar if visible
    if app.sidebar_visible && sidebar_width > 0 {
        render_sidebar(canvas, app, f32::from(source_width), 0.0, sidebar_width, source_height);
    }

    // Render status bar at the bottom
    render_status_bar(canvas, width, height);
}

fn render_source_pane(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    height: u16,
) {
    let border_style = TextStyle { color: COLOR_CYAN, ..Default::default() };
    let inner_width = (width as usize).saturating_sub(2); // borders

    // Top border with title
    let title = format!(" PTX: {} ", app.report.name);
    let title_len = title.len();
    let fill_len = inner_width.saturating_sub(title_len);
    let top_line = format!("┌{}{}┐", title, "─".repeat(fill_len));
    canvas.draw_text(&top_line, Point::new(x, y), &border_style);

    // Source content lines
    let content_height = height.saturating_sub(2); // top + bottom borders
    let scroll = app.source_scroll as usize;
    let lines: Vec<&str> = app.ptx_source.lines().collect();

    for row in 0..content_height {
        let line_idx = scroll + row as usize;
        let cy = y + 1.0 + f32::from(row);

        // Left border
        canvas.draw_text("│", Point::new(x, cy), &border_style);

        if line_idx < lines.len() {
            // Line number
            let line_num = format!("{:4} ", line_idx + 1);
            let linenum_style = TextStyle { color: COLOR_LINENUM, ..Default::default() };
            canvas.draw_text(&line_num, Point::new(x + 1.0, cy), &linenum_style);

            // Highlighted source text
            let (text, color) = highlight_ptx_line(lines[line_idx]);
            let text_style = TextStyle { color, ..Default::default() };
            // Truncate to fit: inner_width - 5 (line number) - 1 (scrollbar)
            let max_text_len = inner_width.saturating_sub(6);
            let display_text: String = text.chars().take(max_text_len).collect();
            canvas.draw_text(&display_text, Point::new(x + 6.0, cy), &text_style);
        }

        // Right border (leave room for scrollbar)
        canvas.draw_text("│", Point::new(x + f32::from(width) - 1.0, cy), &border_style);
    }

    // Bottom border
    let bottom_line = format!("└{}┘", "─".repeat(inner_width));
    canvas.draw_text(&bottom_line, Point::new(x, y + f32::from(height) - 1.0), &border_style);

    // Scrollbar (inside the right border area)
    if app.source_lines > 0 {
        draw_scrollbar(
            canvas,
            x + f32::from(width) - 2.0,
            y + 1.0,
            content_height,
            app.source_scroll as usize,
            app.source_lines,
        );
    }
}

fn draw_scrollbar(
    canvas: &mut DirectTerminalCanvas<'_>,
    x: f32,
    top_y: f32,
    height: u16,
    position: usize,
    total: usize,
) {
    let pos_ratio = position as f32 / total.max(1) as f32;
    #[allow(clippy::cast_sign_loss)]
    let thumb_y = (pos_ratio * f32::from(height - 1)).round() as u16;
    let track_style = TextStyle { color: COLOR_SCROLL_TRACK, ..Default::default() };
    let thumb_style = TextStyle { color: COLOR_SCROLL_THUMB, ..Default::default() };

    for y in 0..height {
        if y == thumb_y {
            canvas.draw_text("\u{2588}", Point::new(x, top_y + f32::from(y)), &thumb_style);
        } else {
            canvas.draw_text("\u{2502}", Point::new(x, top_y + f32::from(y)), &track_style);
        }
    }
}

fn render_status_bar(canvas: &mut DirectTerminalCanvas<'_>, width: u16, height: u16) {
    let status_y = f32::from(height - 3);
    let border_style = TextStyle { color: COLOR_DIM, ..Default::default() };
    let inner_width = (width as usize).saturating_sub(2);

    // Status bar top border
    let top = format!("┌{}┐", "─".repeat(inner_width));
    canvas.draw_text(&top, Point::new(0.0, status_y), &border_style);

    // Status content line
    canvas.draw_text("│", Point::new(0.0, status_y + 1.0), &border_style);

    let key_style = TextStyle { color: COLOR_YELLOW, ..Default::default() };
    let text_style = TextStyle { color: COLOR_TEXT, ..Default::default() };

    let mut cx: f32 = 1.0;
    let items: &[(&str, &str)] =
        &[(" q", ":Quit "), ("s", ":Sidebar "), ("jk", ":Scroll "), ("PgUp/Dn", ":Page ")];

    for &(key, desc) in items {
        canvas.draw_text(key, Point::new(cx, status_y + 1.0), &key_style);
        cx += key.len() as f32;
        canvas.draw_text(desc, Point::new(cx, status_y + 1.0), &text_style);
        cx += desc.len() as f32;
    }

    canvas.draw_text("│", Point::new(f32::from(width) - 1.0, status_y + 1.0), &border_style);

    // Status bar bottom border
    let bottom = format!("└{}┘", "─".repeat(inner_width));
    canvas.draw_text(&bottom, Point::new(0.0, status_y + 2.0), &border_style);
}

#[cfg(test)]
mod tests;
