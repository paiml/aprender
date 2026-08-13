//! TUI command implementation — presentar-terminal powered
//!
//! Interactive terminal user interface for APR model exploration.
//! Contract: contracts/tui-rendering-ux-v1.yaml (tui_model_explorer)
//!
//! Layout (3-zone per contract):
//!   ┌───────────────────────────────────────────┐
//!   │ HEADER: APR Model Inspector - <path>      │
//!   ├───────────────────────────────────────────┤
//!   │ [Overview] [Tensors] [Stats] [Help]       │
//!   ├────────────────────┬──────────────────────┤
//!   │  Primary panel     │  Detail panel        │
//!   │  (navigable list)  │  (context for sel)   │
//!   ├────────────────────┴──────────────────────┤
//!   │ FOOTER: keybindings                       │
//!   └───────────────────────────────────────────┘
//!
//! Keyboard (per contract):
//!   j/↓ = next, k/↑ = prev, Tab = next tab, 1-4 = tab, q = quit, ? = help

use crate::error::{CliError, Result};
use aprender::format::validation::{AprValidator, TensorStats};
use aprender::serialization::apr::AprReader;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{self, ClearType},
};
use humansize::{format_size, BINARY};
use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::ColorMode;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ── Theme (struct literal = const, per contract: no hardcoded ANSI) ──────

const CYAN: Color = Color {
    r: 0.4,
    g: 0.85,
    b: 1.0,
    a: 1.0,
};
const WHITE: Color = Color {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
const DIM: Color = Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};
const YELLOW: Color = Color {
    r: 1.0,
    g: 0.85,
    b: 0.3,
    a: 1.0,
};
const GREEN: Color = Color {
    r: 0.3,
    g: 1.0,
    b: 0.5,
    a: 1.0,
};
const RED: Color = Color {
    r: 1.0,
    g: 0.3,
    b: 0.3,
    a: 1.0,
};
const SELECTED_BG: Color = Color {
    r: 0.15,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};
const HEADER_BG: Color = Color {
    r: 0.1,
    g: 0.12,
    b: 0.18,
    a: 1.0,
};

fn text_style(color: Color) -> TextStyle {
    TextStyle {
        color,
        ..TextStyle::default()
    }
}

fn bold_style(color: Color) -> TextStyle {
    TextStyle {
        color,
        weight: presentar_core::FontWeight::Bold,
        ..TextStyle::default()
    }
}

// ── Tab enum ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Tensors,
    Stats,
    Help,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Overview, Tab::Tensors, Tab::Stats, Tab::Help];

    fn title(self) -> &'static str {
        match self {
            Tab::Overview => " Overview ",
            Tab::Tensors => " Tensors ",
            Tab::Stats => " Stats ",
            Tab::Help => " Help ",
        }
    }

    fn index(self) -> usize {
        match self {
            Tab::Overview => 0,
            Tab::Tensors => 1,
            Tab::Stats => 2,
            Tab::Help => 3,
        }
    }

    fn from_index(i: usize) -> Self {
        match i % 4 {
            0 => Tab::Overview,
            1 => Tab::Tensors,
            2 => Tab::Stats,
            _ => Tab::Help,
        }
    }
}

// ── Tensor info ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
    stats: Option<TensorStats>,
}

// ── Application state ─────────────────────────────────────────────────────

struct App {
    file_path: Option<PathBuf>,
    reader: Option<AprReader>,
    validation_score: Option<u8>,
    current_tab: Tab,
    tensors: Vec<TensorInfo>,
    selected: usize,
    should_quit: bool,
    error_message: Option<String>,
}

impl App {
    fn new(file_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            file_path,
            reader: None,
            validation_score: None,
            current_tab: Tab::Overview,
            tensors: Vec::new(),
            selected: 0,
            should_quit: false,
            error_message: None,
        };
        app.load_model();
        app
    }

    fn load_model(&mut self) {
        let Some(path) = &self.file_path else { return };
        match std::fs::read(path) {
            Ok(data) => {
                let mut validator = AprValidator::new();
                let report = validator.validate_bytes(&data);
                // #1866: percentage of the checks that RAN, not raw awarded
                // points banded against a 100 the checklist cannot reach.
                // The panel used to paint a healthy model "QA Score: 3/100" in
                // red. `None` (nothing ran) hides the panel rather than
                // painting a zero.
                self.validation_score = report
                    .implemented_score()
                    .pct()
                    .map(|pct| pct.round() as u8);
                match AprReader::from_bytes(data) {
                    Ok(reader) => {
                        self.load_tensors(&reader);
                        self.reader = Some(reader);
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Failed to parse model: {e}"));
                    }
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to read file: {e}"));
            }
        }
    }

    fn load_tensors(&mut self, reader: &AprReader) {
        self.tensors.clear();
        for desc in &reader.tensors {
            let stats = reader
                .read_tensor_f32(&desc.name)
                .ok()
                .map(|data| TensorStats::compute(&desc.name, &data));
            self.tensors.push(TensorInfo {
                name: desc.name.clone(),
                shape: desc.shape.clone(),
                dtype: desc.dtype.clone(),
                size_bytes: desc.size,
                stats,
            });
        }
    }

    fn next_tab(&mut self) {
        self.current_tab = Tab::from_index(self.current_tab.index() + 1);
    }

    fn prev_tab(&mut self) {
        self.current_tab = Tab::from_index(self.current_tab.index() + 3);
    }

    fn select_next(&mut self) {
        if !self.tensors.is_empty() {
            self.selected = (self.selected + 1) % self.tensors.len();
        }
    }

    fn select_prev(&mut self) {
        if !self.tensors.is_empty() {
            self.selected = if self.selected == 0 {
                self.tensors.len() - 1
            } else {
                self.selected - 1
            };
        }
    }

    fn selected_tensor(&self) -> Option<&TensorInfo> {
        self.tensors.get(self.selected)
    }

    fn handle_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Tab => self.next_tab(),
            KeyCode::BackTab => self.prev_tab(),
            KeyCode::Char('1') => self.current_tab = Tab::Overview,
            KeyCode::Char('2') => self.current_tab = Tab::Tensors,
            KeyCode::Char('3') => self.current_tab = Tab::Stats,
            KeyCode::Char('?') | KeyCode::Char('4') => self.current_tab = Tab::Help,
            KeyCode::Down | KeyCode::Char('j') => self.select_next(),
            KeyCode::Up | KeyCode::Char('k') => self.select_prev(),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => {
                self.selected = self.tensors.len().saturating_sub(1);
            }
            _ => {}
        }
        false
    }
}

// ── Entry point ───────────────────────────────────────────────────────────

#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "long_running_graceful")]
pub(crate) fn run(file: Option<PathBuf>) -> Result<()> {
    let mut stdout = io::stdout();
    setup_terminal(&mut stdout)?;

    let mut app = App::new(file);
    let color_mode = ColorMode::detect();
    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    let result = run_loop(&mut stdout, &mut app, &mut renderer);

    cleanup_terminal(&mut stdout);
    result
}

fn setup_terminal(stdout: &mut io::Stdout) -> Result<()> {
    terminal::enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to enable raw mode: {e}")))?;
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )
    .map_err(|e| CliError::ValidationFailed(format!("Failed to setup terminal: {e}")))?;
    Ok(())
}

fn cleanup_terminal(stdout: &mut io::Stdout) {
    terminal::disable_raw_mode().ok();
    execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show).ok();
}

fn run_loop(stdout: &mut io::Stdout, app: &mut App, renderer: &mut DiffRenderer) -> Result<()> {
    let render_interval = Duration::from_millis(16); // 60 FPS budget
    let mut last_render = Instant::now();
    let mut force_full = true;

    loop {
        // Process input (non-blocking)
        if event::poll(Duration::from_millis(16))
            .map_err(|e| CliError::ValidationFailed(format!("Event poll error: {e}")))?
        {
            match event::read()
                .map_err(|e| CliError::ValidationFailed(format!("Event read error: {e}")))?
            {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press && app.handle_key(key.code) {
                        return Ok(());
                    }
                }
                Event::Resize(_, _) => {
                    force_full = true;
                }
                _ => {}
            }
        }

        // Rate-limited rendering
        if last_render.elapsed() >= render_interval {
            let (width, height) = terminal::size()
                .map_err(|e| CliError::ValidationFailed(format!("Terminal size error: {e}")))?;
            let mut buffer = CellBuffer::new(width, height);

            draw(app, &mut buffer);

            execute!(stdout, cursor::MoveTo(0, 0)).ok();
            let mut output = Vec::with_capacity(32768);

            if force_full {
                renderer.render_full(&mut buffer, &mut output).ok();
                force_full = false;
            } else {
                renderer.flush(&mut buffer, &mut output).ok();
            }

            stdout.write_all(&output).ok();
            stdout.flush().ok();
            last_render = Instant::now();
        }
    }
}

// ── Drawing ───────────────────────────────────────────────────────────────

fn draw(app: &App, buffer: &mut CellBuffer) {
    let w = buffer.width() as f32;
    let h = buffer.height() as f32;
    if w < 20.0 || h < 8.0 {
        return;
    }

    let mut canvas = DirectTerminalCanvas::new(buffer);

    // 3-zone layout: header(1) + tabs(1) + body(h-3) + footer(1)
    draw_header(app, &mut canvas, w);
    draw_tabs(app, &mut canvas, w);

    let body_rect = Rect::new(0.0, 2.0, w, h - 3.0);

    match app.current_tab {
        Tab::Overview => draw_overview(app, &mut canvas, body_rect),
        Tab::Tensors => draw_tensors(app, &mut canvas, body_rect),
        Tab::Stats => draw_stats(app, &mut canvas, body_rect),
        Tab::Help => draw_help(&mut canvas, body_rect),
    }

    draw_footer(app, &mut canvas, w, h - 1.0);
}

fn draw_header(app: &App, canvas: &mut DirectTerminalCanvas<'_>, width: f32) {
    // Fill header background
    canvas.fill_rect(Rect::new(0.0, 0.0, width, 1.0), HEADER_BG);

    let title = match &app.file_path {
        Some(p) => format!(" APR Model Inspector — {} ", p.display()),
        None => " APR Model Inspector — No file loaded ".to_string(),
    };
    canvas.draw_text(&title, Point::new(0.0, 0.0), &bold_style(WHITE));

    // Right-align version
    let version = format!("v{} ", env!("CARGO_PKG_VERSION"));
    let vx = width - version.len() as f32;
    canvas.draw_text(&version, Point::new(vx, 0.0), &text_style(DIM));
}

fn draw_tabs(app: &App, canvas: &mut DirectTerminalCanvas<'_>, width: f32) {
    let y = 1.0;
    let mut x = 1.0_f32;

    for tab in Tab::ALL {
        let title = tab.title();
        let is_active = tab == app.current_tab;

        if is_active {
            canvas.fill_rect(Rect::new(x, y, title.len() as f32, 1.0), CYAN);
            canvas.draw_text(
                title,
                Point::new(x, y),
                &bold_style(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.1,
                    a: 1.0,
                }),
            );
        } else {
            canvas.draw_text(title, Point::new(x, y), &text_style(DIM));
        }
        x += title.len() as f32 + 1.0;
    }

    let hint = "[1-4/Tab]";
    let hx = width - hint.len() as f32 - 1.0;
    canvas.draw_text(hint, Point::new(hx, y), &text_style(DIM));
}

fn draw_footer(app: &App, canvas: &mut DirectTerminalCanvas<'_>, width: f32, y: f32) {
    canvas.fill_rect(Rect::new(0.0, y, width, 1.0), HEADER_BG);

    let status = match app.current_tab {
        Tab::Overview => " Tab:switch  q:quit",
        Tab::Tensors => " j/k:navigate  Tab:switch  q:quit",
        Tab::Stats => " Tab:switch  q:quit",
        Tab::Help => " Tab:switch  q:quit  ?:help",
    };
    canvas.draw_text(status, Point::new(0.0, y), &text_style(DIM));

    if !app.tensors.is_empty() {
        let info = format!("{} tensors ", app.tensors.len());
        let ix = width - info.len() as f32;
        canvas.draw_text(&info, Point::new(ix, y), &text_style(DIM));
    }
}

// ── Tab panels ────────────────────────────────────────────────────────────

fn draw_overview(app: &App, canvas: &mut DirectTerminalCanvas<'_>, area: Rect) {
    if let Some(ref error) = app.error_message {
        canvas.draw_text(
            error,
            Point::new(area.x + 2.0, area.y + 1.0),
            &text_style(RED),
        );
        return;
    }

    let Some(reader) = &app.reader else {
        canvas.draw_text(
            "No model loaded. Run: apr tui <model.apr>",
            Point::new(area.x + 2.0, area.y + 1.0),
            &text_style(YELLOW),
        );
        return;
    };

    // Border
    canvas.stroke_rect(area, CYAN, 1.0);
    canvas.draw_text(
        " Overview ",
        Point::new(area.x + 2.0, area.y),
        &bold_style(WHITE),
    );

    let lx = area.x + 2.0;
    let vx = area.x + 16.0;
    let mut y = area.y + 1.0;

    let fields: &[(&str, Option<String>)] = &[
        (
            "Model Type:",
            reader
                .metadata
                .get("model_type")
                .map(|v| v.to_string().trim_matches('"').to_string()),
        ),
        (
            "Model Name:",
            reader
                .metadata
                .get("model_name")
                .map(|v| v.to_string().trim_matches('"').to_string()),
        ),
        (
            "Framework:",
            reader
                .metadata
                .get("framework")
                .map(|v| v.to_string().trim_matches('"').to_string()),
        ),
    ];

    for (label, value) in fields {
        if let Some(v) = value {
            canvas.draw_text(label, Point::new(lx, y), &text_style(CYAN));
            canvas.draw_text(v, Point::new(vx, y), &text_style(WHITE));
            y += 1.0;
        }
    }

    y += 1.0;
    canvas.draw_text("Tensors:", Point::new(lx, y), &text_style(CYAN));
    canvas.draw_text(
        &format!("{}", app.tensors.len()),
        Point::new(vx, y),
        &text_style(WHITE),
    );
    y += 1.0;

    let total_size: usize = app.tensors.iter().map(|t| t.size_bytes).sum();
    canvas.draw_text("Total Size:", Point::new(lx, y), &text_style(CYAN));
    canvas.draw_text(
        &format_size(total_size, BINARY),
        Point::new(vx, y),
        &text_style(WHITE),
    );
    y += 1.0;

    if let Some(score) = app.validation_score {
        let score_color = if score >= 90 {
            GREEN
        } else if score >= 70 {
            YELLOW
        } else {
            RED
        };
        canvas.draw_text("QA Score:", Point::new(lx, y), &text_style(CYAN));
        canvas.draw_text(
            &format!("{score}/100"),
            Point::new(vx, y),
            &text_style(score_color),
        );
        y += 1.0;
    }

    if let Some(hp) = reader.metadata.get("hyperparameters") {
        y += 1.0;
        canvas.draw_text("Hyperparameters:", Point::new(lx, y), &bold_style(CYAN));
        y += 1.0;
        if let Some(obj) = hp.as_object() {
            for (k, v) in obj {
                if y >= area.y + area.height - 1.0 {
                    break;
                }
                canvas.draw_text(
                    &format!("{k}: {v}"),
                    Point::new(lx + 2.0, y),
                    &text_style(DIM),
                );
                y += 1.0;
            }
        }
    }
}

fn draw_tensors(app: &App, canvas: &mut DirectTerminalCanvas<'_>, area: Rect) {
    // Two-column layout: 60/40 split (per contract)
    let split = if area.width >= 100.0 {
        area.width * 0.6
    } else {
        area.width * 0.5
    };
    let left = Rect::new(area.x, area.y, split, area.height);
    let right = Rect::new(area.x + split, area.y, area.width - split, area.height);

    // Left: tensor list
    canvas.stroke_rect(left, CYAN, 1.0);
    canvas.draw_text(
        " Tensors (j/k) ",
        Point::new(left.x + 2.0, left.y),
        &bold_style(WHITE),
    );

    let visible_rows = (left.height as usize).saturating_sub(2);
    let scroll = if app.selected >= visible_rows {
        app.selected - visible_rows + 1
    } else {
        0
    };

    for (i, tensor) in app
        .tensors
        .iter()
        .skip(scroll)
        .take(visible_rows)
        .enumerate()
    {
        let y = left.y + 1.0 + i as f32;
        let is_selected = scroll + i == app.selected;
        let shape_str = format!("{:?}", tensor.shape);
        let max_name = (split as usize).saturating_sub(shape_str.len() + 6);
        let line = format!("{} {}", truncate_name(&tensor.name, max_name), shape_str);

        if is_selected {
            canvas.fill_rect(
                Rect::new(left.x + 1.0, y, left.width - 2.0, 1.0),
                SELECTED_BG,
            );
            canvas.draw_text("▸ ", Point::new(left.x + 1.0, y), &text_style(CYAN));
            canvas.draw_text(&line, Point::new(left.x + 3.0, y), &bold_style(WHITE));
        } else {
            canvas.draw_text(
                &format!("  {line}"),
                Point::new(left.x + 1.0, y),
                &text_style(DIM),
            );
        }
    }

    // Right: tensor detail
    canvas.stroke_rect(right, CYAN, 1.0);
    canvas.draw_text(
        " Details ",
        Point::new(right.x + 2.0, right.y),
        &bold_style(WHITE),
    );

    if let Some(tensor) = app.selected_tensor() {
        let dx = right.x + 2.0;
        let vx = right.x + 10.0;
        let mut y = right.y + 1.0;

        let details: &[(&str, String)] = &[
            ("Name:", tensor.name.clone()),
            ("Shape:", format!("{:?}", tensor.shape)),
            ("DType:", tensor.dtype.clone()),
            ("Size:", format_size(tensor.size_bytes, BINARY)),
        ];

        for (label, value) in details {
            if y >= right.y + right.height - 1.0 {
                break;
            }
            canvas.draw_text(label, Point::new(dx, y), &text_style(CYAN));
            canvas.draw_text(value, Point::new(vx, y), &text_style(WHITE));
            y += 1.0;
        }

        if let Some(ref stats) = tensor.stats {
            y += 1.0;
            if y < right.y + right.height - 1.0 {
                canvas.draw_text("Statistics:", Point::new(dx, y), &bold_style(CYAN));
                y += 1.0;
            }

            let stat_lines: &[(&str, String)] = &[
                ("Min:", format!("{:.6}", stats.min)),
                ("Max:", format!("{:.6}", stats.max)),
                ("Mean:", format!("{:.6}", stats.mean)),
                ("Std:", format!("{:.6}", stats.std)),
                ("Zeros:", format!("{}", stats.zero_count)),
                ("NaNs:", format!("{}", stats.nan_count)),
                ("Infs:", format!("{}", stats.inf_count)),
            ];

            for (label, value) in stat_lines {
                if y >= right.y + right.height - 1.0 {
                    break;
                }
                canvas.draw_text(label, Point::new(dx + 2.0, y), &text_style(DIM));
                canvas.draw_text(value, Point::new(dx + 10.0, y), &text_style(WHITE));
                y += 1.0;
            }

            // Sparkline distribution
            if y + 2.0 < right.y + right.height {
                y += 1.0;
                canvas.draw_text("Distribution:", Point::new(dx, y), &text_style(CYAN));
                y += 1.0;
                let spark_width = (right.width as usize).saturating_sub(6);
                let spark = build_sparkline(stats, spark_width);
                canvas.draw_text(&spark, Point::new(dx + 1.0, y), &text_style(CYAN));
            }
        }
    } else {
        canvas.draw_text(
            "Select a tensor",
            Point::new(right.x + 2.0, right.y + 2.0),
            &text_style(DIM),
        );
    }
}

fn draw_stats(app: &App, canvas: &mut DirectTerminalCanvas<'_>, area: Rect) {
    canvas.stroke_rect(area, CYAN, 1.0);
    canvas.draw_text(
        " Statistics ",
        Point::new(area.x + 2.0, area.y),
        &bold_style(WHITE),
    );

    if app.tensors.is_empty() {
        canvas.draw_text(
            "No tensor data available",
            Point::new(area.x + 2.0, area.y + 2.0),
            &text_style(YELLOW),
        );
        return;
    }

    // Column positions (proportional)
    let w = area.width;
    let cx = [
        area.x + 2.0,
        area.x + w * 0.30,
        area.x + w * 0.55,
        area.x + w * 0.70,
        area.x + w * 0.85,
    ];

    // Header
    let y = area.y + 1.0;
    let headers = ["Tensor", "Shape", "Size", "Mean", "Std"];
    for (i, h) in headers.iter().enumerate() {
        canvas.draw_text(h, Point::new(cx[i], y), &bold_style(CYAN));
    }

    // Separator
    let sep = "─".repeat((w as usize).saturating_sub(2));
    canvas.draw_text(
        &sep,
        Point::new(area.x + 1.0, area.y + 2.0),
        &text_style(DIM),
    );

    // Rows
    let visible = (area.height as usize).saturating_sub(4);
    for (i, tensor) in app.tensors.iter().take(visible).enumerate() {
        let row_y = area.y + 3.0 + i as f32;
        if row_y >= area.y + area.height - 1.0 {
            break;
        }

        let name_max = (w * 0.28) as usize;
        canvas.draw_text(
            &truncate_name(&tensor.name, name_max),
            Point::new(cx[0], row_y),
            &text_style(WHITE),
        );
        canvas.draw_text(
            &format!("{:?}", tensor.shape),
            Point::new(cx[1], row_y),
            &text_style(DIM),
        );
        canvas.draw_text(
            &format_size(tensor.size_bytes, BINARY),
            Point::new(cx[2], row_y),
            &text_style(DIM),
        );

        let (mean, std) = match &tensor.stats {
            Some(s) => (format!("{:.4}", s.mean), format!("{:.4}", s.std)),
            None => ("-".to_string(), "-".to_string()),
        };
        canvas.draw_text(&mean, Point::new(cx[3], row_y), &text_style(WHITE));
        canvas.draw_text(&std, Point::new(cx[4], row_y), &text_style(WHITE));
    }
}

fn draw_help(canvas: &mut DirectTerminalCanvas<'_>, area: Rect) {
    canvas.stroke_rect(area, CYAN, 1.0);
    canvas.draw_text(
        " Help ",
        Point::new(area.x + 2.0, area.y),
        &bold_style(WHITE),
    );

    let mut y = area.y + 1.0;

    canvas.draw_text(
        "Keyboard Shortcuts",
        Point::new(area.x + 2.0, y),
        &bold_style(CYAN),
    );
    y += 2.0;

    let bindings: &[(&str, &str)] = &[
        ("1, 2, 3, ?", "Switch to Overview/Tensors/Stats/Help"),
        ("Tab / Shift-Tab", "Next / Previous tab"),
        ("j / Down", "Select next tensor"),
        ("k / Up", "Select previous tensor"),
        ("Home / End", "Jump to first / last tensor"),
        ("q / Esc", "Quit"),
    ];

    for (key, desc) in bindings {
        if y >= area.y + area.height - 1.0 {
            break;
        }
        canvas.draw_text(key, Point::new(area.x + 4.0, y), &text_style(YELLOW));
        canvas.draw_text(desc, Point::new(area.x + 22.0, y), &text_style(WHITE));
        y += 1.0;
    }

    y += 1.0;
    if y < area.y + area.height - 1.0 {
        canvas.draw_text("About", Point::new(area.x + 2.0, y), &bold_style(CYAN));
        y += 2.0;
    }
    if y < area.y + area.height - 1.0 {
        canvas.draw_text(
            "APR Model Inspector — presentar-terminal TUI",
            Point::new(area.x + 4.0, y),
            &text_style(WHITE),
        );
        y += 1.0;
    }
    if y < area.y + area.height - 1.0 {
        canvas.draw_text(
            &format!("Version: {}", env!("CARGO_PKG_VERSION")),
            Point::new(area.x + 4.0, y),
            &text_style(DIM),
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else if max_len < 4 {
        name.chars().take(max_len).collect()
    } else {
        let mut end = max_len - 3;
        while end > 0 && !name.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &name[..end])
    }
}

fn build_sparkline(stats: &TensorStats, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut result = String::with_capacity(width);
    for i in 0..width {
        let x = (i as f64 / width as f64) * 6.0 - 3.0;
        let gaussian = (-x * x / 2.0).exp();
        let idx = (gaussian * 7.0).round() as usize;
        result.push(SPARK[idx.min(7)]);
    }
    result
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_titles() {
        assert_eq!(Tab::Overview.title(), " Overview ");
        assert_eq!(Tab::Help.title(), " Help ");
    }

    #[test]
    fn test_tab_index_roundtrip() {
        for tab in Tab::ALL {
            assert_eq!(Tab::from_index(tab.index()), tab);
        }
    }

    #[test]
    fn test_app_new_no_file() {
        let app = App::new(None);
        assert!(app.file_path.is_none());
        assert!(app.reader.is_none());
        assert!(app.tensors.is_empty());
        assert!(!app.should_quit);
    }

    #[test]
    fn test_app_next_tab() {
        let mut app = App::new(None);
        assert_eq!(app.current_tab, Tab::Overview);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Tensors);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Stats);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Help);
        app.next_tab();
        assert_eq!(app.current_tab, Tab::Overview);
    }

    #[test]
    fn test_app_prev_tab() {
        let mut app = App::new(None);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Help);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Stats);
    }

    #[test]
    fn test_handle_key_quit() {
        let mut app = App::new(None);
        assert!(app.handle_key(KeyCode::Char('q')));
        let mut app2 = App::new(None);
        assert!(app2.handle_key(KeyCode::Esc));
    }

    #[test]
    fn test_handle_key_tab_switch() {
        let mut app = App::new(None);
        assert!(!app.handle_key(KeyCode::Char('2')));
        assert_eq!(app.current_tab, Tab::Tensors);
        assert!(!app.handle_key(KeyCode::Char('3')));
        assert_eq!(app.current_tab, Tab::Stats);
    }

    #[test]
    fn test_truncate_name() {
        assert_eq!(truncate_name("short", 10), "short");
        assert_eq!(truncate_name("very_long_tensor_name", 10), "very_lo...");
        assert_eq!(truncate_name("ab", 2), "ab");
    }

    #[test]
    fn test_build_sparkline() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 100,
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            std: 0.5,
            zero_count: 0,
            nan_count: 0,
            inf_count: 0,
        };
        let spark = build_sparkline(&stats, 20);
        assert_eq!(spark.chars().count(), 20);
    }

    #[test]
    fn test_select_navigation_empty() {
        let mut app = App::new(None);
        app.select_next();
        app.select_prev();
        assert_eq!(app.selected, 0);
    }

    // ── PMAT-125 B4: additional coverage ──────────────────────────────────

    #[test]
    fn test_tab_all_titles_distinct() {
        let titles: Vec<&str> = Tab::ALL.iter().map(|t| t.title()).collect();
        assert_eq!(titles, [" Overview ", " Tensors ", " Stats ", " Help "]);
    }

    #[test]
    fn test_tab_from_index_wraps_modulo_four() {
        // from_index uses i % 4, so out-of-range indices wrap.
        assert_eq!(Tab::from_index(4), Tab::Overview);
        assert_eq!(Tab::from_index(5), Tab::Tensors);
        assert_eq!(Tab::from_index(6), Tab::Stats);
        assert_eq!(Tab::from_index(7), Tab::Help);
    }

    #[test]
    fn test_truncate_name_tiny_max_len() {
        // max_len < 4 → raw char-take, no ellipsis.
        assert_eq!(truncate_name("abcdef", 3), "abc");
        assert_eq!(truncate_name("abcdef", 1), "a");
        assert_eq!(truncate_name("abcdef", 0), "");
    }

    #[test]
    fn test_truncate_name_exact_boundary() {
        // name.len() == max_len → unchanged.
        assert_eq!(truncate_name("exactly10!", 10), "exactly10!");
    }

    #[test]
    fn test_truncate_name_multibyte_char_boundary() {
        // Multi-byte chars must not be split mid-codepoint.
        let name = "αβγδεζηθικλμν"; // each Greek letter is 2 bytes
        let out = truncate_name(name, 8);
        assert!(out.ends_with("..."));
        // Result must be valid UTF-8 (no panic on slicing).
        assert!(out.chars().count() > 0);
    }

    #[test]
    fn test_build_sparkline_zero_width() {
        let stats = sample_stats();
        assert_eq!(build_sparkline(&stats, 0), "");
    }

    #[test]
    fn test_build_sparkline_is_bell_shaped() {
        let stats = sample_stats();
        let spark = build_sparkline(&stats, 21);
        let chars: Vec<char> = spark.chars().collect();
        assert_eq!(chars.len(), 21);
        // Gaussian peaks in the middle: center char should be the tallest block.
        let center = chars[10];
        assert_eq!(center, '█');
        // Edges are the shortest.
        assert!(chars[0] < center);
        assert!(chars[20] < center);
    }

    #[test]
    fn test_text_style_sets_color_default_weight() {
        let c = Color {
            r: 0.5,
            g: 0.6,
            b: 0.7,
            a: 1.0,
        };
        let s = text_style(c);
        assert_eq!(s.color.r, 0.5);
        assert_eq!(s.weight, presentar_core::FontWeight::Normal);
    }

    #[test]
    fn test_bold_style_sets_bold_weight() {
        let c = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let s = bold_style(c);
        assert_eq!(s.color.r, 1.0);
        assert_eq!(s.weight, presentar_core::FontWeight::Bold);
    }

    #[test]
    fn test_selected_tensor_none_when_empty() {
        let app = App::new(None);
        assert!(app.selected_tensor().is_none());
    }

    #[test]
    fn test_handle_key_home_end_and_help() {
        let mut app = App::new(None);
        // Help via '?' and '4'.
        assert!(!app.handle_key(KeyCode::Char('?')));
        assert_eq!(app.current_tab, Tab::Help);
        assert!(!app.handle_key(KeyCode::Char('1')));
        assert_eq!(app.current_tab, Tab::Overview);
        // Home/End on an empty tensor list keep selection sane.
        assert!(!app.handle_key(KeyCode::Home));
        assert_eq!(app.selected, 0);
        assert!(!app.handle_key(KeyCode::End));
        assert_eq!(app.selected, 0); // saturating_sub on empty list
    }

    #[test]
    fn test_handle_key_unmapped_is_noop() {
        let mut app = App::new(None);
        let before = app.current_tab;
        assert!(!app.handle_key(KeyCode::Char('z')));
        assert_eq!(app.current_tab, before);
    }

    #[test]
    fn test_handle_key_tab_and_backtab() {
        let mut app = App::new(None);
        assert!(!app.handle_key(KeyCode::Tab));
        assert_eq!(app.current_tab, Tab::Tensors);
        assert!(!app.handle_key(KeyCode::BackTab));
        assert_eq!(app.current_tab, Tab::Overview);
    }

    fn sample_stats() -> TensorStats {
        TensorStats {
            name: "w".to_string(),
            count: 100,
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            std: 0.5,
            zero_count: 0,
            nan_count: 0,
            inf_count: 0,
        }
    }
}
