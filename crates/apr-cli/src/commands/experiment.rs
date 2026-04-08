//! Interactive experiment browser (ALB-024)
//!
//! Opens a ratatui TUI for browsing SQLite experiment data:
//! - Table of experiments and runs (navigable with arrow keys)
//! - Sparkline + braille loss curve for the selected run
//! - Hyperparameter display
//!
//! ```bash
//! apr experiment view --global
//! apr experiment view --db ./checkpoints/.entrenar/experiments.db
//! apr experiment view --global --json  # Non-interactive JSON dump
//! ```

use crate::CliError;
use entrenar::storage::sqlite::SqliteBackend;
use entrenar::storage::{ExperimentStorage, MetricPoint};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, CliError>;

/// Interactive experiment browser — ratatui TUI with loss curves.
pub(crate) fn experiment_view(db: &Option<PathBuf>, global: bool, json: bool) -> Result<()> {
    let store = open_store(db, global)?;
    let experiments = store
        .list_experiments()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to list experiments: {e}")))?;

    if experiments.is_empty() {
        if json {
            println!("[]");
        } else {
            println!("No experiments found.");
            println!("Run training with `apr train apply` to populate the experiment database.");
        }
        return Ok(());
    }

    // Collect all runs with their metrics
    let mut all_runs = Vec::new();
    for exp in &experiments {
        let runs = store.list_runs(&exp.id).unwrap_or_default();
        for run in runs {
            let loss = store.get_metrics(&run.id, "loss").unwrap_or_default();
            let params = store.get_params(&run.id).unwrap_or_default();
            all_runs.push(RunEntry {
                experiment_name: exp.name.clone(),
                run,
                loss_metrics: loss,
                params,
            });
        }
    }

    if json {
        print_json(&all_runs);
        return Ok(());
    }

    // Interactive TUI via ratatui
    run_tui_browser(&all_runs)
}

/// Single run with associated data for the browser.
struct RunEntry {
    experiment_name: String,
    run: entrenar::storage::Run,
    loss_metrics: Vec<MetricPoint>,
    params: std::collections::HashMap<String, entrenar::storage::ParameterValue>,
}

/// JSON output for non-interactive mode.
fn print_json(runs: &[RunEntry]) {
    let items: Vec<serde_json::Value> = runs
        .iter()
        .map(|r| {
            let loss_values: Vec<f64> = r.loss_metrics.iter().map(|p| p.value).collect();
            let final_loss = loss_values.last().copied();
            serde_json::json!({
                "experiment": r.experiment_name,
                "run_id": r.run.id,
                "status": format!("{:?}", r.run.status),
                "start_time": r.run.start_time.to_rfc3339(),
                "end_time": r.run.end_time.map(|t| t.to_rfc3339()),
                "final_loss": final_loss,
                "num_steps": loss_values.len(),
                "loss_values": loss_values,
                "params": param_map_json(&r.params),
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&items).unwrap_or_default()
    );
}

/// Convert params to clean JSON (unwrap tagged enum).
fn param_map_json(
    params: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in params {
        map.insert(k.clone(), param_to_json(v));
    }
    serde_json::Value::Object(map)
}

/// Unwrap ParameterValue tagged enum to clean JSON.
fn param_to_json(v: &entrenar::storage::ParameterValue) -> serde_json::Value {
    use entrenar::storage::ParameterValue;
    match v {
        ParameterValue::String(s) => serde_json::Value::String(s.clone()),
        ParameterValue::Int(i) => serde_json::json!(i),
        ParameterValue::Float(f) => serde_json::json!(f),
        ParameterValue::Bool(b) => serde_json::json!(b),
        ParameterValue::List(l) => serde_json::Value::Array(l.iter().map(param_to_json).collect()),
        ParameterValue::Dict(d) => {
            let mut map = serde_json::Map::new();
            for (k, v) in d {
                map.insert(k.clone(), param_to_json(v));
            }
            serde_json::Value::Object(map)
        }
    }
}

/// Open SQLite experiment store.
fn open_store(db: &Option<PathBuf>, global: bool) -> Result<SqliteBackend> {
    let db_path = if let Some(p) = db {
        p.clone()
    } else if global {
        dirs::home_dir()
            .map(|h| h.join(".entrenar").join("experiments.db"))
            .ok_or_else(|| {
                CliError::ValidationFailed("Could not determine home directory".into())
            })?
    } else {
        Path::new(".").join(".entrenar").join("experiments.db")
    };

    if !db_path.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Database not found: {}. Run training first or use --global.",
            db_path.display()
        )));
    }

    SqliteBackend::open(db_path.to_string_lossy().as_ref())
        .map_err(|e| CliError::ValidationFailed(format!("Failed to open database: {e}")))
}

// =============================================================================
// Ratatui TUI browser
// =============================================================================

#[cfg(feature = "ratatui")]
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
#[cfg(feature = "ratatui")]
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline as RatatuiSparkline},
    Frame, Terminal,
};
#[cfg(feature = "ratatui")]
use std::io;

#[cfg(not(feature = "ratatui"))]
fn run_tui_browser(runs: &[RunEntry]) -> Result<()> {
    use crossterm::{cursor, event::{self, Event, KeyCode, KeyEventKind}, execute, terminal::{self, ClearType}};
    use presentar_core::{Canvas, Color, Point, Rect, TextStyle, FontWeight};
    use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
    use presentar_terminal::ColorMode;
    use std::io::Write;
    use std::time::{Duration, Instant};

    const CYAN: Color = Color { r: 0.4, g: 0.85, b: 1.0, a: 1.0 };
    const WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    const DIM: Color = Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
    const YELLOW: Color = Color { r: 1.0, g: 0.85, b: 0.3, a: 1.0 };
    const GREEN: Color = Color { r: 0.3, g: 1.0, b: 0.5, a: 1.0 };
    const HEADER_BG: Color = Color { r: 0.1, g: 0.12, b: 0.18, a: 1.0 };
    const SELECTED_BG: Color = Color { r: 0.15, g: 0.2, b: 0.3, a: 1.0 };

    fn ts(color: Color) -> TextStyle { TextStyle { color, ..TextStyle::default() } }
    fn bold(color: Color) -> TextStyle { TextStyle { color, weight: FontWeight::Bold, ..TextStyle::default() } }

    if runs.is_empty() {
        eprintln!("No experiments found.");
        return Ok(());
    }

    // Build table data
    struct Row { name: String, run_id: String, status: String, loss: f64, steps: usize, loss_values: Vec<f64> }
    let rows: Vec<Row> = runs.iter().map(|r| {
        let loss_vals: Vec<f64> = r.loss_metrics.iter().map(|m| m.value).collect();
        let final_loss = loss_vals.last().copied().unwrap_or(0.0);
        Row {
            name: r.experiment_name.clone(),
            run_id: r.run.id.clone(),
            status: format!("{:?}", r.run.status),
            loss: final_loss,
            steps: loss_vals.len(),
            loss_values: loss_vals,
        }
    }).collect();

    let mut selected = 0_usize;
    let mut stdout = std::io::stdout();

    terminal::enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Raw mode: {e}")))?;
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide, terminal::Clear(ClearType::All))
        .map_err(|e| CliError::ValidationFailed(format!("Terminal setup: {e}")))?;

    let color_mode = ColorMode::detect();
    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    let mut force_full = true;

    loop {
        if event::poll(Duration::from_millis(50))
            .map_err(|e| CliError::ValidationFailed(format!("Poll: {e}")))? {
            match event::read().map_err(|e| CliError::ValidationFailed(format!("Read: {e}")))? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        terminal::disable_raw_mode().ok();
                        execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show).ok();
                        return Ok(());
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if selected + 1 < rows.len() { selected += 1; }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if selected > 0 { selected -= 1; }
                    }
                    KeyCode::Home => selected = 0,
                    KeyCode::End => selected = rows.len().saturating_sub(1),
                    _ => {}
                },
                Event::Resize(_, _) => force_full = true,
                _ => {}
            }
        }

        let (width, height) = terminal::size()
            .map_err(|e| CliError::ValidationFailed(format!("Size: {e}")))?;
        let mut buffer = CellBuffer::new(width, height);
        let w = width as f32;
        let h = height as f32;

        if w >= 40.0 && h >= 10.0 {
            let mut c = DirectTerminalCanvas::new(&mut buffer);

            // Header
            c.fill_rect(Rect::new(0.0, 0.0, w, 1.0), HEADER_BG);
            c.draw_text(" Experiment Browser", Point::new(0.0, 0.0), &bold(WHITE));
            c.draw_text(&format!("{} runs ", rows.len()), Point::new(w - 12.0, 0.0), &ts(DIM));

            // Two-column: table (60%) + detail (40%)
            let split = (w * 0.6).floor();

            // Left: run table
            let left = Rect::new(0.0, 1.0, split, h - 2.0);
            c.stroke_rect(left, CYAN, 1.0);
            c.draw_text(" Runs (j/k) ", Point::new(2.0, 1.0), &bold(WHITE));

            // Table header
            c.draw_text("Experiment", Point::new(2.0, 2.0), &bold(CYAN));
            c.draw_text("Loss", Point::new(split * 0.5, 2.0), &bold(CYAN));
            c.draw_text("Steps", Point::new(split * 0.7, 2.0), &bold(CYAN));

            let visible = (left.height as usize).saturating_sub(4);
            let scroll = if selected >= visible { selected - visible + 1 } else { 0 };
            for (i, row) in rows.iter().skip(scroll).take(visible).enumerate() {
                let y = 3.0 + i as f32;
                let is_sel = scroll + i == selected;
                if is_sel {
                    c.fill_rect(Rect::new(1.0, y, split - 2.0, 1.0), SELECTED_BG);
                }
                let fg = if is_sel { WHITE } else { DIM };
                c.draw_text(&truncate(&row.name, (split * 0.45) as usize), Point::new(2.0, y), &ts(fg));
                c.draw_text(&format!("{:.4}", row.loss), Point::new(split * 0.5, y), &ts(fg));
                c.draw_text(&format!("{}", row.steps), Point::new(split * 0.7, y), &ts(fg));
            }

            // Right: detail + sparkline
            let right = Rect::new(split, 1.0, w - split, h - 2.0);
            c.stroke_rect(right, CYAN, 1.0);
            c.draw_text(" Detail ", Point::new(split + 2.0, 1.0), &bold(WHITE));

            if let Some(row) = rows.get(selected) {
                let dx = split + 2.0;
                c.draw_text("Experiment:", Point::new(dx, 2.0), &ts(CYAN));
                c.draw_text(&row.name, Point::new(dx + 14.0, 2.0), &ts(WHITE));
                c.draw_text("Run ID:", Point::new(dx, 3.0), &ts(CYAN));
                c.draw_text(&row.run_id, Point::new(dx + 14.0, 3.0), &ts(WHITE));
                c.draw_text("Status:", Point::new(dx, 4.0), &ts(CYAN));
                c.draw_text(&row.status, Point::new(dx + 14.0, 4.0), &ts(GREEN));
                c.draw_text("Final Loss:", Point::new(dx, 5.0), &ts(CYAN));
                c.draw_text(&format!("{:.6}", row.loss), Point::new(dx + 14.0, 5.0), &ts(WHITE));

                // Loss sparkline
                if !row.loss_values.is_empty() {
                    c.draw_text("Loss Curve:", Point::new(dx, 7.0), &ts(CYAN));
                    let spark_w = ((w - split) as usize).saturating_sub(6);
                    let spark = render_braille(&row.loss_values, spark_w, 3);
                    for (si, line) in spark.iter().enumerate() {
                        c.draw_text(line, Point::new(dx + 1.0, 8.0 + si as f32), &ts(YELLOW));
                    }
                }
            }

            // Footer
            c.fill_rect(Rect::new(0.0, h - 1.0, w, 1.0), HEADER_BG);
            c.draw_text(" j/k:navigate  q:quit", Point::new(0.0, h - 1.0), &ts(DIM));
        }

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
    }
}

/// Pre-computed row data for the browser table.
#[cfg(feature = "ratatui")]
struct TableRow {
    experiment: String,
    run_id: String,
    status: String,
    final_loss: f64,
    steps: usize,
    loss_values: Vec<f64>,
}

/// TUI application state.
#[cfg(feature = "ratatui")]
struct BrowserApp {
    rows: Vec<TableRow>,
    selected: usize,
    should_quit: bool,
}

#[cfg(feature = "ratatui")]
impl BrowserApp {
    fn new(rows: Vec<TableRow>) -> Self {
        Self {
            rows,
            selected: 0,
            should_quit: false,
        }
    }

    fn next(&mut self) {
        if self.selected + 1 < self.rows.len() {
            self.selected += 1;
        }
    }

    fn previous(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }
}

/// Run the interactive TUI experiment browser.
#[cfg(feature = "ratatui")]
fn run_tui_browser(runs: &[RunEntry]) -> Result<()> {
    let table_rows: Vec<TableRow> = runs
        .iter()
        .map(|r| {
            let loss_values: Vec<f64> = r.loss_metrics.iter().map(|p| p.value).collect();
            let final_loss = loss_values.last().copied().unwrap_or(f64::NAN);
            let status = format!("{:?}", r.run.status);
            let steps = loss_values.len();
            TableRow {
                experiment: r.experiment_name.clone(),
                run_id: r.run.id.clone(),
                status,
                final_loss,
                steps,
                loss_values,
            }
        })
        .collect();

    if table_rows.is_empty() {
        println!("No runs found.");
        return Ok(());
    }

    // Setup terminal
    enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to enable raw mode: {e}")))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to enter alt screen: {e}")))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create terminal: {e}")))?;

    let mut app = BrowserApp::new(table_rows);

    // Main loop
    let result = run_event_loop(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();

    result
}

/// Handle a key press event in the experiment browser.
#[cfg(feature = "ratatui")]
fn handle_browser_key(app: &mut BrowserApp, code: KeyCode) {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Up | KeyCode::Char('k') => app.previous(),
        KeyCode::Down | KeyCode::Char('j') => app.next(),
        KeyCode::Home => app.selected = 0,
        KeyCode::End => {
            app.selected = app.rows.len().saturating_sub(1);
        }
        _ => {}
    }
}

#[cfg(feature = "ratatui")]
fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut BrowserApp,
) -> Result<()> {
    loop {
        terminal
            .draw(|f| draw_ui(f, app))
            .map_err(|e| CliError::ValidationFailed(format!("Draw error: {e}")))?;

        if event::poll(std::time::Duration::from_millis(100))
            .map_err(|e| CliError::ValidationFailed(format!("Event poll error: {e}")))?
        {
            if let Event::Key(key) = event::read()
                .map_err(|e| CliError::ValidationFailed(format!("Event read error: {e}")))?
            {
                if key.kind == KeyEventKind::Press {
                    handle_browser_key(app, key.code);
                }
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

#[cfg(feature = "ratatui")]
fn draw_ui(f: &mut Frame, app: &BrowserApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // Run table
            Constraint::Length(8), // Loss sparkline + braille chart
            Constraint::Length(1), // Footer
        ])
        .split(f.area());

    draw_run_table(f, app, chunks[0]);
    draw_loss_panel(f, app, chunks[1]);
    draw_footer(f, chunks[2]);
}

#[cfg(feature = "ratatui")]
fn draw_run_table(f: &mut Frame, app: &BrowserApp, area: ratatui::layout::Rect) {
    let header_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let selected_style = Style::default()
        .fg(Color::White)
        .bg(Color::DarkGray)
        .add_modifier(Modifier::BOLD);
    let normal_style = Style::default().fg(Color::Gray);

    let items: Vec<ListItem> = app
        .rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let loss_str = if row.final_loss.is_finite() {
                format!("{:.6}", row.final_loss)
            } else {
                "-".to_string()
            };
            let marker = if i == app.selected { ">" } else { " " };
            let line = format!(
                "{} {:<20} {:<16} {:>8} {:>10} {:>6}",
                marker,
                truncate(&row.experiment, 20),
                truncate(&row.run_id, 16),
                row.status,
                loss_str,
                row.steps
            );
            let style = if i == app.selected {
                selected_style
            } else {
                normal_style
            };
            ListItem::new(Line::from(Span::styled(line, style)))
        })
        .collect();

    let header = Line::from(Span::styled(
        format!(
            "  {:<20} {:<16} {:>8} {:>10} {:>6}",
            "EXPERIMENT", "RUN", "STATUS", "LOSS", "STEPS"
        ),
        header_style,
    ));

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Experiment Browser ")
        .title_style(Style::default().fg(Color::Cyan));

    // Render header + list
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height > 1 {
        let header_area = ratatui::layout::Rect { height: 1, ..inner };
        let list_area = ratatui::layout::Rect {
            y: inner.y + 1,
            height: inner.height.saturating_sub(1),
            ..inner
        };

        f.render_widget(Paragraph::new(header), header_area);

        // Scroll to keep selected visible
        let visible = list_area.height as usize;
        let offset = app.selected.saturating_sub(visible.saturating_sub(1));
        let visible_items: Vec<ListItem> = items.into_iter().skip(offset).take(visible).collect();
        f.render_widget(List::new(visible_items), list_area);
    }
}

#[cfg(feature = "ratatui")]
fn draw_loss_panel(f: &mut Frame, app: &BrowserApp, area: ratatui::layout::Rect) {
    let row = &app.rows[app.selected];

    if row.loss_values.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Loss History ")
            .title_style(Style::default().fg(Color::Cyan));
        let msg = Paragraph::new("No loss data for this run.")
            .style(Style::default().fg(Color::DarkGray))
            .block(block);
        f.render_widget(msg, area);
        return;
    }

    let first = row.loss_values.first().copied().unwrap_or(0.0);
    let last = row.loss_values.last().copied().unwrap_or(0.0);
    let title = format!(
        " Loss: {} - {} ({} steps) [{:.4} -> {:.4}] ",
        truncate(&row.experiment, 15),
        truncate(&row.run_id, 12),
        row.steps,
        first,
        last
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    // Use ratatui Sparkline for loss data
    // Ratatui sparkline takes u64, so normalize to 0..1000 range
    let min = row
        .loss_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max = row
        .loss_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);

    let spark_data: Vec<u64> = row
        .loss_values
        .iter()
        .map(|v| {
            let norm = ((v - min) / range).clamp(0.0, 1.0);
            (norm * 1000.0) as u64
        })
        .collect();

    // Split inner: sparkline (1 line) + braille (rest)
    let spark_area = ratatui::layout::Rect { height: 1, ..inner };
    let braille_area = ratatui::layout::Rect {
        y: inner.y + 1,
        height: inner.height.saturating_sub(1),
        ..inner
    };

    let sparkline = RatatuiSparkline::default()
        .data(&spark_data)
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, spark_area);

    // Braille chart for remaining space
    if braille_area.height > 0 {
        let chart_lines = render_braille(
            &row.loss_values,
            braille_area.width as usize,
            braille_area.height as usize,
        );
        let braille_text: Vec<Line> = chart_lines
            .into_iter()
            .map(|l| Line::from(Span::styled(l, Style::default().fg(Color::LightBlue))))
            .collect();
        f.render_widget(Paragraph::new(braille_text), braille_area);
    }
}

#[cfg(feature = "ratatui")]
fn draw_footer(f: &mut Frame, area: ratatui::layout::Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(
            " ^/v ",
            Style::default().fg(Color::White).bg(Color::DarkGray),
        ),
        Span::styled(" Navigate  ", Style::default().fg(Color::Gray)),
        Span::styled(" q ", Style::default().fg(Color::White).bg(Color::DarkGray)),
        Span::styled(" Quit  ", Style::default().fg(Color::Gray)),
        Span::styled(
            " Home/End ",
            Style::default().fg(Color::White).bg(Color::DarkGray),
        ),
        Span::styled(" Jump", Style::default().fg(Color::Gray)),
    ]));
    f.render_widget(footer, area);
}

// =============================================================================
// Braille chart renderer
// =============================================================================

/// Render braille chart from data.
fn render_braille(data: &[f64], width: usize, height: usize) -> Vec<String> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![];
    }

    let grid = build_braille_grid(data, width, height);
    let total_dots_h = height * 4;
    let num_points = width * 2;

    (0..height)
        .map(|row| {
            (0..width)
                .map(|col| encode_braille_cell(&grid, col * 2, row * 4, num_points, total_dots_h))
                .collect()
        })
        .collect()
}

/// Build a boolean dot grid from normalized data points.
fn build_braille_grid(data: &[f64], width: usize, height: usize) -> Vec<Vec<bool>> {
    if width == 0 || height == 0 || data.is_empty() {
        return vec![];
    }
    let total_dots_h = height * 4;
    let num_points = width * 2;
    let step = data.len() as f64 / num_points as f64;

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(0.001);

    let mut grid = vec![vec![false; num_points]; total_dots_h];

    for x in 0..num_points.min(data.len()) {
        let idx = if data.len() > num_points {
            (x as f64 * step) as usize
        } else {
            x
        };
        if idx >= data.len() {
            break;
        }
        let norm = ((data[idx] - min) / range).clamp(0.0, 1.0);
        let y = ((1.0 - norm) * (total_dots_h - 1) as f64) as usize;
        grid[y.min(total_dots_h - 1)][x] = true;
    }

    grid
}

/// Encode a 2x4 braille cell from the dot grid.
fn encode_braille_cell(
    grid: &[Vec<bool>],
    x: usize,
    y: usize,
    num_points: usize,
    total_dots_h: usize,
) -> char {
    // Braille dot positions: left column (1,2,3,7), right column (4,5,6,8)
    // Mapped to Unicode offset bits
    const DOT_MAP: [(usize, usize, u32); 8] = [
        (0, 0, 0x01), // dot 1: row+0, col+0
        (1, 0, 0x02), // dot 2: row+1, col+0
        (2, 0, 0x04), // dot 3: row+2, col+0
        (3, 0, 0x40), // dot 7: row+3, col+0
        (0, 1, 0x08), // dot 4: row+0, col+1
        (1, 1, 0x10), // dot 5: row+1, col+1
        (2, 1, 0x20), // dot 6: row+2, col+1
        (3, 1, 0x80), // dot 8: row+3, col+1
    ];

    let mut code: u32 = 0x2800;
    for &(dy, dx, bit) in &DOT_MAP {
        let gy = y + dy;
        let gx = x + dx;
        if gy < total_dots_h && gx < num_points && grid[gy][gx] {
            code |= bit;
        }
    }
    char::from_u32(code).unwrap_or(' ')
}

/// Truncate string.
fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}
