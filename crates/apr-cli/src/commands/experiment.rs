//! Interactive experiment browser (ALB-024)
//!
//! Opens a presentar-terminal TUI for browsing SQLite experiment data:
//! - Table of experiments and runs (navigable with arrow keys)
//! - Braille loss curve for the selected run
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

/// Interactive experiment browser — presentar-terminal TUI with loss curves.
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

    // Interactive TUI
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
// Presentar-terminal TUI browser
// =============================================================================

fn run_tui_browser(runs: &[RunEntry]) -> Result<()> {
    use crossterm::{
        cursor,
        event::{self, Event, KeyCode, KeyEventKind},
        execute,
        terminal::{self, ClearType},
    };
    use presentar_core::{Canvas, Color, FontWeight, Point, Rect, TextStyle};
    use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
    use presentar_terminal::ColorMode;
    use std::io::Write;
    use std::time::Duration;

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
    const HEADER_BG: Color = Color {
        r: 0.1,
        g: 0.12,
        b: 0.18,
        a: 1.0,
    };
    const SELECTED_BG: Color = Color {
        r: 0.15,
        g: 0.2,
        b: 0.3,
        a: 1.0,
    };

    fn ts(color: Color) -> TextStyle {
        TextStyle {
            color,
            ..TextStyle::default()
        }
    }
    fn bold(color: Color) -> TextStyle {
        TextStyle {
            color,
            weight: FontWeight::Bold,
            ..TextStyle::default()
        }
    }

    if runs.is_empty() {
        eprintln!("No experiments found.");
        return Ok(());
    }

    // Build table data
    struct Row {
        name: String,
        run_id: String,
        status: String,
        loss: f64,
        steps: usize,
        loss_values: Vec<f64>,
    }
    let rows: Vec<Row> = runs
        .iter()
        .map(|r| {
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
        })
        .collect();

    let mut selected = 0_usize;
    let mut stdout = std::io::stdout();

    terminal::enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Raw mode: {e}")))?;
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        cursor::Hide,
        terminal::Clear(ClearType::All)
    )
    .map_err(|e| CliError::ValidationFailed(format!("Terminal setup: {e}")))?;

    let color_mode = ColorMode::detect();
    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    let mut force_full = true;

    loop {
        if event::poll(Duration::from_millis(50))
            .map_err(|e| CliError::ValidationFailed(format!("Poll: {e}")))?
        {
            match event::read().map_err(|e| CliError::ValidationFailed(format!("Read: {e}")))? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        terminal::disable_raw_mode().ok();
                        execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show).ok();
                        return Ok(());
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if selected + 1 < rows.len() {
                            selected += 1;
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if selected > 0 {
                            selected -= 1;
                        }
                    }
                    KeyCode::Home => selected = 0,
                    KeyCode::End => selected = rows.len().saturating_sub(1),
                    _ => {}
                },
                Event::Resize(_, _) => force_full = true,
                _ => {}
            }
        }

        let (width, height) =
            terminal::size().map_err(|e| CliError::ValidationFailed(format!("Size: {e}")))?;
        let mut buffer = CellBuffer::new(width, height);
        let w = width as f32;
        let h = height as f32;

        if w >= 40.0 && h >= 10.0 {
            let mut c = DirectTerminalCanvas::new(&mut buffer);

            // Header
            c.fill_rect(Rect::new(0.0, 0.0, w, 1.0), HEADER_BG);
            c.draw_text(" Experiment Browser", Point::new(0.0, 0.0), &bold(WHITE));
            c.draw_text(
                &format!("{} runs ", rows.len()),
                Point::new(w - 12.0, 0.0),
                &ts(DIM),
            );

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
            let scroll = if selected >= visible {
                selected - visible + 1
            } else {
                0
            };
            for (i, row) in rows.iter().skip(scroll).take(visible).enumerate() {
                let y = 3.0 + i as f32;
                let is_sel = scroll + i == selected;
                if is_sel {
                    c.fill_rect(Rect::new(1.0, y, split - 2.0, 1.0), SELECTED_BG);
                }
                let fg = if is_sel { WHITE } else { DIM };
                c.draw_text(
                    &truncate(&row.name, (split * 0.45) as usize),
                    Point::new(2.0, y),
                    &ts(fg),
                );
                c.draw_text(
                    &format!("{:.4}", row.loss),
                    Point::new(split * 0.5, y),
                    &ts(fg),
                );
                c.draw_text(
                    &format!("{}", row.steps),
                    Point::new(split * 0.7, y),
                    &ts(fg),
                );
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
                c.draw_text(
                    &format!("{:.6}", row.loss),
                    Point::new(dx + 14.0, 5.0),
                    &ts(WHITE),
                );

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

// =============================================================================
// PMAT-125 B4: pure-helper coverage
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use entrenar::storage::ParameterValue;
    use std::collections::HashMap;

    #[test]
    fn test_truncate_shorter_unchanged() {
        assert_eq!(truncate("hi", 10), "hi");
        assert_eq!(truncate("exact", 5), "exact");
    }

    #[test]
    fn test_truncate_longer_clipped() {
        assert_eq!(truncate("hello world", 5), "hello");
        assert_eq!(truncate("abcdef", 0), "");
    }

    #[test]
    fn test_param_to_json_scalars() {
        assert_eq!(
            param_to_json(&ParameterValue::String("adam".into())),
            serde_json::json!("adam")
        );
        assert_eq!(param_to_json(&ParameterValue::Int(7)), serde_json::json!(7));
        assert_eq!(
            param_to_json(&ParameterValue::Float(0.5)),
            serde_json::json!(0.5)
        );
        assert_eq!(
            param_to_json(&ParameterValue::Bool(true)),
            serde_json::json!(true)
        );
    }

    #[test]
    fn test_param_to_json_nested() {
        let list = ParameterValue::List(vec![
            ParameterValue::Int(1),
            ParameterValue::String("x".into()),
        ]);
        assert_eq!(param_to_json(&list), serde_json::json!([1, "x"]));

        let mut d = HashMap::new();
        d.insert("lr".to_string(), ParameterValue::Float(0.01));
        let dict = ParameterValue::Dict(d);
        assert_eq!(param_to_json(&dict), serde_json::json!({"lr": 0.01}));
    }

    #[test]
    fn test_param_map_json_unwraps_all_keys() {
        let mut params = HashMap::new();
        params.insert("opt".to_string(), ParameterValue::String("adam".into()));
        params.insert("epochs".to_string(), ParameterValue::Int(3));
        let json = param_map_json(&params);
        assert_eq!(json["opt"], serde_json::json!("adam"));
        assert_eq!(json["epochs"], serde_json::json!(3));
    }

    #[test]
    fn test_param_map_json_empty() {
        let params = HashMap::new();
        let json = param_map_json(&params);
        assert_eq!(json, serde_json::json!({}));
    }

    #[test]
    fn test_render_braille_empty_inputs() {
        assert!(render_braille(&[], 10, 4).is_empty());
        assert!(render_braille(&[1.0, 2.0], 0, 4).is_empty());
        assert!(render_braille(&[1.0, 2.0], 10, 0).is_empty());
    }

    #[test]
    fn test_render_braille_dimensions() {
        let data: Vec<f64> = (0..40).map(|i| (i as f64).sin()).collect();
        let rows = render_braille(&data, 16, 4);
        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert_eq!(row.chars().count(), 16);
            assert!(row.chars().all(|c| (0x2800..=0x28FF).contains(&(c as u32))));
        }
    }

    #[test]
    fn test_build_braille_grid_shape() {
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let grid = build_braille_grid(&data, 8, 4);
        assert_eq!(grid.len(), 16);
        assert!(grid.iter().all(|r| r.len() == 16));
        assert!(grid.iter().any(|r| r.iter().any(|&b| b)));
    }

    #[test]
    fn test_build_braille_grid_empty() {
        assert!(build_braille_grid(&[], 8, 4).is_empty());
        assert!(build_braille_grid(&[1.0], 0, 4).is_empty());
    }

    #[test]
    fn test_build_braille_grid_constant_data_no_div_by_zero() {
        let data = vec![5.0; 10];
        let grid = build_braille_grid(&data, 8, 4);
        assert_eq!(grid.len(), 16);
    }

    #[test]
    fn test_encode_braille_cell_all_unset_is_base() {
        let grid = vec![vec![false; 4]; 8];
        let ch = encode_braille_cell(&grid, 0, 0, 4, 8);
        assert_eq!(ch, '\u{2800}');
    }

    #[test]
    fn test_encode_braille_cell_top_left_dot() {
        let mut grid = vec![vec![false; 4]; 8];
        grid[0][0] = true; // dot 1 → +0x01
        let ch = encode_braille_cell(&grid, 0, 0, 4, 8);
        assert_eq!(ch as u32, 0x2800 + 0x01);
    }

    #[test]
    fn test_encode_braille_cell_full_cell() {
        let mut grid = vec![vec![false; 2]; 4];
        for row in grid.iter_mut() {
            for cell in row.iter_mut() {
                *cell = true;
            }
        }
        let ch = encode_braille_cell(&grid, 0, 0, 2, 4);
        assert_eq!(ch as u32, 0x28FF);
    }
}
