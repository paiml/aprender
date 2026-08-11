
/// Check CI thresholds.
///
/// The report's own verdict is the primary gate: when cbtop has already
/// concluded `Status: FAIL | CI: red` (one or more bricks over budget), `--ci`
/// must fail even if the caller passed no explicit `--throughput` /
/// `--brick-score` number. Previously only the explicit numeric thresholds
/// could set `passed = false`, so `apr cbtop --ci` printed a red report and
/// exited 0 — a gate that could not fail.
fn check_ci_thresholds(report: &HeadlessReport, config: &CbtopConfig) -> bool {
    let mut passed = true;

    if report.ci_result != "green" {
        eprintln!(
            "cbtop: FAIL - report status {} (CI: {}), falsification {}/{} passed",
            report.status,
            report.ci_result,
            report.falsification.passed,
            report.falsification.total_points
        );
        passed = false;
    }

    if let Some(threshold) = config.throughput_threshold {
        if report.throughput.tokens_per_sec < threshold {
            eprintln!(
                "cbtop: FAIL - Throughput {:.1} tok/s < threshold {:.1} tok/s",
                report.throughput.tokens_per_sec, threshold
            );
            passed = false;
        } else {
            eprintln!(
                "cbtop: PASS - Throughput {:.1} tok/s >= threshold {:.1} tok/s",
                report.throughput.tokens_per_sec, threshold
            );
        }
    }

    if let Some(threshold) = config.brick_score_threshold {
        let avg_score = if report.brick_scores.is_empty() {
            0
        } else {
            report.brick_scores.iter().map(|b| b.score).sum::<u32>()
                / report.brick_scores.len() as u32
        };
        if avg_score < threshold {
            eprintln!(
                "cbtop: FAIL - Brick score {} < threshold {}",
                avg_score, threshold
            );
            passed = false;
        } else {
            eprintln!(
                "cbtop: PASS - Brick score {} >= threshold {}",
                avg_score, threshold
            );
        }
    }

    passed
}

/// Format report as JSON
fn format_report_as_json(report: &HeadlessReport) -> String {
    let brick_scores_json: String = report
        .brick_scores
        .iter()
        .map(|b| {
            format!(
                r#"    {{
      "name": "{}",
      "score": {},
      "grade": "{}",
      "budget_us": {:.2},
      "actual_us": {:.2},
      "gap_factor": {:.3}
    }}"#,
                b.name, b.score, b.grade, b.budget_us, b.actual_us, b.gap_factor
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        r#"{{
  "model": "{}",
  "timestamp": "{}",
  "hardware": {{
    "gpu": "{}",
    "cpu": "{}",
    "memory_gb": {}
  }},
  "throughput": {{
    "tokens_per_sec": {:.2},
    "ttft_ms": {:.2},
    "cv_percent": {:.2},
    "p50_us": {:.2},
    "p99_us": {:.2}
  }},
  "brick_scores": [
{}
  ],
  "pmat_scores": {{
    "rust_project_score": {:.1},
    "tdg_score": {:.1},
    "cuda_tdg_score": {:.1},
    "brick_score": {},
    "grade": "{}"
  }},
  "falsification": {{
    "total_points": {},
    "passed": {},
    "failed": {},
    "blocked": {}
  }},
  "status": "{}",
  "ci_result": "{}"
}}"#,
        report.model,
        report.timestamp,
        report.hardware.gpu,
        report.hardware.cpu,
        report.hardware.memory_gb,
        report.throughput.tokens_per_sec,
        report.throughput.ttft_ms,
        report.throughput.cv_percent,
        report.throughput.p50_us,
        report.throughput.p99_us,
        brick_scores_json,
        report.pmat_scores.rust_project_score,
        report.pmat_scores.tdg_score,
        report.pmat_scores.cuda_tdg_score,
        report.pmat_scores.brick_score,
        report.pmat_scores.grade,
        report.falsification.total_points,
        report.falsification.passed,
        report.falsification.failed,
        report.falsification.blocked,
        report.status,
        report.ci_result,
    )
}

/// Print report as plain text
fn print_report_text(report: &HeadlessReport) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  cbtop Headless Benchmark Report");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Model:     {}", report.model);
    println!("  Timestamp: {}", report.timestamp);
    println!();
    println!(
        "  Throughput: {:.1} tok/s",
        report.throughput.tokens_per_sec
    );
    println!("  TTFT:       {:.2} ms", report.throughput.ttft_ms);
    println!("  CV:         {:.2}%", report.throughput.cv_percent);
    println!();
    println!("  Brick Scores:");
    for brick in &report.brick_scores {
        let status = if brick.gap_factor <= 1.0 {
            "✅"
        } else {
            "❌"
        };
        println!(
            "    {} {:12} {:>3} ({}) - {:.1}µs / {:.1}µs ({:.2}x)",
            status,
            brick.name,
            brick.score,
            brick.grade,
            brick.actual_us,
            brick.budget_us,
            brick.gap_factor
        );
    }
    println!();
    println!(
        "  Falsification: {}/{} passed",
        report.falsification.passed, report.falsification.total_points
    );
    println!("  Status: {} | CI: {}", report.status, report.ci_result);
    println!("═══════════════════════════════════════════════════════════════");
}

// ── cbtop TUI (presentar-terminal) ───────────────────────────────────────

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, FontWeight};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};

const CBTOP_CYAN: Color = Color { r: 0.4, g: 0.85, b: 1.0, a: 1.0 };
const CBTOP_WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
const CBTOP_DIM: Color = Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
const CBTOP_GREEN: Color = Color { r: 0.3, g: 1.0, b: 0.5, a: 1.0 };
const CBTOP_YELLOW: Color = Color { r: 1.0, g: 0.85, b: 0.3, a: 1.0 };
const CBTOP_RED: Color = Color { r: 1.0, g: 0.3, b: 0.3, a: 1.0 };
const CBTOP_HEADER_BG: Color = Color { r: 0.1, g: 0.12, b: 0.18, a: 1.0 };
const CBTOP_SELECTED_BG: Color = Color { r: 0.15, g: 0.2, b: 0.3, a: 1.0 };

fn cbtop_ts(color: Color) -> TextStyle {
    TextStyle { color, ..TextStyle::default() }
}
fn cbtop_bold(color: Color) -> TextStyle {
    TextStyle { color, weight: FontWeight::Bold, ..TextStyle::default() }
}

/// Run TUI mode — presentar-terminal powered
fn run_tui(model: Option<&str>, attach: Option<&str>) -> Result<()> {
    use crossterm::{cursor, execute, terminal::{self, ClearType}};
    use presentar_terminal::ColorMode;
    use std::io::Write;
    use std::time::{Duration, Instant};

    if attach.is_some() {
        eprintln!("Warning: --attach is not yet implemented for TUI mode. Flag ignored.");
    }

    let mut app = App::new(model);
    let mut stdout = io::stdout();

    terminal::enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to enable raw mode: {e}")))?;
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide, terminal::Clear(ClearType::All))
        .map_err(|e| CliError::ValidationFailed(format!("Failed to setup terminal: {e}")))?;

    let color_mode = ColorMode::detect();
    let mut renderer = DiffRenderer::with_color_mode(color_mode);
    let render_interval = Duration::from_millis(100);
    let mut last_render = Instant::now();
    let mut force_full = true;

    loop {
        if event::poll(Duration::from_millis(50))
            .map_err(|e| CliError::ValidationFailed(format!("Event poll error: {e}")))?
        {
            match event::read()
                .map_err(|e| CliError::ValidationFailed(format!("Event read error: {e}")))?
            {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if handle_cbtop_key(&mut app, key.code) {
                        terminal::disable_raw_mode().ok();
                        execute!(stdout, terminal::LeaveAlternateScreen, cursor::Show).ok();
                        return Ok(());
                    }
                }
                Event::Resize(_, _) => force_full = true,
                _ => {}
            }
        }

        app.tick();

        if last_render.elapsed() >= render_interval {
            let (width, height) = terminal::size()
                .map_err(|e| CliError::ValidationFailed(format!("Terminal size: {e}")))?;
            let mut buffer = CellBuffer::new(width, height);
            draw_cbtop(&app, &mut buffer);

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

/// Handle cbtop key press. Returns true if should quit.
fn handle_cbtop_key(app: &mut App, code: KeyCode) -> bool {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => return true,
        KeyCode::Char('p') => app.current_view = View::Pipeline,
        KeyCode::Char('b') => app.current_view = View::Budget,
        KeyCode::Char('h') => app.current_view = View::Histogram,
        KeyCode::Char('g') => app.current_view = View::Gpu,
        KeyCode::Char('m') => app.current_view = View::Memory,
        KeyCode::Down | KeyCode::Char('j') => app.next_brick(),
        KeyCode::Up | KeyCode::Char('k') => app.prev_brick(),
        _ => {}
    }
    false
}

/// Draw full cbtop frame into buffer
fn draw_cbtop(app: &App, buffer: &mut CellBuffer) {
    let w = buffer.width() as f32;
    let h = buffer.height() as f32;
    if w < 20.0 || h < 8.0 { return; }

    let mut c = DirectTerminalCanvas::new(buffer);
    draw_cbtop_header(app, &mut c, w);
    draw_cbtop_tabs(app, &mut c, w);

    let body = Rect::new(0.0, 2.0, w, h - 3.0);
    match app.current_view {
        View::Pipeline => draw_cbtop_pipeline(app, &mut c, body),
        View::Budget => draw_cbtop_budget(app, &mut c, body),
        _ => draw_cbtop_stub(app, &mut c, body),
    }

    draw_cbtop_summary(app, &mut c, w, h - 2.0);
    draw_cbtop_footer(&mut c, w, h - 1.0);
}

fn draw_cbtop_header(app: &App, c: &mut DirectTerminalCanvas<'_>, w: f32) {
    c.fill_rect(Rect::new(0.0, 0.0, w, 1.0), CBTOP_HEADER_BG);
    let title = format!(" cbtop — {} │ Layer {}/{}", app.model_name, app.pipeline.layer_idx, app.pipeline.total_layers);
    c.draw_text(&title, Point::new(0.0, 0.0), &cbtop_bold(CBTOP_WHITE));
}

fn draw_cbtop_tabs(app: &App, c: &mut DirectTerminalCanvas<'_>, _w: f32) {
    let mut x = 1.0_f32;
    for (i, tab) in View::titles().iter().enumerate() {
        let is_active = i == app.current_view.index();
        if is_active {
            c.fill_rect(Rect::new(x, 1.0, tab.len() as f32 + 2.0, 1.0), CBTOP_CYAN);
            c.draw_text(&format!(" {tab} "), Point::new(x, 1.0), &cbtop_bold(Color { r: 0.0, g: 0.0, b: 0.1, a: 1.0 }));
        } else {
            c.draw_text(&format!(" {tab} "), Point::new(x, 1.0), &cbtop_ts(CBTOP_DIM));
        }
        x += tab.len() as f32 + 3.0;
    }
}

fn draw_cbtop_pipeline(app: &App, c: &mut DirectTerminalCanvas<'_>, body: Rect) {
    c.stroke_rect(body, CBTOP_CYAN, 1.0);
    c.draw_text(" Pipeline ", Point::new(2.0, body.y), &cbtop_bold(CBTOP_WHITE));
    for (i, brick) in app.pipeline.bricks.iter().enumerate() {
        let y = body.y + 1.0 + i as f32;
        if y >= body.y + body.height - 1.0 { break; }
        let is_sel = i == app.selected_brick;
        let gap = brick.gap_factor();
        let status_color = if gap <= 1.0 { CBTOP_GREEN } else if gap <= 1.5 { CBTOP_YELLOW } else { CBTOP_RED };

        if is_sel {
            c.fill_rect(Rect::new(body.x + 1.0, y, body.width - 2.0, 1.0), CBTOP_SELECTED_BG);
        }

        let bar_width = (gap.min(2.0) * 20.0) as usize;
        let bar: String = "█".repeat(bar_width.min(20));
        let line = format!(" {:12} {} {:.1}µs/{:.1}µs [{:<20}]", brick.name, brick.status(), brick.actual_us, brick.budget_us, bar);
        let fg = if is_sel { CBTOP_WHITE } else { CBTOP_DIM };
        c.draw_text(&line, Point::new(body.x + 1.0, y), &cbtop_ts(fg));
        c.draw_text(&format!(" {:.1}x", gap), Point::new(body.x + 50.0, y), &cbtop_bold(status_color));
    }
}

fn draw_cbtop_budget(app: &App, c: &mut DirectTerminalCanvas<'_>, body: Rect) {
    c.stroke_rect(body, CBTOP_CYAN, 1.0);
    c.draw_text(" Budget vs Actual ", Point::new(2.0, body.y), &cbtop_bold(CBTOP_WHITE));
    for (i, brick) in app.pipeline.bricks.iter().enumerate() {
        let y = body.y + 1.0 + i as f32;
        if y >= body.y + body.height - 1.0 { break; }
        let ratio = brick.actual_us / brick.budget_us;
        let bar_w = (ratio.min(2.0) * (body.width * 0.4) as f64) as usize;
        let bar_color = if ratio <= 1.0 { CBTOP_GREEN } else { CBTOP_RED };
        let bar: String = "█".repeat(bar_w.min(40));
        c.draw_text(&format!("{:12}", brick.name), Point::new(body.x + 2.0, y), &cbtop_ts(CBTOP_CYAN));
        c.draw_text(&bar, Point::new(body.x + 16.0, y), &cbtop_ts(bar_color));
        c.draw_text(&format!(" {:.1}x", ratio), Point::new(body.x + 16.0 + bar_w as f32 + 1.0, y), &cbtop_ts(CBTOP_WHITE));
    }
}

fn draw_cbtop_stub(app: &App, c: &mut DirectTerminalCanvas<'_>, body: Rect) {
    c.stroke_rect(body, CBTOP_CYAN, 1.0);
    let label = match app.current_view {
        View::Histogram => " Latency Histogram ",
        View::Gpu => " GPU Info ",
        View::Memory => " Memory ",
        _ => "",
    };
    c.draw_text(label, Point::new(2.0, body.y), &cbtop_bold(CBTOP_WHITE));
    c.draw_text("Panel rendering in progress (presentar-terminal migration)", Point::new(body.x + 4.0, body.y + 2.0), &cbtop_ts(CBTOP_DIM));
}

fn draw_cbtop_summary(app: &App, c: &mut DirectTerminalCanvas<'_>, w: f32, y: f32) {
    c.fill_rect(Rect::new(0.0, y, w, 1.0), CBTOP_HEADER_BG);
    let throughput = app.pipeline.current_tok_s;
    let thr_color = if throughput >= 100.0 { CBTOP_GREEN } else if throughput >= 30.0 { CBTOP_YELLOW } else { CBTOP_RED };
    c.draw_text(&format!(" {:.1} tok/s", throughput), Point::new(0.0, y), &cbtop_bold(thr_color));
    let avg_gap = if app.pipeline.bricks.is_empty() { 0.0 } else {
        app.pipeline.bricks.iter().map(|b| b.gap_factor()).sum::<f64>() / app.pipeline.bricks.len() as f64
    };
    c.draw_text(&format!("│ Bricks: {} │ Avg gap: {:.2}x", app.pipeline.bricks.len(), avg_gap), Point::new(16.0, y), &cbtop_ts(CBTOP_DIM));
}

fn draw_cbtop_footer(c: &mut DirectTerminalCanvas<'_>, w: f32, y: f32) {
    c.fill_rect(Rect::new(0.0, y, w, 1.0), CBTOP_HEADER_BG);
    c.draw_text(" p:pipeline b:budget h:hist g:gpu m:mem  j/k:nav  q:quit", Point::new(0.0, y), &cbtop_ts(CBTOP_DIM));
}
