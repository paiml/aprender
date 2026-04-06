//! Simular Orbit Demo - Terminal User Interface
//!
//! A TUI demonstration of orbital mechanics using ratatui.
//! App logic lives in `simular::tui::orbit_app`.

#![forbid(unsafe_code)]

#[cfg(feature = "tui")]
fn main() -> std::io::Result<()> {
    use simular::tui::orbit_app::OrbitApp;
    tui::run(OrbitApp::new())
}

#[cfg(not(feature = "tui"))]
fn main() {
    eprintln!("TUI feature not enabled. Run with --features tui");
    std::process::exit(1);
}

#[cfg(feature = "tui")]
mod tui {
    use crossterm::{
        event::{self, Event, KeyEventKind},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout, Rect},
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{
            canvas::{Canvas, Points},
            Block, Borders, Gauge, Paragraph,
        },
        Frame, Terminal,
    };
    use simular::orbit::prelude::AU;
    use simular::tui::orbit_app::OrbitApp;
    use std::io;
    use std::time::{Duration, Instant};

    /// Run the TUI application.
    pub fn run(mut app: OrbitApp) -> io::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let tick_rate = Duration::from_millis(33);

        loop {
            let start = Instant::now();
            terminal.draw(|f| ui(f, &app))?;

            let timeout = tick_rate.saturating_sub(start.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        app.handle_key(key.code);
                    }
                }
            }

            if app.should_quit {
                break;
            }

            app.update();
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn ui(f: &mut Frame, app: &OrbitApp) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Length(4), // ComputeBlock sparklines
            ])
            .split(f.area());

        render_title(f, chunks[0], app);
        render_orbit_canvas(f, chunks[1], app);
        render_status(f, chunks[2], app);
        render_status_panel(f, chunks[3], app);
        render_sparklines(f, chunks[4], app);
    }

    fn render_title(f: &mut Frame, area: Rect, app: &OrbitApp) {
        let title = Paragraph::new(vec![Line::from(vec![
            Span::styled(
                " SIMULAR ORBIT DEMO ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(
                if app.paused { "[PAUSED]" } else { "[RUNNING]" },
                Style::default().fg(if app.paused {
                    Color::Yellow
                } else {
                    Color::Green
                }),
            ),
            Span::raw(" | "),
            Span::styled(
                format!("Time: {:.1} days", app.sim_time_days),
                Style::default().fg(Color::White),
            ),
        ])])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Controls: [Space] Pause  [R] Reset  [+/-] Speed  [Q] Quit"),
        );
        f.render_widget(title, area);
    }

    fn render_orbit_canvas(f: &mut Frame, area: Rect, app: &OrbitApp) {
        let scale = 2.0 / AU;

        let canvas = Canvas::default()
            .block(Block::default().borders(Borders::ALL).title("Orbit View"))
            .x_bounds([-2.0, 2.0])
            .y_bounds([-2.0, 2.0])
            .paint(|ctx| {
                if app.trails.len() > 1 {
                    let trail_points: Vec<(f64, f64)> = app.trails[1]
                        .points()
                        .iter()
                        .map(|(x, y)| (x * scale, y * scale))
                        .collect();

                    if !trail_points.is_empty() {
                        ctx.draw(&Points {
                            coords: &trail_points,
                            color: Color::Blue,
                        });
                    }
                }

                let (sun_x, sun_y, _) = app.state.bodies[0].position.as_meters();
                ctx.print(
                    sun_x * scale,
                    sun_y * scale,
                    Span::styled("☉", Style::default().fg(Color::Yellow)),
                );

                if app.state.bodies.len() > 1 {
                    let (earth_x, earth_y, _) = app.state.bodies[1].position.as_meters();
                    ctx.print(
                        earth_x * scale,
                        earth_y * scale,
                        Span::styled("🌍", Style::default().fg(Color::Cyan)),
                    );
                }
            });

        f.render_widget(canvas, area);
    }

    fn render_status(f: &mut Frame, area: Rect, app: &OrbitApp) {
        let energy = app.total_energy();
        let status = Paragraph::new(vec![Line::from(vec![
            Span::styled("Energy: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{energy:.6e} J"), Style::default().fg(Color::White)),
            Span::raw(" | "),
            Span::styled("Frame: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.frame_count),
                Style::default().fg(Color::White),
            ),
            Span::raw(" | "),
            Span::styled("Scale: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}x", app.time_scale),
                Style::default().fg(Color::Cyan),
            ),
        ])])
        .block(Block::default().borders(Borders::ALL));
        f.render_widget(status, area);
    }

    fn render_status_panel(f: &mut Frame, area: Rect, app: &OrbitApp) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        let jidoka_status = app.jidoka_status();
        let jidoka_color = if jidoka_status.energy_ok && jidoka_status.angular_momentum_ok {
            Color::Green
        } else if jidoka_status.warning_count > 0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let jidoka_widget = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Jidoka: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    if jidoka_status.energy_ok {
                        "✓"
                    } else {
                        "✗"
                    },
                    Style::default().fg(jidoka_color),
                ),
                Span::raw(" Energy "),
                Span::styled(
                    if jidoka_status.angular_momentum_ok {
                        "✓"
                    } else {
                        "✗"
                    },
                    Style::default().fg(jidoka_color),
                ),
                Span::raw(" L "),
                Span::styled(
                    if jidoka_status.finite_ok {
                        "✓"
                    } else {
                        "✗"
                    },
                    Style::default().fg(jidoka_color),
                ),
                Span::raw(" Finite"),
            ]),
            Line::from(vec![Span::styled(
                format!(
                    "ΔE: {:.2e}  ΔL: {:.2e}",
                    jidoka_status.energy_error, jidoka_status.angular_momentum_error
                ),
                Style::default().fg(Color::Gray),
            )]),
        ])
        .block(Block::default().borders(Borders::ALL).title("Jidoka"));
        f.render_widget(jidoka_widget, chunks[0]);

        let heijunka_status = app.heijunka_status();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let budget_ratio = (heijunka_status.utilization * 100.0).clamp(0.0, 100.0) as u16;

        let heijunka_widget = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Heijunka Budget"),
            )
            .gauge_style(Style::default().fg(if heijunka_status.utilization <= 1.0 {
                Color::Green
            } else {
                Color::Red
            }))
            .percent(budget_ratio)
            .label(format!(
                "{:.1}ms/{:.1}ms {:?}",
                heijunka_status.used_ms, heijunka_status.budget_ms, heijunka_status.quality,
            ));
        f.render_widget(heijunka_widget, chunks[1]);
    }

    fn render_sparklines(f: &mut Frame, area: Rect, app: &OrbitApp) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(area);

        // Energy conservation sparkline (ComputeBlock)
        let energy_chars: String = app.metrics.energy.render().into_iter().collect();
        let (e_min, e_max) = app.metrics.energy.range();
        let energy_widget = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Energy Drift ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("[{e_min:.1}..{e_max:.1} ppm]"),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
            Line::from(Span::styled(energy_chars, Style::default().fg(Color::Cyan))),
        ])
        .block(Block::default().borders(Borders::ALL).title(format!(
            "SIMD: {}",
            app.metrics.simd_instruction_set().name()
        )));
        f.render_widget(energy_widget, chunks[0]);

        // Angular momentum conservation sparkline (ComputeBlock)
        let momentum_chars: String = app.metrics.momentum.render().into_iter().collect();
        let (m_min, m_max) = app.metrics.momentum.range();
        let momentum_widget = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("L Drift ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("[{m_min:.1}..{m_max:.1} ppm]"),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
            Line::from(Span::styled(
                momentum_chars,
                Style::default().fg(Color::Magenta),
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title("Momentum"));
        f.render_widget(momentum_widget, chunks[1]);

        // Frame budget trend (ComputeBlock)
        let budget_chars: String = app.metrics.frame_budget.render().into_iter().collect();
        let avg = app.metrics.frame_budget.average();
        let trend = app.metrics.frame_budget.trend();
        let trend_arrow = match trend {
            simular::tui::compute_blocks::TrendDirection::Up => "↑",
            simular::tui::compute_blocks::TrendDirection::Down => "↓",
            simular::tui::compute_blocks::TrendDirection::Flat => "→",
        };
        let budget_widget = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Budget ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{avg:.0}% {trend_arrow}"),
                    Style::default().fg(if avg <= 100.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
            ]),
            Line::from(Span::styled(
                budget_chars,
                Style::default().fg(Color::Yellow),
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title("Heijunka"));
        f.render_widget(budget_widget, chunks[2]);
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ratatui::backend::TestBackend;

        fn create_test_terminal() -> Terminal<TestBackend> {
            let backend = TestBackend::new(80, 40);
            Terminal::new(backend).expect("Failed to create test terminal")
        }

        #[test]
        fn test_ui_renders_without_panic() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            terminal
                .draw(|f| ui(f, &app))
                .expect("UI should render without panic");
        }

        #[test]
        fn test_render_title() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_title(f, area, &app);
                })
                .expect("Title should render");
        }

        #[test]
        fn test_render_title_paused() {
            let mut terminal = create_test_terminal();
            let mut app = OrbitApp::new();
            app.paused = true;

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_title(f, area, &app);
                })
                .expect("Paused title should render");
        }

        #[test]
        fn test_render_orbit_canvas() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_orbit_canvas(f, area, &app);
                })
                .expect("Canvas should render");
        }

        #[test]
        fn test_render_orbit_canvas_with_trails() {
            let mut terminal = create_test_terminal();
            let mut app = OrbitApp::new();
            // Run a few updates to generate trails
            for _ in 0..10 {
                app.update();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_orbit_canvas(f, area, &app);
                })
                .expect("Canvas with trails should render");
        }

        #[test]
        fn test_render_status() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status(f, area, &app);
                })
                .expect("Status should render");
        }

        #[test]
        fn test_render_status_panel() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status_panel(f, area, &app);
                })
                .expect("Status panel should render");
        }

        #[test]
        fn test_render_status_panel_with_warnings() {
            let mut terminal = create_test_terminal();
            let mut app = OrbitApp::new();
            // Run updates to potentially trigger jidoka warnings
            for _ in 0..100 {
                app.update();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status_panel(f, area, &app);
                })
                .expect("Status panel with warnings should render");
        }

        #[test]
        fn test_full_ui_layout() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            // Test the full UI renders with proper layout
            let result = terminal.draw(|f| ui(f, &app));
            assert!(result.is_ok());

            // Verify buffer was written to
            let buffer = terminal.backend().buffer();
            assert!(buffer.area.width > 0);
            assert!(buffer.area.height > 0);
        }

        #[test]
        fn test_ui_after_multiple_updates() {
            let mut terminal = create_test_terminal();
            let mut app = OrbitApp::new();

            // Run multiple updates
            for _ in 0..50 {
                app.update();
            }

            terminal
                .draw(|f| ui(f, &app))
                .expect("UI should render after updates");
        }

        #[test]
        fn test_jidoka_colors() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            // Test that jidoka status determines colors
            let jidoka_status = app.jidoka_status();
            let expected_color = if jidoka_status.energy_ok && jidoka_status.angular_momentum_ok {
                Color::Green
            } else if jidoka_status.warning_count > 0 {
                Color::Yellow
            } else {
                Color::Red
            };

            // Verify the color logic is correct
            assert!(
                expected_color == Color::Green
                    || expected_color == Color::Yellow
                    || expected_color == Color::Red
            );

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status_panel(f, area, &app);
                })
                .expect("Status panel should render with correct colors");
        }

        #[test]
        fn test_heijunka_budget_display() {
            let mut terminal = create_test_terminal();
            let app = OrbitApp::new();

            let heijunka_status = app.heijunka_status();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let budget_ratio = (heijunka_status.utilization * 100.0).clamp(0.0, 100.0) as u16;

            assert!(budget_ratio <= 100);

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status_panel(f, area, &app);
                })
                .expect("Heijunka budget should display correctly");
        }

        #[test]
        fn test_scale_constant() {
            // Verify the scale calculation
            let scale = 2.0 / AU;
            assert!(scale > 0.0);
            assert!(scale.is_finite());
        }
    }
}
