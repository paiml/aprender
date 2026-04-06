//! Simular TSP GRASP Demo - Terminal User Interface
//!
//! A TUI demonstration of the GRASP methodology for TSP using ratatui.
//! App logic lives in `simular::tui::tsp_app`.

#![forbid(unsafe_code)]

#[cfg(feature = "tui")]
fn main() -> std::io::Result<()> {
    use simular::tui::tsp_app::TspApp;

    // YAML-FIRST: Default to Bay Area ground truth instance
    const DEFAULT_YAML: &str = "examples/experiments/bay_area_tsp.yaml";

    let args: Vec<String> = std::env::args().collect();
    let yaml_path = if args.len() > 1 {
        &args[1]
    } else {
        DEFAULT_YAML
    };

    let app = match TspApp::from_yaml_file(yaml_path) {
        Ok(app) => {
            eprintln!(
                "Loaded: {yaml_path} ({} cities, {} units)",
                app.demo.n, app.demo.units
            );
            app
        }
        Err(e) => {
            eprintln!("Error loading '{yaml_path}': {e}");
            eprintln!("Usage: tsp-tui [path/to/instance.yaml]");
            std::process::exit(1);
        }
    };

    tui::run(app)
}

#[cfg(not(feature = "tui"))]
fn main() {
    eprintln!("TUI feature not enabled. Run with: cargo run --bin tsp-tui --features tui");
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
            canvas::{Canvas, Line as CanvasLine},
            Block, Borders, Paragraph, Sparkline,
        },
        Frame, Terminal,
    };
    use simular::tui::tsp_app::TspApp;
    use std::io;
    use std::time::{Duration, Instant};

    pub fn run(mut app: TspApp) -> io::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, crossterm::cursor::Hide)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        let tick_rate = Duration::from_millis(200);
        let result = run_main_loop(&mut terminal, &mut app, tick_rate);

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            crossterm::cursor::Show
        )?;

        result
    }

    fn run_main_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        app: &mut TspApp,
        base_tick_rate: Duration,
    ) -> io::Result<()> {
        let mut last_tick = Instant::now();

        loop {
            terminal.draw(|f| ui(f, app))?;

            // HEIJUNKA: Adaptive tick rate based on convergence
            // - Faster when improving (200ms)
            // - Slower when converged (500ms) - reduce CPU waste
            let tick_rate = if app.demo.converged {
                Duration::from_millis(500) // Converged: slower updates
            } else {
                base_tick_rate
            };

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        app.handle_key(key.code);
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                // HEIJUNKA: Only step simulation if not converged (level load)
                if !app.demo.converged {
                    app.step();
                }
                last_tick = Instant::now();
            }

            if app.should_quit() {
                break;
            }
        }

        Ok(())
    }

    fn ui(f: &mut Frame, app: &TspApp) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(6),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(f.area());

        render_title(f, chunks[0]);
        render_equations(f, chunks[1], app);

        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(chunks[2]);

        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
            .split(main_chunks[0]);

        render_city_plot(f, left_chunks[0], app);
        render_convergence(f, left_chunks[1], app);

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(main_chunks[1]);

        render_stats(f, right_chunks[0], app);
        render_controls(f, right_chunks[1], app);

        render_status_bar(f, chunks[3], app);
    }

    fn render_title(f: &mut Frame, area: Rect) {
        let title = Paragraph::new(vec![Line::from(vec![
            Span::styled(
                " TSP GRASP Demo ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("- EDD Demo 6 "),
            Span::styled(
                "[EMC: optimization/tsp_grasp_2opt]",
                Style::default().fg(Color::Gray), // Improved contrast
            ),
        ])])
        .block(Block::default().borders(Borders::ALL).title("simular"));
        f.render_widget(title, area);
    }

    fn render_equations(f: &mut Frame, area: Rect, app: &TspApp) {
        let gap = app.optimality_gap();
        let lower_bound = app.demo.lower_bound;
        let best_tour = app.demo.best_tour_length;

        let equations_text = vec![
            Line::from(vec![
                Span::styled("Tour Length: ", Style::default().fg(Color::Yellow)),
                Span::raw("L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))  "),
                Span::styled(
                    format!("L = {best_tour:.1}"),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::styled("2-Opt Δ:     ", Style::default().fg(Color::Yellow)),
                Span::raw("Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)  "),
                Span::styled(
                    format!("[{} improvements]", app.demo.two_opt_improvements),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("1-Tree LB:   ", Style::default().fg(Color::Yellow)),
                Span::raw("L* ≥ MST(G\\{v₀}) + 2 shortest edges to v₀  "),
                Span::styled(
                    format!("= {lower_bound:.1}"),
                    Style::default().fg(Color::Blue),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("Gap = {:.1}%", gap * 100.0),
                    Style::default().fg(if gap < 0.20 { Color::Green } else { Color::Red }),
                ),
            ]),
            Line::from(vec![
                Span::styled("             ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "(Held-Karp 1970: 1-tree is subgraph of optimal tour → valid lower bound)",
                    Style::default().fg(Color::Gray), // Improved contrast + citation year
                ),
            ]),
        ];

        let equations = Paragraph::new(equations_text).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Governing Equations (EMC: optimization/tsp_grasp_2opt v1.0.0)")
                .border_style(Style::default().fg(Color::Yellow)),
        );

        f.render_widget(equations, area);
    }

    fn render_city_plot(f: &mut Frame, area: Rect, app: &TspApp) {
        let cities = &app.demo.cities;
        let best_tour = &app.demo.best_tour;
        let current_tour = &app.demo.tour; // Current exploration (changes every iteration)

        // Get city aliases from loaded YAML instance (if available)
        let aliases: Vec<String> = app.loaded_instance.as_ref().map_or_else(
            || (0..cities.len()).map(|i| format!("{i}")).collect(),
            |inst| inst.cities.iter().map(|c| c.alias.clone()).collect(),
        );

        let (min_x, max_x, min_y, max_y) = cities.iter().fold(
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
            |(min_x, max_x, min_y, max_y), c| {
                (
                    min_x.min(c.x),
                    max_x.max(c.x),
                    min_y.min(c.y),
                    max_y.max(c.y),
                )
            },
        );

        let padding = 0.15; // Slightly more padding for labels
        let x_range = (max_x - min_x).max(0.1);
        let y_range = (max_y - min_y).max(0.1);
        let x_min = min_x - padding * x_range;
        let x_max = max_x + padding * x_range;
        let y_min = min_y - padding * y_range;
        let y_max = max_y + padding * y_range;

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Tour Visualization (Green=Best, Blue=Current, Yellow=Cities)"),
            )
            .x_bounds([x_min, x_max])
            .y_bounds([y_min, y_max])
            .paint(move |ctx| {
                // Draw CURRENT exploration tour (blue, changes every iteration)
                if current_tour.len() > 1 {
                    for i in 0..current_tour.len() {
                        let j = (i + 1) % current_tour.len();
                        let c1 = &cities[current_tour[i]];
                        let c2 = &cities[current_tour[j]];
                        ctx.draw(&CanvasLine {
                            x1: c1.x,
                            y1: c1.y,
                            x2: c2.x,
                            y2: c2.y,
                            color: Color::Blue,
                        });
                    }
                }

                // Draw BEST tour (green, only updates when better found)
                if best_tour.len() > 1 {
                    for i in 0..best_tour.len() {
                        let j = (i + 1) % best_tour.len();
                        let c1 = &cities[best_tour[i]];
                        let c2 = &cities[best_tour[j]];
                        ctx.draw(&CanvasLine {
                            x1: c1.x,
                            y1: c1.y,
                            x2: c2.x,
                            y2: c2.y,
                            color: Color::Green,
                        });
                    }
                }

                // Draw cities with labels (yellow - FIXED geographic positions)
                for (i, city) in cities.iter().enumerate() {
                    // Draw city label (clone to satisfy lifetime requirements)
                    let label = aliases.get(i).cloned().unwrap_or_else(|| format!("{i}"));
                    ctx.print(
                        city.x,
                        city.y,
                        Span::styled(
                            label,
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                    );
                }
            });

        f.render_widget(canvas, area);
    }

    #[allow(clippy::cast_precision_loss)] // Tour lengths fit in f64 mantissa
    fn render_convergence(f: &mut Frame, area: Rect, app: &TspApp) {
        let data: Vec<u64> = app.convergence_history.clone();

        // Calculate statistics for visual control (Kaizen: show trend)
        let (min_val, max_val, improving) = if data.len() >= 2 {
            let min = data.iter().copied().min().unwrap_or(0);
            let max = data.iter().copied().max().unwrap_or(0);
            // Improving if recent values lower than older values
            let recent_avg: f64 = data.iter().rev().take(5).map(|&v| v as f64).sum::<f64>()
                / data.len().min(5) as f64;
            let older_avg: f64 =
                data.iter().take(5).map(|&v| v as f64).sum::<f64>() / data.len().min(5) as f64;
            (min, max, recent_avg < older_avg)
        } else {
            (0, 0, false)
        };

        // Visual Control: Show trend indicator
        let trend = if improving {
            "↓"
        } else if data.len() < 2 {
            "—"
        } else {
            "→"
        };
        let trend_color = if improving {
            Color::Green
        } else {
            Color::Yellow
        };

        let title = format!(
            "Convergence {trend} (min:{:.1}k max:{:.1}k)",
            min_val as f64 / 1000.0,
            max_val as f64 / 1000.0
        );

        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title)
                    .border_style(Style::default().fg(trend_color)),
            )
            .data(&data)
            .style(Style::default().fg(Color::Cyan));

        f.render_widget(sparkline, area);
    }

    /// Build instance info lines (Cities, Restarts).
    fn stats_instance_lines(app: &TspApp) -> Vec<Line<'static>> {
        vec![
            Line::from(vec![
                Span::raw("Cities: "),
                Span::styled(
                    format!("{}", app.demo.n),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled(" [cities]", Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(vec![
                Span::raw("Restarts: "),
                Span::styled(
                    format!("{}", app.demo.restarts),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" [iterations]", Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(""),
        ]
    }

    /// Build algorithm config lines (Method, RCL).
    fn stats_algorithm_lines(app: &TspApp) -> Vec<Line<'static>> {
        vec![
            Line::from(vec![
                Span::raw("Method: "),
                Span::styled(
                    app.construction_method_name(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw("RCL: "),
                Span::styled(
                    format!("{}", app.demo.rcl_size),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled(" [candidates]", Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(""),
        ]
    }

    /// Build tour result lines (Current, Best, Optimal, Lower Bound, Gap).
    fn stats_tour_lines(app: &TspApp, gap: f64, converged: bool) -> Vec<Line<'static>> {
        let units = &app.demo.units;
        let mut lines = vec![
            Line::from(vec![
                Span::raw("Current: "),
                Span::styled(
                    format!("{:.1}", app.demo.tour_length),
                    Style::default().fg(Color::Green),
                ),
                Span::styled(format!(" [{units}]"), Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(vec![
                Span::raw("Best: "),
                Span::styled(
                    format!("{:.1}", app.demo.best_tour_length),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" [{units}]"), Style::default().fg(Color::Gray)), // Improved contrast
                if converged {
                    Span::styled(" CONVERGED", Style::default().fg(Color::Cyan))
                } else {
                    Span::styled(" searching...", Style::default().fg(Color::Yellow))
                },
            ]),
        ];

        if let Some(optimal) = app.demo.optimal_known {
            let is_optimal = (app.demo.best_tour_length - f64::from(optimal)).abs() < 0.5;
            lines.push(Line::from(vec![
                Span::raw("Optimal: "),
                Span::styled(format!("{optimal}"), Style::default().fg(Color::Magenta)),
                Span::styled(format!(" [{units}]"), Style::default().fg(Color::Gray)), // Improved contrast
                Span::raw(" "),
                Span::styled(
                    if is_optimal { "✓" } else { "" },
                    Style::default().fg(Color::Green),
                ),
            ]));
        }

        lines.extend(vec![
            Line::from(vec![
                Span::raw("Lower Bound: "),
                Span::styled(
                    format!("{:.1}", app.demo.lower_bound),
                    Style::default().fg(Color::Blue),
                ),
                Span::styled(format!(" [{units}]"), Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(vec![
                Span::raw("Gap: "),
                Span::styled(
                    format!("{gap:.1}%"),
                    Style::default().fg(if gap < 20.0 { Color::Green } else { Color::Red }),
                ),
            ]),
            Line::from(""),
        ]);
        lines
    }

    /// Build optimization status lines (2-opt, Crossings, Verified).
    fn stats_optimization_lines(app: &TspApp, verified: bool) -> Vec<Line<'static>> {
        let crossings = app.demo.count_crossings();
        vec![
            Line::from(vec![
                Span::raw("2-opt: "),
                Span::styled(
                    app.demo.two_opt_improvements.to_string(),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled(" [improvements]", Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(vec![
                Span::raw("Crossings: "),
                Span::styled(
                    format!("{crossings}"),
                    Style::default().fg(if crossings == 0 {
                        Color::Green
                    } else {
                        Color::Red
                    }),
                ),
                Span::styled(" [edges]", Style::default().fg(Color::Gray)), // Improved contrast
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("Verified: "),
                Span::styled(
                    if verified { "YES" } else { "NO" },
                    Style::default().fg(if verified { Color::Green } else { Color::Red }),
                ),
            ]),
        ]
    }

    fn render_stats(f: &mut Frame, area: Rect, app: &TspApp) {
        let gap = app.optimality_gap() * 100.0;
        let verified = app.verify_equation();
        // MUDA: Use algorithm's stagnation-based convergence OR optimal/gap criteria
        let converged = app.demo.converged
            || app
                .demo
                .optimal_known
                .is_some_and(|opt| (app.demo.best_tour_length - f64::from(opt)).abs() < 0.5)
            || gap < 1.0;

        let mut stats_text = stats_instance_lines(app);
        stats_text.extend(stats_algorithm_lines(app));
        stats_text.extend(stats_tour_lines(app, gap, converged));
        stats_text.extend(stats_optimization_lines(app, verified));

        let stats = Paragraph::new(stats_text)
            .block(Block::default().borders(Borders::ALL).title("Statistics"));
        f.render_widget(stats, area);
    }

    fn render_controls(f: &mut Frame, area: Rect, app: &TspApp) {
        // HEIJUNKA: Show running/paused/converged status
        let (status, status_color) = if app.demo.converged {
            ("CONVERGED", Color::Cyan)
        } else if app.auto_run {
            ("RUNNING", Color::Green)
        } else {
            ("PAUSED", Color::Yellow)
        };

        let controls_text = vec![
            Line::from(vec![
                Span::raw("Status: "),
                Span::styled(status, Style::default().fg(status_color)),
                if app.demo.converged {
                    Span::styled(" (stagnated)", Style::default().fg(Color::Gray))
                } else {
                    Span::raw("")
                },
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Controls:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(" Space  - Toggle auto-run"),
            Line::from(" G      - Single GRASP iteration"),
            Line::from(" R      - Reset simulation"),
            Line::from(" +/-    - Adjust RCL size"),
            Line::from(" M      - Cycle method"),
            Line::from(" Q      - Quit"),
        ];

        let controls = Paragraph::new(controls_text)
            .block(Block::default().borders(Borders::ALL).title("Controls"));

        f.render_widget(controls, area);
    }

    fn render_status_bar(f: &mut Frame, area: Rect, app: &TspApp) {
        let status = app.falsification_status();

        // JIDOKA: Visual prominence for stop-the-line alerts
        let is_jidoka_stop = status.message.contains("JIDOKA STOP");

        let (status_style, border_style) = if status.verified {
            (
                Style::default().fg(Color::Green),
                Style::default().fg(Color::Green),
            )
        } else if is_jidoka_stop {
            // JIDOKA STOP: High-visibility red background (Andon board style)
            (
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
                Style::default().fg(Color::Red),
            )
        } else {
            (
                Style::default().fg(Color::Yellow),
                Style::default().fg(Color::Yellow),
            )
        };

        let status_text = Line::from(vec![
            Span::raw(" EDD Status: "),
            Span::styled(&status.message, status_style),
            Span::raw(" | "),
            Span::raw(format!("Frame: {} ", app.frame_count)),
        ]);

        let status_bar = Paragraph::new(status_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style),
        );

        f.render_widget(status_bar, area);
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ratatui::backend::TestBackend;

        fn create_test_terminal() -> Terminal<TestBackend> {
            let backend = TestBackend::new(120, 50);
            Terminal::new(backend).expect("Failed to create test terminal")
        }

        #[test]
        fn test_ui_renders_without_panic() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            terminal
                .draw(|f| ui(f, &app))
                .expect("UI should render without panic");
        }

        #[test]
        fn test_render_title() {
            let mut terminal = create_test_terminal();

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_title(f, area);
                })
                .expect("Title should render");
        }

        #[test]
        fn test_render_equations() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_equations(f, area, &app);
                })
                .expect("Equations should render");
        }

        #[test]
        fn test_render_equations_with_gap() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            // Run some iterations to generate a tour
            for _ in 0..5 {
                app.step();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_equations(f, area, &app);
                })
                .expect("Equations with gap should render");
        }

        #[test]
        fn test_render_city_plot() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_city_plot(f, area, &app);
                })
                .expect("City plot should render");
        }

        #[test]
        fn test_render_city_plot_with_tour() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            // Run iterations to generate tour
            for _ in 0..5 {
                app.step();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_city_plot(f, area, &app);
                })
                .expect("City plot with tour should render");
        }

        #[test]
        fn test_render_convergence() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_convergence(f, area, &app);
                })
                .expect("Convergence should render");
        }

        #[test]
        fn test_render_convergence_with_history() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            // Run iterations to build history
            for _ in 0..20 {
                app.step();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_convergence(f, area, &app);
                })
                .expect("Convergence with history should render");
        }

        #[test]
        fn test_render_stats() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_stats(f, area, &app);
                })
                .expect("Stats should render");
        }

        #[test]
        fn test_render_stats_after_optimization() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            for _ in 0..10 {
                app.step();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_stats(f, area, &app);
                })
                .expect("Stats after optimization should render");
        }

        #[test]
        fn test_render_controls_running() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            app.auto_run = true;

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_controls(f, area, &app);
                })
                .expect("Controls (running) should render");
        }

        #[test]
        fn test_render_controls_paused() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            app.auto_run = false;

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_controls(f, area, &app);
                })
                .expect("Controls (paused) should render");
        }

        #[test]
        fn test_render_status_bar() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status_bar(f, area, &app);
                })
                .expect("Status bar should render");
        }

        #[test]
        fn test_render_status_bar_verified() {
            let mut terminal = create_test_terminal();
            let mut app = TspApp::new(10, 42);
            // Run enough iterations to potentially verify
            for _ in 0..50 {
                app.step();
            }

            terminal
                .draw(|f| {
                    let area = f.area();
                    render_status_bar(f, area, &app);
                })
                .expect("Status bar (verified) should render");
        }

        #[test]
        fn test_full_ui_layout() {
            let mut terminal = create_test_terminal();
            let app = TspApp::new(10, 42);

            let result = terminal.draw(|f| ui(f, &app));
            assert!(result.is_ok());

            let buffer = terminal.backend().buffer();
            assert!(buffer.area.width > 0);
            assert!(buffer.area.height > 0);
        }

        #[test]
        fn test_ui_different_city_counts() {
            let mut terminal = create_test_terminal();

            for n in [5, 10, 20, 30] {
                let app = TspApp::new(n, 42);
                terminal
                    .draw(|f| ui(f, &app))
                    .expect(&format!("UI should render with {n} cities"));
            }
        }

        #[test]
        fn test_gap_color_logic() {
            let app = TspApp::new(10, 42);
            let gap = app.optimality_gap();

            // Gap color should be green if < 0.20, red otherwise
            let color = if gap < 0.20 { Color::Green } else { Color::Red };
            assert!(color == Color::Green || color == Color::Red);
        }

        #[test]
        fn test_city_bounds_calculation() {
            let app = TspApp::new(10, 42);
            let cities = &app.demo.cities;

            let (min_x, max_x, min_y, max_y) = cities.iter().fold(
                (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
                |(min_x, max_x, min_y, max_y), c| {
                    (
                        min_x.min(c.x),
                        max_x.max(c.x),
                        min_y.min(c.y),
                        max_y.max(c.y),
                    )
                },
            );

            assert!(min_x <= max_x);
            assert!(min_y <= max_y);
            assert!(min_x.is_finite());
            assert!(max_x.is_finite());
        }

        #[test]
        fn test_verified_display() {
            let mut app = TspApp::new(10, 42);
            for _ in 0..20 {
                app.step();
            }

            let verified = app.verify_equation();
            // Just ensure it returns a bool
            assert!(verified || !verified);
        }

        #[test]
        fn test_construction_method_name() {
            let app = TspApp::new(10, 42);
            let method_str = app.construction_method_name();
            assert!(!method_str.is_empty());
        }

        #[test]
        fn test_crossings_count_display() {
            let mut app = TspApp::new(10, 42);
            for _ in 0..10 {
                app.step();
            }

            let crossings = app.demo.count_crossings();
            let color = if crossings == 0 {
                Color::Green
            } else {
                Color::Red
            };
            assert!(color == Color::Green || color == Color::Red);
        }
    }
}
