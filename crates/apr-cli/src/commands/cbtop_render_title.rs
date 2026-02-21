
fn ui(f: &mut Frame<'_>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Summary
            Constraint::Length(1), // Status
        ])
        .split(f.area());

    render_title(f, chunks[0], app);
    render_tabs(f, chunks[1], app);

    match app.current_view {
        View::Pipeline => render_pipeline(f, chunks[2], app),
        View::Budget => render_budget(f, chunks[2], app),
        View::Histogram => render_histogram(f, chunks[2], app),
        View::Gpu => render_gpu(f, chunks[2], app),
        View::Memory => render_memory(f, chunks[2], app),
    }

    render_summary(f, chunks[3], app);
    render_status(f, chunks[4], app);
}

fn render_title(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = format!(
        " cbtop - ComputeBrick Pipeline Monitor │ {} │ Layer {}/{} ",
        app.model_name, app.pipeline.layer_idx, app.pipeline.total_layers
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            title,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    f.render_widget(block, area);
}

fn render_tabs(f: &mut Frame<'_>, area: Rect, app: &App) {
    let titles: Vec<Line<'_>> = View::titles().iter().map(|t| Line::from(*t)).collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Views"))
        .select(app.current_view.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

fn render_pipeline(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Transformer Layer Pipeline (Mieruka: Visual Control) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Split into brick list and sparkline
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(inner);

    // Brick list with progress bars
    let items: Vec<ListItem<'_>> = app
        .pipeline
        .bricks
        .iter()
        .enumerate()
        .map(|(i, brick)| {
            let percent = brick.percent_of_budget();
            let bar_len = 20;
            let filled = (percent as usize * bar_len / 100).min(bar_len);
            let bar: String = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(bar_len.saturating_sub(filled))
            );

            let color = if percent <= 100 {
                Color::Green
            } else if percent <= 120 {
                Color::Yellow
            } else {
                Color::Red
            };

            let selected = if i == app.selected_brick {
                "► "
            } else {
                "  "
            };

            let bottleneck = if Some(brick.name) == app.pipeline.bottleneck().map(|b| b.name)
                && brick.gap_factor() > 1.0
            {
                " ← BOTTLENECK"
            } else {
                ""
            };

            let line = Line::from(vec![
                Span::raw(selected),
                Span::styled(
                    format!("{:12}", brick.name),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" │ "),
                Span::styled(
                    format!("{:6.1}µs", brick.actual_us),
                    Style::default().fg(color),
                ),
                Span::raw(" │ "),
                Span::raw(brick.status()),
                Span::raw(" │ "),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(format!(" {:3}%", percent), Style::default().fg(color)),
                Span::styled(
                    bottleneck,
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, chunks[0]);

    // Sparkline for selected brick
    if let Some(brick) = app.pipeline.bricks.get(app.selected_brick) {
        let data = brick.sparkline_data();
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} History ", brick.name)),
            )
            .data(&data)
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(sparkline, chunks[1]);
    }
}

fn render_budget(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Budget Compliance (Jidoka: Stop on Violation) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let items: Vec<ListItem<'_>> = app
        .pipeline
        .bricks
        .iter()
        .map(|brick| {
            let gap = brick.gap_factor();
            let status_color = if gap <= 1.0 {
                Color::Green
            } else if gap <= 1.2 {
                Color::Yellow
            } else {
                Color::Red
            };

            let line = Line::from(vec![
                Span::styled(
                    format!("{:12}", brick.name),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" │ Budget: "),
                Span::styled(
                    format!("{:5.1}µs", brick.budget_us),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" │ Actual: "),
                Span::styled(
                    format!("{:5.1}µs", brick.actual_us),
                    Style::default().fg(status_color),
                ),
                Span::raw(" │ Gap: "),
                Span::styled(format!("{:.2}x", gap), Style::default().fg(status_color)),
                Span::raw(" │ "),
                Span::raw(brick.status()),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, inner);
}

fn render_histogram(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Latency Distribution (p50/p99/p999) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    if let Some(brick) = app.pipeline.bricks.get(app.selected_brick) {
        let mut sorted = brick.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        let p99 = sorted
            .get((sorted.len() as f64 * 0.99) as usize)
            .copied()
            .unwrap_or(0.0);
        let p999 = sorted
            .get((sorted.len() as f64 * 0.999) as usize)
            .copied()
            .unwrap_or(0.0);

        let text = vec![
            Line::from(format!("Brick: {}", brick.name)),
            Line::from(""),
            Line::from(vec![
                Span::raw("  p50:  "),
                Span::styled(format!("{:6.2}µs", p50), Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::raw("  p99:  "),
                Span::styled(format!("{:6.2}µs", p99), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("  p999: "),
                Span::styled(format!("{:6.2}µs", p999), Style::default().fg(Color::Red)),
            ]),
            Line::from(""),
            Line::from(format!("  Samples: {}", brick.samples.len())),
        ];

        let paragraph = Paragraph::new(text);
        f.render_widget(paragraph, inner);
    }
}

fn render_gpu(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" GPU Metrics (Genchi Genbutsu: Real Data) ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Placeholder GPU metrics (would come from nvidia-smi/NVML)
    let text = vec![
        Line::from(Span::styled(
            "GPU Status",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Device:      NVIDIA RTX 4090"),
        Line::from("  Memory:      16.2 / 24.0 GB (67%)"),
        Line::from("  Utilization: 94%"),
        Line::from("  Temperature: 72°C"),
        Line::from("  Power:       385W / 450W"),
        Line::from(""),
        Line::from(Span::styled(
            "CUDA Graphs",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Captured:    Yes"),
        Line::from("  Replay Mode: Active"),
        Line::from("  Overhead:    < 100µs"),
        Line::from(""),
        Line::from(Span::styled(
            "(Real metrics require CUDA connection)",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(text);
    f.render_widget(paragraph, inner);
}

fn render_memory(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Memory Bandwidth ");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let text = vec![
        Line::from(Span::styled(
            "Memory Performance",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Peak Bandwidth:    1008 GB/s"),
        Line::from("  Achieved:          720 GB/s (71%)"),
        Line::from(""),
        Line::from(Span::styled(
            "Per-Brick Bandwidth",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  RmsNorm:   650 GB/s (bandwidth-bound)"),
        Line::from("  QkvBrick:  180 GB/s (compute-bound)"),
        Line::from("  Attention: 420 GB/s (memory-bound)"),
        Line::from("  FfnBrick:  210 GB/s (compute-bound)"),
        Line::from(""),
        Line::from(Span::styled(
            "(Requires ncu profiler for accurate data)",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(text);
    f.render_widget(paragraph, inner);
}

fn render_summary(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(" Summary ");

    let total_budget = app.pipeline.total_budget();
    let total_actual = app.pipeline.total_actual();
    // Model-level budgets for future use (full model = layer × layers)
    let _model_budget = total_budget * app.pipeline.total_layers as f64;
    let _model_actual = total_actual * app.pipeline.total_layers as f64;

    let status_color = if total_actual <= total_budget {
        Color::Green
    } else {
        Color::Red
    };

    let status = if total_actual <= total_budget {
        "✅ PASS"
    } else {
        "❌ FAIL"
    };

    let text = Line::from(vec![
        Span::raw(" Current: "),
        Span::styled(
            format!("{:.1} tok/s", app.pipeline.current_tok_s),
            Style::default().fg(status_color),
        ),
        Span::raw(" │ Target: "),
        Span::styled(
            format!("{:.0} tok/s", app.pipeline.target_tok_s),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" │ Layer: "),
        Span::styled(
            format!("{:.1}µs", total_actual),
            Style::default().fg(status_color),
        ),
        Span::raw("/"),
        Span::styled(
            format!("{:.1}µs", total_budget),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" │ "),
        Span::styled(status, Style::default().fg(status_color)),
    ]);

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

fn render_status(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let status =
        "[Enter] Drill into brick  [p]ipeline  [b]udget  [h]istogram  [g]pu  [m]emory  [q]uit";

    let paragraph = Paragraph::new(status).style(Style::default().fg(Color::DarkGray));
    f.render_widget(paragraph, area);
}
