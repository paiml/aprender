
fn ui(f: &mut Frame, app: &MonitorApp) {
    let area = f.area();

    // Main layout: left (60%) and right (40%) - inspired by simular
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Left side: throughput chart (60%) + GPU (25%) + controls (15%)
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(55),
            Constraint::Percentage(30),
            Constraint::Percentage(15),
        ])
        .split(main_chunks[0]);

    // Right side: metrics (60%) + system (40%)
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(main_chunks[1]);

    // ============================================================
    // LEFT PANEL: Throughput Chart
    // ============================================================
    let throughput_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Throughput ");

    let inner = throughput_block.inner(left_chunks[0]);
    f.render_widget(throughput_block, left_chunks[0]);

    let chart_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // Sparkline
            Constraint::Length(3), // Stats
        ])
        .split(inner);

    // Sparkline
    let data = app.throughput_series.as_u64_vec();
    let sparkline = RatatuiSparkline::default()
        .data(&data)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, chart_layout[0]);

    // Stats below sparkline
    let current = app.metrics.throughput_tok_per_sec;
    let peak = app.peak_throughput;
    let trend = app.throughput_series.trend();
    let trend_color = match trend {
        "↑" => Color::Green,
        "↓" => Color::Red,
        _ => Color::Yellow,
    };

    let stats_text = Paragraph::new(vec![
        Line::from(vec![
            Span::raw("Current: "),
            Span::styled(
                format!("{:.1} tok/s ", current),
                Style::default().fg(Color::Yellow).bold(),
            ),
            Span::styled(trend, Style::default().fg(trend_color).bold()),
            Span::raw("   Peak: "),
            Span::styled(
                format!("{:.1} tok/s", peak),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::raw("Samples: "),
            Span::raw(format!("{}", app.throughput_series.len())),
            Span::raw("   Min: "),
            Span::raw(format!("{:.1}", app.throughput_series.min().unwrap_or(0.0))),
            Span::raw("   Max: "),
            Span::raw(format!("{:.1}", app.throughput_series.max().unwrap_or(0.0))),
        ]),
    ]);
    f.render_widget(stats_text, chart_layout[1]);

    // ============================================================
    // LEFT PANEL: GPU Memory
    // ============================================================
    let gpu_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta))
        .title(" GPU Memory ");

    let gpu_inner = gpu_block.inner(left_chunks[1]);
    f.render_widget(gpu_block, left_chunks[1]);

    let gpu_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Memory bar
            Constraint::Length(1), // Memory text
            Constraint::Length(1), // Spacer
            Constraint::Min(2),    // Status lines
        ])
        .split(gpu_inner);

    // GPU Memory Gauge
    let gpu_percent = if app.metrics.gpu_memory_total_bytes > 0 {
        ((app.metrics.gpu_memory_used_bytes as f64 / app.metrics.gpu_memory_total_bytes as f64)
            * 100.0) as u16
    } else {
        0
    };

    // Use explicit gpu_color() function (PARITY-108c)
    let gauge_color = gpu_color(gpu_percent);

    let gpu_gauge = Gauge::default()
        .percent(gpu_percent)
        .gauge_style(Style::default().fg(gauge_color).bg(Color::DarkGray))
        .label(format!("{}%", gpu_percent));
    f.render_widget(gpu_gauge, gpu_layout[0]);

    // Memory text
    let used = MonitorApp::format_gb(app.metrics.gpu_memory_used_bytes);
    let total = MonitorApp::format_gb(app.metrics.gpu_memory_total_bytes);
    let mem_text =
        Paragraph::new(format!("{} / {}", used, total)).style(Style::default().fg(Color::White));
    f.render_widget(mem_text, gpu_layout[1]);

    // CUDA status and batch info
    let cuda_status = if app.metrics.cuda_path_active {
        Span::styled("● Active", Style::default().fg(Color::Green).bold())
    } else {
        Span::styled("○ Inactive", Style::default().fg(Color::Red))
    };

    let status_text = Paragraph::new(vec![
        Line::from(vec![Span::raw("CUDA: "), cuda_status]),
        Line::from(vec![
            Span::raw("Batch: "),
            Span::styled(
                format!("{}", app.metrics.batch_size),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw("   Queue: "),
            Span::styled(
                format!("{}", app.metrics.queue_depth),
                Style::default().fg(Color::Yellow),
            ),
        ]),
    ]);
    f.render_widget(status_text, gpu_layout[3]);

    // ============================================================
    // LEFT PANEL: Controls
    // ============================================================
    // Use MonitorState enum for status display (PARITY-108a)
    let (status_color, status_icon) = match app.state {
        MonitorState::Paused => (Color::Yellow, "⏸ Paused"),
        MonitorState::Connected => (Color::Green, "● Connected"),
        MonitorState::Disconnected => (Color::Red, "○ Disconnected"),
    };

    let controls = Paragraph::new(vec![Line::from(vec![
        Span::styled(status_icon, Style::default().fg(status_color)),
        Span::raw("   "),
        Span::styled("[q]", Style::default().fg(Color::Cyan)),
        Span::raw(" Quit  "),
        Span::styled("[r]", Style::default().fg(Color::Cyan)),
        Span::raw(" Reset  "),
        Span::styled("[p]", Style::default().fg(Color::Cyan)),
        Span::raw(" Pause"),
    ])])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" Controls "),
    );
    f.render_widget(controls, left_chunks[2]);

    // ============================================================
    // RIGHT PANEL: Metrics
    // ============================================================
    let latency_trend = app.latency_series.trend();
    let latency_color = match latency_trend {
        "↑" => Color::Red,   // Higher latency is bad
        "↓" => Color::Green, // Lower latency is good
        _ => Color::Yellow,
    };

    let metrics_text = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Throughput: "),
            Span::styled(
                format!("{:.1} tok/s", app.metrics.throughput_tok_per_sec),
                Style::default().fg(Color::Yellow).bold(),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Latency P50: "),
            Span::styled(
                format!("{:.1} ms ", app.metrics.latency_p50_ms),
                Style::default().fg(Color::Green),
            ),
            Span::styled(latency_trend, Style::default().fg(latency_color)),
        ]),
        Line::from(vec![
            Span::raw("  Latency P95: "),
            Span::styled(
                format!("{:.1} ms", app.metrics.latency_p95_ms),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Latency P99: "),
            Span::styled(
                format!("{:.1} ms", app.metrics.latency_p99_ms),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Requests: "),
            Span::styled(
                format!("{:>10}", format_number(app.metrics.total_requests)),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Tokens:   "),
            Span::styled(
                format!("{:>10}", format_number(app.metrics.total_tokens)),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Uptime:   "),
            Span::styled(
                format!("{:>10}", app.format_uptime()),
                Style::default().fg(Color::White),
            ),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue))
            .title(" Metrics "),
    );
    f.render_widget(metrics_text, right_chunks[0]);

    // ============================================================
    // RIGHT PANEL: System Info
    // ============================================================
    let model_name = if app.metrics.model_name.is_empty() {
        "N/A".to_string()
    } else {
        app.metrics.model_name.clone()
    };

    let error_line = if let Some(ref error) = app.last_error {
        Line::from(vec![
            Span::raw("  Error: "),
            Span::styled(error.clone(), Style::default().fg(Color::Red)),
        ])
    } else {
        Line::from(vec![
            Span::raw("  Status: "),
            Span::styled("OK", Style::default().fg(Color::Green)),
        ])
    };

    let system_text = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(model_name, Style::default().fg(Color::Cyan).bold()),
        ]),
        Line::from(vec![
            Span::raw("  Server: "),
            Span::styled(app.url.clone(), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        error_line,
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" System "),
    );
    f.render_widget(system_text, right_chunks[1]);
}

/// Format number with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
