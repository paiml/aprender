
fn render_health(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" NODE HEALTH ");

    let statuses = app.health.all_statuses();

    if statuses.is_empty() {
        let no_nodes = Paragraph::new("No nodes registered for health monitoring.")
            .style(Style::default().fg(Color::Yellow))
            .block(block);
        f.render_widget(no_nodes, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Node").style(Style::default().fg(Color::Cyan)),
        Cell::from("State").style(Style::default().fg(Color::Cyan)),
        Cell::from("Latency P50").style(Style::default().fg(Color::Cyan)),
        Cell::from("Latency P99").style(Style::default().fg(Color::Cyan)),
        Cell::from("Queue").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = statuses
        .iter()
        .map(|status| {
            let state_style = match status.state {
                HealthState::Healthy => Style::default().fg(Color::Green),
                HealthState::Degraded => Style::default().fg(Color::Yellow),
                HealthState::Unhealthy => Style::default().fg(Color::Red),
                HealthState::Unknown => Style::default().fg(Color::DarkGray),
            };

            Row::new(vec![
                Cell::from(status.node_id.0.clone()),
                Cell::from(format!("{:?}", status.state)).style(state_style),
                Cell::from(format!("{}ms", status.latency_p50.as_millis())),
                Cell::from(format!("{}ms", status.latency_p99.as_millis())),
                Cell::from(format!("{}", status.queue_depth)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_routing(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" ROUTING DECISIONS ");

    if app.routing_history.is_empty() {
        let no_routing = Paragraph::new("No routing decisions recorded yet.")
            .style(Style::default().fg(Color::Yellow))
            .block(block);
        f.render_widget(no_routing, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Request").style(Style::default().fg(Color::Cyan)),
        Cell::from("Capability").style(Style::default().fg(Color::Cyan)),
        Cell::from("Selected").style(Style::default().fg(Color::Cyan)),
        Cell::from("Score").style(Style::default().fg(Color::Cyan)),
        Cell::from("Reason").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = app
        .routing_history
        .iter()
        .rev()
        .take(20)
        .map(|record| {
            Row::new(vec![
                Cell::from(truncate(&record.request_id, 12)),
                Cell::from(record.capability.clone()),
                Cell::from(record.selected_node.clone()),
                Cell::from(format!("{:.2}", record.score)),
                Cell::from(truncate(&record.reason, 30)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(30),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_circuits(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" CIRCUIT BREAKERS ");

    let circuits = app.circuit_breaker.all_states();

    if circuits.is_empty() {
        let no_circuits = Paragraph::new("No circuit breakers active.")
            .style(Style::default().fg(Color::DarkGray))
            .block(block);
        f.render_widget(no_circuits, area);
        return;
    }

    let header = Row::new(vec![
        Cell::from("Node").style(Style::default().fg(Color::Cyan)),
        Cell::from("State").style(Style::default().fg(Color::Cyan)),
        Cell::from("Failures").style(Style::default().fg(Color::Cyan)),
        Cell::from("Last Failure").style(Style::default().fg(Color::Cyan)),
        Cell::from("Reset In").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = circuits
        .iter()
        .map(|(node_id, state)| {
            let state_style = match state {
                CircuitState::Closed => Style::default().fg(Color::Green),
                CircuitState::HalfOpen => Style::default().fg(Color::Yellow),
                CircuitState::Open => Style::default().fg(Color::Red),
            };

            Row::new(vec![
                Cell::from(node_id.0.clone()),
                Cell::from(format!("{:?}", state)).style(state_style),
                Cell::from("-"), // Would need failure count from circuit breaker
                Cell::from("-"),
                Cell::from("-"),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_stats(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" GATEWAY STATS ");

    let stats = app
        .gateway
        .as_ref()
        .map_or_else(GatewayStats::default, |g| g.stats());

    // Split area for metrics and gauges
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Metrics
            Constraint::Length(3), // Success rate gauge
            Constraint::Min(0),    // Spacer
        ])
        .split(area);

    // Metrics
    let metrics = vec![
        Line::from(vec![
            Span::styled("Total Requests:    ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", stats.total_requests)),
        ]),
        Line::from(vec![
            Span::styled("Successful:        ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{}", stats.successful_requests),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::styled("Failed:            ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{}", stats.failed_requests),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::styled("Total Tokens:      ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", stats.total_tokens)),
        ]),
        Line::from(vec![
            Span::styled("Avg Latency:       ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}ms", stats.avg_latency.as_millis())),
        ]),
        Line::from(vec![
            Span::styled("Active Streams:    ", Style::default().fg(Color::Cyan)),
            Span::raw(format!("{}", stats.active_streams)),
        ]),
    ];

    let metrics_widget = Paragraph::new(metrics).block(block);
    f.render_widget(metrics_widget, chunks[0]);

    // Success rate gauge
    let success_rate = app.success_rate();
    let gauge_color = if success_rate >= 0.99 {
        Color::Green
    } else if success_rate >= 0.95 {
        Color::Yellow
    } else {
        Color::Red
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Success Rate"))
        .gauge_style(Style::default().fg(gauge_color))
        .ratio(success_rate)
        .label(format!("{:.1}%", success_rate * 100.0));

    f.render_widget(gauge, chunks[1]);
}

fn render_policies(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" ACTIVE POLICIES ");

    let header = Row::new(vec![
        Cell::from("Policy").style(Style::default().fg(Color::Cyan)),
        Cell::from("Weight").style(Style::default().fg(Color::Cyan)),
        Cell::from("Status").style(Style::default().fg(Color::Cyan)),
        Cell::from("Description").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = app
        .policies
        .iter()
        .map(|policy| {
            let status_style = if policy.enabled {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            Row::new(vec![
                Cell::from(policy.name.clone()),
                Cell::from(format!("{:.1}", policy.weight)),
                Cell::from(if policy.enabled { "Active" } else { "Disabled" }).style(status_style),
                Cell::from(policy.description.clone()),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(15),
            Constraint::Percentage(10),
            Constraint::Percentage(15),
            Constraint::Percentage(60),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_help(f: &mut Frame<'_>, area: Rect) {
    let help_text = vec![
        Line::from(Span::styled(
            "Federation Gateway Dashboard",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Keyboard Shortcuts",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("1-6, ?    ", Style::default().fg(Color::Yellow)),
            Span::raw("Switch to Catalog/Health/Routing/Circuits/Stats/Policies/Help"),
        ]),
        Line::from(vec![
            Span::styled("Tab       ", Style::default().fg(Color::Yellow)),
            Span::raw("Next tab"),
        ]),
        Line::from(vec![
            Span::styled("Shift+Tab ", Style::default().fg(Color::Yellow)),
            Span::raw("Previous tab"),
        ]),
        Line::from(vec![
            Span::styled("j / Down  ", Style::default().fg(Color::Yellow)),
            Span::raw("Select next row"),
        ]),
        Line::from(vec![
            Span::styled("k / Up    ", Style::default().fg(Color::Yellow)),
            Span::raw("Select previous row"),
        ]),
        Line::from(vec![
            Span::styled("r         ", Style::default().fg(Color::Yellow)),
            Span::raw("Refresh data"),
        ]),
        Line::from(vec![
            Span::styled("h         ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle health panel"),
        ]),
        Line::from(vec![
            Span::styled("c         ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle circuit panel"),
        ]),
        Line::from(vec![
            Span::styled("s         ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle stats panel"),
        ]),
        Line::from(vec![
            Span::styled("q / Esc   ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Panels",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("• Catalog: View registered models and deployments"),
        Line::from("• Health: Monitor node health and latency"),
        Line::from("• Routing: Track routing decisions and scores"),
        Line::from("• Circuits: View circuit breaker states"),
        Line::from("• Stats: Gateway statistics and success rate"),
        Line::from("• Policies: Active routing policies"),
    ];

    let help =
        Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title(" Help "));
    f.render_widget(help, area);
}

fn render_status_bar(f: &mut Frame<'_>, area: Rect, app: &FederationApp) {
    let _status = app
        .status_message
        .as_deref()
        .unwrap_or("Tab to switch views │ q to quit │ ? for help");

    let left = "Federation Gateway v1.0";
    let center = format!(
        "{}/{} nodes healthy",
        app.healthy_node_count(),
        app.total_node_count()
    );
    let right = format!(
        "{:.1} req/s │ {:.1}% success",
        app.requests_per_sec(),
        app.success_rate() * 100.0
    );

    let full_status = format!("{} │ {} │ {}", left, center, right);

    let status_bar = Paragraph::new(full_status).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status_bar, area);
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
