
fn render_tensors(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    // Split into list and detail
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Tensor list
    let items: Vec<ListItem<'_>> = app
        .tensors
        .iter()
        .map(|t| {
            let shape_str = format!("{:?}", t.shape);
            let content = format!("{} {}", t.name, shape_str);
            ListItem::new(content)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Tensors (j/k to navigate) "),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    f.render_stateful_widget(list, chunks[0], &mut app.tensor_list_state);

    // Tensor detail
    let detail_block = Block::default()
        .borders(Borders::ALL)
        .title(" Tensor Details ");

    if let Some(tensor) = app.selected_tensor() {
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Name:  ", Style::default().fg(Color::Cyan)),
                Span::raw(tensor.name.clone()),
            ]),
            Line::from(vec![
                Span::styled("Shape: ", Style::default().fg(Color::Cyan)),
                Span::raw(format!("{:?}", tensor.shape)),
            ]),
            Line::from(vec![
                Span::styled("DType: ", Style::default().fg(Color::Cyan)),
                Span::raw(tensor.dtype.clone()),
            ]),
            Line::from(vec![
                Span::styled("Size:  ", Style::default().fg(Color::Cyan)),
                Span::raw(format_size(tensor.size_bytes, BINARY)),
            ]),
        ];

        if let Some(ref stats) = tensor.stats {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Statistics:",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(format!("  Min:    {:.6}", stats.min)));
            lines.push(Line::from(format!("  Max:    {:.6}", stats.max)));
            lines.push(Line::from(format!("  Mean:   {:.6}", stats.mean)));
            lines.push(Line::from(format!("  Std:    {:.6}", stats.std)));
            lines.push(Line::from(format!("  Zeros:  {}", stats.zero_count)));
            lines.push(Line::from(format!("  NaNs:   {}", stats.nan_count)));
            lines.push(Line::from(format!("  Infs:   {}", stats.inf_count)));
        }

        let detail = Paragraph::new(lines).block(detail_block);
        f.render_widget(detail, chunks[1]);
    } else {
        let no_selection = Paragraph::new("Select a tensor to view details")
            .style(Style::default().fg(Color::DarkGray))
            .block(detail_block);
        f.render_widget(no_selection, chunks[1]);
    }
}

fn render_stats(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Model Statistics ");

    if app.tensors.is_empty() {
        let no_data = Paragraph::new("No tensor data available")
            .style(Style::default().fg(Color::Yellow))
            .block(block);
        f.render_widget(no_data, area);
        return;
    }

    // Create stats table
    let header = Row::new(vec![
        Cell::from("Tensor").style(Style::default().fg(Color::Cyan)),
        Cell::from("Shape").style(Style::default().fg(Color::Cyan)),
        Cell::from("Size").style(Style::default().fg(Color::Cyan)),
        Cell::from("Mean").style(Style::default().fg(Color::Cyan)),
        Cell::from("Std").style(Style::default().fg(Color::Cyan)),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row<'_>> = app
        .tensors
        .iter()
        .map(|t| {
            let mean_str = t
                .stats
                .as_ref()
                .map_or_else(|| "-".to_string(), |s| format!("{:.4}", s.mean));
            let std_str = t
                .stats
                .as_ref()
                .map_or_else(|| "-".to_string(), |s| format!("{:.4}", s.std));

            Row::new(vec![
                Cell::from(truncate_name(&t.name, 30)),
                Cell::from(format!("{:?}", t.shape)),
                Cell::from(format_size(t.size_bytes, BINARY)),
                Cell::from(mean_str),
                Cell::from(std_str),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(25),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn render_help(f: &mut Frame<'_>, area: Rect) {
    let help_text = vec![
        Line::from(Span::styled(
            "Keyboard Shortcuts",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("1, 2, 3, ? ", Style::default().fg(Color::Yellow)),
            Span::raw("Switch to Overview/Tensors/Stats/Help tab"),
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
            Span::raw("Select next tensor"),
        ]),
        Line::from(vec![
            Span::styled("k / Up    ", Style::default().fg(Color::Yellow)),
            Span::raw("Select previous tensor"),
        ]),
        Line::from(vec![
            Span::styled("q / Esc   ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "About",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("APR Model Inspector - Interactive TUI for exploring APR models"),
        Line::from("Part of the apr-cli toolchain"),
        Line::from(""),
        Line::from(format!("Version: {}", env!("CARGO_PKG_VERSION"))),
    ];

    let help =
        Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title(" Help "));
    f.render_widget(help, area);
}

fn render_status(f: &mut Frame<'_>, area: Rect, app: &App) {
    let status = match app.current_tab {
        Tab::Overview => "Press Tab to switch views | q to quit",
        Tab::Tensors => "j/k to navigate | Tab to switch views | q to quit",
        Tab::Stats | Tab::Help => "Tab to switch views | q to quit",
    };

    let status_bar = Paragraph::new(status).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status_bar, area);
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len - 3])
    }
}
