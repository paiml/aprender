//! TUI command implementation
//!
//! Interactive terminal user interface for APR model exploration.
//!
//! Features:
//! - Model overview with metadata
//! - Tensor browser with statistics
//! - Keyboard navigation
//! - Multiple views (Overview, Tensors, Stats)

use crate::error::{CliError, Result};
use aprender::format::validation::{AprValidator, TensorStats};
use aprender::serialization::apr::AprReader;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use humansize::{format_size, BINARY};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Table, Tabs},
    Frame, Terminal,
};
use std::io;
use std::path::PathBuf;

/// Active tab in the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Tensors,
    Stats,
    Help,
}

impl Tab {
    fn titles() -> Vec<&'static str> {
        vec!["Overview [1]", "Tensors [2]", "Stats [3]", "Help [?]"]
    }

    fn index(self) -> usize {
        match self {
            Tab::Overview => 0,
            Tab::Tensors => 1,
            Tab::Stats => 2,
            Tab::Help => 3,
        }
    }
}

/// Tensor info for display
#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
    stats: Option<TensorStats>,
}

/// Application state
struct App {
    /// Model file path
    file_path: Option<PathBuf>,
    /// Model reader (if loaded)
    reader: Option<AprReader>,
    /// Validation score
    validation_score: Option<u8>,
    /// Current tab
    current_tab: Tab,
    /// Tensor list for browsing
    tensors: Vec<TensorInfo>,
    /// Selected tensor index
    tensor_list_state: ListState,
    /// Should quit
    should_quit: bool,
    /// Error message to display
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
            tensor_list_state: ListState::default(),
            should_quit: false,
            error_message: None,
        };
        app.load_model();
        app
    }

    fn load_model(&mut self) {
        let Some(path) = &self.file_path else {
            return;
        };

        // Try to load the model
        match std::fs::read(path) {
            Ok(data) => {
                // Validate
                let mut validator = AprValidator::new();
                let report = validator.validate_bytes(&data);
                self.validation_score = Some(report.total_score);

                // Load reader
                match AprReader::from_bytes(data) {
                    Ok(reader) => {
                        self.load_tensors(&reader);
                        self.reader = Some(reader);
                        if !self.tensors.is_empty() {
                            self.tensor_list_state.select(Some(0));
                        }
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
            // Try to get stats by reading the tensor data
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
        self.current_tab = match self.current_tab {
            Tab::Overview => Tab::Tensors,
            Tab::Tensors => Tab::Stats,
            Tab::Stats => Tab::Help,
            Tab::Help => Tab::Overview,
        };
    }

    fn prev_tab(&mut self) {
        self.current_tab = match self.current_tab {
            Tab::Overview => Tab::Help,
            Tab::Tensors => Tab::Overview,
            Tab::Stats => Tab::Tensors,
            Tab::Help => Tab::Stats,
        };
    }

    fn select_next_tensor(&mut self) {
        if self.tensors.is_empty() {
            return;
        }
        let i = match self.tensor_list_state.selected() {
            Some(i) => {
                if i >= self.tensors.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.tensor_list_state.select(Some(i));
    }

    fn select_prev_tensor(&mut self) {
        if self.tensors.is_empty() {
            return;
        }
        let i = match self.tensor_list_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.tensors.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.tensor_list_state.select(Some(i));
    }

    fn selected_tensor(&self) -> Option<&TensorInfo> {
        self.tensor_list_state
            .selected()
            .and_then(|i| self.tensors.get(i))
    }
}

/// Run the TUI command
pub(crate) fn run(file: Option<PathBuf>) -> Result<()> {
    // Setup terminal
    enable_raw_mode()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to enable raw mode: {e}")))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to setup terminal: {e}")))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create terminal: {e}")))?;

    // Create app and run
    let mut app = App::new(file);
    let res = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();

    res
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal
            .draw(|f| ui(f, app))
            .map_err(|e| CliError::ValidationFailed(format!("Failed to draw: {e}")))?;

        if event::poll(std::time::Duration::from_millis(100))
            .map_err(|e| CliError::ValidationFailed(format!("Event poll error: {e}")))?
        {
            if let Event::Key(key) = event::read()
                .map_err(|e| CliError::ValidationFailed(format!("Event read error: {e}")))?
            {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                        KeyCode::Tab => app.next_tab(),
                        KeyCode::BackTab => app.prev_tab(),
                        KeyCode::Char('1') => app.current_tab = Tab::Overview,
                        KeyCode::Char('2') => app.current_tab = Tab::Tensors,
                        KeyCode::Char('3') => app.current_tab = Tab::Stats,
                        KeyCode::Char('?') => app.current_tab = Tab::Help,
                        KeyCode::Down | KeyCode::Char('j') => app.select_next_tensor(),
                        KeyCode::Up | KeyCode::Char('k') => app.select_prev_tensor(),
                        _ => {}
                    }
                }
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn ui(f: &mut Frame<'_>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title bar
            Constraint::Length(3), // Tabs
            Constraint::Min(0),    // Content
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    // Title bar
    render_title(f, chunks[0], app);

    // Tabs
    render_tabs(f, chunks[1], app);

    // Content based on tab
    match app.current_tab {
        Tab::Overview => render_overview(f, chunks[2], app),
        Tab::Tensors => render_tensors(f, chunks[2], app),
        Tab::Stats => render_stats(f, chunks[2], app),
        Tab::Help => render_help(f, chunks[2]),
    }

    // Status bar
    render_status(f, chunks[3], app);
}

fn render_title(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = match &app.file_path {
        Some(p) => format!(" APR Model Inspector - {} ", p.display()),
        None => " APR Model Inspector - No file loaded ".to_string(),
    };

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
    let titles: Vec<Line<'_>> = Tab::titles().iter().map(|t| Line::from(*t)).collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Navigation"))
        .select(app.current_tab.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

fn render_overview(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Model Overview ");

    if let Some(ref error) = app.error_message {
        let error_text = Paragraph::new(error.as_str())
            .style(Style::default().fg(Color::Red))
            .block(block);
        f.render_widget(error_text, area);
        return;
    }

    let Some(reader) = &app.reader else {
        let no_file = Paragraph::new("No model loaded. Run: apr tui <model.apr>")
            .style(Style::default().fg(Color::Yellow))
            .block(block);
        f.render_widget(no_file, area);
        return;
    };

    // Build overview content
    let mut lines = Vec::new();

    // Model type
    if let Some(model_type) = reader.metadata.get("model_type") {
        lines.push(Line::from(vec![
            Span::styled("Model Type: ", Style::default().fg(Color::Cyan)),
            Span::raw(model_type.to_string().trim_matches('"').to_string()),
        ]));
    }

    // Model name
    if let Some(name) = reader.metadata.get("model_name") {
        lines.push(Line::from(vec![
            Span::styled("Model Name: ", Style::default().fg(Color::Cyan)),
            Span::raw(name.to_string().trim_matches('"').to_string()),
        ]));
    }

    // Framework
    if let Some(fw) = reader.metadata.get("framework") {
        lines.push(Line::from(vec![
            Span::styled("Framework:  ", Style::default().fg(Color::Cyan)),
            Span::raw(fw.to_string().trim_matches('"').to_string()),
        ]));
    }

    lines.push(Line::from(""));

    // Tensor summary
    lines.push(Line::from(vec![
        Span::styled("Tensors:    ", Style::default().fg(Color::Cyan)),
        Span::raw(format!("{}", app.tensors.len())),
    ]));

    // Total size
    let total_size: usize = app.tensors.iter().map(|t| t.size_bytes).sum();
    lines.push(Line::from(vec![
        Span::styled("Total Size: ", Style::default().fg(Color::Cyan)),
        Span::raw(format_size(total_size, BINARY)),
    ]));

    // Validation score
    if let Some(score) = app.validation_score {
        let score_color = if score >= 90 {
            Color::Green
        } else if score >= 70 {
            Color::Yellow
        } else {
            Color::Red
        };
        lines.push(Line::from(vec![
            Span::styled("QA Score:   ", Style::default().fg(Color::Cyan)),
            Span::styled(format!("{score}/100"), Style::default().fg(score_color)),
        ]));
    }

    lines.push(Line::from(""));

    // Hyperparameters
    if let Some(hp) = reader.metadata.get("hyperparameters") {
        lines.push(Line::from(Span::styled(
            "Hyperparameters:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        if let Some(obj) = hp.as_object() {
            for (k, v) in obj {
                lines.push(Line::from(format!("  {k}: {v}")));
            }
        }
    }

    let overview = Paragraph::new(lines).block(block);
    f.render_widget(overview, area);
}

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
                .map(|s| format!("{:.4}", s.mean))
                .unwrap_or_else(|| "-".to_string());
            let std_str = t
                .stats
                .as_ref()
                .map(|s| format!("{:.4}", s.std))
                .unwrap_or_else(|| "-".to_string());

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
        Tab::Stats => "Tab to switch views | q to quit",
        Tab::Help => "Tab to switch views | q to quit",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_titles() {
        let titles = Tab::titles();
        assert_eq!(titles.len(), 4);
        assert!(titles[0].contains("Overview"));
        assert!(titles[1].contains("Tensors"));
        assert!(titles[2].contains("Stats"));
        assert!(titles[3].contains("Help"));
    }

    #[test]
    fn test_tab_index() {
        assert_eq!(Tab::Overview.index(), 0);
        assert_eq!(Tab::Tensors.index(), 1);
        assert_eq!(Tab::Stats.index(), 2);
        assert_eq!(Tab::Help.index(), 3);
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
        assert_eq!(app.current_tab, Tab::Overview);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Help);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Stats);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Tensors);
        app.prev_tab();
        assert_eq!(app.current_tab, Tab::Overview);
    }

    #[test]
    fn test_truncate_name_short() {
        assert_eq!(truncate_name("short", 10), "short");
    }

    #[test]
    fn test_truncate_name_long() {
        assert_eq!(
            truncate_name("this_is_a_very_long_tensor_name", 20),
            "this_is_a_very_lo..."
        );
    }

    #[test]
    fn test_tensor_selection_empty() {
        let mut app = App::new(None);
        app.select_next_tensor();
        assert!(app.tensor_list_state.selected().is_none());
        app.select_prev_tensor();
        assert!(app.tensor_list_state.selected().is_none());
    }

    // TUI frame capture tests using ratatui's TestBackend
    // Converts to probar TuiFrame for assertions
    mod tui_frame_tests {
        use super::*;
        use jugar_probar::tui::TuiFrame;
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        /// Helper to render app to a test backend and capture frame
        fn render_frame(app: &mut App, width: u16, height: u16) -> TuiFrame {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal.draw(|f| ui(f, app)).unwrap();
            // Convert ratatui buffer to probar TuiFrame
            TuiFrame::from_buffer(terminal.backend().buffer(), 0)
        }

        #[test]
        fn test_tui_overview_no_file() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Should show "No model loaded" message
            assert!(
                frame.contains("No model loaded"),
                "Frame should show no model message:\n{}",
                frame.as_text()
            );
        }

        #[test]
        fn test_tui_tabs_displayed() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Should show all tab titles
            assert!(frame.contains("Overview"), "Should show Overview tab");
            assert!(frame.contains("Tensors"), "Should show Tensors tab");
            assert!(frame.contains("Stats"), "Should show Stats tab");
            assert!(frame.contains("Help"), "Should show Help tab");
        }

        #[test]
        fn test_tui_title_bar() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Should show title
            assert!(
                frame.contains("APR Model Inspector"),
                "Should show title bar"
            );
        }

        #[test]
        fn test_tui_help_view() {
            let mut app = App::new(None);
            app.current_tab = Tab::Help;
            let frame = render_frame(&mut app, 80, 24);

            // Should show help content
            assert!(
                frame.contains("Keyboard Shortcuts"),
                "Help should show keyboard shortcuts"
            );
            assert!(frame.contains("q / Esc"), "Help should show quit shortcut");
        }

        #[test]
        fn test_tui_stats_empty() {
            let mut app = App::new(None);
            app.current_tab = Tab::Stats;
            let frame = render_frame(&mut app, 80, 24);

            // Should show no data message
            assert!(
                frame.contains("No tensor data"),
                "Stats should show no data message"
            );
        }

        #[test]
        fn test_tui_tensors_empty() {
            let mut app = App::new(None);
            app.current_tab = Tab::Tensors;
            let frame = render_frame(&mut app, 80, 24);

            // Should show tensor browser (even if empty)
            assert!(
                frame.contains("Tensor") || frame.contains("navigate"),
                "Tensors view should be shown"
            );
        }

        #[test]
        fn test_tui_frame_dimensions() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 100, 30);

            assert_eq!(frame.width(), 100, "Frame width should match");
            assert_eq!(frame.height(), 30, "Frame height should match");
        }
    }
}
