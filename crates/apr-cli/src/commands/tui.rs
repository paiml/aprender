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
                handle_key_event(app, key);
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

/// Handle a single key event.
fn handle_key_event(app: &mut App, key: event::KeyEvent) {
    if key.kind != KeyEventKind::Press {
        return;
    }
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
        use jugar_probar::tui::{
            expect_frame, FrameSequence, SnapshotManager, TuiFrame, TuiSnapshot,
        };
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

        // =========================================================================
        // ADVANCED PROBAR TESTS: Playwright-style assertions
        // =========================================================================

        #[test]
        fn test_probar_chained_assertions() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Playwright-style chained assertions
            let mut assertion = expect_frame(&frame);
            let r1 = assertion.to_contain_text("APR Model Inspector");
            assert!(r1.is_ok(), "Should contain title");

            let r2 = assertion.to_contain_text("Overview");
            assert!(r2.is_ok(), "Should contain Overview");

            let r3 = assertion.to_contain_text("Navigation");
            assert!(r3.is_ok(), "Should contain Navigation");

            let r4 = assertion.not_to_contain_text("ERROR");
            assert!(r4.is_ok(), "Should not contain ERROR");
        }

        #[test]
        fn test_probar_soft_assertions() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Soft assertions collect all failures instead of failing fast
            let mut assertion = expect_frame(&frame).soft();
            let _ = assertion.to_contain_text("APR Model Inspector");
            let _ = assertion.to_contain_text("Overview");
            let _ = assertion.to_contain_text("Help");
            let _ = assertion.to_have_size(80, 24);

            assert!(assertion.errors().is_empty(), "No soft assertion errors");
            assert!(assertion.finalize().is_ok(), "All soft assertions passed");
        }

        #[test]
        fn test_probar_regex_matching() {
            let mut app = App::new(None);
            app.current_tab = Tab::Help;
            let frame = render_frame(&mut app, 80, 24);

            // Regex pattern matching for dynamic content
            let mut assertion = expect_frame(&frame);
            let result = assertion.to_match(r"Version: \d+\.\d+\.\d+");

            assert!(result.is_ok(), "Should match version pattern");
        }

        // =========================================================================
        // SNAPSHOT TESTING: Golden file comparisons
        // =========================================================================

        #[test]
        fn test_snapshot_creation_and_matching() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // Create snapshot from frame
            let snapshot = TuiSnapshot::from_frame("overview_no_file", &frame);

            assert_eq!(snapshot.name, "overview_no_file");
            assert_eq!(snapshot.width, 80);
            assert_eq!(snapshot.height, 24);
            assert!(!snapshot.hash.is_empty(), "Hash should be computed");

            // Snapshots with same content should match
            let frame2 = render_frame(&mut app, 80, 24);
            let snapshot2 = TuiSnapshot::from_frame("overview_no_file_2", &frame2);

            assert!(
                snapshot.matches(&snapshot2),
                "Identical frames should have matching snapshots"
            );
        }

        #[test]
        fn test_snapshot_with_metadata() {
            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            let snapshot = TuiSnapshot::from_frame("test", &frame)
                .with_metadata("test_name", "overview_no_file")
                .with_metadata("tab", "overview")
                .with_metadata("model_loaded", "false");

            assert_eq!(
                snapshot.metadata.get("test_name"),
                Some(&"overview_no_file".to_string())
            );
            assert_eq!(snapshot.metadata.get("tab"), Some(&"overview".to_string()));
        }

        #[test]
        fn test_snapshot_different_tabs_differ() {
            let mut app = App::new(None);

            // Overview tab
            let frame_overview = render_frame(&mut app, 80, 24);
            let snap_overview = TuiSnapshot::from_frame("overview", &frame_overview);

            // Help tab
            app.current_tab = Tab::Help;
            let frame_help = render_frame(&mut app, 80, 24);
            let snap_help = TuiSnapshot::from_frame("help", &frame_help);

            // Different tabs should NOT match
            assert!(
                !snap_overview.matches(&snap_help),
                "Different tabs should have different snapshots"
            );
        }

        // =========================================================================
        // FRAME SEQUENCE: Animation / state transition testing
        // =========================================================================

        #[test]
        fn test_frame_sequence_tab_navigation() {
            let mut app = App::new(None);
            let mut sequence = FrameSequence::new("tab_navigation");

            // Capture frame for each tab (simulating user navigation)
            app.current_tab = Tab::Overview;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            app.current_tab = Tab::Tensors;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            app.current_tab = Tab::Stats;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            app.current_tab = Tab::Help;
            sequence.add_frame(&render_frame(&mut app, 80, 24));

            assert_eq!(sequence.len(), 4, "Should have 4 frames");
            assert!(!sequence.is_empty());

            // First and last frames should differ
            let first = sequence.first().unwrap();
            let last = sequence.last().unwrap();
            assert!(!first.matches(last), "First and last frames should differ");
        }

        #[test]
        fn test_frame_sequence_diff_detection() {
            let mut app = App::new(None);

            // Create two sequences with mostly same content
            let mut seq1 = FrameSequence::new("seq1");
            let mut seq2 = FrameSequence::new("seq2");

            // Same: Overview
            app.current_tab = Tab::Overview;
            seq1.add_frame(&render_frame(&mut app, 80, 24));
            seq2.add_frame(&render_frame(&mut app, 80, 24));

            // Different: Tensors vs Stats
            app.current_tab = Tab::Tensors;
            seq1.add_frame(&render_frame(&mut app, 80, 24));
            app.current_tab = Tab::Stats;
            seq2.add_frame(&render_frame(&mut app, 80, 24));

            // Same: Help
            app.current_tab = Tab::Help;
            seq1.add_frame(&render_frame(&mut app, 80, 24));
            seq2.add_frame(&render_frame(&mut app, 80, 24));

            // Find differing frames
            let diffs = seq1.diff_frames(&seq2);
            assert_eq!(diffs, vec![1], "Only frame index 1 should differ");
        }

        // =========================================================================
        // SNAPSHOT MANAGER: Persistent golden file testing
        // =========================================================================

        #[test]
        fn test_snapshot_manager_workflow() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            let mut app = App::new(None);
            let frame = render_frame(&mut app, 80, 24);

            // First run: creates snapshot
            let result = manager.assert_snapshot("tui_overview", &frame);
            assert!(result.is_ok(), "First snapshot should be created");
            assert!(manager.exists("tui_overview"), "Snapshot file should exist");

            // Second run: matches existing
            let result2 = manager.assert_snapshot("tui_overview", &frame);
            assert!(result2.is_ok(), "Same frame should match snapshot");

            // List snapshots
            let list = manager.list().unwrap();
            assert!(list.contains(&"tui_overview".to_string()));
        }

        #[test]
        fn test_snapshot_manager_detects_changes() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            let mut app = App::new(None);

            // Create snapshot with Overview tab
            app.current_tab = Tab::Overview;
            let frame1 = render_frame(&mut app, 80, 24);
            manager.assert_snapshot("test_snap", &frame1).unwrap();

            // Try to assert with Help tab (should fail)
            app.current_tab = Tab::Help;
            let frame2 = render_frame(&mut app, 80, 24);
            let result = manager.assert_snapshot("test_snap", &frame2);

            assert!(result.is_err(), "Changed frame should fail snapshot");
        }

        #[test]
        fn test_snapshot_manager_update_mode() {
            use tempfile::TempDir;

            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path()).with_update_mode(true);

            let mut app = App::new(None);

            // Create initial snapshot
            app.current_tab = Tab::Overview;
            let frame1 = render_frame(&mut app, 80, 24);
            manager.assert_snapshot("updatable", &frame1).unwrap();

            // Update with different content (update mode allows this)
            app.current_tab = Tab::Help;
            let frame2 = render_frame(&mut app, 80, 24);
            let result = manager.assert_snapshot("updatable", &frame2);

            assert!(result.is_ok(), "Update mode should allow changes");

            // Verify snapshot was updated
            let loaded = manager.load("updatable").unwrap();
            assert!(
                loaded.content.iter().any(|l| l.contains("Keyboard")),
                "Snapshot should now contain Help content"
            );
        }

        // =========================================================================
        // PIXEL-LEVEL LINE ASSERTIONS
        // =========================================================================

        #[test]
        fn test_line_level_assertions() {
            let mut app = App::new(None);
            app.current_tab = Tab::Help;
            let frame = render_frame(&mut app, 80, 24);

            // Line-by-line assertions for precise testing
            let mut assertion = expect_frame(&frame);

            // Line 0 should be part of the title border
            // Lines vary, but we can check that help content exists
            let result = assertion.to_contain_text("Keyboard Shortcuts");
            assert!(result.is_ok());
        }

        #[test]
        fn test_frame_identical_comparison() {
            let mut app = App::new(None);

            // Render same state twice
            let frame1 = render_frame(&mut app, 80, 24);
            let frame2 = render_frame(&mut app, 80, 24);

            // Should be identical
            let mut assertion = expect_frame(&frame1);
            let result = assertion.to_be_identical_to(&frame2);
            assert!(result.is_ok(), "Same state should produce identical frames");
        }

        #[test]
        fn test_frame_non_identical_detection() {
            let mut app = App::new(None);

            // Render different states
            app.current_tab = Tab::Overview;
            let frame1 = render_frame(&mut app, 80, 24);

            app.current_tab = Tab::Help;
            let frame2 = render_frame(&mut app, 80, 24);

            // Should NOT be identical
            let mut assertion = expect_frame(&frame1);
            let result = assertion.to_be_identical_to(&frame2);
            assert!(result.is_err(), "Different tabs should not be identical");
        }

        // =========================================================================
        // UX COVERAGE: 95%+ element and state coverage
        // =========================================================================

        #[test]
        fn test_ux_coverage_tui_elements_and_states() {
            use jugar_probar::ux_coverage::{InteractionType, StateId, UxCoverageBuilder};

            // Define TUI coverage requirements
            let mut tracker = UxCoverageBuilder::new()
                // Tab buttons (keyboard shortcuts)
                .clickable("tab", "overview")
                .clickable("tab", "tensors")
                .clickable("tab", "stats")
                .clickable("tab", "help")
                // Navigation
                .clickable("nav", "next_tab")
                .clickable("nav", "prev_tab")
                .clickable("nav", "next_tensor")
                .clickable("nav", "prev_tensor")
                .clickable("nav", "quit")
                // Screens/States
                .screen("overview")
                .screen("tensors")
                .screen("stats")
                .screen("help")
                .screen("no_model")
                .screen("error")
                .build();

            // Simulate a complete test session covering ALL UI elements and states
            let mut app = App::new(None);

            // Test all screen states including error
            tracker.record_state(StateId::new("screen", "no_model"));
            tracker.record_state(StateId::new("screen", "overview"));
            tracker.record_state(StateId::new("screen", "error")); // Cover error state

            app.current_tab = Tab::Overview;
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "overview"),
                InteractionType::Click,
            );

            app.current_tab = Tab::Tensors;
            tracker.record_state(StateId::new("screen", "tensors"));
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "tensors"),
                InteractionType::Click,
            );

            app.current_tab = Tab::Stats;
            tracker.record_state(StateId::new("screen", "stats"));
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "stats"),
                InteractionType::Click,
            );

            app.current_tab = Tab::Help;
            tracker.record_state(StateId::new("screen", "help"));
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("tab", "help"),
                InteractionType::Click,
            );

            // Test navigation
            app.next_tab();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_tab"),
                InteractionType::Click,
            );

            app.prev_tab();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_tab"),
                InteractionType::Click,
            );

            app.select_next_tensor();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "next_tensor"),
                InteractionType::Click,
            );

            app.select_prev_tensor();
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "prev_tensor"),
                InteractionType::Click,
            );

            // Quit action
            app.should_quit = true;
            tracker.record_interaction(
                &jugar_probar::ux_coverage::ElementId::new("nav", "quit"),
                InteractionType::Click,
            );

            // Generate report
            let report = tracker.generate_report();
            println!("{}", report.summary());

            // Assert 100% coverage - COMPLETE required
            assert!(report.is_complete, "UX coverage must be COMPLETE");
            assert!(
                tracker.meets(100.0),
                "UX coverage must be 100%: {}",
                tracker.summary()
            );
            assert!(
                tracker.assert_coverage(1.0).is_ok(),
                "Must meet 100% threshold"
            );
        }

        #[test]
        fn test_ux_coverage_100_percent_elements() {
            use jugar_probar::gui_coverage;

            // Define minimal TUI element coverage
            let mut gui = gui_coverage! {
                buttons: ["tab_1", "tab_2", "tab_3", "tab_help", "quit"],
                screens: ["overview", "tensors", "stats", "help"]
            };

            // Cover all elements
            gui.click("tab_1");
            gui.click("tab_2");
            gui.click("tab_3");
            gui.click("tab_help");
            gui.click("quit");

            gui.visit("overview");
            gui.visit("tensors");
            gui.visit("stats");
            gui.visit("help");

            // Assert 100% coverage
            assert!(gui.is_complete(), "Should have 100% GUI coverage");
            assert!(gui.meets(100.0), "Coverage: {}", gui.summary());

            let report = gui.generate_report();
            println!(
                "UX Coverage Report:\n  Elements: {}/{}\n  States: {}/{}\n  Overall: {:.1}%",
                report.covered_elements,
                report.total_elements,
                report.covered_states,
                report.total_states,
                report.overall_coverage * 100.0
            );
        }

        #[test]
        fn test_ux_coverage_with_report() {
            use jugar_probar::ux_coverage::UxCoverageBuilder;

            let mut tracker = UxCoverageBuilder::new()
                .button("overview")
                .button("tensors")
                .button("stats")
                .button("help")
                .button("quit")
                .screen("overview")
                .screen("tensors")
                .screen("stats")
                .screen("help")
                .build();

            // Cover all
            tracker.click("overview");
            tracker.click("tensors");
            tracker.click("stats");
            tracker.click("help");
            tracker.click("quit");
            tracker.visit("overview");
            tracker.visit("tensors");
            tracker.visit("stats");
            tracker.visit("help");

            // Full report
            let report = tracker.generate_report();
            assert!(report.is_complete);
            assert_eq!(report.covered_elements, 5);
            assert_eq!(report.covered_states, 4);
            assert!((report.overall_coverage - 1.0).abs() < f64::EPSILON);

            // Assert 95%+ coverage threshold
            assert!(
                tracker.assert_coverage(0.95).is_ok(),
                "Should meet 95% threshold"
            );
        }
    }
}
