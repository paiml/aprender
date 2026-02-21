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

include!("rendering.rs");
include!("tui_03.rs");
