//! Main application loop for the TUI monitor.

use crate::monitor::config::Config;
use crate::monitor::error::Result;
use crate::monitor::input::{Action, InputHandler};
use crate::monitor::layout::LayoutManager;
use crate::monitor::panels::{CpuPanel, MemoryPanel, ProcessPanel};
use crate::monitor::state::State;
use crate::monitor::theme::Theme;
use crate::monitor::types::Collector;

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::{self, stdout};
use std::time::Duration;

/// The main TUI monitoring application.
pub struct App {
    /// Configuration.
    config: Config,
    /// Theme (for future theming support).
    #[allow(dead_code)]
    theme: Theme,
    /// Application state.
    state: State,
    /// Input handler.
    input: InputHandler,
    /// Layout manager.
    layout: LayoutManager,
    /// CPU panel.
    cpu_panel: CpuPanel,
    /// Memory panel.
    memory_panel: MemoryPanel,
    /// Process panel.
    process_panel: ProcessPanel,
}

impl App {
    /// Creates a new application with the given configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
        let theme = Theme::default();
        let state = State::new(config.global.history_size);
        let input = InputHandler::new(config.global.vim_keys);
        let layout = LayoutManager::new();

        Self {
            config,
            theme,
            state,
            input,
            layout,
            cpu_panel: CpuPanel::new(),
            memory_panel: MemoryPanel::new(),
            process_panel: ProcessPanel::new(),
        }
    }

    /// Runs the application main loop.
    ///
    /// # Errors
    ///
    /// Returns an error if terminal setup or rendering fails.
    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout());
        let mut terminal = Terminal::new(backend)?;

        // Run the main loop
        let result = self.main_loop(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    /// The main event loop.
    fn main_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
        let poll_timeout = Duration::from_millis(100);

        loop {
            // Render
            terminal.draw(|frame| {
                self.render(frame);
            })?;

            // Poll for events
            if event::poll(poll_timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        let action = self.input.handle_key(key);
                        self.handle_action(action);
                    }
                }
            }

            // Collect metrics periodically
            self.collect_metrics();

            // Check for quit
            if self.state.should_quit {
                break;
            }
        }

        Ok(())
    }

    /// Handles an input action.
    fn handle_action(&mut self, action: Action) {
        match action {
            Action::Quit => self.state.quit(),
            Action::Help => self.state.toggle_help(),
            Action::Preset(n) => self.layout.switch_to(n as usize),
            Action::Up | Action::Down | Action::Left | Action::Right => {
                // Navigation within panels not yet implemented - reserved for future use
            }
            _ => {}
        }
    }

    /// Collects metrics from all collectors.
    fn collect_metrics(&mut self) {
        // Collect CPU metrics
        if self.cpu_panel.collector.is_available() {
            if let Ok(metrics) = self.cpu_panel.collector.collect() {
                self.state.record("cpu", metrics, self.config.global.history_size);
            }
        }

        // Collect memory metrics
        if self.memory_panel.collector.is_available() {
            if let Ok(metrics) = self.memory_panel.collector.collect() {
                self.state.record("memory", metrics, self.config.global.history_size);
            }
        }
    }

    /// Renders the application.
    fn render(&self, frame: &mut ratatui::Frame) {
        use ratatui::layout::{Constraint, Direction, Layout};
        use ratatui::style::{Color, Style};
        use ratatui::widgets::{Block, Borders, Paragraph};

        let area = frame.area();

        // Calculate layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Percentage(25),
                Constraint::Percentage(45),
            ])
            .split(area);

        // Render CPU panel
        let cpu_block = Block::default()
            .title(" CPU ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let cpu_content = if let Some(metrics) = self.state.latest("cpu") {
            let percent = metrics.get_gauge("cpu.total").unwrap_or(0.0);
            format!("CPU Usage: {percent:.1}%")
        } else {
            "CPU: collecting...".to_string()
        };

        frame.render_widget(Paragraph::new(cpu_content).block(cpu_block), chunks[0]);

        // Render memory panel with detailed swap info
        let mem_block = Block::default()
            .title(" Memory ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green));

        let mem_content = if let Some(metrics) = self.state.latest("memory") {
            let total = metrics.get_counter("memory.total").unwrap_or(0);
            let used = metrics.get_counter("memory.used").unwrap_or(0);
            let swap_total = metrics.get_counter("memory.swap.total").unwrap_or(0);
            let swap_used = metrics.get_counter("memory.swap.used").unwrap_or(0);

            let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
            let used_gb = used as f64 / (1024.0 * 1024.0 * 1024.0);
            let swap_total_gb = swap_total as f64 / (1024.0 * 1024.0 * 1024.0);
            let swap_used_gb = swap_used as f64 / (1024.0 * 1024.0 * 1024.0);

            // macOS-specific: show compressed memory
            #[cfg(target_os = "macos")]
            let extra = {
                let compressed = metrics.get_counter("memory.compressed").unwrap_or(0);
                let compressed_gb = compressed as f64 / (1024.0 * 1024.0 * 1024.0);
                if compressed_gb > 0.1 {
                    format!(" | Compressed: {:.1}G", compressed_gb)
                } else {
                    String::new()
                }
            };
            #[cfg(not(target_os = "macos"))]
            let extra = String::new();

            format!(
                "RAM: {used_gb:.0}/{total_gb:.0}G | Swap: {swap_used_gb:.0}/{swap_total_gb:.0}G{extra}"
            )
        } else {
            "Memory: collecting...".to_string()
        };

        frame.render_widget(Paragraph::new(mem_content).block(mem_block), chunks[1]);

        // Render process panel
        let proc_block = Block::default()
            .title(" Processes ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        let proc_content = format!("Processes: {}", self.process_panel.collector.count());

        frame.render_widget(Paragraph::new(proc_content).block(proc_block), chunks[2]);

        // Render help if visible
        if self.state.show_help {
            // Help overlay rendering is handled by ttop::panels::draw_help()
        }
    }

    /// Returns whether the app should quit.
    #[must_use]
    pub fn should_quit(&self) -> bool {
        self.state.should_quit
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_new() {
        let app = App::new(Config::default());
        assert!(!app.should_quit());
    }

    #[test]
    fn test_app_handle_quit() {
        let mut app = App::new(Config::default());
        app.handle_action(Action::Quit);
        assert!(app.should_quit());
    }

    #[test]
    fn test_app_handle_help() {
        let mut app = App::new(Config::default());
        assert!(!app.state.show_help);

        app.handle_action(Action::Help);
        assert!(app.state.show_help);

        app.handle_action(Action::Help);
        assert!(!app.state.show_help);
    }

    #[test]
    fn test_app_default() {
        let app = App::default();
        assert!(!app.should_quit());
    }
}
