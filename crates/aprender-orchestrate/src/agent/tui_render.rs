//! Agent TUI rendering (presentar-terminal backend).
//!
//! Extracted from `tui.rs` for QA-002 compliance (≤500 lines).
//! Feature-gated behind `presentar-terminal`.

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use presentar_terminal::{CellBuffer, Color, DiffRenderer, Modifiers};

use std::io::{self, Write};
use std::time::Duration;

use super::{AgentDashboardState, StreamEvent};

const CYAN: Color = Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
const GREEN: Color = Color { r: 0.2, g: 0.9, b: 0.2, a: 1.0 };
const YELLOW: Color = Color { r: 1.0, g: 0.9, b: 0.0, a: 1.0 };
const RED: Color = Color { r: 1.0, g: 0.2, b: 0.2, a: 1.0 };

/// Interactive agent TUI dashboard.
pub struct AgentDashboard {
    state: AgentDashboardState,
    buffer: CellBuffer,
    renderer: DiffRenderer,
    width: u16,
    height: u16,
}

impl AgentDashboard {
    /// Create a new dashboard from agent state.
    pub fn new(state: AgentDashboardState) -> Self {
        let (width, height) = crossterm::terminal::size().unwrap_or((80, 24));
        Self {
            state,
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::new(),
            width,
            height,
        }
    }

    /// Run the dashboard loop, receiving events from a channel.
    pub fn run(mut self, rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, cursor::Hide)?;

        let result = self.run_loop(&mut stdout, rx);

        disable_raw_mode()?;
        execute!(stdout, LeaveAlternateScreen, cursor::Show)?;

        result
    }

    fn run_loop(
        &mut self,
        stdout: &mut io::Stdout,
        rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>,
    ) -> anyhow::Result<()> {
        loop {
            self.drain_events(rx);
            self.handle_resize();
            self.flush_frame(stdout)?;

            if Self::poll_quit_key(Duration::from_millis(100))? {
                return Ok(());
            }

            if !self.state.running {
                self.flush_frame(stdout)?;
                Self::wait_for_any_key()?;
                return Ok(());
            }
        }
    }

    fn drain_events(&mut self, rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>) {
        while let Ok(ev) = rx.try_recv() {
            self.state.apply_event(&ev);
        }
    }

    fn handle_resize(&mut self) {
        let (w, h) = crossterm::terminal::size().unwrap_or((80, 24));
        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.buffer.resize(w, h);
            self.renderer.reset();
        }
    }

    fn flush_frame(&mut self, stdout: &mut io::Stdout) -> anyhow::Result<()> {
        self.buffer.clear();
        self.render();
        self.renderer.flush(&mut self.buffer, stdout)?;
        stdout.flush()?;
        Ok(())
    }

    /// Returns `true` if quit key (q/Esc) was pressed.
    fn poll_quit_key(timeout: Duration) -> anyhow::Result<bool> {
        if !event::poll(timeout)? {
            return Ok(false);
        }
        let Event::Key(key) = event::read()? else {
            return Ok(false);
        };
        if key.kind != KeyEventKind::Press {
            return Ok(false);
        }
        Ok(matches!(key.code, KeyCode::Char('q') | KeyCode::Esc))
    }

    /// Block until any key is pressed.
    fn wait_for_any_key() -> anyhow::Result<()> {
        loop {
            if !event::poll(Duration::from_millis(200))? {
                continue;
            }
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind == KeyEventKind::Press {
                return Ok(());
            }
        }
    }

    /// Write a string character-by-character into the cell buffer.
    fn write_str(&mut self, x: u16, y: u16, s: &str, fg: Color) {
        let mut cx = x;
        for ch in s.chars() {
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            self.buffer.update(cx, y, encoded, fg, Color::TRANSPARENT, Modifiers::NONE);
            cx = cx.saturating_add(1);
        }
    }

    fn render(&mut self) {
        self.render_title_bar(0);
        self.render_phase_indicator(2);
        self.render_divider(3);
        self.render_progress_bars(4);
        self.render_token_usage(8);
        self.render_divider(10);
        self.render_tool_log(11);
        self.render_recent_text(self.height.saturating_sub(6));
        self.render_help_bar(self.height.saturating_sub(1));
    }

    fn render_divider(&mut self, row: u16) {
        let w = (self.width as usize).min(80);
        let divider: String = "─".repeat(w);
        self.write_str(0, row, &divider, Color::WHITE);
    }

    fn render_title_bar(&mut self, row: u16) {
        let title = format!(" Agent: {} ", self.state.agent_name);
        for (i, ch) in title.chars().enumerate() {
            if (i as u16) >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(i as u16, row, s, Color::BLACK, CYAN, Modifiers::BOLD);
        }

        let status = if self.state.running { " RUNNING " } else { " DONE " };
        let status_color = if self.state.running { GREEN } else { CYAN };
        let x = self.width.saturating_sub(status.len() as u16 + 1);
        for (i, ch) in status.chars().enumerate() {
            let cx = x + i as u16;
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(cx, row, s, Color::BLACK, status_color, Modifiers::BOLD);
        }
    }

    fn render_phase_indicator(&mut self, row: u16) {
        let phase_str = format!("{:?}", self.state.phase);
        let label = format!(
            "Phase: {} | Iteration: {}/{}",
            phase_str, self.state.iteration, self.state.max_iterations,
        );
        self.write_str(1, row, &label, CYAN);

        if let Some(ref sr) = self.state.stop_reason {
            let reason = format!(" Stop: {sr:?}");
            let x = label.len() as u16 + 3;
            self.write_str(x, row, &reason, YELLOW);
        }
    }

    fn render_progress_bars(&mut self, row: u16) {
        let bar_width = 30usize;

        let iter_pct = self.state.iteration_pct();
        let iter_label = format!("Iterations: {:>3}%", iter_pct);
        self.write_str(1, row, &iter_label, Color::WHITE);
        self.render_bar(iter_label.len() as u16 + 2, row, bar_width, iter_pct);

        let tool_pct = if self.state.max_tool_calls > 0 {
            (self.state.tool_calls * 100) / self.state.max_tool_calls
        } else {
            0
        };
        let tool_label = format!("Tool calls: {:>3}%", tool_pct);
        self.write_str(1, row + 1, &tool_label, Color::WHITE);
        self.render_bar(tool_label.len() as u16 + 2, row + 1, bar_width, tool_pct);

        if self.state.token_budget.is_some() {
            let tok_pct = self.state.token_budget_pct();
            let tok_label = format!("Token budget: {:>3}%", tok_pct);
            self.write_str(1, row + 2, &tok_label, Color::WHITE);
            self.render_bar(tok_label.len() as u16 + 2, row + 2, bar_width, tok_pct);
        }
    }

    fn render_bar(&mut self, x: u16, y: u16, width: usize, pct: u32) {
        let filled = (pct as usize * width) / 100;
        let color = if pct >= 90 {
            RED
        } else if pct >= 70 {
            YELLOW
        } else {
            GREEN
        };
        let bar: String = "█".repeat(filled) + &"░".repeat(width.saturating_sub(filled));
        self.write_str(x, y, &format!("[{bar}]"), color);
    }

    fn render_token_usage(&mut self, row: u16) {
        let total = self.state.usage.input_tokens + self.state.usage.output_tokens;
        let usage_str = format!(
            "Tokens: {} in / {} out = {} total",
            self.state.usage.input_tokens, self.state.usage.output_tokens, total,
        );
        self.write_str(1, row, &usage_str, Color::WHITE);

        if self.state.max_cost_usd > 0.0 {
            let cost_str =
                format!("  Cost: ${:.4} / ${:.4}", self.state.cost_usd, self.state.max_cost_usd,);
            let x = usage_str.len() as u16 + 2;
            self.write_str(x, row, &cost_str, YELLOW);
        }
    }

    fn render_tool_log(&mut self, row: u16) {
        self.write_str(1, row, "Tool Log:", CYAN);

        let max_entries =
            (self.height.saturating_sub(row + 8) as usize).min(self.state.tool_log.len());
        let max_w = self.width.saturating_sub(2) as usize;

        let lines: Vec<String> = self
            .state
            .tool_log
            .iter()
            .rev()
            .take(max_entries)
            .map(|entry| {
                let line = format!("  {} → {}", entry.name, entry.result_summary);
                if line.len() > max_w {
                    format!("{}...", &line[..max_w.saturating_sub(3)])
                } else {
                    line
                }
            })
            .collect();

        for (i, line) in lines.iter().enumerate() {
            self.write_str(1, row + 1 + i as u16, line, Color::WHITE);
        }
    }

    fn render_recent_text(&mut self, row: u16) {
        self.write_str(1, row, "Output:", CYAN);

        let text: String = self.state.recent_text.join("");
        let w = self.width.saturating_sub(4) as usize;
        let max_chars = w * 4;
        let display = if text.len() > max_chars { &text[text.len() - max_chars..] } else { &text };

        for (i, chunk) in display.as_bytes().chunks(w.max(1)).take(4).enumerate() {
            let s = String::from_utf8_lossy(chunk);
            self.write_str(2, row + 1 + i as u16, &s, GREEN);
        }
    }

    fn render_help_bar(&mut self, row: u16) {
        let help = if self.state.running { " q: quit " } else { " Press any key to exit " };
        for (i, ch) in help.chars().enumerate() {
            if (i as u16) >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(i as u16, row, s, Color::BLACK, Color::WHITE, Modifiers::NONE);
        }
    }
}
