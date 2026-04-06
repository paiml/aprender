//! TUI Dashboard for simular.
//!
//! Real-time terminal visualization using ratatui.
//!
//! This module is only available with the `tui` feature.

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};

use super::{SimMetrics, TimeSeries};
use crate::engine::{SimState, SimTime};
use crate::error::{SimError, SimResult};

/// TUI Dashboard for real-time simulation visualization.
pub struct SimularTui {
    /// Terminal instance.
    terminal: Terminal<CrosstermBackend<Stdout>>,
    /// Dashboard state.
    state: DashboardState,
    /// Refresh rate in Hz.
    refresh_hz: u32,
    /// Last frame time.
    last_frame: Instant,
}

/// Dashboard state for tracking visualization data.
#[derive(Debug)]
pub struct DashboardState {
    /// Energy time series.
    pub energy_series: TimeSeries,
    /// Kinetic energy time series.
    pub ke_series: TimeSeries,
    /// Potential energy time series.
    pub pe_series: TimeSeries,
    /// Steps per second time series.
    pub throughput_series: TimeSeries,
    /// Current metrics.
    pub metrics: SimMetrics,
    /// Simulation running state.
    pub running: bool,
    /// Paused state.
    pub paused: bool,
    /// Selected panel.
    pub selected_panel: usize,
    /// Status message.
    pub status: String,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            energy_series: TimeSeries::new("Total Energy", 200),
            ke_series: TimeSeries::new("Kinetic Energy", 200),
            pe_series: TimeSeries::new("Potential Energy", 200),
            throughput_series: TimeSeries::new("Steps/sec", 200),
            metrics: SimMetrics::new(),
            running: true,
            paused: false,
            selected_panel: 0,
            status: "Ready".to_string(),
        }
    }
}

impl DashboardState {
    /// Update dashboard state with new simulation data.
    /// This is the pure logic extracted from `SimularTui::update()`.
    pub fn update_from_sim(&mut self, sim_state: &SimState, time: SimTime, metrics: &SimMetrics) {
        let t = time.as_secs_f64();

        // Update time series
        if let Some(te) = metrics.total_energy {
            self.energy_series.push(t, te);
        }
        if let Some(ke) = metrics.kinetic_energy {
            self.ke_series.push(t, ke);
        }
        if let Some(pe) = metrics.potential_energy {
            self.pe_series.push(t, pe);
        }
        self.throughput_series.push(t, metrics.steps_per_second);

        // Update current metrics
        self.metrics = metrics.clone();
        self.metrics.update_from_state(sim_state, time);
    }

    /// Toggle pause state and update status message.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.status = if self.paused {
            "Paused by user".to_string()
        } else {
            "Resumed".to_string()
        };
    }

    /// Request reset and update status.
    pub fn request_reset(&mut self) {
        self.status = "Reset requested".to_string();
    }

    /// Stop the simulation.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Set status message.
    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    /// Check if paused.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// Check if running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &SimMetrics {
        &self.metrics
    }

    /// Get energy series.
    #[must_use]
    pub const fn energy_series(&self) -> &TimeSeries {
        &self.energy_series
    }

    /// Get kinetic energy series.
    #[must_use]
    pub const fn ke_series(&self) -> &TimeSeries {
        &self.ke_series
    }

    /// Get potential energy series.
    #[must_use]
    pub const fn pe_series(&self) -> &TimeSeries {
        &self.pe_series
    }

    /// Get throughput series.
    #[must_use]
    pub const fn throughput_series(&self) -> &TimeSeries {
        &self.throughput_series
    }

    /// Format controls text for rendering.
    #[must_use]
    pub fn format_controls_text(&self) -> String {
        let status_text = if self.paused {
            "PAUSED"
        } else if self.running {
            "RUNNING"
        } else {
            "STOPPED"
        };

        format!(
            "Status: {}\n\n\
             [Space] Pause/Resume\n\
             [R] Reset\n\
             [Q] Quit\n\n\
             {}",
            status_text, self.status
        )
    }

    /// Format metrics text for rendering.
    #[must_use]
    #[allow(clippy::option_if_let_else)]
    pub fn format_metrics_text(&self) -> String {
        let m = &self.metrics;

        let energy_text = if let Some(te) = m.total_energy {
            format!("Total: {te:.6}")
        } else {
            "Total: N/A".to_string()
        };

        let ke_text = if let Some(ke) = m.kinetic_energy {
            format!("Kinetic: {ke:.6}")
        } else {
            "Kinetic: N/A".to_string()
        };

        let pe_text = if let Some(pe) = m.potential_energy {
            format!("Potential: {pe:.6}")
        } else {
            "Potential: N/A".to_string()
        };

        let drift_text = if let Some(drift) = m.energy_drift {
            format!("Drift: {drift:.2e}")
        } else {
            "Drift: N/A".to_string()
        };

        format!(
            "Time: {:.4}s\n\
             Step: {}\n\
             Bodies: {}\n\n\
             Energy:\n  {}\n  {}\n  {}\n  {}\n\n\
             Throughput: {:.1} steps/s\n\
             Jidoka: {} warnings, {} errors",
            m.time,
            m.step,
            m.body_count,
            energy_text,
            ke_text,
            pe_text,
            drift_text,
            m.steps_per_second,
            m.jidoka_warnings,
            m.jidoka_errors
        )
    }

    /// Format energy chart text for rendering.
    #[must_use]
    pub fn format_energy_chart_text(&self) -> String {
        let energy_data = &self.energy_series;

        if energy_data.is_empty() {
            "No data yet...".to_string()
        } else {
            let min = energy_data.min().unwrap_or(0.0);
            let max = energy_data.max().unwrap_or(0.0);
            let last = energy_data.last_value().unwrap_or(0.0);
            let (t_start, t_end) = energy_data.time_range().unwrap_or((0.0, 0.0));

            format!(
                "Energy over time:\n\n\
                 Range: [{:.4}, {:.4}]\n\
                 Current: {:.6}\n\
                 Time: {:.2}s - {:.2}s\n\
                 Samples: {}",
                min,
                max,
                last,
                t_start,
                t_end,
                energy_data.len()
            )
        }
    }

    /// Get status color based on state.
    #[must_use]
    pub const fn status_color(&self) -> Color {
        if self.paused {
            Color::Yellow
        } else if self.running {
            Color::Green
        } else {
            Color::Red
        }
    }
}

impl SimularTui {
    /// Create and initialize TUI dashboard.
    ///
    /// # Errors
    ///
    /// Returns error if terminal initialization fails.
    pub fn new(refresh_hz: u32) -> SimResult<Self> {
        enable_raw_mode().map_err(|e| SimError::io(format!("Failed to enable raw mode: {e}")))?;

        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| SimError::io(format!("Failed to enter alternate screen: {e}")))?;

        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)
            .map_err(|e| SimError::io(format!("Failed to create terminal: {e}")))?;

        Ok(Self {
            terminal,
            state: DashboardState::default(),
            refresh_hz: refresh_hz.max(1),
            last_frame: Instant::now(),
        })
    }

    /// Update dashboard with new simulation state.
    pub fn update(&mut self, sim_state: &SimState, time: SimTime, metrics: &SimMetrics) {
        self.state.update_from_sim(sim_state, time, metrics);
    }

    /// Render the dashboard.
    ///
    /// # Errors
    ///
    /// Returns error if rendering fails.
    pub fn render(&mut self) -> SimResult<()> {
        let state = &self.state;

        self.terminal
            .draw(|frame| {
                let area = frame.area();

                // Main layout: left (60%) and right (40%)
                let main_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .split(area);

                // Left side: trajectory (70%) and controls (30%)
                let left_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                    .split(main_chunks[0]);

                // Right side: metrics (50%) and energy chart (50%)
                let right_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(main_chunks[1]);

                // Render trajectory placeholder
                let trajectory_block = Block::default()
                    .title(" Trajectory ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan));

                let trajectory_text = Paragraph::new(format!(
                    "Bodies: {}\nTime: {:.3}s\nStep: {}",
                    state.metrics.body_count, state.metrics.time, state.metrics.step
                ))
                .block(trajectory_block)
                .style(Style::default().fg(Color::White));

                frame.render_widget(trajectory_text, left_chunks[0]);

                // Render controls
                let controls = Self::render_controls(state);
                frame.render_widget(controls, left_chunks[1]);

                // Render metrics
                let metrics = Self::render_metrics(state);
                frame.render_widget(metrics, right_chunks[0]);

                // Render energy chart
                let energy_chart = Self::render_energy_chart(state, right_chunks[1]);
                frame.render_widget(energy_chart, right_chunks[1]);
            })
            .map_err(|e| SimError::io(format!("Render failed: {e}")))?;

        Ok(())
    }

    /// Render controls panel.
    fn render_controls(state: &DashboardState) -> Paragraph<'static> {
        let status_color = if state.paused {
            Color::Yellow
        } else if state.running {
            Color::Green
        } else {
            Color::Red
        };

        let status_text = if state.paused {
            "PAUSED"
        } else if state.running {
            "RUNNING"
        } else {
            "STOPPED"
        };

        let text = format!(
            "Status: {}\n\n\
             [Space] Pause/Resume\n\
             [R] Reset\n\
             [Q] Quit\n\n\
             {}",
            status_text, state.status
        );

        Paragraph::new(text)
            .block(
                Block::default()
                    .title(" Controls ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(status_color)),
            )
            .style(Style::default().fg(Color::White))
    }

    /// Render metrics panel.
    #[allow(clippy::option_if_let_else)]
    fn render_metrics(state: &DashboardState) -> Paragraph<'static> {
        let m = &state.metrics;

        let energy_text = if let Some(te) = m.total_energy {
            format!("Total: {te:.6}")
        } else {
            "Total: N/A".to_string()
        };

        let ke_text = if let Some(ke) = m.kinetic_energy {
            format!("Kinetic: {ke:.6}")
        } else {
            "Kinetic: N/A".to_string()
        };

        let pe_text = if let Some(pe) = m.potential_energy {
            format!("Potential: {pe:.6}")
        } else {
            "Potential: N/A".to_string()
        };

        let drift_text = if let Some(drift) = m.energy_drift {
            format!("Drift: {drift:.2e}")
        } else {
            "Drift: N/A".to_string()
        };

        let text = format!(
            "Time: {:.4}s\n\
             Step: {}\n\
             Bodies: {}\n\n\
             Energy:\n  {}\n  {}\n  {}\n  {}\n\n\
             Throughput: {:.1} steps/s\n\
             Jidoka: {} warnings, {} errors",
            m.time,
            m.step,
            m.body_count,
            energy_text,
            ke_text,
            pe_text,
            drift_text,
            m.steps_per_second,
            m.jidoka_warnings,
            m.jidoka_errors
        );

        Paragraph::new(text)
            .block(
                Block::default()
                    .title(" Metrics ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta)),
            )
            .style(Style::default().fg(Color::White))
    }

    /// Render energy chart.
    fn render_energy_chart(state: &DashboardState, _area: Rect) -> Paragraph<'static> {
        // Simplified text-based chart since we can't easily do sparklines
        // with the data we have
        let energy_data = &state.energy_series;

        let chart_text = if energy_data.is_empty() {
            "No data yet...".to_string()
        } else {
            let min = energy_data.min().unwrap_or(0.0);
            let max = energy_data.max().unwrap_or(0.0);
            let last = energy_data.last_value().unwrap_or(0.0);
            let (t_start, t_end) = energy_data.time_range().unwrap_or((0.0, 0.0));

            format!(
                "Energy over time:\n\n\
                 Range: [{:.4}, {:.4}]\n\
                 Current: {:.6}\n\
                 Time: {:.2}s - {:.2}s\n\
                 Samples: {}",
                min,
                max,
                last,
                t_start,
                t_end,
                energy_data.len()
            )
        };

        Paragraph::new(chart_text)
            .block(
                Block::default()
                    .title(" Energy Chart ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Blue)),
            )
            .style(Style::default().fg(Color::White))
    }

    /// Handle input events.
    ///
    /// Returns `Ok(false)` if the application should quit.
    ///
    /// # Errors
    ///
    /// Returns error if event handling fails.
    pub fn handle_events(&mut self) -> SimResult<bool> {
        let frame_duration = Duration::from_millis(1000 / u64::from(self.refresh_hz));
        let elapsed = self.last_frame.elapsed();

        if elapsed < frame_duration {
            let remaining = frame_duration.saturating_sub(elapsed);
            if event::poll(remaining)
                .map_err(|e| SimError::io(format!("Event poll failed: {e}")))?
            {
                if let Event::Key(key) =
                    event::read().map_err(|e| SimError::io(format!("Event read failed: {e}")))?
                {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                return Ok(false);
                            }
                            KeyCode::Char(' ') => {
                                self.state.paused = !self.state.paused;
                                self.state.status = if self.state.paused {
                                    "Paused by user".to_string()
                                } else {
                                    "Resumed".to_string()
                                };
                            }
                            KeyCode::Char('r') => {
                                self.state.status = "Reset requested".to_string();
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        self.last_frame = Instant::now();
        Ok(true)
    }

    /// Check if simulation is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.state.paused
    }

    /// Check if simulation should continue.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.state.running
    }

    /// Set status message.
    pub fn set_status(&mut self, status: impl Into<String>) {
        self.state.status = status.into();
    }

    /// Stop the simulation.
    pub fn stop(&mut self) {
        self.state.running = false;
    }

    /// Restore terminal state.
    fn restore_terminal(&mut self) -> SimResult<()> {
        disable_raw_mode().map_err(|e| SimError::io(format!("Failed to disable raw mode: {e}")))?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .map_err(|e| SimError::io(format!("Failed to leave alternate screen: {e}")))?;
        self.terminal
            .show_cursor()
            .map_err(|e| SimError::io(format!("Failed to show cursor: {e}")))?;
        Ok(())
    }
}

impl Drop for SimularTui {
    fn drop(&mut self) {
        // Best effort to restore terminal
        let _ = self.restore_terminal();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_default() {
        let state = DashboardState::default();
        assert!(state.running);
        assert!(!state.paused);
        assert_eq!(state.selected_panel, 0);
        assert_eq!(state.status, "Ready");
    }

    #[test]
    fn test_dashboard_state_series_initialized() {
        let state = DashboardState::default();
        assert!(state.energy_series.is_empty());
        assert!(state.ke_series.is_empty());
        assert!(state.pe_series.is_empty());
        assert!(state.throughput_series.is_empty());
    }

    #[test]
    fn test_dashboard_state_metrics() {
        let state = DashboardState::default();
        assert_eq!(state.metrics.step, 0);
        assert_eq!(state.metrics.body_count, 0);
    }

    #[test]
    fn test_render_controls_running() {
        let state = DashboardState::default();
        let widget = SimularTui::render_controls(&state);
        // Widget was created successfully - testing the logic branches
        let _ = widget;
    }

    #[test]
    fn test_render_controls_paused() {
        let mut state = DashboardState::default();
        state.paused = true;
        let widget = SimularTui::render_controls(&state);
        let _ = widget;
    }

    #[test]
    fn test_render_controls_stopped() {
        let mut state = DashboardState::default();
        state.running = false;
        let widget = SimularTui::render_controls(&state);
        let _ = widget;
    }

    #[test]
    fn test_render_metrics_empty() {
        let state = DashboardState::default();
        let widget = SimularTui::render_metrics(&state);
        let _ = widget;
    }

    #[test]
    fn test_render_metrics_with_energy() {
        let mut state = DashboardState::default();
        state.metrics.total_energy = Some(100.0);
        state.metrics.kinetic_energy = Some(60.0);
        state.metrics.potential_energy = Some(40.0);
        state.metrics.energy_drift = Some(0.001);
        let widget = SimularTui::render_metrics(&state);
        let _ = widget;
    }

    #[test]
    fn test_render_energy_chart_empty() {
        let state = DashboardState::default();
        let area = ratatui::prelude::Rect::new(0, 0, 100, 50);
        let widget = SimularTui::render_energy_chart(&state, area);
        let _ = widget;
    }

    #[test]
    fn test_render_energy_chart_with_data() {
        let mut state = DashboardState::default();
        state.energy_series.push(0.0, 100.0);
        state.energy_series.push(1.0, 99.0);
        state.energy_series.push(2.0, 98.0);
        let area = ratatui::prelude::Rect::new(0, 0, 100, 50);
        let widget = SimularTui::render_energy_chart(&state, area);
        let _ = widget;
    }

    // Test update logic (can't test full TUI without terminal)
    #[test]
    fn test_dashboard_state_update_manually() {
        let mut state = DashboardState::default();

        // Simulate what update() does
        let t = 1.0;
        state.energy_series.push(t, 100.0);
        state.ke_series.push(t, 60.0);
        state.pe_series.push(t, 40.0);
        state.throughput_series.push(t, 1000.0);

        state.metrics.time = t;
        state.metrics.step = 100;
        state.metrics.total_energy = Some(100.0);
        state.metrics.kinetic_energy = Some(60.0);
        state.metrics.potential_energy = Some(40.0);

        assert_eq!(state.energy_series.len(), 1);
        assert!((state.metrics.time - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dashboard_pause_toggle() {
        let mut state = DashboardState::default();
        assert!(!state.paused);

        state.paused = !state.paused;
        assert!(state.paused);
        state.status = "Paused by user".to_string();
        assert_eq!(state.status, "Paused by user");

        state.paused = !state.paused;
        assert!(!state.paused);
        state.status = "Resumed".to_string();
        assert_eq!(state.status, "Resumed");
    }

    #[test]
    fn test_dashboard_reset_status() {
        let mut state = DashboardState::default();
        state.status = "Reset requested".to_string();
        assert_eq!(state.status, "Reset requested");
    }

    #[test]
    fn test_dashboard_stop() {
        let mut state = DashboardState::default();
        assert!(state.running);
        state.running = false;
        assert!(!state.running);
    }

    #[test]
    fn test_dashboard_selected_panel() {
        let mut state = DashboardState::default();
        assert_eq!(state.selected_panel, 0);
        state.selected_panel = 1;
        assert_eq!(state.selected_panel, 1);
    }

    #[test]
    fn test_time_series_capacity_200() {
        // Verify the default capacity is 200
        let _state = DashboardState::default();
        // Push more than 200 items
        let mut series = TimeSeries::new("test", 200);
        for i in 0..250 {
            series.push(i as f64, i as f64);
        }
        assert_eq!(series.len(), 200);
    }

    // ==========================================================================
    // Tests for extracted DashboardState methods
    // ==========================================================================

    #[test]
    fn test_dashboard_update_from_sim() {
        let mut state = DashboardState::default();
        let mut sim_state = SimState::new();
        sim_state.add_body(
            1.0,
            crate::engine::state::Vec3::zero(),
            crate::engine::state::Vec3::zero(),
        );

        let time = SimTime::from_secs(1.5);
        let mut metrics = SimMetrics::new();
        metrics.total_energy = Some(100.0);
        metrics.kinetic_energy = Some(60.0);
        metrics.potential_energy = Some(40.0);
        metrics.steps_per_second = 1000.0;

        state.update_from_sim(&sim_state, time, &metrics);

        assert_eq!(state.energy_series.len(), 1);
        assert_eq!(state.ke_series.len(), 1);
        assert_eq!(state.pe_series.len(), 1);
        assert_eq!(state.throughput_series.len(), 1);
        assert!(state.metrics.total_energy.is_some());
    }

    #[test]
    fn test_dashboard_update_from_sim_no_energy() {
        let mut state = DashboardState::default();
        let sim_state = SimState::new();
        let time = SimTime::from_secs(1.0);
        let metrics = SimMetrics::new(); // No energy values

        state.update_from_sim(&sim_state, time, &metrics);

        // Only throughput series gets updated (always has a value)
        assert!(state.energy_series.is_empty());
        assert!(state.ke_series.is_empty());
        assert!(state.pe_series.is_empty());
        assert_eq!(state.throughput_series.len(), 1);
    }

    #[test]
    fn test_dashboard_toggle_pause() {
        let mut state = DashboardState::default();
        assert!(!state.paused);
        assert_eq!(state.status, "Ready");

        state.toggle_pause();
        assert!(state.paused);
        assert_eq!(state.status, "Paused by user");

        state.toggle_pause();
        assert!(!state.paused);
        assert_eq!(state.status, "Resumed");
    }

    #[test]
    fn test_dashboard_request_reset() {
        let mut state = DashboardState::default();
        state.request_reset();
        assert_eq!(state.status, "Reset requested");
    }

    #[test]
    fn test_dashboard_stop_method() {
        let mut state = DashboardState::default();
        assert!(state.running);

        state.stop();
        assert!(!state.running);
    }

    #[test]
    fn test_dashboard_set_status() {
        let mut state = DashboardState::default();
        state.set_status("Custom status");
        assert_eq!(state.status, "Custom status");

        state.set_status(String::from("Another status"));
        assert_eq!(state.status, "Another status");
    }

    #[test]
    fn test_dashboard_is_paused() {
        let mut state = DashboardState::default();
        assert!(!state.is_paused());

        state.paused = true;
        assert!(state.is_paused());
    }

    #[test]
    fn test_dashboard_is_running() {
        let mut state = DashboardState::default();
        assert!(state.is_running());

        state.running = false;
        assert!(!state.is_running());
    }

    #[test]
    fn test_dashboard_metrics_accessor() {
        let mut state = DashboardState::default();
        state.metrics.step = 42;
        state.metrics.body_count = 5;

        let m = state.metrics();
        assert_eq!(m.step, 42);
        assert_eq!(m.body_count, 5);
    }

    #[test]
    fn test_dashboard_energy_series_accessor() {
        let mut state = DashboardState::default();
        state.energy_series.push(1.0, 100.0);

        let es = state.energy_series();
        assert_eq!(es.len(), 1);
    }

    #[test]
    fn test_dashboard_ke_series_accessor() {
        let mut state = DashboardState::default();
        state.ke_series.push(1.0, 60.0);

        let ks = state.ke_series();
        assert_eq!(ks.len(), 1);
    }

    #[test]
    fn test_dashboard_pe_series_accessor() {
        let mut state = DashboardState::default();
        state.pe_series.push(1.0, 40.0);

        let ps = state.pe_series();
        assert_eq!(ps.len(), 1);
    }

    #[test]
    fn test_dashboard_throughput_series_accessor() {
        let mut state = DashboardState::default();
        state.throughput_series.push(1.0, 1000.0);

        let ts = state.throughput_series();
        assert_eq!(ts.len(), 1);
    }

    #[test]
    fn test_dashboard_format_controls_text_running() {
        let state = DashboardState::default();
        let text = state.format_controls_text();

        assert!(text.contains("RUNNING"));
        assert!(text.contains("[Space] Pause/Resume"));
        assert!(text.contains("[R] Reset"));
        assert!(text.contains("[Q] Quit"));
        assert!(text.contains("Ready")); // Default status
    }

    #[test]
    fn test_dashboard_format_controls_text_paused() {
        let mut state = DashboardState::default();
        state.paused = true;
        state.status = "Paused by user".to_string();

        let text = state.format_controls_text();
        assert!(text.contains("PAUSED"));
        assert!(text.contains("Paused by user"));
    }

    #[test]
    fn test_dashboard_format_controls_text_stopped() {
        let mut state = DashboardState::default();
        state.running = false;

        let text = state.format_controls_text();
        assert!(text.contains("STOPPED"));
    }

    #[test]
    fn test_dashboard_format_metrics_text_empty() {
        let state = DashboardState::default();
        let text = state.format_metrics_text();

        assert!(text.contains("Time: 0.0000s"));
        assert!(text.contains("Step: 0"));
        assert!(text.contains("Bodies: 0"));
        assert!(text.contains("Total: N/A"));
        assert!(text.contains("Kinetic: N/A"));
        assert!(text.contains("Potential: N/A"));
        assert!(text.contains("Drift: N/A"));
    }

    #[test]
    fn test_dashboard_format_metrics_text_with_values() {
        let mut state = DashboardState::default();
        state.metrics.time = 1.5;
        state.metrics.step = 100;
        state.metrics.body_count = 3;
        state.metrics.total_energy = Some(100.5);
        state.metrics.kinetic_energy = Some(60.3);
        state.metrics.potential_energy = Some(40.2);
        state.metrics.energy_drift = Some(0.001);
        state.metrics.steps_per_second = 500.0;
        state.metrics.jidoka_warnings = 2;
        state.metrics.jidoka_errors = 1;

        let text = state.format_metrics_text();

        assert!(text.contains("Time: 1.5000s"));
        assert!(text.contains("Step: 100"));
        assert!(text.contains("Bodies: 3"));
        assert!(text.contains("Total: 100.500000"));
        assert!(text.contains("Kinetic: 60.300000"));
        assert!(text.contains("Potential: 40.200000"));
        assert!(text.contains("Drift:"));
        assert!(text.contains("500.0 steps/s"));
        assert!(text.contains("2 warnings"));
        assert!(text.contains("1 errors"));
    }

    #[test]
    fn test_dashboard_format_energy_chart_text_empty() {
        let state = DashboardState::default();
        let text = state.format_energy_chart_text();

        assert_eq!(text, "No data yet...");
    }

    #[test]
    fn test_dashboard_format_energy_chart_text_with_data() {
        let mut state = DashboardState::default();
        state.energy_series.push(0.0, 100.0);
        state.energy_series.push(1.0, 99.5);
        state.energy_series.push(2.0, 99.0);

        let text = state.format_energy_chart_text();

        assert!(text.contains("Energy over time:"));
        assert!(text.contains("Range:"));
        assert!(text.contains("Current:"));
        assert!(text.contains("Time:"));
        assert!(text.contains("Samples: 3"));
    }

    #[test]
    fn test_dashboard_status_color_running() {
        let state = DashboardState::default();
        assert_eq!(state.status_color(), Color::Green);
    }

    #[test]
    fn test_dashboard_status_color_paused() {
        let mut state = DashboardState::default();
        state.paused = true;
        assert_eq!(state.status_color(), Color::Yellow);
    }

    #[test]
    fn test_dashboard_status_color_stopped() {
        let mut state = DashboardState::default();
        state.running = false;
        assert_eq!(state.status_color(), Color::Red);
    }

    #[test]
    fn test_dashboard_status_color_paused_takes_precedence() {
        let mut state = DashboardState::default();
        state.paused = true;
        state.running = true;
        // Paused takes precedence over running
        assert_eq!(state.status_color(), Color::Yellow);
    }

    #[test]
    fn test_dashboard_debug_impl() {
        let state = DashboardState::default();
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("DashboardState"));
        assert!(debug_str.contains("running: true"));
        assert!(debug_str.contains("paused: false"));
    }

    #[test]
    fn test_dashboard_multiple_updates() {
        let mut state = DashboardState::default();
        let sim_state = SimState::new();
        let mut metrics = SimMetrics::new();

        // Multiple updates at different times
        for i in 0..10 {
            let time = SimTime::from_secs(i as f64 * 0.1);
            metrics.total_energy = Some(100.0 - i as f64 * 0.1);
            metrics.kinetic_energy = Some(60.0 - i as f64 * 0.05);
            metrics.potential_energy = Some(40.0 - i as f64 * 0.05);
            metrics.steps_per_second = 1000.0 + i as f64 * 10.0;
            state.update_from_sim(&sim_state, time, &metrics);
        }

        assert_eq!(state.energy_series.len(), 10);
        assert_eq!(state.ke_series.len(), 10);
        assert_eq!(state.pe_series.len(), 10);
        assert_eq!(state.throughput_series.len(), 10);
    }

    #[test]
    fn test_dashboard_partial_energy_update() {
        let mut state = DashboardState::default();
        let sim_state = SimState::new();
        let time = SimTime::from_secs(1.0);

        // Only total energy set
        let mut metrics = SimMetrics::new();
        metrics.total_energy = Some(100.0);
        state.update_from_sim(&sim_state, time, &metrics);

        assert_eq!(state.energy_series.len(), 1);
        assert!(state.ke_series.is_empty());
        assert!(state.pe_series.is_empty());
    }

    #[test]
    fn test_dashboard_state_accessors_chain() {
        let state = DashboardState::default();

        // Test chained accessor calls
        let _ = state.metrics().time;
        let _ = state.energy_series().len();
        let _ = state.ke_series().len();
        let _ = state.pe_series().len();
        let _ = state.throughput_series().len();
        let _ = state.is_paused();
        let _ = state.is_running();
        let _ = state.status_color();
    }

    #[test]
    fn test_format_metrics_partial_values() {
        let mut state = DashboardState::default();
        // Only kinetic energy set
        state.metrics.kinetic_energy = Some(50.0);

        let text = state.format_metrics_text();
        assert!(text.contains("Total: N/A"));
        assert!(text.contains("Kinetic: 50.000000"));
        assert!(text.contains("Potential: N/A"));
    }

    #[test]
    fn test_format_energy_chart_single_point() {
        let mut state = DashboardState::default();
        state.energy_series.push(0.0, 100.0);

        let text = state.format_energy_chart_text();
        assert!(text.contains("Samples: 1"));
    }

    #[test]
    fn test_render_controls_all_states() {
        // Running state
        let state_running = DashboardState::default();
        let _ = SimularTui::render_controls(&state_running);

        // Paused state
        let mut state_paused = DashboardState::default();
        state_paused.paused = true;
        let _ = SimularTui::render_controls(&state_paused);

        // Stopped state
        let mut state_stopped = DashboardState::default();
        state_stopped.running = false;
        state_stopped.paused = false;
        let _ = SimularTui::render_controls(&state_stopped);
    }

    #[test]
    fn test_render_metrics_all_branches() {
        // All values present
        let mut state_full = DashboardState::default();
        state_full.metrics.total_energy = Some(100.0);
        state_full.metrics.kinetic_energy = Some(60.0);
        state_full.metrics.potential_energy = Some(40.0);
        state_full.metrics.energy_drift = Some(0.001);
        let _ = SimularTui::render_metrics(&state_full);

        // No values present
        let state_empty = DashboardState::default();
        let _ = SimularTui::render_metrics(&state_empty);

        // Partial values
        let mut state_partial = DashboardState::default();
        state_partial.metrics.total_energy = Some(100.0);
        // kinetic, potential, drift are None
        let _ = SimularTui::render_metrics(&state_partial);
    }

    #[test]
    fn test_render_energy_chart_all_branches() {
        let area = ratatui::prelude::Rect::new(0, 0, 80, 24);

        // Empty data
        let state_empty = DashboardState::default();
        let _ = SimularTui::render_energy_chart(&state_empty, area);

        // With data
        let mut state_data = DashboardState::default();
        for i in 0..50 {
            state_data
                .energy_series
                .push(i as f64 * 0.1, 100.0 - i as f64 * 0.01);
        }
        let _ = SimularTui::render_energy_chart(&state_data, area);
    }

    // =========================================================================
    // Additional tests for DashboardState accessor methods
    // =========================================================================

    #[test]
    fn test_dashboard_toggle_pause_method() {
        let mut state = DashboardState::default();
        assert!(!state.is_paused());
        assert!(state.is_running());

        state.toggle_pause();
        assert!(state.is_paused());
        assert_eq!(state.status, "Paused by user");

        state.toggle_pause();
        assert!(!state.is_paused());
        assert_eq!(state.status, "Resumed");
    }

    #[test]
    fn test_dashboard_request_reset_method() {
        let mut state = DashboardState::default();
        state.request_reset();
        assert_eq!(state.status, "Reset requested");
    }

    #[test]
    fn test_dashboard_stop_method_call() {
        let mut state = DashboardState::default();
        assert!(state.is_running());
        state.stop();
        assert!(!state.is_running());
    }

    #[test]
    fn test_dashboard_set_status_method() {
        let mut state = DashboardState::default();
        state.set_status("Custom status");
        assert_eq!(state.status, "Custom status");

        state.set_status(String::from("String status"));
        assert_eq!(state.status, "String status");
    }

    #[test]
    fn test_dashboard_series_accessors() {
        let mut state = DashboardState::default();
        state.energy_series.push(0.0, 100.0);
        state.ke_series.push(0.0, 60.0);
        state.pe_series.push(0.0, 40.0);
        state.throughput_series.push(0.0, 1000.0);
        state.metrics.step = 42;

        // Test accessors
        assert_eq!(state.energy_series().len(), 1);
        assert_eq!(state.ke_series().len(), 1);
        assert_eq!(state.pe_series().len(), 1);
        assert_eq!(state.throughput_series().len(), 1);
        assert_eq!(state.metrics().step, 42);
    }

    #[test]
    fn test_format_controls_text_when_running() {
        let state = DashboardState::default();
        let text = state.format_controls_text();
        assert!(text.contains("RUNNING"));
        assert!(text.contains("[Space] Pause/Resume"));
        assert!(text.contains("[R] Reset"));
        assert!(text.contains("[Q] Quit"));
    }

    #[test]
    fn test_format_controls_text_when_paused() {
        let mut state = DashboardState::default();
        state.toggle_pause();
        let text = state.format_controls_text();
        assert!(text.contains("PAUSED"));
    }

    #[test]
    fn test_format_controls_text_when_stopped() {
        let mut state = DashboardState::default();
        state.stop();
        let text = state.format_controls_text();
        assert!(text.contains("STOPPED"));
    }
}
