//! Agent TUI Dashboard
//!
//! Interactive terminal dashboard for monitoring agent loop execution.
//! Uses presentar-terminal for rendering and crossterm for terminal control.
//!
//! Launched by `batuta agent status --tui` or during `batuta agent run --stream`.

pub(crate) use crate::agent::driver::StreamEvent;
use crate::agent::manifest::AgentManifest;
use crate::agent::phase::LoopPhase;
use crate::agent::result::{StopReason, TokenUsage};

/// Truncate a string to `max_len`, appending "..." if needed.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Snapshot of agent loop state for TUI rendering.
#[derive(Debug, Clone)]
pub struct AgentDashboardState {
    /// Agent name from manifest.
    pub agent_name: String,
    /// Current loop phase.
    pub phase: LoopPhase,
    /// Current iteration number.
    pub iteration: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Cumulative token usage.
    pub usage: TokenUsage,
    /// Token budget (None = unlimited).
    pub token_budget: Option<u64>,
    /// Tool calls executed.
    pub tool_calls: u32,
    /// Max tool calls allowed.
    pub max_tool_calls: u32,
    /// Recent text fragments.
    pub recent_text: Vec<String>,
    /// Recent tool call log.
    pub tool_log: Vec<ToolLogEntry>,
    /// Accumulated cost (USD).
    pub cost_usd: f64,
    /// Max cost budget.
    pub max_cost_usd: f64,
    /// Whether the loop is still running.
    pub running: bool,
    /// Final stop reason (if completed).
    pub stop_reason: Option<StopReason>,
}

/// A log entry for a tool call.
#[derive(Debug, Clone)]
pub struct ToolLogEntry {
    /// Tool name.
    pub name: String,
    /// Brief input summary.
    pub input_summary: String,
    /// Result summary.
    pub result_summary: String,
}

impl AgentDashboardState {
    /// Create initial state from manifest.
    pub fn from_manifest(manifest: &AgentManifest) -> Self {
        Self {
            agent_name: manifest.name.clone(),
            phase: LoopPhase::Perceive,
            iteration: 0,
            max_iterations: manifest.resources.max_iterations,
            usage: TokenUsage { input_tokens: 0, output_tokens: 0 },
            token_budget: manifest.resources.max_tokens_budget,
            tool_calls: 0,
            max_tool_calls: manifest.resources.max_tool_calls,
            recent_text: Vec::new(),
            tool_log: Vec::new(),
            cost_usd: 0.0,
            max_cost_usd: manifest.resources.max_cost_usd,
            running: true,
            stop_reason: None,
        }
    }

    /// Apply a stream event to update state.
    pub fn apply_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::PhaseChange { phase } => {
                self.phase = phase.clone();
            }
            StreamEvent::TextDelta { text } => {
                self.push_text(text);
            }
            StreamEvent::ToolUseStart { name, .. } => {
                self.push_tool_start(name);
            }
            StreamEvent::ToolUseEnd { name, result, .. } => {
                self.complete_tool(name, result);
            }
            StreamEvent::ContentComplete { stop_reason, usage } => {
                self.usage = usage.clone();
                self.stop_reason = Some(stop_reason.clone());
                self.running = false;
            }
        }
    }

    fn push_text(&mut self, text: &str) {
        self.recent_text.push(text.to_owned());
        if self.recent_text.len() > 20 {
            self.recent_text.remove(0);
        }
    }

    fn push_tool_start(&mut self, name: &str) {
        self.tool_calls += 1;
        self.tool_log.push(ToolLogEntry {
            name: name.to_owned(),
            input_summary: String::new(),
            result_summary: "running...".into(),
        });
        if self.tool_log.len() > 10 {
            self.tool_log.remove(0);
        }
    }

    fn complete_tool(&mut self, name: &str, result: &str) {
        let Some(entry) = self.tool_log.iter_mut().rev().find(|e| e.name == name) else {
            return;
        };
        entry.result_summary = truncate_str(result, 60);
    }

    /// Iteration progress as percentage (0-100).
    pub fn iteration_pct(&self) -> u32 {
        if self.max_iterations == 0 {
            return 0;
        }
        (self.iteration * 100) / self.max_iterations
    }

    /// Token budget usage percentage (0-100), or 0 if unlimited.
    pub fn token_budget_pct(&self) -> u32 {
        let Some(budget) = self.token_budget else {
            return 0;
        };
        if budget == 0 {
            return 0;
        }
        let total = self.usage.input_tokens + self.usage.output_tokens;
        ((total * 100) / budget) as u32
    }
}

// ============================================================================
// TUI rendering (feature-gated)
// ============================================================================
// TUI rendering (feature-gated behind presentar-terminal)
// ============================================================================

#[cfg(feature = "presentar-terminal")]
#[path = "tui_render.rs"]
mod tui_render;

#[cfg(feature = "presentar-terminal")]
pub use tui_render::AgentDashboard;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_state_from_manifest() {
        let manifest = AgentManifest::default();
        let state = AgentDashboardState::from_manifest(&manifest);
        assert!(state.running);
        assert_eq!(state.iteration, 0);
        assert_eq!(state.max_iterations, manifest.resources.max_iterations,);
    }

    #[test]
    fn test_apply_text_delta() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.apply_event(&StreamEvent::TextDelta { text: "hello".into() });
        assert_eq!(state.recent_text.len(), 1);
        assert_eq!(state.recent_text[0], "hello");
    }

    #[test]
    fn test_apply_content_complete() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.apply_event(&StreamEvent::ContentComplete {
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage { input_tokens: 100, output_tokens: 50 },
        });
        assert!(!state.running);
        assert_eq!(state.usage.input_tokens, 100);
        assert!(matches!(state.stop_reason, Some(StopReason::EndTurn)));
    }

    #[test]
    fn test_apply_tool_use_events() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.apply_event(&StreamEvent::ToolUseStart { id: "1".into(), name: "rag".into() });
        assert_eq!(state.tool_calls, 1);
        assert_eq!(state.tool_log.len(), 1);
        assert_eq!(state.tool_log[0].name, "rag");

        state.apply_event(&StreamEvent::ToolUseEnd {
            id: "1".into(),
            name: "rag".into(),
            result: "found 3 results".into(),
        });
        assert_eq!(state.tool_log[0].result_summary, "found 3 results",);
    }

    #[test]
    fn test_apply_phase_change() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.apply_event(&StreamEvent::PhaseChange {
            phase: LoopPhase::Act { tool_name: "rag".into() },
        });
        assert!(matches!(state.phase, LoopPhase::Act { .. }));
    }

    #[test]
    fn test_iteration_pct() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.max_iterations = 10;
        state.iteration = 3;
        assert_eq!(state.iteration_pct(), 30);
    }

    #[test]
    fn test_iteration_pct_zero_max() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.max_iterations = 0;
        assert_eq!(state.iteration_pct(), 0);
    }

    #[test]
    fn test_token_budget_pct() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        state.token_budget = Some(1000);
        state.usage = TokenUsage { input_tokens: 400, output_tokens: 100 };
        assert_eq!(state.token_budget_pct(), 50);
    }

    #[test]
    fn test_token_budget_pct_unlimited() {
        let state = AgentDashboardState::from_manifest(&AgentManifest::default());
        assert_eq!(state.token_budget_pct(), 0);
    }

    #[test]
    fn test_recent_text_capped_at_20() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        for i in 0..25 {
            state.apply_event(&StreamEvent::TextDelta { text: format!("t{i}") });
        }
        assert_eq!(state.recent_text.len(), 20);
        assert_eq!(state.recent_text[0], "t5");
    }

    #[test]
    fn test_tool_log_capped_at_10() {
        let mut state = AgentDashboardState::from_manifest(&AgentManifest::default());
        for i in 0..12 {
            state.apply_event(&StreamEvent::ToolUseStart {
                id: format!("{i}"),
                name: format!("tool_{i}"),
            });
        }
        assert_eq!(state.tool_log.len(), 10);
        assert_eq!(state.tool_log[0].name, "tool_2");
    }
}
