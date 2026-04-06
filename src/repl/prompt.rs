//! Andon-style Prompt (Visual Control)
//!
//! The prompt displays immediate visual feedback about dataset health,
//! following the Andon board concept from Toyota Way.
//!
//! # Prompt Design (from ALIM-SPEC-006)
//!
//! ```text
//! # Healthy dataset (green)
//! alimentar [data.parquet: 1512 rows, A] >
//!
//! # Issues detected (yellow)
//! alimentar [data.parquet: 1512 rows, C!] >
//!
//! # Critical failures (red)
//! alimentar [data.parquet: INVALID] >
//! ```

use std::fmt::Write;

#[cfg(feature = "repl")]
use nu_ansi_term::{Color, Style};
#[cfg(feature = "repl")]
use reedline::Prompt;

use super::session::ReplSession;

/// Health status for Andon display
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Green - Grade A or B, publishable
    Healthy,
    /// Yellow - Grade C or D, needs attention
    Warning,
    /// Red - Grade F or invalid data
    Critical,
    /// No dataset loaded
    None,
}

impl HealthStatus {
    /// Create health status from letter grade
    #[must_use]
    pub fn from_grade(grade: char) -> Self {
        match grade {
            'A' | 'B' => Self::Healthy,
            'C' | 'D' => Self::Warning,
            'F' => Self::Critical,
            _ => Self::None,
        }
    }

    /// Get ANSI color for this status
    #[cfg(feature = "repl")]
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Healthy => Color::Green,
            Self::Warning => Color::Yellow,
            Self::Critical => Color::Red,
            Self::None => Color::Default,
        }
    }

    /// Get plain text indicator
    #[must_use]
    pub fn indicator(&self) -> &'static str {
        match self {
            Self::Warning => "!",
            Self::Critical => "!!",
            Self::Healthy | Self::None => "",
        }
    }
}

/// Andon-style prompt for REPL
pub struct AndonPrompt {
    /// Base prompt text
    #[allow(dead_code)]
    base: String,
}

impl Default for AndonPrompt {
    fn default() -> Self {
        Self::new()
    }
}

impl AndonPrompt {
    /// Create a new Andon prompt
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: "alimentar".to_string(),
        }
    }

    /// Render prompt string from session state (plain text)
    #[must_use]
    pub fn render(session: &ReplSession) -> String {
        let mut prompt = String::from("alimentar");

        if let Some(name) = session.active_name() {
            prompt.push_str(" [");
            prompt.push_str(&name);

            if let Some(rows) = session.active_row_count() {
                let _ = write!(prompt, ": {} rows", rows);
            }

            if let Some(grade) = session.active_grade() {
                let status =
                    HealthStatus::from_grade(grade.to_string().chars().next().unwrap_or(' '));
                let _ = write!(prompt, ", {}{}", grade, status.indicator());
            }

            prompt.push(']');
        }

        prompt.push_str(" > ");
        prompt
    }

    /// Render prompt with colors for terminal
    #[cfg(feature = "repl")]
    #[must_use]
    pub fn render_colored(session: &ReplSession) -> String {
        let mut prompt = Style::new().bold().paint("alimentar").to_string();

        if let Some(name) = session.active_name() {
            prompt.push_str(" [");
            prompt.push_str(&name);

            if let Some(rows) = session.active_row_count() {
                let _ = write!(prompt, ": {} rows", rows);
            }

            if let Some(grade) = session.active_grade() {
                let grade_char = grade.to_string().chars().next().unwrap_or(' ');
                let status = HealthStatus::from_grade(grade_char);
                let grade_colored =
                    status
                        .color()
                        .bold()
                        .paint(format!("{}{}", grade, status.indicator()));
                let _ = write!(prompt, ", {}", grade_colored);
            }

            prompt.push(']');
        }

        prompt.push_str(" > ");
        prompt
    }
}

#[cfg(feature = "repl")]
impl Prompt for AndonPrompt {
    fn render_prompt_left(&self) -> std::borrow::Cow<'_, str> {
        // Basic prompt - session state would need to be passed differently
        // for dynamic updates. This is the static fallback.
        std::borrow::Cow::Borrowed("alimentar > ")
    }

    fn render_prompt_right(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_indicator(
        &self,
        _prompt_mode: reedline::PromptEditMode,
    ) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        _history_search: reedline::PromptHistorySearch,
    ) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("(search) ")
    }
}

/// Session-aware prompt that updates based on state
#[cfg(feature = "repl")]
#[allow(dead_code)]
pub struct SessionPrompt<'a> {
    session: &'a ReplSession,
}

#[cfg(feature = "repl")]
#[allow(dead_code)]
impl<'a> SessionPrompt<'a> {
    /// Create a new session-aware prompt
    #[must_use]
    pub fn new(session: &'a ReplSession) -> Self {
        Self { session }
    }
}

#[cfg(feature = "repl")]
impl Prompt for SessionPrompt<'_> {
    fn render_prompt_left(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Owned(AndonPrompt::render_colored(self.session))
    }

    fn render_prompt_right(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_indicator(
        &self,
        _prompt_mode: reedline::PromptEditMode,
    ) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        _history_search: reedline::PromptHistorySearch,
    ) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("(search) ")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float64Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema as ArrowSchema},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::ArrowDataset;

    fn create_test_dataset() -> ArrowDataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Float64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    None,
                    Some("d"),
                    Some("e"),
                ])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap();

        ArrowDataset::new(vec![batch]).unwrap()
    }

    // HealthStatus tests
    #[test]
    fn test_health_status_from_grade_a() {
        assert_eq!(HealthStatus::from_grade('A'), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_status_from_grade_b() {
        assert_eq!(HealthStatus::from_grade('B'), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_status_from_grade_c() {
        assert_eq!(HealthStatus::from_grade('C'), HealthStatus::Warning);
    }

    #[test]
    fn test_health_status_from_grade_d() {
        assert_eq!(HealthStatus::from_grade('D'), HealthStatus::Warning);
    }

    #[test]
    fn test_health_status_from_grade_f() {
        assert_eq!(HealthStatus::from_grade('F'), HealthStatus::Critical);
    }

    #[test]
    fn test_health_status_from_grade_unknown() {
        assert_eq!(HealthStatus::from_grade('X'), HealthStatus::None);
    }

    #[test]
    fn test_health_status_indicator_healthy() {
        assert_eq!(HealthStatus::Healthy.indicator(), "");
    }

    #[test]
    fn test_health_status_indicator_warning() {
        assert_eq!(HealthStatus::Warning.indicator(), "!");
    }

    #[test]
    fn test_health_status_indicator_critical() {
        assert_eq!(HealthStatus::Critical.indicator(), "!!");
    }

    #[test]
    fn test_health_status_indicator_none() {
        assert_eq!(HealthStatus::None.indicator(), "");
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_health_status_color_healthy() {
        assert_eq!(HealthStatus::Healthy.color(), Color::Green);
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_health_status_color_warning() {
        assert_eq!(HealthStatus::Warning.color(), Color::Yellow);
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_health_status_color_critical() {
        assert_eq!(HealthStatus::Critical.color(), Color::Red);
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_health_status_color_none() {
        assert_eq!(HealthStatus::None.color(), Color::Default);
    }

    // AndonPrompt tests
    #[test]
    fn test_andon_prompt_new() {
        let prompt = AndonPrompt::new();
        assert_eq!(prompt.base, "alimentar");
    }

    #[test]
    fn test_andon_prompt_default() {
        let prompt = AndonPrompt::default();
        assert_eq!(prompt.base, "alimentar");
    }

    #[test]
    fn test_andon_prompt_render_no_dataset() {
        let session = ReplSession::new();
        let rendered = AndonPrompt::render(&session);
        assert_eq!(rendered, "alimentar > ");
    }

    #[test]
    fn test_andon_prompt_render_with_dataset() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        let rendered = AndonPrompt::render(&session);
        assert!(rendered.starts_with("alimentar [test.parquet"));
        assert!(rendered.contains("5 rows"));
        assert!(rendered.ends_with("] > "));
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_colored_no_dataset() {
        let session = ReplSession::new();
        let rendered = AndonPrompt::render_colored(&session);
        assert!(rendered.contains("alimentar"));
        assert!(rendered.ends_with(" > "));
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_colored_with_dataset() {
        let mut session = ReplSession::new();
        session.load_dataset("data.parquet", create_test_dataset());
        let rendered = AndonPrompt::render_colored(&session);
        assert!(rendered.contains("data.parquet"));
        assert!(rendered.contains("5 rows"));
    }

    // Prompt trait tests
    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_prompt_left() {
        use reedline::Prompt;
        let prompt = AndonPrompt::new();
        assert_eq!(prompt.render_prompt_left().as_ref(), "alimentar > ");
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_prompt_right() {
        use reedline::Prompt;
        let prompt = AndonPrompt::new();
        assert_eq!(prompt.render_prompt_right().as_ref(), "");
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_prompt_indicator() {
        use reedline::Prompt;
        let prompt = AndonPrompt::new();
        assert_eq!(
            prompt
                .render_prompt_indicator(reedline::PromptEditMode::Default)
                .as_ref(),
            ""
        );
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_multiline() {
        use reedline::Prompt;
        let prompt = AndonPrompt::new();
        assert_eq!(prompt.render_prompt_multiline_indicator().as_ref(), "... ");
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_history_search() {
        use reedline::Prompt;
        let prompt = AndonPrompt::new();
        let search = reedline::PromptHistorySearch::new(
            reedline::PromptHistorySearchStatus::Passing,
            "test".to_string(),
        );
        assert_eq!(
            prompt
                .render_prompt_history_search_indicator(search)
                .as_ref(),
            "(search) "
        );
    }

    // SessionPrompt tests
    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_new() {
        let session = ReplSession::new();
        let _prompt = SessionPrompt::new(&session);
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_render_prompt_left() {
        use reedline::Prompt;
        let session = ReplSession::new();
        let prompt = SessionPrompt::new(&session);
        let rendered = prompt.render_prompt_left();
        assert!(rendered.contains("alimentar"));
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_render_prompt_right() {
        use reedline::Prompt;
        let session = ReplSession::new();
        let prompt = SessionPrompt::new(&session);
        assert_eq!(prompt.render_prompt_right().as_ref(), "");
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_render_prompt_indicator() {
        use reedline::Prompt;
        let session = ReplSession::new();
        let prompt = SessionPrompt::new(&session);
        assert_eq!(
            prompt
                .render_prompt_indicator(reedline::PromptEditMode::Default)
                .as_ref(),
            ""
        );
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_render_multiline() {
        use reedline::Prompt;
        let session = ReplSession::new();
        let prompt = SessionPrompt::new(&session);
        assert_eq!(prompt.render_prompt_multiline_indicator().as_ref(), "... ");
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_render_history_search() {
        use reedline::Prompt;
        let session = ReplSession::new();
        let prompt = SessionPrompt::new(&session);
        let search = reedline::PromptHistorySearch::new(
            reedline::PromptHistorySearchStatus::Passing,
            "test".to_string(),
        );
        assert_eq!(
            prompt
                .render_prompt_history_search_indicator(search)
                .as_ref(),
            "(search) "
        );
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_session_prompt_with_dataset() {
        use reedline::Prompt;
        let mut session = ReplSession::new();
        session.load_dataset("mydata.csv", create_test_dataset());
        let prompt = SessionPrompt::new(&session);
        let rendered = prompt.render_prompt_left();
        assert!(rendered.contains("mydata.csv"));
        assert!(rendered.contains("5 rows"));
    }
}
