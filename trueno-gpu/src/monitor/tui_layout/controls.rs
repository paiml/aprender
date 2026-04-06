//! Keyboard controls for TUI interaction.

// ============================================================================
// Keyboard Controls
// ============================================================================

/// Keyboard action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Quit application
    Quit,
    /// Force refresh
    Refresh,
    /// Toggle stress test mode
    ToggleStress,
    /// Focus next section
    FocusNext,
    /// Navigate up
    NavigateUp,
    /// Navigate down
    NavigateDown,
    /// Expand/collapse current item
    Expand,
    /// Show help overlay
    Help,
    /// Show alerts panel
    Alerts,
    /// Export metrics to JSON
    Export,
    /// Pause/resume monitoring
    TogglePause,
}

impl KeyAction {
    /// Get keyboard shortcut for this action
    #[must_use]
    pub fn key(&self) -> char {
        match self {
            Self::Quit => 'q',
            Self::Refresh => 'r',
            Self::ToggleStress => 's',
            Self::FocusNext => '\t',
            Self::NavigateUp => '\u{2191}',
            Self::NavigateDown => '\u{2193}',
            Self::Expand => '\n',
            Self::Help => '?',
            Self::Alerts => 'a',
            Self::Export => 'e',
            Self::TogglePause => 'p',
        }
    }

    /// Get description for help display
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Quit => "Quit",
            Self::Refresh => "Refresh",
            Self::ToggleStress => "Stress Test",
            Self::FocusNext => "Focus",
            Self::NavigateUp => "Up",
            Self::NavigateDown => "Down",
            Self::Expand => "Expand",
            Self::Help => "Help",
            Self::Alerts => "Alerts",
            Self::Export => "Export",
            Self::TogglePause => "Pause",
        }
    }
}
