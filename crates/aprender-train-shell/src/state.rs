//! Session state management for the interactive shell.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Session state that persists across commands.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SessionState {
    /// Currently loaded models
    models: HashMap<String, LoadedModel>,
    /// Command history
    history: Vec<HistoryEntry>,
    /// User preferences
    preferences: Preferences,
    /// Session metrics
    metrics: SessionMetrics,
}

impl SessionState {
    /// Create a new empty session state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get loaded models.
    pub fn loaded_models(&self) -> &HashMap<String, LoadedModel> {
        &self.models
    }

    /// Get command history.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Add a model to the session.
    pub fn add_model(&mut self, name: String, model: LoadedModel) {
        self.models.insert(name, model);
    }

    /// Remove a model from the session.
    pub fn remove_model(&mut self, name: &str) -> Option<LoadedModel> {
        self.models.remove(name)
    }

    /// Get a model by name.
    pub fn get_model(&self, name: &str) -> Option<&LoadedModel> {
        self.models.get(name)
    }

    /// Add a command to history.
    pub fn add_to_history(&mut self, entry: HistoryEntry) {
        self.history.push(entry);
    }

    /// Get mutable preferences.
    pub fn preferences_mut(&mut self) -> &mut Preferences {
        &mut self.preferences
    }

    /// Get preferences.
    pub fn preferences(&self) -> &Preferences {
        &self.preferences
    }

    /// Get session metrics.
    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    /// Update metrics after a command.
    pub fn record_command(&mut self, duration_ms: u64, success: bool) {
        self.metrics.total_commands += 1;
        if success {
            self.metrics.successful_commands += 1;
        }
        self.metrics.total_duration_ms += duration_ms;
    }

    /// Save state to a file.
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load state from a file.
    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// A loaded model in the session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadedModel {
    /// Model identifier (HuggingFace ID or path)
    pub id: String,
    /// Local path to cached model
    pub path: PathBuf,
    /// Model architecture
    pub architecture: String,
    /// Number of parameters
    pub parameters: u64,
    /// Number of layers
    pub layers: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Role in session (teacher/student)
    pub role: ModelRole,
}

/// Role of a model in the distillation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelRole {
    /// Teacher model (knowledge source)
    Teacher,
    /// Student model (learning target)
    Student,
    /// No specific role assigned
    #[default]
    None,
}

/// A history entry for a command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    /// The command string
    pub command: String,
    /// Execution timestamp (Unix seconds)
    pub timestamp: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether the command succeeded
    pub success: bool,
}

impl HistoryEntry {
    /// Create a new history entry.
    pub fn new(command: impl Into<String>, duration_ms: u64, success: bool) -> Self {
        Self {
            command: command.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            duration_ms,
            success,
        }
    }
}

/// User preferences.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Preferences {
    /// Default output format
    pub output_format: String,
    /// Whether to show progress bars
    pub show_progress: bool,
    /// Whether to save history automatically
    pub auto_save_history: bool,
    /// Default batch size for operations
    pub default_batch_size: u32,
    /// Default sequence length
    pub default_seq_len: usize,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            output_format: "table".to_string(),
            show_progress: true,
            auto_save_history: true,
            default_batch_size: 32,
            default_seq_len: 512,
        }
    }
}

/// Session-level metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SessionMetrics {
    /// Total commands executed
    pub total_commands: u64,
    /// Successful commands
    pub successful_commands: u64,
    /// Total duration in milliseconds
    pub total_duration_ms: u64,
}

impl SessionMetrics {
    /// Get success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_commands == 0 {
            100.0
        } else {
            (self.successful_commands as f64 / self.total_commands as f64) * 100.0
        }
    }

    /// Get average command duration in milliseconds.
    pub fn avg_duration_ms(&self) -> f64 {
        if self.total_commands == 0 {
            0.0
        } else {
            self.total_duration_ms as f64 / self.total_commands as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_model_management() {
        let mut state = SessionState::new();

        let model = LoadedModel {
            id: "test/model".to_string(),
            path: PathBuf::from("/tmp/model"),
            architecture: "llama".to_string(),
            parameters: 7_000_000_000,
            layers: 32,
            hidden_dim: 4096,
            role: ModelRole::Teacher,
        };

        state.add_model("teacher".to_string(), model.clone());
        assert_eq!(state.loaded_models().len(), 1);
        assert!(state.get_model("teacher").is_some());

        state.remove_model("teacher");
        assert!(state.get_model("teacher").is_none());
    }

    #[test]
    fn test_session_state_history() {
        let mut state = SessionState::new();

        state.add_to_history(HistoryEntry::new("fetch model", 100, true));
        state.add_to_history(HistoryEntry::new("inspect layers", 50, true));

        assert_eq!(state.history().len(), 2);
        assert_eq!(state.history()[0].command, "fetch model");
    }

    #[test]
    fn test_session_metrics() {
        let mut state = SessionState::new();

        state.record_command(100, true);
        state.record_command(200, true);
        state.record_command(150, false);

        assert_eq!(state.metrics().total_commands, 3);
        assert_eq!(state.metrics().successful_commands, 2);
        assert!((state.metrics().success_rate() - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_session_state_serialization_roundtrip() {
        let mut state = SessionState::new();
        state.add_to_history(HistoryEntry::new("test", 100, true));
        state.preferences_mut().default_batch_size = 64;

        let json = serde_json::to_string(&state).expect("JSON serialization should succeed");
        let restored: SessionState =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(state, restored);
    }

    #[test]
    fn test_model_role_default() {
        assert_eq!(ModelRole::default(), ModelRole::None);
    }

    #[test]
    fn test_preferences_default_values() {
        let prefs = Preferences::default();
        assert_eq!(prefs.output_format, "table");
        assert!(prefs.show_progress);
        assert_eq!(prefs.default_batch_size, 32);
    }

    #[test]
    fn test_session_metrics_success_rate_zero() {
        let metrics = SessionMetrics::default();
        assert_eq!(metrics.success_rate(), 100.0);
    }

    #[test]
    fn test_session_metrics_avg_duration_zero() {
        let metrics = SessionMetrics::default();
        assert_eq!(metrics.avg_duration_ms(), 0.0);
    }

    #[test]
    fn test_session_metrics_avg_duration() {
        let mut state = SessionState::new();
        state.record_command(100, true);
        state.record_command(200, true);
        assert_eq!(state.metrics().avg_duration_ms(), 150.0);
    }

    #[test]
    fn test_history_entry_new() {
        let entry = HistoryEntry::new("test command", 50, true);
        assert_eq!(entry.command, "test command");
        assert_eq!(entry.duration_ms, 50);
        assert!(entry.success);
        assert!(entry.timestamp > 0);
    }

    #[test]
    fn test_loaded_model_equality() {
        let model1 = LoadedModel {
            id: "test".to_string(),
            path: PathBuf::from("/tmp"),
            architecture: "llama".to_string(),
            parameters: 7_000_000_000,
            layers: 32,
            hidden_dim: 4096,
            role: ModelRole::None,
        };
        let model2 = model1.clone();
        assert_eq!(model1, model2);
    }

    #[test]
    fn test_model_role_equality() {
        assert_eq!(ModelRole::Teacher, ModelRole::Teacher);
        assert_ne!(ModelRole::Teacher, ModelRole::Student);
        assert_ne!(ModelRole::Student, ModelRole::None);
    }

    #[test]
    fn test_session_state_save_load() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let state_path = temp_dir.path().join("state.json");

        let mut state = SessionState::new();
        state.add_to_history(HistoryEntry::new("test", 100, true));
        state.preferences_mut().default_batch_size = 128;

        state.save(&state_path).expect("save should succeed");
        let loaded = SessionState::load(&state_path).expect("load should succeed");

        assert_eq!(state, loaded);
    }

    #[test]
    fn test_session_state_load_invalid_json() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut file = NamedTempFile::new().expect("temp file creation should succeed");
        file.write_all(b"not valid json").expect("file write should succeed");

        let result = SessionState::load(&file.path().to_path_buf());
        assert!(result.is_err());
    }

    #[test]
    fn test_preferences_all_fields() {
        let prefs = Preferences::default();
        assert_eq!(prefs.output_format, "table");
        assert!(prefs.show_progress);
        assert!(prefs.auto_save_history);
        assert_eq!(prefs.default_batch_size, 32);
        assert_eq!(prefs.default_seq_len, 512);
    }

    #[test]
    fn test_session_state_remove_nonexistent() {
        let mut state = SessionState::new();
        let result = state.remove_model("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_session_state_get_nonexistent() {
        let state = SessionState::new();
        assert!(state.get_model("nonexistent").is_none());
    }

    #[test]
    fn test_session_metrics_fields() {
        let metrics =
            SessionMetrics { total_commands: 10, successful_commands: 8, total_duration_ms: 1000 };
        assert_eq!(metrics.success_rate(), 80.0);
        assert_eq!(metrics.avg_duration_ms(), 100.0);
    }
}
