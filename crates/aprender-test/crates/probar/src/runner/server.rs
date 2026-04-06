//! WASM Development Server
//!
//! HTTP server with hot reload support for WASM development.

use super::config::WasmRunnerConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Debug output configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DebugOutput {
    /// Show console.log from WASM
    pub console: bool,
    /// Show performance metrics
    pub metrics: bool,
    /// Show network requests
    pub network: bool,
    /// Show WASM memory usage
    pub memory: bool,
}

impl Default for DebugOutput {
    fn default() -> Self {
        Self {
            console: true,
            metrics: true,
            network: false,
            memory: false,
        }
    }
}

impl DebugOutput {
    /// Create with all output disabled
    #[must_use]
    pub fn none() -> Self {
        Self {
            console: false,
            metrics: false,
            network: false,
            memory: false,
        }
    }

    /// Create with all output enabled
    #[must_use]
    pub fn all() -> Self {
        Self {
            console: true,
            metrics: true,
            network: true,
            memory: true,
        }
    }

    /// Enable console output
    #[must_use]
    pub fn with_console(mut self) -> Self {
        self.console = true;
        self
    }

    /// Enable metrics output
    #[must_use]
    pub fn with_metrics(mut self) -> Self {
        self.metrics = true;
        self
    }

    /// Enable network output
    #[must_use]
    pub fn with_network(mut self) -> Self {
        self.network = true;
        self
    }

    /// Enable memory output
    #[must_use]
    pub fn with_memory(mut self) -> Self {
        self.memory = true;
        self
    }
}

/// Hot reload event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotReloadEvent {
    /// File changed, triggering rebuild
    FileChanged {
        /// Path to changed file
        path: String,
    },
    /// Rebuild started
    RebuildStarted,
    /// Rebuild completed successfully
    Rebuild {
        /// Build duration
        duration: Duration,
        /// State types preserved
        preserved: Vec<String>,
    },
    /// Rebuild failed
    RebuildFailed {
        /// Error messages
        errors: Vec<String>,
    },
    /// Client connected
    ClientConnected {
        /// Client ID
        id: u32,
    },
    /// Client disconnected
    ClientDisconnected {
        /// Client ID
        id: u32,
    },
}

/// Console message from WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleMessage {
    /// Log level
    pub level: ConsoleLevel,
    /// Message text
    pub text: String,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Console log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsoleLevel {
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Log level
    Log,
    /// Warning level
    Warn,
    /// Error level
    Error,
}

impl ConsoleLevel {
    /// Get the prefix for terminal output
    #[must_use]
    pub fn prefix(&self) -> &'static str {
        match self {
            ConsoleLevel::Debug => "\x1b[34m[DEBUG]\x1b[0m",
            ConsoleLevel::Info => "\x1b[36m[INFO]\x1b[0m",
            ConsoleLevel::Log => "[LOG]",
            ConsoleLevel::Warn => "\x1b[33m[WARN]\x1b[0m",
            ConsoleLevel::Error => "\x1b[31m[ERROR]\x1b[0m",
        }
    }
}

/// WASM development server runner
#[derive(Debug)]
pub struct WasmRunner {
    config: WasmRunnerConfig,
    debug_output: DebugOutput,
    running: bool,
    clients: Vec<u32>,
    next_client_id: u32,
}

impl WasmRunner {
    /// Create a new runner with configuration
    #[must_use]
    pub fn new(config: WasmRunnerConfig) -> Self {
        Self {
            config,
            debug_output: DebugOutput::default(),
            running: false,
            clients: Vec::new(),
            next_client_id: 1,
        }
    }

    /// Create a builder
    #[must_use]
    pub fn builder() -> WasmRunnerBuilder {
        WasmRunnerBuilder::default()
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &WasmRunnerConfig {
        &self.config
    }

    /// Get debug output configuration
    #[must_use]
    pub fn debug_output(&self) -> &DebugOutput {
        &self.debug_output
    }

    /// Set debug output configuration
    pub fn set_debug_output(&mut self, output: DebugOutput) {
        self.debug_output = output;
    }

    /// Check if server is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get connected client count
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Simulate starting the server (for testing)
    pub fn simulate_start(&mut self) {
        self.running = true;
    }

    /// Simulate stopping the server (for testing)
    pub fn simulate_stop(&mut self) {
        self.running = false;
        self.clients.clear();
    }

    /// Simulate client connection (for testing)
    pub fn simulate_client_connect(&mut self) -> u32 {
        let id = self.next_client_id;
        self.next_client_id += 1;
        self.clients.push(id);
        id
    }

    /// Simulate client disconnection (for testing)
    pub fn simulate_client_disconnect(&mut self, id: u32) {
        self.clients.retain(|&c| c != id);
    }

    /// Format a console message for terminal output
    #[must_use]
    pub fn format_console_message(&self, msg: &ConsoleMessage) -> String {
        // Platform-independent timestamp formatting
        #[cfg(not(target_arch = "wasm32"))]
        {
            use chrono::{DateTime, Local};
            let timestamp: DateTime<Local> = msg.timestamp.into();
            format!(
                "[{}] {} {}",
                timestamp.format("%H:%M:%S"),
                msg.level.prefix(),
                msg.text
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            // On WASM, use epoch seconds as timestamp
            let secs = msg
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("[{}] {} {}", secs, msg.level.prefix(), msg.text)
        }
    }

    /// Get server URL
    #[must_use]
    pub fn http_url(&self) -> String {
        format!("http://localhost:{}", self.config.http_port)
    }

    /// Get WebSocket URL
    #[must_use]
    pub fn ws_url(&self) -> String {
        format!("ws://localhost:{}", self.config.ws_port)
    }
}

/// Builder for `WasmRunner`
#[derive(Debug, Clone, Default)]
pub struct WasmRunnerBuilder {
    config: WasmRunnerConfig,
    debug_output: DebugOutput,
}

impl WasmRunnerBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set HTTP port
    #[must_use]
    pub fn http_port(mut self, port: u16) -> Self {
        self.config.http_port = port;
        self
    }

    /// Set WebSocket port
    #[must_use]
    pub fn ws_port(mut self, port: u16) -> Self {
        self.config.ws_port = port;
        self
    }

    /// Enable/disable hot reload
    #[must_use]
    pub fn hot_reload(mut self, enabled: bool) -> Self {
        self.config.hot_reload = enabled;
        self
    }

    /// Enable/disable state preservation
    #[must_use]
    pub fn preserve_state(mut self, enabled: bool) -> Self {
        self.config.preserve_state = enabled;
        self
    }

    /// Enable/disable source maps
    #[must_use]
    pub fn source_maps(mut self, enabled: bool) -> Self {
        self.config.source_maps = enabled;
        self
    }

    /// Set debug output configuration
    #[must_use]
    pub fn debug_output(mut self, output: DebugOutput) -> Self {
        self.debug_output = output;
        self
    }

    /// Build the runner
    #[must_use]
    pub fn build(self) -> WasmRunner {
        let mut runner = WasmRunner::new(self.config);
        runner.debug_output = self.debug_output;
        runner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_output_default() {
        let output = DebugOutput::default();
        assert!(output.console);
        assert!(output.metrics);
        assert!(!output.network);
        assert!(!output.memory);
    }

    #[test]
    fn test_debug_output_none() {
        let output = DebugOutput::none();
        assert!(!output.console);
        assert!(!output.metrics);
        assert!(!output.network);
        assert!(!output.memory);
    }

    #[test]
    fn test_debug_output_all() {
        let output = DebugOutput::all();
        assert!(output.console);
        assert!(output.metrics);
        assert!(output.network);
        assert!(output.memory);
    }

    #[test]
    fn test_debug_output_builders() {
        let output = DebugOutput::none().with_console().with_network();

        assert!(output.console);
        assert!(!output.metrics);
        assert!(output.network);
        assert!(!output.memory);
    }

    #[test]
    fn test_console_level_prefix() {
        assert!(ConsoleLevel::Error.prefix().contains("[ERROR]"));
        assert!(ConsoleLevel::Warn.prefix().contains("[WARN]"));
        assert!(ConsoleLevel::Info.prefix().contains("[INFO]"));
    }

    #[test]
    fn test_wasm_runner_new() {
        let config = WasmRunnerConfig::default();
        let runner = WasmRunner::new(config);
        assert!(!runner.is_running());
        assert_eq!(runner.client_count(), 0);
    }

    #[test]
    fn test_wasm_runner_builder() {
        let runner = WasmRunnerBuilder::new()
            .http_port(9000)
            .ws_port(9001)
            .hot_reload(false)
            .build();

        assert_eq!(runner.config().http_port, 9000);
        assert_eq!(runner.config().ws_port, 9001);
        assert!(!runner.config().hot_reload);
    }

    #[test]
    fn test_wasm_runner_urls() {
        let runner = WasmRunnerBuilder::new()
            .http_port(8080)
            .ws_port(8081)
            .build();

        assert_eq!(runner.http_url(), "http://localhost:8080");
        assert_eq!(runner.ws_url(), "ws://localhost:8081");
    }

    #[test]
    fn test_wasm_runner_simulate() {
        let mut runner = WasmRunner::new(WasmRunnerConfig::default());

        runner.simulate_start();
        assert!(runner.is_running());

        let id1 = runner.simulate_client_connect();
        let id2 = runner.simulate_client_connect();
        assert_eq!(runner.client_count(), 2);

        runner.simulate_client_disconnect(id1);
        assert_eq!(runner.client_count(), 1);

        runner.simulate_stop();
        assert!(!runner.is_running());
        assert_eq!(runner.client_count(), 0);

        // Ensure IDs are unique
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_hot_reload_event_variants() {
        let event = HotReloadEvent::FileChanged {
            path: "src/main.rs".to_string(),
        };
        assert!(matches!(event, HotReloadEvent::FileChanged { .. }));

        let event = HotReloadEvent::Rebuild {
            duration: Duration::from_millis(500),
            preserved: vec!["State".to_string()],
        };
        assert!(matches!(event, HotReloadEvent::Rebuild { .. }));
    }

    #[test]
    fn test_console_message() {
        let msg = ConsoleMessage {
            level: ConsoleLevel::Info,
            text: "Test message".to_string(),
            timestamp: std::time::SystemTime::now(),
        };

        let runner = WasmRunner::new(WasmRunnerConfig::default());
        let formatted = runner.format_console_message(&msg);
        assert!(formatted.contains("[INFO]"));
        assert!(formatted.contains("Test message"));
    }

    #[test]
    fn test_debug_output_with_metrics() {
        let output = DebugOutput::none().with_metrics();
        assert!(!output.console);
        assert!(output.metrics);
        assert!(!output.network);
        assert!(!output.memory);
    }

    #[test]
    fn test_debug_output_with_memory() {
        let output = DebugOutput::none().with_memory();
        assert!(!output.console);
        assert!(!output.metrics);
        assert!(!output.network);
        assert!(output.memory);
    }

    #[test]
    fn test_debug_output_chaining_all() {
        let output = DebugOutput::none()
            .with_console()
            .with_metrics()
            .with_network()
            .with_memory();

        assert!(output.console);
        assert!(output.metrics);
        assert!(output.network);
        assert!(output.memory);
    }

    #[test]
    fn test_console_level_debug_prefix() {
        let prefix = ConsoleLevel::Debug.prefix();
        assert!(prefix.contains("[DEBUG]"));
    }

    #[test]
    fn test_console_level_log_prefix() {
        let prefix = ConsoleLevel::Log.prefix();
        assert!(prefix.contains("[LOG]"));
    }

    #[test]
    fn test_console_level_all_prefixes() {
        // Test all variants have appropriate prefixes
        assert!(ConsoleLevel::Debug.prefix().contains("[DEBUG]"));
        assert!(ConsoleLevel::Info.prefix().contains("[INFO]"));
        assert!(ConsoleLevel::Log.prefix().contains("[LOG]"));
        assert!(ConsoleLevel::Warn.prefix().contains("[WARN]"));
        assert!(ConsoleLevel::Error.prefix().contains("[ERROR]"));
    }

    #[test]
    fn test_wasm_runner_debug_output_accessor() {
        let runner = WasmRunner::new(WasmRunnerConfig::default());
        let debug_output = runner.debug_output();

        // Default debug output has console and metrics enabled
        assert!(debug_output.console);
        assert!(debug_output.metrics);
        assert!(!debug_output.network);
        assert!(!debug_output.memory);
    }

    #[test]
    fn test_wasm_runner_set_debug_output() {
        let mut runner = WasmRunner::new(WasmRunnerConfig::default());

        // Set to all enabled
        runner.set_debug_output(DebugOutput::all());
        assert!(runner.debug_output().console);
        assert!(runner.debug_output().metrics);
        assert!(runner.debug_output().network);
        assert!(runner.debug_output().memory);

        // Set to none
        runner.set_debug_output(DebugOutput::none());
        assert!(!runner.debug_output().console);
        assert!(!runner.debug_output().metrics);
        assert!(!runner.debug_output().network);
        assert!(!runner.debug_output().memory);
    }

    #[test]
    fn test_wasm_runner_builder_preserve_state() {
        let runner = WasmRunnerBuilder::new().preserve_state(false).build();
        assert!(!runner.config().preserve_state);

        let runner2 = WasmRunnerBuilder::new().preserve_state(true).build();
        assert!(runner2.config().preserve_state);
    }

    #[test]
    fn test_wasm_runner_builder_source_maps() {
        let runner = WasmRunnerBuilder::new().source_maps(false).build();
        assert!(!runner.config().source_maps);

        let runner2 = WasmRunnerBuilder::new().source_maps(true).build();
        assert!(runner2.config().source_maps);
    }

    #[test]
    fn test_wasm_runner_builder_debug_output() {
        let runner = WasmRunnerBuilder::new()
            .debug_output(DebugOutput::all())
            .build();

        assert!(runner.debug_output().console);
        assert!(runner.debug_output().metrics);
        assert!(runner.debug_output().network);
        assert!(runner.debug_output().memory);
    }

    #[test]
    fn test_hot_reload_event_rebuild_started() {
        let event = HotReloadEvent::RebuildStarted;
        assert!(matches!(event, HotReloadEvent::RebuildStarted));
    }

    #[test]
    fn test_hot_reload_event_rebuild_failed() {
        let event = HotReloadEvent::RebuildFailed {
            errors: vec![
                "error[E0599]: no method named `foo`".to_string(),
                "error: aborting due to previous error".to_string(),
            ],
        };
        if let HotReloadEvent::RebuildFailed { errors } = event {
            assert_eq!(errors.len(), 2);
            assert!(errors[0].contains("E0599"));
        } else {
            panic!("Expected HotReloadEvent::RebuildFailed");
        }
    }

    #[test]
    fn test_hot_reload_event_client_connected() {
        let event = HotReloadEvent::ClientConnected { id: 42 };
        if let HotReloadEvent::ClientConnected { id } = event {
            assert_eq!(id, 42);
        } else {
            panic!("Expected HotReloadEvent::ClientConnected");
        }
    }

    #[test]
    fn test_hot_reload_event_client_disconnected() {
        let event = HotReloadEvent::ClientDisconnected { id: 99 };
        if let HotReloadEvent::ClientDisconnected { id } = event {
            assert_eq!(id, 99);
        } else {
            panic!("Expected HotReloadEvent::ClientDisconnected");
        }
    }

    #[test]
    fn test_wasm_runner_builder_new_vs_default() {
        let builder1 = WasmRunnerBuilder::new();
        let builder2 = WasmRunnerBuilder::default();

        let runner1 = builder1.build();
        let runner2 = builder2.build();

        assert_eq!(runner1.config().http_port, runner2.config().http_port);
        assert_eq!(runner1.config().ws_port, runner2.config().ws_port);
    }

    #[test]
    fn test_wasm_runner_config_accessor() {
        let runner = WasmRunnerBuilder::new()
            .http_port(5000)
            .ws_port(5001)
            .hot_reload(true)
            .build();

        let config = runner.config();
        assert_eq!(config.http_port, 5000);
        assert_eq!(config.ws_port, 5001);
        assert!(config.hot_reload);
    }

    #[test]
    fn test_wasm_runner_simulate_multiple_clients() {
        let mut runner = WasmRunner::new(WasmRunnerConfig::default());
        runner.simulate_start();

        // Connect multiple clients
        let id1 = runner.simulate_client_connect();
        let id2 = runner.simulate_client_connect();
        let id3 = runner.simulate_client_connect();

        assert_eq!(runner.client_count(), 3);

        // IDs should be sequential and unique
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);

        // Disconnect middle client
        runner.simulate_client_disconnect(id2);
        assert_eq!(runner.client_count(), 2);

        // Disconnect non-existent client (should be no-op)
        runner.simulate_client_disconnect(999);
        assert_eq!(runner.client_count(), 2);
    }

    #[test]
    fn test_console_message_all_levels() {
        let runner = WasmRunner::new(WasmRunnerConfig::default());

        for level in [
            ConsoleLevel::Debug,
            ConsoleLevel::Info,
            ConsoleLevel::Log,
            ConsoleLevel::Warn,
            ConsoleLevel::Error,
        ] {
            let msg = ConsoleMessage {
                level,
                text: format!("Message at {:?} level", level),
                timestamp: std::time::SystemTime::now(),
            };
            let formatted = runner.format_console_message(&msg);
            assert!(formatted.contains(&msg.text));
        }
    }

    #[test]
    fn test_hot_reload_event_serialization() {
        let event = HotReloadEvent::Rebuild {
            duration: Duration::from_millis(1500),
            preserved: vec!["GameState".to_string(), "PlayerData".to_string()],
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("GameState"));
        assert!(json.contains("PlayerData"));
    }

    #[test]
    fn test_console_message_serialization() {
        let msg = ConsoleMessage {
            level: ConsoleLevel::Warn,
            text: "Warning message".to_string(),
            timestamp: std::time::SystemTime::now(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: ConsoleMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.level, ConsoleLevel::Warn);
        assert_eq!(deserialized.text, "Warning message");
    }

    #[test]
    fn test_debug_output_serialization() {
        let output = DebugOutput::all();
        let json = serde_json::to_string(&output).unwrap();
        let deserialized: DebugOutput = serde_json::from_str(&json).unwrap();

        assert!(deserialized.console);
        assert!(deserialized.metrics);
        assert!(deserialized.network);
        assert!(deserialized.memory);
    }

    #[test]
    fn test_wasm_runner_builder_all_options() {
        let runner = WasmRunnerBuilder::new()
            .http_port(7000)
            .ws_port(7001)
            .hot_reload(false)
            .preserve_state(false)
            .source_maps(false)
            .debug_output(DebugOutput::none())
            .build();

        assert_eq!(runner.config().http_port, 7000);
        assert_eq!(runner.config().ws_port, 7001);
        assert!(!runner.config().hot_reload);
        assert!(!runner.config().preserve_state);
        assert!(!runner.config().source_maps);
        assert!(!runner.debug_output().console);
    }

    #[test]
    fn test_console_level_equality() {
        assert_eq!(ConsoleLevel::Debug, ConsoleLevel::Debug);
        assert_eq!(ConsoleLevel::Info, ConsoleLevel::Info);
        assert_eq!(ConsoleLevel::Log, ConsoleLevel::Log);
        assert_eq!(ConsoleLevel::Warn, ConsoleLevel::Warn);
        assert_eq!(ConsoleLevel::Error, ConsoleLevel::Error);

        assert_ne!(ConsoleLevel::Debug, ConsoleLevel::Error);
        assert_ne!(ConsoleLevel::Info, ConsoleLevel::Warn);
    }

    #[test]
    fn test_wasm_runner_initial_state() {
        let runner = WasmRunner::new(WasmRunnerConfig::default());

        assert!(!runner.is_running());
        assert_eq!(runner.client_count(), 0);
    }

    #[test]
    fn test_wasm_runner_url_formats() {
        let runner = WasmRunnerBuilder::new()
            .http_port(3000)
            .ws_port(3001)
            .build();

        assert!(runner.http_url().starts_with("http://"));
        assert!(runner.http_url().contains("localhost"));
        assert!(runner.http_url().contains("3000"));

        assert!(runner.ws_url().starts_with("ws://"));
        assert!(runner.ws_url().contains("localhost"));
        assert!(runner.ws_url().contains("3001"));
    }

    #[test]
    fn test_hot_reload_event_file_changed() {
        let event = HotReloadEvent::FileChanged {
            path: "src/game.rs".to_string(),
        };
        if let HotReloadEvent::FileChanged { path } = event {
            assert_eq!(path, "src/game.rs");
        } else {
            panic!("Expected HotReloadEvent::FileChanged");
        }
    }

    #[test]
    fn test_hot_reload_event_rebuild_details() {
        let event = HotReloadEvent::Rebuild {
            duration: Duration::from_secs(2),
            preserved: vec!["AppState".to_string()],
        };
        if let HotReloadEvent::Rebuild {
            duration,
            preserved,
        } = event
        {
            assert_eq!(duration, Duration::from_secs(2));
            assert_eq!(preserved.len(), 1);
            assert_eq!(preserved[0], "AppState");
        } else {
            panic!("Expected HotReloadEvent::Rebuild");
        }
    }

    #[test]
    fn test_wasm_runner_stop_clears_clients() {
        let mut runner = WasmRunner::new(WasmRunnerConfig::default());
        runner.simulate_start();

        runner.simulate_client_connect();
        runner.simulate_client_connect();
        assert_eq!(runner.client_count(), 2);

        runner.simulate_stop();
        assert!(!runner.is_running());
        assert_eq!(runner.client_count(), 0);
    }

    #[test]
    fn test_wasm_runner_builder_default_debug_output() {
        let runner = WasmRunnerBuilder::default().build();

        // Default is DebugOutput::default() which has console and metrics enabled
        assert!(runner.debug_output().console);
        assert!(runner.debug_output().metrics);
        assert!(!runner.debug_output().network);
        assert!(!runner.debug_output().memory);
    }

    #[test]
    fn test_console_level_serialization_all_variants() {
        for level in [
            ConsoleLevel::Debug,
            ConsoleLevel::Info,
            ConsoleLevel::Log,
            ConsoleLevel::Warn,
            ConsoleLevel::Error,
        ] {
            let json = serde_json::to_string(&level).unwrap();
            let deserialized: ConsoleLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, deserialized);
        }
    }

    #[test]
    fn test_hot_reload_event_all_variants_serialization() {
        // FileChanged
        let event1 = HotReloadEvent::FileChanged {
            path: "test.rs".to_string(),
        };
        let json1 = serde_json::to_string(&event1).unwrap();
        assert!(json1.contains("test.rs"));

        // RebuildStarted
        let event2 = HotReloadEvent::RebuildStarted;
        let json2 = serde_json::to_string(&event2).unwrap();
        assert!(json2.contains("RebuildStarted"));

        // RebuildFailed
        let event3 = HotReloadEvent::RebuildFailed {
            errors: vec!["error".to_string()],
        };
        let json3 = serde_json::to_string(&event3).unwrap();
        assert!(json3.contains("error"));

        // ClientConnected
        let event4 = HotReloadEvent::ClientConnected { id: 1 };
        let json4 = serde_json::to_string(&event4).unwrap();
        assert!(json4.contains('1'));

        // ClientDisconnected
        let event5 = HotReloadEvent::ClientDisconnected { id: 2 };
        let json5 = serde_json::to_string(&event5).unwrap();
        assert!(json5.contains('2'));
    }
}
