//! WASM Runner with Hot Reload (Advanced Feature D)
//!
//! Native WASM development server with automatic hot reload,
//! rich debugging output, and zero external dependencies.

mod builder;
mod config;
mod server;

pub use builder::{BuildCoordinator, BuildEvent, BuildResult, BuildStatus};
pub use config::{OptLevel, RunnerConfig, WasmRunnerConfig};
pub use server::{DebugOutput, HotReloadEvent, WasmRunner, WasmRunnerBuilder};

/// Default port for HTTP server
pub const DEFAULT_HTTP_PORT: u16 = 8080;
/// Default port for WebSocket server
pub const DEFAULT_WS_PORT: u16 = 8081;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-RUN-01: Runner configuration
    // =========================================================================

    #[test]
    fn h0_run_01_default_config() {
        let config = WasmRunnerConfig::default();
        assert_eq!(config.http_port, DEFAULT_HTTP_PORT);
        assert_eq!(config.ws_port, DEFAULT_WS_PORT);
        assert!(config.hot_reload);
    }

    #[test]
    fn h0_run_02_config_builder() {
        let config = WasmRunnerConfig::builder()
            .http_port(9000)
            .ws_port(9001)
            .hot_reload(false)
            .build();

        assert_eq!(config.http_port, 9000);
        assert_eq!(config.ws_port, 9001);
        assert!(!config.hot_reload);
    }

    // =========================================================================
    // H₀-RUN-03: Build coordinator
    // =========================================================================

    #[test]
    fn h0_run_03_build_result_success() {
        let result = BuildResult::success(1024, std::time::Duration::from_millis(500));
        assert!(result.is_success());
        assert_eq!(result.size_bytes(), Some(1024));
    }

    #[test]
    fn h0_run_04_build_result_failure() {
        let result = BuildResult::failure(vec!["error: test".to_string()]);
        assert!(!result.is_success());
        assert!(result.errors().is_some());
    }

    // =========================================================================
    // H₀-RUN-05: Optimization levels
    // =========================================================================

    #[test]
    fn h0_run_05_opt_levels() {
        assert_eq!(OptLevel::Debug.as_str(), "0");
        assert_eq!(OptLevel::Release.as_str(), "3");
        assert_eq!(OptLevel::Size.as_str(), "s");
        assert_eq!(OptLevel::MinSize.as_str(), "z");
    }

    // =========================================================================
    // H₀-RUN-06: Debug output configuration
    // =========================================================================

    #[test]
    fn h0_run_06_debug_output_default() {
        let output = DebugOutput::default();
        assert!(output.console);
        assert!(output.metrics);
    }

    #[test]
    fn h0_run_07_debug_output_none() {
        let output = DebugOutput::none();
        assert!(!output.console);
        assert!(!output.metrics);
        assert!(!output.network);
        assert!(!output.memory);
    }

    // =========================================================================
    // H₀-RUN-08: Hot reload events
    // =========================================================================

    #[test]
    fn h0_run_08_hot_reload_event() {
        let event = HotReloadEvent::Rebuild {
            duration: std::time::Duration::from_millis(100),
            preserved: vec!["AppState".to_string()],
        };

        match event {
            HotReloadEvent::Rebuild {
                duration,
                preserved,
            } => {
                assert_eq!(duration.as_millis(), 100);
                assert_eq!(preserved.len(), 1);
            }
            _ => panic!("Wrong event type"),
        }
    }

    // =========================================================================
    // H₀-RUN-09: Runner creation
    // =========================================================================

    #[test]
    fn h0_run_09_runner_builder() {
        let runner = WasmRunnerBuilder::new()
            .http_port(8888)
            .source_maps(true)
            .build();

        assert_eq!(runner.config().http_port, 8888);
        assert!(runner.config().source_maps);
    }
}
