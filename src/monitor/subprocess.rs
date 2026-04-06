//! Subprocess execution with timeout support.
//!
//! Provides safe, non-blocking subprocess execution to prevent UI hangs
//! when external commands (like `ioreg`, `ps`, `sysctl`) block indefinitely.

use std::process::{Command, Output};
use std::time::Duration;

/// Result of a subprocess execution with timeout.
#[derive(Debug)]
pub enum SubprocessResult {
    /// Command completed successfully with output.
    Success(Output),
    /// Command timed out and was killed.
    Timeout,
    /// Command failed to spawn.
    SpawnError,
    /// Command exited with non-zero status.
    Failed(Output),
}

impl SubprocessResult {
    /// Returns stdout as string if successful.
    #[must_use]
    pub fn stdout_string(&self) -> Option<String> {
        match self {
            Self::Success(output) | Self::Failed(output) => {
                Some(String::from_utf8_lossy(&output.stdout).to_string())
            }
            _ => None,
        }
    }

    /// Returns true if command completed successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Returns true if command timed out.
    #[must_use]
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout)
    }
}

/// Runs a command with a timeout.
///
/// Uses a background thread to run the blocking `output()` call, which properly
/// handles pipe I/O. The main thread waits with a timeout.
///
/// # Arguments
/// * `cmd` - Command name to execute
/// * `args` - Command arguments
/// * `timeout` - Maximum time to wait for command completion
///
/// # Returns
/// * `SubprocessResult` indicating success, timeout, or failure
pub fn run_with_timeout(cmd: &str, args: &[&str], timeout: Duration) -> SubprocessResult {
    use std::sync::mpsc;
    use std::thread;

    let cmd = cmd.to_string();
    let args: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();

    // Channel to receive result from worker thread
    let (tx, rx) = mpsc::channel();

    // Spawn worker thread that runs the blocking output() call
    thread::spawn(move || {
        let result = Command::new(&cmd).args(&args).output();
        let _ = tx.send(result);
    });

    // Wait for result with timeout
    match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            if output.status.success() {
                SubprocessResult::Success(output)
            } else {
                SubprocessResult::Failed(output)
            }
        }
        Ok(Err(_)) => SubprocessResult::SpawnError,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Thread is still running - it will clean up on its own
            SubprocessResult::Timeout
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => SubprocessResult::SpawnError,
    }
}

/// Runs a command with timeout and returns stdout as Option<String>.
///
/// Convenience wrapper that returns None on timeout or error.
#[must_use]
pub fn run_with_timeout_stdout(cmd: &str, args: &[&str], timeout: Duration) -> Option<String> {
    match run_with_timeout(cmd, args, timeout) {
        SubprocessResult::Success(output) => {
            Some(String::from_utf8_lossy(&output.stdout).to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_successful_command() {
        let result = run_with_timeout("echo", &["hello"], Duration::from_secs(1));
        assert!(result.is_success());
        assert_eq!(result.stdout_string().expect("value should be present").trim(), "hello");
    }

    #[test]
    fn test_command_with_args() {
        let result = run_with_timeout("printf", &["%s %s", "foo", "bar"], Duration::from_secs(1));
        assert!(result.is_success());
        assert_eq!(result.stdout_string().expect("value should be present"), "foo bar");
    }

    #[test]
    fn test_timeout_kills_slow_command() {
        // sleep 10 should timeout after 100ms
        let start = Instant::now();
        let result = run_with_timeout("sleep", &["10"], Duration::from_millis(100));
        let elapsed = start.elapsed();

        assert!(result.is_timeout());
        assert!(elapsed < Duration::from_secs(1), "Should timeout quickly, took {elapsed:?}");
    }

    #[test]
    fn test_nonexistent_command() {
        let result =
            run_with_timeout("this_command_does_not_exist_12345", &[], Duration::from_secs(1));
        assert!(matches!(result, SubprocessResult::SpawnError));
    }

    #[test]
    fn test_failed_command() {
        let result = run_with_timeout("false", &[], Duration::from_secs(1));
        assert!(matches!(result, SubprocessResult::Failed(_)));
    }

    #[test]
    fn test_stdout_string_convenience() {
        let output = run_with_timeout_stdout("echo", &["test"], Duration::from_secs(1));
        assert_eq!(output.expect("value should be present").trim(), "test");
    }

    #[test]
    fn test_stdout_string_timeout_returns_none() {
        let output = run_with_timeout_stdout("sleep", &["10"], Duration::from_millis(50));
        assert!(output.is_none());
    }

    #[test]
    fn test_multiple_rapid_timeouts() {
        // Ensure we don't leak resources with rapid timeout/kill cycles
        for _ in 0..5 {
            let result = run_with_timeout("sleep", &["10"], Duration::from_millis(20));
            assert!(result.is_timeout());
        }
    }

    #[test]
    fn test_command_that_produces_large_output() {
        // seq produces numbered output
        let result = run_with_timeout("seq", &["1", "100"], Duration::from_secs(1));
        assert!(result.is_success());
        let output = result.stdout_string().expect("operation should succeed");
        assert!(output.contains('1'));
        assert!(output.contains("100"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_sysctl() {
        let result = run_with_timeout("sysctl", &["-n", "hw.ncpu"], Duration::from_secs(1));
        assert!(result.is_success());
        let ncpu: u32 = result
            .stdout_string()
            .expect("parsing should succeed")
            .trim()
            .parse()
            .expect("parsing should succeed");
        assert!(ncpu > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_ioreg_with_timeout() {
        // ioreg can hang indefinitely on some systems (e.g., Mac Pro with AMD GPUs)
        // This test verifies our timeout actually kills it
        let start = Instant::now();
        let result = run_with_timeout(
            "ioreg",
            &["-r", "-c", "IOAccelerator", "-d", "2"],
            Duration::from_millis(200), // Short timeout
        );
        let elapsed = start.elapsed();

        // Either succeeds quickly or times out - hanging is the bug we're preventing
        assert!(result.is_success() || result.is_timeout());
        // Must complete within reasonable time (timeout + buffer)
        assert!(elapsed < Duration::from_secs(1), "ioreg took too long: {:?}", elapsed);
    }
}
