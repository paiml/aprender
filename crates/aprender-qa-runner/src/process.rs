//! Process lifecycle management (Jidoka)
//!
//! Implements Toyota Way Jidoka principle: stop the line and clean up,
//! never leave defects (orphan processes) in the system.
//!
//! Pattern derived from repartir's task lifecycle management.

use std::process::Child;
use std::sync::{Arc, Mutex, OnceLock};

/// Global registry of spawned child processes for cleanup
static PROCESS_REGISTRY: OnceLock<Arc<Mutex<Vec<Child>>>> = OnceLock::new();

/// Lazily initialize and return the global process registry
fn get_registry() -> &'static Arc<Mutex<Vec<Child>>> {
    PROCESS_REGISTRY.get_or_init(|| Arc::new(Mutex::new(Vec::new())))
}

/// Register a child process for tracking
#[must_use]
pub fn register_child(child: Child) -> usize {
    get_registry().lock().map_or_else(
        |e| {
            eprintln!("[JIDOKA] Process registry lock poisoned: {e}");
            0
        },
        |mut registry| {
            let idx = registry.len();
            registry.push(child);
            idx
        },
    )
}

/// Kill and reap all registered child processes (Jidoka cleanup)
///
/// Returns the number of processes cleaned up.
#[must_use]
pub fn kill_all_registered() -> usize {
    get_registry().lock().map_or_else(
        |e| {
            eprintln!("[JIDOKA] Process registry lock poisoned during cleanup: {e}");
            0
        },
        |mut registry| {
            let count = registry.len();
            for child in registry.iter_mut() {
                if let Err(e) = child.kill() {
                    eprintln!("[JIDOKA] Failed to kill child process: {e}");
                }
                if let Err(e) = child.wait() {
                    eprintln!("[JIDOKA] Failed to reap child process: {e}");
                }
            }
            registry.clear();
            count
        },
    )
}

/// RAII guard that ensures child process cleanup on drop
///
/// Implements Jidoka: if the guard is dropped without explicit completion,
/// the child process is killed and reaped.
pub struct ProcessGuard {
    child: Option<Child>,
    #[allow(dead_code)]
    pid: u32,
}

/// RAII lifecycle operations for managed child processes
impl ProcessGuard {
    /// Create a new process guard from a spawned child
    #[must_use]
    pub fn new(child: Child) -> Self {
        let pid = child.id();
        Self {
            child: Some(child),
            pid,
        }
    }

    /// Wait for the child process to complete
    ///
    /// # Errors
    ///
    /// Returns an error if the process has already been consumed or wait fails.
    pub fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.as_mut().map_or_else(
            || Err(std::io::Error::other("Process already consumed")),
            Child::wait,
        )
    }

    /// Wait for the child and collect output
    ///
    /// # Errors
    ///
    /// Returns an error if the process has already been consumed or wait fails.
    pub fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        self.child.take().map_or_else(
            || Err(std::io::Error::other("Process already consumed")),
            Child::wait_with_output,
        )
    }

    /// Take the child process, preventing automatic cleanup
    ///
    /// Use this when you want to manage the child process manually.
    #[must_use]
    pub fn take(mut self) -> Option<Child> {
        self.child.take()
    }

    /// Get the process ID
    #[must_use]
    pub const fn pid(&self) -> u32 {
        self.pid
    }
}

/// Kill and reap child process on guard drop (Jidoka cleanup)
impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            eprintln!("[JIDOKA] Cleaning up child process {}", self.pid);
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn test_process_guard_wait() {
        let child = Command::new("echo")
            .arg("test")
            .spawn()
            .expect("Failed to spawn");

        let mut guard = ProcessGuard::new(child);
        let status = guard.wait().expect("Wait failed");
        assert!(status.success());
    }

    #[test]
    fn test_process_guard_wait_with_output() {
        use std::process::Stdio;

        let child = Command::new("echo")
            .arg("hello")
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to spawn");

        let guard = ProcessGuard::new(child);
        let output = guard.wait_with_output().expect("Wait failed");
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("hello"));
    }

    #[test]
    fn test_process_guard_take() {
        let child = Command::new("echo")
            .arg("test")
            .spawn()
            .expect("Failed to spawn");

        let guard = ProcessGuard::new(child);
        let mut taken = guard.take().expect("Take failed");
        let status = taken.wait().expect("Wait failed");
        assert!(status.success());
    }

    #[test]
    fn test_process_guard_drop_kills() {
        // Spawn a sleep process that would run for 60 seconds
        let child = Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("Failed to spawn");

        let pid = child.id();
        let guard = ProcessGuard::new(child);

        // Drop the guard - should kill the process
        drop(guard);

        // Verify process is gone (this is platform-specific)
        // On Unix, we can check /proc/{pid} doesn't exist
        #[cfg(unix)]
        {
            use std::path::Path;
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(!Path::new(&format!("/proc/{pid}")).exists());
        }
    }

    #[test]
    fn test_kill_all_registered() {
        // Clear any existing entries
        let _ = kill_all_registered();

        // Register some processes
        let child1 = Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("Failed to spawn");
        let child2 = Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("Failed to spawn");

        let _ = register_child(child1);
        let _ = register_child(child2);

        // Kill all
        let count = kill_all_registered();
        assert_eq!(count, 2);

        // Registry should be empty now
        let count2 = kill_all_registered();
        assert_eq!(count2, 0);
    }

    #[test]
    fn test_process_guard_pid() {
        let child = Command::new("echo")
            .arg("test")
            .spawn()
            .expect("Failed to spawn");

        let expected_pid = child.id();
        let guard = ProcessGuard::new(child);
        assert_eq!(guard.pid(), expected_pid);
    }

    #[test]
    fn test_process_guard_wait_already_consumed() {
        let child = Command::new("echo")
            .arg("test")
            .spawn()
            .expect("Failed to spawn");

        let mut guard = ProcessGuard::new(child);
        // Take the child to consume it
        let taken = guard.child.take();
        assert!(taken.is_some());

        // Now wait should fail with "already consumed"
        let result = guard.wait();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already consumed"));
    }

    #[test]
    fn test_process_guard_wait_with_output_already_consumed() {
        use std::process::Stdio;

        let child = Command::new("echo")
            .arg("test")
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to spawn");

        let mut guard = ProcessGuard::new(child);
        // Take the child to consume it
        let taken = guard.child.take();
        assert!(taken.is_some());

        // Now wait_with_output should fail
        let result = guard.wait_with_output();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already consumed"));
    }
}
