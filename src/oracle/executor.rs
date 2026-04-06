//! Code execution backends for verification
//!
//! Provides executors for running code in different languages
//! with proper I/O capture and timeout handling.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{Error, Language, Result};

use super::ExecutionResult;

/// Code executor trait for running programs
pub trait Executor: Send + Sync {
    /// Execute code with the given input
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    fn execute(&self, code: &str, input: &str, timeout_ms: u64) -> Result<ExecutionResult>;

    /// Get the language this executor handles
    fn language(&self) -> Language;
}

/// Python code executor using system Python interpreter
#[derive(Debug, Default)]
pub struct PythonExecutor {
    /// Path to Python interpreter (default: "python3")
    interpreter: String,
}

impl PythonExecutor {
    /// Create a new Python executor with default interpreter
    #[must_use]
    pub fn new() -> Self {
        Self {
            interpreter: "python3".to_string(),
        }
    }

    /// Create a Python executor with custom interpreter path
    #[must_use]
    pub fn with_interpreter(interpreter: impl Into<String>) -> Self {
        Self {
            interpreter: interpreter.into(),
        }
    }

    /// Check if Python is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        Command::new(&self.interpreter)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }
}

impl Executor for PythonExecutor {
    fn execute(&self, code: &str, input: &str, timeout_ms: u64) -> Result<ExecutionResult> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let start = Instant::now();

        // Create a temporary file for the code with unique name
        let temp_dir = std::env::temp_dir();
        let unique_id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let temp_file = temp_dir.join(format!("verificar_{}_{}.py", std::process::id(), unique_id));

        std::fs::write(&temp_file, code)
            .map_err(|e| Error::Verification(format!("Failed to write temp file: {e}")))?;

        let mut cmd = Command::new(&self.interpreter);
        cmd.arg(&temp_file)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Spawn process
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::Verification(format!("Failed to spawn Python: {e}")))?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(input.as_bytes());
        }

        // Wait with timeout
        let timeout = Duration::from_millis(timeout_ms);
        let output = match wait_with_timeout(child, timeout) {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&temp_file);
                return Err(e);
            }
        };

        let _ = std::fs::remove_file(&temp_file);

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            duration_ms,
        })
    }

    fn language(&self) -> Language {
        Language::Python
    }
}

/// Rust code executor using cargo/rustc
#[derive(Debug, Default)]
pub struct RustExecutor {
    /// Path to rustc (default: "rustc")
    compiler: String,
}

impl RustExecutor {
    /// Create a new Rust executor
    #[must_use]
    pub fn new() -> Self {
        Self {
            compiler: "rustc".to_string(),
        }
    }

    /// Check if rustc is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        Command::new(&self.compiler)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }
}

impl Executor for RustExecutor {
    fn execute(&self, code: &str, input: &str, timeout_ms: u64) -> Result<ExecutionResult> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let start = Instant::now();

        // Create temp directory for compilation with unique names
        let temp_dir = std::env::temp_dir();
        let unique_id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let source_file =
            temp_dir.join(format!("verificar_{}_{}.rs", std::process::id(), unique_id));
        let binary_file = temp_dir.join(format!("verificar_{}_{}", std::process::id(), unique_id));

        std::fs::write(&source_file, code)
            .map_err(|e| Error::Verification(format!("Failed to write temp file: {e}")))?;

        let compile_output = Command::new(&self.compiler)
            .arg(&source_file)
            .arg("-o")
            .arg(&binary_file)
            .output()
            .map_err(|e| Error::Verification(format!("Failed to compile: {e}")))?;

        if !compile_output.status.success() {
            let _ = std::fs::remove_file(&source_file);
            return Ok(ExecutionResult {
                stdout: String::new(),
                stderr: String::from_utf8_lossy(&compile_output.stderr).to_string(),
                exit_code: compile_output.status.code().unwrap_or(-1),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Run binary
        let mut cmd = Command::new(&binary_file);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| Error::Verification(format!("Failed to run binary: {e}")))?;

        // Write input
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(input.as_bytes());
        }

        // Wait with timeout
        let timeout = Duration::from_millis(timeout_ms);
        let output = match wait_with_timeout(child, timeout) {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&source_file);
                let _ = std::fs::remove_file(&binary_file);
                return Err(e);
            }
        };

        // Cleanup
        let _ = std::fs::remove_file(&source_file);
        let _ = std::fs::remove_file(&binary_file);

        Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn language(&self) -> Language {
        Language::Rust
    }
}

/// Wait for a process with timeout using threaded output capture
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output> {
    use std::io::Read;
    use std::sync::mpsc;
    use std::thread;

    // Take stdout and stderr handles before spawning threads
    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    // Spawn threads to read stdout and stderr concurrently
    let stdout_thread = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut stdout) = stdout_handle {
            let _ = stdout.read_to_end(&mut buf);
        }
        buf
    });

    let stderr_thread = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut stderr) = stderr_handle {
            let _ = stderr.read_to_end(&mut buf);
        }
        buf
    });

    // Wait for process with timeout using a channel
    let (tx, rx) = mpsc::channel();
    let wait_thread = thread::spawn(move || {
        let result = child.wait();
        let _ = tx.send(result);
        child
    });

    // Wait with timeout
    match rx.recv_timeout(timeout) {
        Ok(Ok(status)) => {
            // Process finished, collect output
            // The wait_thread returns the child after wait() completes
            // We join the thread and let the child go out of scope (already waited)
            let _ = wait_thread.join();

            let stdout = stdout_thread.join().unwrap_or_default();
            let stderr = stderr_thread.join().unwrap_or_default();

            Ok(std::process::Output {
                status,
                stdout,
                stderr,
            })
        }
        Ok(Err(e)) => {
            let _ = wait_thread.join();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            Err(Error::Verification(format!("Wait error: {e}")))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Timeout - kill the process and wait to avoid zombie
            if let Ok(mut child) = wait_thread.join() {
                let _ = child.kill();
                let _ = child.wait(); // Reap the zombie process
            }
            // Still collect any output that was produced
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            Err(Error::Verification("Execution timed out".to_string()))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = wait_thread.join();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            Err(Error::Verification(
                "Process wait thread disconnected".to_string(),
            ))
        }
    }
}

/// Get an executor for the specified language
#[must_use]
pub fn executor_for(language: Language) -> Option<Box<dyn Executor>> {
    match language {
        Language::Python => Some(Box::new(PythonExecutor::new())),
        Language::Rust => Some(Box::new(RustExecutor::new())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_executor_simple() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let result = executor
            .execute("print('hello')", "", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "hello");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_python_executor_with_input() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let code = "x = input()\nprint(f'got: {x}')";
        let result = executor
            .execute(code, "test", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "got: test");
    }

    #[test]
    fn test_python_executor_error() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let result = executor
            .execute("raise ValueError('oops')", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
        assert!(result.stderr.contains("ValueError"));
    }

    #[test]
    fn test_python_executor_arithmetic() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let code = "print(1 + 2 * 3)";
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "7");
    }

    #[test]
    fn test_executor_for_python() {
        let executor = executor_for(Language::Python);
        assert!(executor.is_some());
        assert_eq!(executor.unwrap().language(), Language::Python);
    }

    #[test]
    fn test_executor_for_rust() {
        let executor = executor_for(Language::Rust);
        assert!(executor.is_some());
        assert_eq!(executor.unwrap().language(), Language::Rust);
    }

    #[test]
    fn test_executor_for_unsupported() {
        let executor = executor_for(Language::Bash);
        assert!(executor.is_none());
    }

    #[test]
    fn test_python_executor_with_interpreter() {
        let executor = PythonExecutor::with_interpreter("python3");
        assert_eq!(executor.interpreter, "python3");
    }

    #[test]
    fn test_python_executor_default() {
        let executor = PythonExecutor::default();
        assert!(executor.interpreter.is_empty() || executor.interpreter == "python3");
    }

    #[test]
    fn test_python_executor_language() {
        let executor = PythonExecutor::new();
        assert_eq!(executor.language(), Language::Python);
    }

    #[test]
    fn test_python_executor_debug() {
        let executor = PythonExecutor::new();
        let debug = format!("{:?}", executor);
        assert!(debug.contains("PythonExecutor"));
    }

    #[test]
    fn test_rust_executor_new() {
        let executor = RustExecutor::new();
        assert_eq!(executor.compiler, "rustc");
    }

    #[test]
    fn test_rust_executor_default() {
        let executor = RustExecutor::default();
        assert!(executor.compiler.is_empty() || executor.compiler == "rustc");
    }

    #[test]
    fn test_rust_executor_language() {
        let executor = RustExecutor::new();
        assert_eq!(executor.language(), Language::Rust);
    }

    #[test]
    fn test_rust_executor_debug() {
        let executor = RustExecutor::new();
        let debug = format!("{:?}", executor);
        assert!(debug.contains("RustExecutor"));
    }

    #[test]
    fn test_rust_executor_is_available() {
        let executor = RustExecutor::new();
        // This may or may not be available depending on the system
        let _ = executor.is_available();
    }

    #[test]
    fn test_rust_executor_simple() {
        let executor = RustExecutor::new();
        if !executor.is_available() {
            eprintln!("rustc not available, skipping test");
            return;
        }

        let code = r#"fn main() { println!("hello from rust"); }"#;
        let result = executor
            .execute(code, "", 10000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "hello from rust");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_rust_executor_compile_error() {
        let executor = RustExecutor::new();
        if !executor.is_available() {
            eprintln!("rustc not available, skipping test");
            return;
        }

        let code = "fn main() { invalid syntax }";
        let result = executor
            .execute(code, "", 10000)
            .expect("execution should return compile error");

        assert_ne!(result.exit_code, 0);
        assert!(!result.stderr.is_empty());
    }

    #[test]
    fn test_rust_executor_with_input() {
        let executor = RustExecutor::new();
        if !executor.is_available() {
            eprintln!("rustc not available, skipping test");
            return;
        }

        let code = r#"
use std::io::{self, BufRead};
fn main() {
    let stdin = io::stdin();
    let line = stdin.lock().lines().next().unwrap().unwrap();
    println!("got: {}", line);
}
"#;
        let result = executor
            .execute(code, "test input", 10000)
            .expect("execution should succeed");

        assert!(result.stdout.contains("got: test input"));
    }

    #[test]
    fn test_python_executor_timeout() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        // Code that takes too long
        let code = "import time; time.sleep(10)";
        let result = executor.execute(code, "", 100); // 100ms timeout

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(err_str.contains("timeout") || err_str.contains("timed out"));
    }

    #[test]
    fn test_python_executor_multiple_lines_output() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let code = "for i in range(3): print(i)";
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        assert!(result.stdout.contains("0"));
        assert!(result.stdout.contains("1"));
        assert!(result.stdout.contains("2"));
    }

    #[test]
    fn test_python_executor_syntax_error() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let code = "def f(: pass"; // syntax error
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
        assert!(result.stderr.contains("SyntaxError"));
    }
}
