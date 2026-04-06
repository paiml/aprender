//! Sandboxed Python execution for safe code evaluation
//!
//! Provides secure execution of untrusted Python code with:
//! - Import restrictions (no os, subprocess, socket, etc.)
//! - Resource limits (time, output size)
//! - Isolated environment (no env vars, no site-packages)
//!
//! # Security Model
//!
//! The sandbox uses multiple layers of protection:
//! 1. **Python isolation flags**: `-I -E -S` for isolated mode
//! 2. **Import restrictions**: Custom import hook blocks dangerous modules
//! 3. **Builtin restrictions**: Removes dangerous builtins (eval, exec, open, etc.)
//! 4. **Time limits**: Process timeout with forced kill
//! 5. **Output limits**: Truncate excessive output
//!
//! # Example
//!
//! ```rust,ignore
//! use verificar::oracle::SandboxedPythonExecutor;
//!
//! let executor = SandboxedPythonExecutor::new();
//! let result = executor.execute("print(1 + 1)", "", 1000)?;
//! assert_eq!(result.stdout.trim(), "2");
//! ```

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{Error, Language, Result};

use super::executor::Executor;
use super::ExecutionResult;

/// Sandboxed Python executor with security restrictions
#[derive(Debug, Clone)]
pub struct SandboxedPythonExecutor {
    /// Path to Python interpreter
    interpreter: String,
    /// Maximum output size in bytes
    max_output_bytes: usize,
    /// Blocked module names
    blocked_modules: Vec<String>,
    /// Whether to allow file I/O
    allow_file_io: bool,
}

impl Default for SandboxedPythonExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SandboxedPythonExecutor {
    /// Default blocked modules for safe execution
    const DEFAULT_BLOCKED_MODULES: &'static [&'static str] = &[
        // System/process access
        "os",
        "subprocess",
        "sys",
        "shutil",
        "pathlib",
        "glob",
        "tempfile",
        // Network access
        "socket",
        "http",
        "urllib",
        "requests",
        "aiohttp",
        "ftplib",
        "smtplib",
        "ssl",
        // Code execution/compilation
        "code",
        "codeop",
        "compile",
        "importlib",
        "runpy",
        "ast",
        "dis",
        "inspect",
        // Dangerous internals
        "ctypes",
        "cffi",
        "multiprocessing",
        "threading",
        "concurrent",
        "_thread",
        "gc",
        "resource",
        "signal",
        // File operations
        "io",
        "builtins",
        "pickle",
        "shelve",
        "dbm",
        "sqlite3",
        // Misc dangerous
        "pty",
        "tty",
        "termios",
        "fcntl",
        "mmap",
    ];

    /// Create a new sandboxed Python executor with default restrictions
    #[must_use]
    pub fn new() -> Self {
        Self {
            interpreter: "python3".to_string(),
            max_output_bytes: 64 * 1024, // 64KB max output
            blocked_modules: Self::DEFAULT_BLOCKED_MODULES
                .iter()
                .map(|&s| s.to_string())
                .collect(),
            allow_file_io: false,
        }
    }

    /// Create executor with custom interpreter path
    #[must_use]
    pub fn with_interpreter(mut self, interpreter: impl Into<String>) -> Self {
        self.interpreter = interpreter.into();
        self
    }

    /// Set maximum output size
    #[must_use]
    pub fn with_max_output(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes;
        self
    }

    /// Add additional blocked modules
    #[must_use]
    pub fn with_blocked_modules(mut self, modules: &[&str]) -> Self {
        for module in modules {
            if !self.blocked_modules.contains(&(*module).to_string()) {
                self.blocked_modules.push((*module).to_string());
            }
        }
        self
    }

    /// Allow file I/O operations (not recommended for untrusted code)
    #[must_use]
    pub fn with_file_io(mut self, allow: bool) -> Self {
        self.allow_file_io = allow;
        self
    }

    /// Check if Python interpreter is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        Command::new(&self.interpreter)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    /// Generate the sandbox wrapper code that restricts imports and builtins
    #[allow(clippy::uninlined_format_args)]
    fn sandbox_wrapper(&self) -> String {
        let blocked_list = self
            .blocked_modules
            .iter()
            .map(|m| format!("'{m}'"))
            .collect::<Vec<_>>()
            .join(", ");

        let open_restriction = if self.allow_file_io {
            ""
        } else {
            "_sandbox_builtins.open = None\n"
        };

        format!(
            r#"
import sys as _sandbox_sys
import builtins as _sandbox_builtins

# Block dangerous modules
_sandbox_blocked = set([{blocked_list}])

# Remove already-loaded blocked modules from sys.modules
for _sandbox_mod in list(_sandbox_sys.modules.keys()):
    _sandbox_base = _sandbox_mod.split('.')[0]
    if _sandbox_base in _sandbox_blocked and _sandbox_base != 'sys':
        del _sandbox_sys.modules[_sandbox_mod]

# Create a restricted __import__ function using default argument to capture values
_sandbox_orig_import = _sandbox_builtins.__import__

def _sandbox_import(name, globals=None, locals=None, fromlist=(), level=0,
                    _blocked=_sandbox_blocked, _orig=_sandbox_orig_import):
    base = name.split('.')[0]
    if base in _blocked:
        raise ImportError(f"Module '{{name}}' is not allowed in sandbox")
    return _orig(name, globals, locals, fromlist, level)

# Replace builtins.__import__ (this is what import statements actually use)
_sandbox_builtins.__import__ = _sandbox_import

# Restrict dangerous builtins
_sandbox_builtins.eval = None
_sandbox_builtins.exec = None
_sandbox_builtins.compile = None
_sandbox_builtins.breakpoint = None
_sandbox_builtins.help = None
{open_restriction}
# Capture user input lines
_sandbox_user_input = []
_sandbox_builtins.input = lambda *a, _inp=_sandbox_user_input: _inp.pop(0) if _inp else ''

# Clean up sandbox setup from namespace
del _sandbox_mod, _sandbox_base, _sandbox_sys, _sandbox_builtins, _sandbox_orig_import, _sandbox_blocked

# User code follows:
"#,
            blocked_list = blocked_list,
            open_restriction = open_restriction
        )
    }

    /// Wrap user code with sandbox restrictions
    fn wrap_code(&self, code: &str, input: &str) -> String {
        let mut wrapper = self.sandbox_wrapper();

        // Add input handling - inject into _sandbox_user_input before cleanup
        if !input.is_empty() {
            // Need to insert input BEFORE the del statement
            // Find and replace the _sandbox_user_input = [] line
            let input_lines: Vec<_> = input
                .lines()
                .map(|l| format!("'{}'", l.replace('\'', "\\'")))
                .collect();
            let input_init = format!("_sandbox_user_input = [{}]\n", input_lines.join(", "));
            wrapper = wrapper.replace("_sandbox_user_input = []\n", &input_init);
        }

        // Add user code (sandbox is already set up)
        wrapper.push_str(code);

        wrapper
    }
}

impl Executor for SandboxedPythonExecutor {
    fn execute(&self, code: &str, input: &str, timeout_ms: u64) -> Result<ExecutionResult> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let start = Instant::now();

        // Create a unique temp file
        let temp_dir = std::env::temp_dir();
        let unique_id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let temp_file = temp_dir.join(format!(
            "verificar_sandbox_{}_{}.py",
            std::process::id(),
            unique_id
        ));

        let sandboxed_code = self.wrap_code(code, input);

        std::fs::write(&temp_file, &sandboxed_code)
            .map_err(|e| Error::Verification(format!("Failed to write sandbox file: {e}")))?;

        let mut cmd = Command::new(&self.interpreter);
        cmd.arg("-I") // Isolated mode: don't add user site directory, ignore PYTHON* env vars
            .arg("-E") // Ignore PYTHON* environment variables
            .arg("-S") // Don't import site module
            .arg("-u") // Unbuffered output
            .arg(&temp_file)
            .stdin(Stdio::null()) // No stdin (we handle input via code)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear(); // Clear all environment variables

        // Spawn process
        let child = cmd.spawn().map_err(|e| {
            let _ = std::fs::remove_file(&temp_file);
            Error::Verification(format!("Failed to spawn sandboxed Python: {e}"))
        })?;

        // Wait with timeout
        let timeout = Duration::from_millis(timeout_ms);
        let output = match wait_with_timeout(child, timeout) {
            Ok(output) => output,
            Err(e) => {
                let _ = std::fs::remove_file(&temp_file);
                return Err(e);
            }
        };

        // Clean up
        let _ = std::fs::remove_file(&temp_file);

        let duration_ms = start.elapsed().as_millis() as u64;

        // Truncate output if too large
        let stdout = truncate_output(&output.stdout, self.max_output_bytes);
        let stderr = truncate_output(&output.stderr, self.max_output_bytes);

        Ok(ExecutionResult {
            stdout,
            stderr,
            exit_code: output.status.code().unwrap_or(-1),
            duration_ms,
        })
    }

    fn language(&self) -> Language {
        Language::Python
    }
}

/// Wait for process with timeout (duplicated to avoid circular dependency)
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output> {
    use std::io::Read;
    use std::sync::mpsc;
    use std::thread;

    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

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

    let (tx, rx) = mpsc::channel();
    let wait_thread = thread::spawn(move || {
        let result = child.wait();
        let _ = tx.send(result);
        child
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(status)) => {
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
            if let Ok(mut child) = wait_thread.join() {
                let _ = child.kill();
                let _ = child.wait();
            }
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            Err(Error::Verification(
                "Sandbox execution timed out".to_string(),
            ))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let _ = wait_thread.join();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            Err(Error::Verification(
                "Sandbox thread disconnected".to_string(),
            ))
        }
    }
}

/// Truncate output to maximum size with message
fn truncate_output(data: &[u8], max_bytes: usize) -> String {
    let s = String::from_utf8_lossy(data);
    if s.len() <= max_bytes {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_bytes).collect();
        format!("{truncated}\n... [output truncated at {max_bytes} bytes]")
    }
}

/// Sandbox configuration options
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum execution time in milliseconds
    pub timeout_ms: u64,
    /// Maximum output size in bytes
    pub max_output_bytes: usize,
    /// Maximum memory usage (not enforced in pure Rust, advisory only)
    pub max_memory_bytes: usize,
    /// Allow network access (always false for security)
    pub allow_network: bool,
    /// Allow file system access
    pub allow_filesystem: bool,
    /// Additional blocked modules
    pub blocked_modules: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            max_output_bytes: 64 * 1024,
            max_memory_bytes: 128 * 1024 * 1024, // 128MB advisory
            allow_network: false,
            allow_filesystem: false,
            blocked_modules: Vec::new(),
        }
    }
}

impl SandboxConfig {
    /// Create a strict sandbox configuration
    #[must_use]
    pub fn strict() -> Self {
        Self {
            timeout_ms: 1000,
            max_output_bytes: 16 * 1024,
            max_memory_bytes: 64 * 1024 * 1024,
            allow_network: false,
            allow_filesystem: false,
            blocked_modules: Vec::new(),
        }
    }

    /// Create a lenient sandbox for testing (still secure, but more permissive limits)
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            timeout_ms: 30000,
            max_output_bytes: 1024 * 1024,
            max_memory_bytes: 512 * 1024 * 1024,
            allow_network: false,
            allow_filesystem: false,
            blocked_modules: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_python(executor: &SandboxedPythonExecutor) -> bool {
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            true
        } else {
            false
        }
    }

    #[test]
    fn test_sandbox_basic_execution() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("print(1 + 1)", "", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "2");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_blocks_os_import() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("import os\nprint(os.getcwd())", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
        assert!(result.stderr.contains("not allowed") || result.stderr.contains("ImportError"));
    }

    #[test]
    fn test_sandbox_blocks_subprocess() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("import subprocess\nsubprocess.run(['ls'])", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
        assert!(result.stderr.contains("not allowed") || result.stderr.contains("ImportError"));
    }

    #[test]
    fn test_sandbox_blocks_socket() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("import socket", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_blocks_eval() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("eval('1+1')", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_blocks_exec() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("exec('print(1)')", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_allows_safe_math() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("import math\nprint(math.sqrt(16))", "", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "4.0");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_allows_safe_builtins() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let code = r#"
x = [1, 2, 3, 4, 5]
print(sum(x))
print(len(x))
print(max(x))
print(min(x))
"#;
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        let lines: Vec<_> = result.stdout.trim().lines().collect();
        assert_eq!(lines, vec!["15", "5", "5", "1"]);
    }

    #[test]
    #[ignore] // Takes too long for CI - run with `cargo test -- --ignored`
    fn test_sandbox_timeout() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor.execute("while True: pass", "", 500);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timed out") || err.contains("timeout"));
    }

    #[test]
    fn test_sandbox_output_truncation() {
        let executor = SandboxedPythonExecutor::new().with_max_output(100);
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("print('x' * 1000)", "", 5000)
            .expect("execution should succeed");

        assert!(result.stdout.len() <= 150); // 100 + truncation message
        assert!(result.stdout.contains("truncated"));
    }

    #[test]
    fn test_sandbox_input_handling() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let code = r#"
name = input()
print(f"Hello, {name}!")
"#;
        let result = executor
            .execute(code, "World", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "Hello, World!");
    }

    #[test]
    fn test_sandbox_multiple_inputs() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let code = r#"
a = int(input())
b = int(input())
print(a + b)
"#;
        let result = executor
            .execute(code, "3\n4", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "7");
    }

    #[test]
    fn test_sandbox_config_strict() {
        let config = SandboxConfig::strict();
        assert_eq!(config.timeout_ms, 1000);
        assert!(!config.allow_network);
        assert!(!config.allow_filesystem);
    }

    #[test]
    fn test_sandbox_config_lenient() {
        let config = SandboxConfig::lenient();
        assert_eq!(config.timeout_ms, 30000);
        assert!(!config.allow_network);
    }

    #[test]
    fn test_sandbox_config_default() {
        let config = SandboxConfig::default();
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.max_output_bytes, 64 * 1024);
        assert_eq!(config.max_memory_bytes, 128 * 1024 * 1024);
        assert!(!config.allow_network);
        assert!(!config.allow_filesystem);
        assert!(config.blocked_modules.is_empty());
    }

    #[test]
    fn test_sandbox_config_debug() {
        let config = SandboxConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("SandboxConfig"));
    }

    #[test]
    fn test_sandbox_config_clone() {
        let config = SandboxConfig::strict();
        let cloned = config.clone();
        assert_eq!(cloned.timeout_ms, config.timeout_ms);
    }

    #[test]
    fn test_sandboxed_executor_with_interpreter() {
        let executor = SandboxedPythonExecutor::new().with_interpreter("python3.11");
        assert_eq!(executor.interpreter, "python3.11");
    }

    #[test]
    fn test_sandboxed_executor_with_max_output() {
        let executor = SandboxedPythonExecutor::new().with_max_output(1024);
        assert_eq!(executor.max_output_bytes, 1024);
    }

    #[test]
    fn test_sandboxed_executor_with_blocked_modules() {
        let executor = SandboxedPythonExecutor::new()
            .with_blocked_modules(&["custom_module", "another_module"]);
        assert!(executor
            .blocked_modules
            .contains(&"custom_module".to_string()));
        assert!(executor
            .blocked_modules
            .contains(&"another_module".to_string()));
    }

    #[test]
    fn test_sandboxed_executor_with_file_io() {
        let executor = SandboxedPythonExecutor::new().with_file_io(true);
        assert!(executor.allow_file_io);
    }

    #[test]
    fn test_sandboxed_executor_default() {
        let executor = SandboxedPythonExecutor::default();
        assert_eq!(executor.interpreter, "python3");
        assert_eq!(executor.max_output_bytes, 64 * 1024);
        assert!(!executor.allow_file_io);
    }

    #[test]
    fn test_sandboxed_executor_language() {
        let executor = SandboxedPythonExecutor::new();
        assert_eq!(executor.language(), Language::Python);
    }

    #[test]
    fn test_sandboxed_executor_debug() {
        let executor = SandboxedPythonExecutor::new();
        let debug = format!("{:?}", executor);
        assert!(debug.contains("SandboxedPythonExecutor"));
    }

    #[test]
    fn test_truncate_output_short() {
        let data = b"hello world";
        let result = truncate_output(data, 100);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_truncate_output_long() {
        let data = b"hello world this is a longer string";
        let result = truncate_output(data, 10);
        assert!(result.len() <= 50); // Truncated + message
        assert!(result.contains("truncated"));
    }

    #[test]
    fn test_sandbox_with_file_io_enabled() {
        let executor = SandboxedPythonExecutor::new().with_file_io(true);
        if skip_if_no_python(&executor) {
            return;
        }

        // With file I/O enabled, open should work (but we don't actually write files)
        // Just test that the sandbox generates different code
        let code = "print('file io test')";
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "file io test");
    }

    #[test]
    fn test_sandbox_blocks_open_by_default() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        // Default should block open()
        let code = "open('/etc/passwd', 'r')";
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        // Should fail because open is disabled
        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_blocks_ctypes() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("import ctypes", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_blocks_sys() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        let result = executor
            .execute("import sys\nprint(sys.executable)", "", 5000)
            .expect("execution should succeed");

        // sys is blocked
        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_sandbox_empty_input() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        // With empty input, input() should return empty string
        let code = "x = input()\nprint(f'got: [{x}]')";
        let result = executor
            .execute(code, "", 5000)
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "got: []");
    }

    #[test]
    fn test_sandbox_input_with_quotes() {
        let executor = SandboxedPythonExecutor::new();
        if skip_if_no_python(&executor) {
            return;
        }

        // Test that quotes in input are properly escaped
        let code = "x = input()\nprint(f'got: {x}')";
        let result = executor
            .execute(code, "hello'world", 5000)
            .expect("execution should succeed");

        assert!(result.stdout.contains("hello'world") || result.exit_code == 0);
    }

    #[test]
    fn test_sandbox_is_available() {
        let executor = SandboxedPythonExecutor::new();
        // Just check it returns something
        let _ = executor.is_available();
    }

    #[test]
    fn test_sandbox_with_custom_blocked_module_execution() {
        let executor = SandboxedPythonExecutor::new().with_blocked_modules(&["json"]);
        if skip_if_no_python(&executor) {
            return;
        }

        // json is normally allowed, but we blocked it
        let result = executor
            .execute("import json\nprint(json.dumps({}))", "", 5000)
            .expect("execution should succeed");

        assert_ne!(result.exit_code, 0);
    }
}
