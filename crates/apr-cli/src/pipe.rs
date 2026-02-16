//! Stdin/stdout pipe support (PMAT-261)
//!
//! Enables POSIX-standard `-` convention for stdin/stdout in CLI commands.
//! Model data from stdin is buffered to a temporary file so that mmap-based
//! operations (GGUF, SafeTensors) work transparently.

use crate::error::CliError;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Check if a path string indicates stdin.
///
/// Recognizes POSIX `-` convention and Linux device paths:
/// `-`, `/dev/stdin`, `/dev/fd/0`, `/proc/self/fd/0`
#[must_use]
pub fn is_stdin(path: &str) -> bool {
    matches!(path, "-" | "/dev/stdin" | "/dev/fd/0" | "/proc/self/fd/0")
}

/// Check if a path string indicates stdout.
///
/// Recognizes POSIX `-` convention and Linux device paths:
/// `-`, `/dev/stdout`, `/dev/fd/1`, `/proc/self/fd/1`
#[must_use]
pub fn is_stdout(path: &str) -> bool {
    matches!(path, "-" | "/dev/stdout" | "/dev/fd/1" | "/proc/self/fd/1")
}

/// Temporary file that holds stdin data for mmap-based operations.
/// Automatically deleted when dropped.
pub struct TempModelFile {
    path: PathBuf,
}

impl TempModelFile {
    /// Get the path to the temporary file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempModelFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Read all of stdin into a temporary file.
///
/// Returns a `TempModelFile` whose path can be passed to any function
/// expecting a file path (mmap, fs::read, etc.).
pub fn read_stdin_to_tempfile() -> Result<TempModelFile, CliError> {
    let mut buf = Vec::new();
    io::stdin()
        .lock()
        .read_to_end(&mut buf)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read stdin: {e}")))?;

    if buf.is_empty() {
        return Err(CliError::ValidationFailed(
            "No data received on stdin. Pipe a model file: cat model.gguf | apr validate -"
                .to_string(),
        ));
    }

    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("apr-stdin-{}.bin", std::process::id()));

    fs::write(&tmp_path, &buf)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write temp file: {e}")))?;

    Ok(TempModelFile { path: tmp_path })
}

/// Resolve an input path: if "-", read stdin to tempfile; otherwise return the path as-is.
///
/// Returns `(resolved_path, Option<TempModelFile>)`. The caller must hold the
/// `TempModelFile` in scope to prevent premature cleanup.
pub fn resolve_input(path_str: &str) -> Result<(PathBuf, Option<TempModelFile>), CliError> {
    if is_stdin(path_str) {
        let tmp = read_stdin_to_tempfile()?;
        let p = tmp.path().to_path_buf();
        Ok((p, Some(tmp)))
    } else {
        Ok((PathBuf::from(path_str), None))
    }
}

/// Write bytes to stdout (for `-` output paths).
pub fn write_stdout(data: &[u8]) -> Result<(), CliError> {
    io::stdout()
        .lock()
        .write_all(data)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write to stdout: {e}")))?;
    io::stdout()
        .lock()
        .flush()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to flush stdout: {e}")))?;
    Ok(())
}

/// Run a command with stdin pipe support: if `file` is `-`, buffer stdin
/// to a tempfile; otherwise pass the path through.
pub fn with_stdin_support<F>(file: &Path, f: F) -> Result<(), CliError>
where
    F: FnOnce(&Path) -> Result<(), CliError>,
{
    let file_str = file.to_string_lossy();
    if is_stdin(&file_str) {
        let tmp = read_stdin_to_tempfile()?;
        f(tmp.path())
    } else {
        f(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_stdin() {
        assert!(is_stdin("-"));
        assert!(is_stdin("/dev/stdin"));
        assert!(is_stdin("/dev/fd/0"));
        assert!(is_stdin("/proc/self/fd/0"));
        assert!(!is_stdin("model.gguf"));
        assert!(!is_stdin(""));
        assert!(!is_stdin("--"));
        assert!(!is_stdin("/dev/fd/1"));
    }

    #[test]
    fn test_is_stdout() {
        assert!(is_stdout("-"));
        assert!(is_stdout("/dev/stdout"));
        assert!(is_stdout("/dev/fd/1"));
        assert!(is_stdout("/proc/self/fd/1"));
        assert!(!is_stdout("output.apr"));
        assert!(!is_stdout("/dev/stdin"));
        assert!(!is_stdout("/dev/fd/0"));
    }

    #[test]
    fn test_resolve_input_file_path() {
        let (path, tmp) = resolve_input("/tmp/nonexistent.gguf").expect("should resolve");
        assert_eq!(path, PathBuf::from("/tmp/nonexistent.gguf"));
        assert!(tmp.is_none());
    }

    #[test]
    fn test_temp_model_file_cleanup() {
        let tmp_path = std::env::temp_dir().join("apr-test-cleanup.bin");
        fs::write(&tmp_path, b"test data").expect("write");
        assert!(tmp_path.exists());

        {
            let _tmp = TempModelFile {
                path: tmp_path.clone(),
            };
            assert!(tmp_path.exists());
        }
        // Dropped — file should be cleaned up
        assert!(!tmp_path.exists());
    }
}
