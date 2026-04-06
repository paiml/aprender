//! SQLite Backend core implementation.
//!
//! Per-project durable experiment store using rusqlite with WAL mode.
//! Follows the `.pmat/context.db` pattern from paiml-mcp-agent-toolkit.

use super::schema;
use crate::storage::Result;
use crate::storage::StorageError;
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// SQLite backend for experiment storage
///
/// Uses a real SQLite database (WAL mode) for durable, per-project
/// experiment tracking. The database is created lazily at
/// `<project>/.entrenar/experiments.db`.
///
/// The `Connection` is wrapped in a `Mutex` to satisfy the `Send + Sync`
/// bounds on `ExperimentStorage`. SQLite in WAL mode handles concurrent
/// readers natively; the mutex serializes writers.
pub struct SqliteBackend {
    pub(crate) conn: Mutex<Connection>,
    pub(crate) path: String,
}

// Manual Debug impl since rusqlite::Connection doesn't implement Debug
impl std::fmt::Debug for SqliteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteBackend").field("path", &self.path).finish_non_exhaustive()
    }
}

impl SqliteBackend {
    /// Open or create a SQLite database at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let conn = Connection::open(path.as_ref())
            .map_err(|e| StorageError::Backend(format!("Failed to open SQLite database: {e}")))?;
        schema::init_schema(&conn)
            .map_err(|e| StorageError::Backend(format!("Failed to initialize schema: {e}")))?;
        Ok(Self { conn: Mutex::new(conn), path: path_str })
    }

    /// Open an in-memory database (for tests — backward compatible)
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| StorageError::Backend(format!("Failed to open in-memory SQLite: {e}")))?;
        schema::init_schema(&conn)
            .map_err(|e| StorageError::Backend(format!("Failed to initialize schema: {e}")))?;
        Ok(Self { conn: Mutex::new(conn), path: ":memory:".to_string() })
    }

    /// Open or create at project-local path: `<project>/.entrenar/experiments.db`
    pub fn open_project<P: AsRef<Path>>(project_dir: P) -> Result<Self> {
        let dir = project_dir.as_ref().join(".entrenar");
        std::fs::create_dir_all(&dir)?;
        Self::open(dir.join("experiments.db"))
    }

    /// Get the database path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Generate a unique ID
    pub(crate) fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must not be before UNIX epoch")
            .as_nanos();
        format!("{ts:x}")
    }

    /// Acquire the connection lock, mapping poison errors to StorageError
    pub(crate) fn lock_conn(
        &self,
    ) -> std::result::Result<std::sync::MutexGuard<'_, Connection>, StorageError> {
        self.conn
            .lock()
            .map_err(|e| StorageError::Backend(format!("Failed to acquire connection lock: {e}")))
    }
}
