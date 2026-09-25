//! SQLite schema definition and migration.
//!
//! Defines all tables for the experiment store and handles schema initialization.

use rusqlite::Connection;

/// Current schema version
pub const CURRENT_VERSION: &str = "1.0.0";

/// Initialize the database schema, creating tables if they don't exist.
///
/// Also configures WAL mode and performance pragmas matching the
/// paiml-mcp-agent-toolkit `.pmat/context.db` pattern.
pub fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Performance pragmas (matching paiml-mcp-agent-toolkit)
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA cache_size = -64000;
         PRAGMA temp_store = MEMORY;",
    )?;

    conn.execute_batch(SCHEMA_SQL)?;
    ensure_liveness_columns(conn)?;

    // Insert schema version if not present
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM schema_version", [], |row| row.get(0))?;
    if count == 0 {
        conn.execute("INSERT INTO schema_version (version) VALUES (?1)", [CURRENT_VERSION])?;
    }

    Ok(())
}

/// Liveness columns on `runs` (EXT-03, aprender#4385).
///
/// Premise recorded at HEAD b474a64a: `runs` carried no pid, host, boot id or
/// process start time, so a crashed trainer left its row `running` forever and
/// nothing could tell it from a live one. The identity is the 4-tuple
/// `(host, boot_id, pid, proc_start_time)`: a pid alone is reused after a
/// reboot or a wrap, and a pid + start time alone collides across hosts.
/// Added by `ALTER TABLE` so databases created before EXT-03 upgrade in place;
/// their rows keep NULLs, which the reaper reads as "identity unrecorded".
pub(crate) const LIVENESS_COLUMNS: [(&str, &str); 4] =
    [("host", "TEXT"), ("boot_id", "TEXT"), ("pid", "INTEGER"), ("proc_start_time", "INTEGER")];

fn ensure_liveness_columns(conn: &Connection) -> Result<(), rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT name FROM pragma_table_info('runs')")?;
    let present: Vec<String> = stmt.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?;
    for (name, ty) in LIVENESS_COLUMNS {
        if !present.iter().any(|p| p == name) {
            conn.execute_batch(&format!("ALTER TABLE runs ADD COLUMN {name} {ty};"))?;
        }
    }
    Ok(())
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config TEXT,
    tags TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    start_time TEXT,
    end_time TEXT,
    tags TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    step INTEGER NOT NULL,
    value REAL NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    sha256 TEXT NOT NULL,
    data BLOB,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256);

CREATE TABLE IF NOT EXISTS span_ids (
    run_id TEXT PRIMARY KEY,
    span_id TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
";
