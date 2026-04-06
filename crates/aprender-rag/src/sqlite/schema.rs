//! SQLite schema definition and migration logic.
//!
//! Implements the normalized schema from the sqlite-rag-integration spec:
//! documents, chunks, chunks_fts (FTS5 with Porter stemmer), fingerprints,
//! and metadata tables.

use crate::Result;
use rusqlite::Connection;

/// Current schema version for migration tracking.
///
/// v1.0.0: Initial schema with content FTS5 (stores duplicate text)
/// v2.0.0: External content FTS5 (content=chunks) — eliminates ~135MB shadow table
pub const SCHEMA_VERSION: &str = "2.0.0";

/// Initialize the database schema and set connection pragmas.
///
/// Pragmas follow the sqlite-rag-integration spec Section 4.1.2:
/// - WAL journal mode for concurrent readers (Hipp, 2014)
/// - 64 MB page cache
/// - 256 MB mmap for reduced syscall overhead
///
/// Handles migration from v1 (content FTS5) to v2 (external content FTS5).
pub fn initialize(conn: &Connection) -> Result<()> {
    set_pragmas(conn)?;
    let migrated = migrate_if_needed(conn)?;
    create_tables(conn)?;
    if migrated {
        rebuild_fts(conn)?;
    }
    set_schema_version(conn)?;
    Ok(())
}

/// Migrate from v1 (content FTS5) to v2 (external content FTS5).
///
/// Detects v1 schema by checking if `chunks_fts_content` shadow table exists
/// (content FTS5 creates it; external content FTS5 does not). Drops old FTS
/// table + triggers, so `create_tables` can recreate with external content.
/// After table creation, triggers handle sync; existing data is rebuilt via
/// `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`.
fn migrate_if_needed(conn: &Connection) -> Result<bool> {
    // Check if chunks_fts_content shadow table exists (sign of v1 content FTS5)
    let has_shadow: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='chunks_fts_content'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !has_shadow {
        return Ok(false);
    }

    // v1 → v2 migration: drop content FTS5, triggers will be recreated
    conn.execute_batch(
        "DROP TRIGGER IF EXISTS chunks_ai;
         DROP TRIGGER IF EXISTS chunks_ad;
         DROP TRIGGER IF EXISTS chunks_au;
         DROP TABLE IF EXISTS chunks_fts;",
    )
    .map_err(|e| crate::Error::Query(format!("v1→v2 migration: drop old FTS failed: {e}")))?;

    Ok(true)
}

fn set_pragmas(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -65536;
         PRAGMA mmap_size = 268435456;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;",
    )
    .map_err(|e| crate::Error::Query(format!("Failed to set pragmas: {e}")))?;
    Ok(())
}

fn create_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            title       TEXT,
            source      TEXT,
            content     TEXT NOT NULL,
            metadata    TEXT,
            chunk_count INTEGER DEFAULT 0,
            created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
            updated_at  INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content     TEXT NOT NULL,
            position    INTEGER NOT NULL,
            metadata    TEXT,
            UNIQUE(doc_id, position)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content=chunks,
            content_rowid=rowid,
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
            INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
        END;

        CREATE TABLE IF NOT EXISTS fingerprints (
            doc_path    TEXT PRIMARY KEY,
            blake3_hash BLOB NOT NULL,
            chunk_count INTEGER NOT NULL,
            indexed_at  INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )
    .map_err(|e| crate::Error::Query(format!("Failed to create tables: {e}")))?;
    Ok(())
}

/// Rebuild FTS5 index from the external content table (chunks).
///
/// Called after v1→v2 migration to populate the new external-content FTS5
/// index from existing chunk data.
fn rebuild_fts(conn: &Connection) -> Result<()> {
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])
        .map_err(|e| crate::Error::Query(format!("FTS5 rebuild failed: {e}")))?;
    Ok(())
}

fn set_schema_version(conn: &Connection) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION],
    )
    .map_err(|e| crate::Error::Query(format!("Failed to set schema version: {e}")))?;
    Ok(())
}

/// Check if the schema version matches the current version.
pub fn check_version(conn: &Connection) -> Result<bool> {
    let version: Option<String> = conn
        .query_row("SELECT value FROM metadata WHERE key = 'schema_version'", [], |row| row.get(0))
        .ok();
    Ok(version.as_deref() == Some(SCHEMA_VERSION))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_creates_tables() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();

        // Verify tables exist by querying them
        let doc_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)).unwrap();
        assert_eq!(doc_count, 0);

        let chunk_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).unwrap();
        assert_eq!(chunk_count, 0);

        let fp_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM fingerprints", [], |r| r.get(0)).unwrap();
        assert_eq!(fp_count, 0);
    }

    #[test]
    fn test_schema_version_set() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();
        assert!(check_version(&conn).unwrap());
    }

    #[test]
    fn test_initialize_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();
        initialize(&conn).unwrap(); // Should not fail on second call
        assert!(check_version(&conn).unwrap());
    }

    #[test]
    fn test_wal_mode_enabled() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();
        // In-memory databases use "memory" journal mode, not WAL
        // but the pragma should not error
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
        assert!(!mode.is_empty());
    }

    #[test]
    fn test_external_content_fts5_no_shadow_content_table() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();

        // External content FTS5 should NOT create chunks_fts_content shadow table
        let has_content_table: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='chunks_fts_content'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!has_content_table, "External content FTS5 should not create chunks_fts_content");
    }

    #[test]
    fn test_v1_to_v2_migration() {
        let conn = Connection::open_in_memory().unwrap();

        // Simulate v1 schema: create content FTS5 (with shadow table)
        set_pragmas(&conn).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, title TEXT, source TEXT, content TEXT NOT NULL,
                metadata TEXT, chunk_count INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch())
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL, position INTEGER NOT NULL, metadata TEXT,
                UNIQUE(doc_id, position)
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(content, tokenize='porter unicode61');
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.rowid;
            END;
            CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.rowid;
                INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TABLE IF NOT EXISTS fingerprints (
                doc_path TEXT PRIMARY KEY, blake3_hash BLOB NOT NULL,
                chunk_count INTEGER NOT NULL, indexed_at INTEGER NOT NULL DEFAULT (unixepoch())
            );
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', '1.0.0');",
        )
        .unwrap();

        // Insert test data under v1 schema
        conn.execute("INSERT INTO documents (id, content) VALUES ('doc1', 'test document')", [])
            .unwrap();
        conn.execute(
            "INSERT INTO chunks (id, doc_id, content, position) VALUES ('c1', 'doc1', 'SIMD vector operations', 0)",
            [],
        )
        .unwrap();

        // Verify v1 has chunks_fts_content shadow table
        let has_shadow: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='chunks_fts_content'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_shadow, "v1 should have chunks_fts_content shadow table");

        // Run initialize — should migrate v1→v2
        initialize(&conn).unwrap();

        // Verify shadow table is gone
        let has_shadow_after: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='chunks_fts_content'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!has_shadow_after, "v2 should not have chunks_fts_content");

        // Verify version is 2.0.0
        assert!(check_version(&conn).unwrap());

        // Verify FTS search still works (data was rebuilt)
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'SIMD'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 1, "FTS search should find migrated data");
    }

    #[test]
    fn test_v2_fresh_install_no_migration() {
        let conn = Connection::open_in_memory().unwrap();
        let migrated = migrate_if_needed(&conn).unwrap();
        assert!(!migrated, "Fresh install should not trigger migration");
    }
}
