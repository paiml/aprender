//! Artifact operations for SQLite Backend.
//!
//! Contains artifact retrieval and listing methods.

use super::backend::SqliteBackend;
use super::types::ArtifactRef;
use crate::storage::{Result, StorageError};
use chrono::Utc;
use rusqlite::params;

impl SqliteBackend {
    /// Get artifact data by SHA-256 hash
    pub fn get_artifact_data(&self, sha256: &str) -> Result<Vec<u8>> {
        let conn = self.lock_conn()?;

        let data: Vec<u8> = conn
            .query_row("SELECT data FROM artifacts WHERE sha256 = ?1 LIMIT 1", [sha256], |row| {
                row.get(0)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::Backend(format!("Artifact not found: {sha256}"))
                }
                _ => StorageError::Backend(format!("Failed to get artifact data: {e}")),
            })?;

        Ok(data)
    }

    /// List artifacts for a run
    pub fn list_artifacts(&self, run_id: &str) -> Result<Vec<ArtifactRef>> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let mut stmt = conn
            .prepare(
                "SELECT id, run_id, path, size_bytes, sha256, created_at FROM artifacts WHERE run_id = ?1 ORDER BY created_at",
            )
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        let rows = stmt
            .query_map(params![run_id], |row| {
                let id: String = row.get(0)?;
                let run_id: String = row.get(1)?;
                let path: String = row.get(2)?;
                let size_bytes: i64 = row.get(3)?;
                let sha256: String = row.get(4)?;
                let created_str: String = row.get(5)?;
                Ok((id, run_id, path, size_bytes, sha256, created_str))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query artifacts: {e}")))?;

        let mut result = Vec::new();
        for row in rows {
            let (id, run_id, path, size_bytes, sha256, created_str) = row
                .map_err(|e| StorageError::Backend(format!("Failed to read artifact row: {e}")))?;
            let created_at = created_str.parse().unwrap_or_else(|_| Utc::now());
            result.push(ArtifactRef {
                id,
                run_id,
                path,
                size_bytes: size_bytes as u64,
                sha256,
                created_at,
            });
        }

        Ok(result)
    }
}
