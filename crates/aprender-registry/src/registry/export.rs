//! Fleet JSONL export (EXT-001 §3.1, row EXT-22, aprender#4404).
//!
//! §3.1: "a nightly JSONL export per host goes to the RAID. Re-exporting is
//! byte-identical (I-7). The fleet merge happens on content keys, never on host ids."
//!
//! One line per registry row: `{"table":…,"key":…,"row":{…}}`. The `row` object holds
//! every column but the host-local autoincrement ids, with keys in sorted order, and the
//! lines are sorted. Nothing in a line depends on when, or on which host, the export ran,
//! so an unchanged registry exports to the same bytes (EXT-INV-007).

use super::database::RegistryDb;
use crate::error::{PachaError, Result};
use rusqlite::types::ValueRef;
use std::collections::BTreeMap;

/// Each exported table: its content key columns and the host-local columns it drops.
struct Table {
    name: &'static str,
    key: &'static [&'static str],
    drop: &'static [&'static str],
}

/// Every table the registry holds. A table missing here is refused, not skipped: an
/// export that silently leaves one out would pass the ledger with rows missing.
const TABLES: &[Table] = &[
    Table { name: "models", key: &["content_hash", "name", "version"], drop: &[] },
    Table { name: "datasets", key: &["content_hash", "name", "version"], drop: &[] },
    Table { name: "dataset_manifests", key: &["canonical_sha256"], drop: &[] },
    Table { name: "recipes", key: &["name", "version"], drop: &[] },
    Table { name: "runs", key: &["id"], drop: &[] },
    Table { name: "lineage", key: &["from_id", "to_id", "edge_type"], drop: &["id"] },
    Table { name: "evals", key: &["model_sha", "suite", "suite_manifest_sha"], drop: &["id"] },
];

fn json_value(v: ValueRef<'_>) -> serde_json::Value {
    match v {
        ValueRef::Null => serde_json::Value::Null,
        ValueRef::Integer(i) => i.into(),
        ValueRef::Real(f) => f.into(),
        ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned().into(),
        ValueRef::Blob(b) => b.iter().map(|x| format!("{x:02x}")).collect::<String>().into(),
    }
}

fn key_part(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

impl RegistryDb {
    fn table_lines(&self, t: &Table, out: &mut Vec<String>) -> Result<()> {
        let mut stmt = self.conn.prepare(&format!("SELECT * FROM {}", t.name))?;
        let cols: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
        for k in t.key {
            if !cols.iter().any(|c| c == k) {
                return Err(PachaError::Validation(format!(
                    "export: table {} has no key column {k}",
                    t.name
                )));
            }
        }
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let mut row = BTreeMap::new();
            for (i, c) in cols.iter().enumerate() {
                if !t.drop.contains(&c.as_str()) {
                    row.insert(c.clone(), json_value(r.get_ref(i)?));
                }
            }
            let key: Vec<String> = t.key.iter().map(|k| key_part(&row[*k])).collect();
            let line = serde_json::json!({ "table": t.name, "key": key.join("|"), "row": row });
            out.push(serde_json::to_string(&line)?);
        }
        Ok(())
    }

    /// The registry as sorted JSONL, one row per line, each line `\n`-terminated.
    ///
    /// # Errors
    ///
    /// Returns an error if a query fails, or the database holds a table the export does
    /// not know (it would otherwise be left out without a sign).
    pub fn export_jsonl(&self) -> Result<String> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
        )?;
        let names = stmt.query_map([], |r| r.get::<_, String>(0))?;
        for n in names {
            let n = n?;
            if !TABLES.iter().any(|t| t.name == n) {
                return Err(PachaError::Validation(format!(
                    "export: table {n} is not in the export (EXT-22): add it with its content key"
                )));
            }
        }
        let mut lines = Vec::new();
        for t in TABLES {
            self.table_lines(t, &mut lines)?;
        }
        lines.sort_unstable();
        Ok(lines.into_iter().map(|l| l + "\n").collect())
    }
}

impl super::Registry {
    /// The registry as sorted JSONL; see [`RegistryDb::export_jsonl`].
    ///
    /// # Errors
    ///
    /// As [`RegistryDb::export_jsonl`].
    pub fn export_jsonl(&self) -> Result<String> {
        self.db.export_jsonl()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{EvalRecord, Registry, RegistryConfig};
    use crate::model::{ModelCard, ModelVersion};
    use tempfile::TempDir;

    fn eval(suite: &str) -> EvalRecord {
        EvalRecord {
            model_sha: "a".repeat(64),
            suite: suite.into(),
            suite_manifest_sha: "b".repeat(64),
            score: 0.5,
            n: 10,
            engine_version: "0.70.0".into(),
            engine_sha: "c".repeat(40),
            host: "lambda".into(),
            ts: chrono::DateTime::from_timestamp(1_790_000_000, 0).unwrap(),
        }
    }

    /// The same rows, written in `order`, into a fresh registry.
    fn registry(order: &[usize]) -> (TempDir, Registry) {
        let dir = TempDir::new().unwrap();
        let r = Registry::open(RegistryConfig::new(dir.path())).unwrap();
        for &i in order {
            match i {
                0 => r.add_lineage_edge("run-a", "model-a", "produced", None).unwrap(),
                1 => r.add_lineage_edge("data-a", "run-a", "consumed", None).unwrap(),
                2 => r.record_eval(&eval("apr-qa")).unwrap(),
                _ => r.record_eval(&eval("perplexity/wikitext2")).unwrap(),
            }
        }
        (dir, r)
    }

    /// FALSIFY-EXT-010 (I-7): the export is a function of the rows, not of the order they
    /// were written in, and the host-local autoincrement ids never leave the host.
    #[test]
    fn falsify_ext_010_export_is_byte_identical_and_drops_host_local_ids() {
        let (_a, a) = registry(&[0, 1, 2, 3]);
        let (_b, b) = registry(&[3, 1, 2, 0]);
        let ea = a.export_jsonl().unwrap();
        assert_eq!(ea, a.export_jsonl().unwrap(), "re-export of an unchanged registry");
        assert_eq!(ea, b.export_jsonl().unwrap(), "same rows, other insertion order");
        assert_eq!(ea.lines().count(), 4);
        for l in ea.lines() {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert!(v["row"].get("id").is_none(), "autoincrement id exported: {l}");
            assert!(!v["key"].as_str().unwrap().is_empty(), "{l}");
        }
        let mut sorted: Vec<&str> = ea.lines().collect();
        sorted.sort_unstable();
        assert_eq!(sorted, ea.lines().collect::<Vec<_>>());

        // A content change is a byte change.
        b.record_eval(&eval("humaneval")).unwrap();
        assert_ne!(ea, b.export_jsonl().unwrap());
    }

    #[test]
    fn ext_22_export_keeps_text_ids_and_refuses_an_unknown_table() {
        let (dir, r) = registry(&[]);
        let v = ModelVersion::new(1, 0, 0);
        r.register_model("m", &v, b"weights", ModelCard::new("m")).unwrap();
        let e = r.export_jsonl().unwrap();
        let row: serde_json::Value = serde_json::from_str(e.trim_end()).unwrap();
        assert_eq!(row["table"], "models");
        assert!(row["key"].as_str().unwrap().ends_with("|m|1.0.0"), "{row}");

        let conn = rusqlite::Connection::open(dir.path().join("registry.db")).unwrap();
        conn.execute_batch("CREATE TABLE shadow (x TEXT)").unwrap();
        let err = r.export_jsonl().unwrap_err().to_string();
        assert!(err.contains("table shadow is not in the export"), "{err}");
    }
}
