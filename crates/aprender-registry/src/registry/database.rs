//! `SQLite` database for registry metadata.

use crate::data::{Dataset, DatasetVersion};
use crate::error::{PachaError, Result};
use crate::experiment::{ExperimentRun, RunId};
use crate::model::{Model, ModelId, ModelStage, ModelVersion};
use crate::recipe::{RecipeReference, RecipeVersion, TrainingRecipe};
use crate::storage::ContentAddress;
use rusqlite::{params, Connection};
use std::path::Path;

/// `SQLite` database for registry metadata.
pub struct RegistryDb {
    pub(super) conn: Connection,
}

/// A stored lineage row: `(from_id, to_id, edge_type, metadata_json)`.
pub type LineageRow = (String, String, String, Option<String>);

impl RegistryDb {
    /// Open or create a database at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.init_schema()?;
        db.init_evals_schema()?;
        db.init_dataset_manifest_schema()?;
        Ok(db)
    }

    /// HELIX-IDEA-007 — atomic point-in-time snapshot via SQLite
    /// `VACUUM INTO 'path'`. The target path MUST NOT already exist;
    /// SQLite refuses to overwrite, and we surface that refusal as
    /// `PachaError::Database` rather than silently truncating.
    ///
    /// Concurrent writers continue against the source DB; their changes
    /// are not visible in the snapshot. Reads block briefly only while
    /// VACUUM INTO copies pages.
    ///
    /// Contract: `contracts/apr-registry-snapshot-v1.yaml`
    /// (FALSIFY-SNAPSHOT-001..003).
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL fails — most commonly when `target`
    /// already exists.
    pub fn vacuum_into<P: AsRef<Path>>(&self, target: P) -> Result<()> {
        // SQLite parameter binding can't be used with VACUUM INTO; the
        // path is a quoted string literal in the SQL grammar. Construct
        // the statement defensively by escaping single quotes.
        let raw = target.as_ref().to_str().ok_or_else(|| {
            PachaError::Validation("snapshot path is not valid UTF-8".to_string())
        })?;
        let escaped = raw.replace('\'', "''");
        let sql = format!("VACUUM INTO '{escaped}'");
        self.conn.execute_batch(&sql)?;
        Ok(())
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r"
            -- Models
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content_size INTEGER NOT NULL,
                card_json TEXT NOT NULL,
                stage TEXT DEFAULT 'development',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(name, version)
            );

            CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
            CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage);

            -- Datasets
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content_size INTEGER NOT NULL,
                datasheet_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(name, version)
            );

            CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);

            -- Recipes
            CREATE TABLE IF NOT EXISTS recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                recipe_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(name, version)
            );

            CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(name);

            -- Experiment Runs
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                recipe_name TEXT,
                recipe_version TEXT,
                hyperparameters_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                run_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_recipe ON runs(recipe_name, recipe_version);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

            -- Lineage edges
            CREATE TABLE IF NOT EXISTS lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                metadata_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_lineage_from ON lineage(from_id);
            CREATE INDEX IF NOT EXISTS idx_lineage_to ON lineage(to_id);
            ",
        )?;
        Ok(())
    }

    // ==================== Models ====================

    /// Insert a model into the database.
    pub fn insert_model(&self, model: &Model) -> Result<()> {
        let card_json = serde_json::to_string(&model.card)?;
        self.conn.execute(
            r"INSERT INTO models (id, name, version, content_hash, content_size, card_json, stage, created_at, updated_at)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                model.id.to_string(),
                model.name,
                model.version.to_string(),
                model.content_address.hash_hex(),
                model.content_address.size(),
                card_json,
                model.stage.to_string(),
                model.created_at.to_rfc3339(),
                model.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Check if a model exists.
    pub fn model_exists(&self, name: &str, version: &ModelVersion) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM models WHERE name = ?1 AND version = ?2",
            params![name, version.to_string()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get a model by name and version.
    pub fn get_model(&self, name: &str, version: &ModelVersion) -> Result<Model> {
        let row = self.conn.query_row(
            r"SELECT id, name, version, content_hash, content_size, card_json, stage, created_at, updated_at
              FROM models WHERE name = ?1 AND version = ?2",
            params![name, version.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                ))
            },
        ).map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => PachaError::NotFound {
                kind: "model".to_string(),
                name: name.to_string(),
                version: version.to_string(),
            },
            e => PachaError::Database(e),
        })?;

        Self::row_to_model(row)
    }

    /// Get a model by ID.
    pub fn get_model_by_id(&self, id: &ModelId) -> Result<Model> {
        let row = self.conn.query_row(
            r"SELECT id, name, version, content_hash, content_size, card_json, stage, created_at, updated_at
              FROM models WHERE id = ?1",
            params![id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                ))
            },
        ).map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => PachaError::NotFound {
                kind: "model".to_string(),
                name: id.to_string(),
                version: "n/a".to_string(),
            },
            e => PachaError::Database(e),
        })?;

        Self::row_to_model(row)
    }

    fn row_to_model(
        row: (String, String, String, String, i64, String, String, String, String),
    ) -> Result<Model> {
        let (
            id_str,
            name,
            version_str,
            hash_hex,
            size,
            card_json,
            stage_str,
            created_str,
            updated_str,
        ) = row;

        // Parse hash from hex
        let hash_bytes = hex_decode(&hash_hex)?;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hash_bytes);

        // Safe conversion: size from DB should always be non-negative
        let size_u64 = u64::try_from(size).unwrap_or(0);

        Ok(Model {
            id: id_str
                .parse()
                .map_err(|_| PachaError::Validation("invalid model id".to_string()))?,
            name,
            version: version_str.parse()?,
            content_address: ContentAddress::new(hash, size_u64, crate::storage::Compression::None),
            card: serde_json::from_str(&card_json)?,
            stage: stage_str.parse()?,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map_err(|_| PachaError::Validation("invalid timestamp".to_string()))?
                .with_timezone(&chrono::Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                .map_err(|_| PachaError::Validation("invalid timestamp".to_string()))?
                .with_timezone(&chrono::Utc),
        })
    }

    /// List all versions of a model.
    pub fn list_model_versions(&self, name: &str) -> Result<Vec<ModelVersion>> {
        let mut stmt =
            self.conn.prepare("SELECT version FROM models WHERE name = ?1 ORDER BY version")?;
        let rows = stmt.query_map(params![name], |row| row.get::<_, String>(0))?;

        let mut versions = Vec::new();
        for row in rows {
            let version_str = row?;
            versions.push(version_str.parse()?);
        }
        Ok(versions)
    }

    /// List all model names.
    pub fn list_model_names(&self) -> Result<Vec<String>> {
        contract_pre_ols_fit!();
        let mut stmt = self.conn.prepare("SELECT DISTINCT name FROM models ORDER BY name")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        let mut names = Vec::new();
        for row in rows {
            names.push(row?);
        }
        Ok(names)
    }

    /// Update model stage.
    pub fn update_model_stage(&self, id: &ModelId, stage: ModelStage) -> Result<()> {
        let updated_at = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE models SET stage = ?1, updated_at = ?2 WHERE id = ?3",
            params![stage.to_string(), updated_at, id.to_string()],
        )?;
        Ok(())
    }

    /// The id of a model whose artifact has BLAKE3 hash `hash_hex`, if any
    /// (the oldest registration wins when two share an artifact).
    pub fn find_model_id_by_content_hash(&self, hash_hex: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM models WHERE content_hash = ?1 ORDER BY created_at LIMIT 1")?;
        let mut rows = stmt.query(params![hash_hex])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    /// Insert one lineage edge `from_id -> to_id` (EXT-001 EXT-05).
    pub fn insert_lineage_edge(
        &self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        metadata_json: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO lineage (from_id, to_id, edge_type, metadata_json) VALUES (?1, ?2, ?3, ?4)",
            params![from_id, to_id, edge_type, metadata_json],
        )?;
        Ok(())
    }

    /// Every lineage edge into `to_id`, oldest first, as
    /// `(from_id, to_id, edge_type, metadata_json)` (EXT-001 EXT-07).
    pub fn lineage_edges_into(&self, to_id: &str) -> Result<Vec<LineageRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT from_id, to_id, edge_type, metadata_json FROM lineage WHERE to_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(params![to_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// The id of a model whose card records `extra.sha256 == sha256_hex`, if
    /// any (oldest first; EXT-001 EXT-07).
    pub fn find_model_id_by_card_sha256(&self, sha256_hex: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM models WHERE json_extract(card_json, '$.extra.sha256') = ?1 ORDER BY created_at LIMIT 1",
        )?;
        let mut rows = stmt.query(params![sha256_hex])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    /// Count lineage edges.
    pub fn count_lineage_edges(&self) -> Result<usize> {
        let count: i64 =
            self.conn.query_row("SELECT COUNT(*) FROM lineage", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    /// Count models.
    pub fn count_models(&self) -> Result<usize> {
        let count: i64 =
            self.conn.query_row("SELECT COUNT(*) FROM models", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    // ==================== Datasets ====================

    /// Insert a dataset into the database.
    pub fn insert_dataset(&self, dataset: &Dataset) -> Result<()> {
        let datasheet_json = serde_json::to_string(&dataset.datasheet)?;
        self.conn.execute(
            r"INSERT INTO datasets (id, name, version, content_hash, content_size, datasheet_json, created_at)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                dataset.id.to_string(),
                dataset.name,
                dataset.version.to_string(),
                dataset.content_address.hash_hex(),
                dataset.content_address.size(),
                datasheet_json,
                dataset.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Check if a dataset exists.
    pub fn dataset_exists(&self, name: &str, version: &DatasetVersion) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM datasets WHERE name = ?1 AND version = ?2",
            params![name, version.to_string()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get a dataset by name and version.
    pub fn get_dataset(&self, name: &str, version: &DatasetVersion) -> Result<Dataset> {
        let row = self
            .conn
            .query_row(
                r"SELECT id, name, version, content_hash, content_size, datasheet_json, created_at
              FROM datasets WHERE name = ?1 AND version = ?2",
                params![name, version.to_string()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => PachaError::NotFound {
                    kind: "dataset".to_string(),
                    name: name.to_string(),
                    version: version.to_string(),
                },
                e => PachaError::Database(e),
            })?;

        let (id_str, name, version_str, hash_hex, size, datasheet_json, created_str) = row;

        let hash_bytes = hex_decode(&hash_hex)?;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hash_bytes);

        // Safe conversion: size from DB should always be non-negative
        let size_u64 = u64::try_from(size).unwrap_or(0);

        Ok(Dataset {
            id: id_str
                .parse()
                .map_err(|_| PachaError::Validation("invalid dataset id".to_string()))?,
            name,
            version: version_str.parse()?,
            content_address: ContentAddress::new(hash, size_u64, crate::storage::Compression::None),
            datasheet: serde_json::from_str(&datasheet_json)?,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map_err(|_| PachaError::Validation("invalid timestamp".to_string()))?
                .with_timezone(&chrono::Utc),
        })
    }

    /// List all dataset names.
    pub fn list_dataset_names(&self) -> Result<Vec<String>> {
        contract_pre_name_resolution!();
        let mut stmt = self.conn.prepare("SELECT DISTINCT name FROM datasets ORDER BY name")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        let mut names = Vec::new();
        for row in rows {
            names.push(row?);
        }
        Ok(names)
    }

    /// List all versions of a dataset.
    pub fn list_dataset_versions(&self, name: &str) -> Result<Vec<DatasetVersion>> {
        let mut stmt =
            self.conn.prepare("SELECT version FROM datasets WHERE name = ?1 ORDER BY version")?;
        let rows = stmt.query_map(params![name], |row| row.get::<_, String>(0))?;

        let mut versions = Vec::new();
        for row in rows {
            let version_str = row?;
            versions.push(version_str.parse()?);
        }
        Ok(versions)
    }

    /// Count datasets.
    pub fn count_datasets(&self) -> Result<usize> {
        let count: i64 =
            self.conn.query_row("SELECT COUNT(*) FROM datasets", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    // ==================== Recipes ====================

    /// Insert a recipe into the database.
    pub fn insert_recipe(&self, recipe: &TrainingRecipe) -> Result<()> {
        let recipe_json = serde_json::to_string(recipe)?;
        self.conn.execute(
            r"INSERT INTO recipes (id, name, version, recipe_json, created_at)
              VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                recipe.id.to_string(),
                recipe.name,
                recipe.version.to_string(),
                recipe_json,
                recipe.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Check if a recipe exists.
    pub fn recipe_exists(&self, name: &str, version: &RecipeVersion) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM recipes WHERE name = ?1 AND version = ?2",
            params![name, version.to_string()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get a recipe by name and version.
    pub fn get_recipe(&self, name: &str, version: &RecipeVersion) -> Result<TrainingRecipe> {
        let recipe_json: String = self
            .conn
            .query_row(
                "SELECT recipe_json FROM recipes WHERE name = ?1 AND version = ?2",
                params![name, version.to_string()],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => PachaError::NotFound {
                    kind: "recipe".to_string(),
                    name: name.to_string(),
                    version: version.to_string(),
                },
                e => PachaError::Database(e),
            })?;

        Ok(serde_json::from_str(&recipe_json)?)
    }

    /// List all recipe names.
    pub fn list_recipe_names(&self) -> Result<Vec<String>> {
        contract_pre_expand_recipe!();
        let mut stmt = self.conn.prepare("SELECT DISTINCT name FROM recipes ORDER BY name")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        let mut names = Vec::new();
        for row in rows {
            names.push(row?);
        }
        Ok(names)
    }

    /// List all versions of a recipe.
    pub fn list_recipe_versions(&self, name: &str) -> Result<Vec<RecipeVersion>> {
        let mut stmt =
            self.conn.prepare("SELECT version FROM recipes WHERE name = ?1 ORDER BY version")?;
        let rows = stmt.query_map(params![name], |row| row.get::<_, String>(0))?;

        let mut versions = Vec::new();
        for row in rows {
            let version_str = row?;
            versions.push(version_str.parse()?);
        }
        Ok(versions)
    }

    /// Count recipes.
    pub fn count_recipes(&self) -> Result<usize> {
        let count: i64 =
            self.conn.query_row("SELECT COUNT(*) FROM recipes", [], |row| row.get(0))?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    // ==================== Experiment Runs ====================

    /// Insert an experiment run.
    pub fn insert_run(&self, run: &ExperimentRun) -> Result<()> {
        contract_pre_configuration!();
        let hyperparams_json = serde_json::to_string(&run.hyperparameters)?;
        let run_json = serde_json::to_string(run)?;
        let (recipe_name, recipe_version) = run
            .recipe
            .as_ref()
            .map_or((None, None), |r| (Some(r.name.clone()), Some(r.version.to_string())));

        self.conn.execute(
            r"INSERT INTO runs (id, recipe_name, recipe_version, hyperparameters_json, status, started_at, finished_at, run_json)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                run.run_id.to_string(),
                recipe_name,
                recipe_version,
                hyperparams_json,
                run.status.to_string(),
                run.started_at.to_rfc3339(),
                run.finished_at.map(|t| t.to_rfc3339()),
                run_json,
            ],
        )?;
        Ok(())
    }

    /// Update an experiment run.
    pub fn update_run(&self, run: &ExperimentRun) -> Result<()> {
        let run_json = serde_json::to_string(run)?;
        self.conn.execute(
            r"UPDATE runs SET status = ?1, finished_at = ?2, run_json = ?3 WHERE id = ?4",
            params![
                run.status.to_string(),
                run.finished_at.map(|t| t.to_rfc3339()),
                run_json,
                run.run_id.to_string(),
            ],
        )?;
        Ok(())
    }

    /// Get an experiment run by ID.
    pub fn get_run(&self, run_id: &RunId) -> Result<ExperimentRun> {
        let run_json: String = self
            .conn
            .query_row(
                "SELECT run_json FROM runs WHERE id = ?1",
                params![run_id.to_string()],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => PachaError::NotFound {
                    kind: "run".to_string(),
                    name: run_id.to_string(),
                    version: "n/a".to_string(),
                },
                e => PachaError::Database(e),
            })?;

        Ok(serde_json::from_str(&run_json)?)
    }

    /// List runs for a recipe.
    pub fn list_runs_for_recipe(&self, recipe_ref: &RecipeReference) -> Result<Vec<ExperimentRun>> {
        contract_pre_expand_recipe!(recipe_ref);
        let mut stmt = self.conn.prepare(
            "SELECT run_json FROM runs WHERE recipe_name = ?1 AND recipe_version = ?2 ORDER BY started_at DESC"
        )?;

        let rows = stmt
            .query_map(params![recipe_ref.name, recipe_ref.version.to_string()], |row| {
                row.get::<_, String>(0)
            })?;

        let mut runs = Vec::new();
        for row in rows {
            let run_json = row?;
            runs.push(serde_json::from_str(&run_json)?);
        }
        Ok(runs)
    }
}

/// Decode hex string to bytes.
fn hex_decode(s: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(s.len() / 2);
    let chars: Vec<char> = s.chars().collect();

    for chunk in chars.chunks(2) {
        if chunk.len() != 2 {
            return Err(PachaError::Validation("invalid hex string".to_string()));
        }
        let high = hex_char_to_nibble(chunk[0])?;
        let low = hex_char_to_nibble(chunk[1])?;
        bytes.push((high << 4) | low);
    }

    Ok(bytes)
}

fn hex_char_to_nibble(c: char) -> Result<u8> {
    match c {
        '0'..='9' => Ok(c as u8 - b'0'),
        'a'..='f' => Ok(c as u8 - b'a' + 10),
        'A'..='F' => Ok(c as u8 - b'A' + 10),
        _ => Err(PachaError::Validation(format!("invalid hex char: {c}"))),
    }
}

// ==================== Evals (EXT-09, aprender#4391) ====================
// Kept at the end of the file so it rebases cleanly past the lineage writer (EXT-05).

const EVALS_DDL: &str = r"
CREATE TABLE IF NOT EXISTS evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_sha TEXT NOT NULL,
    suite TEXT NOT NULL,
    suite_manifest_sha TEXT NOT NULL,
    score REAL NOT NULL,
    n INTEGER NOT NULL,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    host TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evals_model ON evals(model_sha);
";

impl RegistryDb {
    fn init_evals_schema(&self) -> Result<()> {
        self.conn.execute_batch(EVALS_DDL)?;
        Ok(())
    }

    /// Insert one eval row. The caller validates it first.
    pub fn insert_eval(&self, e: &super::evals::EvalRecord) -> Result<()> {
        self.conn.execute(
            "INSERT INTO evals (model_sha, suite, suite_manifest_sha, score, n, engine_version, engine_sha, host, ts)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                e.model_sha,
                e.suite,
                e.suite_manifest_sha,
                e.score,
                i64::try_from(e.n).map_err(|_| PachaError::Validation("eval row: n overflows i64".into()))?,
                e.engine_version,
                e.engine_sha,
                e.host,
                e.ts.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Every eval row for a model sha, oldest first.
    pub fn list_evals_for_model(&self, model_sha: &str) -> Result<Vec<super::evals::EvalRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT model_sha, suite, suite_manifest_sha, score, n, engine_version, engine_sha, host, ts
             FROM evals WHERE model_sha = ?1 ORDER BY ts, id",
        )?;
        let rows = stmt.query_map(params![model_sha], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f64>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, String>(7)?,
                r.get::<_, String>(8)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                model_sha,
                suite,
                suite_manifest_sha,
                score,
                n,
                engine_version,
                engine_sha,
                host,
                ts,
            ) = row?;
            out.push(super::evals::EvalRecord {
                model_sha,
                suite,
                suite_manifest_sha,
                score,
                n: u64::try_from(n)
                    .map_err(|_| PachaError::Validation("eval row: negative n".into()))?,
                engine_version,
                engine_sha,
                host,
                ts: chrono::DateTime::parse_from_rfc3339(&ts)
                    .map_err(|e| PachaError::Validation(format!("eval row: bad ts {ts}: {e}")))?
                    .with_timezone(&chrono::Utc),
            });
        }
        Ok(out)
    }

    /// True when some model card records this sha256 in `extra.sha256` (EXT-05 writes it).
    pub fn model_sha_registered(&self, model_sha: &str) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM models WHERE json_extract(card_json, '$.extra.sha256') = ?1",
            params![model_sha],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }
}

// ==================== Dataset manifests (EXT-08, aprender#4390) ====================

const DATASET_MANIFESTS_DDL: &str = "
    CREATE TABLE IF NOT EXISTS dataset_manifests (
        canonical_sha256 TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT NOT NULL,
        admitted_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(name, version)
    );
";

impl RegistryDb {
    fn init_dataset_manifest_schema(&self) -> Result<()> {
        self.conn.execute_batch(DATASET_MANIFESTS_DDL)?;
        Ok(())
    }

    /// Store an admitted manifest. A repeated canonical hash or name+version is refused.
    pub fn insert_dataset_manifest(
        &self,
        name: &str,
        version: &str,
        admitted: &crate::data::AdmittedManifest,
    ) -> Result<()> {
        let json = serde_json::to_string(admitted)?;
        self.conn
            .execute(
                "INSERT INTO dataset_manifests (canonical_sha256, name, version, admitted_json, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![admitted.canonical_sha256, name, version, json, chrono::Utc::now().to_rfc3339()],
            )
            .map_err(|e| match e {
                rusqlite::Error::SqliteFailure(f, _)
                    if f.code == rusqlite::ErrorCode::ConstraintViolation =>
                {
                    PachaError::AlreadyExists {
                        kind: "dataset manifest".to_string(),
                        name: name.to_string(),
                        version: version.to_string(),
                    }
                }
                other => other.into(),
            })?;
        Ok(())
    }

    /// The admitted manifest with this canonical hash, if registered.
    pub fn get_dataset_manifest(
        &self,
        canonical_sha256: &str,
    ) -> Result<Option<crate::data::AdmittedManifest>> {
        use rusqlite::OptionalExtension;
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT admitted_json FROM dataset_manifests WHERE canonical_sha256 = ?1",
                params![canonical_sha256],
                |r| r.get(0),
            )
            .optional()?;
        json.map(|j| serde_json::from_str(&j).map_err(Into::into)).transpose()
    }

    /// Number of stored dataset manifests.
    pub fn dataset_manifest_count(&self) -> Result<usize> {
        let n: i64 =
            self.conn.query_row("SELECT COUNT(*) FROM dataset_manifests", [], |r| r.get(0))?;
        Ok(usize::try_from(n).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DatasetId, Datasheet};
    use crate::model::ModelCard;
    use tempfile::TempDir;

    fn setup() -> (TempDir, RegistryDb) {
        let dir = TempDir::new().unwrap();
        let db = RegistryDb::open(dir.path().join("test.db")).unwrap();
        (dir, db)
    }

    #[test]
    fn test_db_open() {
        let (_dir, _db) = setup();
    }

    #[test]
    fn test_hex_decode() {
        assert_eq!(hex_decode("00").unwrap(), vec![0]);
        assert_eq!(hex_decode("ff").unwrap(), vec![255]);
        assert_eq!(hex_decode("0123").unwrap(), vec![1, 35]);
        assert_eq!(hex_decode("deadbeef").unwrap(), vec![0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn test_model_crud() {
        let (_dir, db) = setup();

        let model = Model {
            id: ModelId::new(),
            name: "test".to_string(),
            version: ModelVersion::new(1, 0, 0),
            content_address: ContentAddress::from_bytes(b"test"),
            card: ModelCard::new("Test model"),
            stage: ModelStage::Development,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        db.insert_model(&model).unwrap();
        assert!(db.model_exists("test", &ModelVersion::new(1, 0, 0)).unwrap());

        let retrieved = db.get_model("test", &ModelVersion::new(1, 0, 0)).unwrap();
        assert_eq!(retrieved.id, model.id);
        assert_eq!(retrieved.name, model.name);
    }

    #[test]
    fn test_dataset_crud() {
        let (_dir, db) = setup();

        let dataset = Dataset {
            id: DatasetId::new(),
            name: "test-data".to_string(),
            version: DatasetVersion::new(1, 0, 0),
            content_address: ContentAddress::from_bytes(b"data"),
            datasheet: Datasheet::new("Test dataset"),
            created_at: chrono::Utc::now(),
        };

        db.insert_dataset(&dataset).unwrap();
        assert!(db.dataset_exists("test-data", &DatasetVersion::new(1, 0, 0)).unwrap());

        let retrieved = db.get_dataset("test-data", &DatasetVersion::new(1, 0, 0)).unwrap();
        assert_eq!(retrieved.id, dataset.id);
    }

    #[test]
    fn test_recipe_crud() {
        let (_dir, db) = setup();

        let recipe = TrainingRecipe::builder()
            .name("test-recipe")
            .version(RecipeVersion::new(1, 0, 0))
            .description("Test")
            .build();

        db.insert_recipe(&recipe).unwrap();
        assert!(db.recipe_exists("test-recipe", &RecipeVersion::new(1, 0, 0)).unwrap());

        let retrieved = db.get_recipe("test-recipe", &RecipeVersion::new(1, 0, 0)).unwrap();
        assert_eq!(retrieved.id, recipe.id);
    }
}
