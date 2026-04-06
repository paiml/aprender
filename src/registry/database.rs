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
    conn: Connection,
}

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
        Ok(db)
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
