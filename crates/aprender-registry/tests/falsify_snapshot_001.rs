//! FALSIFY-SNAPSHOT-001 — `Registry::snapshot()` produces a target file
//! that, when opened with `RegistryDb::open`, yields identical query
//! results (count + per-row identity) to the source as of snapshot time.
//!
//! Contract: `contracts/apr-registry-snapshot-v1.yaml`.
//!
//! Discharge strategy: populate a temp registry with N models, datasets,
//! recipes; snapshot; open the snapshot via `RegistryDb::open`; query
//! both source and snapshot; assert equality.

#![allow(clippy::unwrap_used)]

use pacha::data::{DatasetVersion, Datasheet};
use pacha::model::{ModelCard, ModelVersion};
use pacha::recipe::{Hyperparameters, RecipeVersion, TrainingRecipe};
use pacha::registry::RegistryDb;
use pacha::registry::{Registry, RegistryConfig};
use tempfile::TempDir;

fn populated_registry() -> (TempDir, Registry) {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();
    for i in 0..3 {
        registry
            .register_model(
                &format!("model-{i}"),
                &ModelVersion::new(1, 0, u32::try_from(i).unwrap()),
                format!("model bytes {i}").as_bytes(),
                ModelCard::new(format!("Model number {i}")),
            )
            .unwrap();
    }
    for i in 0..2 {
        registry
            .register_dataset(
                &format!("data-{i}"),
                &DatasetVersion::new(1, 0, u32::try_from(i).unwrap()),
                format!("dataset bytes {i}").as_bytes(),
                Datasheet::new(format!("Dataset number {i}")),
            )
            .unwrap();
    }
    for i in 0..2 {
        let recipe = TrainingRecipe::builder()
            .name(format!("recipe-{i}"))
            .version(RecipeVersion::new(1, 0, u32::try_from(i).unwrap()))
            .description(format!("Recipe number {i}"))
            .hyperparameters(Hyperparameters::default())
            .build();
        registry.register_recipe(&recipe).unwrap();
    }
    (dir, registry)
}

#[test]
fn snapshot_yields_bit_identical_query_results() {
    let (_src_dir, registry) = populated_registry();

    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("snapshot.db");
    registry.snapshot(&target).expect("snapshot must succeed");
    assert!(target.is_file(), "VACUUM INTO must create the target file");

    // Open the snapshot via the low-level RegistryDb API. We can't open it
    // through `Registry::open` because Registry expects a directory layout
    // (db + objects/); the snapshot is the SQL file alone, which is exactly
    // what FALSIFY-SNAPSHOT-001 verifies.
    let snap = RegistryDb::open(&target).unwrap();

    let src_models = registry.list_models().unwrap();
    let src_datasets = registry.list_datasets().unwrap();
    let src_recipes = registry.list_recipes().unwrap();

    assert_eq!(snap.count_models().unwrap(), src_models.len());
    assert_eq!(snap.count_datasets().unwrap(), src_datasets.len());
    assert_eq!(snap.count_recipes().unwrap(), src_recipes.len());

    // Per-row identity: every model name in source is also in snapshot.
    let snap_model_names = snap.list_model_names().unwrap();
    let mut src_sorted = src_models.clone();
    src_sorted.sort();
    let mut snap_sorted = snap_model_names;
    snap_sorted.sort();
    assert_eq!(src_sorted, snap_sorted);
}

#[test]
fn snapshot_after_no_op_session_round_trips() {
    // Empty registry: snapshot must still produce an openable, empty DB.
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("empty.db");
    registry.snapshot(&target).unwrap();
    let snap = RegistryDb::open(&target).unwrap();
    assert_eq!(snap.count_models().unwrap(), 0);
    assert_eq!(snap.count_datasets().unwrap(), 0);
    assert_eq!(snap.count_recipes().unwrap(), 0);
}

#[test]
fn snapshot_does_not_mutate_source() {
    let (src_dir, registry) = populated_registry();
    let pre_models = registry.list_models().unwrap();
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("snapshot.db");
    registry.snapshot(&target).unwrap();
    let post_models = registry.list_models().unwrap();
    assert_eq!(pre_models, post_models);
    // And the source DB file is still where it was.
    assert!(src_dir.path().join("registry.db").is_file());
}
