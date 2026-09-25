//! EXT-05 (aprender#4387): FALSIFY-EXT-004 (the train-verb delta on a tmp
//! pacha home) and FALSIFY-EXT-009 (engine identity refused at start_run).

use super::*;
use tempfile::TempDir;

fn clean_engine() -> EngineIdentity {
    EngineIdentity {
        apr_version: "0.71.0".into(),
        git_sha: "abc1234".into(),
        dirty: "0".into(),
    }
}

/// (runs, lineage edges, models) in the pacha home.
fn counts(home: &Path) -> (usize, usize, usize) {
    let reg = Registry::open(RegistryConfig::new(home)).expect("registry");
    let backend = PachaBackend::open(
        &RegistryConfig::new(home).db_path(),
        &home.join("tracking-metrics.db"),
    )
    .expect("backend");
    (
        backend.list_runs().expect("runs").len(),
        reg.count_lineage_edges().expect("lineage"),
        reg.storage_stats().expect("stats").model_count,
    )
}

struct Fixture {
    _dir: TempDir,
    home: PathBuf,
    base: PathBuf,
    data: PathBuf,
    out: PathBuf,
}

fn fixture() -> Fixture {
    let dir = TempDir::new().expect("tempdir");
    let home = dir.path().join("pacha");
    let base = dir.path().join("base.safetensors");
    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).expect("mkdir");
    std::fs::write(&base, b"base weights").expect("base");
    std::fs::write(data.join("train.jsonl"), b"{\"text\":\"a\"}\n").expect("data");
    std::fs::write(data.join("val.jsonl"), b"{\"text\":\"b\"}\n").expect("data");
    let out = dir.path().join("adapter.apr");
    Fixture {
        _dir: dir,
        home,
        base,
        data,
        out,
    }
}

/// FALSIFY-EXT-004: with the base pre-registered, one tracked run adds
/// runs +1, lineage +3 (base, dataset, produced) and models +1, and the
/// produced model's recorded sha256 is the output file's.
#[test]
fn falsify_ext_004_train_verb_delta_on_tmp_pacha_home() {
    let f = fixture();
    let reg = Registry::open(RegistryConfig::new(&f.home)).expect("registry");
    let base_id = reg
        .register_model(
            "base",
            &ModelVersion::new(1, 0, 0),
            &std::fs::read(&f.base).expect("read"),
            ModelCard::new("pre-registered base"),
        )
        .expect("pre-register base")
        .to_string();
    drop(reg);
    let before = counts(&f.home);

    let rec = Recorder::start_in(
        &f.home,
        &clean_engine(),
        "finetune",
        Some(&f.base),
        Some(&f.data),
    )
    .expect("start");
    std::fs::write(&f.out, b"adapter produced by training").expect("train writes output");
    let recorded = rec.finish(true, Some(&f.out)).expect("finish");

    let after = counts(&f.home);
    assert_eq!(
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        (1, 3, 1),
        "delta (runs, lineage, models)"
    );
    let want_sha = hash_file(&f.out).expect("hash").1;
    assert_eq!(
        recorded.produced_sha256.as_deref(),
        Some(want_sha.as_str()),
        "output sha"
    );

    // The base edge names the pre-registered model, not a bare hash, and the
    // produced model carries its sha256 and run id.
    let reg = Registry::open(RegistryConfig::new(&f.home)).expect("registry");
    let run = PachaBackend::open(
        &RegistryConfig::new(&f.home).db_path(),
        &f.home.join("tracking-metrics.db"),
    )
    .expect("backend")
    .load_run(&recorded.run_id)
    .expect("run");
    assert_eq!(run.params.get("base"), Some(&base_id));
    assert_eq!(run.params.get("output_sha256"), Some(&want_sha));
    assert!(run.params["dataset"].starts_with("blake3:"));
    assert_eq!(run.tags.get("git_sha").map(String::as_str), Some("abc1234"));
    assert!(matches!(run.status, RunStatus::Completed));
    let produced_id = recorded.produced_model_id.expect("produced");
    let produced = reg
        .get_model_by_id(&produced_id.parse().expect("model id"))
        .expect("produced model");
    assert_eq!(
        produced.card.extra.get("sha256"),
        Some(&serde_json::json!(want_sha))
    );
    assert_eq!(
        produced.card.extra.get("run_id"),
        Some(&serde_json::json!(recorded.run_id))
    );
}

/// A failed run is recorded as failed, with only its input edges and no
/// produced model, even when a partial output file exists.
#[test]
fn a_failed_run_registers_no_model() {
    let f = fixture();
    let before = counts(&f.home);
    let rec = Recorder::start_in(
        &f.home,
        &clean_engine(),
        "finetune",
        Some(&f.base),
        Some(&f.data),
    )
    .expect("start");
    std::fs::write(&f.out, b"partial").expect("partial");
    let recorded = rec.finish(false, Some(&f.out)).expect("finish");
    let after = counts(&f.home);
    assert_eq!(
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        (1, 2, 0)
    );
    assert_eq!(recorded.produced_model_id, None);
}

/// The dataset identity is content-addressed: renaming the directory keeps
/// it, and changing one byte changes it.
#[test]
fn dataset_identity_is_content_addressed() {
    let f = fixture();
    let a = blake3_tree(&f.data).expect("hash");
    let moved = f.data.with_file_name("moved");
    std::fs::rename(&f.data, &moved).expect("rename");
    assert_eq!(blake3_tree(&moved).expect("hash"), a);
    std::fs::write(moved.join("val.jsonl"), b"{\"text\":\"c\"}\n").expect("edit");
    assert_ne!(blake3_tree(&moved).expect("hash"), a);
}

/// An empty placeholder output (the deferred-export CUDA path) is not a model.
#[test]
fn an_empty_output_is_not_a_produced_artifact() {
    let f = fixture();
    std::fs::write(&f.out, b"").expect("touch");
    assert_eq!(produced_artifact(&f.out), None);
    let dir = f.out.with_extension("d");
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("model.safetensors"), b"w").expect("w");
    assert_eq!(produced_artifact(&dir), Some(dir.join("model.safetensors")));
}

/// FALSIFY-EXT-009: a dirty or unidentifiable engine is refused at
/// start_run, the refusal names --no-track, and nothing is written.
#[test]
fn falsify_ext_009_dirty_or_unidentified_engine_refused_at_start_run() {
    let f = fixture();
    for engine in [
        EngineIdentity {
            dirty: "1".into(),
            ..clean_engine()
        },
        EngineIdentity {
            dirty: "unknown".into(),
            ..clean_engine()
        },
        EngineIdentity {
            git_sha: "v0.71.0+no-git".into(),
            dirty: "unknown".into(),
            ..clean_engine()
        },
        EngineIdentity {
            git_sha: String::new(),
            ..clean_engine()
        },
        // Refused on the sha alone, whatever the dirty flag says.
        EngineIdentity {
            git_sha: "v0.71.0+no-git".into(),
            ..clean_engine()
        },
    ] {
        let err = Recorder::start_in(&f.home, &engine, "finetune", Some(&f.base), Some(&f.data))
            .err()
            .unwrap_or_else(|| panic!("{engine:?} must be refused"));
        assert!(err.to_string().contains("--no-track"), "{err}");
        assert!(
            !f.home.exists(),
            "a refused start wrote to pacha: {engine:?}"
        );
    }
    assert!(clean_engine().check().is_ok());
}

/// `--no-track` runs the verb without touching pacha, whatever the engine.
#[test]
fn no_track_trains_without_recording() {
    let mut ran = false;
    tracked("finetune", None, None, None, true, || {
        ran = true;
        Ok(())
    })
    .expect("untracked run");
    assert!(ran);
}
