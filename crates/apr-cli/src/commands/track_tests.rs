//! EXT-05 (aprender#4387): FALSIFY-EXT-004 (the train-verb delta on a tmp
//! pacha home) and FALSIFY-EXT-009 (engine identity refused at start_run).
//! EXT-06 (aprender#4388): FALSIFY-EXT-005 (derivation verbs record parent
//! edges into the produced model).

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
        &[(&f.base, "base"), (&f.data, "dataset")],
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
    let want_sha = format!(
        "{:x}",
        sha2::Sha256::digest(std::fs::read(&f.out).expect("read"))
    );
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
        &[(&f.base, "base"), (&f.data, "dataset")],
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
        let err = Recorder::start_in(
            &f.home,
            &engine,
            "finetune",
            &[(&f.base, "base"), (&f.data, "dataset")],
        )
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

/// FALSIFY-EXT-005: each derivation verb records one `<verb>_from` edge per
/// parent into the produced model (merge: 3 parents), and the produced
/// model's ancestry reaches every parent.
#[test]
fn falsify_ext_005_derivation_verbs_record_parent_edges() {
    for (verb, want_edge, parents) in [
        ("distill", "distilled_from", 1),
        ("merge", "merged_from", 3),
        ("quantize", "quantized_from", 1),
        ("prune", "pruned_from", 1),
    ] {
        assert_eq!(parent_edge(verb), Some(want_edge), "{verb}");
        let f = fixture();
        let files: Vec<PathBuf> = (0..parents)
            .map(|i| {
                let p = f.base.with_file_name(format!("parent{i}.apr"));
                std::fs::write(&p, format!("{verb} parent {i}")).expect("parent");
                p
            })
            .collect();
        let mut inputs: Vec<Input<'_>> = files.iter().map(|p| (p.as_path(), want_edge)).collect();
        inputs.push((&f.base, "base"));
        let rec = Recorder::start_in(&f.home, &clean_engine(), verb, &inputs).expect("start");
        std::fs::write(&f.out, format!("{verb} output")).expect("output");
        let recorded = rec.finish(true, Some(&f.out)).expect("finish");
        let produced = recorded.produced_model_id.expect("produced");

        let reg = Registry::open(RegistryConfig::new(&f.home)).expect("registry");
        let into = reg.lineage_edges_into(&produced).expect("edges");
        let from_parents: Vec<_> = into.iter().filter(|e| e.edge_type == want_edge).collect();
        assert_eq!(from_parents.len(), parents, "{verb}: parent edges");
        assert_eq!(
            into.iter().filter(|e| e.edge_type == "produced").count(),
            1,
            "{verb}: produced edge"
        );
        // Parents point at the model, never at the run; the base points at the run.
        assert!(reg
            .lineage_edges_into(&recorded.run_id)
            .expect("run edges")
            .iter()
            .all(|e| e.edge_type == "base"));
        let ancestry = reg.ancestry(&produced).expect("ancestry");
        for p in &files {
            let node = format!("blake3:{}", blake3_file(p).expect("hash"));
            assert!(
                ancestry.nodes.contains(&node),
                "{verb}: {node} not an ancestor"
            );
        }
        assert!(ancestry.nodes.contains(&recorded.run_id), "{verb}: run");
    }
    assert_eq!(parent_edge("finetune"), None);
}

/// `--no-track` runs the verb without touching pacha, whatever the engine.
#[test]
fn no_track_trains_without_recording() {
    let mut ran = false;
    tracked("finetune", &[], None, true, || {
        ran = true;
        Ok(())
    })
    .expect("untracked run");
    assert!(ran);
}

/// EXT-05 overhead receipt (run by hand, `--ignored --nocapture`): wall
/// time the recorder adds to one run for a base of `EXT05_BASE_MB` (default
/// 1000), a 50 MB dataset and a 50 MB adapter.
#[test]
#[ignore = "timing receipt, not a gate"]
fn ext05_recorder_overhead_receipt() {
    let mb = |v: &str, d| {
        std::env::var(v)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let f = fixture();
    let fill = |p: &Path, n: usize| {
        let chunk: Vec<u8> = (0..1usize << 20).map(|i| (i * 31 % 251) as u8).collect();
        let mut w = std::fs::File::create(p).expect("create");
        for _ in 0..n {
            std::io::Write::write_all(&mut w, &chunk).expect("write");
        }
    };
    fill(&f.base, mb("EXT05_BASE_MB", 1000));
    fill(&f.data.join("train.jsonl"), 50);
    fill(&f.out, 50);
    let t = std::time::Instant::now();
    let rec = Recorder::start_in(
        &f.home,
        &clean_engine(),
        "finetune",
        &[(&f.base, "base"), (&f.data, "dataset")],
    )
    .expect("start");
    let start_s = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    rec.finish(true, Some(&f.out)).expect("finish");
    let finish_s = t.elapsed().as_secs_f64();
    println!(
        "EXT05_OVERHEAD start_s={start_s:.3} finish_s={finish_s:.3} total_s={:.3}",
        start_s + finish_s
    );
}
