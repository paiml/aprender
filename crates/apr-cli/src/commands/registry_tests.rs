//! EXT-07 (aprender#4389): FALSIFY-EXT-006 — `apr registry lineage` walks a
//! 3-hop chain and refuses a cycle.

use super::*;
use pacha::model::{ModelCard, ModelVersion};
use tempfile::TempDir;

/// A pacha home with base -> run -> produced, where produced's card records
/// `sha256` and its file sits at `produced_file`.
struct Chain {
    _dir: TempDir,
    home: PathBuf,
    produced: String,
    produced_file: PathBuf,
}

fn chain() -> Chain {
    let dir = TempDir::new().expect("tempdir");
    let home = dir.path().join("pacha");
    let reg = Registry::open(RegistryConfig::new(&home)).expect("registry");
    let base = reg
        .register_model(
            "base",
            &ModelVersion::new(1, 0, 0),
            b"base",
            ModelCard::new("b"),
        )
        .expect("base")
        .to_string();
    let mut card = ModelCard::new("p");
    card.extra
        .insert("sha256".into(), serde_json::json!("ab".repeat(32)));
    let produced_file = dir.path().join("produced.apr");
    std::fs::write(&produced_file, b"produced").expect("write");
    let produced = reg
        .register_model("produced", &ModelVersion::new(1, 0, 0), b"produced", card)
        .expect("produced")
        .to_string();
    reg.add_lineage_edge(&base, "RUN1", "base", None)
        .expect("edge");
    reg.add_lineage_edge("RUN1", &produced, "produced", None)
        .expect("edge");
    Chain {
        _dir: dir,
        home,
        produced,
        produced_file,
    }
}

/// FALSIFY-EXT-006: a 3-hop fixture gives 3 nodes and 2 edges, whichever
/// way the model is named; a cycle is refused.
#[test]
fn falsify_ext_006_lineage_three_hop_and_cycle_refused() {
    let c = chain();
    let b3 = blake3::hash(b"produced").to_hex().to_string();
    for target in [
        c.produced.clone(),
        b3.clone(),
        format!("blake3:{b3}"),
        "ab".repeat(32),
        c.produced_file.display().to_string(),
    ] {
        let a = lineage_in(&c.home, &target).unwrap_or_else(|e| panic!("{target}: {e}"));
        assert_eq!(a.root, c.produced, "{target}");
        assert_eq!((a.nodes.len(), a.edges.len()), (3, 2), "{target}");
    }

    let reg = Registry::open(RegistryConfig::new(&c.home)).expect("registry");
    reg.add_lineage_edge(&c.produced, "RUN1", "base", None)
        .expect("plant cycle");
    let err = lineage_in(&c.home, &c.produced).expect_err("cycle must be refused");
    assert!(err.to_string().contains("lineage cycle"), "{err}");
}

/// A registered model with no recorded parents is a root, not an error.
#[test]
fn a_registered_root_has_an_empty_ancestry() {
    let c = chain();
    let base = lineage_in(&c.home, &c.produced).expect("chain").nodes[2].clone();
    let a = lineage_in(&c.home, &base).expect("root");
    assert_eq!((a.nodes, a.edges.len()), (vec![base], 0));
}

/// An unknown target is an error, not an empty ancestry.
#[test]
fn an_unknown_target_is_refused() {
    let c = chain();
    let err = lineage_in(&c.home, "no-such-model").expect_err("unknown");
    assert!(err.to_string().contains("no recorded lineage"), "{err}");
    let missing = c.home.with_file_name("nowhere");
    assert!(lineage_in(&missing, &c.produced).is_err());
    assert!(!missing.exists(), "reading lineage created a registry");
}

/// EXT-31 (aprender#4413), FALSIFY-CRUX-P-04-001: one `apr registry lineage`
/// answers which run, dataset sha and base produced a model. The planted
/// failure — a model registered with no producing run — is reported as an
/// orphan, never given a run. Datasets and bases of an UPSTREAM run (the
/// base's own producer) are not attributed to the model.
#[test]
fn falsify_crux_p_04_001_lineage_answers_run_dataset_base_or_orphan() {
    let c = chain();
    let reg = Registry::open(RegistryConfig::new(&c.home)).expect("registry");
    let base = lineage_in(&c.home, &c.produced).expect("chain").nodes[2].clone();
    let d1 = format!("blake3:{}", "d1".repeat(32));
    let d0 = format!("blake3:{}", "d0".repeat(32));
    reg.add_lineage_edge(&d1, "RUN1", "dataset", None)
        .expect("edge");
    // Upstream: the base itself was produced by RUN0 from dataset d0.
    reg.add_lineage_edge("RUN0", &base, "produced", None)
        .expect("edge");
    reg.add_lineage_edge(&d0, "RUN0", "dataset", None)
        .expect("edge");

    let p = provenance(&lineage_in(&c.home, &c.produced).expect("lineage"));
    assert_eq!(p.produced_by.as_deref(), Some("RUN1"));
    assert_eq!(p.datasets, vec![d1]);
    assert_eq!(p.bases, vec![base.clone()]);
    assert!(!p.is_orphan());

    // The base answers with its own run, not the model's.
    let pb = provenance(&lineage_in(&c.home, &base).expect("base"));
    assert_eq!(pb.produced_by.as_deref(), Some("RUN0"));
    assert_eq!(pb.datasets, vec![d0]);
    assert!(pb.bases.is_empty());

    // Planted: registered with no producing run -> orphan, RED.
    let orphan = reg
        .register_model(
            "orphan",
            &ModelVersion::new(1, 0, 0),
            b"orphan",
            ModelCard::new("o"),
        )
        .expect("orphan")
        .to_string();
    let po = provenance(&lineage_in(&c.home, &orphan).expect("orphan"));
    assert!(po.is_orphan(), "{po:?}");
    assert_eq!(po.produced_by, None);
    assert!(po.summary(&orphan).contains("orphan"), "{}", po.summary(&orphan));
    assert!(p.summary(&c.produced).contains("RUN1"));
}
