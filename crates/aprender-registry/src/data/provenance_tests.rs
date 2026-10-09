//! EXT-08 (aprender#4390): canonical manifest hash, origin, I-11 and I-14 at registration.

use super::*;
use crate::data::DatasetVersion;
use crate::registry::{Registry, RegistryConfig};
use proptest::prelude::*;
use tempfile::TempDir;

fn sha(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

fn row(relpath: &str, n: u8, origin: Origin) -> ManifestRow {
    ManifestRow {
        relpath: relpath.into(),
        bytes: u64::from(n) * 10,
        sha256: sha(n),
        origin,
        label: None,
    }
}

fn labeled(mut r: ManifestRow) -> ManifestRow {
    r.label = Some(ExternalLabel::TestResult { test: "falsify_x".into(), passed: true });
    r
}

fn manifest(rows: Vec<ManifestRow>) -> DatasetManifest {
    DatasetManifest { rows }
}

fn registry() -> (TempDir, Registry) {
    let dir = TempDir::new().unwrap();
    let reg = Registry::open(RegistryConfig::new(dir.path())).unwrap();
    (dir, reg)
}

fn v1() -> DatasetVersion {
    DatasetVersion::new(1, 0, 0)
}

proptest! {
    /// §3.3: permuting the rows leaves the canonical hash unchanged.
    #[test]
    fn ext_08_canonical_hash_is_permutation_invariant(
        keys in proptest::collection::btree_set((0u8..=255, 0u8..=255), 1..24),
        seed in any::<u64>(),
    ) {
        let rows: Vec<_> =
            keys.iter().map(|(p, n)| row(&format!("d/{p}.jsonl"), *n, Origin::Human)).collect();
        let base = manifest(rows.clone()).canonical_sha256();
        let mut shuffled = rows;
        // Deterministic Fisher-Yates from the seed.
        let mut s = seed;
        for i in (1..shuffled.len()).rev() {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            shuffled.swap(i, (s >> 33) as usize % (i + 1));
        }
        prop_assert_eq!(manifest(shuffled).canonical_sha256(), base);
    }
}

#[test]
fn ext_08_canonical_hash_sees_every_key_field_and_ignores_provenance() {
    let base = manifest(vec![row("a", 1, Origin::Human), row("b", 2, Origin::Upstream)]);
    let h = base.canonical_sha256();
    assert!(is_sha256_hex(&h));
    let mut m = base.clone();
    m.rows[0].relpath = "a2".into();
    assert_ne!(m.canonical_sha256(), h, "relpath is hashed");
    let mut m = base.clone();
    m.rows[0].bytes += 1;
    assert_ne!(m.canonical_sha256(), h, "bytes are hashed");
    let mut m = base.clone();
    m.rows[0].sha256 = sha(9);
    assert_ne!(m.canonical_sha256(), h, "sha256 is hashed");
    let mut m = base.clone();
    m.rows.pop();
    assert_ne!(m.canonical_sha256(), h, "the row set is hashed");
    let mut m = base;
    m.rows[0] = labeled(ManifestRow { origin: Origin::SelfGenerated, ..m.rows[0].clone() });
    assert_eq!(m.canonical_sha256(), h, "origin and label are provenance, not identity");
}

#[test]
fn falsify_ext_012_sealed_leak_refused() {
    let (_d, reg) = registry();
    let sealed = SealedItems::from_hashes([sha(7), sha(8)]);
    let leak = manifest(vec![
        row("train/a.jsonl", 1, Origin::Human),
        row("train/b.jsonl", 8, Origin::Upstream),
    ]);
    let err = reg.register_dataset_manifest("ds", &v1(), &leak, &sealed).unwrap_err().to_string();
    assert!(err.contains("I-11") && err.contains("train/b.jsonl"), "{err}");
    assert_eq!(reg.dataset_manifest_count().unwrap(), 0, "a contaminated manifest was stored");
    assert!(reg.get_dataset_manifest(&leak.canonical_sha256()).unwrap().is_none());

    // Control: the same registry admits the manifest without the sealed row.
    let clean = manifest(vec![row("train/a.jsonl", 1, Origin::Human)]);
    let got = reg.register_dataset_manifest("ds", &v1(), &clean, &sealed).unwrap();
    assert_eq!(got.sealed_items_checked, 2);
    assert_eq!(got.sealed_set_sha256, sealed.set_sha256());
    assert_eq!(reg.get_dataset_manifest(&got.canonical_sha256).unwrap(), Some(got));
}

#[test]
fn falsify_ext_013_unlabeled_self_generated_refused() {
    let (_d, reg) = registry();
    let none = SealedItems::default();
    let unlabeled =
        manifest(vec![row("a", 1, Origin::Human), row("gen/b", 2, Origin::SelfGenerated)]);
    let err =
        reg.register_dataset_manifest("ds", &v1(), &unlabeled, &none).unwrap_err().to_string();
    assert!(err.contains("I-14") && err.contains("gen/b"), "{err}");
    assert_eq!(
        reg.dataset_manifest_count().unwrap(),
        0,
        "an unlabeled self_generated row was stored"
    );

    // A label that names no resolver is no label.
    for empty in [
        ExternalLabel::HumanHrq { verdict: " ".into() },
        ExternalLabel::PrOutcome { pr: String::new(), merged: true },
        ExternalLabel::TestResult { test: String::new(), passed: true },
        ExternalLabel::LlamaCppVerified { oracle_sha256: "abc".into() },
    ] {
        let mut m = unlabeled.clone();
        m.rows[1].label = Some(empty.clone());
        assert!(m.admit(&none).is_err(), "{empty:?} must not admit a self_generated row");
    }

    // Control: every resolver kind admits it, and the synthetic fraction is recorded.
    for label in [
        ExternalLabel::HumanHrq { verdict: "accept".into() },
        ExternalLabel::PrOutcome { pr: "paiml/aprender#1".into(), merged: false },
        ExternalLabel::TestResult { test: "t".into(), passed: false },
        ExternalLabel::LlamaCppVerified { oracle_sha256: sha(3) },
    ] {
        let mut m = unlabeled.clone();
        m.rows[1].label = Some(label);
        let a = m.admit(&none).unwrap();
        assert!((a.synthetic_fraction - 0.5).abs() < 1e-12, "{}", a.synthetic_fraction);
    }
    // Non-self-generated rows need no label.
    let other = manifest(vec![row("a", 1, Origin::ExternalModel), row("b", 2, Origin::Upstream)]);
    assert_eq!(other.admit(&none).unwrap().synthetic_fraction, 0.0);
}

#[test]
fn ext_08_malformed_manifests_are_refused() {
    let none = SealedItems::default();
    assert!(manifest(vec![]).admit(&none).is_err(), "empty manifest");
    type Plant = (&'static str, fn(&mut ManifestRow));
    let plants: [Plant; 7] = [
        ("empty relpath", |r| r.relpath.clear()),
        ("absolute relpath", |r| r.relpath = "/etc/x".into()),
        ("dotdot relpath", |r| r.relpath = "a/../b".into()),
        ("empty component", |r| r.relpath = "a//b".into()),
        ("control char", |r| r.relpath = "a\nb".into()),
        ("short sha", |r| r.sha256.truncate(63)),
        ("uppercase sha", |r| r.sha256 = r.sha256.to_uppercase()),
    ];
    for (what, plant) in plants {
        let mut m = manifest(vec![row("ok", 1, Origin::Human), row("x", 0xab, Origin::Human)]);
        plant(&mut m.rows[1]);
        assert!(m.admit(&none).is_err(), "{what} must be refused");
    }
    let dup = manifest(vec![row("x", 1, Origin::Human), row("x", 2, Origin::Human)]);
    assert!(dup.admit(&none).is_err(), "duplicate relpath");
}

#[test]
fn ext_08_repeat_registration_is_refused() {
    let (_d, reg) = registry();
    let none = SealedItems::default();
    let m = manifest(vec![row("a", 1, Origin::Human)]);
    reg.register_dataset_manifest("ds", &v1(), &m, &none).unwrap();
    assert!(reg.register_dataset_manifest("ds2", &v1(), &m, &none).is_err(), "same canonical hash");
    let m2 = manifest(vec![row("b", 2, Origin::Human)]);
    assert!(reg.register_dataset_manifest("ds", &v1(), &m2, &none).is_err(), "same name+version");
    assert_eq!(reg.dataset_manifest_count().unwrap(), 1);
}

#[test]
fn ext_08_sealed_store_loads_every_json_manifest() {
    let dir = TempDir::new().unwrap();
    assert!(SealedItems::load_dir(&dir.path().join("absent")).unwrap().is_empty());
    let write = |name: &str, m: &DatasetManifest| {
        std::fs::write(dir.path().join(name), serde_json::to_vec(m).unwrap()).unwrap();
    };
    write("humaneval.json", &manifest(vec![row("p/1", 1, Origin::Upstream)]));
    write(
        "mbpp.json",
        &manifest(vec![row("p/2", 2, Origin::Upstream), row("p/3", 3, Origin::Upstream)]),
    );
    std::fs::write(dir.path().join("README.md"), "not a manifest").unwrap();
    let sealed = SealedItems::load_dir(dir.path()).unwrap();
    assert_eq!(sealed, SealedItems::from_hashes([sha(1), sha(2), sha(3)]));
    assert!(
        manifest(vec![row("t", 3, Origin::Human)]).admit(&sealed).is_err(),
        "loaded hash is sealed"
    );

    std::fs::write(dir.path().join("broken.json"), "{").unwrap();
    assert!(
        SealedItems::load_dir(dir.path()).is_err(),
        "an unreadable sealed manifest is an error, not a gap"
    );
}
