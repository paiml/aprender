//! EXT-11 (aprender#4393): FALSIFY-EXT-014 — `apr model pack` is deterministic, and its
//! manifest validates against `model-release-v1.schema.json`.

use super::super::model_gate::ReleaseManifest;
use super::super::track::{EngineIdentity, Recorder};
use super::*;
use pacha::data::{DatasetManifest, DatasetVersion, ManifestRow, Origin, SealedItems};
use pacha::model::{ModelCard, ModelVersion};
use serde_json::{json, Value};
use tempfile::TempDir;

const SCHEMA: &str = include_str!("../../../../contracts/schemas/model-release-v1.schema.json");

/// A pacha home where a finetune run turned a registered base plus a registered
/// dataset into a produced model.
struct Fixture {
    dir: TempDir,
    home: PathBuf,
    model: String,
    run: String,
    dataset: String,
    base_sha: String,
}

fn fixture() -> Fixture {
    let dir = TempDir::new().expect("tempdir");
    let w = dir.path();
    let home = w.join("pacha");
    let reg = Registry::open(RegistryConfig::new(&home)).expect("registry");

    let base_sha = sha256_hex(b"base-weights");
    let mut card = ModelCard::new("base");
    card.extra.insert("sha256".into(), json!(base_sha));
    reg.register_model("base", &ModelVersion::new(1, 0, 0), b"base-weights", card)
        .expect("base");
    std::fs::write(w.join("base.apr"), b"base-weights").expect("write");

    std::fs::create_dir(w.join("data")).expect("mkdir");
    std::fs::write(w.join("data/train.jsonl"), b"{\"q\":1}\n").expect("write");
    let rows = DatasetManifest {
        rows: vec![ManifestRow {
            relpath: "train.jsonl".into(),
            bytes: 8,
            sha256: sha256_hex(b"{\"q\":1}\n"),
            origin: Origin::Human,
            label: None,
        }],
    };
    let dataset = reg
        .register_dataset_manifest(
            "sft",
            &DatasetVersion::new(1, 0, 0),
            &rows,
            &SealedItems::default(),
        )
        .expect("dataset")
        .canonical_sha256;

    let engine = EngineIdentity {
        apr_version: "0.71.0".into(),
        git_sha: "abc1234".into(),
        dirty: "0".into(),
    };
    let (base, data) = (w.join("base.apr"), w.join("data"));
    let rec = Recorder::start_in(
        &home,
        &engine,
        "finetune",
        &[(base.as_path(), "base"), (data.as_path(), "dataset")],
    )
    .expect("start");
    std::fs::write(w.join("tuned.apr"), b"tuned-weights").expect("write");
    let recorded = rec
        .finish(true, Some(&w.join("tuned.apr")))
        .expect("finish");

    for (name, body) in [
        ("LICENSE.txt", "Apache-2.0 text"),
        ("NOTICE.txt", "upstream notice"),
        ("apr-0.71.0.crate", "tarball"),
        ("README.md", "# card"),
    ] {
        std::fs::write(w.join(name), body).expect("write");
    }
    Fixture {
        home,
        model: recorded.produced_model_id.expect("produced"),
        run: recorded.run_id,
        dataset,
        base_sha,
        dir,
    }
}

impl Fixture {
    fn args(&self, out: &str) -> PackArgs {
        let w = self.dir.path();
        PackArgs {
            model: self.model.clone(),
            line: "paiml/qwen3.5-4b-apr".into(),
            version: "0.1.0-rc.1".into(),
            channel: "rc".into(),
            base_hf_id: "Qwen/Qwen3.5-4B".into(),
            base_revision: "main@0123abc".into(),
            base_sha256: self.base_sha.clone(),
            datasets: vec![self.dataset.clone()],
            engine_tarball: w.join("apr-0.71.0.crate"),
            license: "Apache-2.0".into(),
            license_file: w.join("LICENSE.txt"),
            notice_file: w.join("NOTICE.txt"),
            files: vec![w.join("README.md")],
            out: w.join(out),
        }
    }
}

/// Every file of a release dir, by name.
fn tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|e| {
            let e = e.expect("entry");
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("read"),
            )
        })
        .collect()
}

fn schema_errors(v: &Value) -> Vec<String> {
    let schema: Value = serde_json::from_str(SCHEMA).expect("schema json");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");
    validator.iter_errors(v).map(|e| e.to_string()).collect()
}

/// FALSIFY-EXT-014: two packs of one model are byte-identical, and the manifest
/// validates against its schema and parses as the gate's `ReleaseManifest`.
#[test]
fn falsify_ext_014_pack_byte_identical() {
    let f = fixture();
    let first = pack(&f.home, &f.args("rel-a")).expect("pack a");
    let second = pack(&f.home, &f.args("rel-b")).expect("pack b");
    assert_eq!(first, second);
    let (a, b) = (
        tree(&f.dir.path().join("rel-a")),
        tree(&f.dir.path().join("rel-b")),
    );
    assert_eq!(a, b, "two packs differ");
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        ["LICENSE", "NOTICE", "README.md", MANIFEST, "tuned.apr"]
    );
    assert_eq!(
        a["tuned.apr"], b"tuned-weights",
        "bytes from the pacha store"
    );

    let v: Value = serde_json::from_slice(&a[MANIFEST]).expect("manifest json");
    assert_eq!(schema_errors(&v), Vec::<String>::new());
    assert_eq!(v["lineage"], json!([f.run]));
    assert_eq!(v["datasets"], json!([f.dataset]));
    assert_eq!(
        v["engine"]["apr_version"], "0.71.0",
        "the producing run's engine"
    );
    assert_eq!(v["engine"]["crate_tarball_sha256"], sha256_hex(b"tarball"));
    assert_eq!(
        v["license"]["upstream_notice_sha256"],
        sha256_hex(b"upstream notice")
    );
    assert_eq!(v["gates"], json!({}));
    let names: Vec<&str> = first.files.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        ["LICENSE", "NOTICE", "README.md", "tuned.apr"],
        "sorted"
    );

    let gate: ReleaseManifest = serde_json::from_slice(&a[MANIFEST]).expect("gate reads it");
    assert_eq!(gate.files, first.files);
    assert_eq!(gate.lineage, first.lineage);

    let err = pack(&f.home, &f.args("rel-a")).expect_err("rel-a is not empty");
    assert!(err.to_string().contains("written once"), "{err}");
}

/// The schema refuses each way a manifest can be malformed.
#[test]
fn the_schema_refuses_a_malformed_manifest() {
    let f = fixture();
    let m = pack(&f.home, &f.args("rel")).expect("pack");
    let good = serde_json::to_value(&m).expect("value");
    assert!(schema_errors(&good).is_empty());
    let cases: [(&str, fn(&mut Value)); 9] = [
        ("no lineage", |v| {
            v.as_object_mut().expect("obj").remove("lineage");
        }),
        ("empty lineage", |v| v["lineage"] = json!([])),
        ("no files", |v| v["files"] = json!([])),
        ("short file hash", |v| {
            v["files"][0]["sha256"] = json!("abc")
        }),
        ("path in a file name", |v| {
            v["files"][0]["name"] = json!("a/b")
        }),
        ("bad channel", |v| v["channel"] = json!("beta")),
        ("bad version", |v| v["version"] = json!("1.0")),
        ("unknown gate", |v| v["gates"] = json!({"M9": "r"})),
        ("unknown field", |v| v["built_at"] = json!("2026-09-25")),
    ];
    for (what, mutate) in cases {
        let mut v = good.clone();
        mutate(&mut v);
        assert!(!schema_errors(&v).is_empty(), "{what} passed the schema");
    }
}

/// Channel and version agree: rc needs -rc.N, released has none, nothing is yanked.
#[test]
fn version_and_channel_must_agree() {
    for (version, channel, ok) in [
        ("0.1.0-rc.1", "rc", true),
        ("0.1.0", "released", true),
        ("0.1.0", "rc", false),
        ("0.1.0-rc.1", "released", false),
        ("0.1.0", "yanked", false),
        ("0.1", "released", false),
        ("0.1.0-rc.", "rc", false),
        ("0.1.x", "released", false),
        ("0.1.0.4", "released", false),
    ] {
        assert_eq!(
            check_version(version, channel).is_ok(),
            ok,
            "{version} {channel}"
        );
    }
}

/// Nothing pacha cannot vouch for is packed.
#[test]
fn pack_refuses_what_pacha_cannot_vouch_for() {
    let f = fixture();
    let refused = |mutate: &dyn Fn(&mut PackArgs), want: &str| {
        let mut a = f.args("out");
        mutate(&mut a);
        let err = pack(&f.home, &a).expect_err(want).to_string();
        assert!(err.contains(want), "want `{want}`, got {err}");
        assert!(
            !a.out.exists(),
            "{want}: a refused pack wrote {}",
            a.out.display()
        );
    };
    refused(&|a| a.datasets.clear(), "no --dataset manifest");
    refused(&|a| a.datasets = vec!["cd".repeat(32)], "not registered");
    refused(
        &|a| a.base_sha256 = "ef".repeat(32),
        "pacha records the base",
    );
    refused(&|a| a.base_sha256 = "EF".repeat(32), "64 lowercase hex");
    refused(&|a| a.model = "ab".repeat(32), "not a registered model");
    refused(
        &|a| a.files.push(a.license_file.with_file_name("LICENSE")),
        "No such file",
    );
    refused(&|a| a.base_revision = " ".into(), "must not be empty");
    refused(&|a| a.version = "0.1.0".into(), "does not match");

    std::fs::write(f.dir.path().join("LICENSE"), "dup").expect("write");
    refused(
        &|a| a.files.push(a.license_file.with_file_name("LICENSE")),
        "share this name",
    );
}

/// A registered model no run produced is an orphan (I-1).
#[test]
fn an_orphan_model_is_not_packed() {
    let f = fixture();
    let reg = Registry::open(RegistryConfig::new(&f.home)).expect("registry");
    let orphan = reg
        .register_model(
            "orphan",
            &ModelVersion::new(1, 0, 0),
            b"o",
            ModelCard::new("o"),
        )
        .expect("orphan")
        .to_string();
    let mut a = f.args("out");
    a.model = orphan;
    let err = pack(&f.home, &a).expect_err("orphan").to_string();
    assert!(err.contains("no run recorded as producing it"), "{err}");
}
