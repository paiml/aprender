//! EXT-12 (aprender#4394): `apr model gate` end to end against a real pacha home.

use super::super::model_gate::tests::{fixture, item, Fixture, RUN};
use super::*;
use entrenar::tracking::Run;
use pacha::data::{DatasetManifest, DatasetVersion, ManifestRow, Origin};
use tempfile::TempDir;

struct Cli {
    f: Fixture,
    home: TempDir,
    work: TempDir,
}

impl Cli {
    /// The gate fixture, with its evidence and sealed set written to disk.
    fn new() -> Self {
        let f = fixture();
        let work = TempDir::new().unwrap();
        std::fs::write(
            work.path().join("evidence.json"),
            serde_json::to_vec(&f.evidence).unwrap(),
        )
        .unwrap();
        let sealed = DatasetManifest {
            rows: vec![ManifestRow {
                relpath: "eval/humaneval.jsonl".into(),
                bytes: 7,
                sha256: item(7),
                origin: Origin::Human,
                label: None,
            }],
        };
        std::fs::create_dir(work.path().join("sealed")).unwrap();
        std::fs::write(
            work.path().join("sealed/humaneval.json"),
            serde_json::to_vec(&sealed).unwrap(),
        )
        .unwrap();
        Self {
            f,
            home: TempDir::new().unwrap(),
            work,
        }
    }

    /// Record the lineage run and register the dataset in the pacha home.
    fn populate(&self) {
        let env = PachaEnv::open(self.home.path()).unwrap();
        let mut runs = env.runs;
        runs.save_run(&Run::new(RUN.into(), None, "ext-12".into()))
            .unwrap();
        for a in self.f.env.datasets.values() {
            env.registry
                .register_dataset_manifest(
                    "sft",
                    &DatasetVersion::new(1, 0, 0),
                    &a.manifest,
                    &self.f.sealed,
                )
                .unwrap();
        }
    }

    fn gate(&self, sealed: &str) -> Result<()> {
        run_gate(&GateArgs {
            dir: self.f.dir.path(),
            evidence: &self.work.path().join("evidence.json"),
            sealed: &self.work.path().join(sealed),
            engine_tarball: Some(&self.f.tarball),
            pacha_home: Some(self.home.path()),
            out: None,
            json: true,
        })
    }

    fn receipt(&self) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.f.dir.path().join(RECEIPT)).unwrap()).unwrap()
    }

    fn red(&self) -> Vec<String> {
        self.receipt()["gates"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|g| g["green"] == false)
            .map(|g| g["gate"].as_str().unwrap().to_string())
            .collect()
    }
}

#[test]
fn falsify_ext_015_cli_gates_a_real_pacha_home() {
    let c = Cli::new();
    // An empty pacha home: the lineage run and the dataset are unknown, the receipt is
    // still written, and the command fails naming exactly those gates.
    match c.gate("sealed") {
        Err(CliError::ValidationFailed(m)) => assert!(m.ends_with("M0, M4"), "{m}"),
        r => panic!("an empty pacha home must fail M0 and M4, got {r:?}"),
    }
    assert_eq!(c.red(), ["M0", "M4"]);

    c.populate();
    c.gate("sealed").expect("all seven gates green");
    let r = c.receipt();
    assert_eq!(r["all_green"], true);
    assert_eq!(r["sealed_items_checked"], 1);
    // The gate's own receipt in the release dir does not fail M0 on the next run.
    c.gate("sealed").expect("re-gating is idempotent");
}

#[test]
fn ext_12_cli_refuses_bad_inputs_before_gating() {
    let c = Cli::new();
    c.populate();
    // A sealed dir that does not exist loads as an empty set, which M4 refuses.
    assert!(c.gate("no-such-dir").is_err());
    assert_eq!(c.red(), ["M4"]);

    std::fs::write(c.work.path().join("evidence.json"), br#"{"m9": {}}"#).unwrap();
    assert!(matches!(c.gate("sealed"), Err(CliError::InvalidInput(_))));
}
