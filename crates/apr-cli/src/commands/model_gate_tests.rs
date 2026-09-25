//! EXT-12 (aprender#4394): `apr model gate` M0..M6 — each gate RED on its own plant.

use super::super::model_gate_m2::arms::{Arm, ArmIdentity, ArmRole};
use super::super::model_gate_m2::{ReleaseClass, Suite, SuiteData};
use super::*;
use pacha::data::{DatasetManifest, ManifestRow, Origin};
use std::collections::BTreeMap;
use std::path::PathBuf;
use tempfile::TempDir;

const PRE: M2Prereg = M2Prereg::REX_001;
pub(crate) const RUN: &str = "01J8ZQ5N6W3M4K2P7R9S1T0V8X";

#[derive(Default)]
pub(crate) struct FakeEnv {
    pub(crate) runs: BTreeSet<String>,
    pub(crate) datasets: BTreeMap<String, AdmittedManifest>,
}

impl GateEnv for FakeEnv {
    fn run_resolves(&self, run_id: &str) -> Result<bool, String> {
        Ok(self.runs.contains(run_id))
    }
    fn dataset_manifest(&self, sha: &str) -> Result<Option<AdmittedManifest>, String> {
        Ok(self.datasets.get(sha).cloned())
    }
}

fn sha_of(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

pub(crate) fn item(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

const CARD: &str = "# paiml/qwen3.5-4b-apr v0.1.0-rc.1\n\n\
    A packaging release of Qwen3.5-4B.\n\n\
    1. Parity cosine 0.9993 vs llama.cpp [receipt:m1-parity]\n\
    2. HumanEval pass@1 0.80 [receipt:m2-sealed]\n";

/// A release dir whose every gate is green, plus what it was gated with.
pub(crate) struct Fixture {
    pub(crate) dir: TempDir,
    pub(crate) _tar: TempDir,
    pub(crate) tarball: PathBuf,
    pub(crate) manifest: serde_json::Value,
    pub(crate) evidence: GateEvidence,
    pub(crate) env: FakeEnv,
    pub(crate) sealed: SealedItems,
}

fn file_entry(name: &str, format: &str, bytes: &[u8]) -> serde_json::Value {
    serde_json::json!({"name": name, "format": format, "bytes": bytes.len(), "sha256": sha_of(bytes)})
}

fn cand() -> Vec<bool> {
    (0..100).map(|i| i < 80).collect()
}

pub(crate) fn fixture() -> Fixture {
    let dir = TempDir::new().unwrap();
    let files: [(&str, &str, &[u8]); 4] = [
        ("model.gguf", "gguf", b"GGUF\x03\x00\x00\x00tensors"),
        ("LICENSE", "license", b"Apache License 2.0\n"),
        ("NOTICE", "notice", b"Qwen3.5 NOTICE\n"),
        ("README.md", "card", CARD.as_bytes()),
    ];
    for (n, _, b) in files {
        std::fs::write(dir.path().join(n), b).unwrap();
    }
    let tar = TempDir::new().unwrap();
    let tarball = tar.path().join("aprender-0.71.0.crate");
    std::fs::write(&tarball, b"crate tarball bytes").unwrap();

    let dataset = DatasetManifest {
        rows: vec![ManifestRow {
            relpath: "train/a.jsonl".into(),
            bytes: 10,
            sha256: item(1),
            origin: Origin::Human,
            label: None,
        }],
    };
    let admitted = dataset.admit(&SealedItems::default()).unwrap();
    let mut env = FakeEnv::default();
    env.runs.insert(RUN.into());
    env.datasets
        .insert(admitted.canonical_sha256.clone(), admitted.clone());

    let manifest = serde_json::json!({
        "line": "paiml/qwen3.5-4b-apr",
        "version": "0.1.0-rc.1",
        "channel": "rc",
        "files": files.iter().map(|(n, f, b)| file_entry(n, f, b)).collect::<Vec<_>>(),
        "base": {"hf_id": "Qwen/Qwen3.5-4B", "revision": "a".repeat(40), "sha256": item(9)},
        "lineage": [RUN],
        "datasets": [admitted.canonical_sha256],
        "engine": {"apr_version": "0.71.0", "crate_tarball_sha256": sha_of(b"crate tarball bytes")},
        "gates": {},
        "license": {"spdx_or_name": "Apache-2.0", "upstream_notice_sha256": sha_of(b"Qwen3.5 NOTICE\n")},
        "recipe": {"prompt_sha256": item(8), "decoding": "greedy"},
    });
    let evidence = GateEvidence {
        m1: Some(M1Evidence {
            cosine: 0.9993,
            llama_cpp_commit: "d1d3c3396".into(),
            greedy_token_agreement: Some(0.97),
        }),
        m2: Some(M2Evidence {
            class: ReleaseClass::First,
            suites: vec![Suite {
                name: "humaneval".into(),
                data: SuiteData::Binary {
                    candidate: cand(),
                    baseline: None,
                },
            }],
            arms: vec![Arm {
                identity: ArmIdentity::Artifact {
                    name: "qwen3.5-4b-stock".into(),
                    sha256: item(9),
                },
                role: ArmRole::Blocking,
                suites: vec![Suite {
                    name: "humaneval".into(),
                    data: SuiteData::Binary {
                        candidate: cand(),
                        baseline: Some(cand()),
                    },
                }],
            }],
        }),
        m3: Some(M3Evidence {
            probes: ["hello", "code", "review"]
                .map(|id| Probe {
                    id: id.into(),
                    run_answered: true,
                    serve_answered: true,
                })
                .to_vec(),
            parse_rate: Some(0.95),
        }),
        receipt_ids: vec!["m1-parity".into(), "m2-sealed".into()],
    };
    let mut f = Fixture {
        dir,
        _tar: tar,
        tarball,
        manifest,
        evidence,
        env,
        sealed: SealedItems::from_hashes([item(7)]),
    };
    f.write_manifest();
    f
}

impl Fixture {
    fn write_manifest(&mut self) {
        std::fs::write(
            self.dir.path().join(MANIFEST),
            serde_json::to_vec(&self.manifest).unwrap(),
        )
        .unwrap();
    }

    fn gate(&self) -> GateReceipt {
        run(
            &PRE,
            &GateInputs {
                dir: self.dir.path(),
                evidence: &self.evidence,
                sealed: &self.sealed,
                engine_tarball: Some(&self.tarball),
                env: &self.env,
            },
        )
        .expect("manifest readable")
    }

    fn red(&self) -> Vec<&'static str> {
        self.gate()
            .gates
            .iter()
            .filter(|g| !g.green)
            .map(|g| g.gate)
            .collect()
    }

    /// Replace a release file's bytes and re-hash it in the manifest.
    fn rewrite(&mut self, name: &str, bytes: &[u8]) {
        std::fs::write(self.dir.path().join(name), bytes).unwrap();
        let files = self.manifest["files"].as_array_mut().unwrap();
        let f = files.iter_mut().find(|f| f["name"] == name).unwrap();
        f["bytes"] = bytes.len().into();
        f["sha256"] = sha_of(bytes).into();
        self.write_manifest();
    }
}

#[test]
fn ext_12_green_release_passes_every_gate_deterministically() {
    let f = fixture();
    let r = f.gate();
    let names: Vec<_> = r.gates.iter().map(|g| g.gate).collect();
    assert_eq!(names, ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]);
    assert!(r.all_green, "{:#?}", r.gates);
    assert_eq!(r.schema, "model-gate-receipt-v1");
    assert_eq!(r.sealed_items_checked, 1);
    assert_eq!(r.m3_parse_rate, Some(0.95));
    // The M2 section carries the stock arm's levels for a first release.
    let arm = &r.m2.as_ref().unwrap().arms[0];
    assert_eq!(arm.suites[0].arm_level, 0.8);
    // No timestamp: the same inputs give the same receipt bytes.
    let a = serde_json::to_vec(&r).unwrap();
    assert_eq!(a, serde_json::to_vec(&f.gate()).unwrap());
}

type Plant = (&'static str, &'static str, fn(&mut Fixture));

#[test]
fn falsify_ext_015_each_gate_turns_red_on_its_plant() {
    let plants: [Plant; 23] = [
        ("M0", "one byte of the model flipped", |f| {
            std::fs::write(
                f.dir.path().join("model.gguf"),
                b"GGUF\x03\x00\x00\x00tensorz",
            )
            .unwrap();
        }),
        ("M0", "an unhashed file in the release", |f| {
            std::fs::write(f.dir.path().join("extra.bin"), b"x").unwrap();
        }),
        ("M0", "the engine tarball is not the recorded one", |f| {
            std::fs::write(&f.tarball, b"a different crate").unwrap();
        }),
        ("M0", "a lineage run that does not resolve", |f| {
            f.env.runs.clear();
        }),
        ("M0", "no lineage at all", |f| {
            f.manifest["lineage"] = serde_json::json!([]);
            f.write_manifest();
        }),
        ("M0", "a path-escaping file name", |f| {
            f.manifest["files"][0]["name"] = "../model.gguf".into();
            f.write_manifest();
        }),
        ("M1", "cosine below 0.98", |f| {
            f.evidence.m1.as_mut().unwrap().cosine = 0.979
        }),
        ("M1", "a NaN cosine", |f| {
            f.evidence.m1.as_mut().unwrap().cosine = f64::NAN
        }),
        ("M1", "parity against another llama.cpp", |f| {
            f.evidence.m1.as_mut().unwrap().llama_cpp_commit = "b4000".into();
        }),
        ("M1", "no parity evidence", |f| f.evidence.m1 = None),
        ("M2", "below stock on the sealed suite", |f| {
            let arm = &mut f.evidence.m2.as_mut().unwrap().arms[0];
            arm.suites[0].data = SuiteData::Binary {
                candidate: cand(),
                baseline: Some((0..100).map(|i| i < 95).collect()),
            };
        }),
        ("M2", "no stock arm", |f| {
            f.evidence.m2.as_mut().unwrap().arms.clear()
        }),
        ("M2", "no sealed-suite evidence", |f| f.evidence.m2 = None),
        ("M3", "apr serve misses a probe", |f| {
            f.evidence.m3.as_mut().unwrap().probes[1].serve_answered = false;
        }),
        ("M3", "an empty probe set", |f| {
            f.evidence.m3.as_mut().unwrap().probes.clear()
        }),
        ("M3", "no probe evidence", |f| f.evidence.m3 = None),
        ("M4", "a training row became a sealed item", |f| {
            f.sealed = SealedItems::from_hashes([item(7), item(1)]);
        }),
        ("M4", "an unregistered dataset", |f| f.env.datasets.clear()),
        ("M4", "an empty sealed set checks nothing", |f| {
            f.sealed = SealedItems::default();
        }),
        ("M5", "the NOTICE is not the upstream one", |f| {
            f.manifest["license"]["upstream_notice_sha256"] = item(3).into();
            f.write_manifest();
        }),
        ("M5", "no NOTICE shipped", |f| {
            std::fs::remove_file(f.dir.path().join("NOTICE")).unwrap();
            f.manifest["files"]
                .as_array_mut()
                .unwrap()
                .retain(|x| x["name"] != "NOTICE");
            f.write_manifest();
        }),
        ("M6", "a card figure with no receipt", |f| {
            f.rewrite(
                "README.md",
                format!("{CARD}3. MBPP pass@1 0.61\n").as_bytes(),
            );
        }),
        ("M6", "a card citing an unknown receipt", |f| {
            f.rewrite(
                "README.md",
                format!("{CARD}3. MBPP pass@1 0.61 [receipt:made-up]\n").as_bytes(),
            );
        }),
    ];
    for (gate, what, plant) in plants {
        let mut f = fixture();
        assert!(f.red().is_empty(), "control for {what}");
        plant(&mut f);
        let r = f.gate();
        let red: Vec<_> = r
            .gates
            .iter()
            .filter(|g| !g.green)
            .map(|g| g.gate)
            .collect();
        assert_eq!(red, [gate], "{what}: {:#?}", r.gates);
        assert!(!r.all_green, "{what}");
    }
}

#[test]
fn ext_12_m0_needs_the_engine_tarball() {
    let f = fixture();
    let r = run(
        &PRE,
        &GateInputs {
            dir: f.dir.path(),
            evidence: &f.evidence,
            sealed: &f.sealed,
            engine_tarball: None,
            env: &f.env,
        },
    )
    .unwrap();
    assert!(
        !r.gates[0].green && r.gates[0].findings[0].contains("unverified"),
        "{:?}",
        r.gates[0]
    );
}

#[test]
fn ext_12_card_figures_skip_identifiers_and_list_markers() {
    for (line, figure) in [
        ("# paiml/qwen3.5-4b-apr v0.1.0-rc.1", false),
        ("A packaging release of Qwen3.5-4B.", false),
        ("1. See the lineage", false),
        ("llama.cpp d1d3c3396", false),
        ("cosine 0.9993", true),
        ("measured 2026-09-25", true),
        ("12. pass@1 0.8", true),
        ("scored 80 of 100", true),
        ("(95 % CI)", true),
    ] {
        assert_eq!(states_a_figure(line), figure, "{line:?}");
    }
    assert_eq!(
        receipt_markers("x 1 [receipt: a ] y [receipt:b]"),
        ("x 1  y ".into(), vec!["a", "b"])
    );
}

#[test]
fn ext_12_evidence_file_round_trips_and_refuses_unknown_fields() {
    let f = fixture();
    let json = serde_json::to_string(&f.evidence).unwrap();
    let back: GateEvidence = serde_json::from_str(&json).unwrap();
    assert_eq!(back, f.evidence);
    assert!(
        serde_json::from_str::<GateEvidence>(r#"{"m7": {}}"#).is_err(),
        "a typo is not a skip"
    );
    // An empty evidence file parses, and every evidence-backed gate is RED.
    let empty: GateEvidence = serde_json::from_str("{}").unwrap();
    let mut f = f;
    f.evidence = empty;
    let red = f.red();
    assert_eq!(
        red,
        ["M1", "M2", "M3", "M6"],
        "M6: the card cites receipts nobody vouched for"
    );
}

#[test]
fn ext_12_unreadable_manifest_is_an_error_not_a_receipt() {
    let f = fixture();
    std::fs::write(f.dir.path().join(MANIFEST), b"{").unwrap();
    let got = run(
        &PRE,
        &GateInputs {
            dir: f.dir.path(),
            evidence: &f.evidence,
            sealed: &f.sealed,
            engine_tarball: Some(&f.tarball),
            env: &f.env,
        },
    );
    assert!(got.is_err());
}
