//! EXT-28 (aprender#4410): C2 cell receipts, S-14, and FALSIFY-EXT-022 on
//! measured records.

use super::super::speed_ledger::{parse_row, ratchet, Ratchet};
use super::*;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn sha(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

fn record(arm: &str, tok_s: f64) -> ArmRecord {
    ArmRecord {
        arm: arm.into(),
        version: format!("{arm} 1.0"),
        engine_sha256: sha(1),
        model_sha256: sha(2),
        file_type: 15,
        vision_tensors: 0,
        served_model: "ours".into(),
        ollama_manifest_digest: None,
        conditions: Conditions {
            cpus: "0-15".into(),
            threads: 16,
            concurrency: 1,
            iterations: 3,
            max_tokens: 32,
            prompt_sha256: sha(3),
        },
        timing: Timing {
            load_ms: 1000.0,
            ttft_ms: 500.0,
            itl_ms: 100.0,
            e2e_ms: 3600.0,
            decode_tok_s: tok_s,
            peak_rss_kb: 4_000_000,
        },
        comparator: ComparatorBlock {
            command: vec![arm.into(), "serve".into()],
            version: format!("{arm} 1.0"),
            env_sha256: sha(4),
            artifact_sha256: sha(5),
            log_path: format!("logs/{arm}.server.log"),
            image: None,
            started_utc: "2026-09-26T00:00:00.000Z".into(),
            finished_utc: "2026-09-26T00:01:00.000Z".into(),
        },
    }
}

/// apr and llama.cpp measured, Ollama refused for its vision tower, mistral.rs not run.
fn cell() -> CellReceipt {
    let apr = record("apr", 2.0);
    let mut ollama = record("ollama", 9.0);
    ollama.model_sha256 = sha(6);
    ollama.vision_tensors = 393;
    let mut arms = BTreeMap::new();
    arms.insert(
        "llama.cpp".into(),
        classify(&apr, record("llama.cpp", 10.0)),
    );
    arms.insert("ollama".into(), classify(&apr, ollama));
    arms.insert(
        "mistral.rs".into(),
        ArmOutcome::NotRun {
            reason: "not built".into(),
        },
    );
    arms.insert("apr".into(), ArmOutcome::Measured(apr));
    CellReceipt {
        tag: "v0.70.0".into(),
        cell: "lambda-cpu".into(),
        not_run: None,
        arms,
    }
}

#[test]
fn a_like_for_like_cell_is_green_and_becomes_one_ledger_row() {
    let c = cell();
    assert!(check_cell(&c).is_empty(), "{:?}", check_cell(&c));
    assert!(matches!(c.arms["llama.cpp"], ArmOutcome::Measured(_)));
    let row = ledger_row(&c, &sha(9)).expect("row");
    // The row is a valid EXT-19 ledger line carrying only the measured arms.
    let back = parse_row(&serde_json::to_value(&row).expect("json")).expect("ledger row");
    let Outcome::Measured(m) = back.outcome else {
        panic!("not measured: {back:?}")
    };
    assert_eq!(m.apr_decode_tok_s, 2.0);
    assert_eq!(
        m.arms.iter().map(|a| a.arm.as_str()).collect::<Vec<_>>(),
        ["llama.cpp"]
    );
    // T28: no ratio key anywhere in the receipt or the row.
    let text =
        serde_json::to_string(&c).expect("json") + &serde_json::to_string(&row).expect("json");
    assert!(
        !text.contains("\"ratio\"") && !text.contains("speedup"),
        "{text}"
    );
}

/// S-14: each way an arm can differ from apr's run is refused, never normalised,
/// and a receipt that records it as measured is RED.
#[test]
fn each_unlike_arm_is_refused_and_red_if_recorded_as_measured() {
    let apr = record("apr", 2.0);
    type Plant = (&'static str, fn(&mut ArmRecord));
    let plants: [Plant; 8] = [
        ("m>1", |r| r.conditions.concurrency = 4),
        ("other threads", |r| r.conditions.threads = 32),
        ("other cpus", |r| r.conditions.cpus = "0-31".into()),
        ("other prompt", |r| r.conditions.prompt_sha256 = sha(7)),
        ("other length", |r| r.conditions.max_tokens = 64),
        ("other quant", |r| r.file_type = 7),
        ("vision path", |r| r.vision_tensors = 1),
        ("reference on another file", |r| r.model_sha256 = sha(8)),
    ];
    for (what, plant) in plants {
        let mut r = record("llama.cpp", 10.0);
        plant(&mut r);
        assert!(!unlike(&apr, &r).is_empty(), "{what}: like-for-like");
        assert!(
            matches!(
                classify(&apr, r.clone()),
                ArmOutcome::Refused {
                    record: Some(_),
                    ..
                }
            ),
            "{what}: not refused"
        );
        let mut c = cell();
        c.arms.insert("llama.cpp".into(), ArmOutcome::Measured(r));
        let f = check_cell(&c);
        assert!(f.iter().any(|f| f.contains("S-14")), "{what}: {f:?}");
    }
    // A competitor on its own file is fine; only the reference must share apr's.
    let mut m = record("mistral.rs", 3.0);
    m.model_sha256 = sha(8);
    assert!(unlike(&apr, &m).is_empty());
}

#[test]
fn each_plant_turns_the_cell_red() {
    type Plant = (&'static str, fn(&mut CellReceipt));
    let plants: [Plant; 12] = [
        ("empty tag", |c| c.tag.clear()),
        ("no apr arm", |c| {
            c.arms.remove("apr");
        }),
        ("apr not measured", |c| {
            c.arms
                .insert("apr".into(), ArmOutcome::NotRun { reason: "x".into() });
        }),
        ("a competitor arm missing", |c| {
            c.arms.remove("mistral.rs");
        }),
        ("an unknown arm", |c| {
            c.arms
                .insert("vllm".into(), ArmOutcome::NotRun { reason: "x".into() });
        }),
        ("a blank reason", |c| {
            c.arms.insert(
                "mistral.rs".into(),
                ArmOutcome::NotRun { reason: " ".into() },
            );
        }),
        ("a record under another arm's name", |c| {
            let r = record("apr", 10.0);
            c.arms.insert("llama.cpp".into(), ArmOutcome::Measured(r));
        }),
        ("a zero decode rate", |c| {
            if let Some(ArmOutcome::Measured(r)) = c.arms.get_mut("llama.cpp") {
                r.timing.decode_tok_s = 0.0;
            }
        }),
        ("a NaN TTFT", |c| {
            if let Some(ArmOutcome::Measured(r)) = c.arms.get_mut("apr") {
                r.timing.ttft_ms = f64::NAN;
            }
        }),
        ("a bad comparator hash", |c| {
            if let Some(ArmOutcome::Measured(r)) = c.arms.get_mut("llama.cpp") {
                r.comparator.env_sha256 = "abc".into();
            }
        }),
        ("apr with a vision path", |c| {
            if let Some(ArmOutcome::Measured(r)) = c.arms.get_mut("apr") {
                r.vision_tensors = 3;
            }
        }),
        ("a not_run cell with arms", |c| {
            c.not_run = Some("host down".into())
        }),
    ];
    for (what, plant) in plants {
        let mut c = cell();
        plant(&mut c);
        assert!(!check_cell(&c).is_empty(), "stayed green: {what}");
        assert!(ledger_row(&c, &sha(9)).is_err(), "{what}: became a row");
    }
}

/// A cell that was not run, or whose reference arm was not measured, still covers
/// its (tag, cell) pair — as `not_run`, with the reason.
#[test]
fn unmeasured_cells_become_not_run_rows() {
    let c = CellReceipt {
        tag: "v0.70.0".into(),
        cell: "gx10-cuda".into(),
        not_run: Some("REX-08 has not ruled".into()),
        arms: BTreeMap::new(),
    };
    let row = ledger_row(&c, &sha(9)).expect("row");
    assert!(matches!(&row.outcome, Outcome::NotRun(n) if n.reason.contains("REX-08")));

    let mut c = cell();
    c.arms.insert(
        "llama.cpp".into(),
        ArmOutcome::Refused {
            reason: "cannot load".into(),
            record: None,
        },
    );
    let row = ledger_row(&c, &sha(9)).expect("row");
    assert!(matches!(&row.outcome, Outcome::NotRun(n) if n.reason.contains("cannot load")));
}

// ---- the committed records (evidence/ext-001/EXT-28) ----

fn evidence() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evidence/ext-001/EXT-28")
}

/// A committed cell receipt and the ledger row it becomes, keyed by its bytes.
fn committed(rel: &str) -> (CellReceipt, Row) {
    let p = evidence().join(rel);
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let c: CellReceipt =
        serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let row = ledger_row(&c, &digest).unwrap_or_else(|f| panic!("{}: {f:?}", p.display()));
    (c, row)
}

/// Where a cell's receipt for the release tag lives: the measured cell's release
/// record is its fourth run; every other cell has one receipt under `cells/`.
fn release_receipt(cell: &str) -> String {
    if cell == "lambda-cpu" {
        "lambda-cpu/r4/cell.json".into()
    } else {
        format!("cells/{cell}.json")
    }
}

/// Every provisional cell has a committed receipt for the tag: 100% coverage.
#[test]
fn every_provisional_cell_is_covered() {
    let rows: Vec<Row> = PROVISIONAL_CELLS
        .iter()
        .map(|cell| committed(&release_receipt(cell)).1)
        .collect();
    let tags = vec!["v0.70.0".to_string()];
    let cells: Vec<String> = PROVISIONAL_CELLS.iter().map(|c| (*c).to_string()).collect();
    let missing = super::super::speed_ledger::uncovered(&rows, &tags, &cells).expect("ledger");
    assert!(missing.is_empty(), "{missing:?}");
    assert!(
        rows.iter()
            .any(|r| matches!(r.outcome, Outcome::Measured(_))),
        "no cell was measured"
    );
}

/// FALSIFY-EXT-022, on measured records. Three baseline runs of apr on the
/// lambda-cpu cell arm the ratchet; a fourth unpatched run stays GREEN; the same
/// apr with a planted sleep in the decode loop, measured the same way next to the
/// same llama.cpp, is RED.
#[test]
fn falsify_ext_022_planted_sleep_red() {
    let run = |name: &str| committed(&format!("lambda-cpu/{name}/cell.json"));
    let base: Vec<(CellReceipt, Row)> = ["r1", "r2", "r3"].iter().map(|n| run(n)).collect();
    let (control_cell, control) = run("r4");
    let (plant_cell, plant) = run("plant");
    let apr = |c: &CellReceipt| match &c.arms[APR_ARM] {
        ArmOutcome::Measured(r) => r.clone(),
        o => panic!("apr not measured: {o:?}"),
    };
    // The plant is another apr binary serving the same file under the same conditions.
    let (a, p) = (apr(&base[0].0), apr(&plant_cell));
    assert_ne!(
        a.engine_sha256, p.engine_sha256,
        "the plant is the same binary"
    );
    assert_eq!(a.model_sha256, p.model_sha256);
    assert_eq!(a.conditions, p.conditions);
    assert_eq!(apr(&control_cell).engine_sha256, a.engine_sha256);

    let judge = |last: &Row| {
        let mut rows: Vec<Row> = base.iter().map(|(_, r)| r.clone()).collect();
        rows.push(last.clone());
        let tags: Vec<String> = rows.iter().map(|r| r.tag.clone()).collect();
        ratchet(&rows, &tags, "lambda-cpu")
    };
    let three: Vec<Row> = base.iter().map(|(_, r)| r.clone()).collect();
    let tags: Vec<String> = three.iter().map(|r| r.tag.clone()).collect();
    assert!(matches!(
        ratchet(&three, &tags, "lambda-cpu"),
        Ratchet::Unarmed { records: 3 }
    ));
    let g = judge(&control);
    assert!(matches!(g, Ratchet::Green { .. }), "control: {g:?}");
    let r = judge(&plant);
    assert!(
        matches!(r, Ratchet::Red { .. }),
        "the measured plant stayed green: {r:?}"
    );
}
