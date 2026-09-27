//! EXT-27 (aprender#4409): M1b artifact quality, FALSIFY-EXT-021.

use super::super::model_gate::{receipt_markers, states_a_figure};
use super::*;

const OURS_SHA: &str = "aa";

fn sha(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

fn arm(name: &str, n: u8, kl: f64, top1: f64) -> QualityArm {
    QualityArm {
        arm: name.into(),
        file_sha256: sha(n),
        kl_mean: kl,
        top1,
    }
}

fn record(version: &str, ours_kl: f64, ours_top1: f64) -> QualityRecord {
    QualityRecord {
        version: version.into(),
        llama_cpp_commit: "d1d3c3396aa13a5f239109a822666c4870490ad5".into(),
        corpus_sha256: sha(0xc0),
        reference_sha256: sha(0xbf),
        ours: QualityArm {
            arm: "ours".into(),
            file_sha256: OURS_SHA.repeat(32),
            kl_mean: ours_kl,
            top1: ours_top1,
        },
        arms: vec![
            arm("unsloth-q4_k_m", 1, 0.030, 0.930),
            arm("bartowski-q4_k_m", 2, 0.028, 0.935),
        ],
    }
}

/// A manifest whose only model file is ours.
fn manifest() -> ReleaseManifest {
    serde_json::from_value(serde_json::json!({
        "line": "qwen3.5-4b-apr", "version": "0.2.0", "channel": "stable",
        "files": [{"name": "model.gguf", "format": "gguf", "quant": "Q4_K_M",
                   "bytes": 1, "sha256": OURS_SHA.repeat(32)}],
        "base": {"hf_id": "Qwen/Qwen3.5-4B", "revision": "a".repeat(40), "sha256": sha(9)},
        "lineage": [], "datasets": [],
        "engine": {"apr_version": "0.71.0", "crate_tarball_sha256": sha(8)},
        "license": {"spdx_or_name": "Apache-2.0", "upstream_notice_sha256": sha(7)},
    }))
    .expect("manifest")
}

/// Release 0.2.0 against 0.1.0, with our quant exactly as good as before.
fn evidence() -> M1bEvidence {
    M1bEvidence {
        current: record("0.2.0", 0.031, 0.929),
        baseline: Some(record("0.1.0", 0.031, 0.929)),
        receipt_id: "m1b-quality".into(),
    }
}

fn red(ev: &M1bEvidence) -> Vec<String> {
    let (r, _) = gate(&manifest(), Some(ev));
    assert_eq!(r.gate, "M1b");
    if r.green {
        Vec::new()
    } else {
        r.findings
    }
}

#[test]
fn control_is_green_and_reports_every_arm() {
    let (r, rep) = gate(&manifest(), Some(&evidence()));
    assert!(r.green, "{:?}", r.findings);
    let rep = rep.expect("report");
    assert!(!rep.first_record);
    assert_eq!(rep.gaps.len(), 2);
    let u = &rep.gaps[0];
    assert_eq!(u.arm, "unsloth-q4_k_m");
    assert!((u.kl_gap - 0.001).abs() < 1e-12, "{u:?}");
    assert!((u.top1_gap - 0.001).abs() < 1e-12, "{u:?}");
    assert_eq!(u.baseline_kl_gap, Some(u.kl_gap));
    assert!(!u.widened);
}

#[test]
fn the_first_record_sets_the_baseline() {
    let mut ev = evidence();
    ev.baseline = None;
    // Far behind every arm: an absolute gap is never a floor (R-15).
    ev.current.ours.kl_mean = 0.5;
    ev.current.ours.top1 = 0.5;
    let (r, rep) = gate(&manifest(), Some(&ev));
    assert!(r.green, "{:?}", r.findings);
    let rep = rep.expect("report");
    assert!(rep.first_record);
    assert!(rep
        .gaps
        .iter()
        .all(|g| g.baseline_kl_gap.is_none() && !g.widened));
}

/// FALSIFY-EXT-021: one tensor re-quantized to Q2_K raises our KL and lowers our
/// top-1 agreement; the competitor arms are unchanged, so every gap widens → RED.
#[test]
fn falsify_ext_021_degraded_quant_red() {
    let mut ev = evidence();
    ev.current.ours.kl_mean = 0.045;
    let r = red(&ev);
    assert!(
        r.iter()
            .any(|f| f.contains("unsloth-q4_k_m") && f.contains("KL")),
        "{r:?}"
    );

    let mut ev = evidence();
    ev.current.ours.top1 = 0.920;
    let r = red(&ev);
    assert!(r.iter().any(|f| f.contains("top-1")), "{r:?}");

    // The measured records (evidence/ext-001/EXT-27): the first record is green and
    // sets the baseline; the same quant with blk.10.ffn_down re-quantized to Q2_K,
    // measured by the same llama.cpp on the same corpus and reference, is RED.
    let first = measured("m1b-first-record.json");
    assert!(first.baseline.is_none());
    let (r, rep) = gate(&release_of(&first), Some(&first));
    assert!(r.green, "{:?}", r.findings);
    assert!(rep.expect("report").first_record);

    let plant = measured("m1b-plant-q2k.json");
    assert_eq!(plant.baseline.as_ref(), Some(&first.current));
    assert_ne!(
        plant.current.ours.file_sha256,
        first.current.ours.file_sha256
    );
    let (r, _) = gate(&release_of(&plant), Some(&plant));
    assert!(!r.green, "the measured Q2_K plant stayed green");
    for a in &plant.current.arms {
        assert!(
            r.findings
                .iter()
                .any(|f| f.contains(&format!("KL gap to {} widened", a.arm))),
            "{}: {:?}",
            a.arm,
            r.findings
        );
    }
}

fn measured(name: &str) -> M1bEvidence {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../evidence/ext-001/EXT-27")
        .join(name);
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// A release whose only file is the quant `ev` measured as ours.
fn release_of(ev: &M1bEvidence) -> ReleaseManifest {
    let mut m = manifest();
    m.files[0].sha256.clone_from(&ev.current.ours.file_sha256);
    m
}

/// Movement inside the noise band is not a widening; just past it is.
#[test]
fn the_epsilon_is_the_boundary() {
    let mut ev = evidence();
    ev.current.ours.kl_mean += M1B_KL_EPS * 0.5;
    ev.current.ours.top1 -= M1B_TOP1_EPS * 0.5;
    assert!(red(&ev).is_empty());
    let mut ev = evidence();
    ev.current.ours.kl_mean += M1B_KL_EPS * 2.0;
    assert_eq!(red(&ev).len(), 2, "both arms widen");
    let mut ev = evidence();
    ev.current.ours.top1 -= M1B_TOP1_EPS * 2.0;
    assert_eq!(red(&ev).len(), 2, "both arms widen");
}

/// A gap that narrows is green: getting better never blocks.
#[test]
fn a_narrowing_gap_is_green() {
    let mut ev = evidence();
    ev.current.ours.kl_mean = 0.010;
    ev.current.ours.top1 = 0.990;
    assert!(red(&ev).is_empty());
}

#[test]
fn each_plant_turns_m1b_red() {
    type Plant = (&'static str, fn(&mut M1bEvidence));
    let plants: [Plant; 14] = [
        ("measured against another llama.cpp", |e| {
            e.current.llama_cpp_commit = "b4000".into()
        }),
        ("the corpus changed since the baseline", |e| {
            e.current.corpus_sha256 = sha(0xc1)
        }),
        ("the reference changed since the baseline", |e| {
            e.current.reference_sha256 = sha(0xbe)
        }),
        ("the baseline used another llama.cpp", |e| {
            e.baseline.as_mut().unwrap().llama_cpp_commit = "b4000".into()
        }),
        ("our measured file is not a release file", |e| {
            e.current.ours.file_sha256 = sha(0x11)
        }),
        ("a NaN KL", |e| e.current.ours.kl_mean = f64::NAN),
        ("a negative KL", |e| e.current.arms[0].kl_mean = -0.1),
        ("a top-1 above 1", |e| e.current.arms[1].top1 = 1.01),
        ("a NaN in the baseline", |e| {
            e.baseline.as_mut().unwrap().arms[0].kl_mean = f64::NAN
        }),
        ("no competitor arm", |e| e.current.arms.clear()),
        ("every arm re-pinned: nothing compared", |e| {
            for a in &mut e.current.arms {
                a.file_sha256 = sha(0x33);
            }
        }),
        ("a duplicated arm", |e| {
            let a = e.current.arms[0].clone();
            e.current.arms.push(a);
        }),
        ("an arm file that is not a sha256", |e| {
            e.current.arms[0].file_sha256 = "abc".into()
        }),
        ("no receipt id for the card", |e| {
            e.receipt_id = String::new()
        }),
    ];
    for (what, plant) in plants {
        let mut ev = evidence();
        plant(&mut ev);
        assert!(!red(&ev).is_empty(), "stayed green: {what}");
    }
    let (r, rep) = gate(&manifest(), None);
    assert!(!r.green && rep.is_none(), "absent evidence is RED");
}

/// One re-pinned arm is reported and not compared; the others still gate.
#[test]
fn a_repinned_arm_is_not_compared_but_the_rest_gate() {
    let mut ev = evidence();
    ev.current.arms[0].file_sha256 = sha(0x33);
    ev.current.arms[0].kl_mean = 0.001;
    let (r, rep) = gate(&manifest(), Some(&ev));
    assert!(r.green, "{:?}", r.findings);
    let g = &rep.expect("report").gaps;
    assert_eq!(g[0].baseline_kl_gap, None);
    assert!(g[1].baseline_kl_gap.is_some());
    ev.current.ours.kl_mean = 0.05;
    assert_eq!(red(&ev).len(), 1, "only the compared arm can widen");
}

/// The card: absolute values per arm, every figure line cites the receipt (so M6
/// accepts it), and no ratio or gap is published (T28).
#[test]
fn the_card_renders_absolute_values_per_arm_with_no_ratio() {
    let ev = evidence();
    let md = card_markdown(&ev.current, &ev.receipt_id);
    for name in ["ours", "unsloth-q4_k_m", "bartowski-q4_k_m"] {
        assert!(
            md.lines().any(|l| l.contains(name)),
            "{name} missing:\n{md}"
        );
    }
    assert!(md.contains("0.0310") && md.contains("92.90"), "{md}");
    let mut figures = 0;
    for line in md.lines() {
        let (rest, ids) = receipt_markers(line);
        if states_a_figure(&rest) {
            figures += 1;
            assert_eq!(ids, vec!["m1b-quality"], "{line}");
        }
        let lower = line.to_ascii_lowercase();
        for banned in [
            "ratio", "×", " x ", "faster", "better", "worse", "gap", "vs ",
        ] {
            assert!(!lower.contains(banned), "{banned:?} in {line}");
        }
    }
    assert_eq!(figures, 3, "{md}");
    let (_, rep) = gate(&manifest(), Some(&ev));
    assert_eq!(rep.expect("report").card_markdown, md);
}
