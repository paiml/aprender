//! SIM-3 (FLOW-005, #5511): the `pr-flow` scenario is a contract, run through the shipped binary.
//!
//! Each design must land inside its reference band, Little's law must close, and the same
//! seeds must give the same bytes. A scenario that returned a constant would fail the band for
//! three of the four designs; a scenario that read a clock or an unseeded RNG would fail the
//! byte comparison.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_aprender-simulate");

/// Design, reference merged/h (the +-5% band is the contract).
const WANT: [(&str, f64); 4] = [("D", 20.08), ("A", 4.96), ("G", 12.55), ("E", 21.37)];

fn block(design: &str) -> String {
    let out = Command::new(BIN)
        .args(["pr-flow", "--design", design, "--json", "--sha", "test"])
        .output()
        .expect("spawn the binary under test");
    assert!(
        out.status.success(),
        "pr-flow {design} exited {:?}",
        out.status
    );
    String::from_utf8(out.stdout).expect("utf-8 json")
}

fn num(json: &serde_json::Value, path: &[&str]) -> f64 {
    path.iter()
        .fold(json, |v, k| &v[*k])
        .as_f64()
        .unwrap_or(f64::NAN)
}

#[test]
fn every_design_is_inside_its_reference_band_and_littles_law_closes() {
    for (design, want) in WANT {
        let v: serde_json::Value = serde_json::from_str(&block(design)).expect("json");
        let got = num(&v, &["predicted", "merged_h"]);
        assert!(
            (got - want).abs() <= want * 0.05,
            "{design}: {got:.2} vs {want}"
        );
        let (w_min, l) = (
            num(&v, &["predicted", "W_min_mean"]),
            num(&v, &["predicted", "L"]),
        );
        let lam_w = got * w_min / 60.0;
        assert!(
            (lam_w - l).abs() <= l * 0.005,
            "{design}: lambda*W {lam_w:.3} vs L {l:.3}"
        );
        assert_eq!(v["sim3_contract"], "pass", "{design}");
    }
}

#[test]
fn same_seeds_give_the_same_bytes() {
    for (design, _) in WANT {
        assert_eq!(block(design), block(design), "{design}");
    }
}

#[test]
fn params_file_gives_actual_one_change_and_optimum_with_the_same_bytes() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let file = dir.join("pr-flow-params.json");
    std::fs::write(
        &file,
        r#"{"repo":"x","design":"D","params":{"cap":10,"r_pr":28,"p_red":0.62}}"#,
    )
    .expect("write params");
    let run = |args: &[&str]| {
        let out = Command::new(BIN)
            .args(["pr-flow", "--params"])
            .arg(&file)
            .args(args)
            .output()
            .expect("spawn");
        (
            out.status.success(),
            String::from_utf8(out.stdout).expect("utf-8"),
        )
    };
    let flags = ["--seeds", "2", "--hours", "100", "--warmup", "5", "--json"];
    let (ok, a) = run(&flags);
    assert!(ok);
    assert_eq!(a, run(&flags).1, "same params must give the same bytes");
    let v: serde_json::Value = serde_json::from_str(&a).expect("json");
    for k in ["actual", "optimum", "one_change", "binding_station"] {
        assert!(!v[k].is_null() || k == "binding_station", "missing {k}");
    }
    assert!(num(&v, &["optimum", "merged_h"]) > num(&v, &["actual", "merged_h"]));
    std::fs::write(&file, "not json").expect("write bad");
    let (ok, _) = run(&flags);
    assert!(!ok, "a malformed params file must fail, not print a number");
}

#[test]
fn json_block_reports_the_author_busy_fraction_beside_the_assumption() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_aprender-simulate"))
        .args(["pr-flow", "--design", "D", "--json"])
        .output()
        .expect("run");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    let rho = v["predicted"]["rho_a"].as_f64().expect("rho_a present");
    assert!((0.0..=1.0).contains(&rho), "rho_a {rho}");
    assert!((v["predicted"]["rho_a_assumed"].as_f64().expect("assumed") - 0.95).abs() < 1e-9);
}

#[test]
fn json_law_block_carries_eq7_within_2_percent_of_the_sim_on_design_d() {
    let v: serde_json::Value = serde_json::from_str(&block("D")).expect("json");
    let (sim, law) = (
        num(&v, &["predicted", "W_min_mean"]),
        num(&v, &["law", "W_law_min"]),
    );
    assert!(
        (sim - law).abs() <= law * 0.02,
        "sim W {sim:.1} vs law W {law:.1}"
    );
    let terms = num(&v, &["law", "checks_min"])
        + num(&v, &["law", "t_arm_min"])
        + num(&v, &["law", "rebase_min"])
        + num(&v, &["law", "W_mq_min"]);
    assert!((terms - law).abs() < 1e-9, "terms {terms} sum to {law}");
    assert_eq!(num(&v, &["law", "p_dirty"]), 0.0);
}
