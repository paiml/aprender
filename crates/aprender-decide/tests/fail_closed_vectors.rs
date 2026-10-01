//! FALSIFY-LAYA-GATE-010: both D-19 demo runs are refused by `pack` AND by `verify` on the exact
//! bytes, each with its DOCUMENTED refusal, and nothing is written (plan 08-09; D-07, D-19).
//!
//! ARMED only by BOTH `LAYA_MODEL_DIR` (the declared base snapshot) and
//! `LAYA_FAIL_CLOSED_VECTORS=1`; otherwise it prints
//! `SKIP: fail-closed vectors not armed (LAYA_MODEL_DIR + LAYA_FAIL_CLOSED_VECTORS=1)` and returns
//! (never `#[ignore]`). The run dirs are gitignored local evidence from plan 08-08, so CI compiles
//! this and runs nothing until plan 08-12.
//!
//! ```text
//! LAYA_FAIL_CLOSED_VECTORS=1 \
//! LAYA_MODEL_DIR=~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4e… \
//!   cargo test -p aprender-decide --release --test fail_closed_vectors -- --nocapture
//! ```
//!
//! The vectors are read from `contracts/laya-finetune-gate-v1.yaml` `demo.fail_closed_vectors`
//! and identified by sha256(recipe.json) == the recorded recipe_id — never by re-running them
//! (08-08: MPS training is not bitwise reproducible).
//!
//! THE PER-VECTOR REFUSAL (user decision option A, 2026-09-26; debug session
//! laya-rescore-drift). `pack_rescore_probs_abs` stays 1e-5, so:
//! - early_stopping `3d4b91da…` must refuse with `GateFailed` naming ONLY the ece_post clause
//!   (exit 3), after both 280-row re-scores pass;
//! - fixed_epochs `d0f4e40d…` must refuse with `RescoreDrift` on the FINE-TUNED re-score
//!   (exit 2): its torch fp32 probabilities are up to 3.7e-5 from a float64 reference, so no
//!   fp32 port meets the bar on it, and its gate is never reached.
//!
//! A contract vector with no entry in [`expected`] FAILS the test: a new vector needs a
//! declared refusal before it can be accepted as fail-closed.
//!
//! UNDER laya-finetune-gate-v1 1.4.0 / laya-parity-v1 2.0.0 (plan 08-15) THE TABLE IS UNCHANGED.
//! Neither run dir carries `rescore-noise.json`, so both re-scores are held to the A1 floor
//! (`pack_rescore_probs_abs`, 1.0e-5), which this test ASSERTS and prints as the bound used; and
//! neither recipe carries `seed_selection`, so the legacy rule applies, whose `SeedPolicyMissing`
//! comes only after the gate and is never reached by either vector.

mod common;

use aprender_decide::artifact::artifact_sha256_hex;
use aprender_decide::pack::pack_run_dir;
use aprender_decide::verify::{
    pack_for_serving, verify_path, GateClause, ProbsWhich, VerifyError, VerifyPolicy,
};
use common::{contract, policy, workspace_root};
use std::path::{Path, PathBuf};

const ENV_MODEL: &str = "LAYA_MODEL_DIR";
const ENV_ARM: &str = "LAYA_FAIL_CLOSED_VECTORS";
const ENV_DATA: &str = "LAYA_VECTORS_DATA_DIR";
const DEFAULT_DATA: &str = "data/decide/tweet-stance-16";

/// The documented refusal of one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    /// `GateFailed` with the clause set exactly `[EcePost]` (exit 3).
    GateFailedEcePost,
    /// `RescoreDrift { which: FineTuned }` (exit 2).
    RescoreDriftFineTuned,
}

/// Keyed by the FULL recipe_id (option A).
fn expected(recipe_id: &str) -> Option<Expected> {
    match recipe_id {
        "3d4b91daf86772bcb23e5342c2dff4bb6467f5d9f254f7833ac7f61a8f2f5375" => {
            Some(Expected::GateFailedEcePost)
        }
        "d0f4e40d39425e68d503f557f4f660eb9a73a4fb258da01f777b6f49362bcf20" => {
            Some(Expected::RescoreDriftFineTuned)
        }
        _ => None,
    }
}

/// `(recipe_id, run dir)` of every `demo.fail_closed_vectors` entry.
fn vectors() -> Vec<(String, PathBuf)> {
    let gate = contract("laya-finetune-gate-v1.yaml");
    let entries = gate["demo"]["fail_closed_vectors"]
        .as_sequence()
        .expect("demo.fail_closed_vectors");
    entries
        .iter()
        .map(|e| {
            let s = e.as_str().expect("vector entry is a string");
            let rid = s
                .split("recipe_id ")
                .nth(1)
                .and_then(|t| t.get(..64))
                .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))
                .unwrap_or_else(|| panic!("no recipe_id in {s:?}"))
                .to_string();
            let dir = s
                .split("run dir ")
                .nth(1)
                .and_then(|t| t.split_whitespace().next())
                .unwrap_or_else(|| panic!("no run dir in {s:?}"));
            (rid, workspace_root().join(dir.trim_end_matches('/')))
        })
        .collect()
}

fn report(run: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(run.join("gate-report.json")).expect("gate-report.json"))
        .expect("parse gate-report.json")
}

/// Assert `e` is the vector's documented refusal and return the CLI's line for it.
fn check_refusal(
    side: &str,
    want: Expected,
    e: &VerifyError,
    rep: &serde_json::Value,
    policy: &VerifyPolicy,
) -> String {
    match (want, e) {
        (Expected::GateFailedEcePost, VerifyError::GateFailed(g)) => {
            assert_eq!(e.exit_code(), 3, "{side}");
            let floor = policy.rescore_probs_abs.to_bits();
            assert_eq!(
                g.rescore_bound.to_bits(),
                floor,
                "{side}: no record -> floor"
            );
            assert_eq!(
                g.zs_rescore_bound.to_bits(),
                floor,
                "{side}: no record -> floor"
            );
            assert_eq!((g.noise, g.zs_noise), (None, None), "{side}: no record");
            assert_eq!(g.shipped_seed, None, "{side}: legacy recipe");
            assert_eq!(g.clauses, vec![GateClause::EcePost], "{side}: clause set");
            let reported = rep["fine_tuned"]["ece_post"].as_f64().expect("ece_post");
            assert!(
                (g.recomputed.ece_post - reported).abs() <= policy.metric_recompute_abs,
                "{side}: recomputed ece_post {} vs reported {reported}",
                g.recomputed.ece_post
            );
            assert!(g.recomputed.ece_post > policy.max_ece, "{side}: ece_post");
            assert!(
                g.recomputed.margin >= policy.min_macro_f1_margin,
                "{side}: margin must pass"
            );
            assert_eq!(g.argmax_agree, g.n, "{side}: argmax n/n");
            assert!(g.rescore_max_abs <= policy.rescore_probs_abs, "{side}");
            assert!(g.zs_rescore_max_abs <= policy.rescore_probs_abs, "{side}");
        }
        (
            Expected::RescoreDriftFineTuned,
            VerifyError::RescoreDrift {
                which: ProbsWhich::FineTuned,
                max_abs,
                bound,
                ..
            },
        ) => {
            assert_eq!(e.exit_code(), 2, "{side}");
            assert_eq!(
                bound.to_bits(),
                policy.rescore_probs_abs.to_bits(),
                "{side}: no record -> the floor is the bound used"
            );
            assert!(*max_abs > *bound, "{side}: max_abs");
        }
        (want, other) => panic!(
            "{side}: expected {want:?}, got {} {other} — STOP RULE: an undocumented refusal",
            other.variant_name()
        ),
    }
    format!("REFUSED {} {e}", e.variant_name())
}

#[test]
fn demo_vectors_are_refused_fail_closed() {
    let armed = std::env::var(ENV_ARM).as_deref() == Ok("1");
    let Some(base) = std::env::var_os(ENV_MODEL)
        .filter(|_| armed)
        .map(PathBuf::from)
    else {
        println!("SKIP: fail-closed vectors not armed ({ENV_MODEL} + {ENV_ARM}=1)");
        return;
    };
    let data = std::env::var_os(ENV_DATA)
        .map_or_else(|| workspace_root().join(DEFAULT_DATA), PathBuf::from);
    assert!(
        data.join("eval.jsonl").is_file(),
        "armed, but the data dir {} is missing",
        data.display()
    );
    let policy = policy();
    let vectors = vectors();
    assert!(
        vectors.len() >= 2,
        "fewer than two fail-closed vectors parse"
    );
    let t0 = std::time::Instant::now();
    let mut refused = 0usize;
    for (rid, run) in &vectors {
        let short = &rid[..8];
        let want = expected(rid)
            .unwrap_or_else(|| panic!("vector {rid} has no documented refusal in this test"));
        assert!(
            run.join("recipe.json").is_file(),
            "armed, but the run dir {} is missing",
            run.display()
        );
        // (1) Identify the vector by its files.
        assert_eq!(
            artifact_sha256_hex(&std::fs::read(run.join("recipe.json")).expect("recipe.json")),
            *rid,
            "{short}: recipe_id"
        );
        let rep = report(run);
        assert_eq!(rep["pass"], serde_json::Value::Bool(false), "{short}: pass");
        // 1.4.0: no noise record (-> the floor) and no seed_selection (-> the legacy rule).
        assert!(
            rep.get("rescore_noise_sha256").is_none() && !run.join("rescore-noise.json").exists(),
            "{short}: a 1.x vector carries no rescore-noise record"
        );
        let recipe: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run.join("recipe.json")).expect("recipe.json"))
                .expect("parse recipe.json");
        assert!(
            recipe.get("seed_selection").is_none(),
            "{short}: a 1.x vector carries no seed_selection"
        );

        // (2) pack_for_serving into a scratch dir: the documented refusal, nothing written.
        let out_dir = tempfile::TempDir::new().expect("tempdir");
        let pack_err = pack_for_serving(run, &data, &base, &out_dir.path().join("v.apr"), &policy)
            .expect_err("a fail-closed vector must never pack");
        let line = check_refusal("pack", want, &pack_err, &rep, &policy);
        println!("VECTOR {short} pack: {line}");
        println!(
            "VECTOR {short} bound used: fine_tuned={} zero_shot={} (the A1 floor: no rescore-noise.json)",
            policy.rescore_probs_abs,
            match &pack_err {
                VerifyError::GateFailed(g) => g.zs_rescore_bound.to_string(),
                _ => "not reached".to_string(),
            }
        );
        let left: Vec<_> = std::fs::read_dir(out_dir.path())
            .expect("read scratch dir")
            .collect();
        assert!(left.is_empty(), "{short}: pack wrote {left:?}");

        // (3) The low-level bytes of the same run, in a TempDir, through verify on the file.
        let bytes = pack_run_dir(run, &data).expect("low-level pack");
        let scratch = tempfile::TempDir::new().expect("tempdir");
        let apr = scratch.path().join("vector.apr");
        std::fs::write(&apr, &bytes).expect("write scratch apr");
        let verify_err = verify_path(&apr, run, &data, &base, &policy)
            .expect_err("verify must refuse a fail-closed vector");
        let line = check_refusal("verify", want, &verify_err, &rep, &policy);
        println!("VECTOR {short} verify: {line}");
        match (&pack_err, &verify_err) {
            (VerifyError::GateFailed(p), VerifyError::GateFailed(v)) => {
                assert_eq!(
                    p.artifact_sha256, v.artifact_sha256,
                    "{short}: verify refused other bytes than pack"
                );
                assert_eq!(v.artifact_sha256, artifact_sha256_hex(&bytes), "{short}");
            }
            (
                VerifyError::RescoreDrift {
                    row: pr,
                    max_abs: pm,
                    ..
                },
                VerifyError::RescoreDrift {
                    row: vr,
                    max_abs: vm,
                    ..
                },
            ) => {
                assert_eq!(pr, vr, "{short}: first drifting row");
                assert_eq!(pm.to_bits(), vm.to_bits(), "{short}: max_abs");
            }
            (p, v) => panic!("{short}: pack {p:?} and verify {v:?} disagree"),
        }
        let scratch_path = scratch.path().to_path_buf();
        drop(scratch);
        assert!(!scratch_path.exists(), "{short}: scratch dir left behind");
        refused += 1;
    }
    println!(
        "FAIL-CLOSED VECTORS REFUSED {refused}/{} ({:.0} s, ARCH {})",
        vectors.len(),
        t0.elapsed().as_secs_f64(),
        std::env::consts::ARCH
    );
    assert_eq!(refused, vectors.len());
}
