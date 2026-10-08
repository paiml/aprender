//! FALSIFY-LAYA-GATE-014: pack-for-serving and `pack_laya verify` decide EXACTLY the recorded
//! outcome of the one declared s64 run (laya-finetune-gate-v1 `demo_s64`), on its exact bytes.
//!
//! ARMED only by BOTH `LAYA_MODEL_DIR` (the declared base snapshot) and `LAYA_DEMO_RUN=1`;
//! otherwise it prints `SKIP: demo run not armed (LAYA_MODEL_DIR + LAYA_DEMO_RUN=1)` and passes
//! (never `#[ignore]`). The run dir is gitignored local evidence, so CI compiles this only.
//!
//! ```text
//! LAYA_DEMO_RUN=1 \
//! LAYA_MODEL_DIR=~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4e… \
//!   cargo test -p aprender-decide --release --test demo_run -- --nocapture
//! ```
//!
//! Everything is read from the contract's `demo_s64` block: `run_dir`, `data_dir`, `outcome`
//! (one of `outcome_values`) and `outcome_record`, which plan 08-16 writes ONCE after the run:
//! - `pending` / `aborted`: prints a SKIP line naming the outcome and passes (nothing to decide);
//! - `gate_pass`: `pack_for_serving` into a TempDir, then `verify_path` on THAT file; both
//!   accept, `deploy_eligible`, one artifact sha256 on both sides (and the file's), the shipped
//!   seed equal to `outcome_record.shipped_seed`, and both re-score bounds printed;
//! - `gate_fail`: both refuse `GateFailed` naming exactly `outcome_record.failed_clauses`;
//! - `pack_refused`: both refuse with the variant `outcome_record.refusal`, identically;
//! - any value outside `outcome_values` fails.
//!
//! It writes only inside its TempDirs.

mod common;

use aprender_decide::artifact::artifact_sha256_hex;
use aprender_decide::pack::pack_run_dir;
use aprender_decide::verify::{
    opt_f64, opt_i64, pack_for_serving, verify_path, VerifyError, VerifyPolicy,
};
use std::path::{Path, PathBuf};

const ENV_MODEL: &str = "LAYA_MODEL_DIR";
const ENV_ARM: &str = "LAYA_DEMO_RUN";

/// `demo_s64` of laya-finetune-gate-v1.
struct Declared {
    outcome: String,
    values: Vec<String>,
    record: serde_yaml::Value,
    run_dir: PathBuf,
    data_dir: PathBuf,
}

fn declared() -> Declared {
    let gate = common::contract("laya-finetune-gate-v1.yaml");
    let d = &gate["demo_s64"];
    let path = |k: &str| {
        common::workspace_root().join(
            d[k].as_str()
                .unwrap_or_else(|| panic!("demo_s64.{k}"))
                .trim_end_matches('/'),
        )
    };
    Declared {
        outcome: common::str_at(d, &["outcome"]),
        values: d["outcome_values"]
            .as_sequence()
            .expect("demo_s64.outcome_values")
            .iter()
            .map(|v| v.as_str().expect("an outcome value").to_string())
            .collect(),
        record: d["outcome_record"].clone(),
        run_dir: path("run_dir"),
        data_dir: path("data_dir"),
    }
}

/// Pack for serving into a scratch dir (nothing may be written on refusal), and verify the
/// exact bytes of the same run through `verify_path` on a file. Returns both results and the
/// sha256 of the file verify read.
fn decide_both(
    d: &Declared,
    base: &Path,
    policy: &VerifyPolicy,
) -> (
    Result<aprender_decide::verify::VerifyReport, VerifyError>,
    Result<aprender_decide::verify::VerifyReport, VerifyError>,
    String,
) {
    let out = tempfile::TempDir::new().expect("tempdir");
    let apr = out.path().join("laya-stance-64.apr");
    let pack = pack_for_serving(&d.run_dir, &d.data_dir, base, &apr, policy);
    let file = if pack.is_ok() {
        apr
    } else {
        let left: Vec<_> = std::fs::read_dir(out.path())
            .expect("read scratch dir")
            .collect();
        assert!(left.is_empty(), "a refused pack wrote {left:?}");
        // The low-level bytes of the same run, so verify decides on a real file too.
        let bytes = pack_run_dir(&d.run_dir, &d.data_dir).expect("low-level pack");
        let f = out.path().join("low-level.apr");
        std::fs::write(&f, &bytes).expect("write scratch apr");
        f
    };
    let sha = artifact_sha256_hex(&std::fs::read(&file).expect("read the verified file"));
    let verify = verify_path(&file, &d.run_dir, &d.data_dir, base, policy);
    (pack, verify, sha)
}

#[test]
fn declared_demo_run_matches_its_outcome_of_record() {
    let armed = std::env::var(ENV_ARM).as_deref() == Ok("1");
    let Some(base) = std::env::var_os(ENV_MODEL)
        .filter(|_| armed)
        .map(PathBuf::from)
    else {
        println!("SKIP: demo run not armed ({ENV_MODEL} + {ENV_ARM}=1)");
        return;
    };
    let d = declared();
    assert!(
        d.values.contains(&d.outcome),
        "demo_s64.outcome {:?} is not one of demo_s64.outcome_values {:?}",
        d.outcome,
        d.values
    );
    match d.outcome.as_str() {
        "pending" => {
            println!("SKIP: demo_s64.outcome is pending (plan 08-16 records it)");
            return;
        }
        "aborted" => {
            println!("SKIP: demo_s64.outcome is aborted (no complete run dir)");
            return;
        }
        _ => {}
    }
    assert!(
        d.run_dir.join("gate-report.json").is_file(),
        "armed, but the declared run dir {} is missing",
        d.run_dir.display()
    );
    let policy = common::policy();
    let t0 = std::time::Instant::now();
    let (pack, verify, sha) = decide_both(&d, &base, &policy);
    match d.outcome.as_str() {
        "gate_pass" => {
            let p = pack.unwrap_or_else(|e| {
                panic!(
                    "outcome gate_pass, but pack REFUSED {} {e}",
                    e.variant_name()
                )
            });
            let v = verify.unwrap_or_else(|e| {
                panic!(
                    "outcome gate_pass, but verify REFUSED {} {e}",
                    e.variant_name()
                )
            });
            assert!(p.deploy_eligible && v.deploy_eligible);
            assert_eq!(p.artifact_sha256, v.artifact_sha256, "pack vs verify bytes");
            assert_eq!(v.artifact_sha256, sha, "verify read another file");
            let want = d.record["shipped_seed"]
                .as_i64()
                .expect("demo_s64.outcome_record.shipped_seed");
            assert_eq!(v.shipped_seed, Some(want), "shipped seed vs the record");
            assert_eq!(p.shipped_seed, v.shipped_seed);
            println!(
                "DEMO gate_pass sha256={} shipped_seed={} rescore_max_abs={} rescore_bound={} \
                 zs_rescore_max_abs={} zs_rescore_bound={} noise={} zs_noise={} ece_post={} margin={}",
                v.artifact_sha256,
                opt_i64(v.shipped_seed),
                v.rescore_max_abs,
                v.rescore_bound,
                v.zs_rescore_max_abs,
                v.zs_rescore_bound,
                opt_f64(v.noise),
                opt_f64(v.zs_noise),
                v.recomputed.ece_post,
                v.recomputed.margin
            );
        }
        "gate_fail" => {
            let want: Vec<String> = d.record["failed_clauses"]
                .as_sequence()
                .expect("demo_s64.outcome_record.failed_clauses")
                .iter()
                .map(|c| c.as_str().expect("a clause name").to_string())
                .collect();
            for (side, r) in [("pack", &pack), ("verify", &verify)] {
                match r {
                    Err(VerifyError::GateFailed(g)) => {
                        let got: Vec<String> = g.clauses.iter().map(ToString::to_string).collect();
                        assert_eq!(got, want, "{side}: failed clauses vs the record");
                        println!("DEMO gate_fail {side}: REFUSED GateFailed {g}");
                    }
                    other => panic!("outcome gate_fail, but {side} decided {other:?}"),
                }
            }
            assert_eq!(pack, verify, "pack and verify refused differently");
        }
        "pack_refused" => {
            let want = d.record["refusal"]
                .as_str()
                .expect("demo_s64.outcome_record.refusal");
            for (side, r) in [("pack", &pack), ("verify", &verify)] {
                match r {
                    Err(e) => {
                        assert_eq!(e.variant_name(), want, "{side}: refusal vs the record");
                        println!("DEMO pack_refused {side}: REFUSED {} {e}", e.variant_name());
                    }
                    Ok(rep) => panic!("outcome pack_refused, but {side} accepted {rep:?}"),
                }
            }
            assert_eq!(
                pack, verify,
                "pack and verify refused differently (row / value)"
            );
        }
        other => panic!("demo_s64.outcome {other:?} has no decision rule in this test"),
    }
    println!(
        "DEMO OUTCOME {} decided on the exact bytes ({:.0} s, ARCH {})",
        d.outcome,
        t0.elapsed().as_secs_f64(),
        std::env::consts::ARCH
    );
}
