//! `pack_laya` — the back-office pack / verify CLI for Laya decision models (plan 08-09).
//!
//! ```text
//! pack_laya pack         --run DIR --data DIR --base DIR --out FILE
//! pack_laya verify FILE  --run DIR --data DIR --base DIR
//! pack_laya inspect FILE
//! pack_laya pack-fixture --run DIR --data DIR --out FILE
//! ```
//!
//! `pack` verifies the run BEFORE anything is written (`aprender_decide::verify::pack_for_serving`):
//! production variant, the contract's recipe block, the contract's base (every declared identity
//! field, and the base dir's files against the contract's pins), input hashes, the split,
//! both probability files, a Rust re-score of every eval row from the packed bytes and from
//! the base, and the gate recomputed from those verified probabilities. Only an accepted run
//! is written, atomically; it prints
//! `PACKED <path> sha256=<H> rescore_max_abs=<x> zs_rescore_max_abs=<y> rescore_bound=<b>
//! zs_rescore_bound=<b> noise=<v|null> zs_noise=<v|null> argmax=<n>/<n>` — each re-score's bound
//! is laya-parity-v1 A1's `max(floor, k x noise)`, derived from `rescore-noise.json`, or the floor
//! (noise `null`) when the run carries no record — and `shipped_seed=<s>`, the median-ECE seed
//! verify re-derived from the per-seed files (laya-finetune-gate-v1 1.4.0). A run whose recipe
//! carries no `seed_selection` is refused `SeedPolicyMissing` after its re-scores and gate.
//! A refusal prints ONE `REFUSED <Variant> <detail> (nothing written)` line.
//!
//! `verify` is decide-apr-v1 `deploy_eligibility`, the ONLY eligibility check: it loads the
//! EXACT file through the full load ladder, binds its manifest to the run and data dirs, and
//! re-runs every `pack` check on those bytes (`verify::verify_path`). On accept it prints one
//! JSON line carrying `deploy_eligible`; on refusal the same `REFUSED ...` line as `pack`.
//!
//! `inspect` reads the file through the bounded rung-1 reader and runs load rungs 1-4 (bounded
//! read, header, manifest, structure — which binds every manifest leaf to its sha-bound source),
//! then prints identity only: it has no eligibility field and makes no eligibility claim.
//!
//! `pack-fixture` writes ONLY `synthetic-fixture` artifacts (`verify::pack_fixture`), which
//! every `verify` refuses; any other variant is refused with nothing written.
//!
//! Exit codes (the scripts/laya_train convention): 0 accepted, 3 the recomputed gate failed,
//! 2 every other refusal (including a usage error).
//!
//! THE POLICY IS NOT AN ARGUMENT. Thresholds, `ece_bins`, `gate_metric_recompute_abs`,
//! `calibration_slice_min_per_class`, the `base` block and the `seed_policy` (selection, seeds,
//! rank_scale, tie_break) are read from `contracts/laya-finetune-gate-v1.yaml`, and the
//! re-score floor `pack_rescore_probs_abs`, the
//! noise multiplier `pack_rescore_noise_k` and the ceiling `pack_rescore_bound_max_abs` from
//! `contracts/laya-parity-v1.yaml`, at run time, into the library's typed views and through
//! its one mapping `VerifyPolicy::from_contract_views` (the tests use the same). The CLI
//! accepts only the path arguments above and reads no environment variable: there is no way to
//! hand it another policy.

use aprender_decide::artifact;
use aprender_decide::verify::{
    self, GateContractView, ParityContractView, VerifyError, VerifyPolicy,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: pack_laya pack --run DIR --data DIR --base DIR --out FILE \
                     | verify FILE --run DIR --data DIR --base DIR | inspect FILE \
                     | pack-fixture --run DIR --data DIR --out FILE";

/// A contract file from the workspace root (resolved from this crate's manifest dir at
/// compile time — never from the environment), parsed into the library's typed view.
fn contract<T: serde::de::DeserializeOwned>(name: &str) -> Result<T, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .join(name);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The verify policy, read from the contracts (never literals) through the library's ONE
/// mapping, `VerifyPolicy::from_contract_views`.
fn policy() -> Result<VerifyPolicy, String> {
    let gate: GateContractView = contract("laya-finetune-gate-v1.yaml")?;
    let parity: ParityContractView = contract("laya-parity-v1.yaml")?;
    Ok(VerifyPolicy::from_contract_views(&gate, &parity))
}

/// The ONLY arguments any subcommand takes.
#[derive(Default)]
struct Args {
    operand: Option<PathBuf>,
    run: Option<PathBuf>,
    data: Option<PathBuf>,
    base: Option<PathBuf>,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(argv: &[String]) -> Result<Self, String> {
        let mut a = Self::default();
        let mut it = argv.iter();
        while let Some(arg) = it.next() {
            let slot = match arg.as_str() {
                "--run" => &mut a.run,
                "--data" => &mut a.data,
                "--base" => &mut a.base,
                "--out" => &mut a.out,
                s if s.starts_with('-') => return Err(format!("unknown argument {s}")),
                _ => &mut a.operand,
            };
            if slot.is_some() {
                return Err(format!("{arg} given twice"));
            }
            let value = if arg.starts_with("--") {
                it.next().ok_or_else(|| format!("{arg} needs a value"))?
            } else {
                arg
            };
            *slot = Some(PathBuf::from(value));
        }
        Ok(a)
    }

    fn need(v: Option<&PathBuf>, flag: &str) -> Result<PathBuf, String> {
        v.cloned().ok_or_else(|| format!("{flag} is required"))
    }
}

/// A JSON object from ordered pairs (serde_json's `json!` object form expands to `unwrap`,
/// which this workspace bans).
fn obj(pairs: Vec<(&str, serde_json::Value)>) -> serde_json::Value {
    serde_json::Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn refused(e: &VerifyError) -> ExitCode {
    println!("REFUSED {} {e} (nothing written)", e.variant_name());
    ExitCode::from(u8::try_from(e.exit_code()).unwrap_or(2))
}

fn cmd_pack(a: &Args) -> Result<ExitCode, String> {
    if a.operand.is_some() {
        return Err("pack takes no operand".into());
    }
    let run = Args::need(a.run.as_ref(), "--run")?;
    let data = Args::need(a.data.as_ref(), "--data")?;
    let base = Args::need(a.base.as_ref(), "--base")?;
    let out = Args::need(a.out.as_ref(), "--out")?;
    let policy = policy()?;
    Ok(
        match verify::pack_for_serving(&run, &data, &base, &out, &policy) {
            Ok(r) => {
                println!(
                    "PACKED {} sha256={} rescore_max_abs={} zs_rescore_max_abs={} rescore_bound={} \
                     zs_rescore_bound={} noise={} zs_noise={} shipped_seed={} argmax={}/{}",
                    out.display(),
                    r.artifact_sha256,
                    r.rescore_max_abs,
                    r.zs_rescore_max_abs,
                    r.rescore_bound,
                    r.zs_rescore_bound,
                    verify::opt_f64(r.noise),
                    verify::opt_f64(r.zs_noise),
                    verify::opt_i64(r.shipped_seed),
                    r.argmax_agree,
                    r.n
                );
                ExitCode::SUCCESS
            }
            Err(e) => refused(&e),
        },
    )
}

fn cmd_verify(a: &Args) -> Result<ExitCode, String> {
    if a.out.is_some() {
        return Err("verify takes no --out".into());
    }
    let apr = Args::need(a.operand.as_ref(), "the FILE operand")?;
    let run = Args::need(a.run.as_ref(), "--run")?;
    let data = Args::need(a.data.as_ref(), "--data")?;
    let base = Args::need(a.base.as_ref(), "--base")?;
    let policy = policy()?;
    Ok(
        match verify::verify_path(&apr, &run, &data, &base, &policy) {
            Ok(r) => {
                let line = obj(vec![
                    ("artifact_sha256", r.artifact_sha256.into()),
                    ("deploy_eligible", r.deploy_eligible.into()),
                    (
                        "recomputed",
                        obj(vec![
                            ("zs_macro_f1", r.recomputed.zs_macro_f1.into()),
                            ("ft_macro_f1", r.recomputed.ft_macro_f1.into()),
                            ("ece_post", r.recomputed.ece_post.into()),
                            ("margin", r.recomputed.margin.into()),
                        ]),
                    ),
                    ("rescore_max_abs", r.rescore_max_abs.into()),
                    ("zs_rescore_max_abs", r.zs_rescore_max_abs.into()),
                    ("rescore_bound", r.rescore_bound.into()),
                    ("zs_rescore_bound", r.zs_rescore_bound.into()),
                    ("noise", r.noise.into()),
                    ("zs_noise", r.zs_noise.into()),
                    ("shipped_seed", r.shipped_seed.into()),
                    ("argmax", format!("{}/{}", r.argmax_agree, r.n).into()),
                ]);
                println!("{line}");
                ExitCode::SUCCESS
            }
            Err(e) => refused(&e),
        },
    )
}

fn cmd_inspect(a: &Args) -> Result<ExitCode, String> {
    if a.run.is_some() || a.data.is_some() || a.base.is_some() || a.out.is_some() {
        return Err("inspect takes only the FILE operand".into());
    }
    let apr = Args::need(a.operand.as_ref(), "the FILE operand")?;
    // Rung 1 on the file itself: the declared length is checked before the file is read, and
    // the read is capped at the cap + 1 whatever the metadata says (no unbounded fs::read).
    let file = std::fs::File::open(&apr).map_err(|e| format!("{}: {e}", apr.display()))?;
    let declared = file
        .metadata()
        .map_err(|e| format!("{}: {e}", apr.display()))?
        .len();
    let bytes = match artifact::read_decide_apr_bytes_bounded(file, Some(declared)) {
        Ok(bytes) => bytes,
        Err(e) => return Ok(refused(&VerifyError::Artifact(e))),
    };
    Ok(match artifact::inspect_manifest(&bytes) {
        Ok(m) => {
            let line = obj(vec![
                (
                    "artifact_sha256",
                    artifact::artifact_sha256_hex(&bytes).into(),
                ),
                ("recipe_id", m.recipe_id.into()),
                ("method", m.method.into()),
                ("base", m.base.display().into()),
                ("variant", m.variant.into()),
                ("labels", m.labels.into()),
                (
                    "embedded_gate",
                    obj(vec![
                        ("pass", m.gate.pass.into()),
                        ("margin", m.gate.margin.into()),
                        ("ece_post", m.gate.ece_post.into()),
                    ]),
                ),
                ("schema_version", m.schema_version.into()),
            ]);
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => refused(&VerifyError::Artifact(e)),
    })
}

fn cmd_pack_fixture(a: &Args) -> Result<ExitCode, String> {
    if a.operand.is_some() || a.base.is_some() {
        return Err("pack-fixture takes only --run, --data and --out".into());
    }
    let run = Args::need(a.run.as_ref(), "--run")?;
    let data = Args::need(a.data.as_ref(), "--data")?;
    let out = Args::need(a.out.as_ref(), "--out")?;
    Ok(match verify::pack_fixture(&run, &data, &out) {
        Ok(sha) => {
            println!(
                "PACKED-FIXTURE {} sha256={sha} variant={} (not deployable)",
                out.display(),
                verify::SYNTHETIC_FIXTURE_VARIANT
            );
            ExitCode::SUCCESS
        }
        Err(e) => refused(&e),
    })
}

fn run(argv: &[String]) -> Result<ExitCode, String> {
    let (cmd, rest) = argv.split_first().ok_or(USAGE)?;
    let args = Args::parse(rest)?;
    match cmd.as_str() {
        "pack" => cmd_pack(&args),
        "verify" => cmd_verify(&args),
        "inspect" => cmd_inspect(&args),
        "pack-fixture" => cmd_pack_fixture(&args),
        other => Err(format!("unknown subcommand {other}; {USAGE}")),
    }
}

fn main() -> ExitCode {
    // `args_os`, not `args`: `std::env::args` panics on a non-UTF-8 argument. Every argument
    // here is a path the CLI prints back, so a non-UTF-8 one is a usage refusal (exit 2).
    let argv: Result<Vec<String>, _> = std::env::args_os()
        .skip(1)
        .map(std::ffi::OsString::into_string)
        .collect();
    let result = match argv {
        Ok(argv) => run(&argv),
        Err(_) => Err(format!("an argument is not valid UTF-8; {USAGE}")),
    };
    result.unwrap_or_else(|msg| {
        println!("REFUSED Usage {msg} (nothing written)");
        ExitCode::from(2)
    })
}
