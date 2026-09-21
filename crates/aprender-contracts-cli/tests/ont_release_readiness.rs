//! aprender#3715 + #3745 S2 (#3777) — `release-readiness-v1` on the CLI: the red-turning case table over DERIVED
//! cells.
//!
//! Every case copies `tests/fixtures/ont/release-green/` (a measured inventory, a GENERIC surface — `gen` generates,
//! `look` reads a model, `list` takes none; not apr verbs, so S3's hand-list guard has nothing to find) and the
//! real `contracts/release-readiness-v1.yaml` into a scratch repo. The green base then asks pv for the cells it
//! derives (`pv extract … --surface --cells-out`) and writes one passing row per cell, keyed by `cell_id` — so the
//! fixture cannot drift from the derivation — and each case changes exactly ONE thing.
//!
//! DISCRIMINATION: the green base must PASS with every derived cell carrying a row, so a build that rejects
//! everything fails this file; each single change must FAIL naming what it broke, so a build that accepts
//! everything fails it too. The ONT-001 rule: a shape that has never gone red is not armed.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Map, Value};

const MC: &str = "1111111111111111111111111111111111111111";
const PARENT: &str = "3333333333333333333333333333333333333333";
const BUMP: &str = "2222222222222222222222222222222222222222";
const GB: u64 = 1_000_000_000;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code,
        r.stdout.chars().take(6000).collect::<String>(),
        r.stderr
    )
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("mkdir");
    for e in std::fs::read_dir(from).expect("read fixture dir").flatten() {
        let p = e.path();
        let dst = to.join(e.file_name());
        if p.is_dir() {
            copy_dir(&p, &dst);
        } else {
            std::fs::copy(&p, &dst).expect("copy fixture file");
        }
    }
}

fn pv(cwd: &Path, args: &[&str]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn s(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_string()
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json")
}

fn edit(path: &Path, f: impl FnOnce(&mut Value)) {
    let mut v = read(path);
    f(&mut v);
    std::fs::write(path, serde_json::to_string_pretty(&v).expect("ser")).expect("write");
}

fn models(t: &Path, host: &str) -> PathBuf {
    t.join(format!("evidence/dogfood/models/0.69.1/{host}.json"))
}

fn kernels(t: &Path, host: &str) -> PathBuf {
    t.join(format!("evidence/dogfood/kernels/0.69.1/{host}.json"))
}

fn tokenizer(t: &Path) -> PathBuf {
    t.join("evidence/dogfood/tokenizer/0.69.1/parity.json")
}

fn json_of(r: &Run) -> Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

/// The cells pv derives for the scratch repo, written to `<t>/cells.json` and returned.
fn derive(t: &Path) -> Vec<Value> {
    let contracts = s(&t.join("contracts"));
    let surface = s(&t.join("surface.json"));
    let out = s(&t.join("release.nt"));
    let cells = s(&t.join("cells.json"));
    let r = pv(
        t,
        &[
            "extract",
            &contracts,
            "--release-version",
            "0.69.1",
            "--release-commit",
            MC,
            "--surface",
            &surface,
            "--out",
            &out,
            "--cells-out",
            &cells,
        ],
    );
    assert_eq!(r.code, 0, "{}", show(&r));
    read(&t.join("cells.json"))["cells"]
        .as_array()
        .expect("cells")
        .clone()
}

/// One PASSING row for a derived cell — whatever its class needs to pass, and nothing it does not.
fn pass_row(c: &Value) -> Value {
    let mut r = Map::new();
    r.insert("cell_id".into(), c["id"].clone());
    r.insert("verdict".into(), "pass".into());
    r.insert("answer_chars".into(), 42.into());
    r.insert("rc".into(), 0.into());
    r.insert("wall_ms".into(), 1500.into());
    let generating = c["generates"] == true && c["kind"] == "matrix";
    r.insert(
        "backend".into(),
        if generating { "cuda" } else { "cpu" }.into(),
    );
    r.insert("fallback".into(), false.into());
    if let Some(t) = c["rung_tokens"].as_u64() {
        let declared = c["rung"] == "declared";
        r.insert("max_tokens".into(), 256.into());
        let p = if declared {
            t.saturating_sub(256)
        } else {
            t + 1
        };
        r.insert("prompt_tokens".into(), p.into());
    }
    if c["thinking"] == "on" {
        r.insert("think_closed".into(), true.into());
    }
    if matches!(c["kind"].as_str(), Some("base" | "effect")) {
        let id = c["id"].as_str().unwrap_or_default();
        r.insert("output_sha256".into(), format!("out:{id}").into());
    }
    Value::Object(r)
}

/// The green base: the fixture, the REAL contract, and one passing row per derived cell.
fn green() -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    copy_dir(
        &repo_root().join("tests/fixtures/ont/release-green"),
        t.path(),
    );
    std::fs::copy(
        repo_root().join("contracts/release-readiness-v1.yaml"),
        t.path().join("contracts/release-readiness-v1.yaml"),
    )
    .expect("copy the real contract");
    synthesize(t.path());
    t
}

/// (Re)write every host's rows from the cells pv derives now.
fn synthesize(t: &Path) {
    let cells = derive(t);
    for h in ["lambda", "gx10"] {
        let rows: Vec<Value> = cells
            .iter()
            .filter(|c| c["host"] == h)
            .map(pass_row)
            .collect();
        edit(&models(t, h), |v| v["cells"] = Value::Array(rows));
    }
}

/// The id of the first derived cell `pred` accepts.
fn pick(t: &Path, pred: impl Fn(&Value) -> bool) -> String {
    read(&t.join("cells.json"))["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .find(|c| pred(c))
        .and_then(|c| c["id"].as_str())
        .expect("a derived cell matches")
        .to_string()
}

/// The IRI tail a finding names for a cell id: each segment percent-encoded as the graph does.
fn named(id: &str) -> String {
    let enc = |seg: &str| {
        seg.bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b':' => {
                    (b as char).to_string()
                }
                _ => format!("%{b:02X}"),
            })
            .collect::<String>()
    };
    let segs: Vec<String> = id.split('/').map(enc).collect();
    format!("release-cell/0.69.1/{}", segs.join("/"))
}

/// Edit the one row keyed to `id` on `host`.
fn edit_row(t: &Path, host: &str, id: &str, f: impl FnOnce(&mut Value)) {
    edit(&models(t, host), |v| {
        let row = v["cells"]
            .as_array_mut()
            .expect("cells")
            .iter_mut()
            .find(|r| r["cell_id"] == id)
            .expect("the row exists");
        f(row);
    });
}

fn drop_row(t: &Path, host: &str, id: &str) {
    edit(&models(t, host), |v| {
        v["cells"]
            .as_array_mut()
            .expect("cells")
            .retain(|r| r["cell_id"] != id);
    });
}

/// The release gate as T-1 calls it, plus `extra` flags.
fn gate(t: &Path, extra: &[&str]) -> Run {
    let contracts = s(&t.join("contracts"));
    let dogfood = s(&t.join("dogfood-receipt.json"));
    let surface = s(&t.join("surface.json"));
    let mut args = vec![
        "lint",
        &contracts,
        "--gate",
        "shapes",
        "--shape",
        "release-readiness-v1",
        "--release-version",
        "0.69.1",
        "--release-commit",
        MC,
        "--dogfood-receipt",
        &dogfood,
        "--surface",
        &surface,
    ];
    args.extend_from_slice(extra);
    pv(t, &args)
}

/// RED, and the findings name `needle`.
fn assert_red_naming(r: &Run, needle: &str) -> Value {
    assert_eq!(r.code, 1, "{}", show(r));
    let v = json_of(r);
    assert_eq!(v["verdict"], "Fail", "{}", show(r));
    let msgs: Vec<&str> = v["findings"]
        .as_array()
        .expect("findings")
        .iter()
        .filter_map(|f| f["message"].as_str())
        .collect();
    assert!(
        msgs.iter().any(|m| m.contains(needle)),
        "no finding names {needle}:\n{}",
        msgs.iter().take(40).copied().collect::<Vec<_>>().join("\n")
    );
    v
}

fn gen_cell(
    host: &'static str,
    thinking: &'static str,
    rung: &'static str,
) -> impl Fn(&Value) -> bool {
    move |c| {
        c["command"] == "gen"
            && c["kind"] == "matrix"
            && c["host"] == host
            && c["thinking"] == thinking
            && c["rung"] == rung
    }
}

// ── the green base ───────────────────────────────────────────────────────────────────────────────────────

#[test]
fn the_green_release_passes_with_every_derived_cell_carrying_a_row() {
    let t = green();
    let r = gate(t.path(), &[]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass");
    assert_eq!(
        v["pc_shape"], "fired",
        "the plant fired from the named family"
    );
    assert_eq!(v["pc_extract"]["release-evidence"], "fired");
    assert_eq!(v["pc_extract"]["cli-surface"], "fired");
    let rel = &v["release"];
    let derived = read(&t.path().join("cells.json"))["cells"]
        .as_array()
        .map(Vec::len);
    assert_eq!(
        rel["cells"].as_u64().map(|n| n as usize),
        derived,
        "N/N: every derived cell is a node"
    );
    assert_eq!(rel["cells_with_row"], rel["cells"]);
    assert_eq!(rel["orphan_rows"], 0);
    assert_eq!(
        rel["probe_cells"], 2,
        "`list` takes no model: one probe per host"
    );
    assert!(
        rel["mode_effects"].as_u64() >= Some(1),
        "`gen --chat` is judged by the flag-effect oracle"
    );
    assert_eq!(rel["surface"]["generating_commands"], 1);
    assert_eq!(rel["kernel_cells"], 4);
    assert_eq!(rel["tokenizer_cells"], 1);
    assert_eq!(
        v["armed_shapes"].as_array().map(Vec::len),
        Some(13),
        "--shape arms the whole family"
    );
    let lambda = &rel["projection"]["lambda"];
    assert_eq!(
        lambda["unmeasured_cells"], 0,
        "every class has a measured median"
    );
    assert!(lambda["projected_secs"].as_u64() > Some(0), "{lambda}");
}

// ── #3715's cases, on derived cells ──────────────────────────────────────────────────────────────────────

#[test]
fn a_missing_cell_is_red_naming_it() {
    let t = green();
    let id = pick(t.path(), gen_cell("lambda", "on", "consumer-max"));
    drop_row(t.path(), "lambda", &id);
    assert_red_naming(&gate(t.path(), &[]), &named(&id));
}

#[test]
fn a_skipped_a_fallback_or_a_silent_fallback_cell_is_red_naming_it() {
    for change in ["skip", "fallback", "silent"] {
        let t = green();
        let id = pick(t.path(), gen_cell("gx10", "off", "4k"));
        edit_row(t.path(), "gx10", &id, |r| match change {
            "skip" => r["verdict"] = "skip".into(),
            "fallback" => r["fallback"] = true.into(),
            _ => {
                r.as_object_mut().expect("row").remove("fallback");
            }
        });
        assert_red_naming(&gate(t.path(), &[]), &named(&id));
    }
}

#[test]
fn a_receipt_of_another_commit_or_version_is_stale_and_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        v["apr_sha"] = PARENT.into()
    });
    let v = assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/lambda");
    assert!(
        !v["findings"]
            .to_string()
            .contains("release-host/0.69.1/gx10"),
        "gx10 is untouched"
    );
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        v["version"] = "0.69.0".into()
    });
    assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/gx10");
}

#[test]
fn a_think_on_row_with_an_unclosed_block_and_an_empty_answer_is_red() {
    let t = green();
    let id = pick(t.path(), gen_cell("lambda", "on", "4k"));
    edit_row(t.path(), "lambda", &id, |r| {
        r["think_closed"] = false.into();
        r["answer_chars"] = 0.into();
    });
    let v = assert_red_naming(&gate(t.path(), &[]), &named(&id));
    let msgs = v["findings"].to_string();
    assert!(
        msgs.contains("thinkOk") && msgs.contains("answered"),
        "{msgs}"
    );
}

#[test]
fn a_declared_length_cell_missing_or_an_unmeasured_length_is_red() {
    let t = green();
    let id = pick(t.path(), gen_cell("lambda", "off", "declared"));
    drop_row(t.path(), "lambda", &id);
    assert_red_naming(&gate(t.path(), &[]), &named(&id));
    let t = green();
    for h in ["lambda", "gx10"] {
        edit(&models(t.path(), h), |v| {
            v["inventory"][0]
                .as_object_mut()
                .expect("item")
                .remove("context_length");
        });
    }
    assert_red_naming(&gate(t.path(), &[]), "contextLength");
}

#[test]
fn a_prompt_short_of_its_rung_or_a_duplicate_row_is_red() {
    let t = green();
    let id = pick(t.path(), gen_cell("lambda", "on", "consumer-max"));
    edit_row(t.path(), "lambda", &id, |r| {
        r["prompt_tokens"] = 1000.into()
    });
    let v = assert_red_naming(&gate(t.path(), &[]), &named(&id));
    assert!(v["findings"].to_string().contains("contextMet"));
    let t = green();
    let id = pick(t.path(), gen_cell("lambda", "off", "4k"));
    edit(&models(t.path(), "lambda"), |v| {
        let cells = v["cells"].as_array_mut().expect("cells");
        let dup = cells
            .iter()
            .find(|r| r["cell_id"] == id.as_str())
            .cloned()
            .expect("row");
        cells.push(dup);
    });
    let v = assert_red_naming(&gate(t.path(), &[]), &named(&id));
    assert!(v["findings"].to_string().contains("maxCount"));
}

#[test]
fn a_kernel_green_on_sm_121_and_red_on_sm_89_is_named_before_any_cell() {
    let t = green();
    edit(&kernels(t.path(), "lambda"), |v| {
        v["rows"][0]["verdict"] = "fail".into();
        v["rows"][0]["max_err"] = 0.5.into();
    });
    let id = pick(t.path(), gen_cell("lambda", "off", "4k"));
    edit_row(t.path(), "lambda", &id, |r| r["verdict"] = "fail".into());
    let v = assert_red_naming(
        &gate(t.path(), &[]),
        "release-kernel/0.69.1/lambda/q4k_gemv/q4_k",
    );
    let first = v["findings"][0]["message"].as_str().unwrap_or_default();
    assert!(
        first.contains("release-kernel/0.69.1/lambda"),
        "kernel first: {first}"
    );
    let t = green();
    edit(&kernels(t.path(), "gx10"), |v| {
        v["rows"].as_array_mut().expect("rows").remove(1);
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        "release-kernel/0.69.1/gx10/rmsnorm/f32",
    );
}

#[test]
fn a_required_host_with_no_receipt_or_a_v1_receipt_is_red_naming_the_host() {
    let t = green();
    std::fs::remove_file(models(t.path(), "gx10")).expect("rm");
    assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/gx10");
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        v["schema"] = "apr-model-ladder-receipt/v1".into()
    });
    assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/gx10");
}

#[test]
fn an_inherited_dogfood_or_none_at_all_is_red() {
    let t = green();
    edit(&t.path().join("dogfood-receipt.json"), |v| {
        v["commit"] = PARENT.into()
    });
    assert_red_naming(&gate(t.path(), &[]), "release-dogfood/0.69.1");
    let t = green();
    let contracts = s(&t.path().join("contracts"));
    let surface = s(&t.path().join("surface.json"));
    let r = pv(
        t.path(),
        &[
            "lint",
            &contracts,
            "--gate",
            "shapes",
            "--shape",
            "release-readiness-v1",
            "--release-version",
            "0.69.1",
            "--release-commit",
            MC,
            "--surface",
            &surface,
        ],
    );
    assert_red_naming(&r, "dogfoodReceipt");
}

#[test]
fn a_consumer_with_no_number_or_a_thinking_contradiction_is_red() {
    let t = green();
    edit(&t.path().join("evidence/release/context-rungs.json"), |v| {
        let rec = serde_json::from_str::<Value>(
            r#"{"consumer":"pending","basis":"","source":"fixture://p"}"#,
        )
        .expect("record");
        v["consumers"].as_array_mut().expect("consumers").push(rec);
    });
    assert_red_naming(&gate(t.path(), &[]), "release-context/consumer-max");
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        v["inventory"][0]["thinking_modes"] =
            serde_json::from_str::<Value>(r#"["off"]"#).expect("modes");
    });
    assert_red_naming(&gate(t.path(), &[]), "thinkingContradiction");
}

#[test]
fn t4_receipts_measured_at_the_bump_head_pass_only_with_receipts_commit() {
    let t = green();
    let other = t.path().join("t4-receipts");
    std::fs::create_dir_all(&other).expect("mkdir");
    for h in ["lambda", "gx10"] {
        let dst = other.join(format!("{h}.json"));
        std::fs::rename(models(t.path(), h), &dst).expect("mv");
        edit(&dst, |v| v["apr_sha"] = BUMP.into());
        edit(&kernels(t.path(), h), |v| v["apr_sha"] = BUMP.into());
    }
    edit(&tokenizer(t.path()), |v| v["apr_sha"] = BUMP.into());
    let dir = s(&other);
    let r = gate(t.path(), &["--receipts", &dir, "--receipts-commit", BUMP]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert_red_naming(
        &gate(t.path(), &["--receipts", &dir]),
        "release-host/0.69.1/lambda",
    );
}

#[test]
fn one_token_id_off_llama_cpp_or_no_comparison_is_red_naming_the_model() {
    for change in ["mismatch", "empty", "roundtrip", "absent"] {
        let t = green();
        match change {
            "mismatch" => edit(&tokenizer(t.path()), |v| {
                v["rows"][0]["mismatches"] = 1.into()
            }),
            "empty" => edit(&tokenizer(t.path()), |v| {
                v["rows"][0]["tokens_compared"] = 0.into()
            }),
            "roundtrip" => edit(&tokenizer(t.path()), |v| {
                v["rows"][0]["roundtrip_ok"] = false.into()
            }),
            _ => std::fs::remove_file(tokenizer(t.path())).expect("rm"),
        }
        assert_red_naming(
            &gate(t.path(), &[]),
            "release-tokenizer/0.69.1/tiny-qwen35.gguf",
        );
    }
}

// ── the memory rule (#3710, operator decision "a") ───────────────────────────────────────────────────────
// 16 GB weights + 1 MB/token KV + 1 GB workspace: `4k` (21.1 GB) fits the 24 GB card; `consumer-max` (25 GB)
// and `declared` (27 GB) do not, and are owed on lambda as honest pre-load refusals — and on gx10 as passes.

fn memory_split(t: &Path) {
    for (h, gpu) in [("lambda", 24 * GB), ("gx10", 119 * GB)] {
        edit(&models(t, h), |v| {
            v["gpu_mem_total_bytes"] = gpu.into();
            let item = &mut v["inventory"][0];
            item["weights_bytes"] = (16 * GB).into();
            item["kv_bytes_per_token"] = 1_000_000u64.into();
            item["workspace_bytes"] = GB.into();
        });
    }
    let cells = read(&t.join("cells.json"))["cells"]
        .as_array()
        .expect("cells")
        .clone();
    edit(&models(t, "lambda"), |v| {
        for r in v["cells"].as_array_mut().expect("cells") {
            let spec = cells
                .iter()
                .find(|c| c["id"] == r["cell_id"])
                .expect("spec");
            let need = match (spec["kind"].as_str(), spec["rung"].as_str()) {
                (Some("matrix"), Some("consumer-max")) => 25 * GB,
                (Some("matrix"), Some("declared")) => 27 * GB,
                _ => continue,
            };
            r["verdict"] = "refused".into();
            r["required_bytes"] = need.into();
            r["available_bytes"] = (24 * GB).into();
        }
    });
}

#[test]
fn an_honest_refusal_on_lambda_and_a_pass_on_gx10_is_green() {
    let t = green();
    memory_split(t.path());
    let r = gate(t.path(), &[]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert!(json_of(&r)["release"]["refusal_cells"].as_u64() > Some(0));
}

#[test]
fn an_oom_a_co_tenant_refusal_or_a_rung_that_fits_nowhere_is_red() {
    let t = green();
    memory_split(t.path());
    let id = pick(t.path(), gen_cell("lambda", "on", "declared"));
    edit_row(t.path(), "lambda", &id, |r| {
        r["verdict"] = "error".into();
        r["reason"] = "CUDA_ERROR_OUT_OF_MEMORY after 41 s of prefill".into();
    });
    assert_red_naming(&gate(t.path(), &[]), &named(&id));
    let t = green();
    memory_split(t.path());
    let id = pick(t.path(), gen_cell("lambda", "off", "consumer-max"));
    edit_row(t.path(), "lambda", &id, |r| {
        r["required_bytes"] = (23 * GB).into()
    });
    let v = assert_red_naming(&gate(t.path(), &[]), &named(&id));
    assert!(v["findings"].to_string().contains("arithmeticNamed"));
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "gx10"), |v| {
        v["gpu_mem_total_bytes"] = (24 * GB).into()
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        "release-coverage/0.69.1/tiny-qwen35.gguf/declared",
    );
}

// ── #3745 S2: the cells are DERIVED ───────────────────────────────────────────────────────────────────────

#[test]
fn issue_mutant_1_a_new_flag_on_a_generating_command_owes_cells_nobody_has_run() {
    let t = green();
    edit(&t.path().join("surface.json"), |v| {
        let flag = serde_json::from_str::<Value>(
            r#"{"id":"extra","long":"extra","value_type":"flag","values":["true","false"],"role":"mode"}"#,
        )
        .expect("flag");
        v["commands"][0]["args"]
            .as_array_mut()
            .expect("args")
            .push(flag);
    });
    let v = assert_red_naming(&gate(t.path(), &[]), "--extra");
    assert!(
        v["findings"]
            .to_string()
            .contains("release-cell/0.69.1/gen/"),
        "the owed cells are gen's"
    );
}

#[test]
fn issue_mutant_2_the_prompt_with_chat_cell_missing_is_red_naming_it() {
    // #3743's shape: `-i file --chat` was covered, the `--prompt` shape was not
    let t = green();
    let id = pick(t.path(), |c| {
        c["command"] == "gen"
            && c["kind"] == "matrix"
            && c["host"] == "lambda"
            && c["shape"] == "--prompt"
            && c["args"].to_string().contains("--chat")
    });
    drop_row(t.path(), "lambda", &id);
    let v = assert_red_naming(&gate(t.path(), &[]), &named(&id));
    assert!(
        id.contains("--prompt") && id.contains("--chat"),
        "the cell names both: {id}"
    );
    let _ = v;
}

#[test]
fn no_surface_or_a_surface_that_does_not_say_what_generates_is_red() {
    let t = green();
    let contracts = s(&t.path().join("contracts"));
    let dogfood = s(&t.path().join("dogfood-receipt.json"));
    let r = pv(
        t.path(),
        &[
            "lint",
            &contracts,
            "--gate",
            "shapes",
            "--shape",
            "release-readiness-v1",
            "--release-version",
            "0.69.1",
            "--release-commit",
            MC,
            "--dogfood-receipt",
            &dogfood,
        ],
    );
    assert_red_naming(&r, "release/surface");
    let t = green();
    edit(&t.path().join("surface.json"), |v| {
        v["commands"][0]
            .as_object_mut()
            .expect("cmd")
            .remove("generates");
    });
    assert_red_naming(&gate(t.path(), &[]), "generatesUndeclared");
}

#[test]
fn an_unknown_role_arg_beyond_its_ceiling_grows_the_ratchet_and_is_red() {
    let t = green();
    edit(&t.path().join("surface.json"), |v| {
        let arg = serde_json::from_str::<Value>(
            r#"{"id":"mystery","long":"mystery","value_type":"text","role":"unknown"}"#,
        )
        .expect("arg");
        v["commands"][1]["args"]
            .as_array_mut()
            .expect("args")
            .push(arg);
    });
    assert_red_naming(&gate(t.path(), &[]), "ratchetGrew");
}

#[test]
fn a_flag_whose_output_never_changes_is_red_naming_the_setting() {
    // S2.5: the effect cell prints exactly what its base printed
    let t = green();
    let cells = read(&t.path().join("cells.json"))["cells"]
        .as_array()
        .expect("cells")
        .clone();
    for h in ["lambda", "gx10"] {
        edit(&models(t.path(), h), |v| {
            for r in v["cells"].as_array_mut().expect("cells") {
                let spec = cells
                    .iter()
                    .find(|c| c["id"] == r["cell_id"])
                    .expect("spec");
                if spec["kind"] == "effect" {
                    r["output_sha256"] =
                        format!("out:{}", spec["base"].as_str().unwrap_or_default()).into();
                }
            }
        });
    }
    assert_red_naming(
        &gate(t.path(), &[]),
        "release-effect/0.69.1/gen/--chat%3Dtrue",
    );
}

#[test]
fn a_modelless_command_must_run_or_refuse_by_name_and_a_model_command_must_pass_or_refuse() {
    let t = green();
    let id = pick(t.path(), |c| {
        c["command"] == "list" && c["host"] == "lambda"
    });
    edit_row(t.path(), "lambda", &id, |r| r["verdict"] = "refused".into());
    assert_eq!(
        gate(t.path(), &[]).code,
        0,
        "a typed refusal by name is an answer"
    );
    edit_row(t.path(), "lambda", &id, |r| r["verdict"] = "fail".into());
    assert_red_naming(&gate(t.path(), &[]), &named(&id));
    let t = green();
    let id = pick(t.path(), |c| {
        c["command"] == "look" && c["kind"] == "matrix" && c["host"] == "gx10"
    });
    drop_row(t.path(), "gx10", &id);
    assert_red_naming(&gate(t.path(), &[]), &named(&id));
}

// ── the CLI's own answers ────────────────────────────────────────────────────────────────────────────────

#[test]
fn no_release_subject_declines_and_a_malformed_one_is_the_callers_error() {
    let t = green();
    let contracts = s(&t.path().join("contracts"));
    let r = pv(
        t.path(),
        &[
            "lint",
            &contracts,
            "--gate",
            "shapes",
            "--shape",
            "release-readiness-v1",
        ],
    );
    assert_eq!(
        r.code,
        2,
        "zero cells is a decline, not a pass: {}",
        show(&r)
    );
    for bad in [
        vec!["--gate", "shapes", "--release-version", "0.69.1"],
        vec![
            "--gate",
            "shapes",
            "--release-version",
            "0.69.1",
            "--release-commit",
            "225b2a9ab",
        ],
        vec!["--gate", "sigma", "--shape", "x"],
        vec!["--gate", "shapes", "--shape", "no-such-shape"],
    ] {
        let mut args = vec!["lint", contracts.as_str()];
        args.extend(bad.iter().copied());
        let r = pv(t.path(), &args);
        assert_eq!(r.code, 3, "{bad:?}: {}", show(&r));
    }
}

#[test]
fn pv_extract_writes_the_release_graph_and_the_cell_list_only_where_told() {
    let t = green();
    let contracts = s(&t.path().join("contracts"));
    let refused = pv(
        t.path(),
        &[
            "extract",
            &contracts,
            "--release-version",
            "0.69.1",
            "--release-commit",
            MC,
        ],
    );
    assert_eq!(refused.code, 3, "no --out: {}", show(&refused));
    let cells = read(&t.path().join("cells.json"));
    assert_eq!(cells["schema"], "apr-release-cells/v1");
    assert_eq!(cells["release_commit"], MC);
    let nt = std::fs::read_to_string(t.path().join("release.nt")).expect("written");
    assert!(
        nt.contains("release/Cell") && nt.contains("cli/Command"),
        "cells and the surface are in the graph"
    );
    assert!(!nt.contains("_:"), "no blank nodes (R-15)");
    assert!(
        !t.path().join("contracts/contracts.nt").exists(),
        "the tracked file is untouched"
    );
}
