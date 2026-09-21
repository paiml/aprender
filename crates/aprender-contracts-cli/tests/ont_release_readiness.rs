//! aprender#3715 — `release-readiness-v1` on the CLI: the red-turning case table.
//!
//! Every case copies `tests/fixtures/ont/release-green/` AND the real `contracts/release-readiness-v1.yaml` (so the
//! fixture cannot drift from the shapes it proves) into a scratch repo, changes exactly ONE thing, and runs
//! `pv lint contracts --gate shapes --shape release-readiness-v1 --release-version 0.69.1 --release-commit MC …`.
//!
//! DISCRIMINATION: the green base must PASS with every cell named (48/48, 4 kernel cells) — so a build that
//! rejects everything fails this file — and each single-cell change must FAIL naming that cell — so a build that
//! accepts everything fails it too. The ONT-001 rule: a shape that has never gone red is not armed.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const MC: &str = "1111111111111111111111111111111111111111";
const PARENT: &str = "3333333333333333333333333333333333333333";
const BUMP: &str = "2222222222222222222222222222222222222222";
const MODEL: &str = "tiny-qwen35.gguf";

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code, r.stdout, r.stderr
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

/// The green base: the fixture plus the REAL shape contract.
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
    t
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

/// The release gate as T-1 calls it, plus `extra` flags.
fn gate(t: &Path, extra: &[&str]) -> Run {
    let contracts = s(&t.join("contracts"));
    let dogfood = s(&t.join("dogfood-receipt.json"));
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
    ];
    args.extend_from_slice(extra);
    pv(t, &args)
}

fn json_of(r: &Run) -> Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

fn edit(path: &Path, f: impl FnOnce(&mut Value)) {
    let mut v: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");
    f(&mut v);
    std::fs::write(path, serde_json::to_string_pretty(&v).expect("ser")).expect("write");
}

fn models(t: &Path, host: &str) -> PathBuf {
    t.join(format!("evidence/dogfood/models/0.69.1/{host}.json"))
}

fn kernels(t: &Path, host: &str) -> PathBuf {
    t.join(format!("evidence/dogfood/kernels/0.69.1/{host}.json"))
}

/// Index of the one row for (verb, thinking, rung) in a model receipt.
fn row_index(v: &Value, verb: &str, thinking: &str, rung: &str) -> usize {
    v["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .position(|c| c["verb"] == verb && c["thinking"] == thinking && c["context"] == rung)
        .unwrap_or_else(|| panic!("no row {verb}/{thinking}/{rung}"))
}

fn row<'a>(v: &'a mut Value, verb: &str, thinking: &str, rung: &str) -> &'a mut Value {
    let i = row_index(v, verb, thinking, rung);
    &mut v["cells"][i]
}

fn cell(host: &str, verb: &str, thinking: &str, rung: &str) -> String {
    format!("release-cell/0.69.1/{host}/{MODEL}/{verb}/think-{thinking}/{rung}")
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
        msgs.join("\n")
    );
    v
}

#[test]
fn the_green_release_passes_with_every_cell_named() {
    let t = green();
    let r = gate(t.path(), &[]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass");
    assert_eq!(
        v["pc_shape"], "fired",
        "the plant fired from the named family"
    );
    assert_eq!(
        v["pc_extract"]["release-evidence"], "fired",
        "the extractor's own control (PMAT-3704)"
    );
    let rel = &v["release"];
    assert_eq!(
        rel["cells"], 48,
        "2 hosts × 4 verbs × 2 thinking × (4k, consumer-max, declared)"
    );
    assert_eq!(rel["cells_with_row"], 48);
    assert_eq!(rel["cell_names"].as_array().map(Vec::len), Some(48));
    assert_eq!(
        rel["kernel_cells"], 4,
        "(q4k_gemv@q4_k, rmsnorm@f32) × 2 hosts"
    );
    assert_eq!(rel["orphan_rows"], 0);
    assert_eq!(
        rel["tokenizer_cells"], 1,
        "one distinct model across both hosts (#3726)"
    );
    let armed = v["armed_shapes"].as_array().expect("armed").len();
    assert_eq!(
        armed, 9,
        "--shape arms the whole family whatever armed_shapes says"
    );
}

#[test]
fn a_missing_cell_is_red_naming_it() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        let i = row_index(v, "chat", "on", "consumer-max");
        v["cells"].as_array_mut().expect("cells").remove(i);
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        &cell("lambda", "chat", "on", "consumer-max"),
    );
}

#[test]
fn a_skipped_cell_is_red_naming_it() {
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        row(v, "serve", "off", "4k")["verdict"] = "skip".into();
    });
    assert_red_naming(&gate(t.path(), &[]), &cell("gx10", "serve", "off", "4k"));
}

#[test]
fn a_fallback_cell_is_red_naming_it() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        row(v, "run", "off", "4k")["fallback"] = true.into();
    });
    assert_red_naming(&gate(t.path(), &[]), &cell("lambda", "run", "off", "4k"));
}

#[test]
fn a_cell_that_does_not_say_whether_it_fell_back_is_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        row(v, "code", "off", "4k")
            .as_object_mut()
            .expect("row")
            .remove("fallback");
    });
    assert_red_naming(&gate(t.path(), &[]), &cell("lambda", "code", "off", "4k"));
}

#[test]
fn a_receipt_measured_at_another_commit_is_stale_and_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        v["apr_sha"] = PARENT.into()
    });
    let v = assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/lambda");
    let msgs = v["findings"].to_string();
    assert!(
        msgs.contains(&cell("lambda", "run", "off", "4k")),
        "every lambda cell is stale"
    );
    assert!(
        !msgs.contains("release-host/0.69.1/gx10"),
        "gx10 is untouched"
    );
}

#[test]
fn a_receipt_for_another_version_is_stale_and_red() {
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        v["version"] = "0.69.0".into()
    });
    assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/gx10");
}

#[test]
fn a_think_on_row_with_an_unclosed_block_and_an_empty_answer_is_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        let r = row(v, "chat", "on", "4k");
        r["think_closed"] = false.into();
        r["answer_chars"] = 0.into();
    });
    let v = assert_red_naming(&gate(t.path(), &[]), &cell("lambda", "chat", "on", "4k"));
    let msgs = v["findings"].to_string();
    assert!(msgs.contains("thinkOk"), "names the unclosed block: {msgs}");
    assert!(msgs.contains("answered"), "names the empty answer: {msgs}");
}

#[test]
fn a_missing_think_on_cell_is_red_naming_it() {
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        let i = row_index(v, "code", "on", "declared");
        v["cells"].as_array_mut().expect("cells").remove(i);
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        &cell("gx10", "code", "on", "declared"),
    );
}

#[test]
fn an_absent_declared_length_rung_is_red_and_so_is_an_unmeasured_length() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        let i = row_index(v, "run", "off", "declared");
        v["cells"].as_array_mut().expect("cells").remove(i);
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        &cell("lambda", "run", "off", "declared"),
    );

    // the length is a property of the FILE (one sha256): gx10 dropping it leaves lambda's measurement standing,
    // but gx10's own declared cells have no size to fill and go red …
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        v["inventory"][0]
            .as_object_mut()
            .expect("item")
            .remove("context_length");
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        &cell("gx10", "run", "off", "declared"),
    );
    // … and with no host measuring it, the model itself is named
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
fn two_hosts_that_disagree_on_one_files_length_are_red() {
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        v["inventory"][0]["context_length"] = 4096.into();
    });
    let v = assert_red_naming(&gate(t.path(), &[]), "contextLength");
    assert!(v["findings"].to_string().contains("maxCount"));
}

#[test]
fn a_prompt_short_of_its_rung_is_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        row(v, "serve", "on", "consumer-max")["prompt_tokens"] = 1000.into();
    });
    let v = assert_red_naming(
        &gate(t.path(), &[]),
        &cell("lambda", "serve", "on", "consumer-max"),
    );
    assert!(v["findings"].to_string().contains("contextMet"));
}

#[test]
fn two_rows_for_one_cell_are_ambiguous_and_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        let dup = row(v, "run", "on", "4k").clone();
        v["cells"].as_array_mut().expect("cells").push(dup);
    });
    let v = assert_red_naming(&gate(t.path(), &[]), &cell("lambda", "run", "on", "4k"));
    assert!(v["findings"].to_string().contains("maxCount"));
}

#[test]
fn a_kernel_green_on_sm_121_and_red_on_sm_89_is_named_on_lambda_before_any_model_cell() {
    let t = green();
    edit(&kernels(t.path(), "lambda"), |v| {
        v["rows"][0]["verdict"] = "fail".into();
        v["rows"][0]["max_err"] = 0.5.into();
    });
    // and one model cell too, so the ORDER is observable
    edit(&models(t.path(), "lambda"), |v| {
        row(v, "run", "off", "4k")["verdict"] = "fail".into();
    });
    let v = assert_red_naming(
        &gate(t.path(), &[]),
        "release-kernel/0.69.1/lambda/q4k_gemv/q4_k",
    );
    let findings = v["findings"].as_array().expect("findings");
    let first = findings[0]["message"].as_str().unwrap_or("");
    assert!(
        first.contains("release-kernel/0.69.1/lambda"),
        "kernel first: {first}"
    );
    assert!(
        !v["findings"]
            .to_string()
            .contains("release-kernel/0.69.1/gx10"),
        "gx10's kernel is green"
    );
}

#[test]
fn a_kernel_measured_on_one_arch_only_is_a_missing_row_on_the_other() {
    let t = green();
    edit(&kernels(t.path(), "gx10"), |v| {
        v["rows"].as_array_mut().expect("rows").remove(1); // rmsnorm@f32 unmeasured on gx10
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        "release-kernel/0.69.1/gx10/rmsnorm/f32",
    );
}

#[test]
fn a_required_host_with_no_receipt_is_red_naming_the_host() {
    let t = green();
    std::fs::remove_file(models(t.path(), "gx10")).expect("rm");
    assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/gx10");
}

#[test]
fn an_inherited_dogfood_or_none_at_all_is_red() {
    let t = green();
    edit(&t.path().join("dogfood-receipt.json"), |v| {
        v["commit"] = PARENT.into();
    });
    assert_red_naming(&gate(t.path(), &[]), "release-dogfood/0.69.1");

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
            "--release-version",
            "0.69.1",
            "--release-commit",
            MC,
        ],
    );
    assert_red_naming(&r, "dogfoodReceipt");
}

#[test]
fn a_consumer_with_no_recorded_number_is_red_on_the_derived_rung() {
    let t = green();
    edit(&t.path().join("evidence/release/context-rungs.json"), |v| {
        v["consumers"].as_array_mut().expect("consumers").push(
            serde_json::from_str::<Value>(
                r#"{"consumer": "pending", "basis": "", "source": "fixture://pending"}"#,
            )
            .expect("a consumer record"),
        );
    });
    assert_red_naming(&gate(t.path(), &[]), "release-context/consumer-max");
}

#[test]
fn thinking_that_contradicts_its_template_markers_is_red() {
    let t = green();
    edit(&models(t.path(), "lambda"), |v| {
        // off only, though the template carries the `enable_thinking` switch (#3723's derivation says both)
        v["inventory"][0]["thinking_modes"] =
            serde_json::from_str::<Value>(r#"["off"]"#).expect("modes");
    });
    assert_red_naming(&gate(t.path(), &[]), "thinkingContradiction");
}

#[test]
fn a_v1_receipt_cannot_stand_for_a_host() {
    let t = green();
    edit(&models(t.path(), "gx10"), |v| {
        v["schema"] = "apr-model-ladder-receipt/v1".into();
    });
    assert_red_naming(&gate(t.path(), &[]), "release-host/0.69.1/gx10");
}

#[test]
fn t4_receipts_measured_at_the_bump_head_pass_only_with_receipts_commit() {
    // T-4: the committed receipts were measured at the bump head; R7 proved it equal to MC modulo evidence/
    let t = green();
    let other = t.path().join("t4-receipts");
    std::fs::create_dir_all(&other).expect("mkdir");
    for h in ["lambda", "gx10"] {
        let dst = other.join(format!("{h}.json"));
        std::fs::rename(models(t.path(), h), &dst).expect("mv");
        edit(&dst, |v| v["apr_sha"] = BUMP.into());
        edit(&kernels(t.path(), h), |v| v["apr_sha"] = BUMP.into());
        edit(&tokenizer(t.path()), |v| v["apr_sha"] = BUMP.into());
    }
    let dir = s(&other);
    let r = gate(t.path(), &["--receipts", &dir, "--receipts-commit", BUMP]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let r = gate(t.path(), &["--receipts", &dir]);
    assert_red_naming(&r, "release-host/0.69.1/lambda");
}

#[test]
fn no_release_subject_declines_it_never_passes() {
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
}

#[test]
fn a_malformed_subject_or_a_misplaced_flag_is_the_callers_error() {
    let t = green();
    let contracts = s(&t.path().join("contracts"));
    let partial = pv(
        t.path(),
        &[
            "lint",
            &contracts,
            "--gate",
            "shapes",
            "--release-version",
            "0.69.1",
        ],
    );
    assert_eq!(partial.code, 3, "{}", show(&partial));
    let short = pv(
        t.path(),
        &[
            "lint",
            &contracts,
            "--gate",
            "shapes",
            "--release-version",
            "0.69.1",
            "--release-commit",
            "225b2a9ab",
        ],
    );
    assert_eq!(
        short.code,
        3,
        "a prefix is refused, never matched: {}",
        show(&short)
    );
    let misplaced = pv(
        t.path(),
        &["lint", &contracts, "--gate", "sigma", "--shape", "x"],
    );
    assert_eq!(misplaced.code, 3, "{}", show(&misplaced));
    let unknown = pv(
        t.path(),
        &[
            "lint",
            &contracts,
            "--gate",
            "shapes",
            "--shape",
            "no-such-shape",
        ],
    );
    assert_eq!(unknown.code, 3, "{}", show(&unknown));
}

#[test]
fn pv_extract_writes_the_release_graph_only_where_it_is_told() {
    let t = green();
    let contracts = s(&t.path().join("contracts"));
    let out = t.path().join("release.nt");
    let dogfood = s(&t.path().join("dogfood-receipt.json"));
    let args = [
        "extract",
        &contracts,
        "--release-version",
        "0.69.1",
        "--release-commit",
        MC,
        "--dogfood-receipt",
        &dogfood,
    ];
    let refused = pv(t.path(), &args);
    assert_eq!(refused.code, 3, "no --out: {}", show(&refused));
    assert!(
        !t.path().join("contracts/contracts.nt").exists(),
        "the tracked file is untouched"
    );
    let out_s = s(&out);
    let mut with_out = args.to_vec();
    with_out.extend_from_slice(&["--out", &out_s]);
    let r = pv(t.path(), &with_out);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["release"]["cells"], 48);
    let nt = std::fs::read_to_string(&out).expect("written");
    assert!(nt.contains("release/Cell"), "the cells are in the graph");
    assert!(nt.contains("release/KernelDiffReceipt"));
    assert!(!nt.contains("_:"), "no blank nodes (R-15)");
    assert!(!t.path().join("contracts/contracts.nt").exists());
}

fn tokenizer(t: &Path) -> PathBuf {
    t.join("evidence/dogfood/tokenizer/0.69.1/parity.json")
}

const TOK_CELL: &str = "release-tokenizer/0.69.1/tiny-qwen35.gguf";

#[test]
fn one_token_id_that_disagrees_with_llama_cpp_is_red_naming_the_model() {
    // #3726: apr mapped every non-ASCII byte to id 0 and `apr parity` could not see it (CPU and GPU share it)
    let t = green();
    edit(&tokenizer(t.path()), |v| {
        v["rows"][0]["mismatches"] = 1.into()
    });
    let v = assert_red_naming(&gate(t.path(), &[]), TOK_CELL);
    assert!(v["findings"].to_string().contains("parityExact"));
}

#[test]
fn a_tokenizer_comparison_over_nothing_or_a_broken_round_trip_or_no_receipt_is_red() {
    let t = green();
    edit(&tokenizer(t.path()), |v| {
        v["rows"][0]["tokens_compared"] = 0.into()
    });
    assert_red_naming(&gate(t.path(), &[]), TOK_CELL);

    let t = green();
    edit(&tokenizer(t.path()), |v| {
        v["rows"][0]["roundtrip_ok"] = false.into()
    });
    let v = assert_red_naming(&gate(t.path(), &[]), TOK_CELL);
    assert!(v["findings"].to_string().contains("roundTrip"));

    let t = green();
    std::fs::remove_file(tokenizer(t.path())).expect("rm");
    assert_red_naming(&gate(t.path(), &[]), TOK_CELL);
}

// ── the memory rule (#3710, operator decision "a") ──────────────────────────────────────────────────────
// weights 16 GB + 1 MB/token KV + 1 GB workspace: `4k` (21.1 GB) fits the 24 GB card; `consumer-max` (25 GB) and
// `declared` (27 GB) do not, and are owed on lambda as honest pre-load refusals — and on gx10 (119 GB) as passes.

const GB: u64 = 1_000_000_000;

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
    edit(&models(t, "lambda"), |v| {
        for c in v["cells"].as_array_mut().expect("cells") {
            let need = match c["context"].as_str() {
                Some("consumer-max") => 25 * GB,
                Some("declared") => 27 * GB,
                _ => continue,
            };
            c["verdict"] = "refused".into();
            c["required_bytes"] = need.into();
            c["available_bytes"] = (24 * GB).into();
            c["reason"] =
                "pre-load refusal: KV for this context needs more than the device holds".into();
        }
    });
}

#[test]
fn a_27b_that_refuses_honestly_on_lambda_and_passes_on_gx10_is_green() {
    let t = green();
    memory_split(t.path());
    let r = gate(t.path(), &[]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(
        v["release"]["refusal_cells"], 16,
        "4 verbs × 2 modes × (consumer-max, declared) on lambda"
    );
    assert_eq!(v["release"]["cells"], 48, "a refusal is still an owed cell");
}

#[test]
fn the_gx10_cell_that_covers_the_rung_missing_is_red() {
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "gx10"), |v| {
        let i = row_index(v, "chat", "off", "declared");
        v["cells"].as_array_mut().expect("cells").remove(i);
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        &cell("gx10", "chat", "off", "declared"),
    );
}

#[test]
fn an_oom_mid_run_where_a_refusal_is_owed_is_red() {
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "lambda"), |v| {
        let r = row(v, "serve", "on", "declared");
        r["verdict"] = "error".into();
        r["reason"] = "CUDA_ERROR_OUT_OF_MEMORY after 41 s of prefill".into();
        r.as_object_mut().expect("row").remove("required_bytes");
        r.as_object_mut().expect("row").remove("available_bytes");
    });
    let v = assert_red_naming(
        &gate(t.path(), &[]),
        &cell("lambda", "serve", "on", "declared"),
    );
    assert!(v["findings"]
        .to_string()
        .contains("release-readiness-v1.refusal"));
}

#[test]
fn a_refusal_whose_arithmetic_shows_no_shortfall_is_red() {
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "lambda"), |v| {
        row(v, "code", "off", "consumer-max")["required_bytes"] = (20 * GB).into();
    });
    let v = assert_red_naming(
        &gate(t.path(), &[]),
        &cell("lambda", "code", "off", "consumer-max"),
    );
    assert!(v["findings"].to_string().contains("arithmeticNamed"));
}

#[test]
fn a_cell_that_fits_by_the_arithmetic_but_is_missing_is_red() {
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "lambda"), |v| {
        let i = row_index(v, "run", "on", "4k");
        v["cells"].as_array_mut().expect("cells").remove(i);
    });
    assert_red_naming(&gate(t.path(), &[]), &cell("lambda", "run", "on", "4k"));
}

#[test]
fn a_rung_that_fits_on_no_required_host_is_red_naming_the_rung() {
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "gx10"), |v| {
        v["gpu_mem_total_bytes"] = (24 * GB).into();
    });
    assert_red_naming(
        &gate(t.path(), &[]),
        &format!("release-coverage/0.69.1/{MODEL}/declared"),
    );
}

#[test]
fn a_refusal_because_a_co_tenant_holds_memory_is_red() {
    // required 23 GB ≤ the card's 24 GB total, refused because only 20 GB were free: a busy GPU, not the
    // arithmetic — it must not shrink what the release proves (#3712, aprender-62)
    let t = green();
    memory_split(t.path());
    edit(&models(t.path(), "lambda"), |v| {
        let r = row(v, "chat", "off", "consumer-max");
        r["required_bytes"] = (23 * GB).into();
        r["available_bytes"] = (20 * GB).into();
    });
    let v = assert_red_naming(
        &gate(t.path(), &[]),
        &cell("lambda", "chat", "off", "consumer-max"),
    );
    assert!(v["findings"].to_string().contains("arithmeticNamed"));
}

#[test]
fn a_model_whose_template_always_opens_think_owes_only_the_on_cells() {
    // Qwen3-style prefill that always opens <think> (#3723): ON only — the OFF cells are not owed, so removing
    // every OFF row leaves the release green, and removing an ON row does not
    let t = green();
    for h in ["lambda", "gx10"] {
        edit(&models(t.path(), h), |v| {
            v["inventory"][0]["thinking_modes"] =
                serde_json::from_str::<Value>(r#"["on"]"#).expect("modes");
            v["inventory"][0]["thinking_markers"] =
                serde_json::from_str::<Value>(r#"["<think>"]"#).expect("markers");
            let cells = v["cells"].as_array_mut().expect("cells");
            cells.retain(|c| c["thinking"] == "on");
        });
    }
    let r = gate(t.path(), &[]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert_eq!(
        json_of(&r)["release"]["cells"],
        24,
        "2 hosts × 4 verbs × ON × 3 rungs"
    );
    edit(&models(t.path(), "gx10"), |v| {
        let i = row_index(v, "code", "on", "4k");
        v["cells"].as_array_mut().expect("cells").remove(i);
    });
    assert_red_naming(&gate(t.path(), &[]), &cell("gx10", "code", "on", "4k"));
}
