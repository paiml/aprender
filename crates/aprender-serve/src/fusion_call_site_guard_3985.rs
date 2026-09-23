//! #3985: every ACTIVE fusion decision in `contracts/kernel-fusion-v1.yaml` cites a
//! `call_site` that EXISTS and is the fused kernel's call.
//!
//! `tests/fusion_gate_contract_falsify.rs` claimed to enforce
//! `enforcement.active_must_be_called`, but only checked that the string
//! `call_site:` occurred somewhere in the file. Four ACTIVE entries pointed into
//! `cuda/generate.rs`, a file nothing compiled, at lines that had drifted onto
//! unrelated arms; two more still cited pre-monorepo `realizar/...` paths. That
//! test is also in no CI workflow. This one is a lib test, so CI's `--lib` run
//! executes it.
//!
//! The check per ACTIVE entry: the path exists from the workspace root, the line
//! exists, and within a few lines of it the fused kernel's name appears (case and
//! underscores ignored, so `FusedSwigluKernel` matches `fused_swiglu_into`). A
//! path that moved, a line that drifted and a file that stopped being the call
//! site all fail.

use std::path::Path;

/// The contract, read from the workspace at RUN time (#4048). It used to be
/// `include_str!("../../../contracts/…")`, a path outside the crate, so `cargo test` from
/// the published `aprender-serve` tarball could not even compile this module.
const CONTRACT_PATH: &str = "contracts/kernel-fusion-v1.yaml";

/// Lines searched around the cited one for the kernel's name: one before, three
/// after (a call often spans a `let kernel_type =` line and its arguments).
const WINDOW_BEFORE: usize = 1;
const WINDOW_AFTER: usize = 3;

fn normalise(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

/// `Err(reason)` if `call_site` is not a live call of the kernel `fused` names.
fn check_call_site(root: &Path, fused: &str, call_site: &str) -> Result<(), String> {
    let (path, line) = call_site
        .rsplit_once(':')
        .ok_or_else(|| format!("`{call_site}` has no `:line`"))?;
    let line: usize = line
        .parse()
        .map_err(|_| format!("`{call_site}`: `{line}` is not a line number"))?;
    let src = std::fs::read_to_string(root.join(path))
        .map_err(|e| format!("`{path}` does not exist from the workspace root ({e})"))?;
    let lines: Vec<&str> = src.lines().collect();
    if line == 0 || line > lines.len() {
        return Err(format!(
            "`{path}` has {} lines, cited line {line}",
            lines.len()
        ));
    }
    // `FusedSwigluKernel (path)` -> `FusedSwiglu`
    let kernel = fused.split_whitespace().next().unwrap_or(fused);
    let stem = normalise(kernel.strip_suffix("Kernel").unwrap_or(kernel));
    let lo = line.saturating_sub(1 + WINDOW_BEFORE);
    let hi = (line + WINDOW_AFTER).min(lines.len());
    let window = lines[lo..hi].join("\n");
    if normalise(&window).contains(&stem) {
        Ok(())
    } else {
        Err(format!(
            "`{call_site}` does not call `{kernel}`; lines {}..={hi} read:\n{window}",
            lo + 1
        ))
    }
}

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `true` when this crate is built inside the aprender workspace, `false` out of tree (the
/// crates.io tarball). Keyed on the `contracts/` DIRECTORY, not on the one file: in tree, a
/// missing or renamed `kernel-fusion-v1.yaml` must FAIL the guard, never skip it.
fn in_workspace() -> bool {
    workspace_root().join("contracts").is_dir()
}

/// The contract text, or `None` (after naming the skip) when out of tree.
fn contract_or_skip(test: &str) -> Option<String> {
    if !in_workspace() {
        eprintln!(
            "SKIP {test}: out of tree (no {} beside this crate) - the #3985 guard reads the \
             workspace, which a published crate does not carry (#4048)",
            workspace_root().join("contracts").display()
        );
        return None;
    }
    let path = workspace_root().join(CONTRACT_PATH);
    Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("in tree, {} must be readable: {e}", path.display())),
    )
}

#[test]
fn every_active_fusion_call_site_is_a_live_call_of_its_kernel() {
    let Some(contract) =
        contract_or_skip("every_active_fusion_call_site_is_a_live_call_of_its_kernel")
    else {
        return;
    };
    let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(&contract).expect("parse contract");
    let decisions = doc["fusion_decisions"]
        .as_mapping()
        .expect("fusion_decisions mapping");
    let root = workspace_root();

    let mut checked = 0;
    let mut broken = Vec::new();
    for (key, d) in decisions {
        if d["status"].as_str() != Some("ACTIVE") {
            continue;
        }
        let id = d["id"].as_str().unwrap_or("?");
        let fused = d["kernels"]["fused"].as_str().unwrap_or_default();
        let Some(call_site) = d["call_site"].as_str() else {
            broken.push(format!("\n  - {id} ({key:?}): ACTIVE with no call_site"));
            continue;
        };
        checked += 1;
        if let Err(e) = check_call_site(&root, fused, call_site) {
            broken.push(format!("\n  - {id}: {e}"));
        }
    }
    // Not vacuous: the contract has seven ACTIVE fusions today.
    assert!(checked >= 7, "only {checked} ACTIVE call_sites checked");
    assert!(
        broken.is_empty(),
        "ACTIVE fusion call_site(s) that are not a live call of their kernel:{}",
        broken.concat()
    );
}

/// The case table: the check must reject each way a citation goes stale, and
/// accept the real one.
#[test]
fn the_call_site_check_rejects_each_stale_shape() {
    if contract_or_skip("the_call_site_check_rejects_each_stale_shape").is_none() {
        return;
    }
    let root = workspace_root();
    let live = "crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs";
    let src = std::fs::read_to_string(root.join(live)).expect("read live generator");
    let at = src
        .lines()
        .position(|l| l.contains("KernelType::FusedQKV {"))
        .expect("FusedQKV arm")
        + 1;
    let fused = "FusedQKVKernel (x)";

    // must-match: the real arm.
    assert!(check_call_site(&root, fused, &format!("{live}:{at}")).is_ok());
    // must-not-match: a pre-monorepo path, an out-of-range line, no line at all,
    // a line drifted well away from the arm, the deleted orphan generator.
    for (stale, why) in [
        (format!("realizar/src/cuda/{at}.rs:{at}"), "moved path"),
        (format!("{live}:999999"), "line past end of file"),
        (live.to_string(), "no line number"),
        (format!("{live}:{}", at + 40), "drifted line"),
        (
            "crates/aprender-serve/src/cuda/generate.rs:267".to_string(),
            "deleted orphan",
        ),
    ] {
        assert!(
            check_call_site(&root, fused, &stale).is_err(),
            "{why}: `{stale}` was accepted"
        );
    }
}
