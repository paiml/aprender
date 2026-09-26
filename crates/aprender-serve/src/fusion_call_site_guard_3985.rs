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
//! #3422: a `path:line` citation went RED on an unrelated edit — #4234 moved the
//! arms of `kernels_generate_gemm_cuda.rs` and FUSION-009/010's lines fell off
//! their kernels. A line number is a property of every edit above it, not of the
//! call. So a `call_site` is `path#symbol`, and the check per ACTIVE entry is:
//! the path exists from the workspace root; the symbol occurs there EXACTLY ONCE
//! outside a `//` comment (an anchor that matches twice anchors nothing); and the
//! symbol NAMES the fused kernel as a whole identifier (case and underscores
//! ignored, `Kernel`/`_into` suffixes allowed, so `self.fused_swiglu_into(` names
//! `FusedSwigluKernel`, but `KernelType::FusedGateUpQ4KGemv {` does not name
//! `FusedGateUpKernel`). A `path:line` citation is rejected outright.

use std::path::Path;

const CONTRACT: &str = include_str!("../../../contracts/kernel-fusion-v1.yaml");

fn normalise(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

/// True when some identifier in `symbol` IS the kernel `stem` (normalised), bare
/// or with a `Kernel` / `_into` suffix. Whole identifiers, never substrings:
/// `FusedGateUp` must not be satisfied by `FusedGateUpQ4KGemv`.
fn names_kernel(symbol: &str, stem: &str) -> bool {
    symbol
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map(normalise)
        .any(|t| t == stem || t == format!("{stem}kernel") || t == format!("{stem}into"))
}

/// `Err(reason)` if `call_site` is not a live call of the kernel `fused` names.
fn check_call_site(root: &Path, fused: &str, call_site: &str) -> Result<(), String> {
    let (path, symbol) = call_site.split_once('#').ok_or_else(|| {
        format!("`{call_site}` is not `path#symbol` (a line number drifts on every edit above it, #3422)")
    })?;
    if symbol.trim().is_empty() {
        return Err(format!("`{call_site}` has an empty symbol"));
    }
    let src = std::fs::read_to_string(root.join(path))
        .map_err(|e| format!("`{path}` does not exist from the workspace root ({e})"))?;
    let hits: Vec<(usize, &str)> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(symbol) && !l.trim_start().starts_with("//"))
        .collect();
    match hits.len() {
        0 => {
            return Err(format!(
                "`{path}` has no non-comment occurrence of `{symbol}`"
            ))
        },
        1 => {},
        n => {
            let at: Vec<String> = hits.iter().map(|(i, _)| (i + 1).to_string()).collect();
            return Err(format!(
                "`{symbol}` occurs {n} times in `{path}` (lines {}); an anchor must be unique",
                at.join(", ")
            ));
        },
    }
    // `FusedSwigluKernel (path)` -> `fusedswiglu`
    let kernel = fused.split_whitespace().next().unwrap_or(fused);
    let stem = normalise(kernel.strip_suffix("Kernel").unwrap_or(kernel));
    if names_kernel(symbol, &stem) {
        Ok(())
    } else {
        Err(format!(
            "`{symbol}` (line {} of `{path}`) does not name `{kernel}`",
            hits[0].0 + 1
        ))
    }
}

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn every_active_fusion_call_site_is_a_live_call_of_its_kernel() {
    let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(CONTRACT).expect("parse contract");
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

/// The case table: the check must reject each way a citation goes stale or
/// ambiguous, accept the real one, and keep accepting it after the lines move.
#[test]
fn the_call_site_check_rejects_each_stale_shape() {
    let root = workspace_root();
    let live = "crates/aprender-serve/src/cuda/kernels_generate_gemm_cuda.rs";
    let qkv = "FusedQKVKernel (x)";
    let gate_up = "FusedGateUpKernel (x)";

    // must-match: the real arms, and a call named with an `_into` suffix.
    for (fused, site) in [
        (qkv, format!("{live}#KernelType::FusedQKV {{")),
        (gate_up, format!("{live}#KernelType::FusedGateUp {{")),
        (
            "FusedSwigluKernel (x)",
            "crates/aprender-serve/src/cuda/executor/layers/indexed_ffn.rs#self.fused_swiglu_into("
                .to_string(),
        ),
    ] {
        assert!(
            check_call_site(&root, fused, &site).is_ok(),
            "real call site `{site}` was rejected: {:?}",
            check_call_site(&root, fused, &site)
        );
    }

    // must-not-match against the real tree.
    for (fused, stale, why) in [
        (
            qkv,
            format!("realizar/src/cuda/x.rs#KernelType::FusedQKV {{"),
            "moved path",
        ),
        (qkv, format!("{live}:371"), "a line number, not a symbol"),
        (qkv, live.to_string(), "no symbol at all"),
        (qkv, format!("{live}#"), "empty symbol"),
        (
            qkv,
            format!("{live}#KernelType::FusedNoSuchArm {{"),
            "symbol absent",
        ),
        (qkv, format!("{live}#KernelType::"), "ambiguous: many arms"),
        (
            qkv,
            format!("{live}#KernelType::FusedGateUp {{"),
            "names another kernel",
        ),
        // Substring hole of the line-window check: `FusedGateUp` is a prefix of
        // `FusedGateUpQ4KGemv`, so the Q4K arm must not satisfy FUSION-007.
        (
            gate_up,
            format!("{live}#KernelType::FusedGateUpQ4KGemv {{"),
            "prefix-named kernel",
        ),
        (
            qkv,
            "crates/aprender-serve/src/cuda/generate.rs#KernelType::FusedQKV {".to_string(),
            "deleted orphan",
        ),
    ] {
        assert!(
            check_call_site(&root, fused, &stale).is_err(),
            "{why}: `{stale}` was accepted"
        );
    }

    // Synthetic files: line drift must NOT matter; a comment must not anchor.
    let dir = std::env::temp_dir().join(format!("fusion-guard-3422-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let arm = "        KernelType::FusedQKV { hidden_size, kv_dim } => {\n";
    let drifted = format!("{}{arm}", "// padding\n".repeat(40));
    std::fs::write(dir.join("drifted.rs"), &drifted).expect("write");
    std::fs::write(
        dir.join("comment.rs"),
        "// KernelType::FusedQKV { was here\n",
    )
    .expect("write");
    std::fs::write(dir.join("twice.rs"), format!("{arm}{arm}")).expect("write");
    let drift = check_call_site(&dir, qkv, "drifted.rs#KernelType::FusedQKV {");
    let comment = check_call_site(&dir, qkv, "comment.rs#KernelType::FusedQKV {");
    let twice = check_call_site(&dir, qkv, "twice.rs#KernelType::FusedQKV {");
    std::fs::remove_dir_all(&dir).ok();
    assert!(
        drift.is_ok(),
        "40 lines of drift broke a symbol anchor: {drift:?}"
    );
    assert!(comment.is_err(), "a `//` comment anchored the call site");
    assert!(
        twice.is_err(),
        "a symbol occurring twice was accepted as an anchor"
    );
}
