//! #5914 — the ONE debug-build `pv` pass over this repository's real `contracts/` tree.
//!
//! A debug `pv lint contracts` costs ~28 CPU-s, and seven test binaries used to pay it each (617% CPU on the
//! intel runners). Every other `pv` integration test lints a fixture corpus under `tests/fixtures/ont/`; the
//! repo tree is linted here, once, and each gate's verdict is read from that single report:
//!
//! | gate | was asserted in | asserted here |
//! |---|---|---|
//! | sigma | `ont2b_sigma_gate` | Pass |
//! | relations | `ont4_relations_gate` | Pass, typed relations present, legacy `depends_on` counted |
//! | shapes | `ont4b_shapes_gate` | Pass, plant fired, 73 shapes, 42 plant violations |
//! | refines | `ont4e_refines_gate` | Pass, a Liskov pair checked, `pc_checker` fired |
//! | theorem-pairing, depends-on-present | `ev11_lint_ratchets` | Pass, equal to the baseline file, which is not written |
//!
//! `pv extract contracts --check` (R-18, the tracked graph is fresh) is the one other repo-tree command.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

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

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .current_dir(scratch.path())
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn dir_arg() -> String {
    repo_contracts().to_str().expect("utf-8 path").to_string()
}

/// The single `pv lint contracts --format json` run, plus the baseline bytes before and after it.
fn full_run() -> &'static (Run, Vec<u8>, Vec<u8>) {
    static FULL: OnceLock<(Run, Vec<u8>, Vec<u8>)> = OnceLock::new();
    FULL.get_or_init(|| {
        let baseline = repo_contracts().join("lint-baseline.json");
        let before = std::fs::read(&baseline).expect("baseline readable");
        let r = pv(&["lint", &dir_arg(), "--format", "json"]);
        let after = std::fs::read(&baseline).expect("baseline readable");
        (r, before, after)
    })
}

fn report() -> serde_json::Value {
    let r = &full_run().0;
    assert_eq!(r.code, 0, "{}", show(r));
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

/// One gate's entry in the full report; the full run computes every gate (R-8), so its absence is a failure.
fn gate(name: &str) -> serde_json::Value {
    let v = report();
    let g = v["gates"]
        .as_array()
        .and_then(|gs| gs.iter().find(|g| g["name"] == name))
        .unwrap_or_else(|| panic!("the full run computes the {name} gate: {v}"))
        .clone();
    assert_eq!(g["verdict"], "Pass", "{name}: {g}");
    g
}

#[test]
fn the_repo_corpus_passes_the_sigma_gate() {
    gate("sigma");
}

#[test]
fn the_repo_corpus_passes_the_relations_gate() {
    let g = gate("relations");
    // R-2 by construction: the real corpus must carry typed relations, or this would be a decline
    assert!(
        g["extra"]["relations_n"].as_u64().unwrap_or(0) > 0,
        "the real corpus has typed relations\n{g}"
    );
    // the legacy field is measured, never rewritten: 188 contracts carry it at 3409b29d
    assert!(
        g["extra"]["legacy_depends_on"].as_u64().unwrap_or(0) > 100,
        "metadata.depends_on is counted\n{g}"
    );
}

#[test]
fn the_repo_corpus_passes_with_the_plant_fired_and_the_whole_corpus_as_focus_nodes() {
    let g = gate("shapes");
    let x = &g["extra"];
    assert!(
        x["focus_nodes_n"].as_u64().unwrap_or(0) > 1000,
        "every contract's projection is a focus node\n{g}"
    );
    assert_eq!(x["pc_shape"], "fired", "{g}");
    // ONT-4b: exactly one, from ont:id minCount; ONT-4c1, ONT-4f and ONT-4c add plants (see the history
    // below): 42.
    assert_eq!(x["plant_violations"], 42, "{g}");
    // The shape count is hardcoded on purpose, so adding a shape turns this red and makes you name it.
    // MEASURED: `pv extract contracts --check` reported
    // shapes_n=9, triples=15851 (the 0.69 batch, #3669, pinned pv 0.68.2 built from
    // the tree; was 6 / 15620 before #3600 folded in). The count is deliberately hardcoded rather
    // than derived — a shape added without anyone noticing is the thing this
    // assertion exists to prevent, so adding one is SUPPOSED to turn it red and
    // make you name the new shape here.
    //
    // It did exactly that for #3605's `refusal-receipt-v1`, and a quorum lane
    // caught it rather than I did. Note for whoever merges second: this counter
    // is shared across branches, so a sibling PR that also adds a shape
    // (#3600's parity-receipt-v2) will need the number raised again at merge —
    // that is the ratchet working, not a conflict to route around. It did: the
    // 0.69 batch folded #3600 in and the count went 6 -> 9 with its three shapes. #3715 added the nine-shape
    // `release-readiness-v1` family (shapes_n=18, triples=15863, measured on its branch); it contributes no focus
    // node to a PR's graph — the release evidence is extracted only under `--release-*`. ONT-4f (#4330) adds
    // `github-entities-v1`'s four armed shapes: 22. #3560 R1 adds `examples-well-formed` (reported, not armed): 23. R4 adds `examples-model-current` (reported): 24.
    // ONT-4c (#3847) adds the four armed non-code shapes readme-root, claude-md, model-setfit-slice, csv-train: 28.
    // ONT-4g/ONT-10 (#4502) add binary-target, surface-audit-ledger, binary-apr-http-mcp and fifteen
    // binary-<bin>-surface shapes (apr-cli-apr, apr-corpus-ingest, aprender-cbtop, aprender-compute-xtask, aprender-db,
    // aprender-explain, aprender-orchestrate-aprender-orchestrate, aprender-profile, aprender-ptx-debug, aprender-test-cli, aprender-zram-generator,
    // apr-qa-readme-sync, pv, score, trueno-rag): 46. Each new ONT-10 slice raises this — name it here.
    // The 0.70.0 car (#4429) added, disjoint from these: #3745 S2 (#3777) six DERIVED cell-class shapes (effect,
    // probe, model-cell, effect-cell, crux-verb, crux) and a8's `sampling`, ONT-4c5 (PMAT-3972) `capability-cells`,
    // and ONT-4c4 (#4069) kernel-parity, kernel-timing and kernel-safety: 46 + 8 + 3 = 57.
    // ONT-10 rest (#4588, PR-2) adds eight more binary-<bin>-surface shapes (apr, apr-qa, aprender-data,
    // aprender-present, aprender-ptop, aprender-simulate, aprender-train-lora, aprender-train-shell) and
    // binary-aprender-apr-http-mcp: 57 + 9 = 66.
    // #4455 adds `findings-ledger-v1` (not in `armed_shapes`; it grades zero focus nodes here): 67.
    // KREG-001 (#4539) adds `kernel-registry-v1`'s one shape, a row per dispatchable kernel: 68.
    // #3715 v2 adds `release-readiness-v2{.kernel,.model,.model-smoke,.sanitizer}` (like v1, a focus node only under
    // `--v2-evidence`; the smoke is its own shape so no shape exports two `<shape>/node` subjects) and the
    // per-forward `kernel-registry-v1.op`: 68 + 5 = 73.
    assert_eq!(x["shapes_n"], 73, "{g}");
}

#[test]
fn the_spec_probe_holds_on_the_repo_corpus() {
    // ONT-001 §5 ONT-4e probe: `lint contracts/ --gate refines` reports liskov_pairs_checked>0, pc_checker fired, Pass.
    let g = gate("refines");
    let x = &g["extra"];
    assert!(
        x["liskov_pairs_checked"].as_u64().unwrap_or(0) > 0,
        "liskov_pairs_checked>0: {g}"
    );
    assert_eq!(x["pc_checker"], "fired", "{g}");
}

/// PVL-001 EV-11's probe on the real corpus: both ratchets Pass at the recorded baseline, and no gate writes it.
#[test]
fn the_ratchet_probe_passes_on_the_real_corpus_and_writes_nothing() {
    let (_, before, after) = full_run();
    let doc: serde_json::Value = serde_json::from_slice(before).expect("baseline is JSON");
    assert!(doc["unpaired_theorem_modules"].is_u64(), "{doc}");
    assert!(doc["contracts_without_depends_on"].is_u64(), "{doc}");
    assert_eq!(doc["command"], "make lint-ratchet");
    let pairing = gate("theorem-pairing");
    let depends = gate("depends-on-present");
    assert_eq!(
        pairing["extra"]["unpaired_theorem_modules"], doc["unpaired_theorem_modules"],
        "the recorded baseline is the measured count (`make lint-ratchet` wrote it)"
    );
    assert_eq!(
        depends["extra"]["contracts_without_depends_on"],
        doc["contracts_without_depends_on"]
    );
    assert_eq!(after, before, "a gate wrote the baseline");
}

/// The tracked graph in this repository IS a fresh extraction (R-18), and `shapes.ttl` is the export.
#[test]
fn the_tracked_repo_graph_is_fresh() {
    let r = pv(&["extract", &dir_arg(), "--check"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v: serde_json::Value = serde_json::from_str(&r.stdout).expect("stdout is JSON");
    assert!(v["triples"].as_u64().unwrap_or(0) > 5000, "{}", show(&r));
}
