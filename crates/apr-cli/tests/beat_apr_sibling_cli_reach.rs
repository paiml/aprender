//! FALSIFY-CLI-REACH-002: `trueno-rag`, `trueno-zram` and `simular` capability
//! must be reachable through `apr`.
//!
//! Both crates declared their whole command surface inside a `main.rs`. A
//! command enum in a binary target is importable by nothing, so the standalone
//! binary was the only way to reach any of it -- `apr` had no route at all.
//! Retiring those binaries before exposing the commands would have DELETED the
//! capability rather than relocating it; this test makes that ordering error
//! detectable.
//!
//! Asserts on the OUTPUT of the built `apr`, not on the Rust enums: the enum
//! being correct is not the claim. The claim is that a user typing
//! `apr rag index` reaches the implementation and it does the work.

use std::process::Command;

fn apr() -> Command {
    Command::new(env!("CARGO_BIN_EXE_apr"))
}

/// Top-level commands `trueno-rag` declares under default features.
const RAG_COMMANDS: &[&str] = &[
    "demo",
    "index",
    "query",
    "transcribe",
    "extract-frames",
    "info",
];

/// Top-level commands `trueno-zram` declares.
const ZRAM_COMMANDS: &[&str] = &["create", "remove", "status", "benchmark"];

fn help_body(args: &[&str]) -> String {
    let out = apr().args(args).output().expect("apr help");
    assert!(
        out.status.success(),
        "`apr {}` must succeed, got {:?}\nstderr: {}",
        args.join(" "),
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let help = String::from_utf8_lossy(&out.stdout).into_owned();
    // Control: an empty or error help body would make every `contains` below
    // pass vacuously, which is how a reach test reports green while reaching
    // nothing.
    assert!(
        help.contains("Commands:"),
        "help body has no command list, so the assertions below prove nothing: {help}"
    );
    help
}

#[test]
fn every_trueno_rag_command_is_reachable_through_apr_rag() {
    let help = help_body(&["rag", "--help"]);

    let missing: Vec<&str> = RAG_COMMANDS
        .iter()
        .copied()
        .filter(|c| !help.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "`apr rag` does not offer {missing:?}; they are reachable only from the \
         standalone `trueno-rag` binary.\n{help}"
    );

    // Non-vacuity: the assertion above must be able to FAIL. A `contains` sweep
    // over a body that happens to contain everything excludes no outcome.
    assert!(
        !help.contains("wharrgarbl"),
        "help contains a name no command has, so `contains` proves nothing here"
    );
}

#[test]
fn every_trueno_zram_command_is_reachable_through_apr_zram() {
    let help = help_body(&["zram", "--help"]);

    let missing: Vec<&str> = ZRAM_COMMANDS
        .iter()
        .copied()
        .filter(|c| !help.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "`apr zram` does not offer {missing:?}; they are reachable only from the \
         standalone `trueno-zram` binary.\n{help}"
    );
    assert!(
        !help.contains("wharrgarbl"),
        "help contains a name no command has, so `contains` proves nothing here"
    );
}

/// Help-listing proves clap wiring. It does NOT prove the arm reaches the
/// implementation -- a variant dispatching to `unimplemented!()` would list
/// identically. These next tests run commands that produce output only the real
/// handler can produce.
#[test]
fn apr_rag_info_reaches_the_loader_registry() {
    let out = apr().args(["rag", "info"]).output().expect("apr rag info");
    assert!(
        out.status.success(),
        "apr rag info exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = String::from_utf8_lossy(&out.stdout);

    // The extension list is read from `LoaderRegistry::new()` at runtime; clap
    // cannot produce it.
    assert!(
        body.contains("Supported formats:"),
        "no format list: `apr rag info` did not reach run_info.\n{body}"
    );
    assert!(
        body.contains("srt"),
        "format list does not name the subtitle loader, so the registry was not \
         consulted.\n{body}"
    );
}

#[test]
fn apr_rag_indexes_and_queries_real_documents() {
    let dir = tempfile::tempdir().expect("tempdir");
    let docs = dir.path().join("docs");
    std::fs::create_dir_all(&docs).expect("mkdir docs");
    std::fs::write(
        docs.join("simd.txt"),
        "Trueno provides SIMD acceleration for matrix multiplication.\n",
    )
    .expect("write simd.txt");
    std::fs::write(
        docs.join("frame.txt"),
        "Aprender is a machine learning framework written in pure Rust.\n",
    )
    .expect("write frame.txt");
    let index = dir.path().join("idx");

    let out = apr()
        .args(["rag", "index", "--path"])
        .arg(&docs)
        .arg("--output")
        .arg(&index)
        .output()
        .expect("apr rag index");
    assert!(
        out.status.success(),
        "apr rag index exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        index.join("index.json").is_file(),
        "apr rag index reported success but wrote no index.json"
    );

    let out = apr()
        .args(["rag", "query", "--index"])
        .arg(&index)
        .arg("SIMD matrix")
        .output()
        .expect("apr rag query");
    assert!(
        out.status.success(),
        "apr rag query exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        body.contains("SIMD acceleration for matrix multiplication"),
        "the matching document is absent from the results, so nothing was \
         actually retrieved.\n{body}"
    );
    // Excludes the outcome where retrieval returns everything in file order:
    // the SIMD document must OUTRANK the unrelated one.
    let simd = body
        .find("SIMD acceleration")
        .expect("simd document in results");
    let other = body.find("machine learning framework");
    assert!(
        other.is_none_or(|o| simd < o),
        "the unrelated document outranked the matching one, so scoring did not \
         run.\n{body}"
    );
}

#[test]
fn apr_zram_benchmark_runs_instead_of_panicking() {
    // Regression: `pages` derived short `-p` while `pattern` explicitly claimed
    // it. clap's debug_assert fired before any argument was parsed, so EVERY
    // invocation of this command -- `--help` included -- aborted with exit 101.
    // Nothing ever ran it, so nothing noticed.
    let out = apr()
        .args(["zram", "benchmark", "--pages", "64", "--algorithm", "lz4"])
        .output()
        .expect("apr zram benchmark");
    assert_ne!(
        out.status.code(),
        Some(101),
        "apr zram benchmark aborted on a panic\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "apr zram benchmark exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        body.contains("Compress") && body.contains("Ratio"),
        "no benchmark table: the command did not reach trueno_zram_core.\n{body}"
    );
    // Non-vacuity for the table above: a real run reports a ratio, not a header
    // with nothing under it.
    assert!(
        body.contains('x') && body.lines().count() > 6,
        "benchmark printed a header and no measurements.\n{body}"
    );
}

// A cross-binary byte-identity check (`apr rag query` vs `trueno-rag query`)
// belongs here in spirit, but `CARGO_BIN_EXE_*` only exposes the binaries of
// THIS package, so writing it from apr-cli yields an apr-vs-apr comparison --
// an oracle that agrees with itself by construction. The structural guarantee
// is instead that `apr`'s arm and `trueno-rag`'s `main` both call
// `aprender_rag_cli::dispatch`; the tests above prove that path executes.

/// Top-level commands `simular` declares.
const SIM_COMMANDS: &[&str] = &[
    "run",
    "render",
    "validate",
    "verify",
    "emc-check",
    "emc-validate",
    "list-emc",
];

#[test]
fn every_simular_command_is_reachable_through_apr_sim() {
    let help = help_body(&["sim", "--help"]);

    let missing: Vec<&str> = SIM_COMMANDS
        .iter()
        .copied()
        .filter(|c| !help.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "`apr sim` does not offer {missing:?}; they are reachable only from the \
         standalone `simular` binary.\n{help}"
    );
    assert!(
        !help.contains("wharrgarbl"),
        "help contains a name no command has, so `contains` proves nothing here"
    );
}

#[test]
fn apr_sim_validates_a_real_emc_file() {
    let emc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../aprender-simulate/docs/emc/ml/linear_regression.emc.yaml"
    );
    assert!(
        std::path::Path::new(emc).is_file(),
        "fixture moved: {emc} -- fix the path rather than weakening the test"
    );

    let out = apr()
        .args(["sim", "emc-validate", emc])
        .output()
        .expect("apr sim emc-validate");
    assert!(
        out.status.success(),
        "apr sim emc-validate exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = String::from_utf8_lossy(&out.stdout);
    // EMC-ML-001 comes out of the parsed YAML; clap cannot produce it.
    assert!(
        body.contains("EMC-ML-001"),
        "the EMC id is absent, so the file was never parsed.\n{body}"
    );
    assert!(
        body.contains("PASSED"),
        "schema validation did not report a verdict.\n{body}"
    );
}

/// simular's grammar used to be hand-rolled: `match args[1].as_str()` over a
/// `Vec<String>`. It did not merely duplicate clap, it SILENTLY DISCARDED
/// input -- `--seed notanumber` became `None` and the run proceeded on the
/// default seed, in the one flag that pins reproducibility. Its own unit test
/// asserted that behaviour ("Missing value and invalid value both result in
/// None seed"), which is how the defect survived.
///
/// These assert the grammar is now declarative and rejecting, through `apr`.
#[test]
fn apr_sim_rejects_input_the_hand_rolled_parser_swallowed() {
    for (args, what) in [
        (
            vec!["sim", "run", "experiment.yaml", "--seed", "not-a-number"],
            "a non-numeric seed",
        ),
        (
            vec!["sim", "run", "experiment.yaml", "--seed"],
            "a seed flag with no value",
        ),
        (
            vec!["sim", "run", "experiment.yaml", "--verbse"],
            "a misspelled flag",
        ),
        (
            vec!["sim", "verify", "experiment.yaml", "--runs", "many"],
            "a non-numeric run count",
        ),
    ] {
        let out = apr().args(&args).output().expect("apr sim");
        assert!(
            !out.status.success(),
            "{what} was accepted: `apr {}` exited 0. Silently ignoring it is the \
             defect this test exists for.",
            args.join(" ")
        );
    }

    // Control: a well-formed invocation of the SAME subcommand is not rejected
    // by the parser. Without this, "everything fails" would pass the loop above.
    let out = apr()
        .args(["sim", "run", "--seed", "7", "definitely-not-here.yaml"])
        .output()
        .expect("apr sim run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("invalid value") && !stderr.contains("unexpected argument"),
        "a well-formed invocation was rejected by the PARSER, so the rejections \
         above are not specific to bad input.\nstderr: {stderr}"
    );
}

/// Top-level commands `aprender-cgp` declares.
const CGP_COMMANDS: &[&str] = &[
    "profile", "bench", "roofline", "diff", "contract", "trace", "explain", "tui", "baseline",
    "doctor", "compete",
];

/// Top-level commands `apr-qa` declares.
const QA_PLAYBOOK_COMMANDS: &[&str] = &[
    "certify",
    "run",
    "tools",
    "generate",
    "score",
    "report",
    "list",
    "lock-playbooks",
    "tickets",
    "parity",
    "export-csv",
    "export-evidence",
    "bootstrap",
    "validate-contract",
    "kernel-coverage",
];

#[test]
fn every_cgp_command_is_reachable_through_apr_cgp() {
    let help = help_body(&["cgp", "--help"]);
    let missing: Vec<&str> = CGP_COMMANDS
        .iter()
        .copied()
        .filter(|c| !help.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "`apr cgp` does not offer {missing:?}\n{help}"
    );
    assert!(!help.contains("wharrgarbl"), "contains proves nothing here");
}

#[test]
fn every_apr_qa_command_is_reachable_through_apr_qa_playbook() {
    // Named `qa-playbook`, not `qa`: `apr qa` is already the falsifiable-gates
    // command that takes a model path. Two different tools, two names.
    let help = help_body(&["qa-playbook", "--help"]);
    let missing: Vec<&str> = QA_PLAYBOOK_COMMANDS
        .iter()
        .copied()
        .filter(|c| !help.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "`apr qa-playbook` does not offer {missing:?}\n{help}"
    );
    assert!(!help.contains("wharrgarbl"), "contains proves nothing here");
}

#[test]
fn apr_cgp_doctor_probes_the_real_machine() {
    let out = apr()
        .args(["cgp", "doctor"])
        .output()
        .expect("apr cgp doctor");
    assert!(
        out.status.success(),
        "apr cgp doctor exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        body.contains("System Check"),
        "no system check banner: the arm did not reach cgp.\n{body}"
    );
    // Every probe reports a verdict, so the report is not an empty shell. Which
    // tools are present is machine-dependent; that SOME verdict is rendered is
    // not.
    assert!(
        body.contains("[OK]") || body.contains("[MISSING]"),
        "no probe reported a verdict, so nothing was actually checked.\n{body}"
    );
}

#[test]
fn apr_qa_playbook_list_reads_the_real_registry() {
    let out = apr()
        .args(["qa-playbook", "list"])
        .output()
        .expect("apr qa-playbook list");
    assert!(
        out.status.success(),
        "apr qa-playbook list exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let body = String::from_utf8_lossy(&out.stdout);
    let total = body
        .lines()
        .find_map(|l| l.trim().strip_prefix("Total: "))
        .and_then(|t| t.split_whitespace().next())
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or_else(|| panic!("no `Total: N models` line in output:\n{body}"));
    // Excludes the outcome where the registry loads but is empty -- which is how
    // a scan reports clean having examined nothing.
    assert!(
        total > 0,
        "the registry listed 0 models, so `list` proved nothing.\n{body}"
    );
}
