//! `contracts-pv` first-tranche coverage gate (#2589).
//!
//! WHY THIS FILE EXISTS
//! ====================
//! `pv` is the tool that decides whether a release is contractually sound, and
//! the surface audit reads `cluster contracts-pv: 76 features, 0.0% covered,
//! 0 UNKNOWN hardware`. That last column matters: the features are not
//! unlooked-at, they are **ungated**. The tool that makes every other subsystem
//! prove things proved nothing about itself.
//!
//! Two distinct causes were measured at 4bbfeb07f, and this file addresses both:
//!
//! 1. **pv's CLI tests were DARK.** `crates/aprender-contracts-cli/tests/` held
//!    651 lines of binary-spawning tests, and CI ran **none** of them:
//!    `workspace-test` is `cargo nextest run --workspace --lib` (library targets
//!    only), and the explicit integration list in `ci.yml` names 23 targets,
//!    not one of them from `aprender-contracts-cli`. A test that never executes
//!    is 0% coverage however many lines it has. This PR wires pv's integration
//!    targets into that list.
//!
//! 2. **The tests that did exist EXCLUDED NOTHING.** Nearly every one asserted
//!    `status.success()` on a good contract. That assertion passes unchanged
//!    against `fn main() { std::process::exit(0) }`. Per
//!    `project_assertions_exclude_guard`: an assertion that no wrong behaviour
//!    can violate is not a control. Every test below is written so that some
//!    concrete wrong behaviour makes it RED.
//!
//! SCOPE OF THIS TRANCHE — read `docs/` note in the PR body for what is deferred.
//!   COVERED: `pv validate` field/rule decision table (all 18 diagnostic rules,
//!            both directions), subcommand reachability, contract-input
//!            rejection across every contract-taking subcommand, `--version`
//!            parseability.
//!   NOT COVERED: `pv lint` warning ratchet, `pv diff` semver suggestions,
//!            `pv score`, `pv certify`, `pv coverage`, and the generator
//!            subcommands' output *content* (kani/probar/coq/flux/tla/...).
//!
//! DISCRIMINATION. `gate_self_test` at the bottom is the control on the control:
//! it proves the rule table cannot be satisfied by a `pv` that prints every rule
//! id unconditionally, nor by one that prints none.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

/// The `pv` binary built from THIS tree.
///
/// `CARGO_BIN_EXE_pv` is set by cargo for the crate's own `[[bin]]`, so this
/// cannot silently resolve a stale `pv` off `$PATH` — the exact defect #2552
/// was filed for (PATH pv was 0.49.0 while in-tree was 0.63.0, and the two
/// disagreed on the release-deciding gate).
fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn pv(args: &[&str]) -> (i32, String) {
    let out = Command::new(pv_bin())
        .args(args)
        .output()
        .expect("failed to spawn pv");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

// ============================================================================
// Contract fixture builder
// ============================================================================

/// A minimal contract that `pv validate` accepts with 0 errors and 0 warnings.
///
/// Every case below is this contract with exactly ONE part swapped, so a
/// reported rule is attributable to that swap and to nothing else.
struct Fixture {
    metadata: String,
    equations: String,
    obligations: String,
    ftests: String,
    harnesses: String,
    qa_gate: String,
}

const METADATA_OK: &str = "metadata:\n  version: \"1.0.0\"\n  kind: kernel\n  \
     description: \"pv surface gate fixture\"\n  references:\n    - \"#2589\"\n";
const EQUATIONS_OK: &str = "equations:\n  probe_eq:\n    formula: \"y = x\"\n";
const OBLIGATION_OK: &str =
    "  - type: invariant\n    property: \"P\"\n    formal: \"F-OK\"\n    applies_to: all\n";
const FTEST_OK: &str = "  - id: FALSIFY-PVGATE-001\n    prediction: \"p\"\n    if_fails: \"f\"\n";
const HARNESS_OK: &str =
    "  - id: KANI-PVGATE-001\n    obligation: OB-1\n    property: \"P\"\n    bound: 8\n";
const QA_GATE_OK: &str = "qa_gate:\n  id: F-PVGATE-001\n  name: \"pv gate\"\n  \
     checks:\n    - \"c\"\n  pass_criteria: \"all\"\n";

impl Default for Fixture {
    fn default() -> Self {
        Self {
            metadata: METADATA_OK.to_string(),
            equations: EQUATIONS_OK.to_string(),
            obligations: OBLIGATION_OK.to_string(),
            ftests: FTEST_OK.to_string(),
            harnesses: HARNESS_OK.to_string(),
            qa_gate: QA_GATE_OK.to_string(),
        }
    }
}

impl Fixture {
    fn render(&self) -> String {
        format!(
            "{}\n{}\nproof_obligations:\n{}\nfalsification_tests:\n{}\n\
             kani_harnesses:\n{}\n{}",
            self.metadata,
            self.equations,
            self.obligations,
            self.ftests,
            self.harnesses,
            self.qa_gate,
        )
    }

    /// Write to a temp dir and run `pv validate` on it.
    fn validate(&self) -> (i32, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fixture-v1.yaml");
        std::fs::write(&path, self.render()).expect("write fixture");
        pv(&["validate", path.to_str().expect("utf8 path")])
    }
}

/// Severity a rule is expected to be reported at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sev {
    Error,
    Warn,
}

impl Sev {
    fn tag(self) -> &'static str {
        match self {
            Sev::Error => "[ERROR]",
            Sev::Warn => "[WARN]",
        }
    }
    /// `pv validate` exits nonzero iff at least one ERROR was reported.
    fn expected_rc(self) -> i32 {
        match self {
            Sev::Error => 1,
            Sev::Warn => 0,
        }
    }
}

/// One row of the decision table: a rule id, the severity it must be reported
/// at, and the single-field mutation that must trip it.
struct Case {
    rule: &'static str,
    sev: Sev,
    build: fn(&mut Fixture),
}

/// THE DECISION TABLE.
///
/// Every diagnostic `pv validate` can emit appears exactly once. Each row is a
/// falsifier: it asserts the CLI *reaches* that rule. The library unit tests in
/// `aprender-contracts` call `validate_contract()` directly and stay green even
/// if the CLI stops printing violations entirely — these rows do not.
///
/// Severities and exit codes here were MEASURED against the binary at
/// 4bbfeb07f, not assumed. `SCHEMA-006` is deliberately absent from the trip
/// table and covered separately: tripping it needs a second obligation, which
/// also trips `PROVABILITY-001`, so it cannot be attributed to one swap.
const CASES: &[Case] = &[
    Case {
        rule: "SCHEMA-001",
        sev: Sev::Error,
        build: |f| {
            f.metadata = "metadata:\n  version: \"1.0.0\"\n  kind: kernel\n  \
                 description: \"d\"\n  references: []\n"
                .to_string();
        },
    },
    Case {
        rule: "SCHEMA-002",
        sev: Sev::Error,
        build: |f| {
            f.metadata = "metadata:\n  version: \"\"\n  kind: kernel\n  \
                 description: \"d\"\n  references:\n    - \"r\"\n"
                .to_string();
        },
    },
    Case {
        rule: "SCHEMA-003",
        sev: Sev::Error,
        build: |f| f.equations = "equations: {}\n".to_string(),
    },
    Case {
        rule: "SCHEMA-004",
        sev: Sev::Error,
        build: |f| {
            f.equations = "equations:\n  probe_eq:\n    formula: \"\"\n".to_string();
        },
    },
    Case {
        rule: "SCHEMA-005",
        sev: Sev::Error,
        build: |f| {
            f.obligations = "  - type: invariant\n    property: \"\"\n    \
                 formal: \"F-OK\"\n    applies_to: all\n"
                .to_string();
        },
    },
    Case {
        rule: "SCHEMA-007",
        sev: Sev::Error,
        build: |f| {
            // Same id twice — and a second falsification test keeps the
            // ftest/obligation ratio satisfied, so PROVABILITY-001 stays quiet.
            f.ftests = format!(
                "{FTEST_OK}  - id: FALSIFY-PVGATE-001\n    prediction: \"q\"\n    \
                 if_fails: \"f\"\n"
            );
        },
    },
    Case {
        rule: "SCHEMA-008",
        sev: Sev::Error,
        build: |f| {
            f.ftests = "  - id: FALSIFY-PVGATE-001\n    prediction: \"\"\n    \
                 if_fails: \"f\"\n"
                .to_string();
        },
    },
    Case {
        rule: "SCHEMA-009",
        sev: Sev::Warn,
        build: |f| {
            f.ftests = "  - id: FALSIFY-PVGATE-001\n    prediction: \"p\"\n    \
                 if_fails: \"\"\n"
                .to_string();
        },
    },
    Case {
        rule: "SCHEMA-010",
        sev: Sev::Error,
        build: |f| {
            f.harnesses = format!(
                "{HARNESS_OK}  - id: KANI-PVGATE-001\n    obligation: OB-2\n    \
                 property: \"Q\"\n    bound: 8\n"
            );
        },
    },
    Case {
        rule: "SCHEMA-011",
        sev: Sev::Error,
        build: |f| {
            f.harnesses = "  - id: KANI-PVGATE-001\n    obligation: \"\"\n    \
                 property: \"P\"\n    bound: 8\n"
                .to_string();
        },
    },
    Case {
        rule: "SCHEMA-012",
        sev: Sev::Warn,
        build: |f| {
            f.harnesses =
                "  - id: KANI-PVGATE-001\n    obligation: OB-1\n    property: \"P\"\n".to_string();
        },
    },
    Case {
        rule: "SCHEMA-013",
        sev: Sev::Warn,
        build: |f| f.qa_gate = String::new(),
    },
    Case {
        rule: "SCHEMA-014",
        sev: Sev::Error,
        build: |f| {
            f.obligations = format!("{OBLIGATION_OK}    requires: \"x > 0\"\n");
        },
    },
    Case {
        rule: "SCHEMA-015",
        sev: Sev::Error,
        build: |f| {
            f.obligations = format!("{OBLIGATION_OK}    applies_to_phase: \"phase1\"\n");
        },
    },
    Case {
        rule: "SCHEMA-016",
        sev: Sev::Error,
        build: |f| {
            f.obligations = format!("{OBLIGATION_OK}    parent_contract: \"other-v1\"\n");
        },
    },
    Case {
        rule: "SCHEMA-017",
        sev: Sev::Error,
        build: |f| {
            f.obligations = "  - type: subcontract\n    property: \"P\"\n    \
                 formal: \"F-OK\"\n    applies_to: all\n    \
                 parent_contract: \"not-listed-v1\"\n"
                .to_string();
        },
    },
    Case {
        rule: "PROVABILITY-001",
        sev: Sev::Error,
        build: |f| {
            // Two obligations, one falsification test: unfalsifiable claim.
            f.obligations = format!(
                "{OBLIGATION_OK}  - type: bound\n    property: \"Q\"\n    \
                 formal: \"F-2\"\n    applies_to: all\n"
            );
        },
    },
];

// ---------------------------------------------------------------------------
// The BEAT family — reachable only for `kind: beat-benchmark`
// ---------------------------------------------------------------------------

/// A `beat-benchmark` contract whose `beat:` block is complete and valid.
///
/// The BEAT rules were nearly missed by this gate: `BEAT-002..007` are emitted
/// through a local closure rather than a `rule: "..."` literal, so a scan for
/// that literal saw only `BEAT-001`. The rule-id extractor below therefore takes
/// its universe from every rule-id-SHAPED string literal in the validator, not
/// from one syntactic form.
const BEAT_BLOCK_OK: &str = "beat:\n  pillar: 1\n  incumbent: scikit-learn\n  \
     metric: wall_clock_ratio\n  direction: lower_is_better\n  \
     beat_threshold: 0.9\n  approved_compute: CPU\n  \
     ci_gate_name: \"beat_pv_surface_gate\"\n";

/// Build a beat-benchmark contract body with `beat:` replaced wholesale.
fn beat_fixture(beat_block: &str) -> String {
    format!(
        "metadata:\n  kind: beat-benchmark\n  version: \"1.0.0\"\n  \
         description: \"pv surface gate beat fixture\"\n  references:\n    - \"#2589\"\n\
         \n{EQUATIONS_OK}\n{beat_block}"
    )
}

fn validate_beat(beat_block: &str) -> (i32, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("beat-fixture-v1.yaml");
    std::fs::write(&path, beat_fixture(beat_block)).expect("write beat fixture");
    pv(&["validate", path.to_str().expect("utf8 path")])
}

/// One row of the BEAT decision table.
struct BeatCase {
    rule: &'static str,
    /// The `beat:` block that must trip `rule`.
    block: &'static str,
}

/// THE BEAT DECISION TABLE. Every row omits or corrupts exactly one field.
///
/// `BEAT-002` gets two rows because the rule has two arms — empty and
/// not-an-incumbent — and only the second exercises the `BEAT_INCUMBENTS`
/// membership list.
const BEAT_CASES: &[BeatCase] = &[
    BeatCase {
        rule: "BEAT-001",
        // No `beat:` block at all.
        block: "",
    },
    BeatCase {
        rule: "BEAT-002",
        block: "beat:\n  pillar: 1\n  incumbent: \"\"\n  metric: m\n  \
                direction: lower_is_better\n  beat_threshold: 0.9\n  \
                approved_compute: CPU\n  ci_gate_name: \"g\"\n",
    },
    BeatCase {
        rule: "BEAT-002",
        // Membership arm: a plausible-looking but unlisted incumbent.
        block: "beat:\n  pillar: 1\n  incumbent: THIS-INCUMBENT-DOES-NOT-EXIST\n  \
                metric: m\n  direction: lower_is_better\n  beat_threshold: 0.9\n  \
                approved_compute: CPU\n  ci_gate_name: \"g\"\n",
    },
    BeatCase {
        rule: "BEAT-003",
        block: "beat:\n  pillar: 1\n  incumbent: scikit-learn\n  metric: \"  \"\n  \
                direction: lower_is_better\n  beat_threshold: 0.9\n  \
                approved_compute: CPU\n  ci_gate_name: \"g\"\n",
    },
    BeatCase {
        rule: "BEAT-004",
        block: "beat:\n  pillar: 1\n  incumbent: scikit-learn\n  metric: m\n  \
                direction: sideways_is_better\n  beat_threshold: 0.9\n  \
                approved_compute: CPU\n  ci_gate_name: \"g\"\n",
    },
    BeatCase {
        rule: "BEAT-005",
        block: "beat:\n  pillar: 1\n  incumbent: scikit-learn\n  metric: m\n  \
                direction: lower_is_better\n  approved_compute: CPU\n  \
                ci_gate_name: \"g\"\n",
    },
    BeatCase {
        rule: "BEAT-006",
        block: "beat:\n  pillar: 1\n  incumbent: scikit-learn\n  metric: m\n  \
                direction: lower_is_better\n  beat_threshold: 0.9\n  \
                approved_compute: CPU\n  ci_gate_name: \"\"\n",
    },
    BeatCase {
        rule: "BEAT-007",
        block: "beat:\n  pillar: 1\n  incumbent: scikit-learn\n  metric: m\n  \
                direction: lower_is_better\n  beat_threshold: 0.9\n  \
                approved_compute: TPU\n  ci_gate_name: \"g\"\n",
    },
];

/// Rule ids the tables do not trip directly but which must still never appear
/// on a clean baseline.
///
/// `SCHEMA-006` (duplicate `formal:` predicate) needs a second proof obligation,
/// which unavoidably also trips `PROVABILITY-001` — so it cannot be attributed
/// to a single swap and is asserted absent-from-baseline only.
const ALSO_ABSENT_FROM_BASELINE: &[&str] = &["SCHEMA-006"];

fn all_rule_ids() -> BTreeSet<&'static str> {
    CASES
        .iter()
        .map(|c| c.rule)
        .chain(BEAT_CASES.iter().map(|c| c.rule))
        .chain(ALSO_ABSENT_FROM_BASELINE.iter().copied())
        .collect()
}

// ============================================================================
// A. `pv validate` decision table — the priority-1 target of #2589
// ============================================================================

/// NEGATIVE DIRECTION: a contract with nothing wrong must produce NO rule id
/// and exit 0.
///
/// Without this, a `pv` that printed all 18 rules unconditionally would satisfy
/// every trip case below. This is the half that makes the table discriminating.
#[test]
fn validate_baseline_is_silent_and_exits_zero() {
    let (rc, out) = Fixture::default().validate();
    assert_eq!(
        rc, 0,
        "clean fixture must exit 0.\n--- pv output ---\n{out}"
    );
    assert!(
        out.contains("0 error(s), 0 warning(s)"),
        "clean fixture must report zero of both.\n--- pv output ---\n{out}"
    );
    for rule in all_rule_ids() {
        assert!(
            !out.contains(rule),
            "clean fixture tripped {rule} — either the fixture is not clean or \
             pv reports rules unconditionally.\n--- pv output ---\n{out}"
        );
    }
}

/// POSITIVE DIRECTION: every rule is reachable THROUGH THE CLI, at the right
/// severity, with the exit code that severity implies.
///
/// The exit-code half is what excludes `fn main() { exit(0) }`; the rule-id half
/// is what excludes a `pv` that swallows the violation list.
#[test]
fn validate_reaches_every_rule_at_the_right_severity() {
    let mut failures = Vec::new();
    for case in CASES {
        let mut fixture = Fixture::default();
        (case.build)(&mut fixture);
        let (rc, out) = fixture.validate();

        let expected_line = format!("{} {}:", case.sev.tag(), case.rule);
        if !out.contains(&expected_line) {
            failures.push(format!(
                "{}: expected a `{}` line, got:\n{}",
                case.rule,
                expected_line.trim_end_matches(':'),
                indent(&out)
            ));
            continue;
        }
        if rc != case.sev.expected_rc() {
            failures.push(format!(
                "{}: reported at {:?} so pv must exit {}, got rc={rc}:\n{}",
                case.rule,
                case.sev,
                case.sev.expected_rc(),
                indent(&out)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} validate rules are not reachable through the CLI:\n\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n")
    );
}

/// Every rule-id-SHAPED string literal in the validator source.
///
/// The universe is deliberately taken from the SHAPE (`FAMILY-NNN`), not from
/// one syntactic form. An earlier version of this extractor matched only
/// `rule: "..."` and consequently saw `BEAT-001` but none of `BEAT-002..007`,
/// which are emitted through a local closure — the exact "guard's universe
/// built from the wrong side" failure this repo has hit before. A new rule must
/// ADD to the loop, never quietly fall outside it.
fn declared_rule_ids() -> BTreeSet<String> {
    let src = include_str!("../../aprender-contracts/src/schema/validator.rs");
    let mut ids = BTreeSet::new();
    for chunk in src.split('"').skip(1).step_by(2) {
        if is_rule_id_shaped(chunk) {
            ids.insert(chunk.to_string());
        }
    }
    ids
}

/// `FAMILY-NNN` where FAMILY is upper-case ASCII (possibly hyphenated) and NNN
/// is exactly three digits.
fn is_rule_id_shaped(s: &str) -> bool {
    let Some((family, num)) = s.rsplit_once('-') else {
        return false;
    };
    num.len() == 3
        && num.chars().all(|c| c.is_ascii_digit())
        && !family.is_empty()
        && family
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-')
}

/// The BEAT family, reachable only through `kind: beat-benchmark`.
///
/// This half of the table nearly did not exist: the first extractor missed
/// `BEAT-002..007` entirely. It is also where the brief's `BEAT_INCUMBENTS`
/// question lives — the `BEAT-002` membership row below is what makes that list
/// a load-bearing, falsifiable check rather than dead code.
#[test]
fn validate_reaches_every_beat_rule() {
    let (rc, out) = validate_beat(BEAT_BLOCK_OK);
    assert_eq!(
        rc,
        0,
        "the clean beat-benchmark fixture must validate.\n{}",
        indent(&out)
    );
    for id in all_rule_ids() {
        assert!(
            !out.contains(id),
            "clean beat fixture tripped {id}:\n{}",
            indent(&out)
        );
    }

    let mut failures = Vec::new();
    for case in BEAT_CASES {
        let (rc, out) = validate_beat(case.block);
        let expected = format!("[ERROR] {}:", case.rule);
        if !out.contains(&expected) {
            failures.push(format!(
                "{}: expected `{}` line, got:\n{}",
                case.rule,
                expected.trim_end_matches(':'),
                indent(&out)
            ));
        } else if rc == 0 {
            failures.push(format!(
                "{}: reported as an ERROR but pv still exited 0:\n{}",
                case.rule,
                indent(&out)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} BEAT rules are not reachable through the CLI:\n\n{}",
        failures.len(),
        BEAT_CASES.len(),
        failures.join("\n")
    );
}

/// The table must stay exhaustive as rules are added.
///
/// Without this, adding `SCHEMA-018` to the validator leaves it ungated forever
/// and every test here still passes — the "guard's universe built from the wrong
/// side" failure. The universe is taken from the VALIDATOR SOURCE, not from the
/// table, so a new rule removes nothing from the loop.
#[test]
fn every_rule_in_the_validator_source_appears_in_the_table() {
    let declared = declared_rule_ids();
    assert!(
        declared.len() >= 25,
        "parsed only {} rule ids out of validator.rs — the extractor broke, \
         which would make this guard vacuously green. Found: {declared:?}",
        declared.len()
    );

    let covered = all_rule_ids();
    let missing: Vec<_> = declared
        .iter()
        .filter(|d| !covered.contains(d.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "validator.rs declares rules with no row in the pv-surface decision \
         table: {missing:?}. Add a Case (or ALSO_ABSENT_FROM_BASELINE) so the \
         new rule is gated through the CLI, not only in library unit tests."
    );
}

// ============================================================================
// B. Whole-surface invariants, enumerated AT RUNTIME from the binary
// ============================================================================

/// Read the subcommand list out of `pv --help`.
///
/// Enumerating at runtime rather than hardcoding a list is deliberate: a
/// hardcoded list silently stops covering commands added after it was written.
fn advertised_subcommands() -> Vec<String> {
    let (rc, out) = pv(&["--help"]);
    assert_eq!(rc, 0, "`pv --help` must exit 0, got {rc}:\n{out}");
    let cmds: Vec<String> = commands_section(&out)
        .filter_map(subcommand_name)
        .map(str::to_string)
        .collect();
    assert!(
        cmds.len() >= 30,
        "parsed only {} subcommands out of `pv --help` — the parser broke and \
         every test built on it would be vacuously green:\n{out}",
        cmds.len()
    );
    cmds
}

/// The lines of `--help` between the `Commands:` and `Options:` headings.
fn commands_section(help: &str) -> impl Iterator<Item = &str> {
    help.lines()
        .skip_while(|l| !l.starts_with("Commands:"))
        .skip(1)
        .take_while(|l| !l.starts_with("Options:"))
}

/// The subcommand name on one clap listing line, if it is one.
///
/// clap prints `"  <name>   <description>"`; wrapped description lines are
/// indented further, so only exactly-two-space indentation counts. `help` is
/// excluded — it is clap's own, not part of pv's surface.
fn subcommand_name(line: &str) -> Option<&str> {
    if !line.starts_with("  ") || line.starts_with("   ") {
        return None;
    }
    let word = line.split_whitespace().next()?;
    let plausible = word
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    (word != "help" && plausible).then_some(word)
}

/// Every subcommand `pv --help` ADVERTISES must actually be reachable.
///
/// This is the `apr test llm` defect class (#2527): a command listed in `--help`
/// but permanently unreachable because its feature was never declared, so its
/// own remedy was impossible to run. Advertising is a promise; this checks it.
#[test]
fn every_advertised_subcommand_is_reachable() {
    let mut unreachable = Vec::new();
    for cmd in advertised_subcommands() {
        let (rc, out) = pv(&[&cmd, "--help"]);
        if rc != 0 {
            unreachable.push(format!("pv {cmd} --help -> rc={rc}\n{}", indent(&out)));
        }
    }
    assert!(
        unreachable.is_empty(),
        "{} advertised subcommand(s) are not reachable:\n{}",
        unreachable.len(),
        unreachable.join("\n")
    );
}

/// Every subcommand that takes a contract path must REJECT bad input.
///
/// This is the assertion the pre-existing suite never made. `pv validate <good>
/// == 0` is satisfied by a binary that does nothing; `pv <cmd> <garbage> != 0`
/// is not. Applied across every contract-taking subcommand at once, discovered
/// by parsing each command's own usage line.
#[test]
fn contract_taking_subcommands_reject_unusable_input() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("does-not-exist.yaml");
    let garbage = dir.path().join("garbage.yaml");
    std::fs::write(&garbage, "{{{ not yaml at all: [\n").expect("write");
    let empty = dir.path().join("empty.yaml");
    std::fs::write(&empty, "").expect("write");
    let wrong_shape = dir.path().join("wrong-shape.yaml");
    std::fs::write(&wrong_shape, "hello: world\nlist:\n  - 1\n").expect("write");

    let inputs = [
        ("nonexistent", &missing),
        ("malformed-yaml", &garbage),
        ("empty", &empty),
        ("well-formed-but-not-a-contract", &wrong_shape),
    ];

    let mut checked = 0usize;
    let mut accepted = Vec::new();
    for cmd in advertised_subcommands() {
        let (_, help) = pv(&[&cmd, "--help"]);
        let Some(usage) = help.lines().find(|l| l.starts_with("Usage:")) else {
            continue;
        };
        if !(usage.contains("<CONTRACT>") || usage.contains("<PATH>") || usage.contains("<FILE>")) {
            continue;
        }
        checked += 1;
        for (label, path) in &inputs {
            let (rc, out) = pv(&[&cmd, path.to_str().expect("utf8 path")]);
            if rc == 0 {
                accepted.push(format!(
                    "pv {cmd} <{label}> exited 0 — it accepted unusable input:\n{}",
                    indent(&out)
                ));
            }
        }
    }
    assert!(
        checked >= 10,
        "only {checked} contract-taking subcommands were found; the usage-line \
         parser probably broke, which would make this guard vacuously green"
    );
    assert!(
        accepted.is_empty(),
        "{} subcommand/input pair(s) accepted input they cannot possibly \
         process:\n{}",
        accepted.len(),
        accepted.join("\n")
    );
}

/// `pv --version` must report the version of THIS crate.
///
/// #2552: a `pv` on `$PATH` was 0.49.0 while the in-tree crate was 0.63.0, and
/// the two disagreed on the release-deciding gate — with the stale one cited in
/// the release receipt as evidence of correctness. This is a substring check on
/// the semver only, so the separate work on `--version` IDENTITY (#2559 — making
/// it distinguishable from the `pv(1)` pipe viewer) is free to change the
/// surrounding text without touching this test.
#[test]
fn version_reports_this_crates_version() {
    let (rc, out) = pv(&["--version"]);
    assert_eq!(rc, 0, "`pv --version` must exit 0, got {rc}:\n{out}");
    assert!(
        out.contains(env!("CARGO_PKG_VERSION")),
        "`pv --version` printed {out:?} which does not contain this crate's \
         version {:?}",
        env!("CARGO_PKG_VERSION")
    );
}

// ============================================================================
// C. The control on the control
// ============================================================================

/// Prove the decision table can distinguish the two degenerate `pv`s.
///
/// A rule table is only a control if BOTH degenerate binaries fail it:
///   * a `pv` that prints NO rule ids   -> the trip cases must fail
///   * a `pv` that prints ALL rule ids  -> the baseline case must fail
///
/// The real binary must satisfy both halves simultaneously, i.e. the observed
/// rule set must depend on the input. This test asserts exactly that dependency
/// rather than re-reading the assertions above.
#[test]
fn gate_self_test_rule_output_depends_on_input() {
    let (_, baseline) = Fixture::default().validate();
    let baseline_rules: BTreeSet<&str> = all_rule_ids()
        .into_iter()
        .filter(|r| baseline.contains(r))
        .collect();
    assert!(
        baseline_rules.is_empty(),
        "a `pv` that prints rules unconditionally would pass every trip case; \
         the baseline must print none, saw {baseline_rules:?}"
    );

    let mut tripped: BTreeSet<&str> = BTreeSet::new();
    for case in CASES {
        let mut fixture = Fixture::default();
        (case.build)(&mut fixture);
        let (_, out) = fixture.validate();
        if out.contains(case.rule) {
            tripped.insert(case.rule);
        }
    }
    assert_eq!(
        tripped.len(),
        CASES.len(),
        "a `pv` that prints no rules would pass the baseline case; {} of {} \
         trip cases produced no rule id. Reached: {tripped:?}",
        CASES.len() - tripped.len(),
        CASES.len()
    );
}

fn indent(s: &str) -> String {
    s.lines()
        .map(|l| format!("    {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}
