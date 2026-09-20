//! #3610 — an armed shape that graded ZERO focus nodes must not be counted as clean.
//!
//! **The defect, measured on pv 0.68.2.** `NoFocus` asked the question GLOBALLY —
//! `report.focus_nodes_n == 0` — which is only ever true for a lone contract. In a **directory**
//! one empty shape is invisible: the other shapes carry the total above zero, the empty one
//! contributes no violations, and the aggregate reports **`Pass` while listing it in
//! `armed_shapes`**. A `type: jsonl` contract whose rows violate its own `sh:pattern` passed exactly
//! that way, and `armed_shapes` is the tool's claim about what it measured.
//!
//! `by_shape` had carried `<shape>=0` all along. **Nothing read it** — the same *computed but
//! unexposed* shape as #3597, this time inside the gate itself.
//!
//! **BOTH VENUES, because the lone-contract case already declined correctly and a fix proved only
//! there proves nothing.** The directory case is the one that lied.
//!
//! | fixture | shape reach | armed? | expected |
//! |---|---|---|---|
//! | `shapes-one-empty` | `empty-shape=0`, `tool-status=1` | yes | **exit 2**, decline naming `empty-shape` |
//! | `shapes-one-empty-unarmed` | same | no | **Pass**, `declines: ["empty-shape"]` |
//! | `json-ok` | `tool-status=1` | yes | Pass, `declines: []` |
//!
//! The second row is what keeps the fix from being "refuse whenever anything is empty": an UNARMED
//! shape at zero did not affect the verdict, so it is **reported** rather than refused — and it is
//! reported, not swallowed, which is the half the original defect got wrong.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn all(&self) -> String {
        format!(
            "exit {}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }
    fn extra(&self) -> serde_json::Value {
        serde_json::from_str::<serde_json::Value>(&self.stdout)
            .map(|v| v["extra"].clone())
            .unwrap_or(serde_json::Value::Null)
    }
}

fn shapes_on(fixture: &str) -> Run {
    let contracts = repo_root()
        .join("tests/fixtures/ont")
        .join(fixture)
        .join("contracts");
    let out = Command::new(pv_bin())
        .args([
            "lint",
            contracts.to_str().expect("utf-8 path"),
            "--gate",
            "shapes",
            "--format",
            "json",
        ])
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn an_armed_shape_that_graded_nothing_refuses_in_a_directory() {
    // THE CASE THAT LIED. Before #3610 this was `Pass` with `empty-shape` in `armed_shapes`.
    let r = shapes_on("shapes-one-empty");
    assert_eq!(
        r.code,
        2,
        "a vacuous armed shape must DECLINE, not pass and not fail\n{}",
        r.all()
    );
    // THE REFUSAL IS THE EXIT CODE; THE REPORT IS THE EVIDENCE. Downstream consumers (infra's SLK
    // gate) capture stdout and parse it REGARDLESS of exit code, because pv already exits non-zero
    // on Fail. A refusal that printed only a bare `decline:` line would read to them as "no
    // by_shape" and score UNMEASURED — indistinguishable, from their side, from a broken pv.
    let extra = r.extra();
    assert!(
        !extra.is_null(),
        "stdout must still parse as JSON on the exit-2 path\n{}",
        r.all()
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&r.stdout).expect("json")["verdict"].as_str(),
        Some("Unknown(NoFocus)"),
        "the report must carry the declining verdict\n{}",
        r.all()
    );
    assert!(
        extra["by_shape"]
            .as_array()
            .expect("by_shape")
            .iter()
            .any(|s| s.as_str() == Some("empty-shape=0")),
        "by_shape must NAME the zero-focus shape on the exit-2 path — it is the evidence\n{}",
        r.all()
    );
    assert_eq!(
        extra["declines"][0].as_str(),
        Some("empty-shape"),
        "the refusal must name the shape in declines\n{}",
        r.all()
    );
    assert!(
        !extra["armed_shapes"]
            .as_array()
            .expect("armed_shapes")
            .iter()
            .any(|s| s.as_str() == Some("empty-shape")),
        "a shape that graded nothing has no place in armed_shapes\n{}",
        r.all()
    );
    assert!(
        !extra["not_armed_shapes"]
            .as_array()
            .expect("not_armed_shapes")
            .iter()
            .any(|s| s.as_str() == Some("empty-shape")),
        "nor in not_armed_shapes: filing a vacuity as a policy choice is how it hid\n{}",
        r.all()
    );
}

#[test]
fn an_unarmed_shape_that_graded_nothing_is_reported_not_refused() {
    // The other direction, and the reason this is not "refuse whenever anything is empty".
    let r = shapes_on("shapes-one-empty-unarmed");
    assert_eq!(
        r.code,
        0,
        "an UNARMED empty shape must not refuse\n{}",
        r.all()
    );
    let extra = r.extra();
    assert_eq!(
        extra["declines"].as_array().map(|a| a.len()),
        Some(1),
        "the empty shape must be REPORTED in declines, not swallowed\n{}",
        r.all()
    );
    assert_eq!(
        extra["declines"][0].as_str(),
        Some("empty-shape"),
        "{}",
        r.all()
    );
    assert!(
        !extra["armed_shapes"]
            .as_array()
            .expect("armed_shapes")
            .iter()
            .any(|s| s.as_str() == Some("empty-shape")),
        "a shape that graded nothing has no place in armed_shapes\n{}",
        r.all()
    );
    // THE HALF THIS TEST WAS MISSING. The criterion is "neither list", and only
    // the armed sibling above checked both — so an UNARMED vacuity could sit in
    // `not_armed_shapes`, filed as a deliberate policy choice, which is exactly
    // how the original defect hid. It did: the partition filtered on
    // `vacuous_armed`, and two quorum lanes found it independently while this
    // test stayed green.
    assert!(
        !extra["not_armed_shapes"]
            .as_array()
            .expect("not_armed_shapes")
            .iter()
            .any(|s| s.as_str() == Some("empty-shape")),
        "an UNARMED shape that graded nothing must not be filed under \
         not_armed_shapes either — `declines` is the only list it belongs in\n{}",
        r.all()
    );
}

#[test]
fn a_corpus_whose_shapes_all_graded_something_still_passes_with_an_empty_declines() {
    // The control for the control: without this, a build that declined EVERYTHING would pass the
    // two cases above.
    let r = shapes_on("json-ok");
    assert_eq!(r.code, 0, "{}", r.all());
    assert_eq!(
        r.extra()["declines"].as_array().map(Vec::len),
        Some(0),
        "nothing graded zero here, so declines must be empty\n{}",
        r.all()
    );
}

#[test]
fn the_lone_contract_venue_still_declines_as_it_always_did() {
    // The fix must not regress the venue that was already right — and proving it only there is
    // what would have proved nothing, since that venue never lied.
    let r = shapes_on("json-missing-ref");
    assert_ne!(
        r.code,
        0,
        "a lone contract that cannot be extracted must not pass\n{}",
        r.all()
    );
}

#[test]
fn the_three_answers_remain_distinct() {
    // pass / fail / decline must not collapse: a build that returned one code for everything
    // would satisfy any single assertion above.
    let pass = shapes_on("json-ok").code;
    let fail = shapes_on("json-violation").code;
    let decline = shapes_on("shapes-one-empty").code;
    assert_eq!(
        (pass, fail, decline),
        (0, 1, 2),
        "pass / fail / decline must differ"
    );
    // Every one of the three must also PRINT its report — a consumer that captures stdout and
    // parses it regardless of exit code must get a document in all three cases, not two.
    for fixture in ["json-ok", "json-violation", "shapes-one-empty"] {
        let r = shapes_on(fixture);
        assert!(
            !r.extra().is_null(),
            "{fixture} did not emit a parseable report\n{}",
            r.all()
        );
        assert!(
            r.extra()["by_shape"].is_array(),
            "{fixture}: extra.by_shape must be present whatever the exit code\n{}",
            r.all()
        );
    }
}
