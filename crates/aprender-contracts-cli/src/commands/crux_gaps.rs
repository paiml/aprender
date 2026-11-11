//! `pv crux gaps` — the per-story CRUX verdict, derived from ONE input: the
//! `LIVE-PENDING` markers in each story contract's falsification tests
//! (PMAT-6734, slice 1 of epic #6728).
//!
//! WHY THIS EXISTS. Every `contracts/crux-<CAT>-<NN>-v1.yaml` STATES an
//! `intake_status` (`missing` / `partial` / `supported`) in its metadata. A
//! stated status is a claim a person typed; nothing re-derived it. A story whose
//! falsification tests are all `LIVE-PENDING` placeholders is `missing` no matter
//! what its metadata says. This verb derives the verdict from the tests and sets
//! it beside the stated one, so a disagreement is a row, not a surprise.
//!
//! WHAT IT DOES NOT MEASURE. Surface resolution (does the named command or API
//! exist) and bound-test reachability (does the test the contract names run and
//! pass) are sibling tickets. Without them nothing here proves a test PASSES, so
//! the verb can never print `supported`: a story with no `LIVE-PENDING` test is
//! `not_measured`, and its reason says so.
//!
//! FAIL CLOSED. A file named like a story that does not parse, or has no
//! `metadata.id`, is an error listed by path and the verb exits non-zero. Zero
//! matching files is [`ZeroContracts`] (exit 2). A gate that measures nothing is
//! not a gate.

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::contract_walk::ZeroContracts;
use crate::json_obj::obj;

/// A falsification test whose `test:` text starts with this (after leading
/// whitespace) is a placeholder: the capability does not exist yet.
pub const PENDING_MARKER: &str = "LIVE-PENDING";

/// The verdict derived from the falsification tests. There is deliberately no
/// `Supported`: this slice proves no test passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Every falsification test is pending, or there are none.
    Missing,
    /// Some tests are pending and some are not.
    Partial,
    /// No pending test; nothing here measures whether the real ones pass.
    NotMeasured,
}

impl Verdict {
    /// The lower-case word printed in the table and the JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Partial => "partial",
            Self::NotMeasured => "not_measured",
        }
    }
}

/// One story's row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Story {
    pub id: String,
    pub verdict: Verdict,
    pub reason: String,
    pub competitor: Option<String>,
    pub demand_score: Option<u64>,
    pub github_issue: Option<u64>,
    /// What the metadata STATES, verbatim.
    pub intake_status: Option<String>,
    /// The stated `intake_status` is present and differs from the derived verdict.
    pub disagrees: bool,
}

/// Story files that cannot be used: named like a story, but unreadable, not
/// parseable, or without a valid `metadata.id`. Exit 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoryErrors {
    /// The directory that was asked for.
    pub path: PathBuf,
    /// `(file, why)` per unusable file, in name order.
    pub errors: Vec<(PathBuf, String)>,
}

impl fmt::Display for StoryErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} CRUX story file(s) under {} unusable:",
            self.errors.len(),
            self.path.display()
        )?;
        for (file, why) in &self.errors {
            write!(f, "\n  {}: {why}", file.display())?;
        }
        Ok(())
    }
}

impl std::error::Error for StoryErrors {}

/// Per-verdict counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
    pub missing: usize,
    pub partial: usize,
    pub not_measured: usize,
}

/// Is this falsification test text a `LIVE-PENDING` placeholder? Leading
/// whitespace is ignored; the marker in the middle of the text does not count.
#[must_use]
pub fn is_pending(test: &str) -> bool {
    let _ = test; // RED stub: not implemented yet
    false
}

/// The verdict and its reason from the pending and total test counts. Pure.
#[must_use]
pub fn derive(pending: usize, total: usize) -> (Verdict, String) {
    let _ = (pending, total); // RED stub: not implemented yet
    (Verdict::NotMeasured, String::new())
}

/// Does the stated `intake_status` disagree with the derived verdict? Absent is
/// no claim. `missing` and `partial` compare literally. A stated `supported`
/// against `not_measured` is not a disagreement (nothing here can refute it),
/// against `missing` or `partial` it is.
#[must_use]
pub fn disagrees(stated: Option<&str>, derived: Verdict) -> bool {
    let _ = (stated, derived); // RED stub: not implemented yet
    false
}

fn is_cat(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_uppercase())
}

fn is_num(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Is this file name `crux-<CAT>-<NN>-v1.yaml`?
#[must_use]
pub fn name_matches(name: &str) -> bool {
    name.strip_prefix("crux-")
        .and_then(|rest| rest.strip_suffix("-v1.yaml"))
        .and_then(|mid| mid.split_once('-'))
        .is_some_and(|(cat, nn)| is_cat(cat) && is_num(nn))
}

/// Is this `metadata.id` `CRUX-<CAT>-<NN>`?
#[must_use]
pub fn id_matches(id: &str) -> bool {
    id.strip_prefix("CRUX-")
        .and_then(|rest| rest.split_once('-'))
        .is_some_and(|(cat, nn)| is_cat(cat) && is_num(nn))
}

fn opt_field<T>(
    meta: &Value,
    key: &str,
    conv: fn(&Value) -> Option<T>,
    kind: &str,
) -> Result<Option<T>, String> {
    match meta.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => conv(v)
            .map(Some)
            .ok_or_else(|| format!("metadata.{key} is not {kind}")),
    }
}

fn opt_str(meta: &Value, key: &str) -> Result<Option<String>, String> {
    opt_field(meta, key, |v| v.as_str().map(str::to_owned), "a string")
}

fn opt_uint(meta: &Value, key: &str) -> Result<Option<u64>, String> {
    opt_field(meta, key, Value::as_u64, "a non-negative integer")
}

/// `(pending, total)` over the top-level `falsification_tests` list. Absent or
/// null is zero tests; anything but a list is an error.
fn test_counts(doc: &Value) -> Result<(usize, usize), String> {
    let list = match doc.get("falsification_tests") {
        None | Some(Value::Null) => return Ok((0, 0)),
        Some(Value::Array(list)) => list,
        Some(_) => return Err("falsification_tests is not a list".to_string()),
    };
    let pending = list
        .iter()
        .filter(|t| {
            t.get("test")
                .and_then(Value::as_str)
                .is_some_and(is_pending)
        })
        .count();
    Ok((pending, list.len()))
}

/// One story from its YAML text. Pure over the text.
///
/// # Errors
///
/// A description of why the text is not a usable story: not YAML, no
/// `metadata`, no `metadata.id`, an id that is not `CRUX-<CAT>-<NN>`, a field
/// of the wrong type, or `falsification_tests` that is not a list.
pub fn story_from_yaml(text: &str) -> Result<Story, String> {
    let doc: Value = crate::yaml::from_str(text).map_err(|e| format!("not parseable: {e}"))?;
    let meta = doc.get("metadata").ok_or("no metadata mapping")?;
    let id = opt_str(meta, "id")?.ok_or("metadata.id is missing")?;
    if !id_matches(&id) {
        return Err(format!("metadata.id {id:?} is not CRUX-<CAT>-<NN>"));
    }
    let (pending, total) = test_counts(&doc)?;
    let (verdict, reason) = derive(pending, total);
    let intake_status = opt_str(meta, "intake_status")?;
    Ok(Story {
        id,
        verdict,
        reason,
        competitor: opt_str(meta, "competitor")?,
        demand_score: opt_uint(meta, "demand_score")?,
        github_issue: opt_uint(meta, "github_issue")?,
        disagrees: disagrees(intake_status.as_deref(), verdict),
        intake_status,
    })
}

fn read_story(path: &Path) -> Result<Story, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    story_from_yaml(&text)
}

/// The story files directly under `dir`, in name order. A directory that cannot
/// be read has none.
fn story_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(name_matches)
        })
        .collect();
    files.sort();
    files
}

/// Demand descending (absent is 0), then id ascending.
pub fn sort_stories(stories: &mut [Story]) {
    stories.sort_by(|a, b| {
        b.demand_score
            .unwrap_or(0)
            .cmp(&a.demand_score.unwrap_or(0))
            .then_with(|| a.id.cmp(&b.id))
    });
}

/// Every story under `dir`, sorted.
///
/// # Errors
///
/// [`ZeroContracts`] when no file matches; [`StoryErrors`] when any matching
/// file is unusable (every such file is named).
pub fn gaps_of(dir: &Path) -> Result<Vec<Story>, Box<dyn std::error::Error>> {
    let files = story_files(dir);
    if files.is_empty() {
        return Err(Box::new(ZeroContracts {
            path: dir.to_path_buf(),
            filter: None,
        }));
    }
    let mut stories = Vec::new();
    let mut errors = Vec::new();
    for file in files {
        match read_story(&file) {
            Ok(story) => stories.push(story),
            Err(why) => errors.push((file, why)),
        }
    }
    if !errors.is_empty() {
        return Err(Box::new(StoryErrors {
            path: dir.to_path_buf(),
            errors,
        }));
    }
    sort_stories(&mut stories);
    Ok(stories)
}

/// Stories per derived verdict.
#[must_use]
pub fn counts_of(stories: &[Story]) -> Counts {
    let mut counts = Counts::default();
    for story in stories {
        match story.verdict {
            Verdict::Missing => counts.missing += 1,
            Verdict::Partial => counts.partial += 1,
            Verdict::NotMeasured => counts.not_measured += 1,
        }
    }
    counts
}

/// How many stories state a status that disagrees with the derived verdict.
#[must_use]
pub fn disagreement_count(stories: &[Story]) -> usize {
    stories.iter().filter(|s| s.disagrees).count()
}

const HEADER: [&str; 8] = [
    "ID",
    "VERDICT",
    "DEMAND",
    "COMPETITOR",
    "ISSUE",
    "STATED",
    "DISAGREES",
    "REASON",
];

fn opt_cell<T: ToString>(v: Option<&T>) -> String {
    v.map_or_else(|| "-".to_string(), ToString::to_string)
}

fn row_cells(s: &Story) -> [String; 8] {
    [
        s.id.clone(),
        s.verdict.as_str().to_string(),
        opt_cell(s.demand_score.as_ref()),
        opt_cell(s.competitor.as_ref()),
        opt_cell(s.github_issue.as_ref()),
        opt_cell(s.intake_status.as_ref()),
        if s.disagrees { "yes" } else { "no" }.to_string(),
        s.reason.clone(),
    ]
}

fn format_line(cells: &[String; 8], widths: &[usize; 8]) -> String {
    let padded: Vec<String> = cells
        .iter()
        .zip(widths)
        .map(|(c, &w)| format!("{c:<w$}"))
        .collect();
    format!("{}\n", padded.join("  ").trim_end())
}

/// The plain table: header, one row per story, one summary line.
#[must_use]
pub fn render_table(stories: &[Story]) -> String {
    let rows: Vec<[String; 8]> = stories.iter().map(row_cells).collect();
    let header = HEADER.map(String::from);
    let mut widths = HEADER.map(str::len);
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.len());
        }
    }
    let mut out = format_line(&header, &widths);
    for row in &rows {
        out.push_str(&format_line(row, &widths));
    }
    let c = counts_of(stories);
    out.push_str(&format!(
        "summary: {} stories; missing={} partial={} not_measured={}; disagreements={}\n",
        stories.len(),
        c.missing,
        c.partial,
        c.not_measured,
        disagreement_count(stories)
    ));
    out
}

fn story_json(s: &Story) -> Value {
    obj([
        ("id", Value::from(s.id.clone())),
        ("verdict", Value::from(s.verdict.as_str())),
        ("reason", Value::from(s.reason.clone())),
        ("competitor", Value::from(s.competitor.clone())),
        ("demand_score", Value::from(s.demand_score)),
        ("github_issue", Value::from(s.github_issue)),
        ("intake_status", Value::from(s.intake_status.clone())),
        ("disagrees", Value::from(s.disagrees)),
    ])
}

/// `{ "stories": [...], "counts": {...}, "disagreements": N }`, pretty, one
/// trailing newline. Deterministic: the same stories give the same bytes.
///
/// # Errors
///
/// The `serde_json` error, which a value built here cannot produce.
pub fn render_json(stories: &[Story]) -> Result<String, serde_json::Error> {
    let c = counts_of(stories);
    let doc = obj([
        (
            "stories",
            Value::Array(stories.iter().map(story_json).collect()),
        ),
        (
            "counts",
            obj([
                ("missing", Value::from(c.missing)),
                ("partial", Value::from(c.partial)),
                ("not_measured", Value::from(c.not_measured)),
            ]),
        ),
        ("disagreements", Value::from(disagreement_count(stories))),
    ]);
    Ok(format!("{}\n", serde_json::to_string_pretty(&doc)?))
}

/// `pv crux gaps [--json] [DIR]`.
///
/// # Errors
///
/// See [`gaps_of`].
pub fn run(contract_dir: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let stories = gaps_of(contract_dir)?;
    if json {
        print!("{}", render_json(&stories)?);
    } else {
        print!("{}", render_table(&stories));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A story document. `extra` is extra metadata lines (2-space indented,
    /// newline terminated); `tests` are `test:` strings, double-quoted in YAML.
    /// An empty `tests` omits `falsification_tests` altogether.
    fn story_yaml(id: &str, extra: &str, tests: &[&str]) -> String {
        let head = format!("metadata:\n  id: {id}\n  competitor: burn\n{extra}");
        if tests.is_empty() {
            return head;
        }
        let list = tests
            .iter()
            .enumerate()
            .map(|(i, t)| format!("- id: F-{i}\n  test: \"{t}\"\n"))
            .collect::<String>();
        format!("{head}falsification_tests:\n{list}")
    }

    fn corpus(files: &[(&str, String)]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        for (name, text) in files {
            std::fs::write(tmp.path().join(name), text).expect("fixture is writable");
        }
        tmp
    }

    fn story(extra: &str, tests: &[&str]) -> Story {
        story_from_yaml(&story_yaml("CRUX-A-01", extra, tests)).expect("fixture is a story")
    }

    const PENDING: &str = "LIVE-PENDING - not built yet";
    const REAL: &str = "cargo test -p aprender-compute --lib svd";

    #[test]
    fn all_pending_is_missing() {
        let s = story("", &[PENDING, PENDING, PENDING]);
        assert_eq!(s.verdict, Verdict::Missing);
        assert!(s.reason.contains("3/3"), "{}", s.reason);
    }

    #[test]
    fn mixed_is_partial() {
        let s = story("", &[PENDING, REAL, PENDING]);
        assert_eq!(s.verdict, Verdict::Partial);
        assert!(s.reason.contains("2/3"), "{}", s.reason);
    }

    #[test]
    fn none_pending_is_not_measured_and_never_supported() {
        let s = story("  intake_status: partial\n", &[REAL, REAL]);
        assert_eq!(s.verdict, Verdict::NotMeasured);
        assert!(s.reason.contains("0/2"), "{}", s.reason);
        assert!(
            s.reason.contains(
                "no LIVE-PENDING test; surface and bound tests not measured (sibling tickets)"
            ),
            "{}",
            s.reason
        );
        // The stated column may legitimately echo a stated `supported`; this
        // fixture states `partial`, so the word must appear nowhere.
        let stories = [s];
        assert!(!render_table(&stories).contains("supported"));
        assert!(!render_json(&stories)
            .expect("renders")
            .contains("supported"));
    }

    #[test]
    fn zero_tests_is_missing() {
        let absent = story("", &[]);
        assert_eq!(absent.verdict, Verdict::Missing);
        assert!(absent.reason.contains("no falsification tests"));
        let empty = story_from_yaml("metadata:\n  id: CRUX-A-01\nfalsification_tests: []\n")
            .expect("empty list is a story");
        assert_eq!(empty.verdict, Verdict::Missing);
        let null = story_from_yaml("metadata:\n  id: CRUX-A-01\nfalsification_tests:\n")
            .expect("null list is a story");
        assert_eq!(null.verdict, Verdict::Missing);
    }

    #[test]
    fn leading_whitespace_before_the_marker_is_pending() {
        assert!(is_pending("LIVE-PENDING"));
        assert!(is_pending("   LIVE-PENDING - x"));
        assert!(is_pending("\n\t LIVE-PENDING - x"));
        let s = story("", &["   LIVE-PENDING - x", "\\n  LIVE-PENDING - y"]);
        assert_eq!(s.verdict, Verdict::Missing);
    }

    #[test]
    fn marker_in_the_middle_is_not_pending() {
        assert!(!is_pending("cargo test x # LIVE-PENDING"));
        assert!(!is_pending("see LIVE-PENDING"));
        assert!(!is_pending("live-pending lower case"));
        assert!(!is_pending(""));
        let s = story("", &["cargo test x # LIVE-PENDING"]);
        assert_eq!(s.verdict, Verdict::NotMeasured);
    }

    #[test]
    fn a_test_without_a_string_test_field_is_not_pending() {
        let doc =
            "metadata:\n  id: CRUX-A-01\nfalsification_tests:\n- id: F-1\n- test: 7\n- plain\n";
        let s = story_from_yaml(doc).expect("odd items are still tests");
        assert_eq!(s.verdict, Verdict::NotMeasured);
        assert!(s.reason.contains("0/3"), "{}", s.reason);
    }

    #[test]
    fn disagreement_table_matches_the_rule() {
        use Verdict::{Missing, NotMeasured, Partial};
        let rows = [
            ("missing", Missing, false),
            ("partial", Missing, true),
            ("supported", Missing, true),
            ("bogus", Missing, true),
            ("missing", Partial, true),
            ("partial", Partial, false),
            ("supported", Partial, true),
            ("missing", NotMeasured, true),
            ("partial", NotMeasured, true),
            ("supported", NotMeasured, false),
            ("not_measured", NotMeasured, false),
        ];
        for (stated, derived, want) in rows {
            assert_eq!(
                disagrees(Some(stated), derived),
                want,
                "{stated} vs {derived:?}"
            );
        }
        for derived in [Missing, Partial, NotMeasured] {
            assert!(
                !disagrees(None, derived),
                "absent is no claim ({derived:?})"
            );
        }
    }

    #[test]
    fn disagreement_flows_from_the_yaml() {
        assert!(story("  intake_status: supported\n", &[PENDING]).disagrees);
        assert!(story("  intake_status: partial\n", &[PENDING]).disagrees);
        assert!(!story("  intake_status: missing\n", &[PENDING]).disagrees);
        assert!(!story("  intake_status: supported\n", &[REAL]).disagrees);
        assert!(!story("", &[PENDING]).disagrees);
    }

    #[test]
    fn stories_sort_by_demand_descending_then_id_ascending() {
        let t = [PENDING];
        let tmp = corpus(&[
            (
                "crux-A-01-v1.yaml",
                story_yaml("CRUX-A-01", "  demand_score: 3\n", &t),
            ),
            ("crux-A-02-v1.yaml", story_yaml("CRUX-A-02", "", &t)),
            (
                "crux-A-03-v1.yaml",
                story_yaml("CRUX-A-03", "  demand_score: 5\n", &t),
            ),
            (
                "crux-B-01-v1.yaml",
                story_yaml("CRUX-B-01", "  demand_score: 3\n", &t),
            ),
            (
                "crux-C-01-v1.yaml",
                story_yaml("CRUX-C-01", "  demand_score: 0\n", &t),
            ),
        ]);
        let ids: Vec<String> = gaps_of(tmp.path())
            .expect("corpus reads")
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(
            ids,
            [
                "CRUX-A-03",
                "CRUX-A-01",
                "CRUX-B-01",
                "CRUX-A-02",
                "CRUX-C-01"
            ]
        );
    }

    #[test]
    fn an_unparsable_file_is_an_error_naming_the_path() {
        let tmp = corpus(&[
            ("crux-A-01-v1.yaml", story_yaml("CRUX-A-01", "", &[PENDING])),
            (
                "crux-A-02-v1.yaml",
                "metadata: [unclosed\n  id: {".to_string(),
            ),
        ]);
        let err = gaps_of(tmp.path()).expect_err("must fail closed");
        let text = err.to_string();
        assert!(text.contains("crux-A-02-v1.yaml"), "{text}");
        assert!(text.contains("not parseable"), "{text}");
        assert!(!text.contains("crux-A-01-v1.yaml"), "{text}");
        assert_eq!(crate::contract_walk::exit_code_for(err.as_ref()), 1);
    }

    #[test]
    fn a_file_without_a_usable_metadata_id_is_an_error() {
        let docs = [
            "metadata:\n  competitor: burn\n",
            "other: 1\n",
            "metadata:\n  id: F-CRUX-A-01-QA-001\n",
            "metadata:\n  id: 12\n",
            "metadata:\n  id: CRUX-a-01\n",
            "metadata:\n  id: CRUX-A-\n",
        ];
        for doc in docs {
            assert!(story_from_yaml(doc).is_err(), "{doc:?} must be refused");
        }
        let tmp = corpus(&[("crux-A-01-v1.yaml", docs[0].to_string())]);
        let err = gaps_of(tmp.path()).expect_err("must fail closed");
        let text = err.to_string();
        assert!(
            text.contains("crux-A-01-v1.yaml") && text.contains("metadata.id"),
            "{text}"
        );
    }

    #[test]
    fn wrong_typed_fields_are_errors() {
        let docs = [
            "metadata:\n  id: CRUX-A-01\n  demand_score: high\n",
            "metadata:\n  id: CRUX-A-01\n  github_issue: -3\n",
            "metadata:\n  id: CRUX-A-01\n  intake_status: [a]\n",
            "metadata:\n  id: CRUX-A-01\nfalsification_tests:\n  a: 1\n",
        ];
        for doc in docs {
            assert!(story_from_yaml(doc).is_err(), "{doc:?} must be refused");
        }
        let ok = story_from_yaml(
            "metadata:\n  id: CRUX-A-01\n  demand_score: 4  # high\n  github_issue: 3147\n  competitor:\n",
        )
        .expect("typed fields read");
        assert_eq!(ok.demand_score, Some(4));
        assert_eq!(ok.github_issue, Some(3147));
        assert_eq!(ok.competitor, None);
    }

    #[test]
    fn an_unreadable_file_is_an_error() {
        let tmp = tempfile::tempdir().expect("temp dir");
        std::fs::write(tmp.path().join("crux-A-01-v1.yaml"), [0xff_u8, 0xfe, 0xfd])
            .expect("fixture is writable");
        let text = gaps_of(tmp.path())
            .expect_err("must fail closed")
            .to_string();
        assert!(text.contains("unreadable"), "{text}");
    }

    #[test]
    fn an_empty_or_missing_dir_is_zero_contracts() {
        let tmp = tempfile::tempdir().expect("temp dir");
        for dir in [tmp.path().to_path_buf(), tmp.path().join("absent")] {
            let err = gaps_of(&dir).expect_err("vacuity is an error");
            assert!(err.downcast_ref::<ZeroContracts>().is_some(), "{err}");
            assert_eq!(
                crate::contract_walk::exit_code_for(err.as_ref()),
                crate::contract_walk::ZERO_CONTRACTS_EXIT
            );
        }
        assert!(run(tmp.path(), false).is_err());
        assert!(run(tmp.path(), true).is_err());
    }

    #[test]
    fn non_crux_yaml_files_are_ignored() {
        let tmp = corpus(&[
            ("crux-A-01-v1.yaml", story_yaml("CRUX-A-01", "", &[PENDING])),
            (
                "crux-competitive-research-ux-v1.yaml",
                "not: a story".to_string(),
            ),
            ("crux-A-01-v2.yaml", ":::".to_string()),
            ("crux-a-01-v1.yaml", ":::".to_string()),
            ("crux-A-xx-v1.yaml", ":::".to_string()),
            ("other-v1.yaml", ":::".to_string()),
            ("crux-A-02-v1.yaml.bak", ":::".to_string()),
        ]);
        std::fs::create_dir(tmp.path().join("crux-A-09-v1.yaml"))
            .expect("a directory by that name");
        let stories = gaps_of(tmp.path()).expect("only the one story counts");
        assert_eq!(stories.len(), 1);
        assert_eq!(stories[0].id, "CRUX-A-01");
    }

    #[test]
    fn name_and_id_patterns() {
        for ok in ["crux-A-01-v1.yaml", "crux-AB-123-v1.yaml"] {
            assert!(name_matches(ok), "{ok}");
        }
        for bad in [
            "crux-A-01-v1.yml",
            "crux--01-v1.yaml",
            "crux-A--v1.yaml",
            "crux-A-01-02-v1.yaml",
            "crux-A01-v1.yaml",
            "Crux-A-01-v1.yaml",
            "crux-A-01.yaml",
            "",
        ] {
            assert!(!name_matches(bad), "{bad}");
        }
        assert!(id_matches("CRUX-N-01") && id_matches("CRUX-AB-7"));
        for bad in [
            "CRUX-N",
            "CRUX-N-",
            "CRUX--1",
            "crux-N-01",
            "CRUX-N-01-X",
            "CRUX-n-01",
            "",
        ] {
            assert!(!id_matches(bad), "{bad}");
        }
    }

    #[test]
    fn json_parses_back_and_is_byte_identical_across_runs() {
        let tmp = corpus(&[
            (
                "crux-A-01-v1.yaml",
                story_yaml(
                    "CRUX-A-01",
                    "  demand_score: 5\n  github_issue: 11\n  intake_status: supported\n",
                    &[PENDING],
                ),
            ),
            (
                "crux-A-02-v1.yaml",
                story_yaml(
                    "CRUX-A-02",
                    "  demand_score: 2\n  intake_status: partial\n",
                    &[PENDING, REAL],
                ),
            ),
            ("crux-A-03-v1.yaml", story_yaml("CRUX-A-03", "", &[REAL])),
        ]);
        let first = render_json(&gaps_of(tmp.path()).expect("reads")).expect("renders");
        let second = render_json(&gaps_of(tmp.path()).expect("reads")).expect("renders");
        assert_eq!(first, second, "two runs differ");
        let doc: Value = serde_json::from_str(&first).expect("output is JSON");
        assert_eq!(doc["stories"].as_array().map(Vec::len), Some(3));
        assert_eq!(doc["stories"][0]["id"], "CRUX-A-01");
        assert_eq!(doc["stories"][0]["verdict"], "missing");
        assert_eq!(doc["stories"][0]["github_issue"], 11);
        assert_eq!(doc["stories"][0]["disagrees"], true);
        assert_eq!(doc["stories"][2]["demand_score"], Value::Null);
        assert_eq!(doc["counts"]["missing"], 1);
        assert_eq!(doc["counts"]["partial"], 1);
        assert_eq!(doc["counts"]["not_measured"], 1);
        assert_eq!(doc["disagreements"], 1);
        assert!(first.ends_with("}\n"));
        assert!(run(tmp.path(), true).is_ok());
        assert!(run(tmp.path(), false).is_ok());
    }

    #[test]
    fn table_has_a_header_a_row_per_story_and_one_summary_line() {
        let tmp = corpus(&[
            (
                "crux-A-01-v1.yaml",
                story_yaml(
                    "CRUX-A-01",
                    "  demand_score: 5\n  intake_status: supported\n",
                    &[PENDING],
                ),
            ),
            ("crux-A-02-v1.yaml", story_yaml("CRUX-A-02", "", &[REAL])),
        ]);
        let table = render_table(&gaps_of(tmp.path()).expect("reads"));
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 4, "{table}");
        assert!(lines[0].starts_with("ID") && lines[0].ends_with("REASON"));
        assert!(
            lines[1].starts_with("CRUX-A-01")
                && lines[1].contains("missing")
                && lines[1].contains("yes")
        );
        assert!(
            lines[2].starts_with("CRUX-A-02")
                && lines[2].contains("not_measured")
                && lines[2].contains('-')
        );
        assert_eq!(
            lines[3],
            "summary: 2 stories; missing=1 partial=0 not_measured=1; disagreements=1"
        );
        assert!(!table.lines().any(|l| l.ends_with(' ')), "trailing blanks");
    }

    #[test]
    fn replacing_a_pending_test_by_a_real_one_moves_the_story_up() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let path = tmp.path().join("crux-A-01-v1.yaml");
        let steps = [
            ([PENDING, PENDING], Verdict::Missing),
            ([PENDING, REAL], Verdict::Partial),
            ([REAL, REAL], Verdict::NotMeasured),
        ];
        for (tests, want) in steps {
            std::fs::write(
                &path,
                story_yaml("CRUX-A-01", "  intake_status: missing\n", &tests),
            )
            .expect("fixture is writable");
            let stories = gaps_of(tmp.path()).expect("reads");
            assert_eq!(stories[0].verdict, want, "{tests:?}");
        }
        // The stated `missing` now disagrees with the derived not_measured.
        let last = gaps_of(tmp.path()).expect("reads");
        assert!(last[0].disagrees);
        assert_eq!(
            counts_of(&last),
            Counts {
                missing: 0,
                partial: 0,
                not_measured: 1
            }
        );
    }

    #[test]
    fn story_errors_display_lists_every_file() {
        let err = StoryErrors {
            path: PathBuf::from("contracts"),
            errors: vec![
                (
                    PathBuf::from("contracts/crux-A-01-v1.yaml"),
                    "x".to_string(),
                ),
                (
                    PathBuf::from("contracts/crux-A-02-v1.yaml"),
                    "y".to_string(),
                ),
            ],
        };
        let text = err.to_string();
        assert!(
            text.starts_with("2 CRUX story file(s) under contracts unusable:"),
            "{text}"
        );
        assert!(text.contains("crux-A-01-v1.yaml: x") && text.contains("crux-A-02-v1.yaml: y"));
    }

    #[test]
    fn verdict_words() {
        assert_eq!(Verdict::Missing.as_str(), "missing");
        assert_eq!(Verdict::Partial.as_str(), "partial");
        assert_eq!(Verdict::NotMeasured.as_str(), "not_measured");
    }
}
