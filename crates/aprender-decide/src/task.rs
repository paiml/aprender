//! The `task.json` document (D-05): `{type: "choice", instructions, criteria}`.
//!
//! **The document order of `criteria` is the label index.** That is the one rule this
//! module exists to keep, and it is where it is most easily lost: `serde_json::Map` is
//! a key-sorted `BTreeMap` by default and an insertion-ordered `IndexMap` only under
//! the `preserve_order` feature, which `pmcp` turns on for every build that links it.
//! A library tested alone and the same library linked into a server would then read
//! the same bytes into different label orders (RESEARCH Pitfall 2).
//!
//! So criteria are NEVER read through `serde_json::Value` / `Map`. They are read
//! straight from the raw bytes by a hand-written [`serde::de::Visitor::visit_map`]
//! that pushes `(name, description)` pairs in document order — the same order under
//! both map backings (`tests::serde_backing_canary` records which backing a run had).
//!
//! Refusals ([`TaskError`]): a `type` other than `"choice"`, fewer than 2 criteria, more than
//! [`MAX_CRITERIA`] (refused while reading, before the entry is stored or compared), a
//! duplicate criterion name (found with a set, so the scan is linear), an empty name, a
//! non-string description, and any unknown top-level key (`deny_unknown_fields`). Nothing is
//! defaulted.
//!
//! Contract: `contracts/decide-apr-v1.yaml` `task_json_schema` and
//! `equations.task_order_is_label_index`.

use serde::de::{Deserializer, Error as _, MapAccess, Visitor};
use serde::Deserialize;
use std::collections::HashSet;
use std::fmt;

/// The message the criteria visitor refuses an over-bound entry with. serde's own messages
/// never start with it ("invalid type", "unknown field", ...), so [`Task::from_slice`] maps a
/// data error that does back to the typed [`TaskError::TooManyCriteria`].
const TOO_MANY_CRITERIA: &str = "decide-apr-v1 max_criteria exceeded";

/// The most criteria a task may declare (decide-apr-v1 `constants.max_criteria`, asserted
/// equal by `tests::max_criteria_matches_contract`; the derivation is the contract's).
/// Refused while the criteria object is being read, before the entry that would exceed it
/// is stored or compared, so a hostile task blob costs at most this many entries.
pub const MAX_CRITERIA: usize = 510;

/// One criterion: its name is the label; its description, when present, is rendered
/// into the option text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Criterion {
    /// The label (a criterion name, unique within the task).
    pub name: String,
    /// Optional description; `null` and `""` both mean "name only".
    pub description: Option<String>,
}

/// A parsed, validated `task.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    instructions: String,
    criteria: Vec<Criterion>,
    sha256: String,
}

/// Why a `task.json` was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskError {
    /// Not valid JSON for the schema: malformed, an unknown top-level key, a missing
    /// field, or a description that is neither a string nor `null`.
    Parse(String),
    /// `type` is not `"choice"` (the only task type a decision artifact serves).
    UnsupportedType(String),
    /// Fewer than 2 criteria.
    TooFewCriteria(usize),
    /// The same criterion name appears twice; the second occurrence would silently
    /// shadow or shift a label.
    DuplicateCriterion(String),
    /// More than [`MAX_CRITERIA`] criteria: refused while reading (V1-b).
    TooManyCriteria {
        /// The bound ([`MAX_CRITERIA`]).
        max: usize,
    },
    /// A criterion name is empty.
    EmptyCriterionName {
        /// Its position in document order.
        index: usize,
    },
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "task.json: {e}"),
            Self::UnsupportedType(t) => {
                write!(
                    f,
                    "task.json: type {t:?} is not supported (only \"choice\")"
                )
            }
            Self::TooFewCriteria(n) => {
                write!(f, "task.json: {n} criteria, at least 2 are required")
            }
            Self::DuplicateCriterion(name) => {
                write!(f, "task.json: criterion {name:?} appears more than once")
            }
            Self::TooManyCriteria { max } => write!(
                f,
                "task.json: more than {max} criteria (decide-apr-v1 max_criteria)"
            ),
            Self::EmptyCriterionName { index } => {
                write!(f, "task.json: criterion {index} has an empty name")
            }
        }
    }
}

impl std::error::Error for TaskError {}

/// Criteria in DOCUMENT order, with the first repeated name (if any) recorded so the
/// caller can refuse it as a typed [`TaskError::DuplicateCriterion`].
struct OrderedCriteria {
    pairs: Vec<(String, Option<String>)>,
    first_duplicate: Option<String>,
}

impl<'de> Deserialize<'de> for OrderedCriteria {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedVisitor;

        impl<'de> Visitor<'de> for OrderedVisitor {
            type Value = OrderedCriteria;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a criteria object mapping names to descriptions (string or null)")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut pairs: Vec<(String, Option<String>)> = Vec::new();
                let mut seen: HashSet<String> = HashSet::new();
                let mut first_duplicate = None;
                while let Some((name, description)) = map.next_entry::<String, Option<String>>()? {
                    // V1-b: the entry that would exceed the bound is refused BEFORE it is stored
                    // or compared, and the parse stops here — a hostile criteria object costs at
                    // most MAX_CRITERIA entries.
                    if pairs.len() == MAX_CRITERIA {
                        return Err(A::Error::custom(TOO_MANY_CRITERIA));
                    }
                    if !seen.insert(name.clone()) && first_duplicate.is_none() {
                        first_duplicate = Some(name.clone());
                    }
                    pairs.push((name, description.filter(|d| !d.is_empty())));
                }
                Ok(OrderedCriteria {
                    pairs,
                    first_duplicate,
                })
            }
        }

        deserializer.deserialize_map(OrderedVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskDoc {
    #[serde(rename = "type")]
    kind: String,
    instructions: String,
    criteria: OrderedCriteria,
}

impl Task {
    /// Parse and validate `task.json` from its raw bytes.
    ///
    /// # Errors
    ///
    /// A [`TaskError`] for every refusal listed in the module docs.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, TaskError> {
        let doc: TaskDoc = serde_json::from_slice(bytes).map_err(|e| {
            let message = e.to_string();
            if e.is_data() && message.starts_with(TOO_MANY_CRITERIA) {
                TaskError::TooManyCriteria { max: MAX_CRITERIA }
            } else {
                TaskError::Parse(message)
            }
        })?;
        if doc.kind != "choice" {
            return Err(TaskError::UnsupportedType(doc.kind));
        }
        if let Some(name) = doc.criteria.first_duplicate {
            return Err(TaskError::DuplicateCriterion(name));
        }
        if let Some(index) = doc.criteria.pairs.iter().position(|(n, _)| n.is_empty()) {
            return Err(TaskError::EmptyCriterionName { index });
        }
        if doc.criteria.pairs.len() < 2 {
            return Err(TaskError::TooFewCriteria(doc.criteria.pairs.len()));
        }
        let criteria = doc
            .criteria
            .pairs
            .into_iter()
            .map(|(name, description)| Criterion { name, description })
            .collect();
        let sha256 = crate::digest::sha256_hex(bytes);
        Ok(Self {
            instructions: doc.instructions,
            criteria,
            sha256,
        })
    }

    /// The task instructions (Laya's question text).
    #[must_use]
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// The criteria in label-index order.
    #[must_use]
    pub fn criteria(&self) -> &[Criterion] {
        &self.criteria
    }

    /// The labels (criterion names) in label-index order.
    #[must_use]
    pub fn labels(&self) -> Vec<&str> {
        self.criteria.iter().map(|c| c.name.as_str()).collect()
    }

    /// [`Self::labels`] as owned strings (the manifest's `labels` shape).
    #[must_use]
    pub fn owned_labels(&self) -> Vec<String> {
        self.criteria.iter().map(|c| c.name.clone()).collect()
    }

    /// The option texts in label-index order: `"name: description"`, or `"name"`
    /// when there is no description (Laya `render_options` for `choice`).
    #[must_use]
    pub fn render_options(&self) -> Vec<String> {
        self.criteria
            .iter()
            .map(|c| {
                c.description
                    .as_ref()
                    .map_or_else(|| c.name.clone(), |d| format!("{}: {d}", c.name))
            })
            .collect()
    }

    /// Lower-case hex sha256 of the exact bytes this task was parsed from.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[cfg(test)]
mod tests {
    use super::{Task, TaskError, MAX_CRITERIA};
    use std::time::{Duration, Instant};

    /// A choice task with `n` distinct criteria `c0 .. c{n-1}`.
    fn task_with(n: usize) -> String {
        let criteria: Vec<String> = (0..n).map(|i| format!(r#""c{i}":null"#)).collect();
        format!(
            r#"{{"type":"choice","instructions":"q","criteria":{{{}}}}}"#,
            criteria.join(",")
        )
    }

    /// V1-b / T-08-26-02: `max_criteria` criteria parse; one more is refused as the typed
    /// `TooManyCriteria` while the criteria object is being read; and a 100 000-criteria task
    /// is refused in under one second in a debug build. The timing is asserted BEFORE the
    /// result, so an unbounded quadratic scan is caught by the clock, not only by the result.
    #[test]
    fn too_many_criteria_refused_while_reading() {
        let huge = task_with(100_000);
        let started = Instant::now();
        let result = Task::from_slice(huge.as_bytes());
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "a 100 000-criteria task took {elapsed:?} to refuse"
        );
        assert_eq!(
            result.expect_err("100 000 criteria refused"),
            TaskError::TooManyCriteria { max: MAX_CRITERIA }
        );

        let at = Task::from_slice(task_with(MAX_CRITERIA).as_bytes()).expect("max_criteria parses");
        assert_eq!(at.criteria().len(), MAX_CRITERIA);
        let e = Task::from_slice(task_with(MAX_CRITERIA + 1).as_bytes())
            .expect_err("max_criteria + 1 refused");
        assert_eq!(e, TaskError::TooManyCriteria { max: MAX_CRITERIA });

        // The count refusal wins over a duplicate that only appears past the bound: the entry
        // that would exceed the bound is refused before it is compared with anything.
        let past = task_with(MAX_CRITERIA).replace("}}", r#","c0":null}}"#);
        assert_eq!(
            Task::from_slice(past.as_bytes()).expect_err("over the bound"),
            TaskError::TooManyCriteria { max: MAX_CRITERIA }
        );
    }

    /// The code mirrors decide-apr-v1 `constants.max_criteria` (the contract is the source).
    #[test]
    fn max_criteria_matches_contract() {
        let c = crate::test_support::contract_yaml("decide-apr-v1.yaml");
        let declared = c["constants"]["max_criteria"]
            .as_u64()
            .expect("decide-apr-v1 constants.max_criteria");
        assert_eq!(
            MAX_CRITERIA as u64, declared,
            "task::MAX_CRITERIA vs decide-apr-v1 constants.max_criteria"
        );
    }

    fn labels(json: &str) -> Vec<String> {
        Task::from_slice(json.as_bytes())
            .expect("task parses")
            .labels()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Reports which `serde_json::Map` backing this build compiled. It never fails:
    /// it is the evidence that a run exercised the backing it claims (CLAUDE.md
    /// Verification #2 — prove the mechanism engaged).
    #[test]
    fn serde_backing_canary() {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(r#"{"b":1,"a":2}"#).expect("parse canary map");
        let insertion_ordered = map.keys().next().map(String::as_str) == Some("b");
        println!(
            "serde_json backing: preserve_order={}",
            if insertion_ordered { "ON" } else { "OFF" }
        );
    }

    /// Document order, deliberately NOT sorted order, is the label index.
    #[test]
    fn order_is_document_order() {
        let got = labels(
            r#"{"type":"choice","instructions":"q","criteria":{"zeta":"z","alpha":"a","mid":null}}"#,
        );
        assert_eq!(got, ["zeta", "alpha", "mid"]);
    }

    /// FALSIFY-DECIDE-APR-007's own prediction: the stance demo order.
    #[test]
    fn stance_order_none_against_favor() {
        let got = labels(
            r#"{"type":"choice","instructions":"stance","criteria":{"none":"","against":"opposes","favor":"supports"}}"#,
        );
        assert_eq!(got, ["none", "against", "favor"]);
    }

    #[test]
    fn render_options_name_or_name_colon_description() {
        let t = Task::from_slice(
            br#"{"type":"choice","instructions":"q","criteria":{"b":"bee","a":"","c":null}}"#,
        )
        .expect("task parses");
        assert_eq!(t.render_options(), ["b: bee", "a", "c"]);
        assert_eq!(t.instructions(), "q");
    }

    #[test]
    fn refuses_non_choice_type() {
        let e =
            Task::from_slice(br#"{"type":"score","instructions":"q","criteria":{"a":"","b":""}}"#)
                .expect_err("score refused");
        assert_eq!(e, TaskError::UnsupportedType("score".into()));
    }

    #[test]
    fn refuses_one_criterion() {
        let e = Task::from_slice(br#"{"type":"choice","instructions":"q","criteria":{"a":"x"}}"#)
            .expect_err("one criterion refused");
        assert_eq!(e, TaskError::TooFewCriteria(1));
    }

    #[test]
    fn refuses_duplicate_criterion() {
        let e = Task::from_slice(
            br#"{"type":"choice","instructions":"q","criteria":{"a":"x","b":"y","a":"z"}}"#,
        )
        .expect_err("duplicate refused");
        assert_eq!(e, TaskError::DuplicateCriterion("a".into()));
    }

    /// The FIRST repeated name is named, found through the set whatever its distance from
    /// the original.
    #[test]
    fn refuses_the_first_duplicate_criterion() {
        let e = Task::from_slice(
            br#"{"type":"choice","instructions":"q","criteria":{"a":"","b":"","c":"","b":"","a":""}}"#,
        )
        .expect_err("duplicates refused");
        assert_eq!(e, TaskError::DuplicateCriterion("b".into()));
    }

    #[test]
    fn refuses_empty_name() {
        let e = Task::from_slice(
            br#"{"type":"choice","instructions":"q","criteria":{"a":"x","":"y"}}"#,
        )
        .expect_err("empty name refused");
        assert_eq!(e, TaskError::EmptyCriterionName { index: 1 });
    }

    #[test]
    fn refuses_unknown_top_level_key() {
        let e = Task::from_slice(
            br#"{"type":"choice","instructions":"q","criteria":{"a":"","b":""},"labels":{}}"#,
        )
        .expect_err("unknown key refused");
        assert!(
            matches!(e, TaskError::Parse(ref m) if m.contains("labels")),
            "{e}"
        );
    }

    #[test]
    fn refuses_non_string_description() {
        let e =
            Task::from_slice(br#"{"type":"choice","instructions":"q","criteria":{"a":1,"b":""}}"#)
                .expect_err("numeric description refused");
        assert!(matches!(e, TaskError::Parse(_)), "{e}");
    }

    /// The sha256 is of the exact bytes, so a reformatted task is a different task.
    #[test]
    fn sha256_is_of_the_raw_bytes() {
        let a =
            Task::from_slice(br#"{"type":"choice","instructions":"q","criteria":{"a":"","b":""}}"#)
                .expect("a");
        let b = Task::from_slice(
            br#"{ "type":"choice","instructions":"q","criteria":{"a":"","b":""}}"#,
        )
        .expect("b");
        assert_eq!(a.labels(), b.labels());
        assert_ne!(a.sha256(), b.sha256());
        assert_eq!(a.sha256().len(), 64);
    }
}
