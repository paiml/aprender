//! ONT-001 §5 ONT-4e — preconditions and postconditions as first-class clauses.
//!
//! `requires[]` and `ensures[]` sit beside `invariants[]` with one shape, `{id, statement, formal, formal_status}`
//! (§3.5, the schema row: "`requires[]`/`ensures[]` beside `invariants[]` (ONT-4e), `*[].formal_status`"). The
//! shape is closed: an unknown key is a parse error, so a misspelt `formal_satus` is refused, not silently read as a
//! clause with no status.
//!
//! `formal_status` says whether `formal` is checkable: `parsed` clauses take part in the Liskov check on `refines`
//! (R-20), `prose` clauses make it `Unknown{Prose}`. A `parsed` clause without a `formal` is a contradiction in
//! terms and is refused by [`Clause::defect`].

use serde::{Deserialize, Serialize};

/// Whether a clause's `formal` is in the checkable subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FormalStatus {
    /// `formal` is a proposition the Liskov check can compare.
    Parsed,
    /// `formal` is absent or not checkable; any check that meets it answers `Unknown{Prose}`.
    Prose,
}

/// One `requires[]` / `ensures[]` entry (and an `invariants[]` entry that carries `formal_status`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clause {
    pub id: String,
    pub statement: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formal: Option<String>,
    pub formal_status: FormalStatus,
}

impl Clause {
    /// Why this clause is not well formed, or `None`.
    #[must_use]
    pub fn defect(&self) -> Option<String> {
        if self.id.trim().is_empty() {
            return Some("empty id".into());
        }
        let formal_empty = self.formal.as_deref().is_none_or(|f| f.trim().is_empty());
        (self.formal_status == FormalStatus::Parsed && formal_empty)
            .then(|| format!("{}: formal_status parsed with no formal", self.id))
    }

    /// The proposition the Liskov check compares: `formal` with whitespace runs collapsed. `None` for prose.
    #[must_use]
    pub fn atom(&self) -> Option<String> {
        match self.formal_status {
            FormalStatus::Parsed => self
                .formal
                .as_deref()
                .map(|f| f.split_whitespace().collect::<Vec<_>>().join(" ")),
            FormalStatus::Prose => None,
        }
    }
}
