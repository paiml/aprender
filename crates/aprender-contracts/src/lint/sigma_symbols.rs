//! ONT-001 §5 ONT-2b — the symbol-level check on `formal:`.
//!
//! **What is checked: the operator vocabulary, and nothing else.** Every non-ASCII, non-letter glyph in a `formal:`
//! expression must be a symbol Σ declares. That is the vocabulary Σ owns.
//!
//! **What is NOT checked, deliberately** (plan v2 ruling 2, and each of these is measured, not guessed):
//!
//! - **Applied identifiers.** `encode(x)`, `is_ok(r)`, `predict(m)`, `softmax(v)` — 1325 of the corpus's 2303
//!   expressions call a name Σ does not declare, because those names are CODE, not ontology. Resolving them is
//!   ONT-3a's `extract:code`, fail-closed, against the workspace. A Σ that had to enumerate every function in the
//!   tree would be a second, worse copy of the binding registry.
//! - Grammar, parenthesis balance, arity, types, scope, variable binding, or whether the expression is TRUE.
//! - Bare variables (`x`, `i`, `out_i`, `θ`): a variable is not a symbol.
//!
//! **The opt-out is explicit.** An entry carrying `prose: true` beside its `formal:` is prose the author declared as
//! prose, and its glyphs are not checked. Inferring "this looks like prose" was refused in the plan grill: an
//! inferred rule cannot be shrunk deliberately, and cannot fail on a malformed string.
//!
//! **The debt is counted, not hidden.** `formal_prose` in `contracts/lint-baseline.json` is the number of `formal:`
//! entries carrying NO declared symbol at all — 1536 of 2303 when this row landed. It is shrink-only: the corpus may
//! become more formal over time, never less.

use crate::ontology::sigma::Sigma;

/// One `formal:` expression, and where it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalExpr {
    /// The expression as written.
    pub text: String,
    /// The author's explicit `prose: true` opt-out, read from the sibling key.
    pub prose: bool,
    /// A path like `equations.softmax` for the report.
    pub path: String,
}

/// What the symbol check found in one file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolScan {
    /// Expressions carrying a glyph Σ does not declare, and no `prose: true`.
    pub undeclared: Vec<(String, String, char)>,
    /// Expressions carrying no declared symbol at all — the `formal_prose` debt.
    pub prose_count: usize,
    /// Expressions read.
    pub total: usize,
}

/// Is `c` a glyph the Σ vocabulary owns? Non-ASCII and not a letter: `∀`, `≥`, `‖` yes; `θ`, `σ` no (a Greek letter
/// is a variable name in this corpus — `σ(x)`, `θ` — not an operator).
#[must_use]
pub fn is_operator_glyph(c: char) -> bool {
    !c.is_ascii() && !c.is_alphabetic()
}

/// Collect every `formal:` expression in a raw contract document, with its `prose:` sibling.
#[must_use]
pub fn collect_formals(_doc: &serde_yaml::Value) -> Vec<FormalExpr> {
    Vec::new() // RED
}

/// Check one file's expressions against Σ.
#[must_use]
pub fn scan(_sigma: &Sigma, _exprs: &[FormalExpr]) -> SymbolScan {
    SymbolScan::default() // RED
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigma() -> Sigma {
        Sigma::from_yaml(
            r#"
schema: ont-sigma-v1
symbols:
  - {name: "∀", kind: operator}
  - {name: "≥", kind: operator}
  - {name: len, kind: function}
readers:
  symbols: lint/sigma_symbols.rs
"#,
        )
        .expect("Σ parses")
    }

    fn doc(yaml: &str) -> serde_yaml::Value {
        serde_yaml::from_str(yaml).expect("fixture parses")
    }

    #[test]
    fn a_greek_letter_is_a_variable_not_an_operator() {
        assert!(is_operator_glyph('∀'));
        assert!(is_operator_glyph('≥'));
        assert!(!is_operator_glyph('θ'), "θ is a variable in this corpus");
        assert!(
            !is_operator_glyph('σ'),
            "σ(x) is a function name, not a glyph"
        );
        assert!(!is_operator_glyph('x'));
    }

    #[test]
    fn formals_are_collected_with_their_path_and_prose_flag() {
        let d = doc(r"
equations:
  softmax:
    formal: '∀i: len(out) = len(x)'
  notes:
    formal: 'the kernel is stable — see the paper'
    prose: true
");
        let mut found = collect_formals(&d);
        found.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].path, "equations.notes");
        assert!(found[0].prose, "the explicit opt-out is read");
        assert_eq!(found[1].path, "equations.softmax");
        assert!(!found[1].prose);
    }

    #[test]
    fn a_declared_vocabulary_passes() {
        let d = doc(r"
equations:
  ok:
    formal: '∀i: len(out) ≥ len(x)'
");
        let scan = scan(&sigma(), &collect_formals(&d));
        assert!(scan.undeclared.is_empty(), "{scan:?}");
        assert_eq!(scan.total, 1);
        assert_eq!(scan.prose_count, 0);
    }

    #[test]
    fn an_undeclared_glyph_is_reported_with_the_glyph_itself() {
        let d = doc(r"
equations:
  bad:
    formal: '∀i: out_i ∘ x_i'
");
        let scan = scan(&sigma(), &collect_formals(&d));
        assert_eq!(scan.undeclared.len(), 1, "{scan:?}");
        assert_eq!(scan.undeclared[0].0, "equations.bad");
        assert_eq!(scan.undeclared[0].2, '∘');
    }

    #[test]
    fn the_explicit_prose_marker_exempts_the_expression() {
        let d = doc(r"
equations:
  bad:
    formal: 'the kernel is stable — see the paper'
    prose: true
");
        let scan = scan(&sigma(), &collect_formals(&d));
        assert!(
            scan.undeclared.is_empty(),
            "prose: true exempts it: {scan:?}"
        );
    }

    #[test]
    fn an_expression_with_no_declared_symbol_is_counted_as_prose_debt() {
        let d = doc(r"
equations:
  a:
    formal: 'the output preserves the input ordering'
  b:
    formal: '∀i: len(out) = len(x)'
");
        let scan = scan(&sigma(), &collect_formals(&d));
        assert_eq!(scan.total, 2);
        assert_eq!(scan.prose_count, 1, "only `a` carries no declared symbol");
    }

    #[test]
    fn an_applied_identifier_is_not_checked() {
        // `encode` is code, not ontology: ONT-3a resolves it, fail-closed, against the workspace.
        let d = doc(r"
equations:
  call:
    formal: '∀i: encode(x_i) ≥ 0'
");
        let scan = scan(&sigma(), &collect_formals(&d));
        assert!(scan.undeclared.is_empty(), "{scan:?}");
    }
}
