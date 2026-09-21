//! `apr run --json-schema` / `--grammar` (#3793, #3568 PR 2): the flags read before the model
//! loads, every refusal named in one line, and a SECOND READER.
//!
//! The engine masks each step so the output can only be what the schema allows. That is one
//! reader. The finished text is then checked again by `jsonschema`, a validator that shares
//! nothing with the engine, before `apr run` exits 0. Masking is one reader; this is the
//! other. A Lark grammar has no second reader: nothing independent of the engine parses Lark
//! in this tree, and saying so is better than pretending one ran.

use crate::commands::run::ConstraintArgs;
use crate::error::{CliError, ConstraintRefusal, Result};
use realizar::constrain::{load_grammar, load_schema, ConstraintError, ConstraintRequest};
use realizar::infer::run_report::FinishReason;

/// What removes a `SchemaUnsupported` refusal: the engine (llguidance 1.8.0) cannot enforce the
/// keyword it names, and nothing schedules that.
const SCHEMA_UNSUPPORTED_REMOVED_BY: &str =
    "not scheduled: llguidance 1.8.0 cannot enforce it, so the schema must not use it";

/// What removes `SchemaWithThinking`.
const WITH_THINKING_REMOVED_BY: &str = "#3735 (0.70): constrain the answer after </think>";

/// The request the flags describe, read and checked here, before the model loads: a malformed
/// schema is `SchemaInvalid` with no model loaded and no token generated.
///
/// # Errors
/// [`CliError::ConstraintRefused`] (`SchemaInvalid`) for an unreadable or malformed argument.
pub(crate) fn constraint_request(args: &ConstraintArgs) -> Result<Option<ConstraintRequest>> {
    let request = match (&args.json_schema, &args.grammar) {
        (Some(schema), _) => ConstraintRequest::JsonSchema(load_schema(schema).map_err(refused)?),
        (None, Some(grammar)) => ConstraintRequest::Lark(load_grammar(grammar).map_err(refused)?),
        (None, None) => return Ok(None),
    };
    Ok(Some(request))
}

/// A realizar error as the CLI reports it: a constraint's refusal keeps its name
/// ([`CliError::ConstraintRefused`]); anything else is an inference failure.
pub(crate) fn inference_or_refusal(e: realizar::error::RealizarError) -> CliError {
    match e {
        realizar::error::RealizarError::Constraint(c) => refused(c),
        other => crate::commands::run::inference_error(other),
    }
}

/// A constraint's error as a refusal: its name, its one line, what removes it.
pub(crate) fn refused(e: ConstraintError) -> CliError {
    CliError::ConstraintRefused(refusal_of(&e))
}

/// The refusal for a constraint's error, keyed on the TYPE, never on its text.
pub(crate) fn refusal_of(e: &ConstraintError) -> ConstraintRefusal {
    let (kind, removed_by, finish_reason) = match e {
        ConstraintError::SchemaInvalid(_) => ("SchemaInvalid", None, None),
        ConstraintError::SchemaUnsupported(_) => (
            "SchemaUnsupported",
            Some(SCHEMA_UNSUPPORTED_REMOVED_BY.to_string()),
            None,
        ),
        ConstraintError::UnsupportedPath { removed_by, .. } => {
            ("SchemaUnsupportedPath", Some(removed_by.clone()), None)
        }
        ConstraintError::WithThinking(_) => (
            "SchemaWithThinking",
            Some(WITH_THINKING_REMOVED_BY.to_string()),
            None,
        ),
        ConstraintError::Violation(_) => ("SchemaViolation", None, Some("constraint_complete")),
        ConstraintError::Truncated { .. } => ("Truncated", None, Some("length")),
        ConstraintError::DeadEnd { .. } => ("ConstraintDeadEnd", None, Some("dead_end")),
        ConstraintError::Rejected { .. } => ("ConstraintRejected", None, None),
        ConstraintError::Vocab(_) => ("ConstraintVocab", None, None),
        ConstraintError::NotCompiled => ("StructuredOutputNotCompiled", None, None),
    };
    ConstraintRefusal {
        kind,
        message: e.to_string(),
        removed_by,
        finish_reason,
    }
}

/// The verdict on a constrained run's finished output (#3793): `None` when it stands, else the
/// refusal it earns. A document cut by the budget is `Truncated`; a complete one that fails the
/// second reader is `SchemaViolation`.
pub(crate) fn constraint_verdict(
    request: &ConstraintRequest,
    finish_reason: Option<FinishReason>,
    text: &str,
    max_tokens: usize,
) -> Option<ConstraintRefusal> {
    match finish_reason {
        Some(FinishReason::Length) => Some(refusal_of(&ConstraintError::Truncated { max_tokens })),
        Some(FinishReason::ConstraintComplete) => match request {
            ConstraintRequest::JsonSchema(schema) => second_reader(schema, text)
                .err()
                .map(|why| refusal_of(&ConstraintError::Violation(why))),
            ConstraintRequest::Lark(_) => None,
        },
        // A constrained loop reports only the two above; anything else is not a finished
        // document, and is not reported as one
        other => Some(refusal_of(&ConstraintError::Violation(format!(
            "the constrained loop ended with finish_reason {:?}, not a complete document",
            other.map(FinishReason::as_str)
        )))),
    }
}

/// Validate `text` against `schema` with `jsonschema`, which shares nothing with the engine.
///
/// # Errors
/// Why the text is not a conforming document: not JSON, or the first validation errors.
pub(crate) fn second_reader(
    schema: &serde_json::Value,
    text: &str,
) -> std::result::Result<(), String> {
    let doc: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("the output is not one JSON document: {e}"))?;
    let validator = jsonschema::validator_for(schema)
        .map_err(|e| format!("the second reader cannot compile the schema: {e}"))?;
    let errors: Vec<String> = validator
        .iter_errors(&doc)
        .take(3)
        .map(|e| format!("{e} (at {})", e.instance_path))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the output fails the schema in a validator independent of the engine: {}",
            errors.join("; ")
        ))
    }
}

#[cfg(test)]
#[path = "constrained_run_tests.rs"]
mod tests;
