//! The SECOND READER (#3568): a JSON Schema validator that shares nothing with the engine.
//!
//! The engine masks each step so the output can only be what the schema allows. That is one
//! reader. `jsonschema`, a crate the engine never calls, is the other: it reads the schema
//! before the model loads ([`check_schema`]) and the finished text after the turn
//! ([`second_reader`]). `apr run` and `apr serve` both ship a document only when the two agree.
//!
//! Neither reader fetches. `jsonschema`'s own default retriever follows an `http(s)://` or
//! `file://` `$ref`, and on `apr serve` the schema comes off the network: a request could make
//! the server read its own files or call any address it can reach. A `$ref` the schema does
//! not hold itself is `SchemaUnsupported`, named, before any model work.

use super::ConstraintError;
use jsonschema::error::ValidationErrorKind;
use jsonschema::{ReferencingError, Retrieve, Uri, Validator};
use serde_json::Value;

/// How many of the second reader's errors a refusal quotes.
const QUOTED_ERRORS: usize = 3;

/// A retriever that refuses every resource the schema does not hold itself.
struct NoFetch;

impl Retrieve for NoFetch {
    fn retrieve(&self, uri: &Uri<&str>) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(format!(
            "{} is outside the schema, and no reader fetches it",
            uri.as_str()
        )
        .into())
    }
}

/// Compile `schema` for the second reader, fetching nothing.
fn validator(schema: &Value) -> Result<Validator, ConstraintError> {
    jsonschema::options()
        .with_retriever(NoFetch)
        .build(schema)
        .map_err(|e| match &e.kind {
            ValidationErrorKind::Referencing(ReferencingError::Unretrievable { .. }) => {
                ConstraintError::SchemaUnsupported(format!(
                    "the schema has a $ref to a resource it does not hold: {e}"
                ))
            },
            _ => ConstraintError::SchemaInvalid(format!(
                "the schema is not a valid JSON Schema: {e}"
            )),
        })
}

/// Check `schema` with the second reader, before the model loads: one it cannot read
/// (`{"type": 12}`) is `SchemaInvalid` with no model work done, never `SchemaUnsupported`
/// from the engine after it.
///
/// # Errors
/// `SchemaInvalid` for a malformed schema; `SchemaUnsupported` for a `$ref` outside it.
pub fn check_schema(schema: &Value) -> Result<(), ConstraintError> {
    validator(schema).map(|_| ())
}

/// Validate a finished `text` against `schema` in the second reader.
///
/// # Errors
/// `Violation` when the text is not one JSON document or fails the schema (the first
/// [`QUOTED_ERRORS`] errors, each with its instance path); the schema's own refusal when it
/// does not compile.
pub fn second_reader(schema: &Value, text: &str) -> Result<(), ConstraintError> {
    let doc: Value = serde_json::from_str(text).map_err(|e| {
        ConstraintError::Violation(format!("the output is not one JSON document: {e}"))
    })?;
    let validator = validator(schema)?;
    let errors: Vec<String> = validator
        .iter_errors(&doc)
        .take(QUOTED_ERRORS)
        .map(|e| format!("{e} (at {})", e.instance_path))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(ConstraintError::Violation(format!(
            "the output fails the schema in a validator independent of the engine: {}",
            errors.join("; ")
        )))
    }
}
