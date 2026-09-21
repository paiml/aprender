//! Schema-constrained decoding (#3568): the one hook a generation loop calls.
//!
//! A constrained step is `mask` (every token the constraint forbids goes to
//! `-inf`), then the loop's OWN sampler, unchanged, then `accept`. The mask
//! only removes candidates, so greedy / temperature-0 decoding stays
//! deterministic. The answer is produced right rather than repaired after the
//! fact.
//!
//! The engine sits behind [`TokenConstraint`] so it can be swapped. Today it
//! is `llguidance` (MIT), behind the `structured-output` feature, which apr-cli's
//! `inference` feature turns on (cop ruling on #3568, 2026-09-21). Without that
//! feature, building a constraint refuses by name ([`ConstraintError::NotCompiled`]),
//! and the flag is never silently ignored.
//!
//! What a schema CANNOT say stays the consumer's to check. A grammar enforces
//! what JSON Schema expresses: types, required keys, `additionalProperties`,
//! bounds, `enum`/`const`, `pattern`. Cross-field facts are outside it, for
//! example `correct_index < options.length`, feedback that lines up with its
//! options, or a contiguous `n` across records. A passing constraint says
//! nothing about them (rmedia, #3716).
//!
//! This module has no generation loop in it. It is the trait, the refusals,
//! the schema source, and the vocabulary the engine needs (each token's RAW
//! BYTES, from the same per-token decoding `GGUFModel::decode` uses, so the mask
//! and the decoded text can never disagree about what a token is).

use std::fmt;

#[cfg(feature = "structured-output")]
mod llg;
#[cfg(feature = "structured-output")]
pub use llg::{LlgConstraint, LlgEnv};

/// What every constrained generation loop calls, once per generated token.
pub trait TokenConstraint: Send {
    /// Set every logit this constraint forbids at the current position to `-inf`.
    ///
    /// `Err(DeadEnd)` when nothing is allowed. The loop must then stop: it never
    /// continues unconstrained.
    fn mask(&mut self, logits: &mut [f32]) -> Result<(), ConstraintError>;

    /// Advance past the token the sampler chose. A token the mask forbade is `Rejected`.
    fn accept(&mut self, token: u32) -> Result<(), ConstraintError>;

    /// The output so far is a complete document: end-of-sequence is allowed here.
    fn is_complete(&mut self) -> bool;
}

/// Every way a constrained generation refuses, by name (#3568: "refused in one line").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    /// The schema argument is not a readable JSON Schema: bad JSON, an unreadable
    /// `@path`, or a document no engine can compile.
    SchemaInvalid(String),
    /// A well-formed schema that uses something this engine cannot enforce.
    SchemaUnsupported(String),
    /// No token is allowed at this output position.
    DeadEnd {
        /// Tokens accepted before the dead end.
        position: usize,
        /// The engine's reason.
        reason: String,
    },
    /// The sampler chose a token the constraint does not allow.
    Rejected {
        /// The token that was refused.
        token: u32,
        /// Tokens accepted before it.
        position: usize,
        /// The engine's reason.
        reason: String,
    },
    /// The vocabulary could not be turned into the engine's token table.
    Vocab(String),
    /// This build has no `structured-output` feature.
    NotCompiled,
    /// A generation path that does not apply a constraint yet (#3793). Refused, never run
    /// unconstrained: "a constraint that is silently ignored is decoration".
    UnsupportedPath {
        /// The path, by name (`gguf-cuda`, `apr`, `qwen3-moe`, ...).
        path: String,
        /// What removes this refusal.
        removed_by: String,
    },
    /// A thinking template is selected: 0.69.1 constrains no-think output only (#3735
    /// constrains after `</think>` in 0.70).
    WithThinking(String),
    /// The finished output failed the second reader: a validator independent of the engine.
    Violation(String),
    /// The token budget ran out before the output was a complete document. A cut document
    /// is an error, never a success (rmedia, #3716).
    Truncated {
        /// The budget the loop ran with.
        max_tokens: usize,
    },
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaInvalid(why) => write!(f, "SchemaInvalid: {why}"),
            Self::SchemaUnsupported(why) => write!(f, "SchemaUnsupported: {why}"),
            Self::DeadEnd { position, reason } => write!(
                f,
                "ConstraintDeadEnd: no token is allowed at output position {position}: {reason}"
            ),
            Self::Rejected {
                token,
                position,
                reason,
            } => write!(
                f,
                "ConstraintRejected: token {token} is not allowed at output position {position}: {reason}"
            ),
            Self::Vocab(why) => write!(f, "ConstraintVocab: {why}"),
            Self::NotCompiled => write!(
                f,
                "StructuredOutputNotCompiled: this build has no `structured-output` feature, so a \
                 schema cannot be enforced (refused rather than ignored)"
            ),
            Self::UnsupportedPath { path, removed_by } => write!(
                f,
                "SchemaUnsupportedPath: the {path} generation path does not apply a constraint \
                 yet, so it refuses rather than running unconstrained (removed_by: {removed_by})"
            ),
            Self::WithThinking(why) => write!(f, "SchemaWithThinking: {why}"),
            Self::Violation(why) => write!(f, "SchemaViolation: {why}"),
            Self::Truncated { max_tokens } => write!(
                f,
                "Truncated: the budget of {max_tokens} tokens ran out before the output was a \
                 complete document; a cut document is an error, never a success"
            ),
        }
    }
}

impl std::error::Error for ConstraintError {}

/// Read a schema argument: the schema inline, or `@path` to a file holding it.
///
/// Both forms are accepted on purpose (#3568 design input). A caller once passed
/// a path where a tool expected the schema, the process died on its arguments,
/// and from outside that looked like a lane with nothing to say. Validated here,
/// before the first token.
pub fn load_schema(arg: &str) -> Result<serde_json::Value, ConstraintError> {
    let (text, from) = match arg.strip_prefix('@') {
        Some(path) => (
            std::fs::read_to_string(path).map_err(|e| {
                ConstraintError::SchemaInvalid(format!("cannot read the schema file {path}: {e}"))
            })?,
            format!("the schema file {path}"),
        ),
        None => (arg.to_string(), "the inline schema".to_string()),
    };
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ConstraintError::SchemaInvalid(format!("{from} is not JSON: {e}")))?;
    if !(value.is_object() || value.is_boolean()) {
        return Err(ConstraintError::SchemaInvalid(format!(
            "{from} is JSON but not a schema: a JSON Schema is an object or a boolean"
        )));
    }
    Ok(value)
}

/// Read a grammar argument (`--grammar`): the Lark grammar inline, or `@path` to a file
/// holding it. Checked here, before the first token.
pub fn load_grammar(arg: &str) -> Result<String, ConstraintError> {
    let (text, from) = match arg.strip_prefix('@') {
        Some(path) => (
            std::fs::read_to_string(path).map_err(|e| {
                ConstraintError::SchemaInvalid(format!("cannot read the grammar file {path}: {e}"))
            })?,
            format!("the grammar file {path}"),
        ),
        None => (arg.to_string(), "the inline grammar".to_string()),
    };
    if text.trim().is_empty() {
        return Err(ConstraintError::SchemaInvalid(format!("{from} is empty")));
    }
    Ok(text)
}

/// What a caller asked generation to be constrained to (#3793), read and checked before the
/// model loads, and carried on the inference config.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintRequest {
    /// `--json-schema`: a JSON Schema document.
    JsonSchema(serde_json::Value),
    /// `--grammar`: a Lark grammar.
    Lark(String),
}

impl ConstraintRequest {
    /// Compile this request against a model's indexed vocabulary.
    pub fn compile(
        &self,
        env: &ConstraintEnv,
    ) -> Result<Box<dyn TokenConstraint>, ConstraintError> {
        match self {
            Self::JsonSchema(schema) => env.json_schema(schema),
            Self::Lark(grammar) => env.lark(grammar),
        }
    }
}

/// A vocabulary as a constraint sees it: each token's raw bytes, which tokens
/// are special (control tokens, which never appear in constrained text), and
/// the end-of-sequence id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintVocab {
    /// The bytes each token id decodes to, indexed by token id.
    pub token_bytes: Vec<Vec<u8>>,
    /// Whether each token id is a special (control) token.
    pub special: Vec<bool>,
    /// The end-of-sequence token id.
    pub eos: u32,
}

impl ConstraintVocab {
    /// Check the three parts agree, before any engine sees them.
    pub fn validate(&self) -> Result<(), ConstraintError> {
        if self.token_bytes.is_empty() {
            return Err(ConstraintError::Vocab(
                "the vocabulary is empty".to_string(),
            ));
        }
        if self.special.len() != self.token_bytes.len() {
            return Err(ConstraintError::Vocab(format!(
                "{} token byte strings but {} special flags",
                self.token_bytes.len(),
                self.special.len()
            )));
        }
        if self.eos as usize >= self.token_bytes.len() {
            return Err(ConstraintError::Vocab(format!(
                "eos {} is outside a vocabulary of {}",
                self.eos,
                self.token_bytes.len()
            )));
        }
        Ok(())
    }
}

/// The engine built once per model, from which each request's constraint is made.
///
/// Building it indexes the whole vocabulary (248,320 tokens for Qwen3.5), so a
/// caller builds it once and keeps it.
pub struct ConstraintEnv {
    #[cfg(feature = "structured-output")]
    inner: LlgEnv,
}

impl ConstraintEnv {
    /// Index `vocab` for constrained decoding.
    pub fn new(vocab: &ConstraintVocab) -> Result<Self, ConstraintError> {
        vocab.validate()?;
        #[cfg(feature = "structured-output")]
        {
            Ok(Self {
                inner: LlgEnv::new(vocab)?,
            })
        }
        #[cfg(not(feature = "structured-output"))]
        {
            Err(ConstraintError::NotCompiled)
        }
    }

    /// A constraint that admits exactly the documents `schema` accepts.
    pub fn json_schema(
        &self,
        schema: &serde_json::Value,
    ) -> Result<Box<dyn TokenConstraint>, ConstraintError> {
        #[cfg(feature = "structured-output")]
        {
            Ok(Box::new(self.inner.json_schema(schema)?))
        }
        #[cfg(not(feature = "structured-output"))]
        {
            let _ = schema;
            Err(ConstraintError::NotCompiled)
        }
    }

    /// A constraint that admits exactly the strings a Lark grammar accepts.
    pub fn lark(&self, grammar: &str) -> Result<Box<dyn TokenConstraint>, ConstraintError> {
        #[cfg(feature = "structured-output")]
        {
            Ok(Box::new(self.inner.lark(grammar)?))
        }
        #[cfg(not(feature = "structured-output"))]
        {
            let _ = grammar;
            Err(ConstraintError::NotCompiled)
        }
    }
}

#[cfg(test)]
mod tests;
