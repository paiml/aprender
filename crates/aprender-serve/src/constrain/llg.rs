//! The `llguidance` engine behind [`TokenConstraint`] (#3568).
//!
//! `LlgEnv` indexes a vocabulary once: a byte trie over every token, with each
//! special token prefixed by `TokTrie::SPECIAL_TOKEN_MARKER` so it can never
//! match constrained text. `LlgConstraint` is one request's matcher.
//!
//! The error mapping is structural, not a reading of message text.
//! `load_schema` reports input that is not a schema at all as `SchemaInvalid`.
//! Anything the engine then fails to compile is `SchemaUnsupported`, with the
//! engine's own message verbatim, because llguidance returns untyped errors, and
//! sorting them by substring would be a text match standing in for a type.

use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};

use super::{ConstraintError, ConstraintVocab, TokenConstraint};

/// A vocabulary indexed for llguidance, built once per model.
pub struct LlgEnv {
    factory: Arc<ParserFactory>,
}

impl LlgEnv {
    /// Index `vocab`. Special tokens get the marker byte, so no grammar can emit them as text.
    pub fn new(vocab: &ConstraintVocab) -> Result<Self, ConstraintError> {
        vocab.validate()?;
        let words: Vec<Vec<u8>> = vocab
            .token_bytes
            .iter()
            .zip(&vocab.special)
            .map(|(bytes, &special)| {
                if special {
                    let mut marked = Vec::with_capacity(bytes.len() + 1);
                    marked.push(TokTrie::SPECIAL_TOKEN_MARKER);
                    marked.extend_from_slice(bytes);
                    marked
                } else {
                    bytes.clone()
                }
            })
            .collect();
        let vocab_size = u32::try_from(words.len()).map_err(|_| {
            ConstraintError::Vocab(format!("{} tokens do not fit a u32 id", words.len()))
        })?;
        let info = TokRxInfo::new(vocab_size, vocab.eos);
        let trie = TokTrie::from(&info, &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env).map_err(|e| {
            ConstraintError::Vocab(format!("llguidance refused the vocabulary: {e}"))
        })?;
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
        })
    }

    /// One request's constraint for a JSON schema.
    ///
    /// Compact JSON (`,` and `:`, no free whitespace) unless the caller chose its own
    /// `x-guidance`. llguidance's default is flexible whitespace, and that is the known runaway:
    /// a constrained model emits whitespace until `max_tokens`. This module's own case table
    /// measured it, with `{` followed by 63 bytes of whitespace.
    pub fn json_schema(
        &self,
        schema: &serde_json::Value,
    ) -> Result<LlgConstraint, ConstraintError> {
        // A boolean schema carries no x-guidance, so it would compile with flexible whitespace:
        // `true` (any JSON value) becomes `{}`, which means the same and compiles compact, and
        // `false` admits no document at all, which could only ever be a dead end.
        let mut schema = match schema {
            serde_json::Value::Bool(true) => serde_json::json!({}),
            serde_json::Value::Bool(false) => {
                return Err(ConstraintError::SchemaInvalid(
                    "the schema `false` admits no document, so nothing could ever be generated"
                        .to_string(),
                ))
            },
            other => other.clone(),
        };
        if schema.is_object() && schema.get("x-guidance").is_none() {
            llguidance::JsonCompileOptions {
                whitespace_flexible: false,
                ..Default::default()
            }
            .apply_to(&mut schema);
        }
        self.compile(TopLevelGrammar::from_json_schema(schema), "JSON schema")
    }

    /// One request's constraint for a Lark grammar.
    pub fn lark(&self, grammar: &str) -> Result<LlgConstraint, ConstraintError> {
        self.compile(
            TopLevelGrammar::from_lark(grammar.to_string()),
            "Lark grammar",
        )
    }

    fn compile(
        &self,
        grammar: TopLevelGrammar,
        what: &str,
    ) -> Result<LlgConstraint, ConstraintError> {
        let parser = self.factory.create_parser(grammar).map_err(|e| {
            ConstraintError::SchemaUnsupported(format!(
                "the {what} does not compile for constrained decoding: {e}"
            ))
        })?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(ConstraintError::SchemaUnsupported(format!(
                "the {what} does not start: {e}"
            )));
        }
        // An approximation is a weaker constraint than the caller wrote: refused, never used quietly.
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(ConstraintError::SchemaUnsupported(format!(
                "the {what} compiles only approximately: {}",
                warnings.join("; ")
            )));
        }
        Ok(LlgConstraint {
            matcher,
            position: 0,
        })
    }
}

/// One request's constraint: an llguidance matcher and how far it has advanced.
pub struct LlgConstraint {
    matcher: Matcher,
    position: usize,
}

impl TokenConstraint for LlgConstraint {
    fn mask(&mut self, logits: &mut [f32]) -> Result<(), ConstraintError> {
        // llguidance reports "nothing can extend this" as an error (`NoExtensionBias`), not as
        // an empty mask, so both are a dead end: the loop stops, never continues unconstrained.
        let allowed = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|e| ConstraintError::DeadEnd {
                position: self.position,
                reason: e
                    .to_string()
                    .lines()
                    .next()
                    .unwrap_or("no reason given")
                    .to_string(),
            })?;
        let mut any = false;
        for (id, logit) in logits.iter_mut().enumerate() {
            // A logit row past the tokenizer's vocabulary (padded embeddings) is never allowed.
            let ok = id < allowed.len() && u32::try_from(id).is_ok_and(|t| allowed.is_allowed(t));
            if ok {
                any = true;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }
        if any {
            Ok(())
        } else {
            Err(ConstraintError::DeadEnd {
                position: self.position,
                reason: "the mask allows no token".to_string(),
            })
        }
    }

    fn accept(&mut self, token: u32) -> Result<(), ConstraintError> {
        self.matcher
            .consume_token(token)
            .map_err(|e| ConstraintError::Rejected {
                token,
                position: self.position,
                reason: e.to_string(),
            })?;
        self.position += 1;
        Ok(())
    }

    fn is_complete(&mut self) -> bool {
        self.matcher.is_stopped() || self.matcher.is_accepting().unwrap_or(false)
    }
}
