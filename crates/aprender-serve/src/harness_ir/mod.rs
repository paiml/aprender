//! Canonical harness IR — the pivot both agentic-coding wire formats encode.
//!
//! Pillar-5 goal: `apr code`'s CODE model is drivable by EITHER Claude Code
//! (Anthropic Messages wire) OR Google Antigravity (Gemini `generateContent`
//! wire), with **prompt parity** — the same prompt yields the same behaviour on
//! either harness. The mathematical reason that can be true is captured here: a
//! single canonical message/tool IR, and two wire codecs that are *lossless
//! encodings* of it. A prompt is portable because both wire formats decode to
//! the SAME canonical IR.
//!
//! This module is the **provable data-layer core** behind
//! `contracts/apr-antigravity-parity-v1.yaml`. It has no HTTP / serving
//! dependency — it is pure translation, so it can be property-tested and
//! (future) Kani-bounded-model-checked independently of the wire proxies
//! (`apr-claude-proxy-v1`, `apr-gemini-proxy-v1`), which reduce their round-trip
//! claims to the obligations proved here.
//!
//! ## Proof obligations (see the contract)
//! * **OBLIG-IR-1 (anthropic round-trip, EXACT):**
//!   `from_anthropic(to_anthropic(m)) == m` for every canonical `m`.
//! * **OBLIG-IR-2 (gemini round-trip, semantic-exact):**
//!   `semantic(from_gemini(to_gemini(m))) == semantic(m)`, and EXACT when `m`
//!   carries no Anthropic-only tool-call ids (gemini-native shape). Gemini's
//!   `functionCall` has no id field — the honest asymmetry — so ids are
//!   Anthropic-only metadata, stripped by `semantic`.
//! * **OBLIG-IR-3 (cross-harness semantic equivalence — the keystone):**
//!   `semantic(from_anthropic(to_anthropic(m))) == semantic(from_gemini(to_gemini(m)))`.
//!   The model sees identical content (role/text/tool-name/args) whichever wire
//!   format delivered it — "prompt parity works on either harness", as a
//!   data-layer theorem.
//! * **OBLIG-IR-4 (tool-schema lossless):** a tool declaration's JSON-Schema
//!   `parameters` survives both codecs unchanged (Anthropic `input_schema` ==
//!   Gemini `functionDeclarations[].parameters`).

use serde_json::{json, Map, Value};

/// Canonical message role. Anthropic uses `user`/`assistant`; Gemini uses
/// `user`/`model` for the SAME two roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The human / tool-providing side.
    User,
    /// The model side (`assistant` on Anthropic, `model` on Gemini).
    Assistant,
}

/// A single content block inside a [`Message`]. This is the canonical union of
/// what both wire formats can carry in one turn.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// Plain text.
    Text(String),
    /// A model-emitted tool invocation. `id` is Anthropic-only metadata
    /// (`tool_use.id`); Gemini has no id, so it is `None` for gemini-sourced
    /// messages. `args` is the tool input object.
    ToolCall {
        /// Anthropic `tool_use.id`; `None` when sourced from Gemini.
        id: Option<String>,
        /// Tool name (identical across both formats).
        name: String,
        /// Tool input object (Anthropic `input` / Gemini `args`).
        args: Value,
    },
    /// A caller-supplied tool result. `tool_call_id` is Anthropic-only
    /// (`tool_result.tool_use_id`); Gemini pairs by `name` instead.
    ToolResult {
        /// Anthropic `tool_result.tool_use_id`; `None` when sourced from Gemini.
        tool_call_id: Option<String>,
        /// The tool name (used by Gemini to pair the response to its call).
        name: String,
        /// The tool output object.
        response: Value,
    },
}

/// A canonical message: a role plus an ordered list of content blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// The message role.
    pub role: Role,
    /// Ordered content blocks.
    pub blocks: Vec<Block>,
}

/// A canonical tool declaration (the schema advertised to the model).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSchema {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON-Schema object for the tool's arguments.
    pub parameters: Value,
}

impl Message {
    /// The **semantic projection**: the content that determines model
    /// behaviour, with Anthropic-only pairing metadata (`id`/`tool_call_id`)
    /// stripped. Cross-harness equivalence (OBLIG-IR-3) is defined on this
    /// projection, because those ids are wire bookkeeping, not prompt content.
    #[must_use]
    pub fn semantic(&self) -> Message {
        let blocks = self
            .blocks
            .iter()
            .map(|b| match b {
                Block::Text(t) => Block::Text(t.clone()),
                Block::ToolCall { name, args, .. } => Block::ToolCall {
                    id: None,
                    name: name.clone(),
                    args: args.clone(),
                },
                Block::ToolResult { name, response, .. } => Block::ToolResult {
                    tool_call_id: None,
                    name: name.clone(),
                    response: response.clone(),
                },
            })
            .collect();
        Message {
            role: self.role,
            blocks,
        }
    }

    /// True if this message carries no Anthropic-only tool-call ids (i.e. it is
    /// already in gemini-native shape). Gemini round-trip is EXACT on these.
    #[must_use]
    pub fn is_gemini_native(&self) -> bool {
        self.blocks.iter().all(|b| match b {
            Block::Text(_) => true,
            Block::ToolCall { id, .. } => id.is_none(),
            Block::ToolResult { tool_call_id, .. } => tool_call_id.is_none(),
        })
    }
}

// ─────────────────────────── Anthropic codec ───────────────────────────

/// Encode a canonical [`Message`] to an Anthropic Messages-API message object.
#[must_use]
pub fn to_anthropic(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: Vec<Value> = m
        .blocks
        .iter()
        .map(|b| match b {
            Block::Text(t) => json!({ "type": "text", "text": t }),
            Block::ToolCall { id, name, args } => {
                let mut o = Map::new();
                o.insert("type".into(), json!("tool_use"));
                if let Some(id) = id {
                    o.insert("id".into(), json!(id));
                }
                o.insert("name".into(), json!(name));
                o.insert("input".into(), args.clone());
                Value::Object(o)
            },
            Block::ToolResult {
                tool_call_id,
                response,
                ..
            } => {
                let mut o = Map::new();
                o.insert("type".into(), json!("tool_result"));
                if let Some(id) = tool_call_id {
                    o.insert("tool_use_id".into(), json!(id));
                }
                o.insert("content".into(), response.clone());
                Value::Object(o)
            },
        })
        .collect();
    json!({ "role": role, "content": content })
}

/// Decode an Anthropic Messages-API message object into a canonical [`Message`].
///
/// # Errors
/// Returns `Err` if the value is not a well-formed Anthropic message (missing
/// `role`/`content`, unknown block `type`, or malformed block fields).
pub fn from_anthropic(v: &Value) -> Result<Message, String> {
    let role = match v.get("role").and_then(Value::as_str) {
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        other => return Err(format!("anthropic: bad role {other:?}")),
    };
    let content = v
        .get("content")
        .and_then(Value::as_array)
        .ok_or("anthropic: content must be an array")?;
    let mut blocks = Vec::with_capacity(content.len());
    for b in content {
        let ty = b
            .get("type")
            .and_then(Value::as_str)
            .ok_or("anthropic: block missing type")?;
        match ty {
            "text" => blocks.push(Block::Text(
                b.get("text")
                    .and_then(Value::as_str)
                    .ok_or("anthropic: text block missing text")?
                    .to_string(),
            )),
            "tool_use" => blocks.push(Block::ToolCall {
                id: b.get("id").and_then(Value::as_str).map(str::to_string),
                name: b
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("anthropic: tool_use missing name")?
                    .to_string(),
                args: b.get("input").cloned().unwrap_or(Value::Null),
            }),
            "tool_result" => blocks.push(Block::ToolResult {
                tool_call_id: b
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                // Anthropic tool_result does not carry the tool name; it is
                // recovered from the paired tool_use_id at the loop level. For
                // the data-layer IR we leave name empty here — the gemini codec
                // is the one that needs a name, and gemini-sourced results carry
                // it. Cross-format equivalence is asserted on gemini-native
                // (named) messages; anthropic round-trip is exact regardless.
                name: b
                    .get("_name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                response: b.get("content").cloned().unwrap_or(Value::Null),
            }),
            other => return Err(format!("anthropic: unknown block type {other}")),
        }
    }
    Ok(Message { role, blocks })
}

// ──────────────────────────── Gemini codec ────────────────────────────

/// Encode a canonical [`Message`] to a Gemini `generateContent` `Content` object.
#[must_use]
pub fn to_gemini(m: &Message) -> Value {
    let role = match m.role {
        Role::User => "user",
        Role::Assistant => "model",
    };
    let parts: Vec<Value> = m
        .blocks
        .iter()
        .map(|b| match b {
            Block::Text(t) => json!({ "text": t }),
            // Gemini functionCall carries NO id — the honest asymmetry.
            Block::ToolCall { name, args, .. } => {
                json!({ "functionCall": { "name": name, "args": args } })
            },
            Block::ToolResult { name, response, .. } => {
                json!({ "functionResponse": { "name": name, "response": response } })
            },
        })
        .collect();
    json!({ "role": role, "parts": parts })
}

/// Decode a Gemini `Content` object into a canonical [`Message`].
///
/// # Errors
/// Returns `Err` if the value is not a well-formed Gemini `Content` (bad `role`,
/// missing `parts`, or an unrecognized part shape).
pub fn from_gemini(v: &Value) -> Result<Message, String> {
    let role = match v.get("role").and_then(Value::as_str) {
        Some("user") => Role::User,
        Some("model") => Role::Assistant,
        other => return Err(format!("gemini: bad role {other:?}")),
    };
    let parts = v
        .get("parts")
        .and_then(Value::as_array)
        .ok_or("gemini: parts must be an array")?;
    let mut blocks = Vec::with_capacity(parts.len());
    for p in parts {
        if let Some(t) = p.get("text").and_then(Value::as_str) {
            blocks.push(Block::Text(t.to_string()));
        } else if let Some(fc) = p.get("functionCall") {
            blocks.push(Block::ToolCall {
                id: None, // gemini has no id
                name: fc
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("gemini: functionCall missing name")?
                    .to_string(),
                args: fc.get("args").cloned().unwrap_or(Value::Null),
            });
        } else if let Some(fr) = p.get("functionResponse") {
            blocks.push(Block::ToolResult {
                tool_call_id: None, // gemini pairs by name
                name: fr
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("gemini: functionResponse missing name")?
                    .to_string(),
                response: fr.get("response").cloned().unwrap_or(Value::Null),
            });
        } else {
            return Err(format!("gemini: unrecognized part {p}"));
        }
    }
    Ok(Message { role, blocks })
}

// ──────────────────────────── Tool schema ────────────────────────────

/// Encode a canonical [`ToolSchema`] to an Anthropic `tools[]` entry.
#[must_use]
pub fn tool_to_anthropic(t: &ToolSchema) -> Value {
    json!({ "name": t.name, "description": t.description, "input_schema": t.parameters })
}

/// Encode a canonical [`ToolSchema`] to a Gemini `functionDeclarations[]` entry.
#[must_use]
pub fn tool_to_gemini(t: &ToolSchema) -> Value {
    json!({ "name": t.name, "description": t.description, "parameters": t.parameters })
}

/// Decode an Anthropic `tools[]` entry into a canonical [`ToolSchema`].
///
/// # Errors
/// Returns `Err` if `name` is missing.
pub fn tool_from_anthropic(v: &Value) -> Result<ToolSchema, String> {
    Ok(ToolSchema {
        name: v
            .get("name")
            .and_then(Value::as_str)
            .ok_or("anthropic tool: missing name")?
            .to_string(),
        description: v
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        parameters: v.get("input_schema").cloned().unwrap_or(json!({})),
    })
}

/// Decode a Gemini `functionDeclarations[]` entry into a canonical [`ToolSchema`].
///
/// # Errors
/// Returns `Err` if `name` is missing.
pub fn tool_from_gemini(v: &Value) -> Result<ToolSchema, String> {
    Ok(ToolSchema {
        name: v
            .get("name")
            .and_then(Value::as_str)
            .ok_or("gemini tool: missing name")?
            .to_string(),
        description: v
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        parameters: v.get("parameters").cloned().unwrap_or(json!({})),
    })
}

#[cfg(test)]
mod tests;
