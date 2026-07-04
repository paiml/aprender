//! Falsification tests for the canonical harness-IR round-trip
//! (`contracts/apr-code-harness-ir-v1.yaml`). Each `#[test]` / proptest is
//! referenced by a proof obligation in that contract; together they lift the
//! contract to L2 (falsification tests cover obligations). The honest
//! asymmetries between the wire formats are modelled explicitly:
//!   * Anthropic `tool_use` carries an `id`; Gemini `functionCall` does not.
//!   * Anthropic `tool_result` pairs by `tool_use_id` (no name); Gemini
//!     `functionResponse` pairs by `name` (no id).
//! So per-format round-trip is EXACT on that format's native shape, and the
//! cross-harness equivalence (the keystone) is asserted on the shared core
//! that both formats carry identically: text + tool invocation (name + args).

use super::*;
use proptest::prelude::*;
use serde_json::json;

// ── Fixtures ──

fn bash_call(id: Option<&str>) -> Block {
    Block::ToolCall {
        id: id.map(str::to_string),
        name: "Bash".into(),
        args: json!({ "command": "cargo test", "timeout": 120 }),
    }
}

// ── OBLIG-IR-1: anthropic round-trip is EXACT on anthropic-native messages ──

/// FALSIFY-IR-ANTHROPIC-ROUNDTRIP-001 (unit): a text + tool_use assistant turn
/// survives `from_anthropic ∘ to_anthropic` byte-for-byte.
#[test]
fn falsify_ir_anthropic_roundtrip_001() {
    let m = Message {
        role: Role::Assistant,
        blocks: vec![
            Block::Text("Running the tests.".into()),
            bash_call(Some("toolu_01ABC")),
        ],
    };
    assert_eq!(from_anthropic(&to_anthropic(&m)).unwrap(), m);
}

/// FALSIFY-IR-ANTHROPIC-ROUNDTRIP-001b: a user turn carrying a tool_result
/// (anthropic-native: paired by id, no name) round-trips exactly.
#[test]
fn falsify_ir_anthropic_roundtrip_001b() {
    let m = Message {
        role: Role::User,
        blocks: vec![Block::ToolResult {
            tool_call_id: Some("toolu_01ABC".into()),
            name: String::new(), // anthropic tool_result has no name (pairs by id)
            response: json!({ "stdout": "ok", "code": 0 }),
        }],
    };
    assert_eq!(from_anthropic(&to_anthropic(&m)).unwrap(), m);
}

// ── OBLIG-IR-2: gemini round-trip is EXACT on gemini-native messages ──

/// FALSIFY-IR-GEMINI-ROUNDTRIP-002 (unit): a functionCall + functionResponse
/// pair (gemini-native: no ids, paired by name) round-trips exactly.
#[test]
fn falsify_ir_gemini_roundtrip_002() {
    let call = Message {
        role: Role::Assistant,
        blocks: vec![bash_call(None)], // gemini-native: id=None
    };
    assert!(call.is_gemini_native());
    assert_eq!(from_gemini(&to_gemini(&call)).unwrap(), call);

    let result = Message {
        role: Role::User,
        blocks: vec![Block::ToolResult {
            tool_call_id: None,
            name: "Bash".into(),
            response: json!({ "stdout": "ok" }),
        }],
    };
    assert_eq!(from_gemini(&to_gemini(&result)).unwrap(), result);
}

// ── OBLIG-IR-3: cross-harness SEMANTIC equivalence (the keystone) ──

/// FALSIFY-IR-CROSS-HARNESS-003 (unit): the SAME canonical turn, encoded to
/// Anthropic and to Gemini then decoded back, yields the SAME semantic content.
/// This is "prompt parity works on either harness" at the data layer.
#[test]
fn falsify_ir_cross_harness_003() {
    let m = Message {
        role: Role::Assistant,
        blocks: vec![
            Block::Text("Let me run that.".into()),
            bash_call(Some("toolu_99")), // id present; semantic() strips it
        ],
    };
    let via_anthropic = from_anthropic(&to_anthropic(&m)).unwrap().semantic();
    let via_gemini = from_gemini(&to_gemini(&m)).unwrap().semantic();
    assert_eq!(
        via_anthropic, via_gemini,
        "cross-harness content diverged: anthropic={via_anthropic:?} gemini={via_gemini:?}"
    );
    // and both equal the source semantics
    assert_eq!(via_anthropic, m.semantic());
}

// ── OBLIG-IR-4: tool-schema lossless across both formats ──

/// FALSIFY-IR-TOOL-SCHEMA-004 (unit): a tool's JSON-Schema parameters survive
/// both codecs unchanged, and Anthropic `input_schema` == Gemini `parameters`.
#[test]
fn falsify_ir_tool_schema_004() {
    let t = ToolSchema {
        name: "Edit".into(),
        description: "Replace a string in a file".into(),
        parameters: json!({
            "type": "object",
            "required": ["file_path", "old", "new"],
            "properties": {
                "file_path": { "type": "string" },
                "old": { "type": "string" },
                "new": { "type": "string" },
                "count": { "type": "integer", "minimum": 1 }
            },
            "additionalProperties": false
        }),
    };
    assert_eq!(tool_from_anthropic(&tool_to_anthropic(&t)).unwrap(), t);
    assert_eq!(tool_from_gemini(&tool_to_gemini(&t)).unwrap(), t);
    // The schema body is identical across the two wire formats.
    assert_eq!(
        tool_to_anthropic(&t)["input_schema"],
        tool_to_gemini(&t)["parameters"]
    );
}

// ── L5 binding liveness ──

/// L5 binding-liveness gate: binds each function named in
/// `contracts/binding.yaml` for `apr-code-harness-ir-v1` to a typed function
/// pointer. If any bound function is renamed, removed, or its signature drifts,
/// THIS TEST FAILS TO COMPILE — making the binding registry's `status:
/// implemented` claims compile-enforced, not self-declared. Complements the
/// source-scan verification (`pv proof-status --binding … --verify-bindings .`).
#[test]
fn binding_liveness_l5_bound_functions_exist() {
    let _oblig_ir_1: fn(&serde_json::Value) -> Result<Message, String> = from_anthropic;
    let _oblig_ir_2: fn(&serde_json::Value) -> Result<Message, String> = from_gemini;
    let _oblig_ir_3: fn(&Message) -> Message = Message::semantic;
    let _oblig_ir_4: fn(&ToolSchema) -> serde_json::Value = tool_to_anthropic;
}

// ── Proptest generators ──

prop_compose! {
    fn arb_json_scalar()(choice in 0..4usize, s in "[a-z]{1,6}", n in -100i64..100)
        -> serde_json::Value {
        match choice {
            0 => json!(s),
            1 => json!(n),
            2 => json!(n % 2 == 0),
            _ => serde_json::Value::Null,
        }
    }
}

prop_compose! {
    fn arb_args()(k1 in "[a-z]{1,5}", v1 in arb_json_scalar(), k2 in "[a-z]{1,5}", v2 in arb_json_scalar())
        -> serde_json::Value {
        json!({ k1: v1, k2: v2 })
    }
}

fn arb_role() -> impl Strategy<Value = Role> {
    prop_oneof![Just(Role::User), Just(Role::Assistant)]
}

/// A shared-core block: Text or ToolCall — the content both formats carry
/// identically. Used for the cross-harness equivalence obligation.
fn arb_shared_block() -> impl Strategy<Value = Block> {
    prop_oneof![
        "[a-zA-Z0-9 .]{0,20}".prop_map(Block::Text),
        ("[a-zA-Z_]{1,8}", arb_args(), any::<bool>()).prop_map(|(name, args, has_id)| {
            Block::ToolCall {
                id: has_id.then(|| "toolu_x".to_string()),
                name,
                args,
            }
        }),
    ]
}

proptest! {
    /// OBLIG-IR-1: anthropic round-trip is exact on anthropic-native messages
    /// (tool_use with id, tool_result with id + empty name).
    #[test]
    fn prop_anthropic_roundtrip_exact(
        role in arb_role(),
        blocks in prop::collection::vec(arb_shared_block(), 0..5),
    ) {
        let m = Message { role, blocks };
        prop_assert_eq!(from_anthropic(&to_anthropic(&m)).unwrap(), m);
    }

    /// OBLIG-IR-2: gemini round-trip is exact on gemini-native messages
    /// (functionCall without id). We strip any generated id to gemini-native.
    #[test]
    fn prop_gemini_roundtrip_exact(
        role in arb_role(),
        blocks in prop::collection::vec(arb_shared_block(), 0..5),
    ) {
        // Force gemini-native shape (drop anthropic-only ids).
        let m = Message { role, blocks }.semantic();
        prop_assert!(m.is_gemini_native());
        prop_assert_eq!(from_gemini(&to_gemini(&m)).unwrap(), m);
    }

    /// OBLIG-IR-3 (keystone): for any shared-core turn, the Anthropic and Gemini
    /// wire paths decode to the SAME semantic content — prompt parity on either
    /// harness, as a data-layer property over random inputs.
    #[test]
    fn prop_cross_harness_semantic_equivalence(
        role in arb_role(),
        blocks in prop::collection::vec(arb_shared_block(), 0..5),
    ) {
        let m = Message { role, blocks };
        let via_anthropic = from_anthropic(&to_anthropic(&m)).unwrap().semantic();
        let via_gemini = from_gemini(&to_gemini(&m)).unwrap().semantic();
        prop_assert_eq!(&via_anthropic, &via_gemini);
        prop_assert_eq!(via_anthropic, m.semantic());
    }

    /// OBLIG-IR-4: tool-schema parameters are byte-identical across both wire
    /// formats and survive both round-trips.
    #[test]
    fn prop_tool_schema_lossless(
        name in "[a-zA-Z_]{1,8}",
        desc in "[a-zA-Z ]{0,20}",
        params in arb_args(),
    ) {
        let t = ToolSchema { name, description: desc, parameters: params };
        prop_assert_eq!(tool_from_anthropic(&tool_to_anthropic(&t)).unwrap(), t.clone());
        prop_assert_eq!(tool_from_gemini(&tool_to_gemini(&t)).unwrap(), t.clone());
        prop_assert_eq!(tool_to_anthropic(&t)["input_schema"].clone(), tool_to_gemini(&t)["parameters"].clone());
    }
}
