//! `apr.predict` — subprocess wrapper over `apr predict <model> --input <doc> --json`.
//!
//! Follows the M2 pattern established by `apr.validate`: spawn the CLI with
//! `--json`, pass stdout through verbatim, map non-zero exit to `isError`.
//!
//! Deliberately GENERIC, mirroring the CLI's D-06 decision: a `setfit-apr-v1`
//! classifier is an APR like any other, so it is predicted with the same
//! command as any other model. Routing reads the artifact's typed `model_type`
//! tag and NEVER tensor names, which means a file that merely *looks* like a
//! classifier is refused by the CLI rather than silently scored — that refusal
//! reaches the MCP client unchanged as `isError: true`.
//!
//! Batch shape is intentional. `ClassifyResponse` is batch-shaped and the CLI
//! deliberately rejects line-delimited input (a line-delimited file cannot
//! carry a text containing a newline, so CLI and HTTP would receive different
//! ordered inputs while appearing to agree). One call classifies N texts;
//! response order is request order.
//!
//! # The batch goes through `--input`, NOT one `--text=` per text
//!
//! argv is not a body, and the OS bounds it. `ARG_MAX` on the development host
//! measures 1,048,576 bytes — the SAME number as the contract's
//! `max_request_body_bytes` — and the process environment is charged against
//! that same budget. So a batch that is legal on BOTH contract bounds
//! (`max_batch_texts` = 256, `max_request_body_bytes` = 1 MiB) could not be
//! spawned. Measured, with `--text=` argv:
//!
//! ```text
//! 256 x 4050B | doc 1,037,881B legal | argv 1,038,879B spawn OK
//! 256 x 4080B | doc 1,045,561B legal | argv 1,046,559B spawn E2BIG
//! ```
//!
//! Two properties of that failure make it worse than a size limit: the client
//! saw `Argument list too long` rather than the contract's typed refusal, and
//! the threshold MOVED with the size of the environment, so no fixed number
//! documented here would have been true.
//!
//! `--input` is the door the contract already built for this. It carries the
//! same `ClassifyRequestDocument` that `POST /v1/classify` accepts, so the two
//! surfaces receive byte-identical input; the CLI stats and bounds the file
//! against `MAX_REQUEST_BODY_BYTES` before parsing it. This crate does NOT
//! restate that number — it has no dependency on `aprender-core` and adding one
//! for a single constant would make a thin subprocess wrapper depend on the ML
//! library. Core states plainly that the bound is owed by the READING surface;
//! this is a writing surface, and the reader it hands the file to enforces it.
//!
//! Passing texts as JSON values also removes them from clap's sight entirely,
//! which is what makes a text beginning with `-` representable at all.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use std::io::Write;

use crate::tools::subprocess::run_apr;
use crate::types::{InputSchema, ToolCallResult, ToolDefinition};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.predict";

/// Return the MCP tool definition for `apr.predict`.
///
/// FALSIFY-MCP-008: `inputSchema` and `description` come from build-time
/// codegen constants emitted from `contracts/apr-mcp-tool-schemas-v1.yaml`.
/// Neither may be hand-coded here.
#[must_use]
pub fn predict_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_PREDICT_SCHEMA)
        .expect(
            "FALSIFY-MCP-008: apr.predict codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
        );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_PREDICT_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// A validated call, split into the two things the spawn needs.
///
/// Separated from [`call`] so the request SHAPE is unit-testable without
/// spawning anything or touching the filesystem.
struct PredictRequest {
    /// The model path as it must appear in argv.
    model_arg: String,
    /// The shared `ClassifyRequestDocument`, ready to serialize.
    document: serde_json::Value,
}

/// Validate the MCP arguments into a [`PredictRequest`], or return the
/// client-facing refusal.
fn build_request(args: &serde_json::Value) -> Result<PredictRequest, String> {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return Err("Missing required argument: model_path".to_string());
    };
    let Some(texts) = args.get("texts").and_then(|v| v.as_array()) else {
        return Err("Missing required argument: texts (array of strings)".to_string());
    };
    if texts.is_empty() {
        // An empty batch is a client bug, not an empty result: core refuses it
        // with `ClassifyError::EmptyInput` anyway, and refusing here saves a
        // spawn and a temp file to reach the same verdict.
        return Err("Argument `texts` must contain at least one text".to_string());
    }
    let include_logits = args
        .get("include_logits")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let mut collected: Vec<&str> = Vec::with_capacity(texts.len());
    for (index, value) in texts.iter().enumerate() {
        // Reject a non-string element instead of lossily stringifying it — a
        // silently coerced `42` would be classified as the literal "42" and the
        // caller would never learn their input was not what they sent.
        let Some(text) = value.as_str() else {
            return Err(format!(
                "Argument `texts[{index}]` must be a string, got: {value}"
            ));
        };
        collected.push(text);
    }

    Ok(PredictRequest {
        // A relative path beginning with `-` is a clap FLAG, not a positional.
        // `./` makes it a path again without naming a different file. The texts
        // no longer need this treatment — they are JSON values now, not argv —
        // but the model path is still a positional argument.
        model_arg: if model_path.starts_with('-') {
            format!("./{model_path}")
        } else {
            model_path.to_string()
        },
        // `include_logits` is set in the DOCUMENT rather than by passing
        // `--logits`. On the `--input` path the CLI flag can only turn the flag
        // ON (it never downgrades a document that asked for logits), so a
        // document carrying the client's actual choice is the only spelling
        // that can express `false` as well as `true`.
        document: serde_json::json!({
            "texts": collected,
            "include_logits": include_logits,
        }),
    })
}

/// The argv for a prepared request. Fixed arity: FIVE elements, whatever the
/// batch size, which is the whole point of the `--input` door.
fn argv_for(model_arg: &str, input_path: &str) -> Vec<String> {
    vec![
        "predict".to_string(),
        model_arg.to_string(),
        "--input".to_string(),
        input_path.to_string(),
        "--json".to_string(),
    ]
}

/// Execute `apr.predict` by spawning `apr predict <model> --input <doc> --json`.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let request = match build_request(args) {
        Ok(request) => request,
        Err(message) => return ToolCallResult::error(message),
    };

    let body = match serde_json::to_string(&request.document) {
        Ok(body) => body,
        Err(error) => {
            return ToolCallResult::error(format!(
                "the classify request document did not serialize: {error}"
            ))
        }
    };

    // `tempfile` creates with mode 0600 and an unpredictable name: the batch is
    // client text and must not become world-readable in /tmp for the lifetime
    // of the call. The handle is held until after `run_apr` returns and deletes
    // the file on drop, so no cleanup path can be missed on an error return.
    let mut file = match tempfile::Builder::new()
        .prefix("apr-predict-")
        .suffix(".json")
        .tempfile()
    {
        Ok(file) => file,
        Err(error) => {
            return ToolCallResult::error(format!(
                "could not create the classify request document: {error}"
            ))
        }
    };
    if let Err(error) = file.write_all(body.as_bytes()) {
        return ToolCallResult::error(format!(
            "could not write the classify request document: {error}"
        ));
    }
    if let Err(error) = file.flush() {
        return ToolCallResult::error(format!(
            "could not flush the classify request document: {error}"
        ));
    }

    // The CLI takes a path, so a non-UTF-8 temp dir must be a typed refusal
    // rather than a lossy conversion that names a DIFFERENT file.
    let Some(input_path) = file.path().to_str() else {
        return ToolCallResult::error(
            "the temporary directory's path is not valid UTF-8, so the classify request \
             document cannot be named on the command line"
                .to_string(),
        );
    };

    let argv = argv_for(&request.model_arg, input_path);
    let borrowed: Vec<&str> = argv.iter().map(String::as_str).collect();
    run_apr(&borrowed)
}

/// HELIX-IDEA-002 — unified-signature shim for the inventory dispatcher.
pub fn dispatch(
    args: &serde_json::Value,
    _cancel: &std::sync::mpsc::Receiver<()>,
    _sink: Option<&crate::server::NotificationSink>,
    _token: Option<serde_json::Value>,
) -> ToolCallResult {
    call(args)
}

crate::register_mcp_tool!(
    name: NAME,
    definition: predict_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_fields() {
        let def = predict_tool_definition();
        assert_eq!(def.name, "apr.predict");
        assert_eq!(def.input_schema.schema_type, "object");
        assert!(def.input_schema.properties.contains_key("model_path"));
        assert!(def.input_schema.properties.contains_key("texts"));
        assert!(def
            .input_schema
            .required
            .contains(&"model_path".to_string()));
        assert!(def.input_schema.required.contains(&"texts".to_string()));
    }

    /// The `texts` schema must declare `items`; an array without it leaves the
    /// element type undefined for every client.
    #[test]
    fn texts_schema_declares_string_items() {
        let def = predict_tool_definition();
        let texts = def
            .input_schema
            .properties
            .get("texts")
            .expect("texts property must exist");
        assert_eq!(texts.prop_type, "array");
        let items = texts
            .items
            .as_ref()
            .expect("array schema must declare items");
        assert_eq!(items.item_type, "string");
    }

    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({ "texts": ["hello"] }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("model_path"));
    }

    #[test]
    fn missing_texts_returns_error() {
        let result = call(&serde_json::json!({ "model_path": "m.apr" }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("texts"));
    }

    #[test]
    fn empty_texts_is_refused_not_spawned() {
        let result = call(&serde_json::json!({ "model_path": "m.apr", "texts": [] }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("at least one"));
    }

    /// The batch must not reach argv AT ALL — that is the entire change.
    ///
    /// Regression guard for a MEASURED failure: with one `--text=` per text,
    /// 256 texts of 4080 bytes built a 1,046,559-byte argv and `execve` refused
    /// it with E2BIG, even though the same batch is legal on both contract
    /// bounds (`max_batch_texts` = 256, `max_request_body_bytes` = 1 MiB) and
    /// classifies fine over HTTP. `ARG_MAX` is 1,048,576 on the development
    /// host and the environment is charged against it too, so the ceiling was
    /// not even a fixed number.
    ///
    /// Asserted as a PROPERTY rather than a threshold: argv is fixed-arity and
    /// its size depends only on the two path lengths, so no batch can reach it.
    #[test]
    fn a_maximal_batch_does_not_reach_argv() {
        let texts: Vec<String> = (0..256).map(|_| "x".repeat(4080)).collect();
        let request = build_request(&serde_json::json!({
            "model_path": "m.apr",
            "texts": texts,
        }))
        .expect("a 256-text batch is legal");

        let document = serde_json::to_string(&request.document).expect("document serializes");
        assert!(
            document.len() > 1_000_000,
            "this test is vacuous unless the document is genuinely near the 1 MiB bound: {}",
            document.len()
        );

        let argv = argv_for(&request.model_arg, "/tmp/apr-predict-abcdef.json");
        assert_eq!(argv.len(), 5, "argv arity must not depend on batch size");
        let argv_bytes: usize = argv.iter().map(|a| a.len() + 1).sum();
        assert!(
            argv_bytes < 4096,
            "argv must stay tiny however large the batch is, got {argv_bytes} bytes: {argv:?}"
        );
    }

    /// A text beginning with `-` must survive to the model.
    ///
    /// It used to die in clap: `apr predict m.apr --text "-1 star"` exits with
    /// `error: unexpected argument '-1' found`, because `--text` does not set
    /// `allow_hyphen_values`. Carried as a JSON value in the request document
    /// it is never parsed as argv at all, which is a stronger guarantee than
    /// the `--text=` spelling it replaces.
    #[test]
    fn hyphen_leading_text_is_carried_as_a_json_value() {
        let request = build_request(&serde_json::json!({
            "model_path": "m.apr",
            "texts": ["-1 star", "--> awful"],
        }))
        .expect("valid arguments");
        assert_eq!(
            request.document["texts"],
            serde_json::json!(["-1 star", "--> awful"]),
            "texts must reach the document verbatim and in request order"
        );

        let argv = argv_for(&request.model_arg, "/tmp/doc.json");
        assert!(
            !argv.iter().any(|a| a.starts_with("--text")),
            "no text may appear in argv in any spelling: {argv:?}"
        );
    }

    /// A model path beginning with `-` is a clap flag unless it is re-anchored.
    /// Unlike the texts, it is still a positional argument.
    #[test]
    fn hyphen_leading_model_path_is_re_anchored() {
        let request = build_request(&serde_json::json!({
            "model_path": "-weird.apr",
            "texts": ["ok"],
        }))
        .expect("valid arguments");
        assert_eq!(request.model_arg, "./-weird.apr");
    }

    /// `include_logits` rides in the document, not as `--logits`.
    ///
    /// On the `--input` path the CLI flag can only turn the flag ON, so passing
    /// `--logits` could never express `false` — only the document can.
    #[test]
    fn include_logits_rides_in_the_document() {
        let with = build_request(&serde_json::json!({
            "model_path": "m.apr",
            "texts": ["ok"],
            "include_logits": true,
        }))
        .expect("valid arguments");
        assert_eq!(with.document["include_logits"], serde_json::json!(true));

        let without = build_request(&serde_json::json!({
            "model_path": "m.apr",
            "texts": ["ok"],
        }))
        .expect("valid arguments");
        assert_eq!(without.document["include_logits"], serde_json::json!(false));

        let argv = argv_for(&without.model_arg, "/tmp/doc.json");
        assert!(!argv.iter().any(|a| a == "--logits"));
    }

    /// The argv shape the contract's `cli_mapping` documents.
    #[test]
    fn argv_is_the_documented_shape() {
        let argv = argv_for("m.apr", "/tmp/doc.json");
        assert_eq!(
            argv,
            vec!["predict", "m.apr", "--input", "/tmp/doc.json", "--json"]
        );
    }

    /// The document must be EXACTLY the shape the CLI's `--input` reader takes.
    ///
    /// `ClassifyRequestDocument` is `#[serde(deny_unknown_fields)]`
    /// (aprender-core `setfit/classify.rs:191`), so one extra key here is a
    /// hard rejection at the reader rather than an ignored knob — and this
    /// crate cannot catch that at compile time, because it deliberately does
    /// not depend on `aprender-core`. The literal below is the same one
    /// apr-cli's own `--input` test feeds
    /// (`crates/apr-cli/src/commands/predict_tests.rs:259`), so a drift in
    /// either direction turns this red.
    #[test]
    fn the_document_is_byte_compatible_with_the_cli_reader() {
        let request = build_request(&serde_json::json!({
            "model_path": "m.apr",
            "texts": ["a"],
            "include_logits": true,
        }))
        .expect("valid arguments");
        // Compared as PARSED values, not as bytes. `serde_json::Map` is a
        // BTreeMap without the `preserve_order` feature, so this renders with
        // its keys sorted (`include_logits` first) while apr-cli's literal is
        // written in declaration order. Object key order carries no meaning to
        // any JSON reader; asserting on it would have pinned a detail that is
        // not the claim and gone red on a serde version bump.
        let rendered = serde_json::to_string(&request.document).expect("document serializes");
        let reparsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("what we write must parse back");
        let canonical: serde_json::Value =
            serde_json::from_str(r#"{"texts":["a"],"include_logits":true}"#)
                .expect("apr-cli's own --input literal parses");
        assert_eq!(reparsed, canonical);

        let object = request
            .document
            .as_object()
            .expect("the document is a JSON object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["include_logits", "texts"],
            "deny_unknown_fields means any additional key is a refusal, not a no-op"
        );
    }

    /// A non-string element must be refused, never coerced: a silently
    /// stringified `42` would be classified as the literal "42".
    #[test]
    fn non_string_text_element_is_refused() {
        let result = call(&serde_json::json!({ "model_path": "m.apr", "texts": ["ok", 42] }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("texts[1]"));
    }
}
