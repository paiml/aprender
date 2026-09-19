//! aprender-mcp-setfit — a THIN, single-model MCP server for SetFit classification.
//!
//! # One model, one tool, on purpose
//!
//! This crate is deliberately NOT a general-purpose ML server. The design rule
//! it ships under: business analysts curate MCP connectors to the exact business
//! process, so a server that wraps a machine learning model serves ONE model
//! (at most two) behind one task-shaped tool. The general developer-toolchain
//! surface (ten tools, subprocess wrappers) is `crates/aprender-mcp` — this
//! crate is the template for the curated, deployable kind.
//!
//! # In-process, inside the sanctioned exception
//!
//! Classification calls `aprender-core`'s `VerifiedSetFitModel::classify`
//! directly — the same one implementation `apr predict`, `POST /v1/classify`
//! and `apr.predict`-over-MCP all route to (Phase 4 D-09, OPS-03). The model
//! loads once at startup through `load_setfit_apr`'s eight-rung verification
//! ladder and stays warm; there is no subprocess and no second inference path.
//!
//! # Bounds ownership
//!
//! [`MAX_BATCH_TEXTS`] is enforced by core inside `classify` (re-stated here
//! only for a friendlier error). [`MAX_REQUEST_BODY_BYTES`] is a TRANSPORT
//! bound owed by the reading surface — this server IS a reading surface, so
//! [`precheck`] measures the serialized request document against it, exactly
//! as `apr predict --input` bounds the file it reads and `/v1/classify`
//! bounds the HTTP body.

use std::path::Path;
use std::sync::Arc;

use aprender::setfit::{
    load_setfit_apr, read_setfit_apr_bytes_bounded, ClassifyError, ClassifyRequestDocument,
    SetFitArtifactError, VerifiedSetFitModel, MAX_BATCH_TEXTS,
};
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::Server;

// Transport crates (the Lambda wrapper) hold the loaded model in their state;
// re-exported so they depend on this crate alone, not on aprender-core.
pub use aprender::setfit::VerifiedSetFitModel as Model;

// The contract's request-body bound, re-exported for the same reason: a
// transport must configure it on the surface that reads the bytes, and the
// Lambda crate has no aprender-core dependency to reach it through.
pub use aprender::setfit::MAX_REQUEST_BODY_BYTES;

/// The one tool this server advertises.
pub const TOOL_NAME: &str = "classify";

/// The identity this server reports in `initialize`, and in the Lambda health
/// payload. Lives here rather than in each binary: it is what clients key on
/// and what `scenarios/*.yaml` are written against, so the stdio binary and the
/// deployed Lambda must not be able to advertise different names.
pub const SERVER_NAME: &str = "aprender-setfit-predict";

/// The environment variable naming the model artifact.
///
/// Read at BUILD time by the Lambda's `build.rs` (to embed the bytes) and at
/// RUN time by both transports (as a path). Three readers across two crates and
/// two phases, so the spelling is owned once — a mismatch fails silently, by
/// falling through to a door that then reports the wrong reason.
pub const ENV_MODEL: &str = "APRENDER_SETFIT_MODEL";

/// Tool description served on `tools/list`.
pub const TOOL_DESCRIPTION: &str = "Classify texts with the SetFit classifier this server was \
     deployed with. Each element of `texts` is ONE complete document (e.g. one whole social-media \
     post) and yields exactly one classification — NEVER split a single document into multiple \
     elements (fragments classify worse than the whole) and never join separate documents into \
     one element. Long texts are handled by the model itself (truncation is reported per result). \
     Returns one result per element, in input order: label, per-class probabilities, margin, \
     token_count, truncated. Bounded at 256 texts and ~1 MiB of request body per call.";

/// The MCP argument surface of [`TOOL_NAME`].
///
/// Mirrors [`ClassifyRequestDocument`] field-for-field — including
/// `deny_unknown_fields`, so an unmodeled key is a rejection on THIS surface
/// too, not a silently ignored knob. The mirror exists only because the MCP
/// schema needs a [`JsonSchema`] derive that core's document does not carry;
/// [`ClassifyArgs::into_document`] is the total conversion between them, and
/// a unit test pins the two shapes against each other.
pub use args::ClassifyArgs;

mod args {
    // schemars' JsonSchema derive expands to .unwrap() internally, and the
    // generated impl lands at module scope where a struct-level allow cannot
    // reach it. Scoped to this module ON PURPOSE: aprender-mcp's precedent puts
    // this attribute on individual tool MODULES, so at a crate root it would
    // lift the repo-wide `unwrap()` ban off both loader doors, `precheck` and
    // `build_server` — everything the ban exists to protect.
    #![allow(clippy::disallowed_methods)]

    use super::ClassifyRequestDocument;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct ClassifyArgs {
        /// The ordered texts to classify; order is response order. One COMPLETE
        /// document per element (a whole post, comment, or message) — do not split
        /// a document across elements, and do not concatenate documents into one.
        pub texts: Vec<String>,
        /// Include per-class logits in each result. Defaults to false.
        #[serde(default)]
        pub include_logits: bool,
    }

    impl ClassifyArgs {
        /// Convert into the ONE request document every aprender surface parses.
        #[must_use]
        pub fn into_document(self) -> ClassifyRequestDocument {
            let mut document = ClassifyRequestDocument::new(self.texts);
            if self.include_logits {
                document = document.with_logits();
            }
            document
        }
    }
}

/// Why a model failed to load at startup.
///
/// Crate-local rather than re-using core's [`SetFitArtifactError`] on purpose:
/// core's is `#[non_exhaustive]`, so a new variant there would otherwise become
/// a breaking change in every transport's public error surface.
#[derive(Debug, thiserror::Error)]
pub enum ModelLoadError {
    /// The artifact file could not be opened or statted.
    #[error("cannot read model artifact: {0}")]
    Io(#[from] std::io::Error),
    /// The bytes were read but refused by the verification ladder.
    #[error("model artifact refused: {0}")]
    Artifact(#[from] SetFitArtifactError),
    /// Neither door was open: nothing embedded, and no path configured.
    ///
    /// A distinct variant because it is the likeliest deployment
    /// misconfiguration and it is not a filesystem failure. Reporting it as a
    /// synthetic `io::ErrorKind::NotFound` printed it as "cannot read model
    /// artifact", which sends the reader looking for a file that was never
    /// named.
    #[error("no model: this build embedded none and {ENV_MODEL} is unset")]
    NoModel,
}

/// Load and fully verify a `setfit-apr-v1` artifact from a file path.
///
/// The read is bounded BEFORE the bytes land in memory
/// (`read_setfit_apr_bytes_bounded` with the file's declared length), then the
/// bytes go through `load_setfit_apr`'s verification ladder. This is the only
/// load door; a server must never mint its own.
///
/// # Errors
///
/// [`ModelLoadError::Io`] if the file cannot be opened or statted;
/// [`ModelLoadError::Artifact`] for an oversized read or any rung of the
/// verification ladder refusing the artifact.
pub fn load_model_from_path(path: &Path) -> Result<VerifiedSetFitModel, ModelLoadError> {
    let file = std::fs::File::open(path)?;
    let declared_len = file.metadata()?.len();
    let bytes = read_setfit_apr_bytes_bounded(file, Some(declared_len))?;
    Ok(load_setfit_apr(&bytes)?)
}

/// Load and fully verify a `setfit-apr-v1` artifact from bytes already in
/// memory — the door a deployment that compiles the model into the binary
/// (`include_bytes!`) walks through.
///
/// # Errors
///
/// [`ModelLoadError::Artifact`] for any rung of the verification ladder
/// refusing the artifact.
pub fn load_model_from_bytes(bytes: &[u8]) -> Result<VerifiedSetFitModel, ModelLoadError> {
    Ok(load_setfit_apr(bytes)?)
}

/// Validate a request and convert it to the shared document.
///
/// Order matters and is the same order every other surface applies:
/// batch bound first (friendlier error than core's, but core re-checks),
/// then the transport byte bound this server owes as a reading surface.
///
/// # Errors
///
/// `pmcp::Error::validation` naming the violated bound.
pub fn precheck(args: ClassifyArgs) -> pmcp::Result<ClassifyRequestDocument> {
    if args.texts.len() > MAX_BATCH_TEXTS {
        return Err(pmcp::Error::validation(format!(
            "batch of {} texts exceeds max_batch_texts {MAX_BATCH_TEXTS} \
             (contracts/setfit-apr-v1.yaml item 11); split the batch",
            args.texts.len()
        )));
    }
    let document = args.into_document();
    let body_bytes = serde_json::to_vec(&document)
        .map_err(|e| pmcp::Error::internal(format!("request document serialization: {e}")))?
        .len() as u64;
    if body_bytes > MAX_REQUEST_BODY_BYTES {
        return Err(pmcp::Error::validation(format!(
            "request document is {body_bytes} bytes, over max_request_body_bytes \
             {MAX_REQUEST_BODY_BYTES} (contracts/setfit-apr-v1.yaml item 11); send less text"
        )));
    }
    Ok(document)
}

/// Map a classify failure onto the MCP error taxonomy.
///
/// Input-bound refusals are validation errors (the caller can fix them);
/// everything else — encode/head failures, envelope refusals — is internal.
fn classify_error(error: &ClassifyError) -> pmcp::Error {
    match error {
        ClassifyError::EmptyInput | ClassifyError::BatchTooLarge { .. } => {
            pmcp::Error::validation(error.to_string())
        }
        other => pmcp::Error::internal(other.to_string()),
    }
}

/// Build the MCP server: one classify tool over one loaded model.
///
/// The classify call runs on a blocking thread (`spawn_blocking`) because a
/// 256-text batch is seconds of CPU-bound encoder work and must not stall the
/// async protocol loop.
///
/// # Errors
///
/// `pmcp::Error` if the server builder refuses the configuration.
pub fn build_server(
    model: Arc<VerifiedSetFitModel>,
    name: &str,
    version: &str,
) -> pmcp::Result<Server> {
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool_typed_with_description::<ClassifyArgs, _, _>(
            TOOL_NAME,
            TOOL_DESCRIPTION,
            move |args, _extra| {
                let model = Arc::clone(&model);
                async move {
                    let document = precheck(args)?;
                    let response = tokio::task::spawn_blocking(move || model.classify(&document))
                        .await
                        .map_err(|e| pmcp::Error::internal(format!("classify task join: {e}")))?
                        .map_err(|e| classify_error(&e))?;
                    serde_json::to_value(&response)
                        .map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
                }
            },
        )
        .build()
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally
mod tests {
    use super::*;

    #[test]
    fn an_unknown_argument_key_is_refused_not_ignored() {
        let error = serde_json::from_value::<ClassifyArgs>(serde_json::json!({
            "texts": ["a"],
            "temperature": 0.7
        }))
        .expect_err("an unmodeled key must be a rejection");
        assert!(
            error.to_string().contains("unknown field"),
            "the refusal must name the defect: {error}"
        );
    }

    #[test]
    fn args_convert_to_the_one_shared_document() {
        let args = serde_json::from_value::<ClassifyArgs>(serde_json::json!({
            "texts": ["first", "second"],
            "include_logits": true
        }))
        .expect("well-formed args");
        let expected = ClassifyRequestDocument::new(["first", "second"]).with_logits();
        assert_eq!(args.into_document(), expected);
    }

    #[test]
    fn include_logits_defaults_to_false_like_every_other_surface() {
        let args = serde_json::from_value::<ClassifyArgs>(serde_json::json!({
            "texts": ["a"]
        }))
        .expect("omitting include_logits is legal");
        assert!(!args.include_logits);
        let expected = ClassifyRequestDocument::new(["a"]);
        assert_eq!(args.into_document(), expected);
    }

    #[test]
    fn a_batch_over_the_contract_bound_is_refused_before_any_work() {
        let args = ClassifyArgs {
            texts: vec![String::from("x"); MAX_BATCH_TEXTS + 1],
            include_logits: false,
        };
        let error = precheck(args).expect_err("257 texts must be refused");
        assert!(
            error.to_string().contains("max_batch_texts"),
            "the refusal must name the bound: {error}"
        );
    }

    #[test]
    fn a_single_oversized_text_is_refused_by_the_byte_bound() {
        // One text over 1 MiB: legal on the count bound, illegal on the byte
        // bound. This is the case argv-based transport could never even carry.
        let args = ClassifyArgs {
            texts: vec!["y".repeat(1_100_000)],
            include_logits: false,
        };
        let error = precheck(args).expect_err("an oversized document must be refused");
        assert!(
            error.to_string().contains("max_request_body_bytes"),
            "the refusal must name the bound: {error}"
        );
    }

    #[test]
    fn a_maximal_legal_request_passes_precheck() {
        // 256 texts just under the byte bound together: both bounds satisfied.
        let args = ClassifyArgs {
            texts: vec!["z".repeat(3_500); MAX_BATCH_TEXTS],
            include_logits: false,
        };
        let document = precheck(args).expect("a contract-legal batch must pass");
        assert_eq!(document.texts.len(), MAX_BATCH_TEXTS);
    }

    #[test]
    fn the_tool_schema_is_strict_and_names_both_fields() {
        let schema =
            serde_json::to_value(schemars::schema_for!(ClassifyArgs)).expect("schema serializes");
        assert_eq!(
            schema["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must surface in the advertised schema"
        );
        let required = schema["required"].as_array().expect("required array");
        assert!(required.contains(&serde_json::json!("texts")));
        assert!(
            !required.contains(&serde_json::json!("include_logits")),
            "include_logits is optional on every surface"
        );
    }
}
