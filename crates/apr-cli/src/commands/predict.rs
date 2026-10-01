//! `apr predict` — the GENERIC prediction surface (D-06, OPS-03).
//!
//! # Generic, not `apr setfit predict`
//!
//! A `setfit-apr-v1` artifact is an APR, so it is predicted with the same command
//! as any other model. Routing is by the artifact's own typed tag
//! ([`crate::setfit_tag`]), which is what makes "one implementation per operation"
//! (OPS-03) a structural property rather than a convention: there is no second
//! spelling of this operation to keep in step.
//!
//! # This file DECLARES NO RESPONSE TYPE
//!
//! `--json` prints `serde_json` of `aprender::setfit::ClassifyResponse` verbatim,
//! and the human rendering is a view over that SAME value through its accessors.
//! A local struct here — even one with identical fields — would be a second wire
//! form for the OPS-04 envelope, and the day core added a field the CLI would keep
//! producing a body that no longer matched the HTTP surface while every test in
//! both crates stayed green. `predict_declares_no_response_type` scans this file.
//!
//! # `--input` is the SHARED REQUEST DOCUMENT, not one text per line
//!
//! The flag parses `{"texts": [...], "include_logits": bool}` — core's
//! [`ClassifyRequestDocument`] — and NOT a line-delimited file. A line-delimited
//! format cannot carry a text containing a newline: the library, the CLI and the
//! HTTP surface would each receive a DIFFERENT ordered input set while appearing to
//! agree, which is exactly the defect the cross-AI review (M2) found in the parity
//! harness. A tab, an empty string and non-ASCII have the same problem in weaker
//! forms. The document is one JSON value, so all four survive verbatim.
//!
//! # Both bounds are applied BEFORE the work they bound
//!
//! The `--input` file is refused above `MAX_REQUEST_BODY_BYTES` from its stat'd
//! length, before serde is handed anything (the CLI half of the enforcement core's
//! constant defers to its reading surfaces), and artifact bytes come through
//! [`crate::setfit_io::read_setfit_apr_file_bounded`] — the ONE bounded artifact
//! door. There is no `fs::read` in this file and a test asserts there is not.

use std::path::{Path, PathBuf};

use crate::error::{CliError, Result};
use crate::setfit_tag;

/// What `predict` can do in v1, named in the refusal so the message is actionable.
const SUPPORTED_FAMILIES: &str = "`apr predict` supports `setfit-apr-v1` classifiers in v1. \
     Routing is by the artifact's typed `model_type` tag and NEVER by tensor names, so a file \
     that merely looks like a classifier is treated as a plain APR — run `apr inspect <FILE>` \
     to see what it actually is.";

/// Where the ordered texts came from.
///
/// Resolved BEFORE the model path is opened, and without naming a single core type,
/// so a binary built without the `setfit` feature still refuses a conflicting
/// invocation as a request error rather than reporting it as a missing feature.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RequestSource {
    /// One or more `--text` flags, in the order given.
    Texts(Vec<String>),
    /// A `--input` document path.
    Document(PathBuf),
}

impl RequestSource {
    /// Choose the one source, or refuse.
    ///
    /// Both is a conflict and neither is an omission; each names the flags, because
    /// "invalid arguments" tells an operator nothing they did not already know.
    fn choose(texts: &[String], input: Option<&Path>) -> Result<Self> {
        match (texts.is_empty(), input) {
            (false, Some(path)) => Err(CliError::ValidationFailed(format!(
                "--text and --input name two different ordered input sets and only one can be \
                 classified; drop one (--input {} was given alongside {} --text value(s))",
                path.display(),
                texts.len()
            ))),
            (true, None) => Err(CliError::ValidationFailed(
                "no texts to classify: pass --text <STR> (repeatable) or --input <FILE>, where \
                 the file is a classify request document `{\"texts\": [...], \"include_logits\": \
                 false}` — the SAME document the HTTP surface accepts"
                    .to_string(),
            )),
            (false, None) => Ok(Self::Texts(texts.to_vec())),
            (true, Some(path)) => Ok(Self::Document(path.to_path_buf())),
        }
    }
}

/// Run `apr predict`.
///
/// Order of business, and the order is asserted by a test:
///
/// 1. the REQUEST — a conflicting or empty invocation is the operator's mistake and
///    must not be reported as a problem with the model file;
/// 2. the artifact's typed tag, read from the header and metadata only;
/// 3. the classifier path, or a typed refusal naming what `predict` does support.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a bad request, [`CliError::InvalidFormat`] for
/// a file that is not a tagged SetFit artifact or for an oversized `--input`,
/// [`CliError::ModelLoadFailed`] for an artifact the loader refuses, and
/// [`CliError::FeatureDisabled`] (exit 9) when the binary was built without the
/// `setfit` feature.
pub(crate) fn run(
    path: &Path,
    texts: &[String],
    input: Option<&Path>,
    logits: bool,
    json_output: bool,
) -> Result<()> {
    // (1) THE REQUEST FIRST.
    let source = RequestSource::choose(texts, input)?;

    // (2) THE TYPED TAG. Header + metadata only; no tensor is loaded to decide this.
    let tagged = setfit_tag::read_setfit_tag(path)?.is_some();
    if !tagged {
        return Err(CliError::InvalidFormat(format!(
            "{}: not a SetFit classifier. {SUPPORTED_FAMILIES}",
            path.display()
        )));
    }

    // (3) THE ONE CLASSIFY PATH.
    #[cfg(feature = "setfit")]
    {
        setfit::predict_setfit(path, &source, logits, json_output)
    }
    #[cfg(not(feature = "setfit"))]
    {
        let _ = (&source, logits, json_output);
        Err(CliError::FeatureDisabled(format!(
            "{} is a setfit-apr-v1 classifier, but this binary was built without the `setfit` \
             feature. Rebuild with `--features setfit` (for example \
             `cargo install aprender --features setfit`) to classify with it. \
             `apr inspect` still reports its identity fields without the feature.",
            path.display()
        )))
    }
}

#[cfg(feature = "setfit")]
mod setfit {
    //! Everything that needs the classifier compiled in.

    use std::io::Read as _;

    use aprender::setfit::{
        load_setfit_apr, ClassifyRequestDocument, ClassifyResponse, MAX_REQUEST_BODY_BYTES,
    };
    use colored::Colorize;

    use super::{Path, RequestSource, Result};
    use crate::error::CliError;
    use crate::output;
    use crate::setfit_io;

    /// Load, classify and render.
    pub(super) fn predict_setfit(
        path: &Path,
        source: &RequestSource,
        logits: bool,
        json_output: bool,
    ) -> Result<()> {
        let request = build_request(source, logits)?;

        // THE ONE BOUNDED ARTIFACT DOOR (review B5). `fs::read` on an artifact path
        // is banned crate-wide by `setfit_io`'s module rule and by this file's own
        // source assertion.
        let bytes = setfit_io::read_setfit_apr_file_bounded(path)?;

        // The FULL eight-rung ladder, probe replay included. `apr predict` gets a
        // `VerifiedSetFitModel` or it gets a typed refusal; there is no "has a
        // bundle" tier in between that this command can reach.
        let model = load_setfit_apr(&bytes).map_err(|error| {
            CliError::ModelLoadFailed(format!(
                "{}: the artifact did not pass the load ladder — {error:?}",
                path.display()
            ))
        })?;

        let response = model
            .classify(&request)
            .map_err(|error| CliError::InferenceFailed(format!("{}: {error}", path.display())))?;

        if json_output {
            // VERBATIM. Core's own `Serialize`, no rewrite, no re-key: this is what
            // makes the CLI body and the HTTP body the same bytes for the same model
            // and input, which is what 04-09's parity gate compares.
            let rendered = serde_json::to_string_pretty(&response).map_err(|error| {
                CliError::Aprender(format!(
                    "the classification envelope did not serialize: {error}"
                ))
            })?;
            println!("{rendered}");
        } else {
            render_human(&request, &response);
        }
        Ok(())
    }

    /// Turn the chosen source into the SHARED request document.
    ///
    /// `--logits` sets `include_logits` on both paths. On the `--input` path it can
    /// only turn the flag ON: a document that asked for logits is not silently
    /// downgraded because the flag was omitted, and a document that did not ask can
    /// still be upgraded from the command line without editing the file.
    pub(super) fn build_request(
        source: &RequestSource,
        logits: bool,
    ) -> Result<ClassifyRequestDocument> {
        let mut document = match source {
            RequestSource::Texts(texts) => ClassifyRequestDocument::new(texts.clone()),
            RequestSource::Document(path) => read_request_document(path)?,
        };
        if logits {
            document.include_logits = true;
        }
        Ok(document)
    }

    /// Read `--input` as the shared document, BOUNDED BEFORE THE PARSE.
    ///
    /// `MAX_REQUEST_BODY_BYTES` is core's constant and core states plainly that it
    /// cannot enforce it — the module never sees bytes, only an already-parsed
    /// document — so enforcement is owed by the reading surface. This is that
    /// surface for the CLI. `MAX_BATCH_TEXTS` is checked by core AFTER the document
    /// exists, so it bounds the tokenization but not the parse: a body with ten
    /// million strings is fully materialised before it fires. The bound here is what
    /// stops that, and it is applied to the stat'd length first (free) and then to
    /// the stream (so a length that lies still cannot exhaust memory).
    ///
    /// Deliberately NOT `fs::read`, and deliberately not routed through
    /// `setfit_io`: that door is for ARTIFACT bytes and carries the artifact cap.
    /// A request document is a different resource class with a different bound, and
    /// reading it through the artifact door would apply MAX_ARTIFACT_BYTES to a 1 MiB
    /// resource.
    fn read_request_document(path: &Path) -> Result<ClassifyRequestDocument> {
        let metadata = std::fs::metadata(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                CliError::FileNotFound(path.to_path_buf())
            } else {
                CliError::Io(error)
            }
        })?;
        if !metadata.is_file() {
            return Err(CliError::NotAFile(path.to_path_buf()));
        }
        if metadata.len() > MAX_REQUEST_BODY_BYTES {
            return Err(CliError::InvalidFormat(format!(
                "{}: the request document declares {} bytes against the contracted bound of \
                 {MAX_REQUEST_BODY_BYTES}; refused before parsing, because a batch bound checked \
                 after the document is allocated is not a bound on the work an attacker can \
                 request",
                path.display(),
                metadata.len()
            )));
        }

        let file = std::fs::File::open(path).map_err(CliError::Io)?;
        let mut bytes = Vec::new();
        // `+ 1` so an over-cap stream is DETECTED rather than silently truncated
        // into a document that parses.
        file.take(MAX_REQUEST_BODY_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(CliError::Io)?;
        if bytes.len() as u64 > MAX_REQUEST_BODY_BYTES {
            return Err(CliError::InvalidFormat(format!(
                "{}: the request document's stream exceeded the contracted bound of \
                 {MAX_REQUEST_BODY_BYTES} bytes, so its declared length lied",
                path.display()
            )));
        }

        serde_json::from_slice(&bytes).map_err(|error| {
            CliError::ValidationFailed(format!(
                "{}: not a classify request document — {error}. The expected shape is \
                 {{\"texts\": [\"...\"], \"include_logits\": false}}; unknown keys are refused. \
                 It is deliberately NOT one text per line: a line-delimited file cannot carry a \
                 text containing a newline, so the CLI and the HTTP surface would receive \
                 different inputs while appearing to agree.",
                path.display()
            ))
        })
    }

    /// A view over the SAME envelope the `--json` path serializes.
    ///
    /// Every value is read through an accessor. Nothing is recomputed here — in
    /// particular the margin is the envelope's, not a difference this function takes
    /// between two probabilities it picked out, because a second derivation is a
    /// second answer.
    fn render_human(request: &ClassifyRequestDocument, response: &ClassifyResponse) {
        output::header("Classification");
        let mut rows: Vec<Vec<String>> = Vec::new();
        for (index, result) in response.results().iter().enumerate() {
            let text = request
                .texts
                .get(index)
                .map_or_else(String::new, |t| preview(t));
            let top = result
                .probabilities()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            rows.push(vec![
                index.to_string(),
                text,
                result.label().to_string(),
                format!("{top:.4}"),
                format!("{:.4}", result.margin()),
                result.token_count().to_string(),
                if result.truncated() { "yes" } else { "no" }.to_string(),
            ]);
        }
        println!(
            "{}",
            output::table(
                &["#", "Text", "Label", "P(top)", "Margin", "Tokens", "Trunc"],
                &rows
            )
        );
        println!();
        println!(
            "{}",
            output::kv_table(&[
                ("Artifact", response.artifact_sha256().to_string()),
                ("Backend", response.backend().to_string()),
                ("Schema", response.schema_version().to_string()),
                ("Latency", format!("{:.3} ms", response.latency_ms())),
            ])
        );
        if response.results().iter().any(|r| r.logits().is_some()) {
            println!(
                "{}",
                "  (per-class logits present; use --json to read them)".dimmed()
            );
        }
    }

    /// A single-line, bounded rendering of a text for the human table.
    ///
    /// Control characters are escaped rather than printed: a text containing a
    /// newline is exactly the input this command exists to carry faithfully, and a
    /// table that let it break the row would misreport which text got which label.
    fn preview(text: &str) -> String {
        const MAX: usize = 40;
        let mut out = String::with_capacity(MAX);
        // Count as we go. `out.chars().count()` re-walked the whole accumulating string on
        // every pushed char, making this quadratic in MAX for each of up to 256 rows.
        //
        // The counter tracks RENDERED width, not input chars: an escaped control character
        // occupies two columns, and the bound is on what the table cell displays. Counting
        // inputs instead would let a newline-heavy text render wider than MAX.
        let mut shown = 0_usize;
        for ch in text.chars() {
            if shown >= MAX {
                out.push('…');
                break;
            }
            match ch {
                '\n' => {
                    out.push_str("\\n");
                    shown += 2;
                }
                '\r' => {
                    out.push_str("\\r");
                    shown += 2;
                }
                '\t' => {
                    out.push_str("\\t");
                    shown += 2;
                }
                other => {
                    out.push(other);
                    shown += 1;
                }
            }
        }
        if out.is_empty() {
            "(empty)".to_string()
        } else {
            out
        }
    }
}

#[cfg(test)]
#[path = "predict_tests.rs"]
mod predict_tests;
