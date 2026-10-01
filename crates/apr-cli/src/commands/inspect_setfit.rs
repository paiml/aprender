
// ============================================================================
// APR-05: the SetFit inspection section
// ============================================================================
//
// # What APR-05 asks for, and why it is recoverable OFFLINE
//
// "A user recovers every contracted identity field from the artifact alone."
// Alone means: no network, no sidecar file, no registry, and no training run to
// ask. Everything below is read out of the one `setfit` custom-metadata key the
// writer put there, plus the artifact's own SHA-256.
//
// # Inspection is READ-ONLY metadata work (the APR-04 boundary)
//
// It does not load a tensor, rebuild an encoder, or replay a probe.
// `VerifiedSetFitModel` is the typestate APR-04 gates PREDICTION behind, because
// prediction is a claim about what the model computes. Inspection is a claim about
// what the artifact SAYS, and demanding the verified state for it would mean a
// corrupt artifact could not be inspected — which is precisely when an operator
// most needs to look at it.
//
// The one whole-file read here is for the artifact's SHA-256, and it goes through
// `setfit_io::read_setfit_apr_file_bounded` like every other artifact read in this
// crate (review B5). The cheap header+metadata read above is untouched.
//
// # ONE renderer over the raw document, with the typed parse as a SIGNAL
//
// The section is built by reading named paths out of the recovered
// `serde_json::Value`, and (when the `setfit` feature is compiled in) core's
// `SetFitArtifactDoc` is parsed as well — not to render from, but to REPORT whether
// the document satisfies the closed schema, under `document_schema_valid`.
//
// Two renderers, one for each feature state, would be two answers that can drift,
// and the feature-off one would be the untested one. A single renderer also keeps
// `inspect` useful on an artifact whose document this build cannot parse: a future
// schema version still shows its identity fields AND reports the parse failure,
// instead of showing nothing. `inspect_setfit_key_names_are_the_documents_own`
// pins the key names against `SETFIT_ARTIFACT_DOC_FIELDS` so the renderer's paths
// cannot silently drift from the struct's field names.

/// Build the APR-05 section, or `None` for anything that is not a tagged artifact.
fn build_setfit_inspection(path: &Path, doc: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let doc = doc?;
    let mut section = serde_json::Map::new();

    // --- schema identity -----------------------------------------------------
    copy_paths(
        &mut section,
        doc,
        &[
            ("schema", &["schema"]),
            ("schema_version", &["schema_version"]),
            ("bundle_schema_version", &["bundle_schema_version"]),
            ("format_id", &["format_id"]),
        ],
    );

    // --- encoder + tokenizer revision and hashes -----------------------------
    copy_paths(
        &mut section,
        doc,
        &[
            ("encoder_revision", &["architecture", "source_revision"]),
            (
                "encoder_tokenizer_sha256",
                &["architecture", "tokenizer_sha256"],
            ),
            ("tokenizer_sha256", &["tokenizer_sha256"]),
            ("hidden_act", &["architecture", "hidden_act"]),
        ],
    );
    section.insert(
        "architecture".to_string(),
        pick_object(
            doc,
            &["architecture"],
            &[
                "hidden",
                "heads",
                "head_dim",
                "num_layers",
                "intermediate",
                "vocab",
                "positions",
                "type_vocab_size",
            ],
        ),
    );

    // --- pooling / normalization / truncation policy -------------------------
    section.insert(
        "preprocessing".to_string(),
        pick_object(
            doc,
            &["preprocessing"],
            &[
                "pooling",
                "normalization",
                "l2_epsilon_hex",
                "truncation_max_sequence_length",
                "padding_mode",
                "max_length",
            ],
        ),
    );

    // --- ordered labels and head configuration -------------------------------
    copy_paths(&mut section, doc, &[("ordered_labels", &["ordered_labels"])]);
    section.insert(
        "head".to_string(),
        pick_object(doc, &["head"], &["n_features", "num_labels"]),
    );

    // --- the data fingerprints and the seeds (provenance, bundle field 20) ----
    //
    // BOTH fingerprints, because Phase 2 made them deliberately distinct: the
    // dataset fingerprint digests the whole corpus and the validation-split one
    // digests the split alone, so only carrying both lets a reader tell "a
    // different dataset" from "the same dataset whose validation rows changed".
    copy_paths(
        &mut section,
        doc,
        &[
            (
                "dataset_fingerprint",
                &["provenance", "dataset_fingerprint"],
            ),
            (
                "validation_split_fingerprint",
                &["provenance", "validation_split_fingerprint"],
            ),
            (
                "selection_semantic_hash",
                &["provenance", "selection_semantic_hash"],
            ),
            (
                "selection_ledger_hash",
                &["provenance", "selection_ledger_hash"],
            ),
            ("root_seed", &["root_seed"]),
            (
                "selection_root_seed",
                &["provenance", "selection_root_seed"],
            ),
            ("shots_per_class", &["provenance", "shots_per_class"]),
        ],
    );

    // --- the update-evidence summary and its binding hash --------------------
    section.insert(
        "evidence".to_string(),
        pick_object(
            doc,
            &["evidence"],
            &[
                "verdict",
                "trainable_count",
                "frozen_count",
                "worst_param_name",
                "epsilon_used",
                "calibration_regime_id",
                "contract_version",
                "table_hash",
            ],
        ),
    );

    // --- the artifact's own SHA-256 ------------------------------------------
    let (artifact_sha256, note) = artifact_sha256_of(path);
    section.insert("artifact_sha256".to_string(), artifact_sha256);
    if let Some(note) = note {
        section.insert(
            "artifact_sha256_note".to_string(),
            serde_json::Value::String(note),
        );
    }

    // --- the typed-schema signal ---------------------------------------------
    let (valid, error) = document_schema_validity(doc);
    section.insert("document_schema_valid".to_string(), valid);
    if let Some(error) = error {
        section.insert(
            "document_schema_error".to_string(),
            serde_json::Value::String(error),
        );
    }

    Some(serde_json::Value::Object(section))
}

/// Copy `(output key, document path)` pairs, emitting `null` for anything absent.
///
/// `null` rather than a skipped key, on the same reasoning the provenance block
/// above states for `license` / `data_source` / `data_license`: an auditor greps
/// for a field, and a silently skipped key is indistinguishable from a field this
/// build does not know about.
fn copy_paths(
    out: &mut serde_json::Map<String, serde_json::Value>,
    doc: &serde_json::Value,
    pairs: &[(&str, &[&str])],
) {
    for (key, path) in pairs {
        out.insert((*key).to_string(), at_path(doc, path));
    }
}

/// Read a nested path, or `null`.
fn at_path(doc: &serde_json::Value, path: &[&str]) -> serde_json::Value {
    let mut cursor = doc;
    for segment in path {
        match cursor.get(segment) {
            Some(next) => cursor = next,
            None => return serde_json::Value::Null,
        }
    }
    cursor.clone()
}

/// A sub-object built from an EXPLICIT key list rather than copied wholesale.
///
/// Wholesale copying would republish `requested_config`, every probe's embedding
/// and the whole HF name map into an inspection report — hundreds of kilobytes on a
/// real artifact — and would make the report's shape a function of the artifact's
/// rather than of APR-05's. The named keys are the contracted ones.
fn pick_object(doc: &serde_json::Value, at: &[&str], keys: &[&str]) -> serde_json::Value {
    let source = at_path(doc, at);
    let mut picked = serde_json::Map::new();
    for key in keys {
        picked.insert(
            (*key).to_string(),
            source
                .get(key)
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        );
    }
    serde_json::Value::Object(picked)
}

/// The artifact's SHA-256, and a note when this build cannot compute it.
///
/// With the `setfit` feature compiled in, this is core's `artifact_sha256_hex` over
/// bytes obtained from the ONE bounded artifact door — the same value every other
/// surface in this repository reports for the same file, because it is the same
/// function over the same bytes.
///
/// Without the feature the digest is `null` and the note says so. Inventing a
/// second hashing path here to fill the field would be a second answer to "what is
/// this artifact's identity", and two answers to that question is the failure mode
/// this whole phase exists to prevent.
fn artifact_sha256_of(path: &Path) -> (serde_json::Value, Option<String>) {
    #[cfg(feature = "setfit")]
    {
        match crate::setfit_io::read_setfit_apr_file_bounded(path) {
            Ok(bytes) => (
                serde_json::Value::String(aprender::setfit::artifact_sha256_hex(&bytes)),
                None,
            ),
            Err(error) => (
                serde_json::Value::Null,
                Some(format!(
                    "the artifact bytes could not be read through the bounded door: {error}"
                )),
            ),
        }
    }
    #[cfg(not(feature = "setfit"))]
    {
        let _ = path;
        (
            serde_json::Value::Null,
            Some(
                "this binary was built without the `setfit` feature, so it does not carry the \
                 artifact hashing path; rebuild with `--features setfit` for the digest. Every \
                 field above is recovered from the artifact's own metadata and is complete."
                    .to_string(),
            ),
        )
    }
}

/// Whether the recovered document satisfies core's closed schema.
///
/// A SIGNAL, not a gate: the fields above are rendered either way. Reported as
/// `null` (with a note) when this build has no parser for it, because `false` would
/// be a claim about the artifact that this binary is not in a position to make.
fn document_schema_validity(doc: &serde_json::Value) -> (serde_json::Value, Option<String>) {
    #[cfg(feature = "setfit")]
    {
        match serde_json::from_value::<aprender::setfit::SetFitArtifactDoc>(doc.clone()) {
            Ok(_) => (serde_json::Value::Bool(true), None),
            Err(error) => (
                serde_json::Value::Bool(false),
                Some(format!(
                    "the artifact document does not satisfy setfit-apr-v1's closed schema: \
                     {error}. The fields above are what could be recovered from it."
                )),
            ),
        }
    }
    #[cfg(not(feature = "setfit"))]
    {
        let _ = doc;
        (
            serde_json::Value::Null,
            Some(
                "this binary was built without the `setfit` feature, so it carries no parser for \
                 the artifact document's closed schema and makes no claim about it."
                    .to_string(),
            ),
        )
    }
}

/// Render the APR-05 section for human eyes.
fn output_setfit_text(section: Option<&serde_json::Value>) {
    let Some(section) = section else {
        return;
    };
    output::subheader("SetFit Classifier (setfit-apr-v1)");

    let mut pairs: Vec<(&str, String)> = Vec::new();
    for key in [
        "schema",
        "schema_version",
        "format_id",
        "artifact_sha256",
        "encoder_revision",
        "tokenizer_sha256",
        "root_seed",
    ] {
        pairs.push((key, render_scalar(section.get(key))));
    }
    println!("{}", output::kv_table(&pairs));

    output::subheader("  Labels & Head");
    let mut head_pairs: Vec<(&str, String)> = vec![(
        "ordered_labels",
        render_scalar(section.get("ordered_labels")),
    )];
    for key in ["n_features", "num_labels"] {
        head_pairs.push((
            key,
            render_scalar(section.get("head").and_then(|h| h.get(key))),
        ));
    }
    println!("{}", output::kv_table(&head_pairs));

    output::subheader("  Preprocessing");
    let mut pre_pairs: Vec<(&str, String)> = Vec::new();
    for key in [
        "pooling",
        "normalization",
        "l2_epsilon_hex",
        "truncation_max_sequence_length",
        "padding_mode",
        "max_length",
    ] {
        pre_pairs.push((
            key,
            render_scalar(section.get("preprocessing").and_then(|p| p.get(key))),
        ));
    }
    println!("{}", output::kv_table(&pre_pairs));

    output::subheader("  Provenance & Evidence");
    let mut prov_pairs: Vec<(&str, String)> = Vec::new();
    for key in [
        "dataset_fingerprint",
        "validation_split_fingerprint",
        "selection_semantic_hash",
        "selection_ledger_hash",
        "selection_root_seed",
        "shots_per_class",
    ] {
        prov_pairs.push((key, render_scalar(section.get(key))));
    }
    for key in ["verdict", "calibration_regime_id", "table_hash"] {
        prov_pairs.push((
            key,
            render_scalar(section.get("evidence").and_then(|e| e.get(key))),
        ));
    }
    println!("{}", output::kv_table(&prov_pairs));

    for note in ["artifact_sha256_note", "document_schema_error"] {
        if let Some(serde_json::Value::String(text)) = section.get(note) {
            println!("  ({note}: {text})");
        }
    }
}

/// One JSON value as a single human-readable cell.
fn render_scalar(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => "(absent)".to_string(),
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}
