//! The ONE place `apr-cli` decides "is this APR a SetFit classifier?" (D-04).
//!
//! # Detection reads the TYPED TAG and nothing else
//!
//! `model_type == "setfit"` in the APR v2 metadata record, plus the single custom
//! key `"setfit"` that carries the artifact document. Tensor names are NEVER
//! consulted. That is D-04's negative stated as code: a file that happens to carry
//! `setfit.head.weight` — a hand-assembled APR, a partially converted checkpoint, a
//! deliberately shaped decoy — is a plain APR to every command here, because the
//! only thing that makes an artifact a SetFit artifact is that its writer said so in
//! the field reserved for saying so.
//!
//! Sniffing by tensor name would make the classification a GUESS about content, and
//! a guess that lands on `load_setfit_apr` hands hostile bytes to a rebuild path
//! that a plain `apr inspect` would have merely described.
//!
//! # This is a HEADER + METADATA read, never a tensor load
//!
//! Detection costs 64 bytes plus the metadata block. `apr predict` on a 200 MB
//! artifact that turns out not to be a classifier must not have paid for the
//! tensors to find that out, and `apr inspect` — which never classifies — must not
//! pay for them at all.
//!
//! # The metadata length is bounded by the FILE, not believed
//!
//! `metadata_size` is a `u32` read out of the file being judged, so an attacker
//! controls it up to 4 GiB. It is checked against the stat'd file length before the
//! buffer is allocated: a block cannot be longer than the file that contains it, and
//! that bound needs no invented constant to be true. `read_exact` failing afterwards
//! is a diagnosis, not a defence — by then the allocation has already happened.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use aprender::format::v2::{AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2};

use crate::error::CliError;

/// The value `AprV2Metadata::model_type` carries for a `setfit-apr-v1` artifact.
pub(crate) const SETFIT_MODEL_TYPE: &str = "setfit";

/// The most metadata this module will read to answer "is this a SetFit artifact?".
///
/// Absorbed from a duplicate detector that `serve/handlers.rs` carried until this cleanup;
/// that copy had its own bound and its own opinion, and the two could answer differently for
/// the same file. One detection point (D-04/D-10) means one bound.
///
/// `commands::inspect.rs::read_metadata` now REFERENCES this constant rather than bounding its
/// own read by the stat'd file length alone, so the two readers of one metadata block cannot
/// answer differently about it — the same argument stated just above for the deleted `serve`
/// copy, applied to a second READER instead of to a second detector.
pub(crate) const MAX_TAG_METADATA_BYTES: u64 = 16 * 1024 * 1024;

/// The single custom metadata key the artifact document lives at.
pub(crate) const SETFIT_CUSTOM_KEY: &str = "setfit";

/// What the tag read recovered.
///
/// `doc` is `Option` because the two facts are independent: the tag is what ROUTES,
/// and the document is what a reader then renders. An artifact tagged `setfit` whose
/// custom key is missing is still routed to the SetFit path — where the loader
/// refuses it by name — rather than silently falling back to "plain APR", which
/// would report a corrupt classifier as a healthy generic model.
#[derive(Debug, Clone)]
pub(crate) struct SetFitTag {
    /// The raw value at custom key [`SETFIT_CUSTOM_KEY`], unparsed.
    ///
    /// Deliberately `serde_json::Value` and not a typed document: this module is
    /// compiled with the `setfit` feature OFF as well, where `SetFitArtifactDoc`
    /// cannot be named. Callers that have the feature parse it; `apr inspect`
    /// renders the raw value so identity fields stay recoverable from a binary
    /// built without the classifier.
    pub(crate) doc: Option<serde_json::Value>,
}

/// Read a file's typed SetFit tag, if it has one.
///
/// Returns `Ok(None)` for anything that is readable but not a tagged SetFit APR —
/// a GGUF, a SafeTensors file, a legacy APR, a plain APR v2, or an APR v2 whose
/// metadata block does not parse. Deciding what to say about those belongs to the
/// command, which knows whether it can do something useful with a plain APR.
///
/// # Errors
///
/// [`CliError::FileNotFound`] for an absent path, [`CliError::NotAFile`] for a
/// directory or other non-regular file, and [`CliError::Io`] for a read failure.
///
/// [`CliError::InvalidFormat`] for EITHER of two distinct metadata malformations,
/// which carry distinct messages because they are different facts about the
/// container and a reader must not have to guess which one fired:
///
/// 1. the block the header declares does not fit inside the file; or
/// 2. it fits, but exceeds [`MAX_TAG_METADATA_BYTES`] — the identification cap. This
///    case is deliberately NOT `Ok(None)`: falling through would report a container
///    this function could not identify as one it positively identified as "not
///    SetFit", which is a claim it has no evidence for (WR-08).
pub(crate) fn read_setfit_tag(path: &Path) -> Result<Option<SetFitTag>, CliError> {
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::FileNotFound(path.to_path_buf())
        } else {
            CliError::Io(error)
        }
    })?;
    if !metadata.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    let file_len = metadata.len();

    let mut file = fs::File::open(path).map_err(CliError::Io)?;

    // (1) THE HEADER. A file too short to hold one is not an APR v2 container, which
    //     is a "no" rather than an error: `apr predict` says the same thing about a
    //     GGUF, and both answers are "this is not a SetFit artifact".
    let mut header_bytes = [0_u8; HEADER_SIZE_V2];
    if file.read_exact(&mut header_bytes).is_err() {
        return Ok(None);
    }
    let magic = &header_bytes[0..4];
    if magic != MAGIC_V2 {
        return Ok(None);
    }
    let Ok(header) = AprV2Header::from_bytes(&header_bytes) else {
        return Ok(None);
    };
    if header.metadata_size == 0 {
        return Ok(None);
    }

    // (2) THE BOUND, BEFORE THE ALLOCATION. `metadata_size` is a u32 out of the file
    //     under judgement; a block that claims to run past the end of its own file is
    //     a malformed container, and saying so costs one comparison instead of up to
    //     4 GiB of zeroed memory.
    let declared = u64::from(header.metadata_size);
    let end = header.metadata_offset.saturating_add(declared);
    if end > file_len {
        return Err(CliError::InvalidFormat(format!(
            "{}: the metadata block declares {declared} bytes at offset {} , which runs past the \
             end of a {file_len}-byte file",
            path.display(),
            header.metadata_offset
        )));
    }

    // (2b) THE ABSOLUTE CAP, complementary to (2) rather than a replacement for it.
    //      (2) is tighter for a small file; this is tighter for a large one — a 30 GB APR
    //      whose metadata block declares 20 MiB passes (2) and would otherwise be read in
    //      full just to answer one yes/no question. `setfit-apr-v1` cannot need this much:
    //      the tokenizer bytes and every tensor live in the container's DATA section, never
    //      in metadata.
    //
    //      Over the cap is an ERROR, not `Ok(None)`. It used to be `Ok(None)`, justified by
    //      "the caller falls through to the pre-existing APR path, which has its own and
    //      better diagnosis". MEASURED, that sentence was true of `apr serve` ONLY:
    //
    //        * `apr predict` has no plain-APR path, so it answered "not a SetFit
    //          classifier … run `apr inspect`" for a file that DOES carry the tag, and
    //          `apr inspect` then rendered the full APR-05 section for the same bytes —
    //          the refusal's own text routed the operator into the contradiction;
    //        * `apr eval --task classify` refused its SetFit-only flags with "does not
    //          carry the SetFit tag", which is false of such a file.
    //
    //      One `Ok(None)` therefore produced three statements about one file, two of them
    //      untrue (WR-08). `apr serve` KEEPS its fall-through: `serve/handlers.rs:774`
    //      calls this through `.ok().flatten()`, so an `Err` here still lands on the
    //      plain-APR path exactly as before, deliberately and documented as such. The
    //      other two consumers already `?`-propagate, so they now carry a true statement
    //      with no edit of their own.
    //
    //      The message says NOTHING about whether this is a SetFit artifact: the block
    //      has not been read, so at this point that fact is genuinely unknown.
    if declared > MAX_TAG_METADATA_BYTES {
        return Err(CliError::InvalidFormat(format!(
            "{}: the metadata block declares {declared} bytes, over the \
             {MAX_TAG_METADATA_BYTES} byte (16 MiB) cap this reader will read to identify a \
             container. The block was NOT read, so what this file is has not been determined.",
            path.display()
        )));
    }

    // (3) THE READ, at the size the bound just approved.
    if file.seek(SeekFrom::Start(header.metadata_offset)).is_err() {
        return Ok(None);
    }
    let mut metadata_bytes = vec![0_u8; declared as usize];
    if file.read_exact(&mut metadata_bytes).is_err() {
        return Ok(None);
    }

    // (4) THE TAG. Parse failures are "not a SetFit artifact", not an error: a plain
    //     APR whose metadata this build cannot parse is exactly as un-SetFit as one
    //     whose `model_type` is `qwen2`.
    let Ok(mut meta) = AprV2Metadata::from_json(&metadata_bytes) else {
        return Ok(None);
    };
    if meta.model_type != SETFIT_MODEL_TYPE {
        return Ok(None);
    }
    // `remove`, not `get(..).cloned()`: `meta` is owned and `custom` is never read again, so the
    // document MOVES out. The cloned form deep-copied the whole SetFit document — every probe
    // embedding and the entire HF name map, hundreds of KB — on every `apr predict` and every
    // `apr eval --task classify`, only to drop it: both production callers ask `.is_some()`.
    Ok(Some(SetFitTag {
        doc: meta.custom.remove(SETFIT_CUSTOM_KEY),
    }))
}

/// The ONE fixture builder for tag-detection tests, shared across command modules.
///
/// It lives here rather than in each test module because the two files it produces
/// differ ONLY in `model_type`: a copy in another module would be free to drift in
/// exactly the dimension every one of these tests is sensitive to, and the drifted
/// copy would keep passing.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashMap;
    use std::io::Write as _;
    use std::path::{Path, PathBuf};

    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    use super::SETFIT_CUSTOM_KEY;

    /// An APR v2 container carrying SetFit-SHAPED tensors and a caller-chosen tag.
    ///
    /// The tensor names are the ones a real SetFit artifact uses, so a detector that
    /// sniffed content rather than reading the tag would classify the untagged case
    /// as SetFit — which is the D-04 negative these fixtures exist to express.
    ///
    /// The container is REAL (written by the production writer) but the tensors are
    /// not a model: nothing here can pass the load ladder, and nothing here claims
    /// to. Tag detection and load refusal are what these fixtures test.
    pub(crate) fn write_setfit_shaped_apr(
        dir: &Path,
        name: &str,
        model_type: &str,
        doc: Option<&str>,
    ) -> PathBuf {
        let mut custom: HashMap<String, serde_json::Value> = HashMap::new();
        if let Some(doc) = doc {
            custom.insert(
                SETFIT_CUSTOM_KEY.to_string(),
                serde_json::from_str(doc).expect("the fixture document is valid JSON"),
            );
        }
        let metadata = AprV2Metadata {
            model_type: model_type.to_string(),
            created_at: None,
            custom,
            ..Default::default()
        };
        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("setfit.head.weight".to_string(), vec![2, 4], &[0.5_f32; 8]);
        writer.add_f32_tensor("setfit.head.bias".to_string(), vec![2], &[0.0_f32; 2]);
        let bytes = writer.write().expect("the fixture container is writable");
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).expect("fixture file is creatable");
        file.write_all(&bytes).expect("fixture file is writable");
        file.sync_all().expect("fixture file syncs");
        path
    }

    /// A REAL APR v2 container whose header DECLARES a metadata block over the cap.
    ///
    /// Only the 4-byte `metadata_size` field is rewritten. The DECLARED value is the
    /// whole subject, so the block does not have to actually be that large — and
    /// making it so would put a 16 MiB write into every `cargo test` for no extra
    /// evidence, since nothing ever reads past the refusal.
    ///
    /// The offsets are `AprV2Header::from_bytes`' own — `metadata_offset` a LE `u64`
    /// at 12..20 and `metadata_size` a LE `u32` at 20..24 — CONFIRMED by reading
    /// `apr-format/src/v2/header_impl.rs`, not taken on trust from a plan.
    ///
    /// The file is then EXTENDED past `metadata_offset + declared` so the pre-existing
    /// "runs past the end of its own file" check cannot fire first. That ORDER is what
    /// keeps the two malformations distinguishable, and a fixture that tripped the
    /// earlier check would prove nothing about the cap. `set_len` is sparse, so the
    /// extension is instant and costs no disk.
    pub(crate) fn write_over_cap_apr(
        dir: &Path,
        name: &str,
        model_type: &str,
        declared: u64,
    ) -> PathBuf {
        let path = write_setfit_shaped_apr(
            dir,
            name,
            model_type,
            Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
        );
        let mut bytes = std::fs::read(&path).expect("the honest fixture is readable");
        let metadata_offset = u64::from_le_bytes(
            bytes
                .get(12..20)
                .and_then(|slice| <[u8; 8]>::try_from(slice).ok())
                .expect("an APR v2 header is 64 bytes, so 12..20 is in range"),
        );
        let encoded = u32::try_from(declared)
            .expect("a declared size under test must fit the u32 header field")
            .to_le_bytes();
        bytes
            .get_mut(20..24)
            .expect("an APR v2 header is 64 bytes, so 20..24 is in range")
            .copy_from_slice(&encoded);
        std::fs::write(&path, &bytes).expect("the patched fixture is writable");

        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("the patched fixture reopens for extension");
        file.set_len(metadata_offset.saturating_add(declared).saturating_add(1))
            .expect("sparse extension needs no zero buffer");
        file.sync_all().expect("the extended fixture syncs");
        path
    }
}

#[cfg(test)]
#[path = "setfit_tag_tests.rs"]
mod setfit_tag_tests;
