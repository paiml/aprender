//! What the server actually knows about the model it loaded.
//!
//! `/realize/model` and the Ollama discovery endpoints (`/api/tags`,
//! `/api/show`) report model metadata. Before this type existed they reported
//! *constants*: `size_bytes: 0`, `context_length: 4096`,
//! `quantization: "Q4_K_M"`, `format: "gguf"` and a `content_hash` of
//! `"blake3:0".repeat(16)` — a 128-character string shaped exactly like a real
//! BLAKE3 digest, which a consumer will happily store and compare as
//! provenance. Every one of those was wrong for at least one shipped model,
//! and the hash was wrong for all of them.
//!
//! The rule this type enforces: **every field is either measured or absent.**
//! Each accessor returns `Option`, `None` means "this server does not know",
//! and the handlers omit unknown fields from the JSON rather than substituting
//! a plausible-looking default. An absent field is a fact a client can act on;
//! a fabricated one is not.

use std::path::Path;

/// Measured provenance/metadata for the model this server is serving.
///
/// Built by the process that loaded the model (`apr serve run`), attached to
/// [`super::AppState`] via `with_model_source`, and read by the metadata
/// handlers. Construct with [`ModelSourceInfo::from_path`] so `size_bytes` and
/// `format` come from the file itself; layer on what the loader learned with
/// the `with_*` builders.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelSourceInfo {
    path: Option<String>,
    size_bytes: Option<u64>,
    format: Option<String>,
    quantization: Option<String>,
    architecture: Option<String>,
    /// Context length this server was CONFIGURED with (`--context-length`),
    /// NOT the model's advertised maximum — a server started with
    /// `--context-length 128` must not report 4096, and must not report the
    /// model's 32768 either. These are three different facts.
    context_length: Option<usize>,
    /// Model's own advertised maximum context, when the loader read one.
    model_max_context_length: Option<usize>,
    content_hash: Option<String>,
    parameter_count: Option<u64>,
}

/// Identify a model container from its leading bytes.
///
/// Returns `None` rather than guessing: an unrecognised container is reported
/// as unknown, never as "gguf".
#[must_use]
pub fn detect_format_from_magic(magic: &[u8]) -> Option<&'static str> {
    if magic.starts_with(b"GGUF") {
        return Some("gguf");
    }
    // APR v2 is "APR\0", v1 is "APRN"; both share the 3-byte "APR" prefix.
    if magic.starts_with(&crate::apr::MAGIC_PREFIX) {
        return Some("apr");
    }
    // SafeTensors: 8-byte little-endian header length, then a JSON object.
    if magic.len() > 8 && magic[8] == b'{' {
        return Some("safetensors");
    }
    None
}

impl ModelSourceInfo {
    /// Measure what the file itself tells us: absolute path, byte size, and
    /// container format (from magic bytes, not the extension).
    ///
    /// Unreadable metadata leaves the corresponding field `None`; this never
    /// fails, because a missing metadata field must not stop the server from
    /// serving.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        let size_bytes = std::fs::metadata(path).ok().map(|m| m.len());
        let format = read_magic(path)
            .as_deref()
            .and_then(detect_format_from_magic)
            .map(str::to_string);
        let path_str = std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned();
        Self {
            path: Some(path_str),
            size_bytes,
            format,
            ..Self::default()
        }
    }

    /// Record the quantization actually present in the loaded weights.
    #[must_use]
    pub fn with_quantization(mut self, quantization: impl Into<String>) -> Self {
        self.quantization = Some(quantization.into());
        self
    }

    /// Record the model architecture the loader identified (e.g. `qwen2`).
    #[must_use]
    pub fn with_architecture(mut self, architecture: impl Into<String>) -> Self {
        self.architecture = Some(architecture.into());
        self
    }

    /// Record the context length this server was configured with.
    #[must_use]
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = Some(context_length);
        self
    }

    /// Record the model's own advertised maximum context length.
    #[must_use]
    pub fn with_model_max_context_length(mut self, context_length: usize) -> Self {
        self.model_max_context_length = Some(context_length);
        self
    }

    /// Record a **computed** content hash, e.g. `blake3:<hex>`.
    ///
    /// Only call this with a digest that was actually computed over the model
    /// bytes. There is deliberately no default.
    #[must_use]
    pub fn with_content_hash(mut self, content_hash: impl Into<String>) -> Self {
        self.content_hash = Some(content_hash.into());
        self
    }

    /// Record the parameter count the loader derived from the weights.
    #[must_use]
    pub fn with_parameter_count(mut self, parameter_count: u64) -> Self {
        self.parameter_count = Some(parameter_count);
        self
    }

    /// Absolute path of the served model file, if it came from a file.
    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Size of the model file in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> Option<u64> {
        self.size_bytes
    }

    /// Container format (`gguf`, `apr`, `safetensors`).
    #[must_use]
    pub fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// Quantization of the loaded weights.
    #[must_use]
    pub fn quantization(&self) -> Option<&str> {
        self.quantization.as_deref()
    }

    /// Model architecture.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    /// Context length this server was configured with.
    #[must_use]
    pub fn context_length(&self) -> Option<usize> {
        self.context_length
    }

    /// Model's advertised maximum context length.
    #[must_use]
    pub fn model_max_context_length(&self) -> Option<usize> {
        self.model_max_context_length
    }

    /// Computed content hash of the model bytes.
    #[must_use]
    pub fn content_hash(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    /// Parameter count.
    #[must_use]
    pub fn parameter_count(&self) -> Option<u64> {
        self.parameter_count
    }
}

/// Read the first 16 bytes of a file for format detection.
fn read_magic(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 16];
    let n = file.read(&mut buf).ok()?;
    Some(buf[..n].to_vec())
}

/// Human-readable name for a GGUF quantization type id.
///
/// Derived from the qtype actually stored in the loaded tensors, which is the
/// ground truth — `general.file_type` in GGUF metadata is advisory and is
/// known to go stale when a file is requantized.
#[must_use]
pub fn gguf_qtype_name(qtype: u32) -> Option<&'static str> {
    Some(match qtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_source_knows_nothing_and_admits_it() {
        // The point of the type: an empty source reports absence, not defaults.
        let src = ModelSourceInfo::default();
        assert_eq!(src.size_bytes(), None);
        assert_eq!(src.format(), None);
        assert_eq!(src.quantization(), None);
        assert_eq!(src.context_length(), None);
        assert_eq!(src.content_hash(), None, "never invent a content hash");
    }

    #[test]
    fn detect_format_recognises_the_three_containers() {
        assert_eq!(detect_format_from_magic(b"GGUF\0\0\0\0"), Some("gguf"));
        assert_eq!(detect_format_from_magic(b"APR\0____"), Some("apr"));
        assert_eq!(detect_format_from_magic(b"APRN____"), Some("apr"));
        // SafeTensors: u64 header length then '{'.
        let st = [8u8, 0, 0, 0, 0, 0, 0, 0, b'{', b'"'];
        assert_eq!(detect_format_from_magic(&st), Some("safetensors"));
    }

    #[test]
    fn unrecognised_magic_is_unknown_not_gguf() {
        // The shipped defect reported "gguf" for every model regardless.
        assert_eq!(detect_format_from_magic(b"\x7fELF\0\0\0\0"), None);
        assert_eq!(detect_format_from_magic(b""), None);
    }

    #[test]
    fn from_path_measures_size_and_format_of_a_real_file() {
        let dir = std::env::temp_dir().join(format!(
            "apr-model-source-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("tiny.gguf");
        // 4-byte magic + 12 bytes of payload = 16 bytes on disk.
        std::fs::write(&path, b"GGUF\x03\0\0\0\0\0\0\0\0\0\0\0").expect("write");

        let src = ModelSourceInfo::from_path(&path);
        assert_eq!(
            src.size_bytes(),
            Some(16),
            "size must be the real file size"
        );
        assert_eq!(src.format(), Some("gguf"));
        assert!(src.path().is_some_and(|p| p.ends_with("tiny.gguf")));
        assert_eq!(src.content_hash(), None);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_path_on_a_missing_file_reports_absence_not_zero() {
        let src = ModelSourceInfo::from_path(Path::new("/nonexistent/model.gguf"));
        assert_eq!(src.size_bytes(), None, "a missing file is unknown, not 0");
        assert_eq!(src.format(), None);
    }

    #[test]
    fn builders_record_measured_values() {
        let src = ModelSourceInfo::default()
            .with_quantization("Q4_K")
            .with_architecture("qwen2")
            .with_context_length(128)
            .with_model_max_context_length(32768)
            .with_content_hash("blake3:deadbeef")
            .with_parameter_count(1_500_000_000);
        assert_eq!(src.quantization(), Some("Q4_K"));
        assert_eq!(src.architecture(), Some("qwen2"));
        assert_eq!(src.context_length(), Some(128));
        assert_eq!(src.model_max_context_length(), Some(32768));
        assert_eq!(src.content_hash(), Some("blake3:deadbeef"));
        assert_eq!(src.parameter_count(), Some(1_500_000_000));
    }

    #[test]
    fn qtype_names_cover_the_shipped_quantizations() {
        assert_eq!(gguf_qtype_name(12), Some("Q4_K"));
        assert_eq!(gguf_qtype_name(14), Some("Q6_K"));
        assert_eq!(gguf_qtype_name(0), Some("F32"));
        // Unknown ids must not be reported as some plausible quantization.
        assert_eq!(gguf_qtype_name(9999), None);
    }
}
