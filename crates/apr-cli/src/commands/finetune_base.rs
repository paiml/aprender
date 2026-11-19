//! The base model of `apr finetune`: `.apr`, SafeTensors or GGUF
//! (contracts/apr-recipe-v1.yaml, FALSIFY-APR-RECIPE-011; PMAT-7241).
//!
//! The LoRA/QLoRA pipeline reads APR only. A SafeTensors or GGUF base is taken
//! through the same import path as `apr import` (`aprender::format::apr_import`),
//! which writes row-major APR: a GGUF is transposed at that boundary
//! (LAYOUT-001/002, contracts/tensor-layout-v1.yaml) and no col-major kernel is
//! involved. The format is read from the magic bytes, never the extension.
//!
//! A GGUF with quantized weights is refused: the pipeline would have to dequantize
//! it to F32 and train on values the file only approximates, so the run would not
//! be a run of the model that was named.

use crate::error::{CliError, Result};
use std::io::Read;
use std::path::{Path, PathBuf};

/// GGML dtypes that hold the weight exactly: F32, F16, BF16.
const GGML_EXACT_DTYPES: [u32; 3] = [0, 1, 30];

/// What the magic bytes of a base model file say it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BaseFormat {
    Apr,
    SafeTensors,
    Gguf,
}

impl BaseFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Apr => "apr",
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
        }
    }
}

fn refuse(path: &Path, why: &str) -> CliError {
    CliError::ValidationFailed(format!(
        "base model '{}' cannot be fine-tuned: {why}",
        path.display()
    ))
}

/// The first bytes of a file, and how long the file is.
struct Head {
    bytes: Vec<u8>,
    size: u64,
}

fn read_head(path: &Path) -> Result<Head> {
    let file = std::fs::File::open(path).map_err(|e| refuse(path, &format!("cannot open: {e}")))?;
    let size = file
        .metadata()
        .map_err(|e| refuse(path, &format!("cannot stat: {e}")))?
        .len();
    let mut bytes = Vec::with_capacity(9);
    file.take(9)
        .read_to_end(&mut bytes)
        .map_err(|e| refuse(path, &format!("cannot read: {e}")))?;
    Ok(Head { bytes, size })
}

/// APR (v2 or v1) and GGUF are named by a four-byte magic.
fn magic_format(head: &[u8]) -> Option<BaseFormat> {
    match head.get(..4)? {
        m if m == aprender::format::v2::MAGIC_V2 || m == b"APRN" => Some(BaseFormat::Apr),
        b"GGUF" => Some(BaseFormat::Gguf),
        _ => None,
    }
}

/// SafeTensors: a u64 LE header length, then a JSON object that fits the file.
fn is_safetensors_head(head: &Head) -> bool {
    let Some(len_bytes) = head
        .bytes
        .get(..8)
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
    else {
        return false;
    };
    let header_len = u64::from_le_bytes(len_bytes);
    head.bytes.get(8) == Some(&b'{') && header_len > 1 && header_len.saturating_add(8) <= head.size
}

/// Detect the format from the first bytes of the file. Anything that is not a
/// well-formed APR, GGUF or SafeTensors header is refused, before any training.
pub(crate) fn detect_base_format(path: &Path) -> Result<BaseFormat> {
    let head = read_head(path)?;
    if let Some(format) = magic_format(&head.bytes) {
        return Ok(format);
    }
    if is_safetensors_head(&head) {
        return Ok(BaseFormat::SafeTensors);
    }
    Err(refuse(
        path,
        &format!(
            "its first bytes are not an APR, SafeTensors or GGUF header \
             (file is {} bytes; the file is unsupported or corrupt)",
            head.size
        ),
    ))
}

/// Refuse a GGUF whose weights are quantized. Reads the header only through the
/// importer's own reader, so a corrupt GGUF is refused here with its parse error.
fn check_gguf_trainable(path: &Path) -> Result<()> {
    let reader = aprender::format::gguf::GgufReader::from_file(path).map_err(|e| {
        refuse(
            path,
            &format!("the GGUF header does not parse (corrupt file?): {e}"),
        )
    })?;
    let quantized: Vec<&str> = reader
        .tensors
        .iter()
        .filter(|t| !GGML_EXACT_DTYPES.contains(&t.dtype))
        .map(|t| t.name.as_str())
        .collect();
    if quantized.is_empty() {
        return Ok(());
    }
    Err(refuse(
        path,
        &format!(
            "{} of {} tensors are quantized (first: '{}'). Training would dequantize them to \
             F32 and fit values the file only approximates, so it is refused rather than done \
             silently. Use an F32, F16 or BF16 GGUF, or the .safetensors or .apr the model was \
             made from.",
            quantized.len(),
            reader.tensors.len(),
            quantized.first().copied().unwrap_or("?"),
        ),
    ))
}

/// Detect and check a base without converting it. This is what a real recipe run
/// does before it opens a model, so a bad file is refused first.
pub(crate) fn preflight_base(path: &Path) -> Result<BaseFormat> {
    let format = detect_base_format(path)?;
    if format == BaseFormat::Gguf {
        check_gguf_trainable(path)?;
    }
    Ok(format)
}

/// A base ready for `InstructPipeline::from_apr`: the APR file to read and the
/// format the user supplied. A converted base lives in a scratch directory that
/// goes away with this value.
pub(crate) struct PreparedBase {
    path: PathBuf,
    pub(crate) format: BaseFormat,
    _scratch: Option<tempfile::TempDir>,
}

impl PreparedBase {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// Detect the base format and, for SafeTensors and GGUF, convert it to APR with the
/// same importer `apr import` uses. The detected format is logged on stderr.
pub(crate) fn prepare_base(path: &Path) -> Result<PreparedBase> {
    let format = preflight_base(path)?;
    eprintln!(
        "[base] detected format={} (by magic bytes) file={}",
        format.as_str(),
        path.display()
    );
    if format == BaseFormat::Apr {
        return Ok(PreparedBase {
            path: path.to_path_buf(),
            format,
            _scratch: None,
        });
    }
    let scratch = tempfile::tempdir()
        .map_err(|e| refuse(path, &format!("cannot create a scratch directory: {e}")))?;
    let out = scratch.path().join("base.apr");
    eprintln!(
        "[base] importing {} to row-major APR (same path as `apr import`)",
        format.as_str()
    );
    aprender::format::apr_import(
        &path.display().to_string(),
        &out,
        aprender::format::ImportOptions::default(),
    )
    .map_err(|e| refuse(path, &format!("import to APR failed: {e}")))?;
    Ok(PreparedBase {
        path: out,
        format,
        _scratch: Some(scratch),
    })
}

#[cfg(test)]
#[path = "finetune_base_tests.rs"]
mod tests;
