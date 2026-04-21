//! `apr llava-lint` — CRUX-C-12 LLaVA multi-modal observation linter.
//!
//! Reads a JSON observation file that captures a single LLaVA/SigLIP vision
//! inference run and dispatches four classifiers (image_tokens,
//! caption_parity, mmproj_compat, image_format). Emits a text or `--json`
//! report.
//!
//! Spec: `contracts/crux-C-12-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing sections
//! skip the corresponding classifier):
//!
//!   {
//!     "image_tokens": { "arch": "llava15" | "siglip", "got": 576 },
//!     "caption":      { "apr": "a cat", "golden": "a cat" },
//!     "mmproj":       { "arch": "clip", "projection_dim": 4096,
//!                       "hidden_size": 4096 },
//!     "image_format": { "filename": "photo.jpg" }
//!   }

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::llava_classifier as clf;
use crate::error::{CliError, Result};

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr llava-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let image_tokens = classify_image_tokens(&obs);
    let caption = classify_caption(&obs);
    let mmproj = classify_mmproj(&obs);
    let image_format = classify_image_format(&obs);

    let fail_reasons: Vec<String> = [
        image_tokens.as_ref().and_then(image_tokens_fail_reason),
        caption.as_ref().and_then(caption_fail_reason),
        mmproj.as_ref().and_then(mmproj_fail_reason),
        image_format.as_ref().and_then(image_format_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        image_tokens.as_ref(),
        caption.as_ref(),
        mmproj.as_ref(),
        image_format.as_ref(),
        json,
    );

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn classify_image_tokens(obs: &Value) -> Option<clf::ImageTokenCountOutcome> {
    let sec = obs.get("image_tokens")?.as_object()?;
    let arch_str = sec.get("arch")?.as_str()?;
    let arch = parse_vision_arch(arch_str)?;
    let got = sec.get("got")?.as_u64()? as u32;
    Some(clf::classify_image_token_count(arch, got))
}

fn parse_vision_arch(s: &str) -> Option<clf::VisionArch> {
    match s.to_ascii_lowercase().as_str() {
        "llava15" | "llava-1.5" | "clip" => Some(clf::VisionArch::Llava15),
        "siglip" => Some(clf::VisionArch::Siglip),
        _ => None,
    }
}

fn classify_caption(obs: &Value) -> Option<clf::CaptionParityOutcome> {
    let sec = obs.get("caption")?.as_object()?;
    let apr_cap = sec.get("apr")?.as_str()?;
    let golden = sec.get("golden")?.as_str()?;
    Some(clf::classify_caption_parity(apr_cap, golden))
}

fn classify_mmproj(obs: &Value) -> Option<clf::MmprojCompatOutcome> {
    let sec = obs.get("mmproj")?.as_object()?;
    let arch = sec.get("arch")?.as_str()?;
    let proj_dim = sec.get("projection_dim")?.as_u64()? as u32;
    let hidden = sec.get("hidden_size")?.as_u64()? as u32;
    Some(clf::classify_mmproj_compatibility(arch, proj_dim, hidden))
}

fn classify_image_format(obs: &Value) -> Option<clf::ImageFormatOutcome> {
    let sec = obs.get("image_format")?.as_object()?;
    let filename = sec.get("filename")?.as_str()?;
    Some(clf::classify_image_format(filename))
}

fn image_tokens_fail_reason(o: &clf::ImageTokenCountOutcome) -> Option<String> {
    match o {
        clf::ImageTokenCountOutcome::Ok => None,
        clf::ImageTokenCountOutcome::ZeroImageTokens => Some(
            "FALSIFY-CRUX-C-12-001 image_tokens: projector emitted zero image tokens".to_string(),
        ),
        clf::ImageTokenCountOutcome::Mismatch {
            arch,
            expected,
            got,
        } => Some(format!(
            "FALSIFY-CRUX-C-12-001 image_tokens: {arch:?} expected {expected}, got {got}"
        )),
    }
}

fn caption_fail_reason(o: &clf::CaptionParityOutcome) -> Option<String> {
    match o {
        clf::CaptionParityOutcome::Ok => None,
        clf::CaptionParityOutcome::EmptinessMismatch {
            apr_empty,
            golden_empty,
        } => Some(format!(
            "FALSIFY-CRUX-C-12-002 caption: emptiness mismatch apr_empty={apr_empty} golden_empty={golden_empty}"
        )),
        clf::CaptionParityOutcome::LengthMismatch {
            apr_len,
            golden_len,
        } => Some(format!(
            "FALSIFY-CRUX-C-12-002 caption: length mismatch apr={apr_len} golden={golden_len}"
        )),
        clf::CaptionParityOutcome::ByteDivergence {
            at_index,
            apr_byte,
            golden_byte,
        } => Some(format!(
            "FALSIFY-CRUX-C-12-002 caption: byte divergence at idx {at_index}: apr=0x{apr_byte:02x} golden=0x{golden_byte:02x}"
        )),
    }
}

fn mmproj_fail_reason(o: &clf::MmprojCompatOutcome) -> Option<String> {
    match o {
        clf::MmprojCompatOutcome::Ok => None,
        clf::MmprojCompatOutcome::UnsupportedArch { got } => Some(format!(
            "FALSIFY-CRUX-C-12-003 mmproj: unsupported arch {got:?} (expected clip|siglip)"
        )),
        clf::MmprojCompatOutcome::ZeroDim { which } => Some(format!(
            "FALSIFY-CRUX-C-12-003 mmproj: {which} is zero"
        )),
        clf::MmprojCompatOutcome::ProjectionDimMismatch {
            projection_dim,
            hidden_size,
        } => Some(format!(
            "FALSIFY-CRUX-C-12-003 mmproj: projection_dim {projection_dim} != hidden_size {hidden_size}"
        )),
    }
}

fn image_format_fail_reason(o: &clf::ImageFormatOutcome) -> Option<String> {
    match o {
        clf::ImageFormatOutcome::Ok { .. } => None,
        clf::ImageFormatOutcome::EmptyFilename => Some(
            "FALSIFY-CRUX-C-12-004 image_format: filename is empty".to_string(),
        ),
        clf::ImageFormatOutcome::MissingExtension => Some(
            "FALSIFY-CRUX-C-12-004 image_format: filename has no extension".to_string(),
        ),
        clf::ImageFormatOutcome::UnsupportedExtension { got } => Some(format!(
            "FALSIFY-CRUX-C-12-004 image_format: unsupported extension {got:?} (expected jpg|jpeg|png|bmp)"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &Path,
    image_tokens: Option<&clf::ImageTokenCountOutcome>,
    caption: Option<&clf::CaptionParityOutcome>,
    mmproj: Option<&clf::MmprojCompatOutcome>,
    image_format: Option<&clf::ImageFormatOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "image_tokens": image_tokens.map(|o| format!("{o:?}")),
            "caption":      caption.map(|o| format!("{o:?}")),
            "mmproj":       mmproj.map(|o| format!("{o:?}")),
            "image_format": image_format.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("llava-lint report for {}", path.display());
        print_line("  image_tokens: ", image_tokens.map(|o| format!("{o:?}")));
        print_line("  caption:      ", caption.map(|o| format!("{o:?}")));
        print_line("  mmproj:       ", mmproj.map(|o| format!("{o:?}")));
        print_line("  image_format: ", image_format.map(|o| format!("{o:?}")));
    }
}

fn print_line(prefix: &str, v: Option<String>) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}(missing fields — classifier skipped)"),
    }
}
