//! LLaVA-style multi-modal vision classifier (CRUX-C-12).
//!
//! Four pure, deterministic classifiers that discharge
//! FALSIFY-CRUX-C-12-{001..004} at the PARTIAL_ALGORITHM_LEVEL:
//!
//!   * `classify_image_token_count` — vision projector output length
//!     must match the declared `VisionArch`'s `N_img_tokens` (LLaVA-1.5
//!     = 576, SigLIP = 729).
//!   * `classify_caption_parity` — at temp=0 top_k=1, apr caption bytes
//!     equal the llama-llava-cli golden bytes; first-divergence byte
//!     index reported deterministically.
//!   * `classify_mmproj_compatibility` — mmproj architecture ∈
//!     {"clip", "siglip"} AND `projection_dim == hidden_size`.
//!   * `classify_image_format` — file extension ∈ {jpg, jpeg, png, bmp};
//!     other formats rejected with a specific reason.
//!
//! Full discharge blocks on the `apr run --mmproj --image` surface
//! integrating a real CLIP/SigLIP vision tower.

/// Canonical image-token counts per vision-tower architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionArch {
    Llava15,
    Siglip,
}

impl VisionArch {
    pub const fn expected_image_tokens(self) -> u32 {
        match self {
            VisionArch::Llava15 => 576,
            VisionArch::Siglip => 729,
        }
    }
    pub const fn arch_name(self) -> &'static str {
        match self {
            VisionArch::Llava15 => "clip",
            VisionArch::Siglip => "siglip",
        }
    }
}

pub const LLAVA15_IMAGE_TOKENS: u32 = 576;
pub const SIGLIP_IMAGE_TOKENS: u32 = 729;

/// Supported image-file extensions (lowercase, without leading dot).
pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp"];

/// Outcome of `classify_image_token_count`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageTokenCountOutcome {
    /// `got == arch.expected_image_tokens()`.
    Ok,
    /// `got` disagrees with the arch's expected count.
    Mismatch {
        arch: VisionArch,
        expected: u32,
        got: u32,
    },
    /// `got == 0` — splicing produced no image tokens.
    ZeroImageTokens,
}

/// Vision-projector length gate.
pub fn classify_image_token_count(
    arch: VisionArch,
    got_image_tokens: u32,
) -> ImageTokenCountOutcome {
    if got_image_tokens == 0 {
        return ImageTokenCountOutcome::ZeroImageTokens;
    }
    let expected = arch.expected_image_tokens();
    if got_image_tokens != expected {
        return ImageTokenCountOutcome::Mismatch {
            arch,
            expected,
            got: got_image_tokens,
        };
    }
    ImageTokenCountOutcome::Ok
}

/// Outcome of `classify_caption_parity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptionParityOutcome {
    /// apr caption matches golden byte-for-byte.
    Ok,
    /// Lengths differ.
    LengthMismatch { apr_len: usize, golden_len: usize },
    /// Same length but diverges at some byte.
    ByteDivergence {
        at_index: usize,
        apr_byte: u8,
        golden_byte: u8,
    },
    /// apr caption is empty but golden is not (or vice versa).
    EmptinessMismatch {
        apr_empty: bool,
        golden_empty: bool,
    },
}

/// Byte-for-byte greedy-decode parity at temp=0 top_k=1.
pub fn classify_caption_parity(apr_caption: &str, golden: &str) -> CaptionParityOutcome {
    if apr_caption.is_empty() != golden.is_empty() {
        return CaptionParityOutcome::EmptinessMismatch {
            apr_empty: apr_caption.is_empty(),
            golden_empty: golden.is_empty(),
        };
    }
    let a = apr_caption.as_bytes();
    let g = golden.as_bytes();
    if a.len() != g.len() {
        return CaptionParityOutcome::LengthMismatch {
            apr_len: a.len(),
            golden_len: g.len(),
        };
    }
    for (i, (&ab, &gb)) in a.iter().zip(g.iter()).enumerate() {
        if ab != gb {
            return CaptionParityOutcome::ByteDivergence {
                at_index: i,
                apr_byte: ab,
                golden_byte: gb,
            };
        }
    }
    CaptionParityOutcome::Ok
}

/// Outcome of `classify_mmproj_compatibility`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MmprojCompatOutcome {
    Ok,
    /// Unknown / unsupported vision architecture name.
    UnsupportedArch { got: String },
    /// `projection_dim != language_model.hidden_size`.
    ProjectionDimMismatch {
        projection_dim: u32,
        hidden_size: u32,
    },
    /// `projection_dim == 0` or `hidden_size == 0`.
    ZeroDim { which: &'static str },
}

/// Compatibility gate run BEFORE the first inference step.
pub fn classify_mmproj_compatibility(
    mmproj_arch: &str,
    projection_dim: u32,
    language_hidden_size: u32,
) -> MmprojCompatOutcome {
    let normalized = mmproj_arch.to_ascii_lowercase();
    if normalized != "clip" && normalized != "siglip" {
        return MmprojCompatOutcome::UnsupportedArch {
            got: mmproj_arch.to_string(),
        };
    }
    if projection_dim == 0 {
        return MmprojCompatOutcome::ZeroDim {
            which: "projection_dim",
        };
    }
    if language_hidden_size == 0 {
        return MmprojCompatOutcome::ZeroDim {
            which: "hidden_size",
        };
    }
    if projection_dim != language_hidden_size {
        return MmprojCompatOutcome::ProjectionDimMismatch {
            projection_dim,
            hidden_size: language_hidden_size,
        };
    }
    MmprojCompatOutcome::Ok
}

/// Outcome of `classify_image_format`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageFormatOutcome {
    /// Extension is in the supported set.
    Ok { normalized_ext: String },
    /// Extension is not in the supported set.
    UnsupportedExtension { got: String },
    /// No extension on the filename.
    MissingExtension,
    /// Filename is empty.
    EmptyFilename,
}

/// Image-format gate. Compares the *last* filename component's extension
/// (lowercased) against the supported set.
pub fn classify_image_format(filename: &str) -> ImageFormatOutcome {
    if filename.is_empty() {
        return ImageFormatOutcome::EmptyFilename;
    }
    let last_component = filename
        .rsplit_once('/')
        .map_or(filename, |(_, tail)| tail);
    let last_component = last_component
        .rsplit_once('\\')
        .map_or(last_component, |(_, tail)| tail);
    let ext = match last_component.rsplit_once('.') {
        Some((_, e)) if !e.is_empty() => e.to_ascii_lowercase(),
        _ => return ImageFormatOutcome::MissingExtension,
    };
    if SUPPORTED_IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        ImageFormatOutcome::Ok {
            normalized_ext: ext,
        }
    } else {
        ImageFormatOutcome::UnsupportedExtension { got: ext }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- image token count ------------------------------------------------

    #[test]
    fn image_tokens_ok_llava15() {
        assert_eq!(
            classify_image_token_count(VisionArch::Llava15, 576),
            ImageTokenCountOutcome::Ok
        );
    }

    #[test]
    fn image_tokens_ok_siglip() {
        assert_eq!(
            classify_image_token_count(VisionArch::Siglip, 729),
            ImageTokenCountOutcome::Ok
        );
    }

    #[test]
    fn image_tokens_rejects_zero() {
        assert_eq!(
            classify_image_token_count(VisionArch::Llava15, 0),
            ImageTokenCountOutcome::ZeroImageTokens
        );
    }

    #[test]
    fn image_tokens_rejects_mismatch() {
        assert_eq!(
            classify_image_token_count(VisionArch::Llava15, 729),
            ImageTokenCountOutcome::Mismatch {
                arch: VisionArch::Llava15,
                expected: 576,
                got: 729,
            }
        );
    }

    #[test]
    fn image_tokens_rejects_siglip_with_llava_count() {
        assert_eq!(
            classify_image_token_count(VisionArch::Siglip, 576),
            ImageTokenCountOutcome::Mismatch {
                arch: VisionArch::Siglip,
                expected: 729,
                got: 576,
            }
        );
    }

    #[test]
    fn image_tokens_classifier_is_deterministic() {
        let a = classify_image_token_count(VisionArch::Llava15, 576);
        let b = classify_image_token_count(VisionArch::Llava15, 576);
        assert_eq!(a, b);
    }

    // ---- caption parity ---------------------------------------------------

    #[test]
    fn caption_ok_on_byte_identical() {
        assert_eq!(
            classify_caption_parity("a photo of a cat", "a photo of a cat"),
            CaptionParityOutcome::Ok
        );
    }

    #[test]
    fn caption_ok_on_two_empty() {
        assert_eq!(
            classify_caption_parity("", ""),
            CaptionParityOutcome::Ok
        );
    }

    #[test]
    fn caption_rejects_emptiness_mismatch() {
        assert_eq!(
            classify_caption_parity("", "cat"),
            CaptionParityOutcome::EmptinessMismatch {
                apr_empty: true,
                golden_empty: false,
            }
        );
        assert_eq!(
            classify_caption_parity("cat", ""),
            CaptionParityOutcome::EmptinessMismatch {
                apr_empty: false,
                golden_empty: true,
            }
        );
    }

    #[test]
    fn caption_rejects_length_mismatch() {
        assert_eq!(
            classify_caption_parity("abc", "abcd"),
            CaptionParityOutcome::LengthMismatch {
                apr_len: 3,
                golden_len: 4,
            }
        );
    }

    #[test]
    fn caption_rejects_byte_divergence() {
        assert_eq!(
            classify_caption_parity("a cat", "a bat"),
            CaptionParityOutcome::ByteDivergence {
                at_index: 2,
                apr_byte: b'c',
                golden_byte: b'b',
            }
        );
    }

    #[test]
    fn caption_classifier_is_deterministic() {
        let a = classify_caption_parity("hi", "hi");
        let b = classify_caption_parity("hi", "hi");
        assert_eq!(a, b);
    }

    // ---- mmproj compatibility --------------------------------------------

    #[test]
    fn mmproj_ok_on_clip_matching_dim() {
        assert_eq!(
            classify_mmproj_compatibility("clip", 4096, 4096),
            MmprojCompatOutcome::Ok
        );
    }

    #[test]
    fn mmproj_ok_on_siglip_case_insensitive() {
        assert_eq!(
            classify_mmproj_compatibility("SigLIP", 2048, 2048),
            MmprojCompatOutcome::Ok
        );
    }

    #[test]
    fn mmproj_rejects_unknown_arch() {
        assert_eq!(
            classify_mmproj_compatibility("dinov2", 4096, 4096),
            MmprojCompatOutcome::UnsupportedArch {
                got: "dinov2".to_string()
            }
        );
    }

    #[test]
    fn mmproj_rejects_zero_projection_dim() {
        assert_eq!(
            classify_mmproj_compatibility("clip", 0, 4096),
            MmprojCompatOutcome::ZeroDim {
                which: "projection_dim"
            }
        );
    }

    #[test]
    fn mmproj_rejects_zero_hidden_size() {
        assert_eq!(
            classify_mmproj_compatibility("clip", 4096, 0),
            MmprojCompatOutcome::ZeroDim {
                which: "hidden_size"
            }
        );
    }

    #[test]
    fn mmproj_rejects_dim_mismatch() {
        assert_eq!(
            classify_mmproj_compatibility("clip", 1024, 4096),
            MmprojCompatOutcome::ProjectionDimMismatch {
                projection_dim: 1024,
                hidden_size: 4096,
            }
        );
    }

    #[test]
    fn mmproj_classifier_is_deterministic() {
        let a = classify_mmproj_compatibility("clip", 4096, 4096);
        let b = classify_mmproj_compatibility("clip", 4096, 4096);
        assert_eq!(a, b);
    }

    // ---- image format -----------------------------------------------------

    #[test]
    fn image_format_ok_jpg() {
        assert_eq!(
            classify_image_format("photo.jpg"),
            ImageFormatOutcome::Ok {
                normalized_ext: "jpg".to_string()
            }
        );
    }

    #[test]
    fn image_format_ok_jpeg_case_insensitive() {
        assert_eq!(
            classify_image_format("PHOTO.JPEG"),
            ImageFormatOutcome::Ok {
                normalized_ext: "jpeg".to_string()
            }
        );
    }

    #[test]
    fn image_format_ok_png_with_path() {
        assert_eq!(
            classify_image_format("/tmp/foo/bar.png"),
            ImageFormatOutcome::Ok {
                normalized_ext: "png".to_string()
            }
        );
    }

    #[test]
    fn image_format_ok_bmp() {
        assert_eq!(
            classify_image_format("icon.bmp"),
            ImageFormatOutcome::Ok {
                normalized_ext: "bmp".to_string()
            }
        );
    }

    #[test]
    fn image_format_rejects_unsupported() {
        assert_eq!(
            classify_image_format("clip.mp4"),
            ImageFormatOutcome::UnsupportedExtension {
                got: "mp4".to_string()
            }
        );
        assert_eq!(
            classify_image_format("doc.pdf"),
            ImageFormatOutcome::UnsupportedExtension {
                got: "pdf".to_string()
            }
        );
    }

    #[test]
    fn image_format_rejects_missing_extension() {
        assert_eq!(
            classify_image_format("README"),
            ImageFormatOutcome::MissingExtension
        );
        assert_eq!(
            classify_image_format("README."),
            ImageFormatOutcome::MissingExtension
        );
    }

    #[test]
    fn image_format_rejects_empty_filename() {
        assert_eq!(
            classify_image_format(""),
            ImageFormatOutcome::EmptyFilename
        );
    }

    #[test]
    fn image_format_ignores_directory_dots() {
        // "a.b/c" has no extension on "c"
        assert_eq!(
            classify_image_format("a.b/c"),
            ImageFormatOutcome::MissingExtension
        );
    }

    #[test]
    fn image_format_classifier_is_deterministic() {
        let a = classify_image_format("foo.png");
        let b = classify_image_format("foo.png");
        assert_eq!(a, b);
    }

    // ---- constants ------------------------------------------------------

    #[test]
    fn vision_arch_constants_are_canonical() {
        assert_eq!(VisionArch::Llava15.expected_image_tokens(), 576);
        assert_eq!(VisionArch::Siglip.expected_image_tokens(), 729);
        assert_eq!(VisionArch::Llava15.arch_name(), "clip");
        assert_eq!(VisionArch::Siglip.arch_name(), "siglip");
        assert_eq!(LLAVA15_IMAGE_TOKENS, 576);
        assert_eq!(SIGLIP_IMAGE_TOKENS, 729);
    }

    #[test]
    fn supported_extensions_are_canonical() {
        assert_eq!(
            SUPPORTED_IMAGE_EXTENSIONS,
            &["jpg", "jpeg", "png", "bmp"]
        );
    }
}
