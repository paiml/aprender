//! PNG output encoder.
//!
//! Pure Rust PNG encoding using the `png` crate.
//!
//! # Every encoder setting is pinned
//!
//! APEX-001 EV-2a rule 3. The bytes of a PNG are not determined by its pixels alone: the
//! compression level, the row filter, and whether filtering adapts per row all change the
//! output for identical input, and every one of them has a *default* that the `png` crate is
//! free to change in a patch release. A figure whose hash moves because a dependency was
//! updated is not a reproducible figure.
//!
//! So the encoder states all three explicitly, and writes no timestamp or text chunk:
//! `tIME` would put the wall clock in the file, and `tEXt`/`iTXt`/`zTXt` would put the
//! encoder's identity there. Neither is a property of the image.
//!
//! `Compression::Fast` and `FilterType::NoFilter` are chosen over the smaller
//! `Best`/`Paeth` because the property that matters here is that the bytes are a pure
//! function of the pixels and stay stable across `png` patch versions — not that the file is
//! small. Fewer moving parts in the encoder is fewer things that can change underneath a hash.

use crate::error::Result;
use crate::framebuffer::Framebuffer;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Fix every setting that can move the bytes for a fixed set of pixels.
///
/// One function, called from both entry points, so `write_to_file` and `to_bytes` cannot drift
/// apart — a figure written to disk and the same figure hashed in memory must be the same file.
fn pin<W: std::io::Write>(encoder: &mut png::Encoder<'_, W>) {
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.set_filter(png::FilterType::NoFilter);
    encoder.set_adaptive_filter(png::AdaptiveFilterType::NonAdaptive);
    // No set_time / add_text_chunk / add_ztxt_chunk: a wall clock and an encoder name are not
    // properties of the image, and both would move the hash on every run.
}

/// PNG encoder for framebuffer output.
pub struct PngEncoder;

impl PngEncoder {
    /// Write a framebuffer to a PNG file.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation or PNG encoding fails.
    pub fn write_to_file<P: AsRef<Path>>(fb: &Framebuffer, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        let mut encoder = png::Encoder::new(writer, fb.width(), fb.height());
        pin(&mut encoder);

        let mut writer = encoder.write_header()?;
        // Use compact pixels to handle stride padding
        writer.write_image_data(&fb.to_compact_pixels())?;

        Ok(())
    }

    /// Encode a framebuffer to PNG bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if PNG encoding fails.
    pub fn to_bytes(fb: &Framebuffer) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        {
            let mut encoder = png::Encoder::new(&mut buffer, fb.width(), fb.height());
            pin(&mut encoder);

            let mut writer = encoder.write_header()?;
            // Use compact pixels to handle stride padding
            writer.write_image_data(&fb.to_compact_pixels())?;
        }

        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;

    #[test]
    fn test_png_to_bytes() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::RED);

        let bytes = PngEncoder::to_bytes(&fb).expect("encoding should succeed");
        // PNG magic bytes
        assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn test_png_write_to_file() {
        let mut fb = Framebuffer::new(8, 8).expect("framebuffer creation should succeed");
        fb.clear(Rgba::BLUE);

        let tmp = tempfile::NamedTempFile::new().expect("temp file creation should succeed");
        let path = tmp.path();

        PngEncoder::write_to_file(&fb, path).expect("file write should succeed");

        // Verify file was written and has PNG header
        let data = std::fs::read(path).expect("file read should succeed");
        assert_eq!(&data[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(data.len() > 8);
    }

    #[test]
    fn test_png_roundtrip_dimensions() {
        let mut fb = Framebuffer::new(16, 24).expect("framebuffer creation should succeed");
        fb.clear(Rgba::GREEN);

        let bytes = PngEncoder::to_bytes(&fb).expect("encoding should succeed");

        // Decode to verify dimensions are correct in header
        // PNG IHDR chunk starts at byte 8, width at 16, height at 20
        // IHDR: length(4) + "IHDR"(4) + width(4) + height(4)
        let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);

        assert_eq!(width, 16);
        assert_eq!(height, 24);
    }

    #[test]
    fn test_png_various_sizes() {
        for (w, h) in [(1, 1), (100, 50), (3, 7)] {
            let fb = Framebuffer::new(w, h).expect("framebuffer creation should succeed");
            let bytes = PngEncoder::to_bytes(&fb).expect("encoding should succeed");
            assert!(!bytes.is_empty());
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        }
    }
}
