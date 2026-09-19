//! Text as glyph outlines.
//!
//! APEX-001 EV-2a rule 1. Moved from `rmedia-text` (never published, F-07), so copied.
//!
//! # The font is bytes the caller pins
//!
//! There is deliberately no way to ask this module to find a font. `fontdb.load_system_fonts()`
//! and friends read a machine's installed font set, which is an **undeclared input**: the same
//! source, the same code and the same toolchain then render differently on two machines, and
//! nothing in the output records why. Every entry point here takes `font_bytes`, so the font is
//! as pinned as the source is.
//!
//! # Why outlines at all
//!
//! An SVG carrying `<text font-family="sans-serif">` does not describe a figure; it describes a
//! request that the *viewer* find a font. Two readers see two images and neither is wrong. Glyph
//! outlines close that: the shapes are in the file.

use kurbo::{Affine, BezPath, Vec2};
use skrifa::{
    instance::{LocationRef, Size},
    metrics::GlyphMetrics,
    outline::{DrawSettings, OutlinePen},
    raw::FontRef,
    GlyphId, MetadataProvider,
};

/// One glyph of a shaping run.
///
/// Coordinates are in pixels at the requested size. `x_offset` is the cumulative horizontal
/// advance from the start of the run, precomputed so callers need not re-fold it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapedGlyph {
    /// OpenType glyph index, as `u32` so the `skrifa::GlyphId` type does not cross this boundary.
    pub glyph_id: u32,
    /// Codepoint that produced this glyph.
    pub codepoint: u32,
    /// Advance width in pixels at the requested size.
    pub x_advance: f32,
    /// Cumulative pen position at the START of this glyph, i.e. `Σ x_advance[..i]`.
    pub x_offset: f32,
    /// Vertical baseline offset. Always 0.0: no vertical or stacked layout.
    pub y_offset: f32,
}

/// skrifa `OutlinePen` → kurbo `BezPath` adapter.
///
/// skrifa emits Y-up font units; kurbo and SVG are Y-down, so every node flips Y and callers
/// receive screen-space paths (origin top-left, baseline at `y = 0`).
struct PathSink<'a>(&'a mut BezPath);

impl OutlinePen for PathSink<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((f64::from(x), f64::from(-y)));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((f64::from(x), f64::from(-y)));
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0.quad_to((f64::from(cx), f64::from(-cy)), (f64::from(x), f64::from(-y)));
    }
    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        self.0.curve_to(
            (f64::from(c0x), f64::from(-c0y)),
            (f64::from(c1x), f64::from(-c1y)),
            (f64::from(x), f64::from(-y)),
        );
    }
    fn close(&mut self) {
        self.0.close_path();
    }
}

/// Shape `text` against `font_bytes` at `size_px`.
///
/// ASCII codepoint shaping: each `char` maps 1:1 to a [`ShapedGlyph`]; a codepoint absent from
/// the font's charmap yields glyph 0 (notdef) with the notdef advance.
///
/// Returns an empty `Vec` for empty text, malformed font bytes, or a non-finite/non-positive
/// size. Never panics.
#[must_use]
pub fn shape(text: &str, font_bytes: &[u8], size_px: f64) -> Vec<ShapedGlyph> {
    if text.is_empty() || !size_px.is_finite() || size_px <= 0.0 {
        return Vec::new();
    }
    let Ok(font) = FontRef::new(font_bytes) else {
        return Vec::new();
    };
    // skrifa's raster grid is f32; the public surface is f64 (EV-2c moved these paths to f64) and
    // narrows here, at the one point the upstream API fixes.
    let size_f32 = size_px as f32;
    let charmap = font.charmap();
    let size = Size::new(size_f32);
    let loc = LocationRef::default();
    let metrics: GlyphMetrics = font.glyph_metrics(size, loc);

    let mut pen = 0.0_f32;
    let mut out = Vec::new();
    for ch in text.chars() {
        let gid = charmap.map(ch).unwrap_or(GlyphId::new(0));
        let adv = metrics.advance_width(gid).unwrap_or(0.0);
        out.push(ShapedGlyph {
            glyph_id: gid.to_u32(),
            codepoint: ch as u32,
            x_advance: adv,
            x_offset: pen,
            y_offset: 0.0,
        });
        pen += adv;
    }
    out
}

/// Shape `text` and concatenate every glyph's outline into one [`BezPath`], each translated by
/// its cumulative advance so the result lays out left-to-right.
///
/// Returns an empty path for empty text, a malformed font, or an invalid size — never panics.
///
/// ```
/// use trueno_viz::text::to_path;
/// // No font bytes: an empty path, not a panic and not a fallback font.
/// assert!(to_path("hello", &[], 16.0).is_empty());
/// ```
#[must_use]
pub fn to_path(text: &str, font_bytes: &[u8], size_px: f64) -> BezPath {
    let glyphs = shape(text, font_bytes, size_px);
    if glyphs.is_empty() {
        return BezPath::new();
    }
    let Ok(font) = FontRef::new(font_bytes) else {
        return BezPath::new();
    };
    let size = Size::new(size_px as f32);
    let loc = LocationRef::default();
    let outlines = font.outline_glyphs();

    let mut out = BezPath::new();
    for g in &glyphs {
        let Some(outline) = outlines.get(GlyphId::new(g.glyph_id)) else {
            continue;
        };
        let mut glyph_path = BezPath::new();
        let mut sink = PathSink(&mut glyph_path);
        if outline.draw(DrawSettings::unhinted(size, loc), &mut sink).is_err() {
            continue;
        }
        out.extend(Affine::translate(Vec2::new(f64::from(g.x_offset), 0.0)) * glyph_path);
    }
    out
}

/// The SVG path data (`d` attribute) for `text`, with coordinates quantised to the grid.
///
/// This is the form the SVG writer emits, so a figure carries shapes rather than a font request.
/// See [`crate::quantise`] for why the coordinates are rounded.
#[must_use]
pub fn to_path_data(text: &str, font_bytes: &[u8], size_px: f64) -> String {
    to_path_data_at(text, font_bytes, size_px, 0.0, 0.0)
}

/// As [`to_path_data`], with the run's baseline origin placed at `(x, y)`.
///
/// The translation is applied to the geometry, not emitted as a `transform` attribute, so the
/// numbers in the file are the final coordinates and the quantisation grid applies to them
/// rather than to a pre-transform intermediate.
#[must_use]
pub fn to_path_data_at(text: &str, font_bytes: &[u8], size_px: f64, x: f64, y: f64) -> String {
    let path = to_path(text, font_bytes, size_px);
    if path.is_empty() {
        return String::new();
    }
    let placed = Affine::translate(Vec2::new(x, y)) * path;
    crate::quantise_path_data(&placed.to_svg())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_font_bytes_gives_an_empty_path_not_a_panic() {
        assert!(to_path("hello", &[], 16.0).is_empty());
        assert!(shape("hello", &[], 16.0).is_empty());
    }

    #[test]
    fn an_invalid_size_gives_an_empty_path() {
        assert!(shape("x", &[], f64::NAN).is_empty());
        assert!(shape("x", &[], 0.0).is_empty());
        assert!(shape("x", &[], -1.0).is_empty());
    }

    #[test]
    fn empty_text_gives_an_empty_path() {
        assert!(to_path("", &[], 16.0).is_empty());
    }
}
