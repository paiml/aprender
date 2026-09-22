//! Rasterisation through a pinned, font-free `resvg` (APEX-001 EV-2a rule 3, EV-2d,
//! paiml/aprender#3666).
//!
//! Three rules hold for [`svg_to_png`], in order:
//!
//! 1. **`scale` must be finite and strictly positive.** Anything else (zero, negative, NaN,
//!    infinity) cannot produce a sane pixel size and is refused before any parsing happens.
//! 2. **A pre-scan refuses text, font, and file-reading elements before `usvg` ever builds a
//!    tree.** `usvg::Options::default()` carries no font database (this crate turns off
//!    resvg's `text`/`system-fonts` features — see `Cargo.toml` — so no font database is
//!    compiled in, and "no font database" holds by construction, not by omitting a call). With
//!    no fonts, usvg silently *drops* every `<text>`-family element instead of failing
//!    (usvg-0.45.1 `src/parser/text.rs:144`), which would render a wrong, silently-incomplete
//!    picture rather than an error — and a raster is exactly the artefact nobody re-reads as
//!    text to notice. `<image>`/`<feImage>` are refused for a second, independent reason: the
//!    default href resolver treats the href as a filesystem path and calls `std::fs::read` on
//!    it, and can load an arbitrary *nested* SVG document that this pre-scan never sees
//!    (usvg-0.45.1 `src/parser/image.rs:85-110`) — a host file read and a text-refusal bypass
//!    in one element. `<foreignObject>` carries arbitrary embedded content that usvg also drops
//!    silently, so it is refused rather than inspected. The pre-scan walks `roxmltree`'s own
//!    parse of the document (not `usvg`'s), and treats an element as SVG both when it declares
//!    the SVG namespace and when it declares none at all — `usvg` accepts both forms
//!    (`usvg-0.45.1 src/parser/svgtree/parse.rs:139`), so a document with no `xmlns` must not
//!    slip past a pre-scan that only recognised the namespaced form.
//! 3. **The `image_href_resolver` is replaced too, in defence of rule 2 rather than instead of
//!    it.** If the refused-element list above is ever edited and a hole opens, the resolver
//!    still refuses to read anything from disk: both `resolve_data` and `resolve_string` return
//!    `None` unconditionally.
//!
//! Pixels come off the pinned [`resvg::tiny_skia::Pixmap`] premultiplied; PNG wants straight
//! alpha, so every pixel is demultiplied (a correctly-rounded, deterministic division) on the
//! way into the [`Framebuffer`], and the framebuffer is encoded with this crate's own pinned
//! [`crate::output::PngEncoder`] — never `Pixmap::encode_png`, which is a different encoder with
//! its own, unpinned defaults.
//!
//! [`svg_to_png`] is split into the `scale`/pre-scan checks and a private [`render_unscanned`]
//! that does the actual rendering, so a unit test can observe a property (no font database
//! compiled in) that the pre-scan would otherwise hide from anything reachable through the
//! public entry point.

use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::output::PngEncoder;

/// Elements refused by the pre-scan, grouped by why (see the module doc for the reasoning).
const REFUSED: &[&str] = &[
    // Text family: usvg with no font database silently drops these instead of failing.
    "text",
    "tspan",
    "textPath",
    "tref",
    "altGlyph",
    "textArea",
    // Fonts: no font should ever be resolved on the raster path.
    "font",
    "font-face",
    // Opaque or file-reading content: usvg either drops it silently (foreignObject) or its
    // default resolver reads the filesystem and can load a nested SVG document (image,
    // feImage) that this pre-scan cannot see.
    "foreignObject",
    "image",
    "feImage",
];

/// The SVG namespace URI. `usvg` treats an element as SVG when it declares this namespace OR
/// declares none at all (usvg-0.45.1 `src/parser/svgtree/parse.rs:139`).
const SVG_NS: &str = "http://www.w3.org/2000/svg";

/// Rasterise `svg` to PNG bytes at `scale`, through a pinned, font-free `resvg`.
///
/// See the module documentation for the three rules this function enforces.
///
/// # Errors
///
/// Returns [`Error::Rendering`] if `scale` is not finite and positive, if `svg` does not parse
/// as XML, if `usvg` cannot build a tree from it, or if the computed pixel size is not finite,
/// is smaller than one pixel on either axis, or overflows `u32`. Returns
/// [`Error::InvalidDimensions`] if the computed size cannot be allocated as a pixmap. Returns
/// [`Error::SvgElementRefused`] if the pre-scan finds a refused element (see [`REFUSED`]).
pub fn svg_to_png(svg: &str, scale: f64) -> Result<Vec<u8>> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Rendering(format!(
            "scale must be finite and greater than zero, got {scale}"
        )));
    }

    prescan_refuse(svg)?;

    render_unscanned(svg, scale)
}

/// The rendering mechanism with the pre-scan policy lifted off.
///
/// [`svg_to_png`] is the only production caller, and it always calls [`prescan_refuse`] first;
/// this function does not repeat that check. It exists as a separate, private function so the
/// unit test below can call it directly and observe a property the pre-scan otherwise hides:
/// that no font database is compiled into this crate's `resvg`, so `usvg` silently drops any
/// `<text>`-family element rather than rendering it. Through `svg_to_png` that document would
/// never reach `usvg` at all, because the pre-scan refuses it first.
fn render_unscanned(svg: &str, scale: f64) -> Result<Vec<u8>> {
    let mut opt = resvg::usvg::Options::default();
    // Rule 3: defence in depth. Never read the filesystem, even if REFUSED above is edited.
    opt.image_href_resolver = resvg::usvg::ImageHrefResolver {
        resolve_data: Box::new(
            |_mime: &str, _data: std::sync::Arc<Vec<u8>>, _opt: &resvg::usvg::Options| None,
        ),
        resolve_string: Box::new(|_href: &str, _opt: &resvg::usvg::Options| None),
    };

    let tree = resvg::usvg::Tree::from_str(svg, &opt)
        .map_err(|e| Error::Rendering(format!("usvg failed to build a tree: {e}")))?;

    let w = (f64::from(tree.size().width()) * scale).ceil();
    let h = (f64::from(tree.size().height()) * scale).ceil();
    if !w.is_finite() || !h.is_finite() || w < 1.0 || h < 1.0 {
        return Err(Error::Rendering(format!("computed raster size {w}x{h} is not usable")));
    }
    if w > f64::from(u32::MAX) || h > f64::from(u32::MAX) {
        return Err(Error::Rendering(format!("computed raster size {w}x{h} overflows u32")));
    }
    let (w, h) = (w as u32, h as u32);

    let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)
        .ok_or(Error::InvalidDimensions { width: w, height: h })?;

    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32),
        &mut pixmap.as_mut(),
    );

    let mut fb = Framebuffer::new(w, h)?;
    for y in 0..h {
        let row = fb.row_mut(y).expect("y is within the framebuffer we just allocated");
        for x in 0..w {
            // PNG wants straight alpha; the pixmap holds premultiplied colour.
            let px = pixmap.pixels()[(y as usize) * (w as usize) + (x as usize)].demultiply();
            let o = (x as usize) * 4;
            row[o] = px.red();
            row[o + 1] = px.green();
            row[o + 2] = px.blue();
            row[o + 3] = px.alpha();
        }
    }

    PngEncoder::to_bytes(&fb)
}

/// Walk `svg` as XML (via `roxmltree`, independently of `usvg`'s own parse) and refuse if any
/// descendant element is in [`REFUSED`].
///
/// An element counts only when its namespace is `None` or the SVG namespace — matching exactly
/// what `usvg` itself accepts as an SVG element, so a document with no `xmlns` cannot slip past
/// a check that only recognised the namespaced form.
fn prescan_refuse(svg: &str) -> Result<()> {
    let doc = resvg::usvg::roxmltree::Document::parse(svg)
        .map_err(|e| Error::Rendering(format!("SVG is not well-formed XML: {e}")))?;

    let mut first: Option<String> = None;
    let mut count = 0usize;
    for node in doc.descendants() {
        if !node.is_element() {
            continue;
        }
        let name = node.tag_name();
        let is_svg_ns = matches!(name.namespace(), None | Some(SVG_NS));
        if !is_svg_ns {
            continue;
        }
        if REFUSED.contains(&name.name()) {
            count += 1;
            if first.is_none() {
                first = Some(name.name().to_string());
            }
        }
    }

    if let Some(element) = first {
        return Err(Error::SvgElementRefused { element, count });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::render_unscanned;

    /// Property observed: this crate's `resvg` has no font database compiled in (its
    /// `text`/`system-fonts` features are off), so `usvg` silently *drops* a `<text>` element
    /// rather than rendering it — the document renders identically with or without the text.
    /// No test through `svg_to_png` can observe this, because its pre-scan refuses any
    /// `<text>`-bearing document before `usvg` ever builds a tree, so this test calls the
    /// private `render_unscanned` directly instead. Mutation M2 (turning resvg's default
    /// features back on and loading system fonts inside the renderer) turns this test red,
    /// because a loaded system font would paint glyph pixels and the two byte strings would
    /// diverge.
    #[test]
    fn a_text_element_paints_nothing_because_no_font_database_is_compiled_in() {
        let with_text = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">
            <text x="2" y="14" font-size="12">Hi</text>
        </svg>"#;
        let blank = r#"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">
        </svg>"#;

        let with_text_bytes =
            render_unscanned(with_text, 1.0).expect("rendering a bare <text> element must succeed");
        let blank_bytes = render_unscanned(blank, 1.0)
            .expect("rendering the same document without <text> must succeed");

        assert_eq!(
            with_text_bytes, blank_bytes,
            "a <text> element must contribute no pixels when no font database is compiled in"
        );
    }
}
