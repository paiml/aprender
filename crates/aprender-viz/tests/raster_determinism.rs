#![cfg(feature = "raster")]
//! APEX-001 EV-2d: `raster::svg_to_png` is deterministic and refuses fonts/external resources
//! (paiml/aprender#3666).
//!
//! Three claims are under test here, mirroring `render_determinism.rs`'s split:
//!
//! - **RED (a), same host twice** — rasterising the same SVG into two places gives identical PNG
//!   bytes. Cheap, runs everywhere, and does not by itself prove the image is anything but
//!   consistently wrong (hence the non-blank / multi-colour test below it).
//! - **RED (b), two architectures** — the same SVG rasterised on X64 and ARM64 gives identical
//!   PNG. No single host can see this (a host always agrees with itself); that half lives in CI
//!   (the `determinism` matrix and `scripts/ci/raster-compare.sh`, written outside this ticket).
//!   What is here is the per-host half plus the receipt the two-host job compares.
//! - **RED (c), refusal** — the pre-scan in `raster::svg_to_png` refuses text, font, and
//!   file-reading elements before `usvg` ever builds a tree, including the two ways a naive
//!   namespace check could be bypassed: no `xmlns` at all, and an element nested where it is
//!   never rendered.
//!
//! The fixture is built with this crate's own [`SvgEncoder`] (rect, circle, line, polyline,
//! polygon, path — every element that has anti-aliased or fractional-alpha edges) and then has a
//! hand-written gradient and cubic bezier spliced in before `</svg>`, so tiny-skia's
//! anti-aliasing and gradient pipelines both actually run. It deliberately never calls
//! `SvgEncoder::text`/`text_as_path` — there is no font in this crate to render with.

use std::fs;
use std::path::PathBuf;
use trueno_viz::color::Rgba;
use trueno_viz::error::Error;
use trueno_viz::manifest::digest_bytes;
use trueno_viz::output::SvgEncoder;
use trueno_viz::raster::svg_to_png;

/// The fixture's logical size, before `SCALE` is applied.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 200;
/// The scale `svg_to_png` is asked to rasterise at throughout this file.
const SCALE: f64 = 2.0;

/// The base fixture: several shapes with fractional coordinates, fractional stroke widths, and
/// alpha < 255, built entirely through this crate's own [`SvgEncoder`].
fn base_svg() -> String {
    SvgEncoder::new(WIDTH, HEIGHT)
        .background(Some(Rgba::WHITE))
        .rect(10.3, 15.7, 40.2, 25.9, Rgba::new(200, 30, 30, 180))
        .circle(120.25, 80.75, 22.125, Rgba::new(30, 120, 200, 220))
        .line(5.5, 5.5, 300.25, 190.75, Rgba::new(0, 0, 0, 255), 1.75)
        .polyline(
            &[(10.0, 10.0), (50.0, 90.0), (100.0, 20.0), (200.0, 150.0)],
            Rgba::new(10, 10, 10, 255),
            2.25,
        )
        .polygon(
            &[(200.0, 20.0), (260.0, 20.0), (260.0, 60.0), (200.0, 60.0)],
            Rgba::new(50, 200, 50, 150),
            None,
            1.0,
        )
        .path(
            "M 20 150 L 60 180 L 100 140 Z",
            Some(Rgba::new(255, 165, 0, 200)),
            Some(Rgba::BLACK),
            1.5,
        )
        .render()
}

/// `base_svg()` plus a hand-written gradient-filled rect and a fractional-stroke cubic bezier,
/// spliced in before `</svg>` — so gradients and curve anti-aliasing are exercised too.
fn fixture_svg() -> String {
    let svg = base_svg();
    let extra = r##"  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ff0000"/>
      <stop offset="100%" stop-color="#0000ff"/>
    </linearGradient>
  </defs>
  <rect x="150.3" y="30.7" width="80.4" height="50.9" fill="url(#g)"/>
  <path d="M 20.25 60.5 C 40.75 10.25 90.125 110.875 130.5 60.25" stroke="#222222" stroke-width="1.375" fill="none"/>
"##;
    assert_eq!(svg.matches("</svg>").count(), 1, "fixture SVG must close exactly once");
    svg.replacen("</svg>", &format!("{extra}</svg>"), 1)
}

fn out_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/aprender-viz");
    fs::create_dir_all(&d).expect("create target dir");
    d
}

/// Minimal PNG chunk: `(type, data)`, walked without pulling in a second PNG reader for this.
fn png_chunks(bytes: &[u8]) -> Vec<(&[u8], &[u8])> {
    const SIG_LEN: usize = 8;
    let mut chunks = Vec::new();
    let mut pos = SIG_LEN;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        let ty = &bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start + len;
        assert!(data_end + 4 <= bytes.len(), "truncated PNG chunk {ty:?}");
        chunks.push((ty, &bytes[data_start..data_end]));
        pos = data_end + 4; // skip CRC
    }
    chunks
}

/// Decode PNG bytes to `(width, height, color_type, bit_depth, straight-alpha RGBA pixels)`.
fn decode_png(bytes: &[u8]) -> (u32, u32, png::ColorType, png::BitDepth, Vec<u8>) {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder.read_info().expect("valid PNG header");
    let info = reader.info();
    let (w, h, ct, bd) = (info.width, info.height, info.color_type, info.bit_depth);
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let out = reader.next_frame(&mut buf).expect("decode frame");
    buf.truncate(out.buffer_size());
    (w, h, ct, bd, buf)
}

// --- RED (a): same host, twice --------------------------------------------------------------

/// Rasterising the same SVG into two places gives byte-identical PNGs.
#[test]
fn the_same_svg_rasterises_to_identical_png_bytes_twice() {
    let svg = fixture_svg();
    let a = svg_to_png(&svg, SCALE).expect("rasterise a");
    let b = svg_to_png(&svg, SCALE).expect("rasterise b");
    assert_eq!(digest_bytes(&a), digest_bytes(&b), "two rasters of one SVG differ");
    assert!(!a.is_empty(), "the raster is empty");
}

/// The raster is not blank: some pixel has alpha > 0, and there are at least two distinct
/// opaque colours — so the identical-bytes test above cannot pass vacuously on an empty image.
#[test]
fn the_raster_is_not_blank_and_has_multiple_colours() {
    let png = svg_to_png(&fixture_svg(), SCALE).expect("rasterise");
    let (_, _, ct, _, pixels) = decode_png(&png);
    assert_eq!(ct, png::ColorType::Rgba, "decoded as something other than RGBA");

    let mut any_alpha = false;
    let mut opaque_colours = std::collections::HashSet::new();
    for px in pixels.chunks_exact(4) {
        let (r, g, b, a) = (px[0], px[1], px[2], px[3]);
        if a > 0 {
            any_alpha = true;
        }
        if a == 255 {
            opaque_colours.insert((r, g, b));
        }
    }
    assert!(any_alpha, "every pixel is fully transparent");
    assert!(
        opaque_colours.len() >= 2,
        "only {} distinct opaque colour(s) — the fixture did not actually draw anything varied",
        opaque_colours.len()
    );
}

/// IHDR carries the scaled pixel size and RGBA-8 colour, straight from `svg_to_png`'s own math.
#[test]
fn ihdr_matches_the_scaled_dimensions_and_is_rgba8() {
    let png = svg_to_png(&fixture_svg(), SCALE).expect("rasterise");
    let (w, h, ct, bd, _) = decode_png(&png);
    assert_eq!(w, (f64::from(WIDTH) * SCALE).ceil() as u32);
    assert_eq!(h, (f64::from(HEIGHT) * SCALE).ceil() as u32);
    assert_eq!(w, 640);
    assert_eq!(h, 400);
    assert_eq!(ct, png::ColorType::Rgba);
    assert_eq!(bd, png::BitDepth::Eight);
}

/// The PNG carries no timestamp and no text chunk — same reasoning as
/// `render_determinism.rs::the_png_contains_no_timestamp_and_no_text_chunks`: neither is a
/// property of the image, and either would make two rasters of the same SVG differ for a reason
/// that is not the image.
#[test]
fn the_png_carries_no_time_or_text_chunk() {
    let png = svg_to_png(&fixture_svg(), SCALE).expect("rasterise");
    for (ty, _) in png_chunks(&png) {
        for banned in [b"tIME".as_slice(), b"tEXt", b"iTXt", b"zTXt"] {
            assert_ne!(ty, banned, "PNG carries a {} chunk", String::from_utf8_lossy(banned));
        }
    }
}

// --- RED (c): refusal -------------------------------------------------------------------------

/// A `<text>` element is refused, naming itself and its count.
#[test]
fn a_text_element_is_refused() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
        <text x="1" y="1">hi</text>
    </svg>"#;
    match svg_to_png(svg, SCALE) {
        Err(Error::SvgElementRefused { element, count }) => {
            assert_eq!(element, "text");
            assert_eq!(count, 1);
        }
        other => panic!("expected SvgElementRefused, got {other:?}"),
    }
}

/// A `<text>` nested inside `<defs>` — never directly rendered — is refused anyway. The pre-scan
/// walks every descendant, not just the ones `usvg` would draw.
#[test]
fn text_nested_inside_defs_is_refused() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
        <defs><text x="0" y="0">unreferenced</text></defs>
    </svg>"#;
    assert!(matches!(
        svg_to_png(svg, SCALE),
        Err(Error::SvgElementRefused { ref element, .. }) if element == "text"
    ));
}

/// A `<text>` in a document with NO `xmlns` at all is refused — the grill's bypass input. `usvg`
/// treats a namespace-less element as SVG too, so a pre-scan checking only the namespaced form
/// would miss this.
#[test]
fn text_without_an_xmlns_is_refused() {
    let svg = r#"<svg width="10" height="10"><text>hi</text></svg>"#;
    assert!(matches!(
        svg_to_png(svg, SCALE),
        Err(Error::SvgElementRefused { ref element, .. }) if element == "text"
    ));
}

/// An SVG whose only mention of "text" is inside a comment is NOT refused: comments and CDATA
/// are not elements.
#[test]
fn a_comment_mentioning_text_is_not_refused() {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
        <!-- <text>not an element</text> -->
        <rect x="1" y="1" width="5" height="5" fill="#ff0000"/>
    </svg>"##;
    svg_to_png(svg, SCALE).expect("a comment must not be treated as a refused element");
}

/// An `<image>` whose `href` is a `data:image/svg+xml;base64,...` URI carrying a nested SVG with
/// `<text>` is refused as `image` — by the pre-scan, before `usvg`'s href resolver (which this
/// module also disarms) ever gets a chance to load the nested document.
#[test]
fn image_with_nested_svg_text_is_refused_as_image() {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let nested =
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="5" height="5"><text>hi</text></svg>"#;
    let encoded = STANDARD.encode(nested);
    let svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
            <image x="0" y="0" width="5" height="5" href="data:image/svg+xml;base64,{encoded}"/>
        </svg>"#
    );
    match svg_to_png(&svg, SCALE) {
        Err(Error::SvgElementRefused { element, .. }) => assert_eq!(element, "image"),
        other => panic!("expected SvgElementRefused(image), got {other:?}"),
    }
}

/// `scale` of zero, negative, NaN, or infinite is refused before any parsing happens.
#[test]
fn non_finite_or_non_positive_scale_is_refused() {
    let svg = base_svg();
    for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
        assert!(svg_to_png(&svg, bad).is_err(), "scale {bad} should have been refused");
    }
}

// --- the receipt the two-host job reads ---------------------------------------------------------

/// Write this host's half of the raster determinism receipt.
///
/// Not an assertion — a measurement, mirroring
/// `render_determinism.rs::zz_write_this_hosts_determinism_receipt`. The `determinism` job
/// collects this file from an X64 and an ARM64 runner and compares `png_sha256` between them
/// via `scripts/ci/raster-compare.sh`.
#[test]
fn zz_write_this_hosts_raster_receipt() {
    let dir = out_dir();
    let svg = fixture_svg();
    let png = svg_to_png(&svg, SCALE).expect("rasterise");
    // Recorded from the PNG itself, not from the constants the fixture was built with: the
    // receipt is a measurement, and raster-compare.sh compares these across hosts.
    let (width, height, _, _, _) = decode_png(&png);

    let arch = std::env::consts::ARCH;
    let receipt = format!(
        concat!(
            "{{\n",
            "  \"row\": \"EV-2d\",\n",
            "  \"host_arch\": \"{arch}\",\n",
            "  \"os\": \"{os}\",\n",
            "  \"svg_sha256\": \"{svg_sha}\",\n",
            "  \"png_sha256\": \"{png_sha}\",\n",
            "  \"png_bytes\": {png_bytes},\n",
            "  \"width\": {width},\n",
            "  \"height\": {height},\n",
            "  \"scale\": {scale},\n",
            "  \"resvg\": \"0.45.1\"\n",
            "}}\n"
        ),
        arch = arch,
        os = std::env::consts::OS,
        svg_sha = digest_bytes(svg.as_bytes()),
        png_sha = digest_bytes(&png),
        png_bytes = png.len(),
        width = width,
        height = height,
        scale = SCALE,
    );
    let path = dir.join(format!("raster-{arch}.json"));
    fs::write(&path, receipt).expect("write receipt");
    assert!(path.exists(), "the receipt the two-host job reads was not written");
}
