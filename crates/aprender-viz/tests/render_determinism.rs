//! APEX-001 EV-2a: the render primitive is deterministic (paiml/aprender#3233).
//!
//! Two claims are under test, and they fail differently:
//!
//! - **Same host, twice** — rendering the same figure into two places gives identical bytes.
//!   This catches a wall clock in the file, a hash-ordered iteration, an uninitialised pad byte.
//!   It is cheap and it runs everywhere.
//! - **Two architectures** — the same figure rendered on X64 and on ARM64 gives identical SVG.
//!   This catches the platform libm, and no single-host test can see it, because a host always
//!   agrees with itself. That half lives in CI (the `determinism` matrix and `determinism
//!   (compare)` jobs of `.github/workflows/ci.yml`, deciding through
//!   `scripts/ci/determinism-compare.sh`); what is here is the per-host half plus the receipt
//!   the two-host job compares.
//!
//! The receipt carries the SVG and the PNG as **separate digests**, because the row asserts one
//! and only records the other. `svg_identical` is decided from `svg_sha256` alone: an earlier
//! revision decided it from a manifest root over `fixture.png` only, so the field named SVG and
//! measured PNG, and no SVG byte was ever compared.
//!
//! The fixture puts a **log scale** on the axis on purpose: `LogScale::scale` is the only place
//! in this crate where a transcendental reaches a rendered coordinate, so a figure without one
//! would pass on two architectures while proving nothing about rule 5.

use std::fs;
use std::path::PathBuf;
use trueno_viz::color::Rgba;
use trueno_viz::framebuffer::Framebuffer;
use trueno_viz::manifest::{digest_bytes, Manifest};
use trueno_viz::output::{PngEncoder, SvgEncoder};
use trueno_viz::scale::{LogScale, Scale};

/// The fixture: positions produced through a log scale, drawn into a framebuffer.
///
/// Deliberately not a one-pixel image — the point is to run the transcendental path many times
/// and let any divergence accumulate into the bytes.
fn render_fixture() -> Framebuffer {
    let mut fb = Framebuffer::new(320, 200).expect("framebuffer");
    fb.clear(Rgba::WHITE);

    // 200 marks, each positioned by two log-scale evaluations.
    for (px, py) in log_positions() {
        let (xi, yi) = (px as i32, py as i32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y) = (xi + dx, yi + dy);
                if x >= 0 && y >= 0 {
                    fb.set_pixel(x as u32, y as u32, Rgba::new(31, 119, 180, 255));
                }
            }
        }
    }
    fb
}

/// The x and y positions of the fixture's marks, each produced by a log-scale evaluation.
fn log_positions() -> Vec<(f32, f32)> {
    let sx = LogScale::new((1.0_f32, 10_000.0), (8.0, 312.0)).expect("x scale");
    let sy = LogScale::new((1.0_f32, 1_000.0), (192.0, 8.0)).expect("y scale");
    (0..200)
        .map(|i| {
            let t = f64::from(i) / 199.0;
            let xv = 1.0 + t * 9_999.0;
            let yv = 1.0 + (1.0 - t) * 999.0;
            (sx.scale(xv as f32), sy.scale(yv as f32))
        })
        .collect()
}

/// The same figure as SVG: the artefact EV-2a asserts is identical across architectures.
///
/// Every coordinate in it came out of a log scale, and the writer snaps every emitted number to
/// the coordinate grid, so this string is where rule 5 either holds or does not.
fn render_fixture_svg() -> String {
    let pts = log_positions();
    let mut svg = SvgEncoder::new(320, 200).background(Some(Rgba::WHITE)).polyline(
        &pts,
        Rgba::new(31, 119, 180, 255),
        1.0,
    );
    for &(x, y) in &pts {
        svg = svg.circle(x, y, 1.5, Rgba::new(31, 119, 180, 255));
    }
    svg.render()
}

fn out_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/aprender-viz");
    fs::create_dir_all(&d).expect("create target dir");
    d
}

// --- the row's RED (a): same host, twice ------------------------------------------------------

/// Rendering the same figure into two different directories gives byte-identical PNGs.
///
/// The two directories matter: a path leaking into the output — through a temp name, a recorded
/// filename, or a text chunk — would pass a same-path comparison and fail this one.
#[test]
fn the_same_figure_renders_to_identical_png_bytes_twice() {
    let dir = out_dir();
    let a = dir.join("run-a");
    let b = dir.join("run-b");
    fs::create_dir_all(&a).expect("run-a");
    fs::create_dir_all(&b).expect("run-b");

    let pa = a.join("fixture.png");
    let pb = b.join("fixture.png");
    PngEncoder::write_to_file(&render_fixture(), &pa).expect("write a");
    PngEncoder::write_to_file(&render_fixture(), &pb).expect("write b");

    let (ba, bb) = (fs::read(&pa).expect("read a"), fs::read(&pb).expect("read b"));
    assert_eq!(
        digest_bytes(&ba),
        digest_bytes(&bb),
        "two renders of one figure differ: {} vs {} bytes",
        ba.len(),
        bb.len()
    );
    assert!(!ba.is_empty(), "the fixture rendered to an empty file");
}

/// Rendering the same figure into two different directories gives byte-identical SVG.
///
/// RED (a) for the artefact the row asserts across architectures. Same reasoning as the PNG
/// test above: two directories, so a path or a temp name leaking into the file cannot hide.
#[test]
fn the_same_figure_renders_to_identical_svg_bytes_twice() {
    let dir = out_dir();
    let a = dir.join("run-a");
    let b = dir.join("run-b");
    fs::create_dir_all(&a).expect("run-a");
    fs::create_dir_all(&b).expect("run-b");

    let (pa, pb) = (a.join("fixture.svg"), b.join("fixture.svg"));
    fs::write(&pa, render_fixture_svg()).expect("write a");
    fs::write(&pb, render_fixture_svg()).expect("write b");

    let (ba, bb) = (fs::read(&pa).expect("read a"), fs::read(&pb).expect("read b"));
    assert_eq!(digest_bytes(&ba), digest_bytes(&bb), "two SVG renders of one figure differ");
    assert!(
        ba.len() > 1_000,
        "the SVG fixture is too small to exercise anything: {} bytes",
        ba.len()
    );
}

/// Every number in the SVG is on the coordinate grid, and there is something to check.
///
/// The fixture's coordinates come out of `LogScale` as arbitrary `f32`s, so without the writer's
/// quantisation most of them print with six or more decimals. The count is asserted too: a
/// check over zero numbers would pass on a writer that emits nothing.
#[test]
fn every_number_in_the_svg_is_on_the_grid() {
    let svg = render_fixture_svg();
    let mut checked = 0usize;
    for tok in svg.split(|ch: char| !(ch.is_ascii_digit() || ch == '.' || ch == '-')) {
        if tok.is_empty() || tok == "-" || tok == "." {
            continue;
        }
        let decimals = tok.split_once('.').map_or(0, |(_, frac)| frac.len());
        assert!(decimals <= 3, "`{tok}` is off the 1e-3 grid");
        checked += 1;
    }
    assert!(checked >= 800, "only {checked} numbers were checked — the fixture lost its marks");
}

/// The PNG carries no timestamp and no text chunk.
///
/// `tIME` puts the wall clock in the file and `tEXt`/`iTXt`/`zTXt` put the encoder's identity
/// there. Either makes two renders differ for reasons that are not the image, so the
/// same-host test above would already be red — but it would be red without saying why. This
/// names the cause.
#[test]
fn the_png_contains_no_timestamp_and_no_text_chunks() {
    let bytes = PngEncoder::to_bytes(&render_fixture()).expect("encode");
    for chunk in [b"tIME".as_slice(), b"tEXt", b"iTXt", b"zTXt"] {
        assert!(
            !bytes.windows(4).any(|w| w == chunk),
            "PNG carries a {} chunk, which is not a property of the image",
            String::from_utf8_lossy(chunk)
        );
    }
}

/// Encoding to memory and writing to a file produce the same bytes.
///
/// The two entry points set their options separately; if they drift, a figure hashed in memory
/// and the same figure on disk stop being the same file, and the manifest records a root nothing
/// else can reproduce.
#[test]
fn to_bytes_and_write_to_file_agree() {
    let dir = out_dir();
    let p = dir.join("agree.png");
    let fb = render_fixture();
    PngEncoder::write_to_file(&fb, &p).expect("write");
    let from_file = fs::read(&p).expect("read");
    let from_mem = PngEncoder::to_bytes(&fb).expect("encode");
    assert_eq!(digest_bytes(&from_file), digest_bytes(&from_mem));
}

// --- rule 5: the transcendental path ----------------------------------------------------------

/// The log scale agrees with `libm` exactly, not approximately.
///
/// This is the assertion the two-host job cannot make on its own: it compares two hosts to each
/// other, so it would stay green if BOTH drifted to the platform libm together. This pins the
/// value to the pure-Rust implementation on whichever host runs it.
#[test]
fn the_log_scale_goes_through_libm_exactly() {
    let s = LogScale::new((1.0_f32, 10_000.0), (0.0, 1.0)).expect("scale");
    for v in [1.0_f32, 2.5, 7.0, 100.0, 3_333.0, 9_999.0] {
        let want = {
            let lb = libm::logf(10.0);
            let lmin = libm::logf(1.0) / lb;
            let lmax = libm::logf(10_000.0) / lb;
            let lv = libm::logf(v) / lb;
            (lv - lmin) / (lmax - lmin)
        };
        assert_eq!(
            s.scale(v).to_bits(),
            want.to_bits(),
            "log scale at {v} is not bit-identical to the libm computation"
        );
    }
}

/// Coordinates are snapped to the grid, and the two zeros are one zero.
#[test]
fn coordinates_are_quantised_to_the_grid() {
    use trueno_viz::{quantise, quantise_path_data, COORD_GRID};
    assert_eq!(COORD_GRID, 1e-3);
    // A sub-grid difference — the size of an accumulated ulp — cannot survive.
    assert_eq!(quantise(1.000_000_4), quantise(1.000_000_1));
    // -0.0 and 0.0 must not print differently.
    assert_eq!(quantise(-0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(quantise_path_data("M -0.0 1.00000004 L 2.0005 3"), "M 0 1 L 2.001 3");
}

// --- the manifest, over the rendered tree -----------------------------------------------------

/// The manifest root is a function of content, not of insertion order or of the run.
#[test]
fn the_manifest_root_is_stable_across_runs() {
    let first = {
        let mut m = Manifest::new();
        m.insert("fixture.png", &PngEncoder::to_bytes(&render_fixture()).expect("encode"));
        m.insert("fixture.svg", render_fixture_svg().as_bytes());
        m.root()
    };
    let second = {
        let mut m = Manifest::new();
        m.insert("fixture.png", &PngEncoder::to_bytes(&render_fixture()).expect("encode"));
        m.insert("fixture.svg", render_fixture_svg().as_bytes());
        m.root()
    };
    assert_eq!(first, second);
}

// --- the receipt the two-host job reads -------------------------------------------------------

/// Write this host's half of the determinism receipt.
///
/// Not an assertion — a measurement. The `determinism` job in `.github/workflows/ci.yml` runs this test on an
/// X64 and an ARM64 clean-room runner, collects both files, and merges them into
/// `determinism-receipt.json` with `hosts` of length 2 and `svg_identical` decided by comparing
/// the two `svg_sha256` digests — never the manifest root, which also covers the PNG. A single host cannot set `svg_identical`, and this deliberately does not
/// pretend to: it records what it saw and lets the job that can see both decide.
#[test]
fn zz_write_this_hosts_determinism_receipt() {
    let dir = out_dir();
    let fb = render_fixture();
    let png = PngEncoder::to_bytes(&fb).expect("encode");
    let svg = render_fixture_svg();

    let mut m = Manifest::new();
    m.insert("fixture.png", &png);
    m.insert("fixture.svg", svg.as_bytes());

    let arch = std::env::consts::ARCH;
    let receipt = format!(
        concat!(
            "{{\n",
            "  \"host_arch\": \"{arch}\",\n",
            "  \"os\": \"{os}\",\n",
            "  \"svg_sha256\": \"{svg}\",\n",
            "  \"svg_bytes\": {svg_len},\n",
            "  \"png_sha256\": \"{png}\",\n",
            "  \"png_bytes\": {len},\n",
            "  \"manifest_root\": \"{root}\",\n",
            "  \"coord_grid\": {grid},\n",
            "  \"libm\": \"pure-rust\"\n",
            "}}\n"
        ),
        arch = arch,
        os = std::env::consts::OS,
        svg = digest_bytes(svg.as_bytes()),
        svg_len = svg.len(),
        png = digest_bytes(&png),
        root = m.root(),
        len = png.len(),
        grid = trueno_viz::COORD_GRID,
    );
    let path = dir.join(format!("determinism-{arch}.json"));
    fs::write(&path, receipt).expect("write receipt");
    assert!(path.exists(), "the receipt the two-host job reads was not written");
}
