//! #3590: `GGPlot::title`/`xlab`/`ylab` are drawn, each in its own margin.
//!
//! Host-bound: labels need font bytes and the crate ships none, so this reads a system font
//! (`APR_VIZ_TEST_FONT`, default DejaVu Sans). A font that is not there is a FAIL, not a skip.
#![cfg(feature = "text-path")]

use trueno_viz::grammar::{GGPlot, Geom, Theme};

fn font() -> Vec<u8> {
    let path = std::env::var("APR_VIZ_TEST_FONT")
        .unwrap_or_else(|_| "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf".into());
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path} NOT HELD: {e}"))
}

/// Pixels of the theme's text colour inside the rectangle `x0..x1 × y0..y1`.
fn ink(plot: GGPlot, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> usize {
    let text = Theme::grey().text_color;
    let fb = plot.build().expect("build").to_framebuffer().expect("render");
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            n += usize::from(fb.get_pixel(x, y) == Some(text));
        }
    }
    n
}

fn base() -> GGPlot {
    GGPlot::new()
        .data_xy(&[1.0, 2.0, 3.0], &[3.0, 1.0, 2.0])
        .geom(Geom::point())
        .dimensions(400, 300)
        .font(font())
}

#[test]
#[ignore = "host-bound: needs a font file (APR_VIZ_TEST_FONT, default DejaVu Sans)"]
fn each_label_is_drawn_in_its_own_margin() {
    // Default theme margin is 40 px.
    let top = (40, 0, 360, 40);
    let bottom = (40, 260, 360, 300);
    let left = (0, 40, 40, 260);
    for (name, plot, rect) in [
        ("title", base().title("Title"), top),
        ("xlab", base().xlab("X label"), bottom),
        ("ylab", base().ylab("Y label"), left),
    ] {
        let unlabelled = ink(base(), rect);
        let labelled = ink(plot, rect);
        assert!(
            labelled > unlabelled + 20,
            "{name}: {labelled} text pixels vs {unlabelled} unlabelled"
        );
    }
}
