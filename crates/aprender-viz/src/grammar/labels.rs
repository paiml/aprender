//! The title and axis labels of a [`super::GGPlot`] (#3590): drawn, or refused — never dropped.
//!
//! Labels are glyph outlines from font bytes the caller pins ([`crate::text`] explains why there
//! is no default font). So a label is resolved at `build()`:
//!
//! - without the `text-path` feature there is no way to draw text, and a set label is an error;
//! - with it, a set label needs `.font(bytes)`, and a font that draws no glyph for the label is
//!   an error too — an empty outline would be the same silent drop by another route.

#[cfg(not(feature = "text-path"))]
use crate::error::{Error, Result};

/// The three label strings as the builder received them.
#[derive(Debug, Clone, Default)]
pub(super) struct LabelText {
    pub title: Option<String>,
    pub xlab: Option<String>,
    pub ylab: Option<String>,
}

impl LabelText {
    /// `(builder method, text)` for every label that asks for something to be drawn.
    fn set(&self) -> impl Iterator<Item = (&'static str, &str)> {
        [("title", &self.title), ("xlab", &self.xlab), ("ylab", &self.ylab)]
            .into_iter()
            .filter_map(|(name, text)| Some((name, text.as_deref()?)))
            .filter(|(_, text)| !text.is_empty())
    }
}

/// Labels ready to draw. Without `text-path` the only buildable value is "no labels".
#[cfg(not(feature = "text-path"))]
#[derive(Debug, Default)]
pub(super) struct Labels;

#[cfg(not(feature = "text-path"))]
impl Labels {
    pub(super) fn resolve(text: &LabelText, _font: Option<&[u8]>, _margin: u32) -> Result<Self> {
        match text.set().next() {
            None => Ok(Self),
            Some((name, _)) => Err(Error::Rendering(format!(
                "GGPlot::{name} cannot be drawn: labels are glyph outlines from pinned font bytes; \
                 build aprender-viz with the `text-path` feature and call .font(bytes) (#3590)"
            ))),
        }
    }

    pub(super) fn draw(
        &self,
        _fb: &mut crate::framebuffer::Framebuffer,
        _margin: u32,
        _color: crate::color::Rgba,
    ) {
    }
}

#[cfg(feature = "text-path")]
pub(super) use drawn::Labels;

#[cfg(feature = "text-path")]
mod drawn {
    use kurbo::{Affine, BezPath, PathEl, Point, Shape, Vec2};

    use super::LabelText;
    use crate::color::Rgba;
    use crate::error::{Error, Result};
    use crate::framebuffer::Framebuffer;

    /// Label size as a fraction of the theme margin the label sits in.
    const SIZE_OF_MARGIN: f64 = 0.45;
    /// Smallest label size, so a zero margin still yields a legible label.
    const MIN_SIZE_PX: f64 = 8.0;

    /// Outlines at the text origin (baseline at y = 0, y down), placed at draw time.
    #[derive(Debug, Default)]
    pub(in crate::grammar) struct Labels {
        title: Option<BezPath>,
        xlab: Option<BezPath>,
        ylab: Option<BezPath>,
    }

    impl Labels {
        pub(in crate::grammar) fn resolve(
            text: &LabelText,
            font: Option<&[u8]>,
            margin: u32,
        ) -> Result<Self> {
            let size = (f64::from(margin) * SIZE_OF_MARGIN).max(MIN_SIZE_PX);
            let mut labels = Self::default();
            for (name, s) in text.set() {
                let Some(font) = font else {
                    return Err(Error::Rendering(format!(
                        "GGPlot::{name} needs a font: call .font(bytes) with the font to draw it \
                         in; there is no default font (#3590)"
                    )));
                };
                let path = crate::text::to_path(s, font, size);
                if path.is_empty() {
                    return Err(Error::Rendering(format!(
                        "GGPlot::{name}: the font draws no glyph for {s:?} (#3590)"
                    )));
                }
                let slot = match name {
                    "title" => &mut labels.title,
                    "xlab" => &mut labels.xlab,
                    _ => &mut labels.ylab,
                };
                *slot = Some(path);
            }
            Ok(labels)
        }

        /// Title centred in the top margin, x label in the bottom margin, y label rotated a
        /// quarter turn and centred in the left margin.
        pub(in crate::grammar) fn draw(&self, fb: &mut Framebuffer, margin: u32, color: Rgba) {
            let (w, h, m) = (f64::from(fb.width()), f64::from(fb.height()), f64::from(margin));
            if let Some(p) = &self.title {
                fill_path(fb, &centred(p.clone(), Point::new(w / 2.0, m / 2.0)), color);
            }
            if let Some(p) = &self.xlab {
                fill_path(fb, &centred(p.clone(), Point::new(w / 2.0, h - m / 2.0)), color);
            }
            if let Some(p) = &self.ylab {
                let turned = Affine::rotate(-std::f64::consts::FRAC_PI_2) * p.clone();
                fill_path(fb, &centred(turned, Point::new(m / 2.0, h / 2.0)), color);
            }
        }
    }

    /// `path` translated so its bounding box is centred on `at`.
    fn centred(path: BezPath, at: Point) -> BezPath {
        let c = path.bounding_box().center();
        Affine::translate(Vec2::new(at.x - c.x, at.y - c.y)) * path
    }

    /// Fill `path` with the nonzero winding rule, one sample per pixel centre (no antialiasing,
    /// so the result is exact and platform-independent).
    pub(in crate::grammar) fn fill_path(fb: &mut Framebuffer, path: &BezPath, color: Rgba) {
        let edges = edges(path);
        if edges.is_empty() {
            return;
        }
        let bb = path.bounding_box();
        let y0 = bb.y0.floor().max(0.0) as u32;
        let y1 = bb.y1.ceil().min(f64::from(fb.height())) as u32;
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for py in y0..y1 {
            let sy = f64::from(py) + 0.5;
            xs.clear();
            for &(a, b) in &edges {
                if (a.y <= sy) != (b.y <= sy) {
                    let t = (sy - a.y) / (b.y - a.y);
                    xs.push((a.x + t * (b.x - a.x), if b.y > a.y { 1 } else { -1 }));
                }
            }
            xs.sort_by(|p, q| p.0.total_cmp(&q.0));
            let mut winding = 0;
            for pair in xs.windows(2) {
                winding += pair[0].1;
                if winding == 0 {
                    continue;
                }
                // Pixels whose centre lies in [pair[0].x, pair[1].x).
                let from = (pair[0].0 - 0.5).ceil().max(0.0);
                let to = (pair[1].0 - 0.5).ceil().min(f64::from(fb.width()));
                let mut px = from;
                while px < to {
                    fb.set_pixel(px as u32, py, color);
                    px += 1.0;
                }
            }
        }
    }

    /// The flattened path as closed line segments.
    fn edges(path: &BezPath) -> Vec<(Point, Point)> {
        let mut out = Vec::new();
        let (mut start, mut cur) = (Point::ZERO, Point::ZERO);
        kurbo::flatten(path.iter(), 0.1, |el| match el {
            PathEl::MoveTo(p) => {
                if cur != start {
                    out.push((cur, start));
                }
                start = p;
                cur = p;
            }
            PathEl::LineTo(p) => {
                out.push((cur, p));
                cur = p;
            }
            PathEl::ClosePath => {
                if cur != start {
                    out.push((cur, start));
                }
                cur = start;
            }
            // `flatten` emits only the three above.
            PathEl::QuadTo(..) | PathEl::CurveTo(..) => {}
        });
        if cur != start {
            out.push((cur, start));
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn count(fb: &Framebuffer, color: Rgba) -> usize {
            let mut n = 0;
            for y in 0..fb.height() {
                for x in 0..fb.width() {
                    n += usize::from(fb.get_pixel(x, y) == Some(color));
                }
            }
            n
        }

        #[test]
        fn fill_path_covers_exactly_the_pixel_centres_inside() {
            let mut fb = Framebuffer::new(20, 20).expect("fb");
            let ink = Rgba::rgb(1, 2, 3);
            let mut square = BezPath::new();
            square.move_to((2.0, 3.0));
            square.line_to((12.0, 3.0));
            square.line_to((12.0, 8.0));
            square.line_to((2.0, 8.0));
            square.close_path();
            fill_path(&mut fb, &square, ink);
            assert_eq!(count(&fb, ink), 10 * 5);
            assert_eq!(fb.get_pixel(2, 3), Some(ink));
            assert_eq!(fb.get_pixel(11, 7), Some(ink));
            assert_ne!(fb.get_pixel(12, 7), Some(ink));
            assert_ne!(fb.get_pixel(1, 3), Some(ink));
        }

        #[test]
        fn fill_path_nonzero_keeps_the_hole_of_a_reversed_inner_contour() {
            let mut fb = Framebuffer::new(20, 20).expect("fb");
            let ink = Rgba::rgb(9, 9, 9);
            let mut ring = BezPath::new();
            // Outer square clockwise, inner square counter-clockwise: the inner one cancels.
            for pts in [
                [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                [(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)],
            ] {
                ring.move_to(pts[0]);
                for p in &pts[1..] {
                    ring.line_to(*p);
                }
                ring.close_path();
            }
            fill_path(&mut fb, &ring, ink);
            assert_eq!(count(&fb, ink), 100 - 16);
            assert_ne!(fb.get_pixel(5, 5), Some(ink));
        }

        #[test]
        fn fill_path_clips_to_the_framebuffer() {
            let mut fb = Framebuffer::new(4, 4).expect("fb");
            let ink = Rgba::rgb(7, 7, 7);
            let mut big = BezPath::new();
            big.move_to((-10.0, -10.0));
            big.line_to((10.0, -10.0));
            big.line_to((10.0, 10.0));
            big.line_to((-10.0, 10.0));
            big.close_path();
            fill_path(&mut fb, &big, ink);
            assert_eq!(count(&fb, ink), 16);
        }
    }
}
