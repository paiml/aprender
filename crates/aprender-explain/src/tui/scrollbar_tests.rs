//! `draw_scrollbar` (#5623): thumb row from position/total, track elsewhere, colours.
//!
//! Contract: `contracts/explain-tui-draw-scrollbar-v1.yaml`.

use super::*;

const THUMB: &str = "\u{2588}";
const TRACK: &str = "\u{2502}";

/// Draw into a 6-wide buffer at column 3 starting at row `top`, return the buffer.
fn drawn(top: u16, height: u16, position: usize, total: usize) -> CellBuffer {
    let mut buf = CellBuffer::new(6, 20);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buf);
        draw_scrollbar(&mut canvas, 3.0, f32::from(top), height, position, total);
    }
    buf
}

fn sym(buf: &CellBuffer, x: u16, y: u16) -> String {
    buf.get(x, y).expect("cell").symbol.to_string()
}

/// Rows (absolute) holding the thumb glyph in column 3.
fn thumb_rows(buf: &CellBuffer) -> Vec<u16> {
    (0..20).filter(|&y| sym(buf, 3, y) == THUMB).collect()
}

#[test]
fn thumb_row_is_the_rounded_position_ratio() {
    // (position, total, height) -> expected thumb row relative to top
    for (pos, total, height, want) in [
        (0, 10, 10, 0u16),
        (5, 10, 10, 5),  // 0.5 * 9 = 4.5 rounds away from zero
        (10, 10, 10, 9), // end of content: last row
        (2, 10, 11, 2),  // 0.2 * 10
        (0, 0, 8, 0),    // total 0 clamps to 1 and position 0
    ] {
        let buf = drawn(0, height, pos, total);
        assert_eq!(
            thumb_rows(&buf),
            vec![want],
            "pos={pos} total={total} h={height}"
        );
    }
}

#[test]
fn position_past_the_end_draws_no_thumb_only_track() {
    let buf = drawn(0, 8, 3, 1);
    assert!(thumb_rows(&buf).is_empty());
    assert_eq!(sym(&buf, 3, 7), TRACK);
}

#[test]
fn exactly_one_thumb_and_the_rest_is_track() {
    let buf = drawn(0, 8, 4, 8);
    let rows: Vec<String> = (0..8).map(|y| sym(&buf, 3, y)).collect();
    assert_eq!(rows.iter().filter(|s| *s == THUMB).count(), 1);
    assert_eq!(rows.iter().filter(|s| *s == TRACK).count(), 7);
}

#[test]
fn rows_are_offset_by_top_y_and_nothing_else_is_touched() {
    let buf = drawn(5, 4, 0, 4);
    assert_eq!(sym(&buf, 3, 5), THUMB);
    for y in [6u16, 7, 8] {
        assert_eq!(sym(&buf, 3, y), TRACK);
    }
    // above the bar, below the bar and the neighbouring columns stay blank
    for (x, y) in [(3u16, 4u16), (3, 9), (2, 5), (4, 6)] {
        assert_eq!(sym(&buf, x, y), " ", "({x},{y}) was painted");
    }
}

#[test]
fn thumb_and_track_use_their_own_colours() {
    let buf = drawn(0, 4, 0, 4);
    let thumb = buf.get(3, 0).expect("thumb");
    let track = buf.get(3, 1).expect("track");
    assert!(color_eq(thumb.fg, COLOR_SCROLL_THUMB));
    assert!(color_eq(track.fg, COLOR_SCROLL_TRACK));
    assert!(!color_eq(thumb.fg, track.fg));
}

fn color_eq(a: Color, b: Color) -> bool {
    (a.r - b.r).abs() < 0.01
        && (a.g - b.g).abs() < 0.01
        && (a.b - b.b).abs() < 0.01
        && (a.a - b.a).abs() < 0.01
}

#[test]
fn single_row_bar_is_just_the_thumb() {
    let buf = drawn(2, 1, 7, 9);
    assert_eq!(thumb_rows(&buf), vec![2]);
    assert_eq!(sym(&buf, 3, 3), " ");
}
