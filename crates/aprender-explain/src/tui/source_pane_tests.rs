//! `render_source_pane` (#5601): title, line numbers, scroll offset, truncation, borders.

use super::*;
use crate::analyzer::AnalysisReport;

const W: u16 = 30;

fn app_with(src: &str, scroll: u16) -> TuiApp {
    let report = AnalysisReport {
        name: "k".to_string(),
        ..Default::default()
    };
    let mut app = TuiApp::new(src.to_string(), report);
    app.source_scroll = scroll;
    app
}

/// Render the pane at (0,0) with the given height into a W x 12 buffer.
fn rendered(app: &TuiApp, width: u16, height: u16) -> CellBuffer {
    let mut buf = CellBuffer::new(W, 12);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buf);
        render_source_pane(&mut canvas, app, 0.0, 0.0, width, height);
    }
    buf
}

fn cell(buf: &CellBuffer, x: u16, y: u16) -> String {
    buf.get(x, y).expect("cell").symbol.to_string()
}

fn row(buf: &CellBuffer, y: u16) -> String {
    (0..W)
        .map(|x| buf.get(x, y).expect("cell").symbol.to_string())
        .collect::<String>()
        .trim_end()
        .to_string()
}

#[test]
fn title_carries_the_report_name_and_corners() {
    let buf = rendered(&app_with("ret;", 0), 20, 5);
    let top = row(&buf, 0);
    assert!(top.starts_with("┌ PTX: k "), "{top}");
    assert!(top.ends_with('┐'), "{top}");
    assert_eq!(top.chars().count(), 20, "{top}");
    let bottom = row(&buf, 4);
    assert_eq!(bottom, format!("└{}┘", "─".repeat(18)));
}

#[test]
fn rows_show_one_based_line_numbers_and_text_between_borders() {
    let buf = rendered(&app_with("a\nb\nc", 0), 20, 5);
    let r1 = row(&buf, 1);
    assert!(r1.starts_with("│   1 a"), "{r1}");
    assert_eq!(cell(&buf, 19, 1), "│", "right border");
    assert!(row(&buf, 2).starts_with("│   2 b"), "{}", row(&buf, 2));
    assert!(row(&buf, 3).starts_with("│   3 c"), "{}", row(&buf, 3));
}

#[test]
fn scroll_offsets_the_first_visible_line() {
    let src = "l1\nl2\nl3\nl4\nl5";
    let buf = rendered(&app_with(src, 2), 20, 5);
    assert!(row(&buf, 1).starts_with("│   3 l3"), "{}", row(&buf, 1));
    assert!(row(&buf, 2).starts_with("│   4 l4"), "{}", row(&buf, 2));
    assert!(row(&buf, 3).starts_with("│   5 l5"), "{}", row(&buf, 3));
}

#[test]
fn rows_past_the_end_keep_borders_but_have_no_line_number() {
    let buf = rendered(&app_with("only", 0), 20, 5);
    let r = row(&buf, 2);
    assert!(r.starts_with('│'), "{r}");
    assert!(!r.chars().any(|c| c.is_ascii_digit()), "{r}");
    assert_eq!(cell(&buf, 19, 2), "│", "right border");
}

#[test]
fn long_lines_are_truncated_to_the_inner_width() {
    // width 20 -> inner 18 -> text budget 18 - 6 = 12 chars
    let buf = rendered(&app_with("ABCDEFGHIJKLMNOPQRST", 0), 20, 4);
    let r = row(&buf, 1);
    // text occupies cols 6..=17; col 18 is the scrollbar, col 19 the border
    assert_eq!(cell(&buf, 6, 1), "A");
    assert_eq!(cell(&buf, 17, 1), "L");
    assert_eq!(cell(&buf, 19, 1), "│");
    assert!((20..W).all(|x| cell(&buf, x, 1) == " "), "{r}");
}

#[test]
fn offset_origin_moves_the_whole_pane() {
    let app = app_with("a", 0);
    let mut buf = CellBuffer::new(W, 12);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buf);
        render_source_pane(&mut canvas, &app, 2.0, 3.0, 20, 4);
    }
    assert_eq!(cell(&buf, 2, 3), "┌");
    assert_eq!(cell(&buf, 2, 6), "└");
    assert_eq!(cell(&buf, 0, 3), " ");
    // content row 4: left border at x, number at x+1, text at x+6, right border at x+width-1
    assert_eq!(cell(&buf, 2, 4), "│");
    assert_eq!(cell(&buf, 6, 4), "1");
    assert_eq!(cell(&buf, 8, 4), "a");
    assert_eq!(cell(&buf, 21, 4), "│");
}
