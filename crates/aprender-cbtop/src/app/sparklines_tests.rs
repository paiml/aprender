//! Behavioural tests for `CbtopApp::render_sparklines` (ticket #5595).

use super::CbtopApp;
use presentar_core::TextStyle;
use presentar_terminal::direct::{CellBuffer, DirectTerminalCanvas};
use presentar_terminal::Theme;

const W: u16 = 80;
const H: u16 = 30;

/// Run `render_sparklines` on a fresh buffer; return the text of every row and the final y.
fn run(start_y: f32, height: u16, cpu: &[f64], bricks: &[f64]) -> (Vec<String>, f32) {
    let mut buffer = CellBuffer::new(W, H);
    let mut y = start_y;
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        let style = TextStyle::default();
        CbtopApp::render_sparklines(
            &mut canvas,
            &mut y,
            height,
            cpu,
            bricks,
            40,
            43,
            20,
            &Theme::tokyo_night(),
            &style,
            &style,
        );
    }
    let rows = (0..H)
        .map(|r| {
            (0..W)
                .filter_map(|x| buffer.get(x, r).map(|c| c.symbol.to_string()))
                .collect::<String>()
        })
        .collect();
    (rows, y)
}

fn has(rows: &[String], needle: &str) -> bool {
    rows.iter().any(|r| r.contains(needle))
}

#[test]
fn both_boxes_drawn_and_y_advances_to_end_of_second_box() {
    let (rows, y) = run(2.0, H, &[10.0, 50.0, 90.0], &[1.0, 2.0, 3.0]);
    assert!(rows[2].contains("┌─ CPU History (braille) "), "{}", rows[2]);
    assert!(rows[3].starts_with(" │ "), "{}", rows[3]);
    assert!(
        rows[4].contains('└') && rows[4].contains('┘'),
        "{}",
        rows[4]
    );
    // cpu block advances 1 + 1 + 2 = 4 rows, so the bricks box starts at row 6
    assert!(rows[6].contains("┌─ Bricks/sec (braille) "), "{}", rows[6]);
    assert!(rows[8].contains('└'), "{}", rows[8]);
    assert_eq!(y, 8.0, "second box advances 1 + 1 only");
}

#[test]
fn empty_cpu_data_skips_cpu_box_and_leaves_y() {
    let (rows, y) = run(2.0, H, &[], &[1.0, 2.0]);
    assert!(!has(&rows, "CPU History"));
    assert!(rows[2].contains("┌─ Bricks/sec (braille) "), "{}", rows[2]);
    assert_eq!(y, 4.0);
}

#[test]
fn empty_bricks_data_skips_bricks_box() {
    let (rows, y) = run(2.0, H, &[5.0], &[]);
    assert!(has(&rows, "CPU History"));
    assert!(!has(&rows, "Bricks/sec"));
    assert_eq!(y, 6.0);
}

#[test]
fn both_empty_draws_nothing() {
    let (rows, y) = run(2.0, H, &[], &[]);
    assert!(rows.iter().all(|r| r.trim().is_empty()));
    assert_eq!(y, 2.0);
}

#[test]
fn cpu_box_needs_eight_rows_of_headroom() {
    // y < height - 8 is strict: 22 < 22 is false, 21 < 22 is true
    let (rows, y) = run(22.0, H, &[5.0], &[]);
    assert!(!has(&rows, "CPU History"));
    assert_eq!(y, 22.0);
    let (rows, _) = run(21.0, H, &[5.0], &[]);
    assert!(has(&rows, "CPU History"));
}

#[test]
fn bricks_box_needs_five_rows_of_headroom() {
    // after the cpu box y = start + 4; bricks need y < height - 5 = 25
    let (rows, y) = run(20.0, H, &[5.0], &[1.0]);
    assert!(has(&rows, "Bricks/sec"), "start 20 -> y 24 < 25");
    assert_eq!(y, 26.0);
    let (rows, y) = run(21.0, H, &[5.0], &[1.0]);
    assert!(!has(&rows, "Bricks/sec"), "start 21 -> y 25 is not < 25");
    assert_eq!(y, 25.0);
}
