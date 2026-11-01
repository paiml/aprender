#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! `Menu::paint` draw-command contract, observed through `RecordingCanvas` (#5555).
//!
//! Layout used throughout: the trigger occupies (10, 20, 100x32), so the panel
//! opens at y = 52 and the first item row starts at y = 60 (8px padding).

use presentar_core::draw::DrawCommand;
use presentar_core::{Color, Event, Point, RecordingCanvas, Rect, Widget};
use presentar_widgets::{Button, Menu, MenuItem};

fn painted(menu: &mut Menu) -> Vec<DrawCommand> {
    menu.layout(Rect::new(10.0, 20.0, 100.0, 32.0));
    let mut canvas = RecordingCanvas::new();
    menu.paint(&mut canvas);
    canvas.take_commands()
}

fn texts(cmds: &[DrawCommand]) -> Vec<(String, Point, f32, Color)> {
    cmds.iter()
        .filter_map(|c| match c {
            DrawCommand::Text {
                content,
                position,
                style,
            } => Some((content.clone(), *position, style.size, style.color)),
            _ => None,
        })
        .collect()
}

fn fills(cmds: &[DrawCommand]) -> Vec<(Rect, Color)> {
    cmds.iter()
        .filter_map(|c| match c {
            DrawCommand::Rect { bounds, style, .. } => style.fill.map(|f| (*bounds, f)),
            _ => None,
        })
        .collect()
}

fn open_menu(items: Vec<MenuItem>) -> Menu {
    let mut m = Menu::new().width(200.0).items(items);
    m.show();
    m
}

/// Highlight the row under `y` the way a pointer does (lays the menu out first).
fn hover(m: &mut Menu, y: f32) {
    m.layout(Rect::new(10.0, 20.0, 100.0, 32.0));
    m.event(&Event::MouseMove {
        position: Point::new(50.0, y),
    });
}

#[test]
fn closed_menu_draws_nothing() {
    let mut m = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
    assert!(painted(&mut m).is_empty());
}

#[test]
fn closed_menu_still_paints_its_trigger() {
    let mut m = Menu::new()
        .trigger_widget(Button::new("File"))
        .items(vec![MenuItem::action("Cut", "cut")]);
    let cmds = painted(&mut m);
    assert!(texts(&cmds).iter().any(|t| t.0 == "File"));
    assert!(!texts(&cmds).iter().any(|t| t.0 == "Cut"));
}

#[test]
fn open_menu_draws_shadow_then_panel() {
    let mut m = open_menu(vec![MenuItem::action("Cut", "cut")]);
    let f = fills(&painted(&mut m));
    assert_eq!(f[0].0, Rect::new(12.0, 54.0, 200.0, 48.0));
    assert_eq!(f[0].1, Color::rgba(0.0, 0.0, 0.0, 0.1));
    assert_eq!(f[1].0, Rect::new(10.0, 52.0, 200.0, 48.0));
    assert_eq!(f[1].1, m.background_color);
}

#[test]
fn action_label_and_shortcut_positions() {
    let mut m = open_menu(vec![MenuItem::action("Cut", "cut").shortcut("Ctrl+X")]);
    let t = texts(&painted(&mut m));
    assert_eq!(t[0].0, "Cut");
    assert_eq!(t[0].1, Point::new(22.0, 80.0));
    assert_eq!(t[0].2, 14.0);
    assert_eq!(t[0].3, m.text_color);
    assert_eq!(t[1].0, "Ctrl+X");
    assert_eq!(t[1].1, Point::new(150.0, 80.0));
    assert_eq!(t[1].2, 12.0);
    assert_eq!(t[1].3, m.disabled_color);
}

#[test]
fn highlighted_enabled_item_gets_a_hover_rect() {
    let mut m = open_menu(vec![MenuItem::action("Cut", "cut")]);
    hover(&mut m, 70.0);
    assert_eq!(m.highlighted_index(), Some(0));
    let f = fills(&painted(&mut m));
    assert_eq!(f.len(), 3);
    assert_eq!(f[2].0, Rect::new(10.0, 60.0, 200.0, 32.0));
    assert_eq!(f[2].1, m.hover_color);
}

#[test]
fn unhighlighted_item_gets_no_hover_rect() {
    let mut m = open_menu(vec![MenuItem::action("Cut", "cut")]);
    assert_eq!(fills(&painted(&mut m)).len(), 2);
}

#[test]
fn disabled_items_never_hover_and_use_the_disabled_color() {
    let mut m = open_menu(vec![
        MenuItem::action("Cut", "cut").disabled(true),
        MenuItem::checkbox("Grid", "grid", true).disabled(true),
        MenuItem::submenu("More", vec![]).disabled(true),
    ]);
    for y in [70.0, 102.0, 134.0] {
        hover(&mut m, y);
        assert!(m.highlighted_index().is_some(), "row at y={y} is hovered");
        let cmds = painted(&mut m);
        assert_eq!(fills(&cmds).len(), 2, "row at y={y} must not draw hover");
        assert!(texts(&cmds).iter().all(|t| t.3 == m.disabled_color));
    }
}

#[test]
fn checkbox_marks_checked_and_unchecked() {
    let mut m = open_menu(vec![
        MenuItem::checkbox("A", "a", true),
        MenuItem::checkbox("B", "b", false),
    ]);
    hover(&mut m, 100.0);
    let cmds = painted(&mut m);
    let t = texts(&cmds);
    assert_eq!(t[0].0, "✓");
    assert_eq!(t[0].1, Point::new(22.0, 80.0));
    assert_eq!(t[1].0, "A");
    assert_eq!(t[1].1, Point::new(42.0, 80.0));
    assert_eq!(t[2].0, " ");
    assert_eq!(t[3].0, "B");
    assert_eq!(fills(&cmds)[2].0, Rect::new(10.0, 92.0, 200.0, 32.0));
}

#[test]
fn separator_draws_an_inset_line_and_advances_by_its_height() {
    let mut m = open_menu(vec![
        MenuItem::action("A", "a"),
        MenuItem::Separator,
        MenuItem::action("B", "b"),
    ]);
    let cmds = painted(&mut m);
    let (pts, closed, width) = cmds
        .iter()
        .find_map(|c| match c {
            DrawCommand::Path {
                points,
                closed,
                style,
            } => Some((points.clone(), *closed, style.width)),
            _ => None,
        })
        .expect("separator line");
    assert_eq!(pts, vec![Point::new(18.0, 96.0), Point::new(202.0, 96.0)]);
    assert!(!closed);
    assert_eq!(width, 1.0);
    let t = texts(&cmds);
    assert_eq!(t[1].0, "B");
    assert_eq!(t[1].1.y, 60.0 + 32.0 + 9.0 + 20.0);
}

#[test]
fn submenu_draws_label_and_arrow() {
    let mut m = open_menu(vec![MenuItem::submenu(
        "More",
        vec![MenuItem::action("X", "x")],
    )]);
    hover(&mut m, 70.0);
    let cmds = painted(&mut m);
    let t = texts(&cmds);
    assert_eq!(t[0].0, "More");
    assert_eq!(t[0].1, Point::new(22.0, 80.0));
    assert_eq!(t[1].0, "›");
    assert_eq!(t[1].1, Point::new(190.0, 80.0));
    assert_eq!(fills(&cmds).len(), 3);
}
