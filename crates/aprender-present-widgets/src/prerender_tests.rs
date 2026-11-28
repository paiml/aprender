//! contracts/present-prerender-v1.yaml: what a widget tree prerenders to.

use crate::{
    button::Button, column::Column, container::Container, data_card::DataCard, grid::Grid,
    image::Image, modal::Modal, row::Row, stack::Stack, text::Text, text_input::TextInput,
    tooltip::Tooltip,
};
use presentar_core::{Brick, Widget};

const ATTACK: &str = r#"<script>alert("x&y")</script>'"#;

fn assert_inert(html: &str) {
    assert!(!html.contains("<script"), "raw script tag in {html}");
    assert!(
        html.contains("&lt;script&gt;"),
        "escaped text missing in {html}"
    );
}

fn both(html: &str) {
    let first = html
        .find(">first<")
        .unwrap_or_else(|| panic!("first child missing in {html}"));
    let second = html
        .find(">second<")
        .unwrap_or_else(|| panic!("second child missing in {html}"));
    assert!(first < second, "children out of order in {html}");
}

fn text(s: &str) -> Text {
    Text::new(s)
}

#[test]
fn containers_emit_their_children_in_order() {
    both(
        &Column::new()
            .child(text("first"))
            .child(text("second"))
            .to_html(),
    );
    both(
        &Row::new()
            .child(text("first"))
            .child(text("second"))
            .to_html(),
    );
    both(
        &Stack::new()
            .child(text("first"))
            .child(text("second"))
            .to_html(),
    );
    both(
        &Grid::new(2)
            .child(text("first"))
            .child(text("second"))
            .to_html(),
    );
    both(
        &Modal::new()
            .open(true)
            .content(text("first"))
            .footer(text("second"))
            .to_html(),
    );
}

#[test]
fn nesting_keeps_every_level() {
    let page = Column::new().child(Container::new().child(Row::new().child(text("deep"))));
    assert!(
        page.to_html().contains(">deep</span></div></div></div>"),
        "{}",
        page.to_html()
    );
}

#[test]
fn empty_container_is_a_closed_div() {
    assert_eq!(
        Column::new().to_html(),
        r#"<div class="brick-column" data-testid="column"></div>"#
    );
}

#[test]
fn closed_modal_is_hidden_and_titles_are_escaped() {
    let closed = Modal::new().title(ATTACK).content(text("body")).to_html();
    assert!(
        closed.starts_with(r#"<div class="brick-modal" hidden>"#),
        "{closed}"
    );
    assert_inert(&closed);
    assert!(!Modal::new().open(true).to_html().contains("hidden"));
}

#[test]
fn text_content_is_escaped() {
    assert_inert(&text(ATTACK).to_html());
    assert_inert(&Button::new(ATTACK).to_html());
    assert_inert(&Tooltip::new(ATTACK).to_html());
    assert_inert(&DataCard::new(ATTACK).to_html());
}

#[test]
fn attribute_values_cannot_break_out() {
    let quote = r#"" onload="x"#;
    for html in [
        Image::new(quote).alt(quote).to_html(),
        TextInput::new().value(quote).placeholder(quote).to_html(),
        Column::new().with_test_id(quote).to_html(),
        Text::new("t").with_test_id(quote).to_html(),
    ] {
        assert!(
            !html.contains(r#"" onload=""#),
            "attribute breakout in {html}"
        );
        assert!(
            html.contains("&quot; onload=&quot;x"),
            "escaped value missing in {html}"
        );
    }
}

#[test]
fn a_trait_object_child_is_emitted() {
    let child: Box<dyn Widget> = Box::new(text("boxed"));
    assert!(presentar_core::html::children_html(&[child]).contains(">boxed<"));
}

#[test]
fn escapers_cover_every_metacharacter() {
    use presentar_core::html::{escape_attr, escape_text};
    assert_eq!(escape_text("a<b>&\"'"), "a&lt;b&gt;&amp;\"'");
    assert_eq!(escape_attr("a<b>&\"'"), "a&lt;b&gt;&amp;&quot;&#39;");
}

#[test]
fn a_password_value_is_never_written_into_markup() {
    let html = TextInput::new().obscure(true).value("hunter2").to_html();
    assert!(html.contains(r#"type="password""#), "{html}");
    assert!(!html.contains("hunter2"), "password leaked into {html}");
}
