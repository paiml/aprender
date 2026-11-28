//! Helpers every `Brick::to_html` shares (contracts/present-prerender-v1.yaml).

use crate::widget::Widget;

/// Escape a string for HTML element content.
#[must_use]
pub fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape a string for a quoted HTML attribute value.
#[must_use]
pub fn escape_attr(value: &str) -> String {
    escape_text(value)
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// The HTML of `children`, in order.
#[must_use]
pub fn children_html(children: &[Box<dyn Widget>]) -> String {
    children.iter().map(|child| child.to_html()).collect()
}
