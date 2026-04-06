//! Type-Safe HTML Generation (Zero-JavaScript Policy)
//!
//! Generates valid HTML programmatically with accessibility attributes.

use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};

/// Generated HTML output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedHtml {
    /// Document title
    pub title: String,
    /// Body content (inner HTML)
    pub body_content: String,
    /// Full HTML document
    pub content: String,
    /// Elements in the document
    pub elements: Vec<Element>,
}

/// HTML element types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Element {
    /// Canvas element for WASM rendering
    Canvas {
        /// Element ID
        id: String,
        /// Width in pixels
        width: u32,
        /// Height in pixels
        height: u32,
        /// ARIA role
        role: String,
        /// ARIA label
        aria_label: String,
    },
    /// Div container
    Div {
        /// Element ID
        id: String,
        /// CSS classes
        classes: Vec<String>,
        /// Inner content
        content: String,
    },
    /// Button element
    Button {
        /// Element ID
        id: String,
        /// Button text
        text: String,
        /// ARIA label
        aria_label: String,
    },
    /// Input element
    Input {
        /// Element ID
        id: String,
        /// Input type
        input_type: String,
        /// Placeholder text
        placeholder: String,
        /// ARIA label
        aria_label: String,
    },
}

impl Element {
    /// Render element to HTML string
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Element::Canvas {
                id,
                width,
                height,
                role,
                aria_label,
            } => {
                format!(
                    r#"<canvas id="{id}" width="{width}" height="{height}" role="{role}" aria-label="{aria_label}" tabindex="0"></canvas>"#
                )
            }
            Element::Div {
                id,
                classes,
                content,
            } => {
                let class_attr = if classes.is_empty() {
                    String::new()
                } else {
                    format!(r#" class="{}""#, classes.join(" "))
                };
                format!(r#"<div id="{id}"{class_attr}>{content}</div>"#)
            }
            Element::Button {
                id,
                text,
                aria_label,
            } => {
                format!(r#"<button id="{id}" aria-label="{aria_label}">{text}</button>"#)
            }
            Element::Input {
                id,
                input_type,
                placeholder,
                aria_label,
            } => {
                format!(
                    r#"<input id="{id}" type="{input_type}" placeholder="{placeholder}" aria-label="{aria_label}">"#
                )
            }
        }
    }
}

/// Internal HTML document structure
#[derive(Debug, Clone, Default)]
pub struct HtmlDocument {
    /// Document title
    pub title: String,
    /// Document language
    pub lang: String,
    /// Character encoding
    pub charset: String,
    /// Viewport configuration
    pub viewport: String,
    /// Body elements
    pub elements: Vec<Element>,
}

/// Type-safe HTML builder
#[derive(Debug, Clone, Default)]
pub struct HtmlBuilder {
    document: HtmlDocument,
}

impl HtmlBuilder {
    /// Create a new HTML builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            document: HtmlDocument {
                title: String::new(),
                lang: "en".to_string(),
                charset: "UTF-8".to_string(),
                viewport: "width=device-width, initial-scale=1.0".to_string(),
                elements: Vec::new(),
            },
        }
    }

    /// Set document title (required)
    #[must_use]
    pub fn title(mut self, title: &str) -> Self {
        self.document.title = title.to_string();
        self
    }

    /// Set document language
    #[must_use]
    pub fn lang(mut self, lang: &str) -> Self {
        self.document.lang = lang.to_string();
        self
    }

    /// Add a canvas element for WASM rendering
    #[must_use]
    pub fn canvas(mut self, id: &str, width: u32, height: u32) -> Self {
        self.document.elements.push(Element::Canvas {
            id: id.to_string(),
            width,
            height,
            role: "application".to_string(),
            aria_label: "Application canvas".to_string(),
        });
        self
    }

    /// Add a canvas element with custom accessibility attributes
    #[must_use]
    pub fn canvas_with_a11y(
        mut self,
        id: &str,
        width: u32,
        height: u32,
        role: &str,
        aria_label: &str,
    ) -> Self {
        self.document.elements.push(Element::Canvas {
            id: id.to_string(),
            width,
            height,
            role: role.to_string(),
            aria_label: aria_label.to_string(),
        });
        self
    }

    /// Add a div container
    #[must_use]
    pub fn div(mut self, id: &str, classes: &[&str], content: &str) -> Self {
        self.document.elements.push(Element::Div {
            id: id.to_string(),
            classes: classes.iter().map(|s| (*s).to_string()).collect(),
            content: content.to_string(),
        });
        self
    }

    /// Add a button element
    #[must_use]
    pub fn button(mut self, id: &str, text: &str, aria_label: &str) -> Self {
        self.document.elements.push(Element::Button {
            id: id.to_string(),
            text: text.to_string(),
            aria_label: aria_label.to_string(),
        });
        self
    }

    /// Add an input element
    #[must_use]
    pub fn input(
        mut self,
        id: &str,
        input_type: &str,
        placeholder: &str,
        aria_label: &str,
    ) -> Self {
        self.document.elements.push(Element::Input {
            id: id.to_string(),
            input_type: input_type.to_string(),
            placeholder: placeholder.to_string(),
            aria_label: aria_label.to_string(),
        });
        self
    }

    /// Add a raw element
    #[must_use]
    pub fn element(mut self, element: Element) -> Self {
        self.document.elements.push(element);
        self
    }

    /// Build and validate HTML document
    ///
    /// # Errors
    ///
    /// Returns error if title is empty
    pub fn build(self) -> ProbarResult<GeneratedHtml> {
        // Validation: title is required
        if self.document.title.is_empty() {
            return Err(ProbarError::HtmlGeneration(
                "Document title is required".to_string(),
            ));
        }

        // Generate body content
        let body_content = self
            .document
            .elements
            .iter()
            .map(Element::render)
            .collect::<Vec<_>>()
            .join("\n    ");

        // Generate full HTML
        let content = format!(
            r#"<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="{charset}">
    <meta name="viewport" content="{viewport}">
    <title>{title}</title>
</head>
<body>
    {body}
</body>
</html>"#,
            lang = self.document.lang,
            charset = self.document.charset,
            viewport = self.document.viewport,
            title = self.document.title,
            body = body_content,
        );

        Ok(GeneratedHtml {
            title: self.document.title,
            body_content,
            content,
            elements: self.document.elements,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-HTML-01: HtmlBuilder creation and defaults
    // =========================================================================

    #[test]
    fn h0_html_01_builder_new() {
        let builder = HtmlBuilder::new();
        assert_eq!(builder.document.lang, "en");
        assert_eq!(builder.document.charset, "UTF-8");
    }

    #[test]
    fn h0_html_02_builder_title() {
        let builder = HtmlBuilder::new().title("My App");
        assert_eq!(builder.document.title, "My App");
    }

    #[test]
    fn h0_html_03_builder_lang() {
        let builder = HtmlBuilder::new().lang("es");
        assert_eq!(builder.document.lang, "es");
    }

    // =========================================================================
    // H₀-HTML-04: Canvas element generation
    // =========================================================================

    #[test]
    fn h0_html_04_canvas_element() {
        let html = HtmlBuilder::new()
            .title("Test")
            .canvas("game", 800, 600)
            .build()
            .unwrap();

        assert!(html.content.contains(r#"id="game""#));
        assert!(html.content.contains(r#"width="800""#));
        assert!(html.content.contains(r#"height="600""#));
        assert!(html.content.contains(r#"role="application""#));
        assert!(html.content.contains(r#"aria-label="Application canvas""#));
        assert!(html.content.contains(r#"tabindex="0""#));
    }

    #[test]
    fn h0_html_05_canvas_custom_a11y() {
        let html = HtmlBuilder::new()
            .title("Test")
            .canvas_with_a11y("calc", 400, 300, "img", "Calculator display")
            .build()
            .unwrap();

        assert!(html.content.contains(r#"role="img""#));
        assert!(html.content.contains(r#"aria-label="Calculator display""#));
    }

    // =========================================================================
    // H₀-HTML-06: Other element types
    // =========================================================================

    #[test]
    fn h0_html_06_div_element() {
        let html = HtmlBuilder::new()
            .title("Test")
            .div("container", &["main", "flex"], "Hello")
            .build()
            .unwrap();

        assert!(html
            .content
            .contains(r#"<div id="container" class="main flex">Hello</div>"#));
    }

    #[test]
    fn h0_html_07_button_element() {
        let html = HtmlBuilder::new()
            .title("Test")
            .button("submit", "Submit", "Submit form")
            .build()
            .unwrap();

        assert!(html
            .content
            .contains(r#"<button id="submit" aria-label="Submit form">Submit</button>"#));
    }

    #[test]
    fn h0_html_08_input_element() {
        let html = HtmlBuilder::new()
            .title("Test")
            .input("email", "email", "Enter email", "Email address")
            .build()
            .unwrap();

        assert!(html.content.contains(r#"<input id="email""#));
        assert!(html.content.contains(r#"type="email""#));
        assert!(html.content.contains(r#"placeholder="Enter email""#));
    }

    // =========================================================================
    // H₀-HTML-09: Validation
    // =========================================================================

    #[test]
    fn h0_html_09_empty_title_fails() {
        let result = HtmlBuilder::new().build();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("title is required"));
    }

    #[test]
    fn h0_html_10_valid_html_structure() {
        let html = HtmlBuilder::new()
            .title("Test App")
            .canvas("app", 100, 100)
            .build()
            .unwrap();

        assert!(html.content.starts_with("<!DOCTYPE html>"));
        assert!(html.content.contains("<html lang=\"en\">"));
        assert!(html.content.contains("<head>"));
        assert!(html.content.contains("</head>"));
        assert!(html.content.contains("<body>"));
        assert!(html.content.contains("</body>"));
        assert!(html.content.contains("</html>"));
    }

    // =========================================================================
    // H₀-HTML-11: Element rendering
    // =========================================================================

    #[test]
    fn h0_html_11_element_render_canvas() {
        let elem = Element::Canvas {
            id: "c".to_string(),
            width: 100,
            height: 100,
            role: "img".to_string(),
            aria_label: "Test".to_string(),
        };

        let rendered = elem.render();
        assert!(rendered.contains("<canvas"));
        assert!(rendered.contains("</canvas>"));
    }

    #[test]
    fn h0_html_12_element_render_div_no_classes() {
        let elem = Element::Div {
            id: "d".to_string(),
            classes: vec![],
            content: "Test".to_string(),
        };

        let rendered = elem.render();
        assert_eq!(rendered, r#"<div id="d">Test</div>"#);
    }

    #[test]
    fn h0_html_13_generated_html_fields() {
        let html = HtmlBuilder::new()
            .title("My Title")
            .canvas("c", 10, 10)
            .build()
            .unwrap();

        assert_eq!(html.title, "My Title");
        assert!(!html.body_content.is_empty());
        assert!(!html.elements.is_empty());
    }
}
