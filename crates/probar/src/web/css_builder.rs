//! Type-Safe CSS Generation (Zero-JavaScript Policy)
//!
//! Generates valid CSS programmatically with responsive design support.

use crate::result::ProbarResult;
use serde::{Deserialize, Serialize};

/// Generated CSS output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCss {
    /// CSS content
    pub content: String,
    /// CSS rules in the stylesheet
    pub rules: Vec<CssRule>,
    /// CSS variables defined
    pub variables: Vec<(String, String)>,
}

/// A CSS rule with selector and declarations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CssRule {
    /// CSS selector
    pub selector: String,
    /// Property-value pairs
    pub declarations: Vec<(String, String)>,
}

impl CssRule {
    /// Create a new CSS rule
    #[must_use]
    pub fn new(selector: &str) -> Self {
        Self {
            selector: selector.to_string(),
            declarations: Vec::new(),
        }
    }

    /// Add a declaration
    #[must_use]
    pub fn declaration(mut self, property: &str, value: &str) -> Self {
        self.declarations
            .push((property.to_string(), value.to_string()));
        self
    }

    /// Render rule to CSS string
    #[must_use]
    pub fn render(&self) -> String {
        if self.declarations.is_empty() {
            return String::new();
        }

        let decls = self
            .declarations
            .iter()
            .map(|(prop, val)| format!("    {prop}: {val};"))
            .collect::<Vec<_>>()
            .join("\n");

        format!("{} {{\n{}\n}}", self.selector, decls)
    }
}

/// Type-safe CSS builder
#[derive(Debug, Clone, Default)]
pub struct CssBuilder {
    variables: Vec<(String, String)>,
    rules: Vec<CssRule>,
}

impl CssBuilder {
    /// Create a new CSS builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a CSS variable
    #[must_use]
    pub fn variable(mut self, name: &str, value: &str) -> Self {
        self.variables.push((name.to_string(), value.to_string()));
        self
    }

    /// Add a CSS rule
    #[must_use]
    pub fn rule(mut self, rule: CssRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add reset styles (minimal normalize)
    #[must_use]
    pub fn reset(mut self) -> Self {
        self.rules.push(
            CssRule::new("*, *::before, *::after")
                .declaration("box-sizing", "border-box")
                .declaration("margin", "0")
                .declaration("padding", "0"),
        );
        self
    }

    /// Add responsive canvas styling
    #[must_use]
    pub fn responsive_canvas(mut self, id: &str) -> Self {
        self.rules.push(
            CssRule::new(&format!("#{id}"))
                .declaration("width", "100vw")
                .declaration("height", "100vh")
                .declaration("display", "block")
                .declaration("touch-action", "none"),
        );
        self
    }

    /// Add fullscreen body styling
    #[must_use]
    pub fn fullscreen_body(mut self) -> Self {
        self.rules.push(
            CssRule::new("html, body")
                .declaration("width", "100%")
                .declaration("height", "100%")
                .declaration("margin", "0")
                .declaration("padding", "0")
                .declaration("overflow", "hidden"),
        );
        self
    }

    /// Add a media query rule
    #[must_use]
    pub fn media_query(mut self, query: &str, rule: CssRule) -> Self {
        // Store as a special rule with media query prefix
        let media_rule = CssRule {
            selector: format!("@media {query} {{ {} }}", rule.selector),
            declarations: rule.declarations,
        };
        self.rules.push(media_rule);
        self
    }

    /// Add dark mode support
    #[must_use]
    #[allow(clippy::literal_string_with_formatting_args)]
    pub fn dark_mode(mut self, background: &str, foreground: &str) -> Self {
        self.rules.push(
            CssRule::new("@media (prefers-color-scheme: dark) { :root }")
                .declaration("--bg-color", background)
                .declaration("--fg-color", foreground),
        );
        self
    }

    /// Build the CSS stylesheet
    ///
    /// # Errors
    ///
    /// Currently always succeeds, but returns Result for future validation
    pub fn build(self) -> ProbarResult<GeneratedCss> {
        let mut content = String::new();

        // Generate CSS variables in :root
        if !self.variables.is_empty() {
            content.push_str(":root {\n");
            for (name, value) in &self.variables {
                content.push_str(&format!("    --{name}: {value};\n"));
            }
            content.push_str("}\n\n");
        }

        // Generate rules
        let rule_strings: Vec<String> = self.rules.iter().map(CssRule::render).collect();
        content.push_str(&rule_strings.join("\n\n"));

        Ok(GeneratedCss {
            content,
            rules: self.rules,
            variables: self.variables,
        })
    }
}

/// Pre-built CSS presets
#[allow(dead_code)]
pub mod presets {
    use super::{CssBuilder, CssRule};

    /// WASM application preset - fullscreen canvas with no scrollbars
    #[must_use]
    pub fn wasm_app(canvas_id: &str) -> CssBuilder {
        CssBuilder::new()
            .reset()
            .fullscreen_body()
            .responsive_canvas(canvas_id)
    }

    /// Calculator preset
    #[must_use]
    pub fn calculator() -> CssBuilder {
        CssBuilder::new()
            .reset()
            .variable("primary-color", "#4a90d9")
            .variable("secondary-color", "#2c3e50")
            .variable("bg-color", "#1a1a2e")
            .rule(
                CssRule::new("body")
                    .declaration("font-family", "system-ui, sans-serif")
                    .declaration("background", "var(--bg-color)")
                    .declaration("color", "#fff")
                    .declaration("display", "flex")
                    .declaration("justify-content", "center")
                    .declaration("align-items", "center")
                    .declaration("min-height", "100vh"),
            )
    }

    /// Game preset with loading screen support
    #[must_use]
    pub fn game(canvas_id: &str) -> CssBuilder {
        CssBuilder::new()
            .reset()
            .fullscreen_body()
            .responsive_canvas(canvas_id)
            .rule(
                CssRule::new(".loading")
                    .declaration("position", "fixed")
                    .declaration("inset", "0")
                    .declaration("display", "flex")
                    .declaration("justify-content", "center")
                    .declaration("align-items", "center")
                    .declaration("background", "#000")
                    .declaration("color", "#fff")
                    .declaration("font-size", "1.5rem"),
            )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::wildcard_imports)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-CSS-01: CssBuilder creation
    // =========================================================================

    #[test]
    fn h0_css_01_builder_new() {
        let builder = CssBuilder::new();
        assert!(builder.variables.is_empty());
        assert!(builder.rules.is_empty());
    }

    #[test]
    fn h0_css_02_builder_variable() {
        let css = CssBuilder::new()
            .variable("primary", "#ff0000")
            .build()
            .unwrap();

        assert!(css.content.contains(":root {"));
        assert!(css.content.contains("--primary: #ff0000;"));
    }

    // =========================================================================
    // H₀-CSS-03: CSS rule generation
    // =========================================================================

    #[test]
    fn h0_css_03_rule_render() {
        let rule = CssRule::new("body")
            .declaration("margin", "0")
            .declaration("padding", "0");

        let rendered = rule.render();
        assert!(rendered.contains("body {"));
        assert!(rendered.contains("margin: 0;"));
        assert!(rendered.contains("padding: 0;"));
    }

    #[test]
    fn h0_css_04_rule_empty_declarations() {
        let rule = CssRule::new("div");
        let rendered = rule.render();
        assert!(rendered.is_empty());
    }

    #[test]
    fn h0_css_05_builder_rule() {
        let css = CssBuilder::new()
            .rule(CssRule::new(".test").declaration("color", "red"))
            .build()
            .unwrap();

        assert!(css.content.contains(".test {"));
        assert!(css.content.contains("color: red;"));
    }

    // =========================================================================
    // H₀-CSS-06: Preset methods
    // =========================================================================

    #[test]
    fn h0_css_06_reset() {
        let css = CssBuilder::new().reset().build().unwrap();

        assert!(css.content.contains("box-sizing: border-box;"));
        assert!(css.content.contains("margin: 0;"));
        assert!(css.content.contains("padding: 0;"));
    }

    #[test]
    fn h0_css_07_responsive_canvas() {
        let css = CssBuilder::new().responsive_canvas("game").build().unwrap();

        assert!(css.content.contains("#game {"));
        assert!(css.content.contains("width: 100vw;"));
        assert!(css.content.contains("height: 100vh;"));
        assert!(css.content.contains("touch-action: none;"));
    }

    #[test]
    fn h0_css_08_fullscreen_body() {
        let css = CssBuilder::new().fullscreen_body().build().unwrap();

        assert!(css.content.contains("html, body {"));
        assert!(css.content.contains("overflow: hidden;"));
    }

    #[test]
    fn h0_css_09_dark_mode() {
        let css = CssBuilder::new().dark_mode("#000", "#fff").build().unwrap();

        assert!(css.content.contains("prefers-color-scheme: dark"));
        assert!(css.content.contains("--bg-color: #000;"));
        assert!(css.content.contains("--fg-color: #fff;"));
    }

    // =========================================================================
    // H₀-CSS-10: Preset modules
    // =========================================================================

    #[test]
    fn h0_css_10_preset_wasm_app() {
        let css = presets::wasm_app("app").build().unwrap();

        assert!(css.content.contains("#app {"));
        assert!(css.content.contains("100vw"));
    }

    #[test]
    fn h0_css_11_preset_calculator() {
        let css = presets::calculator().build().unwrap();

        assert!(css.content.contains("--primary-color"));
        assert!(css.content.contains("system-ui"));
    }

    #[test]
    fn h0_css_12_preset_game() {
        let css = presets::game("canvas").build().unwrap();

        assert!(css.content.contains("#canvas"));
        assert!(css.content.contains(".loading"));
    }

    // =========================================================================
    // H₀-CSS-13: Generated output structure
    // =========================================================================

    #[test]
    fn h0_css_13_generated_css_fields() {
        let css = CssBuilder::new()
            .variable("x", "1")
            .rule(CssRule::new("a").declaration("b", "c"))
            .build()
            .unwrap();

        assert_eq!(css.variables.len(), 1);
        assert_eq!(css.rules.len(), 1);
        assert!(!css.content.is_empty());
    }

    #[test]
    fn h0_css_14_chained_methods() {
        let css = CssBuilder::new()
            .reset()
            .fullscreen_body()
            .responsive_canvas("c")
            .variable("color", "blue")
            .build()
            .unwrap();

        // Should have reset + fullscreen_body + responsive_canvas = 3 rules
        assert_eq!(css.rules.len(), 3);
        assert_eq!(css.variables.len(), 1);
    }
}
