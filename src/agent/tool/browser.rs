//! BrowserTool — wraps `jugar_probar::Browser` for headless Chromium.
//!
//! Provides agent access to browser automation: navigate, screenshot,
//! evaluate JS/WASM, click elements. Sovereign privacy tier restricts
//! navigation to localhost/file:// URLs only.
//!
//! Feature-gated: requires `agents-browser` feature for
//! `jugar_probar` access.

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::{Tool, ToolResult};
use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;
use crate::serve::backends::PrivacyTier;

use jugar_probar::{Browser, BrowserConfig, Page};

/// Tool that wraps jugar-probar for headless Chromium automation.
pub struct BrowserTool {
    config: BrowserConfig,
    privacy_tier: PrivacyTier,
    page: Mutex<Option<Page>>,
}

impl BrowserTool {
    /// Create a new BrowserTool with the given privacy tier.
    pub fn new(privacy_tier: PrivacyTier) -> Self {
        Self { config: BrowserConfig::default(), privacy_tier, page: Mutex::new(None) }
    }

    /// Check if a URL is allowed under the current privacy tier.
    fn is_url_allowed(&self, url: &str) -> bool {
        match self.privacy_tier {
            PrivacyTier::Sovereign => {
                url.starts_with("http://localhost")
                    || url.starts_with("http://127.0.0.1")
                    || url.starts_with("https://localhost")
                    || url.starts_with("https://127.0.0.1")
                    || url.starts_with("file://")
            }
            PrivacyTier::Private | PrivacyTier::Standard => true,
        }
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &'static str {
        "browser"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "browser".into(),
            description: "Headless browser automation (navigate, \
                          screenshot, evaluate JS/WASM)"
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "navigate", "screenshot", "evaluate",
                            "eval_wasm", "click", "wait_wasm",
                            "console"
                        ],
                        "description": "Browser action to perform"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (navigate)"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector (screenshot, click)"
                    },
                    "expression": {
                        "type": "string",
                        "description": "JS/WASM expression (evaluate, eval_wasm)"
                    },
                    "clear": {
                        "type": "boolean",
                        "description": "Clear console messages (console)"
                    }
                },
                "required": ["action"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let action = match input.get("action").and_then(|a| a.as_str()) {
            Some(a) => a,
            None => {
                return ToolResult::error("missing required field: action");
            }
        };

        match action {
            "navigate" => self.handle_navigate(&input).await,
            "screenshot" => self.handle_screenshot().await,
            "evaluate" => self.handle_evaluate(&input).await,
            "eval_wasm" => self.handle_eval_wasm(&input).await,
            "click" => self.handle_click(&input).await,
            "wait_wasm" => self.handle_wait_wasm().await,
            "console" => self.handle_console().await,
            _ => ToolResult::error(format!("unknown action: {action}")),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Browser
    }

    fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }
}

impl BrowserTool {
    async fn ensure_page(&self) -> Result<(), ToolResult> {
        let mut guard = self.page.lock().await;
        if guard.is_none() {
            let browser = Browser::launch(self.config.clone())
                .await
                .map_err(|e| ToolResult::error(format!("browser launch: {e}")))?;
            let new_page = browser
                .new_page()
                .await
                .map_err(|e| ToolResult::error(format!("new page: {e}")))?;
            *guard = Some(new_page);
        }
        Ok(())
    }

    async fn handle_navigate(&self, input: &serde_json::Value) -> ToolResult {
        let url = match input.get("url").and_then(|u| u.as_str()) {
            Some(u) => u,
            None => return ToolResult::error("navigate: missing url"),
        };

        if !self.is_url_allowed(url) {
            return ToolResult::error(format!(
                "navigate blocked: URL '{url}' not allowed \
                 under {:?} privacy tier",
                self.privacy_tier
            ));
        }

        if let Err(e) = self.ensure_page().await {
            return e;
        }

        let mut guard = self.page.lock().await;
        let Some(page) = guard.as_mut() else {
            return ToolResult::error("browser page not initialized");
        };
        match page.goto(url).await {
            Ok(()) => {
                let current = page.current_url().to_string();
                ToolResult::success(format!("Navigated to: {current}"))
            }
            Err(e) => ToolResult::error(format!("navigate failed: {e}")),
        }
    }

    async fn handle_screenshot(&self) -> ToolResult {
        if let Err(e) = self.ensure_page().await {
            return e;
        }
        let guard = self.page.lock().await;
        let Some(page) = guard.as_ref() else {
            return ToolResult::error("browser page not initialized");
        };
        match page.screenshot().await {
            Ok(bytes) => {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                ToolResult::success(format!(
                    "Screenshot ({} bytes): \
                     data:image/png;base64,{b64}",
                    bytes.len()
                ))
            }
            Err(e) => ToolResult::error(format!("screenshot failed: {e}")),
        }
    }

    async fn handle_evaluate(&self, input: &serde_json::Value) -> ToolResult {
        let expr = match input.get("expression").and_then(|e| e.as_str()) {
            Some(e) => e,
            None => {
                return ToolResult::error("evaluate: missing expression");
            }
        };

        if let Err(e) = self.ensure_page().await {
            return e;
        }
        let guard = self.page.lock().await;
        let Some(page) = guard.as_ref() else {
            return ToolResult::error("browser page not initialized");
        };
        match page.evaluate(expr).await {
            Ok(val) => ToolResult::success(format!("{val:?}")),
            Err(e) => ToolResult::error(format!("evaluate failed: {e}")),
        }
    }

    async fn handle_eval_wasm(&self, input: &serde_json::Value) -> ToolResult {
        let expr = match input.get("expression").and_then(|e| e.as_str()) {
            Some(e) => e,
            None => {
                return ToolResult::error("eval_wasm: missing expression");
            }
        };

        if let Err(e) = self.ensure_page().await {
            return e;
        }
        let guard = self.page.lock().await;
        let Some(page) = guard.as_ref() else {
            return ToolResult::error("browser page not initialized");
        };
        match page.eval_wasm::<serde_json::Value>(expr).await {
            Ok(val) => ToolResult::success(
                serde_json::to_string_pretty(&val)
                    .unwrap_or_else(|e| format!("{val:?} (serialize error: {e})")),
            ),
            Err(e) => ToolResult::error(format!("eval_wasm failed: {e}")),
        }
    }

    async fn handle_click(&self, input: &serde_json::Value) -> ToolResult {
        let selector = match input.get("selector").and_then(|s| s.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult::error("click: missing selector");
            }
        };

        if let Err(e) = self.ensure_page().await {
            return e;
        }
        let guard = self.page.lock().await;
        let Some(page) = guard.as_ref() else {
            return ToolResult::error("browser page not initialized");
        };
        match page.click(selector).await {
            Ok(()) => ToolResult::success(format!("Clicked: {selector}")),
            Err(e) => ToolResult::error(format!("click failed: {e}")),
        }
    }

    async fn handle_wait_wasm(&self) -> ToolResult {
        if let Err(e) = self.ensure_page().await {
            return e;
        }
        let mut guard = self.page.lock().await;
        let Some(page) = guard.as_mut() else {
            return ToolResult::error("browser page not initialized");
        };
        match page.wait_for_wasm_ready().await {
            Ok(()) => ToolResult::success("WASM runtime ready"),
            Err(e) => ToolResult::error(format!("wait_wasm failed: {e}")),
        }
    }

    async fn handle_console(&self) -> ToolResult {
        if let Err(e) = self.ensure_page().await {
            return e;
        }
        let guard = self.page.lock().await;
        let Some(page) = guard.as_ref() else {
            return ToolResult::error("browser page not initialized");
        };
        let msgs = page.console_messages().await;
        let formatted: Vec<String> =
            msgs.iter().map(|m| format!("[{}] {}", m.level, m.text)).collect();
        ToolResult::success(formatted.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereign_url_restriction() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        assert!(tool.is_url_allowed("http://localhost:8080"));
        assert!(tool.is_url_allowed("http://127.0.0.1:3000"));
        assert!(tool.is_url_allowed("https://localhost"));
        assert!(tool.is_url_allowed("file:///tmp/test.html"));
        assert!(!tool.is_url_allowed("https://example.com"));
        assert!(!tool.is_url_allowed("http://evil.com"));
    }

    #[test]
    fn test_standard_allows_all() {
        let tool = BrowserTool::new(PrivacyTier::Standard);
        assert!(tool.is_url_allowed("https://example.com"));
        assert!(tool.is_url_allowed("http://localhost:8080"));
    }

    #[test]
    fn test_private_allows_all() {
        let tool = BrowserTool::new(PrivacyTier::Private);
        assert!(tool.is_url_allowed("https://example.com"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        assert_eq!(tool.name(), "browser");
        assert_eq!(tool.required_capability(), Capability::Browser);
        assert_eq!(tool.timeout(), std::time::Duration::from_secs(30));
    }

    #[test]
    fn test_definition_schema() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let def = tool.definition();
        assert_eq!(def.name, "browser");
        let props = def.input_schema.get("properties");
        assert!(props.is_some());
        let action = props.expect("props exists").get("action");
        assert!(action.is_some());
    }

    #[tokio::test]
    async fn test_missing_action() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("action"));
    }

    #[tokio::test]
    async fn test_unknown_action() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool.execute(serde_json::json!({"action": "fly"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("unknown action"));
    }

    #[tokio::test]
    async fn test_navigate_missing_url() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool.execute(serde_json::json!({"action": "navigate"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing url"));
    }

    #[tokio::test]
    async fn test_navigate_blocked_by_privacy() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool
            .execute(serde_json::json!({
                "action": "navigate",
                "url": "https://example.com"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("blocked"));
    }

    #[tokio::test]
    async fn test_evaluate_missing_expression() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool.execute(serde_json::json!({"action": "evaluate"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing expression"));
    }

    #[tokio::test]
    async fn test_eval_wasm_missing_expression() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool.execute(serde_json::json!({"action": "eval_wasm"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing expression"));
    }

    #[tokio::test]
    async fn test_click_missing_selector() {
        let tool = BrowserTool::new(PrivacyTier::Sovereign);
        let result = tool.execute(serde_json::json!({"action": "click"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing selector"));
    }
}
