//! Network tool for HTTP requests with privacy-tier enforcement.
//!
//! Allows agents to make HTTP GET/POST requests to allowed hosts.
//! Sovereign tier blocks all network access (Poka-Yoke).
//! Standard/Private tiers allow configured hosts only.

use std::time::Duration;

use async_trait::async_trait;

use crate::agent::capability::Capability;
use crate::agent::driver::ToolDefinition;

use super::{Tool, ToolResult};

/// Maximum response body bytes before truncation.
const MAX_RESPONSE_BYTES: usize = 16384;

/// HTTP network tool with host allowlisting.
pub struct NetworkTool {
    allowed_hosts: Vec<String>,
    timeout: Duration,
}

impl NetworkTool {
    /// Create a new network tool with allowed hosts.
    pub fn new(allowed_hosts: Vec<String>) -> Self {
        Self { allowed_hosts, timeout: Duration::from_secs(30) }
    }

    /// Check if a URL's host is allowed.
    fn is_host_allowed(&self, url: &str) -> bool {
        if self.allowed_hosts.iter().any(|h| h == "*") {
            return true;
        }
        let host = extract_host(url);
        self.allowed_hosts.iter().any(|h| h == &host)
    }
}

/// Extract hostname from a URL string.
fn extract_host(url: &str) -> String {
    url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_string()
}

#[async_trait]
impl Tool for NetworkTool {
    fn name(&self) -> &'static str {
        "network"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "network".into(),
            description: "Make HTTP requests to allowed hosts. \
                Supports GET and POST methods."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method (default: GET)"
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body for POST"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    #[cfg_attr(
        feature = "agents-contracts",
        provable_contracts_macros::contract("agent-loop-v1", equation = "network_host_allowlist")
    )]
    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let Some(url) = input.get("url").and_then(|v| v.as_str()) else {
            return ToolResult::error("missing required field: url");
        };

        if !self.is_host_allowed(url) {
            let host = extract_host(url);
            return ToolResult::error(format!("host '{host}' not in allowed_hosts"));
        }

        let method = input.get("method").and_then(|v| v.as_str()).unwrap_or("GET");

        // Build and execute request
        let client = match reqwest::Client::builder().timeout(self.timeout).build() {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("client error: {e}")),
        };

        let request = match method.to_uppercase().as_str() {
            "GET" => client.get(url),
            "POST" => {
                let body = input.get("body").and_then(|v| v.as_str()).unwrap_or("");
                client.post(url).body(body.to_string())
            }
            other => {
                return ToolResult::error(format!("unsupported method: {other}"));
            }
        };

        match request.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                match response.text().await {
                    Ok(body) => {
                        let truncated = if body.len() > MAX_RESPONSE_BYTES {
                            format!("{}...[truncated]", &body[..MAX_RESPONSE_BYTES])
                        } else {
                            body
                        };
                        ToolResult::success(format!("HTTP {status}\n{truncated}"))
                    }
                    Err(e) => ToolResult::error(format!("HTTP {status}, body read error: {e}")),
                }
            }
            Err(e) => ToolResult::error(format!("request failed: {e}")),
        }
    }

    fn required_capability(&self) -> Capability {
        Capability::Network { allowed_hosts: self.allowed_hosts.clone() }
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_tool_definition() {
        let tool = NetworkTool::new(vec!["api.example.com".into()]);
        let def = tool.definition();
        assert_eq!(def.name, "network");
        assert!(def.description.contains("HTTP"));
    }

    #[test]
    fn test_network_tool_capability() {
        let tool = NetworkTool::new(vec!["localhost".into()]);
        assert_eq!(
            tool.required_capability(),
            Capability::Network { allowed_hosts: vec!["localhost".into()] },
        );
    }

    #[test]
    fn test_extract_host() {
        assert_eq!(extract_host("https://api.example.com/path"), "api.example.com");
        assert_eq!(extract_host("http://localhost:8080/api"), "localhost");
        assert_eq!(extract_host("example.com/foo"), "example.com");
    }

    #[test]
    fn test_host_allowed_specific() {
        let tool = NetworkTool::new(vec!["api.example.com".into()]);
        assert!(tool.is_host_allowed("https://api.example.com/v1"));
        assert!(!tool.is_host_allowed("https://evil.com/hack"));
    }

    #[test]
    fn test_host_allowed_wildcard() {
        let tool = NetworkTool::new(vec!["*".into()]);
        assert!(tool.is_host_allowed("https://anything.com/path"));
    }

    #[tokio::test]
    async fn test_missing_url() {
        let tool = NetworkTool::new(vec!["*".into()]);
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_error);
        assert!(result.content.contains("missing"));
    }

    #[tokio::test]
    async fn test_blocked_host() {
        let tool = NetworkTool::new(vec!["allowed.com".into()]);
        let result = tool.execute(serde_json::json!({"url": "https://blocked.com/api"})).await;
        assert!(result.is_error);
        assert!(result.content.contains("not in allowed_hosts"));
    }

    #[tokio::test]
    async fn test_unsupported_method() {
        let tool = NetworkTool::new(vec!["*".into()]);
        let result = tool
            .execute(serde_json::json!({
                "url": "https://example.com",
                "method": "DELETE"
            }))
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("unsupported method"));
    }
}
