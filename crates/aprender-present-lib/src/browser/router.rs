//! Browser router with History API integration.
//!
//! Provides navigation and URL management for single-page applications.
//!
//! # Example
//!
//! ```ignore
//! use presentar::browser::router::BrowserRouter;
//!
//! let router = BrowserRouter::new();
//! router.navigate("/dashboard");
//! ```

use presentar_core::Router;
use std::sync::Mutex;

/// Browser router that uses the History API.
///
/// In WASM, this interfaces with the browser's history.pushState/replaceState.
/// In non-WASM (tests), this uses an in-memory implementation.
#[derive(Debug)]
pub struct BrowserRouter {
    /// In-memory state for non-WASM environments
    #[cfg(not(target_arch = "wasm32"))]
    state: Mutex<BrowserRouterState>,
}

impl Default for BrowserRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct BrowserRouterState {
    current: String,
    history: Vec<String>,
    history_index: usize,
}

impl BrowserRouter {
    /// Create a new browser router.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self {}
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                state: Mutex::new(BrowserRouterState {
                    current: "/".to_string(),
                    history: vec!["/".to_string()],
                    history_index: 0,
                }),
            }
        }
    }

    /// Get the current pathname.
    #[must_use]
    pub fn pathname(&self) -> String {
        #[cfg(target_arch = "wasm32")]
        {
            self.pathname_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state
                .lock()
                .map(|s| s.current.clone())
                .unwrap_or_else(|_| "/".to_string())
        }
    }

    /// Get the current search query string.
    #[must_use]
    pub fn search(&self) -> String {
        #[cfg(target_arch = "wasm32")]
        {
            self.search_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Parse query from current URL
            let current = self.pathname();
            current
                .find('?')
                .map(|i| current[i..].to_string())
                .unwrap_or_default()
        }
    }

    /// Get the current hash.
    #[must_use]
    pub fn hash(&self) -> String {
        #[cfg(target_arch = "wasm32")]
        {
            self.hash_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Parse hash from current URL
            let current = self.pathname();
            current
                .find('#')
                .map(|i| current[i..].to_string())
                .unwrap_or_default()
        }
    }

    /// Navigate to a new route, adding to history.
    pub fn push(&self, path: &str) {
        #[cfg(target_arch = "wasm32")]
        {
            self.push_wasm(path);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut state) = self.state.lock() {
                // Truncate forward history if we're not at the end
                let idx = state.history_index;
                if idx < state.history.len().saturating_sub(1) {
                    state.history.truncate(idx + 1);
                }
                state.current = path.to_string();
                state.history.push(path.to_string());
                state.history_index = state.history.len() - 1;
            }
        }
    }

    /// Replace the current route without adding to history.
    pub fn replace(&self, path: &str) {
        #[cfg(target_arch = "wasm32")]
        {
            self.replace_wasm(path);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut state) = self.state.lock() {
                state.current = path.to_string();
                let idx = state.history_index;
                if let Some(entry) = state.history.get_mut(idx) {
                    *entry = path.to_string();
                }
            }
        }
    }

    /// Go back in history.
    pub fn back(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.back_wasm();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut state) = self.state.lock() {
                if state.history_index > 0 {
                    state.history_index -= 1;
                    state.current = state.history[state.history_index].clone();
                }
            }
        }
    }

    /// Go forward in history.
    pub fn forward(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.forward_wasm();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut state) = self.state.lock() {
                if state.history_index < state.history.len().saturating_sub(1) {
                    state.history_index += 1;
                    state.current = state.history[state.history_index].clone();
                }
            }
        }
    }

    /// Go to a specific point in history (positive = forward, negative = back).
    pub fn go(&self, delta: i32) {
        #[cfg(target_arch = "wasm32")]
        {
            self.go_wasm(delta);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut state) = self.state.lock() {
                let new_index = if delta >= 0 {
                    state.history_index.saturating_add(delta as usize)
                } else {
                    state.history_index.saturating_sub((-delta) as usize)
                };
                if new_index < state.history.len() {
                    state.history_index = new_index;
                    state.current = state.history[new_index].clone();
                }
            }
        }
    }

    /// Get the history length.
    #[must_use]
    pub fn history_len(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            self.history_len_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state.lock().map(|s| s.history.len()).unwrap_or(0)
        }
    }

    /// Check if we can go back.
    #[must_use]
    pub fn can_go_back(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.history_len() > 1
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state
                .lock()
                .map(|s| s.history_index > 0)
                .unwrap_or(false)
        }
    }

    /// Check if we can go forward.
    #[must_use]
    pub fn can_go_forward(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state
                .lock()
                .map(|s| s.history_index < s.history.len().saturating_sub(1))
                .unwrap_or(false)
        }
        #[cfg(target_arch = "wasm32")]
        {
            false // Can't reliably detect in browser
        }
    }

    // WASM implementations
    #[cfg(target_arch = "wasm32")]
    fn pathname_wasm(&self) -> String {
        web_sys::window()
            .and_then(|w| w.location().pathname().ok())
            .unwrap_or_else(|| "/".to_string())
    }

    #[cfg(target_arch = "wasm32")]
    fn search_wasm(&self) -> String {
        web_sys::window()
            .and_then(|w| w.location().search().ok())
            .unwrap_or_default()
    }

    #[cfg(target_arch = "wasm32")]
    fn hash_wasm(&self) -> String {
        web_sys::window()
            .and_then(|w| w.location().hash().ok())
            .unwrap_or_default()
    }

    #[cfg(target_arch = "wasm32")]
    fn push_wasm(&self, path: &str) {
        if let Some(window) = web_sys::window() {
            if let Ok(history) = window.history() {
                let _ = history.push_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(path));
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn replace_wasm(&self, path: &str) {
        if let Some(window) = web_sys::window() {
            if let Ok(history) = window.history() {
                let _ =
                    history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(path));
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn back_wasm(&self) {
        if let Some(window) = web_sys::window() {
            if let Ok(history) = window.history() {
                let _ = history.back();
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn forward_wasm(&self) {
        if let Some(window) = web_sys::window() {
            if let Ok(history) = window.history() {
                let _ = history.forward();
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn go_wasm(&self, delta: i32) {
        if let Some(window) = web_sys::window() {
            if let Ok(history) = window.history() {
                let _ = history.go_with_delta(delta);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn history_len_wasm(&self) -> usize {
        web_sys::window()
            .and_then(|w| w.history().ok())
            .and_then(|h| h.length().ok())
            .unwrap_or(0) as usize
    }
}

impl Router for BrowserRouter {
    fn navigate(&self, route: &str) {
        self.push(route);
    }

    fn current_route(&self) -> String {
        self.pathname()
    }
}

/// Route matching result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteMatch {
    /// The matched route pattern.
    pub pattern: String,
    /// Extracted path parameters.
    pub params: std::collections::HashMap<String, String>,
}

impl RouteMatch {
    /// Create a new route match.
    #[must_use]
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            params: std::collections::HashMap::new(),
        }
    }

    /// Get a parameter value.
    #[must_use]
    pub fn param(&self, name: &str) -> Option<&str> {
        self.params.get(name).map(String::as_str)
    }
}

/// Pattern-based route matcher.
#[derive(Debug, Clone)]
pub struct RouteMatcher {
    routes: Vec<RoutePattern>,
}

#[derive(Debug, Clone)]
struct RoutePattern {
    pattern: String,
    segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
enum Segment {
    Static(String),
    Param(String),
    Wildcard,
}

impl RouteMatcher {
    /// Create a new route matcher.
    #[must_use]
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Add a route pattern.
    ///
    /// Patterns support:
    /// - Static segments: `/users/list`
    /// - Path parameters: `/users/:id`
    /// - Wildcards: `/files/*`
    pub fn add(&mut self, pattern: &str) -> &mut Self {
        let segments = pattern
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| {
                if s == "*" {
                    Segment::Wildcard
                } else if let Some(name) = s.strip_prefix(':') {
                    Segment::Param(name.to_string())
                } else {
                    Segment::Static(s.to_string())
                }
            })
            .collect();

        self.routes.push(RoutePattern {
            pattern: pattern.to_string(),
            segments,
        });
        self
    }

    /// Match a path against registered routes.
    #[must_use]
    pub fn match_path(&self, path: &str) -> Option<RouteMatch> {
        // Remove query string and hash
        let path = path.split('?').next().unwrap_or(path);
        let path = path.split('#').next().unwrap_or(path);

        let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        for route in &self.routes {
            if let Some(params) = self.try_match(&route.segments, &path_segments) {
                return Some(RouteMatch {
                    pattern: route.pattern.clone(),
                    params,
                });
            }
        }

        None
    }

    fn try_match(
        &self,
        pattern: &[Segment],
        path: &[&str],
    ) -> Option<std::collections::HashMap<String, String>> {
        let mut params = std::collections::HashMap::new();
        let mut path_iter = path.iter();

        for segment in pattern {
            match segment {
                Segment::Static(expected) => {
                    let actual = path_iter.next()?;
                    if *actual != expected {
                        return None;
                    }
                }
                Segment::Param(name) => {
                    let value = path_iter.next()?;
                    params.insert(name.clone(), (*value).to_string());
                }
                Segment::Wildcard => {
                    // Wildcard matches rest of path
                    let rest: Vec<&str> = path_iter.copied().collect();
                    params.insert("*".to_string(), rest.join("/"));
                    return Some(params);
                }
            }
        }

        // Check that we consumed all path segments
        if path_iter.next().is_some() {
            return None;
        }

        Some(params)
    }
}

impl Default for RouteMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BrowserRouter Tests
    // =========================================================================

    #[test]
    fn test_router_new() {
        let router = BrowserRouter::new();
        assert_eq!(router.pathname(), "/");
    }

    #[test]
    fn test_router_push() {
        let router = BrowserRouter::new();
        router.push("/dashboard");
        assert_eq!(router.pathname(), "/dashboard");
    }

    #[test]
    fn test_router_multiple_push() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        router.push("/page3");
        assert_eq!(router.pathname(), "/page3");
        assert_eq!(router.history_len(), 4); // Initial + 3 pushes
    }

    #[test]
    fn test_router_replace() {
        let router = BrowserRouter::new();
        router.push("/original");
        router.replace("/replaced");
        assert_eq!(router.pathname(), "/replaced");
        assert_eq!(router.history_len(), 2); // Initial + 1 push (replace doesn't add)
    }

    #[test]
    fn test_router_back() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        router.back();
        assert_eq!(router.pathname(), "/page1");
    }

    #[test]
    fn test_router_forward() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        router.back();
        router.forward();
        assert_eq!(router.pathname(), "/page2");
    }

    #[test]
    fn test_router_go_positive() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        router.push("/page3");
        router.go(-2);
        assert_eq!(router.pathname(), "/page1");
        router.go(1);
        assert_eq!(router.pathname(), "/page2");
    }

    #[test]
    fn test_router_go_negative() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        router.go(-1);
        assert_eq!(router.pathname(), "/page1");
    }

    #[test]
    fn test_router_can_go_back() {
        let router = BrowserRouter::new();
        assert!(!router.can_go_back());
        router.push("/page1");
        assert!(router.can_go_back());
    }

    #[test]
    fn test_router_can_go_forward() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        assert!(!router.can_go_forward());
        router.back();
        assert!(router.can_go_forward());
    }

    #[test]
    fn test_router_trait_navigate() {
        let router = BrowserRouter::new();
        router.navigate("/test");
        assert_eq!(router.current_route(), "/test");
    }

    #[test]
    fn test_router_back_at_start() {
        let router = BrowserRouter::new();
        router.back(); // Should not panic
        assert_eq!(router.pathname(), "/");
    }

    #[test]
    fn test_router_forward_at_end() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.forward(); // Should not panic
        assert_eq!(router.pathname(), "/page1");
    }

    #[test]
    fn test_router_history_truncation() {
        let router = BrowserRouter::new();
        router.push("/page1");
        router.push("/page2");
        router.push("/page3");
        router.back();
        router.back(); // At page1
        router.push("/new"); // Should truncate page2 and page3
        assert_eq!(router.pathname(), "/new");
        router.forward(); // Should not go anywhere
        assert_eq!(router.pathname(), "/new");
    }

    // =========================================================================
    // RouteMatch Tests
    // =========================================================================

    #[test]
    fn test_route_match_new() {
        let m = RouteMatch::new("/users/:id");
        assert_eq!(m.pattern, "/users/:id");
        assert!(m.params.is_empty());
    }

    #[test]
    fn test_route_match_param() {
        let mut m = RouteMatch::new("/users/:id");
        m.params.insert("id".to_string(), "123".to_string());
        assert_eq!(m.param("id"), Some("123"));
        assert_eq!(m.param("other"), None);
    }

    // =========================================================================
    // RouteMatcher Tests
    // =========================================================================

    #[test]
    fn test_matcher_static_route() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/users/list");

        let result = matcher.match_path("/users/list");
        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern, "/users/list");

        assert!(matcher.match_path("/users").is_none());
        assert!(matcher.match_path("/users/list/extra").is_none());
    }

    #[test]
    fn test_matcher_param_route() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/users/:id");

        let result = matcher.match_path("/users/123");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.pattern, "/users/:id");
        assert_eq!(m.param("id"), Some("123"));
    }

    #[test]
    fn test_matcher_multiple_params() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/users/:userId/posts/:postId");

        let result = matcher.match_path("/users/42/posts/99");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.param("userId"), Some("42"));
        assert_eq!(m.param("postId"), Some("99"));
    }

    #[test]
    fn test_matcher_wildcard() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/files/*");

        let result = matcher.match_path("/files/path/to/file.txt");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.param("*"), Some("path/to/file.txt"));
    }

    #[test]
    fn test_matcher_priority() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/users/me");
        matcher.add("/users/:id");

        // Static routes should be added first to match first
        let result = matcher.match_path("/users/me");
        assert_eq!(result.unwrap().pattern, "/users/me");

        let result = matcher.match_path("/users/123");
        assert_eq!(result.unwrap().pattern, "/users/:id");
    }

    #[test]
    fn test_matcher_with_query_string() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/search");

        let result = matcher.match_path("/search?q=test");
        assert!(result.is_some());
    }

    #[test]
    fn test_matcher_with_hash() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/page");

        let result = matcher.match_path("/page#section");
        assert!(result.is_some());
    }

    #[test]
    fn test_matcher_root() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/");

        // Empty pattern should match root
        assert!(matcher.match_path("/").is_some());
    }

    #[test]
    fn test_matcher_no_match() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/users");
        matcher.add("/posts");

        assert!(matcher.match_path("/comments").is_none());
    }

    #[test]
    fn test_matcher_empty() {
        let matcher = RouteMatcher::new();
        assert!(matcher.match_path("/anything").is_none());
    }

    #[test]
    fn test_matcher_default() {
        let matcher = RouteMatcher::default();
        assert!(matcher.match_path("/anything").is_none());
    }

    #[test]
    fn test_matcher_complex_route() {
        let mut matcher = RouteMatcher::new();
        matcher.add("/api/v1/users/:id/profile");

        let result = matcher.match_path("/api/v1/users/456/profile");
        assert!(result.is_some());
        assert_eq!(result.unwrap().param("id"), Some("456"));
    }

    #[test]
    fn test_router_default() {
        let router = BrowserRouter::default();
        assert_eq!(router.pathname(), "/");
    }
}
