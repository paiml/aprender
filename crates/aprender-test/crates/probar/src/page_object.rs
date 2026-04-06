//! Page Object Model Support (Feature 19)
//!
//! First-class support for the Page Object Model pattern in test automation.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application:
//! - **Poka-Yoke**: Type-safe selectors prevent invalid queries at compile time
//! - **Muda**: Reduce duplication by encapsulating page logic
//! - **Genchi Genbutsu**: Page objects reflect actual page structure

use crate::locator::{Locator, Selector};
use std::collections::HashMap;

/// Trait for page objects representing a page or component in the UI.
///
/// Implement this trait to create reusable page objects that encapsulate
/// the structure and behavior of UI pages.
///
/// # Example
///
/// ```ignore
/// struct LoginPage {
///     username_input: Locator,
///     password_input: Locator,
///     submit_button: Locator,
/// }
///
/// impl PageObject for LoginPage {
///     fn url_pattern(&self) -> &str {
///         "/login"
///     }
///
///     fn is_loaded(&self) -> bool {
///         // Check if page-specific elements are present
///         true
///     }
/// }
///
/// impl LoginPage {
///     pub fn new() -> Self {
///         Self {
///             username_input: Locator::new(Selector::css("input[name='username']")),
///             password_input: Locator::new(Selector::css("input[name='password']")),
///             submit_button: Locator::new(Selector::css("button[type='submit']")),
///         }
///     }
///
///     pub async fn login(&self, username: &str, password: &str) -> ProbarResult<()> {
///         // Implementation
///         Ok(())
///     }
/// }
/// ```
pub trait PageObject {
    /// URL pattern that matches this page (e.g., "/login", "/users/*")
    fn url_pattern(&self) -> &str;

    /// Check if the page is fully loaded and ready for interaction
    fn is_loaded(&self) -> bool {
        true
    }

    /// Optional wait time for page load (in milliseconds)
    fn load_timeout_ms(&self) -> u64 {
        30000
    }

    /// Get the page name for logging/debugging
    fn page_name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Builder for creating page objects with locators
#[derive(Debug, Clone)]
pub struct PageObjectBuilder {
    url_pattern: String,
    locators: HashMap<String, Locator>,
    load_timeout_ms: u64,
}

impl Default for PageObjectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PageObjectBuilder {
    /// Create a new page object builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            url_pattern: String::new(),
            locators: HashMap::new(),
            load_timeout_ms: 30000,
        }
    }

    /// Set the URL pattern
    #[must_use]
    pub fn with_url_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.url_pattern = pattern.into();
        self
    }

    /// Add a locator with a name
    #[must_use]
    pub fn with_locator(mut self, name: impl Into<String>, selector: Selector) -> Self {
        let _ = self
            .locators
            .insert(name.into(), Locator::from_selector(selector));
        self
    }

    /// Set the load timeout
    #[must_use]
    pub const fn with_load_timeout(mut self, timeout_ms: u64) -> Self {
        self.load_timeout_ms = timeout_ms;
        self
    }

    /// Build a simple page object
    #[must_use]
    pub fn build(self) -> SimplePageObject {
        SimplePageObject {
            url_pattern: self.url_pattern,
            locators: self.locators,
            load_timeout_ms: self.load_timeout_ms,
        }
    }
}

/// A simple generic page object implementation
#[derive(Debug, Clone)]
pub struct SimplePageObject {
    url_pattern: String,
    locators: HashMap<String, Locator>,
    load_timeout_ms: u64,
}

impl SimplePageObject {
    /// Create a new simple page object
    #[must_use]
    pub fn new(url_pattern: impl Into<String>) -> Self {
        Self {
            url_pattern: url_pattern.into(),
            locators: HashMap::new(),
            load_timeout_ms: 30000,
        }
    }

    /// Get a locator by name
    #[must_use]
    pub fn locator(&self, name: &str) -> Option<&Locator> {
        self.locators.get(name)
    }

    /// Add a locator
    pub fn add_locator(&mut self, name: impl Into<String>, selector: Selector) {
        let _ = self
            .locators
            .insert(name.into(), Locator::from_selector(selector));
    }

    /// Get all locator names
    #[must_use]
    pub fn locator_names(&self) -> Vec<&str> {
        self.locators.keys().map(String::as_str).collect()
    }
}

impl PageObject for SimplePageObject {
    fn url_pattern(&self) -> &str {
        &self.url_pattern
    }

    fn load_timeout_ms(&self) -> u64 {
        self.load_timeout_ms
    }
}

/// Page object registry for managing multiple pages
#[derive(Debug, Default)]
pub struct PageRegistry {
    pages: HashMap<String, Box<dyn PageObjectInfo>>,
}

/// Trait for type-erased page object info
pub trait PageObjectInfo: std::fmt::Debug + Send + Sync {
    /// Get the URL pattern
    fn url_pattern(&self) -> &str;

    /// Get the page name
    fn page_name(&self) -> &str;

    /// Get the load timeout
    fn load_timeout_ms(&self) -> u64;
}

impl<T: PageObject + std::fmt::Debug + Send + Sync + 'static> PageObjectInfo for T {
    fn url_pattern(&self) -> &str {
        PageObject::url_pattern(self)
    }

    fn page_name(&self) -> &str {
        PageObject::page_name(self)
    }

    fn load_timeout_ms(&self) -> u64 {
        PageObject::load_timeout_ms(self)
    }
}

impl PageRegistry {
    /// Create a new page registry
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a page object
    pub fn register<T: PageObject + std::fmt::Debug + Send + Sync + 'static>(
        &mut self,
        name: impl Into<String>,
        page: T,
    ) {
        let _ = self.pages.insert(name.into(), Box::new(page));
    }

    /// Get a page by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn PageObjectInfo> {
        self.pages.get(name).map(|p| p.as_ref())
    }

    /// List all registered pages
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.pages.keys().map(String::as_str).collect()
    }

    /// Get the number of registered pages
    #[must_use]
    pub fn count(&self) -> usize {
        self.pages.len()
    }
}

/// URL pattern matcher for page objects
#[derive(Debug, Clone)]
pub struct UrlMatcher {
    pattern: String,
    segments: Vec<UrlSegment>,
}

#[derive(Debug, Clone)]
enum UrlSegment {
    Literal(String),
    Wildcard,
    Parameter(String),
}

impl UrlMatcher {
    /// Create a new URL matcher from a pattern
    ///
    /// Patterns support:
    /// - Literal segments: `/login`
    /// - Wildcards: `/users/*`
    /// - Named parameters: `/users/:id`
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        let segments = pattern
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| {
                if s == "*" {
                    UrlSegment::Wildcard
                } else if let Some(name) = s.strip_prefix(':') {
                    UrlSegment::Parameter(name.to_string())
                } else {
                    UrlSegment::Literal(s.to_string())
                }
            })
            .collect();

        Self {
            pattern: pattern.to_string(),
            segments,
        }
    }

    /// Check if a URL matches the pattern
    #[must_use]
    pub fn matches(&self, url: &str) -> bool {
        let url_segments: Vec<&str> = url.split('/').filter(|s| !s.is_empty()).collect();

        // URLs must have the same number of segments as the pattern
        // (wildcards and parameters each consume exactly one segment)
        if url_segments.len() != self.segments.len() {
            return false;
        }

        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                UrlSegment::Literal(lit) => {
                    if url_segments.get(i) != Some(&lit.as_str()) {
                        return false;
                    }
                }
                UrlSegment::Wildcard | UrlSegment::Parameter(_) => {
                    // Matches anything (but requires a value to exist, enforced by length check)
                }
            }
        }

        true
    }

    /// Extract parameters from a URL
    #[must_use]
    pub fn extract_params(&self, url: &str) -> HashMap<String, String> {
        let mut params = HashMap::new();
        let url_segments: Vec<&str> = url.split('/').filter(|s| !s.is_empty()).collect();

        for (i, segment) in self.segments.iter().enumerate() {
            if let UrlSegment::Parameter(name) = segment {
                if let Some(value) = url_segments.get(i) {
                    let _ = params.insert(name.clone(), (*value).to_string());
                }
            }
        }

        params
    }

    /// Get the original pattern
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod page_object_builder_tests {
        use super::*;

        #[test]
        fn test_builder_basic() {
            let page = PageObjectBuilder::new()
                .with_url_pattern("/login")
                .with_load_timeout(5000)
                .build();

            assert_eq!(PageObject::url_pattern(&page), "/login");
            assert_eq!(PageObject::load_timeout_ms(&page), 5000);
        }

        #[test]
        fn test_builder_with_locators() {
            let page = PageObjectBuilder::new()
                .with_url_pattern("/login")
                .with_locator("username", Selector::css("input[name='username']"))
                .with_locator("password", Selector::css("input[name='password']"))
                .build();

            assert!(page.locator("username").is_some());
            assert!(page.locator("password").is_some());
            assert!(page.locator("nonexistent").is_none());
        }

        #[test]
        fn test_default_builder() {
            let builder = PageObjectBuilder::default();
            let page = builder.build();
            assert!(PageObject::url_pattern(&page).is_empty());
        }
    }

    mod simple_page_object_tests {
        use super::*;

        #[test]
        fn test_new() {
            let page = SimplePageObject::new("/dashboard");
            assert_eq!(PageObject::url_pattern(&page), "/dashboard");
            assert_eq!(PageObject::load_timeout_ms(&page), 30000);
        }

        #[test]
        fn test_add_locator() {
            let mut page = SimplePageObject::new("/test");
            page.add_locator("button", Selector::css("button"));

            assert!(page.locator("button").is_some());
            assert!(page.locator_names().contains(&"button"));
        }

        #[test]
        fn test_is_loaded_default() {
            let page = SimplePageObject::new("/test");
            assert!(page.is_loaded());
        }
    }

    mod page_registry_tests {
        use super::*;

        #[test]
        fn test_new_registry() {
            let registry = PageRegistry::new();
            assert_eq!(registry.count(), 0);
        }

        #[test]
        fn test_register_and_get() {
            let mut registry = PageRegistry::new();
            let page = SimplePageObject::new("/login");
            registry.register("login", page);

            assert_eq!(registry.count(), 1);
            assert!(registry.get("login").is_some());
            assert!(registry.get("nonexistent").is_none());
        }

        #[test]
        fn test_list_pages() {
            let mut registry = PageRegistry::new();
            registry.register("login", SimplePageObject::new("/login"));
            registry.register("home", SimplePageObject::new("/"));

            let pages = registry.list();
            assert_eq!(pages.len(), 2);
            assert!(pages.contains(&"login"));
            assert!(pages.contains(&"home"));
        }
    }

    mod url_matcher_tests {
        use super::*;

        #[test]
        fn test_literal_match() {
            let matcher = UrlMatcher::new("/login");
            assert!(matcher.matches("/login"));
            assert!(!matcher.matches("/register"));
            assert!(!matcher.matches("/login/extra"));
        }

        #[test]
        fn test_wildcard_match() {
            let matcher = UrlMatcher::new("/users/*");
            assert!(matcher.matches("/users/123"));
            assert!(matcher.matches("/users/abc"));
            assert!(!matcher.matches("/users"));
            assert!(!matcher.matches("/other/123"));
        }

        #[test]
        fn test_parameter_match() {
            let matcher = UrlMatcher::new("/users/:id");
            assert!(matcher.matches("/users/123"));
            assert!(matcher.matches("/users/abc"));
            assert!(!matcher.matches("/users"));
        }

        #[test]
        fn test_extract_params() {
            let matcher = UrlMatcher::new("/users/:id/posts/:post_id");
            let params = matcher.extract_params("/users/42/posts/100");

            assert_eq!(params.get("id"), Some(&"42".to_string()));
            assert_eq!(params.get("post_id"), Some(&"100".to_string()));
        }

        #[test]
        fn test_complex_pattern() {
            let matcher = UrlMatcher::new("/api/v1/users/:id");
            assert!(matcher.matches("/api/v1/users/123"));
            assert!(!matcher.matches("/api/v2/users/123"));
        }

        #[test]
        fn test_pattern_getter() {
            let matcher = UrlMatcher::new("/test/pattern");
            assert_eq!(matcher.pattern(), "/test/pattern");
        }
    }

    mod page_object_trait_tests {
        use super::*;

        #[derive(Debug)]
        struct TestPage {
            url: String,
            loaded: bool,
        }

        impl PageObject for TestPage {
            fn url_pattern(&self) -> &str {
                &self.url
            }

            fn is_loaded(&self) -> bool {
                self.loaded
            }

            fn load_timeout_ms(&self) -> u64 {
                5000
            }
        }

        #[test]
        fn test_custom_page_object() {
            let page = TestPage {
                url: "/custom".to_string(),
                loaded: true,
            };

            assert_eq!(PageObject::url_pattern(&page), "/custom");
            assert!(PageObject::is_loaded(&page));
            assert_eq!(PageObject::load_timeout_ms(&page), 5000);
        }

        #[test]
        fn test_page_name() {
            let page = SimplePageObject::new("/test");
            // Should return the type name
            assert!(PageObject::page_name(&page).contains("SimplePageObject"));
        }
    }
}
