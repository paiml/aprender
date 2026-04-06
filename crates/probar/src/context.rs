//! Multi-Browser Context Management (Feature 14)
//!
//! Isolated browser contexts for parallel testing.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Muda**: Eliminate waste by reusing browser instances
//! - **Heijunka**: Load balancing across contexts
//! - **Jidoka**: Automatic context cleanup on failure

use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Browser context state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextState {
    /// Context is being created
    Creating,
    /// Context is ready for use
    Ready,
    /// Context is in use
    InUse,
    /// Context is being cleaned up
    Cleaning,
    /// Context is closed
    Closed,
    /// Context has an error
    Error,
}

/// Storage state for a context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageState {
    /// Cookies
    pub cookies: Vec<Cookie>,
    /// Local storage data
    pub local_storage: HashMap<String, HashMap<String, String>>,
    /// Session storage data
    pub session_storage: HashMap<String, HashMap<String, String>>,
}

impl StorageState {
    /// Create empty storage state
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a cookie
    #[must_use]
    pub fn with_cookie(mut self, cookie: Cookie) -> Self {
        self.cookies.push(cookie);
        self
    }

    /// Add local storage item
    #[must_use]
    pub fn with_local_storage(mut self, origin: &str, key: &str, value: &str) -> Self {
        self.local_storage
            .entry(origin.to_string())
            .or_default()
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Add session storage item
    #[must_use]
    pub fn with_session_storage(mut self, origin: &str, key: &str, value: &str) -> Self {
        self.session_storage
            .entry(origin.to_string())
            .or_default()
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Check if storage is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cookies.is_empty() && self.local_storage.is_empty() && self.session_storage.is_empty()
    }

    /// Clear all storage
    pub fn clear(&mut self) {
        self.cookies.clear();
        self.local_storage.clear();
        self.session_storage.clear();
    }
}

/// A browser cookie
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cookie {
    /// Cookie name
    pub name: String,
    /// Cookie value
    pub value: String,
    /// Domain
    pub domain: String,
    /// Path
    pub path: String,
    /// Expiration timestamp (seconds since epoch)
    pub expires: Option<i64>,
    /// HTTP only flag
    pub http_only: bool,
    /// Secure flag
    pub secure: bool,
    /// Same site setting
    pub same_site: SameSite,
}

impl Cookie {
    /// Create a new cookie
    #[must_use]
    pub fn new(name: &str, value: &str, domain: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            domain: domain.to_string(),
            path: "/".to_string(),
            expires: None,
            http_only: false,
            secure: false,
            same_site: SameSite::Lax,
        }
    }

    /// Set path
    #[must_use]
    pub fn with_path(mut self, path: &str) -> Self {
        self.path = path.to_string();
        self
    }

    /// Set expiration
    #[must_use]
    pub const fn with_expires(mut self, expires: i64) -> Self {
        self.expires = Some(expires);
        self
    }

    /// Set HTTP only
    #[must_use]
    pub const fn http_only(mut self) -> Self {
        self.http_only = true;
        self
    }

    /// Set secure
    #[must_use]
    pub const fn secure(mut self) -> Self {
        self.secure = true;
        self
    }

    /// Set same site
    #[must_use]
    pub const fn with_same_site(mut self, same_site: SameSite) -> Self {
        self.same_site = same_site;
        self
    }
}

/// Same site cookie setting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SameSite {
    /// Strict same site
    Strict,
    /// Lax same site
    Lax,
    /// No same site restriction
    None,
}

/// Configuration for a browser context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Context name/ID
    pub name: String,
    /// Viewport width
    pub viewport_width: u32,
    /// Viewport height
    pub viewport_height: u32,
    /// Device scale factor
    pub device_scale_factor: f64,
    /// Is mobile device
    pub is_mobile: bool,
    /// Has touch support
    pub has_touch: bool,
    /// User agent string
    pub user_agent: Option<String>,
    /// Locale
    pub locale: Option<String>,
    /// Timezone
    pub timezone: Option<String>,
    /// Geolocation
    pub geolocation: Option<Geolocation>,
    /// Permissions
    pub permissions: Vec<String>,
    /// Extra HTTP headers
    pub extra_headers: HashMap<String, String>,
    /// Offline mode
    pub offline: bool,
    /// Initial storage state
    pub storage_state: Option<StorageState>,
    /// Accept downloads
    pub accept_downloads: bool,
    /// Record video
    pub record_video: bool,
    /// Record HAR
    pub record_har: bool,
    /// Ignore HTTPS errors
    pub ignore_https_errors: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            viewport_width: 1280,
            viewport_height: 720,
            device_scale_factor: 1.0,
            is_mobile: false,
            has_touch: false,
            user_agent: None,
            locale: None,
            timezone: None,
            geolocation: None,
            permissions: Vec::new(),
            extra_headers: HashMap::new(),
            offline: false,
            storage_state: None,
            accept_downloads: false,
            record_video: false,
            record_har: false,
            ignore_https_errors: false,
        }
    }
}

impl ContextConfig {
    /// Create a new context config
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Self::default()
        }
    }

    /// Set viewport size
    #[must_use]
    pub const fn with_viewport(mut self, width: u32, height: u32) -> Self {
        self.viewport_width = width;
        self.viewport_height = height;
        self
    }

    /// Set device scale factor
    #[must_use]
    pub const fn with_device_scale(mut self, scale: f64) -> Self {
        self.device_scale_factor = scale;
        self
    }

    /// Set as mobile device
    #[must_use]
    pub const fn mobile(mut self) -> Self {
        self.is_mobile = true;
        self.has_touch = true;
        self
    }

    /// Set user agent
    #[must_use]
    pub fn with_user_agent(mut self, user_agent: &str) -> Self {
        self.user_agent = Some(user_agent.to_string());
        self
    }

    /// Set locale
    #[must_use]
    pub fn with_locale(mut self, locale: &str) -> Self {
        self.locale = Some(locale.to_string());
        self
    }

    /// Set timezone
    #[must_use]
    pub fn with_timezone(mut self, timezone: &str) -> Self {
        self.timezone = Some(timezone.to_string());
        self
    }

    /// Set geolocation
    #[must_use]
    pub fn with_geolocation(mut self, lat: f64, lng: f64) -> Self {
        self.geolocation = Some(Geolocation {
            latitude: lat,
            longitude: lng,
            accuracy: 10.0,
        });
        self
    }

    /// Add permission
    #[must_use]
    pub fn with_permission(mut self, permission: &str) -> Self {
        self.permissions.push(permission.to_string());
        self
    }

    /// Add extra header
    #[must_use]
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.extra_headers
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Set offline mode
    #[must_use]
    pub const fn offline(mut self) -> Self {
        self.offline = true;
        self
    }

    /// Set storage state
    #[must_use]
    pub fn with_storage_state(mut self, state: StorageState) -> Self {
        self.storage_state = Some(state);
        self
    }

    /// Enable video recording
    #[must_use]
    pub const fn with_video(mut self) -> Self {
        self.record_video = true;
        self
    }

    /// Enable HAR recording
    #[must_use]
    pub const fn with_har(mut self) -> Self {
        self.record_har = true;
        self
    }

    /// Ignore HTTPS errors
    #[must_use]
    pub const fn ignore_https_errors(mut self) -> Self {
        self.ignore_https_errors = true;
        self
    }
}

/// Geolocation coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Geolocation {
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
    /// Accuracy in meters
    pub accuracy: f64,
}

/// A browser context instance
#[derive(Debug)]
pub struct BrowserContext {
    /// Context ID
    pub id: String,
    /// Configuration
    pub config: ContextConfig,
    /// Current state
    pub state: ContextState,
    /// Creation time
    pub created_at: Instant,
    /// Pages in this context
    pages: Arc<Mutex<Vec<String>>>,
    /// Storage state
    storage: Arc<Mutex<StorageState>>,
    /// Error message if state is Error
    pub error_message: Option<String>,
}

impl BrowserContext {
    /// Create a new context
    #[must_use]
    pub fn new(id: &str, config: ContextConfig) -> Self {
        let storage = config.storage_state.clone().unwrap_or_default();
        Self {
            id: id.to_string(),
            config,
            state: ContextState::Creating,
            created_at: Instant::now(),
            pages: Arc::new(Mutex::new(Vec::new())),
            storage: Arc::new(Mutex::new(storage)),
            error_message: None,
        }
    }

    /// Mark context as ready
    pub fn ready(&mut self) {
        self.state = ContextState::Ready;
    }

    /// Mark context as in use
    pub fn acquire(&mut self) {
        self.state = ContextState::InUse;
    }

    /// Release context back to pool
    pub fn release(&mut self) {
        self.state = ContextState::Ready;
    }

    /// Close the context
    pub fn close(&mut self) {
        self.state = ContextState::Closed;
    }

    /// Set error state
    pub fn set_error(&mut self, message: &str) {
        self.state = ContextState::Error;
        self.error_message = Some(message.to_string());
    }

    /// Check if context is available
    #[must_use]
    pub const fn is_available(&self) -> bool {
        matches!(self.state, ContextState::Ready)
    }

    /// Check if context is in use
    #[must_use]
    pub const fn is_in_use(&self) -> bool {
        matches!(self.state, ContextState::InUse)
    }

    /// Check if context is closed
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        matches!(self.state, ContextState::Closed)
    }

    /// Get age in milliseconds
    #[must_use]
    pub fn age_ms(&self) -> u64 {
        self.created_at.elapsed().as_millis() as u64
    }

    /// Create a new page
    pub fn new_page(&self) -> String {
        let page_id = format!("{}_{}", self.id, uuid::Uuid::new_v4());
        if let Ok(mut pages) = self.pages.lock() {
            pages.push(page_id.clone());
        }
        page_id
    }

    /// Close a page
    pub fn close_page(&self, page_id: &str) {
        if let Ok(mut pages) = self.pages.lock() {
            pages.retain(|p| p != page_id);
        }
    }

    /// Get page count
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// Get storage state
    #[must_use]
    pub fn storage_state(&self) -> StorageState {
        self.storage.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Clear storage
    pub fn clear_storage(&self) {
        if let Ok(mut storage) = self.storage.lock() {
            storage.clear();
        }
    }

    /// Add cookie
    pub fn add_cookie(&self, cookie: Cookie) {
        if let Ok(mut storage) = self.storage.lock() {
            storage.cookies.push(cookie);
        }
    }

    /// Clear cookies
    pub fn clear_cookies(&self) {
        if let Ok(mut storage) = self.storage.lock() {
            storage.cookies.clear();
        }
    }
}

/// Context pool for managing multiple contexts
#[derive(Debug)]
pub struct ContextPool {
    /// All contexts
    contexts: Arc<Mutex<HashMap<String, BrowserContext>>>,
    /// Maximum number of contexts
    max_contexts: usize,
    /// Default configuration
    default_config: ContextConfig,
    /// Context counter
    counter: Arc<Mutex<u64>>,
}

impl Default for ContextPool {
    fn default() -> Self {
        Self::new(10)
    }
}

impl ContextPool {
    /// Create a new context pool
    #[must_use]
    pub fn new(max_contexts: usize) -> Self {
        Self {
            contexts: Arc::new(Mutex::new(HashMap::new())),
            max_contexts,
            default_config: ContextConfig::default(),
            counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Set default configuration
    #[must_use]
    pub fn with_default_config(mut self, config: ContextConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Create a new context
    pub fn create(&self, config: Option<ContextConfig>) -> ProbarResult<String> {
        let mut contexts = self.contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock contexts",
            ))
        })?;

        if contexts.len() >= self.max_contexts {
            return Err(ProbarError::AssertionError {
                message: format!("Maximum contexts ({}) reached", self.max_contexts),
            });
        }

        let id = {
            let mut counter = self.counter.lock().map_err(|_| {
                ProbarError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to lock counter",
                ))
            })?;
            *counter += 1;
            format!("ctx_{}", *counter)
        };

        let mut ctx_config = config.unwrap_or_else(|| self.default_config.clone());
        ctx_config.name = id.clone();

        let mut context = BrowserContext::new(&id, ctx_config);
        context.ready();

        contexts.insert(id.clone(), context);
        Ok(id)
    }

    /// Acquire an available context
    pub fn acquire(&self) -> ProbarResult<String> {
        let mut contexts = self.contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock contexts",
            ))
        })?;

        for (id, context) in contexts.iter_mut() {
            if context.is_available() {
                context.acquire();
                return Ok(id.clone());
            }
        }

        // No available context, try to create one
        drop(contexts);
        let id = self.create(None)?;

        let mut contexts = self.contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock contexts",
            ))
        })?;

        if let Some(context) = contexts.get_mut(&id) {
            context.acquire();
        }

        Ok(id)
    }

    /// Release a context back to the pool
    pub fn release(&self, context_id: &str) -> ProbarResult<()> {
        let mut contexts = self.contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock contexts",
            ))
        })?;

        if let Some(context) = contexts.get_mut(context_id) {
            context.release();
            Ok(())
        } else {
            Err(ProbarError::AssertionError {
                message: format!("Context {} not found", context_id),
            })
        }
    }

    /// Close a context
    pub fn close(&self, context_id: &str) -> ProbarResult<()> {
        let mut contexts = self.contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock contexts",
            ))
        })?;

        if let Some(context) = contexts.get_mut(context_id) {
            context.close();
            Ok(())
        } else {
            Err(ProbarError::AssertionError {
                message: format!("Context {} not found", context_id),
            })
        }
    }

    /// Remove a closed context
    pub fn remove(&self, context_id: &str) -> ProbarResult<()> {
        let mut contexts = self.contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock contexts",
            ))
        })?;

        contexts.remove(context_id);
        Ok(())
    }

    /// Get context count
    #[must_use]
    pub fn count(&self) -> usize {
        self.contexts.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Get available context count
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.contexts
            .lock()
            .map(|c| c.values().filter(|ctx| ctx.is_available()).count())
            .unwrap_or(0)
    }

    /// Get in-use context count
    #[must_use]
    pub fn in_use_count(&self) -> usize {
        self.contexts
            .lock()
            .map(|c| c.values().filter(|ctx| ctx.is_in_use()).count())
            .unwrap_or(0)
    }

    /// Close all contexts
    pub fn close_all(&self) {
        if let Ok(mut contexts) = self.contexts.lock() {
            for context in contexts.values_mut() {
                context.close();
            }
        }
    }

    /// Clear all contexts
    pub fn clear(&self) {
        if let Ok(mut contexts) = self.contexts.lock() {
            contexts.clear();
        }
    }

    /// Get context IDs
    #[must_use]
    pub fn context_ids(&self) -> Vec<String> {
        self.contexts
            .lock()
            .map(|c| c.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Context manager for test isolation
#[derive(Debug)]
pub struct ContextManager {
    /// Context pool
    pool: ContextPool,
    /// Active test contexts (test_id -> context_id)
    active_contexts: Arc<Mutex<HashMap<String, String>>>,
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextManager {
    /// Create a new context manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            pool: ContextPool::new(10),
            active_contexts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with custom pool size
    #[must_use]
    pub fn with_pool_size(pool_size: usize) -> Self {
        Self {
            pool: ContextPool::new(pool_size),
            active_contexts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a context for a test
    pub fn get_context(&self, test_id: &str) -> ProbarResult<String> {
        let mut active = self.active_contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock active contexts",
            ))
        })?;

        // Check if test already has a context
        if let Some(context_id) = active.get(test_id) {
            return Ok(context_id.clone());
        }

        // Acquire a new context
        let context_id = self.pool.acquire()?;
        let _ = active.insert(test_id.to_string(), context_id.clone());
        Ok(context_id)
    }

    /// Release a test's context
    pub fn release_context(&self, test_id: &str) -> ProbarResult<()> {
        let mut active = self.active_contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock active contexts",
            ))
        })?;

        if let Some(context_id) = active.remove(test_id) {
            self.pool.release(&context_id)?;
        }
        Ok(())
    }

    /// Create a new isolated context for a test
    pub fn create_isolated_context(
        &self,
        test_id: &str,
        config: ContextConfig,
    ) -> ProbarResult<String> {
        let context_id = self.pool.create(Some(config))?;

        let mut active = self.active_contexts.lock().map_err(|_| {
            ProbarError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to lock active contexts",
            ))
        })?;

        // Release any existing context
        if let Some(old_id) = active.get(test_id) {
            let _ = self.pool.release(old_id);
        }

        let _ = active.insert(test_id.to_string(), context_id.clone());
        Ok(context_id)
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> ContextPoolStats {
        ContextPoolStats {
            total: self.pool.count(),
            available: self.pool.available_count(),
            in_use: self.pool.in_use_count(),
            active_tests: self.active_contexts.lock().map(|a| a.len()).unwrap_or(0),
        }
    }

    /// Cleanup all resources
    pub fn cleanup(&self) {
        self.pool.close_all();
        self.pool.clear();
        if let Ok(mut active) = self.active_contexts.lock() {
            active.clear();
        }
    }
}

/// Statistics for context pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPoolStats {
    /// Total contexts
    pub total: usize,
    /// Available contexts
    pub available: usize,
    /// In-use contexts
    pub in_use: usize,
    /// Active test count
    pub active_tests: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod storage_state_tests {
        use super::*;

        #[test]
        fn test_new() {
            let state = StorageState::new();
            assert!(state.is_empty());
        }

        #[test]
        fn test_with_cookie() {
            let state =
                StorageState::new().with_cookie(Cookie::new("session", "abc123", "example.com"));
            assert_eq!(state.cookies.len(), 1);
        }

        #[test]
        fn test_with_local_storage() {
            let state =
                StorageState::new().with_local_storage("https://example.com", "key", "value");
            assert!(!state.local_storage.is_empty());
        }

        #[test]
        fn test_clear() {
            let mut state =
                StorageState::new().with_cookie(Cookie::new("session", "abc123", "example.com"));
            state.clear();
            assert!(state.is_empty());
        }
    }

    mod cookie_tests {
        use super::*;

        #[test]
        fn test_new() {
            let cookie = Cookie::new("name", "value", "example.com");
            assert_eq!(cookie.name, "name");
            assert_eq!(cookie.value, "value");
            assert_eq!(cookie.domain, "example.com");
            assert_eq!(cookie.path, "/");
        }

        #[test]
        fn test_with_path() {
            let cookie = Cookie::new("name", "value", "example.com").with_path("/api");
            assert_eq!(cookie.path, "/api");
        }

        #[test]
        fn test_secure_http_only() {
            let cookie = Cookie::new("name", "value", "example.com")
                .secure()
                .http_only();
            assert!(cookie.secure);
            assert!(cookie.http_only);
        }

        #[test]
        fn test_same_site() {
            let cookie =
                Cookie::new("name", "value", "example.com").with_same_site(SameSite::Strict);
            assert!(matches!(cookie.same_site, SameSite::Strict));
        }
    }

    mod context_config_tests {
        use super::*;

        #[test]
        fn test_new() {
            let config = ContextConfig::new("test");
            assert_eq!(config.name, "test");
            assert_eq!(config.viewport_width, 1280);
            assert_eq!(config.viewport_height, 720);
        }

        #[test]
        fn test_with_viewport() {
            let config = ContextConfig::new("test").with_viewport(1920, 1080);
            assert_eq!(config.viewport_width, 1920);
            assert_eq!(config.viewport_height, 1080);
        }

        #[test]
        fn test_mobile() {
            let config = ContextConfig::new("test").mobile();
            assert!(config.is_mobile);
            assert!(config.has_touch);
        }

        #[test]
        fn test_with_geolocation() {
            let config = ContextConfig::new("test").with_geolocation(37.7749, -122.4194);
            assert!(config.geolocation.is_some());
            let geo = config.geolocation.unwrap();
            assert!((geo.latitude - 37.7749).abs() < f64::EPSILON);
        }

        #[test]
        fn test_with_header() {
            let config = ContextConfig::new("test").with_header("Authorization", "Bearer token");
            assert_eq!(
                config.extra_headers.get("Authorization"),
                Some(&"Bearer token".to_string())
            );
        }

        #[test]
        fn test_offline() {
            let config = ContextConfig::new("test").offline();
            assert!(config.offline);
        }
    }

    mod browser_context_tests {
        use super::*;

        #[test]
        fn test_new() {
            let config = ContextConfig::new("test");
            let context = BrowserContext::new("ctx_1", config);
            assert_eq!(context.id, "ctx_1");
            assert!(matches!(context.state, ContextState::Creating));
        }

        #[test]
        fn test_state_transitions() {
            let config = ContextConfig::new("test");
            let mut context = BrowserContext::new("ctx_1", config);

            context.ready();
            assert!(context.is_available());

            context.acquire();
            assert!(context.is_in_use());

            context.release();
            assert!(context.is_available());

            context.close();
            assert!(context.is_closed());
        }

        #[test]
        fn test_error_state() {
            let config = ContextConfig::new("test");
            let mut context = BrowserContext::new("ctx_1", config);

            context.set_error("Connection failed");
            assert!(matches!(context.state, ContextState::Error));
            assert_eq!(context.error_message, Some("Connection failed".to_string()));
        }

        #[test]
        fn test_pages() {
            let config = ContextConfig::new("test");
            let context = BrowserContext::new("ctx_1", config);

            let page1 = context.new_page();
            let page2 = context.new_page();
            assert_eq!(context.page_count(), 2);

            context.close_page(&page1);
            assert_eq!(context.page_count(), 1);

            context.close_page(&page2);
            assert_eq!(context.page_count(), 0);
        }

        #[test]
        fn test_storage() {
            let config = ContextConfig::new("test");
            let context = BrowserContext::new("ctx_1", config);

            context.add_cookie(Cookie::new("session", "abc", "example.com"));
            let storage = context.storage_state();
            assert_eq!(storage.cookies.len(), 1);

            context.clear_cookies();
            let storage = context.storage_state();
            assert!(storage.cookies.is_empty());
        }
    }

    mod context_pool_tests {
        use super::*;

        #[test]
        fn test_new() {
            let pool = ContextPool::new(5);
            assert_eq!(pool.count(), 0);
        }

        #[test]
        fn test_create() {
            let pool = ContextPool::new(5);
            let id = pool.create(None).unwrap();
            assert!(!id.is_empty());
            assert_eq!(pool.count(), 1);
        }

        #[test]
        fn test_max_contexts() {
            let pool = ContextPool::new(2);
            pool.create(None).unwrap();
            pool.create(None).unwrap();

            let result = pool.create(None);
            assert!(result.is_err());
        }

        #[test]
        fn test_acquire_release() {
            let pool = ContextPool::new(5);
            let id = pool.acquire().unwrap();

            assert_eq!(pool.in_use_count(), 1);
            assert_eq!(pool.available_count(), 0);

            pool.release(&id).unwrap();
            assert_eq!(pool.in_use_count(), 0);
            assert_eq!(pool.available_count(), 1);
        }

        #[test]
        fn test_close() {
            let pool = ContextPool::new(5);
            let id = pool.create(None).unwrap();

            pool.close(&id).unwrap();
            assert_eq!(pool.available_count(), 0);
        }

        #[test]
        fn test_close_all() {
            let pool = ContextPool::new(5);
            pool.create(None).unwrap();
            pool.create(None).unwrap();

            pool.close_all();
            assert_eq!(pool.available_count(), 0);
        }

        #[test]
        fn test_clear() {
            let pool = ContextPool::new(5);
            pool.create(None).unwrap();
            pool.clear();
            assert_eq!(pool.count(), 0);
        }
    }

    mod context_manager_tests {
        use super::*;

        #[test]
        fn test_new() {
            let manager = ContextManager::new();
            let stats = manager.stats();
            assert_eq!(stats.total, 0);
        }

        #[test]
        fn test_get_context() {
            let manager = ContextManager::new();
            let ctx_id = manager.get_context("test_1").unwrap();
            assert!(!ctx_id.is_empty());

            // Same test gets same context
            let ctx_id2 = manager.get_context("test_1").unwrap();
            assert_eq!(ctx_id, ctx_id2);

            // Different test gets different context
            let ctx_id3 = manager.get_context("test_2").unwrap();
            assert_ne!(ctx_id, ctx_id3);
        }

        #[test]
        fn test_release_context() {
            let manager = ContextManager::new();
            let _ctx_id = manager.get_context("test_1").unwrap();

            manager.release_context("test_1").unwrap();
            let stats = manager.stats();
            assert_eq!(stats.available, 1);
            assert_eq!(stats.in_use, 0);
        }

        #[test]
        fn test_create_isolated_context() {
            let manager = ContextManager::new();
            let config = ContextConfig::new("isolated")
                .mobile()
                .with_viewport(375, 812);

            let ctx_id = manager.create_isolated_context("test_1", config).unwrap();
            assert!(!ctx_id.is_empty());
        }

        #[test]
        fn test_cleanup() {
            let manager = ContextManager::new();
            manager.get_context("test_1").unwrap();
            manager.get_context("test_2").unwrap();

            manager.cleanup();
            let stats = manager.stats();
            assert_eq!(stats.total, 0);
            assert_eq!(stats.active_tests, 0);
        }

        #[test]
        fn test_stats() {
            let manager = ContextManager::new();
            manager.get_context("test_1").unwrap();
            manager.get_context("test_2").unwrap();

            let stats = manager.stats();
            assert_eq!(stats.total, 2);
            assert_eq!(stats.in_use, 2);
            assert_eq!(stats.active_tests, 2);
        }
    }

    mod context_config_additional_tests {
        use super::*;

        #[test]
        fn test_with_device_scale() {
            let config = ContextConfig::new("test").with_device_scale(2.0);
            assert!((config.device_scale_factor - 2.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_with_user_agent() {
            let config = ContextConfig::new("test").with_user_agent("Custom UA");
            assert_eq!(config.user_agent, Some("Custom UA".to_string()));
        }

        #[test]
        fn test_with_locale() {
            let config = ContextConfig::new("test").with_locale("en-US");
            assert_eq!(config.locale, Some("en-US".to_string()));
        }

        #[test]
        fn test_with_timezone() {
            let config = ContextConfig::new("test").with_timezone("America/New_York");
            assert_eq!(config.timezone, Some("America/New_York".to_string()));
        }

        #[test]
        fn test_with_permission() {
            let config = ContextConfig::new("test")
                .with_permission("geolocation")
                .with_permission("notifications");
            assert_eq!(config.permissions.len(), 2);
            assert!(config.permissions.contains(&"geolocation".to_string()));
        }

        #[test]
        fn test_with_storage_state() {
            let storage =
                StorageState::new().with_cookie(Cookie::new("session", "abc", "example.com"));
            let config = ContextConfig::new("test").with_storage_state(storage);
            assert!(config.storage_state.is_some());
            assert_eq!(config.storage_state.unwrap().cookies.len(), 1);
        }

        #[test]
        fn test_with_video() {
            let config = ContextConfig::new("test").with_video();
            assert!(config.record_video);
        }

        #[test]
        fn test_with_har() {
            let config = ContextConfig::new("test").with_har();
            assert!(config.record_har);
        }

        #[test]
        fn test_ignore_https_errors() {
            let config = ContextConfig::new("test").ignore_https_errors();
            assert!(config.ignore_https_errors);
        }

        #[test]
        fn test_config_default() {
            let config = ContextConfig::default();
            assert!(config.name.is_empty());
            assert_eq!(config.viewport_width, 1280);
            assert!(!config.is_mobile);
            assert!(!config.offline);
        }
    }

    mod storage_state_additional_tests {
        use super::*;

        #[test]
        fn test_with_session_storage() {
            let state =
                StorageState::new().with_session_storage("https://example.com", "key", "value");
            assert!(!state.session_storage.is_empty());
            assert!(state.session_storage.contains_key("https://example.com"));
        }

        #[test]
        fn test_is_empty() {
            let state = StorageState::new();
            assert!(state.is_empty());

            let state_with_cookie =
                StorageState::new().with_cookie(Cookie::new("name", "value", "example.com"));
            assert!(!state_with_cookie.is_empty());
        }
    }

    mod cookie_additional_tests {
        use super::*;

        #[test]
        fn test_with_expires() {
            let cookie = Cookie::new("name", "value", "example.com").with_expires(1234567890);
            assert_eq!(cookie.expires, Some(1234567890));
        }

        #[test]
        fn test_cookie_debug() {
            let cookie = Cookie::new("name", "value", "example.com");
            let debug = format!("{:?}", cookie);
            assert!(debug.contains("Cookie"));
            assert!(debug.contains("name"));
        }

        #[test]
        fn test_same_site_variants() {
            let strict = SameSite::Strict;
            let lax = SameSite::Lax;
            let none = SameSite::None;

            assert!(matches!(strict, SameSite::Strict));
            assert!(matches!(lax, SameSite::Lax));
            assert!(matches!(none, SameSite::None));
        }

        #[test]
        fn test_same_site_debug() {
            assert!(format!("{:?}", SameSite::Strict).contains("Strict"));
            assert!(format!("{:?}", SameSite::Lax).contains("Lax"));
            assert!(format!("{:?}", SameSite::None).contains("None"));
        }
    }

    mod context_state_tests {
        use super::*;

        #[test]
        fn test_all_states() {
            let states = [
                ContextState::Creating,
                ContextState::Ready,
                ContextState::InUse,
                ContextState::Cleaning,
                ContextState::Closed,
                ContextState::Error,
            ];

            for state in &states {
                let debug = format!("{:?}", state);
                assert!(!debug.is_empty());
            }
        }

        #[test]
        fn test_state_clone() {
            let state = ContextState::Ready;
            let cloned = state;
            assert_eq!(state, cloned);
        }

        #[test]
        fn test_state_serialization() {
            let state = ContextState::Ready;
            let serialized = serde_json::to_string(&state).unwrap();
            let deserialized: ContextState = serde_json::from_str(&serialized).unwrap();
            assert_eq!(state, deserialized);
        }
    }

    mod browser_context_additional_tests {
        use super::*;

        #[test]
        fn test_with_storage_state_initialization() {
            let storage =
                StorageState::new().with_cookie(Cookie::new("init", "value", "example.com"));
            let config = ContextConfig::new("test").with_storage_state(storage);
            let context = BrowserContext::new("ctx_1", config);

            let stored = context.storage_state();
            assert_eq!(stored.cookies.len(), 1);
        }

        #[test]
        fn test_context_age() {
            let config = ContextConfig::new("test");
            let context = BrowserContext::new("ctx_1", config);

            // Context was just created, age should be small
            let age = context.age_ms();
            assert!(age < 1000); // Less than 1 second
        }

        #[test]
        fn test_clear_storage() {
            let storage =
                StorageState::new().with_cookie(Cookie::new("test", "value", "example.com"));
            let config = ContextConfig::new("test").with_storage_state(storage);
            let context = BrowserContext::new("ctx_1", config);

            assert!(!context.storage_state().is_empty());
            context.clear_storage();
            assert!(context.storage_state().is_empty());
        }

        #[test]
        fn test_debug() {
            let config = ContextConfig::new("test");
            let context = BrowserContext::new("ctx_1", config);
            let debug = format!("{:?}", context);
            assert!(debug.contains("BrowserContext"));
            assert!(debug.contains("ctx_1"));
        }
    }

    mod geolocation_tests {
        use super::*;

        #[test]
        fn test_geolocation_debug() {
            let geo = Geolocation {
                latitude: 37.7749,
                longitude: -122.4194,
                accuracy: 10.0,
            };
            let debug = format!("{:?}", geo);
            assert!(debug.contains("Geolocation"));
            assert!(debug.contains("37.7749"));
        }

        #[test]
        fn test_geolocation_clone() {
            let geo = Geolocation {
                latitude: 37.7749,
                longitude: -122.4194,
                accuracy: 10.0,
            };
            let cloned = geo;
            assert!((geo.latitude - cloned.latitude).abs() < f64::EPSILON);
        }
    }

    mod pool_stats_tests {
        use super::*;

        #[test]
        fn test_pool_stats_debug() {
            let manager = ContextManager::new();
            manager.get_context("test_1").unwrap();
            let stats = manager.stats();
            let debug = format!("{:?}", stats);
            assert!(debug.contains("total"));
        }
    }

    mod context_pool_error_tests {
        use super::*;

        #[test]
        fn test_release_unknown_context() {
            let pool = ContextPool::new(5);
            let result = pool.release("unknown_ctx");
            assert!(result.is_err());
        }

        #[test]
        fn test_close_unknown_context() {
            let pool = ContextPool::new(5);
            let result = pool.close("unknown_ctx");
            assert!(result.is_err());
        }

        #[test]
        fn test_remove_context() {
            let pool = ContextPool::new(5);
            let id = pool.create(None).unwrap();
            assert_eq!(pool.count(), 1);

            pool.remove(&id).unwrap();
            assert_eq!(pool.count(), 0);
        }

        #[test]
        fn test_with_default_config() {
            let config = ContextConfig::new("default").mobile();
            let pool = ContextPool::new(5).with_default_config(config);
            let _id = pool.create(None).unwrap();
            // Context should use mobile config
            assert_eq!(pool.count(), 1);
        }

        #[test]
        fn test_acquire_creates_new_context() {
            let pool = ContextPool::new(5);
            // First acquire should create a new context
            let id1 = pool.acquire().unwrap();
            assert_eq!(pool.count(), 1);
            assert_eq!(pool.in_use_count(), 1);

            // Second acquire should create another new context
            let id2 = pool.acquire().unwrap();
            assert_eq!(pool.count(), 2);
            assert_eq!(pool.in_use_count(), 2);

            assert_ne!(id1, id2);
        }

        #[test]
        fn test_acquire_reuses_released_context() {
            let pool = ContextPool::new(5);
            let id1 = pool.acquire().unwrap();
            pool.release(&id1).unwrap();

            // Next acquire should reuse the released context
            let id2 = pool.acquire().unwrap();
            assert_eq!(id1, id2);
            assert_eq!(pool.count(), 1);
        }

        #[test]
        fn test_pool_default() {
            let pool = ContextPool::default();
            assert_eq!(pool.count(), 0);
        }
    }

    mod context_manager_error_tests {
        use super::*;

        #[test]
        fn test_release_unknown_test_ok() {
            let manager = ContextManager::new();
            // release_context returns Ok even for unknown tests
            let result = manager.release_context("unknown_test");
            assert!(result.is_ok());
        }

        #[test]
        fn test_with_pool_size() {
            let manager = ContextManager::with_pool_size(3);
            let stats = manager.stats();
            assert_eq!(stats.total, 0);
        }

        #[test]
        fn test_manager_default() {
            let manager = ContextManager::default();
            assert_eq!(manager.stats().total, 0);
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Browser Context Tests (Feature F P0)
    // =========================================================================

    mod h0_context_state_tests {
        use super::*;

        #[test]
        fn h0_ctx_01_state_creating() {
            assert_eq!(ContextState::Creating, ContextState::Creating);
        }

        #[test]
        fn h0_ctx_02_state_ready() {
            assert_eq!(ContextState::Ready, ContextState::Ready);
        }

        #[test]
        fn h0_ctx_03_state_in_use() {
            assert_eq!(ContextState::InUse, ContextState::InUse);
        }

        #[test]
        fn h0_ctx_04_state_cleaning() {
            assert_eq!(ContextState::Cleaning, ContextState::Cleaning);
        }

        #[test]
        fn h0_ctx_05_state_closed() {
            assert_eq!(ContextState::Closed, ContextState::Closed);
        }

        #[test]
        fn h0_ctx_06_state_error() {
            assert_eq!(ContextState::Error, ContextState::Error);
        }

        #[test]
        fn h0_ctx_07_state_inequality() {
            assert_ne!(ContextState::Ready, ContextState::Closed);
        }

        #[test]
        fn h0_ctx_08_state_debug() {
            let debug = format!("{:?}", ContextState::Ready);
            assert!(debug.contains("Ready"));
        }

        #[test]
        fn h0_ctx_09_state_clone() {
            let state = ContextState::InUse;
            let cloned = state;
            assert_eq!(state, cloned);
        }

        #[test]
        fn h0_ctx_10_state_serialize() {
            let state = ContextState::Ready;
            let json = serde_json::to_string(&state).unwrap();
            assert!(!json.is_empty());
        }
    }

    mod h0_same_site_tests {
        use super::*;

        #[test]
        fn h0_ctx_11_same_site_strict() {
            assert_eq!(SameSite::Strict, SameSite::Strict);
        }

        #[test]
        fn h0_ctx_12_same_site_lax() {
            assert_eq!(SameSite::Lax, SameSite::Lax);
        }

        #[test]
        fn h0_ctx_13_same_site_none() {
            assert_eq!(SameSite::None, SameSite::None);
        }

        #[test]
        fn h0_ctx_14_same_site_inequality() {
            assert_ne!(SameSite::Strict, SameSite::Lax);
        }

        #[test]
        fn h0_ctx_15_same_site_debug() {
            let debug = format!("{:?}", SameSite::Strict);
            assert!(debug.contains("Strict"));
        }
    }

    mod h0_cookie_tests {
        use super::*;

        #[test]
        fn h0_ctx_16_cookie_name() {
            let cookie = Cookie::new("session", "abc", "example.com");
            assert_eq!(cookie.name, "session");
        }

        #[test]
        fn h0_ctx_17_cookie_value() {
            let cookie = Cookie::new("session", "abc", "example.com");
            assert_eq!(cookie.value, "abc");
        }

        #[test]
        fn h0_ctx_18_cookie_domain() {
            let cookie = Cookie::new("session", "abc", "example.com");
            assert_eq!(cookie.domain, "example.com");
        }

        #[test]
        fn h0_ctx_19_cookie_default_path() {
            let cookie = Cookie::new("session", "abc", "example.com");
            assert_eq!(cookie.path, "/");
        }

        #[test]
        fn h0_ctx_20_cookie_with_path() {
            let cookie = Cookie::new("session", "abc", "example.com").with_path("/api");
            assert_eq!(cookie.path, "/api");
        }

        #[test]
        fn h0_ctx_21_cookie_expires() {
            let cookie = Cookie::new("session", "abc", "example.com").with_expires(1234567890);
            assert_eq!(cookie.expires, Some(1234567890));
        }

        #[test]
        fn h0_ctx_22_cookie_http_only() {
            let cookie = Cookie::new("session", "abc", "example.com").http_only();
            assert!(cookie.http_only);
        }

        #[test]
        fn h0_ctx_23_cookie_secure() {
            let cookie = Cookie::new("session", "abc", "example.com").secure();
            assert!(cookie.secure);
        }

        #[test]
        fn h0_ctx_24_cookie_same_site() {
            let cookie =
                Cookie::new("session", "abc", "example.com").with_same_site(SameSite::Strict);
            assert_eq!(cookie.same_site, SameSite::Strict);
        }

        #[test]
        fn h0_ctx_25_cookie_default_same_site() {
            let cookie = Cookie::new("session", "abc", "example.com");
            assert_eq!(cookie.same_site, SameSite::Lax);
        }
    }

    mod h0_storage_state_tests {
        use super::*;

        #[test]
        fn h0_ctx_26_storage_new_empty() {
            let state = StorageState::new();
            assert!(state.is_empty());
        }

        #[test]
        fn h0_ctx_27_storage_with_cookie() {
            let state = StorageState::new().with_cookie(Cookie::new("test", "val", "example.com"));
            assert_eq!(state.cookies.len(), 1);
        }

        #[test]
        fn h0_ctx_28_storage_with_local() {
            let state = StorageState::new().with_local_storage("https://ex.com", "key", "value");
            assert!(!state.local_storage.is_empty());
        }

        #[test]
        fn h0_ctx_29_storage_with_session() {
            let state = StorageState::new().with_session_storage("https://ex.com", "key", "value");
            assert!(!state.session_storage.is_empty());
        }

        #[test]
        fn h0_ctx_30_storage_clear() {
            let mut state = StorageState::new().with_cookie(Cookie::new("test", "v", "ex.com"));
            state.clear();
            assert!(state.is_empty());
        }
    }

    mod h0_context_config_tests {
        use super::*;

        #[test]
        fn h0_ctx_31_config_new_name() {
            let config = ContextConfig::new("test");
            assert_eq!(config.name, "test");
        }

        #[test]
        fn h0_ctx_32_config_default_viewport() {
            let config = ContextConfig::default();
            assert_eq!(config.viewport_width, 1280);
            assert_eq!(config.viewport_height, 720);
        }

        #[test]
        fn h0_ctx_33_config_with_viewport() {
            let config = ContextConfig::new("test").with_viewport(1920, 1080);
            assert_eq!(config.viewport_width, 1920);
        }

        #[test]
        fn h0_ctx_34_config_mobile() {
            let config = ContextConfig::new("test").mobile();
            assert!(config.is_mobile);
            assert!(config.has_touch);
        }

        #[test]
        fn h0_ctx_35_config_offline() {
            let config = ContextConfig::new("test").offline();
            assert!(config.offline);
        }

        #[test]
        fn h0_ctx_36_config_with_geolocation() {
            let config = ContextConfig::new("test").with_geolocation(40.7, -74.0);
            assert!(config.geolocation.is_some());
        }

        #[test]
        fn h0_ctx_37_config_with_header() {
            let config = ContextConfig::new("test").with_header("X-Test", "value");
            assert!(config.extra_headers.contains_key("X-Test"));
        }

        #[test]
        fn h0_ctx_38_config_with_video() {
            let config = ContextConfig::new("test").with_video();
            assert!(config.record_video);
        }

        #[test]
        fn h0_ctx_39_config_with_har() {
            let config = ContextConfig::new("test").with_har();
            assert!(config.record_har);
        }

        #[test]
        fn h0_ctx_40_config_ignore_https() {
            let config = ContextConfig::new("test").ignore_https_errors();
            assert!(config.ignore_https_errors);
        }
    }

    mod h0_browser_context_tests {
        use super::*;

        #[test]
        fn h0_ctx_41_context_new_id() {
            let ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            assert_eq!(ctx.id, "ctx_1");
        }

        #[test]
        fn h0_ctx_42_context_initial_state() {
            let ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            assert_eq!(ctx.state, ContextState::Creating);
        }

        #[test]
        fn h0_ctx_43_context_ready() {
            let mut ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            ctx.ready();
            assert!(ctx.is_available());
        }

        #[test]
        fn h0_ctx_44_context_acquire() {
            let mut ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            ctx.ready();
            ctx.acquire();
            assert!(ctx.is_in_use());
        }

        #[test]
        fn h0_ctx_45_context_release() {
            let mut ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            ctx.ready();
            ctx.acquire();
            ctx.release();
            assert!(ctx.is_available());
        }

        #[test]
        fn h0_ctx_46_context_close() {
            let mut ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            ctx.close();
            assert!(ctx.is_closed());
        }

        #[test]
        fn h0_ctx_47_context_error() {
            let mut ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            ctx.set_error("Failed");
            assert_eq!(ctx.state, ContextState::Error);
        }

        #[test]
        fn h0_ctx_48_context_page_count() {
            let ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            assert_eq!(ctx.page_count(), 0);
        }

        #[test]
        fn h0_ctx_49_context_new_page() {
            let ctx = BrowserContext::new("ctx_1", ContextConfig::default());
            let page_id = ctx.new_page();
            assert!(!page_id.is_empty());
            assert_eq!(ctx.page_count(), 1);
        }

        #[test]
        fn h0_ctx_50_context_pool_new() {
            let pool = ContextPool::new(10);
            assert_eq!(pool.count(), 0);
        }
    }
}
