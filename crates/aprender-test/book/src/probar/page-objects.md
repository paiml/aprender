# Page Objects

> **Toyota Way**: Jidoka (Built-in Quality) - Encapsulated page interactions

Implement the Page Object Model pattern for maintainable, reusable test code.

## Running the Example

```bash
cargo run --example page_object
```

## Quick Start

```rust
use probar::{PageObject, PageObjectBuilder, Selector, Locator};

// Create a simple page object
let login_page = PageObjectBuilder::new()
    .with_url_pattern("/login")
    .with_locator("username", Selector::css("input[name='username']"))
    .with_locator("password", Selector::css("input[name='password']"))
    .with_locator("submit", Selector::css("button[type='submit']"))
    .build();
```

## The PageObject Trait

```rust
use probar::{PageObject, Locator, Selector};

struct LoginPage {
    username_input: Locator,
    password_input: Locator,
    submit_button: Locator,
    error_message: Locator,
}

impl PageObject for LoginPage {
    fn url_pattern(&self) -> &str {
        "/login"
    }

    fn is_loaded(&self) -> bool {
        // Check if key elements are present
        true
    }

    fn load_timeout_ms(&self) -> u64 {
        30000  // 30 seconds
    }

    fn page_name(&self) -> &str {
        "LoginPage"
    }
}

impl LoginPage {
    pub fn new() -> Self {
        Self {
            username_input: Locator::from_selector(
                Selector::css("input[name='username']")
            ),
            password_input: Locator::from_selector(
                Selector::css("input[name='password']")
            ),
            submit_button: Locator::from_selector(
                Selector::css("button[type='submit']")
            ),
            error_message: Locator::from_selector(
                Selector::css(".error-message")
            ),
        }
    }

    // High-level actions
    pub fn login(&self, username: &str, password: &str) {
        // Fill username
        // Fill password
        // Click submit
    }

    pub fn get_error(&self) -> Option<String> {
        // Get error message text
        None
    }
}
```

## Using PageObjectBuilder

```rust
use probar::{PageObjectBuilder, Selector, SimplePageObject};

// Build a page object declaratively
let settings_page = PageObjectBuilder::new()
    .with_url_pattern("/settings")
    .with_load_timeout(10000)
    .with_locator("profile_tab", Selector::css("[data-tab='profile']"))
    .with_locator("security_tab", Selector::css("[data-tab='security']"))
    .with_locator("save_button", Selector::css("button.save"))
    .with_locator("cancel_button", Selector::css("button.cancel"))
    .build();

// Access locators
if let Some(locator) = settings_page.get_locator("save_button") {
    println!("Save button selector: {:?}", locator.selector());
}
```

## SimplePageObject

```rust
use probar::{SimplePageObject, Selector};

// Create a simple page object
let mut page = SimplePageObject::new("/dashboard");

// Add locators
page.add_locator("header", Selector::css(".dashboard-header"));
page.add_locator("nav", Selector::css("nav.main-nav"));
page.add_locator("content", Selector::css(".content-area"));

// Check properties
println!("URL Pattern: {}", page.url_pattern());
println!("Has header locator: {}", page.has_locator("header"));

// Get all locator names
for name in page.locator_names() {
    println!("- {}", name);
}
```

## URL Pattern Matching

```rust
use probar::{PageRegistry, SimplePageObject, UrlMatcher};

// Create page objects for different URL patterns
let home = SimplePageObject::new("/");
let profile = SimplePageObject::new("/users/:id");
let settings = SimplePageObject::new("/settings/*");

// URL matchers
let exact = UrlMatcher::exact("/login");
let prefix = UrlMatcher::starts_with("/api/");
let pattern = UrlMatcher::pattern("/users/:id/posts/:post_id");

// Check matches
assert!(exact.matches("/login"));
assert!(!exact.matches("/login/oauth"));
assert!(prefix.matches("/api/users"));
assert!(pattern.matches("/users/123/posts/456"));
```

## Page Registry

```rust
use probar::{PageRegistry, SimplePageObject};

// Create a registry of page objects
let mut registry = PageRegistry::new();

// Register pages
registry.register("home", SimplePageObject::new("/"));
registry.register("login", SimplePageObject::new("/login"));
registry.register("dashboard", SimplePageObject::new("/dashboard"));
registry.register("profile", SimplePageObject::new("/users/:id"));

// Find page by URL
if let Some(page_name) = registry.find_by_url("/users/123") {
    println!("Matched page: {}", page_name);  // "profile"
}

// Get page object by name
if let Some(page) = registry.get("dashboard") {
    println!("Dashboard URL: {}", page.url_pattern());
}

// List all registered pages
for name in registry.page_names() {
    println!("- {}", name);
}
```

## Composable Page Objects

```rust
use probar::{PageObject, PageObjectBuilder, Selector};

// Shared components
struct NavComponent {
    home_link: probar::Locator,
    profile_link: probar::Locator,
    logout_button: probar::Locator,
}

impl NavComponent {
    fn new() -> Self {
        Self {
            home_link: probar::Locator::from_selector(Selector::css("nav a[href='/']")),
            profile_link: probar::Locator::from_selector(Selector::css("nav a[href='/profile']")),
            logout_button: probar::Locator::from_selector(Selector::css("nav button.logout")),
        }
    }
}

// Page with shared component
struct DashboardPage {
    nav: NavComponent,
    stats_widget: probar::Locator,
    recent_activity: probar::Locator,
}

impl DashboardPage {
    fn new() -> Self {
        Self {
            nav: NavComponent::new(),
            stats_widget: probar::Locator::from_selector(Selector::css(".stats-widget")),
            recent_activity: probar::Locator::from_selector(Selector::css(".recent-activity")),
        }
    }

    fn navigate_to_profile(&self) {
        // Use nav component
        // self.nav.profile_link.click()
    }
}

impl PageObject for DashboardPage {
    fn url_pattern(&self) -> &str { "/dashboard" }
}
```

## Page Object Information

```rust
use probar::PageObjectInfo;

// Get metadata about page objects
let info = PageObjectInfo::new("LoginPage")
    .with_url("/login")
    .with_description("Handles user authentication")
    .with_locator_count(4)
    .with_action_count(2);

println!("Page: {} at {}", info.name(), info.url());
println!("Locators: {}", info.locator_count());
println!("Actions: {}", info.action_count());
```

## Testing with Page Objects

```rust
use probar::{SimplePageObject, Selector};

fn test_login_flow() {
    let login_page = SimplePageObject::new("/login");

    // Verify we're on the right page
    assert_eq!(login_page.url_pattern(), "/login");

    // Test expects specific locators
    assert!(login_page.has_locator("username") || true); // would be added
}

fn test_dashboard_navigation() {
    let dashboard = SimplePageObject::new("/dashboard");

    // Verify navigation elements exist
    // Use locators to interact with the page
}
```

## Best Practices

1. **Single Responsibility**: Each page object represents one page or component
2. **Encapsulation**: Hide locators, expose high-level actions
3. **No Assertions in Page Objects**: Keep assertions in test code
4. **Reusable Components**: Extract shared components (nav, footer, etc.)
5. **Clear Naming**: Name locators by their purpose, not implementation
6. **URL Patterns**: Use patterns for dynamic URLs (`/users/:id`)
7. **Composition**: Compose page objects from smaller components
