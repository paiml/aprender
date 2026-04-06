//! Example: Page Object Model (Feature 19)
//!
//! Demonstrates: Page Object pattern for maintainable tests
//!
//! Run with: `cargo run --example page_object`
//!
//! Toyota Way: Jidoka (Built-in Quality) - Encapsulated page interactions

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== Page Object Model Example ===\n");

    // 1. Create a simple page object
    println!("1. Creating page object using builder...");
    let login_page = PageObjectBuilder::new()
        .with_url_pattern("/login")
        .with_locator("username", Selector::Css("#username".into()))
        .with_locator("password", Selector::Css("#password".into()))
        .with_locator("submit", Selector::Css("button[type='submit']".into()))
        .with_locator("error_message", Selector::Css(".error-message".into()))
        .build();

    println!("   Page: LoginPage");
    println!("   URL pattern: /login");
    println!("   Locators: 4");

    // 2. List locators
    println!("\n2. Page locators...");
    for name in ["username", "password", "submit", "error_message"] {
        if let Some(loc) = login_page.locator(name) {
            println!("   {} -> {:?}", name, loc.selector());
        }
    }

    // 3. Get specific locator
    println!("\n3. Getting specific locators...");
    if let Some(loc) = login_page.locator("username") {
        println!("   Username field: {:?}", loc.selector());
    }
    if let Some(loc) = login_page.locator("submit") {
        println!("   Submit button: {:?}", loc.selector());
    }

    // 4. Create another page object
    println!("\n4. Creating dashboard page...");
    let _dashboard_page = PageObjectBuilder::new()
        .with_url_pattern("/dashboard")
        .with_locator("user_menu", Selector::Css(".user-menu".into()))
        .with_locator("logout_btn", Selector::Text("Logout".into()))
        .with_locator("welcome_msg", Selector::Css(".welcome-message".into()))
        .with_locator("nav_items", Selector::Css("nav li".into()))
        .build();

    println!("   Page: DashboardPage");
    println!("   Locators: 4");

    // 5. Selector types demonstration
    println!("\n5. Selector types...");
    let selectors = [
        ("CSS", Selector::Css(".class-name".into())),
        ("XPath", Selector::XPath("//div[@class='test']".into())),
        ("Text", Selector::Text("Click me".into())),
        ("Test ID", Selector::TestId("submit-btn".into())),
        ("Entity", Selector::Entity("player".into())),
    ];

    for (name, sel) in &selectors {
        println!("   {}: {:?}", name, sel);
    }

    // 6. Using Locator with selectors
    println!("\n6. Using Locator struct...");
    let button_locator = Locator::new("button.primary").with_text("Submit");

    println!("   Locator created with text filter");
    println!("   Selector: {:?}", button_locator.selector());

    // 7. Locator options
    println!("\n7. Locator options...");
    let configured_locator = Locator::new("input.search")
        .with_timeout(std::time::Duration::from_secs(5))
        .with_strict(true)
        .with_visible(true);

    println!("   Options: {:?}", configured_locator.options());

    // 8. Page object best practices
    println!("\n8. Page Object best practices...");
    println!("   - One page object per page/component");
    println!("   - Locators define element locations");
    println!("   - Methods perform actions (login, submit, etc.)");
    println!("   - Keep selectors private, expose actions");
    println!("   - Use meaningful locator names");
    println!("   - Prefer data-testid for stability");

    // 9. Example usage pattern
    println!("\n9. Example test pattern...");
    println!("   async fn test_login(page: &Page) {{");
    println!("       let login = LoginPage::new(page);");
    println!("       login.navigate().await?;");
    println!("       login.enter_username(\"test@example.com\").await?;");
    println!("       login.enter_password(\"password123\").await?;");
    println!("       login.submit().await?;");
    println!("       assert!(login.is_logged_in().await?);");
    println!("   }}");

    // 10. Summary
    println!("\n10. Summary...");
    println!("   Page objects created: 2");
    println!("   Total locators: 8");
    println!("   Selector types: 5");
    println!("   Benefits: Maintainability, Reusability, Readability");

    println!("\n✅ Page Object Model example completed!");
    Ok(())
}
