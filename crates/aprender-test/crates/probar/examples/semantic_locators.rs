//! Semantic Locators Demo - Playwright-style Accessible Element Selection
//!
//! Demonstrates PMAT-001 semantic locators for accessible testing:
//! - Role selectors (ARIA roles)
//! - Label selectors (form labels)
//! - Placeholder selectors (input placeholders)
//! - Alt text selectors (image alt attributes)
//!
//! # Running
//!
//! ```bash
//! cargo run --example semantic_locators -p probar
//! ```
//!
//! # Playwright Parity
//!
//! These locators match Playwright's semantic selectors:
//! - `page.getByRole('button')` -> `Locator::by_role("button")`
//! - `page.getByLabel('Username')` -> `Locator::by_label("Username")`
//! - `page.getByPlaceholder('Email')` -> `Locator::by_placeholder("Email")`
//! - `page.getByAltText('Logo')` -> `Locator::by_alt_text("Logo")`

#![allow(clippy::uninlined_format_args)]

use jugar_probar::{Locator, Selector};

fn main() {
    println!("=== Probar Semantic Locators Demo (PMAT-001) ===\n");

    // Demo 1: Role selectors
    demo_role_selectors();

    // Demo 2: Label selectors
    demo_label_selectors();

    // Demo 3: Placeholder selectors
    demo_placeholder_selectors();

    // Demo 4: Alt text selectors
    demo_alt_text_selectors();

    // Demo 5: Static locator constructors
    demo_static_constructors();

    println!("\n=== Semantic Locators Demo Complete ===");
}

fn demo_role_selectors() {
    println!("--- Demo 1: Role Selectors (ARIA Roles) ---\n");

    // Basic role selector
    let button = Selector::role("button");
    println!("Role selector: {:?}", button);
    println!("  Query: {}", button.to_query());
    println!("  Count Query: {}", button.to_count_query());

    // Role with name filter (like Playwright's { name: 'Submit' })
    let submit_btn = Selector::role_with_name("button", "Submit");
    println!("\nRole with name: {:?}", submit_btn);
    println!("  Query: {}", submit_btn.to_query());

    // Common ARIA roles for testing
    println!("\nCommon ARIA roles for game testing:");
    println!("  - button: Interactive buttons");
    println!("  - link: Clickable links");
    println!("  - textbox: Input fields");
    println!("  - checkbox: Toggle controls");
    println!("  - slider: Range inputs");
    println!("  - progressbar: Loading indicators");
    println!("  - alert: Important messages");
    println!("  - dialog: Modal windows");
    println!("  - menu/menuitem: Game menus");

    println!();
}

fn demo_label_selectors() {
    println!("--- Demo 2: Label Selectors (Form Labels) ---\n");

    // Label selector matches form elements by their label text
    let username = Selector::label("Username");
    println!("Label selector: {:?}", username);
    println!("  Query: {}", username.to_query());
    println!("  Count Query: {}", username.to_count_query());

    // How label association works:
    println!("\nLabel association methods:");
    println!("  1. <label for='id'>Text</label><input id='id'>");
    println!("  2. <label>Text<input></label> (nested)");

    // Common label patterns
    let email = Selector::label("Email");
    let password = Selector::label("Password");
    let remember = Selector::label("Remember me");

    println!("\nCommon form labels:");
    println!("  Email: {}", email.to_query());
    println!("  Password: {}", password.to_query());
    println!("  Remember: {}", remember.to_query());

    println!();
}

fn demo_placeholder_selectors() {
    println!("--- Demo 3: Placeholder Selectors ---\n");

    // Placeholder selector for inputs without visible labels
    let search = Selector::placeholder("Search...");
    println!("Placeholder selector: {:?}", search);
    println!("  Query: {}", search.to_query());
    println!("  Count Query: {}", search.to_count_query());

    // Useful for game search/filter inputs
    let filter_items = Selector::placeholder("Filter items");
    let enter_name = Selector::placeholder("Enter player name");

    println!("\nGame UI placeholders:");
    println!("  Filter: {}", filter_items.to_query());
    println!("  Name: {}", enter_name.to_query());

    println!();
}

fn demo_alt_text_selectors() {
    println!("--- Demo 4: Alt Text Selectors (Images) ---\n");

    // Alt text selector for images
    let logo = Selector::alt_text("Company Logo");
    println!("Alt text selector: {:?}", logo);
    println!("  Query: {}", logo.to_query());
    println!("  Count Query: {}", logo.to_count_query());

    // Game-specific images
    let player_avatar = Selector::alt_text("Player Avatar");
    let enemy_sprite = Selector::alt_text("Enemy");
    let power_up = Selector::alt_text("Power Up");

    println!("\nGame image alt texts:");
    println!("  Avatar: {}", player_avatar.to_query());
    println!("  Enemy: {}", enemy_sprite.to_query());
    println!("  Power-up: {}", power_up.to_query());

    println!();
}

fn demo_static_constructors() {
    println!("--- Demo 5: Static Locator Constructors ---\n");

    // Fluent API with static constructors
    let btn = Locator::by_role("button");
    let label = Locator::by_label("Username");
    let placeholder = Locator::by_placeholder("Search");
    let alt = Locator::by_alt_text("Logo");
    let test_id = Locator::by_test_id("submit-btn");
    let text = Locator::by_text("Click me");

    println!("Locator::by_role(\"button\"): {:?}", btn.selector());
    println!("Locator::by_label(\"Username\"): {:?}", label.selector());
    println!(
        "Locator::by_placeholder(\"Search\"): {:?}",
        placeholder.selector()
    );
    println!("Locator::by_alt_text(\"Logo\"): {:?}", alt.selector());
    println!(
        "Locator::by_test_id(\"submit-btn\"): {:?}",
        test_id.selector()
    );
    println!("Locator::by_text(\"Click me\"): {:?}", text.selector());

    // Chaining with role and name
    let submit = Locator::by_role_with_name("button", "Submit");
    println!("\nLocator::by_role_with_name(\"button\", \"Submit\"):");
    println!("  Selector: {:?}", submit.selector());

    // Combined with options
    let custom = Locator::by_role("textbox")
        .with_timeout(std::time::Duration::from_secs(5))
        .with_strict(true);
    println!("\nWith custom options:");
    println!("  Timeout: {:?}", custom.options().timeout);
    println!("  Strict: {}", custom.options().strict);

    println!();
}
