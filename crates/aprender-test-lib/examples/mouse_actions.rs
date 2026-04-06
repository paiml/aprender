//! Mouse Actions Demo - Playwright-style Input Actions
//!
//! Demonstrates PMAT-003 mouse and input actions:
//! - right_click() for context menus
//! - hover() for tooltips/hover states
//! - focus()/blur() for form elements
//! - check()/uncheck() for checkboxes
//! - click_with_options() for custom clicks
//!
//! # Running
//!
//! ```bash
//! cargo run --example mouse_actions -p probar
//! ```
//!

//! # Playwright Parity
//!
//! These actions match Playwright's action API:
//! - `locator.click({ button: 'right' })` -> `locator.right_click()`
//! - `locator.hover()` -> `locator.hover()`
//! - `locator.focus()` -> `locator.focus()`
//! - `locator.check()` -> `locator.check()`

#![allow(clippy::uninlined_format_args, clippy::expect_used)]

use jugar_probar::prelude::*;

fn main() {
    println!("=== Probar Mouse Actions Demo (PMAT-003) ===\n");

    // Demo 1: Right-click
    demo_right_click();

    // Demo 2: Hover
    demo_hover();

    // Demo 3: Focus and Blur
    demo_focus_blur();

    // Demo 4: Check and Uncheck
    demo_check_uncheck();

    // Demo 5: Click with options
    demo_click_options();

    // Demo 6: Scroll into view
    demo_scroll_into_view();

    println!("\n=== Mouse Actions Demo Complete ===");
}

fn demo_right_click() {
    println!("--- Demo 1: Right-Click (Context Menu) ---\n");

    let locator = Locator::new(".game-entity");
    let action = locator.right_click().expect("Should create action");

    println!("Locator::new(\".game-entity\").right_click():");
    println!("  Action type: {:?}", std::mem::discriminant(&action));

    match &action {
        LocatorAction::RightClick { locator } => {
            println!("  Target selector: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Use case: Context menu in game
    println!("\nUse cases:");
    println!("  - Open context menu on game objects");
    println!("  - Show inventory item options");
    println!("  - Display entity properties panel");

    println!();
}

fn demo_hover() {
    println!("--- Demo 2: Hover (Tooltip/Highlight) ---\n");

    let locator = Locator::new(".inventory-item");
    let action = locator.hover().expect("Should create action");

    println!("Locator::new(\".inventory-item\").hover():");
    println!("  Action type: {:?}", std::mem::discriminant(&action));

    match &action {
        LocatorAction::Hover { locator } => {
            println!("  Target selector: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Use case: Hover effects in game UI
    println!("\nUse cases:");
    println!("  - Show item tooltips");
    println!("  - Highlight menu options");
    println!("  - Preview card details");
    println!("  - Display stat information");

    println!();
}

fn demo_focus_blur() {
    println!("--- Demo 3: Focus and Blur ---\n");

    let input = Locator::new("input#player-name");

    // Focus
    let focus_action = input.focus().expect("Should create action");
    println!("Locator::new(\"input#player-name\").focus():");
    match &focus_action {
        LocatorAction::Focus { locator } => {
            println!("  Target: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Blur
    let blur_action = input.blur().expect("Should create action");
    println!("\nLocator::new(\"input#player-name\").blur():");
    match &blur_action {
        LocatorAction::Blur { locator } => {
            println!("  Target: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Use cases
    println!("\nUse cases:");
    println!("  - Test input validation on blur");
    println!("  - Verify focus indicators (accessibility)");
    println!("  - Test form field activation");
    println!("  - Verify keyboard navigation");

    println!();
}

fn demo_check_uncheck() {
    println!("--- Demo 4: Check and Uncheck ---\n");

    let checkbox = Locator::new("input[type='checkbox']#remember-me");

    // Check
    let check_action = checkbox.check().expect("Should create action");
    println!("Locator::new(\"input[type='checkbox']\").check():");
    match &check_action {
        LocatorAction::Check { locator } => {
            println!("  Target: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Uncheck
    let uncheck_action = checkbox.uncheck().expect("Should create action");
    println!("\nLocator::new(\"input[type='checkbox']\").uncheck():");
    match &uncheck_action {
        LocatorAction::Uncheck { locator } => {
            println!("  Target: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Radio buttons
    let radio = Locator::new("input[type='radio']#difficulty-hard");
    let radio_check = radio.check().expect("Should create action");
    println!("\nRadio button check:");
    println!("  Action type: {:?}", std::mem::discriminant(&radio_check));

    // Use cases
    println!("\nUse cases:");
    println!("  - Toggle game settings (sound, music)");
    println!("  - Select difficulty levels");
    println!("  - Enable/disable features");
    println!("  - Accept terms and conditions");

    println!();
}

fn demo_click_options() {
    println!("--- Demo 5: Click with Options ---\n");

    let button = Locator::new("button.action");

    // Default options
    let default_opts = ClickOptions::new();
    println!("ClickOptions::new() defaults:");
    println!("  Button: {:?}", default_opts.button);
    println!("  Click count: {}", default_opts.click_count);
    println!("  Position: {:?}", default_opts.position);
    println!("  Modifiers: {:?}", default_opts.modifiers);

    // Right button click
    let right_opts = ClickOptions::new().button(MouseButton::Right);
    println!("\nClickOptions::new().button(MouseButton::Right):");
    println!("  Button: {:?}", right_opts.button);

    // Double click
    let double_opts = ClickOptions::new().click_count(2);
    println!("\nClickOptions::new().click_count(2):");
    println!("  Click count: {}", double_opts.click_count);

    // Click at specific position
    let pos_opts = ClickOptions::new().position(Point::new(10.0, 20.0));
    println!("\nClickOptions::new().position(Point::new(10.0, 20.0)):");
    println!("  Position: {:?}", pos_opts.position);

    // Click with modifiers
    let mod_opts = ClickOptions::new()
        .modifier(KeyModifier::Shift)
        .modifier(KeyModifier::Control);
    println!("\nClickOptions with Shift+Ctrl:");
    println!("  Modifiers: {:?}", mod_opts.modifiers);

    // Combined options
    let complex_opts = ClickOptions::new()
        .button(MouseButton::Left)
        .click_count(1)
        .position(Point::new(50.0, 50.0))
        .modifier(KeyModifier::Alt);

    let action = button
        .click_with_options(complex_opts)
        .expect("Should create action");
    println!("\nComplex click action:");
    match &action {
        LocatorAction::ClickWithOptions { options, .. } => {
            println!("  Button: {:?}", options.button);
            println!("  Position: {:?}", options.position);
            println!("  Modifiers: {:?}", options.modifiers);
        }
        _ => println!("  Unexpected action type"),
    }

    // MouseButton variants
    println!("\nMouseButton variants:");
    println!("  {:?} (default)", MouseButton::Left);
    println!("  {:?}", MouseButton::Right);
    println!("  {:?}", MouseButton::Middle);

    // KeyModifier variants
    println!("\nKeyModifier variants:");
    println!("  {:?}", KeyModifier::Alt);
    println!("  {:?}", KeyModifier::Control);
    println!("  {:?}", KeyModifier::Meta);
    println!("  {:?}", KeyModifier::Shift);

    println!();
}

fn demo_scroll_into_view() {
    println!("--- Demo 6: Scroll Into View ---\n");

    let footer = Locator::new("footer.page-footer");
    let action = footer.scroll_into_view().expect("Should create action");

    println!("Locator::new(\"footer.page-footer\").scroll_into_view():");
    match &action {
        LocatorAction::ScrollIntoView { locator } => {
            println!("  Target: {:?}", locator.selector());
        }
        _ => println!("  Unexpected action type"),
    }

    // Use cases
    println!("\nUse cases:");
    println!("  - Scroll to bottom of long lists");
    println!("  - Navigate to off-screen elements");
    println!("  - Ensure element visibility before action");
    println!("  - Test lazy-loaded content");

    println!();
}
