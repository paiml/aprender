//! Element Assertions Demo - Playwright-style State Assertions
//!
//! Demonstrates PMAT-004 element state assertions:
//! - toBeEnabled() / toBeDisabled()
//! - toBeChecked()
//! - toBeEditable()
//! - toBeHidden() / toBeFocused() / toBeEmpty()
//! - toHaveValue() / toHaveCSS() / toHaveClass() / toHaveId()
//! - toHaveAttribute()
//!
//! # Running
//!
//! ```bash
//! cargo run --example element_assertions -p probar
//! ```
//!
//! # Playwright Parity
//!
//! These assertions match Playwright's expect API:
//! - `expect(locator).toBeEnabled()` -> `expect(locator).to_be_enabled()`
//! - `expect(locator).toBeChecked()` -> `expect(locator).to_be_checked()`
//! - `expect(locator).toHaveValue('text')` -> `expect(locator).to_have_value("text")`

#![allow(clippy::uninlined_format_args)]

use jugar_probar::{expect, ExpectAssertion, Locator};

fn main() {
    println!("=== Probar Element Assertions Demo (PMAT-004) ===\n");

    // Demo 1: Enabled/Disabled state
    demo_enabled_disabled();

    // Demo 2: Checked state
    demo_checked_state();

    // Demo 3: Editable state
    demo_editable_state();

    // Demo 4: Visibility states
    demo_visibility_states();

    // Demo 5: Value assertions
    demo_value_assertions();

    // Demo 6: CSS assertions
    demo_css_assertions();

    // Demo 7: Class/ID assertions
    demo_class_id_assertions();

    // Demo 8: Attribute assertions
    demo_attribute_assertions();

    // Demo 9: State validation
    demo_state_validation();

    println!("\n=== Element Assertions Demo Complete ===");
}

fn demo_enabled_disabled() {
    println!("--- Demo 1: Enabled/Disabled State ---\n");

    let button = Locator::new("button#submit");

    // toBeEnabled
    let enabled = expect(button.clone()).to_be_enabled();
    println!("expect(button).to_be_enabled():");
    println!("  Assertion type: {:?}", std::mem::discriminant(&enabled));

    // toBeDisabled
    let disabled = expect(button).to_be_disabled();
    println!("\nexpect(button).to_be_disabled():");
    println!("  Assertion type: {:?}", std::mem::discriminant(&disabled));

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify form submit button state");
    println!("  - Check if action is available");
    println!("  - Test disabled UI during loading");

    println!();
}

fn demo_checked_state() {
    println!("--- Demo 2: Checked State ---\n");

    let checkbox = Locator::new("input[type='checkbox']#agree");
    let radio = Locator::new("input[type='radio']#option-a");

    // toBeChecked for checkbox
    let checkbox_checked = expect(checkbox).to_be_checked();
    println!("expect(checkbox).to_be_checked():");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&checkbox_checked)
    );

    // toBeChecked for radio
    let radio_checked = expect(radio).to_be_checked();
    println!("\nexpect(radio).to_be_checked():");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&radio_checked)
    );

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify terms acceptance checkbox");
    println!("  - Check selected game options");
    println!("  - Test toggle switches");
    println!("  - Verify radio button selection");

    println!();
}

fn demo_editable_state() {
    println!("--- Demo 3: Editable State ---\n");

    let input = Locator::new("input#username");
    let textarea = Locator::new("textarea#bio");

    // toBeEditable for input
    let input_editable = expect(input).to_be_editable();
    println!("expect(input).to_be_editable():");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&input_editable)
    );

    // toBeEditable for textarea
    let textarea_editable = expect(textarea).to_be_editable();
    println!("\nexpect(textarea).to_be_editable():");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&textarea_editable)
    );

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify form fields are editable");
    println!("  - Test readonly state after submission");
    println!("  - Check contenteditable elements");

    println!();
}

fn demo_visibility_states() {
    println!("--- Demo 4: Visibility States ---\n");

    let modal = Locator::new(".modal-dialog");
    let input = Locator::new("input#search");
    let container = Locator::new(".empty-container");

    // toBeHidden
    let hidden = expect(modal.clone()).to_be_hidden();
    println!("expect(modal).to_be_hidden():");
    println!("  Assertion type: {:?}", std::mem::discriminant(&hidden));

    // toBeFocused
    let focused = expect(input).to_be_focused();
    println!("\nexpect(input).to_be_focused():");
    println!("  Assertion type: {:?}", std::mem::discriminant(&focused));

    // toBeEmpty
    let empty = expect(container).to_be_empty();
    println!("\nexpect(container).to_be_empty():");
    println!("  Assertion type: {:?}", std::mem::discriminant(&empty));

    // Contrast with toBeVisible
    let visible = expect(modal).to_be_visible();
    println!("\nexpect(modal).to_be_visible():");
    println!("  Assertion type: {:?}", std::mem::discriminant(&visible));

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify modal closed after action");
    println!("  - Check input focus after click");
    println!("  - Test empty state displays");
    println!("  - Verify loading indicators hidden");

    println!();
}

fn demo_value_assertions() {
    println!("--- Demo 5: Value Assertions ---\n");

    let input = Locator::new("input#score");
    let select = Locator::new("select#difficulty");

    // toHaveValue
    let has_value = expect(input).to_have_value("100");
    println!("expect(input).to_have_value(\"100\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_value));

    // toHaveValue for select
    let select_value = expect(select).to_have_value("hard");
    println!("\nexpect(select).to_have_value(\"hard\"):");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&select_value)
    );

    // Validation
    if let ExpectAssertion::HasValue { expected, .. } = &has_value {
        println!("\nValidation examples:");
        println!("  Expected value: {}", expected);
        println!(
            "  Validate \"100\": {:?}",
            has_value.validate("100").is_ok()
        );
        println!("  Validate \"50\": {:?}", has_value.validate("50").is_ok());
    }

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify input values after fill");
    println!("  - Check calculated fields");
    println!("  - Test select/dropdown values");
    println!("  - Verify score displays");

    println!();
}

fn demo_css_assertions() {
    println!("--- Demo 6: CSS Property Assertions ---\n");

    let element = Locator::new(".status-indicator");

    // toHaveCSS for color
    let has_color = expect(element.clone()).to_have_css("color", "rgb(0, 255, 0)");
    println!("expect(element).to_have_css(\"color\", \"rgb(0, 255, 0)\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_color));

    // toHaveCSS for display
    let has_display = expect(element.clone()).to_have_css("display", "flex");
    println!("\nexpect(element).to_have_css(\"display\", \"flex\"):");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&has_display)
    );

    // toHaveCSS for visibility
    let has_visibility = expect(element).to_have_css("visibility", "visible");
    println!("\nexpect(element).to_have_css(\"visibility\", \"visible\"):");
    println!(
        "  Assertion type: {:?}",
        std::mem::discriminant(&has_visibility)
    );

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify status indicator colors");
    println!("  - Check responsive layout changes");
    println!("  - Test animation states");
    println!("  - Verify theme changes");

    println!();
}

fn demo_class_id_assertions() {
    println!("--- Demo 7: Class and ID Assertions ---\n");

    let element = Locator::new("div.game-container");

    // toHaveClass
    let has_class = expect(element.clone()).to_have_class("active");
    println!("expect(element).to_have_class(\"active\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_class));

    // toHaveId
    let has_id = expect(element).to_have_id("main-game");
    println!("\nexpect(element).to_have_id(\"main-game\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_id));

    // Validation for class (checks within class list)
    if let ExpectAssertion::HasClass { expected, .. } = &has_class {
        println!("\nClass validation examples:");
        println!("  Looking for class: {}", expected);
        println!(
            "  In \"btn active primary\": {:?}",
            has_class.validate("btn active primary").is_ok()
        );
        println!(
            "  In \"btn disabled\": {:?}",
            has_class.validate("btn disabled").is_ok()
        );
    }

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify state classes (active, disabled, loading)");
    println!("  - Check dynamic class changes");
    println!("  - Test CSS class toggles");
    println!("  - Verify element identification");

    println!();
}

fn demo_attribute_assertions() {
    println!("--- Demo 8: Attribute Assertions ---\n");

    let button = Locator::new("button#action");
    let link = Locator::new("a.external");
    let input = Locator::new("input#email");

    // toHaveAttribute for aria
    let has_aria = expect(button).to_have_attribute("aria-label", "Submit form");
    println!("expect(button).to_have_attribute(\"aria-label\", \"Submit form\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_aria));

    // toHaveAttribute for href
    let has_href = expect(link).to_have_attribute("href", "https://example.com");
    println!("\nexpect(link).to_have_attribute(\"href\", \"https://example.com\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_href));

    // toHaveAttribute for type
    let has_type = expect(input).to_have_attribute("type", "email");
    println!("\nexpect(input).to_have_attribute(\"type\", \"email\"):");
    println!("  Assertion type: {:?}", std::mem::discriminant(&has_type));

    // Use cases
    println!("\nUse cases:");
    println!("  - Verify ARIA attributes (accessibility)");
    println!("  - Check data-* attributes");
    println!("  - Test href/src values");
    println!("  - Verify input types");

    println!();
}

fn demo_state_validation() {
    println!("--- Demo 9: State Validation ---\n");

    let button = Locator::new("button");

    // Create assertions
    let enabled = expect(button.clone()).to_be_enabled();
    let disabled = expect(button.clone()).to_be_disabled();
    let checked = expect(button.clone()).to_be_checked();
    let visible = expect(button.clone()).to_be_visible();
    let hidden = expect(button).to_be_hidden();

    // Validate states
    println!("validate_state(true/false) examples:\n");

    println!("to_be_enabled():");
    println!(
        "  validate_state(true): {:?}",
        enabled.validate_state(true).is_ok()
    );
    println!(
        "  validate_state(false): {:?}",
        enabled.validate_state(false).is_ok()
    );

    println!("\nto_be_disabled():");
    println!(
        "  validate_state(true): {:?}",
        disabled.validate_state(true).is_ok()
    );
    println!(
        "  validate_state(false): {:?}",
        disabled.validate_state(false).is_ok()
    );

    println!("\nto_be_checked():");
    println!(
        "  validate_state(true): {:?}",
        checked.validate_state(true).is_ok()
    );
    println!(
        "  validate_state(false): {:?}",
        checked.validate_state(false).is_ok()
    );

    println!("\nto_be_visible():");
    println!(
        "  validate_state(true): {:?}",
        visible.validate_state(true).is_ok()
    );
    println!(
        "  validate_state(false): {:?}",
        visible.validate_state(false).is_ok()
    );

    println!("\nto_be_hidden():");
    println!(
        "  validate_state(true): {:?}",
        hidden.validate_state(true).is_ok()
    );
    println!(
        "  validate_state(false): {:?}",
        hidden.validate_state(false).is_ok()
    );

    // Error messages
    println!("\nError message on failure:");
    if let Err(e) = enabled.validate_state(false) {
        println!("  {}", e);
    }

    println!();
}
