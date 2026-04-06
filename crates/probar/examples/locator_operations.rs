//! Locator Operations Demo - Playwright-style Locator Composition
//!
//! Demonstrates PMAT-002 locator operations for complex element selection:
//! - filter() with has/hasText/hasNot/hasNotText
//! - and() for intersection
//! - or() for union
//! - first(), last(), nth() for indexing
//!
//! # Running
//!
//! ```bash
//! cargo run --example locator_operations -p probar
//! ```
//!
//! # Playwright Parity
//!
//! These operations match Playwright's locator API:
//! - `locator.filter({ hasText: 'Submit' })` -> `locator.filter(opts.has_text("Submit"))`
//! - `locator.and(other)` -> `locator.and(other)`
//! - `locator.or(other)` -> `locator.or(other)`
//! - `locator.first()` -> `locator.first()`

#![allow(clippy::uninlined_format_args)]

use jugar_probar::prelude::*;

fn main() {
    println!("=== Probar Locator Operations Demo (PMAT-002) ===\n");

    // Demo 1: Filter operations
    demo_filter_operations();

    // Demo 2: And/Or composition
    demo_and_or_operations();

    // Demo 3: Index operations
    demo_index_operations();

    // Demo 4: Chained operations
    demo_chained_operations();

    println!("\n=== Locator Operations Demo Complete ===");
}

fn demo_filter_operations() {
    println!("--- Demo 1: Filter Operations ---\n");

    // Filter with hasText
    let opts = FilterOptions::new().has_text("Submit");
    println!("FilterOptions::new().has_text(\"Submit\"):");
    println!("  has_text: {:?}", opts.has_text);

    // Filter with hasNotText
    let exclude = FilterOptions::new().has_not_text("Cancel");
    println!("\nFilterOptions::new().has_not_text(\"Cancel\"):");
    println!("  has_not_text: {:?}", exclude.has_not_text);

    // Filter with child locator (has)
    let child = Locator::new(".icon");
    let with_child = FilterOptions::new().has(child);
    println!("\nFilterOptions::new().has(Locator::new(\".icon\")):");
    println!("  has: {:?}", with_child.has.is_some());

    // Filter with excluded child (hasNot)
    let excluded = Locator::new(".disabled");
    let without_child = FilterOptions::new().has_not(excluded);
    println!("\nFilterOptions::new().has_not(Locator::new(\".disabled\")):");
    println!("  has_not: {:?}", without_child.has_not.is_some());

    // Combined filters
    let combined = FilterOptions::new()
        .has_text("Active")
        .has_not_text("Disabled");
    println!("\nCombined filters:");
    println!("  has_text: {:?}", combined.has_text);
    println!("  has_not_text: {:?}", combined.has_not_text);

    // Apply filter to locator
    let buttons = Locator::new("button");
    let active_buttons = buttons.filter(FilterOptions::new().has_text("Active"));
    println!("\nLocator::new(\"button\").filter(has_text(\"Active\")):");
    println!("  Selector: {:?}", active_buttons.selector());

    println!();
}

fn demo_and_or_operations() {
    println!("--- Demo 2: And/Or Composition ---\n");

    // AND: Both selectors must match (intersection)
    let div = Locator::new("div");
    let active = Locator::new(".active");
    let combined = div.and(active);

    println!("Locator::new(\"div\").and(Locator::new(\".active\")):");
    if let Selector::Css(s) = combined.selector() {
        println!("  Combined CSS: {}", s);
    }

    // OR: Either selector can match (union)
    let button = Locator::new("button");
    let link = Locator::new("a.btn");
    let either = button.or(link);

    println!("\nLocator::new(\"button\").or(Locator::new(\"a.btn\")):");
    if let Selector::Css(s) = either.selector() {
        println!("  Combined CSS: {}", s);
    }

    // Multiple OR operations
    let submit = Locator::new("[type='submit']");
    let input_btn = Locator::new("input[type='button']");
    let any_btn = Locator::new("button").or(submit).or(input_btn);

    println!("\nMultiple ORs (any button-like element):");
    if let Selector::Css(s) = any_btn.selector() {
        println!("  Combined CSS: {}", s);
    }

    // AND then OR
    let primary = Locator::new(".btn").and(Locator::new(".primary"));
    let secondary = Locator::new(".btn").and(Locator::new(".secondary"));
    let styled = primary.or(secondary);

    println!("\n(btn AND primary) OR (btn AND secondary):");
    if let Selector::Css(s) = styled.selector() {
        println!("  Combined CSS: {}", s);
    }

    println!();
}

fn demo_index_operations() {
    println!("--- Demo 3: Index Operations ---\n");

    let items = Locator::new("li.menu-item");

    // First element
    let first = items.clone().first();
    println!("Locator::new(\"li.menu-item\").first():");
    if let Selector::Css(s) = first.selector() {
        println!("  CSS: {}", s);
    }

    // Last element
    let last = items.clone().last();
    println!("\nLocator::new(\"li.menu-item\").last():");
    if let Selector::Css(s) = last.selector() {
        println!("  CSS: {}", s);
    }

    // Nth element (0-indexed, but nth-child is 1-indexed)
    let third = items.clone().nth(2);
    println!("\nLocator::new(\"li.menu-item\").nth(2): (3rd element)");
    if let Selector::Css(s) = third.selector() {
        println!("  CSS: {}", s);
    }

    // First item in the list
    let first_item = items.nth(0);
    println!("\nLocator::new(\"li.menu-item\").nth(0): (1st element)");
    if let Selector::Css(s) = first_item.selector() {
        println!("  CSS: {}", s);
    }

    println!();
}

fn demo_chained_operations() {
    println!("--- Demo 4: Chained Operations ---\n");

    // Complex chain: buttons that are active, get the second one
    let active_button = Locator::new("button").and(Locator::new(".active")).nth(1);

    println!("Locator::new(\"button\").and(.active).nth(1):");
    if let Selector::Css(s) = active_button.selector() {
        println!("  CSS: {}", s);
    }

    // Filter then index
    let filtered_first = Locator::new("div.card")
        .filter(FilterOptions::new().has_text("Featured"))
        .first();

    println!("\nLocator::new(\"div.card\").filter(has_text(\"Featured\")).first():");
    println!("  Selector: {:?}", filtered_first.selector());

    // Real-world example: Get the first enabled submit button
    let enabled_submit = Locator::new("button[type='submit']")
        .and(Locator::new(":not([disabled])"))
        .first();

    println!("\nFirst enabled submit button:");
    if let Selector::Css(s) = enabled_submit.selector() {
        println!("  CSS: {}", s);
    }

    // Game UI example: Get last enemy in a list
    let last_enemy = Locator::new(".enemy-list .enemy").last();

    println!("\nLast enemy in enemy list:");
    if let Selector::Css(s) = last_enemy.selector() {
        println!("  CSS: {}", s);
    }

    println!();
}
