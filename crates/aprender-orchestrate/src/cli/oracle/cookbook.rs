//! Cookbook oracle commands
//!
//! This module contains cookbook-related commands for displaying and searching
//! practical recipes for common Sovereign AI Stack patterns.

use crate::ansi_colors::Colorize;
use crate::oracle;
use crate::oracle::svg::shapes::Point;
use crate::oracle::svg::{ShapeHeavyRenderer, TextHeavyRenderer};

use super::types::OracleOutputFormat;

// ============================================================================
// SVG Generation
// ============================================================================

/// Generate an SVG diagram for a recipe
fn generate_recipe_svg(recipe: &oracle::cookbook::Recipe) -> String {
    // Choose renderer based on recipe content
    if recipe.components.len() > 2 {
        // Use shape-heavy renderer for multi-component recipes
        let mut renderer = ShapeHeavyRenderer::new().title(&recipe.title);

        // Add components as a horizontal stack
        let components: Vec<(&str, &str)> =
            recipe.components.iter().map(|c| (c.as_str(), c.as_str())).collect();

        renderer = renderer.horizontal_stack(&components, Point::new(100.0, 200.0));
        renderer.build()
    } else {
        // Use text-heavy renderer for documentation-style recipes
        TextHeavyRenderer::new()
            .title(&recipe.title)
            .heading("Problem")
            .paragraph(&recipe.problem)
            .heading("Components")
            .paragraph(&recipe.components.join(", "))
            .heading("Tags")
            .paragraph(&recipe.tags.join(", "))
            .build()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn cookbook_show_recipe_not_found(id: &str, cookbook: &oracle::cookbook::Cookbook) {
    println!("{} Recipe '{}' not found", "Error:".bright_red().bold(), id);
    println!();
    println!("Available recipes:");
    for r in cookbook.recipes() {
        println!("  {} - {}", r.id.cyan(), r.title.dimmed());
    }
}

fn cookbook_show_help() {
    println!("{}", "Batuta Cookbook".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();
    println!("{}", "Practical recipes for common Sovereign AI Stack patterns".dimmed());
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!("  {} {}", "batuta oracle --cookbook".cyan(), "# List all recipes".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --recipe wasm-zero-js".cyan(),
        "# Show specific recipe".dimmed()
    );
    println!("  {} {}", "batuta oracle --recipes-by-tag wasm".cyan(), "# Find by tag".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --recipes-by-component aprender".cyan(),
        "# Find by component".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --search-recipes \"random forest\"".cyan(),
        "# Search".dimmed()
    );
    println!();
    println!("{} wasm, ml, distributed, quality, transpilation", "Tags:".bright_yellow());
    println!();
}

fn cookbook_display_filter_results(
    recipes: &[&oracle::cookbook::Recipe],
    empty_msg: &str,
    title: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    if recipes.is_empty() {
        println!("{}", empty_msg);
    } else {
        display_recipe_list(recipes, title, format)?;
    }
    Ok(())
}

/// Format a slice of strings by wrapping each item with a prefix/suffix and joining with a separator.
fn format_items(items: &[String], prefix: &str, suffix: &str, sep: &str) -> String {
    items.iter().map(|x| format!("{}{}{}", prefix, x, suffix)).collect::<Vec<_>>().join(sep)
}

// ============================================================================
// Recipe Display Functions
// ============================================================================

fn display_recipe_markdown(recipe: &oracle::cookbook::Recipe) {
    println!("# {}\n", recipe.title);
    println!("**ID:** `{}`\n", recipe.id);
    println!("## Problem\n\n{}\n", recipe.problem);
    println!("## Components\n\n{}\n", format_items(&recipe.components, "`", "`", ", "));
    println!("## Tags\n\n{}\n", format_items(&recipe.tags, "`", "`", ", "));
    println!("## Code\n\n```rust\n{}\n```\n", recipe.code);
    if !recipe.test_code.is_empty() {
        println!("## Tests\n\n```rust\n{}\n```\n", recipe.test_code);
    }
    if !recipe.related.is_empty() {
        println!("## Related Recipes\n\n{}\n", format_items(&recipe.related, "`", "`", ", "));
    }
}

fn display_recipe_code_line(line: &str) {
    #[cfg(feature = "syntect")]
    crate::cli::syntax::print_highlighted_line(line, crate::cli::syntax::Language::Rust, "");
    #[cfg(not(feature = "syntect"))]
    println!("{}", line);
}

fn display_recipe_text(recipe: &oracle::cookbook::Recipe) {
    println!("{} {}", ">>".bright_cyan(), recipe.title.bright_white().bold());
    println!("{}", "─".repeat(60).dimmed());
    println!();
    println!("{} {}", "ID:".bright_yellow(), recipe.id.cyan());
    println!();
    println!("{}", "Problem:".bright_yellow());
    println!("  {}", recipe.problem);
    println!();
    println!("{}", "Components:".bright_yellow());
    for comp in &recipe.components {
        println!("  * {}", comp.cyan());
    }
    println!();
    println!("{}", "Tags:".bright_yellow());
    println!("  {}", format_items(&recipe.tags, "#", "", " ").dimmed());
    println!();
    println!("{}", "Code:".bright_yellow());
    println!("{}", "─".repeat(60).dimmed());
    for line in recipe.code.lines() {
        display_recipe_code_line(line);
    }
    println!("{}", "─".repeat(60).dimmed());
    if !recipe.test_code.is_empty() {
        println!();
        println!("{}", "Tests:".bright_yellow());
        println!("{}", "─".repeat(60).dimmed());
        for line in recipe.test_code.lines() {
            display_recipe_code_line(line);
        }
        println!("{}", "─".repeat(60).dimmed());
    }
    if !recipe.related.is_empty() {
        println!();
        println!("{}", "Related:".bright_yellow());
        for related in &recipe.related {
            println!("  -> {}", related.cyan());
        }
    }
    println!();
}

/// Display a single recipe
fn display_recipe(
    recipe: &oracle::cookbook::Recipe,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Code => {
            println!("{}", recipe.code);
            if !recipe.test_code.is_empty() {
                println!("\n{}", recipe.test_code);
            }
        }
        OracleOutputFormat::CodeSvg => {
            // Output code
            println!("{}", recipe.code);
            if !recipe.test_code.is_empty() {
                println!("\n{}", recipe.test_code);
            }
            // Generate and output SVG diagram
            let svg = generate_recipe_svg(recipe);
            println!("\n<!-- SVG Diagram -->\n{}", svg);
        }
        OracleOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(recipe)?);
        }
        OracleOutputFormat::Markdown => display_recipe_markdown(recipe),
        OracleOutputFormat::Text => display_recipe_text(recipe),
    }
    Ok(())
}

// ============================================================================
// Recipe List Display Functions
// ============================================================================

/// Display recipe list as code output
fn display_recipe_list_code(recipes: &[&oracle::cookbook::Recipe]) {
    for (i, recipe) in recipes.iter().enumerate() {
        if i > 0 {
            println!();
        }
        println!("// --- {} ---", recipe.id);
        println!("{}", recipe.code);
        if !recipe.test_code.is_empty() {
            println!("\n{}", recipe.test_code);
        }
    }
}

/// Display recipe list as code+svg output
fn display_recipe_list_code_svg(recipes: &[&oracle::cookbook::Recipe]) {
    for (i, recipe) in recipes.iter().enumerate() {
        if i > 0 {
            println!();
        }
        println!("// --- {} ---", recipe.id);
        println!("{}", recipe.code);
        if !recipe.test_code.is_empty() {
            println!("\n{}", recipe.test_code);
        }
        let svg = generate_recipe_svg(recipe);
        println!("\n<!-- SVG Diagram: {} -->\n{}", recipe.id, svg);
    }
}

/// Display recipe list as markdown table
fn display_recipe_list_markdown(recipes: &[&oracle::cookbook::Recipe], title: &str) {
    println!("# {}\n", title);
    println!("| ID | Title | Components | Tags |");
    println!("|---|---|---|---|");
    for recipe in recipes {
        println!(
            "| `{}` | {} | {} | {} |",
            recipe.id,
            recipe.title,
            recipe.components.join(", "),
            recipe.tags.join(", ")
        );
    }
}

/// Display recipe list as formatted text
fn display_recipe_list_text(recipes: &[&oracle::cookbook::Recipe], title: &str) {
    println!("{} {}", ">>".bright_cyan(), title.bright_white().bold());
    println!("{}", "─".repeat(60).dimmed());
    println!();
    for recipe in recipes {
        println!("  {} {}", recipe.id.cyan().bold(), format!("- {}", recipe.title).dimmed());
        println!("    {} {}", "Components:".dimmed(), recipe.components.join(", ").bright_blue());
        println!("    {} {}", "Tags:".dimmed(), format_items(&recipe.tags, "#", "", " "));
        println!();
    }
    println!("{} Use {} to view a recipe", "Tip:".bright_yellow(), "--recipe <id>".cyan());
    println!();
}

/// Display a list of recipes
fn display_recipe_list(
    recipes: &[&oracle::cookbook::Recipe],
    title: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Code => display_recipe_list_code(recipes),
        OracleOutputFormat::CodeSvg => display_recipe_list_code_svg(recipes),
        OracleOutputFormat::Json => {
            let json = serde_json::to_string_pretty(recipes)?;
            println!("{}", json);
        }
        OracleOutputFormat::Markdown => display_recipe_list_markdown(recipes, title),
        OracleOutputFormat::Text => display_recipe_list_text(recipes, title),
    }
    Ok(())
}

// ============================================================================
// Public Commands
// ============================================================================

/// Handle cookbook commands
pub fn cmd_oracle_cookbook(
    list_all: bool,
    recipe_id: Option<String>,
    by_tag: Option<String>,
    by_component: Option<String>,
    search: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::cookbook::Cookbook;

    let cookbook = Cookbook::standard();

    if let Some(id) = recipe_id {
        return match cookbook.get(&id) {
            Some(recipe) => display_recipe(recipe, format),
            None => {
                cookbook_show_recipe_not_found(&id, &cookbook);
                Ok(())
            }
        };
    }

    if let Some(tag) = by_tag {
        let recipes = cookbook.find_by_tag(&tag);
        let empty_msg = format!(
            "No recipes found with tag '{}'\n\nAvailable tags: wasm, ml, distributed, quality, transpilation",
            tag
        );
        return cookbook_display_filter_results(
            &recipes,
            &empty_msg,
            &format!("Recipes tagged '{}'", tag),
            format,
        );
    }

    if let Some(component) = by_component {
        let recipes = cookbook.find_by_component(&component);
        return cookbook_display_filter_results(
            &recipes,
            &format!("No recipes found using component '{}'", component),
            &format!("Recipes using '{}'", component),
            format,
        );
    }

    if let Some(query) = search {
        let recipes = cookbook.search(&query);
        return cookbook_display_filter_results(
            &recipes,
            &format!("No recipes found matching '{}'", query),
            &format!("Recipes matching '{}'", query),
            format,
        );
    }

    if list_all {
        let recipes: Vec<_> = cookbook.recipes().iter().collect();
        display_recipe_list(&recipes, "All Cookbook Recipes", format)?;
        return Ok(());
    }

    cookbook_show_help();
    Ok(())
}
