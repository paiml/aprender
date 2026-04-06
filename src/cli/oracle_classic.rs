//! Classic Oracle Commands
//!
//! This module contains the classic oracle query interface, component recommendations,
//! and display functions.

#![cfg(feature = "native")]

use serde::Serialize;

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::oracle::OracleOutputFormat;

// ============================================================================
// Display helpers — reduce structural repetition (PMAT entropy violations)
// ============================================================================

/// Print any `Serialize` value as pretty JSON.
fn print_json<T: Serialize>(data: &T) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(data)?);
    Ok(())
}

/// Print a bold label followed by a styled value.
fn print_label_value(label: &str, value: impl std::fmt::Display) {
    println!("{}: {}", label.bold(), value);
}

/// Print indented code with Rust syntax highlighting using syntect.
fn print_code_block(code: &str) {
    #[cfg(feature = "syntect")]
    {
        super::syntax::print_highlighted(code, super::syntax::Language::Rust, "  ");
    }
    #[cfg(not(feature = "syntect"))]
    {
        for line in code.lines() {
            println!("  {}", line);
        }
    }
}

/// Print a horizontal divider of the given character and width.
fn print_divider(ch: char, width: usize) {
    println!("{}", ch.to_string().repeat(width).dimmed());
}

/// Exit with code 1 and a stderr message when `--format code` has no code to emit.
fn exit_no_code(context: &str) -> ! {
    eprintln!("No code available for {} (try --format text)", context);
    std::process::exit(1)
}

/// Return `Ok(())` early after printing an error when an expected item is missing.
///
/// Usage: `require_found!(value, "Component '{}' not found", name);`
macro_rules! require_found {
    ($opt:expr, $($arg:tt)+) => {
        match $opt {
            Some(v) => v,
            None => {
                println!("{} {}", "\u{274c}".red(), format!($($arg)+));
                return Ok(());
            }
        }
    };
}

/// Options for the oracle command.
///
/// Groups all oracle command parameters to avoid too-many-arguments warnings.
pub struct OracleOptions {
    pub query: Option<String>,
    pub recommend: bool,
    pub problem: Option<String>,
    pub data_size: Option<String>,
    pub integrate: Option<String>,
    pub capabilities: Option<String>,
    pub list: bool,
    pub show: Option<String>,
    pub interactive: bool,
    pub arxiv: bool,
    pub arxiv_live: bool,
    pub arxiv_max: usize,
    pub format: OracleOutputFormat,
}

fn oracle_show_help() {
    println!("{}", "🔮 Batuta Oracle Mode".bright_cyan().bold());
    print_divider('─', 50);
    println!();
    println!("{}", "Query the Sovereign AI Stack for recommendations".dimmed());
    println!();
    println!("{}", "Knowledge Graph:".bright_yellow());
    println!("  {} {}", "batuta oracle".cyan(), "\"How do I train a random forest?\"".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --recommend --problem".cyan(),
        "\"image classification\"".dimmed()
    );
    println!("  {} {}", "batuta oracle --capabilities".cyan(), "aprender".dimmed());
    println!("  {} {}", "batuta oracle --integrate".cyan(), "\"aprender,realizar\"".dimmed());
    println!("  {} {}", "batuta oracle --list".cyan(), "".dimmed());
    println!("  {} {}", "batuta oracle --interactive".cyan(), "".dimmed());
    println!();
    println!("{}", "Cookbook (Practical Recipes):".bright_yellow());
    println!("  {} {}", "batuta oracle --cookbook".cyan(), "# List all recipes".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --recipe".cyan(),
        "wasm-zero-js       # Show recipe".dimmed()
    );
    println!("  {} {}", "batuta oracle --recipes-by-tag".cyan(), "ml        # By tag".dimmed());
    println!("  {} {}", "batuta oracle --search-recipes".cyan(), "\"gpu\"   # Search".dimmed());
    println!();
}

fn oracle_handle_recommend(
    recommender: &oracle::Recommender,
    problem: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let query_text = problem.unwrap_or_else(|| "general ML task".into());
    let response = recommender.query(&query_text);
    display_oracle_response(&response, format)
}

fn oracle_handle_query(
    recommender: &oracle::Recommender,
    query_text: &str,
    data_size: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::OracleQuery;

    let parsed_size = data_size.and_then(|s| parse_data_size(&s));
    let mut oracle_query = OracleQuery::new(query_text);
    if let Some(size) = parsed_size {
        oracle_query = oracle_query.with_data_size(size);
    }
    let response = recommender.query_structured(&oracle_query);
    display_oracle_response(&response, format)
}

/// Dispatch oracle subcommand that requires Option arguments.
fn dispatch_oracle_option(
    opts: &mut OracleOptions,
    recommender: &oracle::Recommender,
) -> Option<anyhow::Result<()>> {
    if let Some(name) = opts.show.take() {
        return Some(display_component_details(recommender, &name, opts.format));
    }
    if let Some(name) = opts.capabilities.take() {
        return Some(display_capabilities(recommender, &name, opts.format));
    }
    if let Some(components) = opts.integrate.take() {
        return Some(display_integration(recommender, &components, opts.format));
    }
    if let Some(query_text) = opts.query.take() {
        return Some(oracle_handle_query(
            recommender,
            &query_text,
            opts.data_size.take(),
            opts.format,
        ));
    }
    if opts.recommend {
        return Some(oracle_handle_recommend(recommender, opts.problem.take(), opts.format));
    }
    None
}

pub fn cmd_oracle(mut opts: OracleOptions) -> anyhow::Result<()> {
    use oracle::Recommender;

    let recommender = Recommender::new();

    let wants_arxiv = opts.arxiv || opts.arxiv_live;
    let arxiv_live = opts.arxiv_live;
    let arxiv_max = opts.arxiv_max;
    let query_text = opts.query.clone();
    let format = opts.format;

    if opts.list {
        return display_component_list(&recommender, opts.format);
    }
    if opts.interactive {
        return run_interactive_oracle(&recommender);
    }
    if let Some(result) = dispatch_oracle_option(&mut opts, &recommender) {
        result?;
        // After displaying the main oracle response, append arXiv papers
        if wants_arxiv {
            if let Some(ref q) = query_text {
                display_arxiv_enrichment(q, arxiv_live, arxiv_max, format)?;
            }
        }
        return Ok(());
    }

    oracle_show_help();
    Ok(())
}

fn display_component_list(
    recommender: &oracle::Recommender,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    println!("{}", "🔮 Sovereign AI Stack Components".bright_cyan().bold());
    print_divider('─', 50);
    println!();

    let components: Vec<_> = recommender.list_components();

    match format {
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => exit_no_code("--list"),
        OracleOutputFormat::Json => {
            print_json(&components)?;
        }
        OracleOutputFormat::Markdown => {
            println!("## Sovereign AI Stack Components\n");
            println!("| Component | Version | Layer | Description |");
            println!("|-----------|---------|-------|-------------|");
            for name in &components {
                if let Some(comp) = recommender.get_component(name) {
                    println!(
                        "| {} | {} | {} | {} |",
                        comp.name, comp.version, comp.layer, comp.description
                    );
                }
            }
        }
        OracleOutputFormat::Text => {
            // Group by layer
            for layer in oracle::StackLayer::all() {
                let layer_components: Vec<_> = components
                    .iter()
                    .filter_map(|name| recommender.get_component(name))
                    .filter(|c| c.layer == layer)
                    .collect();

                if !layer_components.is_empty() {
                    println!("{} {}", "Layer".bold(), format!("{}", layer).cyan());
                    for comp in layer_components {
                        println!(
                            "  {} {} {} - {}",
                            "•".bright_blue(),
                            comp.name.bright_green(),
                            format!("v{}", comp.version).dimmed(),
                            comp.description.dimmed()
                        );
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

fn display_component_details(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let comp = require_found!(recommender.get_component(name), "Component '{}' not found", name);

    match format {
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => exit_no_code("--show"),
        OracleOutputFormat::Json => {
            print_json(&comp)?;
        }
        OracleOutputFormat::Markdown => {
            println!("## {}\n", comp.name);
            println!("**Version:** {}\n", comp.version);
            println!("**Layer:** {}\n", comp.layer);
            println!("**Description:** {}\n", comp.description);
            println!("### Capabilities\n");
            for cap in &comp.capabilities {
                println!(
                    "- **{}**{}",
                    cap.name,
                    cap.description.as_ref().map(|d| format!(": {}", d)).unwrap_or_default()
                );
            }
        }
        OracleOutputFormat::Text => {
            println!("{}", format!("📦 {}", comp.name).bright_cyan().bold());
            print_divider('─', 50);
            println!();
            print_label_value("Version", comp.version.cyan());
            print_label_value("Layer", format!("{}", comp.layer).cyan());
            print_label_value("Description", &comp.description);
            println!();
            println!("{}", "Capabilities:".bright_yellow());
            for cap in &comp.capabilities {
                let desc = cap
                    .description
                    .as_ref()
                    .map(|d| format!(" - {}", d.dimmed()))
                    .unwrap_or_default();
                println!("  {} {}{}", "•".bright_blue(), cap.name.green(), desc);
            }
            println!();
        }
    }

    Ok(())
}

fn display_capabilities(
    recommender: &oracle::Recommender,
    name: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let caps = recommender.get_capabilities(name);

    if caps.is_empty() {
        println!("{} No capabilities found for '{}'", "\u{274c}".red(), name);
        return Ok(());
    }

    match format {
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => exit_no_code("--capabilities"),
        OracleOutputFormat::Json => {
            print_json(&caps)?;
        }
        OracleOutputFormat::Markdown => {
            println!("## Capabilities of {}\n", name);
            for cap in &caps {
                println!("- {}", cap);
            }
        }
        OracleOutputFormat::Text => {
            println!("{}", format!("\u{1f527} Capabilities of {}", name).bright_cyan().bold());
            print_divider('\u{2500}', 50);
            println!();
            for cap in &caps {
                println!("  {} {}", "\u{2022}".bright_blue(), cap.green());
            }
            println!();
        }
    }

    Ok(())
}

fn display_integration(
    recommender: &oracle::Recommender,
    components: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = components.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        println!("{} Please specify two components separated by comma", "\u{274c}".red());
        println!(
            "  Example: {} {}",
            "batuta oracle --integrate".cyan(),
            "\"aprender,realizar\"".dimmed()
        );
        return Ok(());
    }

    let from = parts[0];
    let to = parts[1];

    let pattern = require_found!(
        recommender.get_integration(from, to),
        "No integration pattern found from '{}' to '{}'",
        from,
        to
    );

    match format {
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => match &pattern.code_template {
            Some(template) => println!("{}", template),
            None => exit_no_code("--integrate (no code template for this pair)"),
        },
        OracleOutputFormat::Json => {
            print_json(&pattern)?;
        }
        OracleOutputFormat::Markdown => {
            println!("## Integration: {} \u{2192} {}\n", from, to);
            println!("**Pattern:** {}\n", pattern.pattern_name);
            println!("**Description:** {}\n", pattern.description);
            if let Some(template) = &pattern.code_template {
                println!("### Code Example\n");
                println!("```rust\n{}\n```", template);
            }
        }
        OracleOutputFormat::Text => {
            println!(
                "{}",
                format!("\u{1f517} Integration: {} \u{2192} {}", from, to).bright_cyan().bold()
            );
            print_divider('\u{2500}', 50);
            println!();
            print_label_value("Pattern", pattern.pattern_name.cyan());
            print_label_value("Description", &pattern.description);
            println!();
            if let Some(template) = &pattern.code_template {
                println!("{}", "Code Example:".bright_yellow());
                print_divider('\u{2500}', 40);
                print_code_block(template);
                print_divider('\u{2500}', 40);
            }
            println!();
        }
    }

    Ok(())
}

fn display_response_markdown(response: &oracle::OracleResponse) {
    println!("## Oracle Recommendation\n");
    println!("**Problem Class:** {}\n", response.problem_class);
    if let Some(algo) = &response.algorithm {
        println!("**Algorithm:** {}\n", algo);
    }
    println!("### Primary Recommendation\n");
    println!("- **Component:** {}", response.primary.component);
    if let Some(path) = &response.primary.path {
        println!("- **Module:** `{}`", path);
    }
    println!("- **Confidence:** {:.0}%", response.primary.confidence * 100.0);
    println!("- **Rationale:** {}\n", response.primary.rationale);

    if !response.supporting.is_empty() {
        println!("### Supporting Components\n");
        for rec in &response.supporting {
            println!("- **{}** ({:.0}%): {}", rec.component, rec.confidence * 100.0, rec.rationale);
        }
        println!();
    }

    println!("### Compute Backend\n");
    println!("- **Backend:** {}", response.compute.backend);
    println!("- **Rationale:** {}\n", response.compute.rationale);

    if response.distribution.needed {
        println!("### Distribution\n");
        println!("- **Tool:** {}", response.distribution.tool.as_deref().unwrap_or("N/A"));
        println!("- **Rationale:** {}\n", response.distribution.rationale);
    }

    if let Some(code) = &response.code_example {
        println!("### Code Example\n");
        println!("```rust\n{}\n```\n", code);
    }
}

fn display_response_text_primary(response: &oracle::OracleResponse) {
    println!("{}", "\u{1f3af} Primary Recommendation".bright_yellow().bold());
    print_divider('\u{2500}', 50);
    println!("  {}: {}", "Component".bold(), response.primary.component.bright_green());
    if let Some(path) = &response.primary.path {
        println!("  {}: {}", "Module".bold(), path.cyan());
    }
    println!(
        "  {}: {}",
        "Confidence".bold(),
        format!("{:.0}%", response.primary.confidence * 100.0).bright_green()
    );
    println!("  {}: {}", "Rationale".bold(), response.primary.rationale.dimmed());
    println!();
}

fn display_response_text_supporting(response: &oracle::OracleResponse) {
    if response.supporting.is_empty() {
        return;
    }
    println!("{}", "\u{1f527} Supporting Components".bright_yellow().bold());
    print_divider('\u{2500}', 50);
    for rec in &response.supporting {
        println!(
            "  {} {} ({:.0}%)",
            "\u{2022}".bright_blue(),
            rec.component.green(),
            rec.confidence * 100.0
        );
        println!("    {}", rec.rationale.dimmed());
    }
    println!();
}

fn display_response_text_distribution(response: &oracle::OracleResponse) {
    if !response.distribution.needed {
        return;
    }
    println!("{}", "\u{1f310} Distribution".bright_yellow().bold());
    print_divider('\u{2500}', 50);
    println!(
        "  {}: {}",
        "Tool".bold(),
        response.distribution.tool.as_deref().unwrap_or("N/A").bright_green()
    );
    if let Some(nodes) = response.distribution.node_count {
        println!("  {}: {}", "Nodes".bold(), nodes);
    }
    println!("  {}", response.distribution.rationale.dimmed());
    println!();
}

fn display_response_text(response: &oracle::OracleResponse) {
    println!();
    println!("{}", "\u{1f52e} Oracle Recommendation".bright_cyan().bold());
    print_divider('\u{2550}', 60);
    println!();

    println!(
        "{} {}: {}",
        "\u{1f4ca}".bright_blue(),
        "Problem Class".bold(),
        response.problem_class.cyan()
    );
    if let Some(algo) = &response.algorithm {
        println!("{} {}: {}", "\u{1f9ee}".bright_blue(), "Algorithm".bold(), algo.cyan());
    }
    println!();

    display_response_text_primary(response);
    display_response_text_supporting(response);

    println!("{}", "\u{26a1} Compute Backend".bright_yellow().bold());
    print_divider('\u{2500}', 50);
    println!("  {}: {}", "Backend".bold(), format!("{}", response.compute.backend).bright_green());
    println!("  {}", response.compute.rationale.dimmed());
    println!();

    display_response_text_distribution(response);

    if let Some(code) = &response.code_example {
        println!("{}", "\u{1f4a1} Example Code".bright_yellow().bold());
        print_divider('\u{2500}', 50);
        print_code_block(code);
        print_divider('\u{2500}', 50);
        println!();
    }

    if !response.related_queries.is_empty() {
        println!("{}", "\u{2753} Related Queries".bright_yellow());
        for query in &response.related_queries {
            println!("  {} {}", "\u{2192}".bright_blue(), query.dimmed());
        }
        println!();
    }

    print_divider('\u{2550}', 60);
}

pub fn display_oracle_response(
    response: &oracle::OracleResponse,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => match &response.code_example {
            Some(code) => println!("{}", code),
            None => {
                eprintln!("No code example available for this query (try --format text)");
                std::process::exit(1);
            }
        },
        OracleOutputFormat::Json => {
            print_json(response)?;
        }
        OracleOutputFormat::Markdown => display_response_markdown(response),
        OracleOutputFormat::Text => display_response_text(response),
    }
    Ok(())
}

fn interactive_show_help() {
    println!();
    println!("{}", "Commands:".bright_yellow());
    println!("  {} - Ask a question about the stack", "any text".cyan());
    println!("  {} - List all components", "list".cyan());
    println!("  {} - Show component details", "show <component>".cyan());
    println!("  {} - Show capabilities", "caps <component>".cyan());
    println!("  {} - Exit interactive mode", "exit".cyan());
    println!();
}

/// Check if input is an exit command
fn is_exit_command(input: &str) -> bool {
    input == "exit" || input == "quit"
}

/// Handle an interactive command. Returns Ok(true) to continue, Ok(false) to exit.
fn interactive_handle_command(
    input: &str,
    recommender: &oracle::Recommender,
) -> anyhow::Result<bool> {
    if input.is_empty() {
        return Ok(true);
    }

    if is_exit_command(input) {
        println!();
        println!("{}", "👋 Goodbye!".bright_cyan());
        return Ok(false);
    }

    match input {
        "help" => interactive_show_help(),
        "list" => display_component_list(recommender, OracleOutputFormat::Text)?,
        _ if input.starts_with("show ") => {
            let name = input.strip_prefix("show ").unwrap_or("").trim();
            display_component_details(recommender, name, OracleOutputFormat::Text)?;
        }
        _ if input.starts_with("caps ") => {
            let name = input.strip_prefix("caps ").unwrap_or("").trim();
            display_capabilities(recommender, name, OracleOutputFormat::Text)?;
        }
        _ => {
            let response = recommender.query(input);
            display_oracle_response(&response, OracleOutputFormat::Text)?;
        }
    }
    Ok(true)
}

fn run_interactive_oracle(recommender: &oracle::Recommender) -> anyhow::Result<()> {
    use std::io::{self, Write};

    println!();
    println!("{}", "🔮 Batuta Oracle Mode v1.0".bright_cyan().bold());
    println!("{}", "   Ask questions about the Sovereign AI Stack".dimmed());
    println!("{}", "   Type 'exit' or 'quit' to leave".dimmed());
    println!();

    loop {
        print!("{} ", ">".bright_cyan());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !interactive_handle_command(input.trim(), recommender)? {
            break;
        }
    }

    Ok(())
}

fn parse_data_size(s: &str) -> Option<oracle::DataSize> {
    super::parse_data_size_value(s).map(oracle::DataSize::samples)
}

// ============================================================================
// arXiv enrichment display
// ============================================================================

fn print_arxiv_text(enrichment: &oracle::arxiv::ArxivEnrichment) {
    use oracle::arxiv::ArxivSource;
    println!();
    let source_label = match enrichment.source {
        ArxivSource::Builtin => "Curated Database",
        ArxivSource::Live => "arXiv API",
    };
    println!("{}", format!("Related Papers ({})", source_label).bright_yellow().bold());
    print_divider('\u{2500}', 50);
    for (i, paper) in enrichment.papers.iter().enumerate() {
        println!("  {}. {}", i + 1, paper.title.bright_green());
        println!("     {} ({})", paper.authors.dimmed(), paper.year);
        println!("     {}", paper.url.cyan());
        if !paper.summary.is_empty() {
            println!("     {}", paper.summary.dimmed());
        }
        println!();
    }
}

fn display_arxiv_enrichment(
    query: &str,
    live: bool,
    max: usize,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::arxiv::{ArxivEnricher, ArxivSource};
    use oracle::QueryEngine;

    // Skip for code formats — arXiv papers aren't code
    if matches!(format, OracleOutputFormat::Code | OracleOutputFormat::CodeSvg) {
        return Ok(());
    }

    let engine = QueryEngine::new();
    let parsed = engine.parse(query);
    let enricher = ArxivEnricher::new();

    let enrichment = if live {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(enricher.enrich_live(&parsed, max))
    } else {
        enricher.enrich_builtin(&parsed, max)
    };

    if enrichment.papers.is_empty() {
        return Ok(());
    }

    match format {
        OracleOutputFormat::Json => {
            print_json(&enrichment)?;
        }
        OracleOutputFormat::Markdown => {
            println!();
            println!("### Related Papers\n");
            for paper in &enrichment.papers {
                println!("- **[{}]({})**  ", paper.title, paper.url);
                println!("  {} ({})  ", paper.authors, paper.year);
                if !paper.summary.is_empty() {
                    println!("  > {}\n", paper.summary);
                }
            }
        }
        OracleOutputFormat::Text => {
            print_arxiv_text(&enrichment);
        }
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            // Already handled above
        }
    }

    Ok(())
}
