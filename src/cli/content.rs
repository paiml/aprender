//! Content command implementations
//!
//! This module contains all content creation-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::content;
use std::path::PathBuf;
use std::str::FromStr;

/// Content creation subcommands
#[derive(Debug, Clone, clap::Subcommand)]
pub enum ContentCommand {
    /// Emit a prompt for content generation
    Emit {
        /// Content type (hlo, dlo, bch, blp, pdm)
        #[arg(long, short = 't')]
        r#type: String,

        /// Title or topic
        #[arg(long)]
        title: Option<String>,

        /// Target audience
        #[arg(long)]
        audience: Option<String>,

        /// Target word count
        #[arg(long)]
        word_count: Option<usize>,

        /// Source context paths (comma-separated)
        #[arg(long)]
        source_context: Option<String>,

        /// Show token budget breakdown
        #[arg(long)]
        show_budget: bool,

        /// Output file (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// Course level for detailed outlines (short, standard, extended)
        #[arg(long, short = 'l')]
        level: Option<String>,
    },

    /// Validate generated content
    Validate {
        /// Content type (hlo, dlo, bch, blp, pdm)
        #[arg(long, short = 't')]
        r#type: String,

        /// File to validate
        file: PathBuf,

        /// Use LLM-as-a-Judge for style validation
        #[arg(long)]
        llm_judge: bool,
    },

    /// List available content types
    Types,
}

/// Main content command dispatcher
pub fn cmd_content(command: ContentCommand) -> anyhow::Result<()> {
    match command {
        ContentCommand::Emit {
            r#type,
            title,
            audience,
            word_count,
            source_context,
            show_budget,
            output,
            level,
        } => {
            cmd_content_emit(
                &r#type,
                title,
                audience,
                word_count,
                source_context,
                show_budget,
                output,
                level,
            )?;
        }
        ContentCommand::Validate { r#type, file, llm_judge } => {
            cmd_content_validate(&r#type, &file, llm_judge)?;
        }
        ContentCommand::Types => {
            cmd_content_types()?;
        }
    }
    Ok(())
}

fn cmd_content_emit(
    content_type: &str,
    title: Option<String>,
    audience: Option<String>,
    word_count: Option<usize>,
    _source_context: Option<String>,
    show_budget: bool,
    output: Option<PathBuf>,
    level: Option<String>,
) -> anyhow::Result<()> {
    use content::{ContentType, CourseLevel, EmitConfig, PromptEmitter};

    let ct = ContentType::from_str(content_type).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut config = EmitConfig::new(ct);
    if let Some(t) = title {
        config = config.with_title(t);
    }
    if let Some(a) = audience {
        config = config.with_audience(a);
    }
    if let Some(wc) = word_count {
        config = config.with_word_count(wc);
    }
    if let Some(lvl) = level {
        let course_level: CourseLevel =
            lvl.parse().map_err(|e: content::ContentError| anyhow::anyhow!("{}", e))?;
        config = config.with_course_level(course_level);
    }
    config.show_budget = show_budget;

    let emitter = PromptEmitter::new();
    let prompt = emitter.emit(&config).map_err(|e| anyhow::anyhow!("{}", e))?;

    if let Some(path) = output {
        std::fs::write(&path, &prompt)?;
        println!("{}", format!("Prompt written to: {}", path.display()).green());
    } else {
        println!("{}", prompt);
    }

    Ok(())
}

fn cmd_content_validate(
    content_type: &str,
    file: &std::path::Path,
    _llm_judge: bool,
) -> anyhow::Result<()> {
    use content::{ContentType, ContentValidator};

    let ct = ContentType::from_str(content_type).map_err(|e| anyhow::anyhow!("{}", e))?;

    let content = std::fs::read_to_string(file)?;
    let validator = ContentValidator::new(ct);
    let result = validator.validate(&content);

    println!("{}", format!("Validating {} as {}...", file.display(), ct.name()).cyan());
    println!();
    println!("{}", result.format_display());

    if result.has_critical() || result.has_errors() {
        anyhow::bail!("Validation failed with critical errors");
    }

    Ok(())
}

fn cmd_content_types() -> anyhow::Result<()> {
    use content::ContentType;

    println!("{}", "Content Types".bright_cyan().bold());
    println!("{}", "=".repeat(60));
    println!();
    println!("{:<6} {:<22} {:<20} Target Length", "Code", "Name", "Output Format");
    println!("{}", "-".repeat(60));

    for ct in ContentType::all() {
        let range = ct.target_length();
        let length_str = if range.start == 0 && range.end == 0 {
            "N/A".to_string()
        } else {
            let unit = if matches!(ct, ContentType::HighLevelOutline | ContentType::DetailedOutline)
            {
                "lines"
            } else {
                "words"
            };
            format!("{}-{} {}", range.start, range.end, unit)
        };
        println!("{:<6} {:<22} {:<20} {}", ct.code(), ct.name(), ct.output_format(), length_str);
    }

    println!();
    println!("{}", "Usage:".bright_yellow());
    println!("  batuta content emit -t hlo --title \"My Course\"");
    println!("  batuta content emit -t bch --title \"Chapter 1\" --word-count 4000");
    println!("  batuta content validate -t bch chapter.md");

    Ok(())
}
