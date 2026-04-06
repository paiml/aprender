# `batuta content`

Content creation tooling for generating structured prompts for educational and technical content.

## Overview

The `content` command provides tools for generating LLM prompts that follow Toyota Way principles, ensuring high-quality, structured content generation.

## Subcommands

### `batuta content emit`

Generate a structured prompt for content creation.

```bash
batuta content emit [OPTIONS] --type <TYPE>
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-t` | Content type: `hlo`, `dlo`, `bch`, `blp`, `pdm` |
| `--title` | | Title or topic for the content |
| `--audience` | | Target audience |
| `--word-count` | | Target word count |
| `--level` | `-l` | Course level for detailed outlines: `short`, `standard`, `extended` |
| `--source-context` | | Source context paths (comma-separated) |
| `--show-budget` | | Show token budget breakdown |
| `--output` | `-o` | Output file (default: stdout) |

**Content Types:**

| Code | Name | Format | Length |
|------|------|--------|--------|
| `hlo` | High-Level Outline | YAML/Markdown | 200-1000 lines |
| `dlo` | Detailed Outline | YAML/Markdown | 200-1000 lines |
| `bch` | Book Chapter | Markdown (mdBook) | 2000-5000 words |
| `blp` | Blog Post | Markdown (Zola) | 1000-2500 words |
| `pdm` | Presentar Demo | YAML/Markdown | N/A |

### Course Levels

For detailed outlines (`dlo`), configure the course structure using `--level`:

| Level | Weeks | Modules | Videos/Module | Weekly Objectives |
|-------|-------|---------|---------------|-------------------|
| `short` | 1 | 2 | 3 | No |
| `standard` | 3 | 3 | 5 | Yes (3 per week) |
| `extended` | 6 | 6 | 5 | Yes (3 per week) |

All courses include:
- Course description (2-3 sentences)
- 3 course-level learning objectives
- Per module: videos + quiz + reading + lab

**Examples:**

```bash
# Short course (1 week, 2 modules)
batuta content emit -t dlo --title "Quick Start" --level short

# Standard course (3 weeks, 3 modules) - default
batuta content emit -t dlo --title "Complete Course"

# Extended course (6 weeks, 6 modules)
batuta content emit -t dlo --title "Masterclass" --level extended

# Book chapter with audience
batuta content emit -t bch --title "Error Handling" --audience "Beginners"

# Blog post with word count
batuta content emit -t blp --title "Why Rust?" --word-count 1500
```

### `batuta content validate`

Validate generated content against quality constraints.

```bash
batuta content validate --type <TYPE> <FILE>
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-t` | Content type to validate against |
| `--llm-judge` | | Use LLM-as-a-Judge for style validation |

**Example:**

```bash
batuta content validate -t bch chapter.md
```

### `batuta content types`

List all available content types.

```bash
batuta content types
```

## Toyota Way Integration

The content module implements Toyota Way principles:

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | LLM-as-a-Judge validation catches quality issues |
| **Poka-Yoke** | Structural constraints in templates prevent mistakes |
| **Genchi Genbutsu** | Source context mandate grounds content in reality |
| **Heijunka** | Token budgeting levels context usage |
| **Kaizen** | Dynamic template composition enables improvement |

## Output Schema (Detailed Outline)

```yaml
type: detailed_outline
version: "1.0"
course:
  title: string
  description: string (2-3 sentences)
  duration_weeks: int
  total_modules: int
  learning_objectives:
    - objective: string
    - objective: string
    - objective: string
weeks:  # Only for standard/extended
  - week: 1
    learning_objectives:
      - objective: string
      - objective: string
      - objective: string
modules:
  - id: module_1
    week: 1
    title: string
    description: string
    learning_objectives:
      - objective: string
    videos:
      - id: video_1_1
        title: string
        duration_minutes: int (5-15)
    reading:
      title: string
      duration_minutes: int (15-30)
    quiz:
      title: string
      num_questions: int (5-10)
    lab:
      title: string
      duration_minutes: int (30-60)
```

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [CLI Overview](./cli-overview.md)
