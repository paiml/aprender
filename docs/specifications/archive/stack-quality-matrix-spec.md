# Stack Quality Matrix Specification
## PAIML Stack A+ Quality Enforcement System

**Version:** 1.0.0
**Date:** 2025-12-07
**Authors:** Pragmatic AI Labs
**Status:** Draft
**GitHub Issues:** #10 (Stack Orchestration)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Quality Matrix Definition](#3-quality-matrix-definition)
4. [Command Specifications](#4-command-specifications)
5. [Scoring Categories](#5-scoring-categories)
6. [Implementation Design](#6-implementation-design)
7. [Toyota Way Analysis](#7-toyota-way-analysis)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [References](#9-references)

---

## 1. Executive Summary

The **Stack Quality Matrix** is a comprehensive quality enforcement system that ensures every component in the PAIML stack meets **A+ quality standards** before release. It integrates with `pmat` (Pragmatic Metrics and Testing) to validate four critical dimensions:

| Dimension | Tool | A+ Threshold | Description |
|-----------|------|--------------|-------------|
| **Rust Project Score** | `pmat rust-project-score` | 105-114/114 | Code quality, testing, docs |
| **Repository Score** | `pmat repo-score` | 95-110/110 | CI/CD, hygiene, automation |
| **README Score** | `pmat repo-score` (Category A) | 18-20/20 | Documentation completeness |
| **Hero Image** | `batuta stack quality` | Present & Valid | Visual branding presence |

### Value Proposition

- **Zero substandard releases** - All components validated before publishing
- **Consistent quality bar** - Every PAIML crate meets identical A+ standards
- **Visual verification** - Hero images ensure professional presentation
- **Automated enforcement** - No manual quality checks required

### Key Command

```bash
# Validate entire stack quality
batuta stack quality --verify

# Example output
PAIML Stack Quality Matrix
══════════════════════════════════════════════════════════════════

Component          Rust    Repo    README   Hero     Status
─────────────────────────────────────────────────────────────────
trueno            107/114  98/110   20/20    ✓       A+ ✅
aprender          109/114  96/110   19/20    ✓       A+ ✅
batuta            104/114  94/110   18/20    ✗       A  ⚠️
entrenar           89/114  82/110   15/20    ✗       B+ ❌

Summary: 2/4 components meet A+ quality standard
Action Required: batuta, entrenar need attention before release
```

---

## 2. Introduction

### 2.1 Purpose

This specification defines the **Stack Quality Matrix** system, which enforces A+ quality standards across all PAIML stack components. It ensures:

1. Every released crate meets production-grade quality thresholds
2. Documentation is complete and professionally presented
3. Visual branding (hero images) is consistent
4. Quality gates prevent substandard releases

### 2.2 Problem Statement

**Current State:**
- Quality varies significantly across PAIML stack components
- Some crates lack hero images or professional documentation
- No unified enforcement of quality standards
- Manual inspection required for quality validation

**Incident Analysis:**
```
Component: alimentar v0.2.0
Issue: Released with 67% test coverage, no hero image
Impact: Users perceive inconsistent quality across stack
Root Cause: No automated quality enforcement
```

### 2.3 Goals

1. **Unified Quality Bar** - All components meet A+ standards
2. **Automated Validation** - Quality checked on every release
3. **Visual Consistency** - Hero images required for branding
4. **Release Blocking** - Substandard components cannot publish

### 2.4 PAIML Stack Components (25 Total)

```
LAYER 0 - COMPUTE PRIMITIVES (5)
├── trueno              SIMD tensor operations
├── trueno-viz          Visualization
├── trueno-db           Time-series database
├── trueno-graph        Graph database
└── trueno-rag          RAG pipeline

LAYER 1 - ML ALGORITHMS (3)
├── aprender            ML algorithms, loss functions
├── aprender-shell      Interactive shell
└── aprender-tsp        TSP solver

LAYER 2 - TRAINING & INFERENCE (2)
├── entrenar            Training orchestration
└── realizar            GGUF inference

LAYER 3 - TRANSPILERS (4)
├── depyler             Python to Rust transpiler
├── decy                Compiler/analyzer
├── ruchy               Scripting runtime
└── bashrs              Bash to Rust (planned)

LAYER 4 - ORCHESTRATION (3)
├── batuta              Stack orchestration (this crate)
├── repartir            Distributed computing
└── pforge              Project forge

LAYER 5 - QUALITY (3)
├── certeza             Quality validation
├── renacer             Distributed tracing
└── pmat                Pragmatic metrics

LAYER 6 - DATA & MLOPS (2)
├── alimentar           Data loading
└── pacha               MCP agent toolkit

LAYER 7 - PRESENTATION (3)
├── presentar           Documentation generation
├── sovereign-ai-stack-book   Book/examples
└── [3 cookbooks]       apr/alm/pres-cookbook
```

---

## 3. Quality Matrix Definition

### 3.1 A+ Quality Thresholds

| Metric | Score Range | Grade | Status |
|--------|-------------|-------|--------|
| **Rust Project Score** | 105-114 | A+ | Required for release |
| | 95-104 | A | Acceptable with waiver |
| | 85-94 | A- | Minimum production (PMAT standard) |
| | <85 | B or below | **Release blocked** |
| **Repository Score** | 95-110 | A+ | Required for release |
| | 90-94 | A | Acceptable with waiver |
| | 85-89 | A- | Minimum production |
| | <85 | B or below | **Release blocked** |
| **README Score** | 18-20 | A+ | Required for release |
| | 16-17 | A | Acceptable |
| | 14-15 | A- | Minimum |
| | <14 | Below | **Release blocked** |
| **Hero Image** | Present | Pass | Required |
| | Missing | Fail | **Release blocked** |

### 3.2 Grading Algorithm

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityGrade {
    APlus,    // A+ (95-100% of max)
    A,        // A  (90-94%)
    AMinus,   // A- (85-89%) - PMAT minimum
    BPlus,    // B+ (80-84%)
    B,        // B  (70-79%)
    C,        // C  (60-69%)
    D,        // D  (50-59%)
    F,        // F  (0-49%)
}

impl QualityGrade {
    pub fn from_rust_project_score(score: u32) -> Self {
        const MAX: u32 = 114;
        match score {
            105..=114 => Self::APlus,
            95..=104 => Self::A,
            85..=94 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    pub fn from_repo_score(score: u32) -> Self {
        const MAX: u32 = 110;
        match score {
            95..=110 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    pub fn from_readme_score(score: u32) -> Self {
        const MAX: u32 = 20;
        match score {
            18..=20 => Self::APlus,
            16..=17 => Self::A,
            14..=15 => Self::AMinus,
            12..=13 => Self::BPlus,
            10..=11 => Self::B,
            8..=9 => Self::C,
            6..=7 => Self::D,
            _ => Self::F,
        }
    }

    pub fn is_release_ready(&self) -> bool {
        matches!(self, Self::APlus | Self::A | Self::AMinus)
    }

    pub fn is_a_plus(&self) -> bool {
        matches!(self, Self::APlus)
    }
}
```

### 3.3 Composite Quality Score

The **Stack Quality Index (SQI)** is a weighted composite score:

```
SQI = (0.40 × Rust Score) + (0.30 × Repo Score) + (0.20 × README Score) + (0.10 × Hero)

Where:
- Rust Score normalized to 0-100
- Repo Score normalized to 0-100
- README Score normalized to 0-100
- Hero = 100 if present, 0 if missing
```

**SQI Grading:**
| SQI Range | Grade | Release Status |
|-----------|-------|----------------|
| 95-100 | A+ | Approved |
| 90-94 | A | Approved |
| 85-89 | A- | Approved (minimum) |
| <85 | Below | **Blocked** |

---

## 4. Command Specifications

### 4.1 `batuta stack quality` - Quality Matrix Check

**Purpose:** Validate all PAIML stack components against the A+ quality matrix.

**Usage:**

```bash
# Check all components
batuta stack quality

# Check specific component
batuta stack quality --component trueno

# Strict A+ only (fail if any component below A+)
batuta stack quality --strict

# Output as JSON for CI
batuta stack quality --format json

# Verify hero images exist
batuta stack quality --verify-hero

# Show detailed breakdown
batuta stack quality --verbose

# Check and generate report
batuta stack quality --report quality-report.md
```

**CLI Definition:**

```rust
#[derive(clap::Args)]
pub struct QualityCommand {
    /// Specific component to check
    #[arg(long)]
    component: Option<String>,

    /// Require A+ for all components
    #[arg(long)]
    strict: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Verify hero images exist and are valid
    #[arg(long)]
    verify_hero: bool,

    /// Show detailed score breakdown
    #[arg(long, short)]
    verbose: bool,

    /// Generate markdown report
    #[arg(long)]
    report: Option<PathBuf>,

    /// Minimum grade required (default: A-)
    #[arg(long, default_value = "a-minus")]
    min_grade: QualityGrade,
}
```

**Output Example (Text):**

```
PAIML Stack Quality Matrix
══════════════════════════════════════════════════════════════════════════════

Scanning 25 components...

LAYER 0 - COMPUTE PRIMITIVES
─────────────────────────────────────────────────────────────────────────────
  Component          Rust     Repo     README   Hero     SQI    Grade
  ─────────────────────────────────────────────────────────────────────────
  trueno            107/114   98/110   20/20    ✓        97.2   A+ ✅
  trueno-viz        105/114   95/110   18/20    ✓        95.1   A+ ✅
  trueno-db         106/114   97/110   19/20    ✓        96.4   A+ ✅
  trueno-graph      108/114   96/110   20/20    ✓        97.0   A+ ✅
  trueno-rag        104/114   94/110   18/20    ✓        94.2   A  ⚠️

LAYER 1 - ML ALGORITHMS
─────────────────────────────────────────────────────────────────────────────
  aprender          109/114   99/110   20/20    ✓        98.1   A+ ✅
  aprender-shell    102/114   92/110   17/20    ✓        92.8   A  ⚠️
  aprender-tsp      105/114   95/110   18/20    ✓        95.1   A+ ✅

LAYER 4 - ORCHESTRATION
─────────────────────────────────────────────────────────────────────────────
  batuta            104/114   94/110   18/20    ✗        91.8   A  ⚠️
    └── ⚠️  Missing hero image at README.md or docs/hero.svg

LAYER 6 - DATA & MLOPS
─────────────────────────────────────────────────────────────────────────────
  entrenar           89/114   82/110   15/20    ✗        79.6   B+ ❌
    └── ❌ Rust score below A- threshold (85)
    └── ❌ Repo score below A- threshold (85)
    └── ❌ README score below A- threshold (14)
    └── ❌ Missing hero image

══════════════════════════════════════════════════════════════════════════════
SUMMARY
══════════════════════════════════════════════════════════════════════════════

Quality Distribution:
  A+  ████████████████████  18 components (72%)
  A   ████████              5 components (20%)
  A-  ██                    1 component  (4%)
  B+  █                     1 component  (4%)

Stack Quality Index: 94.2 (A)

Release Status: ⚠️  CONDITIONAL
  ✓ 24/25 components meet minimum A- standard
  ✗ 1 component (entrenar) below threshold

Recommended Actions:
  1. Add hero image to batuta README.md
  2. Improve entrenar test coverage (currently 67%, need 90%+)
  3. Add missing README sections to entrenar
  4. Run: batuta stack quality --component entrenar --verbose
```

**Output Example (JSON):**

```json
{
  "timestamp": "2025-12-07T10:30:00Z",
  "stack_quality_index": 94.2,
  "overall_grade": "A",
  "release_ready": true,
  "release_ready_strict": false,
  "summary": {
    "total_components": 25,
    "a_plus": 18,
    "a": 5,
    "a_minus": 1,
    "below_threshold": 1
  },
  "components": [
    {
      "name": "trueno",
      "layer": "compute",
      "scores": {
        "rust_project": { "score": 107, "max": 114, "grade": "A+" },
        "repo": { "score": 98, "max": 110, "grade": "A+" },
        "readme": { "score": 20, "max": 20, "grade": "A+" },
        "hero_image": { "present": true, "path": "docs/hero.svg", "valid": true }
      },
      "sqi": 97.2,
      "grade": "A+",
      "release_ready": true,
      "issues": []
    },
    {
      "name": "entrenar",
      "layer": "data_mlops",
      "scores": {
        "rust_project": { "score": 89, "max": 114, "grade": "B+" },
        "repo": { "score": 82, "max": 110, "grade": "B" },
        "readme": { "score": 15, "max": 20, "grade": "A-" },
        "hero_image": { "present": false, "path": null, "valid": false }
      },
      "sqi": 79.6,
      "grade": "B+",
      "release_ready": false,
      "issues": [
        {
          "type": "rust_score_below_threshold",
          "message": "Rust project score 89 below A- threshold (85)",
          "severity": "error",
          "recommendation": "Improve test coverage and documentation"
        },
        {
          "type": "missing_hero_image",
          "message": "No hero image found",
          "severity": "error",
          "recommendation": "Add hero.svg to docs/ or image at top of README.md"
        }
      ]
    }
  ],
  "blocked_releases": ["entrenar"],
  "warnings": ["batuta", "aprender-shell", "trueno-rag"]
}
```

### 4.2 `batuta stack quality-gate` - CI/CD Integration

**Purpose:** Exit with non-zero status if quality standards not met.

```bash
# Use in CI/CD
batuta stack quality-gate --min-grade a-minus

# Strict A+ enforcement
batuta stack quality-gate --strict

# Check specific component before release
batuta stack quality-gate --component trueno
```

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | All components meet quality threshold |
| 1 | One or more components below threshold |
| 2 | Configuration or execution error |

---

## 5. Scoring Categories

### 5.1 Rust Project Score (0-114)

Evaluated via `pmat rust-project-score`. Categories:

| Category | Max Points | A+ Requirement |
|----------|------------|----------------|
| Rust Tooling Compliance | 25 | ≥23 (clippy clean, rustfmt, audit) |
| Code Quality | 26 | ≥24 (low complexity, no unsafe) |
| Testing Excellence | 20 | ≥18 (90%+ coverage, mutation testing) |
| Documentation | 15 | ≥14 (rustdoc, README, changelog) |
| Performance | 10 | ≥9 (criterion benchmarks) |
| Dependency Health | 12 | ≥11 (minimal deps, no vulnerabilities) |
| Formal Verification | 8 | ≥5 (Miri clean) |
| **TOTAL** | **114** | **≥105 for A+** |

**Verification Command:**
```bash
pmat rust-project-score --path ../trueno
```

### 5.2 Repository Score (0-110)

Evaluated via `pmat repo-score`. Categories:

| Category | Max Points | A+ Requirement |
|----------|------------|----------------|
| A: Documentation Quality | 20 | ≥18 (README complete, no broken links) |
| B: Pre-commit Hooks | 20 | ≥18 (hooks present, fast) |
| C: Repository Hygiene | 10 | ≥9 (no cruft, no IDE files) |
| D: Build & Test Automation | 25 | ≥23 (Makefile, all targets) |
| E: Continuous Integration | 20 | ≥18 (workflows configured) |
| F: PMAT Compliance | 5 | ≥5 (.pmat-gates.toml valid) |
| Bonus: Advanced Testing | 10 | ≥5 (property/fuzz/mutation) |
| **TOTAL** | **110** | **≥95 for A+** |

**Verification Command:**
```bash
pmat repo-score ../trueno
```

### 5.3 README Score (0-20)

Subset of Repository Score, Category A:

| Subcategory | Max Points | Requirement |
|-------------|------------|-------------|
| A1: README Accuracy | 10 | File exists, no broken links |
| A2: README Comprehensiveness | 10 | 5 required sections present |

**Required Sections (2 points each):**
1. **Overview/Description** - Project purpose and features
2. **Installation** - Clear installation instructions
3. **Usage** - Getting started examples
4. **Contributing** - How to contribute
5. **License** - License information

**Perfect README Template:**
```markdown
# Project Name

![Hero Image](docs/hero.svg)

[![Crates.io](https://img.shields.io/crates/v/project.svg)](https://crates.io/crates/project)
[![Documentation](https://docs.rs/project/badge.svg)](https://docs.rs/project)
[![CI](https://github.com/paiml/project/workflows/CI/badge.svg)](https://github.com/paiml/project/actions)

## Overview

Brief description of what the project does and why it matters.

## Installation

```bash
cargo add project
```

## Usage

```rust
use project::Feature;

let result = Feature::new().process();
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT OR Apache-2.0
```

### 5.4 Hero Image Requirements

**Purpose:** Visual branding and professional presentation.

**Requirements:**

| Attribute | Requirement |
|-----------|-------------|
| Location | `docs/hero.svg` (preferred) or `docs/hero.png`, OR first image in README.md |
| Format | SVG (preferred), PNG, JPG, or WebP |
| Minimum Size | 800x400 pixels (for raster formats) |
| Maximum Size | 2MB file size |
| Content | Project logo, architecture diagram, or feature showcase |
| Alt Text | Descriptive alt text required |

**Detection Algorithm:**

```rust
pub struct HeroImageResult {
    pub present: bool,
    pub path: Option<PathBuf>,
    pub format: Option<ImageFormat>,
    pub dimensions: Option<(u32, u32)>,
    pub file_size: Option<u64>,
    pub valid: bool,
    pub issues: Vec<String>,
}

impl HeroImageResult {
    pub fn detect(repo_path: &Path) -> Self {
        // Priority 1: Check docs/hero.svg (preferred)
        let hero_path = repo_path.join("docs/hero.svg");
        if hero_path.exists() {
            return Self::validate_image(&hero_path);
        }

        // Priority 2: Check docs/hero.* (other formats)
        for ext in &["png", "jpg", "jpeg", "webp"] {
            let path = repo_path.join(format!("docs/hero.{}", ext));
            if path.exists() {
                return Self::validate_image(&path);
            }
        }

        // Priority 3: Parse README.md for first image
        let readme_path = repo_path.join("README.md");
        if readme_path.exists() {
            if let Some(img_path) = Self::extract_first_image(&readme_path) {
                let full_path = repo_path.join(&img_path);
                if full_path.exists() {
                    return Self::validate_image(&full_path);
                }
            }
        }

        // No hero image found
        Self {
            present: false,
            path: None,
            format: None,
            dimensions: None,
            file_size: None,
            valid: false,
            issues: vec!["No hero image found".to_string()],
        }
    }

    fn validate_image(path: &Path) -> Self {
        let mut issues = Vec::new();

        // Check file size
        let file_size = std::fs::metadata(path).map(|m| m.len()).ok();
        if let Some(size) = file_size {
            if size > 2 * 1024 * 1024 {
                issues.push(format!("Image too large: {} bytes (max 2MB)", size));
            }
        }

        // Check dimensions (requires image crate)
        let dimensions = Self::get_dimensions(path);
        if let Some((w, h)) = dimensions {
            if w < 800 || h < 400 {
                issues.push(format!("Image too small: {}x{} (min 800x400)", w, h));
            }
        }

        Self {
            present: true,
            path: Some(path.to_path_buf()),
            format: Self::detect_format(path),
            dimensions,
            file_size,
            valid: issues.is_empty(),
            issues,
        }
    }
}
```

---

## 6. Implementation Design

### 6.1 Core Data Structures

```rust
/// Quality matrix result for a single component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentQuality {
    /// Component name
    pub name: String,

    /// Layer in the stack
    pub layer: StackLayer,

    /// Repository path
    pub path: PathBuf,

    /// Individual scores
    pub rust_project_score: Score,
    pub repo_score: Score,
    pub readme_score: Score,
    pub hero_image: HeroImageResult,

    /// Composite metrics
    pub stack_quality_index: f64,
    pub grade: QualityGrade,

    /// Release eligibility
    pub release_ready: bool,

    /// Issues found
    pub issues: Vec<QualityIssue>,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Score with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub value: u32,
    pub max: u32,
    pub grade: QualityGrade,
    pub breakdown: Option<ScoreBreakdown>,
}

/// Stack-wide quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackQualityReport {
    pub timestamp: DateTime<Utc>,
    pub components: Vec<ComponentQuality>,
    pub summary: QualitySummary,
    pub stack_quality_index: f64,
    pub overall_grade: QualityGrade,
    pub release_ready: bool,
    pub blocked_components: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySummary {
    pub total_components: usize,
    pub a_plus_count: usize,
    pub a_count: usize,
    pub a_minus_count: usize,
    pub below_threshold_count: usize,
    pub missing_hero_count: usize,
    pub avg_rust_score: f64,
    pub avg_repo_score: f64,
    pub avg_readme_score: f64,
}
```

### 6.2 Integration with PMAT

```rust
use std::process::Command;

pub struct PmatClient {
    pmat_path: PathBuf,
}

impl PmatClient {
    /// Run pmat rust-project-score
    pub fn rust_project_score(&self, repo_path: &Path) -> Result<RustProjectScore> {
        let output = Command::new(&self.pmat_path)
            .args(["rust-project-score", "--path", repo_path.to_str().unwrap()])
            .args(["--format", "json"])
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("pmat rust-project-score failed"));
        }

        let score: RustProjectScore = serde_json::from_slice(&output.stdout)?;
        Ok(score)
    }

    /// Run pmat repo-score
    pub fn repo_score(&self, repo_path: &Path) -> Result<RepoScore> {
        let output = Command::new(&self.pmat_path)
            .args(["repo-score", repo_path.to_str().unwrap()])
            .args(["--format", "json"])
            .output()?;

        if !output.status.success() {
            return Err(anyhow!("pmat repo-score failed"));
        }

        let score: RepoScore = serde_json::from_slice(&output.stdout)?;
        Ok(score)
    }

    /// Extract README score from repo-score
    pub fn readme_score(&self, repo_score: &RepoScore) -> ReadmeScore {
        let category_a = repo_score.categories.get("documentation").unwrap();
        ReadmeScore {
            accuracy: category_a.subcategories.get("a1_accuracy").unwrap().score,
            comprehensiveness: category_a.subcategories.get("a2_comprehensiveness").unwrap().score,
            total: category_a.score,
            max: 20,
        }
    }
}
```

### 6.3 Quality Matrix Runner

```rust
pub struct QualityMatrixRunner {
    pmat: PmatClient,
    config: QualityConfig,
}

impl QualityMatrixRunner {
    /// Check quality for all stack components
    pub async fn check_all(&self) -> Result<StackQualityReport> {
        let mut components = Vec::new();

        for component in PAIML_CRATES {
            let path = self.find_component_path(component)?;
            let quality = self.check_component(component, &path).await?;
            components.push(quality);
        }

        let summary = self.calculate_summary(&components);
        let sqi = self.calculate_stack_quality_index(&components);

        Ok(StackQualityReport {
            timestamp: Utc::now(),
            components,
            summary,
            stack_quality_index: sqi,
            overall_grade: QualityGrade::from_sqi(sqi),
            release_ready: sqi >= 85.0,
            blocked_components: self.find_blocked(&components),
            recommendations: self.generate_recommendations(&components),
        })
    }

    /// Check quality for single component
    pub async fn check_component(
        &self,
        name: &str,
        path: &Path,
    ) -> Result<ComponentQuality> {
        // Run PMAT checks in parallel
        let (rust_score, repo_score) = tokio::join!(
            self.pmat.rust_project_score(path),
            self.pmat.repo_score(path),
        );

        let rust_score = rust_score?;
        let repo_score = repo_score?;
        let readme_score = self.pmat.readme_score(&repo_score);
        let hero_image = HeroImageResult::detect(path);

        let sqi = self.calculate_sqi(&rust_score, &repo_score, &readme_score, &hero_image);
        let grade = QualityGrade::from_sqi(sqi);
        let issues = self.collect_issues(&rust_score, &repo_score, &readme_score, &hero_image);

        Ok(ComponentQuality {
            name: name.to_string(),
            layer: StackLayer::from_component(name),
            path: path.to_path_buf(),
            rust_project_score: rust_score.into(),
            repo_score: repo_score.into(),
            readme_score: readme_score.into(),
            hero_image,
            stack_quality_index: sqi,
            grade,
            release_ready: grade.is_release_ready(),
            issues,
            recommendations: self.generate_component_recommendations(&issues),
        })
    }

    /// Calculate Stack Quality Index
    fn calculate_sqi(
        &self,
        rust: &RustProjectScore,
        repo: &RepoScore,
        readme: &ReadmeScore,
        hero: &HeroImageResult,
    ) -> f64 {
        let rust_normalized = (rust.total as f64 / 114.0) * 100.0;
        let repo_normalized = (repo.total as f64 / 110.0) * 100.0;
        let readme_normalized = (readme.total as f64 / 20.0) * 100.0;
        let hero_normalized = if hero.valid { 100.0 } else { 0.0 };

        (0.40 * rust_normalized)
            + (0.30 * repo_normalized)
            + (0.20 * readme_normalized)
            + (0.10 * hero_normalized)
    }
}
```

### 6.4 Hero Image Generator (Bonus Feature)

```rust
/// Generate placeholder hero image for projects without one
pub struct HeroImageGenerator {
    template_path: PathBuf,
}

impl HeroImageGenerator {
    /// Generate hero image for a component
    pub fn generate(&self, component: &str, output_path: &Path) -> Result<()> {
        // Use SVG template with component name
        let svg = format!(r#"
<svg width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#1a1a2e"/>
  <text x="50%" y="40%" text-anchor="middle" fill="#eaeaea"
        font-family="monospace" font-size="72" font-weight="bold">
    {component}
  </text>
  <text x="50%" y="55%" text-anchor="middle" fill="#888888"
        font-family="monospace" font-size="24">
    PAIML Sovereign AI Stack
  </text>
  <text x="50%" y="90%" text-anchor="middle" fill="#4a4a6a"
        font-family="monospace" font-size="14">
    https://github.com/paiml/{component}
  </text>
</svg>
        "#, component = component);

        std::fs::write(output_path.with_extension("svg"), svg)?;
        Ok(())
    }
}
```

---

## 7. Toyota Way Analysis & Scientific Validation

### 7.1 Jidoka (Built-in Quality) - Stop the Line

**Principle:** Quality must be built in, not inspected in. Defects should be detected immediately, stopping the production line to fix the root cause (Liker, 2004).

**Implementation:** The `batuta stack quality-gate` command implements the "Andon Cord" for the software delivery pipeline. As supported by Duvall et al. (2007) and Humble & Farley (2010), automated quality gates that block release upon failure reduce downstream defects and technical debt.

```rust
fn check_release_readiness(component: &ComponentQuality) -> Result<(), ReleaseBlocker> {
    // STOP THE LINE: Any quality failure blocks release
    if !component.grade.is_release_ready() {
        return Err(ReleaseBlocker::GradeBelowThreshold {
            component: component.name.clone(),
            grade: component.grade,
            required: QualityGrade::AMinus,
        });
    }

    if !component.hero_image.valid {
        return Err(ReleaseBlocker::MissingHeroImage {
            component: component.name.clone(),
        });
    }

    Ok(())
}
```

### 7.2 Genchi Genbutsu (Go and See) - Empirical Verification

**Principle:** Decisions must be based on deep personal understanding of the facts (Genchi Genbutsu), not assumptions or reports (Liker, 2004).

**Implementation:** Instead of relying on manual checklists or self-reported status, `batuta` performs direct, empirical verification of the codebase using `pmat`. This aligns with findings by Nagappan et al. (2005) on the predictive value of objective code metrics for defect density.

```bash
# Don't assume quality - verify it empirically
pmat rust-project-score --path ../trueno
pmat repo-score ../trueno

# Batuta integrates these directly
batuta stack quality --verbose
```

### 7.3 Kaizen (Continuous Improvement) - Metrics for Growth

**Principle:** Continuous improvement (Kaizen) requires standardization and measurement. "Where there is no standard, there can be no Kaizen" (Ohno).

**Implementation:** The **Stack Quality Index (SQI)** provides the standard against which improvement is measured. By tracking these metrics over time, teams can identify regression or stagnation. This data-driven approach to process improvement is central to Lean Software Development (Poppendieck, 2003) and Toyota Kata (Rother, 2009).

```rust
pub struct QualityHistory {
    pub component: String,
    pub measurements: Vec<QualityMeasurement>,
}

impl QualityHistory {
    /// Generate improvement recommendations
    pub fn improvement_areas(&self) -> Vec<String> {
        let latest = self.measurements.last().unwrap();
        let mut areas = Vec::new();

        // Identify trending down metrics
        if self.is_declining("rust_project_score") {
            areas.push("Rust project score trending down - review test coverage".into());
        }

        // Identify consistently low metrics
        if latest.readme_score < 16 {
            areas.push("README score below A - add missing sections".into());
        }

        areas
    }
}
```

### 7.4 Visual Management (Mieruka)

**Principle:** Make problems visible so they are not hidden.

**Implementation:** The Hero Image requirement and the colored CLI output serve as "visual controls" (Mieruka). Research by Lethbridge et al. (2003) highlights that documentation presentation significantly impacts its utility and maintenance. Visual branding (Hero Image) signals professionalism and reduces cognitive load for new users.

---

## 8. Acceptance Criteria

### 8.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| QM-01 | Integrate pmat rust-project-score | P0 | Pending |
| QM-02 | Integrate pmat repo-score | P0 | Pending |
| QM-03 | Extract README score from repo-score | P0 | Pending |
| QM-04 | Detect hero image presence | P0 | Pending |
| QM-05 | Calculate Stack Quality Index | P0 | Pending |
| QM-06 | Generate text report | P0 | Pending |
| QM-07 | Generate JSON output | P1 | Pending |
| QM-08 | Quality gate exit codes for CI | P1 | Pending |
| QM-09 | Hero image validation (size, format) | P1 | Pending |
| QM-10 | Generate markdown report | P2 | Pending |
| QM-11 | Hero image generator | P2 | Pending |

### 8.2 Non-Functional Requirements

| ID | Requirement | Target | Status |
|----|-------------|--------|--------|
| NFR-01 | Full stack scan in <60 seconds | ≤60s | Pending |
| NFR-02 | Parallel PMAT execution | Yes | Pending |
| NFR-03 | Cache PMAT results (5 min TTL) | Yes | Pending |
| NFR-04 | Clear error messages | Yes | Pending |
| NFR-05 | Test coverage ≥90% | ≥90% | Pending |

### 8.3 Verification Commands

```bash
# Verify command works
batuta stack quality

# Verify JSON output
batuta stack quality --format json | jq '.stack_quality_index'

# Verify CI gate
batuta stack quality-gate --min-grade a-minus && echo "PASS" || echo "FAIL"

# Verify strict mode
batuta stack quality --strict
```

---

## 9. References

### 9.1 Peer-Reviewed Academic & Industry Support

1.  **Liker, J. K. (2004).** *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. (Foundational principles for Jidoka and Genchi Genbutsu).
2.  **Poppendieck, M., & Poppendieck, T. (2003).** *Lean Software Development: An Agile Toolkit*. Addison-Wesley Professional. (Application of Toyota principles to software engineering).
3.  **Duvall, P. M., Matyas, S., & Glover, A. (2007).** *Continuous Integration: Improving Software Quality and Reducing Risk*. Addison-Wesley Professional. (Support for automated quality gates).
4.  **Humble, J., & Farley, D. (2010).** *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*. Addison-Wesley Professional. (pipeline automation validation).
5.  **Nagappan, N., Ball, T., & Zeller, A. (2005).** "Mining metrics to predict component failures". *Proceedings of the 27th international conference on Software engineering* (ICSE '05), 452–461. (Validation of metrics for quality prediction).
6.  **McConnell, S. (2004).** *Code Complete: A Practical Handbook of Software Construction, Second Edition*. Microsoft Press. (Best practices for construction quality).
7.  **Lethbridge, T. C., Singer, J., & Forward, A. (2003).** "How Software Engineers Use Documentation: The State of the Practice". *IEEE Software*, 20(6), 35-39. (Importance of documentation quality).
8.  **Martin, R. C. (2008).** *Clean Code: A Handbook of Agile Software Craftsmanship*. Prentice Hall. (Code quality standards).
9.  **Rother, M. (2009).** *Toyota Kata: Managing People for Improvement, Adaptiveness and Superior Results*. McGraw-Hill. (Continuous improvement methodology).
10. **Beck, K. (2002).** *Test Driven Development: By Example*. Addison-Wesley Professional. (Testing excellence and quality assurance).

### 9.2 PMAT Documentation

- [PMAT Repo Score Specification](https://pmat-book.paiml.com/ch31-repo-score)
- [PMAT Rust Project Score](https://pmat-book.paiml.com/ch33-rust-project-score)
- [PMAT Quality Gates](https://pmat-book.paiml.com/ch40-quality-gates)

### 9.3 Related Specifications

- [Batuta Stack Specification](./batuta-stack-spec.md)
- [100-Point QA Checklist](./batuta-stack-0.1-100-point-qa-checklist.md)
- [Stack Tree View](./stack-tree-view.md)

---

**END OF SPECIFICATION**

**Document Version:** 1.0.0
**Last Updated:** 2025-12-07
**Authors:** Pragmatic AI Labs
**License:** MIT (code), CC-BY-4.0 (documentation)
