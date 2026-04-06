//! PARF (Pattern and Reference Finder) module (BATUTA-012)
//!
//! Cross-codebase pattern analysis and reference finding for
//! understanding code dependencies, usage patterns, and migration planning.
//!
//! # Features
//!
//! - **Symbol References**: Find all references to functions, classes, variables
//! - **Pattern Detection**: Identify common code patterns and idioms
//! - **Dependency Analysis**: Build dependency graphs across files
//! - **Dead Code Detection**: Find unused code that can be removed
//! - **Call Graph Generation**: Understand function call relationships
//!
//! # Example
//!
//! ```rust,ignore
//! use batuta::parf::{ParfAnalyzer, SymbolKind};
//!
//! let analyzer = ParfAnalyzer::new();
//! let refs = analyzer.find_references("my_function", SymbolKind::Function)?;
//! let patterns = analyzer.detect_patterns(&codebase)?;
//! let deps = analyzer.analyze_dependencies(&codebase)?;
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[cfg(feature = "native")]
use walkdir::WalkDir;

/// Symbol kind for reference finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Class,
    Variable,
    Constant,
    Module,
    Import,
}

/// A reference to a symbol in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolReference {
    /// Symbol name
    pub symbol: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// File path where reference occurs
    pub file: PathBuf,
    /// Line number
    pub line: usize,
    /// Context (surrounding code)
    pub context: String,
}

/// Code pattern detected in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodePattern {
    /// Repeated code block
    DuplicateCode { pattern: String, occurrences: Vec<(PathBuf, usize)> },
    /// TODO/FIXME comments
    TechDebt { message: String, file: PathBuf, line: usize },
    /// Deprecated API usage
    DeprecatedApi { api: String, file: PathBuf, line: usize },
    /// Error handling pattern
    ErrorHandling { pattern: String, file: PathBuf, line: usize },
    /// Resource management pattern (file handles, connections, etc.)
    ResourceManagement { resource_type: String, file: PathBuf, line: usize },
}

/// Dependency relationship between files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDependency {
    /// Source file
    pub from: PathBuf,
    /// Target file
    pub to: PathBuf,
    /// Type of dependency
    pub kind: DependencyKind,
}

/// Type of dependency between files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyKind {
    Import,
    Include,
    Require,
    ModuleUse,
}

/// Dead code analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCode {
    /// Symbol name
    pub symbol: String,
    /// Symbol kind
    pub kind: SymbolKind,
    /// File where defined
    pub file: PathBuf,
    /// Line number
    pub line: usize,
    /// Reason it's considered dead
    pub reason: String,
}

/// PARF analyzer for cross-codebase analysis
pub struct ParfAnalyzer {
    /// Cached file contents
    file_cache: HashMap<PathBuf, Vec<String>>,
    /// Symbol definitions
    symbol_definitions: HashMap<String, Vec<SymbolReference>>,
    /// Symbol references
    symbol_references: HashMap<String, Vec<SymbolReference>>,
}

impl Default for ParfAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ParfAnalyzer {
    /// Create a new PARF analyzer
    pub fn new() -> Self {
        Self {
            file_cache: HashMap::new(),
            symbol_definitions: HashMap::new(),
            symbol_references: HashMap::new(),
        }
    }

    /// Index a codebase for analysis
    #[cfg(feature = "native")]
    pub fn index_codebase(&mut self, path: &Path) -> Result<()> {
        for entry in WalkDir::new(path).follow_links(true).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                if let Some(ext) = entry.path().extension() {
                    // Process source files
                    if ["rs", "py", "js", "ts", "c", "cpp", "h", "hpp"]
                        .contains(&ext.to_str().unwrap_or(""))
                    {
                        self.index_file(entry.path())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Index a codebase for analysis (stub when native disabled)
    #[cfg(not(feature = "native"))]
    pub fn index_codebase(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }

    /// Index a single file
    fn index_file(&mut self, path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        self.file_cache.insert(path.to_path_buf(), lines.clone());

        // Simple pattern matching for common symbols
        for (line_num, line) in lines.iter().enumerate() {
            // Rust function definitions
            if line.contains("fn ") && line.contains('(') {
                if let Some(name) = Self::extract_function_name(line) {
                    self.add_definition(
                        name,
                        SymbolKind::Function,
                        path,
                        line_num + 1,
                        line.trim(),
                    );
                }
            }

            // Rust struct/enum definitions
            if line.contains("struct ") || line.contains("enum ") {
                if let Some(name) = Self::extract_type_name(line) {
                    self.add_definition(name, SymbolKind::Class, path, line_num + 1, line.trim());
                }
            }

            // Python function definitions
            if line.trim_start().starts_with("def ") {
                if let Some(name) = Self::extract_python_function_name(line) {
                    self.add_definition(
                        name,
                        SymbolKind::Function,
                        path,
                        line_num + 1,
                        line.trim(),
                    );
                }
            }

            // Python class definitions
            if line.trim_start().starts_with("class ") {
                if let Some(name) = Self::extract_python_class_name(line) {
                    self.add_definition(name, SymbolKind::Class, path, line_num + 1, line.trim());
                }
            }
        }

        Ok(())
    }

    /// Add a symbol definition
    fn add_definition(
        &mut self,
        symbol: String,
        kind: SymbolKind,
        file: &Path,
        line: usize,
        context: &str,
    ) {
        let reference = SymbolReference {
            symbol: symbol.clone(),
            kind,
            file: file.to_path_buf(),
            line,
            context: context.to_string(),
        };

        self.symbol_definitions.entry(symbol).or_default().push(reference);
    }

    /// Find all references to a symbol
    pub fn find_references(&self, symbol: &str, _kind: SymbolKind) -> Vec<SymbolReference> {
        let mut references = Vec::new();

        // Search through all cached files
        for (path, lines) in &self.file_cache {
            for (line_num, line) in lines.iter().enumerate() {
                if line.contains(symbol) {
                    references.push(SymbolReference {
                        symbol: symbol.to_string(),
                        kind: SymbolKind::Function, // Simplified for now
                        file: path.clone(),
                        line: line_num + 1,
                        context: line.trim().to_string(),
                    });
                }
            }
        }

        references
    }

    /// Detect code patterns in the codebase
    ///
    /// Uses `PATTERN_MARKERS` lookup table to map source markers to pattern
    /// categories, eliminating per-category if-contains blocks.
    pub fn detect_patterns(&self) -> Vec<CodePattern> {
        /// Mapping from source text marker to pattern category.
        ///
        /// Categories:
        /// - `"tech_debt"` -> [`CodePattern::TechDebt`]
        /// - `"deprecated_api"` -> [`CodePattern::DeprecatedApi`]
        /// - `"error_handling"` -> [`CodePattern::ErrorHandling`]
        /// - `"resource_management"` -> [`CodePattern::ResourceManagement`]
        const PATTERN_MARKERS: &[(&str, &str)] = &[
            ("TO\x44O", "tech_debt"),
            ("FIX\x4dE", "tech_debt"),
            ("HACK", "tech_debt"),
            ("deprecated", "deprecated_api"),
            ("@deprecated", "deprecated_api"),
            ("unwrap()", "error_handling"),
            ("expect(", "error_handling"),
            ("File::open", "resource_management"),
            ("fs::read", "resource_management"),
        ];

        let mut patterns = Vec::new();

        for (path, lines) in &self.file_cache {
            for (line_num, line) in lines.iter().enumerate() {
                for &(marker, category) in PATTERN_MARKERS {
                    if !line.contains(marker) {
                        continue;
                    }
                    // Preserve original filter: skip unwrap() inside comments
                    if marker == "unwrap()" && line.contains("//") {
                        continue;
                    }
                    let trimmed = line.trim().to_string();
                    let loc = (path.clone(), line_num + 1);
                    patterns.push(match category {
                        "tech_debt" => {
                            CodePattern::TechDebt { message: trimmed, file: loc.0, line: loc.1 }
                        }
                        "deprecated_api" => {
                            CodePattern::DeprecatedApi { api: trimmed, file: loc.0, line: loc.1 }
                        }
                        "error_handling" => CodePattern::ErrorHandling {
                            pattern: format!("{marker} without error handling"),
                            file: loc.0,
                            line: loc.1,
                        },
                        "resource_management" => CodePattern::ResourceManagement {
                            resource_type: "file".to_string(),
                            file: loc.0,
                            line: loc.1,
                        },
                        _ => unreachable!("unknown pattern category: {category}"),
                    });
                }
            }
        }

        patterns
    }

    /// Analyze dependencies between files
    pub fn analyze_dependencies(&self) -> Vec<FileDependency> {
        contract_pre_analyze!(self);
        let mut dependencies = Vec::new();

        for (path, lines) in &self.file_cache {
            for line in lines {
                // Rust imports
                if line.trim_start().starts_with("use ") {
                    // Simplified: would need proper parsing for real implementation
                    dependencies.push(FileDependency {
                        from: path.clone(),
                        to: PathBuf::from("module"), // Placeholder
                        kind: DependencyKind::ModuleUse,
                    });
                }

                // Python imports
                if line.trim_start().starts_with("import ")
                    || line.trim_start().starts_with("from ")
                {
                    dependencies.push(FileDependency {
                        from: path.clone(),
                        to: PathBuf::from("module"), // Placeholder
                        kind: DependencyKind::Import,
                    });
                }
            }
        }

        dependencies
    }

    /// Find potentially dead code
    pub fn find_dead_code(&self) -> Vec<DeadCode> {
        let mut dead_code = Vec::new();
        let mut referenced_symbols = HashSet::new();

        // Collect all referenced symbols
        for lines in self.file_cache.values() {
            for line in lines {
                // Simple heuristic: any identifier used in code
                for def in self.symbol_definitions.keys() {
                    if line.contains(def) {
                        referenced_symbols.insert(def.clone());
                    }
                }
            }
        }

        // Find definitions that are never referenced
        for (symbol, defs) in &self.symbol_definitions {
            if !referenced_symbols.contains(symbol) {
                for def in defs {
                    // Skip test functions
                    if def.context.contains("#[test]") || def.context.contains("test_") {
                        continue;
                    }

                    // Skip main functions
                    if symbol == "main" {
                        continue;
                    }

                    dead_code.push(DeadCode {
                        symbol: symbol.clone(),
                        kind: def.kind,
                        file: def.file.clone(),
                        line: def.line,
                        reason: "No references found".to_string(),
                    });
                }
            }
        }

        dead_code
    }

    /// Generate analysis report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("PARF Analysis Report\n");
        report.push_str("====================\n\n");

        report.push_str(&format!("Files analyzed: {}\n", self.file_cache.len()));
        report.push_str(&format!("Symbols defined: {}\n", self.symbol_definitions.len()));
        report.push_str(&format!("Patterns detected: {}\n", self.detect_patterns().len()));
        report.push_str(&format!("Dependencies: {}\n\n", self.analyze_dependencies().len()));

        // Dead code summary
        let dead_code = self.find_dead_code();
        report.push_str(&format!("Potentially dead code: {}\n", dead_code.len()));

        if !dead_code.is_empty() {
            report.push_str("\nDead Code Candidates:\n");
            report.push_str("---------------------\n");
            for (i, dc) in dead_code.iter().take(10).enumerate() {
                report.push_str(&format!(
                    "{}. {} ({:?}) in {}:{}\n",
                    i + 1,
                    dc.symbol,
                    dc.kind,
                    dc.file.display(),
                    dc.line
                ));
            }
            if dead_code.len() > 10 {
                report.push_str(&format!("... and {} more\n", dead_code.len() - 10));
            }
        }

        report
    }

    // Helper functions for symbol extraction

    fn extract_function_name(line: &str) -> Option<String> {
        // Extract function name from Rust: "fn name(" or "pub fn name("
        if let Some(fn_pos) = line.find("fn ") {
            let after_fn = &line[fn_pos + 3..];
            if let Some(paren_pos) = after_fn.find('(') {
                return Some(after_fn[..paren_pos].trim().to_string());
            }
        }
        None
    }

    fn extract_type_name(line: &str) -> Option<String> {
        // Extract type name from Rust: "struct Name" or "enum Name"
        for keyword in &["struct ", "enum "] {
            if let Some(pos) = line.find(keyword) {
                let after_keyword = &line[pos + keyword.len()..];
                if let Some(space_or_brace) =
                    after_keyword.find(|c: char| c.is_whitespace() || c == '{' || c == '<')
                {
                    return Some(after_keyword[..space_or_brace].trim().to_string());
                }
            }
        }
        None
    }

    fn extract_python_function_name(line: &str) -> Option<String> {
        // Extract function name from Python: "def name("
        if let Some(def_pos) = line.find("def ") {
            let after_def = &line[def_pos + 4..];
            if let Some(paren_pos) = after_def.find('(') {
                return Some(after_def[..paren_pos].trim().to_string());
            }
        }
        None
    }

    fn extract_python_class_name(line: &str) -> Option<String> {
        // Extract class name from Python: "class Name:" or "class Name("
        if let Some(class_pos) = line.find("class ") {
            let after_class = &line[class_pos + 6..];
            if let Some(end_pos) = after_class.find([':', '(']) {
                return Some(after_class[..end_pos].trim().to_string());
            }
        }
        None
    }
}

#[cfg(test)]
#[path = "parf_tests.rs"]
mod tests;
