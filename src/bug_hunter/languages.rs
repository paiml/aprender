//! Multi-Language Pattern Support
//!
//! Provides language-specific bug patterns for Python, TypeScript, Go, and Rust.

use super::types::{DefectCategory, FindingSeverity};

/// Supported programming languages for bug hunting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    Go,
}

impl Language {
    /// Detect language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "ts" | "tsx" | "js" | "jsx" => Some(Language::TypeScript),
            "go" => Some(Language::Go),
            _ => None,
        }
    }

    /// Get file extensions for this language.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::Python => &["py"],
            Language::TypeScript => &["ts", "tsx", "js", "jsx"],
            Language::Go => &["go"],
        }
    }

    /// Get glob patterns for this language.
    pub fn glob_patterns(&self) -> Vec<&'static str> {
        match self {
            Language::Rust => vec!["**/*.rs"],
            Language::Python => vec!["**/*.py"],
            Language::TypeScript => vec!["**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"],
            Language::Go => vec!["**/*.go"],
        }
    }
}

/// A language-specific pattern.
pub struct LangPattern {
    pub pattern: &'static str,
    pub category: DefectCategory,
    pub severity: FindingSeverity,
    pub suspiciousness: f64,
    pub language: Option<Language>, // None = applies to all languages
}

/// Get patterns applicable to a specific language.
pub fn patterns_for_language(
    lang: Language,
) -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    let mut patterns = vec![];

    // Universal patterns (all languages)
    patterns.extend(universal_patterns());

    // Language-specific patterns
    match lang {
        Language::Rust => patterns.extend(rust_patterns()),
        Language::Python => patterns.extend(python_patterns()),
        Language::TypeScript => patterns.extend(typescript_patterns()),
        Language::Go => patterns.extend(go_patterns()),
    }

    patterns
}

/// Patterns that apply to all languages.
fn universal_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Universal debt markers
        ("TODO", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("FIXME", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("HACK", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("XXX", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("BUG", DefectCategory::LogicErrors, FindingSeverity::High, 0.7),
        // Hidden debt euphemisms
        ("placeholder", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("stub", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("dummy", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("temporary", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("hardcoded", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("workaround", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("tech debt", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
    ]
}

/// Rust-specific patterns.
fn rust_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        ("unwrap()", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.4),
        ("expect(", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        // SAFETY: no actual unsafe code -- string pattern for defect detection
        ("unsafe {", DefectCategory::MemorySafety, FindingSeverity::High, 0.7),
        ("transmute", DefectCategory::MemorySafety, FindingSeverity::High, 0.8),
        ("panic!", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("unreachable!", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("unimplemented!", DefectCategory::HiddenDebt, FindingSeverity::Critical, 0.9),
        ("todo!", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("#[ignore]", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        (".unwrap_or_else(|_|", DefectCategory::SilentDegradation, FindingSeverity::High, 0.7),
        ("Err(_) => {}", DefectCategory::SilentDegradation, FindingSeverity::High, 0.75),
    ]
}

/// Python-specific patterns.
fn python_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Exception handling
        ("except:", DefectCategory::SilentDegradation, FindingSeverity::High, 0.8),
        ("except Exception:", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.6),
        ("except BaseException:", DefectCategory::SilentDegradation, FindingSeverity::High, 0.8),
        ("pass  # TODO", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        // Security
        ("eval(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Critical, 0.95),
        ("exec(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Critical, 0.95),
        ("pickle.loads", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        ("shell=True", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.85),
        ("__import__", DefectCategory::SecurityVulnerabilities, FindingSeverity::Medium, 0.6),
        // Anti-patterns
        ("global ", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("import *", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("assert ", DefectCategory::TestDebt, FindingSeverity::Low, 0.3), // in production code
        ("# type: ignore", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.5),
        // Test debt
        ("@pytest.mark.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("@unittest.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("raise NotImplementedError", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
        // Threading issues
        ("threading.Thread(", DefectCategory::ConcurrencyBugs, FindingSeverity::Medium, 0.5),
    ]
}

/// TypeScript/JavaScript-specific patterns.
fn typescript_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Type safety
        ("any", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.5),
        ("as any", DefectCategory::TypeErrors, FindingSeverity::High, 0.7),
        ("// @ts-ignore", DefectCategory::TypeErrors, FindingSeverity::High, 0.75),
        ("// @ts-nocheck", DefectCategory::TypeErrors, FindingSeverity::Critical, 0.9),
        ("@ts-expect-error", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.5),
        // Security
        ("eval(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Critical, 0.95),
        ("innerHTML", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        (
            "dangerouslySetInnerHTML",
            DefectCategory::SecurityVulnerabilities,
            FindingSeverity::High,
            0.8,
        ),
        ("document.write", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        // Anti-patterns
        ("console.log", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        ("debugger", DefectCategory::LogicErrors, FindingSeverity::High, 0.7),
        ("== null", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("!= null", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        // Test debt
        ("it.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("describe.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("test.skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        (".only(", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        // Promise anti-patterns
        (".catch(() => {", DefectCategory::SilentDegradation, FindingSeverity::High, 0.75),
        (".catch(e => {})", DefectCategory::SilentDegradation, FindingSeverity::High, 0.8),
    ]
}

/// Go-specific patterns.
fn go_patterns() -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Error handling
        ("_ = err", DefectCategory::SilentDegradation, FindingSeverity::Critical, 0.9),
        ("err != nil { return", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3), // OK pattern, low priority
        ("panic(", DefectCategory::LogicErrors, FindingSeverity::High, 0.7),
        ("log.Fatal", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        // Concurrency
        ("go func()", DefectCategory::ConcurrencyBugs, FindingSeverity::Medium, 0.5),
        ("sync.Mutex", DefectCategory::ConcurrencyBugs, FindingSeverity::Low, 0.3),
        ("data race", DefectCategory::ConcurrencyBugs, FindingSeverity::Critical, 0.95),
        // Security
        ("sql.Query(", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.7), // potential SQL injection
        ("http.Get(", DefectCategory::SecurityVulnerabilities, FindingSeverity::Medium, 0.5), // SSRF potential
        ("exec.Command(", DefectCategory::SecurityVulnerabilities, FindingSeverity::High, 0.8),
        // Anti-patterns
        ("interface{}", DefectCategory::TypeErrors, FindingSeverity::Medium, 0.4),
        ("//nolint", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        // Test debt
        ("t.Skip", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("testing.Short()", DefectCategory::TestDebt, FindingSeverity::Low, 0.3),
    ]
}

/// Get all supported file extensions as glob patterns.
pub fn all_language_globs() -> Vec<String> {
    vec![
        "**/*.rs".to_string(),
        "**/*.py".to_string(),
        "**/*.ts".to_string(),
        "**/*.tsx".to_string(),
        "**/*.js".to_string(),
        "**/*.jsx".to_string(),
        "**/*.go".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("go"), Some(Language::Go));
        assert_eq!(Language::from_extension("txt"), None);
    }

    #[test]
    fn test_patterns_for_rust() {
        let patterns = patterns_for_language(Language::Rust);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "unwrap()"));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_patterns_for_python() {
        let patterns = patterns_for_language(Language::Python);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "eval("));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_patterns_for_typescript() {
        let patterns = patterns_for_language(Language::TypeScript);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "as any"));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_patterns_for_go() {
        let patterns = patterns_for_language(Language::Go);
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "panic("));
        assert!(patterns.iter().any(|(p, _, _, _)| *p == "TODO")); // universal
    }

    #[test]
    fn test_all_language_globs() {
        let globs = all_language_globs();
        assert!(globs.contains(&"**/*.rs".to_string()));
        assert!(globs.contains(&"**/*.py".to_string()));
        assert!(globs.contains(&"**/*.go".to_string()));
    }

    // =========================================================================
    // Coverage gap: Language::from_extension edge cases
    // =========================================================================

    #[test]
    fn test_language_from_extension_tsx() {
        assert_eq!(Language::from_extension("tsx"), Some(Language::TypeScript));
    }

    #[test]
    fn test_language_from_extension_js() {
        assert_eq!(Language::from_extension("js"), Some(Language::TypeScript));
    }

    #[test]
    fn test_language_from_extension_jsx() {
        assert_eq!(Language::from_extension("jsx"), Some(Language::TypeScript));
    }

    #[test]
    fn test_language_from_extension_case_insensitive() {
        assert_eq!(Language::from_extension("RS"), Some(Language::Rust));
        assert_eq!(Language::from_extension("PY"), Some(Language::Python));
        assert_eq!(Language::from_extension("TS"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("GO"), Some(Language::Go));
    }

    #[test]
    fn test_language_from_extension_unknown() {
        assert_eq!(Language::from_extension("c"), None);
        assert_eq!(Language::from_extension("java"), None);
        assert_eq!(Language::from_extension("rb"), None);
        assert_eq!(Language::from_extension(""), None);
    }

    // =========================================================================
    // Coverage gap: Language::extensions()
    // =========================================================================

    #[test]
    fn test_language_extensions_rust() {
        assert_eq!(Language::Rust.extensions(), &["rs"]);
    }

    #[test]
    fn test_language_extensions_python() {
        assert_eq!(Language::Python.extensions(), &["py"]);
    }

    #[test]
    fn test_language_extensions_typescript() {
        let exts = Language::TypeScript.extensions();
        assert_eq!(exts, &["ts", "tsx", "js", "jsx"]);
    }

    #[test]
    fn test_language_extensions_go() {
        assert_eq!(Language::Go.extensions(), &["go"]);
    }

    // =========================================================================
    // Coverage gap: Language::glob_patterns()
    // =========================================================================

    #[test]
    fn test_language_glob_patterns_rust() {
        assert_eq!(Language::Rust.glob_patterns(), vec!["**/*.rs"]);
    }

    #[test]
    fn test_language_glob_patterns_python() {
        assert_eq!(Language::Python.glob_patterns(), vec!["**/*.py"]);
    }

    #[test]
    fn test_language_glob_patterns_typescript() {
        let patterns = Language::TypeScript.glob_patterns();
        assert_eq!(patterns.len(), 4);
        assert!(patterns.contains(&"**/*.ts"));
        assert!(patterns.contains(&"**/*.tsx"));
        assert!(patterns.contains(&"**/*.js"));
        assert!(patterns.contains(&"**/*.jsx"));
    }

    #[test]
    fn test_language_glob_patterns_go() {
        assert_eq!(Language::Go.glob_patterns(), vec!["**/*.go"]);
    }

    // =========================================================================
    // Coverage gap: all_language_globs completeness
    // =========================================================================

    #[test]
    fn test_all_language_globs_complete() {
        let globs = all_language_globs();
        assert_eq!(globs.len(), 7);
        assert!(globs.contains(&"**/*.ts".to_string()));
        assert!(globs.contains(&"**/*.tsx".to_string()));
        assert!(globs.contains(&"**/*.js".to_string()));
        assert!(globs.contains(&"**/*.jsx".to_string()));
    }

    // =========================================================================
    // Coverage gap: pattern content verification
    // =========================================================================

    #[test]
    fn test_universal_patterns_content() {
        let patterns = universal_patterns();
        // Verify all expected universal patterns are present
        let names: Vec<&str> = patterns.iter().map(|(p, _, _, _)| *p).collect();
        assert!(names.contains(&"TODO"));
        assert!(names.contains(&"FIXME"));
        assert!(names.contains(&"HACK"));
        assert!(names.contains(&"XXX"));
        assert!(names.contains(&"BUG"));
        assert!(names.contains(&"placeholder"));
        assert!(names.contains(&"stub"));
        assert!(names.contains(&"dummy"));
        assert!(names.contains(&"temporary"));
        assert!(names.contains(&"hardcoded"));
        assert!(names.contains(&"workaround"));
        assert!(names.contains(&"tech debt"));
    }

    #[test]
    fn test_rust_patterns_content() {
        let patterns = rust_patterns();
        let names: Vec<&str> = patterns.iter().map(|(p, _, _, _)| *p).collect();
        // SAFETY: no actual unsafe code -- asserting pattern detection includes unsafe keyword
        assert!(names.contains(&"unsafe {"));
        assert!(names.contains(&"transmute"));
        assert!(names.contains(&"panic!"));
        assert!(names.contains(&"unreachable!"));
        assert!(names.contains(&"unimplemented!"));
        assert!(names.contains(&"todo!"));
        assert!(names.contains(&"#[ignore]"));
        assert!(names.contains(&".unwrap_or_else(|_|"));
        assert!(names.contains(&"Err(_) => {}"));
    }

    #[test]
    fn test_python_patterns_content() {
        let patterns = python_patterns();
        let names: Vec<&str> = patterns.iter().map(|(p, _, _, _)| *p).collect();
        assert!(names.contains(&"except:"));
        assert!(names.contains(&"except Exception:"));
        assert!(names.contains(&"except BaseException:"));
        assert!(names.contains(&"pickle.loads"));
        assert!(names.contains(&"shell=True"));
        assert!(names.contains(&"__import__"));
        assert!(names.contains(&"global "));
        assert!(names.contains(&"import *"));
        assert!(names.contains(&"# type: ignore"));
        assert!(names.contains(&"@pytest.mark.skip"));
        assert!(names.contains(&"@unittest.skip"));
        assert!(names.contains(&"raise NotImplementedError"));
        assert!(names.contains(&"threading.Thread("));
    }

    #[test]
    fn test_typescript_patterns_content() {
        let patterns = typescript_patterns();
        let names: Vec<&str> = patterns.iter().map(|(p, _, _, _)| *p).collect();
        assert!(names.contains(&"// @ts-ignore"));
        assert!(names.contains(&"// @ts-nocheck"));
        assert!(names.contains(&"@ts-expect-error"));
        assert!(names.contains(&"innerHTML"));
        assert!(names.contains(&"dangerouslySetInnerHTML"));
        assert!(names.contains(&"document.write"));
        assert!(names.contains(&"console.log"));
        assert!(names.contains(&"debugger"));
        assert!(names.contains(&"== null"));
        assert!(names.contains(&"!= null"));
        assert!(names.contains(&"it.skip"));
        assert!(names.contains(&"describe.skip"));
        assert!(names.contains(&"test.skip"));
        assert!(names.contains(&".only("));
        assert!(names.contains(&".catch(() => {"));
        assert!(names.contains(&".catch(e => {})"));
    }

    #[test]
    fn test_go_patterns_content() {
        let patterns = go_patterns();
        let names: Vec<&str> = patterns.iter().map(|(p, _, _, _)| *p).collect();
        assert!(names.contains(&"_ = err"));
        assert!(names.contains(&"panic("));
        assert!(names.contains(&"log.Fatal"));
        assert!(names.contains(&"go func()"));
        assert!(names.contains(&"sync.Mutex"));
        assert!(names.contains(&"data race"));
        assert!(names.contains(&"sql.Query("));
        assert!(names.contains(&"http.Get("));
        assert!(names.contains(&"exec.Command("));
        assert!(names.contains(&"interface{}"));
        assert!(names.contains(&"//nolint"));
        assert!(names.contains(&"t.Skip"));
        assert!(names.contains(&"testing.Short()"));
    }

    // =========================================================================
    // Coverage gap: pattern severity/category verification
    // =========================================================================

    #[test]
    fn test_patterns_have_valid_suspiciousness() {
        for lang in [Language::Rust, Language::Python, Language::TypeScript, Language::Go] {
            let patterns = patterns_for_language(lang);
            for (name, _cat, _sev, sus) in &patterns {
                assert!(
                    *sus >= 0.0 && *sus <= 1.0,
                    "Pattern '{}' has invalid suspiciousness: {}",
                    name,
                    sus
                );
            }
        }
    }

    #[test]
    fn test_language_equality() {
        assert_eq!(Language::Rust, Language::Rust);
        assert_ne!(Language::Rust, Language::Python);
        assert_ne!(Language::Python, Language::TypeScript);
        assert_ne!(Language::TypeScript, Language::Go);
    }
}
