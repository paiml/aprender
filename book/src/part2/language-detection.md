# Language Detection

Language detection is the first sub-phase of Analysis. It identifies every programming language present in the source project and calculates line-count statistics.

## Detection Method

Batuta uses a two-layer detection strategy:

1. **File extension mapping** -- `.py` to Python, `.c`/`.h` to C, `.sh` to Shell, etc.
2. **Content inspection** -- shebang lines (`#!/usr/bin/env python3`) disambiguate extensionless scripts

The `Language` enum in `src/types.rs` covers all supported languages:

```rust
pub enum Language {
    Python, C, Cpp, Rust, Shell,
    JavaScript, TypeScript, Go, Java,
    Other(String),
}
```

Parsing from strings is case-insensitive with common aliases:

```rust
// All of these resolve to Language::Shell
"shell".parse::<Language>()  // Ok(Shell)
"bash".parse::<Language>()   // Ok(Shell)
"sh".parse::<Language>()     // Ok(Shell)
```

## Multi-Language Projects

Most real projects contain multiple languages. Batuta produces a `LanguageStats` vector sorted by line count:

```rust
pub struct LanguageStats {
    pub language: Language,
    pub file_count: usize,
    pub line_count: usize,
    pub percentage: f64,
}
```

The language with the highest percentage becomes the `primary_language`, which determines the default transpiler in Phase 2.

## Example Output

```bash
$ batuta analyze --languages ./my-project

Language Analysis
-----------------
Python     |  142 files |  28,400 lines |  72.3%  (primary)
Shell      |   18 files |   4,200 lines |  10.7%
C          |   12 files |   3,800 lines |   9.7%
JavaScript |    8 files |   2,900 lines |   7.3%
```

## Supported Extensions

| Extension | Language | Notes |
|-----------|----------|-------|
| `.py` | Python | Includes `.pyw`, `.pyi` stubs |
| `.c`, `.h` | C | Header files counted separately |
| `.cpp`, `.cc`, `.cxx`, `.hpp` | C++ | All common variants |
| `.sh`, `.bash` | Shell | Also detects via shebang |
| `.rs` | Rust | Detected but not transpiled |
| `.js`, `.mjs` | JavaScript | ESM and CJS |
| `.ts`, `.tsx` | TypeScript | Including JSX variant |
| `.go` | Go | Single extension |
| `.java` | Java | Single extension |

## Mixed-Language Handling

When a project contains multiple transpilable languages (e.g., Python and Shell), Batuta processes each language with its corresponding transpiler in Phase 2. The `primary_language` sets the default, but all detected languages are stored in the analysis results for per-file transpiler dispatch.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
