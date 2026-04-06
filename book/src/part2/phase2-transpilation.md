# Phase 2: Transpilation

Phase 2 converts source code from the detected language into Rust using external transpiler tools. It dispatches each file to the appropriate transpiler based on the language map produced by Phase 1.

## Transpiler Dispatch

The `TranspilationStage` reads the `primary_language` from `PipelineContext` and selects the matching tool from the `ToolRegistry`:

| Language | Transpiler | Command |
|----------|-----------|---------|
| Python | Depyler | `depyler transpile --input <src> --output <dst> --format project` |
| C / C++ | Decy | `decy transpile --input <src> --output <dst>` |
| Shell | Bashrs | `bashrs build <src> -o <dst> --target posix --verify strict` |

The `ToolRegistry::get_transpiler_for_language()` method performs the lookup:

```rust
pub fn get_transpiler_for_language(&self, lang: &Language) -> Option<&ToolInfo> {
    match lang {
        Language::C | Language::Cpp => self.decy.as_ref(),
        Language::Python => self.depyler.as_ref(),
        Language::Shell => self.bashrs.as_ref(),
        _ => None,
    }
}
```

## Pipeline Context Flow

Phase 2 receives the context from Phase 1 and adds file mappings:

```
PipelineContext {
    primary_language: Some(Python),    // <-- from Phase 1
    file_mappings: [                   // <-- populated by Phase 2
        ("src/main.py", "src/main.rs"),
        ("src/utils.py", "src/utils.rs"),
    ],
}
```

These mappings are consumed by Phase 4 (Validation) for equivalence checking.

## Parallel File Processing

For multi-file projects, transpilation processes files independently. Each file is dispatched to its language-specific transpiler in parallel, with results collected and merged into the pipeline context.

## Jidoka Stop-on-Error

If any file fails to transpile, the `ValidationStrategy::StopOnError` setting halts the pipeline. The error includes the specific file and transpiler output:

```
Error: Stage 'Transpilation' failed
  Caused by: depyler exited with code 1
  File "complex_class.py", line 42
    Unsupported Python feature: metaclass with __prepare__
```

The workflow state records the failure, and Phase 3 refuses to start until the issue is resolved.

## Sub-Topics

- [Tool Selection](./tool-selection.md) -- how transpilers are detected and validated
- [Incremental Compilation](./incremental.md) -- only retranspile changed files
- [Caching Strategy](./caching.md) -- cross-run persistence of transpilation results
- [Error Handling](./error-handling.md) -- Jidoka error patterns

## CLI Usage

```bash
# Transpile the entire project
batuta transpile --incremental --cache

# Transpile specific modules
batuta transpile --modules auth,api

# Force retranspilation of all files
batuta transpile --force
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
