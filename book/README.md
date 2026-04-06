# Realizar Documentation Book

This directory contains the [mdBook](https://rust-lang.github.io/mdBook/) documentation for Realizar.

## Building the Book

Install mdBook (if not already installed):

```bash
cargo install mdbook
```

Build the book:

```bash
mdbook build book/
# or use Makefile
make book-build
```

The HTML output will be in `book/book/index.html`.

## Serving Locally

Serve the book with live reload:

```bash
mdbook serve book/
# or use Makefile
make book-serve
```

Then open http://localhost:3000 in your browser.

## Structure

```
book/
├── book.toml              # Configuration
├── README.md              # This file
├── src/
│   ├── SUMMARY.md         # Table of contents (200+ chapters)
│   ├── introduction.md    # Introduction (WRITTEN)
│   ├── architecture/      # Design philosophy, feature flags
│   ├── formats/           # GGUF, Safetensors parsing
│   ├── quantization/      # Q4_0, Q8_0, K-quants
│   ├── transformer/       # Attention, RoPE, FFN, KV cache
│   ├── tokenization/      # BPE, SentencePiece algorithms
│   ├── generation/        # Sampling strategies
│   ├── api/               # REST API with Axum
│   ├── cli/               # Command-line interface
│   ├── gpu/               # Trueno acceleration
│   ├── tdd/               # EXTREME TDD methodology
│   ├── phases/            # Development phases (1-4)
│   ├── quality/           # Quality gates
│   ├── performance/       # Benchmarks
│   ├── examples/          # Real-world use cases
│   ├── tools/             # cargo tools, pmat
│   ├── best-practices/    # Design patterns
│   ├── decisions/         # Architecture decisions
│   └── appendix/          # Glossary, contributing (WRITTEN)
└── book/                  # Generated HTML (gitignored)
```

## Gating Approach

Realizar follows **trueno's gating approach** for documentation:

### The Gate System

1. **Chapter structure exists upfront** - All 200+ chapters created as placeholders
2. **Content added ONLY when feature is complete** - Gates prevent premature documentation
3. **All code examples are test-backed** - Zero tolerance for hallucination
4. **Phased development** - Chapters marked with Phase 1-4 status

### Why Gating?

- ✅ **Prevents hallucinated documentation** - Can't document what doesn't exist
- ✅ **Enforces test-backed examples** - Every code snippet validated by tests
- ✅ **Clear development phases** - Know which chapters are Phase 1, 2, 3, or 4
- ✅ **Quality matches code quality** - Same EXTREME TDD standards

### Chapter Lifecycle

```
┌─────────────────┐
│  Placeholder    │ ← All chapters start here
│  "[Content to   │
│   be added]"    │
└────────┬────────┘
         │
         │ Implementation complete?
         │ Tests passing?
         │ Examples validated?
         │
         ▼
┌─────────────────┐
│  Full Content   │ ← Chapter written with:
│  Test-backed    │   - Actual code
│  Examples       │   - Test results
│  Zero halluc.   │   - Benchmarks
└─────────────────┘
```

### Phase Markers

Chapters are marked by development phase:

- **Phase 1 (COMPLETE)**: GGUF, Safetensors, Transformer, Tokenization, API, CLI
- **Phase 2 (Optimization)**: Advanced quantization, Flash Attention, Streaming
- **Phase 3 (Advanced Models)**: MQA, GQA, Vision models
- **Phase 4 (Production)**: Multi-model serving, Batching, Monitoring
- **Documentation**: General chapters (always relevant)

## Quality Enforcement

All documentation goes through quality gates:

```bash
# Build book (validates structure)
make book-build

# Validate code examples are test-backed
make book-validate

# Full quality gates (includes book validation)
make quality-gates
```

### What Gets Validated:

1. **Book structure** - `book.toml`, `SUMMARY.md` valid
2. **Code blocks** - All ```rust blocks must reference actual code
3. **Identifiers** - Functions/structs exist in `src/`, `tests/`, `examples/`
4. **Zero hallucination** - Warnings for potentially untested code

See [Contributing to This Book](src/appendix/contributing.md) for full guidelines.

## Current Status

| Section | Chapters | Written | Placeholders |
|---------|----------|---------|--------------|
| **Introduction** | 1 | 1 | 0 |
| **Core Architecture** | 5 | 0 | 5 |
| **Model Formats** | 12 | 0 | 12 |
| **Quantization** | 10 | 0 | 10 |
| **Transformer** | 16 | 0 | 16 |
| **Tokenization** | 8 | 0 | 8 |
| **Text Generation** | 8 | 0 | 8 |
| **REST API & CLI** | 17 | 0 | 17 |
| **GPU Acceleration** | 4 | 0 | 4 |
| **EXTREME TDD** | 10 | 0 | 10 |
| **Development Phases** | 18 | 0 | 18 |
| **Quality, Performance, Examples** | 27 | 0 | 27 |
| **Tools & Best Practices** | 16 | 0 | 16 |
| **Design Decisions** | 6 | 0 | 6 |
| **Appendix** | 6 | 1 | 5 |
| **TOTAL** | **164** | **2** | **162** |

## Contributing

To add or update documentation:

1. **Check the gate** - Is the feature implemented and tested?
2. **Edit markdown files** in `book/src/`
3. **Use actual code only** - Reference `src/`, `tests/`, `examples/`
4. **Validate**: `make book-validate` (test-backed code)
5. **Build**: `make book-build` (structure valid)
6. **Test**: `make quality-gates` (all gates pass)
7. **Submit PR** with documentation changes

### Documentation Standards

All documentation MUST:

- ✅ Include ONLY test-backed code examples
- ✅ Reference actual source files with line numbers
- ✅ Show RED-GREEN-REFACTOR cycles where applicable
- ✅ Link to tests that validate the examples
- ✅ Include performance metrics from actual benchmarks
- ✅ Pass `make book-validate` (zero hallucination check)

See [Contributing to This Book](src/appendix/contributing.md) for detailed guidelines.

## Content Sources

Documentation will be sourced from:

- `src/**/*.rs` - Production code with rustdoc comments
- `tests/**/*.rs` - Unit, property, integration tests
- `examples/**/*.rs` - Working examples
- `benches/**/*.rs` - Performance benchmarks
- `CLAUDE.md` - Development guide and metrics
- `README.md` - Project overview
- `CHANGELOG.md` - Version history

**NO hypothetical code. NO untested examples. NO hallucinations.**

## Examples of Good Chapter Content

### ✅ Phase 1 (Complete) - Can Be Written

```markdown
# GGUF Format

## Implementation (Phase 1 - COMPLETE)

Our GGUF parser is implemented in `src/gguf.rs` and handles GGUF v3 format:

\`\`\`rust
// From src/gguf.rs:245-260 (test-backed)
pub fn from_bytes(data: &[u8]) -> Result<GGUFModel> {
    let mut cursor = Cursor::new(data);
    let magic = Self::read_u32(&mut cursor)?;
    // ... actual implementation
}
\`\`\`

**Tests**: `tests/unit_gguf.rs:123-145` validates metadata parsing
**Coverage**: 95% (region), includes Array type edge cases
**Mutation Score**: Property tests catch all mutants
```

### ❌ Phase 2 (Not Started) - Keep Placeholder

```markdown
# Flash Attention

> **Status**: Phase 2 (Optimization)
>
> This chapter will be written when the corresponding feature is implemented.

[Content to be added]
```

## Deployment

The book can be deployed to GitHub Pages:

```bash
# Build for production
make book-build

# Deploy (future)
# Will be automated via GitHub Actions
```

CI workflow: `.github/workflows/mdbook.yml` (to be created)

## Anti-Hallucination Guarantee

**Every code example in this book**:

- ✅ Comes from actual source code in `src/`
- ✅ Is validated by tests in `tests/`
- ✅ Can be reproduced by running `cargo test`
- ✅ Is checked by `make book-validate` in pre-commit
- ❌ Is NEVER hypothetical or illustrative
- ❌ Is NEVER from outdated code
- ❌ Is NEVER hallucinated by AI

**If you find a code example that doesn't exist in the codebase, that's a bug. Please file an issue!**

---

**Built following EXTREME TDD principles. Zero tolerance for documentation defects.** 📚✅
