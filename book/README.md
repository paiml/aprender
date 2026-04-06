# Trueno Documentation Book

This directory contains the [mdBook](https://rust-lang.github.io/mdBook/) documentation for Trueno.

## Building the Book

Install mdBook (if not already installed):

```bash
cargo install mdbook
```

Build the book:

```bash
mdbook build book/
```

The HTML output will be in `book/book/index.html`.

## Serving Locally

Serve the book with live reload:

```bash
mdbook serve book/
```

Then open http://localhost:3000 in your browser.

## Structure

```
book/
├── book.toml              # Configuration
├── src/
│   ├── SUMMARY.md         # Table of contents
│   ├── introduction.md    # Introduction chapter
│   ├── getting-started/   # Installation, quick start, tutorials
│   ├── architecture/      # Multi-backend design, SIMD/GPU details
│   ├── api-reference/     # API documentation
│   ├── performance/       # Benchmarks, optimization guide
│   ├── safety/            # Safety model, testing
│   ├── examples/          # Real-world use cases
│   ├── development/       # Contributing, EXTREME TDD
│   ├── advanced/          # SIMD intrinsics, GPU shaders
│   ├── ecosystem/         # Integration with Ruchy, Depyler, etc.
│   ├── specifications/    # Design specs, academic foundations
│   └── appendix/          # Glossary, references, changelog
└── book/                  # Generated HTML (gitignored)
```

## Content Sources

Documentation is migrated from:
- `docs/BENCHMARKS.md` → `performance/benchmarks.md`
- `docs/SIMD_PERFORMANCE.md` → `performance/simd-performance.md`
- `docs/specifications/*.md` → `specifications/*.md`
- Inline code documentation
- CLAUDE.md development guide

## Contributing

To add or update documentation:

1. Edit markdown files in `book/src/`
2. Update `book/src/SUMMARY.md` if adding new chapters
3. Test locally with `mdbook serve book/`
4. Submit PR with documentation changes

All documentation should:
- Include runnable code examples (where applicable)
- Link to related chapters
- Follow Trueno's style guide
- Be validated by CI (spelling, links, code examples)

## Deployment

The book is automatically deployed to GitHub Pages on every push to `main`:

<https://paiml.github.io/trueno/>

CI workflow: `.github/workflows/mdbook.yml`
