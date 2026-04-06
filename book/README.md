# Trueno-DB Book

Comprehensive documentation for the Trueno-DB GPU-accelerated database engine.

## Quick Start

### Build the book

```bash
make book
```

Output: `book/book/index.html`

### Serve locally

```bash
make book-serve
```

Opens browser at `http://localhost:3000`

### Watch mode (auto-rebuild)

```bash
make book-watch
```

## Structure

```
book/
├── book.toml           # mdBook configuration
├── src/                # Markdown source files
│   ├── SUMMARY.md      # Table of contents
│   ├── introduction.md # Introduction
│   ├── architecture/   # System design
│   ├── components/     # Component deep dives
│   ├── tdd/            # EXTREME TDD methodology
│   ├── toyota/         # Toyota Way principles
│   ├── quality/        # Quality gates
│   ├── academic/       # Research papers
│   ├── dev/            # Developer guide
│   ├── case-studies/   # Real-world examples
│   ├── performance/    # Benchmarking
│   ├── troubleshooting/
│   └── appendix/
└── book/               # Generated HTML (gitignored)
```

## Content Status

✅ **Complete**:
- Introduction
- Architecture: Cost-Based Backend Selection
- Case Studies: CORE-001 (Arrow Storage Backend)
- System Overview

🚧 **In Progress**:
- All other chapters have placeholder content

## Contributing

To add new content:

1. Edit markdown files in `src/`
2. Update `SUMMARY.md` if adding new pages
3. Run `make book` to rebuild
4. Verify changes with `make book-serve`

## Academic Foundation

All chapters reference peer-reviewed research:
- Leis et al. (2014) - Morsel-driven parallelism
- Funke et al. (2018) - GPU paging
- Gregg & Hazelwood (2011) - PCIe bottlenecks
- Breß et al. (2014) - Heterogeneous query processing
- Neumann (2011) - JIT compilation
- Wu et al. (2012) - Kernel fusion

## License

MIT License - same as parent project
