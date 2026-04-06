# Repartir Book

Comprehensive documentation for the Repartir distributed computing library.

## Building the Book

Install mdBook:

```bash
cargo install mdbook
```

Build and serve locally:

```bash
cd book
mdbook serve
```

The book will be available at `http://localhost:3000`.

## Build for Production

```bash
mdbook build
```

Output will be in `book/book/`.

## Contributing

When adding new chapters:

1. Create the markdown file in `src/`
2. Add it to `src/SUMMARY.md`
3. Test with `mdbook build`
4. Verify all links work

## Structure

```
book/
├── book.toml              # Configuration
├── src/
│   ├── SUMMARY.md        # Table of contents
│   ├── introduction.md   # Landing page
│   ├── getting-started/  # Installation, quick start
│   ├── architecture/     # System design
│   ├── executors/        # CPU, GPU, Remote execution
│   ├── scheduler/        # Task scheduling
│   ├── messaging/        # PubSub, PushPull
│   ├── examples/         # Code examples
│   ├── development/      # Contributing guide
│   ├── advanced/         # Advanced topics
│   ├── ecosystem/        # PAIML stack integration
│   ├── specifications/   # Design specs
│   └── appendix/         # Glossary, references, FAQ
└── book/                 # Generated output (git-ignored)
```
