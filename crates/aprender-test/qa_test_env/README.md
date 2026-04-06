# QA Test Environment

This directory contains **placeholder files** for QA testing of Probar's development server and validation features.

## Purpose

These empty files simulate a minimal web project structure for testing:

- `probar serve` - WASM development server
- `probar serve --validate` - Module import validation
- `probar serve tree` - File tree visualization
- `probar serve score` - Project scoring

## File Structure

```
qa_test_env/
├── index.html      # Empty HTML file (placeholder)
├── script.js       # Empty JS file (placeholder)
├── style.css       # Empty CSS file (placeholder)
└── pkg/
    └── app.wasm    # Empty WASM file (placeholder)
```

## Usage in Testing

### Test 1: Serve Command

```bash
probar serve -d ./qa_test_env --port 8080
```

### Test 2: File Tree

```bash
probar serve tree ./qa_test_env
```

### Test 3: Validation (Expected to Pass)

```bash
probar serve -d ./qa_test_env --validate
```

### Test 4: Score (Expected: Low Score)

```bash
probar serve score ./qa_test_env
```

## Why Empty Files?

Empty files are sufficient for testing:
- File existence checks
- MIME type detection
- Directory traversal
- Server response headers

They do NOT test:
- Actual WASM execution
- JavaScript parsing
- HTML rendering

For full integration tests, use `crates/showcase-calculator/` which contains a complete, working WASM application.

## Related

- See `docs/qa/qa-verification-PROBAR-005-impl.md` for QA verification report
- See `docs/qa/qa-handoff-report-PROBAR-005.md` for handoff checklist
