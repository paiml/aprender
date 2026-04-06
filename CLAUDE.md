# provable-contracts

Papers to Math to Contracts in Code. Rust library and CLI for converting
research papers into provable kernel implementations via YAML contracts
with Kani bounded model checking.

## Code Search Policy

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

```bash
# Find functions by intent
pmat query "contract validation" --limit 10

# Find with fault patterns (--faults)
pmat query "unwrap" --faults --exclude-tests

# Regex search
pmat query --regex "fn\s+verify_\w+" --limit 10

# Literal search
pmat query --literal "kani::proof" --limit 10
```

See `pmat query --help` for full options.

## Project Structure

- `crates/provable-contracts/` — Library crate (schema, scaffold, kani, probar)
- `crates/provable-contracts-cli/` — CLI binary (`pv` command)
- `contracts/` — YAML kernel contracts
- `docs/specifications/pv-spec.md` — Canonical specification (ONE spec)

## Work Tracking

All work tracked via `pmat work`:

```bash
pmat work list          # See all tickets
pmat work start PMAT-N  # Start work
pmat work complete PMAT-N  # Mark done
```

## Zero-Tolerance: No Stubs or SATD

**CRITICAL: Never commit code containing stub markers or self-admitted technical debt.**

Blocked patterns in all `.rs` files:
- `unimplemented!()` / `todo!()` — replace with complete implementations
- `FIXME` / `HACK` / `XXX` — resolve before committing
- `TODO: Replace` / `Wire up:` — artifacts from `pv generate` scaffolds

After `pv generate` produces scaffold files, **immediately** implement all trait
methods, property tests, falsification tests, and Kani harnesses with real logic.
The pre-commit hook scans all staged `.rs` files and rejects commits containing
any of these markers (enforced at `PMAT_MAX_SATD_COMMENTS=0`).

## Quality Gates

```bash
pmat comply check       # Check compliance
pmat quality-gate       # Run quality gates
cargo kani              # Run Kani proof harnesses
```
