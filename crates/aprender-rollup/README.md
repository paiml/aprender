# aprender-rollup

Rollup bot core: picks small green PRs to land as one merge-queue entry
(T-rollup, #5394, PMAT-5737). Part of the [paiml/aprender](https://github.com/paiml/aprender)
monorepo. `publish = false`, build tooling, never installed. This crate is library-only; the CLI
ships as the `aprender-rollup` bin of `aprender-ci-tools` (`cargo run -p aprender-ci-tools --bin aprender-rollup -- plan`),
so the shrink-only [[bin]] allowlist (FALSIFY-MONO-011) does not grow.

| Subcommand | Effect |
|------------|--------|
| `aprender-rollup plan` | prints the rollup it would build (read-only) |
| `aprender-rollup build <dir>` | assembles the rollup locally in worktree `<dir>`; never pushes or closes |
