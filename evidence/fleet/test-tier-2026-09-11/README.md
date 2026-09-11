# Per-Module Test Catch Density Analysis

## Method
1. **Parse JUnit (`junit.xml`)**: We parsed the provided junit.xml into rows comprising `crate`, `binary` (often matching crate if no binary slash), `test_path` (`module::fn`), and `seconds`.
2. **Find Valid Modules**: We ran a file search (`find . -name '*.rs'`) in the repository and mapped each file to its `crate::module` structure based on Rust path conventions (`src/foo/bar.rs` -> `foo::bar`, `src/lib.rs` -> root module). Tests in `tests/` directories were mapped to `not_measured`.
3. **Map JUnit Tests**: Each JUnit test was mapped to its longest valid `crate::module` prefix. Tests with no module prefix or in `tests::`/`cov_tests::` submodules in `lib.rs` fell back to the root module.
4. **Catch Touches**: We analyzed `git log origin/main --since=60.days --format=%h --grep='^fix' -i`. For each commit, we parsed diffs (`git show -U0 -p`) and extracted modified files that had hunks matching `#[test]` or `fn test_`. We counted 1 touch per test hunk towards the file's `crate::module`.
5. **Aggregation**: We aggregated total tests, total seconds, and total touches for each `crate::module`. Density was calculated as `touches / seconds` (handling divisions by zero appropriately).
6. **Selection**: Modules were sorted by density (descending), then touches (descending), then seconds (ascending). We calculated the cumulative % of touches and found the cutoffs for 80%, 90%, and 95%.
7. **Filterset**: We generated a nextest filterset DSL string that unions `package(crate) & test(/^module(::|$)/)` for non-root modules, `package(crate) & test(/^(tests::|cov_tests::)?[^:]+$/)` for root modules, and `test(~falsif)` for all designed catches.

## Cut Lines
* **80% Line**: Index 222, covering **0.28%** of total execution seconds.
* **90% Line**: Index 268, covering **0.91%** of total execution seconds.
* **95% Line**: Index 299, covering **1.63%** of total execution seconds.

## PR Tier Filterset
The generated filterset is located in `pr-tier-filterset.txt`. It selects precisely the modules above the 80% line plus any test containing `falsif`.

### Selection Statistics
* **Total Suite**: 82,203 tests, 4,390.613 seconds
* **Filterset Matches**: 3,920 tests, 737.721 seconds
* **Reduction**: Selects **4.77%** of tests and **16.80%** of seconds vs the full suite.

## Caveats & Considerations
* **Binary Tests vs Lib Tests**: Some tests touched in fix commits were in binary targets and thus excluded from the `--lib` junit metrics (resulting in 0 seconds and infinite density). They are properly accounted for via the filterset.
* **Heavy Tests**: The `falsif` tests account for ~670 seconds alone. The actual modules in the top 80% cost only ~12 seconds.
* **Root Module Complexity**: Identifying tests perfectly bound to the root module (vs its implicit submodules) is handled heuristically via regex in the filterset.
* **Zero Touches**: Modules with zero historical touches are completely excluded from this PR tier, though a reviewer might still expect them to run if manually modifying those files.
