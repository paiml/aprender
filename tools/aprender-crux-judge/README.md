# aprender-crux-judge

A hand-written Rust port of the CRUX inference judge, its answer oracles and
its prompt certifier. It reads a CRUX run manifest and writes the receipt
JSON and the Markdown table, byte-identical to the Python judge it ports.
The three Python modules stay for now: their callers (the `check_crux_*.sh` case
tables, `crux_inference_dogfood.sh`, `crux_sweep_shards.sh`) are gate and
release-decision paths, and a gate keeps its `.py` until a released tool carries
the port (N-1). They switch to this binary, and the `.py` are deleted, in the
first change after that release:

| Python module (kept, N-1) | Rust module | Subcommands |
|-----------------------|-------------|-------------|
| `scripts/lib/crux_inference_judge.py` | `collect.rs`, `judge.rs`, `main.rs` | `collect` |
| `scripts/lib/crux_oracles.py` | `oracles.rs`, `sandbox.rs` | `eval`, `extract`, `lint` |
| `scripts/lib/crux_prompt_certify.py` | `certify.rs` | `certify`, `check` |

`serve_routes.rs` ports the one table the judge reads from
`scripts/lib/crux_serve_routes.py`, which stays: the serve driver uses it.

`pyval.rs`, `pyjson.rs`, `pyio.rs`, `pyre.rs` and `pyerr.rs` carry the
Python semantics the judge leans on: dynamic values and their errors,
`json.loads`/`json.dump`, strict UTF-8 file reads with universal newlines,
`re` character classes, and exception kinds and messages.

The crate is not a workspace member (the root `Cargo.toml` excludes it). Its
binary-debt row is in `contracts/binary-debt-v1.yaml`. Scripts reach it
through `scripts/lib/crux_judge_bin.sh`, which builds it from the calling
tree (`cargo build --release --locked`, target dir inside the crate) and
exports `$CRUX_JUDGE`; `CRUX_JUDGE_BIN=<path>` skips the build. It has no
caller yet: `scripts/crux_inference_dogfood.sh`, `scripts/crux_sweep_shards.sh`
and the two case tables switch to it in the N-1 change above.

## Usage

```
aprender-crux-judge collect --manifest MANIFEST --prompts PROMPTS --meta META \
    --out-json OUT_JSON --out-md OUT_MD [--certification CERTIFICATION]
aprender-crux-judge sandbox
aprender-crux-judge eval <prompt.json> <reply.json>
aprender-crux-judge extract <prompt.json> <reply.json>
aprender-crux-judge lint <prompts.json>
aprender-crux-judge certify --prompts P --inventory I --apr-commit SHA -o OUT MANIFEST...
aprender-crux-judge check --prompts P --receipt R
```

`collect` takes the same flags and has the same exit codes as the Python
judge: 0 PASS, 1 RED, 2 DECLINE.

`eval` and `lint` are the oracle module's CLI: one JSON verdict line, or the
schema errors of a v2 prompt set; rc 0 correct (or valid), 1 wrong (or
invalid), 2 usage. `extract` is new: it prints the JSON unit a same-file
comparison compares, which the oracle case table used to read by importing
the module. `certify` and `check` are the certifier's CLI, flags and exit
codes unchanged.

`sandbox` exists only in the port. It probes the `code_tests` sandbox and
prints the interpreter version a receipt should record, for example
`sandbox: Python 3.13.1; prlimit cpu/as/fsize, unshare -rn, python3 -I, env PATH=/usr/bin:/bin`,
and exits 0. When there is no sandbox it prints
`sandbox_unavailable: <why>` and exits 2.

## The one Python seam: `run_python_cell`

A `code_tests` prompt asks the model for Python, so the reply can only be
tested by running it. `sandbox::run_python_cell` is the single place the
judge starts an interpreter. It runs the model's reply plus the prompt's
asserts and nothing else. This is the operator's external-validation
exception to "no Python": no `.py` file is added to the repo, and the cell
file exists only in a fresh temporary directory. A follow-up replaces the
seam with a Python-free executor.

The cell is `code + "\n\n" + tests + "\nprint('CRUX_TESTS_PASSED')\n"`. It is
judged correct only when the interpreter exits 0 and its stdout, with
trailing whitespace stripped, ends with the sentinel.

### Sandbox properties

| Property | How | Case-table rows (`oracles::tests::sandbox_case_table`) |
|----------|-----|-----------------------------------------|
| Fresh 0700 tmpdir holding only `cell.py`, removed afterwards | `fresh_dir` + `remove_dir_all` | `listdir == ['cell.py']` and mode 0700 → PASS; planted `listdir` and mode checks → AssertionError; `sandbox::tests::the_tmpdir_is_fresh_and_removed` |
| No network | `unshare -rn` (only `lo`) | only `lo` → PASS; `connect` → `Network is unreachable` |
| Isolated interpreter | `python3 -I` | `-I` flags → PASS; planted check → AssertionError |
| Scrubbed environment | `env_clear`, `PATH=/usr/bin:/bin` | env ⊆ {PATH, LC_CTYPE} → PASS; planted `HOME` → AssertionError |
| No stdin | `/dev/null` | fd 0 is the `/dev/null` device → PASS; planted check → AssertionError |
| Address space 1 GiB | `prlimit --as` | `bytearray(2<<30)` → MemoryError |
| CPU seconds | `prlimit --cpu=t:t` | `getrlimit(RLIMIT_CPU) == (t, t)` → PASS; planted → AssertionError. The limit is read, not provoked: a busy-loop row flaked to `tests_timeout` on a loaded host |
| File size 16 MiB | `prlimit --fsize` | 32 MiB write → `File too large` |
| Wall clock `t + 5` s | killed by the judge | `sleep(60)`, t=1 → `tests_timeout` |
| Sentinel | printed after the last assert | failing assert → AssertionError; `sys.exit(0)` before it → `rc=0` |

Every property has a planted-failure row, and the table runs against the
real sandbox. A host without python3, `prlimit` or a working `unshare -rn`
FAILS the tests; it does not skip them, since a sandbox test that skips
measures nothing.

Outside the sandbox (`code_tests_outside_the_sandbox`): no interpreter, a
failed probe, a spawn error and a tmpdir error all give
`sandbox_unavailable`. That is not_measured and never correct.

### Where the seam differs from the Python oracle

- **`prlimit` instead of `preexec_fn`.** The limits are set by `prlimit`
  before `unshare -rn` starts the interpreter, rather than by `setrlimit` in
  the forked child. The values are the oracle's: CPU `(t, t)`, AS `1<<30`,
  FSIZE `1<<24`. A CPU overrun ends in SIGKILL (rc -9) because the soft and
  hard limits are equal.
- **stdin is `/dev/null`.** The Python oracle let the cell inherit the
  judge's stdin.
- **The probe is stricter.** Before the first cell, `python3 -I -V` runs
  under the full sandbox (prlimit, unshare, scrubbed env) and must print a
  version. The Python oracle probed only `unshare -rn true`.
- **`timeout_s`** is read with Python's `int()` rules (`" 1_0 "` is 10;
  `"1__0"` and `None` crash as `int()` would). -1 means no CPU limit and a
  4 s wall bound, as `setrlimit` reads -1 as RLIM_INFINITY. Any other
  negative value crashes with the oracle's
  `SubprocessError: Exception occurred in preexec_fn.` A value between
  2^63 and 2^64 also crashes here, where Python would run with that limit.
- **`tests_timeout` is kept** as its own reason, for golden parity. It is
  never correct: it is a RED reason, not not_measured.
- **The sentinel weakness is inherited** and kept for parity. The cell
  prints the sentinel itself, so a reply that runs
  `print('CRUX_TESTS_PASSED', flush=True)` and then `os._exit(0)` passes
  without its asserts running. A case-table row pins this behaviour so a
  later fix shows up as a deliberate change.
- **The interpreter is the first `python3` on the judge's PATH** (absolute
  entries only). The Python oracle ran its own interpreter (`sys.executable`).
- **The sandbox is probed once per process.** One failed probe makes every
  `code_tests` row of that run `sandbox_unavailable`. The Python oracle
  re-probed for each prompt.
- **Inherited gaps, kept for parity:** a cell that forks a child which
  outlives it is held only by its rlimits (the wall-clock kill reaches the
  interpreter, not its children), and the cell's stdout and stderr are read
  into memory without a cap.

## Other divergences from the Python judge

None of these changes any golden. Each is pinned by a source comment at the
place it happens.

- **Sort order on mixed types** (`pyval::py_sorted_by`). CPython's timsort
  may compare a different pair first, so the operands in a TypeError
  message can appear in the other order.
- **Lone surrogates in JSON** (`pyjson`). A `"\ud800"` escape cannot live
  in a Rust `String`; the port declines with a ValueError where Python
  carries the surrogate on.
- **NaN dict keys** (`pyval::hkey`). Every NaN is its own key. CPython also
  matches the same NaN object by identity.
- **Unicode 15.0/15.1.** The tables are generated from UCD 14.0 plus the two
  digit runs Unicode 15.0 added (15.1 added none). Other 15.0/15.1 code
  points count as unassigned, so `repr` escapes them where CPython 3.13
  would print them.
- **A crash prints `crash: <Kind>: <msg>`** and exits 1, with no traceback.
  The kind is the bare class name: `JSONDecodeError`, where the traceback
  ended `json.decoder.JSONDecodeError`. This holds for every subcommand.
- **Program names.** Usage lines name `aprender-crux-judge`, where argparse
  named `crux_prompt_certify.py` or the oracle CLI named its file.
- **Path arguments.** Python opens an `int` or `bool` as a file descriptor;
  the port raises TypeError (`pyio::path_arg`).
- **Command-line errors.** The usage line and the invalid-choice text match
  argparse; other usage-error texts may differ. A non-UTF-8 argument is
  refused with exit 2.
- **The `.npy` header** is read by a narrow `ast.literal_eval` (`judge.rs`):
  str keys and values, ints, True/False/None, tuples and lists. Anything
  else is a SyntaxError here, where Python may accept it.
- **`struct.error`** is reported with the kind `error`, and the edge cases
  of the array dimensions (negative, huge, bool) are approximated with the
  messages `struct` would raise.
- **Negative-control turns stored as a dict** raise "not subscriptable",
  where Python 3.12+ raises KeyError on the slice. No manifest reaches it.
- **Error order.** On pathological inputs with more than one fault, the port
  may report a different fault first.
- **`json.dump` errors** leave no partial output file: the port serializes
  before it writes.

## Acceptance

The port is accepted only when it is byte-identical to the Python judge on
the 11 golden cases: receipt JSON, Markdown and exit code, with `judged_at`
masked, against the receipts the Python judge wrote under CPython 3.13.1.
Each case must also leave stderr empty and print the same Markdown to stdout.
The `code_tests` cases must run their asserts in the sandbox. Both case
tables must be green against this binary: `scripts/check_crux_oracles.sh`
(oracles, lint and the certifier) and `scripts/check_crux_inference_judge.sh`
(the judge, with its mutants rebuilt from mutated copies of these sources).

Before the Python modules were deleted, `certify` was run against the
Python certifier on the real certification manifests (receipt and stdout
identical, all four `check` combinations rc 0), and `eval`/`lint` against
the Python oracle CLI on 620 inputs; the only difference was the crash form
above.

```
cargo test --release --manifest-path tools/aprender-crux-judge/Cargo.toml
cargo clippy --release --all-targets --manifest-path tools/aprender-crux-judge/Cargo.toml -- -D warnings
```

The crate is outside the workspace, so the workspace lint run never reaches
it. Its own `clippy.toml` carries the repo's `unwrap()` ban (`disallowed-methods`,
GH-41), so the clippy command above enforces it, tests included. Clippy reads
only the nearest config file, so this one shadows the root `.clippy.toml` here.

## Regenerating the Unicode tables

```
bash tools/aprender-crux-judge/gen_unicode_tables.sh <UCD dir> > tools/aprender-crux-judge/src/unicode_tables.rs
```

`<UCD dir>` holds `UnicodeData.txt` and `CaseFolding.txt`. The script needs
GNU awk. Each generated table carries `#[rustfmt::skip]`, so `cargo fmt`
leaves the output byte-identical to what the script writes.
