# impl receipt — #4511 ttop quality (leaks, defects, pv ontology, --timings)

Operator (verbatim, via the cop): "dedicate a worker to quality. it appears to have memory
leaks and defects and is a critical tool for us as it proves TUI visualization and also we
use for monitoring. one improveemnt would be to enforce the "pv ontology" onto it. lets
attempt to fit into .70" and "and instrumentation that does granular timing, i.e. we have
many tools for this: renacer, tracing, etc".

Worker: aprender-57 (author model claude-opus-5-5). Base: `origin/main`. This branch is its
own batch and is never folded into the release car. The work was first built on the 0.70.0
car and replayed onto main as 5 commits, plus one fix commit for the lock and build.rs fmt
(see §6).

## 1. In the workspace (item 1)

`crates/aprender-viz-ttop` was on the root `exclude` list, so CI never built, tested or linted it.
It is now a workspace member. `workspace-test` (`--lib`) picks up its unit tests, and the
ci fragments below run its gates. The workspace fmt pass touched 4 generated/test files in a
style-only commit.

## 2. Leak gate (item 2)

`aprender-viz-ttop --soak-frames N` runs the real loop headless:
- the collector thread
- snapshot apply
- `ui::draw` and the diff
- bytes discarded instead of written

Every 10 frames it prints `{"frame":N,"rss_kib":K}`. Frame 1 starts after the first collector
snapshot is applied, because that pass can take 9.4 s on a loaded host (§5).

`scripts/ttop_soak_gate.sh` drops frames before 100. It computes growth as the median of the
last 5 samples minus the median of the first 5. Growth over 8 MiB is RED. Fewer than 10
samples, or no RSS, is CANNOT MEASURE (rc 2), never green.

| Proof | Result |
|---|---|
| `--self-test` (9 rows: flat, noise, 64/8/12 KiB/frame, warm-up spike, short, null RSS, empty) | PASS |
| real binary, 900 frames | GREEN 3/3 on the main base (see §5) |
| `--mutant`: plants `std::mem::forget(vec![7u8; 64 * 1024])` per frame, rebuilds | RED at 31 MiB growth, which is the required result (the mutant is caught). main.rs is restored, checked by cksum |

## 3. Defects, each with a test the mutant fails (item 3)

| # | Defect | Fix | Test | Mutant |
|---|---|---|---|---|
| D1 | panics on non-ASCII process/mount/iface names: byte slicing mid-char at 13 sites | `floor_char_boundary` | `utf8_boundary_tests` (9 tests fail before the fix, pass after) | the fail-first run |
| D2 | the per-interface history never forgets removed interfaces | `retain` against live ifaces | `iface_history_tests` | fail-first |
| D3 | an unbounded snapshot channel: a stalled UI (XOFF, stuck pty, slow ssh) queued one full snapshot per tick | `sync_channel(1)` + `try_send` | `a_stalled_consumer_holds_at_most_one_pending_snapshot` | `sync_channel(1_000_000)`: caught |
| D4 | stale glyphs outside the layout after a resize (the diff only rewrites drawn cells) | `needs_full_repaint` includes `size != last_size` | `a_resize_forces_a_full_repaint` | drop the size term: caught |
| D5 | an unknown `--explode` panel silently fell back to CPU | `parse_panel_type` returns `Err` | `unknown_panel_is_an_error_not_a_silent_fallback` | `"cpu" \| _ =>`: caught |
| D6 | an error path (`?`) left the collector thread running | `StopOnDrop` guard | `dropping_the_guard_stops_the_collector` | `store(true)`: caught. The first version of this test SURVIVED the mutant: it also dropped `rx`, and a disconnected channel ends the loop by itself. The test now keeps `rx` alive |
| D7 | a panic left the terminal raw, with the cursor hidden and the alt screen still up | panic hook calls `cleanup_terminal` before the default hook | **no CI test** (it needs a pty) | one-off pty proof: a panic planted after `setup_terminal`, run under `script -qfec`. With the hook, the output contains `ESC[?1049l`. With the hook removed, it does not |

Runtime mutants were run on the car base by `ttop-mut.sh` and `ttop-v5.sh`. Every mutant was
restored, checked by cksum.

## 4. pv ontology (item 4)

`contracts/aprender-viz-ttop-surface-v1.yaml` (`pv validate`: 0 errors, 0 warnings) declares:
- `cli_flags`, including `--timings`
- the key bindings for each mode: normal, help, signal_confirmation, exploded, filter_input
- `panels`

`tests/ttop_surface_gate.rs` checks both directions:
- flags against `--help`
- keys against the `app.rs` handlers
- panels against `PanelType::all()` through `parse_panel_type`
- every flag has a `docs/audits/surface_audit.csv` row (14 aprender-viz-ttop rows: 13 flags + the base command)

Its parsers carry a case table. That table caught the old help parser dropping any flag whose
description contained ", ".

## 5. --timings (item 6)

Off by default. `time(None, ..)` is exactly the closure: no `Instant::now`, no span.

When on:
- every frame phase is timed: `input`, `apply`, `layout_render`, `diff`, `write`
- every collector phase is timed: `cpu`, `mem`, `process`, `disk`, `net`, `gpu`, `analyzers`
- each phase is entered as a `tracing` span `ttop.phase`, so any subscriber sees the same
  boundaries, renacer's OTLP exporter included
- on exit it prints `{"timings":P,"n":N,"p50_us":A,"p99_us":B,"max_us":C}` per phase

Samples are kept in a ring of 4096 per phase, so memory stays bounded.

`scripts/ttop_timings_gate.sh` checks three rules:
- T1: every phase is reported with n > 0
- T2: all fields are present and p50 ≤ p99 ≤ max
- T3: nothing is printed without `--timings`

Its `--mutant` plants a 25 ms sleep in `layout_render`. The report must put it there and not in
`diff`. On the car run, layout_render p50 was 99540 µs and diff p50 was 1115 µs, so the mutant
was caught. The unit test `a_planted_sleep_lands_in_its_own_phase` does the same in-process.

No timing assertion is wired into required CI (`check_no_timing_in_required.sh` passes). The
541 fragment runs only the two classifiers' case tables.

Run on the main base (v7, three repetitions, load average around 30–50): soak GREEN 3/3 (RSS growth 1920 / 2304 / 2304 KiB, limit 8192); timings GREEN 3/3 (every phase reported). The first v6 run on this base was RED (9216 KiB, collector phases missing) because frame 1 started before the first snapshot existed. The soak now waits for it (§2). Mutants on this base: planted leak caught; planted 25 ms found in layout_render (p50 101205 µs vs diff 1363 µs).

## 6. Replay onto main

The branch was first cut from the car (228 commits ahead of main). It was replayed with
`cherry-pick -x`. `crates/aprender-viz-ttop/Cargo.toml` conflicted only in `[build-dependencies]`;
the resolution keeps `aprender-build-sha` and drops the car-only `provable-contracts` line,
which main's build.rs does not use. The replayed Cargo.lock carried the car's entry, so
`--locked` refused to run until it was regenerated.

## 7. Checks on the main base

The following are green:
- fmt
- `cargo test -p aprender-viz-ttop --lib --test ttop_surface_gate`
- present-terminal `collect_timings`/`iface_history`/`utf8_boundary` (11 tests)
- clippy `-D warnings` on aprender-viz-ttop `--all-targets` and on present-terminal `--features ptop --lib`
- the guards: explicit-test-commands, no-timing-in-required, bin-cli-tests-wired,
  pass-grep-anchored, sourced-libs-option-neutral
- `cargo deny check advisories`

## 8. Not done / known

- **Item 5, fleet dogfood.** The fleet's `ttop` is a hand-installed crates.io 2.0.0 in `~/.cargo/bin`,
  with no infra manifest. After merge, the nightly ships `aprender-viz-ttop` (#4189 derives the
  bins from workspace metadata). Swapping it in is a handoff to fleet-bins.
- **renacer soak and BrickProfiler.** Not run. The tracing spans are the integration point, but
  no renacer capture is recorded here.
- **D7** has no CI test (see §3).
- **present-terminal `--features ptop --tests` clippy.** 94 findings, all in test code (`unwrap`
  in `#[cfg(test)]` and integration tests). They are pre-existing, and no CI job lints them.
  Out of scope.
- **`brick_interface::p8_frame_budget_80x24`** (a pre-existing timing test in an integration
  target outside required CI) failed at 6093 µs against a 5000 µs budget on a host at load
  average around 42.
- **`aprender-contracts` `lint::tests::lint_empty_dir`** fails on main at aca6f2d7f6 (this
  branch touches no file in that crate): `pv lint` on an empty dir prints `decline: 0 contracts`.
