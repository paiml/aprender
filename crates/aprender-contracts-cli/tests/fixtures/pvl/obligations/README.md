# PVL-001 EV-10 golden fixtures — `pv obligations --gate` vs `pv-obligation-gate.py`

`tests/pvl_obligations_golden.rs` holds `pv obligations <fixture> --gate` to `<fixture>.stdout`
byte for byte and to `<fixture>.rc`. Both files were written by the Python script it replaces,
never by pv.

| fixture | what | script verdict |
|---|---|---|
| `pmat/` | pmat's 35 `contracts/*.yaml` + `binding.yaml`, verbatim, and the 10 `src/` files that `git grep -l -E "fn <name>\b" -- src/` returns for the 5 function-valued `applies_to` (all in `tdg-grade-order-v1.yaml`, `proved_type: Grade`) | 0 problems over 35, exit 0 |
| `broken/` | one planted defect per check, plus a control for every rule that must stay silent | 6 problems over 4, exit 1 |

## Provenance

- pmat (`paiml/paiml-mcp-agent-toolkit`) at `d70a78f67` (2026-09-15).
- script `scripts/pv-obligation-gate.py`, sha256
  `e4fa971964f7d902d2491ef81eb667f8f1eee68a4ec411590b138c70dfad4e12` (last changed in `7c93535e`).
- PyYAML 6.0.3. The script's `pv validate` resolved to the pv built from this tree (first on `PATH`).

## Why the `src/` files end in `.rs.txt`

They are pmat's sources, and ten aprender gates scan every tracked `*.rs`. They were renamed
after copying and not otherwise changed. Neither side cares about the extension: the script's `git grep … -- src/` searches
every tracked file under `src/`, and so does pv. A name only reaches the output inside a check-3b
problem (`['src/gate.rs.txt']` in `broken.stdout`).

## `broken/`, line by line

- `a-invalid-v1.yaml`: no `metadata:`, so `pv validate` fails (check 1).
- `b-hidden-v1.yaml`: two `test:` entries under `falsification:` (check 2). The `action:`-only
  entry is an alert threshold and is not counted.
- `c-unbound-v1.yaml`: `ghost_fn_xyz` names no `fn` (check 3a). `bounded` is only found as a
  prefix of `fn bounded_v2`, and `\b` refuses that match. The controls are `all`, the equation
  `relu`, `real_fn` in a nested file, and an obligation with no `applies_to`.
- `d-proved-v1.yaml`: `grade_gate`'s file only says `GradeTable`, and `grade_prefixed`'s file
  only says `TdgGrade`. Neither says the word `Grade` (check 3b, one case for each side of
  `\b`). The control is `grade_ok`, whose file does say `Grade`.
- Excluded, and never reported:
  - `z-binding.yaml` (its name ends in `binding.yaml`);
  - `.hidden-v1.yaml` (`glob`'s `*` skips dotfiles);
  - `sub/nested-v1.yaml` (the walk is not recursive);
  - `notes.yml` (not `.yaml`).

## Re-recording

The fixture must be tracked (`git add`) first, because the script's `git grep` only sees
tracked files.

```bash
cargo build -p aprender-contracts-cli --bin pv
S=~/src/paiml-mcp-agent-toolkit/scripts/pv-obligation-gate.py   # sha256 as above
for fx in pmat broken; do
  (cd crates/aprender-contracts-cli/tests/fixtures/pvl/obligations/$fx &&
   PATH="$PWD/../../../../../../../target/debug:$PATH" python3 "$S" > ../$fx.stdout; echo $? > ../$fx.rc)
done
```

## Where pv deliberately differs

These are documented in `src/commands/obligations.rs`. No fixture here exercises them, because the
script has no verdict to record:

- zero contracts is `decline:` at exit 2 (PVL-1), where the script reports 0 over 0 at exit 0;
- non-YAML input, a non-mapping file, or a non-string `applies_to`/`proved_type` is a named
  problem, where the script dies with a traceback;
- `src/` is walked on disk, so untracked files count.
