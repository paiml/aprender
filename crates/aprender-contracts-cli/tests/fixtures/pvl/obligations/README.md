# PVL-001 EV-10 golden fixtures — `pv obligations --gate` vs `pv-obligation-gate.py`

`tests/pvl_obligations_golden.rs` holds `pv obligations <fixture> --gate` to `<fixture>.stdout`
byte for byte and to `<fixture>.rc`. Both files were written by the Python script it replaces,
never by pv.

| fixture | what | script verdict |
|---|---|---|
| `pmat/` | pmat's 35 `contracts/*.yaml` + `binding.yaml`, verbatim, and the 10 `src/` files that `git grep -l -E "fn <name>\b" -- src/` returns for the 5 function-valued `applies_to` (all in `tdg-grade-order-v1.yaml`, `proved_type: Grade`) | 0 problems over 35, exit 0 |
| `broken/` | one planted defect per check, plus a control for every rule that must stay silent | 6 problems over 4, exit 1 |
| `edge/` | YAML the script reads through PyYAML and Python truthiness: merge keys and falsy values | 3 problems over 4, exit 1 |

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

## `edge/`, line by line

- `e-merge-v1.yaml`: two entries under `falsification:` have no `test:` of their own. `F-1`
  merges one in with `<<: *hidden`, and `F-2` merges it through `chained`, a mapping that
  merges in turn. PyYAML resolves both, so the script counts 2 (check 2). `F-0` holds the
  templates one level down and is not counted.
- `f-falsy-v1.yaml`: `proved_type: 0`, and `applies_to` set to `0`, `[]`, `{}`, `false` and `""`.
  Python treats every one of them as absent, so the only problem is `pv validate` (check 1).
  `real_fn` is bound and never checked against the falsy type.
- `g-unused-type-v1.yaml`: `proved_type: 7`, which no function target reaches, so the script
  never evaluates it and reports nothing.
- `h-false-doc-v1.yaml`: the document is `false`. `yaml.safe_load(...) or {}` makes it `{}`, so
  only `pv validate` fails.

## Re-recording

The fixture must be tracked (`git add`) first, because the script's `git grep` only sees
tracked files.

```bash
cargo build -p aprender-contracts-cli --bin pv
S=~/src/paiml-mcp-agent-toolkit/scripts/pv-obligation-gate.py   # sha256 as above
for fx in pmat broken edge; do
  (cd crates/aprender-contracts-cli/tests/fixtures/pvl/obligations/$fx &&
   PATH="$PWD/../../../../../../../target/debug:$PATH" python3 "$S" > ../$fx.stdout; echo $? > ../$fx.rc)
done
```

## Where pv deliberately differs

These are documented in `src/commands/obligations.rs`. No fixture here exercises them, because the
script has no verdict to record:

- zero contracts is `decline:` at exit 2 (PVL-1), where the script reports 0 over 0 at exit 0;
- an input the script dies on with a traceback is a named problem. The module doc lists each
  one, and `a_non_string_proved_type_is_named_where_a_bound_fn_reaches_it` tests one;
- YAML 1.2 (`serde_yaml`) against PyYAML's YAML 1.1: a plain `no` is a string here;
- a document `serde_yaml` refuses and PyYAML reads (a duplicate key, an integer beyond 64 bits)
  fails `pv validate` on both sides. pv reports only that, as the script does, but cannot run
  checks 2 and 3 on it, so a problem PyYAML would also find there goes unnamed (the verdict
  agrees). `a_document_serde_yaml_refuses_fails_validate_so_the_verdict_agrees` holds both
  shapes to the script's measured output;
- `src/` is walked on disk, so untracked files count.
