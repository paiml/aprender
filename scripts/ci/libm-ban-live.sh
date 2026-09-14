#!/usr/bin/env bash
# APEX-001 EV-2a rule 5: prove the disallowed-methods ban is LIVE, not merely written down.
#
# WHY THIS EXISTS
#
# A clippy `disallowed-methods` entry with a wrong path is a silent no-op. Measured on this
# tree, banning `f64::log10`:
#
#     f64::log10                  -> lint fires
#     <f64>::log10                -> resolves, NO lint, NO warning
#     std::f64::log10             -> resolves, NO lint, NO warning
#     core::f64::log10            -> resolves, NO lint, NO warning
#     std::primitive::f64::log10  -> warns "not a reachable function"
#
# Three of five plausible spellings enforce nothing and say nothing. So the ban cannot be
# checked by reading the file, by grepping it, or by reviewing the diff: it has to be checked
# by writing a violation and watching the build fail.
#
# It also catches the other way this rule dies: someone deletes
# `#![deny(clippy::disallowed_methods)]` from lib.rs, or adds a crate-level `allow`, and the
# list stops biting while still reading correctly.
#
# Exit 0 = every banned method is enforced. Exit 1 = at least one is not.

set -euo pipefail

CRATE_DIR="crates/aprender-viz"
LIB="$CRATE_DIR/src/lib.rs"
MARKER="__apex_ev2a_libm_ban_probe"

[ -f "$LIB" ] || { printf 'libm-ban-live: %s not found; run from the repo root\n' "$LIB" >&2; exit 1; }

# Every method the ban list names. Each must produce its own lint.
METHODS_F32="ln log10 log2 exp"
METHODS_F64="ln log10 log2 exp"

# Restore by TRUNCATING back to the byte length recorded before planting, never by
# `git checkout -- "$LIB"`. In CI those are equivalent; on a developer's machine they are not,
# and the git form silently discards every uncommitted edit in the file — measured the hard
# way while writing this script, which reverted the very `#![deny]` it exists to test.
ORIG_LEN=$(wc -c < "$LIB")

restore() {
  if [ -n "${PLANTED:-}" ]; then
    # `truncate` is coreutils; fall back to a portable rewrite if it is absent.
    if command -v truncate >/dev/null 2>&1; then
      truncate -s "$ORIG_LEN" "$LIB"
    else
      head -c "$ORIG_LEN" "$LIB" > "$LIB.restore" && mv "$LIB.restore" "$LIB"
    fi
    PLANTED=""
  fi
}
trap restore EXIT INT TERM

printf 'libm-ban-live: planting a violation of every banned method into %s\n' "$LIB"

cat >> "$LIB" <<'PROBE'

// __apex_ev2a_libm_ban_probe — planted by scripts/ci/libm-ban-live.sh, removed by it.
// If you are reading this in a committed file, the script died between planting and
// restoring; delete this block.
#[allow(dead_code, missing_docs)]
pub fn __apex_ev2a_libm_ban_probe(a: f32, b: f64) -> (f32, f64) {
    let x = a.ln() + a.log10() + a.log2() + a.exp() + a.powf(2.0) + a.powi(2);
    let y = b.ln() + b.log10() + b.log2() + b.exp() + b.powf(2.0) + b.powi(2);
    (x, y)
}
PROBE
PLANTED=1

# The plant must actually be there — a setup step that silently did nothing would make the
# rest of this script assert about an unmodified tree.
grep -q "$MARKER" "$LIB" || { printf 'libm-ban-live: FAIL — the probe was not planted\n' >&2; exit 1; }

# Touch so cargo cannot serve a cached result for the old file contents.
touch "$LIB"

set +e
OUT=$(cargo clippy -p aprender-viz --lib --message-format short 2>&1)
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  printf 'libm-ban-live: FAIL — clippy ACCEPTED a file calling every banned method.\n' >&2
  printf '  The ban list in %s/.clippy.toml is not in force. Either the paths are\n' "$CRATE_DIR" >&2
  printf '  spelled in a form clippy resolves but does not lint, or the\n' >&2
  printf '  #![deny(clippy::disallowed_methods)] in src/lib.rs is gone.\n' >&2
  exit 1
fi

# A path clippy cannot resolve at all is loud — and is still a dead ban. Fail on it directly.
if printf '%s\n' "$OUT" | grep -q "does not refer to a reachable function"; then
  printf 'libm-ban-live: FAIL — a banned path does not resolve, so it bans nothing:\n' >&2
  printf '%s\n' "$OUT" | grep "does not refer to a reachable function" >&2
  exit 1
fi

# Not just "it failed" — it must have failed for THIS reason, once per banned method.
#
# The pattern is the FULL diagnostic, `disallowed method \`f32::ln\``, not the bare method
# name. Matching the bare name made this check pass for the wrong reason: clippy's own
# "`std::f32::ln` does not refer to a reachable function" warning CONTAINS the substring
# `f32::ln`, so a misspelled, completely dead ban satisfied the test that existed to catch it.
# Measured while writing this script.
hit() { printf '%s\n' "$OUT" | grep -q "disallowed method \`$1\`"; }

MISSING=""
for t in f32 f64; do
  for m in ln log10 log2 exp powf powi; do
    hit "$t::$m" || MISSING="$MISSING $t::$m"
  done
done

HITS=$(printf '%s\n' "$OUT" | grep -c "disallowed method" || true)

if ! printf '%s\n' "$OUT" | grep -q "error: use of a disallowed method"; then
  printf 'libm-ban-live: FAIL — the bans fired as WARNINGS, not errors.\n' >&2
  printf '  `#![deny(clippy::disallowed_methods)]` is missing from %s, so clippy\n' "$LIB" >&2
  printf '  reports every violation and still exits 0. The list reads correctly and\n' >&2
  printf '  enforces nothing.\n' >&2
  exit 1
fi

if [ -n "$MISSING" ]; then
  printf 'libm-ban-live: FAIL — clippy failed, but these bans produced no diagnostic:%s\n' "$MISSING" >&2
  printf '  A banned path clippy resolves but does not lint is a silent no-op.\n' >&2
  printf '  clippy output was:\n%s\n' "$OUT" >&2
  exit 1
fi

restore
touch "$LIB"

# And the tree as committed must still be clean, or the ban is red for a reason that is not
# the plant.
if ! cargo clippy -p aprender-viz --lib >/dev/null 2>&1; then
  printf 'libm-ban-live: FAIL — the unplanted tree does not pass clippy.\n' >&2
  exit 1
fi

printf 'libm-ban-live: ok — every banned method is enforced (%s disallowed_methods diagnostics), and the clean tree passes.\n' "$HITS"
