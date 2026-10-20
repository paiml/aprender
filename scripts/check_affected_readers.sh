#!/usr/bin/env bash
# FLOW-010 E5: a crate whose Rust reads OTHER crates' files at run time (join("crates"), ../<crate>/..,
# CARGO_MANIFEST_DIR/../..) is a reader edge that cargo metadata cannot see. scripts/affected_set.sh must
# list it in CORE or READERS, or a change to the crate it reads would skip its tests. RED on a reader
# neither list names. Usage: check_affected_readers.sh [crates-dir]   (self-test: --self-test)
set -euo pipefail
dir="${1:-crates}"
SET="${AFFECTED_SET_SH:-scripts/affected_set.sh}"
listed() { sed -n 's/^CORE="${AFFECTED_CORE:-\(.*\)}"$/\1/p; s/^READERS="${AFFECTED_READERS:-\(.*\)}"$/\1/p' "$SET" | tr ' ' '\n' | grep -v '^$' | sort -u; }
# reader patterns; lines about target/, .config, contracts/, docs/ read non-crate paths (an unknown root path is ALL anyway)
readers() {
  grep -rEn --include='*.rs' 'join\("crates"\)|\.\./aprender-|MANIFEST_DIR.*\.\.|join\("\.\./\.\."\)' "$1" 2>/dev/null \
    | grep -vE 'target|\.config|contracts|docs/|^\S+:[0-9]+:\s*//' | awk -F/ '{print $2}' | sort -u || true
}
# aprender-bench-tokenizer joins ../.. only to find tokenizer.json / models/ at the repo root, never another crate's file
ALLOW=" aprender-bench-tokenizer "
check() {
  local bad="" l; l="$(listed)"
  for c in $(readers "$1"); do
    case "$ALLOW" in *" $c "*) continue ;; esac
    printf '%s\n' "$l" | grep -qx "$c" || bad="$bad $c"
  done
  if [ -n "$bad" ]; then echo "RED: crates read other crates' files but are in neither CORE nor READERS (scripts/affected_set.sh):$bad"; return 1; fi
  echo "ok: every cross-crate reader is listed"
}
if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT
  mkdir -p "$t/crates/zz-new/tests" "$t/crates/aprender-serve/src"
  echo 'fn f() { let _ = root.join("crates"); }' >"$t/crates/zz-new/tests/x.rs"
  echo 'fn f() { let _ = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."); }' >"$t/crates/aprender-serve/src/y.rs"
  cd "$t"; SET="${OLDPWD}/${SET}"
  if (check crates >/dev/null 2>&1); then echo "FAIL self-test: an unlisted reader was not caught"; exit 1; fi
  rm -r crates/zz-new
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a listed reader was flagged"; exit 1; }
  echo "ok: self-test"; exit 0
fi
check "$dir"
