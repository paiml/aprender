#!/usr/bin/env bash
# FLOW-010 E5: a crate whose Rust can reach OTHER crates' files at run time is a reader edge that cargo metadata
# cannot see. scripts/affected_set.sh must list it in READERS (the only list that puts a crate in a non-ALL set),
# or a change to the crate it reads would skip its tests. RED on a crate with any path-escape form and no entry.
# Over-approximates on purpose: a false positive only adds a crate to READERS (more tests run); a miss skips tests.
# Flagged in any .rs: a string holding `..`, parent(), ancestors(), current_dir, a "crates/" string or "crates",
# CARGO_WORKSPACE_DIR, workspace_path_or_skip, read_dir(...) naming MANIFEST/crates/root/workspace, format!(...)
# naming MANIFEST_DIR or crates/. Comment-only lines and files under a target/ path component are skipped.
# Usage: check_affected_readers.sh [crates-dir] | --self-test | --list
set -euo pipefail
SET="${AFFECTED_SET_SH:-scripts/affected_set.sh}"
# the exact file:line exemptions (a crate-wide exemption hides a later reader in the same crate)
ALLOW="${AFFECTED_READERS_ALLOW:-crates/aprender-bench-tokenizer/src/lib.rs:50}"   # joins ../.. only for <root>/tokenizer.json and <root>/models/
PAT='"\.\."|"[^"]*(\.\./|/\.\.)[^"]*"|\bparent\(\)|ancestors\(\)|current_dir|"[^"]*crates/|"crates"|CARGO_WORKSPACE_DIR|workspace_path_or_skip|read_dir\([^)]*(MANIFEST|crates|root|workspace)|format!\([^)]*(MANIFEST_DIR|crates/)'
# grep that is empty on rc 1 (no match) and FAILS on rc>=2 (unreadable dir, bad option): the old `|| true` read an error as "no readers"
g() { local r=0; grep "$@" || r=$?; [ "$r" -le 1 ]; }
listed() { sed -n 's/^READERS="${AFFECTED_READERS:-\(.*\)}"$/\1/p' "$SET" | tr ' ' '\n' | grep -v '^$' | sort -u; }
hits() { # $1 = crates dir; prints crate names; a grep error (rc>=2) fails the pipeline, only rc 1 (no match) is empty
  g -rEn --include='*.rs' "$PAT" "$1" 2>/dev/null \
    | g -vE '^[^:]*/target/' | g -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
    | awk -F: -v allow=" $ALLOW " '{ k = $1 ":" $2; if (index(allow, " " k " ")) next; n = split($1, a, "/"); print a[2] }' | sort -u
}
build_listed() { sed -n 's/^BUILD_CONTRACTS="\${AFFECTED_BUILD_CONTRACTS:-\(.*\)}"$/\1/p' "$SET" | tr ' ' '\n' | grep -v '^$' | sort -u; }
# Root contracts/<file> that a build.rs names (any quoted *-vN.yaml) or an include*! macro embeds: compiled into a crate, so ALL.
build_inputs() { # $1 = crates dir; root contracts dir = ${CONTRACTS_DIR:-contracts}
  { g -rhE --include=build.rs '[A-Za-z0-9_-]+\.yaml"' "$1" 2>/dev/null | g -vE '^\s*//' | g -oE '[A-Za-z0-9_-]+\.yaml"' | tr -d '"' &&
    g -rhE --include='*.rs' 'include(_str|_bytes)?!' "$1" 2>/dev/null | g -vE '^\s*//' | g -oE 'contracts/[A-Za-z0-9_-]+\.yaml' | sed 's#contracts/##'
  } | sort -u | while read -r f; do if [ -f "${CONTRACTS_DIR:-contracts}/$f" ]; then echo "$f"; fi; done | g -v 'binding\.yaml$'
}
check() {
  local bad="" l c h b; l="$(listed)"
  [ -d "$1" ] || { echo "RED: crates dir '$1' is not a directory (nothing was scanned)"; return 1; }
  h="$(hits "$1")" || { echo "RED: the READERS scan of '$1' failed (grep error); not reading that as no readers"; return 1; }
  b="$(build_inputs "$1")" || { echo "RED: the BUILD_CONTRACTS scan of '$1' failed (grep error); not reading that as no inputs"; return 1; }
  for c in $h; do grep -qx "$c" <<<"$l" || bad="$bad $c"; done
  local bl; bl="$(build_listed)"
  for c in $b; do grep -qx "$c" <<<"$bl" || bad="$bad contracts/$c(BUILD_CONTRACTS)"; done
  if [ -n "$bad" ]; then echo "RED: crates can reach other crates' files but are not in READERS (scripts/affected_set.sh):$bad"; return 1; fi
  echo "ok: every crate with a path-escape form is in READERS"
}
if [ "${1:-}" = "--list" ]; then hits crates; exit 0; fi
if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT
  SET="$(pwd)/$SET"; cd "$t"
  fail=0
  fresh() { rm -rf -- "${t:?}/crates"; mkdir -p "$t/crates/$1/src"; }   # one unlisted crate holding $2
  expect_red() { # name crate body
    fresh "$2"; printf '%s\n' "fn f() { $3 }" >"$t/crates/$2/src/x.rs"
    if (check crates >/dev/null 2>&1); then echo "FAIL self-test: not caught: $1"; fail=1; fi
  }
  expect_red 'join("crates")' zz-a 'let _ = root.join("crates");'
  expect_red 'MANIFEST_DIR.parent().parent()' zz-b 'let _ = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent();'
  expect_red 'root.join("crates/aprender-core")' zz-c 'let _ = root.join("crates/aprender-core");'
  expect_red 'format! with ../../crates' zz-d 'let _ = format!("{}/../../crates/x", env!("CARGO_MANIFEST_DIR"));'
  expect_red 'ancestors().nth(2)' zz-e 'let _ = here.ancestors().nth(2);'
  expect_red 'read_dir("../")' zz-f 'let _ = std::fs::read_dir("../");'
  expect_red 'current_dir().join("..")' zz-g 'let _ = std::env::current_dir().unwrap().join("..");'
  expect_red 'relative read_to_string' zz-h 'let _ = std::fs::read_to_string("../../crates/x/src/lib.rs");'
  expect_red 'MANIFEST_DIR.parent().join(apr-cli)' zz-i 'let _ = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("apr-cli");'
  expect_red 'workspace_path_or_skip!' zz-j 'let _ = workspace_path_or_skip!("crates/x/..");'
  expect_red 'read_dir(workspace_root)' zz-k 'let _ = std::fs::read_dir(workspace_root);'
  expect_red 'CARGO_WORKSPACE_DIR' zz-l 'let _ = env!("CARGO_WORKSPACE_DIR");'
  expect_red 'format! with MANIFEST_DIR' zz-m 'let _ = format!("{}/x", env!("CARGO_MANIFEST_DIR"));'
  expect_red 'a *contracts*/*target* crate name' zz-contracts-target-new 'let _ = root.join("crates");'
  expect_red 'a line that mentions target/contracts' zz-n 'let _ = root.join("crates"); /* target contracts docs/ */'
  # the bench-tokenizer exemption is the one line, not the crate
  fresh aprender-bench-tokenizer; printf 'fn f() {}\nfn g() { let _ = root.join("crates"); }\n' >"$t/crates/aprender-bench-tokenizer/src/lib.rs"
  if (AFFECTED_READERS_ALLOW="crates/aprender-bench-tokenizer/src/lib.rs:1" check crates >/dev/null 2>&1); then echo "FAIL self-test: an allow entry exempted a different line"; fail=1; fi
  # a listed reader, a comment-only mention and a target/ file are green
  fresh aprender-serve; echo 'fn f() { let _ = root.join("crates"); }' >"$t/crates/aprender-serve/src/y.rs"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a listed reader was flagged"; fail=1; }
  fresh zz-o; echo '// root.join("crates")' >"$t/crates/zz-o/src/y.rs"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a comment-only line was flagged"; fail=1; }
  # fail closed (#6069): a missing crates dir and a grep error (rc 2) are RED, never "no readers"; no match (rc 1) stays green
  fresh zz-p; echo 'fn f() {}' >"$t/crates/zz-p/src/y.rs"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a tree with no reader forms was flagged"; fail=1; }
  if (check "$t/no-such-crates" >/dev/null 2>&1); then echo "FAIL self-test: a missing crates dir read as no readers"; fail=1; fi
  mkdir -p "$t/shim"; printf '#!/bin/sh\nexit 2\n' >"$t/shim/grep"; chmod +x "$t/shim/grep"
  if (PATH="$t/shim:$PATH" check crates >/dev/null 2>&1); then echo "FAIL self-test: a grep error (rc 2) read as no readers"; fail=1; fi
  [ "$fail" -eq 0 ] && echo "ok: self-test"
  exit "$fail"
fi
check "${1:-crates}"
