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
# crates/ dirs outside the workspace (excluded): a change there is unplaceable -> ALL, so they need no READERS entry
NONMEMBER="${AFFECTED_NONMEMBER:-facades}"
PAT='"\.\."|"[^"]*(\.\./|/\.\.)[^"]*"|\bparent[[:space:]]*\(|Path::parent\(|Component::ParentDir|show-toplevel|\.pop\(\);[^;]*\.push\(|ancestors\(\)|current_dir|"[^"]*crates/|"crates"|CARGO_WORKSPACE_DIR|workspace_path_or_skip|read_dir\([^)]*(MANIFEST|crates|root|workspace)|format!\([^)]*(MANIFEST_DIR|crates/)'
listed() { sed -n 's/^READERS="${AFFECTED_READERS:-\(.*\)}"$/\1/p' "$SET" | tr ' ' '\n' | grep -v '^$' | sort -u; }
hits() { # $1 = crates dir; prints crate names
  { grep -rEn --include='*.rs' "$PAT" "$1" 2>/dev/null || true
    # file-level: CARGO_MANIFEST_DIR and .pop( anywhere in one file (rustfmt may wrap the let or split pop/push across lines)
    for f in $(grep -rlE --include='*.rs' 'CARGO_MANIFEST_DIR' "$1" 2>/dev/null || true); do
      grep -qE '\.pop[[:space:]]*\(' "$f" && echo "$f:0:pop-with-MANIFEST_DIR"
    done; } \
    | grep -vE '^[^:]*/target/' | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
    | awk -F: -v allow=" $ALLOW " -v nm=" $NONMEMBER " '{ k = $1 ":" $2; if (index(allow, " " k " ")) next; n = split($1, a, "/"); if (index(nm, " " a[2] " ")) next; print a[2] }' | sort -u
}
nonmember_in_workspace() { # a NONMEMBER dir that cargo metadata lists is a member: its exemption would hide a reader
  local m d
  m="$({ if [ -n "${AFFECTED_META_JSON:-}" ]; then cat "$AFFECTED_META_JSON"; else cargo metadata --no-deps --format-version 1 --offline 2>/dev/null; fi; } || true)"
  [ -n "$m" ] || return 0
  for d in $NONMEMBER; do
    if printf '%s' "$m" | grep -qF "/crates/$d/"; then echo "RED: NONMEMBER dir crates/$d is a workspace member (cargo metadata lists it): drop it from NONMEMBER"; return 1; fi
  done
}
check() {
  nonmember_in_workspace || return 1
  local bad="" l c; l="$(listed)"
  for c in $(hits "$1"); do printf '%s\n' "$l" | grep -qx "$c" || bad="$bad $c"; done
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
  expect_red 'PathBuf pop then push' zz-p 'let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR")); p.pop(); p.push("aprender-core");'
  expect_red 'Component::ParentDir' zz-q 'let _ = Component::ParentDir;'
  expect_red 'git rev-parse --show-toplevel' zz-r 'let _ = Command::new("git").args(["rev-parse", "--show-toplevel"]);'
  expect_red 'Path::parent(x)' zz-s 'let _ = Path::parent(&here);'
  expect_red '.parent ()' zz-t 'let _ = here.parent ();'
  expect_red 'a non-member dir is only exempt when named' facades-x 'let _ = here.ancestors();'
  expect_red 'rustfmt-wrapped let + split pop/push' zz-u $'let mut m =\n        PathBuf::from(env!("CARGO_MANIFEST_DIR"));\n    m.pop();\n    m.push("aprender-core");'
  expect_red 'spaced .pop (' zz-w $'let mut m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));\n    m.pop ();'
  printf '{"packages":[{"manifest_path":"/x/crates/facades/zz-evil/Cargo.toml"}]}' >"$t/meta.json"; fresh zz-v
  if (AFFECTED_META_JSON="$t/meta.json" check crates >/dev/null 2>&1); then echo "FAIL self-test: a NONMEMBER dir listed by cargo metadata passed"; fail=1; fi
  # the bench-tokenizer exemption is the one line, not the crate
  fresh aprender-bench-tokenizer; printf 'fn f() {}\nfn g() { let _ = root.join("crates"); }\n' >"$t/crates/aprender-bench-tokenizer/src/lib.rs"
  if (AFFECTED_READERS_ALLOW="crates/aprender-bench-tokenizer/src/lib.rs:1" check crates >/dev/null 2>&1); then echo "FAIL self-test: an allow entry exempted a different line"; fail=1; fi
  # a listed reader, a comment-only mention and a target/ file are green
  fresh aprender-serve; echo 'fn f() { let _ = root.join("crates"); }' >"$t/crates/aprender-serve/src/y.rs"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a listed reader was flagged"; fail=1; }
  fresh zz-o; echo '// root.join("crates")' >"$t/crates/zz-o/src/y.rs"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a comment-only line was flagged"; fail=1; }
  [ "$fail" -eq 0 ] && echo "ok: self-test"
  exit "$fail"
fi
check "${1:-crates}"
