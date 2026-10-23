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
excluded() { # the root Cargo.toml exclude = [...] entries: separate workspaces, not members; a diff there is unplaceable, so affected_set.sh answers ALL
  sed -n '/^exclude *= *\[/,/^\]/p' "${AFFECTED_ROOT_MANIFEST:-Cargo.toml}" 2>/dev/null | grep -v '^[[:space:]]*#' | grep -oE "[\"'][^\"']+[\"']" | tr -d "\"'" || true
}
pkg_of() { # $1 = a directory holding a flagged file, $2 = crates dir; prints the cargo PACKAGE name of the nearest Cargo.toml (affected_set.sh READERS holds package names, not directory names)
  local d="$1" e
  while [ "$d" != "$2" ] && [ "$d" != . ] && [ "$d" != / ]; do
    if [ -f "$d/Cargo.toml" ]; then
      while IFS= read -r e; do case "$d/" in "$e"/*) return 0 ;; esac; done < <(excluded)
      awk '/^\[package\]/{p=1;next} /^\[/{p=0} p && /^name *=/{gsub(/^name *= *"|".*$/,""); print; exit}' "$d/Cargo.toml"; return 0
    fi
    d="${d%/*}"
  done
  local top="${1#"$2"/}"; echo "${top%%/*}"   # no manifest on the way up: the top directory under the crates dir
}
hits() { # $1 = crates dir; prints package names
  g -rEn --include='*.rs' "$PAT" "$1" 2>/dev/null \
    | g -vE '^[^:]*/target/' | g -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
    | awk -F: -v allow=" $ALLOW " '{ k = $1 ":" $2; if (index(allow, " " k " ")) next; d = $1; sub("/[^/]*$", "", d); print d }' | sort -u \
    | while IFS= read -r d; do pkg_of "$d" "$1"; done | sort -u
}
build_listed() { # full repo-relative paths: contracts/<BUILD_CONTRACTS entry>, docs/<BUILD_DOCS entry>
  sed -n 's/^BUILD_CONTRACTS="\${AFFECTED_BUILD_CONTRACTS:-\(.*\)}"$/\1/p' "$SET" | tr ' ' '\n' | grep -v '^$' | sed 's#^#contracts/#' || true
  sed -n 's/^BUILD_DOCS="\${AFFECTED_BUILD_DOCS:-\(.*\)}"$/\1/p' "$SET" | tr ' ' '\n' | grep -v '^$' | sed 's#^#docs/#' || true
}
# Any contracts/** or docs/** file (nested too) that a build.rs names or an include*! macro embeds (the macro may span lines): compiled into a crate, so ALL.
# Over-approximates on purpose (a path anywhere in the 300 chars after the macro name); a file that does not exist is dropped. Paths relative to the repo root = ${AFFECTED_ROOT_DIR:-.}.
build_inputs() { # $1 = crates dir
  local p='(contracts|docs)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+' f
  {
    g -rzohE --include='*.rs' 'include(_str|_bytes)?!.{0,300}' "$1" 2>/dev/null | tr '\0' '\n' | g -oE "$p"   # -z: one record per file, so a macro split over lines still matches
    g -rlE --include=build.rs '(contracts|docs)/|[A-Za-z0-9_-]+\.yaml"' "$1" 2>/dev/null | while IFS= read -r f; do
      grep -vE '^\s*//' "$f" | grep -oE "$p" || true
      grep -vE '^\s*//' "$f" | grep -oE '[A-Za-z0-9_-]+\.yaml"' | tr -d '"' | sed 's#^#contracts/#' || true; done
  } | sort -u | while read -r f; do if [ -f "${AFFECTED_ROOT_DIR:-.}/$f" ]; then echo "$f"; fi; done | g -vE '(^|/)binding\.yaml$|/model-families/'
}
check() {
  local bad="" l c h b; l="$(listed)"
  [ -d "$1" ] || { echo "RED: crates dir '$1' is not a directory (nothing was scanned)"; return 1; }
  h="$(hits "$1")" || { echo "RED: the READERS scan of '$1' failed (grep error); not reading that as no readers"; return 1; }
  b="$(build_inputs "$1")" || { echo "RED: the BUILD_CONTRACTS scan of '$1' failed (grep error); not reading that as no inputs"; return 1; }
  for c in $h; do grep -qx "$c" <<<"$l" || bad="$bad $c"; done
  local bl; bl="$(build_listed)"
  for c in $b; do grep -qx "$c" <<<"$bl" || bad="$bad $c(BUILD_CONTRACTS/BUILD_DOCS)"; done
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
  # the package name, not the directory, is what READERS holds: crates/zz-dir/inner has package zz-pkg
  fresh zz-dir; mkdir -p "$t/crates/zz-dir/inner/src"; printf '[package]\nname = "zz-pkg"\n' >"$t/crates/zz-dir/inner/Cargo.toml"
  echo 'fn f() { let _ = root.join("crates"); }' >"$t/crates/zz-dir/inner/src/x.rs"
  (check crates >/dev/null 2>&1) && { echo "FAIL self-test: a reader whose directory is listed but whose package is not was passed"; fail=1; }
  out=$(check crates 2>&1 || true); grep -q 'zz-pkg' <<<"$out" || { echo "FAIL self-test: the RED did not name the package"; fail=1; }
  # ...but a root-manifest exclude entry (a separate workspace) is not a member and is skipped
  printf 'exclude = [\n    # a comment "with quotes"\n    "crates/zz-dir",\n]\n' >"$t/Cargo.toml"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: an excluded workspace was flagged"; fail=1; }
  rm -f -- "${t:?}/Cargo.toml"
  # a nested contracts/ or docs/ file that a macro embeds (split over lines too) must be listed
  mkdir -p "$t/contracts/sub" "$t/docs/d"; : >"$t/contracts/sub/n.yaml"; : >"$t/docs/d/m.md"
  fresh zz-p; printf 'const A: &str = include_str!(\n    "contracts/sub/n.yaml");\n' >"$t/crates/zz-p/src/x.rs"
  (check crates >/dev/null 2>&1) && { echo "FAIL self-test: a nested contracts/ include_str! (multi-line) was passed unlisted"; fail=1; }
  sed 's#^BUILD_CONTRACTS="${AFFECTED_BUILD_CONTRACTS:-#&sub/n.yaml #; s#^BUILD_DOCS="${AFFECTED_BUILD_DOCS:-#&d/m.md #' "$SET" >"$t/set-listed.sh"
  (SET="$t/set-listed.sh" check crates >/dev/null 2>&1) || { echo "FAIL self-test: a listed nested build input was flagged"; fail=1; }
  fresh zz-q; printf 'const A: &str = include_str!("docs/d/m.md");\n' >"$t/crates/zz-q/src/x.rs"
  (check crates >/dev/null 2>&1) && { echo "FAIL self-test: an include_str!ed docs/ file was passed unlisted"; fail=1; }
  fresh zz-r
  printf 'const A: &str = include_str!("docs/d/m.md");\n' >"$t/crates/zz-r/src/x.rs"
  (SET="$t/set-listed.sh" check crates >/dev/null 2>&1) || { echo "FAIL self-test: a listed docs build input was flagged"; fail=1; }
  # fail closed (#6069): a missing crates dir and a grep error (rc 2) are RED, never "no readers"; no match (rc 1) stays green
  fresh zz-s; echo 'fn f() {}' >"$t/crates/zz-s/src/y.rs"
  (check crates >/dev/null 2>&1) || { echo "FAIL self-test: a tree with no reader forms was flagged"; fail=1; }
  if (check "$t/no-such-crates" >/dev/null 2>&1); then echo "FAIL self-test: a missing crates dir read as no readers"; fail=1; fi
  mkdir -p "$t/shim"; printf '#!/bin/sh\nexit 2\n' >"$t/shim/grep"; chmod +x "$t/shim/grep"
  if (PATH="$t/shim:$PATH" check crates >/dev/null 2>&1); then echo "FAIL self-test: a grep error (rc 2) read as no readers"; fail=1; fi
  [ "$fail" -eq 0 ] && echo "ok: self-test"
  exit "$fail"
fi
check "${1:-crates}"
