#!/usr/bin/env bash
# review_tier.sh -- the review tier of a change, derived from its changed paths (FLOW-011 rev6 section 4, MT-13).
#
#   bash scripts/review_tier.sh BASE HEAD [--table FILE]   -> prints light | standard | sensitive
#   bash scripts/review_tier.sh --self-test
#
# RULES. The tier comes from .pmat/review-tiers.yaml and the diff, never from the PR text: there is no flag, label
# or trailer that sets it. A diff spanning tiers takes the highest. A path matching `sensitive` is sensitive,
# then one matching `light` is light; any other path is standard unless its diff is comment-only, which is
# checked here by stripping blank lines and whole-line comments from both sides and comparing (never by the
# author's word). A table that cannot be read, or lists nothing, is an error (exit 2): an unreadable table must
# not quietly make every change light. The table is read from BASE, not HEAD, so a PR cannot loosen the rules it
# is judged by; a BASE without a table (bootstrap) makes every path sensitive, never the
# working tree's copy, which the PR could have loosened; --table FILE is the explicit override.
set -euo pipefail

TABLE=${REVIEW_TIER_TABLE:-.pmat/review-tiers.yaml}
TABLE_FILE=

die() { echo "review_tier: $*" >&2; exit 2; }

# list_of KEY FILE -> the quoted globs under `KEY:` in the table, one per line.
list_of() {
  awk -v k="$1:" '
    $0 ~ /^[a-z]/ { on = ($1 == k); next }
    # Anything in a list that is not exactly `  - "glob"` (unquoted, trailing comment, wrong indent) would be
    # dropped silently and make its paths fall to standard: refuse the table instead.
    on && $0 !~ /^[ \t]*(#.*)?$/ {
      if ($0 !~ /^  - "[^"]+"$/) { print "review_tier: malformed table line: " $0 > "/dev/stderr"; bad = 1; exit 2 }
      s = $0; sub(/^  - "/, "", s); sub(/"$/, "", s); print s
    }
    END { exit bad }
  ' "$2"
}

matches_any() { # PATH, globs on stdin
  local f=$1 g
  while IFS= read -r g; do
    [[ -n $g ]] || continue
    # shellcheck disable=SC2053
    [[ $f == $g ]] && return 0
  done
  return 1
}

# strip_comments EXT -> stdin without blank lines and whole-line comments of that file type.
strip_comments() {
  case $1 in
    # `///` and `//!` are doc comments: rustdoc runs the examples in them, so they are code to a reviewer.
    rs) grep -Ev '^[[:space:]]*(//([^/!].*)?)?$' || true ;;
    c|h|cu|js|ts|go|java) grep -Ev '^[[:space:]]*(//.*)?$' || true ;;
    # Line 1 `#!` is the interpreter, not a comment: changing it changes what runs.
    sh|py|yaml|yml|toml|cmd|mk|rb) awk 'NR == 1 && /^#!/ { print; next } !/^[ \t]*(#.*)?$/' ;;
    *) cat ;;
  esac
}

comment_only() { # BASE HEAD PATH -> 0 when the change to PATH touches only blank lines and comments
  local ext=${3##*.} before after b_raw a_raw
  [[ $3 == Makefile ]] && ext=mk
  b_raw=$(git show "$1:$3" 2>/dev/null; echo x) && b_raw=${b_raw%x} || b_raw=
  a_raw=$(git show "$2:$3" 2>/dev/null; echo x) && a_raw=${a_raw%x} || a_raw=
  # A mode change (chmod +x) or an added/removed file changes what runs without changing a line.
  [[ $(git ls-tree "$1" -- "$3" | cut -d' ' -f1) == "$(git ls-tree "$2" -- "$3" | cut -d' ' -f1)" ]] || return 1
  # Neither side readable (a path git could not name): nothing was compared, so it is not comment-only.
  git cat-file -e "$1:$3" 2>/dev/null || git cat-file -e "$2:$3" 2>/dev/null || return 1
  before=$(printf '%s' "$b_raw" | strip_comments "$ext")
  after=$(printf '%s' "$a_raw" | strip_comments "$ext")
  [[ $before == "$after" ]]
}

read_by_code() { # BASE HEAD PATH -> 0 when some file other than PATH names PATH
  local t
  for t in "$1" "$2"; do
    git grep -qF -e "$3" "$t" -- crates src tests scripts ci .github Makefile build.rs ':!'"$3" 2>/dev/null && return 0
  done
  return 1
}

rank() { case $1 in light) echo 0 ;; standard) echo 1 ;; *) echo 2 ;; esac; }

tier_of() { # BASE HEAD PATH
  if matches_any "$3" <<< "$SENS"; then echo sensitive
  elif matches_any "$3" <<< "$STD"; then echo standard   # carve-outs of a light glob (e.g. include_str! docs)
  elif matches_any "$3" <<< "$LIGHT"; then
    # Light is an allowlist that fails closed: a file is light only if no code names it. A path that any
    # source, script, workflow or Makefile at BASE or HEAD mentions (include_str!, include!, a gate reading it) is standard.
    if read_by_code "$1" "$2" "$3"; then echo standard; else echo light; fi
  elif comment_only "$1" "$2" "$3"; then echo light
  else echo standard
  fi
}

SENS=; LIGHT=; STD=
derive() { # BASE HEAD
  local f t best=light n=0 missing=0
  local tbl; tbl=$(mktemp); trap 'rm -f "${tbl:?}"' RETURN
  # A BASE without the table (the PR that introduces it, an old base) is never judged by the working tree's copy,
  # which the PR itself could have loosened: every path is sensitive. --table FILE is the explicit override.
  if [[ -n $TABLE_FILE ]]; then cp "$TABLE_FILE" "$tbl" 2>/dev/null || die "table $TABLE_FILE is unreadable"
  elif ! git show "$1:$TABLE" > "$tbl" 2>/dev/null; then missing=1; fi
  if [[ $missing -eq 0 ]]; then
    SENS=$(list_of sensitive "$tbl") || die "table $TABLE is malformed"; LIGHT=$(list_of light "$tbl") || die "table $TABLE is malformed"
    STD=$(list_of standard "$tbl") || die "table $TABLE is malformed"
    [[ -n $SENS && -n $LIGHT ]] || die "table $TABLE lists no sensitive or no light paths"
  fi
  while IFS= read -r f; do
    [[ -n $f ]] || continue
    n=$((n + 1))
    if [[ $missing -eq 1 ]]; then t=sensitive; else t=$(tier_of "$1" "$2" "$f"); fi
    (( $(rank "$t") > $(rank "$best") )) && best=$t
  done < <(git -c core.quotePath=false diff --name-only -z --no-renames "$1" "$2" | tr '\0' '\n')
  [[ $n -gt 0 ]] || die "no changed paths between $1 and $2"
  echo "$best"
}

self_test() {
  local d fail=0 pass=0 base
  d=$(mktemp -d); trap 'rm -rf "${d:?}"' RETURN
  git -C "$d" init -q; git -C "$d" config user.email t@t; git -C "$d" config user.name t
  mkdir -p "$d/.pmat" "$d/.github/workflows" "$d/crates/x/src" "$d/docs" "$d/scripts" "$d/bin"
  cp "$TABLE" "$d/.pmat/review-tiers.yaml"
  unset REVIEW_TIER_TABLE   # the rows run children that must read the table from the scratch repo
  printf 'fn a() {}\n// note\n' > "$d/crates/x/src/lib.rs"
  printf 'name: ci\n' > "$d/.github/workflows/ci.yml"
  printf '# doc\n' > "$d/docs/a.md"
  printf '' > "$d/scripts/.keep"; printf 'echo hi\n' > "$d/bin/run.sh"
  printf 'doc\n' > "$d/docs/read.md"; printf 'const D: &str = include_str!("../../../docs/read.md");\n' > "$d/crates/x/src/inc.rs"
  git -C "$d" add -A; git -C "$d" commit -qm base; base=$(git -C "$d" rev-parse HEAD)
  expect() { # NAME WANT -- mutation commands run in $d
    local name=$1 want=$2 got
    git -C "$d" add -A; git -C "$d" commit -qm "$name" --allow-empty
    got=$(cd "$d" && TABLE=.pmat/review-tiers.yaml bash "$SELF" "$base" HEAD 2>&1) || true
    if [[ $got == "$want" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $name: want $want, got $got"; fi
    git -C "$d" reset -q --hard "$base"
  }
  SELF=$(realpath "${BASH_SOURCE[0]}")
  printf '# more\n' >> "$d/docs/a.md";                                          expect docs-only light
  printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs";                              expect code-change standard
  printf '// only a comment\n\n' >> "$d/crates/x/src/lib.rs";                    expect comment-only-rs light
  printf '// c\nfn b() {}\n' >> "$d/crates/x/src/lib.rs";                        expect comment-plus-code standard
  printf '# c\n' >> "$d/bin/run.sh";                                         expect comment-only-sh light
  printf 'on: push\n' >> "$d/.github/workflows/ci.yml";                          expect workflow-change sensitive
  printf '# just a comment\n' >> "$d/.github/workflows/ci.yml";                  expect workflow-comment-only-is-still-sensitive sensitive
  printf '# d\n' >> "$d/docs/a.md"; printf 'on: push\n' >> "$d/.github/workflows/ci.yml"; expect docs-plus-workflow-highest-wins sensitive
  printf '# d\n' >> "$d/docs/a.md"; printf 'fn b() {}\n' >> "$d/crates/x/src/lib.rs";   expect docs-plus-code-standard standard
  printf 'x\n' > "$d/scripts/check_new.sh";                                      expect new-gate-script sensitive
  printf 'version: 1\n' > "$d/.pmat/review-tiers.yaml";                          expect table-itself sensitive
  printf 'version: 1\nsensitive:\n  - "nothing"\nlight:\n  - "*"\n' > "$d/.pmat/review-tiers.yaml"; printf 'on: push\n' >> "$d/.github/workflows/ci.yml"; expect loosened-table-in-same-pr-ignored sensitive
  printf 'x\n' > "$d/scripts/review_tier.sh";                                  expect the-tier-script-itself sensitive
  printf 'on: push\n' > "$d/.github/workflows/é.yml";                           expect non-ascii-workflow-name sensitive
  printf 'x\n' > "$d/crates/x/src/é.rs";                                        expect non-ascii-code-path standard
  printf '/// ```\n/// assert!(false);\n/// ```\nfn a() {}\n' > "$d/crates/x/src/lib.rs"; expect doc-comment-change-standard standard
  printf '//! crate docs\n' >> "$d/crates/x/src/lib.rs";                        expect inner-doc-comment-standard standard
  printf '#!/bin/bash\necho hi\n' > "$d/bin/run.sh";                        expect shebang-change-standard standard
  mkdir -p "$d/.claude/skills/x" "$d/notes"
  printf 'read by a gate\n' > "$d/.claude/skills/x/SKILL.md";                  expect gate-read-claude-md sensitive
  mkdir -p "$d/.agents"; printf 'x\n' > "$d/.agents/a.md"; expect agents-dir sensitive
  mkdir -p "$d/scripts/ci" "$d/scripts/lib" "$d/.config"
  printf 'x\n' > "$d/scripts/ci/fat_driver.py";                               expect scripts-ci sensitive
  printf 'x\n' > "$d/scripts/lib/x.py";                                        expect scripts-lib sensitive
  printf 'x\n' > "$d/.config/nextest.toml";                                    expect nextest-config sensitive
  printf 'x\n' > "$d/.github/pr-review.pub";                                    expect pr-review-key sensitive
  printf 'x\n' > "$d/scripts/guard_tree.sh";                                   expect guard-tree sensitive
  printf 'x\n' > "$d/scripts/cargo_step.sh";                                   expect cargo-step sensitive
  printf 'x\n' > "$d/scripts/new.baseline.txt";                               expect baseline-txt sensitive
  printf 'x\n' > "$d/.pmat-gates.toml";                                        expect pmat-gates-toml sensitive
  printf 'x\n' > "$d/.clippy.toml";                                            expect clippy-toml sensitive
  mkdir -p "$d/docs/roadmaps"; printf 'x\n' > "$d/docs/roadmaps/roadmap-base.yaml"; expect roadmap-base sensitive
  printf 'x\n' > "$d/README.md";                                               expect readme-light light
  printf 'x\n' > "$d/notes/n.md";                                              expect stray-md-standard standard
  chmod +x "$d/bin/run.sh";                                                 expect mode-only-change standard
  mkdir -p "$d/docs/audits" "$d/docs/roadmaps" "$d/docs/tools" "$d/crates/x/src/lib" "$d/sub"
  printf 'x\n' > "$d/docs/audits/quorum-1.json";                                expect docs-audit-receipt sensitive
  printf 'x\n' > "$d/docs/CODEOWNERS";                                          expect docs-codeowners sensitive
  printf 'x\n' > "$d/docs/roadmaps/roadmap.yaml";                               expect roadmap-yaml-sensitive sensitive
  printf 'x\n' > "$d/docs/tools/run.sh";                                        expect script-under-docs-not-light standard
  printf 'x\n' > "$d/crates/x/clippy.toml";                                     expect crate-clippy-toml sensitive
  printf 'x\n' > "$d/rust-toolchain";                                           expect rust-toolchain-file sensitive
  printf 'x\n' > "$d/.gitmodules";                                              expect gitmodules sensitive
  mkdir -p "$d/sub"; printf 'x\n' > "$d/sub/.gitignore";                                           expect nested-gitignore sensitive
  printf 'x\n' > "$d/GNUmakefile";                                              expect gnumakefile sensitive
  mkdir -p "$d/sub"; printf 'x\n' > "$d/sub/rules.mk";                                             expect mk-file sensitive
  printf 'x\n' > "$d/crates/x/Cargo.lock";                                      expect nested-cargo-lock sensitive
  printf 'more\n' >> "$d/docs/read.md";                                       expect doc-read-by-code-standard standard
  printf 'doc\n' > "$d/docs/unread.md"; expect unread-doc-light light
  mkdir -p "$d/docs/specifications"; printf 'x\n' > "$d/docs/specifications/s.md";   expect spec-md-not-light standard
  printf '#!/bin/sh\nx\n' > "$d/crates/x/data.txt"; printf '# data\n' >> "$d/crates/x/data.txt"; expect txt-hash-line-is-data standard
  printf 'x\n' > "$d/key.pem";                                                   expect key-file sensitive
  # an author-set tier is not read: the commit message and a trailer say light, the path says sensitive
  printf 'on: push\n' >> "$d/.github/workflows/ci.yml"
  git -C "$d" add -A; git -C "$d" commit -qm "tier: light
PMAT-Tier: light"
  got=$(cd "$d" && bash "$SELF" "$base" HEAD 2>&1) || true
  if [[ $got == sensitive ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL author-set-tier: got $got"; fi
  git -C "$d" reset -q --hard "$base"
  # an empty diff has no tier: error, never light
  if (cd "$d" && bash "$SELF" HEAD HEAD >/dev/null 2>&1); then fail=$((fail + 1)); echo "FAIL empty-diff accepted"; else pass=$((pass + 1)); fi
  # a BASE without the table is never judged by the PR's own (loosened) copy: sensitive
  git -C "$d" checkout -q --orphan nobase; git -C "$d" rm -rqf . ; printf 'x\n' > "$d/f"; git -C "$d" add -A; git -C "$d" commit -qm nobase
  nb=$(git -C "$d" rev-parse HEAD); mkdir -p "$d/.pmat"
  printf 'version: 1\nsensitive:\n  - "nothing"\nlight:\n  - "*"\n' > "$d/.pmat/review-tiers.yaml"; printf 'on: push\n' > "$d/ci.yml"
  git -C "$d" add -A; git -C "$d" commit -qm loosened
  got=$(cd "$d" && bash "$SELF" "$nb" HEAD 2>&1) || true
  if [[ $got == sensitive ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL missing-base-table: want sensitive, got $got"; fi
  git -C "$d" checkout -q -f master 2>/dev/null || git -C "$d" checkout -q -f main; git -C "$d" reset -q --hard "$base"
  # a malformed table line (unquoted, trailing comment) is an error, never a silently dropped glob
  printf '# t\n' >> "$d/docs/a.md"; git -C "$d" add -A; git -C "$d" commit -qm tbl   # a real diff, so only the table can fail
  for bad in '  - .github/*' '  - ".github/*"  # gate'; do
    printf 'version: 1\nsensitive:\n  - "ci/*"\n%s\nlight:\n  - "docs/*"\n' "$bad" > "$d/bad.yaml"
    if (cd "$d" && bash "$SELF" "$base" HEAD --table bad.yaml >/dev/null 2>&1); then fail=$((fail + 1)); echo "FAIL malformed-table-accepted: $bad"; else pass=$((pass + 1)); fi
  done
  git -C "$d" reset -q --hard "$base"
  # an unreadable or empty table is an error, never a quiet light
  printf 'version: 1\n' > "$d/.pmat/review-tiers.yaml"; git -C "$d" add -A; git -C "$d" commit -qm emptytable; base=$(git -C "$d" rev-parse HEAD)
  printf '# x\n' >> "$d/docs/a.md"; git -C "$d" add -A; git -C "$d" commit -qm e
  if (cd "$d" && bash "$SELF" "$base" HEAD >/dev/null 2>&1); then
    fail=$((fail + 1)); echo "FAIL empty-table accepted"; else pass=$((pass + 1)); fi
  echo "review_tier self-test: $pass passed, $fail failed"
  [[ $fail -eq 0 ]]
}

# Not a guard: its proof is check_review_tier.sh. A `--self-test)` case arm here would make
# check_guards_are_wired.sh count this helper as a guard that no workflow names.
[[ ${1:-} == --self-test ]] && { self_test; exit $?; }
# --classify PATH: is PATH sensitive in the working-tree table (used by check_review_tier.sh to scan CI surfaces).
if [[ ${1:-} == --classify ]]; then
  SENS=$(list_of sensitive "${REVIEW_TIER_TABLE:-.pmat/review-tiers.yaml}") || die "table is malformed"; [[ -n $SENS ]] || die "table lists no sensitive paths"
  if matches_any "${2:?usage: review_tier.sh --classify PATH}" <<< "$SENS"; then echo sensitive; else echo other; fi
  exit 0
fi
case ${1:-} in
  "") die "usage: review_tier.sh BASE HEAD [--table FILE] | --self-test" ;;
  *)
    [[ ${3:-} == --table ]] && TABLE_FILE=${4:?}
    derive "$1" "${2:?usage: review_tier.sh BASE HEAD}" ;;
esac
