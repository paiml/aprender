#!/usr/bin/env bash
# check_workflow_cargo_packages.sh — every `-p <pkg>` a workflow hands cargo must
# name a real workspace member.
#
# WHY THIS EXISTS (YOGA-NIGHTLY-001 R-8, paiml/infra).
#
# .github/workflows/silicon-nightly.yml has run every night since 2026-08-29 and
# has NEVER been green. Both of its silicon legs end the same way:
#
#     error: package ID specification `aprender-primitives` did not match any packages
#     ##[error]Process completed with exit code 101
#
# There is no `aprender-primitives` in this workspace and there never has been —
# `git log -S` finds it introduced by the workflow itself (#2772) and nowhere
# else. The lane that exists to prove x86_64 and aarch64 are still tested has
# proved nothing on either, eight scheduled runs in a row, while the ledger it
# guards said the axes were covered.
#
# THE CLASS, NOT THE SYMPTOM. A package name in a workflow is a reference into
# the workspace with NOTHING checking it. Rename or delete a crate and every
# workflow that names it dies at runtime — nightly, on a self-hosted runner, hours
# after the merge, in a lane nobody reads until someone asks why it is red. Every
# other reference in this repo is compiled or linted; this one is a string.
#
# WHAT IT CHECKS. For every workflow, every logical command line containing
# `cargo`, every `-p <tok>` / `--package <tok>` / `--package=<tok>`: the token
# must be a workspace member as `cargo metadata` reports them.
#
# WHAT IT DOES NOT. Tokens carrying `$` (a shell variable or a `${{ }}`
# expression) cannot be resolved statically; they are counted and printed, never
# silently dropped. `-p` on a logical line with no `cargo` in it is not cargo's
# `-p` — `mkdir -p` is the obvious one.
#
# LOGICAL LINES, NOT PHYSICAL ONES. A `cargo test \` continued onto the next line
# puts the `-p` on a line with no `cargo` on it. Reading physical lines would
# make those references invisible, which is the exact shape of the census defect
# in paiml/infra: a detector that cannot see a form is indistinguishable from a
# repo that does not contain it.
#
# ABSENT INSTRUMENT IS A NO-GO. Without `cargo metadata` there is no list to
# compare against; this exits 2 rather than reporting a clean bill it could not
# have measured.
#
#   bash scripts/check_workflow_cargo_packages.sh              # scan
#   bash scripts/check_workflow_cargo_packages.sh --self-test  # fixtures
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WF_DIR="${WORKFLOW_DIR:-$REPO_ROOT/.github/workflows}"

# Package references in one file: `<file>\t<line>\t<token>` per hit.
# Joins backslash continuations first, so a wrapped `cargo` invocation is one
# logical line and its `-p` is attributed to the line the command STARTED on.
refs_in() {
    awk '
        {
            raw = $0
            # STRIP COMMENTS FIRST. A `#` at line start or after whitespace opens
            # a comment in BOTH languages here — YAML outside a `run:` block and
            # sh inside one — and prose describing a cargo invocation is not a
            # cargo invocation. Three of this guard'"'"'s first five findings were
            # sentences in ci.yml comments quoting `cargo test -p X`.
            sub(/(^|[[:space:]])#.*$/, "", raw)
        }
        # accumulate a logical line
        buf == "" { start = FNR }
        {
            sub(/[[:space:]]*\\[[:space:]]*$/, " ", raw)
            cont = ($0 ~ /\\[[:space:]]*$/)
            buf = buf raw
            if (cont) next
        }
        {
            if (buf ~ /cargo/) {
                n = split(buf, w, /[[:space:]]+/)
                for (i = 1; i <= n; i++) {
                    tok = ""
                    if (w[i] == "-p" || w[i] == "--package") { tok = w[i+1] }
                    else if (w[i] ~ /^--package=/) { tok = substr(w[i], 11) }
                    else if (w[i] ~ /^-p=/)        { tok = substr(w[i], 4) }
                    if (tok == "") continue
                    gsub(/^["'"'"']|["'"'"']$/, "", tok)
                    if (tok == "") continue
                    # a path, a flag, or an empty tail is not a package name
                    if (tok ~ /^-/ || tok ~ /\//) continue
                    print FILENAME "\t" start "\t" tok
                }
            }
            buf = ""
        }
    ' "$1"
}

selftest() {
    _tmp="$(mktemp -d)" || return 2
    # One fixture per FORM VARIANT: inline, continued, --package=, a dynamic
    # token, and a `-p` that belongs to mkdir rather than cargo.
    cat > "$_tmp/wf.yml" <<'FIXTURE'
jobs:
  a:
    steps:
      - run: mkdir -p /tmp/not-a-package
      - run: cargo test -p real-one --lib
      - run: |
          nice -n 19 cargo test \
            -p continued-one --release --lib
      - run: cargo build --package=equals-one
      - run: cargo test -p "${{ matrix.crate }}" --lib
FIXTURE
    _got="$(refs_in "$_tmp/wf.yml" | cut -f3 | LC_ALL=C sort | paste -sd, -)"
    # `${_tmp:?}`, not a bare expansion: bashrs SEC011 refuses an unvalidated
    # `rm -rf` on a variable, and it is right to — an empty _tmp would aim this
    # at the working directory.
    rm -rf "${_tmp:?}"
    _want='${{,continued-one,equals-one,real-one'
    if [ "$_got" != "$_want" ]; then
        printf 'INSTRUMENT BROKEN — the extractor did not find its own fixtures.\n'
        printf '  want: %s\n  got:  %s\n' "$_want" "$_got"
        return 2
    fi
    printf 'self-test: 5 forms, extractor found 4 references and skipped `mkdir -p`\n'
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    printf -- '-- instrument self-test --\n'
    selftest; exit $?
fi

printf '== every cargo -p in a workflow must name a real crate ==\n'
printf -- '-- instrument self-test --\n'
selftest || exit 2

command -v cargo >/dev/null 2>&1 || {
    printf '\nNO-GO: cargo is not on PATH, so there is no member list to compare\n'
    printf 'against. An absent instrument is a refusal, never a pass.\n'; exit 2; }
command -v jq >/dev/null 2>&1 || {
    printf '\nNO-GO: jq is not on PATH; cargo metadata cannot be read.\n'; exit 2; }

MEMBERS="$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.packages[].name' | LC_ALL=C sort -u)"
n_members="$(printf '%s\n' "$MEMBERS" | grep -c . || true)"
printf '\nworkspace members: %s\n' "$n_members"
if [ "${n_members:-0}" -lt 2 ]; then
    printf 'NO-GO: cargo metadata reported %s member(s). A comparison against an\n' "${n_members:-0}"
    printf 'empty list passes everything; refusing.\n'; exit 2
fi

[ -d "$WF_DIR" ] || { printf 'NO-GO: %s does not exist.\n' "$WF_DIR"; exit 2; }

files=0; refs=0; dynamic=0; bad=0
printf -- '\n-- references --\n'
for wf in "$WF_DIR"/*.yml "$WF_DIR"/*.yaml; do
    [ -f "$wf" ] || continue
    files=$((files + 1))
    while IFS="$(printf '\t')" read -r f ln tok; do
        [ -n "$tok" ] || continue
        case "$tok" in
            *'$'*) dynamic=$((dynamic + 1)); continue ;;
        esac
        refs=$((refs + 1))
        if printf '%s\n' "$MEMBERS" | grep -qxF -- "$tok"; then continue; fi
        bad=$((bad + 1))
        printf '  VIOLATION %s:%s  `-p %s` names no workspace member\n' \
            "${f#"$REPO_ROOT"/}" "$ln" "$tok"
    done <<EOF
$(refs_in "$wf")
EOF
done

printf -- '\n-- denominators --\n'
printf '%s workflow file(s), %s static package reference(s), %s dynamic (skipped), %s violation(s)\n' \
    "$files" "$refs" "$dynamic" "$bad"

# "0 violations" over 0 references is the failure this whole guard is modelled
# on. A repo whose workflows build nothing is not the state we are in.
if [ "$refs" -eq 0 ]; then
    printf '\nNO-GO: 0 static package references found across %s workflow file(s).\n' "$files"
    printf 'The extractor is broken, not the workflows. Fix it rather than this number.\n'
    exit 2
fi

if [ "$bad" -ne 0 ]; then
    printf '\nFAIL: a workflow hands cargo a package that does not exist.\n'
    printf 'It will fail at RUNTIME — nightly, on a self-hosted runner, in a lane\n'
    printf 'whose whole purpose is to tell you something stopped being tested.\n'
    exit 1
fi
printf '\nOK: every static `-p` in every workflow names a real workspace member.\n'
