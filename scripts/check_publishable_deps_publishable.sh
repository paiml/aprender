#!/usr/bin/env bash
# check_publishable_deps_publishable.sh — a publishable crate may not depend on
# an unpublishable one.
#
# WHY THIS EXISTS
# ---------------
# #2493 routed `apr qa-playbook` by adding a hard dependency from `apr-cli` to
# `aprender-qa-cli`. Five of the six sibling CLIs it routed are publishable;
# `aprender-qa-cli` is not -- it carries
#
#     publish = false  # Internal QA harness; reached through `apr qa`
#
# and has since at least 0.61.0. Nothing checked the combination, so the tree
# built, tested and merged clean while `apr-cli` had become impossible to
# publish:
#
#     error: failed to prepare local package for uploading
#     Caused by: no matching package named `aprender-qa-cli` found
#
# That is invisible to every normal gate. `cargo build`, `cargo test`, `cargo
# clippy` all resolve the path dependency locally and pass. It only appears at
# `cargo publish` -- i.e. during a release cascade, which is the worst possible
# time to find it. The v0.50.0 cascade died this way at 29 of 68 crates.
#
# NOTE `optional = true` DOES NOT HELP -- verified. cargo validates optional
# dependencies at publish time too, so feature-gating cannot launder an
# unpublishable dependency into a publishable crate.
#
# Text-only: reads `cargo metadata`, builds nothing.
#
#   bash scripts/check_publishable_deps_publishable.sh              # check
#   bash scripts/check_publishable_deps_publishable.sh --self-test  # case table

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Scans EVERY workspace given, as ONE graph (aprender#2559).
#
# This used to be a single `cargo metadata` on the repo root. `crates/facades/`
# is a second workspace, `exclude`d from the root, so its three publishable
# crates were outside the universe and this guard could say nothing about them
# -- the same "universe built from the wrong side" defect that let the release
# cascade report success while never shipping them.
#
# Merging the documents rather than running the scan twice is the point. Run
# separately, the facade workspace is VACUOUSLY green: `provable-contracts`
# depends on `aprender-contracts`, which is not a MEMBER of the facade
# workspace, so it never appears in that document's `unpub` set no matter what
# its publish flag says. The cross-workspace edge is exactly the edge that
# matters -- a `publish = false` on `aprender-contracts` would make the facade
# unpublishable, and only a merged graph can see it.
scan() {
    local d
    for d in "$@"; do
        ( cd "$d" && cargo metadata --no-deps --format-version 1 2>/dev/null )
    done | python3 -c '
import json,sys
# One JSON document per workspace, concatenated on stdin.
dec=json.JSONDecoder()
buf=sys.stdin.read()
members={}
i=0
while True:
    while i < len(buf) and buf[i].isspace():
        i += 1
    if i >= len(buf):
        break
    md,i = dec.raw_decode(buf,i)
    for p in md["packages"]:
        members[p["name"]]=p
# publish == [] means `publish = false`. publish == None means publishable.
unpub={n for n,p in members.items() if p.get("publish")==[]}
bad=[]
for n,p in members.items():
    if p.get("publish")==[]:
        continue
    for d in p.get("dependencies",[]) or []:
        if d.get("kind")=="dev":       # dev-deps are stripped from the published manifest
            continue
        if d["name"] in unpub:
            bad.append((n,d["name"]))
print(len(members))
for n,dn in sorted(set(bad)):
    print(f"{n}\t{dn}")
'
}

if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
    trap 'rm -rf "${TD:?}"' EXIT

    mk() { # dir name publish_false dep
        mkdir -p "$TD/$1/src"; : > "$TD/$1/src/lib.rs"
        { printf '[package]\nname = "%s"\nversion = "0.0.0"\nedition = "2021"\n' "$2"
          [ "$3" = "yes" ] && printf 'publish = false\n'
          printf '\n[dependencies]\n'
          [ -n "${4:-}" ] && printf '%s = { path = "../%s", version = "0.0.0" }\n' "$4" "$4"
        } > "$TD/$1/Cargo.toml"
    }

    # Row 1: publishable -> unpublishable MUST be reported (the #2493 defect).
    W="$TD/w1"; mkdir -p "$W"
    printf '[workspace]\nmembers = ["libx","appy"]\nresolver = "2"\n' > "$W/Cargo.toml"
    mk w1/libx libx yes ""
    mk w1/appy appy no libx
    got=$(scan "$W" | tail -n +2)
    if printf '%s' "$got" | grep -q "^appy	libx$"; then
        printf 'ok    row 1 publishable -> unpublishable is reported\n'
    else
        printf 'FAIL  row 1 not reported; got [%s]\n' "$got"; fails=1
    fi

    # Row 2 is the CONTROL. Without it row 1 passes even if this flagged every
    # dependency it saw, and the guard could never go green.
    W="$TD/w2"; mkdir -p "$W"
    printf '[workspace]\nmembers = ["libx","appy"]\nresolver = "2"\n' > "$W/Cargo.toml"
    mk w2/libx libx no ""
    mk w2/appy appy no libx
    if [ -z "$(scan "$W" | tail -n +2)" ]; then
        printf 'ok    row 2 publishable -> publishable is NOT reported\n'
    else
        printf 'FAIL  row 2 false positive on an all-publishable workspace\n'; fails=1
    fi

    # Row 3: an UNPUBLISHABLE crate may freely depend on an unpublishable one --
    # it is never uploaded, so cargo never has to resolve it from the registry.
    W="$TD/w3"; mkdir -p "$W"
    printf '[workspace]\nmembers = ["libx","appy"]\nresolver = "2"\n' > "$W/Cargo.toml"
    mk w3/libx libx yes ""
    mk w3/appy appy yes libx
    if [ -z "$(scan "$W" | tail -n +2)" ]; then
        printf 'ok    row 3 unpublishable -> unpublishable is NOT reported\n'
    else
        printf 'FAIL  row 3 flagged a pair that never reaches crates.io\n'; fails=1
    fi

    # Rows 4 and 5 are aprender#2559: the edge that CROSSES a workspace
    # boundary. `w4` is the "root", `w4x` an EXCLUDED second workspace whose
    # crate depends on an unpublishable crate in the first. This is the exact
    # shape of crates/facades -> aprender-contracts.
    W="$TD/w4"; mkdir -p "$W"
    printf '[workspace]\nmembers = ["libx"]\nresolver = "2"\n' > "$W/Cargo.toml"
    mk w4/libx libx yes ""
    WX="$TD/w4x"; mkdir -p "$WX"
    printf '[workspace]\nmembers = ["face"]\nresolver = "2"\n' > "$WX/Cargo.toml"
    mkdir -p "$WX/face/src"; : > "$WX/face/src/lib.rs"
    printf '[package]\nname = "face"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nlibx = { path = "../../w4/libx", version = "0.0.0" }\n' \
        > "$WX/face/Cargo.toml"

    # Row 4 is the CONTROL FOR THE SCOPE, and it is the row that matters: scanning
    # only the "root" workspace reports NOTHING, because `face` is not in it. A
    # guard that passes here is not lenient, it is BLIND -- and that is precisely
    # what shipped.
    if [ -z "$(scan "$W" | tail -n +2)" ]; then
        printf 'ok    row 4 the root workspace alone CANNOT see the cross-workspace edge\n'
    else
        printf 'FAIL  row 4 expected the single-workspace scan to be blind here\n'; fails=1
    fi

    # Row 5: with both workspaces merged, the same edge IS reported.
    if scan "$W" "$WX" | tail -n +2 | grep -q "^face	libx$"; then
        printf 'ok    row 5 scanning BOTH workspaces reports face -> libx\n'
    else
        printf 'FAIL  row 5 merged scan missed the cross-workspace edge; got [%s]\n' \
            "$(scan "$W" "$WX" | tail -n +2)"; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (5/5)\n'
    exit 0
fi

printf '=== a publishable crate may not depend on an unpublishable one ===\n'
# BOTH workspaces. crates/facades is `exclude`d from the root, so naming it here
# is the only way it enters the universe -- cargo will never volunteer it.
OUT="$(scan "$REPO_ROOT" "$REPO_ROOT/crates/facades")"
TOTAL="$(printf '%s' "$OUT" | head -1)"
VIOL="$(printf '%s' "$OUT" | tail -n +2 | grep . || true)"

# Vacuity: an enumeration that saw no crates would report no violations and
# look like a pass -- the shrunken-universe defect this repo keeps paying for.
if [ "${TOTAL:-0}" -lt 15 ]; then
    printf '\nFAIL (vacuity): cargo metadata reported %s member(s), expected 15+.\n' "${TOTAL:-0}"
    printf 'The ENUMERATION is broken, not the code.\n'
    exit 1
fi

printf '%s workspace member(s) scanned\n' "$TOTAL"

if [ -n "$VIOL" ]; then
    printf '\nFAIL: these publishable crates depend on unpublishable ones:\n\n'
    printf '%s\n' "$VIOL" | while IFS=$'\t' read -r a b; do
        printf '  %s  ->  %s   (publish = false)\n' "$a" "$b"
    done
    printf '\nThe dependent CANNOT be published: cargo resolves every non-dev\n'
    printf 'dependency against the registry, and `optional = true` does not help.\n'
    printf 'Either drop the dependency, or make the dependency publishable and\n'
    printf 'add it to scripts/cascade-publish.sh ahead of its dependents.\n'
    exit 1
fi

printf 'PASS: every publishable crate depends only on publishable crates.\n'
exit 0
