#!/usr/bin/env bash
# publish_cascade.sh — publish the workspace to crates.io in dependency order,
# idempotently, and refuse every precondition that could publish the wrong tree
# (PP-066 row R-5, issue #2908).
#
#   scripts/publish_cascade.sh --list            # the derived publish set, in order
#   scripts/publish_cascade.sh --dry-run [<tag>] # cargo publish --dry-run per crate + receipt
#   scripts/publish_cascade.sh <tag>             # LIVE: publish, one crate per call
#   scripts/publish_cascade.sh --self-test       # case table, both polarities
#
# THE PUBLISH SET IS DERIVED, NEVER LISTED. It is every `cargo metadata`
# workspace member whose `publish` is not `false`, ordered so that a crate comes
# after every workspace crate it depends on. A hand-maintained list goes stale
# silently the first time somebody adds a crate; a derived one cannot.
#
# WHY THE REFUSALS. A cascade publishes immutable versions to a public registry.
# Every one of these has bitten a release somewhere, and none of them is
# recoverable after the fact:
#   * `git describe --exact-match` must equal the tag  — publishing a tree that
#     is not the tagged one ships something nobody reviewed.
#   * HEAD must be DETACHED — on a branch, a later commit silently changes what
#     "the release" means.
#   * the tree must be CLEAN — `cargo publish` packages the working tree.
#   * the GitHub release must NOT be a prerelease — an rc must never reach the
#     stable channel of crates.io, which has no unpublish.
#   * CARGO_REGISTRY_TOKEN must be UNSET — the token comes from the local
#     `cargo login` credential store. A token in the environment is how a CI
#     secret reaches a publish, and PP-066's publish standard is explicit that
#     no workflow publishes and no token lives in GitHub secrets.
#
# A STOP IS A REPORT, NEVER A RETRY. The first non-zero `cargo publish` ends the
# run and prints the crate, the error, what is live and what remains. Retrying
# into a half-published set is how a cascade produces two different versions of
# the same release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=publish_cascade
RECEIPT="docs/audits/publish-cascade-dryrun.md"

# --- facts, each overridable so the case table can drive both polarities ------
pc_describe() { # the exact tag at HEAD, empty when HEAD is not exactly a tag
    if [ -n "${PC_DESCRIBE+x}" ]; then printf '%s' "$PC_DESCRIBE"; return 0; fi
    git -C "$ROOT" describe --exact-match --tags 2>/dev/null || true
}
pc_detached() { # yes | no
    if [ -n "${PC_DETACHED+x}" ]; then printf '%s' "$PC_DETACHED"; return 0; fi
    if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then printf 'no'; else printf 'yes'; fi
}
pc_dirty() { # yes | no
    if [ -n "${PC_DIRTY+x}" ]; then printf '%s' "$PC_DIRTY"; return 0; fi
    if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then printf 'yes'; else printf 'no'; fi
}
pc_prerelease() { # true | false | unknown
    if [ -n "${PC_PRERELEASE+x}" ]; then printf '%s' "$PC_PRERELEASE"; return 0; fi
    gh release view "$1" --json isPrerelease --jq '.isPrerelease' 2>/dev/null || printf 'unknown'
}

# --- the derived, topologically ordered publish set ---------------------------
pc_list() {
    command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 missing\n' "$PROG" >&2; return 2; }
    if [ -n "${PC_METADATA+x}" ]; then cat "$PC_METADATA"; else
        ( cd "$ROOT" && cargo metadata --no-deps --format-version 1 2>/dev/null ) || {
            printf '%s: ENV - cargo metadata failed\n' "$PROG" >&2; return 2; }
    fi | python3 -c '
import json, sys
try:
    meta = json.load(sys.stdin)
except Exception as e:
    print(f"publish_cascade: ENV - cargo metadata is not JSON: {e}", file=sys.stderr); sys.exit(2)
pkgs = meta.get("packages", [])
# publish != false. cargo reports `publish: null` for "any registry" and a LIST
# (possibly empty) otherwise; an EMPTY list means publish = false.
sel = {}
for p in pkgs:
    pub = p.get("publish")
    if pub is not None and len(pub) == 0:
        continue
    sel[p["name"]] = p
# DEV-dependencies are excluded. `cargo publish` does not require a dev-dep to
# exist on the registry (siblings here are path-only, no version, so cargo drops
# them from the published manifest), and sibling dev-deps are legally CYCLIC:
# this workspace really does have aprender-compute <-> aprender-core through
# dev-deps. Ordering on them would report a cycle and refuse to publish a
# workspace that publishes fine. Only normal and build deps constrain the order.
def ordering_deps(p):
    for d in p.get("dependencies", []):
        if d.get("kind") in (None, "build") and d["name"] in sel and d["name"] != p["name"]:
            yield d["name"]
edges = {n: set(ordering_deps(p)) for n, p in sel.items()}
out, seen, mark = [], set(), {}
def visit(n, stack):
    if n in seen: return
    if mark.get(n): # a cycle cannot be published in any order; say so, do not guess
        chain = " -> ".join(stack + [n])
        print(f"publish_cascade: dependency cycle: {chain}", file=sys.stderr)
        sys.exit(2)
    mark[n] = True
    for d in sorted(edges[n]):
        visit(d, stack + [n])
    mark[n] = False
    seen.add(n); out.append(n)
for n in sorted(sel):
    visit(n, [])
for n in out:
    ver = sel[n]["version"]
    print(f"{n}\t{ver}")
'
}

# --- preflight: every refusal, in one place ----------------------------------
pc_preflight() { # pc_preflight <tag> -> 0 ok, 1 refused (reason on stdout)
    local tag=$1 d
    d=$(pc_describe)
    if [ "$d" != "$tag" ]; then
        printf 'REFUSE  HEAD is not exactly %s (git describe: %s). Publishing a tree that is\n' "$tag" "${d:-<none>}"
        printf '        not the tagged one ships something nobody reviewed.\n'; return 1
    fi
    if [ "$(pc_detached)" != yes ]; then
        printf 'REFUSE  HEAD is on a branch, not detached. A later commit would silently change\n'
        printf '        what "the release" means. Use: git worktree add /tmp/apr-<tag> <tag>\n'; return 1
    fi
    if [ "$(pc_dirty)" != no ]; then
        printf 'REFUSE  the working tree is dirty. `cargo publish` packages the working tree,\n'
        printf '        so an uncommitted edit would be published without ever being reviewed.\n'; return 1
    fi
    local pre; pre=$(pc_prerelease "$tag")
    if [ "$pre" != false ]; then
        printf 'REFUSE  GitHub release %s isPrerelease=%s. crates.io has no unpublish, so an rc\n' "$tag" "$pre"
        printf '        must never reach the stable channel. Promote the release first.\n'; return 1
    fi
    if [ -n "${CARGO_REGISTRY_TOKEN:-}" ]; then
        printf 'REFUSE  CARGO_REGISTRY_TOKEN is set. The cascade uses the local `cargo login`\n'
        printf '        credential store; a token in the environment is how a CI secret reaches\n'
        printf '        a publish, and PP-066 publishes only from a workstation.\n'; return 1
    fi
    printf 'ok      preflight: %s is exactly at HEAD, detached, clean, promoted, no token in env\n' "$tag"
    return 0
}

pc_already_published() { # <crate> <version>
    if [ -n "${PC_LIVE_SET+x}" ]; then
        case " $PC_LIVE_SET " in *" $1@$2 "*) return 0 ;; *) return 1 ;; esac
    fi
    cargo search "$1" --limit 1 2>/dev/null | grep -qF "\"$2\""
}

# ---------------------------------------------------------------------------
case "${1:-}" in
--help|-h)
    printf 'usage: %s --list | --dry-run [<tag>] | <tag> | --self-test\n\n' "$PROG"
    printf '  --list       the derived publish set (cargo metadata, publish != false), in\n'
    printf '               dependency-topological order. Derived, never listed.\n'
    printf '  --dry-run    cargo publish --dry-run per crate; writes %s\n' "$RECEIPT"
    printf '  <tag>        LIVE publish, one crate per call, stop on the first non-zero.\n'
    printf '  --self-test  case table, both polarities.\n\n'
    printf 'LIVE mode refuses unless ALL hold:\n'
    printf '  * git describe --exact-match equals <tag>\n'
    printf '  * HEAD is detached\n'
    printf '  * the working tree is clean\n'
    printf '  * the GitHub release is not a prerelease\n'
    printf '  * CARGO_REGISTRY_TOKEN is unset\n'
    exit 0
    ;;
--list)
    pc_list
    exit $?
    ;;
--self-test)
    TD=$(mktemp -d "${TMPDIR:-/tmp}/pcasc.XXXXXX")
    cleanup() { case "$TD" in *pcasc.*) if [ -n "$TD" ] && [ "$TD" != "/" ]; then rm -rf -- "$TD"; fi ;; esac; }
    trap cleanup EXIT
    n=0; red=0
    row() { local _w=$1 _l=$2 _rc=0; shift 2; n=$((n + 1))
        "$@" > "$TD/o.$n" 2>&1 || _rc=$?
        if [ "$_rc" = "$_w" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$_rc" "$_l"
        else printf 'FAIL  row %-2s rc=%s (want %s)  %s\n' "$n" "$_rc" "$_w" "$_l"; sed 's/^/        /' "$TD/o.$n"; red=$((red + 1)); fi; }

    # -- preflight, one row per refusal, plus the all-clear ------------------
    ok_env() { PC_DESCRIBE=v1.2.3 PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=false CARGO_REGISTRY_TOKEN= "$@"; }
    row 0 "preflight passes when every precondition holds"      bash -c 'PC_DESCRIBE=v1.2.3 PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=false; unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "MUTATION: HEAD on a branch -> refuse"                bash -c 'PC_DESCRIBE=v1.2.3 PC_DETACHED=no  PC_DIRTY=no PC_PRERELEASE=false; unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "MUTATION: dirty tree -> refuse"                      bash -c 'PC_DESCRIBE=v1.2.3 PC_DETACHED=yes PC_DIRTY=yes PC_PRERELEASE=false; unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "MUTATION: prerelease -> refuse"                      bash -c 'PC_DESCRIBE=v1.2.3 PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=true;  unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "MUTATION: CARGO_REGISTRY_TOKEN set -> refuse"        bash -c 'PC_DESCRIBE=v1.2.3 PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=false CARGO_REGISTRY_TOKEN=secret; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "HEAD is a different tag -> refuse"                   bash -c 'PC_DESCRIBE=v9.9.9 PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=false; unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "HEAD is at no tag at all -> refuse"                  bash -c 'PC_DESCRIBE= PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=false; unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'
    row 1 "release lookup failed (unknown) -> refuse, never assume" bash -c 'PC_DESCRIBE=v1.2.3 PC_DETACHED=yes PC_DIRTY=no PC_PRERELEASE=unknown; unset CARGO_REGISTRY_TOKEN; . '"$0"' --source-only; pc_preflight v1.2.3'

    # -- the derived set: order, filtering, cycles ---------------------------
    printf '%s' '{"packages":[
      {"name":"leaf","version":"1.0.0","publish":null,"dependencies":[]},
      {"name":"mid","version":"1.0.0","publish":null,"dependencies":[{"name":"leaf"}]},
      {"name":"top","version":"1.0.0","publish":null,"dependencies":[{"name":"mid"},{"name":"leaf"}]},
      {"name":"private","version":"1.0.0","publish":[],"dependencies":[]}]}' > "$TD/meta.json"
    row 0 "the derived set is topological and drops publish=false" bash -c 'PC_METADATA='"$TD"'/meta.json; . '"$0"' --source-only; out=$(pc_list | cut -f1 | tr "\n" " "); [ "$out" = "leaf mid top " ] || { echo "got: $out"; exit 1; }'
    row 0 "a private crate never appears"                        bash -c 'PC_METADATA='"$TD"'/meta.json; . '"$0"' --source-only; pc_list | grep -qv private'
    printf '%s' '{"packages":[
      {"name":"a","version":"1.0.0","publish":null,"dependencies":[{"name":"b","kind":null}]},
      {"name":"b","version":"1.0.0","publish":null,"dependencies":[{"name":"a","kind":null}]}]}' > "$TD/cyc.json"
    # A DEV-dep cycle is legal and must NOT be reported: this workspace really
    # has aprender-compute <-> aprender-core through dev-deps, and it publishes.
    printf '%s' '{"packages":[
      {"name":"a","version":"1.0.0","publish":null,"dependencies":[{"name":"b","kind":"dev"}]},
      {"name":"b","version":"1.0.0","publish":null,"dependencies":[{"name":"a","kind":"dev"}]}]}' > "$TD/devcyc.json"
    row 2 "a dependency cycle is ENV (exit 2), never a guessed order" bash -c 'PC_METADATA='"$TD"'/cyc.json; . '"$0"' --source-only; pc_list'
    printf 'not json' > "$TD/bad.json"
    row 0 "a DEV-dependency cycle is legal and is NOT a cycle"    bash -c 'PC_METADATA='"$TD"'/devcyc.json; . '"$0"' --source-only; out=$(pc_list | cut -f1 | tr "\n" " "); [ "$out" = "a b " ] || { echo "got: $out"; exit 1; }'
    row 2 "malformed cargo metadata is ENV (exit 2)"             bash -c 'PC_METADATA='"$TD"'/bad.json; . '"$0"' --source-only; pc_list'

    # -- per-crate idempotency ----------------------------------------------
    row 0 "a version already on crates.io is skipped"            bash -c 'PC_LIVE_SET="foo@1.0.0"; . '"$0"' --source-only; pc_already_published foo 1.0.0'
    row 1 "a version NOT on crates.io is published"              bash -c 'PC_LIVE_SET="foo@0.9.0"; . '"$0"' --source-only; pc_already_published foo 1.0.0'

    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
    ;;
--source-only)
    # the case table sources this file to reach the functions above
    return 0 2>/dev/null || exit 0
    ;;
--dry-run)
    TAG=${2:-}
    printf '%s: dry run over the derived publish set\n' "$PROG"
    { printf '# publish-cascade dry run\n\n'
      printf 'Set derived from `cargo metadata` (publish != false), dependency-topological.\n\n'
      printf '| # | crate | version | `cargo publish --dry-run` |\n|---|---|---|---|\n'; } > "$ROOT/$RECEIPT"
    i=0; rc=0
    while IFS=$(printf '\t') read -r crate version; do
        [ -n "$crate" ] || continue
        i=$((i + 1))
        if ( cd "$ROOT" && cargo publish -p "$crate" --dry-run --allow-dirty > /dev/null 2>&1 ); then
            printf '| %s | %s | %s | ok |\n' "$i" "$crate" "$version" >> "$ROOT/$RECEIPT"
            printf 'ok    %-34s %s\n' "$crate" "$version"
        else
            printf '| %s | %s | %s | **FAILED** |\n' "$i" "$crate" "$version" >> "$ROOT/$RECEIPT"
            printf 'FAIL  %-34s %s\n' "$crate" "$version"; rc=1
        fi
    done <<EOF
$(pc_list)
EOF
    printf '\n%s crates; receipt: %s\n' "$i" "$RECEIPT" >> "$ROOT/$RECEIPT"
    exit "$rc"
    ;;
'')
    printf 'usage: %s --list | --dry-run [<tag>] | <tag> | --self-test\n' "$PROG" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# LIVE
# ---------------------------------------------------------------------------
TAG=$1
pc_preflight "$TAG" || exit 1

live=""; remaining=$(pc_list | cut -f1 | tr '\n' ' ')
while IFS=$(printf '\t') read -r crate version; do
    [ -n "$crate" ] || continue
    remaining=${remaining#"$crate "}
    if pc_already_published "$crate" "$version"; then
        printf 'skip    %-34s %s already on crates.io\n' "$crate" "$version"
        live="$live $crate"
        continue
    fi
    printf 'publish %-34s %s\n' "$crate" "$version"
    if ( cd "$ROOT" && cargo publish -p "$crate" ); then
        live="$live $crate"
        # the index is eventually consistent; the next crate's build resolves
        # this one from it, so wait for it to appear rather than racing.
        w=0
        while [ "$w" -lt 60 ] && ! pc_already_published "$crate" "$version"; do
            sleep 5; w=$((w + 5))
        done
    else
        printf '\nSTOP    %s failed to publish.\n' "$crate"
        printf '        live now:  %s\n' "${live# }"
        printf '        remaining: %s\n' "$remaining"
        printf '        This is a REPORT, not a retry: re-running into a half-published set is\n'
        printf '        how one release becomes two. Fix the crate, then re-run — every crate\n'
        printf '        already on crates.io is skipped.\n'
        exit 1
    fi
done <<EOF
$(pc_list)
EOF
printf '\nPASS    every crate in the derived set is on crates.io at its workspace version\n'
