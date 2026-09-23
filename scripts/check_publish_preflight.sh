#!/usr/bin/env bash
# check_publish_preflight.sh — the ONLY way into `cargo publish` (F-9, PMAT-745).
#
# WHY. Until 0.65.0 the cascade ran `cargo publish --allow-dirty --locked` with
# no precondition of its own: a dirty tree, an untagged commit, a commit that
# was not on main, or a release whose dogfood verdict was NO-GO could all be
# uploaded to an immutable registry. The operator's release rule reads
# "publish only through a workflow gate; never --allow-dirty". This script is
# that gate: `scripts/cascade-publish.sh` calls it before the first upload and
# refuses to continue on any non-zero exit, and the `--allow-dirty` is gone.
#
# RULES (each prints its own line; the verdict is the AND of all of them)
#   R1  the tree is clean: no tracked change, no untracked file. cargo package
#       ships every non-ignored file in the tree, so an untracked file is a
#       file the registry would receive that git never saw.
#   R2  the version comes from `cargo metadata` (the root manifest), never
#       from an argument.
#   R3  the tag `v<version>` points at HEAD: the crate that is uploaded is the
#       commit that is tagged, not a neighbour of it.
#   R4  HEAD's changes are on the main ref: nothing publishes that main does not carry. Either HEAD
#       is an ancestor of main, or -- main's merge queue SQUASHES (a ruleset), so a release cut never
#       becomes an ancestor -- main CONTAINS the cut's changes: the published-path diff
#       merge-base(HEAD, main)..HEAD over crates/ src/ Cargo.toml reverse-applies cleanly to main's
#       tree, and its Cargo.lock change does too (hunk level) or is present SEMANTICALLY
#       (scripts/lib/lock_contained.py over ladder_equiv's canonical lock delta). main MOVING ON
#       after the cut is expected; a cut change missing on main, or reverted by it, is refused.
#       --via SHA (the PR head main squashed): when main has EDITED OVER a cut change afterwards (the
#       cut's own lines superseded, not lost), containment is proven through history instead: HEAD is
#       an ancestor of SHA, and SHA's published changes are contained in main by the same test.
#   R6  no versioned sibling dev-dependency lies on a CYCLE. cargo keeps a versioned
#       dev-dependency in the published manifest and resolves it on the registry,
#       so two siblings that name each other can never be uploaded first
#       (PMAT-955: 0.65.0 stuck at 48/74 on aprender-core <-> aprender-test-lib).
#       The rule judges the GRAPH, not the shape (#3468): an edge a -dev,versioned-> b
#       refuses only when b reaches a over normal + build + versioned-dev edges.
#       v0.68.1 carried four such edges out of aprender-compute (#3307), none on a
#       cycle, all four publishable first - and the shape rule stopped the train.
#   R5  the newest dogfood receipt (`.dogfood/receipt-*.json`, written by
#       scripts/dogfood.sh) says `verdict: GO` for THIS commit and THIS
#       version. A stale receipt, a NO-GO, or no receipt at all refuses.
#   R7  the model matrix (#3717, #3712): the per-host receipts COMMITTED in this tree for this
#       version (evidence/dogfood/models/<version>/, put there by the bump, #3708) are judged by
#       scripts/check_model_ladder.sh -- the same judge autopilot's T-1 `models` step runs on its
#       fresh measurement. Red, a missing receipt, a missing judge or a judge DECLINE (exit 2)
#       refuses: a decline is not a pass.
#
# EXIT  0 every rule holds · 1 a rule refused · 2 the box cannot answer
#       (no git/cargo/python3, not a repository). 2 is not a pass.
#
# SEAMS (the selftest builds a throwaway repository and drives every rule to
# both verdicts through them; production never sets them):
#   PUBLISH_PREFLIGHT_ROOT         repository root (default: this script's repo)
#   PUBLISH_PREFLIGHT_MAIN_REF     the main ref for R4 (default: origin/main)
#   PUBLISH_PREFLIGHT_RECEIPT_DIR  the dogfood receipt dir (default: $ROOT/.dogfood)
#   PUBLISH_PREFLIGHT_LADDER_JUDGE the R7 judge (default: $ROOT/scripts/check_model_ladder.sh)
#
# USAGE
#   bash scripts/check_publish_preflight.sh             # the gate
#   bash scripts/check_publish_preflight.sh --selftest  # case table, both polarities
#   bash scripts/check_publish_preflight.sh --receipt-only  # R2+R5 only: T-1, before the tag (#3708)
#   bash scripts/check_publish_preflight.sh --scope crux-smoke [--cut-commit SHA]
#       R7 under a RECORDED operator emergency scope (contracts/model-capability-ladder-v1.yaml
#       `ladder.emergency_scopes`; 0.69.1 only): the judge's own `--scope` path
#       (scripts/lib/crux_smoke_scope.py) decides R7 from CRUX smoke receipts bound to the CUT --
#       the commit the release binary was built from -- instead of the model matrix. The cut
#       defaults to HEAD; when HEAD is not the cut (receipts committed on top, or main's squash
#       of it), every PUBLISHED path -- crates/ src/ Cargo.toml Cargo.lock, R4's set -- must be
#       equal to the cut's, or the published source is not the smoked binary's. The
#       model-matrix rows are still printed, as EVIDENCE, never as the verdict.
#       --scope-tree DIR: the tree the scope is READ from (its contract entry, reader and committed
#       CRUX receipts), default the published tree. A release tagged exactly at the cut predates its
#       own scope ruling, so the ruling is read from main's checkout; that tree's HEAD must be ON
#       the main ref, so no branch can carry a ruling of its own.
#       --crux DIR: the CRUX smoke receipts, from a PATH (the judge's own --crux), so they need not be
#       committed before publish. The certification that decides WHICH cells are owed is still read
#       from the scope tree (evidence/crux/<version>/prompt-certification.json), never from DIR.
set -uo pipefail

PROG=${0##*/}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

die_env() { printf '%s: ENV %s\n' "$PROG" "$*" >&2; exit 2; }

# #3957 F1b, operator ruling (a) 2026-09-23: DEFER is ABOLISHED. A receipt with ANY deferred
# row is refused, whatever the phase. The two rows below are unmeasurable before a publish by
# construction, so they are OPEN post-publish obligations (dogfood.sh POST_PUBLISH_OBLIGATIONS)
# -- accepted in a pre-publish receipt only, and never read as passed. `coverage` is gone from
# the list: it was deferred WORK, and it is RED now.
PREPUBLISH_OPEN_OBLIGATIONS="publish-dry-run declared:check_multiplatform_dogfood"
# HISTORY (superseded by the line above). The only rows a pre-publish dogfood receipt could DEFER (PMAT-745): both need the
# crate to be ON the registry before they can be measured, so before a cascade
# they are recorded with their obligation instead of failing by construction.
# `coverage` added 2026-09-22 by operator ruling for 0.69.1 (#3839). It is the FIRST row
# here that is deferred because it FAILS rather than because it is unmeasurable before
# publication -- the other two need the published crate to exist. That is a real widening
# of this whitelist and it is temporary: #3839 owes the repaired measurement, a re-derived
# COV_FLOOR, and the REMOVAL of `coverage` from this line. The whitelist stays a whitelist,
# so this does not loosen anything else; a row not named here is still refused whatever it
# is called.

root_version() { # root -> the root manifest's package version, from cargo metadata
    local root="$1"
    cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null \
    | python3 -c '
import json, os, sys
try:
    m = json.load(sys.stdin)
except ValueError:
    sys.exit(1)   # cargo metadata printed nothing: no version, no stack trace
root = os.path.realpath(sys.argv[1])
for p in m.get("packages", []):
    if os.path.realpath(p["manifest_path"]) == root:
        print(p["version"]); sys.exit(0)
sys.exit(1)' "$root/Cargo.toml"
}

newest_receipt() { # dir -> path of the newest receipt-*.json, or nothing
    local dir="$1"
    [ -d "$dir" ] || return 0
    find "$dir" -maxdepth 1 -name 'receipt-*.json' -type f 2>/dev/null | LC_ALL=C sort | tail -n 1
}

# R5, ONE function for both ends of the train (#3708). The full gate calls it at
# T-4, before the first upload; `--receipt-only` calls it at T-1, right after the
# pre-publish dogfood and before the tag exists, so a receipt this rule will
# refuse at T-4 is refused while no public tag has been cut. Same file, same
# function, same verdict: the two ends cannot disagree.
# rule_r5 root head version -> prints its row; 0 accepted, 1 refused
rule_r5() {
    local root="$1" head="$2" version="$3" rdir receipt verdict rcommit rversion rphase rdeferred ropen bad_defer g
    rdir="${PUBLISH_PREFLIGHT_RECEIPT_DIR:-$root/.dogfood}"
    receipt="$(newest_receipt "$rdir")"
    if [ -z "$receipt" ]; then
        echo "FAIL  R5 no dogfood receipt under $rdir (run scripts/dogfood.sh on this commit)"
        return 1
    else
        read -r verdict rcommit rversion rphase rdeferred ropen < <(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("UNREADABLE - - - - -"); sys.exit(0)
deferred = d.get("deferred") or []
opened = d.get("open_obligations") or []
print(d.get("verdict") or "-", d.get("commit") or "-", d.get("version") or "-",
      d.get("phase") or "full", ",".join(str(x) for x in deferred) or "-",
      ",".join(str(x) for x in opened) or "-")' "$receipt")
        # #3957 F1b: any deferred row refuses. An OPEN row is accepted only in a pre-publish
        # receipt and only for the closed list; the list is a whitelist, so a row not named in
        # it is refused whatever it is called.
        bad_defer=""
        if [ "$rdeferred" != "-" ]; then
            printf 'FAIL  R5 dogfood receipt %s DEFERS [%s] -- DEFER is abolished (#3957 F1b): a row is measured or it is RED\n' \
                "$(basename "$receipt")" "$rdeferred"
            return 1
        fi
        if [ "$ropen" != "-" ] && [ "$rphase" != pre-publish ]; then
            printf 'FAIL  R5 dogfood receipt %s carries OPEN obligations outside the pre-publish phase: %s -- an unmet obligation (#3957 F1b)\n' \
                "$(basename "$receipt")" "$ropen"
            return 1
        elif [ "$ropen" != "-" ]; then
            # Split on commas into an array: an unquoted expansion would also
            # glob, and a gate should not depend on what files sit in its cwd.
            local -a open_rows=()
            IFS=, read -r -a open_rows <<< "$ropen"
            for g in "${open_rows[@]}"; do
                case " $PREPUBLISH_OPEN_OBLIGATIONS " in *" $g "*) ;; *) bad_defer="$bad_defer $g" ;; esac
            done
        fi
        if [ -n "$bad_defer" ]; then
            printf 'FAIL  R5 dogfood receipt %s carries an OPEN obligation this gate does not accept:%s (accepted in --phase pre-publish: %s)\n' \
                "$(basename "$receipt")" "$bad_defer" "$PREPUBLISH_OPEN_OBLIGATIONS"
            return 1
        elif [ "$verdict" = GO ] && [ "$rcommit" = "$head" ] && [ "$rversion" = "$version" ]; then
            echo "ok    R5 dogfood receipt $(basename "$receipt"): GO for ${head:0:9} at $version (phase $rphase$([ "$ropen" = - ] || printf ', OPEN post-publish obligations: %s' "$ropen"))"
        else
            printf 'FAIL  R5 dogfood receipt %s: verdict=%s commit=%s version=%s (need GO, %s, %s)\n' \
                "$(basename "$receipt")" "$verdict" "${rcommit:0:9}" "$rversion" "${head:0:9}" "${version:-?}"
            return 1
        fi
    fi
    return 0
}

# R7, the T-4 end of the model matrix (#3717): the committed receipts, the T-1 judge.
# rule_r7 root version -> prints its row; 0 accepted, 1 refused
rule_r7() {
    local root="$1" version="$2" judge out rc
    judge="${PUBLISH_PREFLIGHT_LADDER_JUDGE:-$root/scripts/check_model_ladder.sh}"
    if [ ! -f "$judge" ]; then
        echo "FAIL  R7 no model-matrix judge at $judge: the committed receipts cannot be judged"
        return 1
    fi
    if [ -n "${SCOPE:-}" ]; then
        rule_r7_scope "$root" "$version" "$judge" "$3"
        return $?
    fi
    out="$(cd "$root" && bash "$judge" --version "$version" 2>&1)"; rc=$?
    case "$rc" in
        0) echo "ok    R7 model matrix green for $version (committed receipts, $(basename "$judge"))"; return 0 ;;
        2) echo "FAIL  R7 the model-matrix judge DECLINED (rc 2), and a decline is not a pass: $(tail -n 1 <<< "$out")" ;;
        *) printf 'FAIL  R7 model matrix NOT green for %s (rc %s):\n%s\n' "$version" "$rc" \
               "$(grep -E '^FAIL' <<< "$out" | head -n 10 | sed 's/^/        /')" ;;
    esac
    return 1
}

# R7 under a recorded operator emergency scope (0.69.1: CRUX smoke only). The scope is READ by the
# judge (`--scope`, scripts/lib/crux_smoke_scope.py), never re-implemented here: it refuses another
# release, receipts from another binary, and a missing host. This rule adds the one binding the judge
# cannot see: the source being PUBLISHED (crates/ src/ Cargo.toml Cargo.lock, the paths R4 judges) is
# the source the smoked binary was built from. Scripts, contracts and evidence may differ: the scope's
# own contract entry and reader arrive after the cut.
# rule_r7_scope root version judge -> prints its rows; 0 accepted, 1 refused
rule_r7_scope() {
    local root="$1" version="$2" judge="$3" main_ref="$4" cut head out rc ev evrc stree sjudge shead
    head="$(git -C "$root" rev-parse HEAD 2>/dev/null)"
    stree="${SCOPE_TREE:-$root}"
    sjudge="${PUBLISH_PREFLIGHT_LADDER_JUDGE:-$stree/scripts/check_model_ladder.sh}"
    if [ "$stree" != "$root" ]; then
        shead="$(git -C "$stree" rev-parse HEAD 2>/dev/null)"
        if [ -z "$shead" ] || ! git -C "$root" merge-base --is-ancestor "$shead" "$main_ref" 2>/dev/null; then
            echo "FAIL  R7 OPERATOR EMERGENCY SCOPE $SCOPE: the scope tree $stree (HEAD ${shead:0:12}) is not on $main_ref -- a ruling is read from main, never from a branch"
            return 1
        fi
    fi
    cut="$(git -C "$root" rev-parse --verify --quiet "${CUT_COMMIT:-HEAD}^{commit}" 2>/dev/null)"
    if [ -z "$cut" ]; then
        echo "FAIL  R7 OPERATOR EMERGENCY SCOPE $SCOPE: the cut ${CUT_COMMIT:-HEAD} does not resolve in this tree"
        return 1
    fi
    if [ "$cut" != "$head" ] && ! git -C "$root" diff --quiet "$cut" "$head" -- crates src Cargo.toml Cargo.lock 2>/dev/null; then
        printf 'FAIL  R7 OPERATOR EMERGENCY SCOPE %s: HEAD %s differs from the cut %s in PUBLISHED paths -- the published source is not the smoked binary'"'"'s:\n%s\n' \
            "$SCOPE" "${head:0:12}" "${cut:0:12}" \
            "$(git -C "$root" diff --name-only "$cut" "$head" -- crates src Cargo.toml Cargo.lock | head -n 10 | sed 's/^/        /')"
        return 1
    fi
    local cruxargs=()
    if [ -n "${CRUX:-}" ]; then
        local cdir; cdir="$(cd "$CRUX" 2>/dev/null && pwd)"
        if [ -z "$cdir" ]; then
            echo "FAIL  R7 OPERATOR EMERGENCY SCOPE $SCOPE: --crux $CRUX is not a directory"
            return 1
        fi
        cruxargs=(--crux "$cdir")
        echo "        receipts from $cdir; certification from $stree/evidence/crux/$version/prompt-certification.json"
    fi
    out="$(cd "$stree" && CRUX_CERT="${CRUX:+$stree/evidence/crux/$version/prompt-certification.json}" \
        bash "$sjudge" --version "$version" --scope "$SCOPE" --cut-commit "$cut" "${cruxargs[@]}" 2>&1)"; rc=$?
    grep -E '^OPERATOR EMERGENCY SCOPE' <<< "$out" | head -n 1 | sed 's/^/        /'
    # The model matrix, reported as EVIDENCE only: under the scope it is not the verdict, and a
    # stale or red row must still be visible.
    ev="$(cd "$root" && bash "$judge" --version "$version" 2>&1)"; evrc=$?
    if [ "$evrc" != 0 ]; then
        printf '        evidence only (NOT the verdict under the emergency scope): model matrix rc %s\n%s\n' "$evrc" \
            "$(grep -E '^FAIL' <<< "$ev" | head -n 10 | sed 's/^/          evidence /')"
    fi
    case "$rc" in
        0) echo "ok    R7 OPERATOR EMERGENCY SCOPE $SCOPE: CRUX smoke satisfied at the cut ${cut:0:12} ($(basename "$judge") --scope); the model matrix was NOT the gate for $version"; return 0 ;;
        2) echo "FAIL  R7 OPERATOR EMERGENCY SCOPE $SCOPE: the judge DECLINED (rc 2), and a decline is not a pass: $(tail -n 1 <<< "$out")" ;;
        *) printf 'FAIL  R7 OPERATOR EMERGENCY SCOPE %s NOT satisfied for %s (rc %s):\n%s\n' "$SCOPE" "$version" "$rc" \
               "$(grep -E '^FAIL' <<< "$out" | head -n 10 | sed 's/^/        /')" ;;
    esac
    return 1
}

# R4's content arm: prints one reason per cut change NOT contained in main; nothing when all are.
# r4_contained root head main_ref
r4_contained() {
    local root="$1" head="$2" main_ref="$3" mb idx patch lp rc
    mb="$(git -C "$root" merge-base "$head" "$main_ref" 2>/dev/null)" || { echo "no merge-base between HEAD and $main_ref"; return 0; }
    idx="$(mktemp)"; patch="$(mktemp)"; lp="$(mktemp)"
    # main's tree in a throwaway index: nothing in any working tree is touched
    GIT_INDEX_FILE="$idx" git -C "$root" read-tree "$main_ref" || { echo "cannot read $main_ref's tree"; rm -f "$idx" "$patch" "$lp"; return 0; }
    git -C "$root" diff --binary "$mb" "$head" -- crates src Cargo.toml > "$patch"
    # -C0: the cut's ADDED lines must exist on main (removed ones must be gone); the surrounding context
    # may differ, because main moving on next to a cut change is expected, not a missing change.
    if [ -s "$patch" ] && ! GIT_INDEX_FILE="$idx" git -C "$root" apply --cached --reverse --check -C0 "$patch" 2>/dev/null; then
        # name the files whose change is not on main; if every file alone passes, the whole still did
        # not -- say so rather than pass (fail closed)
        local named
        named="$(git -C "$root" diff --name-only "$mb" "$head" -- crates src Cargo.toml | while IFS= read -r f; do
            git -C "$root" diff --binary "$mb" "$head" -- "$f" > "$lp"
            GIT_INDEX_FILE="$idx" git -C "$root" apply --cached --reverse --check -C0 "$lp" 2>/dev/null \
                || echo "$f: the cut's change is not on main (missing, reverted, or edited over)"
        done)"
        if [ -n "$named" ]; then printf '%s\n' "$named"
        else echo "the cut's published changes do not reverse-apply to main as a whole (no single file names it)"; fi
    fi
    if ! git -C "$root" diff --quiet "$mb" "$head" -- Cargo.lock; then
        git -C "$root" diff "$mb" "$head" -- Cargo.lock > "$lp"
        if ! GIT_INDEX_FILE="$idx" git -C "$root" apply --cached --reverse --check "$lp" 2>/dev/null; then
            local a b c
            a="$(mktemp)"; b="$(mktemp)"; c="$(mktemp)"
            git -C "$root" show "$mb:Cargo.lock" > "$a" 2>/dev/null
            git -C "$root" show "$head:Cargo.lock" > "$b" 2>/dev/null
            git -C "$root" show "$main_ref:Cargo.lock" > "$c" 2>/dev/null
            PYTHONPATH="$SCRIPT_DIR/lib" python3 "$SCRIPT_DIR/lib/lock_contained.py" "$a" "$b" "$c" | sed 's/^/Cargo.lock: /'
            rm -f "$a" "$b" "$c"
        fi
    fi
    rm -f "$idx" "$patch" "$lp"
    return 0
}

gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" main_ref="${PUBLISH_PREFLIGHT_MAIN_REF:-origin/main}"
    local fails=0 status version tags head
    for t in git cargo python3; do
        command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
    done
    if [ -z "$root" ]; then
        root="$(cd -- "$SCRIPT_DIR/.." && pwd)"
    fi
    git -C "$root" rev-parse --verify --quiet HEAD >/dev/null || die_env "$root is not a git repository with a HEAD"
    head="$(git -C "$root" rev-parse HEAD)"

    # R1 clean tree
    status="$(git -C "$root" status --porcelain --untracked-files=all 2>/dev/null)"
    if [ -n "$status" ]; then
        printf 'FAIL  R1 the tree is not clean; cargo package would ship what git never saw:\n%s\n' \
            "$(printf '%s\n' "$status" | sed 's/^/        /' | head -n 20)"
        fails=1
    else
        echo "ok    R1 clean tree (no tracked change, no untracked file)"
    fi

    # R2 version from cargo metadata
    version="$(root_version "$root")" || version=""
    if [ -z "$version" ]; then
        echo "FAIL  R2 cargo metadata names no version for the root manifest"
        fails=1
    else
        echo "ok    R2 version $version (cargo metadata, root manifest)"
    fi

    # R3 the tag points at HEAD
    tags="$(git -C "$root" tag --points-at HEAD 2>/dev/null)"
    # -F: the version is a string, not a pattern. With -x alone `v1-2-3` on HEAD
    # satisfied `v1.2.3` (second review of #2859, tag-regex-injection).
    if [ -n "$version" ] && printf '%s\n' "$tags" | grep -Fqx -- "v$version"; then
        echo "ok    R3 tag v$version points at HEAD ${head:0:9}"
    else
        printf 'FAIL  R3 tag v%s does not point at HEAD %s (tags here: %s)\n' \
            "${version:-?}" "${head:0:9}" "${tags:-none}"
        fails=1
    fi

    # R4 HEAD is on main: by ancestry, or by CONTENT when main squash-merged it
    local r4diff
    if ! git -C "$root" rev-parse --verify --quiet "${main_ref}^{commit}" >/dev/null; then
        echo "FAIL  R4 the main ref $main_ref does not exist here"
        fails=1
    elif git -C "$root" merge-base --is-ancestor "$head" "$main_ref" 2>/dev/null; then
        echo "ok    R4 HEAD is an ancestor of $main_ref"
    else
        r4diff="$(r4_contained "$root" "$head" "$main_ref")"
        local via viadiff
        if [ -n "$r4diff" ] && [ -n "${VIA:-}" ]; then
            via="$(git -C "$root" rev-parse --verify --quiet "${VIA}^{commit}")"
            if [ -z "$via" ]; then
                r4diff="$(printf '%s\n--via %s does not resolve here' "$r4diff" "$VIA")"
            elif ! git -C "$root" merge-base --is-ancestor "$head" "$via" 2>/dev/null; then
                r4diff="$(printf '%s\n--via %s does not contain HEAD %s (HEAD is not its ancestor)' "$r4diff" "${via:0:9}" "${head:0:9}")"
            else
                viadiff="$(r4_contained "$root" "$via" "$main_ref")"
                if [ -z "$viadiff" ]; then
                    echo "ok    R4 cut changes contained in main $(git -C "$root" rev-parse --short=9 "$main_ref") via squashed PR head ${via:0:9}: HEAD ${head:0:9} is its ancestor, and every published change of it is on main (main edited over some cut lines afterwards: $(printf '%s\n' "$r4diff" | grep -c .) superseded)"
                    r4diff=""; via="done"
                else
                    r4diff="$(printf '%s\n--via %s is itself not contained in main:\n%s' "$r4diff" "${via:0:9}" "$viadiff")"
                fi
            fi
        fi
        if [ "${via:-}" = done ]; then
            :
        elif [ -z "$r4diff" ]; then
            echo "ok    R4 cut changes contained in main $(git -C "$root" rev-parse --short=9 "$main_ref"): HEAD ${head:0:9} is not an ancestor (squash queue), but every published change since merge-base $(git -C "$root" merge-base "$head" "$main_ref" | cut -c1-9) is on main"
        else
            printf 'FAIL  R4 HEAD %s is not an ancestor of %s, and main does not contain the cut'"'"'s changes:\n%s\n' \
                "${head:0:9}" "$main_ref" "$(printf '%s\n' "$r4diff" | head -n 12 | sed 's/^/        /')"
            fails=1
        fi
    fi

    # R5 dogfood receipt: GO, this commit, this version
    rule_r5 "$root" "$head" "$version" || fails=1

    # R6 no versioned sibling dev-dependency lies on a cycle (PMAT-955, #3468). A
    # dev-dependency with a version is kept in the published manifest and resolved
    # on the registry at publish time; a path-only one is stripped. The edge is a
    # defect only when its target can reach its source: then neither crate can be
    # uploaded first. Acyclic edges are printed, so the publish order that must
    # honour them is visible in the receipt.
    r6="$(cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("UNREADABLE"); sys.exit(0)
pk = {p["name"]: p for p in d.get("packages", []) if p.get("publish") != []}
edges = {n: set() for n in pk}
vdev = []
for n, p in pk.items():
    for dep in p.get("dependencies", []):
        t = dep.get("name")
        if t not in pk or t == n:
            continue
        if dep.get("kind") == "dev":
            if (dep.get("req") or "*") == "*":
                continue
            vdev.append((n, t, dep.get("req")))
        edges[n].add(t)
def reaches(src, dst):
    seen, stack = set(), [src]
    while stack:
        x = stack.pop()
        for y in edges.get(x, ()):
            if y == dst:
                return True
            if y not in seen:
                seen.add(y); stack.append(y)
    return False
for n, t, req in sorted(vdev):
    print("%s %s -> %s %s" % ("CYCLE" if reaches(t, n) else "ACYCLIC", n, t, req))')"
    if [ "$r6" = UNREADABLE ]; then
        echo "FAIL  R6 cargo metadata is unreadable, so sibling dev-dependencies cannot be judged"
        fails=1
    elif grep -q '^CYCLE ' <<< "$r6"; then
        printf 'FAIL  R6 a versioned sibling dev-dependency lies on a cycle (kept in the published manifest; neither crate can be uploaded first):\n%s\n' \
            "$(printf '%s\n' "$r6" | sed -n 's/^CYCLE //p')"
        fails=1
    elif [ -n "$r6" ]; then
        printf 'ok    R6 %s versioned sibling dev-dependency edge(s), none on a cycle (the target publishes first):\n%s\n' \
            "$(printf '%s\n' "$r6" | grep -c '^ACYCLIC ')" "$(printf '%s\n' "$r6" | sed -n 's/^ACYCLIC /        /p')"
    else
        echo "ok    R6 no sibling dev-dependency carries a version (path-only, stripped at publish)"
    fi

    # R7 the model matrix, re-read at T-4 through the T-1 judge (#3717)
    rule_r7 "$root" "$version" "$main_ref" || fails=1

    if [ "$fails" -ne 0 ]; then
        echo "REFUSE $PROG: publishing is not allowed from this tree (see the FAIL rows)."
        return 1
    fi
    if [ -n "${SCOPE:-}" ]; then
        echo "PASS  $PROG: clean, versioned, tagged, on $main_ref, dogfood GO, OPERATOR EMERGENCY SCOPE $SCOPE satisfied (the model matrix was NOT the gate)"
    else
        echo "PASS  $PROG: clean, versioned, tagged, on $main_ref, dogfood GO, model matrix green"
    fi
    return 0
}

# --receipt-only (#3708): R2 + R5 and nothing else. R1/R3/R4/R6 describe the tree
# being UPLOADED and the tag it is uploaded from; at T-1 there is no tag yet, so
# they are judged at T-4 by the full gate, as before.
receipt_gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" head version
    for t in git cargo python3; do
        command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
    done
    if [ -z "$root" ]; then
        root="$(cd -- "$SCRIPT_DIR/.." && pwd)"
    fi
    git -C "$root" rev-parse --verify --quiet HEAD >/dev/null || die_env "$root is not a git repository with a HEAD"
    head="$(git -C "$root" rev-parse HEAD)"
    version="$(root_version "$root")" || version=""
    if [ -z "$version" ]; then
        echo "FAIL  R2 cargo metadata names no version for the root manifest"
        echo "REFUSE $PROG --receipt-only: no version to judge the dogfood receipt against."
        return 1
    fi
    echo "ok    R2 version $version (cargo metadata, root manifest)"
    if ! rule_r5 "$root" "$head" "$version"; then
        echo "REFUSE $PROG --receipt-only: the T-4 publish gate would refuse this receipt (R5)."
        return 1
    fi
    echo "PASS  $PROG --receipt-only: R5 holds for ${head:0:9} at $version (R1/R3/R4/R6 are judged at publish)"
    return 0
}

# --------------------------------------------------------------- selftest ---
selftest() {
    local tmp pass=0 fail=0
    tmp="$(mktemp -d)"
    case "$tmp" in /tmp/*|/var/folders/*|/mnt/*) : ;; *) die_env "mktemp gave ${tmp:-<empty>}, refusing to rm -rf it" ;; esac
    # SEC011: the delete is guarded by the same case the creation was, and an
    # empty or root path is left alone rather than deleted carefully.
    _rm_scratch() {
        local victim="${tmp:-}"
        [ -n "$victim" ] || return 0
        [ "$victim" != "/" ] || return 0
        case "$victim" in
            /tmp/?*|/var/folders/?*|/mnt/?*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
            *) return 0 ;;
        esac
    }
    trap _rm_scratch RETURN

    # A throwaway package repository with a lockfile committed, so `cargo
    # metadata --offline` writes nothing into the tree it is judging.
    build_repo() { # dir
        local d="$1"
        mkdir -p "$d/src"
        printf '[package]\nname = "preflight-fixture"\nversion = "1.2.3"\nedition = "2021"\n\n[dependencies]\n' > "$d/Cargo.toml"
        printf 'pub fn f() {}\n' > "$d/src/lib.rs"
        printf '.dogfood/\n' > "$d/.gitignore"
        write_judge "$d"
        git -C "$d" init -q -b fixture-main
        git -C "$d" -c user.name=t -c user.email=t@t config commit.gpgsign false
        ( cd "$d" && cargo metadata --no-deps --offline --format-version 1 >/dev/null 2>&1 )
        git -C "$d" add -A
        git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'fixture' >/dev/null
        git -C "$d" tag v1.2.3
        mkdir -p "$d/.dogfood"
        write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    }
    # R7's judge, committed in every fixture: it must be asked about THIS version (1.2.3), and it
    # answers FX_LADDER_RC (default 0). A judge asked about anything else is red.
    # Under `--scope` (0.69.1 emergency scope) it must be called `--version 1.2.3 --scope crux-smoke
    # --cut-commit <the cut, full sha>` and answers FX_SCOPE_RC; the cut it expects is FX_EXPECT_CUT
    # (default HEAD), so a preflight that passes HEAD when the cut is below it goes red.
    write_judge() { # dir
        mkdir -p "$1/scripts"
        cat > "$1/scripts/check_model_ladder.sh" <<'FXJUDGE'
#!/usr/bin/env bash
if [ "${3:-}" = "--scope" ]; then
    [ "${1:-} ${2:-} ${4:-} ${5:-}" = "--version 1.2.3 crux-smoke --cut-commit" ] || { echo "FAIL  judge scope call: $*"; exit 1; }
    [ "${6:-}" = "$(git rev-parse "${FX_EXPECT_CUT:-HEAD}")" ] || { echo "FAIL  judge asked about cut ${6:-}"; exit 1; }
    if [ -n "${FX_EXPECT_CRUX:-}" ]; then
        [ "${7:-} ${8:-}" = "--crux $FX_EXPECT_CRUX" ] || { echo "FAIL  judge got receipts [${7:-} ${8:-}], not --crux $FX_EXPECT_CRUX"; exit 1; }
        [ "${CRUX_CERT:-}" = "$PWD/evidence/crux/1.2.3/prompt-certification.json" ] || { echo "FAIL  certification read from [${CRUX_CERT:-}], not the scope tree"; exit 1; }
    else
        [ -z "${7:-}" ] || { echo "FAIL  judge got unexpected args: $*"; exit 1; }
    fi
    echo "OPERATOR EMERGENCY SCOPE: CRUX smoke only -- release 1.2.3 (fixture)"
    [ "${FX_SCOPE_RC:-0}" = 0 ] || echo "FAIL  host gx10 has no CRUX receipt"
    exit "${FX_SCOPE_RC:-0}"
fi
[ "${1:-} ${2:-}" = "--version 1.2.3" ] || { echo "FAIL  judge asked about: $*"; exit 1; }
[ "${FX_LADDER_RC:-0}" = 0 ] || echo "FAIL  fx-rung red on lambda"
exit "${FX_LADDER_RC:-0}"
FXJUDGE
    }
    write_receipt() { # dir, verdict, commit, version [, phase, deferred-json-array, open-obligations-json-array]
        printf '{"crate":"preflight-fixture","version":"%s","timestamp":"20260903T000000Z","commit":"%s","gates":[],"phase":"%s","deferred":%s,"open_obligations":%s,"verdict":"%s"}\n' \
            "$4" "$3" "${5:-full}" "${6:-[]}" "${7:-[]}" "$2" > "$1/.dogfood/receipt-20260903T000000Z.json"
    }
    # A throwaway WORKSPACE: root package plus members a and b, where a has a
    # dev-dependency on b declared either path-only or with a version (R6).
    build_ws_repo() { # dir, devdep-suffix ('' | ', version = "1.2.3"') [, b-depends-on-a: yes]
        local d="$1" suffix="$2" back="${3:-no}"
        mkdir -p "$d/src" "$d/a/src" "$d/b/src"
        printf '[workspace]\nmembers = ["a", "b"]\n\n[package]\nname = "preflight-fixture"\nversion = "1.2.3"\nedition = "2021"\n\n[dependencies]\n' > "$d/Cargo.toml"
        printf 'pub fn f() {}\n' > "$d/src/lib.rs"
        printf '[package]\nname = "fx-a"\nversion = "1.2.3"\nedition = "2021"\n\n[dependencies]\n\n[dev-dependencies]\nfx-b = { path = "../b"%s }\n' "$suffix" > "$d/a/Cargo.toml"
        printf 'pub fn a() {}\n' > "$d/a/src/lib.rs"
        printf '[package]\nname = "fx-b"\nversion = "1.2.3"\nedition = "2021"\n' > "$d/b/Cargo.toml"
        if [ "$back" = yes ]; then
            printf '\n[dependencies]\nfx-a = { path = "../a", version = "1.2.3" }\n' >> "$d/b/Cargo.toml"
        fi
        printf 'pub fn b() {}\n' > "$d/b/src/lib.rs"
        printf '.dogfood/\n' > "$d/.gitignore"
        write_judge "$d"
        git -C "$d" init -q -b fixture-main
        git -C "$d" -c user.name=t -c user.email=t@t config commit.gpgsign false
        ( cd "$d" && cargo metadata --no-deps --offline --format-version 1 >/dev/null 2>&1 )
        git -C "$d" add -A
        git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'fixture' >/dev/null
        git -C "$d" tag v1.2.3
        mkdir -p "$d/.dogfood"
        write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    }
    row() { # name, expect(0|1), needle, dir [, gate|receipt_gate]
        local name="$1" expect="$2" needle="$3" d="$4" mode="${5:-gate}" out rc=0
        out="$( PUBLISH_PREFLIGHT_ROOT="$d" PUBLISH_PREFLIGHT_MAIN_REF=fixture-main "$mode" 2>&1 )" || rc=$?
        if [ "$rc" != "$expect" ]; then
            printf '  BROKE %-36s expected exit %s got %s\n' "$name" "$expect" "$rc"; fail=$((fail + 1)); return 0
        fi
        case "$out" in
            *"$needle"*) printf '  ok    %-36s exit=%s said %s\n' "$name" "$expect" "$needle"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-36s exit %s but never said %s\n' "$name" "$expect" "$needle"; fail=$((fail + 1)) ;;
        esac
    }

    local d
    d="$tmp/ok"; build_repo "$d"
    row all_rules_hold                 0 "PASS" "$d"

    d="$tmp/dirty"; build_repo "$d"; printf 'pub fn g() {}\n' >> "$d/src/lib.rs"
    row tracked_change_refuses         1 "FAIL  R1" "$d"

    d="$tmp/untracked"; build_repo "$d"; printf 'x\n' > "$d/stray.out"
    row untracked_file_refuses         1 "FAIL  R1" "$d"

    d="$tmp/notag"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    row missing_tag_refuses            1 "FAIL  R3" "$d"

    d="$tmp/tagelse"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    printf 'pub fn h() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'second' >/dev/null
    git -C "$d" tag v1.2.3 HEAD~1; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row tag_on_another_commit_refuses  1 "FAIL  R3" "$d"

    d="$tmp/branch"; build_repo "$d"; git -C "$d" checkout -q -b topic
    printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'topic' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row head_off_main_refuses          1 "FAIL  R4" "$d"

    # R4 by CONTAINMENT: main's merge queue squashes, so the cut is never an ancestor of main; the cut's
    # changes since merge-base must be ON main. main moving on afterwards is expected.
    gcommit() { git -C "$1" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q "${@:2}" >/dev/null; }
    sq() { # dir: topic commit squash-merged onto fixture-main, HEAD left on the topic (the cut)
        local d="$1"; build_repo "$d"; git -C "$d" checkout -q -b topic
        printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; gcommit "$d" -am 'topic'
        git -C "$d" checkout -q fixture-main; git -C "$d" merge -q --squash topic >/dev/null; gcommit "$d" -m 'squash'
    }
    sq_finish() { git -C "$1" checkout -q topic; git -C "$1" tag -f v1.2.3 >/dev/null; write_receipt "$1" GO "$(git -C "$1" rev-parse HEAD)" 1.2.3; }
    d="$tmp/squash"; sq "$d"; sq_finish "$d"
    row squash_contained_passes        0 "ok    R4 cut changes contained in main" "$d"
    d="$tmp/squash-moved"; sq "$d"; printf 'pub fn z() {}\n' >> "$d/src/lib.rs"; printf 'n\n' > "$d/NOTES.md"; git -C "$d" add -A; gcommit "$d" -m 'main moves on'; sq_finish "$d"
    row squash_then_main_moves_on_passes 0 "ok    R4 cut changes contained in main" "$d"
    d="$tmp/squash-revert"; sq "$d"
    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t revert --no-edit HEAD >/dev/null \
        || { printf '  BROKE fixture: git revert failed\n'; fail=$((fail + 1)); }
    sq_finish "$d"
    row cut_change_reverted_on_main_refuses 1 "src/lib.rs: the cut's change is not on main" "$d"
    # a crate edit that exists ONLY on the cut: a second topic commit main never received
    d="$tmp/squash-partial"; sq "$d"; git -C "$d" checkout -q topic
    printf 'pub fn only_on_cut() {}\n' > "$d/src/extra.rs"; printf 'mod extra;\n' >> "$d/src/lib.rs"; git -C "$d" add -A; gcommit "$d" -m 'cut only'; sq_finish "$d"
    row crate_edit_only_on_the_cut_refuses 1 "src/extra.rs: the cut's change is not on main" "$d"
    # (a cut change missing on main entirely: head_off_main_refuses above)
    # --via: main squashed a PR head that CONTAINS the cut and then edited over one of the cut's own lines
    d="$tmp/via"; build_repo "$d"; git -C "$d" checkout -q -b rel
    printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; gcommit "$d" -am 'cut'; local vcut; vcut="$(git -C "$d" rev-parse HEAD)"
    git -C "$d" checkout -q -b pr; sed -i 's/pub fn k() {}/pub fn k() -> u8 { 1 }/' "$d/src/lib.rs"; gcommit "$d" -am 'pr edits the cut line'
    git -C "$d" checkout -q fixture-main; git -C "$d" merge -q --squash pr >/dev/null; gcommit "$d" -m 'squash pr'
    git -C "$d" checkout -q rel; git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$vcut" 1.2.3
    row superseded_cut_line_refuses_without_via 1 "src/lib.rs: the cut's change is not on main" "$d"
    VIA=pr row superseded_cut_line_passes_via_pr_head 0 "via squashed PR head" "$d"
    VIA=fixture-main~1 row via_not_containing_the_cut_refuses 1 "does not contain HEAD" "$d"
    git -C "$d" checkout -q -b pr2 pr; printf 'pub fn unmerged() {}\n' >> "$d/src/lib.rs"; gcommit "$d" -am 'pr2 not on main'; git -C "$d" checkout -q rel
    VIA=pr2 row via_not_on_main_refuses 1 "is itself not contained in main" "$d"
    # Cargo.lock, semantically (lock_contained.py over ladder_equiv's canonical delta), hermetic:
    lk() { printf '[[package]]\nname = "app"\nversion = "1.0.0"\ndependencies = [%s]\n%s' "$1" "$2"; }
    ext() { printf '\n[[package]]\nname = "%s"\nversion = "%s"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "%s"\n' "$1" "$2" "$3"; }
    local lt="$tmp/lock"; mkdir -p "$lt"
    lk '"zune-core"' "$(ext zune-core 0.4.0 aa)" > "$lt/mb"
    lk '"zune-core", "gif"' "$(ext zune-core 0.4.0 aa; ext gif 0.13.3 bb)" > "$lt/cut"
    lk '"zune-core 0.4.0", "gif", "tiff"' "$(ext zune-core 0.4.0 aa; ext gif 0.13.3 bb; ext tiff 0.9.0 cc)" > "$lt/main_ok"   # disambiguated text + moved on
    cp "$lt/mb" "$lt/main_reverted"
    lk '"zune-core", "gif"' "$(ext zune-core 0.4.0 aa; ext gif 0.12.0 dd)" > "$lt/main_otherver"
    lrow() { # name expect(0|1) main-file needle
        local out rc=0; out="$(PYTHONPATH="$SCRIPT_DIR/lib" python3 "$SCRIPT_DIR/lib/lock_contained.py" "$lt/mb" "$lt/cut" "$lt/$3" 2>&1)" || rc=$?
        if [ "$rc" = "$2" ] && { [ -z "$4" ] || printf '%s' "$out" | grep -qF "$4"; }; then
            printf '  ok    %-36s exit=%s %s\n' "$1" "$2" "$4"; pass=$((pass + 1))
        else printf '  BROKE %-36s expected %s got %s: %s\n' "$1" "$2" "$rc" "$out"; fail=$((fail + 1)); fi
    }
    lrow lock_dedupe_and_moved_on_contained 0 main_ok ""
    lrow lock_change_reverted_refuses 1 main_reverted "adds dependency app -> gif; main does not have it"
    lrow lock_other_version_refuses   1 main_otherver "the cut adds gif 0.13.3"

    d="$tmp/nogo"; build_repo "$d"; write_receipt "$d" NO-GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row dogfood_no_go_refuses          1 "FAIL  R5" "$d"

    d="$tmp/stale"; build_repo "$d"; write_receipt "$d" GO 0000000000000000000000000000000000000000 1.2.3
    row dogfood_stale_commit_refuses   1 "FAIL  R5" "$d"

    d="$tmp/otherver"; build_repo "$d"; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.4
    row dogfood_other_version_refuses  1 "FAIL  R5" "$d"

    d="$tmp/noreceipt"; build_repo "$d"
    # SEC010: validate before rm -- $d is $tmp (mktemp -d) plus a literal, but
    # guard it explicitly rather than relying on the assignment above.
    stale_receipt="$d/.dogfood/receipt-20260903T000000Z.json"
    case "$stale_receipt" in
        *..*) echo "ERROR: fixture path must not contain '..'" >&2; exit 1 ;;
    esac
    rm -f "$stale_receipt"
    rmdir "$d/.dogfood"
    row dogfood_receipt_absent_refuses 1 "FAIL  R5" "$d"

    d="$tmp/unreadable"; build_repo "$d"; printf '{not json' > "$d/.dogfood/receipt-20260903T000000Z.json"
    row dogfood_receipt_unreadable_refuses 1 "FAIL  R5" "$d"

    # R2: a virtual manifest has no root package, so cargo metadata names no version.
    d="$tmp/norootpkg"; build_repo "$d"
    printf '[workspace]\nmembers = []\n' > "$d/Cargo.toml"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'virtual' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row no_root_package_refuses        1 "FAIL  R2" "$d"

    # R3: the version is a string, not a pattern -- `v1-2-3` must not satisfy `v1.2.3`.
    d="$tmp/lookalike"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null; git -C "$d" tag v1-2-3
    row tag_lookalike_refuses          1 "FAIL  R3" "$d"

    # R5 (#3957 F1b, operator ruling (a)): DEFER is abolished -- ANY deferred row refuses,
    # including the two registry-bound rows and coverage, which this list used to admit. Those
    # two rows are now OPEN post-publish obligations, accepted in pre-publish only.
    d="$tmp/prepub-ok"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 pre-publish '[]' '["publish-dry-run","declared:check_multiplatform_dogfood"]'
    row prepublish_open_obligations_pass 0 "PASS" "$d"

    d="$tmp/prepub-defer-registry"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 pre-publish '["publish-dry-run","declared:check_multiplatform_dogfood"]'
    row prepublish_registry_deferral_refuses 1 "DEFER is abolished" "$d"

    d="$tmp/prepub-defer-coverage"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 pre-publish '["coverage"]'
    row prepublish_coverage_deferral_refuses 1 "DEFER is abolished" "$d"

    d="$tmp/prepub-open-coverage"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 pre-publish '[]' '["publish-dry-run","coverage"]'
    row prepublish_unlisted_open_refuses 1 "OPEN obligation this gate does not accept: coverage" "$d"

    d="$tmp/full-open"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 full '[]' '["publish-dry-run"]'
    row open_outside_prepublish_refuses 1 "OPEN obligations outside the pre-publish phase" "$d"

    d="$tmp/prepub-bad"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 pre-publish '["publish-dry-run","bashrs"]'
    row prepublish_unexpected_deferral_refuses 1 "FAIL  R5" "$d"

    d="$tmp/fullphase-defer"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 full '["publish-dry-run"]'
    row deferral_outside_prepublish_refuses 1 "FAIL  R5" "$d"

    # R6, both polarities (PMAT-955), and the graph not the shape (#3468):
    # a -dev,versioned-> b with b -> a is the 0.65.0 cycle and refuses; the same
    # edge with no way back is v0.68.1's aprender-compute and passes, NAMED.
    d="$tmp/devdep_cycle"; build_ws_repo "$d" ', version = "1.2.3"' yes
    row versioned_sibling_devdep_on_a_cycle_refuses 1 "FAIL  R6" "$d"
    d="$tmp/devdep_version"; build_ws_repo "$d" ', version = "1.2.3"'
    row versioned_sibling_devdep_acyclic_passes 0 "fx-a -> fx-b" "$d"
    d="$tmp/devdep_path"; build_ws_repo "$d" ''
    row pathed_sibling_devdep_passes   0 "PASS" "$d"

    # R7 (#3717): the committed model-matrix receipts, judged by the T-1 judge. all_rules_hold above
    # is the green row (the stub refuses any version but 1.2.3, so it also proves the argument).
    d="$tmp/r7-red"; build_repo "$d"
    FX_LADDER_RC=1 row r7_model_matrix_red_refuses       1 "FAIL  R7 model matrix NOT green" "$d"
    d="$tmp/r7-decline"; build_repo "$d"
    FX_LADDER_RC=2 row r7_judge_decline_refuses         1 "FAIL  R7 the model-matrix judge DECLINED" "$d"
    d="$tmp/r7-absent"; build_repo "$d"; git -C "$d" rm -q scripts/check_model_ladder.sh
    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'no judge' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row r7_judge_absent_refuses        1 "FAIL  R7 no model-matrix judge" "$d"

    # R7 under the recorded operator EMERGENCY SCOPE (--scope crux-smoke). The scope's own must-REDs
    # (another release, receipts from another binary, a host missing) live in the judge's reader,
    # scripts/lib/crux_smoke_scope.py, and its table; these rows prove the preflight ASKS it, with the
    # cut, takes its verdict, keeps the model matrix as evidence only, and binds published == smoked.
    d="$tmp/sc-green"; build_repo "$d"
    FX_LADDER_RC=1 SCOPE=crux-smoke row scope_green_over_a_red_matrix_passes 0 "OPERATOR EMERGENCY SCOPE crux-smoke satisfied" "$d"
    FX_LADDER_RC=1 SCOPE=crux-smoke row scope_keeps_the_matrix_as_evidence 0 "evidence FAIL  fx-rung red on lambda" "$d"
    FX_SCOPE_RC=1 SCOPE=crux-smoke row scope_red_refuses 1 "FAIL  R7 OPERATOR EMERGENCY SCOPE crux-smoke NOT satisfied" "$d"
    FX_SCOPE_RC=1 SCOPE=crux-smoke row scope_red_names_the_reason 1 "host gx10 has no CRUX receipt" "$d"
    FX_SCOPE_RC=2 SCOPE=crux-smoke row scope_decline_refuses 1 "the judge DECLINED" "$d"
    FX_LADDER_RC=1 row no_scope_matrix_still_gates 1 "FAIL  R7 model matrix NOT green" "$d"
    # the receipts committed on top of the cut: HEAD differs from the cut ONLY under evidence/
    d="$tmp/sc-evidence"; build_repo "$d"; cut="$(git -C "$d" rev-parse HEAD)"
    mkdir -p "$d/evidence/crux/1.2.3"; printf '{}\n' > "$d/evidence/crux/1.2.3/lambda-gpu.json"
    git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'receipts' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke CUT_COMMIT="$cut" row scope_cut_below_evidence_commit_passes 0 "satisfied at the cut ${cut:0:12}" "$d"
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke row scope_head_is_not_the_cut_refuses 1 "judge asked about cut" "$d"
    # a SOURCE change on top of the cut: what would be published is not what was smoked
    d="$tmp/sc-source"; build_repo "$d"; cut="$(git -C "$d" rev-parse HEAD)"
    printf 'pub fn g() {}\n' >> "$d/src/lib.rs"
    git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'src' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke CUT_COMMIT="$cut" row scope_source_change_over_the_cut_refuses 1 "differs from the cut ${cut:0:12} in PUBLISHED paths" "$d"
    # scripts/contracts arriving after the cut (the scope's own reader and entry do) are not published
    d="$tmp/sc-tooling"; build_repo "$d"; cut="$(git -C "$d" rev-parse HEAD)"
    printf '# tooling\n' > "$d/scripts/new_tool.sh"; mkdir -p "$d/contracts"; printf 'x: 1\n' > "$d/contracts/c.yaml"
    git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'tooling' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke CUT_COMMIT="$cut" row scope_tooling_after_the_cut_passes 0 "satisfied at the cut ${cut:0:12}" "$d"
    # the scope READ from main's checkout while the published tree is exactly the cut (older)
    d="$tmp/sc-tree"; build_repo "$d"; cut="$(git -C "$d" rev-parse HEAD)"
    git -C "$d" checkout -q -b rel; git -C "$d" checkout -q fixture-main
    printf 'ruling: 1\n' > "$d/RULING"; git -C "$d" add -A; gcommit "$d" -m 'ruling on main'
    git -C "$d" worktree add -q "$tmp/sc-tree-main" fixture-main 2>/dev/null || git -C "$d" worktree add -q --detach "$tmp/sc-tree-main" fixture-main
    git -C "$d" checkout -q rel
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke CUT_COMMIT="$cut" SCOPE_TREE="$tmp/sc-tree-main" row scope_tree_on_main_passes 0 "satisfied at the cut ${cut:0:12}" "$d"
    git -C "$d" checkout -q -b rogue; printf 'forged\n' > "$d/RULING"; git -C "$d" add -A; gcommit "$d" -m 'rogue ruling'
    git -C "$d" worktree add -q --detach "$tmp/sc-tree-rogue" rogue; git -C "$d" checkout -q rel
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke CUT_COMMIT="$cut" SCOPE_TREE="$tmp/sc-tree-rogue" row scope_tree_off_main_refuses 1 "is not on fixture-main" "$d"
    # --crux: receipts from a PATH, certification still from the (scope) tree
    d="$tmp/sc-crux"; build_repo "$d"; mkdir -p "$tmp/sc-crux-receipts"
    FX_EXPECT_CRUX="$tmp/sc-crux-receipts" SCOPE=crux-smoke CRUX="$tmp/sc-crux-receipts" row scope_crux_path_passes_through 0 "receipts from $tmp/sc-crux-receipts" "$d"
    FX_EXPECT_CRUX="$tmp/sc-crux-receipts" SCOPE=crux-smoke CRUX="$tmp/sc-crux-receipts" row scope_crux_path_green 0 "OPERATOR EMERGENCY SCOPE crux-smoke satisfied" "$d"
    SCOPE=crux-smoke CRUX="$tmp/no-such-dir" row scope_crux_missing_dir_refuses 1 "is not a directory" "$d"
    d="$tmp/sc-badcut"; build_repo "$d"
    SCOPE=crux-smoke CUT_COMMIT=0123456789abcdef0123456789abcdef01234567 row scope_unresolvable_cut_refuses 1 "does not resolve" "$d"

    # --receipt-only (#3708): the T-1 end of R5. An UNTAGGED tree with a GO
    # receipt passes it (the full gate refuses the same tree on R3 -- the row
    # above -- which is why T-1 cannot run the full gate), and every receipt
    # the full gate refuses on R5 it refuses too, with the same row.
    d="$tmp/ro-untagged-go"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    row receipt_only_untagged_go_passes 0 "PASS  $PROG --receipt-only" "$d" receipt_gate
    d="$tmp/ro-nogo"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    write_receipt "$d" NO-GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row receipt_only_no_go_refuses      1 "FAIL  R5" "$d" receipt_gate
    d="$tmp/ro-stale"; build_repo "$d"; write_receipt "$d" GO 0000000000000000000000000000000000000000 1.2.3
    row receipt_only_stale_commit_refuses 1 "FAIL  R5" "$d" receipt_gate
    d="$tmp/ro-otherver"; build_repo "$d"; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.4
    row receipt_only_other_version_refuses 1 "FAIL  R5" "$d" receipt_gate
    d="$tmp/ro-absent"; build_repo "$d"; rm -f "$d/.dogfood/receipt-20260903T000000Z.json"
    row receipt_only_absent_refuses     1 "FAIL  R5 no dogfood receipt" "$d" receipt_gate
    d="$tmp/ro-baddefer"; build_repo "$d"
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3 pre-publish '["publish-dry-run","bashrs"]'
    row receipt_only_unexpected_deferral_refuses 1 "FAIL  R5" "$d" receipt_gate

    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

SCOPE=""; CUT_COMMIT=""; SCOPE_TREE=""; VIA=""; CRUX=""; MODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --scope) [ $# -ge 2 ] || { printf '%s: --scope needs a name\n' "$PROG" >&2; exit 2; }; SCOPE="$2"; shift 2 ;;
        --cut-commit) [ $# -ge 2 ] || { printf '%s: --cut-commit needs a sha\n' "$PROG" >&2; exit 2; }; CUT_COMMIT="$2"; shift 2 ;;
        --crux) [ $# -ge 2 ] || { printf '%s: --crux needs a directory\n' "$PROG" >&2; exit 2; }; CRUX="$2"; shift 2 ;;
        --via) [ $# -ge 2 ] || { printf '%s: --via needs a sha\n' "$PROG" >&2; exit 2; }; VIA="$2"; shift 2 ;;
        --scope-tree) [ $# -ge 2 ] || { printf '%s: --scope-tree needs a directory\n' "$PROG" >&2; exit 2; }; SCOPE_TREE="$2"; shift 2 ;;
        *) [ -z "$MODE" ] || { printf '%s: unexpected argument %s\n' "$PROG" "$1" >&2; exit 2; }; MODE="$1"; shift ;;
    esac
done
[ -z "$CUT_COMMIT$SCOPE_TREE$CRUX" ] || [ -n "$SCOPE" ] || { printf '%s: --cut-commit/--scope-tree/--crux are only meaningful with --scope\n' "$PROG" >&2; exit 2; }
case "$MODE" in
    --selftest) selftest ;;
    --receipt-only) receipt_gate ;;
    '')         gate ;;
    -h|--help)  sed -n '2,48p' "$0" ;;
    *)          printf '%s: unknown argument %s\n' "$PROG" "$MODE" >&2; exit 2 ;;
esac
