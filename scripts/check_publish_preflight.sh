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
#   R4  HEAD is an ancestor of origin/main OR of origin/release/<version>: nothing
#       publishes from a topic branch. Main (P5, operator ruling 2026-10-08): the
#       release script tags a commit on main and creates no release ref. The release
#       branch (#4286): RC binaries ship from it before merge-back, and main ancestry
#       is enforced at merge-back (#4224). A commit on neither refuses.
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
#   R8  the release's evidence graded by pv's `release-readiness-v1` SHACL shape (#3715 done_when 4),
#       through scripts/release/release_readiness.sh -- the same wrapper autopilot's T-1 `models` step
#       calls on its fresh receipts; here it reads the COMMITTED receipts at HEAD, with the dogfood
#       receipt R5 judged. Any non-zero from the wrapper refuses. The wrapper's committed DEFAULT_MODE
#       is `enforce` (#3715 B1, operator ruling 2026-09-28; there is no report mode): a Fail verdict
#       refuses, and a decline, a caller error or a missing pv is a non-zero too.
#       Under a RECORDED operator emergency scope (the ladder contract records exactly NAME for exactly
#       this release; engaged by `--scope NAME`, or with no flag by that record itself -- C280.4, Q1,
#       0.70.1) the wrapper still runs and its verdict and rc are printed as EVIDENCE, as the model
#       matrix is under R7's scope: it is not the gate for that release. A missing wrapper still
#       refuses. Without a recorded scope R8 is unchanged.
#
# EXIT  0 every rule holds · 1 a rule refused · 2 the box cannot answer
#       (no git/cargo/python3, not a repository). 2 is not a pass.
#
# SEAMS (the selftest builds a throwaway repository and drives every rule to
# both verdicts through them; production never sets them):
#   PUBLISH_PREFLIGHT_ROOT         repository root (default: this script's repo)
#   PUBLISH_PREFLIGHT_RELEASE_REF  R4's release ref (default: origin/release/<R2 version>;
#                                  the selftest leaves it unset so the derivation is tested)
#   PUBLISH_PREFLIGHT_RECEIPT_DIR  the dogfood receipt dir (default: $ROOT/.dogfood)
#   PUBLISH_PREFLIGHT_LADDER_JUDGE the R7 judge (default: $ROOT/scripts/check_model_ladder.sh)
#   PUBLISH_PREFLIGHT_READINESS    the R8 wrapper (default: $ROOT/scripts/release/release_readiness.sh)
#
# USAGE
#   bash scripts/check_publish_preflight.sh             # the gate
#   bash scripts/check_publish_preflight.sh --selftest  # case table, both polarities
#   bash scripts/check_publish_preflight.sh --receipt-only  # R2+R5 only: T-1, before the tag (#3708)
#   bash scripts/check_publish_preflight.sh --graph-only    # R2+R6 only: the rc cut (#4287)
#   bash scripts/check_publish_preflight.sh --scope crux-smoke [--cut-commit SHA]
#       R7 under a RECORDED operator emergency scope (contracts/model-capability-ladder-v1.yaml
#       `ladder.emergency_scopes`; only the release its entry names -- 0.69.1, 0.70.1): the judge's own `--scope` path
#       (scripts/lib/crux_smoke_scope.py) decides R7 from CRUX smoke receipts bound to the CUT --
#       the commit the release binary was built from -- instead of the model matrix. The cut
#       defaults to HEAD; when HEAD is not the cut (receipts committed on top, or main's squash
#       of it), every PUBLISHED path -- crates/ src/ Cargo.toml Cargo.lock, R4's set -- must be
#       equal to the cut's, or the published source is not the smoked binary's. The
#       model-matrix rows are still printed, as EVIDENCE, never as the verdict. When the contract
#       records NAME for this release, R8's release-readiness verdict is printed as EVIDENCE too.
#       With NO --scope (cascade-publish.sh and autopilot's T-4 run this gate bare), a scope the
#       contract RECORDS for this release engages by itself, as the judge's own auto-scope does for
#       the dogfood (#4086; Q1): it is printed `SCOPED:` and the cut is HEAD. No record = the full
#       gate. An unusable record (two for one release, no name, an unreadable contract) refuses.
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
        rule_r7_scope "$root" "$version" "$judge"
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

# R6, ONE function for both ends (#4287): the full gate at T-4, and --graph-only at the
# rc cut, so a publish-graph defect is found on the rc and not at the final tag.
# rule_r6 root -> prints its row; 0 accepted, 1 refused
rule_r6() {
    local root=$1
    # R6 no versioned sibling dev-dependency lies on a cycle (PMAT-955, #3468). A
    # dev-dependency with a version is kept in the published manifest and resolved
    # on the registry at publish time; a path-only one is stripped. The edge is a
    # defect only when its target can reach its source: then neither crate can be
    # uploaded first. Acyclic edges are printed, so the publish order that must
    # honour them is visible in the receipt.
    local r6
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
        return 1
    elif grep -q '^CYCLE ' <<< "$r6"; then
        printf 'FAIL  R6 a versioned sibling dev-dependency lies on a cycle (kept in the published manifest; neither crate can be uploaded first):\n%s\n' \
            "$(printf '%s\n' "$r6" | sed -n 's/^CYCLE //p')"
        return 1
    elif [ -n "$r6" ]; then
        printf 'ok    R6 %s versioned sibling dev-dependency edge(s), none on a cycle (the target publishes first):\n%s\n' \
            "$(printf '%s\n' "$r6" | grep -c '^ACYCLIC ')" "$(printf '%s\n' "$r6" | sed -n 's/^ACYCLIC /        /p')"
    else
        echo "ok    R6 no sibling dev-dependency carries a version (path-only, stripped at publish)"
    fi
    return 0
}

# R7 under a recorded operator emergency scope (0.69.1: CRUX smoke only). The scope is READ by the
# judge (`--scope`, scripts/lib/crux_smoke_scope.py), never re-implemented here: it refuses another
# release, receipts from another binary, and a missing host. This rule adds the one binding the judge
# cannot see: the source being PUBLISHED (crates/ src/ Cargo.toml Cargo.lock, the paths R4 judges) is
# the source the smoked binary was built from. Scripts, contracts and evidence may differ: the scope's
# own contract entry and reader arrive after the cut.
# rule_r7_scope root version judge -> prints its rows; 0 accepted, 1 refused
rule_r7_scope() {
    local root="$1" version="$2" judge="$3" cut head out rc ev evrc
    head="$(git -C "$root" rev-parse HEAD 2>/dev/null)"
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
    out="$(cd "$root" && bash "$judge" --version "$version" --scope "$SCOPE" --cut-commit "$cut" 2>&1)"; rc=$?
    grep -E '^OPERATOR EMERGENCY SCOPE' <<< "$out" | head -n 1 | sed 's/^/        /'
    # The model matrix, reported as EVIDENCE only: under the scope it is not the verdict, and a
    # stale or red row must still be visible.
    # `--scope none`: since #4086 a bare call on a release with a RECORDED scope judges the scope again,
    # and the matrix rows (the old failures, C280.6) would never be printed.
    ev="$(cd "$root" && bash "$judge" --version "$version" --scope none 2>&1)"; evrc=$?
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

# C280 (operator, 2026-10-03, 0.70.1): a scope is RECORDED only when the ladder contract names exactly
# this scope for exactly this release, as read by the judge's own reader (crux_smoke_scope.recorded_scope).
# `--scope` alone is not a record. Anything unreadable is "not recorded", so R8 stays enforced.
# scope_recorded root version scope -> 0 recorded, 1 not
scope_recorded() {
    local root="$1" version="$2" scope="$3"
    [ -n "$scope" ] || return 1
    ( cd "$root" && PYTHONDONTWRITEBYTECODE=1 python3 -c 'import sys, yaml
sys.path.insert(0, "scripts/lib"); import crux_smoke_scope
name, why = crux_smoke_scope.recorded_scope(yaml.safe_load(open(sys.argv[3]))["ladder"], sys.argv[1])
sys.exit(0 if why is None and name == sys.argv[2] else 1)' "$version" "$scope" "${POL_LADDER:-contracts/model-capability-ladder-v1.yaml}" ) >/dev/null 2>&1
}

# Q1 (operator, 2026-10-03, 0.70.1): the scope RECORDED for this release, for a caller that passed no
# --scope. cascade-publish.sh and autopilot's T-4 run this gate bare, as the dogfood runs the judge bare,
# and the judge's own auto-scope (#4086) already serves that caller. Read by the judge's own reader;
# never inferred. No contract at all = no record: the full gate.
# recorded_scope_name root version -> sets REC_NAME (the name) or REC_WHY (why it is unusable);
#   0 one record, 1 no record, 2 unusable
recorded_scope_name() {
    local root="$1" version="$2" rec_out rec_rc py
    REC_NAME=""; REC_WHY=""
    [ -f "$root/contracts/model-capability-ladder-v1.yaml" ] || return 1
    py='import sys
try:
    import yaml
    sys.path.insert(0, "scripts/lib"); import crux_smoke_scope
    name, why = crux_smoke_scope.recorded_scope(yaml.safe_load(open(sys.argv[2]))["ladder"], sys.argv[1])
except Exception as e:
    print(f"the ladder contract could not be loaded ({type(e).__name__}: {e})"); sys.exit(2)
if why:
    print(why); sys.exit(2)
if not name:
    sys.exit(1)
print(name)'
    # bashrs PERF002: not a loop body -- the reader runs once, from gate().
    # bashrs disable-next-line=PERF002
    rec_out="$(cd "$root" && PYTHONDONTWRITEBYTECODE=1 python3 -c "$py" "$version" "${POL_LADDER:-contracts/model-capability-ladder-v1.yaml}" 2>/dev/null)"; rec_rc=$?
    case "$rec_rc" in
        0) REC_NAME="$rec_out" ;;
        1) return 1 ;;
        *) REC_WHY="${rec_out:-the reader exited $rec_rc and printed no reason}"; return 2 ;;
    esac
}

# The STANDING RELEASE POLICY (`ladder.release_policy`, scripts/lib/release_policy.sh): from its `since` on,
# a release is judged on CRUX smoke as if the contract recorded a `crux-smoke` entry for it. Both scope readers
# above read POL_LADDER: the contract itself, or a copy carrying the entry the policy grants this version.
# Without the reader in this tree, a contract that HAS a policy block is unreadable, never "no policy".
# policy_ladder root version -> sets POL_LADDER (absolute), POL_APPLIES, POL_WHY;
#   0 ok, 1 a per-release entry names a covered version, 2 unreadable
policy_ladder() {
    local root="$1" version="$2" out prc
    POL_LADDER="$root/contracts/model-capability-ladder-v1.yaml"; POL_APPLIES=0; POL_WHY=""
    [ -f "$POL_LADDER" ] || return 0
    if [ ! -f "$root/scripts/lib/release_policy.sh" ]; then
        if grep -q '^  release_policy:' "$POL_LADDER"; then POL_WHY="the contract has a release_policy block but $root has no scripts/lib/release_policy.sh to read it"; return 2; fi
        return 0
    fi
    # shellcheck source=lib/release_policy.sh
    . "$root/scripts/lib/release_policy.sh" || { POL_WHY="$root/scripts/lib/release_policy.sh could not be loaded"; return 2; }
    out="$(mktemp)" || { POL_WHY="mktemp failed, so the policy ladder could not be written"; return 2; }
    release_policy_ladder "$POL_LADDER" "$version" > "$out"; prc=$?
    POL_LADDER="$(cat "$out")"; rm -f "${out:?}"
    POL_APPLIES="$RP_APPLIES"; POL_WHY="$RP_WHY"
    [ "$prc" = 0 ] || POL_LADDER="$root/contracts/model-capability-ladder-v1.yaml"
    return "$prc"
}

# rm_pol_ladder root -- removes policy_ladder's copy, never the contract
rm_pol_ladder() {
    if [ -n "${POL_LADDER:-}" ] && [ "$POL_LADDER" != "$1/contracts/model-capability-ladder-v1.yaml" ] && [ -f "$POL_LADDER" ]; then rm -f "${POL_LADDER:?}"; fi
    POL_LADDER=""
}

# R8, release-readiness-v1 (#3715): the committed receipts at HEAD, graded by the shape.
# Under a RECORDED operator emergency scope (C280.4) the wrapper still runs and its verdict is printed
# as EVIDENCE, exactly as the model matrix is under R7's scope: it is not the gate for that release.
# Without a recorded scope R8 is unchanged: ENFORCE PASS or refuse.
# rule_r8 root version head -> prints its rows; 0 accepted, 1 refused
rule_r8() {
    local root="$1" version="$2" head="$3" wrapper receipt out rc
    wrapper="${PUBLISH_PREFLIGHT_READINESS:-$root/scripts/release/release_readiness.sh}"
    if [ ! -f "$wrapper" ]; then
        echo "FAIL  R8 no release-readiness wrapper at $wrapper: the release evidence cannot be graded"
        return 1
    fi
    # The STANDING RELEASE POLICY covers this version: R8 is not run at all (no evidence state on the
    # release path). CRUX smoke under R7 is the gate; the larger rows are the nightly's, ticketed on red.
    if [ "${POL_APPLIES:-0}" = 1 ]; then
        echo "ok    R8 STANDING RELEASE POLICY covers $version: release-readiness was not run; CRUX smoke (R7) is the release gate"
        R8_SCOPED=1
        return 0
    fi
    receipt="$(newest_receipt "${PUBLISH_PREFLIGHT_RECEIPT_DIR:-$root/.dogfood}")"
    out="$(bash "$wrapper" --root "$root" --version "$version" --commit "$head" ${receipt:+--dogfood-receipt "$receipt"} 2>&1)"; rc=$?
    if scope_recorded "$root" "$version" "${SCOPE:-}"; then
        printf '        evidence only (NOT the verdict under the emergency scope): release-readiness wrapper rc %s\n%s\n' "$rc" \
            "$(printf '%s\n' "$out" | sed 's/^/          evidence /')"
        echo "ok    R8 OPERATOR EMERGENCY SCOPE $SCOPE recorded for $version: release-readiness ran (rc $rc), printed above as EVIDENCE; it is NOT the gate for $version"
        R8_SCOPED=1
        return 0
    fi
    printf '%s\n' "$out"
    if [ "$rc" -ne 0 ]; then
        echo "FAIL  R8 the release-readiness wrapper exited $rc (1 Fail, 2 could not judge, 3 caller error): none is a pass"
        return 1
    fi
    # R10/L19 (operator 2026-09-28): report-only is a waiver and a waiver is a stop. rc 0 alone is not
    # a pass: the wrapper must have printed the enforced Pass for exactly this version and HEAD.
    # A here-string, never `printf | grep -q`: grep -q exits at its first match, printf takes SIGPIPE,
    # and pipefail turns a FOUND WARN row into a miss (row r8_warn_ahead_of_2mib_refuses).
    if grep -qE '^WARN +R8 ' <<< "$out"; then
        echo "FAIL  R8 the release-readiness wrapper printed a WARN R8 row: report-only is a waiver, not a pass"
        return 1
    fi
    if ! grep -qF "ok    R8 #3715 ENFORCE PASS version=$version commit=$head pv=" <<< "$out"; then
        echo "FAIL  R8 the release-readiness wrapper exited 0 without '#3715 ENFORCE PASS' for $version at $head"
        return 1
    fi
    return 0
}

gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" release_ref r4_ref r4_on
    local fails=0 status version tags head
    R8_SCOPED=""
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

    # The standing release policy: what both scope readers below read (policy_ladder).
    POL_LADDER=""
    local pol_bad=0
    if [ -n "$version" ]; then
        local prc=0
        policy_ladder "$root" "$version" || prc=$?
        case "$prc" in
            0) [ "$POL_APPLIES" != 1 ] || echo "POLICY: crux-smoke -- the standing release policy in contracts/model-capability-ladder-v1.yaml covers $version: CRUX smoke under R7 is the release gate, R8 is not run, the larger ladder rows are nightly" ;;
            *) echo "FAIL  R7/R8 the standing release policy cannot be applied to $version, so neither the scope nor the full gate can be chosen: $POL_WHY"
               fails=1; pol_bad=1 ;;
        esac
    fi

    # Q1: no --scope on a release whose contract RECORDS one -> that scope, for R7 and R8, at the cut
    # HEAD -- as the judge's own auto-scope (#4086) does. An unusable record refuses.
    if [ -z "${SCOPE:-}" ] && [ -n "$version" ] && [ "$pol_bad" = 0 ]; then
        local rrc
        recorded_scope_name "$root" "$version"; rrc=$?
        case "$rrc" in
            0) SCOPE="$REC_NAME"
               echo "SCOPED: $SCOPE -- the operator emergency scope recorded for release $version in contracts/model-capability-ladder-v1.yaml applies (no --scope given): R7 and R8 judge under it, at the cut HEAD" ;;
            1) : ;;
            *) echo "FAIL  R7/R8 the emergency scope recorded for $version is unusable, so neither the scope nor the full gate can be chosen: $REC_WHY"
               fails=1 ;;
        esac
    fi

    # R3 the tag points at HEAD
    tags="$(git -C "$root" tag --points-at HEAD 2>/dev/null)"
    # -F: the version is a string, not a pattern. With -x alone `v1-2-3` on HEAD
    # satisfied `v1.2.3` (second review of #2859, tag-regex-injection).
    # A release rehearsal (scripts/release/rehearse.sh, APR-071 B1/H10) cuts no tag: its tag step
    # writes "<tag> <commit>" for the tag it would make, and R3 judges THAT tag against HEAD. Only
    # under RELEASE_REHEARSAL=1; otherwise the variable is ignored and the real tag is required.
    if [ "${RELEASE_REHEARSAL:-}" = 1 ] && [ -n "${PUBLISH_PREFLIGHT_WOULD_TAG:-}" ]; then
        if [ -n "$version" ] && [ "$PUBLISH_PREFLIGHT_WOULD_TAG" = "v$version $head" ]; then
            echo "ok    R3 would-be tag v$version names HEAD ${head:0:9} (rehearsal: the tag is a WOULD line)"
        else
            printf 'FAIL  R3 would-be tag (%s) is not v%s on HEAD %s\n' \
                "$PUBLISH_PREFLIGHT_WOULD_TAG" "${version:-?}" "${head:0:9}"
            fails=1
        fi
    elif [ -n "$version" ] && grep -Fqx -- "v$version" <<<"$tags"; then
        echo "ok    R3 tag v$version points at HEAD ${head:0:9}"
    else
        printf 'FAIL  R3 tag v%s does not point at HEAD %s (tags here: %s)\n' \
            "${version:-?}" "${head:0:9}" "${tags:-none}"
        fails=1
    fi

    # R4 HEAD is on main or on the release branch of THIS version (P5, operator ruling
    # 2026-10-08). The release script tags a commit on main and creates no release ref, so
    # a release-branch-only R4 refused every unattended pass (D1); an rc fixed on the
    # release branch before merge-back (#4286) still passes. A commit on neither refuses.
    # No version, no release ref to judge: refuse.
    release_ref="${PUBLISH_PREFLIGHT_RELEASE_REF:-origin/release/${version:-?}}"
    r4_on=""
    if [ -n "$version" ]; then
        for r4_ref in "$release_ref" origin/main; do
            if git -C "$root" rev-parse --verify --quiet "${r4_ref}^{commit}" >/dev/null \
               && git -C "$root" merge-base --is-ancestor "$head" "$r4_ref" 2>/dev/null; then
                r4_on="$r4_ref"
                break
            fi
        done
    fi
    if [ -n "$r4_on" ]; then
        echo "ok    R4 HEAD is an ancestor of $r4_on"
    else
        echo "FAIL  R4 HEAD ${head:0:9} is an ancestor of neither origin/main nor $release_ref (or the ref does not exist)"
        fails=1
    fi

    # R5 dogfood receipt: GO, this commit, this version
    rule_r5 "$root" "$head" "$version" || fails=1

    rule_r6 "$root" || fails=1

    # R7 the model matrix, re-read at T-4 through the T-1 judge (#3717)
    rule_r7 "$root" "$version" || fails=1

    # R8 release-readiness-v1 over the committed evidence (#3715)
    rule_r8 "$root" "$version" "$head" || fails=1

    rm_pol_ladder "$root"
    if [ "$fails" -ne 0 ]; then
        echo "REFUSE $PROG: publishing is not allowed from this tree (see the FAIL rows)."
        return 1
    fi
    if [ -n "${SCOPE:-}" ]; then
        echo "PASS  $PROG: clean, versioned, tagged, on $release_ref, dogfood GO, OPERATOR EMERGENCY SCOPE $SCOPE satisfied (the model matrix was NOT the gate${R8_SCOPED:+; release-readiness was NOT the gate})"
    else
        echo "PASS  $PROG: clean, versioned, tagged, on $release_ref, dogfood GO, model matrix green"
    fi
    return 0
}

# --graph-only (#4287): R2 + R6 on PUBLISH_PREFLIGHT_ROOT, the rc cut's end of the
# publish graph. R1/R3/R4/R5/R7 describe the upload (a tag, main or the release branch,
# receipts) and are judged at T-4 as before.
graph_gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" version
    for t in cargo python3; do
        command -v "$t" >/dev/null 2>&1 || die_env "$t is not on PATH"
    done
    if [ -z "$root" ]; then
        root="$(cd -- "$SCRIPT_DIR/.." && pwd)"
    fi
    [ -f "$root/Cargo.toml" ] || die_env "$root has no Cargo.toml"
    version="$(root_version "$root")" || version=""
    if [ -z "$version" ]; then
        echo "FAIL  R2 cargo metadata names no version for the root manifest"
        echo "REFUSE $PROG --graph-only: no version to judge."
        return 1
    fi
    echo "ok    R2 version $version (cargo metadata, root manifest)"
    if ! rule_r6 "$root"; then
        echo "REFUSE $PROG --graph-only: the publish graph of $version cannot be uploaded (R6)."
        return 1
    fi
    echo "PASS  $PROG --graph-only: R6 holds at $version (R1/R3/R4/R5/R7 are judged at publish)"
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
        git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
        git -C "$d" update-ref refs/remotes/origin/main HEAD
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
if [ "${3:-}" = "--scope" ] && [ "${4:-}" != none ]; then
    [ "${1:-} ${2:-} ${4:-} ${5:-}" = "--version 1.2.3 crux-smoke --cut-commit" ] || { echo "FAIL  judge scope call: $*"; exit 1; }
    [ "${6:-}" = "$(git rev-parse "${FX_EXPECT_CUT:-HEAD}")" ] || { echo "FAIL  judge asked about cut ${6:-}"; exit 1; }
    echo "OPERATOR EMERGENCY SCOPE: CRUX smoke only -- release 1.2.3 (fixture)"
    [ "${FX_SCOPE_RC:-0}" = 0 ] || echo "FAIL  host gx10 has no CRUX receipt"
    exit "${FX_SCOPE_RC:-0}"
fi
[ "${1:-} ${2:-}" = "--version 1.2.3" ] || { echo "FAIL  judge asked about: $*"; exit 1; }
# the real judge's #4086: a bare call on a release the contract records a scope for judges that SCOPE,
# not the matrix; `--scope none` forces the matrix
if [ -z "${3:-}" ] && grep -qs 'release: "1.2.3"' contracts/model-capability-ladder-v1.yaml; then
    echo "SCOPED: crux-smoke (fixture)"; exit "${FX_SCOPE_RC:-0}"
fi
[ "${FX_LADDER_RC:-0}" = 0 ] || echo "FAIL  fx-rung red on lambda"
exit "${FX_LADDER_RC:-0}"
FXJUDGE
        # R8's wrapper: it must be asked about THIS root, version 1.2.3, HEAD and the newest dogfood
        # receipt, and it answers FX_READINESS_RC (default 0, with the enforced-Pass line). FX_READINESS_WARN=1
        # prints a WARN R8 row and exits 0: only a hand-edited wrapper can, and R8 refuses it (R10).
        # FX_READINESS_NO_ENFORCE=1 exits 0 without the enforced-Pass line: refused too.
        mkdir -p "$1/scripts/release"
        cat > "$1/scripts/release/release_readiness.sh" <<'FXREADY'
#!/usr/bin/env bash
me="$(cd "$(dirname "$0")/../.." && pwd -P)"
[ "${1:-}" = --root ] && [ "${2:-}" -ef "$me" ] || { echo "FAIL  R8 wrapper asked about root ${2:-}, not $me"; exit 3; }
want="--version 1.2.3 --commit $(git -C "$me" rev-parse HEAD) --dogfood-receipt"
case "${*:3}" in "$want "*.dogfood/receipt-*.json) : ;; *) echo "FAIL  R8 wrapper asked: $*"; exit 3 ;; esac
[ "${FX_READINESS_WARN:-0}" = 1 ] && echo "WARN  R8 REPORT-ONLY release-readiness-v1 for 1.2.3: Fail, 3 violation(s): cell=3"
# FX_READINESS_WARN_BIG=1: a WARN R8 row, then 2 MiB, then the Pass lines and exit 0 (a pipe into grep -q loses the WARN)
[ "${FX_READINESS_WARN_BIG:-0}" = 1 ] && { echo "WARN  R8 REPORT-ONLY release-readiness-v1 for 1.2.3: Fail, 1 violation(s): cell=1"; head -c 2097152 /dev/zero | tr '\0' '\n'; }
[ "${FX_READINESS_RC:-0}" = 0 ] && [ "${FX_READINESS_WARN:-0}" = 0 ] && echo "ok    R8 release-readiness-v1 for 1.2.3: Pass"
[ "${FX_READINESS_RC:-0}" = 0 ] && [ "${FX_READINESS_WARN:-0}" = 0 ] && [ "${FX_READINESS_NO_ENFORCE:-0}" = 0 ] \
    && echo "ok    R8 #3715 ENFORCE PASS version=1.2.3 commit=$(git -C "$me" rev-parse HEAD) pv=pv-fixture out_sha256=0"
exit "${FX_READINESS_RC:-0}"
FXREADY
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
        git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
        mkdir -p "$d/.dogfood"
        write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    }
    row() { # name, expect(0|1), needle, dir [, gate|receipt_gate]
        local name="$1" expect="$2" needle="$3" d="$4" mode="${5:-gate}" out rc=0
        out="$( PUBLISH_PREFLIGHT_ROOT="$d" "$mode" 2>&1 )" || rc=$?
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

    # R3 in a rehearsal (B1, E1 #3998): no tag exists, the tag step's WOULD line names it. The would-be
    # tag must be v<version> on HEAD; outside a rehearsal the same line is ignored and the real tag rules.
    d="$tmp/would-tag"; build_repo "$d"; git -C "$d" tag -d v1.2.3 >/dev/null
    RELEASE_REHEARSAL=1 PUBLISH_PREFLIGHT_WOULD_TAG="v1.2.3 $(git -C "$d" rev-parse HEAD)" \
        row rehearsal_would_tag_on_head_holds 0 "ok    R3 would-be tag v1.2.3" "$d"
    RELEASE_REHEARSAL=1 PUBLISH_PREFLIGHT_WOULD_TAG="v1.2.3 $(git -C "$d" rev-parse HEAD~0^{tree})" \
        row rehearsal_would_tag_elsewhere_refuses 1 "FAIL  R3 would-be tag" "$d"
    RELEASE_REHEARSAL=1 PUBLISH_PREFLIGHT_WOULD_TAG="v1.2.4 $(git -C "$d" rev-parse HEAD)" \
        row rehearsal_would_tag_other_version_refuses 1 "FAIL  R3 would-be tag" "$d"
    PUBLISH_PREFLIGHT_WOULD_TAG="v1.2.3 $(git -C "$d" rev-parse HEAD)" \
        row would_tag_outside_rehearsal_ignored 1 "FAIL  R3 tag v1.2.3 does not point" "$d"

    d="$tmp/branch"; build_repo "$d"; git -C "$d" checkout -q -b topic
    printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'topic' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row head_on_neither_main_nor_release_refuses 1 "FAIL  R4 HEAD" "$d"

    # R4 (P5, operator ruling 2026-10-08): main OR the release branch. The release script
    # tags a commit on main and creates no release ref (D1), so that cut passes, also after
    # main moves on. An rc commit on release/1.2.3 that main does not contain yet still
    # passes (#4286). A commit on neither refuses, with both refs, with neither, and when
    # the only release branch that holds it is another version's.
    d="$tmp/rc-on-release"; build_repo "$d"; git -C "$d" checkout -q -b release-1.2.3
    printf 'pub fn r() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'rc fix' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    ! git -C "$d" merge-base --is-ancestor HEAD origin/main || { echo "  BROKE fixture: rc commit is on main"; fail=$((fail + 1)); }
    row rc_on_release_not_main_passes  0 "ok    R4 HEAD is an ancestor of origin/release/1.2.3" "$d"
    d="$tmp/on-main"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
    row on_main_without_release_ref_passes 0 "ok    R4 HEAD is an ancestor of origin/main" "$d"
    d="$tmp/main-moved-on"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
    git -C "$d" update-ref refs/remotes/origin/main "$(git -C "$d" -c user.name=t -c user.email=t@t commit-tree 'HEAD^{tree}' -p HEAD -m 'later on main')"
    row main_moved_past_head_passes    0 "ok    R4 HEAD is an ancestor of origin/main" "$d"
    d="$tmp/no-refs"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
    git -C "$d" update-ref -d refs/remotes/origin/main
    row neither_ref_exists_refuses     1 "FAIL  R4 HEAD" "$d"
    d="$tmp/other-release"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
    git -C "$d" checkout -q -b release-1.2.4
    printf 'pub fn o() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'rc 1.2.4' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; git -C "$d" update-ref refs/remotes/origin/release/1.2.4 HEAD
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row other_versions_release_refuses 1 "FAIL  R4 HEAD" "$d"

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
    # --graph-only (#4287), both polarities: the rc cut refuses the same cycle, and passes
    # an untagged tree off its release branch that the full gate would refuse on R3/R4.
    d="$tmp/devdep_cycle"; row graph_only_cycle_refuses      1 "FAIL  R6" "$d" graph_gate
    d="$tmp/devdep_version"; git -C "$d" tag -d v1.2.3 >/dev/null
    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m 'off release' >/dev/null
    row graph_only_acyclic_untagged_passes 0 "PASS  $PROG --graph-only" "$d" graph_gate
    row graph_only_control_full_gate_refuses 1 "FAIL  R3" "$d"

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

    # R8 (#3715): the release-readiness wrapper over the committed evidence. all_rules_hold above is the
    # green row (the stub exits 3 unless it is asked about this root, 1.2.3, HEAD and the dogfood receipt).
    d="$tmp/r8"; build_repo "$d"
    row r8_pass_is_named                  0 "ok    R8 release-readiness-v1 for 1.2.3: Pass" "$d"
    FX_READINESS_WARN=1 row r8_report_mode_warn_refuses 1 "report-only is a waiver" "$d"
    FX_READINESS_WARN_BIG=1 row r8_warn_ahead_of_2mib_refuses 1 "report-only is a waiver" "$d"
    FX_READINESS_NO_ENFORCE=1 row r8_rc0_without_enforce_pass_refuses 1 "without '#3715 ENFORCE PASS'" "$d"
    FX_READINESS_RC=1 row r8_enforced_fail_refuses 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    FX_READINESS_RC=2 row r8_could_not_judge_refuses 1 "FAIL  R8 the release-readiness wrapper exited 2" "$d"
    FX_READINESS_RC=3 row r8_caller_error_refuses 1 "FAIL  R8 the release-readiness wrapper exited 3" "$d"
    d="$tmp/r8-absent"; build_repo "$d"; git -C "$d" rm -q scripts/release/release_readiness.sh
    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'no wrapper' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row r8_wrapper_absent_refuses      1 "FAIL  R8 no release-readiness wrapper" "$d"
    # R8 under a RECORDED operator emergency scope (C280.4, 0.70.1): the wrapper still runs, its verdict
    # and rc are printed as EVIDENCE (as R7's model matrix is), and it is not the gate. "Recorded" = the
    # ladder contract names exactly this scope for exactly this release, read by the judge's OWN reader
    # (copied from this checkout). `--scope` alone is not a record, and a record alone relaxes nothing.
    record_scope() { # dir, release... -- one crux-smoke entry per release given, committed on top
        local d="$1"; shift
        # bashrs SEC010: self-test fixture: $d is under this script's own mktemp -d dir.
        # bashrs disable-next-line=SEC010
        mkdir -p "$d/contracts" "$d/scripts/lib"
        # bashrs SEC010,SEC014: the sources are this checkout's own reader files; $d is the mktemp -d fixture above.
        # bashrs disable-next-line=SEC010,SEC014
        cp -- "$SCRIPT_DIR/lib/crux_smoke_scope.py" "$SCRIPT_DIR/lib/model_ladder_crux.py" "$d/scripts/lib/"
        printf 'ladder:\n  emergency_scopes:\n' > "$d/contracts/model-capability-ladder-v1.yaml"
        printf '    - name: crux-smoke\n      release: "%s"\n' "$@" >> "$d/contracts/model-capability-ladder-v1.yaml"
        git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'scope' >/dev/null
        git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
        git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    }
    d="$tmp/r8-scoped"; build_repo "$d"; record_scope "$d" 1.2.3
    FX_READINESS_RC=1 SCOPE=crux-smoke row r8_scoped_fail_is_evidence 0 "ok    R8 OPERATOR EMERGENCY SCOPE crux-smoke recorded for 1.2.3: release-readiness ran (rc 1)" "$d"
    FX_READINESS_RC=1 SCOPE=crux-smoke row r8_scoped_prints_the_wrapper_rc 0 "evidence only (NOT the verdict under the emergency scope): release-readiness wrapper rc 1" "$d"
    FX_READINESS_WARN=1 SCOPE=crux-smoke row r8_scoped_prints_the_wrapper_rows 0 "evidence WARN  R8 REPORT-ONLY release-readiness-v1 for 1.2.3: Fail" "$d"
    FX_READINESS_NO_ENFORCE=1 SCOPE=crux-smoke row r8_scoped_no_enforce_pass_is_evidence 0 "release-readiness ran (rc 0)" "$d"
    FX_READINESS_RC=2 SCOPE=crux-smoke row r8_scoped_could_not_judge_is_evidence 0 "release-readiness ran (rc 2)" "$d"
    FX_READINESS_RC=3 SCOPE=crux-smoke row r8_scoped_caller_error_is_evidence 0 "release-readiness ran (rc 3)" "$d"
    SCOPE=crux-smoke row r8_scoped_pass_is_printed_too 0 "evidence ok    R8 #3715 ENFORCE PASS version=1.2.3" "$d"
    FX_READINESS_RC=1 SCOPE=crux-smoke row r8_scoped_gate_names_readiness 0 "release-readiness was NOT the gate" "$d"
    # ... and the other way: every case that is not a recorded scope keeps R8 exactly as it was
    # (a record with NO --scope engages the scope since Q1: its rows are the auto_scope_* block below)
    FX_READINESS_RC=1 SCOPE=other-scope row r8_unrecorded_scope_name_refuses 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    FX_READINESS_RC=1 SCOPE=crux-smoke row r8_scope_without_a_contract_refuses 1 "FAIL  R8 the release-readiness wrapper exited 1" "$tmp/r8"
    d="$tmp/r8-other-release"; build_repo "$d"; record_scope "$d" 0.69.1
    FX_READINESS_RC=1 SCOPE=crux-smoke row r8_scope_of_another_release_refuses 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    FX_READINESS_NO_ENFORCE=1 SCOPE=crux-smoke row r8_scope_of_another_release_no_enforce_refuses 1 "without '#3715 ENFORCE PASS'" "$d"
    d="$tmp/r8-two-records"; build_repo "$d"; record_scope "$d" 1.2.3 1.2.3
    FX_READINESS_RC=1 SCOPE=crux-smoke row r8_two_records_for_one_release_refuse 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    d="$tmp/r8-scoped-absent"; build_repo "$d"; record_scope "$d" 1.2.3; git -C "$d" rm -q scripts/release/release_readiness.sh
    git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'no wrapper' >/dev/null
    git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    SCOPE=crux-smoke row r8_scoped_wrapper_absent_refuses 1 "FAIL  R8 no release-readiness wrapper" "$d"
    # Q1: NO --scope on a release whose contract records one. cascade-publish.sh and autopilot's T-4 run
    # this gate bare; the record alone engages the scope, as the judge's #4086 auto-scope does, at the cut HEAD.
    d="$tmp/auto-scoped"; build_repo "$d"; record_scope "$d" 1.2.3
    row auto_scope_is_printed 0 "SCOPED: crux-smoke -- the operator emergency scope recorded for release 1.2.3" "$d"
    FX_READINESS_RC=1 row auto_scope_r8_readiness_is_evidence 0 "ok    R8 OPERATOR EMERGENCY SCOPE crux-smoke recorded for 1.2.3: release-readiness ran (rc 1)" "$d"
    FX_READINESS_WARN=1 row auto_scope_r8_warn_is_evidence 0 "evidence WARN  R8 REPORT-ONLY release-readiness-v1 for 1.2.3: Fail" "$d"
    FX_LADDER_RC=1 row auto_scope_r7_judges_the_scope_at_head 0 "CRUX smoke satisfied at the cut" "$d"
    FX_SCOPE_RC=1 row auto_scope_red_refuses 1 "FAIL  R7 OPERATOR EMERGENCY SCOPE crux-smoke NOT satisfied" "$d"
    FX_SCOPE_RC=2 row auto_scope_decline_refuses 1 "the judge DECLINED" "$d"
    FX_READINESS_RC=1 row auto_scope_pass_names_both 0 "release-readiness was NOT the gate" "$d"
    # ... and the other way: no record for THIS release, or a record that cannot be used, never relaxes
    d="$tmp/auto-other-release"; build_repo "$d"; record_scope "$d" 0.69.1
    FX_READINESS_RC=1 row auto_scope_other_release_r8_enforced 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    FX_LADDER_RC=1 row auto_scope_other_release_r7_is_matrix 1 "FAIL  R7 model matrix NOT green" "$d"
    FX_READINESS_RC=1 row auto_scope_no_contract_r8_enforced 1 "FAIL  R8 the release-readiness wrapper exited 1" "$tmp/r8"
    d="$tmp/auto-two"; build_repo "$d"; record_scope "$d" 1.2.3 1.2.3
    row auto_scope_two_records_refuse 1 "FAIL  R7/R8 the emergency scope recorded for 1.2.3 is unusable" "$d"
    d="$tmp/auto-unreadable"; build_repo "$d"; record_scope "$d" 1.2.3
    printf 'ladder: [\n' > "$d/contracts/model-capability-ladder-v1.yaml"
    git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'unreadable' >/dev/null
    git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
    # bashrs PERF002: not a loop body -- one fixture line, run once.
    # bashrs disable-next-line=PERF002
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row auto_scope_unreadable_contract_refuses 1 "the ladder contract could not be loaded" "$d"
    # The STANDING RELEASE POLICY (`ladder.release_policy`): from `since` on, the release is scoped with no
    # per-release record. The fixture carries this checkout's own reader (scripts/lib/release_policy*).
    record_policy() { # dir, since [, per-release entry release | nolib]
        local d="$1"
        # bashrs SEC010: self-test fixture: $d is under this script's own mktemp -d dir.
        # bashrs disable-next-line=SEC010
        mkdir -p "$d/contracts" "$d/scripts/lib"
        # bashrs SEC010,SEC014: the sources are this checkout's own reader files; $d is the mktemp -d fixture above.
        # bashrs disable-next-line=SEC010,SEC014
        cp -- "$SCRIPT_DIR/lib/crux_smoke_scope.py" "$SCRIPT_DIR/lib/model_ladder_crux.py" "$d/scripts/lib/"
        if [ "${3:-}" != nolib ]; then
            # bashrs disable-next-line=SEC010,SEC014
            cp -- "$SCRIPT_DIR/lib/release_policy.sh" "$SCRIPT_DIR"/lib/release_policy_*.awk "$d/scripts/lib/"
        fi
        { printf 'ladder:\n  release_policy:\n    name: crux-smoke\n    since: "%s"\n    date: "2026-10-07"\n' "$2"
          printf '    quote: "q"\n    hosts: [lambda, gx10]\n    thinking: ["off", "on"]\n    larger_rows: nightly\n'
          printf '    red_row_needs: ticket\n    ticket_owner: "#1"\n    release_notes: known_failures\n  emergency_scopes:\n'
          case "${3:-}" in ''|nolib) : ;; *) printf '    - name: crux-smoke\n      release: "%s"\n' "$3" ;; esac
        } > "$d/contracts/model-capability-ladder-v1.yaml"
        git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'policy' >/dev/null
        git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
        # bashrs disable-next-line=PERF002
        git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    }
    d="$tmp/pol-cov"; build_repo "$d"; record_policy "$d" 1.2.0
    row policy_covers_is_printed 0 "POLICY: crux-smoke -- the standing release policy in contracts/model-capability-ladder-v1.yaml covers 1.2.3" "$d"
    row policy_covers_engages_the_scope 0 "SCOPED: crux-smoke -- the operator emergency scope recorded for release 1.2.3" "$d"
    # policy_ladder writes the granted copy with mktemp; a failed mktemp is an unreadable policy, never "no policy".
    TMPDIR="$tmp/no-such-dir" row policy_mktemp_failed_refuses 1 "mktemp failed, so the policy ladder could not be written" "$d"
    FX_LADDER_RC=1 row policy_r7_judges_the_scope 0 "CRUX smoke satisfied at the cut" "$d"
    FX_SCOPE_RC=1 row policy_scope_red_refuses 1 "FAIL  R7 OPERATOR EMERGENCY SCOPE crux-smoke NOT satisfied" "$d"
    FX_READINESS_RC=1 row policy_r8_is_not_the_gate 0 "release-readiness was NOT the gate" "$d"
    FX_READINESS_RC=1 row policy_r8_is_not_run 0 "ok    R8 STANDING RELEASE POLICY covers 1.2.3: release-readiness was not run" "$d"
    d="$tmp/pol-rc"; build_repo "$d"; record_policy "$d" 1.2.3
    row policy_since_equal_covers 0 "POLICY: crux-smoke" "$d"
    d="$tmp/pol-later"; build_repo "$d"; record_policy "$d" 1.2.4
    FX_READINESS_RC=1 row policy_later_r8_enforced 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    FX_LADDER_RC=1 row policy_later_r7_is_matrix 1 "FAIL  R7 model matrix NOT green" "$d"
    d="$tmp/pol-dup"; build_repo "$d"; record_policy "$d" 1.2.0 1.2.3
    row policy_and_record_refuse 1 "one release takes one ruling" "$d"
    FX_READINESS_RC=1 row policy_and_record_no_scope 1 "FAIL  R8 the release-readiness wrapper exited 1" "$d"
    d="$tmp/pol-nolib"; build_repo "$d"; record_policy "$d" 1.2.0 nolib
    row policy_without_reader_refuses 1 "has no scripts/lib/release_policy.sh to read it" "$d"
    # the model matrix is still EVIDENCE on a recorded release, scope engaged by the record or by the flag
    d="$tmp/auto-scoped"
    FX_LADDER_RC=1 row auto_scope_keeps_the_matrix_as_evidence 0 "evidence FAIL  fx-rung red on lambda" "$d"
    FX_LADDER_RC=1 SCOPE=crux-smoke row scope_on_a_recorded_release_keeps_matrix 0 "evidence FAIL  fx-rung red on lambda" "$d"
    # the wrapper's own table: modes, exit mapping, the receipts-commit rule (runs wherever this selftest runs)
    if ( TMPDIR="${TMPDIR:-/tmp}" bash "$SCRIPT_DIR/release/release_readiness.sh" --selftest >/dev/null 2>&1 ); then
        printf '  ok    %-36s release_readiness.sh --selftest green\n' r8_wrapper_selftest; pass=$((pass + 1))
    else
        printf '  BROKE %-36s release_readiness.sh --selftest RED\n' r8_wrapper_selftest; fail=$((fail + 1))
    fi

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
    # bashrs SEC010: self-test fixture: $d is under this script's own mktemp -d dir.
    # bashrs disable-next-line=SEC010
    mkdir -p "$d/evidence/crux/1.2.3"; printf '{}\n' > "$d/evidence/crux/1.2.3/lambda-gpu.json"
    git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'receipts' >/dev/null
    git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD  # R4 (#4286): the release branch carries the commit
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
    # bashrs SEC010: self-test fixture: $d is under this script's own mktemp -d dir.
    # bashrs disable-next-line=SEC010
    printf '# tooling\n' > "$d/scripts/new_tool.sh"; mkdir -p "$d/contracts"; printf 'x: 1\n' > "$d/contracts/c.yaml"
    git -C "$d" add -A; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm 'tooling' >/dev/null
    git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD  # R4 (#4286): the release branch carries the commit
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    FX_EXPECT_CUT="$cut" SCOPE=crux-smoke CUT_COMMIT="$cut" row scope_tooling_after_the_cut_passes 0 "satisfied at the cut ${cut:0:12}" "$d"
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

SCOPE=""; CUT_COMMIT=""; MODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --scope) [ $# -ge 2 ] || { printf '%s: --scope needs a name\n' "$PROG" >&2; exit 2; }; SCOPE="$2"; shift 2 ;;
        --cut-commit) [ $# -ge 2 ] || { printf '%s: --cut-commit needs a sha\n' "$PROG" >&2; exit 2; }; CUT_COMMIT="$2"; shift 2 ;;
        *) [ -z "$MODE" ] || { printf '%s: unexpected argument %s\n' "$PROG" "$1" >&2; exit 2; }; MODE="$1"; shift ;;
    esac
done
[ -z "$CUT_COMMIT" ] || [ -n "$SCOPE" ] || { printf '%s: --cut-commit is only meaningful with --scope\n' "$PROG" >&2; exit 2; }
case "$MODE" in
    --selftest) selftest ;;
    --receipt-only) receipt_gate ;;
    --graph-only) graph_gate ;;
    '')         gate ;;
    -h|--help)  sed -n '2,48p' "$0" ;;
    *)          printf '%s: unknown argument %s\n' "$PROG" "$MODE" >&2; exit 2 ;;
esac
