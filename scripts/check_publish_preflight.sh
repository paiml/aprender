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
#   R4  HEAD is an ancestor of release/<version> (origin/release/X.Y.Z): nothing
#       publishes from a topic branch. NOT main (cop ruling 2026-09-24, #4286): RC
#       binaries ship from the release branch before merge-back, and main ancestry
#       is enforced at merge-back (#4224). A missing release ref refuses.
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
#       (no git/cargo/jq/iconv, not a repository). 2 is not a pass.
#
# SEAMS (the selftest builds a throwaway repository and drives every rule to
# both verdicts through them; production never sets them):
#   PUBLISH_PREFLIGHT_ROOT         repository root (default: this script's repo)
#   PUBLISH_PREFLIGHT_RELEASE_REF  the ref for R4 (default: origin/release/<R2 version>;
#                                  the selftest leaves it unset so the derivation is tested)
#   PUBLISH_PREFLIGHT_RECEIPT_DIR  the dogfood receipt dir (default: $ROOT/.dogfood)
#   PUBLISH_PREFLIGHT_LADDER_JUDGE the R7 judge (default: $ROOT/scripts/check_model_ladder.sh)
#
# USAGE
#   bash scripts/check_publish_preflight.sh             # the gate
#   bash scripts/check_publish_preflight.sh --selftest  # case table, both polarities
#   bash scripts/check_publish_preflight.sh --receipt-only  # R2+R5 only: T-1, before the tag (#3708)
#   bash scripts/check_publish_preflight.sh --graph-only    # R2+R6 only: the rc cut (#4287)
#   bash scripts/check_publish_preflight.sh --scope crux-smoke [--cut-commit SHA]
#       R7 under a RECORDED operator emergency scope (contracts/model-capability-ladder-v1.yaml
#       `ladder.emergency_scopes`; 0.69.1 only): the judge's own `--scope` path
#       (scripts/lib/crux_smoke_scope.py) decides R7 from CRUX smoke receipts bound to the CUT --
#       the commit the release binary was built from -- instead of the model matrix. The cut
#       defaults to HEAD; when HEAD is not the cut (receipts committed on top, or main's squash
#       of it), every PUBLISHED path -- crates/ src/ Cargo.toml Cargo.lock, R4's set -- must be
#       equal to the cut's, or the published source is not the smoked binary's. The
#       model-matrix rows are still printed, as EVIDENCE, never as the verdict.
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

# ---- JSON reads, in jq (#4352: python3 is out of the release path) ----
# Each keeps the python it replaced, case for case (parity table in #4352), with one deliberate
# difference: where python parsed the JSON and then crashed on its shape (a receipt that is not an
# object, a package without a name, "dependencies": null), its empty stdout read as a verdict --
# for R6 as "ok, no sibling dev-dependency". Here any such document is UNREADABLE, and refused.
# One cosmetic difference: a list or object in a receipt field prints as JSON (["x"]), not python's
# repr (['x']). It only reaches a FAIL message; no such value is ever a GO, a sha or a version.
PP_JQ='def obj: if type == "object" then . else error("not an object") end;
def key($k): obj | if has($k) then .[$k] else error("missing key \($k)") end;
def pystr: if . == null then "None" elif . == true then "True" elif . == false then "False"
    elif type == "string" then .
    elif type == "number" and isnan then "nan"
    elif type == "number" and isinfinite then (if . > 0 then "inf" else "-inf" end)
    else tojson end;
def truthy: . != null and . != false and . != 0 and . != "" and . != [] and . != {};
def str: if type == "string" then . else error("not a string") end;
def pyiter: if type == "array" then .[] elif type == "object" then keys_unsorted[]
    elif type == "string" then explode[] | [.] | implode else error("not iterable") end;
def one: input as $d | if ([inputs] | length) > 0 then error("extra data") else $d end;'
# pp_manifest_rows: stdin `cargo metadata` -> one `manifest<US>1<US>version` row per package, in
# order (`<US>0<US>` when it has no version: python's KeyError fires only on the match).
pp_manifest_rows() {
    jq -rn "$PP_JQ"' one | obj | (if has("packages") then .packages else [] end)
        | [pyiter] | .[] | "\(key("manifest_path") | str)\u001f\(if has("version") then "1\u001f\(.version | pystr)" else "0\u001f" end)"'
}
pp_realpath() { realpath -m -- "$1" 2> /dev/null || readlink -f -- "$1"; }  # os.path.realpath
# pp_receipt_fields FILE -> `verdict commit version phase deferred open` (a falsy field is "-", phase
# "full"; deferred and open_obligations comma-joined, #3957), or `UNREADABLE - - - - -` for anything python could not read
# (missing, not UTF-8, not JSON) and anything it read and then crashed on (not an object).
pp_receipt_fields() {
    # python's open(encoding="utf-8") refuses a BOM that jq would skip: refuse it here too
    { iconv -f UTF-8 -t UTF-8 < "$1" > /dev/null 2>&1 \
        && [ "$(head -c 3 "$1" | od -An -tx1 | tr -d ' \n')" != efbbbf ] \
        && jq -rn "$PP_JQ"' one | obj
            | def f($k; $d): if has($k) and (.[$k] | truthy) then .[$k] | pystr else $d end;
            def l($k): (if has($k) and (.[$k] | truthy) then .[$k] else [] end)
                | [pyiter | pystr] | join(",") | if . == "" then "-" else . end;
            [f("verdict"; "-"), f("commit"; "-"), f("version"; "-"), f("phase"; "full"),
             l("deferred"), l("open_obligations")] | join(" ")' < "$1" 2> /dev/null
    } || echo "UNREADABLE - - - - -"
}
# pp_devdep_edges: stdin `cargo metadata` -> `CYCLE|ACYCLIC <crate> -> <sibling> <req>` for every
# VERSIONED dev-dependency between publishable siblings, sorted; CYCLE when the sibling can reach
# the crate back over normal, build and versioned dev edges. UNREADABLE for bad metadata.
read -r -d '' PP_R6_JQ <<'JQ' || :
        def reaches($s; $d; $E): {seen: {}, stack: [$s], found: false}
          | until(.found or (.stack | length) == 0;
              .stack[-1] as $x | .stack |= .[:-1]
              | reduce ($E[$x] | keys_unsorted[]) as $y (.;
                  if .found then . elif $y == $d then .found = true
                  elif .seen[$y] then . else .seen[$y] = true | .stack += [$y] end))
          | .found;
        one | obj
        | (reduce ((if has("packages") then .packages else [] end) | [pyiter] | .[] | obj
                   | select((if has("publish") then .publish else null end) != [])) as $p
             ({}; .[$p | key("name") | str] = $p)) as $pk
        | (reduce ($pk | to_entries[]) as $e ({edges: ($pk | map_values({})), vdev: []};
             reduce ($e.value | if has("dependencies") then .dependencies else [] end | [pyiter] | .[] | obj) as $dep (.;
               ($dep | if has("name") then .name else null end) as $t
               | ($dep | if has("kind") then .kind else null end) as $kind
               | if ($t | type) != "string" or ($pk | has($t) | not) or $t == $e.key then .
                 elif $kind == "dev" and (($dep.req | if truthy then . else "*" end) == "*") then .
                 else (if $kind == "dev" then .vdev += [[$e.key, $t, $dep.req]] else . end)
                      | .edges[$e.key][$t] = true
                 end))) as $g
        | $g.vdev | sort | .[]
        | "\(if reaches(.[1]; .[0]; $g.edges) then "CYCLE" else "ACYCLIC" end) \(.[0]) -> \(.[1]) \(.[2] | pystr)"
JQ
pp_devdep_edges() {
    jq -rn "$PP_JQ$PP_R6_JQ" 2> /dev/null || echo UNREADABLE
}
# ---- end JSON reads ----

root_version() { # root -> the root manifest's package version, from cargo metadata
    local root="$1" want mp have v
    want="$(pp_realpath "$root/Cargo.toml")"
    while IFS=$'\x1f' read -r mp have v; do
        [ "$mp" = "$want" ] || [ "$(pp_realpath "$mp")" = "$want" ] || continue
        [ "$have" = 1 ] || return 1
        printf '%s\n' "$v"; return 0
    done < <(cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null | pp_manifest_rows)
    return 1   # cargo metadata printed nothing, or no package is the root manifest
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
        read -r verdict rcommit rversion rphase rdeferred ropen < <(pp_receipt_fields "$receipt")
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
    r6="$(cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null | pp_devdep_edges)"
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

gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" release_ref
    local fails=0 status version tags head
    for t in git cargo jq iconv; do
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

    # R4 HEAD is on the release branch of THIS version (#4286), not main: main is
    # merge-back's check (#4224). No version, no release ref to judge: refuse.
    release_ref="${PUBLISH_PREFLIGHT_RELEASE_REF:-origin/release/${version:-?}}"
    if [ -n "$version" ] \
       && git -C "$root" rev-parse --verify --quiet "${release_ref}^{commit}" >/dev/null \
       && git -C "$root" merge-base --is-ancestor "$head" "$release_ref" 2>/dev/null; then
        echo "ok    R4 HEAD is an ancestor of $release_ref"
    else
        echo "FAIL  R4 HEAD ${head:0:9} is not an ancestor of $release_ref (or that ref does not exist)"
        fails=1
    fi

    # R5 dogfood receipt: GO, this commit, this version
    rule_r5 "$root" "$head" "$version" || fails=1

    rule_r6 "$root" || fails=1

    # R7 the model matrix, re-read at T-4 through the T-1 judge (#3717)
    rule_r7 "$root" "$version" || fails=1

    if [ "$fails" -ne 0 ]; then
        echo "REFUSE $PROG: publishing is not allowed from this tree (see the FAIL rows)."
        return 1
    fi
    if [ -n "${SCOPE:-}" ]; then
        echo "PASS  $PROG: clean, versioned, tagged, on $release_ref, dogfood GO, OPERATOR EMERGENCY SCOPE $SCOPE satisfied (the model matrix was NOT the gate)"
    else
        echo "PASS  $PROG: clean, versioned, tagged, on $release_ref, dogfood GO, model matrix green"
    fi
    return 0
}

# --graph-only (#4287): R2 + R6 on PUBLISH_PREFLIGHT_ROOT, the rc cut's end of the
# publish graph. R1/R3/R4/R5/R7 describe the upload (a tag, the release branch, receipts) and are
# judged at T-4 as before.
graph_gate() {
    local root="${PUBLISH_PREFLIGHT_ROOT:-}" version
    for t in cargo jq; do
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
    for t in git cargo jq iconv; do
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

# ------------------------------------------------------------- JSON-read rows ---
# pp_io_rows (#4352): each jq read above against the output the python it replaced gave on the
# same input (the full parity table is in #4352; these are the rows that pin each rule). Then,
# unless PP_NO_MUTANTS=1, every mutant below is applied to a copy of THIS file's JSON-read region
# and must turn at least one row red -- a row table no mutant can fail proves nothing.
pp_io_rows() {
    local t pass=0 fail=0 m anchor repl src head tail cp rc
    t="$(mktemp -d)" || die_env "mktemp failed"
    case "$t" in /tmp/?*|/var/folders/?*|/mnt/?*) : ;; *) die_env "mktemp gave ${t:-<empty>}" ;; esac
    io() { # want label cmd... (stdout compared exactly)
        local want="$1" label="$2" got; shift 2
        got="$("$@" 2> /dev/null)"
        if [ "$got" = "$want" ]; then pass=$((pass + 1)); printf 'ok    io %s\n' "$label"
        else fail=$((fail + 1)); printf 'FAIL  io %s: want [%s] got [%s]\n' "$label" "$want" "$got"; fi
    }
    rec() { printf '%s' "$2" > "$t/$1.json"; }
    rec plain '{"verdict":"GO","commit":"abc","version":"1.2.3","phase":"pre-publish","deferred":["publish-dry-run"]}'
    rec falsy '{"verdict":0,"commit":"","version":null,"phase":"","deferred":[]}'
    rec pyish '{"verdict":true,"commit":false,"version":NaN,"deferred":[null,true,"x"]}'
    rec strdefer '{"verdict":"GO","deferred":"ab"}'
    rec open '{"verdict":"GO","phase":"pre-publish","deferred":[],"open_obligations":["x","y"]}'
    rec notobj '[1]'
    rec twodocs '{"verdict":"GO"}{"verdict":"GO"}'
    printf '\357\273\277{"verdict":"GO"}' > "$t/bom.json"
    printf '{"verdict":"G\377O"}' > "$t/badutf8.json"
    io "GO abc 1.2.3 pre-publish publish-dry-run -" receipt_plain        pp_receipt_fields "$t/plain.json"
    io "- - - full - -"                 receipt_falsy_is_dash_phase_full pp_receipt_fields "$t/falsy.json"
    io "True - nan full None,True,x -"  receipt_python_str_of_values   pp_receipt_fields "$t/pyish.json"
    io "GO - - full a,b -"              receipt_string_deferral_iterates pp_receipt_fields "$t/strdefer.json"
    io "GO - - pre-publish - x,y"       receipt_open_obligations_joined pp_receipt_fields "$t/open.json"
    io "UNREADABLE - - - - -"           receipt_not_object_unreadable  pp_receipt_fields "$t/notobj.json"
    io "UNREADABLE - - - - -"           receipt_extra_data_unreadable  pp_receipt_fields "$t/twodocs.json"
    io "UNREADABLE - - - - -"           receipt_bom_unreadable         pp_receipt_fields "$t/bom.json"
    io "UNREADABLE - - - - -"           receipt_not_utf8_unreadable    pp_receipt_fields "$t/badutf8.json"
    io "UNREADABLE - - - - -"           receipt_missing_unreadable     pp_receipt_fields "$t/absent.json"
    printf '%s' '{"packages":[{"name":"a","publish":null,"dependencies":[{"name":"b","req":"^1","kind":"dev"},{"name":"c","kind":"dev"},{"name":"c","req":"*","kind":"dev"},{"name":"a","req":"^1","kind":"dev"}]},{"name":"b","dependencies":[{"name":"a","req":"^1","kind":null}]},{"name":"c","dependencies":[]},{"name":"d","dependencies":[{"name":"c","req":"=2","kind":"dev"},{"name":"x","req":"^1","kind":"dev"}]},{"name":"x","publish":[],"dependencies":[{"name":"d","req":"^1"}]}]}' > "$t/r6.json"
    io $'CYCLE a -> b ^1\nACYCLIC d -> c =2' r6_versioned_dev_edges_classified pp_devdep_edges < "$t/r6.json"
    io "UNREADABLE"                     r6_bad_metadata_unreadable     pp_devdep_edges < "$t/notobj.json"
    printf '%s' '{"packages":[{"manifest_path":"/m/a","version":"1.2.3"},{"manifest_path":"/m/b"}]}' > "$t/mf.json"
    printf '%s' '{"packages":[{"manifest_path":7,"version":"1"}]}' > "$t/mf7.json"
    io ""                               manifest_path_not_string_refused pp_manifest_rows < "$t/mf7.json"
    io $'/m/a\x1f1\x1f1.2.3\n/m/b\x1f0\x1f' manifest_rows_mark_missing_version pp_manifest_rows < "$t/mf.json"
    if [ "${PP_NO_MUTANTS:-}" != 1 ]; then
        src="$(cat "$0")"; head="${src%%"# ---- end JSON reads ----"*}"; tail="${src#"$head"}"
        while IFS='~' read -r anchor repl; do
            [ -n "$anchor" ] || continue
            m="${head#*"$anchor"}"
            if [ "$m" = "$head" ] || [ "${m#*"$anchor"}" != "$m" ]; then
                fail=$((fail + 1)); printf 'FAIL  mutant anchor absent or not unique: %s\n' "$anchor"; continue
            fi
            cp="$t/mutant.sh"; printf '%s%s%s%s' "${head%%"$anchor"*}" "$repl" "$m" "$tail" > "$cp"
            PP_NO_MUTANTS=1 bash "$cp" --io-selftest > "$t/mutant.out" 2>&1; rc=$?
            if [ "$rc" -ne 0 ] && grep -q '^FAIL  io ' "$t/mutant.out"; then
                pass=$((pass + 1)); printf 'ok    mutant killed: %s\n' "$anchor"
            else fail=$((fail + 1)); printf 'FAIL  mutant SURVIVED: %s\n' "$anchor"; fi
        done <<'MUTANTS'
!= efbbbf ]~!= 000000 ]
iconv -f UTF-8 -t UTF-8 <~cat <
 and . != 0 and ~ and 
elif type == "number" and isnan then "nan"~elif false then "nan"
then error("extra data")~then $d
error("not a string")~.
if reaches(.[1]; .[0]; $g.edges) then "CYCLE"~if false then "CYCLE"
elif $kind == "dev" and~elif false and
"$PP_JQ$PP_R6_JQ" 2> /dev/null || echo UNREADABLE~"$PP_JQ$PP_R6_JQ" 2> /dev/null || :
else "0\u001f" end~else "1\u001f" end
MUTANTS
    fi
    if [ -n "$t" ] && [ "$t" != "/" ]; then
        case "$t" in /tmp/?*|/var/folders/?*|/mnt/?*) rm -rf -- "$t" || return 2 ;; esac
    fi
    printf -- '--- io %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
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

    d="$tmp/branch"; build_repo "$d"; git -C "$d" checkout -q -b topic
    printf 'pub fn k() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'topic' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    row head_off_release_branch_refuses 1 "FAIL  R4 HEAD" "$d"

    # R4 (#4286, cop ruling 2026-09-24): the release branch, not main. An rc commit on
    # release/1.2.3 that main does not contain yet passes; main containing HEAD does not
    # rescue a missing release ref; another version's release branch does not count.
    d="$tmp/rc-on-release"; build_repo "$d"; git -C "$d" checkout -q -b release-1.2.3
    printf 'pub fn r() {}\n' >> "$d/src/lib.rs"; git -C "$d" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qam 'rc fix' >/dev/null
    git -C "$d" tag -f v1.2.3 >/dev/null; git -C "$d" update-ref refs/remotes/origin/release/1.2.3 HEAD
    write_receipt "$d" GO "$(git -C "$d" rev-parse HEAD)" 1.2.3
    ! git -C "$d" merge-base --is-ancestor HEAD fixture-main || { echo "  BROKE fixture: rc commit is on main"; fail=$((fail + 1)); }
    row rc_on_release_not_main_passes  0 "ok    R4 HEAD is an ancestor of origin/release/1.2.3" "$d"
    d="$tmp/no-release-ref"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
    row release_ref_absent_on_main_refuses 1 "FAIL  R4" "$d"
    d="$tmp/other-release"; build_repo "$d"; git -C "$d" update-ref -d refs/remotes/origin/release/1.2.3
    git -C "$d" update-ref refs/remotes/origin/release/1.2.4 HEAD
    row other_versions_release_refuses 1 "not an ancestor of origin/release/1.2.3" "$d"

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

    pp_io_rows || fail=$((fail + 1))

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
    --io-selftest) pp_io_rows ;;
    --receipt-only) receipt_gate ;;
    --graph-only) graph_gate ;;
    '')         gate ;;
    -h|--help)  sed -n '2,48p' "$0" ;;
    *)          printf '%s: unknown argument %s\n' "$PROG" "$MODE" >&2; exit 2 ;;
esac
