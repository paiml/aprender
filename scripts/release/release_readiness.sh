#!/usr/bin/env bash
# release_readiness.sh — R8: the release's evidence graded by pv's `release-readiness-v1` SHACL shape
# (aprender#3715 done_when 4). ONE caller contract for both ends of the train:
#   T-1  scripts/release/autopilot.sh `models` step: the receipts it just measured at the release commit.
#   T-4  scripts/check_publish_preflight.sh R8: the receipts COMMITTED in the bump, at the tagged HEAD.
#
# WHY A WRAPPER. The shape grades absence as a violation (a missing, skipped, stale or fallback cell is a
# missing edge `minCount 1` rejects). Both call sites must ask it the same question with the same subject,
# the same receipts-commit rule and the same exit mapping — two hand-copied `pv lint` lines would drift.
#
# MODE. `enforce`, and nothing else (flipped 2026-09-28 for 0.70, #3715 B1; the report arm was deleted
#   for R10: "report-only is not an option", L19: a silent report-only check = hard stop).
#   Measured 2026-09-25 with pv 0.69.3 on 0.69.1 @d8a6df53a: rc 1, 1101 violations, 1008 of them cells with
#   no `ont:release/row`, because models_t1 measured rungs only. models_t1 now passes --cells on both hosts,
#   and the operator's de-claim of Qwen3 on gx10 (#4590) is DATA the released pv reads (rung hosts + the
#   receipt's inventory), so a Fail STOPs the train before the tag.
#   RELEASE_READINESS_MODE may only say `enforce`; `report`, as the env value OR as a committed DEFAULT_MODE,
#   or any other value, is a caller error (3). A gate an env var or a one-word edit can weaken is theater.
#   Could-not-judge (a decline (pv 2), an unknown exit, a missing pv, or a "Fail" exit carrying no Fail
#   verdict) is exit 2, a stop. A caller error (3) means this script was called wrong: a wiring defect.
#
# RECEIPTS-COMMIT (T-4). The committed receipts were measured at the binary's commit (their `apr_sha`),
# which is never the tagged HEAD: the receipts are committed on top. pv takes `--receipts-commit` on trust
# (it does not diff trees), so THIS script earns it: every host receipt names ONE full apr_sha X, and
# `git diff X <release-commit>` is empty outside evidence/. Otherwise the flag is withheld, pv grades the
# receipts against the release commit, and a stale receipt is a violation — the row says which.
#
# STANDING RELEASE POLICY (`ladder.release_policy` in contracts/model-capability-ladder-v1.yaml, read by
#   scripts/lib/release_policy.sh). For a version it covers, CRUX smoke on lambda and gx10 is the release gate
#   and every row this shape grades above it is nightly-only: no release step runs this script for that
#   version (autopilot skips its readiness step, preflight R8 is not run), so the grade is the nightly's.
#   The verdict lines say so ("nightly-only"); the exit code is unchanged, so a red row is still red and
#   opens or updates its ticket. There is no third state. A policy that cannot be read is exit 2 (not
#   judged), and a per-release emergency_scopes entry for a covered version is exit 1 (one release, one ruling).
#
# EXIT  0 Pass · 1 Fail ·
#       2 pv declined / could not judge / is missing ·
#       3 caller error. Every non-zero STOPs the caller.
#
# SEAMS (the selftest drives every row through them; production sets neither):
#   RELEASE_READINESS_PV    the pv binary (default: scripts/pv_bin.sh, built from THIS tree)
#
# USAGE
#   release_readiness.sh --version V --commit SHA40 [--root DIR] [--receipts DIR] [--dogfood-receipt F] [--out F]
#   release_readiness.sh --selftest
set -uo pipefail

PROG=${0##*/}
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
DEFAULT_MODE=enforce  # flipped for 0.70 (#3715 B1, operator 2026-09-28): models_t1 now measures --cells on both hosts
SHAPE=release-readiness-v1

caller_error() { printf 'FAIL  R8 %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# resolve_mode committed-default env-value -> prints `enforce`; 1 on anything else. There is no report mode
# (operator R10, 2026-09-28: "report-only is not an option"; L19: a silent report-only check = hard stop).
resolve_mode() {
    [ "$1" = enforce ] || return 1
    case "${2:-}" in
        ''|enforce) printf 'enforce\n' ;;
        *) return 1 ;;
    esac
}

# receipts_commit root receipts-dir release-commit -> prints X when earned; always prints a reason on fd 3
receipts_commit() {
    local root="$1" dir="$2" commit="$3" shas x
    shas="$(python3 - "$dir" <<'PY' 2>/dev/null
import glob, json, os, sys
out = set()
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
if not files:
    print("NONE"); sys.exit(0)
for f in files:
    try:
        out.add(str(json.load(open(f)).get("apr_sha", "")))
    except Exception:
        out.add("")
print("\n".join(sorted(out)))
PY
)" || shas=""
    if [ "$shas" = NONE ] || [ -z "$shas" ]; then
        echo "        receipts-commit withheld: no readable receipt in $dir" >&3; return 1
    fi
    if [ "$(printf '%s\n' "$shas" | wc -l)" -ne 1 ]; then
        echo "        receipts-commit withheld: the host receipts name different apr_sha values ($(printf '%s' "$shas" | tr '\n' ' '))" >&3; return 1
    fi
    x="$shas"
    case "$x" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) : ;;
        *) echo "        receipts-commit withheld: apr_sha '${x}' is not a full 40-hex sha" >&3; return 1 ;;
    esac
    if [ "$x" = "$commit" ]; then
        echo "        receipts measured AT the release commit (no --receipts-commit needed)" >&3; return 1
    fi
    if ! git -C "$root" cat-file -e "$x^{commit}" 2>/dev/null; then
        echo "        receipts-commit withheld: apr_sha ${x:0:12} does not resolve in this tree" >&3; return 1
    fi
    if ! git -C "$root" diff --quiet "$x" "$commit" -- . ':(exclude)evidence' 2>/dev/null; then
        echo "        receipts-commit withheld: ${x:0:12} differs from ${commit:0:12} outside evidence/ ($(git -C "$root" diff --name-only "$x" "$commit" -- . ':(exclude)evidence' | wc -l) file(s)), so its receipts are stale for this release" >&3
        return 1
    fi
    echo "        receipts-commit ${x:0:12}: equal to ${commit:0:12} outside evidence/" >&3
    printf '%s\n' "$x"
}

# summarize pv-output-file -> "N violation(s): shape=n ..." for a Fail verdict; 1 when it is not one
summarize() {
    python3 - "$1" <<'PY'
import json, sys
t = open(sys.argv[1], errors="replace").read()
try:
    j = json.loads(t[t.index("{"):t.rindex("}") + 1])
except Exception:
    sys.exit(1)
if j.get("verdict") != "Fail":
    sys.exit(1)
x = j.get("extra", {})
# the family's base shape (no ".<member>") is the cell shape: "release-readiness-v1=1008" -> cell=1008
shapes = " ".join((s.split(".", 1)[1] if "." in s.split("=")[0] else "cell=" + s.rsplit("=", 1)[-1])
                  for s in x.get("by_shape", []) if not s.endswith("=0"))
print("%s violation(s): %s" % (x.get("violations", "?"), shapes or "(no per-shape counts)"))
PY
}

judge() { # judge mode root version commit receipts dogfood out
    local mode="$1" root="$2" version="$3" commit="$4" receipts="$5" dogfood="$6" out="$7" surface="${8:-}"
    local scope="${9:-}"
    local pv pvv rc rx sum label crux cf e
    local -a args
    pv="${RELEASE_READINESS_PV:-}"
    if [ -z "$pv" ]; then
        # shellcheck source=scripts/pv_bin.sh
        if ! ( cd "$root" && . scripts/pv_bin.sh >/dev/null 2>&1 ); then
            echo "FAIL  R8 no pv built from this tree (scripts/pv_bin.sh refused): the release evidence cannot be graded"; return 2
        fi
        pv="$(cd "$root" && . scripts/pv_bin.sh >/dev/null 2>&1 && printf '%s' "$PV")"
    fi
    if [ -z "$pv" ] || [ ! -x "$pv" ]; then
        echo "FAIL  R8 pv '${pv:-<unset>}' is not executable: the release evidence cannot be graded"; return 2
    fi
    args=(lint --gate shapes --shape "$SHAPE" --release-version "$version" --release-commit "$commit")
    [ -n "$receipts" ] && args+=(--receipts "$receipts")
    [ -n "$dogfood" ] && args+=(--dogfood-receipt "$dogfood")
    [ -n "$surface" ] && args+=(--surface "$surface")
    # pv reads every entry of evidence/crux/<V>/ as a crux-inference-receipt/v1 and refuses the whole grade
    # (rc 3) on the prompt certification and its inventory, which are committed beside the receipts and
    # judged by their own gate. model_ladder_cells.py skips the same names. pv is graded on the rest, via symlinks.
    crux="$root/evidence/crux/$version"
    if compgen -G "$crux/prompt-certification*" > /dev/null; then
        cf="$(mktemp -d "${TMPDIR:-/tmp}/release-readiness-crux.XXXXXX")" || { echo "FAIL  R8 mktemp failed: the crux receipts cannot be graded"; return 2; }
        for e in "$crux"/*; do
            case "${e##*/}" in prompt-certification*) ;; *) ln -s "$e" "$cf/${e##*/}" ;; esac
        done
        args+=(--crux-receipts "$cf")
    fi
    if rx="$(receipts_commit "$root" "${receipts:-$root/evidence/dogfood/models/$version}" "$commit" 3>"$out")"; then
        args+=(--receipts-commit "$rx")
    fi
    cat -- "$out"
    pvv="$("$pv" --version 2>/dev/null | head -n 1 | tr -s ' \t' '_')"
    ( cd "$root" && "$pv" "${args[@]}" ) > "$out" 2>&1; rc=$?
    label="$SHAPE for $version at ${commit:0:12}${scope}"
    case "$rc" in
        0) echo "ok    R8 $label: Pass"
           # #3715 B1 (operator 2026-09-28): the ONE line autopilot cut_tag() requires before `git tag`.
           # Printed only for an enforced Pass, so a weakened run, a skipped step or an absent log
           # all leave the tag refused. out_sha256 is pv's verdict output, for the receipt.
           if [ "$mode" = enforce ]; then
               echo "ok    R8 #3715 ENFORCE PASS version=$version commit=$commit pv=${pvv:-unknown} out_sha256=$(sha256sum < "$out" | cut -c1-64)"
           fi
           return 0 ;;
        1)
            if ! sum="$(summarize "$out")"; then
                echo "FAIL  R8 $label: pv exited 1 with no Fail verdict in its output, so it did not judge"; tail -n 3 "$out" | sed 's/^/        /'; return 2
            fi
            echo "FAIL  R8 $label: Fail, $sum"; return 1 ;;
        2) echo "FAIL  R8 $label: pv DECLINED (rc 2), and a decline is not a pass: $(tail -n 1 "$out")"; return 2 ;;
        3) echo "FAIL  R8 $label: pv refused the call (rc 3, caller error): $(tail -n 1 "$out")"; return 3 ;;
        *) echo "FAIL  R8 $label: pv exited $rc, which is outside its 0/1/2/3 contract: $(tail -n 1 "$out")"; return 2 ;;
    esac
}

# policy_scope root version -> rc 0 and RR_SCOPE (the verdict-line suffix; empty when the policy does not
# cover version), rc 1 a per-release entry for a covered version, rc 2 not measured. RP_WHY says why.
# Called in the current shell (not in $( )) so RR_SCOPE and RP_WHY survive.
policy_scope() {
    local root="$1" version="$2" lib pf eff prc
    RR_SCOPE="" RP_WHY=""
    lib="$(dirname -- "$SCRIPT_PATH")/../lib/release_policy.sh"
    # shellcheck source=scripts/lib/release_policy.sh
    . "$lib" || { RP_WHY="cannot load $lib"; return 2; }
    pf="$(mktemp "${TMPDIR:-/tmp}/release-readiness-policy.XXXXXX")" || { RP_WHY="mktemp failed"; return 2; }
    release_policy_ladder "$root/contracts/model-capability-ladder-v1.yaml" "$version" > "$pf"; prc=$?
    eff="$(head -n 1 "$pf")"; rm -f -- "${pf:?}"
    [ "$prc" = 0 ] || return "$prc"
    if [ "$RP_APPLIES" = 1 ]; then
        # the synthesized ladder copy is the CRUX judge's input at the cut, not this script's
        case "$eff" in "${TMPDIR:-/tmp}"/ladder-policy.*) rm -f -- "${eff:?}" ;; esac
        RR_SCOPE=" (nightly-only: the standing release policy covers $version; CRUX smoke is its release gate, every row above it is graded here at night, and a red row is a ticket)"
        echo "      R8 nightly-only: the standing release policy covers $version; the rows above CRUX smoke are not release gates"
    fi
    return 0
}

main() {
    local root="" version="" commit="" receipts="" dogfood="" surface="" out="" mode tmpout rc prc
    while [ $# -gt 0 ]; do
        case "$1" in
            --root|--version|--commit|--receipts|--dogfood-receipt|--surface|--out)
                [ $# -ge 2 ] || caller_error "$1 needs a value" ;;
        esac
        case "$1" in
            --root) root="$2"; shift 2 ;;
            --version) version="$2"; shift 2 ;;
            --commit) commit="$2"; shift 2 ;;
            --receipts) receipts="$2"; shift 2 ;;
            --dogfood-receipt) dogfood="$2"; shift 2 ;;
            --surface) surface="$2"; shift 2 ;;
            --out) out="$2"; shift 2 ;;
            *) caller_error "unknown argument $1" ;;
        esac
    done
    [ -n "$version" ] || caller_error "--version is required"
    [ -n "$commit" ] || caller_error "--commit is required"
    mode="$(resolve_mode "$DEFAULT_MODE" "${RELEASE_READINESS_MODE:-}")" \
        || caller_error "RELEASE_READINESS_MODE='${RELEASE_READINESS_MODE:-}' / DEFAULT_MODE='$DEFAULT_MODE': the only mode is enforce; there is no report mode (may only strengthen)"
    [ -n "$root" ] || root="$(cd -- "$(dirname -- "$SCRIPT_PATH")/../.." && pwd)"
    git -C "$root" rev-parse --verify --quiet HEAD >/dev/null || { echo "FAIL  R8 $root is not a git repository"; exit 2; }
    # The standing release policy: read before pv, so an unreadable policy is never graded as "no policy".
    policy_scope "$root" "$version"; prc=$?
    case "$prc" in
        0) ;;
        1) echo "FAIL  R8 the standing release policy refuses $version: $RP_WHY"; exit 1 ;;
        *) echo "FAIL  R8 the standing release policy cannot be judged for $version (not measured): $RP_WHY"; exit 2 ;;
    esac
    [ -z "$dogfood" ] || [ -f "$dogfood" ] || caller_error "--dogfood-receipt $dogfood does not exist"
    # #3745: pv derives the release cells from the candidate's `apr surface --json` -- no surface, no cells,
    # no pass. T-1 passes the one it just took from the release-built apr; T-4 reads the committed copy.
    # It lives outside the receipts dir, because pv reads every *.json there as a host receipt.
    [ -n "$surface" ] || surface="$root/evidence/release/surface/$version.json"
    [ -f "$surface" ] || surface=""
    if [ -n "$out" ]; then tmpout="$out"; else tmpout="$(mktemp)"; fi
    judge "$mode" "$root" "$version" "$commit" "$receipts" "$dogfood" "$tmpout" "$surface" "$RR_SCOPE"; rc=$?
    [ -n "$out" ] || rm -f -- "$tmpout"
    exit "$rc"
}

# --------------------------------------------------------------- selftest ---
selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) [ -d "$tmp" ] && rm -rf -- "$tmp" ;; esac; }
selftest() {
    local tmp pass=0 fail=0 d x c
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. An empty or '/' value must never reach it. $tmp is this
    # function's own mktemp -d (checked above); selftest_cleanup sees it through bash's dynamic scope.
    trap selftest_cleanup RETURN
    # the stub pv: records its argv, answers FX_PV_RC with FX_PV_BODY
    cat > "$tmp/pv" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = --version ] && { echo "pv 9.9.9-stub"; exit 0; }
printf '%s\n' "$*" > "${FX_ARGS:?}"
case "${FX_PV_BODY:-fail}" in
    fail) printf '{"gate":"shapes","verdict":"Fail","extra":{"violations":3,"by_shape":["release-readiness-v1.kernel=0","release-readiness-v1=3"]}}\nreject: lint failed\n' ;;
    pass) printf '{"gate":"shapes","verdict":"Pass","extra":{"violations":0,"by_shape":[]}}\n' ;;
    junk) printf 'thread main panicked\n' ;;
esac
exit "${FX_PV_RC:-1}"
STUB
    chmod +x "$tmp/pv"
    g() { git -C "$1" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t -c commit.gpgsign=false "${@:2}"; }
    mk() { # dir [subdir]: a repo with src/, the ladder ($MK_LADDER; default: no policy) (and an empty subdir) at
           # commit A, receipts naming A committed on top (HEAD = B)
        mkdir -p "$1/src" "$1/contracts" "$1/evidence/dogfood/models/1.2.3" "$1/${2:-src}"; printf 'a\n' > "$1/src/f"
        printf '%s' "${MK_LADDER:-$'ladder:\n  emergency_scopes:\n'}" > "$1/contracts/model-capability-ladder-v1.yaml"
        git init -q -b main "$1"; g "$1" add -A; g "$1" commit -qm A
        local a; a="$(git -C "$1" rev-parse HEAD)"
        printf '{"apr_sha":"%s"}\n' "$a" > "$1/evidence/dogfood/models/1.2.3/lambda.json"
        printf '{"apr_sha":"%s"}\n' "$a" > "$1/evidence/dogfood/models/1.2.3/gx10.json"
        g "$1" add -A; g "$1" commit -qm receipts
    }
    row() { # name expect-rc needle dir [env...] -- runs main in a subshell
        local name="$1" expect="$2" needle="$3" dir="$4" o rc=0; shift 4
        o="$( env TMPDIR="$tmp" FX_ARGS="$tmp/args" RELEASE_READINESS_PV="$tmp/pv" "$@" bash "$SCRIPT_PATH" --root "$dir" --version 1.2.3 --commit "$(git -C "$dir" rev-parse HEAD)" 2>&1 )" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-44s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$needle" in
            !*) case "$o" in
                    *"${needle#!}"*) printf '  BROKE %-44s exit %s but said %s\n%s\n' "$name" "$rc" "${needle#!}" "$o"; fail=$((fail + 1)) ;;
                    *) printf '  ok    %-44s exit=%s never said %s\n' "$name" "$rc" "${needle#!}"; pass=$((pass + 1)) ;;
                esac
                return 0 ;;
        esac
        case "$o" in
            *"$needle"*) printf '  ok    %-44s exit=%s said %s\n' "$name" "$rc" "$needle"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-44s exit %s but never said %s\n%s\n' "$name" "$rc" "$needle" "$o"; fail=$((fail + 1)) ;;
        esac
    }
    argrow() { # name needle(present|!absent)
        local name="$1" want="$2"
        case "$want" in
            !*) if grep -qF -- "${want#!}" "$tmp/args"; then printf '  BROKE %-44s pv was called with %s\n' "$name" "${want#!}"; fail=$((fail + 1)); else printf '  ok    %-44s pv not called with %s\n' "$name" "${want#!}"; pass=$((pass + 1)); fi ;;
            *) if grep -qF -- "$want" "$tmp/args"; then printf '  ok    %-44s pv called with %s\n' "$name" "$want"; pass=$((pass + 1)); else printf '  BROKE %-44s pv never called with %s: %s\n' "$name" "$want" "$(cat "$tmp/args")"; fail=$((fail + 1)); fi ;;
        esac
    }

    d="$tmp/r"; mk "$d"; x="$(git -C "$d" rev-parse HEAD~1)"; c="$(git -C "$d" rev-parse HEAD)"
    row pass_is_ok                               0 "ok    R8 release-readiness-v1 for 1.2.3" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    row pass_prints_the_enforce_line             0 "ok    R8 #3715 ENFORCE PASS version=1.2.3 commit=$(git -C "$d" rev-parse HEAD) pv=pv_9.9.9-stub out_sha256=" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    argrow pass_asks_the_shape                   "--gate shapes --shape release-readiness-v1 --release-version 1.2.3 --release-commit $c"
    argrow receipts_commit_earned_is_passed      "--receipts-commit $x"
    argrow no_surface_is_not_invented            "!--surface"
    d="$tmp/v"; mk "$d" evidence/release/surface; printf '{}\n' > "$d/evidence/release/surface/1.2.3.json"
    row committed_surface_runs                   0 "ok    R8 release-readiness-v1 for 1.2.3" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    argrow committed_surface_is_passed           "--surface $d/evidence/release/surface/1.2.3.json"
    argrow no_certification_keeps_pv_crux_default "!--crux-receipts"
    d="$tmp/c"; mk "$d" evidence/crux/1.2.3
    printf '{}\n' > "$d/evidence/crux/1.2.3/r.json"; printf '[]\n' > "$d/evidence/crux/1.2.3/prompt-certification-inventory.json"
    row certification_beside_crux_runs           0 "ok    R8 release-readiness-v1 for 1.2.3" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    cx="$(sed -n 's/.*--crux-receipts \([^ ]*\).*/\1/p' "$tmp/args")"
    if [ -n "$cx" ] && [ -e "$cx/r.json" ] && ! compgen -G "$cx/prompt-certification*" > /dev/null; then
        printf '  ok    %-44s pv graded r.json without the certification\n' certification_is_not_a_crux_receipt; pass=$((pass + 1))
    else
        printf '  BROKE %-44s crux dir %s: %s\n' certification_is_not_a_crux_receipt "${cx:-<not passed>}" "$(ls "${cx:-/nonexistent}" 2>&1 | tr '\n' ' ')"; fail=$((fail + 1))
    fi
    d="$tmp/r"
    row bare_fail_refuses                        1 "FAIL  R8 release-readiness-v1 for 1.2.3" "$d" FX_PV_RC=1
    row bare_fail_names_the_count                1 "3 violation(s): cell=3" "$d" FX_PV_RC=1
    row fail_under_enforce_refuses               1 "FAIL  R8 release-readiness-v1 for 1.2.3" "$d" FX_PV_RC=1 RELEASE_READINESS_MODE=enforce
    row bare_decline_refuses                     2 "pv DECLINED" "$d" FX_PV_RC=2 FX_PV_BODY=junk
    row decline_under_enforce_refuses            2 "pv DECLINED" "$d" FX_PV_RC=2 FX_PV_BODY=junk RELEASE_READINESS_MODE=enforce
    row caller_error_is_never_downgraded         3 "caller error" "$d" FX_PV_RC=3 FX_PV_BODY=junk
    row exit1_without_a_fail_verdict_declines    2 "no Fail verdict" "$d" FX_PV_RC=1 FX_PV_BODY=junk RELEASE_READINESS_MODE=enforce
    row exit_outside_contract_declines           2 "outside its 0/1/2/3 contract" "$d" FX_PV_RC=101 FX_PV_BODY=junk RELEASE_READINESS_MODE=enforce
    row missing_pv_declines                      2 "is not executable" "$d" RELEASE_READINESS_PV="$tmp/nope" RELEASE_READINESS_MODE=enforce
    row bare_missing_pv_refuses                  2 "is not executable" "$d" RELEASE_READINESS_PV="$tmp/nope"
    row bare_no_verdict_refuses                  2 "no Fail verdict" "$d" FX_PV_RC=1 FX_PV_BODY=junk
    row bare_outside_contract_refuses            2 "outside its 0/1/2/3 contract" "$d" FX_PV_RC=101 FX_PV_BODY=junk
    row env_may_not_weaken_to_report             3 "may only strengthen" "$d" RELEASE_READINESS_MODE=report FX_PV_RC=1
    row env_may_not_weaken_to_bogus              3 "may only strengthen" "$d" RELEASE_READINESS_MODE=off
    # a source change between the receipts' commit and the release commit: the flag is WITHHELD
    d="$tmp/s"; mk "$d"; printf 'b\n' > "$d/src/f"; g "$d" commit -qam src
    row source_drift_withholds_receipts_commit   0 "differs from" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    argrow source_drift_not_passed               "!--receipts-commit"
    # two hosts measured at different commits: withheld
    d="$tmp/m"; mk "$d"; printf '{"apr_sha":"%s"}\n' "$(printf '0%.0s' $(seq 40))" > "$d/evidence/dogfood/models/1.2.3/gx10.json"; g "$d" commit -qam split
    row split_apr_sha_withholds                  0 "different apr_sha values" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    argrow split_apr_sha_not_passed              "!--receipts-commit"
    # the standing release policy, one row per state. pol dir since [extra-ladder-lines]
    pol() {
        local l
        printf -v l 'ladder:\n  release_policy:\n    name: crux-smoke\n    since: "%s"\n    date: "2026-10-07"\n    quote: '"'"'"q"'"'"'\n    hosts: [lambda, gx10]\n    thinking: ["off", "on"]\n    larger_rows: nightly\n    red_row_needs: ticket\n    release_notes: known_failures\n%s  emergency_scopes:\n%s' \
            "$2" "${4:-}" "${3:-}"
        MK_LADDER="$l" mk "$1"
    }
    d="$tmp/pc"; pol "$d" 1.0.0
    row policy_covered_pass_is_nightly_only      0 "Pass" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    row policy_covered_marks_the_rows            0 "(nightly-only: the standing release policy covers 1.2.3" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    row policy_covered_keeps_the_enforce_line    0 "ok    R8 #3715 ENFORCE PASS version=1.2.3 commit=$(git -C "$d" rev-parse HEAD) pv=" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    row policy_covered_fail_stays_red            1 "3 violation(s): cell=3" "$d" FX_PV_RC=1
    row policy_covered_fail_is_marked            1 "(nightly-only: the standing release policy covers 1.2.3" "$d" FX_PV_RC=1
    row policy_covered_decline_stays_a_stop      2 "pv DECLINED" "$d" FX_PV_RC=2 FX_PV_BODY=junk
    row policy_covered_never_warns               0 "!WARN" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    if compgen -G "$tmp/ladder-policy.*" > /dev/null; then
        printf '  BROKE %-44s %s\n' policy_copy_is_removed "$(ls "$tmp"/ladder-policy.* | tr '\n' ' ')"; fail=$((fail + 1))
    else
        printf '  ok    %-44s no synthesized ladder left behind\n' policy_copy_is_removed; pass=$((pass + 1))
    fi
    d="$tmp/pn"; pol "$d" 2.0.0
    row policy_not_covering_is_not_marked        0 "!nightly-only" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    row policy_not_covering_fail_is_unmarked     1 "!nightly-only" "$d" FX_PV_RC=1
    d="$tmp/r"
    row no_policy_block_is_not_marked            0 "!nightly-only" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    d="$tmp/pe"; pol "$d" 1.0.0 '    - name: crux-smoke
      release: "1.2.3"
'
    row policy_per_release_entry_refuses         1 "the standing release policy refuses 1.2.3" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    d="$tmp/pu"; pol "$d" 1.0.0 '' '    bogus_key: x
'
    row policy_unreadable_is_not_measured        2 "cannot be judged for 1.2.3 (not measured)" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    d="$tmp/pl"; mk "$d"; g "$d" rm -q contracts/model-capability-ladder-v1.yaml; g "$d" commit -qm noladder
    row policy_without_a_ladder_is_not_measured  2 "cannot read the ladder" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    # the committed default is what a bare run uses; flipping it is a reviewed commit
    if [ "$DEFAULT_MODE" = enforce ]; then
        printf '  ok    %-44s DEFAULT_MODE=enforce\n' committed_default_is_enforce; pass=$((pass + 1))
    else
        printf '  BROKE %-44s DEFAULT_MODE=%s\n' committed_default_is_enforce "$DEFAULT_MODE"; fail=$((fail + 1))
    fi
    if [ "$(resolve_mode enforce '')" = enforce ] && [ "$(resolve_mode enforce enforce)" = enforce ] \
        && ! resolve_mode enforce report >/dev/null && ! resolve_mode report '' >/dev/null && ! resolve_mode report enforce >/dev/null; then
        printf '  ok    %-44s report is refused as env AND as a committed default\n' mode_has_no_report_arm; pass=$((pass + 1))
    else
        printf '  BROKE %-44s\n' mode_has_no_report_arm; fail=$((fail + 1))
    fi
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --selftest) selftest ;;
    -h|--help) sed -n '2,45p' "$0" ;;
    *) main "$@" ;;
esac
