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
# MODE. `report` (the committed default below) or `enforce`.
#   Measured 2026-09-25 with pv 0.69.3 on 0.69.1 @d8a6df53a: rc 1, 1101 violations, 1008 of them cells with
#   no `ont:release/row` — the #3712 cells[] producer does not exist yet, so under `enforce` NO release can
#   pass. Until #3712 lands, a Fail verdict is PRINTED IN FULL as a WARN row and the train continues; the
#   flip to `enforce` is a one-line reviewed commit to DEFAULT_MODE, never an environment variable.
#   RELEASE_READINESS_MODE may only STRENGTHEN the committed mode (`enforce`); asking for `report` over an
#   `enforce` default, or any other value, is a caller error (3). A gate an env var can weaken is theater.
#   Could-not-judge under `report` (cop ruling 2026-09-25 09:15Z: "record the rc and receipt, don't
#   stop"): a decline (pv 2), an unknown exit, a missing pv, or a "Fail" exit carrying no Fail verdict is
#   printed as its FAIL row PLUS a `WARN  R8 REPORT-ONLY could not judge (rc 2)` row, and exits 0. The
#   rc is on the page, not hidden. Under `enforce` each is exit 2. A caller error (3) stops in either
#   mode: it means this script was called wrong, and that is a wiring defect, not a verdict.
#
# RECEIPTS-COMMIT (T-4). The committed receipts were measured at the binary's commit (their `apr_sha`),
# which is never the tagged HEAD: the receipts are committed on top. pv takes `--receipts-commit` on trust
# (it does not diff trees), so THIS script earns it: every host receipt names ONE full apr_sha X, and
# `git diff X <release-commit>` is empty outside evidence/. Otherwise the flag is withheld, pv grades the
# receipts against the release commit, and a stale receipt is a violation — the row says which.
#
# EXIT  0 Pass, or a Fail verdict under `report` (WARN row) · 1 Fail under `enforce` ·
#       2 pv declined / could not judge / is missing, under `enforce` (a WARN + 0 under `report`) ·
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
DEFAULT_MODE=report   # flip to `enforce` when #3712's cells[] producer lands (aprender#3715 done_when 5b)
SHAPE=release-readiness-v1

caller_error() { printf 'FAIL  R8 %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# resolve_mode committed-default env-value -> prints the mode; 1 on a value that would weaken or is unknown
resolve_mode() {
    case "${2:-}" in
        '') printf '%s\n' "$1" ;;
        enforce) printf 'enforce\n' ;;
        report) [ "$1" = report ] || return 1; printf 'report\n' ;;
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
    local mode="$1" root="$2" version="$3" commit="$4" receipts="$5" dogfood="$6" out="$7"
    local pv rc rx sum label
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
    if rx="$(receipts_commit "$root" "${receipts:-$root/evidence/dogfood/models/$version}" "$commit" 3>"$out")"; then
        args+=(--receipts-commit "$rx")
    fi
    cat -- "$out"
    ( cd "$root" && "$pv" "${args[@]}" ) > "$out" 2>&1; rc=$?
    label="$SHAPE for $version at ${commit:0:12}"
    case "$rc" in
        0) echo "ok    R8 $label: Pass"; return 0 ;;
        1)
            if ! sum="$(summarize "$out")"; then
                echo "FAIL  R8 $label: pv exited 1 with no Fail verdict in its output, so it did not judge"; tail -n 3 "$out" | sed 's/^/        /'; return 2
            fi
            if [ "$mode" = enforce ]; then
                echo "FAIL  R8 $label: Fail, $sum"; return 1
            fi
            echo "WARN  R8 REPORT-ONLY $label: Fail, $sum (not a stop until #3712 lands; DEFAULT_MODE in $PROG)"
            return 0 ;;
        2) echo "FAIL  R8 $label: pv DECLINED (rc 2), and a decline is not a pass: $(tail -n 1 "$out")"; return 2 ;;
        3) echo "FAIL  R8 $label: pv refused the call (rc 3, caller error): $(tail -n 1 "$out")"; return 3 ;;
        *) echo "FAIL  R8 $label: pv exited $rc, which is outside its 0/1/2/3 contract: $(tail -n 1 "$out")"; return 2 ;;
    esac
}

main() {
    local root="" version="" commit="" receipts="" dogfood="" out="" mode tmpout rc
    while [ $# -gt 0 ]; do
        case "$1" in
            --root|--version|--commit|--receipts|--dogfood-receipt|--out)
                [ $# -ge 2 ] || caller_error "$1 needs a value" ;;
        esac
        case "$1" in
            --root) root="$2"; shift 2 ;;
            --version) version="$2"; shift 2 ;;
            --commit) commit="$2"; shift 2 ;;
            --receipts) receipts="$2"; shift 2 ;;
            --dogfood-receipt) dogfood="$2"; shift 2 ;;
            --out) out="$2"; shift 2 ;;
            *) caller_error "unknown argument $1" ;;
        esac
    done
    [ -n "$version" ] || caller_error "--version is required"
    [ -n "$commit" ] || caller_error "--commit is required"
    mode="$(resolve_mode "$DEFAULT_MODE" "${RELEASE_READINESS_MODE:-}")" \
        || caller_error "RELEASE_READINESS_MODE='${RELEASE_READINESS_MODE:-}' may only strengthen the committed mode ($DEFAULT_MODE) to enforce"
    [ -n "$root" ] || root="$(cd -- "$(dirname -- "$SCRIPT_PATH")/../.." && pwd)"
    git -C "$root" rev-parse --verify --quiet HEAD >/dev/null || { echo "FAIL  R8 $root is not a git repository"; exit 2; }
    [ -z "$dogfood" ] || [ -f "$dogfood" ] || caller_error "--dogfood-receipt $dogfood does not exist"
    if [ -n "$out" ]; then tmpout="$out"; else tmpout="$(mktemp)"; fi
    judge "$mode" "$root" "$version" "$commit" "$receipts" "$dogfood" "$tmpout"; rc=$?
    if [ "$rc" = 2 ] && [ "$mode" = report ]; then
        echo "WARN  R8 REPORT-ONLY could not judge (rc 2), recorded and not a stop until #3712 lands (DEFAULT_MODE in $PROG)"
        rc=0
    fi
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
    mk() { # dir: a repo with src/ at commit A, receipts naming A committed on top (HEAD = B)
        mkdir -p "$1/src" "$1/evidence/dogfood/models/1.2.3"; printf 'a\n' > "$1/src/f"
        git init -q -b main "$1"; g "$1" add -A; g "$1" commit -qm A
        local a; a="$(git -C "$1" rev-parse HEAD)"
        printf '{"apr_sha":"%s"}\n' "$a" > "$1/evidence/dogfood/models/1.2.3/lambda.json"
        printf '{"apr_sha":"%s"}\n' "$a" > "$1/evidence/dogfood/models/1.2.3/gx10.json"
        g "$1" add -A; g "$1" commit -qm receipts
    }
    row() { # name expect-rc needle dir [env...] -- runs main in a subshell
        local name="$1" expect="$2" needle="$3" dir="$4" o rc=0; shift 4
        o="$( env FX_ARGS="$tmp/args" RELEASE_READINESS_PV="$tmp/pv" "$@" bash "$SCRIPT_PATH" --root "$dir" --version 1.2.3 --commit "$(git -C "$dir" rev-parse HEAD)" 2>&1 )" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-44s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
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
    argrow pass_asks_the_shape                   "--gate shapes --shape release-readiness-v1 --release-version 1.2.3 --release-commit $c"
    argrow receipts_commit_earned_is_passed      "--receipts-commit $x"
    row fail_under_report_warns                  0 "WARN  R8 REPORT-ONLY" "$d" FX_PV_RC=1
    row fail_under_report_names_the_count        0 "3 violation(s): cell=3" "$d" FX_PV_RC=1
    row fail_under_enforce_refuses               1 "FAIL  R8 release-readiness-v1 for 1.2.3" "$d" FX_PV_RC=1 RELEASE_READINESS_MODE=enforce
    row decline_under_report_is_recorded         0 "REPORT-ONLY could not judge (rc 2)" "$d" FX_PV_RC=2 FX_PV_BODY=junk
    row decline_under_report_names_the_decline   0 "pv DECLINED" "$d" FX_PV_RC=2 FX_PV_BODY=junk
    row decline_under_enforce_refuses            2 "pv DECLINED" "$d" FX_PV_RC=2 FX_PV_BODY=junk RELEASE_READINESS_MODE=enforce
    row caller_error_is_never_downgraded         3 "caller error" "$d" FX_PV_RC=3 FX_PV_BODY=junk
    row exit1_without_a_fail_verdict_declines    2 "no Fail verdict" "$d" FX_PV_RC=1 FX_PV_BODY=junk RELEASE_READINESS_MODE=enforce
    row exit_outside_contract_declines           2 "outside its 0/1/2/3 contract" "$d" FX_PV_RC=101 FX_PV_BODY=junk RELEASE_READINESS_MODE=enforce
    row missing_pv_declines                      2 "is not executable" "$d" RELEASE_READINESS_PV="$tmp/nope" RELEASE_READINESS_MODE=enforce
    row missing_pv_under_report_is_recorded      0 "REPORT-ONLY could not judge (rc 2)" "$d" RELEASE_READINESS_PV="$tmp/nope"
    row missing_pv_under_report_names_the_cause  0 "is not executable" "$d" RELEASE_READINESS_PV="$tmp/nope"
    row no_verdict_under_report_is_recorded      0 "REPORT-ONLY could not judge (rc 2)" "$d" FX_PV_RC=1 FX_PV_BODY=junk
    row outside_contract_under_report_recorded   0 "REPORT-ONLY could not judge (rc 2)" "$d" FX_PV_RC=101 FX_PV_BODY=junk
    row env_may_not_weaken_to_bogus              3 "may only strengthen" "$d" RELEASE_READINESS_MODE=off
    # a source change between the receipts' commit and the release commit: the flag is WITHHELD
    d="$tmp/s"; mk "$d"; printf 'b\n' > "$d/src/f"; g "$d" commit -qam src
    row source_drift_withholds_receipts_commit   0 "differs from" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    argrow source_drift_not_passed               "!--receipts-commit"
    # two hosts measured at different commits: withheld
    d="$tmp/m"; mk "$d"; printf '{"apr_sha":"%s"}\n' "$(printf '0%.0s' $(seq 40))" > "$d/evidence/dogfood/models/1.2.3/gx10.json"; g "$d" commit -qam split
    row split_apr_sha_withholds                  0 "different apr_sha values" "$d" FX_PV_RC=0 FX_PV_BODY=pass
    argrow split_apr_sha_not_passed              "!--receipts-commit"
    # the committed default is what a bare run uses; flipping it is a reviewed commit
    if [ "$(resolve_mode report '')" = report ] && [ "$(resolve_mode enforce '')" = enforce ] \
        && ! resolve_mode enforce report >/dev/null && [ "$(resolve_mode report enforce)" = enforce ]; then
        printf '  ok    %-44s report over enforce refused, enforce over report taken\n' mode_env_only_strengthens; pass=$((pass + 1))
    else
        printf '  BROKE %-44s\n' mode_env_only_strengthens; fail=$((fail + 1))
    fi
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --selftest) selftest ;;
    -h|--help) sed -n '2,37p' "$0" ;;
    *) main "$@" ;;
esac
