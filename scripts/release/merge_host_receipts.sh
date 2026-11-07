#!/usr/bin/env bash
# merge_host_receipts.sh -- APR-RELEASE-001 T-2 FALLBACK (REL1, epic #6728): merge per-host dogfood
# preflight logs of the SAME commit into ONE verdict file, by rule, never by hand.
#   merge_host_receipts.sh --commit SHA --out FILE --log HOST=PATH [--log HOST=PATH ...]
#                          [--arch HOST=x86_64 ...] [--x86-cuda-ruling PATH]
#   merge_host_receipts.sh --self-test        (runs falsify_merge_host_receipts.sh)
# Exit 0 GO (verdict file written), 1 NO-GO (nothing written), 2 refused / UNKNOWN (nothing written).
# t2_preflight.sh decide() judges ONE log and is NOT changed or sourced here. A log carries its commit in
# a header line `# receipt-commit: <40-hex>` (the host runner prepends it) or t2_preflight's own
# `t2_preflight: judging <sha>`; a log without one is refused, never guessed from its file name.
# Rules: (a) every header sha == --commit. (b) each [FAIL] row is (i) the pre-bump class decide() tolerates
# (version-unpublished; the dogfood-gates roll-up when its RED count equals the declared [FAIL] rows), or
# (ii) green in another log AND its text matches the environment-only allowlist below. Anything else is an
# assertion FAIL and the merge is NO-GO. (c) a row red on every log that ran it is NO-GO. (d) every log
# declares the same gate count, has its VERDICT line and rows >= declared, and the union of row names
# covers that count. (e) the verdict names the receipt path and host per row. (f) the C14 row
# `model-parity` must be [ OK ] on a host given --arch HOST=x86_64 with `--features cuda`, else exit 2
# unless --x86-cuda-ruling PATH names a quorum receipt that contains the commit sha.
# The pre-bump model-ladder deferral is NOT merged here (it needs the ladder log): a red ladder row fails.
set -uo pipefail

# Environment-only reason texts (extend only by a gate-content PR; each is a host fact, not an assertion).
ENV_ALLOW=('Feature not enabled: cuda' 'NOT HELD under' 'is not on this host' 'GitHub read failed')

# merge <commit> <out> <ruling|''> [HOST=ARCH ...] -- HOST=PATH...
merge() {
    local commit=$1 out=$2 ruling=$3; shift 3
    local -a archs=() logs=()
    while [ "$#" -gt 0 ] && [ "$1" != -- ]; do archs+=("$1"); shift; done
    [ "${1:-}" = -- ] && shift
    logs=("$@")
    [ "${#logs[@]}" -ge 2 ] || { echo "merge_host_receipts: need at least two --log HOST=PATH" >&2; return 2; }
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "merge_host_receipts: --commit must be a 40-hex sha" >&2; return 2; }
    local -A hostlog=() declared=() rowstat=() fail_ok=() seen=() okany=()
    local -a names=()
    local spec host path hs n rows nm st line a first='' red dn x86cuda=0 why='' hit t
    for spec in "${logs[@]}"; do
        host=${spec%%=*}; path=${spec#*=}
        [[ "$host" =~ ^[A-Za-z0-9_.-]+$ ]] && [ "$path" != "$spec" ] || { echo "merge_host_receipts: bad --log '$spec'" >&2; return 2; }
        [ -z "${hostlog[$host]:-}" ] || { echo "merge_host_receipts: host $host given twice" >&2; return 2; }
        [ -r "$path" ] || { echo "UNKNOWN $commit reason=no-log host=$host"; return 2; }
        hostlog[$host]=$path
        # (a) header sha
        hs=$(sed -nE 's/^# receipt-commit: ([0-9a-f]{40}).*/\1/p; s/^t2_preflight: judging ([0-9a-f]{40}).*/\1/p' "$path" | head -n 1)
        [ -n "$hs" ] || { echo "UNKNOWN $commit reason=no-header-sha host=$host"; return 2; }
        [ "$hs" = "$commit" ] || { echo "NO-GO $commit reason=sha-mismatch host=$host header=$hs"; return 1; }
        grep -qE '^VERDICT:' "$path" || { echo "UNKNOWN $commit reason=no-verdict-line host=$host"; return 2; }
        n=$(grep -oE '[0-9]+ declared gate\(s\) discovered' "$path" | head -1 | cut -d' ' -f1)
        [ -n "$n" ] || { echo "UNKNOWN $commit reason=no-declared-gate-count host=$host"; return 2; }
        declared[$host]=$n
        rows=$(grep -cE '^[[:space:]]*\[( OK |[A-Z]+)\]' "$path")
        [ "$rows" -ge "$n" ] || { echo "UNKNOWN $commit reason=rows=$rows<declared=$n host=$host"; return 2; }
    done
    # (d) one declared count across hosts
    for host in "${!hostlog[@]}"; do
        if [ -z "$first" ]; then first=${declared[$host]}
        elif [ "${declared[$host]}" != "$first" ]; then echo "NO-GO $commit reason=declared-count-differs host=$host"; return 1; fi
    done
    # collect rows
    for host in "${!hostlog[@]}"; do
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*\[(\ OK |[A-Z]+)\][[:space:]]+([^[:space:]]+)(.*)$ ]] || continue
            st=${BASH_REMATCH[1]// /}; nm=${BASH_REMATCH[2]}
            if [ -z "${seen[$nm]:-}" ]; then
                seen["$nm"]=1; names+=("$nm")
            fi
            rowstat[$host|$nm]=$st
            if [ "$st" = OK ]; then
                okany["$nm"]=1
            fi
            if [ "$nm" = model-parity ] && [ "$st" = OK ] && [[ "$line" == *'--features cuda'* ]]; then
                for a in "${archs[@]}"; do [ "$a" = "$host=x86_64" ] && x86cuda=1; done
            fi
        done < "${hostlog[$host]}"
    done
    # (b)/(c) judge each FAIL row
    for host in "${!hostlog[@]}"; do
        path=${hostlog[$host]}
        red=$(grep -E '^[[:space:]]*\[FAIL\]' "$path")
        dn=$(grep -cE '^[[:space:]]*\[FAIL\] declared:' <<< "$red")
        while IFS= read -r line; do
            [ -n "$line" ] || continue
            read -r _ nm _ <<< "$line"
            case "$nm" in
                version-unpublished) continue ;;
                dogfood-gates)
                    if [[ "$line" =~ discovered,\ ([0-9]+)\ RED ]] && [ "${BASH_REMATCH[1]}" -ge 1 ] && [ "${BASH_REMATCH[1]}" -eq "$dn" ]; then continue; fi
                    echo "NO-GO $commit reason=roll-up-unexplained host=$host receipt=$path"; return 1 ;;
            esac
            if [ -z "${okany[$nm]:-}" ]; then
                echo "NO-GO $commit reason=red-on-every-host row=$nm host=$host receipt=$path"; return 1
            fi
            hit=0
            for t in "${ENV_ALLOW[@]}"; do [[ "$line" == *"$t"* ]] && hit=1; done
            if [ "$hit" -ne 1 ]; then
                echo "NO-GO $commit reason=assertion-fail-not-environment row=$nm host=$host receipt=$path"; return 1
            fi
            fail_ok[$host|$nm]=1
        done <<< "$red"
    done
    [ "${#names[@]}" -ge "$first" ] || { echo "UNKNOWN $commit reason=union-rows=${#names[@]}<declared=$first"; return 2; }
    # (f) x86_64 CUDA C14
    if [ "$x86cuda" -ne 1 ]; then
        if [ -n "$ruling" ] && [ -s "$ruling" ] && grep -qF -- "$commit" "$ruling"; then
            why=" x86_cuda_ruling=$ruling"
        else
            echo "REFUSED $commit reason=no-x86_64-cuda-C14-green (give --arch HOST=x86_64 with a green model-parity --features cuda, or --x86-cuda-ruling PATH naming the commit)"
            return 2
        fi
    fi
    # (e) emit
    {
        printf 'GO %s merged hosts=%s declared=%s rows=%s%s\n' "$commit" "$(printf '%s,' "${!hostlog[@]}" | sed 's/,$//')" "$first" "${#names[@]}" "$why"
        for nm in "${names[@]}"; do
            for host in "${!hostlog[@]}"; do
                [ -n "${rowstat[$host|$nm]:-}" ] || continue
                printf 'ROW %s %s host=%s receipt=%s%s\n' "$nm" "${rowstat[$host|$nm]}" "$host" "${hostlog[$host]}" "${fail_ok[$host|$nm]:+ env-only-waived-by-other-host}"
            done
        done
    } > "$out" || return 2
    head -n 1 "$out"
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    exec bash "$(dirname "$0")/falsify_merge_host_receipts.sh"
fi

commit=''; out=''; ruling=''; archs=(); logs=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --commit) commit=${2:-}; shift 2 ;;
        --out) out=${2:-}; shift 2 ;;
        --log) logs+=("${2:-}"); shift 2 ;;
        --arch) archs+=("${2:-}"); shift 2 ;;
        --x86-cuda-ruling) ruling=${2:-}; shift 2 ;;
        *) echo "usage: merge_host_receipts.sh --commit SHA --out FILE --log HOST=PATH ... [--arch HOST=x86_64] [--x86-cuda-ruling PATH]" >&2; exit 2 ;;
    esac
done
[ -n "$out" ] || { echo "merge_host_receipts: --out FILE required" >&2; exit 2; }
rm -f -- "${out:?}"
merge "$commit" "$out" "$ruling" "${archs[@]}" -- "${logs[@]}"
rc=$?
[ "$rc" -eq 0 ] || rm -f -- "${out:?}"
exit "$rc"
