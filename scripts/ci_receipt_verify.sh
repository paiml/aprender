#!/usr/bin/env bash
# ci_receipt_verify.sh - does a signed batch receipt cover THIS tree? (PMAT-5737, receipt format v1)
#
#   ci_receipt_verify.sh <owner/repo> <receipt> <receipt.sig> [allowed_signers] [toolchain-id-script]
#   ci_receipt_verify.sh --self-test
#
# Run from the checked-out repo. The tree asked about is `git rev-parse HEAD^{tree}` of the cwd.
# Defaults: ci/receipt/allowed_signers (COMMITTED at this tree, never fetched with the receipt) and
# scripts/ci/toolchain_id.sh. Prints exactly one line on stdout:
#   RECEIPT_OK=1
#   RECEIPT_OK=0 reason=<why>
# exit 0 on RECEIPT_OK=1, 1 otherwise. FAIL CLOSED: a missing tool, file, parse problem or any error is
# RECEIPT_OK=0, and the caller runs the full gate. A receipt never skips anything this script did not prove.
# (f), never on a tag, is the caller's (ci_test_tier.sh refuses a tag ref).
#
# The format (8 lines, LF, in this order, final newline, no trailing spaces):
#   pmat-receipt v1 / repo= / tree= / toolchain= / manifest= / suite= / tickets= / ts=
# The signature: ssh-keygen -Y sign -n pmat-5737-receipt, principal pmat-5737-batcher.
#
# Every refusal is ONE line ending in a `# R-<NAME>` marker.
set -uo pipefail

PRINCIPAL=pmat-5737-batcher
NAMESPACE=pmat-5737-receipt
KEYS=(repo tree toolchain manifest suite tickets ts)

no() { echo "RECEIPT_OK=0 reason=$*"; exit 1; }

verify() {
    local repo=${1:-} receipt=${2:-} sig=${3:-} allowed=${4:-ci/receipt/allowed_signers} tcid=${5:-scripts/ci/toolchain_id.sh}
    local line i key head_tree got_tc
    [ -n "$repo" ] && [ -n "$receipt" ] && [ -n "$sig" ] || no "usage: <owner/repo> <receipt> <receipt.sig>" # R-USAGE
    [ -f "$receipt" ] || no "no receipt file" # R-NORECEIPT
    [ -f "$sig" ] || no "no receipt.sig file" # R-NOSIG
    [ -f "$allowed" ] || no "no committed allowed_signers" # R-NOALLOWED
    command -v ssh-keygen >/dev/null 2>&1 || no "ssh-keygen missing" # R-NOSSHKEYGEN
    ! grep -q 'PLACEHOLDER' "$allowed" || no "allowed_signers holds the PLACEHOLDER, not a real key" # R-PLACEHOLDER

    # (a) the signature, against the committed allowed_signers
    ssh-keygen -Y verify -f "$allowed" -I "$PRINCIPAL" -n "$NAMESPACE" -s "$sig" < "$receipt" >/dev/null 2>&1 \
        || no "signature does not verify" # R-SIG

    # (b) shape: exactly 8 lines, LF only, final newline, header, keys in order, no trailing space
    [ "$(wc -l < "$receipt" | tr -d ' ')" = 8 ] || no "receipt is not exactly 8 LF-terminated lines" # R-LINECOUNT
    [ "$(tail -c1 "$receipt" | od -An -c | tr -d ' ')" = '\n' ] || no "receipt has no final newline" # R-FINALNL
    ! grep -q $'\r' "$receipt" || no "receipt has a CR" # R-CR
    ! grep -qE '[[:space:]]$' "$receipt" || no "receipt has trailing whitespace" # R-TRAILSP
    mapfile -t lines < "$receipt"
    [ "${#lines[@]}" -eq 8 ] || no "receipt does not parse into 8 lines" # R-PARSE
    [ "${lines[0]}" = "pmat-receipt v1" ] || no "line 1 is not 'pmat-receipt v1'" # R-HEADER
    declare -A kv=()
    for i in 0 1 2 3 4 5 6; do
        line=${lines[$((i + 1))]}; key=${KEYS[$i]}
        [[ $line == "$key="* ]] || no "line $((i + 2)) is not $key=" # R-KEYORDER
        kv[$key]=${line#*=}
    done
    [[ ${kv[repo]} =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || no "repo= is not owner/name" # R-REPOFORM
    [[ ${kv[tree]} =~ ^[0-9a-f]{40}$ ]] || no "tree= is not 40-hex" # R-TREEFORM
    [[ ${kv[toolchain]} =~ ^[0-9a-f]{64}$ ]] || no "toolchain= is not sha256 hex" # R-TCFORM
    [[ ${kv[manifest]} =~ ^[0-9a-f]{64}$ ]] || no "manifest= is not sha256 hex" # R-MANFORM
    [ -n "${kv[suite]}" ] || no "suite= is empty" # R-SUITEFORM
    [[ ${kv[tickets]} =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]] || no "tickets= is not a comma list" # R-TICKFORM
    [ "$(printf '%s' "${kv[tickets]}" | tr ',' '\n' | sort -u | paste -sd, -)" = "${kv[tickets]}" ] || no "tickets= is not sorted and unique" # R-TICKSORT
    [[ ${kv[ts]} =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || no "ts= is not UTC ISO-8601" # R-TSFORM

    # (c) the repo
    [ "${kv[repo]}" = "$repo" ] || no "repo= ${kv[repo]} is not $repo" # R-REPO
    # (d) the tree
    head_tree=$(git rev-parse 'HEAD^{tree}' 2>/dev/null) || no "cannot read HEAD^{tree}" # R-HEADTREE
    [ "${kv[tree]}" = "$head_tree" ] || no "tree= ${kv[tree]} is not HEAD's tree $head_tree" # R-TREE
    # (e) the toolchain
    [ -f "$tcid" ] || no "no toolchain-id script" # R-NOTCID
    got_tc=$(bash "$tcid" 2>/dev/null) || no "toolchain-id script failed" # R-TCRUN
    [[ $got_tc =~ ^[0-9a-f]{64}$ ]] || no "toolchain-id script printed no sha256" # R-TCOUT
    [ "${kv[toolchain]}" = "$got_tc" ] || no "toolchain= differs from this CI's toolchain" # R-TOOLCHAIN
    echo "RECEIPT_OK=1"
    exit 0
}

self_test() {
    local td n=0 red=0 out rc here tree tc man
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    td=$(mktemp -d "${TMPDIR:-/tmp}/receipt-verify.XXXXXX") || { echo "self-test: no tempdir" >&2; return 2; }
    trap 'rm -rf "${td:?}"' RETURN
    command -v ssh-keygen >/dev/null 2>&1 || { echo "self-test: ssh-keygen missing" >&2; return 2; }
    # throwaway keys, born and deleted inside $td
    ssh-keygen -q -t ed25519 -N '' -C t -f "$td/k" >/dev/null 2>&1 || { echo "self-test: keygen failed" >&2; return 2; }
    ssh-keygen -q -t ed25519 -N '' -C t -f "$td/other" >/dev/null 2>&1 || { echo "self-test: keygen failed" >&2; return 2; }
    printf '%s namespaces="%s" %s\n' "$PRINCIPAL" "$NAMESPACE" "$(cut -d' ' -f1,2 "$td/k.pub")" > "$td/allowed"
    printf '%s namespaces="%s" %s\n' "$PRINCIPAL" "$NAMESPACE" "PLACEHOLDER-NOT-A-KEY" > "$td/allowed.placeholder"
    tc=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    man=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" %s\n' "$tc" > "$td/tcid.sh"
    printf '#!/usr/bin/env bash\nexit 1\n' > "$td/tcid-bad.sh"
    # a throwaway repo: its HEAD tree is what a receipt must name
    git init -q -b main "$td/r" || { echo "self-test: git init failed" >&2; return 2; }
    ( cd "$td/r" && export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t \
        GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null \
      && printf 'x\n' > f && git add f && git commit -q -m c ) || { echo "self-test: commit failed" >&2; return 2; }
    tree=$(git -C "$td/r" rev-parse 'HEAD^{tree}')
    # mk NAME [key=value ...] [--key K] [--ns N] [--extra] [--sed EXPR] [--strip-nl]: write and sign a receipt
    mk() {
        local name=$1 key=$td/k ns=$NAMESPACE extra=0 sedx="" strip=0 k; shift
        local -A o=([repo]=paiml/aprender [tree]=$tree [toolchain]=$tc [manifest]=$man [suite]="make test-fast" [tickets]=PMAT-1,PMAT-2 [ts]=2026-10-19T00:00:00Z)
        while [ $# -gt 0 ]; do
            case $1 in
                --key) key=$2; shift 2 ;;
                --ns) ns=$2; shift 2 ;;
                --extra) extra=1; shift ;;
                --sed) sedx=$2; shift 2 ;;
                --strip-nl) strip=1; shift ;;
                *) o[${1%%=*}]=${1#*=}; shift ;;
            esac
        done
        { printf 'pmat-receipt v1\n'; for k in "${KEYS[@]}"; do printf '%s=%s\n' "$k" "${o[$k]}"; done
          if [ "$extra" = 1 ]; then printf 'extra=1\n'; fi; } > "$td/$name"
        if [ -n "$sedx" ]; then sed -i "$sedx" "$td/$name"; fi
        if [ "$strip" = 1 ]; then truncate -s -1 "$td/$name"; fi
        rm -f "${td:?}/$name.sig"
        ssh-keygen -q -Y sign -f "$key" -n "$ns" "$td/$name" >/dev/null 2>&1
    }
    chk() { # chk WANT LABEL NAME [allowed] [tcid]  (runs inside the throwaway repo)
        local want=$1 label=$2 name=$3 allowed=${4:-$td/allowed} tcs=${5:-$td/tcid.sh}
        n=$((n + 1)); rc=0
        out=$(cd "$td/r" && bash "$here/ci_receipt_verify.sh" paiml/aprender "$td/$name" "$td/$name.sig" "$allowed" "$tcs" 2>&1) || rc=$?
        if { [ "$want" = ok ] && [ "$rc" = 0 ] && [ "$out" = "RECEIPT_OK=1" ]; } || { [ "$want" = no ] && [ "$rc" = 1 ] && [[ $out == "RECEIPT_OK=0 reason="* ]]; }; then
            printf 'ok    row %-2s %s\n' "$n" "$label"
        else printf 'FAIL  row %-2s (wanted %s, rc=%s, got "%s")  %s\n' "$n" "$want" "$rc" "$out" "$label"; red=1; fi
    }
    mk good
    chk ok "green path: signed, right repo, tree, toolchain" good
    mk wrongtree tree=0123456789abcdef0123456789abcdef01234567
    chk no "MUTANT wrong tree, validly signed" wrongtree
    mk wrongrepo repo=paiml/other
    chk no "MUTANT wrong repo, validly signed" wrongrepo
    mk wrongtc toolchain=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    chk no "MUTANT wrong toolchain, validly signed" wrongtc
    mk tamper
    sed -i 's/^suite=.*/suite=true/' "$td/tamper"
    chk no "MUTANT a line tampered after signing" tamper
    mk extra --extra
    chk no "MUTANT extra (9th) line, validly signed" extra
    mk wrongns --ns other-namespace
    chk no "MUTANT wrong signing namespace" wrongns
    mk unknownkey --key "$td/other"
    chk no "MUTANT signed by a key that is not in allowed_signers" unknownkey
    chk no "MUTANT placeholder allowed_signers (the committed state)" good "$td/allowed.placeholder"
    mk nosig
    rm -f "${td:?}/nosig.sig"
    chk no "MUTANT missing receipt.sig" nosig
    mk badver --sed '1s/v1/v2/'
    chk no "MUTANT header pmat-receipt v2, validly signed" badver
    mk badorder --sed '3{h;d};4{G}'
    chk no "MUTANT keys out of order (toolchain before tree), validly signed" badorder
    mk badtick tickets=PMAT-2,PMAT-1
    chk no "MUTANT tickets= unsorted, validly signed" badtick
    mk badts ts=yesterday
    chk no "MUTANT ts= not UTC ISO-8601, validly signed" badts
    mk crlf --sed 's/$/\r/'
    chk no "MUTANT CRLF line endings, validly signed" crlf
    mk nonl --strip-nl
    chk no "MUTANT no final newline, validly signed" nonl
    mk trail --sed '8s/$/ /'
    chk no "MUTANT trailing space, validly signed" trail
    chk no "MUTANT missing receipt file (error is fail closed)" absent
    chk no "MUTANT toolchain-id script fails (error is fail closed)" good "$td/allowed" "$td/tcid-bad.sh"
    chk no "the COMMITTED allowed_signers fails closed" good "$here/../ci/receipt/allowed_signers"
    printf '\n%s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
verify "$@"
