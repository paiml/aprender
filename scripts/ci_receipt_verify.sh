#!/usr/bin/env bash
# ci_receipt_verify.sh - does a signed batch receipt in the PR body cover THIS merge tree?
# (PMAT-5737, FLOW-009 receipt format flow009.receipt/1, RC-1..RC-5; infra PMAT-1889 owns the signer.)
#
#   ci_receipt_verify.sh <owner/repo> <event.json> [pubkey-file] [toolchain-id-script] [manifest-id-script]
#   ci_receipt_verify.sh --self-test
#
# Run from the checked-out PR MERGE commit. <event.json> is $GITHUB_EVENT_PATH: the receipt is read from
# .pull_request.body (RC-3: never an API call, never a comment, never a ref) and the PR head from
# .pull_request.head.sha. Defaults: .flow009/batcher.pub (COMMITTED at this tree), scripts/ci/toolchain_id.sh,
# scripts/ci/manifest_id.sh. Prints on stdout:
#   RECEIPT_OK=1
#   RECEIPT_OK=0 reason=<why>
# On RECEIPT_OK=1 a second stdout line `coverage=<gate_results.coverage>` follows when that value is numeric (sovereign-ci.yml
# reads it); absent or non-numeric => no such line. FLOW009_PUBKEY (env), when set, is THE pubkey path and overrides the
# argument and the default: the caller passes the BASE commit's copy so a PR cannot sign with a key it committed itself.
# exit 0 on RECEIPT_OK=1, 1 otherwise. FAIL CLOSED (PG-3): a missing tool or file, a parse problem or any error is
# RECEIPT_OK=0 and the caller runs the full gate -- never red. Not a tag and not a push: that is the caller's
# (ci/sections.yml runs this for pull_request only; ci_test_tier.sh refuses a tag ref again).
#
# The receipt: a fenced block whose info string is exactly `flow009.receipt/1`, ONE per body, holding a JSON object
# with exactly the RC-1 keys: schema repo batch base_sha tree commits gate_results toolchain manifest steps host
# started finished sig. Signed bytes = `jq -S -c 'del(.sig)'` + LF (the BYTE-IDENTICAL canonicalisation infra's
# fleet/receipts/receipt.sh signs). sig = `ssh-keygen -Y sign -n flow009-receipt` armor, newlines escaped in JSON.
# Verified (RC-5): sig; tree == HEAD^{tree}; commits[].patch_id multiset == `git patch-id --stable` over each
# commit of merge-base($RECEIPT_BASE or HEAD^1, PR head)..PR head; each commits[].ticket ^PMAT-[0-9]+$ and in that commit's message;
# toolchain; manifest. Also required: schema, repo == <owner/repo>, gate_results.tests == "pass", steps sha256 hex.
# base_sha and commits[].sha are informational and never compared.
#
# Every refusal is ONE line ending in a `# R-<NAME>` marker.
set -uo pipefail

PRINCIPAL=flow009-batcher
NAMESPACE=flow009-receipt
SCHEMA=flow009.receipt/1
FENCE='```flow009.receipt/1'
CLOSE='```'
TOP_KEYS='["base_sha","batch","commits","finished","gate_results","host","manifest","repo","schema","sig","started","steps","toolchain","tree"]'

no() { echo "RECEIPT_OK=0 reason=$*"; exit 1; }

verify() {
    local repo=${1:-} event=${2:-} pub=${3:-.flow009/batcher.pub} tcid=${4:-scripts/ci/toolchain_id.sh} manid=${5:-scripts/ci/manifest_id.sh}
    local cov td fences rcpt keyline head_tree head_sha base merge_base got i n pid tk sha ok
    # FLOW009_PUBKEY (trust boundary, PMAT-1889): when set it IS the pubkey path and wins over the argument and the default,
    # so a caller can pin the BASE commit's key and a PR can never supply its own. Set but unreadable => refused below.
    if [ -n "${FLOW009_PUBKEY:-}" ]; then pub=$FLOW009_PUBKEY; fi
    [ -n "$repo" ] && [ -n "$event" ] || no "usage: <owner/repo> <event.json>" # R-USAGE
    [ -f "$event" ] || no "no event file" # R-NOEVENT
    command -v jq >/dev/null 2>&1 || no "jq missing" # R-NOJQ
    command -v ssh-keygen >/dev/null 2>&1 || no "ssh-keygen missing" # R-NOSSHKEYGEN
    command -v git >/dev/null 2>&1 || no "git missing" # R-NOGIT
    td=$(mktemp -d "${TMPDIR:-/tmp}/receipt-v.XXXXXX") || no "no tempdir" # R-TMP
    trap 'rm -rf "${td:?}"' EXIT

    # (a) transport: the PR body, exactly one fenced flow009.receipt/1 block
    jq -e '.pull_request | type == "object"' "$event" >/dev/null 2>&1 || no "event has no pull_request" # R-NOPR
    jq -r '.pull_request.body // ""' "$event" 2>/dev/null | tr -d '\r' > "$td/body" || no "cannot read PR body" # R-BODY
    fences=$(grep -c -x -F -e "$FENCE" "$td/body" || true)
    [ "$fences" = 1 ] || no "PR body has $fences flow009.receipt/1 fenced blocks, want exactly 1" # R-FENCE
    awk -v f="$FENCE" -v c="$CLOSE" '$0 == f && !s {s=1; next} s && $0 == c {closed=1; exit} s {print} END {exit closed ? 0 : 1}' "$td/body" > "$td/receipt.json" \
        || no "receipt block is not closed" # R-UNCLOSED
    [ -s "$td/receipt.json" ] || no "empty receipt block" # R-EMPTYBLOCK
    rcpt=$td/receipt.json

    # (b) shape: exact keys, exact types, no unknown or missing key
    jq -e . "$rcpt" >/dev/null 2>&1 || no "receipt is not JSON" # R-JSON
    jq -e --argjson k "$TOP_KEYS" 'type == "object" and (keys == $k)' "$rcpt" >/dev/null 2>&1 \
        || no "receipt keys are not exactly the RC-1 set" # R-KEYS
    jq -e --arg s "$SCHEMA" '.schema == $s' "$rcpt" >/dev/null 2>&1 || no "schema is not $SCHEMA" # R-SCHEMA
    jq -e '[.repo,.batch,.base_sha,.tree,.toolchain,.manifest,.steps,.host,.started,.finished,.sig] | all(type == "string")' "$rcpt" >/dev/null 2>&1 \
        || no "a scalar field is not a string" # R-TYPES
    jq -e '(.gate_results | type == "object" and (keys == ["coverage","tests"]))' "$rcpt" >/dev/null 2>&1 \
        || no "gate_results keys are not exactly tests, coverage" # R-GATEKEYS
    jq -e '.gate_results.tests == "pass"' "$rcpt" >/dev/null 2>&1 || no "gate_results.tests is not pass" # R-TESTSPASS
    jq -e '(.commits | type == "array" and length > 0 and all(type == "object" and (keys == ["patch_id","sha","ticket"]) and ([.[]] | all(type == "string"))))' "$rcpt" >/dev/null 2>&1 \
        || no "commits is not a non-empty array of exactly {ticket, patch_id, sha}" # R-COMMITKEYS
    jq -e '(.tree | test("^[0-9a-f]{40}$")) and (.toolchain | test("^[0-9a-f]{64}$")) and (.manifest | test("^[0-9a-f]{64}$")) and (.steps | test("^[0-9a-f]{64}$"))' "$rcpt" >/dev/null 2>&1 \
        || no "tree/toolchain/manifest/steps is not lowercase hex of the right length" # R-HEXFORM
    jq -e '.commits | all((.ticket | test("^PMAT-[0-9]+$")) and (.patch_id | test("^[0-9a-f]{40}$")))' "$rcpt" >/dev/null 2>&1 \
        || no "a commit ticket is not PMAT-<digits> or a patch_id is not 40-hex" # R-TICKETFORM

    # (c) the signature, against the COMMITTED public key (never one that travelled with the receipt)
    [ -f "$pub" ] || no "no committed .flow009/batcher.pub" # R-NOPUB
    keyline=$(grep -v -E '^[[:space:]]*(#|$)' "$pub" | head -n 2)
    [ "$(printf '%s\n' "$keyline" | grep -c .)" = 1 ] || no "batcher.pub is a placeholder or not exactly one key line" # R-PLACEHOLDER
    [[ $keyline =~ ^ssh-ed25519[[:space:]]+[A-Za-z0-9+/]+=*([[:space:]].*)?$ ]] || no "batcher.pub line is not an ssh-ed25519 key" # R-PUBFORM
    printf '%s namespaces="%s" %s\n' "$PRINCIPAL" "$NAMESPACE" "$(printf '%s' "$keyline" | cut -d' ' -f1,2)" > "$td/allowed"
    jq -r '.sig' "$rcpt" > "$td/sig" 2>/dev/null || no "cannot read sig" # R-SIGREAD
    [ -s "$td/sig" ] || no "empty sig" # R-SIGEMPTY
    jq -S -c 'del(.sig)' "$rcpt" > "$td/canon" 2>/dev/null || no "cannot canonicalise" # R-CANON
    ssh-keygen -Y verify -f "$td/allowed" -I "$PRINCIPAL" -n "$NAMESPACE" -s "$td/sig" < "$td/canon" >/dev/null 2>&1 \
        || no "signature does not verify" # R-SIG

    # (d) the repo and the tree
    [ "$(jq -r .repo "$rcpt")" = "$repo" ] || no "repo is not $repo" # R-REPO
    head_tree=$(git rev-parse 'HEAD^{tree}' 2>/dev/null) || no "cannot read HEAD^{tree}" # R-HEADTREE
    [ "$(jq -r .tree "$rcpt")" = "$head_tree" ] || no "tree is not HEAD's tree $head_tree" # R-TREE

    # (e) the commits: patch-id multiset and tickets, over merge-base(HEAD^1, PR head)..PR head
    head_sha=$(jq -r '.pull_request.head.sha // ""' "$event")
    [[ $head_sha =~ ^[0-9a-f]{40}$ ]] || no "event has no PR head sha" # R-NOHEAD
    git cat-file -e "${head_sha}^{commit}" 2>/dev/null || no "PR head commit is not in this checkout" # R-HEADMISSING
    base=$(git rev-parse -q --verify "${RECEIPT_BASE:-HEAD^1}" 2>/dev/null) || no "no base ref (RECEIPT_BASE, else the merge commit's first parent)" # R-NOBASE
    merge_base=$(git merge-base "$base" "$head_sha" 2>/dev/null) || no "no merge-base for base and PR head" # R-MERGEBASE
    git rev-list --reverse --no-merges "${merge_base}..${head_sha}" > "$td/revs" 2>/dev/null || no "cannot list PR commits" # R-REVLIST
    : > "$td/gitcommits"
    : > "$td/used"
    while read -r got; do
        [ -n "$got" ] || continue
        i=$(git diff-tree -p --no-color "$got" 2>/dev/null | git patch-id --stable | cut -d' ' -f1)
        [[ $i =~ ^[0-9a-f]{40}$ ]] || no "commit $got has no patch-id (empty diff?)" # R-NOPATCHID
        printf '%s %s\n' "$i" "$got" >> "$td/gitcommits"
    done < "$td/revs"
    n=$(jq '.commits | length' "$rcpt")
    [ "$(grep -c . "$td/gitcommits")" = "$n" ] || no "receipt names $n commits, the PR has $(grep -c . "$td/gitcommits")" # R-COMMITCOUNT
    [ "$(jq -r '.commits[].patch_id' "$rcpt" | sort | paste -sd, -)" = "$(cut -d' ' -f1 "$td/gitcommits" | sort | paste -sd, -)" ] \
        || no "commits[].patch_id multiset differs from git patch-id --stable over the PR" # R-PATCHIDS
    for ((i = 0; i < n; i++)); do
        ok=0
        pid=$(jq -r ".commits[$i].patch_id" "$rcpt"); tk=$(jq -r ".commits[$i].ticket" "$rcpt")
        while read -r got sha; do
            [ "$got" = "$pid" ] || continue
            if grep -q -x -F -e "$sha" "$td/used"; then continue; fi
            if git log -1 --format=%B "$sha" | grep -q -E "(^|[^0-9A-Za-z])${tk}([^0-9]|$)"; then ok=1; echo "$sha" >> "$td/used"; break; fi
        done < "$td/gitcommits"
        [ "$ok" = 1 ] || no "ticket $tk is not in the message of the commit with patch_id $pid" # R-TICKET
    done

    # (f) the toolchain and the manifest, as THIS checkout computes them
    # The helpers run from the BASE commit's trusted dir in CI, so they must be told the PR checkout: their own
    # location is not the tree the receipt was signed over (rust-toolchain.toml, the gate files, cargo metadata).
    root=$(git rev-parse --show-toplevel 2>/dev/null) || no "cannot find the checkout root" # R-ROOT
    [ -f "$tcid" ] || no "no toolchain-id script" # R-NOTCID
    got=$(bash "$tcid" "$root" 2>/dev/null) || no "toolchain-id script failed" # R-TCRUN
    [[ $got =~ ^[0-9a-f]{64}$ ]] || no "toolchain-id script printed no sha256" # R-TCOUT
    [ "$(jq -r .toolchain "$rcpt")" = "$got" ] || no "toolchain differs from this CI's toolchain" # R-TOOLCHAIN
    [ -f "$manid" ] || no "no manifest-id script" # R-NOMAN
    got=$(bash "$manid" "$root" 2>/dev/null) || no "manifest-id script failed" # R-MANRUN
    [[ $got =~ ^[0-9a-f]{64}$ ]] || no "manifest-id script printed no sha256" # R-MANOUT
    [ "$(jq -r .manifest "$rcpt")" = "$got" ] || no "manifest differs from this checkout's gate definition and test list" # R-MANIFEST
    echo "RECEIPT_OK=1"
    # sovereign-ci.yml reads ONE `coverage=<float>` stdout line on exit 0; absent or non-numeric => no line (still OK).
    cov=$(jq -r '.gate_results.coverage // ""' "$rcpt")
    if [[ $cov =~ ^[0-9]+([.][0-9]+)?$ ]]; then echo "coverage=$cov"; fi
    exit 0
}

self_test() {
    local td n=0 red=0 out rc here tc man tree head pid1 pid2 pidx
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    td=$(mktemp -d "${TMPDIR:-/tmp}/receipt-verify.XXXXXX") || { echo "self-test: no tempdir" >&2; return 2; }
    trap 'rm -rf "${td:?}"' RETURN
    command -v ssh-keygen >/dev/null 2>&1 && command -v jq >/dev/null 2>&1 || { echo "self-test: ssh-keygen or jq missing" >&2; return 2; }
    # throwaway keys, born and deleted inside $td
    ssh-keygen -q -t ed25519 -N '' -C t -f "$td/k" >/dev/null 2>&1 || { echo "self-test: keygen failed" >&2; return 2; }
    ssh-keygen -q -t ed25519 -N '' -C t -f "$td/other" >/dev/null 2>&1 || { echo "self-test: keygen failed" >&2; return 2; }
    cp "$td/k.pub" "$td/pub"
    printf '# PLACEHOLDER\n' > "$td/pub.placeholder"
    tc=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    man=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" %s\n' "$tc" > "$td/tcid.sh"
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" %s\n' "$man" > "$td/manid.sh"
    printf '#!/usr/bin/env bash\nexit 1\n' > "$td/bad.sh"
    # a throwaway repo shaped like a PR merge checkout: main -> merge(main, PR head); PR = 2 commits
    git init -q -b main "$td/r" || { echo "self-test: git init failed" >&2; return 2; }
    ( cd "$td/r" && export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t \
        GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null \
      && printf 'x\n' > f && git add f && git commit -q -m base \
      && git checkout -q -b pr && printf 'a\n' > a && git add a && git commit -q -m 'PMAT-11: add a' \
      && printf 'b\n' > b && git add b && git commit -q -m 'PMAT-22: add b' \
      && git checkout -q main && printf 'm\n' > m && git add m && git commit -q -m 'main moves' \
      && git merge -q --no-ff -m merge pr ) || { echo "self-test: repo build failed" >&2; return 2; }
    tree=$(git -C "$td/r" rev-parse 'HEAD^{tree}'); head=$(git -C "$td/r" rev-parse pr)
    pid1=$(git -C "$td/r" diff-tree -p "pr~1" | git patch-id --stable | cut -d' ' -f1)
    pid2=$(git -C "$td/r" diff-tree -p "pr" | git patch-id --stable | cut -d' ' -f1)
    pidx=$(git -C "$td/r" diff-tree -p "main~1" | git patch-id --stable | cut -d' ' -f1)
    jq -n --arg t "$tree" --arg tc "$tc" --arg m "$man" --arg p1 "$pid1" --arg p2 "$pid2" '{
      schema: "flow009.receipt/1", repo: "paiml/aprender", batch: "batch/aprender/1", base_sha: "0000",
      tree: $t, commits: [{ticket: "PMAT-11", patch_id: $p1, sha: "x"}, {ticket: "PMAT-22", patch_id: $p2, sha: "y"}],
      gate_results: {tests: "pass", coverage: "95.1"}, toolchain: $tc, manifest: $m,
      steps: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      host: "lambda-labs", started: "2026-10-19T00:00:00Z", finished: "2026-10-19T00:10:00Z"}' > "$td/base.json"
    # mk NAME FILTER [--key K] [--ns N] [--post FILTER] [--fences N] [--nofence]: NAME.event = an event file whose
    # body carries the receipt. FILTER edits the receipt BEFORE signing, --post AFTER (the old sig stays).
    mk() {
        local name=$1 filt=$2 key=$td/k ns=$NAMESPACE post=. fences=1 nofence=0 f; shift 2
        while [ $# -gt 0 ]; do
            case $1 in
                --key) key=$2; shift 2 ;;
                --ns) ns=$2; shift 2 ;;
                --post) post=$2; shift 2 ;;
                --fences) fences=$2; shift 2 ;;
                --nofence) nofence=1; shift ;;
                *) shift ;;
            esac
        done
        jq "$filt" "$td/base.json" > "$td/$name.pre"
        jq -S -c 'del(.sig)' "$td/$name.pre" > "$td/$name.canon"
        rm -f "${td:?}/$name.canon.sig"
        ssh-keygen -q -Y sign -f "$key" -n "$ns" "$td/$name.canon" >/dev/null 2>&1
        jq --rawfile s "$td/$name.canon.sig" '. + {sig: $s}' "$td/$name.pre" | jq "$post" > "$td/$name.json"
        { printf 'Batch PR\r\n\r\n'
          if [ "$nofence" = 0 ]; then
              for ((f = 0; f < fences; f++)); do printf '```flow009.receipt/1\r\n'; cat "$td/$name.json"; printf '\r\n```\r\n\r\n'; done
          else printf '```json\n%s\n```\n' "$(cat "$td/$name.json")"; fi; } > "$td/$name.body"
        jq -n --rawfile b "$td/$name.body" --arg h "$head" '{pull_request: {body: $b, head: {sha: $h}}}' > "$td/$name.event"
    }
    chk() { # (env FP=<path> sets FLOW009_PUBKEY, cov=<v> the expected coverage line) chk WANT LABEL NAME [pub] [tcid] [manid]  (runs inside the throwaway repo) -- set `why=<fragment>` before a no-row: the reason must contain it
        local want=$1 label=$2 name=$3 pub=${4:-$td/pub} tcs=${5:-$td/tcid.sh} mans=${6:-$td/manid.sh}
        n=$((n + 1)); rc=0
        out=$(cd "$td/r" && env ${FP:+FLOW009_PUBKEY="$FP"} bash "$here/$(basename "${BASH_SOURCE[0]}")" paiml/aprender "$td/$name.event" "$pub" "$tcs" "$mans" 2>&1) || rc=$?
        if { [ "$want" = ok ] && [ "$rc" = 0 ] && [ "$out" = "RECEIPT_OK=1"$'\n'"coverage=${cov:-95.1}" ]; } || { [ "$want" = no ] && [ "$rc" = 1 ] && [[ $out == "RECEIPT_OK=0 reason="*"${why:-}"* ]]; }; then
            printf 'ok    row %-2s %s\n' "$n" "$label"
        else printf 'FAIL  row %-2s (wanted %s, rc=%s, got "%s")  %s\n' "$n" "$want" "$rc" "$out" "$label"; red=1; fi
    }
    mk good .
    chk ok "green path: signed, CRLF body, 2 commits, tree, toolchain, manifest" good
    mk wrongtree '.tree="0123456789abcdef0123456789abcdef01234567"'
    why="HEAD's tree" chk no "MUTANT wrong tree, validly signed" wrongtree
    mk swapped ".commits[0].patch_id=\"$pidx\""
    why='multiset' chk no "MUTANT a patch_id swapped for another commit's, validly signed" swapped
    mk missing '.commits |= .[0:1]'
    why='receipt names' chk no "MUTANT a PR commit missing from the receipt, validly signed" missing
    mk extra ".commits += [{ticket:\"PMAT-33\",patch_id:\"$pidx\",sha:\"z\"}]"
    why='receipt names' chk no "MUTANT an extra commit in the receipt, validly signed" extra
    mk wrongticket '.commits[0].ticket="PMAT-99"'
    why='is not in the message' chk no "MUTANT a ticket not in that commit's message, validly signed" wrongticket
    mk badticket '.commits[0].ticket="JIRA-11"'
    why='not PMAT-<digits>' chk no "MUTANT a ticket not PMAT-<digits>, validly signed" badticket
    mk prefixticket '.commits[0].ticket="PMAT-1"'
    why='is not in the message' chk no "MUTANT ticket PMAT-1 is only a PREFIX of PMAT-11 in the message, validly signed" prefixticket
    mk wrongtc '.toolchain="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"'
    why='toolchain differs' chk no "MUTANT wrong toolchain, validly signed" wrongtc
    mk wrongman '.manifest="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"'
    why='manifest differs' chk no "MUTANT wrong manifest, validly signed" wrongman
    mk wrongschema '.schema="flow009.receipt/2"'
    why='schema is not' chk no "MUTANT wrong schema, validly signed" wrongschema
    mk extrakey '.surprise=1'
    why='exactly the RC-1' chk no "MUTANT an extra top-level key, validly signed" extrakey
    mk missingkey 'del(.host)'
    why='exactly the RC-1' chk no "MUTANT a missing top-level key (host), validly signed" missingkey
    mk gatekey '.gate_results.extra="x"'
    why='gate_results keys' chk no "MUTANT an unknown key inside gate_results, validly signed" gatekey
    mk commitkey '.commits[0].extra="x"'
    why='exactly {ticket' chk no "MUTANT an unknown key inside a commit, validly signed" commitkey
    mk testsfail '.gate_results.tests="fail"'
    why='not pass' chk no "MUTANT gate_results.tests=fail, validly signed" testsfail
    mk wrongrepo '.repo="paiml/other"'
    why='repo is not' chk no "MUTANT wrong repo, validly signed" wrongrepo
    mk edited . --post '.tree="0123456789abcdef0123456789abcdef01234567"'
    why='signature does not verify' chk no "MUTANT body-edited tree with the OLD sig" edited
    mk editedhost . --post '.host="evil"'
    why='signature does not verify' chk no "MUTANT body-edited host with the old sig (any field is covered)" editedhost
    mk wrongns . --ns other-namespace
    why='signature does not verify' chk no "MUTANT wrong signing namespace" wrongns
    mk unknownkey . --key "$td/other"
    why='signature does not verify' chk no "MUTANT signed by a key that is not the committed batcher.pub" unknownkey
    why='placeholder' chk no "MUTANT placeholder batcher.pub" good "$td/pub.placeholder"
    why='signature does not verify' chk no "a test-key receipt against the COMMITTED .flow009/batcher.pub" good "$here/../.flow009/batcher.pub"
    mk nofence . --nofence
    why='0 flow009.receipt/1' chk no "MUTANT no flow009.receipt/1 fenced block (a plain json fence)" nofence
    mk twofences . --fences 2
    why='2 flow009.receipt/1' chk no "MUTANT two flow009.receipt/1 blocks" twofences
    # shape/transport layers: each row isolates ONE refusal (several later checks would also catch some of them)
    sed '$d' "$td/good.body" | sed '$d' | sed '$d' | sed '$d' > "$td/unclosed.body"
    jq -n --rawfile b "$td/unclosed.body" --arg h "$head" '{pull_request: {body: $b, head: {sha: $h}}}' > "$td/unclosed.event"
    why='not closed' chk no "MUTANT the fenced block is never closed" unclosed
    jq -n --arg h "$head" '{pull_request: {body: "```flow009.receipt/1\n```\n", head: {sha: $h}}}' > "$td/emptyblock.event"
    why='empty receipt block' chk no "MUTANT an empty fenced block" emptyblock
    jq -n --arg h "$head" '{pull_request: {body: "```flow009.receipt/1\nnot json\n```\n", head: {sha: $h}}}' > "$td/notjson.event"
    why='not JSON' chk no "MUTANT a fenced block that is not JSON" notjson
    mk numtype '.batch=7'
    why='not a string' chk no "MUTANT batch is a number, validly signed" numtype
    mk uphex ".tree=\"$(printf '%s' "$tree" | tr a-f A-F)\""
    why='lowercase hex' chk no "MUTANT tree in upper-case hex, validly signed" uphex
    why='no committed' chk no "MUTANT no committed batcher.pub file" good "$td/absent.pub"
    printf 'ssh-rsa AAAAB3Nza\n' > "$td/pub.rsa"
    why='not an ssh-ed25519' chk no "MUTANT batcher.pub holds a non-ed25519 key line" good "$td/pub.rsa"
    jq 'del(.pull_request.head)' "$td/good.event" > "$td/nohead2.event"
    why='no PR head sha' chk no "MUTANT event carries no PR head" nohead2
    printf '#!/usr/bin/env bash\nprintf "not-a-hash\\n"\n' > "$td/junk.sh"
    why='printed no sha256' chk no "MUTANT toolchain-id prints junk" good "$td/pub" "$td/junk.sh"
    why='printed no sha256' chk no "MUTANT manifest-id prints junk" good "$td/pub" "$td/tcid.sh" "$td/junk.sh"
    why='no toolchain-id script' chk no "MUTANT toolchain-id script absent" good "$td/pub" "$td/absent.sh"
    why='no manifest-id script' chk no "MUTANT manifest-id script absent" good "$td/pub" "$td/tcid.sh" "$td/absent.sh"
    why='toolchain-id script failed' chk no "MUTANT toolchain-id script fails (fail closed)" good "$td/pub" "$td/bad.sh"
    why='manifest-id script failed' chk no "MUTANT manifest-id script fails (fail closed)" good "$td/pub" "$td/tcid.sh" "$td/bad.sh"
    # B1 (#5944): CI runs the helpers from the trusted dir, so they only see the PR tree if handed its root. These stubs
    # print their sha ONLY when argv[1] is the checkout root (the throwaway repo holds file f); a verifier that omits
    # the root argument gets exit 1 here, as the real helpers' wrong-root hash would mismatch.
    printf '#!/usr/bin/env bash\n[ -f "${1:-/nonexistent}/f" ] || exit 1\nprintf "%%s\\n" %s\n' "$tc" > "$td/tcroot.sh"
    printf '#!/usr/bin/env bash\n[ -f "${1:-/nonexistent}/f" ] || exit 1\nprintf "%%s\\n" %s\n' "$man" > "$td/manroot.sh"
    chk ok "the toolchain and manifest helpers are handed the PR checkout root" good "$td/pub" "$td/tcroot.sh" "$td/manroot.sh"
    why='no event file' chk no "MUTANT missing event file (fail closed)" absent
    jq -n '{push: {}}' > "$td/nopr.event"
    why='no pull_request' chk no "MUTANT event with no pull_request" nopr
    jq --arg h "$(printf '%040d' 0)" '.pull_request.head.sha=$h' "$td/good.event" > "$td/nohead.event"
    why='not in this checkout' chk no "MUTANT PR head sha not in the checkout" nohead
    # trust boundary (PMAT-1889): the key comes from the caller (FLOW009_PUBKEY = the BASE commit's copy), never the PR tree
    mkdir -p "$td/r/.flow009" && cp "$td/other.pub" "$td/r/.flow009/batcher.pub"
    mk selfsigned . --key "$td/other"
    FP=$td/pub why='signature does not verify' chk no "MUTANT the PR tree's own batcher.pub holds the attacker key that signed the receipt; FLOW009_PUBKEY pins the real key" selfsigned .flow009/batcher.pub
    FP=$td/absent.pub why='no committed' chk no "MUTANT FLOW009_PUBKEY points at a missing file (never falls back to the tree copy)" selfsigned .flow009/batcher.pub
    FP=$td/absent.pub why='no committed' chk no "MUTANT FLOW009_PUBKEY missing beats a valid pubkey argument" good "$td/pub"
    FP=$td/pub chk ok "the real key pinned by FLOW009_PUBKEY verifies a good receipt although the tree key is the attacker's" good .flow009/batcher.pub
    # coverage= line: exactly one, the receipt's value; none when absent or non-numeric (still OK)
    cov=95.1 chk ok "green path prints RECEIPT_OK=1 then exactly one coverage=95.1 line" good
    out=$(cd "$td/r" && bash "$here/$(basename "${BASH_SOURCE[0]}")" paiml/aprender "$td/good.event" "$td/pub" "$td/tcid.sh" "$td/manid.sh" 2>/dev/null)
    n=$((n + 1)); if [ "$(printf '%s\n' "$out" | grep -c '^coverage=')" = 1 ]; then printf 'ok    row %-2s exactly one ^coverage= line on stdout\n' "$n"; else printf 'FAIL  row %-2s coverage= line count\n' "$n"; red=1; fi
    mk cov87 '.gate_results.coverage="87.25"'
    cov=87.25 chk ok "coverage=87.25 is the receipt's own value" cov87
    mk covnan '.gate_results.coverage="n/a"'
    n=$((n + 1)); out=$(cd "$td/r" && bash "$here/$(basename "${BASH_SOURCE[0]}")" paiml/aprender "$td/covnan.event" "$td/pub" "$td/tcid.sh" "$td/manid.sh" 2>/dev/null); rc=$?
    if [ "$rc" = 0 ] && [ "$out" = "RECEIPT_OK=1" ]; then printf 'ok    row %-2s non-numeric coverage: verify OK, no coverage= line\n' "$n"; else printf 'FAIL  row %-2s non-numeric coverage (rc=%s "%s")\n' "$n" "$rc" "$out"; red=1; fi
    printf '\nthis self-test ran %s checks, %s failed\n' "$n" "$red"; [ "$red" -eq 0 ]
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
verify "$@"
