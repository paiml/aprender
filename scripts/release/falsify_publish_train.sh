#!/usr/bin/env bash
# falsify_publish_train.sh -- the case table and mutants for publish_train.sh (0.73.0 plan N4/A1/A4)
#
#   bash scripts/release/falsify_publish_train.sh              # every row
#   bash scripts/release/falsify_publish_train.sh --mutants    # the table plus one mutant per check
#   PUBLISH_TRAIN_SCRIPT=/path/to/mutant.sh bash ...           # run the table on another copy
#
# Evidence is produced by the REAL package_lane.sh (stub cargo, fixture metadata), so the two scripts are tested against each
# other's file formats. cargo, curl and sleep are stubs: no row touches crates.io, the network or the clock. The stub curl is
# an independent model of the sparse index (paths 1/x, 2/yy, 3/z/zzz, wx/yz/wxyz written out by hand, not computed), and the
# stub `cargo publish` records, for every upload, whether the crates it depends on were already visible in that index.
#
# Exit 0 = every row held (and, with --mutants, every mutant was caught). 1 = a row or mutant failed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SUBJECT="${PUBLISH_TRAIN_SCRIPT:-$HERE/publish_train.sh}"
LANE="$HERE/package_lane.sh"
TMP=$(mktemp -d) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
FX="$TMP/fx"
mkdir -p "$FX/bin"
pass=0
fail=0
rc=0
SHA=""
IDXURL="http://idx.test"

ok() { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); printf 'FAIL %s -- %s\n' "$1" "$2" >&2; }
expect_rc() { if [ "$rc" -eq "$2" ]; then ok; else bad "$1" "rc=$rc, wanted $2: $(tail -n 3 "$TMP/err" | tr '\n' ' ')"; fi; }
expect_err() { if grep -qF -- "$2" "$TMP/err"; then ok; else bad "$1" "stderr lacks '$2': $(tail -n 3 "$TMP/err" | tr '\n' ' ')"; fi; }
no_publish() { # name: the stub saw no `cargo publish`
    if grep -q '^publish ' "$STUB_LOG" 2> /dev/null; then bad "$1" "cargo publish was called: $(grep '^publish ' "$STUB_LOG" | head -n 2 | tr '\n' ';')"; else ok; fi
}
published() { # the crates `cargo publish` was called for, in call order
    sed -nE 's#^publish .*--manifest-path [^ ]*/([^/ ]+)/Cargo.toml.*#\1#p' "$STUB_LOG" | tr '\n' ' ' | sed 's/ $//'
}
expect_published() { local got; got=$(published); if [ "$got" = "$2" ]; then ok; else bad "$1" "published '$got', wanted '$2'"; fi; }

# ---- the stubs
cat > "$FX/stub-cargo" << 'SH'
#!/usr/bin/env bash
# stub cargo: `package ...` -> T/package/<name>-1.0.0.crate (deterministic bytes); `publish ...` -> logs, then writes the index entry
verb=$1
printf '%s %s\n' "$verb" "$*" | sed 's/^\([a-z]*\) [a-z]* /\1 /' >> "${STUB_LOG:?}"
shift
man="" tgt=""
while [ $# -gt 0 ]; do case "$1" in --manifest-path) man=$2; shift ;; --target-dir) tgt=$2; shift ;; esac; shift; done
name=$(basename "$(dirname "$man")")
case "$verb" in
package)
    mkdir -p "$tgt/package"
    printf 'crate:%s%s\n' "$name" "${STUB_SALT:-}" > "$tgt/package/$name-1.0.0.crate"
    ;;
publish)
    printf 'token-set=%s\n' "$([ -n "${CARGO_REGISTRY_TOKEN:-}" ] && echo yes || echo no)" >> "${STUB_LOG}.tok"
    # every upstream of this crate (hand-written graph x <- yy <- zzz <- wxyz) must already be visible
    case "$name" in yy) need="x" ;; zzz) need="yy" ;; wxyz) need="zzz" ;; *) need="" ;; esac
    if [ -n "$need" ] && ! [ -s "${STUB_IDX:?}/$(case "$need" in x) echo 1/x ;; yy) echo 2/yy ;; zzz) echo 3/z/zzz ;; esac)" ]; then
        echo "$name uploaded before $need was visible" >> "${STUB_LOG}.viol"
    fi
    if [ "${STUB_PUB_FAIL:-}" = "$name" ]; then echo "error: failed to publish $name: 503" >&2; exit 101; fi
    path=$(case "$name" in x) echo 1/x ;; yy) echo 2/yy ;; zzz) echo 3/z/zzz ;; wxyz) echo wx/yz/wxyz ;; esac)
    if [ "${STUB_PUB_NOINDEX:-}" != "$name" ]; then
        mkdir -p "$STUB_IDX/$(dirname "$path")"
        printf '{"name":"%s","vers":"1.0.0","yanked":false}\n' "$name" >> "$STUB_IDX/$path"
    fi
    if [ "${STUB_PUB_EXISTS:-}" = "$name" ]; then
        echo "error: failed to publish to registry at https://crates.io" >&2
        echo "Caused by: the remote server responded with an error: crate version \`1.0.0\` is already uploaded" >&2
        exit 101
    fi
    ;;
*) exit 64 ;;
esac
SH
cat > "$FX/stub-curl" << 'SH'
#!/usr/bin/env bash
# stub curl: serves $STUB_IDX/<path> for "$IDX_URL/<path>"; ".delay.<name>" = answer 404 for that many first reads of <name>
url=""
for a in "$@"; do url=$a; done
p=${url#http://idx.test/}
name=${p##*/}
echo "$name" >> "${STUB_LOG}.curl"
if [ -f "$STUB_IDX/.delay.$name" ]; then
    left=$(cat "$STUB_IDX/.delay.$name")
    if [ "$left" -gt 0 ]; then echo $((left - 1)) > "$STUB_IDX/.delay.$name"; printf '\n404'; exit 0; fi
fi
if [ -s "$STUB_IDX/$p" ]; then printf '%s\n200' "$(cat "$STUB_IDX/$p")"; else printf '\n404'; fi
SH
cat > "$FX/bin/sleep" << 'SH'
#!/usr/bin/env bash
echo "$*" >> "${STUB_LOG}.sleep"
SH
chmod +x "$FX/stub-cargo" "$FX/stub-curl" "$FX/bin/sleep"
export CARGO="$FX/stub-cargo" CURL="$FX/stub-curl" PATH="$FX/bin:$PATH"
export STUB_LOG="$FX/stub.log" STUB_IDX="$FX/idx"

jq -R -s --arg fx "$FX" '{packages: (split("\n") | map(select(length > 0)) | map(split("|") | {
    name: .[0], version: .[1], manifest_path: ($fx + "/" + .[0] + "/Cargo.toml"), publish: null,
    dependencies: ((.[2] // "") | split(",") | map(select(length > 0)) | map({name: ., kind: null}))
}))}' > "$FX/chain.json" << 'EOF'
x|1.0.0|
yy|1.0.0|x
zzz|1.0.0|yy
wxyz|1.0.0|zzz
EOF
jq '.packages |= map(if .name == "zzz" then .version = "2.0.0" else . end)' "$FX/chain.json" > "$FX/chain-v2.json"

# ---- helpers
run() {
    case "${TOK_MODE:-}" in
        none) env -u CARGO_REGISTRY_TOKEN bash "$SUBJECT" "$@" > "$TMP/outp" 2> "$TMP/err"; rc=$? ;;
        empty) env "${TOKVAR}=" bash "$SUBJECT" "$@" > "$TMP/outp" 2> "$TMP/err"; rc=$? ;;
        *) bash "$SUBJECT" "$@" > "$TMP/outp" 2> "$TMP/err"; rc=$? ;;
    esac
}
reset() { rm -rf -- "${FX:?}/idx" "${STUB_LOG:?}" "${STUB_LOG}.tok" "${STUB_LOG}.viol" "${STUB_LOG}.curl" "${STUB_LOG}.sleep"; mkdir -p "$FX/idx"; : > "$STUB_LOG"; }
# fresh evidence from the real package_lane.sh into $TMP/ev$1
evidence() {
    local o="$TMP/ev$1"
    rm -rf -- "${o:?}"
    bash "$LANE" --metadata "$FX/chain.json" --root "$FX/repo" --out "$o" --sha "$SHA" > /dev/null 2>&1 || { echo "evidence build failed" >&2; exit 2; }
    : > "$STUB_LOG"
    echo "$o"
}
# rebless DIR: recompute SHA256SUMS over order.txt + crates/*.crate and the receipt hashes, as a consistent forger/bug would
rebless() {
    local d=$1 o s
    (cd "$d" && sha256sum order.txt crates/*.crate > SHA256SUMS 2> /dev/null)
    o=$(sha256sum < "$d/order.txt"); s=$(sha256sum < "$d/SHA256SUMS")
    jq --arg o "${o%% *}" --arg s "${s%% *}" '.order_sha256 = $o | .sums_sha256 = $s' "$d/package-lane.json" > "$d/p.tmp" && mv "$d/p.tmp" "$d/package-lane.json"
}
go() { run --evidence "$1" --sha "$SHA" --root "$FX/repo" --metadata "$FX/chain.json" --index-url "$IDXURL" --poll-max 4 --poll-interval 0 "${@:2}"; }
TOKVAR=CARGO_REGISTRY_TOKEN
TOKVAL="SEKRET-VALUE-$$"
export_token() { export "$TOKVAR=$TOKVAL"; }

table() {
    local e n
    export_token

    # ---- happy path
    reset; e=$(evidence 1)
    go "$e" --receipt "$TMP/r1.json"
    expect_rc "happy rc" 0
    expect_published "publishes in order.txt order" "x yy zzz wxyz"
    if [ "$(tr '\n' ' ' < "$e/order.txt" | sed 's/ $//')" = "x yy zzz wxyz" ]; then ok; else bad "fixture" "order.txt is not the chain order"; fi
    if [ "$(grep -c '^publish .*--no-verify' "$STUB_LOG")" -eq 4 ] && [ "$(grep -c '^publish .*--locked' "$STUB_LOG")" -eq 4 ]; then ok; else bad "publish flags" "every upload needs --no-verify --locked"; fi
    if grep -E -q -- '--allow-dirty|--token' "$STUB_LOG"; then bad "publish flags" "--allow-dirty or --token passed"; else ok; fi
    if [ -s "$STUB_LOG.viol" ]; then bad "poll before next" "$(cat "$STUB_LOG.viol")"; else ok; fi
    if [ "$(grep -c yes "$STUB_LOG.tok")" -eq 4 ]; then ok; else bad "token" "cargo did not see CARGO_REGISTRY_TOKEN"; fi
    if grep -rqF "$TOKVAL" "$TMP/err" "$TMP/outp" "$TMP/r1.json"; then bad "token" "the token value was printed"; else ok; fi
    if jq -e --arg s "$SHA" '.lane == "publish-train" and .sha == $s and (.published | length) == 4 and .dry_run == false' "$TMP/r1.json" > /dev/null 2>&1; then ok; else bad "receipt" "wrong"; fi
    # a clean second run is a no-op (A4 idempotent)
    : > "$STUB_LOG"
    go "$e"; expect_rc "rerun of a finished train" 0; no_publish "rerun publishes nothing"
    expect_err "rerun says skip" "skip x 1.0.0: already in the index"

    # ---- index polling is bounded and not a tight loop; late convergence works
    reset; e=$(evidence 2); echo 2 > "$FX/idx/.delay.yy"; mkdir -p "$FX/idx"
    go "$e"; expect_rc "late index convergence" 0
    n=$(grep -c '^yy$' "$STUB_LOG.curl")
    if [ "$n" -eq 3 ]; then ok; else bad "late convergence" "yy read $n times, wanted 3 (pre-check + 2 polls)"; fi
    reset; e=$(evidence 3)
    STUB_PUB_NOINDEX=yy go "$e" --poll-interval 7
    expect_rc "index never converges" 1; expect_err "never-converge named" "yy 1.0.0 never appeared in the index after 4 read(s)"
    expect_published "stops before the dependents" "x yy"
    n=$(grep -c '^yy$' "$STUB_LOG.curl")
    if [ "$n" -eq 5 ]; then ok; else bad "poll bound" "yy read $n times, wanted 1 + poll-max 4"; fi
    if [ "$(grep -c '^7$' "$STUB_LOG.sleep")" -eq 3 ]; then ok; else bad "poll sleeps" "wanted 3 sleeps of 7s between 4 reads, got: $(tr '\n' ' ' < "$STUB_LOG.sleep")"; fi

    # ---- already published / resume (A4)
    reset; e=$(evidence 4)
    mkdir -p "$FX/idx/1"; printf '{"name":"x","vers":"1.0.0","yanked":false}\n' > "$FX/idx/1/x"
    go "$e"; expect_rc "x already in the index" 0; expect_published "x skipped, the rest published" "yy zzz wxyz"
    reset; e=$(evidence 5); echo 1 > "$FX/idx/.delay.yy"
    STUB_PUB_EXISTS=yy go "$e"; expect_rc "cargo says already uploaded" 0
    expect_published "already-uploaded does not stop the train" "x yy zzz wxyz"
    reset; e=$(evidence 6)
    STUB_PUB_FAIL=zzz go "$e"; expect_rc "publish failure stops the train" 1; expect_err "failure names the crate" "cargo publish failed for zzz"
    expect_published "nothing after the failure" "x yy zzz"
    : > "$STUB_LOG"; rm -f "$FX/stub.log.tok"
    go "$e"; expect_rc "rerun after partial failure" 0; expect_published "rerun resumes at the failed crate, no duplicate upload" "zzz wxyz"
    reset; e=$(evidence 7)
    mkdir -p "$FX/idx/1"; printf '{"name":"x","vers":"0.9.0","yanked":false}\n{"name":"x","vers":"1.0.0","yanked":true}\n' > "$FX/idx/1/x"
    go "$e"; expect_rc "only other/yanked versions in the index" 0; expect_published "a yanked or different version is not 'published'" "x yy zzz wxyz"

    # ---- dry run
    reset; e=$(evidence 8)
    TOK_MODE=none go "$e" --dry-run --receipt "$TMP/r8.json"
    expect_rc "dry-run needs no token" 0; no_publish "dry-run publishes nothing"
    expect_err "dry-run names the plan" "dry-run: would publish wxyz 1.0.0"
    if [ -e "$FX/idx/wx/yz/wxyz" ]; then bad "dry-run" "the index changed"; else ok; fi
    if jq -e '.dry_run == true and (.published | length) == 0' "$TMP/r8.json" > /dev/null 2>&1; then ok; else bad "dry-run receipt" "wrong"; fi
    if grep -c '^package ' "$STUB_LOG" | grep -qx 4; then ok; else bad "dry-run" "the bytes were not re-checked"; fi
    reset; e=$(evidence 9); printf 'x' >> "$e/crates/yy-1.0.0.crate"
    go "$e" --dry-run; expect_rc "dry-run still refuses a tampered crate" 1

    # ---- no token
    reset; e=$(evidence 10)
    TOK_MODE=none go "$e"; expect_rc "real run without a token" 2; expect_err "token refusal named" "CARGO_REGISTRY_TOKEN"
    no_publish "no token: nothing published"
    TOK_MODE=empty go "$e"; expect_rc "empty token" 2

    # ---- out of order
    reset; e=$(evidence 11); printf 'wxyz\nzzz\nyy\nx\n' > "$e/order.txt"
    go "$e"; expect_rc "reversed order.txt, stale hashes" 1; expect_err "stale hashes caught" "SHA256SUMS does not verify"; no_publish "out of order (stale): nothing published"
    reset; e=$(evidence 12); printf 'yy\nx\nzzz\nwxyz\n' > "$e/order.txt"; rebless "$e"
    go "$e"; expect_rc "out-of-order with consistent hashes" 1; expect_err "order pair named" "yy (#1) is listed before its dependency x (#2)"
    no_publish "out of order (reblessed): nothing published"

    # ---- tampered / missing / extra crates
    reset; e=$(evidence 13); printf 'x' >> "$e/crates/zzz-1.0.0.crate"
    go "$e"; expect_rc "tampered crate" 1; expect_err "tamper named" "SHA256SUMS does not verify"; no_publish "tampered: nothing published"
    reset; e=$(evidence 14); printf 'x' >> "$e/crates/zzz-1.0.0.crate"; rebless "$e"
    go "$e"; expect_rc "tampered crate with consistent hashes" 1; expect_err "bytes mismatch named" "re-packages to different bytes"; no_publish "tampered (reblessed): nothing published"
    reset; e=$(evidence 15); printf 'x' >> "$e/crates/zzz-1.0.0.crate"; rebless "$e"
    go "$e" --no-rebuild-check; expect_rc "--no-rebuild-check skips only the bytes comparison" 0
    expect_err "skip is announced" "bytes NOT compared"
    reset; e=$(evidence 16); rm -f "$e/crates/yy-1.0.0.crate"
    go "$e"; expect_rc "missing crate" 1; no_publish "missing: nothing published"
    reset; e=$(evidence 17); rm -f "$e/crates/yy-1.0.0.crate"; rebless "$e"
    go "$e"; expect_rc "missing crate with consistent hashes" 1; expect_err "missing named" "crate missing from the evidence: no crates/yy-*.crate"
    reset; e=$(evidence 18); printf 'crate:ghost\n' > "$e/crates/ghost-1.0.0.crate"; rebless "$e"
    go "$e"; expect_rc "extra crate" 1; expect_err "extra named" "crates/ghost-1.0.0.crate is not listed in order.txt"
    reset; e=$(evidence 19); printf 'crate:ghost\n' > "$e/crates/ghost-1.0.0.crate"
    go "$e"; expect_rc "extra crate not in SHA256SUMS" 1; no_publish "extra: nothing published"
    reset; e=$(evidence 20)
    STUB_SALT=drift go "$e"; expect_rc "re-package differs from evidence" 1; expect_err "drift named" "re-packages to different bytes"; no_publish "drift: nothing published"
    reset; e=$(evidence 21)
    run --evidence "$e" --sha "$SHA" --root "$FX/repo" --metadata "$FX/chain-v2.json" --index-url "$IDXURL" --poll-max 4 --poll-interval 0
    expect_rc "evidence for another version than the manifest" 1; expect_err "version named" "evidence holds a different version of zzz"; no_publish "version: nothing published"

    # ---- receipt / sha binding
    reset; e=$(evidence 22); jq '.sha = "0000000000000000000000000000000000000001"' "$e/package-lane.json" > "$e/p.tmp" && mv "$e/p.tmp" "$e/package-lane.json"
    go "$e"; expect_rc "yesterday's receipt (other sha)" 1; expect_err "receipt refusal named" "not a green package-order receipt for $SHA"; no_publish "other sha: nothing published"
    reset; e=$(evidence 23); jq '.status = "red"' "$e/package-lane.json" > "$e/p.tmp" && mv "$e/p.tmp" "$e/package-lane.json"
    go "$e"; expect_rc "red receipt" 1; no_publish "red receipt: nothing published"
    reset; e=$(evidence 24); jq '.crates = 3' "$e/package-lane.json" > "$e/p.tmp" && mv "$e/p.tmp" "$e/package-lane.json"
    go "$e"; expect_rc "receipt crate count disagrees" 1; expect_err "count named" "receipt says 3 crates"
    reset; e=$(evidence 25); jq '.order_sha256 = "00"' "$e/package-lane.json" > "$e/p.tmp" && mv "$e/p.tmp" "$e/package-lane.json"
    go "$e"; expect_rc "receipt order hash disagrees" 1; expect_err "order hash named" "order_sha256"
    reset; e=$(evidence 26); jq '.sums_sha256 = "00"' "$e/package-lane.json" > "$e/p.tmp" && mv "$e/p.tmp" "$e/package-lane.json"
    go "$e"; expect_rc "receipt sums hash disagrees" 1; expect_err "sums hash named" "sums_sha256"
    # repo moved on after the evidence was made: receipt matches --sha but HEAD does not
    reset; e=$(evidence 27); rm -rf -- "$TMP/repo2"; cp -r "$FX/repo" "$TMP/repo2"
    git -C "$TMP/repo2" -c user.email=f@x -c user.name=falsify -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --allow-empty -m moved
    go "$e" --root "$TMP/repo2"; expect_rc "repo HEAD is not the receipt sha" 1; expect_err "HEAD named" "repo HEAD is"; no_publish "HEAD moved: nothing published"
    reset; e=$(evidence 28); mkdir -p "$TMP/nogit"
    go "$e" --root "$TMP/nogit"; expect_rc "root is not a git checkout" 1
    reset; e=$(evidence 29); rm -f "$e/package-lane.json"
    go "$e"; expect_rc "no receipt" 1; expect_err "incomplete named" "evidence incomplete"
    reset; e=$(evidence 30); : > "$e/order.txt"
    go "$e"; expect_rc "empty order.txt" 1

    # ---- usage
    reset; e=$(evidence 31)
    run --sha "$SHA"; expect_rc "no --evidence" 2
    run --evidence "$e"; expect_rc "no --sha" 2
    run --evidence "$e" --sha nothex; expect_rc "bad --sha" 2
    run --evidence "$e" --sha "$SHA" --token abc; expect_rc "there is no --token flag" 2
    run --evidence "$e" --sha "$SHA" --poll-max 0; expect_rc "bad --poll-max" 2
    run --evidence "$e" --sha "$SHA" --poll-interval x; expect_rc "bad --poll-interval" 2
    run --evidence "$TMP/nope" --sha "$SHA"; expect_rc "evidence dir missing" 2
    run --evidence "$e" --sha; expect_rc "flag without value" 2
    no_publish "usage errors publish nothing"
}

setup_repo() {
    mkdir -p "$FX/repo" && git -C "$FX/repo" init -q && git -C "$FX/repo" -c user.email=f@x -c user.name=falsify -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --allow-empty -m fx || return 1
    SHA=$(git -C "$FX/repo" rev-parse HEAD)
}

mutants() {
    local m n=0 caught=0 tgt
    declare -a MUT=(
        # sed expression@@label (applied to a copy of the subject)
        's/publish --no-verify --locked/publish --locked/@@--no-verify dropped from publish'
        's/publish --no-verify --locked/publish --no-verify/@@--locked dropped from publish'
        's/^        if \[\[ "\$rc" -eq 0 \]\]; then/        if false; then/@@already-in-index skip dead'
        's/wait_visible "\$c" "\$ver" || die/true || die/@@wait for the index dead'
        's/already (uploaded|exists)/never-matches-xyz/@@already-uploaded not tolerated'
        's/ || die "SHA256SUMS does not verify (a listed file is missing or changed)"/ || true/@@SHA256SUMS check dead'
        's/\.status == "green" and \.sha == \$s/.status == "green"/@@receipt sha binding dead'
        's/ || die "order\.txt is not a valid dependency order"/ || true/@@order check dead'
        's/\[\[ "\${have%% \*}" == "\${want%% \*}" \]\] ||/true ||/@@bytes comparison dead'
        's/\[\[ "\$exp_c" -ge 1 \]\] ||/true ||/@@missing-crate check dead'
        's/-z "\${CARGO_REGISTRY_TOKEN:-}"/-z "x\${CARGO_REGISTRY_TOKEN:-}x"/@@token requirement dead'
        's/^        if \[\[ "\$dry" -eq 1 \]\]; then/        if false; then/@@dry-run still publishes'
        's/sleep "\$poll_int"/true/@@poll without sleeping'
        's/\[\[ "\$head" == "\$sha" \]\] || die/true || die/@@HEAD binding dead'
        's/grep -qxF -- "\$c" <<< "\$list" || die/true || die/@@extra-crate check dead'
        '/^train()/,/^}/ s#for c in "\${ORDER\[@\]}"#for c in $(printf "%s\\n" "${ORDER[@]}" | tac)#@@train ignores order.txt order'
        's/\[\[ "\$f_count" -eq "\$n" \]\] ||/true ||/@@crate count check dead'
    )
    for m in "${MUT[@]}"; do
        n=$((n + 1)); tgt="$TMP/mut$n.sh"
        sed -e "${m%%@@*}" "$HERE/publish_train.sh" > "$tgt"
        if cmp -s "$tgt" "$HERE/publish_train.sh"; then printf 'MUTANT %s did not apply: %s\n' "$n" "${m#*@@}" >&2; fail=$((fail + 1)); continue; fi
        # the mutant runs the whole table in a child falsifier so its counters stay separate
        if PUBLISH_TRAIN_SCRIPT="$tgt" bash "${BASH_SOURCE[0]}" > /dev/null 2>&1; then
            printf 'MUTANT SURVIVED: %s\n' "${m#*@@}" >&2; fail=$((fail + 1))
        else caught=$((caught + 1)); fi
    done
    printf 'mutants: %s caught\n' "$caught"
}

setup_repo || { echo "cannot build the fixture repo" >&2; exit 2; }
table
printf 'falsify_publish_train: %s rows held, %s failed\n' "$pass" "$fail"
if [ "${1:-}" = "--mutants" ] && [ -z "${PUBLISH_TRAIN_SCRIPT:-}" ]; then mutants; fi
[ "$fail" -eq 0 ]
