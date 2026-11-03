#!/usr/bin/env bash
# falsify_package_lane.sh -- the case table and mutants for package_lane.sh (0.73.0 plan N1/A3, defect #3892)
#
#   bash scripts/release/falsify_package_lane.sh               # every row, fixtures + the real repo graph (list mode)
#   bash scripts/release/falsify_package_lane.sh --mutants     # the table plus one mutant per check
#   PACKAGE_LANE_SCRIPT=/path/to/mutant.sh bash ...            # run the table on another copy
#
# Fixture rows feed hand-written `cargo metadata` documents through --metadata and stub cargo through CARGO=, so no row
# packages anything for real. The real-repo rows run --list-only only (cargo metadata, no cargo package). The one real
# package run is a separate, documented command: package_lane.sh --out DIR --only a,b,c.
# Defect #3892: aprender-profile requires aprender-db, yet scripts/cascade-publish.sh TIERS[] lists profile (T3) long before db (T13).
#
# Exit 0 = every row held (and, with --mutants, every mutant was caught). 1 = a row or mutant failed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
ROOT=$(cd "$HERE/../.." && pwd) || exit 2
SUBJECT="${PACKAGE_LANE_SCRIPT:-$HERE/package_lane.sh}"
TMP=$(mktemp -d) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
FX="$TMP/fx"
mkdir -p "$FX"
pass=0
fail=0
rc=0

ok() { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); printf 'FAIL %s -- %s\n' "$1" "$2" >&2; }

# mk_meta: stdin = one crate per line "name|version|publish|dep,dev:dep,build:dep,ren:dep"; stdout = a metadata document.
# publish: "-" unset, "false" -> [], anything else -> [that registry]
mk_meta() {
    jq -R -s --arg fx "$FX" '{packages: (split("\n") | map(select(length > 0)) | map(split("|") | {
        name: .[0], version: .[1], manifest_path: ($fx + "/" + .[0] + "/Cargo.toml"),
        publish: (if .[2] == "-" then null elif .[2] == "false" then [] else [.[2]] end),
        dependencies: ((.[3] // "") | split(",") | map(select(length > 0)) | map(
            if startswith("dev:") then {name: .[4:], kind: "dev"}
            elif startswith("build:") then {name: .[6:], kind: "build"}
            elif startswith("ren:") then {name: .[4:], kind: null, rename: "alias"}
            else {name: ., kind: null} end))
    }))}'
}

# run NAME ARGS... : runs the subject, sets rc, keeps stderr in $TMP/err
run() { bash "$SUBJECT" "$@" > "$TMP/outp" 2> "$TMP/err"; rc=$?; }
expect_rc() { # name want
    if [ "$rc" -eq "$2" ]; then ok; else bad "$1" "rc=$rc, wanted $2: $(tail -n 3 "$TMP/err" | tr '\n' ' ')"; fi
}
expect_err() { # name pattern
    if grep -qF -- "$2" "$TMP/err"; then ok; else bad "$1" "stderr lacks '$2': $(tail -n 3 "$TMP/err" | tr '\n' ' ')"; fi
}
expect_order() { # name outdir "space separated order"
    local got
    got=$(tr '\n' ' ' < "$2/order.txt" 2> /dev/null | sed 's/ $//')
    if [ "$got" = "$3" ]; then ok; else bad "$1" "order '$got', wanted '$3'"; fi
}
meta() { mk_meta > "$FX/$1.json"; }
fresh() { rm -rf -- "${TMP:?}/o$1"; echo "$TMP/o$1"; }

table() {
    local o f

    # ---- order rows
    meta chain <<'EOF'
a|1.0.0|-|b
b|1.0.0|-|c
c|1.0.0|-|
EOF
    o=$(fresh 1); run --metadata "$FX/chain.json" --out "$o" --list-only
    expect_rc "chain rc" 0; expect_order "chain: dependencies first" "$o" "c b a"

    # THE #3892 SHAPE: profile requires db; a tier table puts profile first
    meta d3892 <<'EOF'
aprender-core|0.72.3|-|
aprender-profile|0.72.3|-|aprender-core,aprender-db
aprender-db|0.72.3|-|aprender-core
EOF
    o=$(fresh 2); run --metadata "$FX/d3892.json" --out "$o" --list-only
    expect_rc "#3892 derive rc" 0; expect_order "#3892: db is ordered before profile" "$o" "aprender-core aprender-db aprender-profile"
    printf 'aprender-core\naprender-profile\naprender-db\n' > "$FX/tiers3892.txt"
    run --metadata "$FX/d3892.json" --verify-order "$FX/tiers3892.txt"
    expect_rc "#3892 tier-shaped order refused" 1; expect_err "#3892 names the pair" "aprender-profile (#2) is listed before its dependency aprender-db (#3)"
    printf 'aprender-core\naprender-db\naprender-profile\n' > "$FX/good3892.txt"
    run --metadata "$FX/d3892.json" --verify-order "$FX/good3892.txt"; expect_rc "#3892 graph order accepted" 0

    meta diamond <<'EOF'
top|1.0.0|-|right,left
right|1.0.0|-|base
left|1.0.0|-|base
base|1.0.0|-|
EOF
    o=$(fresh 3); run --metadata "$FX/diamond.json" --out "$o" --list-only
    expect_order "diamond: level then name" "$o" "base left right top"
    meta diamond_rev <<'EOF'
base|1.0.0|-|
left|1.0.0|-|base
right|1.0.0|-|base
top|1.0.0|-|left,right
EOF
    f=$(fresh 4); run --metadata "$FX/diamond_rev.json" --out "$f" --list-only
    if cmp -s "$o/order.txt" "$f/order.txt"; then ok; else bad "determinism" "input order changed the output order"; fi

    meta cycle <<'EOF'
a|1.0.0|-|b
b|1.0.0|-|a
c|1.0.0|-|
EOF
    o=$(fresh 5); run --metadata "$FX/cycle.json" --out "$o" --list-only
    expect_rc "cycle refused" 1; expect_err "cycle named" "dependency cycle"
    if [ -e "$o/order.txt" ] && [ -s "$o/order.txt" ]; then bad "cycle" "an order.txt was written for a cyclic graph"; else ok; fi

    meta devedge <<'EOF'
a|1.0.0|-|dev:b
b|1.0.0|-|a
EOF
    o=$(fresh 6); run --metadata "$FX/devedge.json" --out "$o" --list-only
    expect_rc "dev-dependency cycle is not a cycle" 0; expect_order "dev edge carries no order" "$o" "a b"

    meta buildedge <<'EOF'
a|1.0.0|-|build:b
b|1.0.0|-|
EOF
    o=$(fresh 7); run --metadata "$FX/buildedge.json" --out "$o" --list-only
    expect_order "build-dependency is an edge" "$o" "b a"

    meta renamed <<'EOF'
a|1.0.0|-|ren:b
b|1.0.0|-|
EOF
    o=$(fresh 8); run --metadata "$FX/renamed.json" --out "$o" --list-only
    expect_order "renamed dependency orders by package name" "$o" "b a"

    meta unpub <<'EOF'
a|1.0.0|-|x,y,ext
x|1.0.0|false|a
y|1.0.0|other|
z|1.0.0|crates-io|
EOF
    o=$(fresh 9); run --metadata "$FX/unpub.json" --out "$o" --list-only
    expect_order "publish=false / other registry excluded, outside deps ignored" "$o" "a z"

    meta ws1 <<'EOF'
facade|0.4.0|-|upstream
EOF
    meta ws2 <<'EOF'
upstream|0.72.3|-|
EOF
    o=$(fresh 10); run --metadata "$FX/ws1.json" --metadata "$FX/ws2.json" --out "$o" --list-only
    expect_order "edge across two workspaces" "$o" "upstream facade"

    # ---- vacuity, --only, bad input
    meta none <<'EOF'
x|1.0.0|false|
EOF
    o=$(fresh 11); run --metadata "$FX/none.json" --out "$o" --list-only
    expect_rc "empty set is red" 1; expect_err "empty set named vacuity" "vacuity"
    o=$(fresh 12); run --metadata "$FX/chain.json" --out "$o" --list-only --min-crates 4
    expect_rc "below --min-crates is red" 1
    o=$(fresh 13); run --metadata "$FX/chain.json" --out "$o" --list-only --only a,c
    expect_rc "--only rc" 0; expect_order "--only restricts the set" "$o" "a c"
    o=$(fresh 14); run --metadata "$FX/chain.json" --out "$o" --list-only --only a,nope
    expect_rc "--only unknown name is red" 1; expect_err "--only names the unknown" "nope"
    printf 'not json' > "$FX/junk.json"
    o=$(fresh 15); run --metadata "$FX/junk.json" --out "$o" --list-only
    expect_rc "non-metadata input is red" 1; expect_err "non-metadata named" "not a cargo metadata document"

    # ---- verify-order rows (A3)
    printf '# comment\n\nc\nb\na\n' > "$FX/ok.txt"
    run --metadata "$FX/chain.json" --verify-order "$FX/ok.txt"; expect_rc "valid order, comments and blanks tolerated" 0
    printf 'a\nb\nc\n' > "$FX/rev.txt"
    run --metadata "$FX/chain.json" --verify-order "$FX/rev.txt"; expect_rc "dependent before dependency refused" 1; expect_err "reversed names pair" "a (#1) is listed before its dependency b (#2)"
    printf 'c\nb\nb\na\n' > "$FX/dup.txt"
    run --metadata "$FX/chain.json" --verify-order "$FX/dup.txt"; expect_rc "duplicate refused" 1; expect_err "duplicate named" "duplicate entry: b"
    printf 'c\nb\na\nghost\n' > "$FX/unk.txt"
    run --metadata "$FX/chain.json" --verify-order "$FX/unk.txt"; expect_rc "unknown crate refused" 1; expect_err "unknown named" "ghost"
    printf 'c\nb\n' > "$FX/miss.txt"
    run --metadata "$FX/chain.json" --verify-order "$FX/miss.txt"; expect_rc "missing crate refused" 1; expect_err "missing named" "missing from the order: a"
    : > "$FX/empty.txt"
    run --metadata "$FX/chain.json" --verify-order "$FX/empty.txt"; expect_rc "empty order file refused" 1

    # ---- usage
    run --metadata "$FX/chain.json" --list-only; expect_rc "no --out" 2
    run --metadata "$FX/chain.json" --out "$TMP/u" --frobnicate x; expect_rc "unknown argument" 2
    run --metadata "$FX/chain.json" --out "$TMP/u" --sha nothex; expect_rc "bad --sha" 2
    run --metadata "$FX/chain.json" --out "$TMP/u" --min-crates x; expect_rc "bad --min-crates" 2
    run --metadata "$FX/chain.json" --out; expect_rc "flag without value" 2
}

# a throwaway git checkout + a stub cargo for the package-mode rows
package_rows() {
    local repo="$FX/repo" head o
    mkdir -p "$repo" && git -C "$repo" init -q && git -C "$repo" -c user.email=f@x -c user.name=falsify -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q --allow-empty -m fx || return 1
    head=$(git -C "$repo" rev-parse HEAD)
    cat > "$FX/stub-cargo" <<'SH'
#!/usr/bin/env bash
# stub: `cargo package --no-verify --locked --manifest-path M --target-dir T` -> T/package/<name>-<ver>.crate
[ "$1" = package ] || exit 64
printf '%s\n' "$*" >> "${STUB_LOG:?}"
shift
man="" tgt=""
while [ $# -gt 0 ]; do case "$1" in --manifest-path) man=$2; shift ;; --target-dir) tgt=$2; shift ;; esac; shift; done
name=$(basename "$(dirname "$man")")
[ "${STUB_FAIL:-}" = "$name" ] && { echo "error: failed to package $name" >&2; exit 101; }
[ "${STUB_NOFILE:-}" = "$name" ] && exit 0
mkdir -p "$tgt/package"
printf 'crate:%s\n' "$name" > "$tgt/package/$name-1.0.0.crate"
SH
    chmod +x "$FX/stub-cargo"
    export CARGO="$FX/stub-cargo" STUB_LOG="$FX/stub.log"

    : > "$STUB_LOG"; o=$(fresh 20)
    run --metadata "$FX/chain.json" --root "$repo" --out "$o" --sha "$head"
    expect_rc "package happy rc" 0
    if (cd "$o" && sha256sum --quiet -c SHA256SUMS > /dev/null 2>&1); then ok; else bad "SHA256SUMS" "does not verify"; fi
    if [ "$(wc -l < "$o/SHA256SUMS")" -eq 4 ]; then ok; else bad "SHA256SUMS" "expected order.txt + 3 crates"; fi
    if jq -e --arg s "$head" '.lane == "package-order" and .sha == $s and .scope == "full" and .status == "green" and .crates == 3' "$o/package-lane.json" > /dev/null 2>&1; then ok; else bad "receipt" "missing or wrong"; fi
    if [ "$(sed 's/ .*//; s/^package //' "$STUB_LOG" | wc -l)" -eq 3 ] && [ "$(grep -c -- '--no-verify' "$STUB_LOG")" -eq 3 ] && [ "$(grep -c -- '--locked' "$STUB_LOG")" -eq 3 ]; then ok; else bad "cargo flags" "every crate must be packaged with --no-verify --locked"; fi
    if grep -q -- '--allow-dirty' "$STUB_LOG"; then bad "cargo flags" "--allow-dirty was passed"; else ok; fi
    local seq
    seq=$(sed -E 's#.*--manifest-path [^ ]*/([^/ ]+)/Cargo.toml.*#\1#' "$STUB_LOG" | tr '\n' ' ' | sed 's/ $//')
    if [ "$seq" = "$(tr '\n' ' ' < "$o/order.txt" | sed 's/ $//')" ]; then ok; else bad "package order" "cargo ran in '$seq', not the order.txt order"; fi
    # A2-shaped tamper: a changed .crate, then a changed order.txt, must both fail the sums
    cp -r "$o" "$TMP/tamper1"; printf 'x' >> "$TMP/tamper1/crates/b-1.0.0.crate"
    if (cd "$TMP/tamper1" && sha256sum --quiet -c SHA256SUMS > /dev/null 2>&1); then bad "tamper" "altered .crate still verifies"; else ok; fi
    cp -r "$o" "$TMP/tamper2"; printf 'a\nb\nc\n' > "$TMP/tamper2/order.txt"
    if (cd "$TMP/tamper2" && sha256sum --quiet -c SHA256SUMS > /dev/null 2>&1); then bad "tamper" "altered order.txt still verifies"; else ok; fi

    # an --only run must not leave the receipt of a full run: scope names the restriction
    o=$(fresh 25)
    run --metadata "$FX/chain.json" --root "$repo" --out "$o" --sha "$head" --only a,c
    expect_rc "--only package rc" 0
    if jq -e '.scope == "only:a,c" and .crates == 2 and .status == "green"' "$o/package-lane.json" > /dev/null 2>&1; then ok; else bad "receipt scope" "an --only run must carry scope only:a,c, got $(jq -c .scope "$o/package-lane.json" 2> /dev/null)"; fi

    # a failing crate: no SHA256SUMS, no receipt, and a stale receipt from an earlier run is gone
    o=$(fresh 21); mkdir -p "$o"; echo '{"stale":true}' > "$o/package-lane.json"; echo stale > "$o/SHA256SUMS"
    STUB_FAIL=b run --metadata "$FX/chain.json" --root "$repo" --out "$o"
    expect_rc "failed package is red" 1; expect_err "failure names the crate" "cargo package failed for b"
    if [ -e "$o/package-lane.json" ] || [ -e "$o/SHA256SUMS" ]; then bad "red run" "left a receipt or SHA256SUMS behind"; else ok; fi

    o=$(fresh 22)
    STUB_NOFILE=c run --metadata "$FX/chain.json" --root "$repo" --out "$o"
    expect_rc "success without a .crate is red" 1; expect_err "no-file named" "wrote no c-1.0.0.crate"
    if [ -e "$o/package-lane.json" ]; then bad "no-file run" "wrote a receipt"; else ok; fi

    o=$(fresh 23)
    run --metadata "$FX/chain.json" --root "$repo" --out "$o" --sha 0000000000000000000000000000000000000000
    expect_rc "--sha other than HEAD is red" 1
    if [ -e "$o/package-lane.json" ]; then bad "sha mismatch" "wrote a receipt"; else ok; fi

    run --metadata "$FX/chain.json" --root "$repo" --out "$repo/evidence"
    expect_rc "--out inside the repo refused" 2
    if [ -e "$repo/evidence" ]; then bad "out inside repo" "created the directory"; else ok; fi

    mkdir -p "$FX/nogit"
    run --metadata "$FX/chain.json" --root "$FX/nogit" --out "$(fresh 24)"
    expect_rc "no git checkout, no sha to bind" 1
    unset CARGO STUB_LOG
}

# the real repo graph, list mode only
real_rows() {
    local o n
    o=$(fresh 30); run --root "$ROOT" --out "$o" --list-only
    expect_rc "real repo: derive + self-verify" 0
    n=$(wc -l < "$o/order.txt" 2> /dev/null || echo 0)
    if [ "$n" -ge 70 ]; then ok; else bad "real repo" "only $n crates enumerated"; fi
    local pd pp
    pd=$(grep -n -x 'aprender-db' "$o/order.txt" | cut -d: -f1); pp=$(grep -n -x 'aprender-profile' "$o/order.txt" | cut -d: -f1)
    if [ -n "$pd" ] && [ -n "$pp" ] && [ "$pd" -lt "$pp" ]; then ok; else bad "real repo #3892" "aprender-db (#${pd:-none}) must come before aprender-profile (#${pp:-none})"; fi
    if grep -q -x 'provable-contracts' "$o/order.txt"; then ok; else bad "real repo" "the crates/facades workspace is not enumerated"; fi
    # the real TIERS[] table of cascade-publish.sh, flattened, must be refused for exactly the #3892 pair (skipped once the table is retired)
    if grep -qE '^TIERS\[[0-9]+\]=' "$ROOT/scripts/cascade-publish.sh" 2> /dev/null; then
        sed -nE 's/^TIERS\[([0-9]+)\]="(.*)"$/\1 \2/p' "$ROOT/scripts/cascade-publish.sh" | sort -n | cut -d' ' -f2- | tr ' ' '\n' > "$TMP/tiers.txt"
        run --root "$ROOT" --verify-order "$TMP/tiers.txt"
        expect_rc "real TIERS[] order is refused (#3892)" 1; expect_err "real TIERS[] names profile before db" "aprender-profile ("
        if grep -qE 'aprender-profile \(.*before its dependency aprender-db' "$TMP/err"; then ok; else bad "real TIERS[]" "profile/db pair not named"; fi
    fi
}

run_all() { fail=0; pass=0; table; package_rows; real_rows; }

mutants() {
    local m n=0 caught=0 tgt
    declare -a MUT=(
        # sed expression@@label (applied to a copy of the subject)
        's/\$pos\[\.\] > \$pos\[\$c\]/$pos[.] < $pos[$c]/@@verify comparator inverted'
        's/select(\.kind != "dev" and /select(/@@dev edges counted'
        's/\$fix\.lv\[\.\], \.\]/.]/@@level ordering dropped'
        's/select(\$pos\[\.\] == null) | "missing/select($pos[.] == 9999) | "missing/@@missing check dead'
        's/ --no-verify / /@@--no-verify dropped'
        's/ --locked / /@@--locked dropped'
        's/\[\[ "\$n" -ge "\$floor" \]\] || die/true || die/@@vacuity floor dead'
        's/select(publishable)/select(true)/@@publish filter dropped'
        's/scope="only:$only"/scope=full/@@scope always full'
        's/\.publish == null or (\.publish | index("crates-io") != null)/true/@@publishable() always true'
    )
    # Every copy lives at $TMP/mroot/scripts/release/<name>.sh, the layout of the real repo, so the copy's default root is a
    # real directory rather than "/" -- and the table still passes --root "$ROOT" wherever a row needs the real repo.
    mkdir -p "$TMP/mroot/scripts/release"
    # CONTROL: the unmutated script, copied the same way, must pass every row; otherwise a "caught" mutant proves nothing.
    ctl="$TMP/mroot/scripts/release/control.sh"
    cp "$HERE/package_lane.sh" "$ctl"
    if ! PACKAGE_LANE_SCRIPT="$ctl" bash "${BASH_SOURCE[0]}" > "$TMP/ctl.out" 2> "$TMP/ctl.err"; then
        printf 'CONTROL FAILED: the unmutated copy does not pass the table; every verdict below would be vacuous:\n' >&2
        head -n 5 "$TMP/ctl.err" >&2; fail=$((fail + 1)); return
    fi
    printf 'control: unmutated copy passes every row (%s)\n' "$(tail -n 1 "$TMP/ctl.out")"
    for m in "${MUT[@]}"; do
        n=$((n + 1)); tgt="$TMP/mroot/scripts/release/mut$n.sh"
        sed -e "${m%%@@*}" "$HERE/package_lane.sh" > "$tgt"
        if cmp -s "$tgt" "$HERE/package_lane.sh"; then printf 'MUTANT %s did not apply: %s\n' "$n" "${m#*@@}" >&2; fail=$((fail + 1)); continue; fi
        # the mutant runs the whole table in a child falsifier so its counters stay separate
        if PACKAGE_LANE_SCRIPT="$tgt" bash "${BASH_SOURCE[0]}" > /dev/null 2> "$TMP/mut$n.err"; then
            printf 'MUTANT SURVIVED: %s\n' "${m#*@@}" >&2; fail=$((fail + 1))
        elif grep -qF 'no Cargo.toml under' "$TMP/mut$n.err"; then
            printf 'MUTANT %s "caught" only by a harness artifact (no Cargo.toml under): %s\n' "$n" "${m#*@@}" >&2; fail=$((fail + 1))
        else
            caught=$((caught + 1))
            printf 'mutant %s (%s) caught by: %s\n' "$n" "${m#*@@}" "$(grep -m1 '^FAIL ' "$TMP/mut$n.err" | cut -c1-110)"
        fi
    done
    printf 'mutants: %s of %s caught\n' "$caught" "$n"
}

run_all
printf 'falsify_package_lane: %s rows held, %s failed\n' "$pass" "$fail"
if [ "${1:-}" = "--mutants" ] && [ -z "${PACKAGE_LANE_SCRIPT:-}" ]; then mutants; fi
[ "$fail" -eq 0 ]
