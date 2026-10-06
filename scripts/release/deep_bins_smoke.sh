#!/usr/bin/env bash
# deep_bins_smoke.sh -- the deep-bins-smoke lane's producer logic, in bash + jq (#4718).
#
# WHY. nightly_train.sh reads one verdict lane per release-day deep check. deep-bins-smoke
#   ("every workspace [[bin]] builds and answers --version at THIS commit", #4189) had no
#   producer on main, so it printed not_measured every night. autopilot's T-1 step runs it
#   through scripts/nightly_manifest.py; operator C301 bans new Python on a build or gate
#   path, so the nightly producer carries the same verdicts here instead.
#
# USAGE
#   deep_bins_smoke.sh bins METADATA.json [list|cargo]
#       list  (default) bin names, comma-joined, sorted
#       cargo the cargo arguments that build exactly those bins (-p ... --bin ... --features ...)
#   deep_bins_smoke.sh smoke SHA BINS BIN_DIR [VERSION]
#       one TSV row per bin: bin, verdict, reason, first --version line. Exit 0 iff every bin
#       is green AND there is at least one bin. A smoke over zero bins measured nothing: exit 1.
#   deep_bins_smoke.sh meta METADATA.json version|target_dir
#       the root package version every bin must print, or cargo's target dir
#   deep_bins_smoke.sh --self-test
#
# THE SET (same rule as nightly_manifest.py workspace_bins): every workspace member's [[bin]],
#   one per name. A name defined twice resolves to the package NOT at the workspace root (the
#   facade), so one cargo build never has two outputs racing for target/release/<bin>. Two
#   non-root packages defining one name is an error, never a silent pick.
#
# THE VERDICT per bin (same reasons as nightly_manifest.py record):
#   missing-artifact  BIN_DIR/<bin> is not a file
#   version-failed    --version (empty cwd, no stdin, 20 s) exits non-zero, times out, or does
#                     not name VERSION as a whole version
#   version-no-sha    the line names no "(<7-40 hex>" build SHA (#4219)
#   version-mismatch  the SHA it names is not a prefix of SHA
#   green             otherwise
#
# Errors are carried by return status, never `set -e`: every probe of a bin is EXPECTED to fail
# for some bins, and -e would turn the first red bin into a missing row for every bin after it.

dbs_bins() {
    local meta="$1" fmt="${2:-list}"
    [ -f "$meta" ] || { echo "deep_bins_smoke: no metadata file: $meta" >&2; return 2; }
    jq -r --arg fmt "$fmt" '
      (.workspace_root + "/Cargo.toml") as $root
      | (.workspace_members // []) as $mem
      | [ .packages[]
          | select(($mem | length) == 0 or (.id as $i | $mem | index($i)))
          | . as $p
          | .targets[]
          | select(.kind | index("bin"))
          # a package whose build SHA is behind an optional `build-sha` feature (#4604 C2) is built
          # with it, as binary-release.yml builds it; without it --version reads "+no-git" (#4219)
          | {b: .name, pkg: $p.name,
             f: ((.["required-features"] // []) + (if ($p.features // {}) | has("build-sha") then ["build-sha"] else [] end)
                 | reduce .[] as $q ([]; if index([$q]) then . else . + [$q] end)),
             root: ($p.manifest_path == $root)} ]
      | group_by(.b)
      | map( (map(select(.root | not))) as $nr
             | if ($nr | map(.pkg) | unique | length) > 1
               then error("bin \(.[0].b) is defined by both \($nr[0].pkg) and \($nr[1].pkg)")
               elif ($nr | length) > 0 then $nr[0] else .[0] end )
      | sort_by(.b)
      | if $fmt == "list" then map(.b) | join(",")
        elif $fmt == "cargo" then
          ( reduce .[] as $x ([]; if index($x.pkg) then . else . + [$x.pkg] end) | map("-p", .) )
          + ( map(. as $x | ["--bin", $x.b] + ($x.f | map("--features", "\($x.pkg)/\(.)"))) | add // [] )
          | join(" ")
        else error("format must be list or cargo, got \($fmt)") end
    ' "$meta"
}

# dbs_meta META version|target_dir -> the workspace ROOT package version (what every bin must print),
#   or cargo's own target dir (so a runner that redirects target-dir is still read, #4659). Empty = error.
dbs_meta() {
    local meta="$1" what="$2" out
    [ -f "$meta" ] || { echo "deep_bins_smoke: no metadata file: $meta" >&2; return 2; }
    case "$what" in
        version) out=$(jq -r '(.workspace_root + "/Cargo.toml") as $r | [.packages[] | select(.manifest_path == $r) | .version][0] // empty' "$meta") || return 2 ;;
        target_dir) out=$(jq -r '.target_directory // empty' "$meta") || return 2 ;;
        *) echo "deep_bins_smoke: meta wants version or target_dir, got $what" >&2; return 2 ;;
    esac
    [ -n "$out" ] || { echo "deep_bins_smoke: metadata has no $what" >&2; return 1; }
    printf '%s\n' "$out"
}

# dbs_probe EXE -> rc on line 1, first non-empty output line on line 2 (stdout, else stderr)
dbs_probe() {
    local exe="$1" cwd out rc
    cwd=$(mktemp -d) || return 2
    out=$(cd "$cwd" && timeout 20 "$exe" --version </dev/null 2>"$cwd/.err"); rc=$?
    [ -n "$out" ] || out=$(cat "$cwd/.err")
    rm -rf -- "${cwd:?}"
    printf '%s\n%s\n' "$rc" "$(printf '%s\n' "$out" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' | sed -n 1p)"
}

dbs_verdict() {
    local sha="$1" version="$2" exe="$3" probe rc line vsha
    if [ ! -f "$exe" ]; then printf 'red\tmissing-artifact\t\n'; return 0; fi
    probe=$(dbs_probe "$exe")
    rc=$(printf '%s\n' "$probe" | sed -n 1p)
    line=$(printf '%s\n' "$probe" | sed -n 2p)
    if [ "$rc" != 0 ]; then printf 'red\tversion-failed\t%s\n' "rc=$rc: $line"; return 0; fi
    # grep reads all of its input here (no -q), so the printf upstream never takes a SIGPIPE.
    if [ -n "$version" ] && [ -z "$(printf '%s \n' "$line" | grep -oP "(?<![\\d.])\\Q${version}\\E(?![\\d])")" ]; then
        printf 'red\tversion-failed\t%s\n' "wanted $version: $line"; return 0
    fi
    vsha=$(printf '%s \n' "$line" | grep -oP '\(\K[0-9a-f]{7,40}(?=[\s),])' | sed -n 1p)
    if [ -z "$vsha" ]; then printf 'red\tversion-no-sha\t%s\n' "$line"; return 0; fi
    case "$sha" in
        "$vsha"*) printf 'green\t-\t%s\n' "$line" ;;
        *) printf 'red\tversion-mismatch\t%s\n' "prints $vsha, built at ${sha:0:9}: $line" ;;
    esac
}

dbs_smoke() {
    local sha="$1" bins="$2" dir="$3" version="${4:-}" b n=0 red=0 v
    local -a list=()
    [ -n "$sha" ] || { echo "deep_bins_smoke: smoke needs a SHA" >&2; return 2; }
    dir=$(cd "$dir" 2>/dev/null && pwd) || { echo "deep_bins_smoke: no bin dir: $3" >&2; return 2; }
    IFS=',' read -r -a list <<< "$bins"
    printf 'bin\tverdict\treason\tversion_output\n'
    for b in "${list[@]}"; do
        [ -n "$b" ] || continue
        n=$((n + 1))
        v=$(dbs_verdict "$sha" "$version" "$dir/$b")
        case "$v" in green*) ;; *) red=$((red + 1)) ;; esac
        printf '%s\t%s\n' "$b" "$v"
    done
    if [ "$n" -eq 0 ]; then echo "SMOKE RED: zero bins: a smoke over nothing measured nothing" >&2; return 1; fi
    if [ "$red" -gt 0 ]; then echo "SMOKE RED: $red of $n bins" >&2; return 1; fi
    echo "SMOKE GREEN: $n of $n bins at ${sha:0:9}" >&2
    return 0
}

dbs_self_test() {
    local t rc fail=0 sha=0123456789abcdef0123456789abcdef01234567
    t=$(mktemp -d) || return 2
    expect() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: want [$2] got [$3]"; fail=1; fi; }
    mkbin() { printf '#!/bin/sh\n%s\n' "$2" > "$t/bin/$1"; chmod +x "$t/bin/$1"; }
    mkdir -p "$t/bin"
    cat > "$t/meta.json" <<'J'
{"workspace_root":"/w","target_directory":"/tgt","workspace_members":["a","f","c","x"],
 "packages":[
  {"id":"f","name":"aprender","version":"0.70.2","manifest_path":"/w/Cargo.toml","targets":[{"name":"apr","kind":["bin"]}]},
  {"id":"a","name":"apr-cli","version":"9.9.9","manifest_path":"/w/crates/apr-cli/Cargo.toml","targets":[{"name":"apr","kind":["bin"]},{"name":"apr_cli","kind":["lib"]}]},
  {"id":"c","name":"aprender-contracts-cli","manifest_path":"/w/crates/c/Cargo.toml","features":{"cli":[],"build-sha":["dep:aprender-build-sha"],"update-check":[]},"targets":[{"name":"pv","kind":["bin"],"required-features":["cli"]},{"name":"pv-sat","kind":["bin"],"required-features":["build-sha"]}]},
  {"id":"x","name":"zz","manifest_path":"/w/crates/zz/Cargo.toml","targets":[{"name":"aprender-zz","kind":["bin"]}]},
  {"id":"dep","name":"notmember","manifest_path":"/r/x/Cargo.toml","targets":[{"name":"evil","kind":["bin"]}]}]}
J
    expect "bins: members only, one per name, sorted" "apr,aprender-zz,pv,pv-sat" "$(dbs_bins "$t/meta.json")"
    expect "bins: facade loses to the member package; features scoped to their package; a package with an optional build-sha builds with it, once (#4219)" \
        "-p apr-cli -p zz -p aprender-contracts-cli --bin apr --bin aprender-zz --bin pv --features aprender-contracts-cli/cli --features aprender-contracts-cli/build-sha --bin pv-sat --features aprender-contracts-cli/build-sha" \
        "$(dbs_bins "$t/meta.json" cargo)"
    jq '.packages[3].targets[0].name = "apr"' "$t/meta.json" > "$t/dup.json"
    rc=0; dbs_bins "$t/dup.json" >/dev/null 2>&1 || rc=$?
    expect "bins: two non-root packages for one name is an error" "5" "$rc"
    expect "meta: version is the workspace ROOT package's" "0.70.2" "$(dbs_meta "$t/meta.json" version)"
    expect "meta: target_dir is cargo's own" "/tgt" "$(dbs_meta "$t/meta.json" target_dir)"
    rc=0; jq 'del(.target_directory)' "$t/meta.json" > "$t/notd.json"; dbs_meta "$t/notd.json" target_dir >/dev/null 2>&1 || rc=$?
    expect "meta: an absent field is an error, never an empty answer" "1" "$rc"
    rc=0; dbs_bins "$t/none.json" >/dev/null 2>&1 || rc=$?
    expect "bins: a missing metadata file is an error" "2" "$rc"

    mkbin good "echo 'good 0.70.2 (0123456 2026-10-04)'"
    mkbin long "echo 'long 0.70.2 (0123456789abcdef0123456789abcdef01234567)'"
    mkbin nosha "echo 'nosha 0.70.2'"
    mkbin other "echo 'other 0.70.2 (fedcba9 x)'"
    mkbin mid "echo 'mid 0.70.2 (3456789abc)'"
    mkbin badrc "echo 'badrc 0.70.2 (0123456)'; exit 3"
    mkbin oldver "echo 'oldver 0.70.20 (0123456)'"
    mkbin prefix "echo 'prefix 10.70.2 (0123456)'"
    mkbin stderr "echo 'stderr 0.70.2 (0123456)' >&2"
    mkbin cwd "if test -e Cargo.toml; then echo 'cwd 0.70.2 (0123456)'; else echo 'cwd 0.70.2 nosha'; fi"
    v() { dbs_verdict "$sha" 0.70.2 "$t/bin/$1" | cut -f1,2; }
    expect "verdict: version + sha prefix is green" "green	-" "$(v good)"
    expect "verdict: a full 40-hex sha is green" "green	-" "$(v long)"
    expect "verdict: no build SHA is red (#4219)" "red	version-no-sha" "$(v nosha)"
    expect "verdict: another commit's SHA is red" "red	version-mismatch" "$(v other)"
    expect "verdict: a SHA found mid-hash, not at the start, is red" "red	version-mismatch" "$(v mid)"
    expect "verdict: non-zero exit is red" "red	version-failed" "$(v badrc)"
    expect "verdict: 0.70.20 is not 0.70.2" "red	version-failed" "$(v oldver)"
    expect "verdict: 10.70.2 is not 0.70.2" "red	version-failed" "$(v prefix)"
    expect "verdict: stderr-only output is read" "green	-" "$(v stderr)"
    expect "verdict: a missing executable is red" "red	missing-artifact" "$(v absent)"
    : > "$t/bin/Cargo.toml"
    expect "verdict: runs from an empty cwd, not the caller's" "red	version-no-sha" "$(cd "$t/bin" && v cwd)"

    rc=0; dbs_smoke "$sha" good,stderr "$t/bin" 0.70.2 >/dev/null 2>&1 || rc=$?
    expect "smoke: all green exits 0" "0" "$rc"
    rc=0; dbs_smoke "$sha" good,nosha "$t/bin" 0.70.2 >/dev/null 2>&1 || rc=$?
    expect "smoke: one red bin exits 1" "1" "$rc"
    rc=0; dbs_smoke "$sha" "" "$t/bin" 0.70.2 >/dev/null 2>&1 || rc=$?
    expect "smoke: zero bins is RED, never a vacuous pass" "1" "$rc"
    rc=0; dbs_smoke "$sha" good "$t/nodir" >/dev/null 2>&1 || rc=$?
    expect "smoke: a missing bin dir is an error" "2" "$rc"
    expect "smoke: a relative bin dir resolves before the empty-cwd probe" "1" \
        "$(cd "$t" && dbs_smoke "$sha" good bin 0.70.2 2>/dev/null | grep -c '	green	')"
    expect "smoke: every bin gets a row, not just the first red" "2" \
        "$(dbs_smoke "$sha" nosha,other "$t/bin" 2>/dev/null | grep -c '	red	')"
    rm -rf -- "${t:?}"
    if [ "$fail" -eq 0 ]; then echo "deep_bins_smoke self-test: PASS"; else echo "deep_bins_smoke self-test: FAIL"; fi
    return "$fail"
}

dbs_main() {
    case "${1:-}" in
        bins) shift; dbs_bins "$@" ;;
        meta) shift; dbs_meta "$@" ;;
        smoke) shift; dbs_smoke "$@" ;;
        --self-test) dbs_self_test ;;
        *) echo "usage: deep_bins_smoke.sh bins META [list|cargo] | meta META version|target_dir | smoke SHA BINS DIR [VERSION] | --self-test" >&2; return 2 ;;
    esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    dbs_main "$@"
    exit $?
fi
