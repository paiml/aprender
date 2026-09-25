#!/usr/bin/env bash
# check_release_models_t1.sh -- the model matrix runs at T-1 on BOTH hosts before any tag, and every
# way it can fail to prove the release is a STOP, never a pass (#3717; #3712 done_when 3).
#
# WHAT THIS RUNS. The real scripts/release/autopilot.sh `models` step (from/to = models), which runs
# the real scripts/release/models_t1.sh, in a throwaway fixture:
#   - a local bare origin whose main holds a parent (9.9.8) and the release commit (9.9.9);
#   - the "gx10" leg through an ssh stub that runs the remote script locally, under a fake gx10
#     $HOME whose ~/src/aprender is a clone checked out at the OLD parent commit, with a stale
#     rel-9.9.7 dir beside it;
#   - a build stub that writes an `apr` whose --version is baked from the tree it was built IN
#     (Cargo.toml version + `git rev-parse --short=9 HEAD`), so a build of the wrong tree is visible;
#   - committed stubs for scripts/model_ladder.sh (writes <out>/<host>.json with the apr_version it
#     measured by) and scripts/check_model_ladder.sh (the judge: 0 green, 1 red/missing, 2 decline).
#   The measuring and the judging are #3712's side (a frozen interface); this table proves the
#   wiring around them.
#
# THE ROWS
#   green-pair       both hosts green: "MODELS GO"; gx10's receipt was measured by
#                    "apr 9.9.9 (<release sha9>)" although its checkout sat at the parent; the gx10
#                    build went to the PER-RELEASE dir, never ~/src/aprender/target; rel-9.9.7 is gone.
#   gx10-unreachable ssh rc 255: STOP, "unreachable over SSH".
#   build-fails      lambda's build fails: STOP, "BUILD-FAILED".
#   missing-receipt  lambda's model_ladder writes nothing: STOP, "wrote no receipt".
#   red-cell         a red cell on gx10: STOP, "the judge found red".
#   judge-decline    the judge exits 2: STOP, "the judge DECLINED".
#   stale-binary     gx10 builds an apr that is not the release: STOP, "NOT-THE-RELEASE", before
#                    any measuring.
#   disk-refusal     gx10 has less free space than a fresh target needs: STOP, "ENV:", nothing built.
#   oom-victim       the fleet GPU rule (cop ruling, one lock owner): both hosts' ladders run under
#                    `choom -n 1000`, and this wrapper takes NO flock -- the GPU lock is the
#                    ladder's own, per apr call, and a second one here would deadlock it. No build
#                    runs under a flock.
# T-4, R7: rule_r7() is EXTRACTED from scripts/check_publish_preflight.sh and run on a tagged tree
# with the judge stub and committed receipts -- cargo-free, so every decision can be mutated here.
# (That the full gate CALLS rule_r7 is proved by that script's own --selftest rows r7_*, which need
# a Rust-package fixture and run in ci.yml.)
#   r7-green         both committed receipts green: "ok    R7".
#   r7-red           a red receipt: "FAIL  R7 model matrix NOT green".
#   r7-missing       gx10's receipt absent from the tree: FAIL, naming it.
#   r7-decline       the judge exits 2: "FAIL  R7 the model-matrix judge DECLINED".
#   r7-no-judge      no judge in the tree: "FAIL  R7 no model-matrix judge".
# THE MUTANTS -- one per row (each must turn its row RED), plus the autopilot's own STOP
#   shared-target    gx10 builds into ~/src/aprender/target (#3717's veto)  -> green-pair
#   no-255-reason    leg_reason's ssh-255 branch deleted                     -> gx10-unreachable
#   build-ignored    lambda's build failure no longer stops the leg          -> build-fails
#   no-receipt-ok    a missing receipt no longer NO-GOes by itself           -> missing-receipt
#   red-is-go        the judge's rc 1 read as GO                             -> red-cell
#   decline-is-go    the judge's rc 2 read as GO                             -> judge-decline
#   no-version-proof gx10's pre-measure `apr --version` proof deleted        -> stale-binary
#   no-disk-check    gx10's free-space refusal deleted                       -> disk-refusal
#   autopilot-no-die the autopilot `models` step no longer dies on NO-GO     -> red-cell
#   no-choom         gx10's ladder runs without `choom -n 1000`              -> oom-victim
#   wrapper-flock    lambda's ladder wrapped in `flock /tmp/apr-gpu.lock`    -> oom-victim
#   r7-no-version    the judge is called without --version <v>              -> r7-green
#   r7-red-is-go     the judge's rc 1 read as ok                             -> r7-red
#   r7-no-fail-lines the judge's FAIL lines dropped from the refusal         -> r7-missing
#   r7-decline-is-go the judge's rc 2 read as ok                             -> r7-decline
#   r7-no-judge-ok   a missing judge no longer refuses by itself             -> r7-no-judge
#
# Exit 0 = every row green and every mutant killed. 1 = a row RED or a mutant survived.
# 2 = ENV: a subject or a mutation anchor is missing -- the table judged nothing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
AUTOPILOT="$ROOT/scripts/release/autopilot.sh"
MODELS="$ROOT/scripts/release/models_t1.sh"
PARAMS="$ROOT/scripts/release/lib_release_params.sh"
PREFLIGHT="$ROOT/scripts/check_publish_preflight.sh"
A_SHARED='export CARGO_TARGET_DIR="\$dir/target"'
A_255='elif [ "$1" = "$REMOTE_HOST" ] && [ "$2" = 255 ]; then'
A_BUILD='        || { echo "MODELS-LEG $LOCAL_HOST BUILD-FAILED"; return 3; }'
A_NORCPT='        echo "MODELS $h NO-GO: no receipt -- $(leg_reason "$h" "$rc")"; nogo=1; continue'
A_RED='    *) echo "MODELS NO-GO: the judge found red or missing cells (rc $jrc):"'
A_DECLINE='    2) echo "MODELS NO-GO: the judge DECLINED (rc 2)'
A_PROOF='[ "\$got" = "$want" ] || { echo "MODELS-LEG $REMOTE_HOST NOT-THE-RELEASE'
A_DISK='if [ -z "\$free_kib" ] || [ \$(( free_kib + have_kib )) -lt $NEED_KIB ]; then'
A_DIE='  [ $rc -eq 0 ] || die "T-1 model matrix NO-GO'
A_CHOOM_R='choom -n 1000 -- bash scripts/model_ladder.sh --host $REMOTE_HOST'
A_CHOOM_L='    choom -n 1000 -- bash scripts/model_ladder.sh --host "$LOCAL_HOST"'
R_VER='    out="$(cd "$root" && bash "$judge" --version "$version" 2>&1)"; rc=$?'
R_RED='        *) printf '"'"'FAIL  R7 model matrix NOT green for %s (rc %s):\n%s\n'"'"' "$version" "$rc" \'
R_LINES='               "$(grep -E '"'"'^FAIL'"'"' <<< "$out" | head -n 10 | sed '"'"'s/^/        /'"'"')" ;;'
R_DECLINE='        2) echo "FAIL  R7 the model-matrix judge DECLINED'
R_NOJUDGE='    if [ ! -f "$judge" ]; then'

env_die() { printf 'ENV   %s -- the table judged nothing, not a pass\n' "$*" >&2; exit 2; }
for f in "$AUTOPILOT" "$MODELS" "$PARAMS" "$PREFLIGHT"; do [ -r "$f" ] || env_die "no $f"; done
for t in git python3; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done
for a in "$A_SHARED" "$A_255" "$A_BUILD" "$A_NORCPT" "$A_RED" "$A_DECLINE" "$A_PROOF" "$A_DISK" "$A_CHOOM_R" "$A_CHOOM_L"; do
    grep -qF -- "$a" "$MODELS" || env_die "models_t1.sh has no '$a' line -- the subject moved"
done
grep -qF -- "$A_DIE" "$AUTOPILOT" || env_die "autopilot.sh has no '$A_DIE' line -- the subject moved"
for a in "$R_VER" "$R_RED" "$R_LINES" "$R_DECLINE" "$R_NOJUDGE"; do
    grep -qF -- "$a" "$PREFLIGHT" || env_die "check_publish_preflight.sh has no '$a' line -- the subject moved"
done
grep -q '^rule_r7() {' "$PREFLIGHT" || env_die "check_publish_preflight.sh has no rule_r7() -- the subject moved"

TMP=$(mktemp -d) || exit 2
# SEC011: validate before rm -rf. An empty or '/' value must never reach it.
cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
trap cleanup EXIT

# Hermetic git: no global hooks, signing or identity from the operator's config.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid
export GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid

mkdir -p "$TMP/bin"
# gh: milestone 7 titled 9.9.9, PR 99 merged as $FX_MC
cat > "$TMP/bin/gh" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  api) printf '7\n' ;;
  pr)  printf '%s\n' "$FX_MC" ;;
  *)   exit 1 ;;
esac
STUB
# ssh: "gx10 bash -s" runs the script here, as gx10 (FX_HOST), under the fake gx10 HOME
cat > "$TMP/bin/ssh" <<'STUB'
#!/usr/bin/env bash
while [ "${1:-}" = -o ]; do shift 2; done
host=${1:-}; shift
[ "${FX_SSH_DOWN:-0}" = 1 ] && { echo "ssh: connect to host $host port 22: No route to host" >&2; exit 255; }
[ "$host" = gx10 ] && [ "$*" = "bash -s" ] || exit 255
exec env HOME="$FX_GX10_HOME" FX_HOST=gx10 bash -s
STUB
# the build (only `metadata` and `build` are called): `build` writes an apr whose --version is
# baked from the tree it ran in; FX_BUILD_FAIL_HOST / FX_STALE_HOST pick a host to break
cat > "$TMP/bin/cargo" <<'STUB'
#!/usr/bin/env bash
host=${FX_HOST:-lambda}
tdir=${CARGO_TARGET_DIR:-$PWD/target}
v=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
case "${1:-}" in
  metadata)
    printf '{"packages":[{"name":"fx","version":"%s","manifest_path":"%s/Cargo.toml"}],"target_directory":"%s"}\n' "$v" "$PWD" "$tdir" ;;
  build)
    printf 'build host=%s oom=%s flock=%s\n' "$host" "${FX_OOM:-none}" "${FX_FLOCK:-0}" >> "$FX_LOG"
    [ "${FX_BUILD_FAIL_HOST:-}" = "$host" ] && { echo "error: could not compile (fixture)"; exit 101; }
    s=$(git rev-parse --short=9 HEAD); [ "${FX_STALE_HOST:-}" = "$host" ] && s=deadbeef0
    mkdir -p "$tdir/release"
    printf '#!/usr/bin/env bash\nprintf "apr %s (%s)\\n"\n' "$v" "$s" > "$tdir/release/apr"
    chmod +x "$tdir/release/apr"
    printf '%s\n' "$PWD" > "$tdir/BUILT_FROM" ;;
  *) exit 0 ;;
esac
STUB
# choom / flock: record the call in $FX_LOG, then run the command with FX_OOM / FX_FLOCK set, so the
# ladder and the build can report what wrapped them
cat > "$TMP/bin/choom" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = -n ] || exit 64
n=$2; shift 2; [ "${1:-}" = -- ] && shift
printf 'choom host=%s n=%s %s\n' "${FX_HOST:-lambda}" "$n" "$*" >> "$FX_LOG"
FX_OOM=$n exec "$@"
STUB
cat > "$TMP/bin/flock" <<'STUB'
#!/usr/bin/env bash
printf 'flock host=%s %s\n' "${FX_HOST:-lambda}" "$*" >> "$FX_LOG"
while [ "${1#-}" != "${1:-}" ]; do [ "$1" = -w ] && shift; shift; done
shift
FX_FLOCK=1 exec "$@"
STUB
chmod +x "$TMP/bin/gh" "$TMP/bin/ssh" "$TMP/bin/cargo" "$TMP/bin/choom" "$TMP/bin/flock"
# the judge stub (check_model_ladder.sh's frozen interface): every required host's receipt present,
# green and for this version -> 0; otherwise 1 with a FAIL line each; FX_JUDGE_DECLINE=1 -> 2
cat > "$TMP/judge-stub.sh" <<'STUB'
#!/usr/bin/env bash
v=""; d=""
while [ $# -gt 0 ]; do case "$1" in --version) v=$2; shift 2 ;; --receipts) d=$2; shift 2 ;; *) shift ;; esac; done
[ -n "$v" ] || { echo "decline: no --version given"; exit 2; }
[ -n "$d" ] || d="evidence/dogfood/models/$v"
[ "${FX_JUDGE_DECLINE:-0}" = 1 ] && { echo "decline: fixture ladder unreadable"; exit 2; }
python3 - "$d" "$v" <<'PY'
import json, os, sys
d, v = sys.argv[1:3]; rc = 0
for h in ("lambda", "gx10"):
    f = os.path.join(d, h + ".json")
    if not os.path.exists(f): print("FAIL  %s no receipt at %s" % (h, f)); rc = 1; continue
    r = json.load(open(f))
    if r.get("version") != v or r.get("red") != 0: print("FAIL  %s fx-rung red or stale" % h); rc = 1
sys.exit(rc)
PY
STUB

# fixture NAME AUTOPILOT MODELS_T1 -> $TMP/NAME: origin (parent 9.9.8 + release 9.9.9), the main
# checkout, the fake gx10 home. Returns 2 if it cannot be built.
fixture() {
    local d="$TMP/$1" r
    r="$d/repo"
    mkdir -p "$r/scripts/release" "$d/ap" || return 2
    cp -- "$2" "$r/scripts/release/autopilot.sh" && cp -- "$3" "$r/scripts/release/models_t1.sh" \
        && cp -- "$PARAMS" "$r/scripts/release/lib_release_params.sh" || return 2
    printf '#!/usr/bin/env bash\nexit 0\n' > "$r/scripts/bump-version.sh"
    # model_ladder.sh stub: measures by the apr in the target dir; FX_NO_RECEIPT_HOST / FX_RED_HOST
    cat > "$r/scripts/model_ladder.sh" <<'STUB'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in --host) h=$2; shift 2 ;; --out) o=$2; shift 2 ;; *) shift ;; esac; done
printf 'ladder host=%s oom=%s flock=%s\n' "$h" "${FX_OOM:-none}" "${FX_FLOCK:-0}" >> "$FX_LOG"
[ "${FX_NO_RECEIPT_HOST:-}" = "$h" ] && { echo "fixture: measured nothing on $h"; exit 2; }
apr="${CARGO_TARGET_DIR:-$PWD/target}/release/apr"
v=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
red=0; [ "${FX_RED_HOST:-}" = "$h" ] && red=1
mkdir -p "$o"
printf '{"host":"%s","version":"%s","apr_version":"%s","executed":2,"red":%s}\n' "$h" "$v" "$("$apr" --version)" "$red" > "$o/$h.json"
[ "$red" = 0 ]
STUB
    cp -- "$TMP/judge-stub.sh" "$r/scripts/check_model_ladder.sh" || return 2
    printf 'target/\n.dogfood/\n' > "$r/.gitignore"
    printf '[package]\nname = "fx"\nversion = "9.9.8"\nedition = "2021"\n' > "$r/Cargo.toml"
    git init -q --bare -b main "$d/origin.git" && git -C "$r" init -q -b main \
        && git -C "$r" add -A && git -C "$r" commit -q -m parent \
        && sed -i 's/^version = "9.9.8"$/version = "9.9.9"/' "$r/Cargo.toml" \
        && git -C "$r" commit -q -am 'release: 9.9.9' \
        && git -C "$r" remote add origin "$d/origin.git" && git -C "$r" push -q origin main \
        && git -C "$r" fetch -q origin || return 2
    git -C "$r" rev-parse HEAD > "$d/mc"
    # the fake gx10: its checkout sits at the PARENT, with a stale release dir beside it
    mkdir -p "$d/gx10/src" "$d/gx10/.cache/aprender-release/rel-9.9.7/target" || return 2
    git clone -q "$d/origin.git" "$d/gx10/src/aprender" && git -C "$d/gx10/src/aprender" checkout -q --detach HEAD^ || return 2
}

# models NAME AUTOPILOT MODELS_T1 [ENV=VAL ...] -> runs `autopilot.sh 9.9.9 99 models models`
models() {
    local n=$1 d="$TMP/$1"; shift
    fixture "$n" "$1" "$2" || return 2
    : > "$d/wrap.log"
    shift 2
    ( export RELEASE_AP="$d/ap" RELEASE_EPIC=9002 CARGO_HOME="$TMP/cargo-home" PATH="$TMP/bin:$PATH" \
          FX_MC="$(cat "$d/mc")" FX_GX10_HOME="$d/gx10" MODELS_T1_NEED_KIB=1 FX_LOG="$d/wrap.log"
      for kv in "$@"; do export "${kv?}"; done
      bash "$d/repo/scripts/release/autopilot.sh" 9.9.9 99 models models ) > "$d/out.log" 2>&1
    printf '%s\n' "$?" > "$d/rc"
}

fails=0; rows=0
row() { # row NAME RC MESSAGE
    rows=$((rows + 1))
    if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"
    elif [ "$2" = 2 ]; then env_die "row $1 could not build its fixture"
    else printf 'FAIL  %s: %s\n' "$1" "$3" >&2; fails=$((fails + 1)); fi
}

# ---- row bodies: TAG AUTOPILOT MODELS_T1 -> 0 green, 1 red (prints why), 2 fixture --------------
green_pair() {
    local n="green-$1" d; d="$TMP/green-$1"
    models "$n" "$2" "$3" || return 2
    [ "$(cat "$d/rc")" = 0 ] || { printf 'autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(tail -n 1 "$d/ap/STATUS" 2>/dev/null)"; return 1; }
    grep -qF "MODELS GO at $(cat "$d/mc") on lambda and gx10" "$d/ap/STATUS" || { printf 'no "MODELS GO" line\n'; return 1; }
    grep -qF "apr 9.9.9 ($(cut -c1-9 "$d/mc"))" "$d/ap/models-t1/gx10.json" 2>/dev/null \
        || { printf 'gx10 was not measured by the release binary: %s\n' "$(cat "$d/ap/models-t1/gx10.json" 2>/dev/null)"; return 1; }
    [ -f "$d/gx10/.cache/aprender-release/rel-9.9.9/target/BUILT_FROM" ] || { printf 'gx10 did not build into the per-release dir\n'; return 1; }
    [ ! -e "$d/gx10/src/aprender/target/BUILT_FROM" ] || { printf 'gx10 built into ~/src/aprender/target\n'; return 1; }
    [ ! -e "$d/gx10/.cache/aprender-release/rel-9.9.7" ] || { printf 'the previous release dir rel-9.9.7 was not deleted\n'; return 1; }
    return 0
}
stops() { # TAG AUTOPILOT MODELS_T1 NEEDLE ENV... -> 0 when the step STOPped naming NEEDLE
    local n="$1" d="$TMP/$1" needle=$4 a=$2 m=$3; shift 4
    models "$n" "$a" "$m" "$@" || return 2
    [ "$(cat "$d/rc")" != 0 ] || { printf 'autopilot exited 0 (%s)\n' "$*"; return 1; }
    grep -q 'STOP T-1 model matrix NO-GO' "$d/ap/STATUS" || { printf 'no T-1 STOP: %s\n' "$(tail -n 1 "$d/ap/STATUS")"; return 1; }
    grep -qF -- "$needle" "$d/ap/STATUS" || { printf 'the STOP never said "%s": %s\n' "$needle" "$(grep '^.*MODELS' "$d/ap/STATUS" | tail -n 2 | tr '\n' ' ')"; return 1; }
    return 0
}
unreachable()   { stops "unreach-$1" "$2" "$3" "MODELS gx10 NO-GO: no receipt -- unreachable over SSH" FX_SSH_DOWN=1; }
build_fails()   { stops "build-$1" "$2" "$3" "MODELS lambda NO-GO: no receipt -- BUILD-FAILED" FX_BUILD_FAIL_HOST=lambda; }
missing()       { stops "missing-$1" "$2" "$3" "MODELS lambda NO-GO: no receipt -- model_ladder.sh wrote no receipt" FX_NO_RECEIPT_HOST=lambda; }
red_cell()      { stops "red-$1" "$2" "$3" "MODELS NO-GO: the judge found red" FX_RED_HOST=gx10; }
decline()       { stops "decline-$1" "$2" "$3" "MODELS NO-GO: the judge DECLINED" FX_JUDGE_DECLINE=1; }
stale()         { stops "stale-$1" "$2" "$3" "MODELS gx10 NO-GO: no receipt -- NOT-THE-RELEASE" FX_STALE_HOST=gx10; }
disk() {
    local rc; stops "disk-$1" "$2" "$3" "MODELS gx10 NO-GO: no receipt -- ENV:" MODELS_T1_NEED_KIB=999999999999; rc=$?
    [ "$rc" = 0 ] || return "$rc"
    [ ! -e "$TMP/disk-$1/gx10/.cache/aprender-release/rel-9.9.9/target/BUILT_FROM" ] || { printf 'gx10 built despite the refusal\n'; return 1; }
    grep -q 'rc=2' "$TMP/disk-$1/ap/STATUS" || { printf 'models_t1.sh did not exit 2 (ENV)\n'; return 1; }
    return 0
}

oom_victim() {
    local n="oom-$1" d; d="$TMP/oom-$1"
    models "$n" "$2" "$3" || return 2
    [ "$(cat "$d/rc")" = 0 ] || { printf 'autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(tail -n 1 "$d/ap/STATUS" 2>/dev/null)"; return 1; }
    local h
    for h in lambda gx10; do
        grep -qx "ladder host=$h oom=1000 flock=0" "$d/wrap.log" \
            || { printf '%s ladder did not run under choom -n 1000 alone: %s\n' "$h" "$(grep "^ladder host=$h" "$d/wrap.log" | tr '\n' ' ')"; return 1; }
        grep -qx "build host=$h oom=none flock=0" "$d/wrap.log" \
            || { printf '%s build ran wrapped: %s\n' "$h" "$(grep "^build host=$h" "$d/wrap.log" | tr '\n' ' ')"; return 1; }
    done
    ! grep -q '^flock ' "$d/wrap.log" || { printf 'the wrapper took a flock (it would deadlock the ladder'"'"'s own): %s\n' "$(grep '^flock ' "$d/wrap.log" | head -n 1)"; return 1; }
    return 0
}

# ---- the rows ---------------------------------------------------------------------------------
for spec in "green-pair green_pair" "gx10-unreachable unreachable" "build-fails build_fails" \
            "missing-receipt missing" "red-cell red_cell" "judge-decline decline" \
            "stale-binary stale" "disk-refusal disk" "oom-victim oom_victim"; do
    set -- $spec
    msg=$($2 real "$AUTOPILOT" "$MODELS"); row "$1" "$?" "$msg"
done

# ---- T-4: R7, extracted and run on a tagged tree's COMMITTED receipts -------------------------
# r7 FNSRC NAME HOSTS RED [ENV=VAL ...] -> rule_r7's output in $TMP/r7-NAME/out, its rc in .../rc.
# HOSTS: the hosts whose receipt is committed; RED: the one committed red ("-" for none);
# "nojudge" as a host drops the judge from the tree.
r7() {
    local src=$1 d="$TMP/r7-$2" hosts=$3 redh=$4 h; shift 4
    mkdir -p "$d/tree/scripts" "$d/tree/evidence/dogfood/models/9.9.9" || return 2
    sed -n '/^rule_r7() {/,/^}/p' "$src" > "$d/fn.sh" || return 2
    [ -s "$d/fn.sh" ] || return 2
    case " $hosts " in *" nojudge "*) ;; *) cp -- "$TMP/judge-stub.sh" "$d/tree/scripts/check_model_ladder.sh" ;; esac
    for h in $hosts; do
        [ "$h" = nojudge ] && continue
        printf '{"host":"%s","version":"9.9.9","apr_version":"apr 9.9.9 (fixture)","executed":2,"red":%s}\n' \
            "$h" "$([ "$redh" = "$h" ] && echo 1 || echo 0)" > "$d/tree/evidence/dogfood/models/9.9.9/$h.json"
    done
    ( for kv in "$@"; do export "${kv?}"; done
      unset PUBLISH_PREFLIGHT_LADDER_JUDGE
      . "$d/fn.sh"; cd / && rule_r7 "$d/tree" 9.9.9 ) > "$d/out" 2>&1
    printf '%s\n' "$?" > "$d/rc"
}
r7_expect() { # NAME WANT_RC NEEDLE -> 0 when rule_r7 answered WANT_RC and said NEEDLE
    local d="$TMP/r7-$1"
    [ "$(cat "$d/rc")" = "$2" ] || { printf 'rule_r7 rc=%s, want %s: %s\n' "$(cat "$d/rc")" "$2" "$(head -n 1 "$d/out")"; return 1; }
    grep -qF -- "$3" "$d/out" || { printf 'rule_r7 never said "%s": %s\n' "$3" "$(head -n 2 "$d/out" | tr '\n' ' ')"; return 1; }
    return 0
}
r7_green()   { r7 "$2" "green-$1" "lambda gx10" - || return 2; r7_expect "green-$1" 0 "ok    R7 model matrix green for 9.9.9"; }
r7_red()     { r7 "$2" "red-$1" "lambda gx10" gx10 || return 2; r7_expect "red-$1" 1 "FAIL  R7 model matrix NOT green"; }
r7_missing() { r7 "$2" "missing-$1" "lambda" - || return 2; r7_expect "missing-$1" 1 "gx10 no receipt at evidence/dogfood/models/9.9.9/gx10.json"; }
r7_decline() { r7 "$2" "decline-$1" "lambda gx10" - FX_JUDGE_DECLINE=1 || return 2; r7_expect "decline-$1" 1 "FAIL  R7 the model-matrix judge DECLINED"; }
r7_nojudge() { r7 "$2" "nojudge-$1" "lambda gx10 nojudge" - || return 2; r7_expect "nojudge-$1" 1 "FAIL  R7 no model-matrix judge"; }
for spec in "r7-green r7_green" "r7-red r7_red" "r7-missing r7_missing" "r7-decline r7_decline" "r7-no-judge r7_nojudge"; do
    set -- $spec
    msg=$($2 real "$PREFLIGHT"); row "$1" "$?" "$msg"
done

# ---- the mutants ------------------------------------------------------------------------------
# mutant NAME SRC ANCHOR REPLACEMENT ROWFN [autopilot] -- replace ANCHOR's first line, run ROWFN
mutant() {
    local name=$1 src=$2 anchor=$3 repl=$4 fn=$5 which=${6:-models} m="$TMP/m-$1.sh" msg mrc
    python3 - "$src" "$m" "$anchor" "$repl" <<'PY' || env_die "$name mutant"
import sys
src, dst, anchor, repl = sys.argv[1:5]
s = open(src).read()
assert s.count(anchor) == 1, anchor
open(dst, "w").write(s.replace(anchor, repl, 1))
PY
    cmp -s "$src" "$m" && env_die "$name mutant is identical to its source"
    case "$which" in
        autopilot) msg=$($fn "m-$name" "$m" "$MODELS"); mrc=$? ;;
        preflight) msg=$($fn "m-$name" "$m"); mrc=$? ;;
        *)         msg=$($fn "m-$name" "$AUTOPILOT" "$m"); mrc=$? ;;
    esac
    [ "$mrc" = 2 ] && env_die "$name mutant could not build its fixture"
    [ "$mrc" != 0 ]; row "mutant $name is killed by $fn (${msg:-survived})" "$?" "the mutant PASSED -- the row does not discriminate"
}
mutant shared-target    "$MODELS" "$A_SHARED" 'export CARGO_TARGET_DIR="\$repo/target"' green_pair
mutant no-255-reason    "$MODELS" "$A_255" 'elif false; then' unreachable
mutant build-ignored    "$MODELS" "$A_BUILD" '        || true' build_fails
mutant no-receipt-ok    "$MODELS" "$A_NORCPT" '        continue' missing
mutant red-is-go        "$MODELS" "$A_RED" '    *) ;; 99) echo' red_cell
mutant decline-is-go    "$MODELS" "$A_DECLINE" '    2) ;; 98) echo "' decline
mutant no-version-proof "$MODELS" "$A_PROOF" 'true || { echo "MODELS-LEG $REMOTE_HOST NOT-THE-RELEASE' stale
mutant no-disk-check    "$MODELS" "$A_DISK" 'if false; then' disk
mutant autopilot-no-die "$AUTOPILOT" "$A_DIE" '  [ $rc -eq $rc ] || die "T-1 model matrix NO-GO' red_cell autopilot
mutant no-choom         "$MODELS" "$A_CHOOM_R" 'bash scripts/model_ladder.sh --host $REMOTE_HOST' oom_victim
mutant wrapper-flock    "$MODELS" "$A_CHOOM_L" '    flock /tmp/apr-gpu.lock choom -n 1000 -- bash scripts/model_ladder.sh --host "$LOCAL_HOST"' oom_victim
mutant r7-no-version    "$PREFLIGHT" "$R_VER" '    out="$(cd "$root" && bash "$judge" 2>&1)"; rc=$?' r7_green preflight
mutant r7-red-is-go     "$PREFLIGHT" "$R_RED" '        *) return 0; printf '"'"'FAIL  R7 model matrix NOT green for %s (rc %s):\n%s\n'"'"' "$version" "$rc" \' r7_red preflight
mutant r7-no-fail-lines "$PREFLIGHT" "$R_LINES" '               "" ;;' r7_missing preflight
mutant r7-decline-is-go "$PREFLIGHT" "$R_DECLINE" '        2) return 0; echo "FAIL  R7 the model-matrix judge DECLINED' r7_decline preflight
mutant r7-no-judge-ok   "$PREFLIGHT" "$R_NOJUDGE" '    if false; then' r7_nojudge preflight

# VACUITY FLOOR: a table that ran fewer rows than it declares is not a pass.
[ "$rows" -ge 30 ] || { printf 'VACUOUS %s row(s) ran, fewer than the 30 declared\n' "$rows" >&2; exit 1; }
[ "$fails" -eq 0 ] || { printf 'RED   %s of %s row(s) failed\n' "$fails" "$rows" >&2; exit 1; }
printf 'PASS  %s row(s): the model matrix runs at T-1 on both hosts, every failure to prove the release STOPs before the tag, and R7 refuses the same failures at T-4 (#3717)\n' "$rows"
