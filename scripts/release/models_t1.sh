#!/usr/bin/env bash
# models_t1.sh -- the model matrix at T-1, on BOTH required hosts, at the release commit, before any
# tag (#3717; #3712 done_when 3). autopilot.sh's `models` step runs it from the release worktree.
#
# WHY. v0.69.0 reached its publish step with Q4_K models red on CUDA. The ladder receipts were not a
# precondition of the bump and were measured after the tag was public (#3712, #3708). This measures
# the commit that will be tagged, on lambda (here) and gx10 (over the operator-authorized
# lambda->gx10 SSH), in parallel, and judges both receipts with the judge T-4's R7 re-runs.
#
# PER HOST -- the measuring itself is scripts/model_ladder.sh (#3712's side, a frozen interface):
#   1. build `apr --features cuda` (binary-release.yml's command) FROM THE RELEASE COMMIT:
#        lambda  this worktree, into autopilot's CARGO_TARGET_DIR
#        gx10    a detached worktree at the release commit and a PER-RELEASE target dir,
#                $HOME/.cache/aprender-release/rel-<v> on gx10, derived there at run time (#3618:
#                no host path is a literal under scripts/release/). Every other rel-* is deleted
#                first. NEVER ~/src/aprender/target: a shared cache under an old checkout is how
#                a stale instrument agrees with itself (#3600).
#   2. prove the binary before measuring: `apr --version` must read "apr <v> (<release sha9>)"
#   3. `choom -n 1000 -- bash scripts/model_ladder.sh --host <id> --out <dir>`: the fleet GPU rule
#      (cop, 2026-09-21, after gx10's 15:56Z global OOM killed 18 CI containers) makes THIS run,
#      never the CI pool, the OOM victim; oom_score_adj is inherited across fork, so every apr the
#      ladder starts is covered. The GPU LOCK is model_ladder.sh's own, per apr call (#3712 row B):
#      this wrapper takes NO flock -- a second flock on the same file here would deadlock the
#      ladder's inner one (cop ruling, one owner). The build runs under neither.
#      The receipt's apr_version must be the line proved in step 2.
#   Both legs pass --cells (#3715 B1): the receipts carry cells[], the unit release-readiness-v1 grades at
#   the readiness step. A leg without it measures rungs only and every owed cell is a violation there.
# THEN scripts/check_model_ladder.sh --version <v> --receipts <out> judges both receipts.
#
# An unreachable host, a failed build, a binary that is not the release, a missing receipt, a red
# cell and a judge DECLINE (exit 2) are each NO-GO. None is a pass, and nothing is retried here.
#
# MEASURE MODE -- env MODELS_T1_MEASURE (from 0.71.0 a release ships on CRUX smoke, the standing
# `ladder.release_policy` in contracts/model-capability-ladder-v1.yaml):
#   unset|ladder  the above, unchanged.
#   crux          the same build + binary proof per leg, then
#                 `choom -n 1000 -- bash scripts/crux_sweep_shards.sh <v> --host <id> --apr <proved apr>
#                  --out <dir> --backend gpu --certification evidence/crux/<v>/prompt-certification.json`.
#                 That certification must be committed at the release commit (prepare_bump carries it
#                 forward), else exit 2 before any leg. Receipts: <out>/<host>-gpu.json (the local leg
#                 sweeps into <out>/lambda-crux/ and copies its receipt up: the sweep's meta/plan files
#                 beside it would be globbed by the judge). Each receipt's .apr.version_line must be the
#                 proved line. The judge runs as `check_model_ladder.sh --version <v> --crux <out>
#                 --cut-commit <sha>` with CRUX_CERT set, and its rc 0 counts only when it printed a
#                 `POLICY: ` or `SCOPED: ` line -- otherwise it judged no CRUX scope and crux mode
#                 measured nothing the release gate reads.
#   anything else exit 2.
#
# usage: [MODELS_T1_MEASURE=ladder|crux] models_t1.sh <version> <release-commit> <out-dir>
# exit:  0 GO on both hosts  ·  1 NO-GO  ·  2 usage/ENV (the caller STOPs on any non-zero)
set -uo pipefail
LOCAL_HOST=lambda
REMOTE_HOST=gx10
# A fresh `cargo build --release -p apr-cli --bin apr --features cuda --locked` target, in KiB: the
# gx10 leg REFUSES up front, as ENV, when its filesystem has less free (an existing rel-<v> dir for
# this same version counts, since it is reused). Half-building on a 97-99% disk fails later and
# worse. MEASURED 2026-09-21 on lambda (x86_64) at 629143eeb: a fresh target is 5,285,126,283 bytes
# (5,161,257 KiB, 244 s) and the worktree checkout 363,372 KiB, so 5,524,629 KiB. gx10 is aarch64 and
# was not measured; re-measure when the tree grows. MODELS_T1_NEED_KIB overrides it (the seam
# scripts/check_release_models_t1.sh uses to drive both sides of the refusal).
NEED_KIB=${MODELS_T1_NEED_KIB:-5524629}

# mt_receipt_fields RECEIPT -> "<apr_version>\t<executed>\t<red>" ('-' for a missing key), or
# "UNREADABLE\t-\t-". jq, not python3 (#4352); the values print as python's str() did, except a
# list/object value prints as JSON, not a python repr (display only: it never equals $want).
mt_receipt_fields() {
    # python read the file as locale UTF-8, so a BOM made it UNREADABLE; jq would skip it
    { [ -f "$1" ] && [ -r "$1" ] && [ "$(head -c 3 -- "$1")" != $'\xef\xbb\xbf' ] \
        && jq -rs 'if length != 1 then "UNREADABLE\t-\t-" else .[0] | if type != "object" then empty else
            . as $d | ["apr_version", "executed", "red"]
            | map(. as $k | if ($d | has($k)) | not then "-" else $d[$k]
                  | if type == "string" then . elif . == null then "None" elif . == true then "True"
                    elif . == false then "False" else tojson end end)
            | join("\t") end end' -- "$1" 2>/dev/null; } || printf 'UNREADABLE\t-\t-\n'
}

# mt_crux_fields RECEIPT -> "<apr.version_line>\t<summary.cells>\t<summary.RED>" for a
# crux-inference-receipt/v1, printed as mt_receipt_fields prints ('-' for a missing key), or
# "UNREADABLE\t-\t-". Same BOM rule.
mt_crux_fields() {
    { [ -f "$1" ] && [ -r "$1" ] && [ "$(head -c 3 -- "$1")" != $'\xef\xbb\xbf' ] \
        && jq -rs 'if length != 1 then "UNREADABLE\t-\t-" else .[0] | if type != "object" then empty else
            [(.apr | if type == "object" and has("version_line") then .version_line else "-" end),
             (.summary | if type == "object" and has("cells") then .cells else "-" end),
             (.summary | if type == "object" and has("RED") then .RED else "-" end)]
            | map(if type == "string" then . elif . == null then "None" elif . == true then "True"
                  elif . == false then "False" else tojson end)
            | join("\t") end end' -- "$1" 2>/dev/null; } || printf 'UNREADABLE\t-\t-\n'
}

# mt_fetch_commit REPO VER SHA -> rc 0 when SHA is in REPO after fetching origin main and, when it
# exists, origin release/VER. A patch release's commit is on its release branch, not on main, so a
# fetch of main alone left the remote leg FETCH-FAILED on every release cut from a release branch.
# Defined here and sent ahead of the remote leg with `declare -f`, so the self-test runs the same code.
mt_fetch_commit() {
    git -C "$1" fetch -q origin main || return 1
    git -C "$1" fetch -q origin "+refs/heads/release/$2:refs/remotes/origin/release/$2" 2> /dev/null || true
    git -C "$1" cat-file -e "$3^{commit}" 2> /dev/null
}

mt_self_test() {
    local d fail=0 want got
    d=$(mktemp -d) || return 2
    while IFS='~' read -r want got; do
        printf '%s' "$got" > "$d/r.json"
        got=$(mt_receipt_fields "$d/r.json" | tr '\t' '|')
        if [ "$got" = "$want" ]; then echo "  ok   receipt -> $want"; else echo "  FAIL receipt: wanted $want, got $got"; fail=1; fi
    done <<'EOF'
apr 1.0.0 (abc)|3|0~{"apr_version":"apr 1.0.0 (abc)","executed":3,"red":0}
apr 1.0.0 (abc)|-|-~{"apr_version":"apr 1.0.0 (abc)"}
None|True|False~{"apr_version":null,"executed":true,"red":false}
UNREADABLE|-|-~not json
UNREADABLE|-|-~
UNREADABLE|-|-~{} {}
EOF
    printf '\xef\xbb\xbf{"apr_version":"apr 1.0.0 (abc)"}' > "$d/r.json"
    got=$(mt_receipt_fields "$d/r.json" | tr '\t' '|')
    if [ "$got" = "UNREADABLE|-|-" ]; then echo "  ok   a BOM-prefixed receipt is UNREADABLE, as python read it"; else echo "  FAIL BOM receipt gave '$got'"; fail=1; fi
    got=$(mt_receipt_fields "$d/absent.json" | tr '\t' '|')
    if [ "$got" = "UNREADABLE|-|-" ]; then echo "  ok   a missing receipt is UNREADABLE, once"; else echo "  FAIL missing receipt gave '$got'"; fail=1; fi
    while IFS='~' read -r want got; do
        printf '%s' "$got" > "$d/c.json"
        got=$(mt_crux_fields "$d/c.json" | tr '\t' '|')
        if [ "$got" = "$want" ]; then echo "  ok   crux receipt -> $want"; else echo "  FAIL crux receipt: wanted $want, got $got"; fail=1; fi
    done <<'EOF'
apr 1.0.0 (abc)|12|0~{"apr":{"version_line":"apr 1.0.0 (abc)"},"summary":{"cells":12,"RED":0,"verdict":"GREEN"}}
apr 1.0.0 (abc)|-|-~{"apr":{"version_line":"apr 1.0.0 (abc)"}}
-|4|1~{"apr":"apr 1.0.0 (abc)","summary":{"cells":4,"RED":1}}
None|-|-~{"apr":{"version_line":null},"summary":[]}
UNREADABLE|-|-~not json
UNREADABLE|-|-~
UNREADABLE|-|-~{} {}
EOF
    printf '\xef\xbb\xbf{"apr":{"version_line":"apr 1.0.0 (abc)"}}' > "$d/c.json"
    got=$(mt_crux_fields "$d/c.json" | tr '\t' '|')
    if [ "$got" = "UNREADABLE|-|-" ]; then echo "  ok   a BOM-prefixed crux receipt is UNREADABLE"; else echo "  FAIL BOM crux receipt gave '$got'"; fail=1; fi
    # mt_fetch_commit against a local origin: a commit on main or on release/VER is found, one on
    # another branch or on the release branch of another version is not
    local o="$d/origin" c n=0 which sha s_main s_rel s_other
    local -a g=(git -c user.name=t -c user.email=t@t -c init.defaultBranch=main)
    if "${g[@]}" init -q "$o" && "${g[@]}" -C "$o" commit -q --allow-empty -m m && s_main=$(git -C "$o" rev-parse HEAD) \
        && "${g[@]}" -C "$o" checkout -q -b release/9.9.9 && "${g[@]}" -C "$o" commit -q --allow-empty -m r \
        && s_rel=$(git -C "$o" rev-parse HEAD) && "${g[@]}" -C "$o" checkout -q -b other main \
        && "${g[@]}" -C "$o" commit -q --allow-empty -m x && s_other=$(git -C "$o" rev-parse HEAD) \
        && "${g[@]}" -C "$o" checkout -q main; then
        while IFS='~' read -r want ver which what; do
            case "$which" in main) sha="$s_main" ;; rel) sha="$s_rel" ;; *) sha="$s_other" ;; esac
            # a fresh clone of main per case, so a fetch in one case cannot satisfy the next
            n=$((n + 1)); c="$d/clone-$n"
            git clone -q --single-branch -b main "file://$o" "$c" 2> /dev/null || { echo "  FAIL fetch: clone"; fail=1; continue; }
            if mt_fetch_commit "$c" "$ver" "$sha"; then got=found; else got=refused; fi
            if [ "$got" = "$want" ]; then echo "  ok   fetch: $what -> $want"; else echo "  FAIL fetch: $what wanted $want, got $got"; fail=1; fi
        done <<'FETCH'
found~9.9.9~main~a commit on main
found~9.9.9~rel~a commit on release/9.9.9
refused~9.9.8~rel~a commit on the release branch of another version
refused~9.9.9~other~a commit on neither
FETCH
    else
        echo "  FAIL fetch: could not build the local origin"; fail=1
    fi
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "models_t1 self-test: PASS"; else echo "models_t1 self-test: FAIL"; fi
    return "$fail"
}
if [ "${1:-}" = --self-test ]; then mt_self_test; exit $?; fi

USAGE="usage: [MODELS_T1_MEASURE=ladder|crux] models_t1.sh <version> <release-commit> <out-dir>"
MEASURE=${MODELS_T1_MEASURE:-ladder}
case $MEASURE in
    ladder|crux) ;;
    *) echo "models_t1: MODELS_T1_MEASURE='$MEASURE' is neither ladder nor crux" >&2; echo "$USAGE" >&2; exit 2 ;;
esac
[ $# -eq 3 ] || { echo "$USAGE" >&2; exit 2; }
ver=$1; out=$3
[[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "models_t1: version '$ver' is not X.Y.Z" >&2; exit 2; }
sha=$(git rev-parse --verify --quiet "$2^{commit}") || { echo "models_t1: '$2' is not a commit in $(pwd)" >&2; exit 2; }
[ "$(git rev-parse HEAD)" = "$sha" ] \
    || { echo "models_t1: $(pwd) is at $(git rev-parse --short=9 HEAD), not the release commit ${sha:0:9}" >&2; exit 2; }
sha9=$(git rev-parse --short=9 "$sha")
want="apr $ver ($sha9)"
# crux mode: the receipt names (<host><RSUF>.json: LREC, RREC, each set from one variable), the measurer leg_reason names, the certification
RSUF=""
LREC="$LOCAL_HOST.json"
RREC="$REMOTE_HOST.json"
MEASURER=model_ladder.sh; CERT="evidence/crux/$ver/prompt-certification.json"
if [ "$MEASURE" = crux ]; then
    RSUF=-gpu
    LREC="$LOCAL_HOST-gpu.json"
    RREC="$REMOTE_HOST-gpu.json"
    MEASURER=crux_sweep_shards.sh
    git cat-file -e "$sha:$CERT" 2>/dev/null \
        || { echo "MODELS NO-GO: no prompt certification for $ver at $CERT -- CRUX smoke cannot be planned"; exit 2; }
fi
mkdir -p "$out" || exit 2
if [ "$MEASURE" = crux ]; then
    # the judge globs every *.json in $out: a ladder receipt left by an earlier pass would read as a bad CRUX one
    rm -f -- "${out:?}/$LOCAL_HOST.json" "${out:?}/$REMOTE_HOST.json" "${out:?}/$LREC" \
        "${out:?}/$RREC" "${out:?}/$RREC.part"
    rm -rf -- "${out:?}/$LOCAL_HOST-crux"
else
rm -f -- "$out/$LOCAL_HOST.json" "$out/$REMOTE_HOST.json" "$out/$REMOTE_HOST.json.part"
fi

local_leg() {
    local tdir got
    cargo build --release -p apr-cli --bin apr --features cuda --locked \
        || { echo "MODELS-LEG $LOCAL_HOST BUILD-FAILED"; return 3; }
    tdir=${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 | jq -r .target_directory)}
    got=$("$tdir/release/apr" --version 2>/dev/null | head -n 1)
    [ "$got" = "$want" ] || { echo "MODELS-LEG $LOCAL_HOST NOT-THE-RELEASE: '$got' (want '$want')"; return 3; }
    if [ "$MEASURE" = crux ]; then
        local lrc cdir="$out/$LOCAL_HOST-crux"
        choom -n 1000 -- bash scripts/crux_sweep_shards.sh "$ver" --host "$LOCAL_HOST" --apr "$tdir/release/apr" --out "$cdir" --backend gpu --certification "$CERT"; lrc=$?
        [ ! -f "$cdir/$LREC" ] || cp -- "$cdir/$LREC" "$out/$LREC"
        return "$lrc"
    fi
    choom -n 1000 -- bash scripts/model_ladder.sh --host "$LOCAL_HOST" --cells --out "$out"
}

remote_leg() {
    # mt_fetch_commit goes first on the stream, so the remote leg runs the code the self-test ran
    { declare -f mt_fetch_commit; cat <<HOST
set -u
repo="\$HOME/src/aprender"; base="\$HOME/.cache/aprender-release"; dir="\$base/rel-$ver"
mkdir -p "\$base" || exit 3
for d in "\$base"/rel-*; do
  [ -d "\$d" ] && [ "\$d" != "\$dir" ] || continue
  git -C "\$repo" worktree remove --force "\$d/wt" > /dev/null 2>&1
  rm -rf -- "\$d"
done
free_kib=\$(df -Pk "\$base" | awk 'NR == 2 {print \$4}')
have_kib=0; [ -d "\$dir" ] && have_kib=\$(du -sk "\$dir" | awk '{print \$1}')
if [ -z "\$free_kib" ] || [ \$(( free_kib + have_kib )) -lt $NEED_KIB ]; then
  echo "MODELS-LEG $REMOTE_HOST ENV: \$(( (free_kib + have_kib) / 1048576 )) GiB usable under \$base, a fresh cuda release target needs \$(( $NEED_KIB / 1048576 )) GiB -- refused before building"
  exit 4
fi
mt_fetch_commit "\$repo" "$ver" "$sha" \
  || { echo "MODELS-LEG $REMOTE_HOST FETCH-FAILED: $sha9 is not reachable from origin/main or origin/release/$ver there"; exit 3; }
git -C "\$repo" worktree remove --force "\$dir/wt" > /dev/null 2>&1
git -C "\$repo" worktree prune
git -C "\$repo" worktree add -q --detach "\$dir/wt" "$sha" || { echo "MODELS-LEG $REMOTE_HOST WORKTREE-FAILED"; exit 3; }
cd "\$dir/wt" || exit 3
export CARGO_TARGET_DIR="\$dir/target"
cargo build --release -p apr-cli --bin apr --features cuda --locked > "\$dir/build.log" 2>&1 \
  || { tail -n 20 "\$dir/build.log"; echo "MODELS-LEG $REMOTE_HOST BUILD-FAILED"; exit 3; }
got=\$("\$CARGO_TARGET_DIR/release/apr" --version 2>/dev/null | head -n 1)
[ "\$got" = "$want" ] || { echo "MODELS-LEG $REMOTE_HOST NOT-THE-RELEASE: '\$got' (want '$want')"; exit 3; }
rm -rf -- "\$dir/out"
case $MEASURE in
crux) choom -n 1000 -- bash scripts/crux_sweep_shards.sh "$ver" --host $REMOTE_HOST --apr "\$CARGO_TARGET_DIR/release/apr" --out "\$dir/out" --backend gpu --certification "$CERT" ;;
*) choom -n 1000 -- bash scripts/model_ladder.sh --host $REMOTE_HOST --cells --out "\$dir/out" ;;
esac; lrc=\$?
if [ -f "\$dir/out/$RREC" ]; then
  echo "---RECEIPT $REMOTE_HOST---"; cat "\$dir/out/$RREC"; echo "---END RECEIPT---"
fi
git -C "\$repo" worktree remove --force "\$dir/wt" > /dev/null 2>&1
exit \$lrc
HOST
    } | ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "bash -s"
}

# leg_reason HOST RC -> why a leg produced no usable receipt, from its own log
leg_reason() {
    local r
    r=$(grep -E "^MODELS-LEG $1 " "$out/$1.log" | tail -n 1)
    if [ -n "$r" ]; then printf '%s (rc %s)' "${r#MODELS-LEG $1 }" "$2"
    elif [ "$1" = "$REMOTE_HOST" ] && [ "$2" = 255 ]; then printf 'unreachable over SSH (ssh rc 255)'
    else printf '%s wrote no receipt (rc %s)' "$MEASURER" "$2"; fi
}

local_leg > "$out/$LOCAL_HOST.log" 2>&1 & lpid=$!
remote_leg > "$out/$REMOTE_HOST.log" 2>&1 & rpid=$!
wait "$lpid"; lrc=$?
wait "$rpid"; rrc=$?
sed -n "/^---RECEIPT $REMOTE_HOST---\$/,/^---END RECEIPT---\$/p" "$out/$REMOTE_HOST.log" | sed '1d;$d' > "$out/$RREC.part"
if [ -s "$out/$RREC.part" ]; then mv -- "$out/$RREC.part" "$out/$RREC"
else rm -f -- "$out/$RREC.part"; fi

nogo=0; env=0
for h in "$LOCAL_HOST" "$REMOTE_HOST"; do
    grep -qE "^MODELS-LEG $h ENV:" "$out/$h.log" && env=1
done
for hr in "$LOCAL_HOST $lrc" "$REMOTE_HOST $rrc"; do
    set -- $hr; h=$1; rc=$2; receipt="$out/$h$RSUF.json"
    if [ ! -s "$receipt" ]; then
        echo "MODELS $h NO-GO: no receipt -- $(leg_reason "$h" "$rc")"; nogo=1; continue
    fi
    if [ "$MEASURE" = crux ]; then
        IFS=$'\t' read -r av cells red < <(mt_crux_fields "$receipt")
        if [ "$av" != "$want" ]; then
            echo "MODELS $h NO-GO: the receipt was measured by '$av', not '$want'"; nogo=1; continue
        fi
        echo "MODELS $h measured by $want: cells=$cells red=$red (crux_sweep_shards rc $rc)"; continue
    fi
    # tab-separated: apr_version itself contains spaces ("apr <v> (<sha9>)")
    IFS=$'\t' read -r av executed red < <(mt_receipt_fields "$receipt")
    if [ "$av" != "$want" ]; then
        echo "MODELS $h NO-GO: the receipt was measured by '$av', not '$want'"; nogo=1; continue
    fi
    echo "MODELS $h measured by $want: executed=$executed red=$red (model_ladder rc $rc)"
done

# MODELS_T1_SCOPE (optional) is passed to the judge as --scope. The nightly sets it to `none`: it judges
# the full ladder even where a release scope or the standing CRUX-smoke release policy covers $ver.
if [ "$MEASURE" = crux ]; then
    CRUX_CERT="$CERT" bash scripts/check_model_ladder.sh --version "$ver" --crux "$out" --cut-commit "$sha" ${MODELS_T1_SCOPE:+--scope "$MODELS_T1_SCOPE"} > "$out/judge.log" 2>&1; jrc=$?
else
bash scripts/check_model_ladder.sh --version "$ver" ${MODELS_T1_SCOPE:+--scope "$MODELS_T1_SCOPE"} --receipts "$out" > "$out/judge.log" 2>&1; jrc=$?
fi
case $jrc in
    0) # crux mode: a pass of no CRUX scope (the full ladder, with no ladder receipts here) judged nothing we measured
       if [ "$MEASURE" = crux ] && ! grep -qE '^(POLICY|SCOPED): ' "$out/judge.log"; then
           echo "MODELS NO-GO: the judge judged no CRUX scope for $ver (no POLICY:/SCOPED: line) -- crux mode measured nothing the release gate reads"; nogo=1
       fi ;;
    2) echo "MODELS NO-GO: the judge DECLINED (rc 2), and a decline is not a pass: $(tail -n 1 "$out/judge.log")"; nogo=1 ;;
    *) echo "MODELS NO-GO: the judge found red or missing cells (rc $jrc):"
       grep -E '^FAIL' "$out/judge.log" | head -n 40 | sed 's/^/MODELS   /'; nogo=1 ;;
esac
[ "$env" -eq 0 ] || exit 2
[ "$nogo" -eq 0 ] || exit 1
if [ "$MEASURE" = crux ]; then
    echo "MODELS GO (CRUX smoke) on $LOCAL_HOST and $REMOTE_HOST at $sha9: the judge passed both receipts ($want)"; exit 0
fi
echo "MODELS GO on $LOCAL_HOST and $REMOTE_HOST at $sha9: the judge passed both receipts ($want)"
exit 0
