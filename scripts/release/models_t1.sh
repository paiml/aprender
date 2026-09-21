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
# THEN scripts/check_model_ladder.sh --version <v> --receipts <out> judges both receipts.
#
# An unreachable host, a failed build, a binary that is not the release, a missing receipt, a red
# cell and a judge DECLINE (exit 2) are each NO-GO. None is a pass, and nothing is retried here.
#
# usage: models_t1.sh <version> <release-commit> <out-dir>
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

[ $# -eq 3 ] || { echo "usage: models_t1.sh <version> <release-commit> <out-dir>" >&2; exit 2; }
ver=$1; out=$3
[[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "models_t1: version '$ver' is not X.Y.Z" >&2; exit 2; }
sha=$(git rev-parse --verify --quiet "$2^{commit}") || { echo "models_t1: '$2' is not a commit in $(pwd)" >&2; exit 2; }
[ "$(git rev-parse HEAD)" = "$sha" ] \
    || { echo "models_t1: $(pwd) is at $(git rev-parse --short=9 HEAD), not the release commit ${sha:0:9}" >&2; exit 2; }
sha9=$(git rev-parse --short=9 "$sha")
want="apr $ver ($sha9)"
mkdir -p "$out" || exit 2
rm -f -- "$out/$LOCAL_HOST.json" "$out/$REMOTE_HOST.json" "$out/$REMOTE_HOST.json.part"

local_leg() {
    local tdir got
    cargo build --release -p apr-cli --bin apr --features cuda --locked \
        || { echo "MODELS-LEG $LOCAL_HOST BUILD-FAILED"; return 3; }
    tdir=${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')}
    got=$("$tdir/release/apr" --version 2>/dev/null | head -n 1)
    [ "$got" = "$want" ] || { echo "MODELS-LEG $LOCAL_HOST NOT-THE-RELEASE: '$got' (want '$want')"; return 3; }
    choom -n 1000 -- bash scripts/model_ladder.sh --host "$LOCAL_HOST" --out "$out"
}

remote_leg() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "bash -s" <<HOST
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
git -C "\$repo" fetch -q origin main && git -C "\$repo" cat-file -e "$sha^{commit}" \
  || { echo "MODELS-LEG $REMOTE_HOST FETCH-FAILED: $sha9 is not reachable from origin/main there"; exit 3; }
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
choom -n 1000 -- bash scripts/model_ladder.sh --host $REMOTE_HOST --out "\$dir/out"; lrc=\$?
if [ -f "\$dir/out/$REMOTE_HOST.json" ]; then
  echo "---RECEIPT $REMOTE_HOST---"; cat "\$dir/out/$REMOTE_HOST.json"; echo "---END RECEIPT---"
fi
git -C "\$repo" worktree remove --force "\$dir/wt" > /dev/null 2>&1
exit \$lrc
HOST
}

# leg_reason HOST RC -> why a leg produced no usable receipt, from its own log
leg_reason() {
    local r
    r=$(grep -E "^MODELS-LEG $1 " "$out/$1.log" | tail -n 1)
    if [ -n "$r" ]; then printf '%s (rc %s)' "${r#MODELS-LEG $1 }" "$2"
    elif [ "$1" = "$REMOTE_HOST" ] && [ "$2" = 255 ]; then printf 'unreachable over SSH (ssh rc 255)'
    else printf 'model_ladder.sh wrote no receipt (rc %s)' "$2"; fi
}

local_leg > "$out/$LOCAL_HOST.log" 2>&1 & lpid=$!
remote_leg > "$out/$REMOTE_HOST.log" 2>&1 & rpid=$!
wait "$lpid"; lrc=$?
wait "$rpid"; rrc=$?
sed -n "/^---RECEIPT $REMOTE_HOST---\$/,/^---END RECEIPT---\$/p" "$out/$REMOTE_HOST.log" | sed '1d;$d' > "$out/$REMOTE_HOST.json.part"
if [ -s "$out/$REMOTE_HOST.json.part" ]; then mv -- "$out/$REMOTE_HOST.json.part" "$out/$REMOTE_HOST.json"
else rm -f -- "$out/$REMOTE_HOST.json.part"; fi

nogo=0; env=0
for h in "$LOCAL_HOST" "$REMOTE_HOST"; do
    grep -qE "^MODELS-LEG $h ENV:" "$out/$h.log" && env=1
done
for hr in "$LOCAL_HOST $lrc" "$REMOTE_HOST $rrc"; do
    set -- $hr; h=$1; rc=$2; receipt="$out/$h.json"
    if [ ! -s "$receipt" ]; then
        echo "MODELS $h NO-GO: no receipt -- $(leg_reason "$h" "$rc")"; nogo=1; continue
    fi
    # tab-separated: apr_version itself contains spaces ("apr <v> (<sha9>)")
    IFS=$'\t' read -r av executed red < <(python3 -c '
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: print("UNREADABLE\t-\t-"); sys.exit(0)
print("\t".join(str(d.get(k, "-")) for k in ("apr_version", "executed", "red")))' "$receipt")
    if [ "$av" != "$want" ]; then
        echo "MODELS $h NO-GO: the receipt was measured by '$av', not '$want'"; nogo=1; continue
    fi
    echo "MODELS $h measured by $want: executed=$executed red=$red (model_ladder rc $rc)"
done

bash scripts/check_model_ladder.sh --version "$ver" --receipts "$out" > "$out/judge.log" 2>&1; jrc=$?
case $jrc in
    0) ;;
    2) echo "MODELS NO-GO: the judge DECLINED (rc 2), and a decline is not a pass: $(tail -n 1 "$out/judge.log")"; nogo=1 ;;
    *) echo "MODELS NO-GO: the judge found red or missing cells (rc $jrc):"
       grep -E '^FAIL' "$out/judge.log" | head -n 40 | sed 's/^/MODELS   /'; nogo=1 ;;
esac
[ "$env" -eq 0 ] || exit 2
[ "$nogo" -eq 0 ] || exit 1
echo "MODELS GO on $LOCAL_HOST and $REMOTE_HOST at $sha9: the judge passed both receipts ($want)"
exit 0
