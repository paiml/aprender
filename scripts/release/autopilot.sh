#!/usr/bin/env bash
# Release autopilot (PMAT-1098; APR-RELEASE-001 §4 T-0..T-4, rev 2026-09-16), first run for 0.68.2
# (EPIC #3477). Fail-closed: every irreversible step sits behind the repo's own gate. Terminal states
# only in STATUS, full logs beside it. Never --force, never --allow-dirty, never a skip.
# The train's identity is DERIVED (#3618, scripts/release/lib_release_params.sh): the version is the
# one argument; milestone, epic and the state dir AP are read from GitHub and the repo, never literals.
#
#   autopilot.sh <version> <bump-pr> [from-step] [to-step]
#   steps: wait deep dogfood models tag cleanroom assets preflight dryrun cascade install hosts postpub ledger close
#   T-4 for THIS train (operator 2026-09-17): cascade DRY-RUN receipt, then STOP and report — the cascade
#   itself is the operator's step. Default to-step is dryrun; `cascade` and later run only when named.
#   T-1 'ci / deep' has no workflow on main, so `deep` runs the equivalent locally on the release commit.
#   T-3 dispatches paiml/infra clean-room.yml with -f ref=<tag> (infra#621) and records the run id;
#   cascade-publish.sh refuses without a green `clean-room (aprender)` on exactly the tag commit.
#   (§4.1 freeze and §4.2 bump are prepare_bump.sh: they need review, so they are not in here)
set -uo pipefail
# D2/D3: $0-derived root, resolved BEFORE any cd and before the params that need it.
# NOT `git rev-parse --show-toplevel` -- git refuses a container bind-mounted tree (#3586).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || { echo "cannot resolve the repo root from $0" >&2; exit 2; }
# THE HOSTS THIS TRAIN VISITS, declared ONCE and walked by the hosts step below (#3732). The dogfood
# matrix is derived from what the train reaches, so `autopilot.sh --visited` prints exactly these,
# from these variables, and scripts/check_dogfood_matrix_is_visited.sh judges the gate's matrix
# against them: a host visited but not demanded, or demanded but not visited, is RED.
ASSET_HOSTS="gx10 aarch64-unknown-linux-gnu|yoga x86_64-unknown-linux-gnu"   # the CUDA release asset runs here
INSTALLER_HOSTS="intel --cpu|gx10 --cpu"                                        # install.sh at the tag runs here
TRAIN_HOST="${RELEASE_TRAIN_HOST:-lambda}"  # the host this train runs on (APR-RELEASE-001; ledger.py: host_class lambda)
matrix_hosts() { sed -n 's/^HOSTS="\(.*\)"$/\1/p' "$REPO_ROOT/scripts/check_multiplatform_dogfood.sh" | head -n 1; }
if [ "${1:-}" = "--visited" ]; then
  # no gh, no network, nothing run: one line per (host, how) the hosts step would reach
  printf '%s\n' "$ASSET_HOSTS" | tr '|' '\n' | while read -r h _; do printf 'VISITED %s release-asset\n' "$h"; done
  printf '%s\n' "$INSTALLER_HOSTS" | tr '|' '\n' | while read -r h _; do printf 'VISITED %s installer\n' "$h"; done
  mh=$(matrix_hosts); [ -n "$mh" ] || { echo "autopilot --visited: cannot read HOSTS from scripts/check_multiplatform_dogfood.sh" >&2; exit 2; }
  for h in $mh; do if [ "$h" = "$TRAIN_HOST" ]; then printf 'VISITED %s host-receipt-local\n' "$h"; else printf 'VISITED %s host-receipt-ssh\n' "$h"; fi; done
  exit 0
fi
# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
release_params "${1:-}" "$REPO_ROOT" || { echo "usage: autopilot.sh <version> <bump-pr> [from-step] [to-step]" >&2; exit 2; }
STATUS="$AP/STATUS"; LOG="$AP/autopilot.log"
PR="${2:?usage: autopilot.sh <version> <bump-pr> [from-step] [to-step]}"; FROM="${3:-wait}"; TO="${4:-dryrun}"
STEPS=(wait deep dogfood models tag cleanroom assets preflight dryrun cascade install hosts postpub ledger close)
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATUS" >> "$LOG"; }
die() { say "STOP $*"; exit 1; }
run_step() { # run_step <name>: true when <name> is at or after FROM and at or before TO
    local s seen=0 past=0
    for s in "${STEPS[@]}"; do
        [ "$s" = "$FROM" ] && seen=1
        [ "$s" = "$1" ] && { [ $seen = 1 ] && [ $past = 0 ]; return; }
        [ "$s" = "$TO" ] && past=1
    done
    return 1
}
# D1: pure-bash membership. `producer | grep -q` returns 141 on SIGPIPE under pipefail,
# so the old form could fail step-name validation for a reason unrelated to the step name.
case " ${STEPS[*]} " in *" $FROM "*) ;; *) die "unknown step '$FROM' (${STEPS[*]})" ;; esac
case " ${STEPS[*]} " in *" $TO "*) ;; *) die "unknown to-step '$TO' (${STEPS[*]})" ;; esac
# the milestone and epic, derived once, before any step can need them (#3618)
MS=$(release_milestone_number) || die "milestone $V: cannot resolve exactly one"
EPIC=$(release_epic_number) || die "release epic for $V: cannot resolve exactly one"
say "PARAMS V=$V T=$T milestone=#$MS epic=#$EPIC AP=$AP"
CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
export PATH="$CARGO_BIN:$PATH"
unset CARGO_REGISTRY_TOKEN
WT="$AP/wt"
say "START autopilot pid=$$ pr=#$PR from=$FROM to=$TO"

# 1. wait: the bump PR merges; its merge commit is the release commit
if run_step wait; then
  while :; do
    s=$(gh pr view "$PR" --repo $REPO --json state -q .state) || s=unknown
    [ "$s" = MERGED ] && break; [ "$s" = CLOSED ] && die "#$PR closed unmerged"
    sleep 300
  done
fi
MC=$(gh pr view "$PR" --repo $REPO --json mergeCommit -q .mergeCommit.oid)
[ -n "$MC" ] || die "#$PR has no merge commit"
say "RELEASE COMMIT $MC (#$PR)"
cd "$REPO_ROOT" || die "no repo"
git fetch -q origin main >> "$LOG" 2>&1 || die "fetch failed"
git merge-base --is-ancestor "$MC" origin/main || die "merge commit $MC not on origin/main"
if [ ! -d "$WT" ] || [ "$(git -C "$WT" rev-parse HEAD 2>/dev/null)" != "$MC" ]; then
  [ -d "$WT" ] && git worktree remove --force "$WT" >> "$LOG" 2>&1
  git worktree add --detach "$WT" "$MC" >> "$LOG" 2>&1 || die "worktree add failed"
fi
cd "$WT" || die "cd $WT"
v=$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["packages"][0]["version"])')
[ "$v" = "$V" ] || die "release commit carries version $v, not $V"
bash scripts/bump-version.sh --check >> "$LOG" 2>&1 || die "bump-version.sh --check: the workspaces disagree on the version"
export CARGO_TARGET_DIR="$REPO_ROOT/target"


# 1b. deep (T-1): no `ci / deep` workflow exists on main, so the local equivalent runs on THIS commit.
#     doctests + examples must be rc=0. `--no-default-features` carries the standing #3176 class
#     (every error inside aprender-distribute, identical at v0.66/v0.67): recorded, not blocking;
#     any error OUTSIDE that crate is RED and stops the train.
if run_step deep; then
  cargo test --doc --workspace --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute > "$AP/deep-doctests.log" 2>&1; rc=$?
  say "DEEP doctests rc=$rc: $(grep -E '^test result' "$AP/deep-doctests.log" | awk '{p+=$4; f+=$6} END {print p" passed, "f" failed"}')"
  [ $rc -eq 0 ] || die "T-1 doctests RED ($AP/deep-doctests.log)"
  cargo check --workspace --no-default-features > "$AP/deep-nodefault.log" 2>&1; rc=$?
  other=$(grep -E '^error' -A3 "$AP/deep-nodefault.log" | grep -E '^\s+--> ' | grep -vc 'crates/aprender-distribute/' || true)
  say "DEEP --no-default-features rc=$rc errors_outside_aprender-distribute=$other (standing #3176 class is inside it)"
  [ "$other" = 0 ] || die "T-1 --no-default-features RED outside the #3176 class ($AP/deep-nodefault.log)"
  cargo build --workspace --examples > "$AP/deep-examples.log" 2>&1; rc=$?
  say "DEEP examples build rc=$rc"
  [ $rc -eq 0 ] || die "T-1 examples RED ($AP/deep-examples.log)"
  say "DEEP GO at $MC (doctests, examples green; --no-default-features within #3176)"
fi

# 2. dogfood: the R5 receipt, pre-publish, FULL, on THIS commit -- never inherited (#3708)
#    The T-2 inheritance (operator 2026-09-17) was withdrawn by the cop's ruling on #3708 (2026-09-21).
#    It copied the PARENT's preflight GO forward when the bump diff was version-surface only, but the
#    parent's receipt is at the OLD version, so version-keyed rows (check_model_ladder) were never
#    measured at the release version, and R5 at T-4 refused it anyway: v0.69.0 stopped at the publish
#    preflight 43 min after tagging and ran the real dogfood then, with the tag already public.
#    Now the real dogfood runs here, and R5 is judged HERE by the same function T-4 uses
#    (check_publish_preflight.sh --receipt-only), on the same receipt file in this worktree, so a
#    receipt T-4 would refuse stops the train before any tag exists.
if run_step dogfood; then
  bash scripts/dogfood.sh --phase pre-publish > "$AP/dogfood-pre-publish.log" 2>&1; rc=$?
  grep -E 'VERDICT' "$AP/dogfood-pre-publish.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "dogfood pre-publish NO-GO rc=$rc ($AP/dogfood-pre-publish.log)"
  [ -z "$(git status --porcelain)" ] || die "tree dirty after dogfood: $(git status --porcelain | head -3 | tr '\n' ' ')"
  bash scripts/check_publish_preflight.sh --receipt-only > "$AP/dogfood-r5.log" 2>&1; rc=$?
  tail -2 "$AP/dogfood-r5.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "T-1 R5 refused the dogfood receipt rc=$rc: the T-4 publish gate would refuse it too, so nothing is tagged ($AP/dogfood-r5.log)"
  say "DOGFOOD GO at $MC (R5 holds at T-1)"
fi


# 2b. models (#3717, #3712 done_when 3): the model matrix, measured AT THIS COMMIT on BOTH hosts
#     before any tag -- lambda here, gx10 over the operator-authorized lambda->gx10 SSH, each with an
#     apr built from $MC and proved to be it. One red cell, an unreachable host, a failed build, a
#     missing receipt or a judge decline is a STOP here, with no tag cut. The same judge re-reads the
#     receipts committed in the bump at T-4 (check_publish_preflight.sh R7).
if run_step models; then
  bash scripts/release/models_t1.sh "$V" "$MC" "$AP/models-t1" > "$AP/models-t1.log" 2>&1; rc=$?
  grep -E '^MODELS ' "$AP/models-t1.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "T-1 model matrix NO-GO rc=$rc: nothing is tagged ($AP/models-t1.log)"
  say "MODELS GO at $MC on lambda and gx10"
fi
# 3. tag + release (binary-release.yml fires on release: published, from the TAG's workflow file)
# cut_tag <version> <tag> <commit> -- PMAT-3459. The milestone gate lives INSIDE the
# function that tags, ahead of `git tag`, so the tag cannot be cut without it: there is
# no path through cut_tag() that reaches `git tag` with the gate unsatisfied. v0.68.1
# was tagged 15:06:29Z by a copy of this script that read no milestone at all, six
# minutes after #3455 merged the gate (`grep -c check_milestone_cut autopilot.sh` = 0).
# Fail-closed on BOTH non-zero codes, and they are different failures:
#   1 = the milestone holds open item(s)      -> no tag, no publish
#   2 = the gate could not judge (Unknown)    -> no tag. Never a silent pass.
# scripts/check_tag_step_gated.sh runs this function against stubs and requires each
# of those three paths, plus a gate-call-removed MUTANT, to behave as stated.
cut_tag() {
    local v=$1 t=$2 mc=$3 rc=0
    bash "$REPO_ROOT/scripts/check_milestone_cut.sh" "$v" >> "$LOG" 2>&1 || rc=$?
    case "$rc" in
        0) say "MILESTONE-GATE $v clean at the cut (check_milestone_cut.sh rc=0)" ;;
        1) die "milestone $v still holds open item(s) -- no tag, no publish (check_milestone_cut.sh rc=1)" ;;
        *) die "milestone $v could not be judged (check_milestone_cut.sh rc=$rc) -- no tag; Unknown is not a pass" ;;
    esac
    git tag -a "$t" -m "aprender $t" "$mc" >> "$LOG" 2>&1 || die "tag failed"
    git push origin "$t" >> "$LOG" 2>&1 || die "tag push failed"
}
if run_step tag; then
  git rev-parse -q --verify "refs/tags/$T" > /dev/null && die "tag $T already exists locally"
  [ -f "$AP/release_notes.md" ] || die "no $AP/release_notes.md (prepare_bump.sh writes it from CHANGELOG [$V])"
  cut_tag "$V" "$T" "$MC"
  say "TAGGED $T at $MC"
  gh release create "$T" --repo $REPO --verify-tag --title "aprender $V" --notes-file "$AP/release_notes.md" >> "$LOG" 2>&1 || die "gh release create failed"
  say "RELEASED $(gh release view "$T" --repo $REPO --json url -q .url)"
fi


# 3b. cleanroom (T-3): dispatch paiml/infra clean-room.yml ON THE TAG (infra#621 ref input), record the
#     run id, wait, require the `clean-room (aprender)` job green. The run's own first step asserts
#     HEAD == the ref; cascade-publish.sh re-derives all of this fail-closed before T-4.
if run_step cleanroom; then
  # B2-cpu: paiml/infra clean-room.yml on the tag. Attach to a run already dispatched (cleanroom-attach) or dispatch.
  if [ -s "$AP/cleanroom-attach" ]; then
    crun=$(cat "$AP/cleanroom-attach"); say "CLEANROOM attached to run $crun"
  else
    t0=$(date -u +%s)  # bashrs disable-line=DET002
    gh workflow run clean-room.yml --repo $INFRA -f repos=aprender -f ref="$T" >> "$LOG" 2>&1 || die "clean-room.yml dispatch on $T failed"
    say "CLEANROOM dispatched on $T"
    crun=""; for _ in $(seq 1 30); do
      crun=$(gh run list --repo $INFRA --workflow clean-room.yml --event workflow_dispatch --limit 10 --json databaseId,createdAt --jq "[.[] | select((.createdAt | fromdateiso8601) >= $t0 - 30)] | sort_by(.createdAt) | last | .databaseId // empty")
      [ -n "$crun" ] && break; sleep 20
    done
    [ -n "$crun" ] || die "no clean-room.yml workflow_dispatch run appeared after the dispatch"
    say "CLEANROOM RUN $crun"
  fi
  # B2-gpu: paiml/aprender b2-gpu.yml on the same ref (the org GPU runner groups admit aprender only). In parallel.
  if [ -s "$AP/b2gpu-attach" ]; then
    grun=$(cat "$AP/b2gpu-attach"); say "B2-GPU attached to run $grun"
  else
    t1=$(date -u +%s)  # bashrs disable-line=DET002
    gh workflow run b2-gpu.yml --repo $REPO --ref main -f ref="$T" >> "$LOG" 2>&1 || die "b2-gpu.yml dispatch on $T failed (is aprender#3467 merged?)"
    grun=""; for _ in $(seq 1 30); do
      grun=$(gh run list --repo $REPO --workflow b2-gpu.yml --event workflow_dispatch --limit 10 --json databaseId,createdAt --jq "[.[] | select((.createdAt | fromdateiso8601) >= $t1 - 30)] | sort_by(.createdAt) | last | .databaseId // empty")
      [ -n "$grun" ] && break; sleep 20
    done
    [ -n "$grun" ] || die "no b2-gpu.yml run appeared after the dispatch"
    say "B2-GPU RUN $grun"
  fi
  # the JOB conclusion, not the run status: a sibling job that can never start must not hold the verdict hostage
  jc=""; for _ in $(seq 1 240); do
    jc=$(gh run view "$crun" --repo $INFRA --json jobs --jq '.jobs[] | select(.name=="clean-room (aprender)") | select(.status=="completed") | .conclusion' | head -1)
    [ -n "$jc" ] && break; sleep 60
  done
  [ "$jc" = success ] || die "clean-room (aprender) on $T concluded '${jc:-absent}' (run $crun)"
  say "B2-CPU GREEN on $T (infra run $crun)"
  gc=""; for _ in $(seq 1 120); do
    gs=$(gh run view "$grun" --repo $REPO --json status,conclusion,headSha --jq '"\(.status) \(.conclusion)"'); case "$gs" in completed*) gc=${gs#completed }; break;; esac; sleep 60
  done
  [ "$gc" = success ] || die "b2-gpu on $T concluded '${gc:-absent}' (aprender run $grun)"
  ok=0; for _ in 1 2 3 4 5 6; do gh run view "$grun" --repo $REPO --log > "$AP/b2gpu-run.log" 2>/dev/null; grep -q "tested-sha: $MC" "$AP/b2gpu-run.log" && { ok=1; break; }; sleep 30; done; [ $ok = 1 ] || die "b2-gpu run $grun did not test $MC"
  printf '%s\n' "$crun" > "$AP/cleanroom-run-id"; printf '%s\n' "$grun" > "$AP/b2gpu-run-id"
  say "CLEANROOM GREEN on $T: B2-cpu infra run $crun + B2-gpu aprender run $grun, both on $MC"
fi

# 4. assets: the release run completes and all sixteen assets are on the release, checked by command
if run_step assets; then
  run=""; for _ in $(seq 1 40); do
    run=$(gh run list --repo $REPO --workflow binary-release.yml --event release --limit 10 --json databaseId,headBranch --jq ".[] | select(.headBranch==\"$T\") | .databaseId" | head -1)
    [ -n "$run" ] && break; sleep 30
  done
  [ -n "$run" ] || die "no binary-release run for $T after 20 min"
  say "ASSET RUN $run"
  for _ in $(seq 1 240); do
    s=$(gh run view "$run" --repo $REPO --json status -q .status)
    [ "$s" = completed ] && break
    sleep 60
  done
  c=$(gh run view "$run" --repo $REPO --json conclusion -q .conclusion)
  gh api "repos/$REPO/actions/runs/$run/jobs?per_page=100" --jq '.jobs[] | "  \(.name) = \(.conclusion)"' >> "$STATUS"
  [ "$c" = success ] || die "binary-release run $run concluded '$c' (jobs above)"
  bash scripts/check_release_assets.sh "$T" > "$AP/assets.log" 2>&1; rc=$?
  tail -3 "$AP/assets.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "check_release_assets.sh $T rc=$rc (1 = missing, 2 = ENV)"
  say "ASSETS all present on $T (run $run)"
fi

# 5. preflight (R1-R6; R5 reads the pre-publish receipt in this worktree)
if run_step preflight; then
  bash scripts/check_publish_preflight.sh > "$AP/preflight.log" 2>&1; rc=$?
  tail -3 "$AP/preflight.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "publish preflight refused rc=$rc"
  say "PREFLIGHT PASS"
fi


# 5b. dryrun (T-4 receipt, operator 2026-09-17): the cascade's own --check on the tagged tree, recorded.
#     It is a RECEIPT, not a stop. APR-RELEASE-001 at the release commit says T-4 is "Unattended under
#     standing operator authorization (2026-09-17)" (spec line 174 and the T-4 row), 0.68.1 shipped that
#     way (attended_min 0), and the operator reconfirmed it verbatim on 2026-09-20: "tell agent to
#     auto-publish and never wait for me." The 0.68.2 run logged the old wording and did NOT stop
#     (TO=close), but the line read as a park and cost a minute of reading -- so it now says what it is.
if run_step dryrun; then
  bash scripts/cascade-publish.sh --check > "$AP/cascade-check.log" 2>&1; rc=$?
  behind=$(grep -cE "\(want ${V//./\\.}\)" "$AP/cascade-check.log" || true)
  tail -3 "$AP/cascade-check.log" >> "$STATUS"
  say "DRYRUN cascade-publish.sh --check rc=$rc crates_behind=$behind ($AP/cascade-check.log)"
  say "DRYRUN-RECEIPT T-4 proceeds unattended (standing authorization 2026-09-17, reconfirmed by the operator 2026-09-20 \"never wait for me\"); receipts: dogfood GO, tag $T, clean-room run $(cat "$AP/cleanroom-run-id" 2>/dev/null), assets, preflight, dry-run"
fi

# 6. crates.io cascade: multi-pass drain, then the check is the verdict (pass 1 exiting 1 is normal)
if run_step cascade; then
  bash scripts/cascade-drain.sh --target "$V" --passes 30 > "$AP/cascade.log" 2>&1; rc=$?
  say "CASCADE drain rc=$rc"
  bash scripts/cascade-publish.sh --check > "$AP/cascade-check.log" 2>&1; rc2=$?
  behind=$(grep -cE "\(want ${V//./\\.}\)" "$AP/cascade-check.log" || true)
  say "CASCADE check rc=$rc2 crates_behind=$behind"
  [ "$behind" = 0 ] || die "cascade incomplete: $behind crate(s) behind ($AP/cascade-check.log)"
  say "PUBLISHED $V on crates.io"
fi

# 7. install: the published crate installs on the train host. The post-publish dogfood is NOT here any
#    more (#3731): it judges the host receipts, and those are written by `hosts`, so it runs in
#    `postpub`, after them. Here it could only ever read receipts that did not exist yet.
if run_step install; then
  sleep "${RELEASE_INDEX_SETTLE_S:-180}"  # the sparse index lags a publish; the case table sets 0
  cargo install aprender --version "$V" --force > "$AP/install.log" 2>&1; rc=$?
  say "INSTALL cargo install aprender --version $V rc=$rc: $("$CARGO_BIN/apr" --version 2>&1 | head -1)"
  [ $rc -eq 0 ] || die "post-publish install failed"
fi

# 8. hosts: the RELEASE ASSET (not a local build) downloads, verifies and runs on gx10 and yoga; the
#    installer runs on intel and gx10; and every host in the multi-platform gate's matrix produces
#    its post-publish receipt in the gate's schema (#3731).
#    INFRASTRUCTURE IS NEVER A HOST VERDICT (#3544 item 4). The receipt dir is created before the
#    first write, and every write target is proved writable before a host runs into it, so a missing
#    dir is ONE error naming the dir -- measured 2026-09-20 (v0.68.2): with no dir, every
#    `> $AP/receipts/...` failed and four green hosts were reported as four failed receipts. An ssh
#    that cannot connect (rc 255) and a host that returns no parseable receipt are INFRA lines too,
#    kept apart from the host count. Either kind still STOPs the train: fail closed, honestly named.
if run_step hosts; then
  RDIR="$AP/receipts"
  mkdir -p "$RDIR/dogfood" || die "INFRA cannot create the receipt dir $RDIR/dogfood -- no host was judged"
  # receipt_to <file>: <file> can be written, or the step dies naming the directory it could not write
  receipt_to() { : > "$1" 2> /dev/null || die "INFRA the receipt dir $(dirname "$1") is missing or unwritable -- a write into it failed, so no host was judged (not a host verdict)"; }
  fails=0; infra=""
  SSH_FAILED=255  # ssh(1) exits 255 when IT failed (no route, refused, auth) -- never the remote command's verdict
  while IFS= read -r hp; do
    set -- $hp; h=$1; tgt=$2; a="apr-$T-$tgt-cuda"
    receipt_to "$RDIR/$h.txt"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "bash -s" > "$RDIR/$h.txt" 2>&1 <<HOST
set -e; d=\$(mktemp -d); cd "\$d"
u=https://github.com/$REPO/releases/download/$T
curl -sSfLO "\$u/$a.tar.gz"; curl -sSfLO "\$u/$a.tar.gz.sha256"
sha256sum -c "$a.tar.gz.sha256"; tar xzf "$a.tar.gz"
echo "version: \$("./$a/apr" --version | head -1)"
echo "devices:"; "./$a/apr" devices --json || echo "(apr devices --json failed)"
cd /; rm -rf -- "\$d"
HOST
    rc=$?
    if [ "$rc" -eq "$SSH_FAILED" ]; then infra="$infra $h(ssh-255)"; say "INFRA $h unreachable: ssh rc=255 -- not a host verdict"; continue; fi
    say "HOST $h rc=$rc: $(grep -m1 '^version:' "$RDIR/$h.txt")"
    [ $rc -eq 0 ] || fails=$((fails + 1))
  done <<< "$(printf '%s\n' "$ASSET_HOSTS" | tr '|' '\n')"
  # T-2 installer rows (APR-RELEASE-001 §4, 2026-09-16, #3366): install.sh at THE TAG, on a clean intel
  # (x86_64) and a clean gx10 (aarch64): asset resolves, sha256 verifies (the script refuses otherwise),
  # `apr --version` prints the version. Fetched from the URL the script advertises, pinned to the tag.
  while IFS= read -r hv; do
    set -- $hv; h=$1; variant=$2
    receipt_to "$RDIR/install-$h.txt"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "bash -s" > "$RDIR/install-$h.txt" 2>&1 <<HOST
set -e; d=\$(mktemp -d); export INSTALL_DIR="\$d/bin"
curl -LsSf https://raw.githubusercontent.com/$REPO/$T/scripts/install.sh | sh -s -- --version $T $variant  # bashrs disable-line=SEC002,SEC008,SEC015
v=\$("\$INSTALL_DIR/apr" --version | head -1); echo "installer: \$v"
case "\$v" in *"$V"*) ;; *) echo "VERSION MISMATCH: \$v"; exit 1;; esac
cd /; rm -rf -- "\$d"
HOST
    rc=$?
    if [ "$rc" -eq "$SSH_FAILED" ]; then infra="$infra install-$h(ssh-255)"; say "INFRA $h unreachable: ssh rc=255 -- not a host verdict"; continue; fi
    say "INSTALLER $h rc=$rc: $(grep -m1 '^installer:' "$RDIR/install-$h.txt")"
    [ $rc -eq 0 ] || fails=$((fails + 1))
  done <<< "$(printf '%s\n' "$INSTALLER_HOSTS" | tr '|' '\n')"
  # The post-publish host receipts (#3731, #3544 item 2). The matrix is READ from the gate that
  # judges it (check_multiplatform_dogfood.sh HOSTS), never listed a second time here. Each host runs
  # scripts/release/host_receipt.sh from a KIT archived from the release commit -- never its own
  # checkout (gx10's was 24 days stale) -- in parallel; the train host runs it locally, the rest over
  # ssh. Its GPU measurements queue at --gpu-prio 1: these receipts are release-blocking (rule rev 5).
  # A receipt's CONTENT is judged by the gate in `postpub`, not here: here a host either returned
  # a parseable receipt for ($V, itself), or it is an INFRA line.
  rhosts=$(matrix_hosts)
  [ -n "$rhosts" ] || die "INFRA cannot read the HOSTS matrix from scripts/check_multiplatform_dogfood.sh"
  train_host="$TRAIN_HOST"
  # The kit is the release commit's manifest and its WHOLE scripts/ tree (~1.5 MB gzipped): the block
  # producers source and read helpers beside them, and a hand-kept file list had already drifted --
  # the 0.68.2 first-green run's parity died on gx10 with "scripts/perf_isolation.sh: No such file".
  # It also carries the JSON TWINS of the two YAML files the block helpers read (scripts/lib/yaml_twin.py),
  # written HERE, where PyYAML exists: mini's python3 cannot import yaml, and the twins let it read them
  # with the standard library. Twins live only in the kit, never in the tree, so they cannot drift.
  receipt_to "$RDIR/kit.tar"
  ks="$RDIR/kit"
  if [ -n "$ks" ] && [ "$ks" != "/" ]; then rm -rf -- "$ks"; fi
  mkdir -p "$RDIR/kit" || die "INFRA cannot create the kit dir $RDIR/kit"
  git archive --format=tar "$MC" -- Cargo.toml scripts | tar -C "$ks" -xf - \
    && ( cd "$ks" && python3 scripts/lib/yaml_twin.py write scripts/perf-matrix.yaml scripts/perf-receipt-fields.yaml ) >> "$LOG" 2>&1 \
    && tar -C "$ks" -cf "$RDIR/kit.tar" . \
    || die "INFRA cannot build the host-receipt kit (with its JSON twins) from $MC"
  rpids=""
  for h in $rhosts; do
    receipt_to "$RDIR/dogfood/$h.out"
    if [ "$h" = "$train_host" ]; then
      bash -c 'd=$(mktemp -d) || exit 2; tar -C "$d" -xf "$1" || exit 2; bash "$d/scripts/release/host_receipt.sh" "$2" "$3" --gpu-prio 1; rc=$?; if [ -n "$d" ] && [ "$d" != "/" ]; then rm -rf -- "$d"; fi; exit $rc' \
        _ "$RDIR/kit.tar" "$V" "$h" > "$RDIR/dogfood/$h.out" 2>&1 &
    else
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
        'd=$(mktemp -d) || exit 2; tar -C "$d" -xf - || exit 2; bash "$d/scripts/release/host_receipt.sh" '"$V $h"' --gpu-prio 1; rc=$?; if [ -n "$d" ] && [ "$d" != "/" ]; then rm -rf -- "$d"; fi; exit $rc' \
        < "$RDIR/kit.tar" > "$RDIR/dogfood/$h.out" 2>&1 &
    fi
    rpids="$rpids $h:$!"
  done
  for hp in $rpids; do
    h=${hp%%:*}; wait "${hp#*:}"; rc=$?
    if [ "$rc" -eq "$SSH_FAILED" ] && [ "$h" != "$train_host" ]; then infra="$infra receipt-$h(ssh-255)"; say "INFRA $h unreachable: ssh rc=255, no receipt -- not a host verdict"; continue; fi
    # parsed, never grepped: the ---RECEIPT--- block must be JSON naming this host and this version
    # (python, never jq: with its exit-status flag, jq 1.6 passes an absent receipt, #3554)
    line=$(python3 - "$RDIR/dogfood/$h.out" "$h" "$V" "$RDIR/dogfood/$h.json" <<'PY'
import json, sys
out, h, v, dst = sys.argv[1:5]
lines = open(out, errors="replace").read().splitlines()
try:
    a = lines.index(f"---RECEIPT {h}---"); b = lines.index("---END RECEIPT---", a)
except ValueError:
    print(f"no receipt block in {out} (last line: {(lines[-1] if lines else '<empty>')[:160]})"); sys.exit(3)
try:
    r = json.loads("\n".join(lines[a + 1:b]))
except ValueError as e:
    print(f"the receipt block does not parse: {e}"); sys.exit(3)
if not isinstance(r, dict) or r.get("host") != h or r.get("version_tested") != v:
    print(f"the receipt names host={r.get('host') if isinstance(r, dict) else '?'} version={r.get('version_tested') if isinstance(r, dict) else '?'}, not {h} {v}"); sys.exit(3)
json.dump(r, open(dst, "w"), indent=2); open(dst, "a").write("\n")
blk = lambda k: "yes" if r.get(k) else ("refused" if r.get(k + "_attempt") else "none")
print(f"install_rc={r.get('install_rc')} generate={'sane' if (r.get('generate') or {}).get('output_sane') else ('present' if r.get('generate') else 'null')} "
      f"bench={blk('bench')} parity={blk('parity')} unmeasured={len(r.get('unmeasured') or [])}")
PY
    ); prc=$?
    if [ $prc -ne 0 ]; then infra="$infra receipt-$h"; say "INFRA $h returned no receipt (host_receipt rc=$rc): $line -- not a host verdict"; continue; fi
    say "HOST-RECEIPT $h $line ($RDIR/dogfood/$h.json)"
  done
  [ -z "$infra" ] || die "INFRA fault(s):$infra -- infrastructure, not host verdicts; $fails host/installer receipt(s) failed besides ($RDIR/)"
  [ $fails -eq 0 ] || die "$fails host/installer receipt(s) failed ($RDIR/)"
fi

# 8b. postpub (#3731; #3544 item 1): the post-publish dogfood, AFTER `hosts`, reading the receipts that
#     step just wrote (DOGFOOD_RECEIPTS_DIR; the tag's tree cannot contain files made after it). FAIL
#     CLOSED: a NO-GO is a STOP, not a line in a log. And the receipt is read, not only the rc: a
#     DEFER that survives into post-publish is RED even if something upstream exits 0 over it.
if run_step postpub; then
  [ -d "$AP/receipts/dogfood" ] || die "INFRA no $AP/receipts/dogfood -- the hosts step has not run, so there is nothing to judge"
  DOGFOOD_RECEIPTS_DIR="$AP/receipts/dogfood" bash scripts/dogfood.sh --phase post-publish > "$AP/dogfood-post-publish.log" 2>&1; rc=$?
  grep -E 'VERDICT' "$AP/dogfood-post-publish.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "post-publish dogfood NO-GO rc=$rc ($AP/dogfood-post-publish.log)"
  line=$(python3 - "$PWD/.dogfood" "$MC" <<'PY'
import glob, json, os, sys
d, mc = sys.argv[1:3]
rs = []
for p in sorted(glob.glob(os.path.join(d, "receipt-*.json"))):
    try: r = json.load(open(p))
    except ValueError: continue
    if r.get("phase") == "post-publish" and r.get("commit") == mc: rs.append((p, r))
if not rs: print(f"no post-publish receipt for {mc} under {d}"); sys.exit(1)
p, r = rs[-1]
if r.get("verdict") != "GO": print(f"{os.path.basename(p)} verdict={r.get('verdict')}"); sys.exit(1)
if r.get("deferred"): print(f"{os.path.basename(p)} still DEFERS {', '.join(r['deferred'])} after the publish"); sys.exit(1)
print(os.path.basename(p))
PY
  ) || die "post-publish dogfood refused on its receipt: $line"
  say "DOGFOOD post-publish GO at $MC ($line; host receipts $AP/receipts/dogfood)"
  # A block that runs as REPORT is named in the release notes, never only in a log (cop ruling on
  # #3731): the gate's REPORT rows over these same receipts are appended to the release, once.
  DOGFOOD_RECEIPTS_DIR="$AP/receipts/dogfood" bash scripts/check_multiplatform_dogfood.sh > "$AP/multiplatform.log" 2>&1
  grep -E '^REPORT ' "$AP/multiplatform.log" > "$AP/multiplatform-report.txt" || true
  if [ -s "$AP/multiplatform-report.txt" ]; then
    gh release view "$T" --repo $REPO --json body -q .body > "$AP/release-notes.md" \
      || die "cannot read the $T release notes to name its REPORT rows ($AP/multiplatform-report.txt)"
    if ! grep -qF '## Post-publish host receipts' "$AP/release-notes.md"; then
      { printf '\n## Post-publish host receipts: rows that ran as REPORT\n\n'; sed 's/^REPORT */- /' "$AP/multiplatform-report.txt"; } >> "$AP/release-notes.md"
      gh release edit "$T" --repo $REPO --notes-file "$AP/release-notes.md" >> "$LOG" 2>&1 \
        || die "cannot write the REPORT rows into the $T release notes ($AP/multiplatform-report.txt)"
    fi
    say "REPORT rows named in the $T release notes: $(grep -c . "$AP/multiplatform-report.txt")"
  fi
fi

# 8c. ledger (#3731, cop ruling 2026-09-21): the host receipts and the ledger record leave $AP -- which
#     is reclaimable -- on a `ledger/<V>` branch, with a PR opened UNARMED (the cop folds and arms it).
#     A ledger left as a hand step is how 0.68.x ledgers went missing, so a ledger that cannot be
#     committed, pushed or opened STOPs the train here, before `close` declares it done. Never a
#     force push: a re-run builds on the branch it already pushed.
if run_step ledger; then
  nrec=0; for f in "$AP"/receipts/dogfood/*.json; do [ -f "$f" ] && nrec=$((nrec + 1)); done
  [ "$nrec" -gt 0 ] || die "INFRA no host receipts under $AP/receipts/dogfood -- the hosts step has not run, so there is nothing to ledger"
  day=$(date -u +%F)  # bashrs disable-line=DET002
  python3 "$REPO_ROOT/scripts/release/ledger.py" "$AP" "$MC" "$T" "$V" "$STATUS" >> "$LOG" 2>&1 || die "ledger.py wrote no ledger record ($LOG)"
  rec="$AP/${MC:0:9}-lambda-vector-train.json"
  [ -s "$rec" ] || die "no ledger record at $rec"
  lb="ledger/$V"; lw="$AP/ledger-wt"; lbase=origin/main
  git fetch -q origin main >> "$LOG" 2>&1 || die "fetch of main failed; nothing ledgered"
  if git ls-remote --exit-code --heads origin "$lb" > /dev/null 2>&1; then
    git fetch -q origin "+refs/heads/$lb:refs/remotes/origin/$lb" >> "$LOG" 2>&1 || die "fetch of the existing $lb failed"
    lbase="origin/$lb"
  fi
  if [ -d "$lw" ]; then git worktree remove --force "$lw" >> "$LOG" 2>&1 || die "cannot remove the stale $lw"; fi
  git worktree add -q -B "$lb" "$lw" "$lbase" >> "$LOG" 2>&1 || die "cannot create the $lb worktree at $lw"
  mkdir -p "$lw/evidence/dogfood/$V" "$lw/docs/build-ledger/$day" \
    && cp -- "$AP"/receipts/dogfood/*.json "$lw/evidence/dogfood/$V/" && cp -- "$rec" "$lw/docs/build-ledger/$day/" \
    || die "cannot stage the receipts and the ledger record in $lw"
  git -C "$lw" add -- "evidence/dogfood/$V" "docs/build-ledger/$day" || die "git add failed in $lw"
  if ! git -C "$lw" diff --cached --quiet; then
    git -C "$lw" commit -q -m "ledger: $V release train -- $nrec host receipt(s) + the train record (#3731)" >> "$LOG" 2>&1 || die "the ledger commit failed in $lw"
  fi
  git -C "$lw" push -q origin "$lb" >> "$LOG" 2>&1 || die "cannot push $lb: the receipts exist only in $AP"
  lpr=$(gh pr list --repo $REPO --head "$lb" --state open --json number -q '.[0].number') || die "cannot list the PRs of $lb"
  if [ -z "$lpr" ]; then
    lpr=$(gh pr create --repo $REPO --base main --head "$lb" --title "ledger: $V release train receipts" \
      --body "The $T release train's ledger, opened by scripts/release/autopilot.sh (#3731): evidence/dogfood/$V/ holds the $nrec post-publish host receipt(s) the train produced and judged, and docs/build-ledger/$day/ holds the train record. Opened UNARMED; the cop folds and arms it. Refs #$EPIC") \
      || die "cannot open the ledger PR for $lb"
  fi
  say "LEDGER PR $lpr ($lb): evidence/dogfood/$V/ ($nrec receipt(s)) + docs/build-ledger/$day/ -- opened unarmed"
fi

# 9. close: the epic hears the receipts and closes once it is the last open item; then the milestone
if run_step close; then
  # SEC010 (bashrs-gate): read the run id with the read builtin, not $(cat "$AP/…") -- AP is derived
  # now (#3655), so a cat over it inside the message is flagged as a path-traversal risk.
  cleanroom_run_id=unknown; IFS= read -r cleanroom_run_id < "$AP/cleanroom-run-id" 2>/dev/null || cleanroom_run_id=unknown
  gh issue comment "$EPIC" --repo $REPO --body "$T released: $(gh release view "$T" --repo $REPO --json url -q .url). Receipts: pre-publish dogfood GO at \`$MC\`, all release assets verified by \`scripts/check_release_assets.sh $T\`, crates.io cascade complete, the CUDA asset downloaded, verified and run on gx10 and yoga. clean-room (aprender) green on the tag (run ${cleanroom_run_id}), install.sh receipts on intel and gx10, post-publish dogfood GO over the host receipts of every host in the multi-platform matrix. Logs: $AP on $(hostname)." >> "$LOG" 2>&1 || say "WARN epic comment failed"
  # The epic is IN the milestone, so "0 open items" could never hold while it was open, and nothing
  # here closed it: 0.68.2's #3477 was closed by hand (#3708). Every OTHER item must be closed first;
  # then the epic, then the milestone. Any other open item refuses both.
  others=$(gh api "repos/$REPO/issues?milestone=$MS&state=open&per_page=100" --paginate --jq ".[] | select(.number != $EPIC) | .number" | tr '\n' ' ') \
    || die "cannot read milestone $V's open items; closing neither the epic nor the milestone"
  [ -z "${others// /}" ] || die "milestone $V still has open item(s) besides epic #$EPIC: $others; closing neither"
  estate=$(gh issue view "$EPIC" --repo $REPO --json state -q .state) || die "cannot read epic #$EPIC's state"
  if [ "$estate" = OPEN ]; then
    gh issue close "$EPIC" --repo $REPO --reason completed >> "$LOG" 2>&1 || die "closing epic #$EPIC failed"
    say "EPIC #$EPIC closed"
  fi
  open=$(gh api "repos/$REPO/milestones/$MS" --jq .open_issues); [ "$open" = 0 ] || die "milestone $V still has $open open item(s) after the epic; not closing"
  gh api -X PATCH "repos/$REPO/milestones/$MS" -f state=closed >> "$LOG" 2>&1 && say "MILESTONE $V closed"
  # the ledger record and the host receipts were committed and opened as a PR by the `ledger` step (#3731)
  say "DONE $V released, published, installed, receipts taken"
fi
