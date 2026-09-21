#!/usr/bin/env bash
# Release autopilot (PMAT-1098; APR-RELEASE-001 §4 T-0..T-4, rev 2026-09-16), first run for 0.68.2
# (EPIC #3477). Fail-closed: every irreversible step sits behind the repo's own gate. Terminal states
# only in STATUS, full logs beside it. Never --force, never --allow-dirty, never a skip.
# The train's identity is DERIVED (#3618, scripts/release/lib_release_params.sh): the version is the
# one argument; milestone, epic and the state dir AP are read from GitHub and the repo, never literals.
#
#   autopilot.sh <version> <bump-pr> [from-step] [to-step]
#   steps: wait deep dogfood tag cleanroom assets preflight cascade install hosts close
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
# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
release_params "${1:-}" "$REPO_ROOT" || { echo "usage: autopilot.sh <version> <bump-pr> [from-step] [to-step]" >&2; exit 2; }
STATUS="$AP/STATUS"; LOG="$AP/autopilot.log"
PR="${2:?usage: autopilot.sh <version> <bump-pr> [from-step] [to-step]}"; FROM="${3:-wait}"; TO="${4:-dryrun}"
STEPS=(wait deep dogfood tag cleanroom assets preflight dryrun cascade install hosts close)
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

# 2. dogfood: the R5 receipt, pre-publish, FULL, on THIS commit
if run_step dogfood; then
  # Inherited T-2 (operator 2026-09-17): if the bump diff touches only the version surface, the parent-sha
  # preflight GO is inherited and recorded; any other path -> full T-2.
  parent=$(git rev-parse "$MC^1"); inherit=0
  if [ -f "$AP/preflight-$parent.verdict" ] && grep -q '^GO ' "$AP/preflight-$parent.verdict"; then
    other=$(git diff --name-only "$parent" "$MC" | grep -vE '^(Cargo\.toml|Cargo\.lock|CHANGELOG\.md|crates/[^/]+/Cargo\.toml|crates/facades/[^/]+/Cargo\.toml|crates/facades/Cargo\.lock)$' | wc -l)
    [ "$other" -eq 0 ] && inherit=1
  fi
  if [ $inherit -eq 1 ]; then
    cp "$AP/preflight-$parent.log" "$AP/dogfood-pre-publish.log"
    printf 'inherited_from: %s\nrelease_commit: %s\nbump_diff: version surface only\n' "$parent" "$MC" > "$AP/dogfood-inherited.receipt"
    say "DOGFOOD GO at $MC (INHERITED from parent $parent preflight GO; bump diff = version surface only; receipt $AP/dogfood-inherited.receipt)"
  else
  bash scripts/dogfood.sh --phase pre-publish > "$AP/dogfood-pre-publish.log" 2>&1; rc=$?
  grep -E 'VERDICT' "$AP/dogfood-pre-publish.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "dogfood pre-publish NO-GO rc=$rc ($AP/dogfood-pre-publish.log)"
  [ -z "$(git status --porcelain)" ] || die "tree dirty after dogfood: $(git status --porcelain | head -3 | tr '\n' ' ')"
  say "DOGFOOD GO at $MC"
  fi
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

# 7. install + post-publish dogfood (recorded; the host receipts below are the fail-closed part)
if run_step install; then
  sleep 180
  cargo install aprender --version "$V" --force > "$AP/install.log" 2>&1; rc=$?
  say "INSTALL cargo install aprender --version $V rc=$rc: $("$CARGO_BIN/apr" --version 2>&1 | head -1)"
  [ $rc -eq 0 ] || die "post-publish install failed"
  bash scripts/dogfood.sh --phase post-publish > "$AP/dogfood-post-publish.log" 2>&1; rc=$?
  grep -E 'VERDICT' "$AP/dogfood-post-publish.log" >> "$STATUS"
  say "DOGFOOD post-publish rc=$rc"
fi

# 8. hosts: the RELEASE ASSET (not a local build) downloads, verifies and runs on gx10 and yoga
if run_step hosts; then
  # the receipt dir must exist BEFORE the first redirect: without it every `> $AP/receipts/...`
  # fails and four green hosts are reported as four failed receipts (measured 2026-09-20, v0.68.2).
  mkdir -p "$AP/receipts"
  fails=0
  for hp in "gx10 aarch64-unknown-linux-gnu" "yoga x86_64-unknown-linux-gnu"; do
    set -- $hp; h=$1; tgt=$2; a="apr-$T-$tgt-cuda"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "bash -s" > "$AP/receipts/$h.txt" 2>&1 <<HOST
set -e; d=\$(mktemp -d); cd "\$d"
u=https://github.com/$REPO/releases/download/$T
curl -sSfLO "\$u/$a.tar.gz"; curl -sSfLO "\$u/$a.tar.gz.sha256"
sha256sum -c "$a.tar.gz.sha256"; tar xzf "$a.tar.gz"
echo "version: \$("./$a/apr" --version | head -1)"
echo "devices:"; "./$a/apr" devices --json || echo "(apr devices --json failed)"
cd /; rm -rf -- "\$d"
HOST
    rc=$?; say "HOST $h rc=$rc: $(grep -m1 '^version:' "$AP/receipts/$h.txt")"
    [ $rc -eq 0 ] || fails=$((fails + 1))
  done
  # T-2 installer rows (APR-RELEASE-001 §4, 2026-09-16, #3366): install.sh at THE TAG, on a clean intel
  # (x86_64) and a clean gx10 (aarch64): asset resolves, sha256 verifies (the script refuses otherwise),
  # `apr --version` prints the version. Fetched from the URL the script advertises, pinned to the tag.
  for hv in "intel --cpu" "gx10 --cpu"; do
    set -- $hv; h=$1; variant=$2
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "bash -s" > "$AP/receipts/install-$h.txt" 2>&1 <<HOST
set -e; d=\$(mktemp -d); export INSTALL_DIR="\$d/bin"
curl -LsSf https://raw.githubusercontent.com/$REPO/$T/scripts/install.sh | sh -s -- --version $T $variant  # bashrs disable-line=SEC002,SEC008,SEC015
v=\$("\$INSTALL_DIR/apr" --version | head -1); echo "installer: \$v"
case "\$v" in *"$V"*) ;; *) echo "VERSION MISMATCH: \$v"; exit 1;; esac
cd /; rm -rf -- "\$d"
HOST
    rc=$?; say "INSTALLER $h rc=$rc: $(grep -m1 '^installer:' "$AP/receipts/install-$h.txt")"
    [ $rc -eq 0 ] || fails=$((fails + 1))
  done
  [ $fails -eq 0 ] || die "$fails host/installer receipt(s) failed ($AP/receipts/)"
fi

# 9. close: the epic hears the receipts; the milestone closes only when nothing is left on it
if run_step close; then
  # SEC010 (bashrs-gate): read the run id with the read builtin, not $(cat "$AP/…") -- AP is derived
  # now (#3655), so a cat over it inside the message is flagged as a path-traversal risk.
  cleanroom_run_id=unknown; IFS= read -r cleanroom_run_id < "$AP/cleanroom-run-id" 2>/dev/null || cleanroom_run_id=unknown
  gh issue comment "$EPIC" --repo $REPO --body "$T released: $(gh release view "$T" --repo $REPO --json url -q .url). Receipts: pre-publish dogfood GO at \`$MC\`, all release assets verified by \`scripts/check_release_assets.sh $T\`, crates.io cascade complete, the CUDA asset downloaded, verified and run on gx10 and yoga. clean-room (aprender) green on the tag (run ${cleanroom_run_id}), install.sh receipts on intel and gx10. Logs: $AP on $(hostname)." >> "$LOG" 2>&1 || say "WARN epic comment failed"
  open=$(gh api "repos/$REPO/milestones/$MS" --jq .open_issues); [ "$open" = 0 ] || die "milestone $V still has $open open item(s); not closing"
  gh api -X PATCH "repos/$REPO/milestones/$MS" -f state=closed >> "$LOG" 2>&1 && say "MILESTONE $V closed"
  python3 "$REPO_ROOT/scripts/release/ledger.py" "$AP" "$MC" "$T" "$V" "$STATUS" >> "$LOG" 2>&1 && say "LEDGER record written in $AP (commit under docs/build-ledger/$(date -u +%F)/ via a docs PR)" || say "WARN ledger record not written"  # bashrs disable-line=DET002
  say "DONE $V released, published, installed, receipts taken"
fi
