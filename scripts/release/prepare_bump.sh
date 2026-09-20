#!/usr/bin/env bash
# prepare_bump.sh — §4.2 of docs/specifications/06x-release-schedule.md for 0.68.2: the bump and the
# CHANGELOG section. Two invocations, because the section needs a human-quality summary:
#   prepare_bump.sh           worktree at origin/main, bump-version.sh 0.68.2 + --check, CHANGELOG [0.68.2]
#                             drafted from the milestone's merged PRs (with a placeholder --ship refuses)
#   prepare_bump.sh --ship    pre-push checks, commit, push, PR in milestone 0.68.1 with auto-merge;
#                             prints the autopilot launch line for that PR
# §4.1 (freeze: open milestone items move to 0.69.0 with slipped_from:) is done by hand before this.
set -uo pipefail
V=0.68.2; MS=0.68.2; REPO=paiml/aprender; AP=/mnt/nvme-raid0/agent-wt/rel-0682-autopilot; LAST_TAG=v0.68.1
B="$AP/bump"; BR="PMAT-3477-release-$V"; MARK='<!-- one-paragraph summary of the train: EDIT BEFORE --ship -->'
die() { printf 'STOP %s\n' "$*" >&2; exit 1; }
# D4/D5/D6/D7 (PMAT-3459): $0-derived root and CARGO_HOME-relative cargo. Root resolved
# BEFORE any cd. NOT `git rev-parse --show-toplevel` — refused on a bind-mounted tree (#3586).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || die "cannot resolve the repo root from $0"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

if [ "${1:-}" != "--ship" ]; then
  cd "$REPO_ROOT" || die "no repo"
  git fetch -q origin main || die "fetch failed"
  [ -e "$B" ] && die "$B exists: review it, or 'git worktree remove' it to start over"
  git worktree add -q -b "$BR" "$B" origin/main || die "worktree add failed"
  cd "$B" || die "cd $B"
  bash scripts/bump-version.sh "$V" > "$AP/bump.log" 2>&1 || die "bump-version.sh $V failed ($AP/bump.log)"
  bash scripts/bump-version.sh --check >> "$AP/bump.log" 2>&1 || die "bump-version.sh --check failed after the bump"
  since=$(git log -1 --format=%cI "$LAST_TAG") || die "no tag $LAST_TAG"
  gh pr list --repo $REPO --state merged --search "merged:>=$since" --limit 500 --json number,title > "$AP/merged.json" || die "gh pr list failed"
  python3 - "$AP/merged.json" "$V" "$MARK" "$AP/section.md" <<'PY'
import datetime, json, re, sys
prs, v, mark, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
groups = {"Added": [], "Fixed": [], "Changed": []}
for p in sorted(prs, key=lambda p: p["number"]):
    m = re.match(r"(\w+)(\([^)]*\))?!?:", p["title"])
    kind = m.group(1).lower() if m else ""
    g = "Fixed" if kind == "fix" else "Added" if kind == "feat" else "Changed"
    groups[g].append(f"- {p['title']} (#{p['number']})")
day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
lines = [f"## [{v}] - {day}", "", mark, ""]
for g in ("Added", "Fixed", "Changed"):
    if groups[g]:
        lines += [f"### {g}", ""] + groups[g] + [""]
open(out, "w").write("\n".join(lines))
print(f"{len(prs)} merged PRs since the last tag: " + ", ".join(f"{k} {len(x)}" for k, x in groups.items()))
PY
  python3 - CHANGELOG.md "$AP/section.md" <<'PY'
import sys
p, sec = sys.argv[1], open(sys.argv[2]).read()
s = open(p).read(); anchor = "## [Unreleased]\n"
assert s.count(anchor) == 1, "CHANGELOG has no single [Unreleased] anchor"
open(p, "w").write(s.replace(anchor, anchor + "\n" + sec.rstrip("\n") + "\n", 1))
PY
  printf 'REVIEW %s/CHANGELOG.md [%s]: replace the placeholder with the train summary, curate the bullets, then run: %s --ship\n' "$B" "$V" "$0"
  exit 0
fi

cd "$B" || die "no $B (run without --ship first)"
# T-2 PREFLIGHT BEFORE THE BUMP (operator 2026-09-17): the bump PR is refused without a GO receipt for its parent sha.
parent=$(git rev-parse origin/main); rcpt="$AP/preflight-$parent.verdict"
[ -f "$rcpt" ] && grep -q '^GO ' "$rcpt" || die "no T-2 GO receipt for origin/main $parent ($rcpt) — run: $AP/t2_preflight.sh"
grep -qF "$MARK" CHANGELOG.md && die "CHANGELOG [$V] still carries the EDIT placeholder"
bash scripts/bump-version.sh --check > /dev/null 2>&1 || die "bump-version.sh --check failed"
awk -v h="## [$V]" 'index($0, h) == 1 {f = 1; next} f && /^## \[/ {exit} f' CHANGELOG.md > "$AP/release_notes.md"
[ -s "$AP/release_notes.md" ] || die "the CHANGELOG [$V] section is empty"
cargo_bin() { "${CARGO_HOME:-$HOME/.cargo}"/bin/cargo "$@"; }
cargo_bin fmt --all -- --check > /dev/null 2>&1 || die "cargo fmt --check failed"
cargo_bin deny check advisories > "$AP/deny.log" 2>&1 || die "cargo deny check advisories failed ($AP/deny.log)"
cargo_bin nextest --version > /dev/null 2>&1; cargo_bin "test" --quiet --package aprender-contracts --lib > "$AP/contracts.log" 2>&1 || die "aprender-contracts lib tests failed ($AP/contracts.log)"
git add -A
git commit -q -F - <<MSG || die "commit refused"
release: $V

bump-version.sh $V (every workspace, facades included; --check green) and the CHANGELOG [$V] section,
per docs/specifications/06x-release-schedule.md §4.2. After this merges, rel-0682-autopilot runs APR-RELEASE-001 §4 T-1..T-4 + close:
deep (T-1, local), pre-publish dogfood, tag + release, clean-room.yml dispatched on the tag (T-3, run id recorded), assets by command, preflight, cascade (T-4, automated), install, host + installer receipts, close.

Pmat-Ticket: PMAT-3477
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
git push -q -u origin "$BR" || die "push failed"
{ printf 'Release bump for **%s** (06x release schedule §4.2): `bump-version.sh %s` across every workspace, and the CHANGELOG section below.\n\nWhen this merges, `rel-0682-autopilot/autopilot.sh` runs the rest of the train, fail-closed (APR-RELEASE-001 §4): T-1 deep run, pre-publish dogfood, tag + release, `clean-room.yml` dispatched on the tag with the run id recorded (T-3), all release assets checked with `scripts/check_release_assets.sh`, publish preflight, the crates.io cascade (T-4, automated per operator 2026-09-13), `install.sh --version` receipts on intel and gx10 plus the CUDA asset receipts on gx10 and yoga, and the epic and milestone close.\n\n' "$V" "$V"; cat "$AP/release_notes.md"; printf '\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n'; } > "$AP/pr_body.md"
url=$(gh pr create --repo $REPO --base main --head "$BR" --milestone "$MS" --title "release: $V" --body-file "$AP/pr_body.md") || die "gh pr create failed"
n=${url##*/}
ARM="$REPO_ROOT/scripts/arm_pr_automerge.sh"; [ -f "$ARM" ] || ARM=/mnt/nvme-raid0/agent-wt/wedge-3292/scripts/arm_pr_automerge.sh
armed=0; for _ in $(seq 1 30); do bash "$ARM" "$n" > "$AP/arm.log" 2>&1; rc=$?; [ $rc -eq 0 ] && { armed=1; break; }; [ $rc -eq 3 ] || break; sleep 120; done
[ $armed = 1 ] && printf 'ARMED #%s (guard-tree green on the head)\n' "$n" || printf 'WARN not armed: %s\n' "$(tail -1 "$AP/arm.log")"
printf 'BUMP PR #%s (%s)\nlaunch: setsid nohup %s/autopilot.sh %s > /dev/null 2>&1 < /dev/null & disown\n' "$n" "$url" "$AP" "$n"
