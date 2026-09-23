#!/usr/bin/env bash
# check_no_shadowed_repo_skill.sh -- release tooling refuses to run while a USER-scope skill shadows a REPO skill
# (#4045 M8; the #2361 class: ~/.claude/skills/dogfood shadowed the repo's release-certifying skill, so hardening the
# repo copy edited a file that never ran).
#
# A skill is a directory holding SKILL.md. A user-scope skill (<home>/.claude/skills/*/SKILL.md) SHADOWS a repo skill
# (<repo>/.claude/skills/*/SKILL.md) when they share a directory name OR a frontmatter `name:`. Each collision is
# named with both paths. A user-scope directory WITHOUT SKILL.md is not a skill and shadows nothing.
#
#   bash scripts/check_no_shadowed_repo_skill.sh [--repo <dir>] [--home <dir>]    # exit 0 none, 1 shadowed, 2 usage
#   bash scripts/check_no_shadowed_repo_skill.sh --self-test
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; HOMEDIR="$HOME"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --home) HOMEDIR="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    *) echo "usage: $0 [--repo <dir>] [--home <dir>] | --self-test" >&2; exit 2 ;;
  esac
done

shadows() { # shadows <repo> <home> -> one "SHADOWED ..." line per collision; rc 1 if any
  python3 - "$1/.claude/skills" "$2/.claude/skills" <<'PY'
import glob, os, re, sys
def skills(root):
    out = {}
    for f in glob.glob(os.path.join(root, "*", "SKILL.md")):
        d = os.path.basename(os.path.dirname(f))
        m = re.search(r"^name:\s*(\S+)", open(f, encoding="utf-8", errors="replace").read(), re.M)
        out[f] = {d, m.group(1)} if m else {d}
    return out
repo, user = skills(sys.argv[1]), skills(sys.argv[2])
bad = 0
for uf, un in sorted(user.items()):
    for rf, rn in sorted(repo.items()):
        both = sorted(un & rn)
        if both:
            print("SHADOWED %s: the user-scope skill %s claims the repo skill %s -- the release must run the repo copy"
                  % ("/".join(both), os.path.dirname(uf), os.path.dirname(rf)))
            bad = 1
sys.exit(bad)
PY
}

if [ "$SELF_TEST" = 1 ]; then
  T=$(mktemp -d); bad=0
  mk() { mkdir -p "$1"; printf -- '---\nname: %s\n---\n' "$2" > "$1/SKILL.md"; }
  row() { # row <name> <want rc> <needle|-> ; uses $T/r and $T/h
    local out rc; out=$(shadows "$T/r" "$T/h" 2>&1); rc=$?
    if [ "$rc" = "$2" ] && { [ "$3" = - ] || grep -qF "$3" <<< "$out"; }; then echo "ok    $1"
    else echo "FAIL  $1 -- rc $rc: $out"; bad=1; fi
  }
  mk "$T/r/.claude/skills/dogfood" dogfood; mk "$T/r/.claude/skills/pre-release" pre-release
  mk "$T/h/.claude/skills/quorum-review" quorum-review
  row "no-collision" 0 -
  mkdir -p "$T/h/.claude/skills/dogfood"; echo x > "$T/h/.claude/skills/dogfood/README.md"
  row "a-dir-without-SKILL.md-is-not-a-skill" 0 -
  mk "$T/h/.claude/skills/dogfood" dogfood-user
  row "same-directory-shadows" 1 "SHADOWED dogfood"
  [ -n "$T" ] && rm -rf -- "${T:?}/h/.claude/skills/dogfood"; mk "$T/h/.claude/skills/release-x" pre-release
  row "same-name-shadows" 1 "SHADOWED pre-release"
  [ -n "$T" ] && rm -rf -- "${T:?}/h/.claude/skills/release-x"
  row "cleared-again" 0 -
  if [ -n "$T" ] && [ "$T" != "/" ] && [ -d "$T" ]; then rm -rf -- "$T"; fi
  echo "check_no_shadowed_repo_skill self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
  exit "$bad"
fi

out=$(shadows "$REPO" "$HOMEDIR"); rc=$?
[ -n "$out" ] && printf '%s\n' "$out"
[ "$rc" = 0 ] && echo "ok    no user-scope skill shadows a repo skill ($REPO/.claude/skills vs $HOMEDIR/.claude/skills)"
exit "$rc"
