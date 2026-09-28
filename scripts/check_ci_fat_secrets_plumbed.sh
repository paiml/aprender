#!/usr/bin/env bash
# check_ci_fat_secrets_plumbed.sh — every `secrets.<NAME>` a section in
# ci/sections.yml reads is plumbed into the fat jobs of .github/workflows/ci.yml
# as `FAT_SECRET_<NAME>: ${{ secrets.<NAME> }}` (#4510).
#
# WHY THIS EXISTS
# ---------------
# fat_driver.py builds a section's `secrets` context from the FAT_SECRET_*
# environment of the fat job, so the two files are two hand-written lists of
# the same names with nothing tying them together. #4441 plumbed
# `FAT_SECRET_PR_REVIEW_SIGNING_KEY_B: ${{ secrets.PR_REVIEW_SIGNING_KEY_B }}`
# (the `64` cut off) while the section reads `secrets.PR_REVIEW_SIGNING_KEY_B64`.
# GitHub expands an unknown secret to "", and fat_driver expands an unplumbed
# one to "", so neither complained: the CI receipt signer simply had an empty
# key on every PR, and arm 4 of `present` went RED fleet-wide with nothing
# pointing at the cause.
#
# GITHUB_TOKEN is exempt: fat_driver supplies it itself.
#
# Text-only: reads the two files, builds nothing.
#
#   bash scripts/check_ci_fat_secrets_plumbed.sh
#   bash scripts/check_ci_fat_secrets_plumbed.sh --self-test
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECTIONS_REL="ci/sections.yml"
WORKFLOW_REL=".github/workflows/ci.yml"

# section_secrets <sections.yml> -> the distinct secret names the sections
# read, one per line, GITHUB_TOKEN excluded. rc 2 when the file is unreadable.
section_secrets() {
  [ -r "$1" ] || return 2
  grep -oE 'secrets\.[A-Za-z0-9_]+' "$1" | sed 's/^secrets\.//' | grep -vx 'GITHUB_TOKEN' | sort -u
  return 0
}

# check_pair <sections.yml> <ci.yml> -> prints one FAIL line per defect.
# rc 0 = every read secret is plumbed under its own name and every FAT_SECRET_
# line names the secret it claims; 1 = a defect; 2 = a file is unreadable.
check_pair() {
  [ -r "$1" ] && [ -r "$2" ] || return 2
  bad=0
  names="$(section_secrets "$1")"
  for n in $names; do
    if ! grep -qE "^[[:space:]]*FAT_SECRET_${n}:[[:space:]]*\\\$\{\{[[:space:]]*secrets\.${n}[[:space:]]*\}\}[[:space:]]*$" "$2"; then
      printf 'FAIL: %s reads secrets.%s but no `FAT_SECRET_%s: ${{ secrets.%s }}` line plumbs it -- the section sees "".\n' \
        "$SECTIONS_REL" "$n" "$n" "$n"
      bad=1
    fi
  done
  # The reverse: a FAT_SECRET_ key that names one secret and reads another.
  while IFS= read -r line; do
    key="$(printf '%s\n' "$line" | sed -E 's/^[[:space:]]*FAT_SECRET_([A-Za-z0-9_]+):.*/\1/')"
    val="$(printf '%s\n' "$line" | grep -oE 'secrets\.[A-Za-z0-9_]+' | sed 's/^secrets\.//')"
    if [ "$key" != "$val" ]; then
      printf 'FAIL: %s plumbs FAT_SECRET_%s from secrets.%s -- the key and the secret must be the same name.\n' \
        "$WORKFLOW_REL" "$key" "${val:-<none>}"
      bad=1
    fi
  done < <(grep -E '^[[:space:]]*FAT_SECRET_[A-Za-z0-9_]+:' "$2")
  return "$bad"
}

self_test() {
  printf '=== case table: check_ci_fat_secrets_plumbed.sh ===\n'
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:?}"' RETURN
  fails=0
  assert() {  # assert <label> <expected_rc> <actual_rc>
    if [ "$2" -eq "$3" ]; then
      printf '  ok   %-58s rc=%s\n' "$1" "$3"
    else
      printf '  FAIL %-58s expected rc=%s got rc=%s\n' "$1" "$2" "$3"
      fails=$((fails + 1))
    fi
  }
  printf '        env:\n          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n          K: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n' > "$tmp/sections.yml"
  printf '    env:\n      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n' > "$tmp/head.yml"

  cp "$tmp/head.yml" "$tmp/ok.yml"
  printf '      FAT_SECRET_PR_REVIEW_SIGNING_KEY_B64: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n' >> "$tmp/ok.yml"
  check_pair "$tmp/sections.yml" "$tmp/ok.yml" > /dev/null; assert 'plumbed under its own name' 0 $?

  # The #4441 defect, byte for byte.
  cp "$tmp/head.yml" "$tmp/trunc.yml"
  printf '      FAT_SECRET_PR_REVIEW_SIGNING_KEY_B: ${{ secrets.PR_REVIEW_SIGNING_KEY_B }}\n' >> "$tmp/trunc.yml"
  check_pair "$tmp/sections.yml" "$tmp/trunc.yml" > /dev/null; assert 'the #4441 truncation (_B for _B64) is RED' 1 $?

  cp "$tmp/head.yml" "$tmp/none.yml"
  check_pair "$tmp/sections.yml" "$tmp/none.yml" > /dev/null; assert 'a read secret with no plumbing is RED' 1 $?

  cp "$tmp/ok.yml" "$tmp/mismatch.yml"
  printf '      FAT_SECRET_OTHER: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n' >> "$tmp/mismatch.yml"
  check_pair "$tmp/sections.yml" "$tmp/mismatch.yml" > /dev/null; assert 'a key naming a different secret is RED' 1 $?

  printf '      K: ${{ secrets.GITHUB_TOKEN }}\n' > "$tmp/token-only.yml"
  check_pair "$tmp/token-only.yml" "$tmp/head.yml" > /dev/null; assert 'GITHUB_TOKEN needs no FAT_SECRET_ line' 0 $?

  check_pair "$tmp/absent.yml" "$tmp/ok.yml" > /dev/null 2>&1; assert 'an unreadable sections file is not clean' 2 $?

  if [ "$fails" -gt 0 ]; then
    printf '\nFAIL: %s case(s) failed. The guard does not do what it claims.\n' "$fails"
    return 1
  fi
  printf 'PASS: all cases behave as declared.\n'
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

printf '=== every secret ci/sections.yml reads is plumbed into ci.yml (check_ci_fat_secrets_plumbed.sh) ===\n'
names="$(section_secrets "$REPO_ROOT/$SECTIONS_REL")"
check_pair "$REPO_ROOT/$SECTIONS_REL" "$REPO_ROOT/$WORKFLOW_REL"
rc=$?
if [ "$rc" -eq 2 ]; then
  printf 'FAIL: %s or %s is unreadable -- a check that read nothing certifies nothing.\n' "$SECTIONS_REL" "$WORKFLOW_REL"
  exit 1
fi
[ "$rc" -eq 0 ] || exit 1
printf 'PASS: %s secret(s) read by sections, each plumbed under its own name: %s\n' \
  "$(printf '%s\n' "$names" | grep -c .)" "$(printf '%s' "$names" | tr '\n' ' ')"
exit 0
