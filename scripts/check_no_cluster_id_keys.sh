#!/usr/bin/env bash
#
# check_no_cluster_id_keys.sh — T1: nothing may KEY on `cluster_id`.
#
# WHY
# ---
# docs/audits/surface_audit.csv carries two cluster columns. `cluster_label` is a
# name a human gave a group of features and is stable across re-runs.
# `cluster_id` is a k-means label, and k-means labels PERMUTE: change the input
# and cluster 3 becomes cluster 9. An id baked into a contract, a gate, a waiver
# or a ticket therefore silently RE-POINTS at a different set of features the next
# time the surface moves — the stale-hardcoded-list failure class this repo keeps
# re-finding, wearing a new hat. Nothing announces the re-point; the obligation
# just quietly starts describing something else.
#
# So `cluster_id` is PROVENANCE, never an identity. This guard refuses any use of
# it as a key.
#
# THE RULE, EXACTLY
# -----------------
# Over the tracked surfaces where THIS ledger's cluster columns can be declared or
# consumed -- contracts/, scripts/, .claude/, docs/audits/, .github/ --
# a line mentioning `cluster_id` FAILS when it matches any keying form:
#
#   K1  a YAML/mapping key:            `cluster_id:` at the head of a line
#   K2  an identity field's value:     `key:`/`id:`/`gate:`/`keyed_by:`/`group_by:`/
#                                      `identifier:`/`name:`/`cluster_key:` = cluster_id
#   K3  a dict/index subscript:        `["cluster_id"]`, `['cluster_id']`, `[cluster_id]`
#   K4  a CLI/query key:               `--by cluster_id`, `--key cluster_id`,
#                                      `group by cluster_id`, `keyed on cluster_id`,
#                                      `sort by cluster_id`, `join on cluster_id`
#
# and does NOT carry the escape pragma
#
#       cluster-id-guard allow (<reason>)
#
# The pragma exists for the ONE legitimate read: the coverage gate proves the
# id->label map is a bijection, which requires reading the column. That is
# validation, not keying. Every pragma must state a reason on the same line.
#
# ALLOWED WITHOUT A PRAGMA, because none of them is a key:
#   * prose in a comment or in Markdown ("never key on cluster_id")
#   * the CSV header and the COLUMNS schema list that declares the column exists
#   * `cluster_label` — a different token entirely, and the one to use
#
# SCOPE, and why it is not "everything"
# -------------------------------------
# `cluster_id` is a common local variable name. docs/specifications/ carries an
# unrelated Python spec with `self.cluster_stats[cluster_id]`, which is not this
# column and not this repo's ledger. Widening the universe to all of docs/ makes
# the guard red for a reason that has nothing to do with the surface audit, and a
# guard that reds for the wrong reason gets switched off. So the universe is the
# surfaces where a floor, a gate or an obligation over THIS ledger is declared.
#
# SELF-TEST — the case table is the guard
# ---------------------------------------
#   bash scripts/check_no_cluster_id_keys.sh --self-test
# The apr-invocation guards in this repo had their pattern wrong five times; every
# one was caught by a must-match/must-not-match table and none by review. So this
# guard ships one, runs it in CI beside itself, and a pattern edit that stops
# catching a listed form fails there rather than in six months.

set -uo pipefail

REPO_ROOT="${CLUSTER_ID_GUARD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SELF_REL="scripts/check_no_cluster_id_keys.sh"

PRAGMA='cluster-id-guard[: ]*allow'

# An optional surrounding quote, either kind. Built with printf rather than the
# usual '"'"' dance: that construct is correct bash but bashrs's parser reports
# SC1078 (unclosed double quote) on every one of them, and five spurious errors
# in a guard is how a guard stops being read.
SQ="$(printf '\047')"
Q="[\"${SQ}]?"

# K1..K4. Deliberately anchored: `cluster_id` inside a word (`cluster_idea`) is
# not this token, and `cluster_label` never matches.
K1='^[[:space:]]*-?[[:space:]]*cluster_id[[:space:]]*:'
# K2 requires the token to be the WHOLE value, not merely to follow the field.
# Without that, a GitHub Actions step called `name: cluster_id key guard case
# table` reads as an identity keyed on the id -- a false positive of exactly
# the kind that gets a guard switched off. An identity field's value is
# `cluster_id` and nothing else, so the terminator is EOL, a comment, or one of
# , } ) for inline flow mappings.
K2="\\<(key|id|gate|gate_id|keyed_by|group_by|identifier|name|cluster_key|primary_key)\\>[[:space:]]*[:=][[:space:]]*${Q}cluster_id${Q}[[:space:]]*([,})]|#|$)"
K3="\\[[[:space:]]*${Q}cluster_id${Q}[[:space:]]*\\]"
K4="(--by|--key|--group-by|group by|keyed on|keyed by|sort by|join on|index on|indexed by)[[:space:]]+${Q}cluster_id\\>"

KEYING="(${K1})|(${K2})|(${K3})|(${K4})"

# --------------------------------------------------------------------------
# classify_line — the single decision function. Both the scan and the case
# table call THIS, so the table cannot drift away from what CI enforces.
# echoes KEY | PRAGMA | OK
# --------------------------------------------------------------------------
classify_line() {
  local line="$1"
  if ! grep -Eq 'cluster_id\>' <<<"$line"; then printf 'OK\n'; return 0; fi
  if ! grep -Eq "$KEYING" <<<"$line"; then printf 'OK\n'; return 0; fi
  if grep -Eq "$PRAGMA" <<<"$line"; then printf 'PRAGMA\n'; return 0; fi
  printf 'KEY\n'
}

scan() {
  local root="$1" rc=0 file line n
  local files
  # Tracked files only, and only the surfaces where a key could be DECLARED.
  # `git grep -n` so a path with a space cannot split a filename.
  files="$(git -C "$root" grep -In --no-color -E 'cluster_id' -- \
             'contracts/*' 'scripts/*' '.claude/*' 'docs/audits/*' '.github/*' \
             ':(exclude)docs/audits/surface_audit.csv' 2>/dev/null)"
  local total=0 pragmas=0 offenders=0
  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    file="${hit%%:*}"
    [ "$file" = "$SELF_REL" ] && continue        # this guard quotes every form
    line="${hit#*:}"; n="${line%%:*}"; line="${line#*:}"
    total=$((total + 1))
    case "$(classify_line "$line")" in
      KEY)
        offenders=$((offenders + 1)); rc=1
        printf '  FAIL %s:%s\n       %s\n' "$file" "$n" "$line" ;;
      PRAGMA) pragmas=$((pragmas + 1)) ;;
    esac
  done <<<"$files"

  if [ "$rc" -ne 0 ]; then
    printf '\n  %s line(s) KEY on cluster_id. k-means ids permute on re-run, so an\n' "$offenders"
    printf '  obligation keyed on one silently re-points at a different cluster the\n'
    printf '  next time the surface moves. Use cluster_label, which is human-owned.\n'
    printf '  A genuine read (validation, provenance) may carry the pragma:\n'
    printf '      cluster-id-guard allow (<reason>)\n'
    return 1
  fi
  printf '  cluster_id key guard PASS  %s mention(s) scanned, %s pragma-exempt, 0 keying\n' \
    "$total" "$pragmas"
  # Vacuity guard: a scan that found no mentions at all proves nothing about the
  # pattern. The repo always has at least the schema declaration and this file's
  # own documentation, so zero means the file selector broke.
  if [ "$total" -lt 1 ]; then
    printf '  FAIL: scanned 0 lines mentioning cluster_id. The universe is empty, so\n'
    printf '        nothing was checked. Fix the path selector before trusting a PASS.\n'
    return 1
  fi
  return 0
}

# --------------------------------------------------------------------------
# Case table. must-match (KEY) / must-not-match (OK) / pragma.
# --------------------------------------------------------------------------
self_test() {
  local failed=0 want got line
  printf 'check_no_cluster_id_keys.sh --self-test\n\n'
  printf '  %-4s %-8s %s\n' "want" "got" "line"

  run_case() {
    want="$1"; line="$2"
    got="$(classify_line "$line")"
    if [ "$got" = "$want" ]; then
      printf '  %-4s %-8s OK    %s\n' "$want" "$got" "$line"
    else
      printf '  %-4s %-8s WRONG %s\n' "$want" "$got" "$line"
      failed=1
    fi
  }

  # --- must be caught (K1)
  run_case KEY '  cluster_id: 7'
  run_case KEY '- cluster_id: 3'
  run_case KEY 'cluster_id: 11'
  # --- must be caught (K2)
  run_case KEY '    key: cluster_id'
  run_case KEY '    gate_id: "cluster_id"'
  run_case KEY '  group_by: cluster_id'
  run_case KEY "  keyed_by: 'cluster_id'"
  run_case KEY '  identifier = cluster_id'
  run_case KEY '  floor: {key: cluster_id}'
  run_case KEY '    key: cluster_id   # with a trailing comment'
  # --- must be caught (K3)
  run_case KEY 'floors[row["cluster_id"]] += 1'
  run_case KEY "counts[r['cluster_id']] = n"
  run_case KEY 'x = by[cluster_id]'
  # --- must be caught (K4)
  run_case KEY 'pv coverage --by cluster_id'
  run_case KEY '# ...then group by cluster_id and sum'
  run_case KEY 'the floor is keyed on cluster_id today'
  run_case KEY 'sort by cluster_id'

  # --- must NOT be caught: prose, schema declaration, the sibling column
  run_case OK  '# never key a contract on cluster_id -- the labels permute'
  run_case OK  '   "top_competitor", "in_dogfood_skill", "cluster_id", "cluster_label",'
  run_case OK  'binary,feature,quality_1_10,cluster_id,cluster_label,confidence'
  run_case OK  '  cluster_label: apr-lint-diag'
  run_case OK  '    key: cluster_label'
  run_case OK  'floors[row["cluster_label"]] += 1'
  run_case OK  'A cluster_idea is not this token'
  run_case OK  'cluster_id is provenance, never an identity'
  # A field whose value only BEGINS with the token is prose, not a key. This is
  # the GitHub Actions step that runs this very guard, and the first draft of K2
  # went red on it.
  run_case OK  '      - name: cluster_id key guard case table'
  run_case OK  '  description: cluster_id may not be used as a key'

  # --- pragma: a keying form is admissible only with a stated reason
  run_case PRAGMA 'ids[lab].add(r["cluster_id"])  # noqa: cluster-id-guard allow (validation, not a key)'
  run_case PRAGMA '  cluster_id: 7   # cluster-id-guard allow (provenance column declaration)'

  printf '\n'
  if [ "$failed" -eq 0 ]; then
    printf 'CASE TABLE PASS: every keying form caught, every non-key form left alone,\n'
    printf 'and the pragma recognised. A guard that caught nothing would pass the\n'
    printf 'must-not-match half too, which is why both halves are asserted.\n'
    return 0
  fi
  printf 'CASE TABLE FAIL\n'
  return 1
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

printf 'T1 guard: nothing may key on cluster_id\n'
scan "$REPO_ROOT"
exit $?
