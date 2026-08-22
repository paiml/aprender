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
# THE RULE IS AN ALLOWLIST. IT USED TO BE A BLACKLIST, AND THAT WAS WRONG.
# -----------------------------------------------------------------------
# The first version enumerated four keying SYNTAXES — a YAML mapping key, an
# identity field's value, a dict subscript, a CLI/query key — and passed
# everything else. "Nothing may key on this" is a universal claim, and a list of
# four syntaxes cannot carry one: `df.groupby("cluster_id")`, `r.cluster_id`,
# `row.get("cluster_id")`, `sort_values(by="cluster_id")`, `WHERE cluster_id = 3`
# and `cluster_id == 7` all key on the column and all walked past it. A reviewer
# extracted the function and demonstrated exactly that. The blacklist is
# unbounded; the allowlist is five lines long.
#
# So the rule is inverted. **A standalone `cluster_id` token is a KEY unless the
# line says otherwise**, and there are only three ways to say otherwise:
#
#   A1  the token is not standalone — `check_no_cluster_id_keys.sh`,
#       `CLUSTER_ID_GUARD_ROOT`, `cluster_idea` are different identifiers
#
#   A2  the token sits in a position NO PARSER IN SCOPE READS AS AN IDENTITY:
#         * after a comment marker (`#`, `//`, `<!--`) on that line, or
#         * inside backticks
#       This is not "we trust prose". It is mechanical: in YAML a backtick is a
#       literal character, in Python 3 backticks are a syntax error, and in bash
#       backticks are command substitution — so in every language this guard
#       scans, a backticked token is not an identifier and cannot function as a
#       key. Text after a comment marker is read by nothing at all.
#
#   A3  the line declares the SCHEMA — it names three or more of the ledger's
#       column headers together. Declaring that the column exists is the one
#       parsed position that must write the token down. A key names ONE column;
#       it never names the whole header row.
#
# and the ONE escape for a genuine read:
#
#       cluster-id-guard allow (<reason>)
#
# There are five of those in the tree today: the coverage gate proving the
# id->label map is a bijection, and the contract's falsification test doing the
# same. That is the whole allowlist. Every pragma must state a reason.
#
# HONEST LIMIT
# ------------
# Prose can still DESCRIBE a key — a Markdown table cell reading "floor key:
# `cluster_id`" is exempt under A2. It cannot BE one: the declaration that a
# machine reads lives in a parsed position, where the default is deny. The guard
# is about what runs, not about what is written about what runs.
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
BT='`'

# A1 — the token, standalone. `cluster_idea` and `foo_cluster_id` are different
# identifiers; `CLUSTER_ID_GUARD_ROOT` is a different case.
STANDALONE='\<cluster_id\>'

# A2b — backticked. Checked on the token itself, so `key: `cluster_id`` is exempt
# for the reason stated above: no parser in scope would read that as the column.
BACKTICKED="${BT}[[:space:]]*cluster_id[[:space:]]*${BT}"

# A3 — the schema declaration. Three or more ledger column headers on one line.
# `cluster_id` and `cluster_label` alone do NOT qualify: `key: cluster_id,
# cluster_label` is a composite key, not a header row.
LEDGER_COLS='binary|feature|quality_1_10|verified_hardware|top_competitor|in_dogfood_skill|cluster_id|cluster_label|evidence_path|confidence'

# --------------------------------------------------------------------------
# classify_line — the single decision function. Both the scan and the case
# table call THIS, so the table cannot drift away from what CI enforces.
# echoes KEY | PRAGMA | OK
# --------------------------------------------------------------------------
classify_line() {
  local line="$1" bare ncols
  # A1: not this token at all.
  grep -Eq "$STANDALONE" <<<"$line" || { printf 'OK\n'; return 0; }
  # The pragma is checked BEFORE comments are stripped, because the pragma lives
  # in a comment. Order matters here and nowhere else.
  if grep -Eq "$PRAGMA" <<<"$line"; then printf 'PRAGMA\n'; return 0; fi
  # A3: a schema declaration names the header row, not one column.
  ncols="$(grep -oE "\\<(${LEDGER_COLS})\\>" <<<"$line" | sort -u | wc -l)"
  if [ "$ncols" -ge 3 ]; then printf 'OK\n'; return 0; fi
  # A2a: drop everything from the first comment marker. What remains is what a
  # parser sees.
  bare="$(sed -E 's@(#|//|<!--).*$@@' <<<"$line")"
  grep -Eq "$STANDALONE" <<<"$bare" || { printf 'OK\n'; return 0; }
  # A2b: every surviving occurrence is backticked. Compare occurrence counts so
  # one backticked mention cannot launder a bare one on the same line.
  local n_tok n_bt
  n_tok="$(grep -oE "$STANDALONE" <<<"$bare" | wc -l)"
  n_bt="$(grep -oE "$BACKTICKED" <<<"$bare" | wc -l)"
  if [ "$n_bt" -ge "$n_tok" ]; then printf 'OK\n'; return 0; fi
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
    printf '  The rule is an ALLOWLIST: a standalone token is a key unless it is\n'
    printf '  backticked, after a comment marker, or in a schema header line.\n'
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
#
# The must-match half is now the interesting one: every row below the "forms the
# old four-syntax blacklist walked past" divider was GREEN before the inversion.
# --------------------------------------------------------------------------
self_test() {
  local failed=0 want got line
  printf 'check_no_cluster_id_keys.sh --self-test\n\n'
  printf '  %-6s %-8s %s\n' "want" "got" "line"

  run_case() {
    want="$1"; line="$2"
    got="$(classify_line "$line")"
    if [ "$got" = "$want" ]; then
      printf '  %-6s %-8s OK    %s\n' "$want" "$got" "$line"
    else
      printf '  %-6s %-8s WRONG %s\n' "$want" "$got" "$line"
      failed=1
    fi
  }

  # --- forms the OLD blacklist already caught, which must stay caught
  run_case KEY '  cluster_id: 7'
  run_case KEY '- cluster_id: 3'
  run_case KEY 'cluster_id: 11'
  run_case KEY '    key: cluster_id'
  run_case KEY '    gate_id: "cluster_id"'
  run_case KEY '  group_by: cluster_id'
  run_case KEY "  keyed_by: ${SQ}cluster_id${SQ}"
  run_case KEY '  identifier = cluster_id'
  run_case KEY '  floor: {key: cluster_id}'
  run_case KEY 'floors[row["cluster_id"]] += 1'
  run_case KEY "counts[r[${SQ}cluster_id${SQ}]] = n"
  run_case KEY 'x = by[cluster_id]'
  run_case KEY 'pv coverage --by cluster_id'
  run_case KEY 'sort by cluster_id'

  # --- forms the old four-syntax blacklist WALKED PAST. Each of these was GREEN
  #     before the inversion; each keys an obligation on the permuting id.
  run_case KEY 'for cid, grp in df.groupby("cluster_id"):'
  run_case KEY '    floors = ledger.sort_values(by="cluster_id")'
  run_case KEY '    if row.get("cluster_id") == 3:'
  run_case KEY '    return r.cluster_id'
  run_case KEY '    obligations = {r.cluster_id: r for r in rows}'
  run_case KEY '    SELECT gates FROM ledger WHERE cluster_id = 3'
  run_case KEY '    if cluster_id == 7: fail("cluster 7 is ungated")'
  run_case KEY '    setattr(floor, "cluster_id", 3)'
  run_case KEY '    cluster_id, label = row'
  run_case KEY '    waiver_for(cluster_id=9)'
  run_case KEY '    yaml.safe_load(f)["floors"]["by"] = "cluster_id"'
  run_case KEY '      - name: cluster_id key guard case table'

  # --- must NOT be caught. A2a comment position: nothing parses it.
  run_case OK  '# never key a contract on cluster_id -- the labels permute'
  run_case OK  '  // the cluster_id column is provenance only'
  run_case OK  '<!-- cluster_id permutes; use the label -->'
  run_case OK  '  cluster_label: apr-lint-diag   # not cluster_id, deliberately'
  # --- A2b backtick position: not an identifier in yaml, python or bash.
  run_case OK  'A `cluster_id` is provenance, never an identity'
  run_case OK  '| **T1** | `cluster_id` is a k-means label and permutes |'
  # --- A3 schema declaration: three or more ledger columns on one line.
  run_case OK  '   "top_competitor", "in_dogfood_skill", "cluster_id", "cluster_label",'
  run_case OK  'binary,feature,quality_1_10,cluster_id,cluster_label,confidence'
  # --- A1 a different identifier entirely.
  run_case OK  '  cluster_label: apr-lint-diag'
  run_case OK  '    key: cluster_label'
  run_case OK  'floors[row["cluster_label"]] += 1'
  run_case OK  'A cluster_idea is not this token'
  run_case OK  'IDGUARD="scripts/check_no_cluster_id_keys.sh"'
  run_case OK  'CLUSTER_ID_GUARD_ROOT="$td" bash "$td/scripts/x.sh"'
  # --- a backticked mention may not launder a bare one on the same line.
  run_case KEY '  the durable key is `cluster_label`; key: cluster_id'
  # --- a token BEFORE the comment marker is still parsed.
  run_case KEY '  key: cluster_id   # with a trailing comment'

  # --- pragma: a keying form is admissible only with a stated reason
  run_case PRAGMA 'ids[lab].add(r["cluster_id"])  # cluster-id-guard allow (validation, not a key)'
  run_case PRAGMA '  cluster_id: 7   # cluster-id-guard allow (provenance column declaration)'

  printf '\n'
  if [ "$failed" -eq 0 ]; then
    printf 'CASE TABLE PASS: every keying form caught -- including the twelve the\n'
    printf 'four-syntax blacklist walked past -- every non-key position left alone,\n'
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
