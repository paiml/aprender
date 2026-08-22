#!/usr/bin/env bash
#
# check_dogfood_coverage.sh — the blocking apr-dogfood coverage gate.
#
# WHY THIS EXISTS
# ---------------
# The 0.63.0 surface audit measured 142 of 830 features covered by a gate: 17.1%.
# 27 of 28 shipped binaries have no gate at all. Where gates DO look they find
# defects at 72.5%; where they do not, the surface scores 6 by default — and a 6
# does not mean "fine", it means UNLOOKED-AT. So coverage itself is the gate.
#
# This script fails the build when the audited surface gets WORSE:
#   G2.1 freshness       the ledger is behind the code it claims to describe
#   G2.2 reconciliation  a ledger row vanished, taking its defect with it
#   G2.3 floors          coverage fell, or a dark-row count rose
#   G2.4 waivers         a quality<=4 feature has neither a gate nor a waiver
#
# THE COMPARAND LIVES ON PROTECTED main — THIS IS THE WHOLE POINT
# ---------------------------------------------------------------
# The multi-platform dogfood gate had its FLOOR and its UNIVERSE as literals in
# one file, so a single commit editing both defeated it, and the gate reported
# green while measuring nothing. A floor a PR can rewrite in the same commit that
# breaks it is not a floor.
#
# So every floor here is DERIVED, at run time, from
#
#     git show "${BASE_REF}:docs/audits/surface_audit.csv"
#
# where BASE_REF is `origin/main` — a ref a pull request cannot rewrite, because
# the required checks have already run on everything that reached it. There is no
# baseline NUMBER in this repository for a PR to edit. To lower a floor you must
# land the change on main first.
#
# BOOTSTRAP, AND WHY IT CANNOT BE RE-ENTERED
# ------------------------------------------
# On the one PR that first ADDS the ledger, origin/main does not have it yet.
# That PR compares against its own HEAD COMMIT (never the working tree, so a
# mutation still turns the gate red) and prints a BOOTSTRAP banner. The branch is
# taken only when the ledger is genuinely absent from main AND this branch adds
# it. If the ledger is missing from main and this branch does not add it — the
# shape you would get by DELETING the ledger to escape the gate — the run is a
# hard failure, not a bootstrap.
#
# SELF-TEST
# ---------
#   bash scripts/check_dogfood_coverage.sh --self-test
# builds a scratch repository with a protected `main`, then runs THIS script
# against it under each of the three registered mutations and under a no-op.
# A gate that fires on the no-op as well as the mutation is not measuring the
# mutation, so both verdicts are asserted every time.

set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="${DOGFOOD_GATE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEDGER="docs/audits/surface_audit.csv"
CONTRACT="contracts/apr-dogfood-coverage-v1.yaml"
THE44="docs/audits/dogfood-the-44.yaml"
GATE_PY="scripts/lib/dogfood_coverage_gate.py"
BASE_REF="${DOGFOOD_BASE_REF:-origin/main}"

fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Resolve the comparand. Exhaustive over three cases so that "the ledger is
# missing" can never fall through to "no floor to check".
# ---------------------------------------------------------------------------
resolve_base_ref() {
  local root="$1" ref="$2"
  # An UNRESOLVABLE base ref is not a bootstrap. CI checks this repo out at
  # fetch-depth 1, so origin/main is absent until the step fetches it; if that
  # fetch ever breaks, falling through to BOOTSTRAP would silently downgrade the
  # gate to comparing the branch against itself, permanently and quietly. A
  # missing comparand must be loud.
  if ! git -C "$root" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null 2>&1; then
    printf 'UNRESOLVABLE\t%s\n' "$ref"; return 0
  fi
  if git -C "$root" cat-file -e "${ref}:${LEDGER}" 2>/dev/null; then
    printf 'ARMED\t%s\n' "$ref"; return 0
  fi
  if git -C "$root" cat-file -e "HEAD:${LEDGER}" 2>/dev/null; then
    printf 'BOOTSTRAP\tHEAD\n'; return 0
  fi
  printf 'ABSENT\t-\n'; return 0
}

# ---------------------------------------------------------------------------
# G2.1 — freshness. The ledger cites a file and a line for every one of its
# rows. If any cited file has changed since the commit the ledger says it was
# measured at, the ledger is describing code that no longer exists.
#
# Two ways to cheat, both closed:
#   backdating measured_commit  -> MORE files have changed since -> redder
#   forward-dating it to HEAD   -> caught by the earned-date check below: you
#                                  cannot claim a measurement at a commit that
#                                  is newer than the last commit which actually
#                                  touched the ledger.
# ---------------------------------------------------------------------------
evidence_files() {
  local root="$1"
  python3 - "$root/$LEDGER" <<'PY'
import csv, re, sys
pat = re.compile(r"^([^\s:]+\.(?:rs|toml|yml|yaml|md|sh|py))(?::[\d-]+)?")
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
files, unresolved = set(), 0
for r in rows:
    m = pat.match(r["evidence_path"].strip())
    if m and "/" in m.group(1):
        files.add(m.group(1))
    else:
        unresolved += 1
# A universe assembled by a regex must say how much it dropped. An extractor
# that silently resolves nothing would otherwise report a perfectly fresh ledger.
print(f"#UNRESOLVED\t{unresolved}\t{len(rows)}")
for f in sorted(files):
    print(f)
PY
}

#
# The rule is PR-SCOPED and fully DERIVED from git:
#
#   if this branch changes a file the ledger cites as evidence,
#   this branch must also change the ledger.
#
# Two properties this buys, both deliberate:
#
#   * Nothing self-declared is load-bearing. An earlier draft compared against
#     `measured_commit:` from the contract — a value the same pull request can
#     edit. Forward-dating it to HEAD made the check vacuous, and the obvious
#     patch for that (a "you may not claim a date newer than the ledger" rule)
#     turned out to reject a legitimate re-measure that produced an identical
#     ledger. A derived quantity has neither problem.
#
#   * It can never go red for someone else's commit. A whole-history rule
#     ("no evidence file has moved since the ledger was written") reds every
#     unrelated PR the moment a hot file like commands_enum.rs is touched by
#     anyone, which is how a required check gets switched off.
#
# HONEST LIMIT: this forces the ledger to be TOUCHED, not to be CORRECT. It
# cannot tell a real re-audit from a one-character edit. It is a forcing
# function against silent drift, not a proof of accuracy.
#
check_freshness() {
  local root="$1" base="$2" evfile changed n_ev n_changed unresolved total measured provenance
  evfile="$(mktemp)"; changed="$(mktemp)"

  evidence_files "$root" > "$evfile" 2>/dev/null
  unresolved="$(awk -F'\t' '/^#UNRESOLVED/{print $2}' "$evfile")"
  total="$(awk -F'\t' '/^#UNRESOLVED/{print $3}' "$evfile")"
  sed -i '/^#UNRESOLVED/d' "$evfile"
  n_ev="$(wc -l < "$evfile")"

  if [ "${unresolved:-1}" -ne 0 ]; then
    printf '  G2.1 freshness       FAIL  %s of %s rows cite an evidence path that resolves to no file\n' \
      "$unresolved" "$total"
    rm -f "$evfile" "$changed"; return 1
  fi
  # Vacuity guard: an empty universe would make every ledger eternally fresh.
  if [ "$n_ev" -lt 1 ]; then
    printf '  G2.1 freshness       FAIL  the evidence universe is empty; nothing was checked\n'
    rm -f "$evfile" "$changed"; return 1
  fi

  # -------------------------------------------------------------------------
  # measured_commit is documentation, not a comparand. Two properties are worth
  # asserting about it, and exactly ONE of them is decidable everywhere:
  #
  #   SHAPE      it must be present, and it must BE an object name. A deleted
  #              line, an empty value, `PLACEHOLDER`, a branch name or a typo
  #              are all provenance defects, and a repository of any depth can
  #              say so.
  #
  #   EXISTENCE  whether that object is IN this repository. A shallow clone
  #              cannot answer it: `git cat-file -e` returns the same 128 for
  #              "you invented this SHA" and for "you were cloned with
  #              --depth=1 and this commit is two commits back". This job is
  #              checked out at fetch-depth 1 -- resolve_base_ref() above says
  #              so in as many words, and the earlier version of THIS check
  #              did not carry the reasoning across. Asking there produced a
  #              verdict about the CLONE, not about the ledger, which is how
  #              this gate went red on the very pull request that introduced
  #              it: `measured_commit cbb1ccd78 is not a commit in this repo`
  #              with all three coverage floors green, and every later step in
  #              the job skipped.
  #
  # So shape is enforced always; existence is enforced only where it is
  # decidable, and the PASS line names which of the two it managed to apply. A
  # check that silently narrows its own scope is theater. One that prints the
  # scope it actually had is a measurement.
  #
  # HONEST LIMIT: in a shallow clone a well-formed SHA that names nothing gets
  # through. Closing that would need history CI deliberately does not fetch.
  # The shape half still rejects the placeholder, the typo and the deleted
  # line, which is every provenance defect seen so far.
  # -------------------------------------------------------------------------
  measured="$(grep -m1 -E '^[[:space:]]*measured_commit:' "$root/$CONTRACT" 2>/dev/null \
              | sed -E "s/.*measured_commit:[[:space:]]*//; s/[[:space:]]*(#.*)?$//; s/^['\"]//; s/['\"]$//")"
  if [ -z "$measured" ]; then
    printf '  G2.1 freshness       FAIL  %s carries no measured_commit: value. The\n' "$CONTRACT"
    printf '                             ledger must say which commit it was measured at; deleting\n'
    printf '                             the line is not a way to stop being asked.\n'
    rm -f "$evfile" "$changed"; return 1
  fi
  if ! [[ "$measured" =~ ^[0-9a-f]{7,40}$ ]]; then
    printf '  G2.1 freshness       FAIL  measured_commit %s is not an object name\n' "$measured"
    printf '                             (expected 7-40 lowercase hex digits)\n'
    rm -f "$evfile" "$changed"; return 1
  fi
  # PRESENCE is conclusive anywhere; only ABSENCE needs the depth caveat, so ask
  # in that order. Asking `--is-shallow-repository` FIRST throws away a verdict
  # the repository was able to give: this dev box carries a graft ~740 commits
  # back, deep enough to resolve everything anyone cites, and a shallow-first
  # test downgraded every local run to UNCHECKED for no reason.
  if git -C "$root" cat-file -e "${measured}^{commit}" 2>/dev/null; then
    provenance="shape ok, object present"
  elif [ "$(git -C "$root" rev-parse --is-shallow-repository 2>/dev/null)" != "true" ]; then
    printf '  G2.1 freshness       FAIL  measured_commit %s is not a commit in this repo\n' "$measured"
    printf '                             (complete clone, so this is the ledger and not the depth)\n'
    rm -f "$evfile" "$changed"; return 1
  else
    provenance="shape ok; absence NOT conclusive (shallow clone, object may be beyond it)"
  fi

  # Everything this branch changed, committed or not, against its merge base.
  { git -C "$root" diff --name-only "$base" HEAD 2>/dev/null
    git -C "$root" diff --name-only HEAD 2>/dev/null; } | sort -u > "$changed"

  n_changed="$(grep -Fxc -f "$evfile" "$changed" 2>/dev/null || true)"
  n_changed="${n_changed:-0}"
  if [ "$n_changed" -gt 0 ] && ! grep -Fxq "$LEDGER" "$changed"; then
    printf '  G2.1 freshness       FAIL  this branch changes %s cited evidence file(s) and does\n' "$n_changed"
    printf '                             NOT touch %s:\n' "$LEDGER"
    grep -Fx -f "$evfile" "$changed" | head -10 | sed 's/^/        - /'
    printf '                             Audited code moved; re-run the audit for those rows. A ledger\n'
    printf '                             left behind describes a surface that is no longer shipping.\n'
    rm -f "$evfile" "$changed"; return 1
  fi
  printf '  G2.1 freshness       PASS  %s cited evidence files, %s changed by this branch\n' \
    "$n_ev" "$n_changed"
  printf '                             measured_commit %s: %s\n' "$measured" "$provenance"
  rm -f "$evfile" "$changed"; return 0
}

# ---------------------------------------------------------------------------
# The gate proper.
# ---------------------------------------------------------------------------
run_gate() {
  local root="$1" resolution mode ref basecsv rc frc
  [ -f "$root/$LEDGER" ]   || fail "no ledger at $root/$LEDGER"
  [ -f "$root/$CONTRACT" ] || fail "no contract at $root/$CONTRACT"
  [ -f "$root/$THE44" ]    || fail "no triage file at $root/$THE44"

  resolution="$(resolve_base_ref "$root" "$BASE_REF")"
  mode="${resolution%%$'\t'*}"; ref="${resolution##*$'\t'}"

  case "$mode" in
    UNRESOLVABLE)
      fail "cannot resolve the comparand ref <${ref}>.
      Every floor this gate applies is derived from that ref, so without it there
      is nothing to compare against and a PASS would be meaningless. In CI, fetch
      it first:  git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main" ;;
    ABSENT)
      fail "the ledger is absent from ${BASE_REF} AND this branch does not add it.
      That is the shape of a ledger DELETED to escape its own gate. If the audit
      is genuinely being retired, retire this check in the same commit." ;;
    BOOTSTRAP)
      printf '  BOOTSTRAP: %s has no ledger yet, so the comparand is the HEAD\n' "$BASE_REF"
      printf '             COMMIT of this branch, never the working tree. Once this\n'
      printf '             lands, every later run compares against protected %s.\n\n' "$BASE_REF" ;;
    ARMED)
      # Only origin/main earns the word "protected". Saying it of whatever
      # DOGFOOD_BASE_REF happens to hold would be the gate lying about its own
      # guarantee, which is worse than not printing a banner at all.
      local note="protected; a pull request cannot rewrite it"
      [ "$ref" = "origin/main" ] || note="OVERRIDDEN via DOGFOOD_BASE_REF -- NOT a protected ref"
      printf '  comparand: %s @ %s (%s)\n\n' \
        "$ref" "$(git -C "$root" rev-parse --short "$ref" 2>/dev/null)" "$note" ;;
  esac

  basecsv="$(mktemp)"
  git -C "$root" show "${ref}:${LEDGER}" > "$basecsv" 2>/dev/null \
    || { rm -f "$basecsv"; fail "could not read ${ref}:${LEDGER}"; }

  # Freshness diffs against the branch's merge base, so it sees this branch's
  # changes and nobody else's -- in BOOTSTRAP too, where `ref` is HEAD itself
  # and a HEAD..HEAD diff would be empty and therefore vacuous.
  local fbase
  fbase="$(git -C "$root" merge-base "$BASE_REF" HEAD 2>/dev/null)" || fbase=""
  [ -n "$fbase" ] || fbase="$ref"
  check_freshness "$root" "$fbase"; frc=$?

  python3 "$root/$GATE_PY" --base "$basecsv" --head "$root/$LEDGER" --the44 "$root/$THE44"
  rc=$?
  rm -f "$basecsv"

  if [ "$frc" -ne 0 ] || [ "$rc" -ne 0 ]; then
    printf '\nDOGFOOD COVERAGE GATE: FAIL\n'
    return 1
  fi
  printf '\nDOGFOOD COVERAGE GATE: PASS\n'
  return 0
}

# ---------------------------------------------------------------------------
# --self-test: three registered mutations plus a no-op, in a scratch repo whose
# `main` carries the ledger so the ARMED path is what gets exercised.
# ---------------------------------------------------------------------------
selftest_build_repo() {
  local td="$1" i
  mkdir -p "$td/docs/audits" "$td/contracts" "$td/scripts/lib" "$td/crates/demo/src"
  cp "$SELF" "$td/scripts/check_dogfood_coverage.sh"
  cp "$REPO_ROOT/$GATE_PY" "$td/$GATE_PY"

  for i in 1 2 3; do printf 'fn f%s() {}\n' "$i" > "$td/crates/demo/src/m$i.rs"; done

  {
    printf 'binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,evidence_path,confidence\n'
    printf 'demo,demo alpha,2,x86_64-linux,none,no,crates/demo/src/m1.rs:1,high\n'
    printf 'demo,demo beta,6,UNKNOWN,none,yes,crates/demo/src/m2.rs:1,high\n'
    printf 'demo,demo gamma,6,UNKNOWN,none,yes,crates/demo/src/m3.rs:1,high\n'
    for i in $(seq 1 200); do
      printf 'demo,demo pad%s,6,UNKNOWN,none,no,crates/demo/src/m1.rs:1,medium\n' "$i"
    done
  } > "$td/$LEDGER"

  printf 'metadata:\n  baselines:\n    measured_commit: PLACEHOLDER\n' > "$td/$CONTRACT"
  # `printf` and a leading `-`: "- feature:" is parsed as an OPTION, not a
  # format. Always route caller-shaped text through a %s.
  local q="'"
  { printf 'features:\n'
    printf '%s\n' "- feature: ${q}demo alpha${q}" '  issue: null' '  waiver: null'
  } > "$td/$THE44"

  # `git init` is NOT hermetic on a developer box: this machine sets
  # init.templatedir=~/.git-templates, so a scratch repo silently inherits the
  # pmat pre-commit hooks and every fixture commit fails. The first draft of
  # this self-test reported all three mutations RED -- for that reason, not for
  # the mutation. An empty template plus a dead hooksPath isolates the fixture.
  git -C "$td" init -q -b main --template= || return 1
  git -C "$td" config core.hooksPath /dev/null
  git -C "$td" config user.email t@t
  git -C "$td" config user.name t
  git -C "$td" config commit.gpgsign false
  selftest_commit "$td" base || return 1
  # A second commit so measured_commit has somewhere older to be backdated TO,
  # and so the evidence set has genuinely moved between the two.
  printf 'fn f1() { let _changed = 1; }\n' > "$td/crates/demo/src/m1.rs"
  selftest_commit "$td" 'touch evidence' || return 1
  sed -i "s/PLACEHOLDER/$(git -C "$td" rev-parse HEAD)/" "$td/$CONTRACT"
  selftest_commit "$td" 'record measured_commit' || return 1
}

# A fixture commit that fails must ABORT the self-test. An unchecked `git
# commit` leaves the scratch repo with no HEAD, every run then fails for
# "ledger absent from main", and the three mutations look perfectly RED while
# proving nothing at all.
selftest_commit() {
  local td="$1" msg="$2" out rc
  git -C "$td" add -A
  out="$(git -C "$td" commit -qm "$msg" --no-verify 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    printf 'FIXTURE COMMIT FAILED (%s): %s\n' "$msg" "$out" >&2
    return 1
  fi
  return 0
}

selftest_run() {
  local td="$1" label="$2" expect="$3" out rc
  out="$(DOGFOOD_GATE_ROOT="$td" DOGFOOD_BASE_REF=main \
         bash "$td/scripts/check_dogfood_coverage.sh" 2>&1)"; rc=$?
  local verdict="GREEN"; [ "$rc" -ne 0 ] && verdict="RED"
  if [ "$verdict" = "$expect" ]; then
    printf '  %-46s %-5s (expected %s)  OK\n' "$label" "$verdict" "$expect"
    return 0
  fi
  printf '  %-46s %-5s (expected %s)  MISMATCH\n' "$label" "$verdict" "$expect"
  printf '%s\n' "$out" | sed 's/^/        | /'
  return 1
}

# A sed that fails to match reads exactly like a passing gate. Every mutation is
# applied through this, which refuses to continue unless the file actually moved.
selftest_mutate() {
  local file="$1" desc="$2"; shift 2
  local before after
  before="$(md5sum "$file" | cut -d' ' -f1)"
  "$@"
  after="$(md5sum "$file" | cut -d' ' -f1)"
  if [ "$before" = "$after" ]; then
    printf '  MUTATION DID NOT ENGAGE: %s left %s byte-identical.\n' "$desc" "$file" >&2
    printf '  Refusing to report a verdict from a mutation that never applied.\n' >&2
    return 1
  fi
  printf '  mutation engaged: %-38s %s -> %s\n' "$desc" "${before:0:8}" "${after:0:8}"
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  TD="$(mktemp -d)"
  [ -n "${TD:-}" ] && [ -d "$TD" ] || fail "could not create a scratch dir"
  trap 'rm -rf "${TD:?}" "${TD:?}-shallow"' EXIT
  FAILED=0

  printf 'check_dogfood_coverage.sh --self-test\n'
  printf 'Scratch repo with a protected `main` carrying the ledger, so the ARMED\n'
  printf 'comparand path is what is under test.\n\n'

  # --- baseline: clean tree must be GREEN, or every RED below proves nothing.
  selftest_build_repo "$TD" || fail "could not build the fixture repository"
  selftest_run "$TD" "clean tree" "GREEN" || FAILED=1

  # --- no-op discrimination: rewrite the ledger with identical bytes.
  cp "$TD/$LEDGER" "$TD/.ledger.bak"; cp "$TD/.ledger.bak" "$TD/$LEDGER"
  selftest_run "$TD" "no-op rebuild (identical bytes)" "GREEN" || FAILED=1
  rm -f "$TD/.ledger.bak"
  printf '\n'

  # --- M1 (G2.1): leave the ledger behind HEAD by moving audited code without
  #     re-auditing it. This is what "the CSV is backdated" actually looks like
  #     in a repository: the evidence moved and the ledger did not follow.
  printf 'M1  G2.1 freshness — move cited evidence, leave the ledger behind HEAD\n'
  selftest_mutate "$TD/crates/demo/src/m2.rs" "edit cited evidence m2.rs" \
    sed -i 's/fn f2() {}/fn f2() { let moved = 1; }/' "$TD/crates/demo/src/m2.rs" \
    && selftest_run "$TD" "evidence moved, ledger untouched" "RED" || FAILED=1
  # ...and the SAME evidence change WITH a ledger update must be GREEN, or the
  # gate is just banning edits rather than requiring a re-audit.
  # quality 6 so it adds no triage obligation, and a real hardware value so it
  # does not trip the UNKNOWN ratchet -- this case is about freshness alone.
  printf 'demo,demo delta,6,x86_64-linux,none,no,crates/demo/src/m2.rs:1,high\n' >> "$TD/$LEDGER"
  selftest_run "$TD" "evidence moved, ledger re-audited" "GREEN" || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER" "crates/demo/src/m2.rs"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M2 (G2.2): delete a ledger row for a live command.
  printf 'M2  G2.2 reconciliation — delete one row for a live command\n'
  selftest_mutate "$TD/$LEDGER" "delete the \`demo beta\` row" \
    sed -i '/^demo,demo beta,/d' "$TD/$LEDGER" \
    && selftest_run "$TD" "row deleted" "RED" || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M3 (G2.3): flip one in_dogfood_skill yes -> no.
  printf 'M3  G2.3 floors — flip one in_dogfood_skill yes -> no\n'
  selftest_mutate "$TD/$LEDGER" "ungate \`demo gamma\`" \
    sed -i 's|^demo,demo gamma,6,UNKNOWN,none,yes,|demo,demo gamma,6,UNKNOWN,none,no,|' "$TD/$LEDGER" \
    && selftest_run "$TD" "coverage dropped" "RED" || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M4: the anti-vacuity mutation. Lower the floor AND break it in one
  #     commit — the exact move that defeated the multi-platform gate. It must
  #     still be RED, because the floor is not in this tree.
  printf 'M4  anti-vacuity — lower the floor and break it in ONE commit\n'
  git -C "$TD" checkout -q -b attacker
  sed -i 's|^demo,demo gamma,6,UNKNOWN,none,yes,|demo,demo gamma,6,UNKNOWN,none,no,|' "$TD/$LEDGER"
  sed -i 's/^    measured_commit: .*/    measured_commit: PLACEHOLDER/' "$TD/$CONTRACT"
  selftest_commit "$TD" 'lower the floor and break it in one commit' \
    || fail "fixture commit failed"
  sed -i "s/PLACEHOLDER/$(git -C "$TD" rev-parse HEAD)/" "$TD/$CONTRACT"
  selftest_commit "$TD" 'record measured_commit' || fail "fixture commit failed"
  printf '  committed on branch `attacker`; `main` still holds the original ledger\n'
  selftest_run "$TD" "floor edited in the same commit" "RED" || FAILED=1
  git -C "$TD" checkout -q main
  printf '\n'

  # --- M5 (G2.1 provenance): the measured_commit line. Three shapes must be RED
  #     in any repository, and the fourth case is the regression test for the
  #     defect that made this gate red on its own pull request -- a well-formed
  #     SHA absent ONLY because the clone is shallow must NOT be red. Both
  #     halves are asserted, because a check that reds on the shallow clone as
  #     well as on the bad value is measuring the checkout, not the ledger.
  printf 'M5  G2.1 provenance — measured_commit shape, and depth-independence\n'
  BOGUS_SHA=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
  selftest_mutate "$TD/$CONTRACT" "measured_commit -> PLACEHOLDER" \
    sed -i 's/^    measured_commit: .*/    measured_commit: PLACEHOLDER/' "$TD/$CONTRACT" \
    && selftest_run "$TD" "measured_commit is not an object name" "RED" || FAILED=1
  git -C "$TD" checkout -q -- "$CONTRACT"
  selftest_mutate "$TD/$CONTRACT" "delete the measured_commit line" \
    sed -i '/^    measured_commit:/d' "$TD/$CONTRACT" \
    && selftest_run "$TD" "measured_commit line deleted" "RED" || FAILED=1
  git -C "$TD" checkout -q -- "$CONTRACT"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1

  # The discriminating pair. Same contract, same SHA, two checkout depths.
  selftest_mutate "$TD/$CONTRACT" "measured_commit -> absent object" \
    sed -i "s/^    measured_commit: .*/    measured_commit: $BOGUS_SHA/" "$TD/$CONTRACT" \
    && selftest_run "$TD" "SHA names nothing, full clone" "RED" || FAILED=1
  selftest_commit "$TD" 'record an absent measured_commit' || fail "fixture commit failed"
  SHALLOW="${TD}-shallow"
  rm -rf "${SHALLOW:?}"
  # `--no-local` and a file:// URL: a local-path clone HARDLINKS the whole object
  # store and silently ignores --depth, which would leave the fixture deep and
  # this case asserting nothing. The depth is verified below, not assumed.
  if git clone -q --no-local --depth=1 --template= "file://$TD" "$SHALLOW" 2>/dev/null \
     && [ "$(git -C "$SHALLOW" rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
    printf '  shallow fixture built: depth-1 clone, is-shallow-repository=true\n'
    selftest_run "$SHALLOW" "same SHA, depth-1 clone (must NOT be red)" "GREEN" || FAILED=1
  else
    printf '  SHALLOW FIXTURE NOT BUILT — the depth-independence case did not run.\n' >&2
    printf '  A case that did not run is not a case that passed.\n' >&2
    FAILED=1
  fi
  rm -rf "${SHALLOW:?}"
  git -C "$TD" reset -q --hard HEAD~1
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  if [ "$FAILED" -eq 0 ]; then
    printf 'SELF-TEST PASS: 4 registered mutation groups RED, the anti-vacuity mutation\n'
    printf 'RED, the shallow-clone case GREEN, and every clean/no-op/restored tree GREEN.\n'
    exit 0
  fi
  printf 'SELF-TEST FAIL\n'
  exit 1
fi

printf 'apr-dogfood coverage gate\n'
run_gate "$REPO_ROOT"
exit $?
