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
#   G2.5 per-cluster     a cluster lost a gate, or the zero-gate cluster count
#                        rose; at --release, every cluster_label must carry >= 1
#                        gate. Cluster coverage is reported only BESIDE the
#                        feature fraction, and that pairing is enforced (T2)
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
# against it under each of the six registered mutations and under a no-op.
# A gate that fires on the no-op as well as the mutation is not measuring the
# mutation, so both verdicts are asserted every time.
#
# The registered mutations, each RED with a paired GREEN restore:
#   M1  G2.1  move cited evidence, leave the ledger behind HEAD
#   M2  G2.2  delete a ledger row for a live command
#   M3  G2.3  flip one in_dogfood_skill yes -> no
#   M5  G2.5  move a cluster's ONLY gate to another cluster (totals unchanged)
#   M6  T2    emit cluster coverage with no feature fraction beside it
#   M7  T1    key a contract on the permuting id instead of the label
#   M8  G2.5  move a gated feature into a zero-gate cluster and write a NEW gate
#             in the cluster it left, so every count-based floor still holds --
#             the relabelling attack, GREEN before the membership ratchet existed
#   M8b G2.5  DECLARE that same move and arm the release: still RED, because the
#             gate that walked in is inherited, not earned. Paired GREEN: at the
#             same release arm, WRITE a gate in the zero cluster instead
#   M9  T2    state a cluster ratio in the RECEIPT with no feature fraction --
#             the channel enforce_pairing() did not cover
#   M10 T1    key on the id with a form the four-syntax blacklist walked past
#   M4        anti-vacuity: lower the floor AND break it in one commit

set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="${DOGFOOD_GATE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEDGER="docs/audits/surface_audit.csv"
CONTRACT="contracts/apr-dogfood-coverage-v1.yaml"
THE44="docs/audits/dogfood-the-44.yaml"
GATE_PY="scripts/lib/dogfood_coverage_gate.py"
IDGUARD="scripts/check_no_cluster_id_keys.sh"
REASSIGN="docs/audits/cluster_reassignments.yaml"
SKILL=".claude/skills/apr-dogfood/SKILL.md"
BASE_REF="${DOGFOOD_BASE_REF:-origin/main}"

# T2 is enforced wherever a cluster-level ratio can be EMITTED, not at one call
# site. The gate's own report is checked inside the python module; these are the
# other channels. $DOGFOOD_RECEIPT is appended when set, so a produced receipt is
# held to the same rule as the template it came from.
#   the contract  — the numbers a reader quotes
#   the skill     — the allocation table and the receipt template
#   the receipt   — the artifact the release decision is made on
# Written as an `if` rather than `[ ... ] && arr+=(...)`: bashrs parses the
# array-append parentheses on a test line as SC1028 and reports two errors that
# are not there. A guard that ships spurious errors stops being read.
PAIR_SURFACES=("$CONTRACT" "$SKILL")
if [ -n "${DOGFOOD_RECEIPT:-}" ]; then
  PAIR_SURFACES+=("$DOGFOOD_RECEIPT")
fi

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
      # NAME THE PATH THAT RAN. An earlier draft of this banner described
      # `read_ledger(..., allow_legacy=True)` as the active branch here. It is
      # not: resolve_base_ref takes THIS fallback first, so the comparand is this
      # branch's own HEAD commit, which already carries the 10-column ledger --
      # the legacy-schema branch never executes and the per-cluster ratchet IS
      # armed, from HEAD. A window that is honest about being open is fine; a
      # window whose description does not match its code is not.
      printf '  BOOTSTRAP: %s carries no ledger yet, so the comparand is the HEAD\n' "$BASE_REF"
      printf '             COMMIT of this branch, never the working tree — a mutation in\n'
      printf '             the tree still turns this gate red.\n'
      printf '             PATH TAKEN: resolve_base_ref -> BOOTSTRAP. Because HEAD already\n'
      printf '             carries the 10-column ledger, read_ledger() takes the CLUSTERED\n'
      printf '             branch and allow_legacy does NOT engage; every ratchet below,\n'
      printf '             per-cluster and membership included, is armed from HEAD.\n'
      printf '             The SCHEMA UPGRADE banner belongs to a different case — a main\n'
      printf '             that carries an 8-column ledger — and is not what ran here.\n'
      printf '             Once this lands, every later run compares against protected %s.\n\n' "$BASE_REF" ;;
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

  # --pair-scan carries T2 to the other channels. Paths are made absolute against
  # $root so the self-test's scratch repo scans ITS files, not this checkout's.
  local -a pairargs=()
  local surface
  for surface in "${PAIR_SURFACES[@]}"; do
    case "$surface" in
      /*) pairargs+=(--pair-scan "$surface") ;;
      *)  pairargs+=(--pair-scan "$root/$surface") ;;
    esac
  done
  python3 "$root/$GATE_PY" --base "$basecsv" --head "$root/$LEDGER" \
    --the44 "$root/$THE44" --reassignments "$root/$REASSIGN" \
    --comparand-source "$mode" "${pairargs[@]}"
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
  cp "$REPO_ROOT/$IDGUARD" "$td/$IDGUARD"

  for i in 1 2 3; do printf 'fn f%s() {}\n' "$i" > "$td/crates/demo/src/m$i.rs"; done

  # Three clusters, shaped for the per-cluster mutations:
  #   solo-cluster    2 features, exactly ONE gate -> M5 removes its only gate
  #   pair-cluster    1 feature,  exactly ONE gate
  #   pad-cluster   200 features, ZERO gates       -> M5 moves the gate HERE, so the
  #                                                   overall and per-binary counts
  #                                                   are unchanged and only the
  #                                                   per-cluster floor can explain
  #                                                   the RED
  {
    printf 'binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,cluster_id,cluster_label,evidence_path,confidence\n'
    printf 'demo,demo alpha,2,x86_64-linux,none,no,0,solo-cluster,crates/demo/src/m1.rs:1,high\n'
    printf 'demo,demo beta,6,UNKNOWN,none,yes,0,solo-cluster,crates/demo/src/m2.rs:1,high\n'
    printf 'demo,demo gamma,6,UNKNOWN,none,yes,1,pair-cluster,crates/demo/src/m3.rs:1,high\n'
    for i in $(seq 1 200); do
      printf 'demo,demo pad%s,6,UNKNOWN,none,no,2,pad-cluster,crates/demo/src/m1.rs:1,medium\n' "$i"
    done
  } > "$td/$LEDGER"

  printf 'metadata:\n  baselines:\n    measured_commit: PLACEHOLDER\n' > "$td/$CONTRACT"
  # `printf` and a leading `-`: "- feature:" is parsed as an OPTION, not a
  # format. Always route caller-shaped text through a %s.
  local q="'"
  # The waiver is present so DOGFOOD_RELEASE=1 is usable in this fixture: M8b
  # needs the release arm to be decided by the PER-CLUSTER rule, and an untriaged
  # entry would make it red for G2.4 instead and prove nothing about G2.5.
  { printf 'features:\n'
    printf '%s\n' "- feature: ${q}demo alpha${q}" '  issue: null' \
                   "  waiver: ${q}fixture: alpha is deliberately broken-and-ungated${q}"
  } > "$td/$THE44"

  # No reassignment has happened, so the log is empty. M8 mutates the LEDGER and
  # leaves this alone; M8b adds the matching entry.
  printf 'reassignments: []\n' > "$td/$REASSIGN"

  # The two other channels a cluster ratio can be emitted through. Both carry the
  # pairing, so the T2 scan is not vacuous before M9 removes it.
  mkdir -p "$td/$(dirname "$SKILL")"
  { printf '# fixture skill\n\n'
    printf 'Allocation: clusters gated 2/3, features gated 2/203.\n'
  } > "$td/$SKILL"
  { printf '# fixture receipt\n\n'
    printf 'Coverage: clusters gated 2/3, features gated 2/203.\n'
  } > "$td/docs/audits/receipt.md"

  # `git init` is NOT hermetic on a developer box: this machine sets
  # init.templatedir=~/.git-templates, so a scratch repo silently inherits the
  # analyser's pre-commit hooks and every fixture commit fails. The first draft of
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

# The optional 4th argument is a REQUIRED MARKER in the output. A RED verdict
# alone does not say WHICH floor fired: M5 deliberately keeps the overall and
# per-binary counts constant, so if it went red for some other reason the
# per-cluster floor would still be unproven. Asserting the finding text is what
# makes the mutation attributable to the gate it is registered against.
selftest_run() {
  local td="$1" label="$2" expect="$3" marker="${4:-}" out rc
  # DOGFOOD_RECEIPT points the T2 scan at the fixture's receipt, so the receipt
  # channel is exercised by every run and M9 has something to break.
  # SELFTEST_RELEASE arms DOGFOOD_RELEASE for the one mutation that needs it.
  out="$(DOGFOOD_GATE_ROOT="$td" DOGFOOD_BASE_REF=main \
         DOGFOOD_RECEIPT="$td/docs/audits/receipt.md" \
         DOGFOOD_RELEASE="${SELFTEST_RELEASE:-}" \
         bash "$td/scripts/check_dogfood_coverage.sh" 2>&1)"; rc=$?
  local verdict="GREEN"; [ "$rc" -ne 0 ] && verdict="RED"
  if [ "$verdict" != "$expect" ]; then
    printf '  %-46s %-5s (expected %s)  MISMATCH\n' "$label" "$verdict" "$expect"
    printf '%s\n' "$out" | sed 's/^/        | /'
    return 1
  fi
  if [ -n "$marker" ] && ! grep -qF "$marker" <<<"$out"; then
    printf '  %-46s %-5s but the output never says <%s>  MISATTRIBUTED\n' \
      "$label" "$verdict" "$marker"
    printf '%s\n' "$out" | sed 's/^/        | /'
    return 1
  fi
  printf '  %-46s %-5s (expected %s)  OK%s\n' "$label" "$verdict" "$expect" \
    "$([ -n "$marker" ] && printf ', attributed')"
  return 0
}

# The T1 guard runs over the FIXTURE repo, so it needs its own runner. Same
# contract as selftest_run: a verdict plus an optional attribution marker.
idguard_run() {
  local td="$1" label="$2" expect="$3" out rc
  out="$(CLUSTER_ID_GUARD_ROOT="$td" bash "$td/scripts/check_no_cluster_id_keys.sh" 2>&1)"
  rc=$?
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
  printf 'demo,demo delta,6,x86_64-linux,none,no,1,pair-cluster,crates/demo/src/m2.rs:1,high\n' >> "$TD/$LEDGER"
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
    sed -i 's|^demo,demo gamma,6,UNKNOWN,none,yes,1,|demo,demo gamma,6,UNKNOWN,none,no,1,|' "$TD/$LEDGER" \
    && selftest_run "$TD" "coverage dropped" "RED" || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M5 (G2.5): remove the only gate from a 1-gate cluster, and ADD one
  #     somewhere else so the overall and per-binary counts do not move. That is
  #     precisely the trade a binary-level floor cannot see: `aprender-orchestrate`
  #     is three unrelated subsystems, so one gate on Pacha makes all 184 features
  #     look touched. Only the per-cluster floor can explain this RED, which is
  #     why the finding text is asserted too.
  printf 'M5  G2.5 per-cluster — move a cluster\x27s ONLY gate to another cluster\n'
  selftest_mutate "$TD/$LEDGER" "ungate solo-cluster, gate a pad row" \
    sed -i -e 's|^demo,demo beta,6,UNKNOWN,none,yes,0,|demo,demo beta,6,UNKNOWN,none,no,0,|' \
           -e 's|^demo,demo pad2,6,UNKNOWN,none,no,2,|demo,demo pad2,6,UNKNOWN,none,yes,2,|' \
           "$TD/$LEDGER" \
    && selftest_run "$TD" "gate moved between clusters" "RED" \
         "G2.5 per-cluster FAIL: gates in cluster \`solo-cluster\`" || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M8 (G2.5 membership): THE RELABELLING ATTACK.
  #     Move a GATED feature into the zero-gate cluster, and write a NEW gate in
  #     the cluster it left so that cluster's floor still holds. Every count the
  #     first version of this gate ratcheted is satisfied afterwards: solo-cluster
  #     still has one gate, pad-cluster's gate count went UP, the zero-gate
  #     cluster count went DOWN, no label vanished, per-binary and overall both
  #     rose. It was GREEN. And nothing had been proved about pad-cluster's 200
  #     features -- the zero stopped EXISTING instead of being closed. That is the
  #     same move as deleting a losing benchmark row.
  #     Only the membership ratchet can explain this RED, so the finding is asserted.
  m8_relabel() {
    sed -i 's|^demo,demo beta,6,UNKNOWN,none,yes,0,solo-cluster,|demo,demo beta,6,UNKNOWN,none,yes,2,pad-cluster,|' "$TD/$LEDGER"
    printf 'demo,demo epsilon,6,x86_64-linux,none,yes,0,solo-cluster,crates/demo/src/m3.rs:1,high\n' >> "$TD/$LEDGER"
  }
  printf 'M8  G2.5 membership — relabel a gated feature into the zero-gate cluster\n'
  selftest_mutate "$TD/$LEDGER" "move the gate into pad-cluster, backfill solo" \
    m8_relabel \
    && selftest_run "$TD" "gated feature relabelled, no declaration" "RED" \
         "G2.5 membership FAIL" || FAILED=1

  # --- M8b: DECLARE that same move, and arm the release. The declaration makes
  #     the move legible and legal -- clusters are derived, so a re-cluster must
  #     stay possible -- and it still does not lift pad-cluster off zero, because
  #     the gate that walked in is evidence about the cluster it came FROM.
  #     A declaration buys legibility; only writing a gate buys evidence.
  printf 'M8b G2.5 earned — declaring the move does not lift the cluster off zero\n'
  m8b_declare() {
    { printf 'reassignments:\n'
      printf '  - binary: demo\n'
      printf '    feature: demo beta\n'
      printf '    from: solo-cluster\n'
      printf '    to: pad-cluster\n'
      printf '    reason: fixture -- re-clustered, beta dispatches through the pad path\n'
    } > "$TD/$REASSIGN"
  }
  selftest_mutate "$TD/$REASSIGN" "declare the reassignment" m8b_declare || FAILED=1
  SELFTEST_RELEASE=1 selftest_run "$TD" "declared move, release armed" "RED" \
    "carry no EARNED gate" || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER" "$REASSIGN"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  # ...and the paired GREEN that says what the arm actually wants: WRITE a gate
  # in the zero cluster. Same release arm, same fixture, opposite verdict.
  selftest_mutate "$TD/$LEDGER" "write a gate on a pad-cluster row" \
    sed -i 's|^demo,demo pad1,6,UNKNOWN,none,no,2,|demo,demo pad1,6,UNKNOWN,none,yes,2,|' "$TD/$LEDGER" \
    && SELFTEST_RELEASE=1 selftest_run "$TD" "gate WRITTEN in the zero cluster, release armed" "GREEN" \
    || FAILED=1
  git -C "$TD" checkout -q -- "$LEDGER"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M9 (T2, the RECEIPT channel). enforce_pairing() used to be applied to ONE
  #     generated string -- this gate's own report. The failure it exists to
  #     prevent is a RECEIPT that states cluster coverage without the feature
  #     fraction: the gate then looks STRICTER than before while measuring LESS.
  #     That is vacuity one level up, which is the whole point of T2.
  printf 'M9  T2 — state a cluster ratio in the RECEIPT with no feature %% beside it\n'
  selftest_mutate "$TD/docs/audits/receipt.md" "drop the feature fraction from the receipt" \
    sed -i 's|, features gated 2/203||' "$TD/docs/audits/receipt.md" \
    && selftest_run "$TD" "receipt states the proxy alone" "RED" \
         "G2.6 T2 FAIL" || FAILED=1
  git -C "$TD" checkout -q -- "docs/audits/receipt.md"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  # The same removal in the SKILL, because "one channel" was the defect.
  selftest_mutate "$TD/$SKILL" "drop the feature fraction from the skill" \
    sed -i 's|, features gated 2/203||' "$TD/$SKILL" \
    && selftest_run "$TD" "skill states the proxy alone" "RED" \
         "G2.6 T2 FAIL" || FAILED=1
  git -C "$TD" checkout -q -- "$SKILL"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'
  # --- M6 (T2): report CLUSTER coverage without the FEATURE fraction. This is
  #     the most important mutation in the table. Cluster coverage is a PROXY:
  #     one gate in a 95-member cluster is 1%, not "covered". A receipt that
  #     prints "5 of 14 clusters gated" alone rebuilds the vacuity failure one
  #     level up, and looks STRICTER than what it replaced while measuring less.
  #     The pairing is therefore mechanical: the gate reads back its own report.
  printf 'M6  T2 — emit cluster coverage with no feature %% beside it\n'
  selftest_mutate "$TD/$GATE_PY" "delete the feature fraction from the emitter" \
    sed -i 's|"features gated {}/{} ({:.1f}%)"\.format(|"".format(|' "$TD/$GATE_PY" \
    && selftest_run "$TD" "feature fraction dropped from the report" "RED" \
         "G2.6 T2 FAIL (the per-cluster report)" || FAILED=1
  cp "$REPO_ROOT/$GATE_PY" "$TD/$GATE_PY"
  selftest_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M7 (T1): key a contract on cluster_id instead of cluster_label. k-means
  #     ids permute on re-run, so an obligation keyed on one silently re-points
  #     at a different set of features next time the surface moves. Run against
  #     the FIXTURE repo, because the guard scans tracked files.
  #     The two column names are held in variables rather than written out as a
  #     keying form. That is not evasion, it is the guard finding its FIRST real
  #     violation: written literally, `sed 's|key: X|key: Y|'` in this harness is
  #     itself a line that keys on the permuting id, and the guard went red on its
  #     own mutation harness the first time it ran. Naming a column in a variable
  #     is not keying an obligation on it; the fixture still writes a genuine
  #     violation to disk, which is what the mutation asserts.
  CLUSTER_COL_PREFIX='cluster_'
  DURABLE_KEY="${CLUSTER_COL_PREFIX}label"
  PERMUTING_KEY="${CLUSTER_COL_PREFIX}id"
  printf 'M7  T1 — key a contract on the permuting id instead of the label\n'
  printf 'obligations:\n  - floor:\n      key: %s\n' "$DURABLE_KEY" \
    > "$TD/contracts/cluster-floor.yaml"
  git -C "$TD" add -A >/dev/null 2>&1
  selftest_commit "$TD" 'a contract keyed on the durable label' || fail "fixture commit failed"
  idguard_run "$TD" "contract keyed on the durable label" "GREEN" || FAILED=1
  selftest_mutate "$TD/contracts/cluster-floor.yaml" "swap the key to the permuting id" \
    sed -i "s|key: ${DURABLE_KEY}|key: ${PERMUTING_KEY}|" "$TD/contracts/cluster-floor.yaml" \
    && { git -C "$TD" add -A >/dev/null 2>&1
         selftest_commit "$TD" 'key it on the permuting id instead' >/dev/null
         idguard_run "$TD" "contract keyed on the permuting id" "RED"; } || FAILED=1
  sed -i "s|key: ${PERMUTING_KEY}|key: ${DURABLE_KEY}|" "$TD/contracts/cluster-floor.yaml"
  git -C "$TD" add -A >/dev/null 2>&1
  selftest_commit "$TD" 'restore the durable key' >/dev/null
  idguard_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1

  # --- M10 (T1, the form the BLACKLIST walked past). The first version of the
  #     guard recognised four keying syntaxes and passed everything else. This is
  #     one of the twelve ordinary constructs it did not recognise: a pandas-style
  #     group-by is not a YAML key, not an identity field, not a `["..."]`
  #     subscript and not a `--by` flag, so it keyed an obligation on the
  #     permuting id and stayed GREEN. The guard is now an ALLOWLIST, so this is
  #     RED without anyone having had to predict the syntax.
  #     The token is assembled from a variable for the same reason M7 does it:
  #     written literally, this harness would itself be a keying line, and the
  #     guard scans scripts/.
  printf 'M10 T1 — key on the id with a form the four-syntax blacklist missed\n'
  m10_groupby() {
    printf 'floors = ledger.groupby("%s").size()\n' "$PERMUTING_KEY" \
      > "$TD/scripts/floor_by_cluster.py"
  }
  printf '# derives the per-cluster floor\n' > "$TD/scripts/floor_by_cluster.py"
  git -C "$TD" add -A >/dev/null 2>&1
  selftest_commit "$TD" 'a floor script that keys on nothing' >/dev/null
  idguard_run "$TD" "floor script keys on nothing" "GREEN" || FAILED=1
  selftest_mutate "$TD/scripts/floor_by_cluster.py" "group the floor by the permuting id" \
    m10_groupby \
    && { git -C "$TD" add -A >/dev/null 2>&1
         selftest_commit "$TD" 'group the floor by the permuting id' >/dev/null
         idguard_run "$TD" "floor grouped by the permuting id" "RED"; } || FAILED=1
  printf '# derives the per-cluster floor\n' > "$TD/scripts/floor_by_cluster.py"
  git -C "$TD" add -A >/dev/null 2>&1
  selftest_commit "$TD" 'restore the floor script' >/dev/null
  idguard_run "$TD" "restored (discrimination check)" "GREEN" || FAILED=1
  printf '\n'

  # --- M4: the anti-vacuity mutation. Lower the floor AND break it in one
  #     commit — the exact move that defeated the multi-platform gate. It must
  #     still be RED, because the floor is not in this tree.
  printf 'M4  anti-vacuity — lower the floor and break it in ONE commit\n'
  git -C "$TD" checkout -q -b attacker
  sed -i 's|^demo,demo gamma,6,UNKNOWN,none,yes,1,|demo,demo gamma,6,UNKNOWN,none,no,1,|' "$TD/$LEDGER"
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
    printf 'SELF-TEST PASS: every registered mutation RED (M5/M6/M8/M8b/M9 attributed\n'
    printf 'by finding text, not by exit code alone), the provenance group RED on a\n'
    printf 'placeholder / a deleted line / an absent SHA in a FULL clone, the\n'
    printf 'anti-vacuity mutation RED, and every clean/no-op/restored tree GREEN --\n'
    printf 'including the three GREENs that say what the arms actually want: a\n'
    printf 're-audited ledger (M1), a gate WRITTEN in the zero cluster rather than\n'
    printf 'moved into it (M8b), and the same absent SHA in a depth-1 clone.\n'
    exit 0
  fi
  printf 'SELF-TEST FAIL\n'
  exit 1
fi

printf 'apr-dogfood coverage gate\n'
run_gate "$REPO_ROOT"
exit $?
