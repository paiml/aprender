#!/usr/bin/env bash
# check_roadmap_completion_is_cited.sh — APR-PERF-GATE-001 (#2706), PERF-044.
# A roadmap `status: completed` is a claim. It is evidence only if something
# can prove it.
#
#   bash scripts/check_roadmap_completion_is_cited.sh            # gate
#   bash scripts/check_roadmap_completion_is_cited.sh --selftest # case table
#   bash scripts/check_roadmap_completion_is_cited.sh --update   # re-baseline
#
# WHY THIS EXISTS
# ---------------
# This epic's rule is that a performance number is evidence only if something
# can prove how it was measured. docs/roadmaps/roadmap.yaml fails the same test
# one level up. PERF-040 reconciled the 26 entries carrying `github_issue: 2706`
# against origin/main and found drift in BOTH directions. The sharp case:
#
#   PERF-004 was marked `completed` while none of its three artifacts
#   (perf-receipt-fields.yaml, lib/perf_receipt.py,
#   check_perf_receipt_fields_have_producers.sh) are on main, and its
#   PR #2716 is CLOSED unmerged. `gh pr view 2716 --json state,mergedAt`
#   -> {"state":"CLOSED","mergedAt":null}.
#
# Seven further entries said `planned` for work that had landed.
# `grep -rn roadmap scripts/ .github/ Makefile .githooks/` returned one
# unrelated comment: no guard covered roadmap shape at all.
#
# WHAT A CITATION IS, MECHANICALLY
# --------------------------------
# A `proof:<ref>` token anywhere in the entry's own prose, where <ref> is
# either
#
#   a repo-relative PATH   -> must EXIST in the working tree and be NON-EMPTY
#   #<n> / PR#<n>          -> must be a PullRequest whose state is MERGED
#
# Both are DEREFERENCEABLE, which is the whole point. Three alternatives were
# weighed and rejected:
#
#   a commit SHA        rejected, same reason the sibling PERF-010 guard
#                       rejected it: it records WHEN, not WHAT. Nothing can
#                       dereference it to an artifact.
#   free-form prose     rejected. The 142 notes that exist today already name
#                       things like `commands/serve.rs` and `realizar api.rs`
#                       -- neither is a real repo path (the first is
#                       crates/apr-cli/src/commands/serve.rs, the second is not
#                       a path at all). A detector firing on prose fragments
#                       would be wrong in both directions, and a guard that
#                       cries wolf is deleted.
#   a NEW yaml key      rejected, and this is the load-bearing one. See below.
#
# WHY THE CITATION LIVES IN PROSE AND NOT IN ITS OWN FIELD
# --------------------------------------------------------
# The analyser owns these files, and `"$PMAT" work edit` reparses and REWRITES
# the whole file. An unknown key does not survive. Reproduced here rather than
# taken on faith, and the result is worse than the issue reported -- editing an
# UNRELATED ticket destroys the key on EVERY ticket:
#
#   RT-001: status completed, evidence: [scripts/foo.sh, PR#1234]
#   RT-002: status planned
#   $ "$PMAT" work edit RT-002 --status in_progress
#   ✓ Updated ticket: RT-002
#   $ diff before after
#   -  evidence:            <- RT-001's, deleted by an edit to RT-002
#   -  - scripts/foo.sh
#   -  - PR#1234
#   -  status: planned      <- and note `in_progress` was normalised
#   +  status: inprogress      to `inprogress` on the way in
#
# `notes:` survived that round trip byte-for-byte, and 614 of 640 top-level
# items already carry the key. A long note is REFLOWED, so the token has to
# survive folding too; it does, because a YAML emitter may only break a line at
# an existing space and a `proof:` token contains none. Verified by round trip:
# a 380-char note whose fold lands immediately after `proof:PR#2733` reparses
# with both tokens intact.
#
# That same reflow is why this guard PARSES the YAML instead of grepping it. A
# grep for `proof:` over the raw text is defeated by a fold; hand-rolling a
# YAML reader in bash is the construct check_no_hand_rolled_parsers.sh bans.
# python3 + PyYAML is the shape scripts/perf_gate.sh already uses for
# perf-matrix.yaml. If PyYAML is missing this guard FAILS -- an unparsed
# roadmap is UNMEASURED, and unmeasured is never "no unproven claims".
#
# THE UNIVERSE, AND THE FREE PASS IT WOULD OTHERWISE GIVE
# -------------------------------------------------------
# The issue names docs/roadmaps/roadmap.yaml. That file holds 289 of the
# repository's 1033 `completed` claims. There are 21 analyser work-contract files:
# one per crate under */docs/roadmaps/, plus book/. Scoping to the file the
# issue happened to name would hand 744 completed claims -- 72% of them -- a
# free pass, which is the universe defect this epic has now paid for four
# times. So the scan is DERIVED:
#
#   working tree (find) UNION the index (git ls-files), because a tracked-only
#   universe is a free pass for an untracked file, and untracked is exactly how
#   a new roadmap arrives;
#   every roadmap*.yaml / ROADMAP*.yaml;
#   IN SCOPE iff the file declares a top-level `roadmap:` key -- the analyser work
#   contract schema. Five files in this tree are named roadmap.yaml and are a
#   different schema entirely (`project:`/`milestones:`/`epics:`/`items:`);
#   they are reported out of scope WITH their top-level keys, never skipped
#   silently. An in-scope file that will not PARSE is a hard failure.
#
# Status matching is normalised through the analyser's own alias table
# (`"$PMAT" work list-statuses`): done/finished/closed all mean completed, and
# case, hyphens and underscores are insignificant. That is not hypothetical --
# docs/roadmaps/roadmap.yaml writes `in_progress` while every other file in the
# tree writes `inprogress`. A naive `== "completed"` filter is a universe hole.
#
# Subtasks and phases carry their own `status` and are scanned too. They have
# no `notes` field, so a completed subtask cites through its own title or
# through its PARENT's notes -- without that a subtask claim would be
# unprovable BY CONSTRUCTION, and a gate whose remedy is impossible gets
# deleted rather than satisfied.
#
# THE ENTRIES THAT CITE NOTHING -- THE DESIGN DECISION
# ----------------------------------------------------
# 1033 completed claims exist and 1033 of them cite nothing. Failing them on
# day one is unshippable; passing them quietly is the "gates that cannot fail"
# pattern this epic exists to remove. Neither is taken.
#
# They are FROZEN into a shrink-only baseline, keyed by entry id -- the
# construction scripts/lib_baseline_ratchet.sh exists for, and the one
# check_perf_claims_cite_receipts.sh uses for the same "must cite" shape one
# level down. The consequence is the point:
#
#   * an entry ALREADY claiming completed is recorded, not blessed;
#   * an entry NEWLY flipped to completed is not in the baseline, so it must
#     cite or it fails -- which is precisely the PERF-004 shape;
#   * a NEW completed entry likewise;
#   * the baseline may only SHRINK, compared against merge-base(HEAD,
#     origin/main) with the origin/main tip as the shallow-checkout fallback --
#     a ref this branch cannot rewrite. Appending is REFUSED, so the gate
#     cannot be laundered in the commit that breaks it.
#
# Keyed by ID and never by line number, because the analyser REFLOWS the file: the
# round trip above moved 13516 lines to 12804 and would have invalidated every
# line-keyed entry in one unrelated edit.
#
# A DANGLING CITATION FAILS EVEN AT A BASELINED ENTRY, and unlike the sibling
# guard it fails even when a second token resolves. An uncited entry is
# inherited debt: visible, recorded, ratcheted down. A `proof:` token pointing
# at a path that is not there is an ACTIVE FORGERY -- someone typed it -- and it
# buys the reader's trust with a file nobody can open. The remedy is to delete
# the dead token, which costs nothing, so there is no reason to soften this.
#
# WHAT THIS GUARD DOES NOT DO. It does not decide whether the artifact actually
# discharges the claim; a reviewer does that. It decides only whether anything
# at all can be dereferenced.
#
# STATED RESIDUAL HOLES, rather than hidden ones:
#   * a path added in the SAME commit as the completed claim resolves. That is
#     the honest flow -- land the work and tick the box -- and demanding the
#     artifact be on main FIRST would force a two-PR dance for every completion
#     and get this guard routed around. The `touch` version of that dodge is
#     closed by requiring the path to be NON-EMPTY.
#   * two entries in one file sharing an id are keyed by positional ordinal
#     (`id [#2]`), so reordering duplicated ids churns those keys. Two exist
#     today, both in aprender-orchestrate (PMAT-139, PMAT-141).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_REL="scripts/roadmap_uncited_completion_baseline.txt"

# Minimum in-scope roadmap files before the scan is believed. 21 today; a floor
# of 10 catches a broken universe without breaking on a crate being retired.
# A universe that collapses to zero sweeps clean and READS AS A PASS, which is
# the failure this epic keeps finding.
MIN_ROADMAPS=10

# ---------------------------------------------------------------- scanning --
# Emit one TAB-separated record per status-bearing node whose status means
# completed:
#
#     <file>\t<key>\t<token>|<token>|...
#
# TAB, not `:`, because ids contain both colons and spaces in this tree (the
# longest is 167 characters). No id in any roadmap file contains a tab or a
# newline, which is what makes TAB safe -- checked over all 1477 entries.
#
# The scan is a FUNCTION taking a root so the case table can drive it over a
# fixture tree. A table that could only exercise the regex, and never the
# schema detection or the status aliases, would leave the hard halves unproven.
scan_roadmaps() { # scan_roadmaps <root> ; stdout records, stderr diagnostics
    local root="$1"
    python3 - "$root" <<'PY'
import os, re, sys

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "FATAL PyYAML is not importable, so every roadmap in this tree is\n"
        "      UNPARSED. An unmeasured roadmap is not 'no unproven claims'.\n"
        "      Install it (uv pip install pyyaml) rather than skipping.\n")
    sys.exit(3)

root = sys.argv[1]

# the analyser's own alias table, from its `work list-statuses` subcommand.
# Case, hyphens and underscores are insignificant there, so they are
# insignificant here.
DONE = {"completed", "done", "finished", "closed"}
def is_completed(s):
    if not isinstance(s, str):
        return False
    return re.sub(r"[\s_-]+", "", s).strip().lower() in DONE

TOKEN = re.compile(r"proof:([^\s,;]+)", re.IGNORECASE)
def tokens(*texts):
    out = []
    for t in texts:
        if not isinstance(t, str):
            continue
        for m in TOKEN.finditer(t):
            # Trailing markdown/prose punctuation is not part of the reference,
            # so `(proof:scripts/x.sh).` and `proof:scripts/x.sh,` both resolve.
            ref = m.group(1).rstrip(".,;:)]}'\"`")
            if ref:
                out.append(ref)
    return out

# THE UNIVERSE. find(working tree) UNION git ls-files(index): a tracked-only
# universe hands an untracked file a free pass, and untracked is how a new
# roadmap arrives.
# A path is a candidate if its BASENAME looks like a roadmap, OR if it is any
# .yaml sitting in a docs/roadmaps/ directory -- pmat's own convention for
# where a work contract lives. The second arm is not redundant: the first,
# alone, gave a free pass to any pmat-schema file under docs/roadmaps/ that was
# not called roadmap*.yaml, which the case table caught as two red rows.
def is_candidate(rel):
    d, fn = os.path.split(rel)
    if not fn.lower().endswith(".yaml"):
        return False
    if re.fullmatch(r"(?i)roadmap.*\.yaml", fn):
        return True
    return d.replace(os.sep, "/").endswith("docs/roadmaps")

cand = set()
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in (".git", "target", "node_modules")]
    for fn in filenames:
        rel = os.path.relpath(os.path.join(dirpath, fn), root)
        if is_candidate(rel):
            cand.add(rel)
import subprocess
try:
    idx = subprocess.run(["git", "-C", root, "ls-files"],
                         capture_output=True, text=True, timeout=120)
    if idx.returncode == 0:
        for line in idx.stdout.splitlines():
            if is_candidate(line):
                cand.add(line)
except Exception:
    pass

in_scope, out_scope, unparsed = [], [], []
for rel in sorted(cand):
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        continue
    try:
        raw = open(p, encoding="utf-8", errors="replace").read()
    except OSError as e:
        unparsed.append((rel, f"unreadable: {e}"))
        continue
    # Cheap schema probe FIRST, so a file of a different schema that happens
    # not to parse is classified out rather than reddening the gate. A file
    # claiming the analyser contract schema and then failing to parse is a hard
    # failure; one that never claimed it is not this guard's business.
    if not re.search(r"(?m)^roadmap:", raw):
        top = re.findall(r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", raw)[:5]
        out_scope.append((rel, "no top-level 'roadmap:' key; top keys: "
                               + (", ".join(top) or "none")))
        continue
    try:
        doc = yaml.safe_load(raw)
    except Exception as e:
        unparsed.append((rel, str(e).splitlines()[0]))
        continue
    if not isinstance(doc, dict) or not isinstance(doc.get("roadmap"), list):
        out_scope.append((rel, "'roadmap:' is not a list of work items"))
        continue
    in_scope.append((rel, doc))

for rel, why in out_scope:
    sys.stderr.write(f"SCOPE out  {rel}  ({why})\n")
for rel, why in unparsed:
    sys.stderr.write(f"UNPARSED   {rel}  ({why})\n")
sys.stderr.write(f"COUNT {len(in_scope)}\n")
if unparsed:
    sys.exit(4)

records = []
for rel, doc in in_scope:
    seen = {}
    def emit(key, toks):
        n = seen.get(key, 0) + 1
        seen[key] = n
        # Positional ordinal ONLY on a repeat, so the common case never carries
        # one and a duplicated id cannot shelter under one baseline line.
        k = key if n == 1 else f"{key} [#{n}]"
        records.append(f"{rel}\t{k}\t{'|'.join(toks)}")

    for item in doc["roadmap"]:
        if not isinstance(item, dict):
            continue
        iid = item.get("id")
        iid = iid if isinstance(iid, str) and iid.strip() else "<no id>"
        notes = item.get("notes")
        title = item.get("title")
        if is_completed(item.get("status")):
            emit(iid, tokens(notes, title))
        # Subtasks and phases carry their own status and have no notes field of
        # their own; the parent's notes are therefore in their citation scope,
        # or the claim would be unprovable by construction.
        for st in (item.get("subtasks") or []):
            if isinstance(st, dict) and is_completed(st.get("status")):
                sid = st.get("id") if isinstance(st.get("id"), str) else "<no id>"
                emit(f"sub:{iid}/{sid}", tokens(st.get("title"), notes))
        for ph in (item.get("phases") or []):
            if isinstance(ph, dict) and is_completed(ph.get("status")):
                pn = ph.get("name") if isinstance(ph.get("name"), str) else "<no name>"
                emit(f"phase:{iid}/{pn}", tokens(pn, notes))

for r in records:
    print(r)
PY
}

# ------------------------------------------------------------- resolution --
# A path reference resolves iff it exists AND is non-empty. Emptiness matters:
# the one dodge the working-tree rule would otherwise leave open is `touch`ing
# the cited artifact in the commit that claims completion.
resolve_path_ref() { # resolve_path_ref <root> <ref>  -> rc 0 if it resolves
    local root="$1"
    local ref="$2"
    case "$ref" in
        /*|*..*) return 1 ;;   # absolute or escaping: not a repo artifact
    esac
    if [ -d "$root/$ref" ]; then
        [ -n "$(ls -A "$root/$ref" 2>/dev/null)" ] && return 0
        return 1
    fi
    [ -s "$root/$ref" ]
}

# A PR reference resolves iff GitHub says MERGED. Nothing else counts: #2716 is
# CLOSED with mergedAt null and it is the reason this guard exists.
#
# `gh` unavailable or unauthenticated is a HARD FAILURE, never a skip -- but
# only for an entry that actually cites a PR. Nothing in the tree cites one
# today, so the gate needs no network on the current corpus, and the first
# entry that does cite one pays for the verification honestly.
PR_CACHE_DIR=""
resolve_pr_ref() { # resolve_pr_ref <root> <number> -> rc 0 MERGED, 1 not, 2 unverifiable
    local root="$1"
    local num="$2"
    local cache state
    if [ -n "$PR_CACHE_DIR" ] && [ -f "$PR_CACHE_DIR/$num" ]; then
        cache=$(cat "$PR_CACHE_DIR/$num")
        [ "$cache" = MERGED ] && return 0
        [ "$cache" = UNVERIFIABLE ] && return 2
        return 1
    fi
    if ! command -v gh >/dev/null 2>&1; then
        [ -n "$PR_CACHE_DIR" ] && printf 'UNVERIFIABLE\n' > "$PR_CACHE_DIR/$num"
        return 2
    fi
    state=$(gh pr view "$num" --json state --jq .state 2>/dev/null) || state=""
    if [ -z "$state" ]; then
        [ -n "$PR_CACHE_DIR" ] && printf 'UNVERIFIABLE\n' > "$PR_CACHE_DIR/$num"
        return 2
    fi
    [ -n "$PR_CACHE_DIR" ] && printf '%s\n' "$state" > "$PR_CACHE_DIR/$num"
    [ "$state" = MERGED ] && return 0
    return 1
}

# classify_record <root> <tokens-field> -> uncited | cited | dangling:<detail>
#
# Separate `local` statements, never `local a="$1" b="$a/x"`: bash declares
# every name in one `local` before evaluating any right-hand side, so the
# second reads an unset variable and `set -u` aborts the function mid-way --
# a silent-pass shape the sibling guard's case table caught the hard way.
classify_record() {
    local root="$1"
    local field="$2"
    local tok ref bad="" good=0
    [ -n "$field" ] || { printf 'uncited\n'; return 0; }
    # Herestring, not `printf | while`: a `producer | consumer` pair where the
    # consumer can exit early returns 141 under pipefail though the consumer
    # SUCCEEDED. Five instances of that shape were found in this repository.
    while IFS= read -r tok; do
        [ -n "$tok" ] || continue
        case "$tok" in
            [Pp][Rr]'#'*|'#'*)
                ref="${tok##*#}"
                case "$ref" in
                    ''|*[!0-9]*) bad="${bad}${bad:+, }${tok} (not a PR number)" ; continue ;;
                esac
                resolve_pr_ref "$root" "$ref"
                case $? in
                    0) good=1 ;;
                    1) bad="${bad}${bad:+, }${tok} (not MERGED)" ;;
                    *) bad="${bad}${bad:+, }${tok} (UNVERIFIABLE: gh could not answer)" ;;
                esac ;;
            *)
                if resolve_path_ref "$root" "$tok"; then
                    good=1
                else
                    bad="${bad}${bad:+, }${tok} (absent or empty)"
                fi ;;
        esac
    done <<< "$(printf '%s' "$field" | tr '|' '\n')"
    # A dead token FAILS even beside a live one, and even at a baselined entry.
    # An uncited entry is inherited debt; a proof: token nobody can dereference
    # is an active forgery, and deleting it costs nothing.
    if [ -n "$bad" ]; then printf 'dangling:%s\n' "$bad"; return 0; fi
    if [ "$good" -eq 1 ]; then printf 'cited\n'; return 0; fi
    printf 'uncited\n'
}

# ---------------------------------------------------------------- selftest --
# A guard observed only passing proves nothing. This table drives the REAL
# scan_roadmaps / classify_record over a fixture tree, so the schema probe, the
# status alias table and the reference resolver are each exercised, and every
# row states the answer it must NOT give.
if [ "${1:-}" = "--selftest" ] || [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d) || exit 2
    trap 'rm -rf "${TD:?}"' EXIT
    t=0
    f=0

    mk() { # mk <relpath> ; body on stdin
        mkdir -p "$TD/$(dirname "$1")"
        cat > "$TD/$1"
    }

    row() { # row <label> <want> <got>
        t=$((t + 1))
        if [ "$2" = "$3" ]; then
            printf '  ok    %-10s %s\n' "$2" "$1"
        else
            printf '  FAIL  want %-10s got %-10s %s\n' "$2" "$3" "$1"
            f=$((f + 1))
        fi
    }

    # --- the fixture tree ---------------------------------------------------
    mkdir -p "$TD/scripts" "$TD/emptydir" "$TD/fulldir"
    printf 'real content\n' > "$TD/scripts/real_artifact.sh"
    printf '' > "$TD/scripts/empty_artifact.sh"
    printf 'x\n' > "$TD/fulldir/thing.txt"

    mk docs/roadmaps/roadmap.yaml <<'YAML'
roadmap_version: '1.0'
roadmap:
- id: R-CITED
  status: completed
  notes: 'landed; proof:scripts/real_artifact.sh'
- id: R-DANGLING
  status: completed
  notes: 'landed; proof:scripts/never_existed.sh'
- id: R-EMPTY-ARTIFACT
  status: completed
  notes: 'landed; proof:scripts/empty_artifact.sh'
- id: R-UNCITED
  status: completed
  notes: 'landed, trust me'
- id: R-NO-NOTES
  status: completed
- id: R-PLANNED
  status: planned
  notes: 'nothing here'
- id: R-CANCELLED
  status: cancelled
  notes: 'dropped'
- id: R-INPROGRESS
  status: In-Progress
  notes: 'working'
- id: R-ALIAS-DONE
  status: done
  notes: 'finished ages ago'
- id: R-ALIAS-CLOSED
  status: CLOSED
  notes: 'proof:scripts/real_artifact.sh'
- id: R-UNDERSCORE
  status: Completed
  notes: 'proof:scripts/real_artifact.sh'
- id: R-PUNCT
  status: completed
  notes: 'see (proof:scripts/real_artifact.sh), and also the changelog.'
- id: R-MIXED
  status: completed
  notes: 'proof:scripts/real_artifact.sh proof:scripts/never_existed.sh'
- id: R-DIR-FULL
  status: completed
  notes: 'proof:fulldir'
- id: R-DIR-EMPTY
  status: completed
  notes: 'proof:emptydir'
- id: R-ABSOLUTE
  status: completed
  notes: 'proof:/etc/passwd'
- id: R-ESCAPE
  status: completed
  notes: 'proof:../../../etc/passwd'
- id: R-SUBTASK-PARENT
  status: planned
  notes: 'the parent is not done; proof:scripts/real_artifact.sh'
  subtasks:
  - id: S-DONE
    status: completed
  - id: S-PLANNED
    status: planned
- id: R-PHASE-PARENT
  status: planned
  notes: 'no citation anywhere'
  phases:
  - name: Phase one
    status: completed
- id: R-DUP
  status: completed
  notes: 'first'
- id: R-DUP
  status: completed
  notes: 'second'
YAML

    # A file of a DIFFERENT schema that merely shares the name. Five exist in
    # the real tree; one of them does not even parse as YAML.
    mk crates/other/roadmap.yaml <<'YAML'
project: something-else
milestones:
- name: m1
  status: completed
  criteria:
    - `backticks` break the parser, as one real file in this tree does
YAML

    # --- run the scan -------------------------------------------------------
    SCAN=$(scan_roadmaps "$TD" 2>"$TD/scan.err")
    scan_rc=$?
    row 'scan of a healthy fixture tree exits 0' 0 "$scan_rc"

    verdict_of() { # verdict_of <key>
        local rec toks
        rec=$(awk -F'\t' -v k="$1" '$2==k {print; exit}' <<< "$SCAN")
        if [ -z "$rec" ]; then printf 'absent\n'; return 0; fi
        toks=$(cut -f3 <<< "$rec")
        classify_record "$TD" "$toks"
    }
    short() { # short <verdict>   -> collapse dangling:<detail> to dangling
        case "$1" in dangling:*) printf 'dangling\n' ;; *) printf '%s\n' "$1" ;; esac
    }

    printf -- '--- case table: what makes a completed claim provable ---------------\n'
    row 'resolving path citation'              cited    "$(short "$(verdict_of R-CITED)")"
    row 'citation to a path that is not there' dangling "$(short "$(verdict_of R-DANGLING)")"
    row 'citation to an EMPTY file (the touch dodge)' dangling "$(short "$(verdict_of R-EMPTY-ARTIFACT)")"
    row 'completed, prose but no citation'     uncited  "$(short "$(verdict_of R-UNCITED)")"
    row 'completed with no notes at all'       uncited  "$(short "$(verdict_of R-NO-NOTES)")"
    row 'trailing prose punctuation stripped'  cited    "$(short "$(verdict_of R-PUNCT)")"
    row 'one dead token beside a live one'     dangling "$(short "$(verdict_of R-MIXED)")"
    row 'non-empty directory citation'         cited    "$(short "$(verdict_of R-DIR-FULL)")"
    row 'empty directory citation'             dangling "$(short "$(verdict_of R-DIR-EMPTY)")"
    row 'absolute path is not a repo artifact' dangling "$(short "$(verdict_of R-ABSOLUTE)")"
    row 'path escaping the repo root'          dangling "$(short "$(verdict_of R-ESCAPE)")"

    printf -- '--- case table: the universe (what must and must NOT be gated) ------\n'
    # DISCRIMINATION. Without these the table above passes just as well for a
    # guard that flags every entry it sees.
    row 'planned entry is not gated'           absent   "$(short "$(verdict_of R-PLANNED)")"
    row 'cancelled entry is not gated'         absent   "$(short "$(verdict_of R-CANCELLED)")"
    row 'in-progress entry is not gated'       absent   "$(short "$(verdict_of R-INPROGRESS)")"
    # pmat's alias table: these three all MEAN completed, and a naive
    # `== "completed"` filter is a hole a one-word edit walks through.
    row 'alias `done` is gated'                uncited  "$(short "$(verdict_of R-ALIAS-DONE)")"
    row 'alias `CLOSED` is gated'              cited    "$(short "$(verdict_of R-ALIAS-CLOSED)")"
    row 'capitalised `Completed` is gated'     cited    "$(short "$(verdict_of R-UNDERSCORE)")"
    row 'completed SUBTASK is gated'           cited    "$(short "$(verdict_of 'sub:R-SUBTASK-PARENT/S-DONE')")"
    row 'planned subtask is not gated'         absent   "$(short "$(verdict_of 'sub:R-SUBTASK-PARENT/S-PLANNED')")"
    row 'completed PHASE is gated'             uncited  "$(short "$(verdict_of 'phase:R-PHASE-PARENT/Phase one')")"
    row 'duplicate id gets a positional key'   uncited  "$(short "$(verdict_of 'R-DUP [#2]')")"

    n_out=$(grep -c '^SCOPE out  crates/other/roadmap.yaml' "$TD/scan.err" || true)
    row 'foreign schema is reported out of scope, not skipped' 1 "$n_out"
    n_rec=$(grep -c 'crates/other/roadmap.yaml' <<< "$SCAN" || true)
    row 'foreign schema contributes no records' 0 "$n_rec"

    printf -- '--- case table: unmeasured is a FAILURE, never a skip ---------------\n'
    # A file that CLAIMS the analyser schema and then will not parse must be loud.
    # This is the branch that, taken quietly, disarms the whole guard.
    mk docs/roadmaps/broken.yaml <<'YAML'
roadmap:
- id: X
  status: completed
  notes: "unterminated
YAML
    scan_roadmaps "$TD" >/dev/null 2>"$TD/scan2.err"
    row 'in-scope file that will not parse exits 4' 4 "$?"
    n_unp=$(grep -c '^UNPARSED   docs/roadmaps/broken.yaml' "$TD/scan2.err" || true)
    row 'and it is NAMED in the diagnostics'   1 "$n_unp"
    rm -f "$TD/docs/roadmaps/broken.yaml"

    printf -- '--- case table: PR citations ---------------------------------------\n'
    # Seeded through the resolver's own cache so the row is deterministic and
    # needs no network. The live `gh` arm runs below when it can.
    PR_CACHE_DIR="$TD/prcache"; mkdir -p "$PR_CACHE_DIR"
    printf 'MERGED\n' > "$PR_CACHE_DIR/2733"
    printf 'CLOSED\n' > "$PR_CACHE_DIR/2716"
    printf 'OPEN\n'   > "$PR_CACHE_DIR/9001"
    row 'PR#<merged> resolves'        cited    "$(short "$(classify_record "$TD" 'PR#2733')")"
    row 'PR#<closed unmerged> fails'  dangling "$(short "$(classify_record "$TD" 'PR#2716')")"
    row 'PR#<open> fails'             dangling "$(short "$(classify_record "$TD" 'PR#9001')")"
    row 'bare #<merged> form resolves' cited   "$(short "$(classify_record "$TD" '#2733')")"
    row 'PR#notanumber fails'         dangling "$(short "$(classify_record "$TD" 'PR#abc')")"
    printf 'UNVERIFIABLE\n' > "$PR_CACHE_DIR/4242"
    row 'gh cannot answer -> FAIL, never a pass' dangling "$(short "$(classify_record "$TD" 'PR#4242')")"

    # The live arm. It proves only that `gh pr view --json state` still answers
    # what the cache rows above assume; the MERGED/not-MERGED DECISION is
    # already covered without it.
    rm -rf "$PR_CACHE_DIR"
    PR_CACHE_DIR="$TD/prcache2"; mkdir -p "$PR_CACHE_DIR"
    if gh auth status >/dev/null 2>&1; then
        row 'live: PR #2733 is MERGED'          cited    "$(short "$(classify_record "$TD" 'PR#2733')")"
        row 'live: PR #2716 is CLOSED unmerged' dangling "$(short "$(classify_record "$TD" 'PR#2716')")"
    else
        printf '  REPORT live gh arm did not run (gh unauthenticated). The MERGED vs\n'
        printf '         not-MERGED decision is covered by the cache rows above; only\n'
        printf '         the gh invocation itself is unexercised here.\n'
    fi
    PR_CACHE_DIR=""

    printf '  %s case(s), %s failure(s)\n' "$t" "$f"
    if [ "$f" -ne 0 ]; then printf 'SELFTEST FAILED\n'; exit 1; fi
    printf 'SELFTEST PASSED — no roadmap in this tree was scanned; run with no\n'
    printf '                  arguments for that.\n'
    exit 0
fi

# ------------------------------------------------------------------- gate --
printf -- '=== a completed roadmap status cites something (PERF-044) ===========\n'

SCAN_ERR=$(mktemp) || exit 2
CURRENT=$(mktemp) || exit 2
trap 'rm -f "${SCAN_ERR:?}" "${CURRENT:?}"' EXIT
RECORDS=$(scan_roadmaps "$REPO_ROOT" 2>"$SCAN_ERR")
SCAN_RC=$?

grep -E '^(SCOPE out|UNPARSED) ' "$SCAN_ERR" || true
N_SCOPE=$(sed -n 's/^COUNT //p' "$SCAN_ERR" | tail -1)
[ -n "$N_SCOPE" ] || N_SCOPE=0

if [ "$SCAN_RC" -ne 0 ]; then
    grep -E '^FATAL' "$SCAN_ERR" || true
    printf 'FAIL  the roadmap scan did not complete (rc=%s). An unparsed roadmap is\n' "$SCAN_RC"
    printf '      UNMEASURED, and unmeasured is never "no unproven claims".\n'
    exit 1
fi

# VACUITY. A universe that collapsed scans clean and reads as a pass — the
# exact failure this epic keeps finding. Fix the scan, never this number.
if [ "$N_SCOPE" -lt "$MIN_ROADMAPS" ]; then
    printf 'FAIL  universe collapsed to %s in-scope roadmap file(s), expected %s+.\n' \
        "$N_SCOPE" "$MIN_ROADMAPS"
    printf '      The scan is broken, not the tree.\n'
    exit 1
fi
printf 'universe: %s analyser work-contract roadmap file(s)\n' "$N_SCOPE"

BASELINE="$REPO_ROOT/$BASELINE_REL"

if [ "${1:-}" = "--update" ]; then
    {
        printf '# Roadmap entries claiming `status: completed` that cite nothing, as of\n'
        printf '# the day check_roadmap_completion_is_cited.sh was written. RECORDED,\n'
        printf '# NOT BLESSED — PERF-004 is in this file and is known to be FALSE: its\n'
        printf '# three artifacts are absent from main and its PR #2716 is CLOSED\n'
        printf '# unmerged. This file is the census of an unproven backlog.\n#\n'
        printf '# THE RATCHET: it may only SHRINK. An entry leaves by citing a\n'
        printf '# `proof:<path>` or `proof:PR#<n>` in its notes, or by having its\n'
        printf '# status corrected. It is compared against origin/main by\n'
        printf '# check_roadmap_completion_is_cited.sh and by\n'
        printf '# check_baseline_ratchets.sh, so an APPEND IS REFUSED — a new\n'
        printf '# completed claim cannot be laundered in the commit that makes it.\n#\n'
        printf '# Keyed by <file><TAB><entry id>, never by line number: the analyser reflows\n'
        printf '# these files (13516 -> 12804 lines in one unrelated edit) and a\n'
        printf '# line-keyed baseline would be invalidated wholesale.\n#\n'
        printf '# A DANGLING citation is never baselined — see the guard header.\n'
        printf '# Regenerate with: bash scripts/check_roadmap_completion_is_cited.sh --update\n'
        while IFS= read -r rec; do
            [ -n "$rec" ] || continue
            if [ "$(classify_record "$REPO_ROOT" "$(cut -f3 <<< "$rec")")" = uncited ]; then
                cut -f1,2 <<< "$rec"
            fi
        done <<< "$RECORDS" | LC_ALL=C sort -u
    } > "$BASELINE"
    printf 'baseline written: %s uncited completed claim(s)\n' \
        "$(grep -cvE '^[[:space:]]*(#|$)' "$BASELINE" || true)"
    exit 0
fi

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL  %s is missing. Growth is UNMEASURED without it.\n' "$BASELINE_REL"
    printf '      Run --update once to establish it, or retire this guard in the\n'
    printf '      same commit if it is genuinely being retired.\n'
    exit 1
fi

rc=0
n_total=0
n_cited=0
n_known=0
n_new=0
n_dang=0
while IFS= read -r rec; do
    [ -n "$rec" ] || continue
    n_total=$((n_total + 1))
    loc=$(cut -f1,2 <<< "$rec")
    verdict=$(classify_record "$REPO_ROOT" "$(cut -f3 <<< "$rec")")
    case "$verdict" in
        cited)
            n_cited=$((n_cited + 1)) ;;
        dangling:*)
            printf 'FAIL  %s\n      cites a proof that does not resolve: %s\n' \
                "$(tr '\t' ' ' <<< "$loc")" "${verdict#dangling:}"
            n_dang=$((n_dang + 1))
            rc=1 ;;
        uncited)
            printf '%s\n' "$loc" >> "$CURRENT"
            if grep -qxF "$loc" "$BASELINE" 2>/dev/null; then
                n_known=$((n_known + 1))
            else
                printf 'FAIL  %s\n      claims completed and cites nothing, and is NOT in the\n' \
                    "$(tr '\t' ' ' <<< "$loc")"
                printf '      frozen baseline — so it is a NEW unprovable claim.\n'
                n_new=$((n_new + 1))
                rc=1
            fi ;;
    esac
done <<< "$RECORDS"

printf 'completed claims: %s   cited: %s   baselined (must shrink): %s   NEW: %s   dangling: %s\n' \
    "$n_total" "$n_cited" "$n_known" "$n_new" "$n_dang"

# Stale baseline entries: an id that no longer claims completed must be pruned,
# or the ratchet silently re-admits a claim under that id later.
stale=0
while IFS= read -r loc; do
    [ -n "$loc" ] || continue
    case "$loc" in '#'*) continue ;; esac
    grep -qxF "$loc" "$CURRENT" 2>/dev/null || stale=$((stale + 1))
done < "$BASELINE"
if [ "$stale" -gt 0 ]; then
    printf 'REPORT %s baselined entr(ies) no longer claim completed-and-uncited.\n' "$stale"
    printf '       Prune with --update, or the ratchet re-admits a claim there later.\n'
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet: a commit that appends one baseline
# line AND flips one entry to completed satisfies it. Twelve guards in this
# repository called themselves shrink-only and 12 of 12 accepted exactly that
# commit. So growth is compared against merge-base(HEAD, origin/main), with the
# origin/main tip as the shallow-checkout fallback — a ref this branch cannot
# rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1

# THE ONE-COMMIT BOOTSTRAP is handled by the library, not here. The commit
# introducing a baseline cannot have it at either protected ref, so the library
# reports BOOTSTRAP -- loudly, once, and only while the file is genuinely new.
baseline_ratchet_check "$REPO_ROOT" "$BASELINE_REL" set || rc=1

printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no NEW completed claim cites nothing, and every proof: resolves.\n'
else
    printf 'FAIL  a `status: completed` is a claim, and this one has nothing behind\n'
    printf '      it. Put a `proof:<repo/path>` or `proof:PR#<n>` in the entry notes\n'
    printf '      — the path must exist and be non-empty, the PR must be MERGED. If\n'
    printf '      the work did not actually land, the fix is the STATUS, not the\n'
    printf '      citation: PERF-004 is why this guard exists.\n'
fi
exit "$rc"
