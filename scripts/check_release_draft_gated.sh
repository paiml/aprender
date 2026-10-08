#!/usr/bin/env bash
# check_release_draft_gated.sh -- the case table for #4690: no public GitHub release before its
# assets, its clean-room and its preflight. Two halves, both counted:
#
# STRUCTURE, over the whole autopilot (so a step this table does not run is still judged): the only
# thing that may take a release public is publish_release() -- no draft set by value in any spelling
# (`--draft=X`, `-f draft=X`, a JSON body, X a literal or a variable) and no `gh api` write to a
# release anywhere else (an edit of the notes is allowed); every `gh release create` is a
# --draft; publish_release is called once, from the publish step; and `publish` comes after
# cleanroom, assets and preflight both in STEPS and in the file.
#
# BEHAVIOUR: it EXTRACTS the autopilot's own tag, cleanroom, assets, preflight and publish step
# bodies (and publish_release()) and RUNS them against a stub `gh` that records every call and
# models the release: created draft or public, assets built only by a binary-release run (a
# dispatch, or `release: published` when the release goes public), `gh api` writes, failing edits,
# a missing release (rc 1, as gh does) and `--jq` filters (run through jq over a two-job run, so a
# filter that drops the job name reads the wrong job). A row may run several invocations in turn
# ("a b+c" = run steps a b, then a resumed run of c, sharing state). Each row reads the call log:
#   draft              the release never went public
#   public-gated       it went public after a green clean-room read, a PASS preflight and assets rc 0
#   public-EARLY       it went public before one of those three
# followed by `stop` (a run died) or `ran`. Then each mutant is applied to a copy of the autopilot
# and must turn at least one row WRONG.
#
# THE CASCADE (D4, P7 WIRE). cascade-publish.sh re-runs check_publish_preflight.sh before every pass
# (F-9), and under the standing release policy that preflight's R7 judges CRUX smoke: it must read the
# receipts the preflight step reads, or it refuses after the release is public. The cascade rows run
# the autopilot's cascade step against the REAL call sites the receipts travel through, extracted and
# never re-implemented: cascade-drain.sh's line that starts each cascade-publish.sh pass, and
# cascade-publish.sh's gate block. The preflight they reach is the preflight step's stub, red unless it
# is handed the T-1 receipts exactly when the policy covers the release. A row that ran the cascade
# appends `crates-published` (a pass got past the preflight) or `crates-refused`. Mutants of the drain
# script and of cascade-publish.sh must turn a row WRONG too, as the autopilot's do. Because those rows
# run two extracted lines, one more STRUCTURE row judges both files whole: nothing in either clears
# the environment or unsets, exports, declares or assigns MODEL_LADDER_CRUX_DIR or CRUX_CERT.
#
# MODE. guard_tree runs every scripts/check_*.sh, this one included. REPORT by default (L31: a new
# check blocks only after three green nights): a wrong row or a surviving mutant prints a REPORT line
# and exits 0. RELEASE_DRAFT_GATED_ENFORCE=1 makes it exit 1. ENV (rc 2) is rc 2 in both modes.
# Exit 0 = every row as expected and every mutant killed (or report mode) · 1 = a row or a mutant
# landed wrong under ENFORCE · 2 = the autopilot has none of the step bodies this table runs, or jq
# is absent.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUTOPILOT="${1:-$HERE/release/autopilot.sh}"
DRAIN="$HERE/cascade-drain.sh" CASCADE="$HERE/cascade-publish.sh"
command -v jq > /dev/null || { printf 'ENV   jq is not on PATH: the stub cannot apply --jq filters\n' >&2; exit 2; }
T=$(mktemp -d) || exit 2
trap 'rm -rf "${T:?}"' EXIT

# finish BAD: the verdict in the current mode (see MODE above).
finish() {
  [ "$1" = 0 ] && exit 0
  [ "${RELEASE_DRAFT_GATED_ENFORCE:-0}" = 1 ] && exit 1
  printf 'REPORT: a row or a mutant landed wrong -- report mode (L31: blocks only after three green nights; RELEASE_DRAFT_GATED_ENFORCE=1 enforces), #4690\n'
  exit 0
}
mkdir -p "$T/bin"

# The stub gh. State lives in $W/state/{release,built,dispatched,published-event}; every call appends
# to $W/calls.
cat > "$T/bin/gh" <<'STUB'
#!/usr/bin/env bash
S="$W/state"; a=" $* "
log() { printf '%s\n' "$*" >> "$W/calls"; }
public() { # the release goes public: `release: published` fires binary-release, which builds the assets
  log PUBLIC "$1"; echo public > "$S/release"; echo 1 > "$S/built"; echo 1 > "$S/published-event"
}
jqarg() { local p="" x; for x in "$@"; do case "$p" in --jq|-q) printf '%s' "$x"; return ;; esac; p=$x; done; }
case "$1 $2" in
  "release create")
    case "$a" in *" --draft "*) log CREATE draft; echo draft > "$S/release" ;; *) public create ;; esac ;;
  "release edit")
    [ -s "$S/release" ] || { echo "release not found" >&2; exit 1; }
    case "$a" in
      *" --draft=false "*) [ "$FX_EDIT" = fail ] && { log EDIT-FAILED; echo "HTTP 502" >&2; exit 1; }; public edit ;;
      *) log EDIT "$*" ;;
    esac ;;
  "release view")
    [ -s "$S/release" ] || { echo "release not found" >&2; exit 1; }
    case "$a" in *" isDraft "*) [ "$(cat "$S/release")" = draft ] && echo true || echo false ;;
                 *) echo "https://example.invalid/release" ;; esac ;;
  "workflow run")
    log DISPATCH "$3"
    case "$3" in binary-release.yml) echo 1 > "$S/dispatched"; echo 1 > "$S/built" ;; esac ;;
  "run list")
    case "$a" in
      *" clean-room.yml "*) echo 111 ;;
      *" b2-gpu.yml "*) echo 222 ;;
      *" binary-release.yml --event release "*) [ -s "$S/published-event" ] && echo 333 ;;
      *" binary-release.yml --event workflow_dispatch "*) [ -s "$S/dispatched" ] && echo 333 ;;
    esac ;;
  "run view")
    case "$3$a" in
      # another job of the same run is green: a filter that drops the job name reads that one
      111*" jobs "*) log CLEANROOM-READ "$FX_CLEANROOM"
        printf '{"jobs":[{"name":"clean-room (other)","status":"completed","conclusion":"success"},{"name":"clean-room (aprender)","status":"completed","conclusion":"%s"}]}' \
          "$FX_CLEANROOM" | jq -r "$(jqarg "$@")" ;;
      # no id: gh falls back to picking a recent run, which is some other run's verdict, not ours
      " run view "*" jobs "*) echo success ;;
      222*" --log "*) echo "tested-sha: deadbeef" ;;
      222*) echo "completed success" ;;
      333*" status "*) echo completed ;;
      333*" conclusion "*) echo success ;;
    esac ;;
  "api "*)
    case "$a" in
      *"draft=false"*|*" -X PATCH "*"/releases"*|*" --method PATCH "*"/releases"*) public api ;;
      *) log API "$2" ;;
    esac ;;
esac
exit 0
STUB
chmod +x "$T/bin/gh"

# structure <autopilot> -> number of WRONG structural rows; prints each
structure() {
  local ap=$1 wrong=0 code hits calls order
  srow() { if [ -z "$2" ]; then printf '  ok    %-48s\n' "$1"
           else printf '  WRONG %-48s %s\n' "$1" "$2"; wrong=$((wrong + 1)); fi; }
  # every non-comment line outside publish_release(), numbered, with `\` continuations joined
  code=$(awk '/^publish_release\(\) \{/,/^\}/ { next } /^[[:space:]]*#/ { next }
    { l = (b == "" ? NR ": " : b) $0; if (sub(/\\$/, "", l)) { b = l; next }; b = ""; print l }' "$ap")
  # draft set by value in any spelling (--draft=X, -f draft=X, a JSON "draft": X), or any gh api write
  # to a release -- whatever the value, so a variable cannot hide it; a plain `--draft` is a draft
  hits=$( { grep -E -- '--draft=|(^|[^-[:alnum:]_])"?draft"?[[:space:]]*[=:]' <<< "$code"
            grep -E 'gh api' <<< "$code" | grep -F releases \
              | grep -E '(^|[[:space:]])(-X|--method|-f|-F|--field|--raw-field|--input)([[:space:]]|=)'; } \
          | head -n 3 | tr '\n' ' ')
  srow structure_public_only_inside_publish_release "$hits"
  hits=$(grep -E 'gh release create' <<< "$code" | grep -v -- '--draft' | head -n 3 | tr '\n' ' ')
  srow structure_every_release_create_is_a_draft "$hits"
  calls=$(awk '/^publish_release\(\) \{/ { next } /^[[:space:]]*#/ { next }
    $0 == "if run_step publish; then" { inp = 1 }
    /(^|[^_[:alnum:]])publish_release([^_[:alnum:]]|$)/ { n++; if (inp) k++ }
    inp && /^fi$/ { inp = 0 }
    END { printf "%d %d", n, k }' "$ap")
  [ "$calls" = "1 1" ] && calls="" || calls="calls (total, in the publish step) = $calls, want 1 1"
  srow structure_publish_release_called_once_from_publish "$calls"
  order=$(awk '
    /^STEPS=\(/ { s = $0; gsub(/^STEPS=\(|\).*$/, "", s); n = split(s, w, " "); for (i = 1; i <= n; i++) p[w[i]] = i }
    /^if run_step [a-z]+; then$/ { f[$3 == "" ? "" : substr($3, 1, length($3) - 1)] = NR }
    END {
      ok = p["cleanroom"] && p["assets"] && p["preflight"] && p["publish"] > p["cleanroom"] && p["publish"] > p["assets"] && p["publish"] > p["preflight"]
      ok = ok && f["cleanroom"] && f["assets"] && f["preflight"] && f["publish"] > f["cleanroom"] && f["publish"] > f["assets"] && f["publish"] > f["preflight"]
      if (!ok) printf "STEPS cleanroom=%d assets=%d preflight=%d publish=%d; file lines %d %d %d %d", p["cleanroom"], p["assets"], p["preflight"], p["publish"], f["cleanroom"], f["assets"], f["preflight"], f["publish"]
    }' "$ap")
  srow structure_publish_after_its_gates "$order"
  # D4: the receipts reach the preflight only if nothing between resets them. The cascade rows run two
  # extracted lines, so this judges the WHOLE drain and cascade-publish.sh: no `env -i` (or `env -`),
  # and no unset, export, declaration, assignment or `env -u` of MODEL_LADDER_CRUX_DIR or CRUX_CERT
  hits=$(awk 'FNR == 1 { f = FILENAME; sub(/.*\//, "", f) } /^[[:space:]]*#/ { next }
    /(^|[^-_[:alnum:]])e[n]v[[:space:]]+(-[[:alpha:]]*i|--ignore-environment|-)([[:space:]]|$)/ \
    || /(^|[^_[:alnum:]])(unset|export|readonly|declare|typeset|local)[[:space:]][^#]*(MODEL_LADDER_CRUX_DIR|CRUX_CERT)/ \
    || /(^|[^_[:alnum:]{$])(MODEL_LADDER_CRUX_DIR|CRUX_CERT)\+?=/ \
    || /(^|[^-_[:alnum:]])e[n]v[[:space:]][^#]*(-u[[:space:]]*|--unset[[:space:]=])(MODEL_LADDER_CRUX_DIR|CRUX_CERT)/ { print f ":" FNR }' \
    "$DRAIN" "$CASCADE" | head -n 3 | tr '\n' ' ')
  srow structure_drain_and_cascade_keep_the_crux_vars "$hits"
  return "$wrong"
}

# extract <autopilot> -> the harness body: publish_release() if present, then each step body in order
extract() {
  local ap=$1 s
  awk '/^publish_release\(\) \{/,/^\}/' "$ap"
  for s in tag cleanroom assets preflight publish cascade; do
    awk -v h="if run_step $s; then" '$0 == h { on = 1 } on { print } on && /^fi$/ { exit }' "$ap"
  done
}

# drain_pass <drain> -> the drain's line that starts each cascade-publish.sh pass, continuations joined
drain_pass() {
  awk '/^[[:space:]]*#/ { next }
    c { print; if (!/\\$/) exit; next }
    /bash scripts\/cascade-publish\.sh/ { print; c = 1; if (!/\\$/) exit }' "$1"
}
# gate_block <cascade> -> cascade-publish.sh's gate: from its THE GATE comment to the esac that closes it
gate_block() { awk '/^# THE GATE \(F-9/ { on = 1 } on { print } on && /^esac$/ { exit }' "$1"; }

# run <autopilot> <invocations> <FX vars...> -> "<verdict> <stop|ran>"
# <invocations>: step lists joined by "+", each run as its own resumed autopilot over shared state.
# FX_ASSETS: check_release_assets.sh rc once built (missing build is always rc 1) · FX_CLEANROOM: the
# job conclusion · FX_PREFLIGHT: preflight rc · FX_EDIT: ok | fail (the publishing edit) · FX_PRE:
# pre-state for a run that starts late (draft-built | public-built | gone: built, no release) ·
# FX_CRUN: cleanroom-run-id file (1 = present) · FX_PASS: preflight-pass line (ok | stale | prior |
# none; prior = an earlier run's PASS for this tag and commit, with no PREFLIGHT line of its own) ·
# FX_POLICY: 1 = the standing release policy covers the release. Reads $DRAIN and $CASCADE.
run() {
  local ap=$1 inv=$2 W on
  W=$(mktemp -d "$T/w.XXXXXX") || return 2
  mkdir -p "$W/state" "$W/ap" "$W/scripts/release"
  : > "$W/calls"
  printf 'notes\n' > "$W/ap/release_notes.md"
  local FX_ASSETS=0 FX_CLEANROOM=success FX_PREFLIGHT=0 FX_EDIT=ok FX_PRE=none FX_CRUN=1 FX_PASS=ok FX_POLICY=0 kv
  shift 2; for kv in "$@"; do local "${kv?}"; done
  case "$FX_PRE" in
    draft-built)  echo draft > "$W/state/release"; echo 1 > "$W/state/built"; echo 1 > "$W/state/dispatched" ;;
    public-built) echo public > "$W/state/release"; echo 1 > "$W/state/built" ;;
    gone)         echo 1 > "$W/state/built"; echo 1 > "$W/state/dispatched" ;;
  esac
  [ "$FX_CRUN" = 1 ] && [ "$FX_PRE" != none ] && echo 111 > "$W/ap/cleanroom-run-id"
  case "$FX_PASS" in
    # a run that starts at `publish` inherits an earlier run's preflight: that run's PASS is in the log
    ok)    [ "$FX_PRE" != none ] && { echo "PASS v0.0.0 deadbeef" > "$W/ap/preflight-pass"; echo "PREFLIGHT rc=0" >> "$W/calls"; } ;;
    stale) echo "PASS v0.0.0 cafef00d" > "$W/ap/preflight-pass" ;;
    prior) echo "PASS v0.0.0 deadbeef" > "$W/ap/preflight-pass" ;;
  esac
  printf '#!/usr/bin/env bash\nif [ -s %q ]; then rc=%s; else rc=1; fi\necho "ASSETS-CHECK rc=$rc" >> %q\nexit $rc\n' \
    "$W/state/built" "$FX_ASSETS" "$W/calls" > "$W/scripts/check_release_assets.sh"
  # the preflight stub goes red unless it is handed the T-1 CRUX receipts and the bump's certification
  # exactly when the standing release policy covers the release (FX_POLICY), and nothing otherwise
  printf '#!/usr/bin/env bash\nrc=%s\nif [ %q = 1 ]; then [ "${MODEL_LADDER_CRUX_DIR:-}" = %q ] && [ "${CRUX_CERT:-}" = %q ] || rc=1\nelse [ -z "${MODEL_LADDER_CRUX_DIR:-}${CRUX_CERT:-}" ] || rc=1; fi\necho "PREFLIGHT rc=$rc" >> %q\nexit $rc\n' \
    "$FX_PREFLIGHT" "$FX_POLICY" "$W/ap/models-t1" "$W/evidence/crux/0.0.0/prompt-certification.json" "$W/calls" > "$W/scripts/check_publish_preflight.sh"
  # the drain: the real drain's pass line, so whatever it strips from the environment is stripped here
  { printf '#!/usr/bin/env bash\nTARGET=0.0.0\ncd %q || exit 2\necho DRAIN >> %q\n' "$W" "$W/calls"
    drain_pass "$DRAIN"; } > "$W/scripts/cascade-drain.sh"
  # the cascade: the real gate block (clean-room stubbed green, the preflight is the stub above); a
  # pass that gets past it publishes, and --check reports a crate behind until one has
  { printf '#!/usr/bin/env bash\nMODE="${1:-publish}" REPO_ROOT=%q TARGET_VERSION=0.0.0\n' "$W"
    cat <<'STUB'
clean_room_gate() { return 0; }
STUB
    gate_block "$CASCADE"
    printf 'case "$MODE" in --check|--order-check) grep -qx "CRATES published" %q || echo "aprender-core 0.0.0-rc (want 0.0.0)"; exit 0 ;; esac\n' "$W/calls"
    printf 'echo "CRATES published" >> %q\n' "$W/calls"
  } > "$W/scripts/cascade-publish.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$W/scripts/release/tag_coverage_gate.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$W/scripts/release/rc_publish_gate.sh"
  {
    printf 'set -uo pipefail\ncd %q || exit 2\n' "$W"
    printf 'REPO=paiml/aprender INFRA=paiml/infra V=0.0.0 T=v0.0.0 MC=deadbeef AP_POLICY=%s\n' "$FX_POLICY"
    printf 'AP=%q LOG=%q STATUS=%q WT=%q\n' "$W/ap" "$W/log" "$W/status" "$W"
    printf 'say() { printf "SAY %%s\\n" "$*" >> "$LOG"; }\n'
    printf 'die() { printf "STOP %%s\\n" "$*" >> %q; exit 1; }\n' "$W/calls"
    printf 'sleep() { :; }\n'
    printf 'cut_tag() { printf "TAG %%s\\n" "$2" >> %q; }\n' "$W/calls"
    cat <<'STUB'
ap_known_failures() { :; }
STUB
    printf 'git() { [ "$1" = rev-parse ] && return 1; return 0; }\n'
    printf 'run_step() { case " $ON " in *" $1 "*) return 0 ;; esac; return 1; }\n'
    extract "$ap"
  } > "$W/harness.sh"
  while IFS= read -r on; do
    PATH="$T/bin:$PATH" W="$W" ON="$on" FX_CLEANROOM="$FX_CLEANROOM" FX_EDIT="$FX_EDIT" bash "$W/harness.sh" > /dev/null 2>&1
  done <<< "$(tr '+' '\n' <<< "$inv")"
  awk '
    $1 == "CLEANROOM-READ" && $2 == "success" { c = 1 }
    $1 == "PREFLIGHT" && $2 == "rc=0" { p = 1 }
    $1 == "ASSETS-CHECK" && $2 == "rc=0" { a = 1 }
    $1 == "PUBLIC" && v == "" { v = (c && p && a) ? "public-gated" : "public-EARLY" }
    $1 == "STOP" { s = "stop" }
    /^DRAIN$/ { d = 1 }
    /^CRATES published$/ { k = 1 }
    END { printf "%s %s%s", (v == "" ? "draft" : v), (s == "" ? "ran" : s), (d ? (k ? " crates-published" : " crates-refused") : "") }' "$W/calls"
}

ALL="tag cleanroom assets preflight publish"
CASES="all_green_is_public_only_after_the_gates|$ALL||public-gated ran
the_tag_step_alone_leaves_a_draft|tag||draft ran
a_missing_asset_stops_before_publish|$ALL|FX_ASSETS=1|draft stop
an_unreadable_release_stops_before_publish|$ALL|FX_ASSETS=2|draft stop
a_red_clean_room_stops_before_publish|$ALL|FX_CLEANROOM=failure|draft stop
a_cancelled_clean_room_stops_before_publish|$ALL|FX_CLEANROOM=cancelled|draft stop
a_red_preflight_stops_before_publish|$ALL|FX_PREFLIGHT=1|draft stop
a_red_preflight_then_a_resume_at_publish_stays_draft|tag cleanroom assets preflight+publish|FX_PREFLIGHT=1 FX_PASS=prior|draft stop
publish_alone_all_green_publishes|publish|FX_PRE=draft-built|public-gated ran
publish_alone_refuses_a_missing_asset|publish|FX_PRE=draft-built FX_ASSETS=1|draft stop
publish_alone_refuses_an_unreadable_release|publish|FX_PRE=draft-built FX_ASSETS=2|draft stop
publish_alone_refuses_a_red_clean_room|publish|FX_PRE=draft-built FX_CLEANROOM=failure|draft stop
publish_alone_refuses_a_cancelled_clean_room|publish|FX_PRE=draft-built FX_CLEANROOM=cancelled|draft stop
publish_alone_refuses_an_empty_clean_room_conclusion|publish|FX_PRE=draft-built FX_CLEANROOM=|draft stop
publish_alone_refuses_no_clean_room_run|publish|FX_PRE=draft-built FX_CRUN=0|draft stop
publish_alone_refuses_no_preflight_pass|publish|FX_PRE=draft-built FX_PASS=none|draft stop
publish_alone_refuses_another_commits_pass|publish|FX_PRE=draft-built FX_PASS=stale|draft stop
publish_alone_refuses_a_release_already_public|publish|FX_PRE=public-built|draft stop
publish_alone_refuses_a_missing_release|publish|FX_PRE=gone|draft stop
a_failed_publishing_edit_stops|publish|FX_PRE=draft-built FX_EDIT=fail|draft stop
policy_preflight_judges_the_t1_crux_receipts|preflight|FX_POLICY=1|draft ran
policy_cascade_preflight_reads_the_t1_crux_receipts|tag cleanroom assets preflight publish cascade|FX_POLICY=1|public-gated ran crates-published
policy_cascade_resumed_alone_reads_the_t1_crux_receipts|cascade|FX_PRE=public-built FX_POLICY=1|draft ran crates-published
no_policy_cascade_preflight_gets_no_crux_env|tag cleanroom assets preflight publish cascade||public-gated ran crates-published
a_red_cascade_preflight_publishes_no_crate|cascade|FX_PRE=public-built FX_PREFLIGHT=1|draft stop crates-refused"

table() { # table <autopilot> -> number of WRONG rows; prints each row
  local ap=$1 wrong=0 name on fx want got
  structure "$ap" || wrong=$((wrong + $?))
  while IFS='|' read -r name on fx want; do
    [ -n "$name" ] || continue
    # shellcheck disable=SC2086 # fx is a word list of NAME=value pairs by construction
    got=$(run "$ap" "$on" $fx)
    if [ "$got" = "$want" ]; then printf '  ok    %-48s %s\n' "$name" "$got"
    else printf '  WRONG %-48s got [%s] want [%s]\n' "$name" "$got" "$want"; wrong=$((wrong + 1)); fi
  done <<< "$CASES"
  return "$wrong"
}

grep -q '^if run_step tag; then$' "$AUTOPILOT" \
  || { printf 'ENV   %s has no tag step body: this table would judge nothing\n' "$AUTOPILOT" >&2; exit 2; }
n=$(awk '/^[[:space:]]*#/ { next } /bash scripts\/cascade-publish\.sh/ { n++ } END { print n + 0 }' "$DRAIN")
if [ "$n" != 1 ]; then
  printf 'ENV   %s starts cascade-publish.sh from %s line(s), want 1: the cascade rows would judge one of them\n' "$DRAIN" "$n" >&2
  exit 2
fi
gate=$(gate_block "$CASCADE")
if ! grep -q 'scripts/check_publish_preflight\.sh' <<<"$gate"; then
  printf 'ENV   %s has no THE GATE block that runs check_publish_preflight.sh: the cascade rows would judge nothing\n' "$CASCADE" >&2
  exit 2
fi
bad=0
table "$AUTOPILOT" || bad=1
rows=$(grep -c '|' <<< "$CASES")
[ "$rows" -ge 25 ] || { printf 'VACUOUS %s row(s), fewer than the 25 declared\n' "$rows"; bad=1; }
[ "${1:-}" = "" ] || finish "$bad"   # an explicit autopilot (e.g. origin/main's) runs the table only

# mutant [drain|cascade] <name> <sed expression>: applied to a copy of the autopilot (or of the drain,
# or of cascade-publish.sh), the table must go WRONG. nearmiss, same arguments: a change that keeps the
# receipts flowing, which every row must still hold (the must-not-match half of the structural case table)
killed=0 total=0 held=0 near=0
mutant() {
  local of=autopilot src m out rc ap="$AUTOPILOT" dr="$DRAIN" ca="$CASCADE"
  local kind="${KIND:-mutant}"
  case "$1" in drain|cascade) of=$1; shift ;; esac
  case "$of" in autopilot) src="$AUTOPILOT" ;; drain) src="$DRAIN" ;; cascade) src="$CASCADE" ;; esac
  m="$T/m-$1.sh"
  if [ "$kind" = mutant ]; then total=$((total + 1)); else near=$((near + 1)); fi
  sed "$2" "$src" > "$m"
  if cmp -s "$src" "$m"; then printf '  INCONCLUSIVE %s %s changed nothing\n' "$kind" "$1"; bad=1; return; fi
  bash -n "$m" || { printf '  BROKEN   %s %s does not parse\n' "$kind" "$1"; bad=1; return; }
  case "$of" in autopilot) ap="$m" ;; drain) dr="$m" ;; cascade) ca="$m" ;; esac
  if out=$(DRAIN="$dr" CASCADE="$ca" table "$ap"); then rc=0; else rc=$?; fi
  if [ "$kind" = nearmiss ]; then
    if [ "$rc" = 0 ]; then held=$((held + 1)); printf '  held     near-miss %s\n' "$1"
    else printf '  WRONG    near-miss %s (%s)\n' "$1" "$(grep -m1 -o 'WRONG [a-z0-9_]*' <<< "$out")"; bad=1; fi
  elif [ "$rc" = 0 ]; then printf '  SURVIVED mutant %s\n' "$1"; bad=1
  else killed=$((killed + 1)); printf '  killed   mutant %s (%s)\n' "$1" "$(grep -m1 -o 'WRONG [a-z0-9_]*' <<< "$out")"; fi
}
nearmiss() { local KIND=nearmiss; mutant "$@"; }
mutant create-public       '/gh release create "\$T"/s/ --draft//'
mutant no-asset-dispatch   '/gh workflow run binary-release.yml/s/^  gh workflow run binary-release.yml.*/  :/'
mutant assets-await-publish 's/--workflow binary-release.yml --event workflow_dispatch/--workflow binary-release.yml --event release/'
mutant crun-unchecked      's/^    \[ -n "\$crun" \] || die/    true || die/'
mutant cleanroom-unchecked 's/^    \[ "\$jc" = success \] || die/    true || die/'
mutant cleanroom-any-job   '/^    jc=\$(gh run view "\$crun"/s/ | select(.name=="clean-room (aprender)")//'
mutant preflight-unchecked 's/^    grep -qxF "PASS \$t \$mc"/    true || grep -qxF "PASS $t $mc"/'
mutant pass-any-commit     's/grep -qxF "PASS \$t \$mc"/grep -qF "PASS $t"/'
mutant pass-not-written    's/^  printf .PASS %s %s\\n. "\$T" "\$MC" > "\$AP\/preflight-pass"/  :/'
mutant pass-not-truncated  's/^  : > "\$AP\/preflight-pass".*/  :/'
mutant assets-unchecked    's/^    \[ "\$rc" -eq 0 \] || die "check_release_assets.sh \$t/    true || die "check_release_assets.sh $t/'
mutant assets-env-is-ok    's/^    \[ "\$rc" -eq 0 \] || die "check_release_assets.sh \$t/    [ "$rc" -ne 1 ] || die "check_release_assets.sh $t/'
mutant draft-unchecked     's/^    \[ "\$d" = true \] || die/    true || die/'
mutant edit-failure-ignored '/gh release edit "\$t" --repo "\$REPO" --draft=false/s/|| die "publishing the draft \$t failed"/|| true/'
mutant api-public-in-tag   '/^  say "DRAFTED/i\  gh api -X PATCH "repos/$REPO/releases/1" -f draft=false >> "$LOG" 2>\&1'
mutant edit-in-dryrun      '/^if run_step dryrun; then$/a\  gh release edit "$T" --repo "$REPO" --draft=false >> "$LOG" 2>\&1'
mutant publish-in-preflight '/^  printf .PASS %s %s\\n. "\$T" "\$MC" > "\$AP\/preflight-pass"/i\  publish_release "$T" "$MC"'
mutant steps-publish-first '/^STEPS=/s/preflight publish/publish preflight/'
mutant headers-swapped     's/^if run_step preflight; then$/if run_step PUBX; then/; s/^if run_step publish; then$/if run_step preflight; then/; s/^if run_step PUBX; then$/if run_step publish; then/'
mutant draft-by-variable   '/^if run_step dryrun; then$/a\  F=false; gh release edit "$T" --repo "$REPO" --draft=$F >> "$LOG" 2>\&1'
mutant publish-by-variable '/^if run_step dryrun; then$/a\  PRF=publish_release; $PRF "$T" "$MC"'
mutant api-field-variable  '/^if run_step cascade; then$/a\  gh api "repos/$REPO/releases/$RID" -F draft=$F > /dev/null'
mutant api-input-body      '/^if run_step cascade; then$/a\  gh api --method PATCH "repos/$REPO/releases/$RID" --input "$AP/body.json" > /dev/null'
mutant cleanroom-not-failure 's/^    \[ "\$jc" = success \] || die/    [ "$jc" != failure ] || die/'
PRE='/^if run_step preflight; then$/,/^fi$/' CAS='/^if run_step cascade; then$/,/^fi$/'
mutant policy-crux-dir-dropped "$PRE"'s/^    MODEL_LADDER_CRUX_DIR="\$AP\/models-t1" CRUX_CERT=/    CRUX_CERT=/'
mutant policy-cert-not-the-bumps "$PRE"'s/CRUX_CERT="\$WT\/evidence\/crux\/\$V\/prompt-certification.json"/CRUX_CERT="\$AP\/models-t1\/prompt-certification.json"/'
mutant policy-env-always "$PRE"'s/^  if \[ "\$AP_POLICY" = 1 \]; then$/  if true; then/'
mutant cascade-crux-dir-dropped "$CAS"'s/^    MODEL_LADDER_CRUX_DIR="\$AP\/models-t1" CRUX_CERT=/    CRUX_CERT=/'
mutant cascade-cert-not-the-bumps "$CAS"'s/CRUX_CERT="\$WT\/evidence\/crux\/\$V\/prompt-certification.json"/CRUX_CERT="\$AP\/models-t1\/prompt-certification.json"/'
mutant cascade-env-always "$CAS"'s/^  if \[ "\$AP_POLICY" = 1 \]; then$/  if true; then/'
mutant cascade-env-never "$CAS"'s/^  if \[ "\$AP_POLICY" = 1 \]; then$/  if false; then/'
mutant drain drain-strips-the-receipts 's/( unset CARGO_REGISTRY_TOKEN; bash scripts\/cascade-publish.sh )/( unset CARGO_REGISTRY_TOKEN MODEL_LADDER_CRUX_DIR CRUX_CERT; bash scripts\/cascade-publish.sh )/'
mutant drain drain-clean-env 's/( unset CARGO_REGISTRY_TOKEN; bash scripts\/cascade-publish.sh )/( unset CARGO_REGISTRY_TOKEN; env -i PATH="$PATH" bash scripts\/cascade-publish.sh )/'
mutant cascade cascade-preflight-env-dropped 's/^    if ! bash "\$REPO_ROOT\/scripts\/check_publish_preflight.sh"; then$/    if ! env -u MODEL_LADDER_CRUX_DIR -u CRUX_CERT bash "$REPO_ROOT\/scripts\/check_publish_preflight.sh"; then/'
mutant cascade cascade-preflight-skipped 's/^    if ! bash "\$REPO_ROOT\/scripts\/check_publish_preflight.sh"; then$/    if false; then/'
# a reset anywhere in either file, not only on the two lines the cascade rows run
mutant drain drain-exports-empty-crux-dir '1a\export MODEL_LADDER_CRUX_DIR=""'
mutant drain drain-unsets-the-cert '1a\unset CRUX_CERT'
mutant cascade cascade-reexecs-clean '1a\[ -n "${CASCADE_CLEAN:-}" ] || exec env -i CASCADE_CLEAN=1 PATH="$PATH" HOME="$HOME" bash "$0" "$@"'
mutant cascade cascade-assigns-the-cert '1a\CRUX_CERT=/dev/null'
mutant cascade cascade-wraps-bash '1a\bash() { env -u CRUX_CERT bash "$@"; }'
nearmiss drain drain-reads-the-receipts '1a\: "${MODEL_LADDER_CRUX_DIR:-}" "${CRUX_CERT:-}"'
nearmiss drain drain-touches-other-vars '1a\unset CARGO_TOKEN_OLD; export CARGO_TERM_COLOR=never'
nearmiss cascade cascade-env-without-reset '1a\env PATH="$PATH" true'
nearmiss cascade cascade-echoes-the-cert '1a\[ -z "${CRUX_CERT:-}" ] || echo "crux cert: $CRUX_CERT, env: $(printenv CRUX_CERT)"'
nearmiss cascade cascade-comment-names-them '1a\# a hand run: MODEL_LADDER_CRUX_DIR=x CRUX_CERT=y bash scripts/cascade-publish.sh, never env -i'

printf 'mutants: %s/%s killed, near-misses: %s/%s held\n' "$killed" "$total" "$held" "$near"
[ "$bad" = 0 ] && printf 'PASS  %s row(s) + 5 structural, every mutant killed and every near-miss held: the GitHub release is a draft until clean-room, assets and preflight are green (#4690), and the cascade'"'"'s own preflight reads the T-1 CRUX receipts (D4)\n' "$rows"
finish "$bad"
