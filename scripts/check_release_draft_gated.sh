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
# Not wired as a blocking guard yet (L31: a new check blocks after three green nights); it lives in
# scripts/release/ so guard_tree does not pick it up.
# Exit 0 = every row as expected and every mutant killed · 1 = a row or a mutant landed wrong ·
# 2 = the autopilot has none of the step bodies this table runs, or jq is absent.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUTOPILOT="${1:-$HERE/release/autopilot.sh}"
command -v jq > /dev/null || { printf 'ENV   jq is not on PATH: the stub cannot apply --jq filters\n' >&2; exit 2; }
T=$(mktemp -d) || exit 2
trap 'rm -rf "${T:?}"' EXIT
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
  return "$wrong"
}

# extract <autopilot> -> the harness body: publish_release() if present, then each step body in order
extract() {
  local ap=$1 s
  awk '/^publish_release\(\) \{/,/^\}/' "$ap"
  for s in tag cleanroom assets preflight publish; do
    awk -v h="if run_step $s; then" '$0 == h { on = 1 } on { print } on && /^fi$/ { exit }' "$ap"
  done
}

# run <autopilot> <invocations> <FX vars...> -> "<verdict> <stop|ran>"
# <invocations>: step lists joined by "+", each run as its own resumed autopilot over shared state.
# FX_ASSETS: check_release_assets.sh rc once built (missing build is always rc 1) · FX_CLEANROOM: the
# job conclusion · FX_PREFLIGHT: preflight rc · FX_EDIT: ok | fail (the publishing edit) · FX_PRE:
# pre-state for a run that starts late (draft-built | public-built | gone: built, no release) ·
# FX_CRUN: cleanroom-run-id file (1 = present) · FX_PASS: preflight-pass line (ok | stale | prior |
# none; prior = an earlier run's PASS for this tag and commit, with no PREFLIGHT line of its own).
run() {
  local ap=$1 inv=$2 W on
  W=$(mktemp -d "$T/w.XXXXXX") || return 2
  mkdir -p "$W/state" "$W/ap" "$W/scripts/release"
  : > "$W/calls"
  printf 'notes\n' > "$W/ap/release_notes.md"
  local FX_ASSETS=0 FX_CLEANROOM=success FX_PREFLIGHT=0 FX_EDIT=ok FX_PRE=none FX_CRUN=1 FX_PASS=ok kv
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
  printf '#!/usr/bin/env bash\necho "PREFLIGHT rc=%s" >> %q\nexit %s\n' "$FX_PREFLIGHT" "$W/calls" "$FX_PREFLIGHT" > "$W/scripts/check_publish_preflight.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$W/scripts/release/tag_coverage_gate.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$W/scripts/release/rc_publish_gate.sh"
  {
    printf 'set -uo pipefail\ncd %q || exit 2\n' "$W"
    printf 'REPO=paiml/aprender INFRA=paiml/infra V=0.0.0 T=v0.0.0 MC=deadbeef\n'
    printf 'AP=%q LOG=%q STATUS=%q WT=%q\n' "$W/ap" "$W/log" "$W/status" "$W"
    printf 'say() { printf "SAY %%s\\n" "$*" >> "$LOG"; }\n'
    printf 'die() { printf "STOP %%s\\n" "$*" >> %q; exit 1; }\n' "$W/calls"
    printf 'sleep() { :; }\n'
    printf 'cut_tag() { printf "TAG %%s\\n" "$2" >> %q; }\n' "$W/calls"
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
    END { printf "%s %s", (v == "" ? "draft" : v), (s == "" ? "ran" : s) }' "$W/calls"
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
a_failed_publishing_edit_stops|publish|FX_PRE=draft-built FX_EDIT=fail|draft stop"

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
bad=0
table "$AUTOPILOT" || bad=1
rows=$(grep -c '|' <<< "$CASES")
[ "$rows" -ge 20 ] || { printf 'VACUOUS %s row(s), fewer than the 20 declared\n' "$rows"; bad=1; }
[ "${1:-}" = "" ] || exit "$bad"   # an explicit autopilot (e.g. origin/main's) runs the table only

# mutant <name> <sed expression>: applied to a copy of the autopilot, the table must go WRONG
killed=0 total=0
mutant() {
  local m="$T/m-$1.sh"
  total=$((total + 1))
  sed "$2" "$AUTOPILOT" > "$m"
  if cmp -s "$AUTOPILOT" "$m"; then printf '  INCONCLUSIVE mutant %s changed nothing\n' "$1"; bad=1; return; fi
  bash -n "$m" || { printf '  BROKEN   mutant %s does not parse\n' "$1"; bad=1; return; }
  if table "$m" > /dev/null; then printf '  SURVIVED mutant %s\n' "$1"; bad=1
  else killed=$((killed + 1)); printf '  killed   mutant %s (%s)\n' "$1" "$(table "$m" | grep -m1 -o 'WRONG [a-z_]*')"; fi
}
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

printf 'mutants: %s/%s killed\n' "$killed" "$total"
[ "$bad" = 0 ] && printf 'PASS  %s row(s) + 4 structural and every mutant killed: the GitHub release is a draft until clean-room, assets and preflight are green (#4690)\n' "$rows"
exit "$bad"
