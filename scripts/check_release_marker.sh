#!/usr/bin/env bash
# check_release_marker.sh - the case table for the release-open marker writer (#4670, R10).
#
# R10 (release_timer_guard.sh) and the tool-installing timers both READ the release-open marker.
# scripts/release/release_marker.sh is what WRITES it: the release autopilot sets it at its first step
# and clears it when the release is live. This table drives the real writer over throwaway hosts:
#
#   set on a host                        -> marker holds the version, read back
#   set over a stale marker              -> WARN (another version, or older than 24h), then set
#   clear our own marker                 -> gone, read back absent
#   clear another release's marker       -> WARN, left in place
#   + every refusal the writer makes, each with its own row.
#
#   bash scripts/check_release_marker.sh             # the table
#   bash scripts/check_release_marker.sh --mutants   # delete each `# R-*` line in a copy;
#                                                    # each must turn the table red
#
# No real host: `--local` runs the real body under a throwaway XDG_STATE_HOME, and `--ssh` goes
# through a stub ssh that runs the same body under a per-host one (or prints a canned answer).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WRITER="${RELEASE_MARKER_SUBJECT:-$HERE/release/release_marker.sh}"

if [ "${1:-}" = "--mutants" ]; then
  src="$HERE/release/release_marker.sh"
  mapfile -t markers < <(grep -o '# R-[A-Z]*$' "$src" | sed 's/^# //')
  [ "${#markers[@]}" -gt 0 ] || { echo "check_release_marker: no R-* markers in $src (vacuous)"; exit 1; }
  # A red table kills every mutant for free: the unmutated table must be green first.
  bash "$0" >/dev/null 2>&1 || { echo "check_release_marker: the unmutated table is red - mutants not_measured"; exit 1; }
  work="$(mktemp -d)"
  killed=0
  for m in "${markers[@]}"; do
    grep -v "# $m\$" "$src" >"$work/writer.sh"
    if cmp -s "$src" "$work/writer.sh"; then
      echo "  SURVIVED $m (the mutation changed nothing)"
      continue
    fi
    if RELEASE_MARKER_SUBJECT="$work/writer.sh" bash "$0" >/dev/null 2>&1; then
      echo "  SURVIVED $m (the table stayed green without it)"
    else
      killed=$((killed+1)); echo "  killed   $m"
    fi
  done
  rm -rf "${work:?}"
  echo "check_release_marker: mutants $killed/${#markers[@]} killed"
  [ "$killed" -eq "${#markers[@]}" ]
  exit $?
fi

[ -f "$WRITER" ] || { echo "check_release_marker: missing $WRITER"; exit 1; }

T="$(mktemp -d)"
trap 'chmod -R u+w "${T:?}" 2>/dev/null; rm -rf "${T:?}"' EXIT

fails=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
want() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi; }
# want_out <name> <substring>: the verdict names its own cause, not a neighbour's.
want_out() { case "$OUT" in *"$2"*) ok "$1";; *) bad "$1" "output containing '$2'" "$OUT";; esac; }
no_out()   { case "$OUT" in *"$2"*) bad "$1" "no '$2'" "$OUT";; *) ok "$1";; esac; }

# The stub ssh: `ssh -o .. -o .. <host> bash -s -- <args>`. A host with a canned answer prints it;
# host `down` is unreachable (255); any other host runs the real body under $T/hosts/<host>.
mkdir -p "$T/bin" "$T/canned" "$T/hosts/local"
cat >"$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
while [ "$1" = -o ]; do shift 2; done
h=$1; shift
root="$(dirname "$(dirname "$0")")"
[ "$h" = down ] && { echo "ssh: connect to host down port 22: No route to host" >&2; exit 255; }
[ -f "$root/canned/$h" ] && { cat >/dev/null; printf '%b' "$(cat "$root/canned/$h")"; exit 0; }
XDG_STATE_HOME="$root/hosts/$h" "$@"
EOF
chmod +x "$T/bin/ssh"
export PATH="$T/bin:$PATH"
export XDG_STATE_HOME="$T/hosts/local"

mk() { echo "$T/hosts/$1/apr/release-open"; }           # mk <host>: that host's marker path
plant() { mkdir -p "$T/hosts/$1/apr"; printf '%s\n' "$2" >"$(mk "$1")"; }
writer() { OUT="$(timeout 20 bash "$WRITER" "$@" 2>&1)"; RC=$?; }
content() { if [ -e "$(mk "$1")" ]; then cat "$(mk "$1")"; else echo absent; fi; }

echo "check_release_marker: the writer of the release-open marker"

# -- set ------------------------------------------------------------------------------------------
writer set 0.70.2 --local trainhost
want "S1 set on the local host -> exit 0" 0 "$RC"
want "S1 the marker holds the version" "0.70.2" "$(content local)"
want_out "S1 says it read the marker back" "release 0.70.2 marked open (read back)"
no_out "S1 a fresh host warns nothing" "WARN"

writer set 0.70.2 --ssh h1
want "S2 set over ssh -> exit 0" 0 "$RC"
want "S2 the remote marker holds the version" "0.70.2" "$(content h1)"

plant h2 0.70.1
writer set 0.70.2 --ssh h2
want "S3 set over a marker for another version -> exit 0" 0 "$RC"
want_out "S3 warns the stale version" "stale release-open marker for 0.70.1 (not 0.70.2)"
want "S3 the marker now holds the new version" "0.70.2" "$(content h2)"

plant h3 0.70.2; touch -d '2 days ago' "$(mk h3)"
writer set 0.70.2 --ssh h3
want "S4 set over the same version, 2 days old -> exit 0" 0 "$RC"
want_out "S4 warns the stale age" "over 24h"

plant h4 0.70.2
writer set 0.70.2 --ssh h4
want "S5 set over a fresh marker for the same version -> exit 0" 0 "$RC"
no_out "S5 warns nothing" "WARN"

# -- clear ----------------------------------------------------------------------------------------
writer clear 0.70.2 --local trainhost --ssh h1
want "C1 clear our own markers -> exit 0" 0 "$RC"
want "C1 the local marker is gone" "absent" "$(content local)"
want "C1 the remote marker is gone" "absent" "$(content h1)"
want_out "C1 says it read back absent" "marker cleared (read back absent)"

plant h5 0.71.0
writer clear 0.70.2 --ssh h5
want "C2 clear meets another release's marker -> exit 0" 0 "$RC"
want_out "C2 warns it is stale for this release" "stale release-open marker for 0.71.0 (not 0.70.2)"
want_out "C2 says it left it" "left the marker for 0.71.0 in place"
want "C2 the other release's marker is untouched" "0.71.0" "$(content h5)"

writer clear 0.70.2 --ssh h6
want "C3 clear with no marker -> exit 0" 0 "$RC"

# -- refusals -------------------------------------------------------------------------------------
writer set 0.70.2 --ssh down
want "F1 host unreachable -> exit 2" 2 "$RC"
want_out "F1 names the exit" "set exited 255"

printf 'FOUND-NONE\nREADBACK 0.70.2\n' >"$T/canned/cut"
writer set 0.70.2 --ssh cut
want "F2 answer cut off before END -> exit 2" 2 "$RC"
want_out "F2 names END" "stopped before END"

mkdir -p "$T/hosts/ro"; chmod 500 "$T/hosts/ro"
writer set 0.70.2 --ssh ro
want "F3 the state dir is not writable -> exit 2" 2 "$RC"
want_out "F3 names the write" "could not write the release-open marker"

printf 'FOUND-NONE\nREADBACK 0.69.0\nEND\n' >"$T/canned/liar"
writer set 0.70.2 --ssh liar
want "F4 read back another version after set -> exit 2" 2 "$RC"
want_out "F4 names the read-back" "read back \"READBACK 0.69.0\", not 0.70.2"

printf 'FOUND 0.70.2 5\nREADBACK 0.70.2\nEND\n' >"$T/canned/sticky"
writer clear 0.70.2 --ssh sticky
want "F5 our marker still there after clear -> exit 2" 2 "$RC"
want_out "F5 names the leftover" "still there after clear"

printf 'FOUND 0.71.0 5\nREADBACK-ABSENT\nEND\n' >"$T/canned/eaten"
writer clear 0.70.2 --ssh eaten
want "F6 another release's marker vanished under clear -> exit 2" 2 "$RC"
want_out "F6 names the foreign marker" "the marker for 0.71.0 changed under clear 0.70.2"

writer set 0.70.2 --ssh h7 --ssh down
want "F7 one host done, one unreachable -> exit 2" 2 "$RC"
want "F7 the reachable host is still marked" "0.70.2" "$(content h7)"

# -- caller errors --------------------------------------------------------------------------------
writer open 0.70.2 --ssh h1
want "E1 an op that is not set/clear -> exit 3" 3 "$RC"
want_out "E1 names the op" "must be set or clear, not 'open'"

writer set --ssh h1
want "E2 no version -> exit 3" 3 "$RC"
want_out "E2 names the version" "'--ssh' is not a version"

for flag in --local --ssh; do
  writer set 0.70.2 "$flag"
  want "E3 $flag with no host name -> exit 3" 3 "$RC"
  want_out "E3 $flag names the missing host name" "$flag needs a host name"
done

writer set 0.70.2 --ssh h1 --bogus
want "E4 an unknown argument -> exit 3" 3 "$RC"
want_out "E4 names the argument" "unknown argument '--bogus'"

writer set 0.70.2
want "E5 no host named -> exit 3" 3 "$RC"
want_out "E5 names the vacuous run" "marking zero hosts is vacuous"

if [ "$fails" -eq 0 ]; then
  echo "check_release_marker: OK - set reads back, stale warns, clear leaves only others' markers"
  exit 0
fi
echo "check_release_marker: $fails assertion(s) FAILED"
exit 1
