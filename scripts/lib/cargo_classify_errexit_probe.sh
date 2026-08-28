#!/usr/bin/env bash
# Row C19's probe. EXECUTED, never sourced, so `set` here is correct.
#
# Every caller of scripts/cargo_classify.sh sources it into a shell running
# `set -euo pipefail`, and the first draft did not survive that: an assignment
# takes the exit status of its command substitution, so a grep that merely
# FOUND NOTHING -- the ordinary case for a CODE log -- aborted the classifier
# before it printed anything. Callers compared the empty string, fell to the
# CODE branch, and were right by accident.
#
# Reproducing that needs a REAL errexit shell. `( set -e; ... )` inside a
# command substitution does not reproduce it, so a probe written that way
# passes vacuously -- measured, not assumed.
#
#   $1  path to scripts/cargo_classify.sh
#   $2  path to a log fixture
# stdout: whatever classify_cargo_failure printed (empty if it was aborted)
set -euo pipefail
. "$1" || exit 1
classify_cargo_failure "$2"
