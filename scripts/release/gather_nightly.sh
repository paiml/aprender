#!/usr/bin/env bash
# gather_nightly.sh -- both hosts' nightly certifications into ONE root, for the release gate's nightly half (#4117).
#
# The 0.70 release gate (APR-RELEASE-001 §14.1) is judged on one host, and each host's night lives in ITS OWN
# APR_NIGHTLY_ROOT (default ~/.cache/aprender-nightly, paiml/infra#959's timer). This copies, per host, every
# <root>/<sha>/<host>/ night's verdict.json and the receipts it names (ladder/, crux/, prompt-certification.json)
# -- never the night's checkout or target -- into <dst>/<sha>/<host>/. The remote host is read over the same
# operator-authorized SSH the models step uses. A verdict records its receipts relative to itself (#4117), so the
# copy is admissible where it lands.
#
# It does NOT admit anything: age, green and ancestry are scripts/lib/nightly_admission.py's, run by the judge
# (check_model_ladder.sh --nightly) or by prepare_bump --ship. Zero nights is not an error here; the admission
# refuses it by name.
#
#   bash scripts/release/gather_nightly.sh <dst> <local host> <remote host>
# exit: 0 gathered · 1 a host's root could not be read (or the SSH failed) · 2 usage
# Callers: scripts/release/models_t1.sh (T-1), scripts/release/prepare_bump.sh --ship.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: gather_nightly.sh <dst> <local host> <remote host>" >&2; exit 2; }
dst=$1; local_host=$2; remote_host=$3
case "$dst" in ''|/|*..*) echo "gather_nightly: refusing destination '$dst' (empty, / or containing ..)" >&2; exit 2 ;; esac
# bashrs SEC010: $dst is the caller's required destination, validated above (no '..', never / or empty).
# bashrs disable-next-line=SEC010
mkdir -p "$dst" || exit 1
# The picker runs on BOTH hosts, so it is plain python with no single quote in it (it rides in a heredoc).
PICK='import os, shutil, sys, glob
root, host, dst = sys.argv[1:4]
n = 0
for v in glob.glob(os.path.join(root, "*", host, "verdict.json")):
    src = os.path.dirname(v); to = os.path.join(dst, os.path.basename(os.path.dirname(src)), host)
    os.makedirs(to, exist_ok=True); shutil.copy2(v, to); n += 1
    for part in ("ladder", "crux"):
        if os.path.isdir(os.path.join(src, part)):
            shutil.copytree(os.path.join(src, part), os.path.join(to, part), dirs_exist_ok=True)
    if os.path.isfile(os.path.join(src, "prompt-certification.json")):
        shutil.copy2(os.path.join(src, "prompt-certification.json"), to)
print("%s night(s) of %s" % (n, host))'
python3 -c "$PICK" "${APR_NIGHTLY_ROOT:-$HOME/.cache/aprender-nightly}" "$local_host" "$dst" || exit 1
# bashrs SEC010: the archive is this script's own remote tar of <sha>/<host>/ dirs; $dst is validated above.
# bashrs disable-next-line=SEC010
ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" "bash -s" <<HOST | tar -xzf - -C "$dst" || exit 1
set -u
t=\$(mktemp -d) || exit 3
python3 -c '$PICK' "\${APR_NIGHTLY_ROOT:-\$HOME/.cache/aprender-nightly}" $remote_host "\$t" >&2 || exit 3
tar -czf - -C "\$t" .
rm -rf -- "\$t"
HOST
exit 0
