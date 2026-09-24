#!/usr/bin/env bash
# gather_nightly.sh -- both hosts' nightly certifications into ONE root, for the release gate's nightly half (#4117).
#
# The 0.70 release gate (APR-RELEASE-001 §14.1) is judged on one host, and each host's night lives in ITS OWN
# APR_NIGHTLY_ROOT (default ~/.cache/aprender-nightly, paiml/infra#959's timer). This copies, per host, every
# <root>/<sha>/<host>/ night's verdict.json and EXACTLY the receipts it names (its ladder receipt, each CRUX lane's
# receipt, its certification), at the same relative place -- never the night's checkout, target or shard dirs --
# into <dst>/<sha>/<host>/. The remote host is read over the same
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
PICK='import json, os, shutil, sys, glob
root, host, dst = sys.argv[1:4]
n = skipped = 0
for v in glob.glob(os.path.join(root, "*", host, "verdict.json")):
    src = os.path.dirname(v); to = os.path.join(dst, os.path.basename(os.path.dirname(src)), host)
    try:
        doc = json.load(open(v))
    except (OSError, ValueError):
        doc = {}
    named = [(doc.get("ladder") or {}).get("receipt"), (doc.get("certification") or {}).get("path")]
    named += [(x or {}).get("receipt") for x in ((doc.get("crux") or {}).get("lanes") or {}).values()]
    os.makedirs(to, exist_ok=True); shutil.copy2(v, to); n += 1
    for p in named:
        # exactly the receipts the verdict NAMES, at the same relative place; an absolute path (a night from before
        # #4117) or one that climbs out of the night dir is not copied -- admission then refuses it by name
        if not p or os.path.isabs(p) or os.path.normpath(p).startswith(".."):
            skipped += bool(p); continue
        if os.path.isfile(os.path.join(src, p)):
            os.makedirs(os.path.dirname(os.path.join(to, p)) or to, exist_ok=True)
            shutil.copy2(os.path.join(src, p), os.path.join(to, p))
print("%s night(s) of %s, %s named path(s) not relocatable" % (n, host, skipped))'
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
