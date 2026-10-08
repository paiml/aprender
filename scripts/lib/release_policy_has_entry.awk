# release_policy_has_entry.awk -- exit 0 when the top-level `emergency_scopes` list has an entry
# whose release is the same X.Y.Z as -v v=VERSION (see release_policy.sh), else exit 1. Pre-release and
# build suffixes are dropped on both sides: 0.71.0-rc.1 is a 0.71.0 cut, so an entry for either spelling
# is a per-release ruling for that release.
# The judge reads the ladder as YAML, so this matches every shape YAML reads as that release: the key
# first in its item ("- release:") or later, the value double-, single- or un-quoted, with a trailing
# comment or blanks. A missed shape would let a per-release entry ride beside the standing policy.
/^  emergency_scopes:[ ]*$/ { ins = 1; next }
ins && /^  [^ ]/ { ins = 0 }
ins && match($0, /^    (- |  )release:[ ]/) {
    r = substr($0, RLENGTH + 1); sub(/[ \t]+#.*$/, "", r); sub(/[ \t]+$/, "", r)
    if (r ~ /^".*"$/ || r ~ /^'.*'$/) r = substr(r, 2, length(r) - 2)
    sub(/[-+].*$/, "", r); c = v; sub(/[-+].*$/, "", c)
    if (r == c) found = 1
}
END { exit found ? 0 : 1 }
