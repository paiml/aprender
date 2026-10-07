# release_policy_has_entry.awk -- exit 0 when the top-level `emergency_scopes` list has an entry
# whose release is exactly -v v=VERSION (see release_policy.sh), else exit 1.
/^  emergency_scopes:[ ]*$/ { ins = 1; next }
ins && /^  [^ ]/ { ins = 0 }
ins && $0 ~ /^      release: / { r = substr($0, 16); gsub(/"/, "", r); if (r == v) found = 1 }
END { exit found ? 0 : 1 }
