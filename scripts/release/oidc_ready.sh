#!/usr/bin/env bash
# oidc_ready.sh -- Verify release.yml OIDC Trusted Publishing readiness and derive the
# per-crate Trusted Publishing list from scripts/cascade-publish.sh TIERS[] (PMAT-6905 / #6907).
#
# Usage:
#   bash scripts/release/oidc_ready.sh [--root DIR]       # print TIERS crates + check workflow & doc
#   bash scripts/release/oidc_ready.sh --attest FILE      # also require crates.io TP attestation
#   bash scripts/release/oidc_ready.sh --workflow-only    # workflow+doc checks only (rc 0 possible)
#   bash scripts/release/oidc_ready.sh --crates [--root DIR]
#   bash scripts/release/oidc_ready.sh --print-doc [--root DIR]
#   bash scripts/release/oidc_ready.sh --self-test        # case table + mutants
#   bash scripts/release/oidc_ready.sh --mutants          # run mutant suite only
#
# Exit:
#   0  all checks pass (or --crates / --print-doc succeeded)
#   1  check failed (missing OIDC config, broken token_auth fallback, doc drift, or vacuous TIERS)
#   2  caller/environment error (bad args, missing files)
set -uo pipefail

SELF_PATH=$(readlink -f -- "${BASH_SOURCE[0]}") || exit 2
HERE=${SELF_PATH%/*}
DEFAULT_ROOT=${HERE%/scripts/release}

EXPECTED_OWNER="paiml"
EXPECTED_REPO="aprender"
EXPECTED_WORKFLOW="release.yml"
EXPECTED_ENV="infra-read"
EXPECTED_AUTH_ACTION="rust-lang/crates-io-auth-action@"

usage() {
  cat >&2 <<'EOF'
usage: oidc_ready.sh [--root DIR] [--attest FILE | --workflow-only] [--crates | --print-doc]
       oidc_ready.sh --self-test | --mutants
EOF
  exit 2
}

validate_dir() {
  local d=$1
  case "$d" in
    /*) ;;
    *) printf 'oidc_ready: refused non-absolute path: %s\n' "$d" >&2; return 2 ;;
  esac
  case "$d" in
    *..*) printf 'oidc_ready: refused path containing ..: %s\n' "$d" >&2; return 2 ;;
  esac
  [ -d "$d" ] || { printf 'oidc_ready: missing directory: %s\n' "$d" >&2; return 2; }
  return 0
}

_OIDC_ST_TMP=""
_OIDC_MUT_TMP=""
cleanup_tmpdirs() {
  [ -z "${_OIDC_ST_TMP:-}" ] || rm -rf -- "${_OIDC_ST_TMP:?}"
  [ -z "${_OIDC_MUT_TMP:-}" ] || rm -rf -- "${_OIDC_MUT_TMP:?}"
}
trap cleanup_tmpdirs EXIT

mk_tmpdir() {
  local d
  d=$(mktemp -d) || return 2
  case "$d" in
    /tmp/*|/var/folders/*|"${TMPDIR:-/tmp}"/*) ;;
    *) printf 'oidc_ready: refused tmpdir outside standard temp root: %s\n' "$d" >&2; return 2 ;;
  esac
  case "$d" in
    *..*) printf 'oidc_ready: refused tmpdir containing ..: %s\n' "$d" >&2; return 2 ;;
  esac
  printf '%s\n' "$d"
}

# Print "<tier>\t<crate>" for every crate in scripts/cascade-publish.sh TIERS[].
# Exits 1 if 0 crates are parsed (anti-vacuity).
extract_tiers_rows() {
  local casc=$1 rows count
  [ -f "$casc" ] || { printf 'FAIL  TIERS: missing cascade script %s\n' "$casc" >&2; return 2; }
  rows=$(awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*TIERS\[[0-9]+\]="[^"]*"/ {
      line = $0
      sub(/^[[:space:]]*TIERS\[/, "", line)
      tier = line
      sub(/\].*$/, "", tier)
      sub(/^[^"]*"/, "", line)
      sub(/".*$/, "", line)
      n = split(line, a, /[[:space:]]+/)
      for (i = 1; i <= n; i++) {
        if (a[i] != "") printf "%s\t%s\n", tier, a[i]
      }
    }
  ' "$casc" | sort -t $'\t' -k1,1n -s)
  count=$(printf '%s\n' "$rows" | grep -c . || true)
  if [ "$count" -lt 1 ]; then
    printf 'FAIL  TIERS: parsed 0 crates from %s (vacuity guard)\n' "$casc" >&2
    return 1
  fi
  printf '%s\n' "$rows"
  return 0
}

# Strip YAML comments (full-line and trailing whitespace + #).
strip_yaml_comments() {
  local wf=$1
  awk '
    /^[[:space:]]*#/ { print ""; next }
    {
      line = $0
      sub(/[[:space:]]+#.*$/, "", line)
      print line
    }
  ' "$wf"
}

# Check 1: workflow has permissions: id-token: write (top-level or jobs.release.permissions).
check_id_token() {
  local wf=$1
  strip_yaml_comments "$wf" | awk '
    /^permissions:[[:space:]]*\{.*id-token:[[:space:]]*write/ { found = 1 }
    /^permissions:[[:space:]]*$/ { in_perm = 1; perm_indent = 0; next }
    /^    permissions:[[:space:]]*\{.*id-token:[[:space:]]*write/ && in_rel { found = 1 }
    /^    permissions:[[:space:]]*$/ && in_rel { in_perm = 1; perm_indent = 4; next }
    /^jobs:[[:space:]]*$/ { in_jobs = 1; next }
    in_jobs && /^  release:[[:space:]]*$/ { in_rel = 1; next }
    in_jobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ && !/^  release:[[:space:]]*$/ { in_rel = 0 }
    in_perm {
      if ($0 ~ /^[[:space:]]*$/) next
      match($0, /^[ ]*/)
      ind = RLENGTH
      if (ind <= perm_indent) {
        in_perm = 0
      } else if ($0 ~ /^[[:space:]]+id-token:[[:space:]]*write[[:space:]]*$/) {
        found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

# Check 2: jobs.release specifies environment: infra-read.
check_environment() {
  local wf=$1 env_name=$2
  strip_yaml_comments "$wf" | awk -v want="$env_name" '
    /^jobs:[[:space:]]*$/ { in_jobs = 1; next }
    in_jobs && /^  release:[[:space:]]*$/ { in_rel = 1; next }
    in_jobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ && !/^  release:[[:space:]]*$/ { in_rel = 0; in_env = 0 }
    in_jobs && /^[A-Za-z0-9_-]+:/ { in_jobs = 0; in_rel = 0; in_env = 0 }
    in_rel && /^    environment:[[:space:]]*/ {
      val = $0
      sub(/^    environment:[[:space:]]*/, "", val)
      sub(/[[:space:]]+$/, "", val)
      if (val == want) { found = 1 }
      else if (val == "") { in_env = 1 }
      next
    }
    in_env {
      if ($0 ~ /^[[:space:]]*$/) next
      match($0, /^[ ]*/)
      if (RLENGTH <= 4) { in_env = 0; next }
      if ($0 ~ /^[[:space:]]+name:[[:space:]]*/) {
        val = $0
        sub(/^[[:space:]]+name:[[:space:]]*/, "", val)
        sub(/[[:space:]]+$/, "", val)
        if (val == want) found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

# Check 3: jobs.release has an auth step using rust-lang/crates-io-auth-action@...,
# with an id: and gated on !inputs.dry_run && !inputs.token_auth.
# Echoes the step id on stdout if found.
find_auth_step_id() {
  local wf=$1 action_prefix=$2
  strip_yaml_comments "$wf" | awk -v act="$action_prefix" '
    function flush_step() {
      if (in_step && has_uses && step_id != "" && has_if_dry && has_if_tok) {
        found_id = step_id
      }
      step_id = ""; has_uses = 0; has_if_dry = 0; has_if_tok = 0
    }
    /^jobs:[[:space:]]*$/ { in_jobs = 1; next }
    in_jobs && /^  release:[[:space:]]*$/ { in_rel = 1; next }
    in_jobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ && !/^  release:[[:space:]]*$/ {
      flush_step(); in_rel = 0; in_steps = 0; in_step = 0
    }
    in_rel && /^    steps:[[:space:]]*$/ { in_steps = 1; next }
    in_steps && /^    [A-Za-z0-9_-]+:/ { flush_step(); in_steps = 0; in_step = 0 }
    in_steps && /^      -[[:space:]]+/ {
      flush_step()
      in_step = 1
      line = $0
      sub(/^      -[[:space:]]+/, "", line)
      if (line ~ /^id:[[:space:]]*/) { sub(/^id:[[:space:]]*/, "", line); sub(/[[:space:]]+$/, "", line); step_id = line }
      if (line ~ /^uses:[[:space:]]*/ && index(line, act) > 0) has_uses = 1
      if (line ~ /^if:[[:space:]]*/) {
        if (index(line, "!inputs.dry_run") > 0) has_if_dry = 1
        if (index(line, "!inputs.token_auth") > 0) has_if_tok = 1
      }
      next
    }
    in_step {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      if (line ~ /^id:[[:space:]]*/) { sub(/^id:[[:space:]]*/, "", line); sub(/[[:space:]]+$/, "", line); step_id = line }
      if (line ~ /^uses:[[:space:]]*/ && index(line, act) > 0) has_uses = 1
      if (line ~ /^if:[[:space:]]*/) {
        if (index(line, "!inputs.dry_run") > 0) has_if_dry = 1
        if (index(line, "!inputs.token_auth") > 0) has_if_tok = 1
      }
    }
    END {
      flush_step()
      if (found_id != "") { print found_id; exit 0 }
      exit 1
    }
  '
}

# Check 4: CARGO_REGISTRY_TOKEN in jobs.release wires steps.<auth_id>.outputs.token
# AND preserves inputs.token_auth + secrets.CARGO_REGISTRY_TOKEN so token_auth=true still works.
check_token_wiring() {
  local wf=$1 auth_id=$2
  local line
  line=$(strip_yaml_comments "$wf" | awk '
    /^jobs:[[:space:]]*$/ { in_jobs = 1; next }
    in_jobs && /^  release:[[:space:]]*$/ { in_rel = 1; next }
    in_jobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ && !/^  release:[[:space:]]*$/ { in_rel = 0 }
    in_rel && /^[[:space:]]+CARGO_REGISTRY_TOKEN:[[:space:]]*/ { print; exit }
  ')
  [ -n "$line" ] || { printf 'FAIL  token-wiring: no CARGO_REGISTRY_TOKEN env in jobs.release\n'; return 1; }
  case "$line" in
    *"steps.${auth_id}.outputs.token"*) ;;
    *) printf 'FAIL  token-wiring: CARGO_REGISTRY_TOKEN does not reference steps.%s.outputs.token\n' "$auth_id"; return 1 ;;
  esac
  case "$line" in
    *"inputs.token_auth"*"secrets.CARGO_REGISTRY_TOKEN"*) ;;
    *) printf 'FAIL  token-wiring: CARGO_REGISTRY_TOKEN does not preserve token_auth=true with secrets.CARGO_REGISTRY_TOKEN\n'; return 1 ;;
  esac
  return 0
}

# Check 5: docs/release/oidc-trusted-publisher.md matches TIERS[] and states exact crates.io settings.
check_doc_sync() {
  local doc=$1 tiers_rows=$2
  [ -f "$doc" ] || { printf 'FAIL  doc-sync: missing %s\n' "$doc"; return 1; }
  local want_crates doc_crates missing extra
  want_crates=$(printf '%s\n' "$tiers_rows" | cut -f2 | sort -u)
  doc_crates=$(awk -F'|' -v own="$EXPECTED_OWNER" -v repo="$EXPECTED_REPO" -v wf="$EXPECTED_WORKFLOW" -v env="$EXPECTED_ENV" '
    /^\|[[:space:]]*[0-9]+[[:space:]]*\|/ {
      crate = $3; gsub(/[`[:space:]]/, "", crate)
      o = $4; gsub(/[`[:space:]]/, "", o)
      r = $5; gsub(/[`[:space:]]/, "", r)
      w = $6; gsub(/[`[:space:]]/, "", w)
      e = $7; gsub(/[`[:space:]]/, "", e)
      if (crate != "" && o == own && r == repo && w == wf && e == env) print crate
    }
  ' "$doc" | sort -u)
  [ -n "$doc_crates" ] || { printf 'FAIL  doc-sync: no valid crate settings rows parsed from %s\n' "$doc"; return 1; }
  missing=$(comm -23 <(printf '%s\n' "$want_crates") <(printf '%s\n' "$doc_crates"))
  extra=$(comm -13 <(printf '%s\n' "$want_crates") <(printf '%s\n' "$doc_crates"))
  if [ -n "$missing" ] || [ -n "$extra" ]; then
    [ -z "$missing" ] || printf 'FAIL  doc-sync: TIERS crate(s) missing or misconfigured in %s:\n%s\n' "$doc" "$(printf '%s\n' "$missing" | sed 's/^/  /')"
    [ -z "$extra" ] || printf 'FAIL  doc-sync: extra crate(s) in %s not in TIERS:\n%s\n' "$doc" "$(printf '%s\n' "$extra" | sed 's/^/  /')"
    return 1
  fi
  return 0
}

print_doc() {
  local root=$1 casc rows count
  casc="$root/scripts/cascade-publish.sh"
  rows=$(extract_tiers_rows "$casc") || return $?
  count=$(printf '%s\n' "$rows" | grep -c . || true)

  cat <<EOF
# crates.io Trusted Publishing (OIDC) Setup for aprender

This document specifies the crates.io Trusted Publishing configuration required so
\`.github/workflows/${EXPECTED_WORKFLOW}\` can publish all \`${count}\` crates without a stored
\`CARGO_REGISTRY_TOKEN\` secret (when \`token_auth=false\`, the default).

> **Generated list:** The crate table below is derived directly from \`TIERS[]\` in
> \`scripts/cascade-publish.sh\` by \`bash scripts/release/oidc_ready.sh --print-doc\`.
> Do not edit the crate list by hand; run \`bash scripts/release/oidc_ready.sh\` to verify
> workflow readiness and doc sync.

## Required GitHub Actions Configuration (\`.github/workflows/${EXPECTED_WORKFLOW}\`)

- **Workflow permissions:** \`id-token: write\`
- **Job environment (\`jobs.release\`):** \`${EXPECTED_ENV}\`
- **Auth step:** \`rust-lang/crates-io-auth-action\` gated by \`!inputs.dry_run && !inputs.token_auth\`
- **Fallback preserved:** \`token_auth=true\` continues to use \`secrets.CARGO_REGISTRY_TOKEN\`

## Exact crates.io Trusted Publishing Settings (Per Crate)

Configure a GitHub Actions Trusted Publisher on \`https://crates.io/crates/<crate>/settings\`
for **every** crate below with these exact values:

- **Repository owner:** \`${EXPECTED_OWNER}\`
- **Repository name:** \`${EXPECTED_REPO}\`
- **Workflow filename:** \`${EXPECTED_WORKFLOW}\`
- **Environment name:** \`${EXPECTED_ENV}\` (must match \`jobs.release.environment\`; leaving it blank causes crates.io OIDC exchange to fail)

## Crate List (${count} crates from \`TIERS[]\`)

| Tier | Crate | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- | --- |
EOF
  printf '%s\n' "$rows" | awk -F'\t' -v own="$EXPECTED_OWNER" -v repo="$EXPECTED_REPO" -v wf="$EXPECTED_WORKFLOW" -v env="$EXPECTED_ENV" '{
    printf "| %s | `%s` | `%s` | `%s` | `%s` | `%s` |\n", $1, $2, own, repo, wf, env
  }'
  return 0
}

run_checks() {
  local root=$1 quiet=${2:-0}
  local casc wf doc rows count auth_id rc=0
  casc="$root/scripts/cascade-publish.sh"
  wf="$root/.github/workflows/$EXPECTED_WORKFLOW"
  doc="$root/docs/release/oidc-trusted-publisher.md"

  [ -f "$wf" ] || { printf 'FAIL  workflow: missing %s\n' "$wf"; return 2; }

  rows=$(extract_tiers_rows "$casc") || return $?
  count=$(printf '%s\n' "$rows" | grep -c . || true)

  if [ "$quiet" = 0 ]; then
    printf '=== TIERS crate list (%s crates from %s) ===\n' "$count" "$casc"
    printf '%s\n' "$rows" | cut -f2
    printf '\n=== OIDC workflow checks (%s) ===\n' "$wf"
  fi

  if check_id_token "$wf"; then
    printf 'ok    id-token: write present in permissions\n'
  else
    printf 'FAIL  id-token: permissions must declare id-token: write\n'
    rc=1
  fi

  if check_environment "$wf" "$EXPECTED_ENV"; then
    printf 'ok    environment: jobs.release uses %s\n' "$EXPECTED_ENV"
  else
    printf 'FAIL  environment: jobs.release must declare environment: %s\n' "$EXPECTED_ENV"
    rc=1
  fi

  if auth_id=$(find_auth_step_id "$wf" "$EXPECTED_AUTH_ACTION"); then
    printf 'ok    auth-step: %s step (id=%s) gated on !inputs.dry_run && !inputs.token_auth\n' "$EXPECTED_AUTH_ACTION" "$auth_id"
    if check_token_wiring "$wf" "$auth_id"; then
      printf 'ok    token-wiring: CARGO_REGISTRY_TOKEN wires steps.%s.outputs.token and preserves token_auth=true\n' "$auth_id"
    else
      rc=1
    fi
  else
    printf 'FAIL  auth-step: missing %s step (with id and !inputs.dry_run && !inputs.token_auth)\n' "$EXPECTED_AUTH_ACTION"
    rc=1
  fi

  if check_doc_sync "$doc" "$rows"; then
    printf 'ok    doc-sync: %s matches all %s TIERS crates with (%s/%s, %s, %s)\n' \
      "$doc" "$count" "$EXPECTED_OWNER" "$EXPECTED_REPO" "$EXPECTED_WORKFLOW" "$EXPECTED_ENV"
  else
    rc=1
  fi

  return "$rc"
}

# crates.io Trusted Publisher state. The crates.io trusted_publishing API needs an owner session,
# so it is NOT read here (no credentials are read or sent). The state is an attestation file:
# one crate name per line (# comments ok) that the crates.io owner confirmed on
# https://crates.io/crates/<crate>/settings. Every TIERS crate absent from it is MISSING.
# Prints the per-crate setting once if any is missing. rc 1 while any is missing.
check_tp_state() {
  local rows=$1 attest=$2 missing=0 crate have=""
  if [ -n "$attest" ]; then
    [ -r "$attest" ] || { printf 'FAIL  crates.io: unreadable attestation file %s\n' "$attest"; return 1; }
    have=$(sed -e 's/#.*$//' -e 's/[[:space:]]//g' "$attest" | grep -v '^$' || true)
  fi
  printf '\n=== crates.io Trusted Publisher state (attestation: %s) ===\n' "${attest:-none}"
  while IFS=$'\t' read -r _ crate; do
    [ -n "$crate" ] || continue
    if printf '%s\n' "$have" | grep -qxF -- "$crate"; then
      printf 'ok    tp: %s\n' "$crate"
    else
      printf 'MISSING tp: %s\n' "$crate"
      missing=$((missing + 1))
    fi
  done <<< "$rows"
  if [ "$missing" -gt 0 ]; then
    printf 'FAIL  crates.io: %s crate(s) lack a Trusted Publisher. Add on https://crates.io/crates/<crate>/settings:\n' "$missing"
    printf '      owner=%s repo=%s workflow=%s environment=%s\n' \
      "$EXPECTED_OWNER" "$EXPECTED_REPO" "$EXPECTED_WORKFLOW" "$EXPECTED_ENV"
    return 1
  fi
  return 0
}

seed_fixture() {
  local d=$1 mode=${2:-good}
  mkdir -p "$d/scripts" "$d/.github/workflows" "$d/docs/release" || return 2
  if [ "$mode" = empty_tiers ]; then
    printf '#!/usr/bin/env bash\necho no tiers here\n' > "$d/scripts/cascade-publish.sh"
  else
    cat > "$d/scripts/cascade-publish.sh" <<'EOF'
#!/usr/bin/env bash
declare -A TIERS
TIERS[1]="crate-a crate-b"
TIERS[2]="crate-c"
EOF
  fi

  case "$mode" in
    good|empty_tiers|doc_missing|doc_wrong_env)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  contents: write
  id-token: write   # Trusted Publishing
jobs:
  release:
    runs-on: [self-hosted, Linux, X64, clean-room]
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    missing_id_token)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  contents: write
jobs:
  release:
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    id_token_read)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: read
jobs:
  release:
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    id_token_commented)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  contents: write
  # id-token: write
jobs:
  release:
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    missing_env)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: write
jobs:
  release:
    runs-on: [self-hosted]
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    wrong_env)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: write
jobs:
  release:
    environment: production
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    missing_auth)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: write
jobs:
  release:
    environment: infra-read
    steps:
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    auth_unconditional)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: write
jobs:
  release:
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
    oidc_not_wired)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: write
jobs:
  release:
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ (inputs.token_auth && !inputs.dry_run) && secrets.CARGO_REGISTRY_TOKEN || '' }}
        run: echo ok
EOF
      ;;
    token_auth_broken)
      cat > "$d/.github/workflows/release.yml" <<'EOF'
name: Release
permissions:
  id-token: write
jobs:
  release:
    environment: infra-read
    steps:
      - id: auth
        if: ${{ !inputs.dry_run && !inputs.token_auth }}
        uses: rust-lang/crates-io-auth-action@3268eb413133e2960fa4ae0e963ca76380c54cd9
      - name: release
        env:
          CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}
        run: echo ok
EOF
      ;;
  esac

  case "$mode" in
    empty_tiers)
      printf '| Tier | Crate | Owner | Repository | Workflow | Environment |\n' > "$d/docs/release/oidc-trusted-publisher.md"
      ;;
    doc_missing)
      print_doc "$d" | grep -v 'crate-b' > "$d/docs/release/oidc-trusted-publisher.md"
      ;;
    doc_wrong_env)
      print_doc "$d" | sed 's/`infra-read`/`wrong-env`/g' > "$d/docs/release/oidc-trusted-publisher.md"
      ;;
    *)
      print_doc "$d" > "$d/docs/release/oidc-trusted-publisher.md"
      ;;
  esac
}

self_test() {
  local tmp pass=0 fail=0
  tmp=$(mk_tmpdir) || return 2
  _OIDC_ST_TMP=$tmp

  row() {
    local name=$1 mode=$2 want_rc=$3 want_sub=$4 fx="$tmp/$1" out rc=0
    seed_fixture "$fx" "$mode" || { printf '  BROKE %-40s fixture failed\n' "$name"; fail=$((fail + 1)); return; }
    out=$(bash "$SELF_PATH" --workflow-only --root "$fx" 2>&1) || rc=$?
    if [ "$rc" = "$want_rc" ] && { [ -z "$want_sub" ] || grep -qF -- "$want_sub" <<<"$out"; }; then
      printf '  ok    %s\n' "$name"
      pass=$((pass + 1))
    else
      printf '  BROKE %-40s rc=%s (want %s) out=[%s]\n' "$name" "$rc" "$want_rc" "$(printf '%s' "$out" | tr '\n' '|' | head -c 160)"
      fail=$((fail + 1))
    fi
  }

  printf '=== oidc_ready.sh case table ===\n'
  row green_valid_workflow_and_tiers     good               0 "crate-c"
  row red_missing_id_token               missing_id_token   1 "id-token"
  row red_id_token_read_only             id_token_read      1 "id-token"
  row red_id_token_commented_out         id_token_commented 1 "id-token"
  row red_missing_environment            missing_env        1 "infra-read"
  row red_wrong_environment              wrong_env          1 "infra-read"
  row red_missing_auth_step              missing_auth       1 "crates-io-auth-action@"
  row red_auth_missing_token_auth_guard  auth_unconditional 1 "crates-io-auth-action@"
  row red_oidc_token_not_wired           oidc_not_wired     1 "outputs.token"
  row red_token_auth_broken              token_auth_broken  1 "token_auth=true"
  row red_empty_tiers                    empty_tiers        1 "vacuity guard"
  row red_doc_missing_crate              doc_missing        1 "doc-sync:"
  row red_doc_wrong_settings             doc_wrong_env      1 "doc-sync:"

  arow() {
    local name=$1 want_rc=$2 want_sub=$3 content=$4 mode=${5:-file} fx="$tmp/good" out rc=0 af="$tmp/attest.txt"
    printf '%s' "$content" > "$af"
    case "$mode" in
      none) out=$(bash "$SELF_PATH" --root "$fx" 2>&1) || rc=$? ;;
      unreadable) out=$(bash "$SELF_PATH" --attest "$tmp/no-such-file" --root "$fx" 2>&1) || rc=$? ;;
      *) out=$(bash "$SELF_PATH" --attest "$af" --root "$fx" 2>&1) || rc=$? ;;
    esac
    if [ "$rc" = "$want_rc" ] && { [ -z "$want_sub" ] || grep -qF -- "$want_sub" <<<"$out"; }; then
      printf '  ok    %s\n' "$name"; pass=$((pass + 1))
    else
      printf '  BROKE %-40s rc=%s (want %s) out=[%s]\n' "$name" "$rc" "$want_rc" "$(printf '%s' "$out" | tr '\n' '|' | head -c 160)"
      fail=$((fail + 1))
    fi
  }
  seed_fixture "$tmp/good" good || fail=$((fail + 1))
  arow red_no_attestation_file        1 "MISSING tp: crate-a" ""  none
  arow red_empty_attestation          1 "owner=paiml repo=aprender workflow=release.yml environment=infra-read" "# none
"
  arow red_one_crate_unattested       1 "MISSING tp: crate-c" "crate-a
crate-b
"
  arow red_unreadable_attestation     1 "unreadable"          "" unreadable
  arow green_all_attested             0 "ok    tp: crate-c"   "crate-a # ok
crate-b
crate-c
"
  printf -- '--- %s/%s case rows ---\n' "$pass" "$((pass + fail))"

  if [ "${OIDC_MUTANTS:-1}" = "1" ]; then
    mutants || fail=$((fail + 1))
  fi

  rm -rf -- "${tmp:?}"
  [ "$fail" -eq 0 ]
}

# One mutant per check/guard; each must turn --self-test RED.
OIDC_MUTANT_TABLE='m1_tiers_vacuous_pass s/if \[ "\$count" -lt 1 \]; then/if false; then/
m2_id_token_skipped s/if check_id_token "\$wf"; then/if true; then/
m3_yaml_comments_unstripped s/^strip_yaml_comments() {$/strip_yaml_comments() { awk "{print}" "\$1"; return 0;/
m4_environment_skipped s/if check_environment "\$wf" "\$EXPECTED_ENV"; then/if true; then/
m5_auth_step_skipped s/if auth_id=\$(find_auth_step_id "\$wf" "\$EXPECTED_AUTH_ACTION"); then/if auth_id="auth"; then/
m6_auth_condition_ignored s/if (in_step && has_uses && step_id != "" && has_if_dry && has_if_tok)/if (in_step \&\& has_uses \&\& step_id != "")/
m7_oidc_token_wiring_skipped s/\*"steps\.\${auth_id}\.outputs\.token"\*)/\*"steps.${auth_id}.outputs.token"\*|*)/
m8_token_auth_preserved_skipped s/\*"inputs\.token_auth"\*"secrets\.CARGO_REGISTRY_TOKEN"\*)/\*"inputs.token_auth"*"secrets.CARGO_REGISTRY_TOKEN"*|*)/
m9_doc_sync_skipped s/if check_doc_sync "\$doc" "\$rows"; then/if true; then/
m10_final_verdict_ignored s/return "\$rc"/return 0/
m11_tp_missing_not_counted s/missing=\$((missing + 1))/missing=0/
m12_tp_always_attested s/if printf .%s.n. "\$have" | grep -qxF -- "\$crate"; then/if true; then/
m13_tp_check_skipped s/^    check_tp_state "\$(extract_tiers_rows/    true "$(extract_tiers_rows/'

mutants() {
  local tmp pass=0 fail=0 name expr m out rc
  tmp=$(mk_tmpdir) || return 2
  _OIDC_MUT_TMP=$tmp
  m="$tmp/oidc_ready.sh"
  cp "$SELF_PATH" "$m"
  if ! OIDC_MUTANTS=0 bash "$m" --self-test > /dev/null 2>&1; then
    printf '  BROKE %-40s unmutated baseline failed in %s\n' baseline "$tmp"
    rm -rf -- "${tmp:?}"
    return 1
  fi

  printf '=== oidc_ready.sh mutants ===\n'
  while IFS=' ' read -r name expr; do
    [ -n "$name" ] || continue
    sed -e "$expr" "$SELF_PATH" > "$m"
    if cmp -s "$SELF_PATH" "$m"; then
      printf '  BROKE %-40s mutant pattern matched nothing\n' "$name"
      fail=$((fail + 1))
      continue
    fi
    if ! bash -n "$m" 2>/dev/null; then
      printf '  BROKE %-40s mutant failed bash -n\n' "$name"
      fail=$((fail + 1))
      continue
    fi
    rc=0
    out=$(OIDC_MUTANTS=0 bash "$m" --self-test 2>&1) || rc=$?
    case "$rc:$out" in
      0:*)
        printf '  BROKE %-40s SURVIVED (case table stayed green)\n' "$name"
        fail=$((fail + 1))
        ;;
      *"  BROKE "*)
        printf '  ok    %-40s killed\n' "$name"
        pass=$((pass + 1))
        ;;
      *)
        printf '  BROKE %-40s exited %s without BROKE row\n' "$name" "$rc"
        fail=$((fail + 1))
        ;;
    esac
  done <<< "$OIDC_MUTANT_TABLE"

  printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
  rm -rf -- "${tmp:?}"
  [ "$fail" -eq 0 ]
}

MODE="check"
ATTEST=""
WF_ONLY=0
ROOT="$DEFAULT_ROOT"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      shift
      [ "$#" -gt 0 ] || usage
      ROOT=$1
      ;;
    --attest)
      shift
      [ "$#" -gt 0 ] || usage
      ATTEST=$1
      ;;
    --workflow-only) WF_ONLY=1 ;;
    --crates) MODE="crates" ;;
    --print-doc) MODE="print-doc" ;;
    --self-test) MODE="self-test" ;;
    --mutants) MODE="mutants" ;;
    *) usage ;;
  esac
  shift
done

case "$MODE" in
  self-test) self_test; exit $? ;;
  mutants) mutants; exit $? ;;
  crates)
    validate_dir "$ROOT" || exit $?
    extract_tiers_rows "$ROOT/scripts/cascade-publish.sh" | cut -f2
    exit "${PIPESTATUS[0]}"
    ;;
  print-doc)
    validate_dir "$ROOT" || exit $?
    print_doc "$ROOT"
    exit $?
    ;;
  check)
    validate_dir "$ROOT" || exit $?
    run_checks "$ROOT" 0
    wrc=$?
    [ "$wrc" -eq 0 ] || exit "$wrc"
    [ "$WF_ONLY" = 1 ] && exit 0
    check_tp_state "$(extract_tiers_rows "$ROOT/scripts/cascade-publish.sh")" "$ATTEST"
    exit $?
    ;;
esac
