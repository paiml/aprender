//! CWE-targeted bash mutation for ShellSafetyBench
//!
//! Generates safe/unsafe shell script pairs with specific CWE patterns.
//! Used by bashrs ShellSafetyBench (spec S14.9, Steps 7.2/7.2b).
//!
//! ## Supported CWEs
//!
//! **In-distribution** (covered by bashrs linter):
//! - CWE-78: OS Command Injection (unquoted variables)
//! - CWE-94: Code Injection (eval usage)
//! - CWE-330: Insufficient Randomness ($RANDOM, $$)
//! - CWE-362: TOCTOU Race Condition (missing -p/-f flags)
//! - CWE-798: Hard-coded Credentials
//! - CWE-829: Inclusion of Untrusted Functionality (curl|bash)
//!
//! **Out-of-distribution** (eval-only, not in bashrs linter):
//! - CWE-426: Untrusted Search Path
//! - CWE-77: Command Injection (xargs without -0)
//! - CWE-116: Improper Output Encoding (log injection)
//! - CWE-250: Execution with Unnecessary Privileges

use serde::Serialize;

/// A CWE-targeted mutation: safe version + mutated unsafe version
#[derive(Debug, Clone, Serialize)]
pub struct CweMutation {
    /// Safe (original) script
    pub safe_script: String,
    /// Mutated (unsafe) script
    pub unsafe_script: String,
    /// CWE identifier (e.g., "CWE-78")
    pub cwe: String,
    /// CWE numeric ID
    pub cwe_id: u32,
    /// Human-readable vulnerability name
    pub vulnerability: String,
    /// Which mutation was applied
    pub mutation_description: String,
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64) — no external dependency needed
// ---------------------------------------------------------------------------

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Avoid zero-state (xorshift degenerates)
        Self {
            state: if seed == 0 {
                0x5851_F42D_4C95_7F2D
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    /// Pick a random element from a slice.
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let idx = (self.next() as usize) % items.len();
        &items[idx]
    }
}

// ---------------------------------------------------------------------------
// Template expansion engine
// ---------------------------------------------------------------------------

/// A base template with placeholders like `{VAR}`, `{CMD}`, etc.
struct BaseTemplate {
    safe: &'static str,
    vuln: &'static str,
    desc: &'static str,
}

/// A substitution rule: placeholder string → list of possible values.
struct Substitution {
    placeholder: &'static str,
    values: &'static [&'static str],
}

/// Expand base templates into `count` unique `CweMutation` entries by
/// substituting placeholders with randomly-selected values.
fn expand_templates(
    bases: &[BaseTemplate],
    count: usize,
    seed: u64,
    subs: &[Substitution],
    cwe_tag: &str,
    cwe_id: u32,
    vuln_name: &str,
) -> Vec<CweMutation> {
    let mut rng = Xorshift64::new(seed.wrapping_add(u64::from(cwe_id)));
    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::with_capacity(count);

    // First pass: emit each base template once (identity substitution) to
    // guarantee coverage even when count <= bases.len().
    for base in bases {
        if results.len() >= count {
            break;
        }
        let safe = apply_subs(base.safe, subs, &mut rng);
        let vuln = apply_subs(base.vuln, subs, &mut rng);
        let desc = apply_subs(base.desc, subs, &mut rng);
        let key = format!("{safe}||{vuln}");
        if seen.insert(key) {
            results.push(CweMutation {
                safe_script: safe,
                unsafe_script: vuln,
                cwe: cwe_tag.to_string(),
                cwe_id,
                vulnerability: vuln_name.to_string(),
                mutation_description: desc,
            });
        }
    }

    // Second pass: keep generating random variants until we have enough.
    let mut attempts = 0u64;
    while results.len() < count && attempts < (count as u64) * 20 {
        attempts += 1;
        let base = &bases[(rng.next() as usize) % bases.len()];
        let safe = apply_subs(base.safe, subs, &mut rng);
        let vuln = apply_subs(base.vuln, subs, &mut rng);
        let desc = apply_subs(base.desc, subs, &mut rng);
        let key = format!("{safe}||{vuln}");
        if seen.insert(key) {
            results.push(CweMutation {
                safe_script: safe,
                unsafe_script: vuln,
                cwe: cwe_tag.to_string(),
                cwe_id,
                vulnerability: vuln_name.to_string(),
                mutation_description: desc,
            });
        }
    }

    results.truncate(count);
    results
}

/// Replace all `{PLACEHOLDER}` occurrences in `text` with random values.
fn apply_subs(text: &str, subs: &[Substitution], rng: &mut Xorshift64) -> String {
    let mut result = text.to_string();
    for sub in subs {
        // Replace each occurrence independently so the same placeholder in one
        // template can get different values.
        while result.contains(sub.placeholder) {
            let val = rng.pick(sub.values);
            result = result.replacen(sub.placeholder, val, 1);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate CWE-targeted mutations for bash scripts.
///
/// Returns pairs of (safe, unsafe) scripts for each CWE pattern.
pub fn generate_cwe_mutations(cwe_targets: &[u32], count: usize, seed: u64) -> Vec<CweMutation> {
    let mut results = Vec::new();
    let n = cwe_targets.len().max(1);
    let base_per_cwe = count / n;
    let remainder = count % n;

    for (i, &cwe_id) in cwe_targets.iter().enumerate() {
        // Distribute the remainder across the first `remainder` CWEs
        let this_count = base_per_cwe + usize::from(i < remainder);
        let mutations = match cwe_id {
            78 => generate_cwe_78(this_count, seed),
            94 => generate_cwe_94(this_count, seed),
            330 => generate_cwe_330(this_count, seed),
            362 => generate_cwe_362(this_count, seed),
            798 => generate_cwe_798(this_count, seed),
            829 => generate_cwe_829(this_count, seed),
            377 => generate_cwe_377(this_count, seed),
            732 => generate_cwe_732(this_count, seed),
            // OOD CWEs
            426 => generate_cwe_426(this_count, seed),
            77 => generate_cwe_77(this_count, seed),
            116 => generate_cwe_116(this_count, seed),
            250 => generate_cwe_250(this_count, seed),
            _ => Vec::new(),
        };
        results.extend(mutations);
    }

    results.truncate(count);
    results
}

/// All in-distribution CWE IDs (covered by bashrs linter)
pub fn in_distribution_cwes() -> Vec<u32> {
    vec![78, 94, 330, 362, 377, 732, 798, 829]
}

/// All out-of-distribution CWE IDs (eval-only)
pub fn ood_cwes() -> Vec<u32> {
    vec![426, 77, 116, 250]
}

// ===========================================================================
// In-Distribution CWEs
// ===========================================================================

/// CWE-78: OS Command Injection via unquoted variables
#[allow(clippy::too_many_lines)]
fn generate_cwe_78(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\nrm -rf \"${dir:?}\"/tmp",
            vuln: "#!/bin/sh\nrm -rf $dir/tmp",
            desc: "Unquoted variable in rm command allows path injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncp \"${src}\" \"${dst}\"",
            vuln: "#!/bin/sh\ncp $src $dst",
            desc: "Unquoted variables in cp allow argument injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind \"${search_dir}\" -name '*.log'",
            vuln: "#!/bin/sh\nfind $search_dir -name '*.log'",
            desc: "Unquoted variable in find allows path traversal",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ngrep \"${pattern}\" \"${file}\"",
            vuln: "#!/bin/sh\ngrep $pattern $file",
            desc: "Unquoted grep arguments allow injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 644 \"${target_file}\"",
            vuln: "#!/bin/sh\nchmod 644 $target_file",
            desc: "Unquoted variable in chmod allows arg injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nmv \"${old_name}\" \"${new_name}\"",
            vuln: "#!/bin/sh\nmv $old_name $new_name",
            desc: "Unquoted variables in mv allow injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ntar czf backup.tar.gz \"${backup_dir}\"",
            vuln: "#!/bin/sh\ntar czf backup.tar.gz $backup_dir",
            desc: "Unquoted variable in tar allows injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nssh \"${remote_host}\" 'uptime'",
            vuln: "#!/bin/sh\nssh $remote_host 'uptime'",
            desc: "Unquoted variable in ssh allows host injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncat \"${logfile}\" | head -n 100",
            vuln: "#!/bin/sh\ncat $logfile | head -n 100",
            desc: "Unquoted variable in cat allows filename injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nwc -l \"${input_file}\"",
            vuln: "#!/bin/sh\nwc -l $input_file",
            desc: "Unquoted variable in wc allows argument injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsort \"${data_file}\" > \"${output_file}\"",
            vuln: "#!/bin/sh\nsort $data_file > $output_file",
            desc: "Unquoted variables in sort/redirect allow injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nhead -n 10 \"${report}\"",
            vuln: "#!/bin/sh\nhead -n 10 $report",
            desc: "Unquoted variable in head allows filename injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ntouch \"${marker_file}\"",
            vuln: "#!/bin/sh\ntouch $marker_file",
            desc: "Unquoted variable in touch allows path injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchown root:root \"${config_file}\"",
            vuln: "#!/bin/sh\nchown root:root $config_file",
            desc: "Unquoted variable in chown allows argument injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n{CMD} \"${VAR}\"",
            vuln: "#!/bin/sh\n{CMD} ${VAR}",
            desc: "Unquoted variable in {CMD} allows {ATTACK} injection",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{VAR}",
            values: &[
                "dir",
                "path",
                "file",
                "input",
                "output",
                "src",
                "dst",
                "name",
                "target",
                "prefix",
                "suffix",
                "archive",
                "config",
                "data",
                "log",
                "report",
                "backup",
                "cache",
                "temp",
                "home_dir",
                "work_dir",
                "build_dir",
                "deploy_dir",
                "user_input",
                "remote_path",
            ],
        },
        Substitution {
            placeholder: "{CMD}",
            values: &[
                "ls",
                "cat",
                "stat",
                "file",
                "du",
                "df",
                "tail",
                "head",
                "wc",
                "md5sum",
                "sha256sum",
                "readlink",
                "realpath",
                "basename",
                "dirname",
                "test -f",
                "test -d",
            ],
        },
        Substitution {
            placeholder: "{ATTACK}",
            values: &["path", "argument", "filename", "glob", "command"],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-78",
        78,
        "OS Command Injection",
    )
}

/// CWE-94: Code Injection via eval
#[allow(clippy::too_many_lines)]
fn generate_cwe_94(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\necho \"$user_input\"",
            vuln: "#!/bin/sh\neval \"$user_input\"",
            desc: "eval of user input allows arbitrary code execution",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nresult=$(echo \"$cmd_name\")",
            vuln: "#!/bin/sh\nresult=$(eval \"$cmd_name\")",
            desc: "eval in command substitution allows injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\necho \"Setting config: ${key}=${value}\"",
            vuln: "#!/bin/sh\neval \"${key}=${value}\"",
            desc: "eval of config values allows code injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n. /etc/app/config.sh",
            vuln: "#!/bin/sh\n. \"$CONFIG_PATH\"",
            desc: "source of variable path allows code injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nexport {VAR}=\"${{{VAR}}}\"",
            vuln: "#!/bin/sh\neval \"export {VAR}=${{1}}\"",
            desc: "eval of export with positional param allows injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s\\n' \"${msg}\"",
            vuln: "#!/bin/sh\neval echo \"${msg}\"",
            desc: "eval echo allows command substitution injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncase \"$action\" in\n  start) do_start;;\n  stop) do_stop;;\nesac",
            vuln: "#!/bin/sh\neval \"do_$action\"",
            desc: "eval of action name allows arbitrary function execution",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nawk -v pat=\"$pattern\" '$0 ~ pat' data.txt",
            vuln: "#!/bin/sh\neval \"grep $pattern data.txt\"",
            desc: "eval of grep pattern allows command injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nset -- \"$@\"\nfor arg do printf '%s\\n' \"$arg\"; done",
            vuln: "#!/bin/sh\neval \"echo $*\"",
            desc: "eval of $* allows injection via arguments",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nreadonly {VAR}=\"safe_value\"",
            vuln: "#!/bin/sh\neval \"{VAR}=${{untrusted}}\"",
            desc: "eval of variable assignment from untrusted source",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nif [ \"$mode\" = debug ]; then set -x; fi",
            vuln: "#!/bin/sh\neval \"set $mode_flags\"",
            desc: "eval of shell flags allows arbitrary option injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncommand -v \"${tool}\" >/dev/null",
            vuln: "#!/bin/sh\neval \"type ${tool}\" >/dev/null",
            desc: "eval of type command allows injection via tool name",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n[ -f \"${path}\" ] && cat \"${path}\"",
            vuln: "#!/bin/sh\neval \"cat ${path}\"",
            desc: "eval of cat with variable path allows injection",
        },
    ];

    let subs = &[Substitution {
        placeholder: "{VAR}",
        values: &[
            "LANG", "MODE", "CONFIG", "OUTPUT", "FORMAT", "LEVEL", "PREFIX", "SUFFIX", "NAME",
            "TYPE", "ENCODING", "STYLE",
        ],
    }];

    expand_templates(BASES, count, seed, subs, "CWE-94", 94, "Code Injection")
}

/// CWE-330: Insufficient Randomness ($RANDOM, $$)
#[allow(clippy::too_many_lines)]
fn generate_cwe_330(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\ntmpdir=$(mktemp -d)",
            vuln: "#!/bin/sh\ntmpdir=/tmp/work_$RANDOM",
            desc: "$RANDOM is predictable (15-bit LFSR), use mktemp",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ntoken=$(head -c 32 /dev/urandom | base64)",
            vuln: "#!/bin/sh\ntoken=session_$$",
            desc: "PID-based token is predictable",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nlog_file=$(mktemp /tmp/log.XXXXXX)",
            vuln: "#!/bin/sh\nlog_file=/tmp/log_$(date +%s)",
            desc: "Timestamp-based filename is predictable",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsalt=$(head -c 16 /dev/urandom | xxd -p)",
            vuln: "#!/bin/sh\nsalt=$RANDOM$RANDOM",
            desc: "Double $RANDOM is only 30 bits of entropy",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n{ITEM}=$(mktemp /tmp/{ITEM}.XXXXXX)",
            vuln: "#!/bin/sh\n{ITEM}=/tmp/{ITEM}_$RANDOM",
            desc: "$RANDOM-based {ITEM} filename is predictable",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsession_id=$(head -c 24 /dev/urandom | base64)",
            vuln: "#!/bin/sh\nsession_id=sess_$$_$RANDOM",
            desc: "PID+RANDOM session ID has only ~47 bits of entropy",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nnonce=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \\n')",
            vuln: "#!/bin/sh\nnonce=$(date +%s%N)",
            desc: "Nanosecond timestamp as nonce is predictable",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nport=$(shuf -i 49152-65535 -n 1)",
            vuln: "#!/bin/sh\nport=$((RANDOM + 1024))",
            desc: "$RANDOM port selection is predictable and biased",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncookie=$(head -c 32 /dev/urandom | base64 | tr -d '/+=')",
            vuln: "#!/bin/sh\ncookie=cookie_$(date +%s)_$$",
            desc: "Timestamp+PID cookie is trivially guessable",
        },
        BaseTemplate {
            safe: "#!/bin/sh\niv=$(openssl rand -hex 16)",
            vuln: "#!/bin/sh\niv=$(printf '%032x' $RANDOM)",
            desc: "$RANDOM IV is cryptographically weak",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nbackup_suffix=$(head -c 4 /dev/urandom | xxd -p)",
            vuln: "#!/bin/sh\nbackup_suffix=$(date +%H%M%S)",
            desc: "Time-based backup suffix is predictable",
        },
        BaseTemplate {
            safe: "#!/bin/sh\njob_id=$(uuidgen)",
            vuln: "#!/bin/sh\njob_id=job_$$",
            desc: "PID-based job ID is predictable and reused",
        },
        BaseTemplate {
            safe: "#!/bin/sh\napi_nonce=$(openssl rand -base64 18)",
            vuln: "#!/bin/sh\napi_nonce=$RANDOM$RANDOM$RANDOM",
            desc: "Triple $RANDOM is only 45 bits of entropy",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nworkdir=$(mktemp -d /tmp/{ITEM}.XXXXXX)",
            vuln: "#!/bin/sh\nworkdir=/tmp/{ITEM}_$$",
            desc: "PID-based workdir name is predictable and races",
        },
        BaseTemplate {
            safe: "#!/bin/sh\npasswd=$(head -c 18 /dev/urandom | base64)",
            vuln: "#!/bin/sh\npasswd=pw_$(date +%s | md5sum | head -c 12)",
            desc: "Hashing a timestamp does not add entropy",
        },
    ];

    let subs = &[Substitution {
        placeholder: "{ITEM}",
        values: &[
            "cache", "lock", "pid", "sock", "fifo", "spool", "data", "state", "run", "build",
            "deploy", "stage", "test", "bench", "upload", "download", "archive", "snapshot",
            "dump",
        ],
    }];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-330",
        330,
        "Insufficient Randomness",
    )
}

/// CWE-362: TOCTOU Race Condition (missing -p/-f flags)
#[allow(clippy::too_many_lines)]
fn generate_cwe_362(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\nmkdir -p \"${output_dir}\"",
            vuln: "#!/bin/sh\nmkdir \"${output_dir}\"",
            desc: "mkdir without -p fails if dir exists (not idempotent)",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nrm -f \"${lockfile}\"",
            vuln: "#!/bin/sh\nrm \"${lockfile}\"",
            desc: "rm without -f fails if file missing (not idempotent)",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nln -sf \"${target}\" \"${link_name}\"",
            vuln: "#!/bin/sh\nln -s \"${target}\" \"${link_name}\"",
            desc: "ln -s without -f fails if link exists",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nmkdir -p /var/run/app && chmod 755 /var/run/app",
            vuln: "#!/bin/sh\nmkdir /var/run/app && chmod 755 /var/run/app",
            desc: "mkdir without -p causes TOCTOU race on re-run",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nmkdir -p \"${DIR}/{SUBDIR}\"",
            vuln: "#!/bin/sh\nmkdir \"${DIR}/{SUBDIR}\"",
            desc: "mkdir without -p on {SUBDIR} fails if exists",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nrm -f \"${DIR}/{FNAME}\"",
            vuln: "#!/bin/sh\nrm \"${DIR}/{FNAME}\"",
            desc: "rm without -f on {FNAME} errors if absent",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nln -sf /usr/bin/{TOOL} /usr/local/bin/{TOOL}",
            vuln: "#!/bin/sh\nln -s /usr/bin/{TOOL} /usr/local/bin/{TOOL}",
            desc: "ln -s without -f for {TOOL} symlink fails on re-run",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncp -f \"${src}\" \"${dst}\"",
            vuln: "#!/bin/sh\nif [ ! -f \"${dst}\" ]; then cp \"${src}\" \"${dst}\"; fi",
            desc: "TOCTOU: check-then-copy races with concurrent access",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nmkdir -p /opt/{APP}/data /opt/{APP}/logs /opt/{APP}/tmp",
            vuln: "#!/bin/sh\nmkdir /opt/{APP}/data\nmkdir /opt/{APP}/logs\nmkdir /opt/{APP}/tmp",
            desc: "Multiple mkdir without -p for {APP} not idempotent",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ninstall -d -m 755 \"${cache_dir}\"",
            vuln: "#!/bin/sh\nif [ ! -d \"${cache_dir}\" ]; then mkdir \"${cache_dir}\"; chmod 755 \"${cache_dir}\"; fi",
            desc: "TOCTOU: test -d races with concurrent dir creation",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nmv -f \"${src}\" \"${dst}\"",
            vuln: "#!/bin/sh\nif [ ! -f \"${dst}\" ]; then mv \"${src}\" \"${dst}\"; fi",
            desc: "TOCTOU: existence check before mv races on busy systems",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nmkdir -p \"$HOME/.config/{APP}\"",
            vuln: "#!/bin/sh\nmkdir \"$HOME/.config/{APP}\"",
            desc: "mkdir without -p for user config dir not idempotent",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nrm -rf \"${build_dir}\" && mkdir -p \"${build_dir}\"",
            vuln: "#!/bin/sh\nif [ -d \"${build_dir}\" ]; then rm -r \"${build_dir}\"; fi\nmkdir \"${build_dir}\"",
            desc: "Check-then-remove-then-create has double TOCTOU race",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{SUBDIR}",
            values: &[
                "logs", "tmp", "cache", "data", "run", "state", "spool", "backup", "build",
                "deploy", "dist", "output", "staging",
            ],
        },
        Substitution {
            placeholder: "{FNAME}",
            values: &[
                "lockfile",
                "pidfile",
                "socket",
                "marker",
                "flag",
                "stamp",
                "token",
                "sentinel",
                "checkpoint",
                "progress",
            ],
        },
        Substitution {
            placeholder: "{TOOL}",
            values: &[
                "python3", "node", "ruby", "gcc", "clang", "go", "rustc", "cargo", "npm", "pip",
                "java", "perl",
            ],
        },
        Substitution {
            placeholder: "{APP}",
            values: &[
                "myapp",
                "webapp",
                "api",
                "worker",
                "scheduler",
                "monitor",
                "proxy",
                "gateway",
                "service",
                "daemon",
                "agent",
                "collector",
            ],
        },
        Substitution {
            placeholder: "{DIR}",
            values: &[
                "/var/run",
                "/var/tmp",
                "/opt/app",
                "/srv/data",
                "/usr/local/share",
                "/etc/app",
                "/home/deploy",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-362",
        362,
        "TOCTOU Race Condition",
    )
}

/// CWE-798: Hard-coded Credentials
#[allow(clippy::too_many_lines)]
fn generate_cwe_798(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\nDB_PASS=\"${DB_PASSWORD:?}\"",
            vuln: "#!/bin/sh\nDB_PASS=\"s3cret123\"",
            desc: "Hard-coded database password",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nAPI_KEY=\"$(vault kv get -field=key secret/api)\"",
            vuln: "#!/bin/sh\nAPI_KEY=\"sk-abc123def456\"",
            desc: "Hard-coded API key",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ntoken=\"${GITHUB_TOKEN:?}\"",
            vuln: "#!/bin/sh\ntoken=\"FAKE_TOKEN_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\"",
            desc: "Hard-coded GitHub token",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n{CRED_VAR}=\"${{{CRED_ENV}:?}}\"",
            vuln: "#!/bin/sh\n{CRED_VAR}=\"{CRED_VALUE}\"",
            desc: "Hard-coded {CRED_TYPE} credential",
        },
        BaseTemplate {
            safe: "#!/bin/sh\naws_key=$(aws configure get aws_access_key_id)",
            vuln: "#!/bin/sh\naws_key=\"FAKEKEYID00000EXAMPLE\"",
            desc: "Hard-coded AWS access key ID",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsecret=$(cat /run/secrets/db_password)",
            vuln: "#!/bin/sh\nsecret=\"P@ssw0rd!\"",
            desc: "Hard-coded secret instead of file-based secret",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nSMTP_PASS=\"$(pass show email/smtp)\"",
            vuln: "#!/bin/sh\nSMTP_PASS=\"mailpass123\"",
            desc: "Hard-coded SMTP password",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nSSH_KEY=\"${SSH_PRIVATE_KEY_PATH:?}\"",
            vuln: "#!/bin/sh\nSSH_KEY=\"-----BEGIN RSA PRIVATE KEY-----\"",
            desc: "Hard-coded SSH private key material",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nJWT_SECRET=\"$(vault kv get -field=jwt secret/auth)\"",
            vuln: "#!/bin/sh\nJWT_SECRET=\"super_secret_jwt_key_123\"",
            desc: "Hard-coded JWT signing secret",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nDB_URL=\"postgresql://$DB_USER:$DB_PASS@$DB_HOST/$DB_NAME\"",
            vuln: "#!/bin/sh\nDB_URL=\"postgresql://admin:password@db.example.com/prod\"",
            desc: "Hard-coded database connection string with credentials",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nSTRIPE_KEY=\"${STRIPE_SECRET_KEY:?}\"",
            vuln: "#!/bin/sh\nSTRIPE_KEY=\"EXAMPLE_FAKE_KEY_not_a_real_key\"",
            desc: "Hard-coded payment API key",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nENCRYPT_KEY=$(head -c 32 /dev/urandom | base64)",
            vuln: "#!/bin/sh\nENCRYPT_KEY=\"aGVsbG93b3JsZDEyMzQ1Njc4OTA=\"",
            desc: "Hard-coded encryption key",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nSLACK_WEBHOOK=\"${SLACK_WEBHOOK_URL:?}\"",
            vuln: "#!/bin/sh\nSLACK_WEBHOOK=\"https://hooks.slack.com/services/T00/B00/xxxx\"",
            desc: "Hard-coded Slack webhook URL",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nREDIS_PASS=\"${REDIS_PASSWORD:-}\"",
            vuln: "#!/bin/sh\nREDIS_PASS=\"redis_secret_42\"",
            desc: "Hard-coded Redis password",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{CRED_VAR}",
            values: &[
                "DB_PASS",
                "API_TOKEN",
                "SECRET_KEY",
                "AUTH_TOKEN",
                "OAUTH_SECRET",
                "DEPLOY_KEY",
                "ACCESS_TOKEN",
                "SERVICE_KEY",
                "MASTER_KEY",
                "ROOT_PASS",
                "ADMIN_TOKEN",
            ],
        },
        Substitution {
            placeholder: "{CRED_ENV}",
            values: &[
                "DB_PASSWORD",
                "API_TOKEN_SECRET",
                "SECRET_KEY_FILE",
                "AUTH_TOKEN_VAR",
                "OAUTH_CLIENT_SECRET",
                "DEPLOY_KEY_PATH",
                "ACCESS_TOKEN_ENV",
                "SERVICE_KEY_REF",
            ],
        },
        Substitution {
            placeholder: "{CRED_VALUE}",
            values: &[
                "xoxb-1234-5678-abcdef",
                "gho_xxxxxxxxxxxx",
                "FAKE_PAY_KEY_abcdef",
                "AIzaSyXXXXXXXXXXXXXXXXXXXXXXX",
                "dop_v1_xxxxx",
                "glpat-xxxxxxxxxxxxxxxxxxxx",
                "npm_xxxxxxxxxxxx",
                "pypi-AgEIcHlwaS5vcmc_xxxxx",
                "FAKECLOUD1234EXAMPLE",
                "FAKE_VCS_ABCDEFxxxxxxxxx",
                "sq0atp-xxxxxxxxxxxxxxxx",
            ],
        },
        Substitution {
            placeholder: "{CRED_TYPE}",
            values: &[
                "database",
                "API",
                "OAuth",
                "service",
                "deployment",
                "authentication",
                "encryption",
                "access",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-798",
        798,
        "Hard-coded Credentials",
    )
}

/// CWE-829: Inclusion of Untrusted Functionality (curl|bash)
#[allow(clippy::too_many_lines)]
fn generate_cwe_829(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\ncurl -fsSL https://example.com/install.sh -o /tmp/install.sh\nsha256sum -c /tmp/install.sh.sha256\nsh /tmp/install.sh",
            vuln: "#!/bin/sh\ncurl -fsSL https://example.com/install.sh | sh",
            desc: "Piping curl to shell bypasses integrity verification",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nwget -q https://example.com/setup.sh -O /tmp/setup.sh\nchmod +x /tmp/setup.sh\n/tmp/setup.sh",
            vuln: "#!/bin/sh\nwget -q -O- https://example.com/setup.sh | bash",
            desc: "Piping wget to bash allows MitM code execution",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncurl -fsSL https://{DOMAIN}/{SCRIPT} -o /tmp/{SCRIPT}\nsha256sum -c /tmp/{SCRIPT}.sha256\nsh /tmp/{SCRIPT}",
            vuln: "#!/bin/sh\ncurl -fsSL https://{DOMAIN}/{SCRIPT} | sh",
            desc: "Piping curl|sh for {SCRIPT} from {DOMAIN} bypasses verification",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncurl -fsSL {URL} -o /tmp/script.sh\ngpg --verify /tmp/script.sh.sig /tmp/script.sh\nsh /tmp/script.sh",
            vuln: "#!/bin/sh\ncurl -fsSL {URL} | bash -s --",
            desc: "Piping curl to bash -s bypasses GPG verification",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nwget -q {URL} -O /tmp/bootstrap.sh\nexpected_hash=\"abc123\"\nactual_hash=$(sha256sum /tmp/bootstrap.sh | cut -d' ' -f1)\n[ \"$expected_hash\" = \"$actual_hash\" ] && sh /tmp/bootstrap.sh",
            vuln: "#!/bin/sh\nwget -q -O- {URL} | sh -s",
            desc: "Piping wget to sh -s bypasses hash verification",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncurl -fsSL https://{DOMAIN}/get-{TOOL}.sh -o /tmp/get-{TOOL}.sh\nchmod +x /tmp/get-{TOOL}.sh\n/tmp/get-{TOOL}.sh",
            vuln: "#!/bin/sh\ncurl -fsSL https://{DOMAIN}/get-{TOOL}.sh | sudo sh",
            desc: "Piping curl to sudo sh for {TOOL} is extremely dangerous",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ngit clone --depth 1 https://github.com/{REPO}.git /tmp/repo\ncd /tmp/repo && make install",
            vuln: "#!/bin/sh\ncurl -fsSL https://raw.githubusercontent.com/{REPO}/main/install.sh | bash",
            desc: "Piping raw GitHub content to bash bypasses repo integrity",
        },
        BaseTemplate {
            safe: "#!/bin/sh\npip install {TOOL} --require-hashes -r requirements.txt",
            vuln: "#!/bin/sh\ncurl -fsSL https://bootstrap.pypa.io/get-{TOOL}.py | python3",
            desc: "Piping curl to python3 for {TOOL} bypasses package verification",
        },
        BaseTemplate {
            safe: "#!/bin/sh\napt-get install -y {TOOL}",
            vuln: "#!/bin/sh\ncurl -fsSL https://deb.{DOMAIN}/setup | sudo bash",
            desc: "Piping setup script to sudo bash instead of using package manager",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncurl -fsSL {URL} -o /tmp/agent.sh\nchmod +x /tmp/agent.sh\nsha256sum --check /tmp/agent.sh.sha256\n/tmp/agent.sh",
            vuln: "#!/bin/sh\ncurl {URL} | sh",
            desc: "Piping curl (without -f) to sh ignores HTTP errors and integrity",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsnap install {TOOL}",
            vuln: "#!/bin/sh\ncurl -sSL https://{DOMAIN}/install-{TOOL}.sh | sh -",
            desc: "curl|sh install of {TOOL} instead of snap package",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nnpm install -g {TOOL}",
            vuln: "#!/bin/sh\ncurl -fsSL https://raw.githubusercontent.com/{REPO}/main/install.sh | sh",
            desc: "curl|sh for npm tool {TOOL} bypasses registry verification",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{DOMAIN}",
            values: &[
                "get.docker.com",
                "install.python-poetry.org",
                "raw.githubusercontent.com",
                "sh.rustup.rs",
                "get.helm.sh",
                "deb.nodesource.com",
                "rpm.nodesource.com",
                "packages.gitlab.com",
                "cli.github.com",
                "apt.releases.hashicorp.com",
            ],
        },
        Substitution {
            placeholder: "{SCRIPT}",
            values: &[
                "install.sh",
                "setup.sh",
                "bootstrap.sh",
                "init.sh",
                "configure.sh",
                "deploy.sh",
                "update.sh",
                "migrate.sh",
            ],
        },
        Substitution {
            placeholder: "{TOOL}",
            values: &[
                "docker",
                "poetry",
                "rustup",
                "helm",
                "nvm",
                "rvm",
                "sdkman",
                "volta",
                "deno",
                "bun",
                "terraform",
                "kubectl",
            ],
        },
        Substitution {
            placeholder: "{REPO}",
            values: &[
                "nvm-sh/nvm",
                "pyenv/pyenv",
                "rbenv/rbenv",
                "asdf-vm/asdf",
                "junegunn/fzf",
                "ohmyzsh/ohmyzsh",
            ],
        },
        Substitution {
            placeholder: "{URL}",
            values: &[
                "https://get.docker.com",
                "https://install.python-poetry.org",
                "https://sh.rustup.rs",
                "https://get.helm.sh/helm-install.sh",
                "https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-829",
        829,
        "Inclusion of Untrusted Functionality",
    )
}

/// CWE-377: Insecure Temporary File
#[allow(clippy::too_many_lines)]
fn generate_cwe_377(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\ntmp=$(mktemp)",
            vuln: "#!/bin/sh\ntmp=/tmp/myapp_tmp",
            desc: "Predictable temp file allows symlink attack",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nworkdir=$(mktemp -d)",
            vuln: "#!/bin/sh\nworkdir=/tmp/workdir",
            desc: "Predictable temp directory allows pre-creation attack",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n{ITEM}=$(mktemp /tmp/{ITEM}.XXXXXX)",
            vuln: "#!/bin/sh\n{ITEM}=/tmp/{ITEM}_file",
            desc: "Predictable {ITEM} temp file allows symlink race",
        },
        BaseTemplate {
            safe:
                "#!/bin/sh\ntmpdir=$(mktemp -d /tmp/{APP}.XXXXXX)\ntrap 'rm -rf \"$tmpdir\"' EXIT",
            vuln: "#!/bin/sh\ntmpdir=/tmp/{APP}\nmkdir -p \"$tmpdir\"",
            desc: "Predictable {APP} temp dir with no cleanup trap",
        },
        BaseTemplate {
            safe: "#!/bin/sh\noutput=$(mktemp)\n{CMD} > \"$output\"",
            vuln: "#!/bin/sh\noutput=/tmp/output.txt\n{CMD} > \"$output\"",
            desc: "Predictable output file for {CMD} allows data theft",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfifo=$(mktemp -u /tmp/fifo.XXXXXX)\nmkfifo -m 600 \"$fifo\"",
            vuln: "#!/bin/sh\nfifo=/tmp/myfifo\nmkfifo \"$fifo\"",
            desc: "Predictable FIFO name with default perms allows hijack",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsock=$(mktemp -u /tmp/sock.XXXXXX)\ntrap 'rm -f \"$sock\"' EXIT",
            vuln: "#!/bin/sh\nsock=/tmp/app.sock",
            desc: "Predictable socket path allows pre-bind attack",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncache=$(mktemp /tmp/cache.XXXXXX)\nchmod 600 \"$cache\"",
            vuln: "#!/bin/sh\ncache=/tmp/cache\ntouch \"$cache\"",
            desc: "Predictable cache file with default permissions",
        },
        BaseTemplate {
            safe:
                "#!/bin/sh\nlock=$(mktemp /tmp/lock.XXXXXX)\ntrap 'rm -f \"$lock\"' EXIT INT TERM",
            vuln: "#!/bin/sh\nlock=/tmp/app.lock",
            desc: "Predictable lock file allows denial of service",
        },
        BaseTemplate {
            safe:
                "#!/bin/sh\npipe=$(mktemp -u)\nmkfifo -m 600 \"$pipe\"\ntrap 'rm -f \"$pipe\"' EXIT",
            vuln: "#!/bin/sh\npipe=/tmp/datapipe\nmkfifo \"$pipe\"",
            desc: "Predictable named pipe allows data interception",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nstaging=$(mktemp -d /tmp/deploy.XXXXXX)",
            vuln: "#!/bin/sh\nstaging=/tmp/deploy_staging",
            desc: "Predictable staging dir allows content substitution",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nlog=$(mktemp /tmp/log.XXXXXX)\nexec > \"$log\" 2>&1",
            vuln: "#!/bin/sh\nlog=/tmp/script.log\nexec > \"$log\" 2>&1",
            desc: "Predictable log file allows output interception",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ndump=$(mktemp /tmp/dump.XXXXXX)\n{CMD} > \"$dump\"",
            vuln: "#!/bin/sh\n{CMD} > /tmp/dump.out",
            desc: "Predictable dump file for {CMD} allows data theft",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{ITEM}",
            values: &[
                "report", "config", "state", "session", "data", "audit", "trace", "diff", "patch",
                "manifest", "index", "digest",
            ],
        },
        Substitution {
            placeholder: "{APP}",
            values: &[
                "myapp", "deploy", "builder", "tester", "monitor", "agent", "backup", "restore",
                "migrate", "updater",
            ],
        },
        Substitution {
            placeholder: "{CMD}",
            values: &[
                "pg_dump",
                "mysqldump",
                "tar czf -",
                "df -h",
                "ps aux",
                "env",
                "printenv",
                "ip addr",
                "netstat -tlnp",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-377",
        377,
        "Insecure Temporary File",
    )
}

/// CWE-732: Incorrect Permission Assignment
#[allow(clippy::too_many_lines)]
fn generate_cwe_732(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 600 /etc/app/secrets.conf",
            vuln: "#!/bin/sh\nchmod 777 /etc/app/secrets.conf",
            desc: "World-writable permissions on secrets file",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 700 /var/run/app",
            vuln: "#!/bin/sh\nchmod 666 /var/run/app",
            desc: "World-readable/writable runtime directory",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 600 \"${SECRET_FILE}\"",
            vuln: "#!/bin/sh\nchmod 644 \"${SECRET_FILE}\"",
            desc: "World-readable permissions on secret file",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ninstall -m 600 {FNAME} /etc/{APP}/",
            vuln: "#!/bin/sh\ncp {FNAME} /etc/{APP}/\nchmod 777 /etc/{APP}/{FNAME}",
            desc: "World-writable {FNAME} for {APP} allows tampering",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 700 \"$HOME/.ssh\"",
            vuln: "#!/bin/sh\nchmod 755 \"$HOME/.ssh\"",
            desc: "SSH directory readable by others leaks key metadata",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 600 \"$HOME/.ssh/id_rsa\"",
            vuln: "#!/bin/sh\nchmod 644 \"$HOME/.ssh/id_rsa\"",
            desc: "World-readable SSH private key",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 640 /etc/{APP}/{FNAME}\nchown root:{APP} /etc/{APP}/{FNAME}",
            vuln: "#!/bin/sh\nchmod 666 /etc/{APP}/{FNAME}",
            desc: "World-writable {FNAME} for {APP} instead of group ownership",
        },
        BaseTemplate {
            safe: "#!/bin/sh\numask 077\ntouch /tmp/sensitive_output",
            vuln: "#!/bin/sh\numask 000\ntouch /tmp/sensitive_output",
            desc: "umask 000 creates world-writable files by default",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ninstall -m 750 -d /var/log/{APP}",
            vuln: "#!/bin/sh\nmkdir /var/log/{APP}\nchmod 777 /var/log/{APP}",
            desc: "World-writable log directory for {APP}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 400 /etc/ssl/private/{FNAME}",
            vuln: "#!/bin/sh\nchmod 644 /etc/ssl/private/{FNAME}",
            desc: "World-readable TLS private key {FNAME}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 640 /etc/shadow",
            vuln: "#!/bin/sh\nchmod 644 /etc/shadow",
            desc: "World-readable shadow file exposes password hashes",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 600 \"$HOME/.{APP}rc\"",
            vuln: "#!/bin/sh\nchmod 666 \"$HOME/.{APP}rc\"",
            desc: "World-writable {APP} config file allows manipulation",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchmod 755 /usr/local/bin/{TOOL}",
            vuln: "#!/bin/sh\nchmod 777 /usr/local/bin/{TOOL}",
            desc: "World-writable binary {TOOL} allows replacement",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{APP}",
            values: &[
                "myapp",
                "nginx",
                "postgres",
                "redis",
                "grafana",
                "prometheus",
                "jenkins",
                "vault",
                "consul",
                "nomad",
                "traefik",
                "caddy",
            ],
        },
        Substitution {
            placeholder: "{FNAME}",
            values: &[
                "secrets.conf",
                "credentials.yml",
                "tls.key",
                "auth.json",
                "db.conf",
                "api_keys.env",
                "token.pem",
                "password.txt",
                "cert.key",
                "config.ini",
            ],
        },
        Substitution {
            placeholder: "{TOOL}",
            values: &[
                "deploy.sh",
                "backup.sh",
                "healthcheck",
                "cron-job",
                "service-wrapper",
                "init-script",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-732",
        732,
        "Incorrect Permission Assignment",
    )
}

// ===========================================================================
// Out-of-Distribution CWEs (eval-only)
// ===========================================================================

/// CWE-426: Untrusted Search Path
#[allow(clippy::too_many_lines)]
fn generate_cwe_426(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\n/usr/bin/python3 script.py",
            vuln: "#!/bin/sh\npython3 script.py",
            desc: "Relative command path allows PATH hijacking",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nPATH=/usr/bin:/bin\nexport PATH\ngcc -o app app.c",
            vuln: "#!/bin/sh\nPATH=.:$PATH\nexport PATH\ngcc -o app app.c",
            desc: "Current directory in PATH allows trojan binary execution",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n/bin/ls /var/log",
            vuln: "#!/bin/sh\nls /var/log",
            desc: "Relative 'ls' can be hijacked via PATH manipulation",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nLD_LIBRARY_PATH=/usr/lib\nexport LD_LIBRARY_PATH\n./app",
            vuln: "#!/bin/sh\nLD_LIBRARY_PATH=.:/usr/lib\nexport LD_LIBRARY_PATH\n./app",
            desc: "Current dir in LD_LIBRARY_PATH allows library hijack",
        },
        BaseTemplate {
            safe: "#!/bin/sh\n/usr/bin/{TOOL} {ARGS}",
            vuln: "#!/bin/sh\n{TOOL} {ARGS}",
            desc: "Relative {TOOL} path allows PATH hijacking",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nPATH=/usr/bin:/usr/local/bin:/bin\nexport PATH\n{TOOL} {ARGS}",
            vuln: "#!/bin/sh\nPATH=/tmp:$PATH\nexport PATH\n{TOOL} {ARGS}",
            desc: "Writable /tmp in PATH allows trojan {TOOL}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncommand -v /usr/bin/{TOOL} >/dev/null && /usr/bin/{TOOL} {ARGS}",
            vuln: "#!/bin/sh\nwhich {TOOL} && {TOOL} {ARGS}",
            desc: "'which' follows PATH order, may find trojan {TOOL}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nPATH=/usr/bin:/bin\nexport PATH\nLD_LIBRARY_PATH=/usr/lib:/lib\nexport LD_LIBRARY_PATH",
            vuln: "#!/bin/sh\n# PATH and LD_LIBRARY_PATH inherited from caller",
            desc: "Inherited PATH/LD_LIBRARY_PATH may contain untrusted directories",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nPATH=/usr/bin:/bin exec /usr/bin/{TOOL}",
            vuln: "#!/bin/sh\nexec {TOOL}",
            desc: "exec without resetting PATH allows {TOOL} hijack",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nenv -i PATH=/usr/bin:/bin HOME=\"$HOME\" /usr/bin/{TOOL}",
            vuln: "#!/bin/sh\nenv PATH=\".:$PATH\" {TOOL}",
            desc: "Passing dot-PATH through env for {TOOL}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nPYTHONPATH=/usr/lib/python3/dist-packages\nexport PYTHONPATH\npython3 app.py",
            vuln: "#!/bin/sh\nPYTHONPATH=.:$PYTHONPATH\nexport PYTHONPATH\npython3 app.py",
            desc: "Current dir in PYTHONPATH allows malicious module import",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nNODE_PATH=/usr/lib/node_modules\nexport NODE_PATH\nnode app.js",
            vuln: "#!/bin/sh\nNODE_PATH=./node_modules:$NODE_PATH\nexport NODE_PATH\nnode app.js",
            desc: "Relative NODE_PATH allows malicious package injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nPERL5LIB=/usr/share/perl5\nexport PERL5LIB\nperl script.pl",
            vuln: "#!/bin/sh\nPERL5LIB=.:$PERL5LIB\nexport PERL5LIB\nperl script.pl",
            desc: "Current dir in PERL5LIB allows malicious module loading",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{TOOL}",
            values: &[
                "gcc", "make", "git", "curl", "tar", "gzip", "python3", "node", "ruby", "perl",
                "java", "go", "rustc", "clang", "awk", "sed",
            ],
        },
        Substitution {
            placeholder: "{ARGS}",
            values: &[
                "--version",
                "-h",
                "input.txt",
                "-o output",
                "--config cfg.yml",
                "-f Makefile",
                "src/main.c",
                "app.py",
                "script.rb",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-426",
        426,
        "Untrusted Search Path",
    )
}

/// CWE-77: Command Injection (xargs, indirect execution)
#[allow(clippy::too_many_lines)]
fn generate_cwe_77(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\nfind . -name '*.txt' -print0 | xargs -0 rm",
            vuln: "#!/bin/sh\nfind . -name '*.txt' | xargs rm",
            desc: "xargs without -0 allows filename injection via newlines/spaces",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind /tmp -type f -print0 | xargs -0 chmod 644",
            vuln: "#!/bin/sh\nfind /tmp -type f | xargs chmod 644",
            desc: "Missing -print0/-0 allows injection via crafted filenames",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s\\0' \"$@\" | xargs -0 grep pattern",
            vuln: "#!/bin/sh\necho \"$@\" | xargs grep pattern",
            desc: "Word-splitting user args through xargs without null delimiter",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nwhile IFS= read -r file; do\n  process \"$file\"\ndone < filelist.txt",
            vuln: "#!/bin/sh\nfor file in $(cat filelist.txt); do\n  process $file\ndone",
            desc: "Command substitution + unquoted var allows injection via filenames",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind \"{DIR}\" -name '*.{EXT}' -print0 | xargs -0 {CMD}",
            vuln: "#!/bin/sh\nfind \"{DIR}\" -name '*.{EXT}' | xargs {CMD}",
            desc: "xargs without -0 on {EXT} files allows filename injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind . -type f -print0 | xargs -0 -I{} cp {} backup/",
            vuln: "#!/bin/sh\nfind . -type f | xargs -I{} cp {} backup/",
            desc: "xargs -I without -0 allows injection via crafted filenames",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nwhile IFS= read -r -d '' line; do\n  echo \"$line\"\ndone < <(find . -print0)",
            vuln: "#!/bin/sh\nfor line in $(find .); do\n  echo $line\ndone",
            desc: "for-in with find output splits on spaces and glob-expands",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind . -name '*.{EXT}' -exec {CMD} {} +",
            vuln: "#!/bin/sh\n{CMD} $(find . -name '*.{EXT}')",
            desc: "Command substitution of find splits filenames on whitespace",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind \"{DIR}\" -maxdepth 1 -print0 | xargs -0 ls -la",
            vuln: "#!/bin/sh\nls -la $(find \"{DIR}\" -maxdepth 1)",
            desc: "Command substitution of find in ls allows glob injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind . -name '*.{EXT}' -print0 | xargs -0 wc -l",
            vuln: "#!/bin/sh\nwc -l $(find . -name '*.{EXT}')",
            desc: "Unquoted find substitution in wc allows filename injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind . -type f -newer reference -print0 | xargs -0 {CMD}",
            vuln: "#!/bin/sh\n{CMD} $(find . -type f -newer reference)",
            desc: "Unquoted find in {CMD} splits filenames with spaces",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nwhile IFS= read -r name; do\n  process \"$name\"\ndone < names.txt",
            vuln: "#!/bin/sh\nprocess $(cat names.txt)",
            desc: "Unquoted cat substitution word-splits and glob-expands",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind . -name '*.bak' -print0 | xargs -0 rm -f",
            vuln: "#!/bin/sh\nrm -f $(find . -name '*.bak')",
            desc: "Unquoted find in rm allows deletion of unintended files",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{DIR}",
            values: &[
                "/tmp",
                "/var/log",
                "/home/user",
                "/opt/data",
                "/srv/uploads",
                "/usr/local/share",
                "/var/spool",
            ],
        },
        Substitution {
            placeholder: "{EXT}",
            values: &[
                "log", "txt", "csv", "json", "xml", "yaml", "conf", "bak", "tmp", "dat",
            ],
        },
        Substitution {
            placeholder: "{CMD}",
            values: &[
                "rm",
                "chmod 644",
                "chown root",
                "gzip",
                "sha256sum",
                "wc -l",
                "head -1",
                "cat",
                "file",
                "stat",
            ],
        },
    ];

    expand_templates(BASES, count, seed, subs, "CWE-77", 77, "Command Injection")
}

/// CWE-116: Improper Output Encoding (log injection)
#[allow(clippy::too_many_lines)]
fn generate_cwe_116(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s\\n' \"$user_input\" >> /var/log/app.log",
            vuln: "#!/bin/sh\necho $user_input >> /var/log/app.log",
            desc: "Unquoted echo allows log injection via newlines and escape sequences",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf 'User: %s Action: %s\\n' \"$user\" \"$action\" >> audit.log",
            vuln: "#!/bin/sh\necho \"User: $user Action: $action\" >> audit.log",
            desc: "Unvalidated user/action fields allow log forging",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s\\n' \"${msg}\" | tee -a output.log",
            vuln: "#!/bin/sh\necho -e \"$msg\" | tee -a output.log",
            desc: "echo -e interprets escape sequences from untrusted input",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '[%s] %s: %s\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"$level\" \"$msg\" >> {LOGFILE}",
            vuln: "#!/bin/sh\necho \"[$(date)] $level: $msg\" >> {LOGFILE}",
            desc: "Unquoted log fields allow injection into {LOGFILE}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf 'IP=%s Method=%s Path=%s\\n' \"$ip\" \"$method\" \"$path\" >> access.log",
            vuln: "#!/bin/sh\necho \"IP=$ip Method=$method Path=$path\" >> access.log",
            desc: "Unvalidated HTTP fields allow access log forging",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s\\n' \"$hostname\" >> /var/log/{LOGFILE}",
            vuln: "#!/bin/sh\necho -e $hostname >> /var/log/{LOGFILE}",
            desc: "echo -e of hostname allows escape sequence injection in {LOGFILE}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf 'ERROR: %s (code=%s)\\n' \"$errmsg\" \"$errcode\" >&2",
            vuln: "#!/bin/sh\necho \"ERROR: $errmsg (code=$errcode)\" >&2",
            desc: "Unvalidated error message allows stderr log injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf 'Event: %s\\n' \"${event}\" >> /var/log/{LOGFILE}",
            vuln: "#!/bin/sh\necho -e \"Event: ${event}\" >> /var/log/{LOGFILE}",
            desc: "echo -e of event data injects control chars into {LOGFILE}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nlogger -t myapp -- \"$msg\"",
            vuln: "#!/bin/sh\nlogger -t myapp $msg",
            desc: "Unquoted syslog message allows field splitting and injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s|%s|%s\\n' \"$timestamp\" \"$user\" \"$query\" >> {LOGFILE}",
            vuln: "#!/bin/sh\necho \"$timestamp|$user|$query\" >> {LOGFILE}",
            desc: "Pipe-delimited log with unvalidated fields allows {LOGFILE} forging",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '{\"user\":\"%s\",\"action\":\"%s\"}\\n' \"$user\" \"$action\" >> {LOGFILE}",
            vuln: "#!/bin/sh\necho '{\"user\":\"'$user'\",\"action\":\"'$action'\"}' >> {LOGFILE}",
            desc: "JSON log with unescaped fields allows structure injection",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf 'AUDIT: %s performed %s on %s\\n' \"$who\" \"$what\" \"$target\" >> {LOGFILE}",
            vuln: "#!/bin/sh\necho \"AUDIT: $who performed $what on $target\" >> {LOGFILE}",
            desc: "Unvalidated audit fields allow log record forgery",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nprintf '%s\\n' \"$filename\" | sed 's/[^a-zA-Z0-9._-]/_/g'",
            vuln: "#!/bin/sh\necho $filename",
            desc: "Unsanitized filename echo allows terminal escape injection",
        },
    ];

    let subs = &[Substitution {
        placeholder: "{LOGFILE}",
        values: &[
            "app.log",
            "audit.log",
            "access.log",
            "error.log",
            "auth.log",
            "security.log",
            "debug.log",
            "event.log",
            "transaction.log",
            "query.log",
            "activity.log",
        ],
    }];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-116",
        116,
        "Improper Output Encoding",
    )
}

/// CWE-250: Execution with Unnecessary Privileges
#[allow(clippy::too_many_lines)]
fn generate_cwe_250(count: usize, seed: u64) -> Vec<CweMutation> {
    static BASES: &[BaseTemplate] = &[
        BaseTemplate {
            safe: "#!/bin/sh\ninstall -m 644 config.conf /etc/app/",
            vuln: "#!/bin/sh\nsudo cp config.conf /etc/app/",
            desc: "Unnecessary sudo for file copy when install suffices",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsu -c 'systemctl restart app' appuser",
            vuln: "#!/bin/sh\nsudo systemctl restart app",
            desc: "Running as root when specific user suffices",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nchown appuser:appgroup /var/run/app.pid",
            vuln: "#!/bin/sh\nsudo chmod 777 /var/run/app.pid",
            desc: "Using sudo + world-writable instead of proper ownership",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncap_add NET_BIND_SERVICE /usr/bin/app",
            vuln: "#!/bin/sh\nsudo /usr/bin/app",
            desc: "Running entire app as root instead of adding specific capability",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsu -c '{CMD}' {USER}",
            vuln: "#!/bin/sh\nsudo {CMD}",
            desc: "Unnecessary sudo for {CMD} when {USER} suffices",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nrunuser -u {USER} -- {CMD}",
            vuln: "#!/bin/sh\nsudo -u root {CMD}",
            desc: "Running {CMD} as root instead of {USER}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsetcap cap_net_bind_service=+ep /usr/bin/{TOOL}",
            vuln: "#!/bin/sh\nsudo /usr/bin/{TOOL}",
            desc: "Running {TOOL} as root instead of granting specific capability",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ninstall -o {USER} -g {USER} -m 644 app.conf /etc/{APP}/",
            vuln: "#!/bin/sh\nsudo cp app.conf /etc/{APP}/\nsudo chmod 777 /etc/{APP}/app.conf",
            desc: "sudo + chmod 777 for {APP} config instead of proper install",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsudo -u {USER} {CMD}",
            vuln: "#!/bin/sh\nsudo {CMD}",
            desc: "Running {CMD} as root instead of least-privilege {USER}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsu -c 'systemctl restart {APP}' {USER}",
            vuln: "#!/bin/sh\nsudo systemctl restart {APP}",
            desc: "Restarting {APP} as root instead of service user",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nsetpriv --reuid={USER} --regid={USER} --clear-groups {CMD}",
            vuln: "#!/bin/sh\nsudo su -c '{CMD}'",
            desc: "sudo su chain for {CMD} instead of setpriv",
        },
        BaseTemplate {
            safe: "#!/bin/sh\nfind /var/log/{APP} -name '*.log' -mtime +30 -delete",
            vuln: "#!/bin/sh\nsudo find / -name '*.log' -mtime +30 -delete",
            desc: "sudo find from / instead of scoped log cleanup for {APP}",
        },
        BaseTemplate {
            safe: "#!/bin/sh\ncrontab -u {USER} -l | grep -q '{CMD}' || echo '0 2 * * * {CMD}' | crontab -u {USER} -",
            vuln: "#!/bin/sh\nsudo crontab -l | grep -q '{CMD}' || echo '0 2 * * * sudo {CMD}' | sudo crontab -",
            desc: "Root crontab with sudo instead of user-specific crontab",
        },
    ];

    let subs = &[
        Substitution {
            placeholder: "{CMD}",
            values: &[
                "service nginx reload",
                "pg_dump mydb",
                "tail -f /var/log/app.log",
                "journalctl -u app",
                "kill -HUP $(cat /var/run/app.pid)",
                "logrotate /etc/logrotate.d/app",
                "certbot renew",
            ],
        },
        Substitution {
            placeholder: "{USER}",
            values: &[
                "appuser", "www-data", "nobody", "daemon", "postgres", "redis", "nginx", "deploy",
                "service", "monitor",
            ],
        },
        Substitution {
            placeholder: "{TOOL}",
            values: &[
                "nginx",
                "node",
                "python3",
                "java",
                "caddy",
                "envoy",
                "haproxy",
                "prometheus",
            ],
        },
        Substitution {
            placeholder: "{APP}",
            values: &[
                "myapp",
                "webapp",
                "api",
                "worker",
                "scheduler",
                "monitor",
                "proxy",
                "gateway",
                "backend",
                "frontend",
            ],
        },
    ];

    expand_templates(
        BASES,
        count,
        seed,
        subs,
        "CWE-250",
        250,
        "Execution with Unnecessary Privileges",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_cwe_78_mutations() {
        let mutations = generate_cwe_78(5, 42);
        assert!(!mutations.is_empty());
        assert!(mutations.len() <= 5);
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-78");
            assert_eq!(m.cwe_id, 78);
            assert!(!m.safe_script.is_empty());
            assert!(!m.unsafe_script.is_empty());
            assert_ne!(m.safe_script, m.unsafe_script);
        }
    }

    #[test]
    fn test_generate_cwe_94_mutations() {
        let mutations = generate_cwe_94(3, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-94");
            assert!(m.unsafe_script.contains("eval") || m.unsafe_script.contains(". \"$"));
        }
    }

    #[test]
    fn test_generate_ood_cwe_426() {
        let mutations = generate_cwe_426(4, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-426");
            assert_eq!(m.cwe_id, 426);
        }
    }

    #[test]
    fn test_generate_ood_cwe_77() {
        let mutations = generate_cwe_77(4, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-77");
        }
    }

    #[test]
    fn test_generate_ood_cwe_116() {
        let mutations = generate_cwe_116(3, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-116");
        }
    }

    #[test]
    fn test_generate_ood_cwe_250() {
        let mutations = generate_cwe_250(4, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-250");
        }
    }

    #[test]
    fn test_generate_cwe_mutations_mixed() {
        let mutations = generate_cwe_mutations(&[78, 94, 426, 77], 20, 42);
        assert!(!mutations.is_empty());
        assert!(mutations.len() <= 20);
        // Should have multiple CWE types
        let cwes: std::collections::HashSet<&str> =
            mutations.iter().map(|m| m.cwe.as_str()).collect();
        assert!(cwes.len() > 1);
    }

    #[test]
    fn test_in_distribution_cwes() {
        let ids = in_distribution_cwes();
        assert!(ids.contains(&78));
        assert!(ids.contains(&94));
        assert!(ids.contains(&330));
        assert!(ids.contains(&362));
    }

    #[test]
    fn test_ood_cwes() {
        let ids = ood_cwes();
        assert!(ids.contains(&426));
        assert!(ids.contains(&77));
        assert!(ids.contains(&116));
        assert!(ids.contains(&250));
    }

    #[test]
    fn test_ood_disjoint_from_in_distribution() {
        let id = in_distribution_cwes();
        let ood = ood_cwes();
        for cwe in &ood {
            assert!(
                !id.contains(cwe),
                "OOD CWE {cwe} should not be in-distribution"
            );
        }
    }

    #[test]
    fn test_cwe_mutation_serialization() {
        let mutations = generate_cwe_78(1, 42);
        let json = serde_json::to_string(&mutations[0]).expect("serialize");
        assert!(json.contains("CWE-78"));
        assert!(json.contains("safe_script"));
        assert!(json.contains("unsafe_script"));
    }

    // --- New tests for parameterized expansion ---

    #[test]
    fn test_xorshift64_deterministic() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_xorshift64_zero_seed() {
        let mut rng = Xorshift64::new(0);
        // Should not degenerate (non-zero state)
        let v = rng.next();
        assert_ne!(v, 0);
    }

    #[test]
    fn test_expand_templates_basic() {
        let bases = &[BaseTemplate {
            safe: "safe {X}",
            vuln: "vuln {X}",
            desc: "desc {X}",
        }];
        let subs = &[Substitution {
            placeholder: "{X}",
            values: &["alpha", "beta", "gamma"],
        }];
        let results = expand_templates(bases, 3, 42, subs, "CWE-999", 999, "Test");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(!r.safe_script.contains("{X}"));
            assert!(!r.unsafe_script.contains("{X}"));
        }
    }

    #[test]
    fn test_expand_templates_uniqueness() {
        let bases = &[BaseTemplate {
            safe: "safe {X} {Y}",
            vuln: "vuln {X} {Y}",
            desc: "desc",
        }];
        let subs = &[
            Substitution {
                placeholder: "{X}",
                values: &["a", "b", "c", "d", "e"],
            },
            Substitution {
                placeholder: "{Y}",
                values: &["1", "2", "3", "4", "5"],
            },
        ];
        let results = expand_templates(bases, 20, 42, subs, "CWE-999", 999, "Test");
        // All entries should be unique
        let mut seen = std::collections::HashSet::new();
        for r in &results {
            let key = format!("{}||{}", r.safe_script, r.unsafe_script);
            assert!(seen.insert(key), "Duplicate mutation found");
        }
    }

    #[test]
    fn test_cwe_78_can_generate_100() {
        let mutations = generate_cwe_78(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-78 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_94_can_generate_100() {
        let mutations = generate_cwe_94(100, 42);
        assert!(
            mutations.len() >= 30,
            "Expected >=30 CWE-94 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_330_can_generate_100() {
        let mutations = generate_cwe_330(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-330 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_362_can_generate_100() {
        let mutations = generate_cwe_362(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-362 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_798_can_generate_100() {
        let mutations = generate_cwe_798(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-798 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_829_can_generate_100() {
        let mutations = generate_cwe_829(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-829 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_377_can_generate_100() {
        let mutations = generate_cwe_377(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-377 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_732_can_generate_100() {
        let mutations = generate_cwe_732(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-732 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_426_can_generate_100() {
        let mutations = generate_cwe_426(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-426 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_77_can_generate_100() {
        let mutations = generate_cwe_77(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-77 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_116_can_generate_100() {
        let mutations = generate_cwe_116(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-116 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_cwe_250_can_generate_100() {
        let mutations = generate_cwe_250(100, 42);
        assert!(
            mutations.len() >= 50,
            "Expected >=50 CWE-250 mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_generate_all_cwes_500() {
        let all_cwes: Vec<u32> = {
            let mut v = in_distribution_cwes();
            v.extend(ood_cwes());
            v
        };
        let mutations = generate_cwe_mutations(&all_cwes, 500, 42);
        assert!(
            mutations.len() >= 400,
            "Expected >=400 total mutations, got {}",
            mutations.len()
        );
    }

    #[test]
    fn test_deterministic_output() {
        let m1 = generate_cwe_78(50, 42);
        let m2 = generate_cwe_78(50, 42);
        assert_eq!(m1.len(), m2.len());
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.safe_script, b.safe_script);
            assert_eq!(a.unsafe_script, b.unsafe_script);
        }
    }

    #[test]
    fn test_different_seeds_different_output() {
        let m1 = generate_cwe_78(50, 42);
        let m2 = generate_cwe_78(50, 99);
        // After the first ~14 base templates (which are deterministic), variants should differ
        let mut any_different = false;
        for (a, b) in m1.iter().zip(m2.iter()) {
            if a.safe_script != b.safe_script {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different output"
        );
    }

    #[test]
    fn test_safe_unsafe_always_differ() {
        let all_cwes: Vec<u32> = {
            let mut v = in_distribution_cwes();
            v.extend(ood_cwes());
            v
        };
        let mutations = generate_cwe_mutations(&all_cwes, 200, 42);
        for m in &mutations {
            assert_ne!(
                m.safe_script, m.unsafe_script,
                "Safe and unsafe must differ for {} - {}",
                m.cwe, m.mutation_description
            );
        }
    }

    #[test]
    fn test_all_mutations_have_shebang() {
        let all_cwes: Vec<u32> = {
            let mut v = in_distribution_cwes();
            v.extend(ood_cwes());
            v
        };
        let mutations = generate_cwe_mutations(&all_cwes, 200, 42);
        for m in &mutations {
            assert!(
                m.safe_script.starts_with("#!/bin/sh"),
                "Safe script missing shebang: {}",
                &m.safe_script[..m.safe_script.len().min(40)]
            );
            // Some unsafe scripts intentionally have comment-only bodies
            // but should still start with shebang
            assert!(
                m.unsafe_script.starts_with("#!/bin/sh"),
                "Unsafe script missing shebang: {}",
                &m.unsafe_script[..m.unsafe_script.len().min(40)]
            );
        }
    }
}
