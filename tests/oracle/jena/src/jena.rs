//! Apache Jena 5.6.0, pinned by the sha256 of its binary distribution, run on a JVM the harness checks itself.
//!
//! - The zip is fetched once into the cache and verified by sha256 on EVERY run, then unpacked into the run's
//!   own fresh work dir, so the jars that run are the ones the digest covers. A wrong digest is NOT MEASURED
//!   (FALSIFY-CRUXSHACL-023), never a re-download that is trusted.
//! - The JVM is `ONT_ORACLE_JAVA`, else `java` on PATH; its `-version` must be 17 or newer (Jena 5 needs 17).
//!   Jena's own `bin/` scripts are never used: they follow `JAVA_HOME`, which on the measuring host named a
//!   Java 11 and failed with `UnsupportedClassVersionError`. No JVM, or an old one, is NOT MEASURED
//!   (FALSIFY-CRUXSHACL-022). `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS`, `JDK_JAVA_OPTIONS` and `CLASSPATH` are
//!   cleared so the environment cannot change what runs.
//! - `shacl validate` prints its report as Turtle and has no output-format flag, so the report is read back
//!   through `riot --output=nt`.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::nt::{self, Triple};

pub const JENA_VERSION: &str = "5.6.0";
pub const JENA_URL: &str = "https://archive.apache.org/dist/jena/binaries/apache-jena-5.6.0.zip";
pub const JENA_ZIP_SHA256: &str =
    "fb26f8753ed3c4f8e04607a60d9605e18cf6e22143a47fc7334dcd87f655d888";
/// The SHACL-SHACL graph Jena ships inside its pinned `jena-shacl` jar (subject S-SHSH).
pub const SHSH_IN_JAR: &str = "std/shacl-shacl.ttl";
const MIN_JAVA: u32 = 17;

pub fn sha256(path: &Path) -> Option<String> {
    let out = Command::new("sha256sum").arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(String::from)
}

/// `Err` unless the file's sha256 is `want`.
pub fn verify_pin(path: &Path, want: &str) -> Result<(), String> {
    match sha256(path) {
        Some(got) if got == want => Ok(()),
        Some(got) => Err(format!("{} sha256 is {got}, pinned {want}", path.display())),
        None => Err(format!("{} unreadable", path.display())),
    }
}

/// The major version in a `java -version` first line: `openjdk version "17.0.16"` → 17, `"1.8.0_x"` → 8.
pub fn java_major(line: &str) -> Option<u32> {
    let v = line.split('"').nth(1)?;
    let mut it = v.split(['.', '-', '+', '_']);
    let first: u32 = it.next()?.parse().ok()?;
    if first == 1 {
        it.next()?.parse().ok()
    } else {
        Some(first)
    }
}

/// The JVM to use and its `-version` line. `Err` is NOT MEASURED.
pub fn find_java(cmd: &str) -> Result<String, String> {
    let out = Command::new(cmd)
        .arg("-version")
        .env_remove("JAVA_TOOL_OPTIONS")
        .env_remove("_JAVA_OPTIONS")
        .env_remove("JDK_JAVA_OPTIONS")
        .output()
        .map_err(|e| format!("no JVM: `{cmd} -version`: {e}"))?;
    let line = String::from_utf8_lossy(&out.stderr)
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    if !out.status.success() {
        return Err(format!(
            "no JVM: `{cmd} -version` exited {:?}",
            out.status.code()
        ));
    }
    match java_major(&line) {
        Some(m) if m >= MIN_JAVA => Ok(line),
        Some(m) => Err(format!(
            "JVM too old: `{cmd}` is Java {m}, Jena {JENA_VERSION} needs {MIN_JAVA}+ ({line})"
        )),
        None => Err(format!("cannot read the Java version from {line:?}")),
    }
}

fn cache_dir() -> Result<PathBuf, String> {
    let c = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or("no cache dir")?
        .join(format!("ont-oracle/jena-{JENA_VERSION}"));
    std::fs::create_dir_all(&c).map_err(|e| format!("{}: {e}", c.display()))?;
    Ok(c)
}

/// The pinned zip in the cache, fetched if absent or wrong. `Err` is NOT MEASURED.
pub fn pinned_zip() -> Result<PathBuf, String> {
    let zip = cache_dir()?.join(format!("apache-jena-{JENA_VERSION}.zip"));
    if verify_pin(&zip, JENA_ZIP_SHA256).is_err() {
        let ok = Command::new("curl")
            .args(["-sfL", "-o"])
            .arg(&zip)
            .arg(JENA_URL)
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            return Err(format!("{JENA_URL} unreachable"));
        }
    }
    verify_pin(&zip, JENA_ZIP_SHA256).map(|()| zip)
}

/// `HH:MM:SS LEVEL logger :: text`, the form of Jena's command-line log lines.
pub fn is_log(l: &str) -> bool {
    let b = l.as_bytes();
    b.len() > 9
        && b[8] == b' '
        && b[..8].iter().enumerate().all(|(i, c)| {
            if i == 2 || i == 5 {
                *c == b':'
            } else {
                c.is_ascii_digit()
            }
        })
        && ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
            .iter()
            .any(|lv| l[9..].trim_start().starts_with(lv))
}

/// Jena's command-line logging prints to STDOUT, inside the output graph: riot's `ERROR riot :: [line: 7, …]`
/// came on stdout with stderr empty (measured 2026-10-09). Reading stderr alone, a strict read was never strict.
/// Splits stdout into `(output, log lines)`; a log line is never parsed as RDF.
pub fn split_log(stdout: &str) -> (String, Vec<String>) {
    let mut out = String::new();
    let mut log = Vec::new();
    for l in stdout.lines() {
        if is_log(l) {
            log.push(l.to_string());
        } else {
            out.push_str(l);
            out.push('\n');
        }
    }
    (out, log)
}

/// A Jena command rejected its input: an ERROR line, or a failing exit that no WARN explains.
fn rejected(exit_ok: bool, clean: bool, diag: &str) -> bool {
    diag.lines().any(|l| l.contains("ERROR")) || (!exit_ok && clean)
}

pub struct Jena {
    java: String,
    lib: PathBuf,
}

/// A fresh, exclusively created work dir (a planted path or symlink makes `create_dir` fail, never reuse).
pub fn fresh_dir(tag: &str) -> Result<PathBuf, String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let d = std::env::temp_dir().join(format!("jena-oracle-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&d)
        .map_err(|e| format!("cannot create a fresh work dir {}: {e}", d.display()))?;
    Ok(d)
}

impl Jena {
    /// Unpack the pinned zip into `work` (its digest re-checked first) on a JVM [`find_java`] accepted.
    pub fn open(java_cmd: &str, zip: &Path, work: &Path) -> Result<Jena, String> {
        verify_pin(zip, JENA_ZIP_SHA256)?;
        let ok = Command::new("unzip")
            .arg("-qo")
            .arg(zip)
            .arg("-d")
            .arg(work)
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            return Err("unzip of the pinned zip failed".into());
        }
        let lib = work.join(format!("apache-jena-{JENA_VERSION}/lib"));
        if !lib.join(format!("jena-shacl-{JENA_VERSION}.jar")).is_file() {
            return Err(format!("{} has no jena-shacl jar", lib.display()));
        }
        Ok(Jena {
            java: java_cmd.to_string(),
            lib,
        })
    }

    fn cmd(&self, main: &str) -> Command {
        let mut c = Command::new(&self.java);
        c.env_remove("JAVA_TOOL_OPTIONS")
            .env_remove("_JAVA_OPTIONS")
            .env_remove("JDK_JAVA_OPTIONS")
            .env_remove("CLASSPATH")
            .arg("-cp")
            .arg(format!("{}/*", self.lib.display()))
            .arg(main);
        c
    }

    /// `riot --output=nt [--base=B] FILE`: `(exit ok, clean, N-Triples, diagnostics)`. The diagnostics are the
    /// log lines split out of stdout, then stderr; `clean` is no WARN or ERROR among them.
    pub fn riot(
        &self,
        file: &Path,
        base: Option<&str>,
    ) -> Result<(bool, bool, String, String), String> {
        let mut c = self.cmd("riotcmd.riot");
        c.arg("--output=nt");
        if let Some(b) = base {
            c.arg(format!("--base={b}"));
        }
        let o = c.arg(file).output().map_err(|e| format!("java: {e}"))?;
        let (out, mut diag) = split_log(&String::from_utf8_lossy(&o.stdout));
        diag.extend(String::from_utf8_lossy(&o.stderr).lines().map(String::from));
        let clean = !diag
            .iter()
            .any(|l| l.contains("WARN") || l.contains("ERROR"));
        Ok((o.status.success(), clean, out, diag.join("\n")))
    }

    /// The graph Jena reads from a file, as triples. `Err` when Jena rejects it, or, if `strict`, warns about it
    /// (a W3C case with a deliberately ill-formed literal is read with `strict` off). riot exits 1 on a WARN
    /// alone (measured), so the exit status cannot tell the two apart: an ERROR line, or a failing exit with no
    /// diagnostic at all, is a rejection.
    pub fn read(
        &self,
        file: &Path,
        base: Option<&str>,
        strict: bool,
    ) -> Result<Vec<Triple>, String> {
        let (ok, clean, nt_text, err) = self.riot(file, base)?;
        if rejected(ok, clean, &err) || (strict && !clean) {
            let first = err
                .lines()
                .find(|l| l.contains("WARN") || l.contains("ERROR"))
                .or(err.lines().next());
            return Err(format!(
                "riot {}: {}",
                file.display(),
                first.unwrap_or("non-zero exit")
            ));
        }
        nt::parse(&nt_text).map_err(|e| format!("riot's N-Triples for {}: {e}", file.display()))
    }

    /// `shacl validate`, its report read back as triples.
    pub fn validate(
        &self,
        shapes: &Path,
        data: &Path,
        work: &Path,
        tag: &str,
    ) -> Result<Vec<Triple>, String> {
        let o = self
            .cmd("shacl.shacl")
            .arg("validate")
            .arg("--shapes")
            .arg(shapes)
            .arg("--data")
            .arg(data)
            .output()
            .map_err(|e| format!("java: {e}"))?;
        let (report, mut diag) = split_log(&String::from_utf8_lossy(&o.stdout));
        diag.extend(String::from_utf8_lossy(&o.stderr).lines().map(String::from));
        let clean = !diag
            .iter()
            .any(|l| l.contains("WARN") || l.contains("ERROR"));
        if rejected(o.status.success(), clean, &diag.join("\n")) || report.trim().is_empty() {
            let first = diag
                .iter()
                .find(|l| l.contains("ERROR"))
                .or(diag.first())
                .map_or("", String::as_str);
            return Err(format!(
                "shacl validate exited {:?}: {first}",
                o.status.code()
            ));
        }
        let ttl = work.join(format!("report-{tag}.ttl"));
        std::fs::write(&ttl, report).map_err(|e| format!("{}: {e}", ttl.display()))?;
        // A report may echo an ill-formed literal from the data as sh:value; that is not the report's fault.
        self.read(&ttl, None, false)
    }

    /// `jena.rdfcompare a b` on two N-Triples files: Jena's own isomorphism test (exit 0 equal, 1 unequal).
    pub fn rdfcompare(&self, a: &Path, b: &Path) -> Result<bool, String> {
        let o = self
            .cmd("jena.rdfcompare")
            .arg(a)
            .arg(b)
            .args(["N-TRIPLES", "N-TRIPLES"])
            .output()
            .map_err(|e| format!("java: {e}"))?;
        match o.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            c => Err(format!("rdfcompare exited {c:?}")),
        }
    }

    /// Extract the SHACL-SHACL graph from the pinned jena-shacl jar into `work`.
    pub fn shsh(&self, work: &Path) -> Result<PathBuf, String> {
        let jar = self.lib.join(format!("jena-shacl-{JENA_VERSION}.jar"));
        let o = Command::new("unzip")
            .arg("-p")
            .arg(&jar)
            .arg(SHSH_IN_JAR)
            .output()
            .map_err(|e| format!("unzip: {e}"))?;
        if !o.status.success() || o.stdout.is_empty() {
            return Err(format!("{SHSH_IN_JAR} not in {}", jar.display()));
        }
        let p = work.join("shacl-shacl.ttl");
        std::fs::write(&p, &o.stdout).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(p)
    }
}
