//! The command line of `apr plot` (feature `plot-cli`): `--data file.csv --aes x=a,y=b`, read
//! into [`PlotArgs`] and drawn by [`super::run`]. Each argument carries its role type, which
//! `apr surface` reads. Only what [`super::run`] draws is accepted: the `point` geom with `x` and
//! `y` mapped; any other geom or aesthetic is refused by name with the supported values.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use batuta_common::cli_roles::{FreeText, InputFile};

use super::{Env, Output, PlotArgs, PlotError};

/// The geoms [`super::run`] draws.
const GEOMS: &[&str] = &["point"];
/// The aesthetics [`super::run`] maps.
const AESTHETICS: &[&str] = &["x", "y"];

/// Arguments of `apr plot`.
#[derive(Debug, Clone, clap::Args)]
pub struct PlotFlags {
    /// CSV file with a header row (`#` lines are skipped). Values are zero or have a magnitude
    /// from 1e-5 up to but excluding 1e16, the span whose axis labels were compared with ggplot2
    #[arg(long, required_unless_present = "spec")]
    pub data: Option<InputFile>,

    /// Aesthetic mappings, e.g. x=height,y=weight
    #[arg(long, required_unless_present = "spec", conflicts_with = "spec")]
    pub aes: Option<FreeText>,

    /// A ggplot expression, e.g. "ggplot(df, aes(x, y)) + geom_point()", or a YAML file (name
    /// ending .yaml/.yml) with `data`, `aes` and `layers` (instead of --aes/--geom)
    #[arg(long)]
    pub spec: Option<FreeText>,

    /// Geom of the single layer, one of the proven geoms [default: point]
    #[arg(long)]
    pub geom: Option<FreeText>,

    /// Width: cells for term, pixels for svg
    #[arg(long)]
    pub width: Option<u32>,

    /// Height: lines for term, pixels for svg
    #[arg(long)]
    pub height: Option<u32>,

    /// Render target
    #[arg(long, value_enum, default_value_t = Output::Term)]
    pub output: Output,

    /// Never emit colour escape sequences
    #[arg(long)]
    pub no_color: bool,

    /// Facet as `wrap=column`: one panel per value of the column
    #[arg(long)]
    pub facet: Option<FreeText>,
    /// Horizontal reference line at this y (repeatable)
    #[arg(long, allow_negative_numbers = true)]
    pub hline: Vec<f64>,

    /// Vertical reference line at this x (repeatable)
    #[arg(long, allow_negative_numbers = true)]
    pub vline: Vec<f64>,

    /// Reference line y = slope * x + intercept, as slope,intercept (repeatable)
    #[arg(long, allow_hyphen_values = true)]
    pub abline: Vec<FreeText>,

    /// Plot title
    #[arg(long)]
    pub title: Option<FreeText>,

    /// X axis label
    #[arg(long)]
    pub xlab: Option<FreeText>,

    /// Y axis label
    #[arg(long)]
    pub ylab: Option<FreeText>,
}

/// Parse `x=col,y=col` into a mapping.
///
/// # Errors
///
/// A pair without `=`, an empty side, or the same aesthetic twice.
pub fn parse_aes(s: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for pair in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (k, v) = pair
            .split_once('=')
            .map(|(k, v)| (k.trim(), v.trim()))
            .filter(|(k, v)| !k.is_empty() && !v.is_empty())
            .ok_or_else(|| format!("--aes: expected aesthetic=column, got '{pair}'"))?;
        if out.insert(k.to_string(), v.to_string()).is_some() {
            return Err(format!("--aes: aesthetic '{k}' is mapped twice"));
        }
    }
    Ok(out)
}

/// The facet forms [`super::run`] draws.
const FACETS: &[&str] = &["wrap"];

/// `wrap=column` into the column; any other form is refused by name.
fn parse_facet(s: &str) -> Result<String, String> {
    let (form, col) = s.split_once('=').map_or((s, ""), |(f, c)| (f.trim(), c.trim()));
    if !FACETS.contains(&form) {
        return Err(refuse("facet", form, FACETS));
    }
    if col.is_empty() {
        return Err("--facet: expected wrap=column".into());
    }
    Ok(col.to_string())
}

/// Parse `slope,intercept`.
///
/// # Errors
///
/// Not exactly two comma-separated finite numbers.
pub fn parse_abline(s: &str) -> Result<(f64, f64), String> {
    let bad = || format!("--abline: expected slope,intercept (two numbers), got '{s}'");
    let (m, b) = s.split_once(',').ok_or_else(bad)?;
    let num = |t: &str| t.trim().parse::<f64>().ok().filter(|v| v.is_finite());
    Ok((num(m).ok_or_else(bad)?, num(b).ok_or_else(bad)?))
}

/// A cursor over a ggplot expression; every error names the 1-based character position.
struct Cursor<'a> {
    chars: Vec<char>,
    at: usize,
    _src: &'a str,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self { chars: src.chars().collect(), at: 0, _src: src }
    }

    fn fail<T>(&self, want: &str) -> Result<T, String> {
        let found =
            self.chars.get(self.at).map_or("end of input".to_string(), |c| format!("'{c}'"));
        Err(format!("--spec: position {}: expected {want}, found {found}", self.at + 1))
    }

    fn ws(&mut self) {
        while self.chars.get(self.at).is_some_and(|c| c.is_whitespace()) {
            self.at += 1;
        }
    }

    fn eat(&mut self, c: char) -> Result<(), String> {
        self.ws();
        if self.chars.get(self.at) == Some(&c) {
            self.at += 1;
            Ok(())
        } else {
            self.fail(&format!("'{c}'"))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        self.ws();
        let start = self.at;
        while self.chars.get(self.at).is_some_and(|c| c.is_alphanumeric() || "_.".contains(*c)) {
            self.at += 1;
        }
        if start == self.at {
            return self.fail("a name");
        }
        Ok(self.chars[start..self.at].iter().collect())
    }

    fn keyword(&mut self, word: &str) -> Result<(), String> {
        self.ws();
        let start = self.at;
        let got = self.ident()?;
        if got == word {
            Ok(())
        } else {
            self.at = start;
            self.fail(&format!("'{word}'"))
        }
    }

    /// `col` or `x = col` / `y = col`; a positional argument takes `slot`'s aesthetic.
    fn aes_arg(&mut self, slot: &str) -> Result<(String, String), String> {
        let first = self.ident()?;
        self.ws();
        if self.chars.get(self.at) == Some(&'=') {
            self.at += 1;
            Ok((first, self.ident()?))
        } else {
            Ok((slot.to_string(), first))
        }
    }
}

/// Parse `ggplot(df, aes(x, y)) + geom_point()` (positional or `x = col` arguments) into the
/// geom name and the `x`/`y` column mapping.
///
/// # Errors
///
/// Anything outside that shape, with the 1-based character position of the first bad token.
pub fn parse_spec(src: &str) -> Result<(String, BTreeMap<String, String>), String> {
    let mut c = Cursor::new(src);
    c.keyword("ggplot")?;
    c.eat('(')?;
    c.ident()?;
    c.eat(',')?;
    c.keyword("aes")?;
    c.eat('(')?;
    let mut aes = BTreeMap::new();
    for (i, slot) in ["x", "y"].into_iter().enumerate() {
        if i == 1 {
            c.eat(',')?;
        }
        let (k, v) = c.aes_arg(slot)?;
        if aes.insert(k.clone(), v).is_some() {
            return Err(format!("--spec: position {}: aesthetic '{k}' is mapped twice", c.at));
        }
    }
    c.eat(')')?;
    c.eat(')')?;
    c.eat('+')?;
    c.ws();
    let geom = c.ident()?;
    let Some(geom) = geom.strip_prefix("geom_").map(str::to_string) else {
        c.at -= geom.chars().count();
        return c.fail("a geom_ call");
    };
    c.eat('(')?;
    c.eat(')')?;
    c.ws();
    if c.at < c.chars.len() {
        return c.fail("end of input");
    }
    Ok((geom, aes))
}

/// The YAML spec. A key the verb cannot draw is named in [`parse_spec_file`], not ignored.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSpec {
    data: Option<PathBuf>,
    aes: BTreeMap<String, String>,
    layers: Vec<FileLayer>,
    // Keys of the older spec this verb does not draw yet: read only to be refused.
    scales: Option<serde::de::IgnoredAny>,
    coord: Option<serde::de::IgnoredAny>,
    facet: Option<serde::de::IgnoredAny>,
    labs: Option<serde::de::IgnoredAny>,
    theme: Option<serde::de::IgnoredAny>,
}

/// One layer of the YAML spec: only its geom is drawn.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileLayer {
    geom: String,
}

/// Whether a `--spec` value names a YAML file rather than holding a ggplot expression.
fn is_yaml_path(spec: &str) -> bool {
    Path::new(spec)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"))
}

/// Read a YAML spec into the geom, the aesthetics and its `data:` (relative to the spec file).
///
/// # Errors
///
/// An unreadable or malformed file, an unknown key or layer key, a key the verb does not draw
/// (`scales`, `coord`, `facet`, `labs`, `theme`), or other than exactly one layer.
pub fn parse_spec_file(
    path: &Path,
) -> Result<(String, BTreeMap<String, String>, Option<PathBuf>), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read spec {}: {e}", path.display()))?;
    let spec: FileSpec = serde_saphyr::from_str(&text)
        .map_err(|e| format!("invalid plot spec {}: {e}", path.display()))?;
    let drawn = [
        ("scales", spec.scales.is_some()),
        ("coord", spec.coord.is_some()),
        ("facet", spec.facet.is_some()),
        ("labs", spec.labs.is_some()),
        ("theme", spec.theme.is_some()),
    ];
    if let Some((key, _)) = drawn.iter().find(|(_, set)| *set) {
        return Err(format!(
            "spec {}: '{key}' is not drawn by this verb yet; supported keys: data, aes, layers",
            path.display()
        ));
    }
    let [layer] = <[FileLayer; 1]>::try_from(spec.layers).map_err(|l| {
        format!("spec {}: expected exactly one layer, got {}", path.display(), l.len())
    })?;
    let data = spec.data.map(|d| path.parent().unwrap_or(Path::new("")).join(d));
    Ok((layer.geom, spec.aes, data))
}

fn refuse(what: &str, got: &str, supported: &[&str]) -> String {
    format!("unsupported {what} '{got}'; supported: {}", supported.join(", "))
}

impl PlotFlags {
    /// The library arguments these flags ask for.
    ///
    /// # Errors
    ///
    /// A geom other than `point`, a malformed `--aes`, an aesthetic other than `x` and `y`, or
    /// no mapping for `x` or `y`.
    pub fn plot_args(&self) -> Result<PlotArgs, String> {
        let mut spec_data = None;
        let (geom, mut aes) = match (&self.spec, &self.aes) {
            (Some(spec), _) if is_yaml_path(spec) => {
                let (g, a, d) = parse_spec_file(Path::new(&**spec))?;
                spec_data = d;
                (g, a)
            }
            (Some(spec), _) => parse_spec(spec)?,
            (None, aes) => (
                self.geom.as_deref().unwrap_or("point").to_string(),
                parse_aes(aes.as_deref().unwrap_or_default())?,
            ),
        };
        if !GEOMS.contains(&geom.as_str()) {
            return Err(refuse("geom", &geom, GEOMS));
        }
        if let Some(other) = aes.keys().find(|a| !AESTHETICS.contains(&a.as_str())) {
            return Err(refuse("aesthetic", other, AESTHETICS));
        }
        let mut mapped = |a: &str| {
            aes.remove(a).ok_or_else(|| format!("aes needs a mapping for '{a}' (e.g. {a}=column)"))
        };
        let (x, y) = (mapped("x")?, mapped("y")?);
        let facet = self.facet.as_deref().map(parse_facet).transpose()?;
        let finite = |flag: &str, v: &[f64]| {
            v.iter()
                .all(|x| x.is_finite())
                .then(|| v.to_vec())
                .ok_or_else(|| format!("{flag}: value must be a finite number"))
        };
        Ok(PlotArgs {
            data: match (&self.data, spec_data) {
                (Some(d), _) => d.clone().into_path_buf(),
                (None, Some(d)) => d,
                (None, None) => return Err("no data: give --data or `data:` in the spec".into()),
            },
            x,
            y,
            width: self.width,
            height: self.height,
            output: self.output,
            no_color: self.no_color,
            facet,
            hlines: finite("--hline", &self.hline)?,
            vlines: finite("--vline", &self.vline)?,
            ablines: self.abline.iter().map(|a| parse_abline(a)).collect::<Result<_, _>>()?,
            title: self.title.as_deref().map(str::to_string),
            xlab: self.xlab.as_deref().map(str::to_string),
            ylab: self.ylab.as_deref().map(str::to_string),
        })
    }
}

/// Run `apr plot`: read the flags, then draw with [`super::run`].
///
/// # Errors
///
/// [`PlotError::Usage`] (exit 2) for a refusal of [`PlotFlags::plot_args`] or a size out of
/// bounds; [`PlotError::Failed`] (exit 1) for a failure of the input or the output.
pub fn run(
    flags: &PlotFlags,
    env: &Env,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), PlotError> {
    super::run(&flags.plot_args().map_err(PlotError::Usage)?, env, stdout, stderr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::path::PathBuf;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        plot: PlotFlags,
    }

    fn parse(argv: &[&str]) -> Result<PlotFlags, clap::Error> {
        Cli::try_parse_from(std::iter::once("plot").chain(argv.iter().copied())).map(|c| c.plot)
    }

    fn flags(aes: &str, extra: &[&str]) -> PlotFlags {
        let mut v = vec!["--data", "d.csv", "--aes", aes];
        v.extend_from_slice(extra);
        parse(&v).expect("parses")
    }

    #[test]
    fn args_parse_with_defaults() {
        let a = parse(&["--data", "d.csv", "--aes", "x=a,y=b"]).expect("parses");
        assert_eq!(a.output, Output::Term);
        assert!(a.geom.is_none() && !a.no_color && a.width.is_none() && a.height.is_none());
        let extra: Vec<&str> = "--geom point --width 60 --height 20 --output svg --no-color"
            .split_whitespace()
            .collect();
        let a = flags("x=a,y=b", &extra);
        assert_eq!((a.width, a.height, a.output), (Some(60), Some(20), Output::Svg));
        assert!(a.no_color && a.geom.as_deref() == Some("point"));
    }

    #[test]
    fn args_reject_missing_and_unknown() {
        assert!(parse(&[]).is_err(), "needs --data and --aes");
        assert!(parse(&["--data", "d.csv"]).is_err(), "needs --aes");
        assert!(parse(&["--aes", "x=a,y=b"]).is_err(), "needs --data");
        assert!(parse(&["--data", "d", "--aes", "x=a,y=b", "--output", "png"]).is_err());
        assert!(parse(&["--data", "d", "--aes", "x=a,y=b", "--spec", "p"]).is_err(), "exclusive");
    }

    #[test]
    fn aes_parses_and_rejects_malformed() {
        let m = parse_aes(" x = a , y=b ,").expect("parses");
        assert_eq!(m.get("x").map(String::as_str), Some("a"));
        assert_eq!(m.get("y").map(String::as_str), Some("b"));
        assert_eq!(m.len(), 2);
        let e = parse_aes("x").expect_err("no =");
        assert_eq!(e, "--aes: expected aesthetic=column, got 'x'");
        assert!(parse_aes("x=").is_err());
        assert!(parse_aes("=a").is_err());
        assert_eq!(
            parse_aes("x=a,x=b").expect_err("twice"),
            "--aes: aesthetic 'x' is mapped twice"
        );
    }

    #[test]
    fn spec_parses_and_names_the_position_of_a_malformed_one() {
        let (g, m) = parse_spec(" ggplot(df, aes(h, w)) + geom_point() ").expect("parses");
        assert_eq!((g.as_str(), m["x"].as_str(), m["y"].as_str()), ("point", "h", "w"));
        let (_, m) = parse_spec("ggplot(df, aes(y = w, x = h)) + geom_point()").expect("named");
        assert_eq!((m["x"].as_str(), m["y"].as_str()), ("h", "w"));
        let (_, m) = parse_spec("ggplot(df, aes(h.cm, w_2)) + geom_point()").expect("dotted");
        assert_eq!((m["x"].as_str(), m["y"].as_str()), ("h.cm", "w_2"));
        let bad = |s: &str| parse_spec(s).expect_err("refused");
        assert_eq!(
            bad("ggplot(df, aes(x, y) + geom_point()"),
            "--spec: position 22: expected ')', found '+'"
        );
        assert_eq!(
            bad("ggplot(df, aes(x y)) + geom_point()"),
            "--spec: position 18: expected ',', found 'y'"
        );
        assert_eq!(
            bad("ggplot(df, aes(x, y))"),
            "--spec: position 22: expected '+', found end of input"
        );
        assert_eq!(
            bad("ggplot(df, aes(x, y)) + point()"),
            "--spec: position 25: expected a geom_ call, found 'p'"
        );
        assert!(bad("ggplot(df, aes(x, y)) + geom_point() x").contains("position 38"));
        assert!(bad("plot(df)").contains("position 1"));
    }

    #[test]
    fn spec_renders_and_a_bad_one_is_refused_before_reading() {
        let dir = std::env::temp_dir().join(format!("plot-cli-spec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let csv = dir.join("d.csv");
        std::fs::write(&csv, "x,y\n1.5,-2.25\n2.25,0.5\n3,1.75\n").expect("write csv");
        let path = csv.to_str().expect("utf8");
        let go = |spec: &str| {
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let f = parse(&["--data", path, "--output", "svg", "--spec", spec]).expect("parses");
            (
                run(&f, &Env::default(), &mut out, &mut err),
                String::from_utf8_lossy(&out).into_owned(),
            )
        };
        let (r, svg) = go("ggplot(df, aes(x, y)) + geom_point()");
        assert_eq!(r, Ok(()));
        assert!(svg.starts_with("<svg"), "{svg:.200}");
        let (r, out) = go("ggplot(df, aes(x, y) + geom_point()");
        assert_eq!(r, Err("--spec: position 22: expected ')', found '+'".into()));
        assert!(out.is_empty());
        let (r, _) = go("ggplot(df, aes(x, y)) + geom_line()");
        assert_eq!(r, Err("unsupported geom 'line'; supported: point".into()));
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }

    fn yaml_in(name: &str, body: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("plot-cli-yaml-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        std::fs::write(dir.join("d.csv"), "x,y\n1.5,-2.25\n2.25,0.5\n3,1.75\n").expect("csv");
        let spec = dir.join("p.yaml");
        std::fs::write(&spec, body).expect("spec");
        (dir, spec)
    }

    fn draw_yaml(spec: &Path, extra: &[&str]) -> (Result<(), String>, String) {
        let s = spec.to_str().expect("utf8");
        let mut argv = vec!["--output", "svg", "--spec", s];
        argv.extend_from_slice(extra);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let r = run(&parse(&argv).expect("parses"), &Env::default(), &mut out, &mut err);
        (r, String::from_utf8_lossy(&out).into_owned())
    }

    #[test]
    fn yaml_spec_draws_data_aes_and_a_point_layer() {
        let (dir, spec) = yaml_in(
            "ok",
            "data: d.csv\naes:\n  x: x\n  y: y\nlayers:\n  - geom: point\nfacet: null\n",
        );
        let (r, svg) = draw_yaml(&spec, &[]);
        assert_eq!(r, Ok(()));
        assert!(svg.starts_with("<svg"), "{svg:.200}");
        // --data wins over the spec's data:, and a spec without data: needs --data.
        let csv = dir.join("d.csv");
        let (r, _) = draw_yaml(&spec, &["--data", csv.to_str().expect("utf8")]);
        assert_eq!(r, Ok(()));
        std::fs::write(&spec, "aes:\n  x: x\n  y: y\nlayers:\n  - geom: point\n").expect("w");
        let (r, _) = draw_yaml(&spec, &[]);
        assert_eq!(r, Err("no data: give --data or `data:` in the spec".into()));
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }

    #[test]
    fn yaml_spec_refuses_what_is_not_drawn_and_the_aes_and_geom_rules_still_hold() {
        let head = "data: d.csv\naes:\n  x: x\n  y: y\nlayers:\n  - geom: point\n";
        for (extra, key) in [
            ("labs:\n  title: T\n", "labs"),
            ("scales:\n  x: log10\n", "scales"),
            ("coord:\n  type: cartesian\n", "coord"),
            ("facet: wrap\n", "facet"),
            ("theme: bw\n", "theme"),
        ] {
            let (dir, spec) = yaml_in(key, &format!("{head}{extra}"));
            let (r, out) = draw_yaml(&spec, &[]);
            let e = r.expect_err("refused");
            assert!(e.contains(&format!("'{key}' is not drawn")), "{e}");
            assert!(out.is_empty());
            std::fs::remove_dir_all(&dir).expect("rm tmp dir");
        }
        let (dir, spec) =
            yaml_in("geom", "data: d.csv\naes:\n  x: x\n  y: y\nlayers:\n  - geom: line\n");
        assert_eq!(
            draw_yaml(&spec, &[]).0,
            Err("unsupported geom 'line'; supported: point".into())
        );
        let two = "data: d.csv\naes:\n  x: x\n  y: y\nlayers:\n  - geom: point\n  - geom: point\n";
        std::fs::write(&spec, two).expect("w");
        assert!(draw_yaml(&spec, &[]).0.expect_err("two").contains("exactly one layer, got 2"));
        let stat = "data: d.csv\naes:\n  x: x\n  y: y\nlayers:\n  - geom: point\n    stat: bin\n";
        std::fs::write(&spec, stat).expect("w");
        assert!(draw_yaml(&spec, &[]).0.expect_err("stat").contains("invalid plot spec"));
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }

    #[test]
    fn yaml_spec_malformed_or_missing_is_refused() {
        let (dir, spec) = yaml_in("bad", "data: [unclosed\n");
        let e = draw_yaml(&spec, &[]).0.expect_err("malformed");
        assert!(e.starts_with("invalid plot spec "), "{e}");
        std::fs::write(&spec, "bogus: 1\n").expect("w");
        assert!(draw_yaml(&spec, &[]).0.expect_err("unknown key").contains("invalid plot spec"));
        std::fs::remove_file(&spec).expect("rm");
        assert!(draw_yaml(&spec, &[]).0.expect_err("missing").contains("cannot read spec"));
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }

    #[test]
    fn flags_map_onto_the_library_arguments() {
        let a = flags(" y = w , x = h ", &["--width", "60", "--height", "20", "--no-color"]);
        let want = PlotArgs {
            data: PathBuf::from("d.csv"),
            x: "h".into(),
            y: "w".into(),
            width: Some(60),
            height: Some(20),
            output: Output::Term,
            no_color: true,
            ..PlotArgs::default()
        };
        assert_eq!(a.plot_args(), Ok(want));
        let svg = flags("x=a,y=b", &["--output", "svg", "--geom", "point"]).plot_args();
        assert_eq!(svg.map(|p| (p.output, p.no_color)), Ok((Output::Svg, false)));
    }

    #[test]
    fn facet_wrap_draws_strips_and_other_forms_are_refused() {
        let dir = std::env::temp_dir().join(format!("plot-cli-facet-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let csv = dir.join("f.csv");
        std::fs::write(&csv, "x,y,g\n1,2,b\n2,3,a\n3,1,b\n4,5,c\n").expect("write csv");
        let path = csv.to_str().expect("utf8");
        let go = |facet: &str| {
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let f =
                parse(&["--data", path, "--aes", "x=x,y=y", "--output", "svg", "--facet", facet])
                    .expect("parses");
            let r = run(&f, &Env::default(), &mut out, &mut err);
            (r, String::from_utf8_lossy(&out).into_owned())
        };
        let (r, svg) = go("wrap=g");
        assert_eq!(r, Ok(()));
        assert_eq!(svg.matches(r#"fill="rgb(217,217,217)""#).count(), 3, "{svg}");
        for level in [">a<", ">b<", ">c<"] {
            assert!(svg.contains(level), "{level}");
        }
        assert_eq!(go("grid=g").0, Err("unsupported facet 'grid'; supported: wrap".into()));
        assert!(go("wrap=zz").0.expect_err("unknown").contains("unknown column 'zz'"));
        assert!(go("wrap=").0.is_err());
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }

    #[test]
    fn only_the_drawn_geom_and_aesthetics_are_accepted() {
        let refused =
            |aes: &str, extra: &[&str]| flags(aes, extra).plot_args().expect_err("refused");
        assert_eq!(
            refused("x=a,y=b", &["--geom", "line"]),
            "unsupported geom 'line'; supported: point"
        );
        assert_eq!(
            refused("x=a,y=b,colour=c", &[]),
            "unsupported aesthetic 'colour'; supported: x, y"
        );
        assert_eq!(refused("y=b", &[]), "aes needs a mapping for 'x' (e.g. x=column)");
        assert_eq!(refused("x=a", &[]), "aes needs a mapping for 'y' (e.g. y=column)");
        assert!(refused("x", &[]).contains("expected aesthetic=column"));
    }

    #[test]
    fn reference_lines_and_labels_map_onto_the_library_arguments() {
        let a = flags(
            "x=a,y=b",
            &[
                "--hline", "-1.5", "--hline", "2", "--vline", "3", "--abline", "0.5,-1", "--title",
                "T", "--xlab", "X", "--ylab", "Y",
            ],
        )
        .plot_args()
        .expect("maps");
        assert_eq!(
            (a.hlines, a.vlines, a.ablines),
            (vec![-1.5, 2.0], vec![3.0], vec![(0.5, -1.0)])
        );
        assert_eq!(
            (a.title.as_deref(), a.xlab.as_deref(), a.ylab.as_deref()),
            (Some("T"), Some("X"), Some("Y"))
        );
    }

    #[test]
    fn malformed_abline_and_non_finite_lines_are_refused() {
        let refused = |extra: &[&str]| flags("x=a,y=b", extra).plot_args().expect_err("refused");
        for bad in ["1", "a,b", "1,2,3", "1,", "inf,0", "1,NaN"] {
            assert_eq!(
                refused(&["--abline", bad]),
                format!("--abline: expected slope,intercept (two numbers), got '{bad}'")
            );
        }
        assert_eq!(refused(&["--hline", "inf"]), "--hline: value must be a finite number");
        assert_eq!(refused(&["--vline", "NaN"]), "--vline: value must be a finite number");
    }

    #[test]
    fn lines_and_labels_reach_the_svg() {
        let dir = std::env::temp_dir().join(format!("plot-cli-lines-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let csv = dir.join("d.csv");
        std::fs::write(&csv, "x,y\n1,1\n2,2\n3,3\n").expect("write csv");
        let path = csv.to_str().expect("utf8");
        let svg = |extra: &[&str]| {
            let mut argv = vec!["--data", path, "--aes", "x=x,y=y", "--output", "svg"];
            argv.extend_from_slice(extra);
            let (mut out, mut err) = (Vec::new(), Vec::new());
            run(&parse(&argv).expect("parses"), &Env::default(), &mut out, &mut err)
                .expect("draws");
            String::from_utf8(out).expect("utf8")
        };
        let plain = svg(&[]);
        for (extra, label) in [
            (vec!["--hline", "2"], None),
            (vec!["--vline", "2"], None),
            (vec!["--abline", "1,0"], None),
            (vec!["--title", "TTLX"], Some("TTLX")),
            (vec!["--xlab", "XLBX"], Some("XLBX")),
            (vec!["--ylab", "YLBX"], Some("YLBX")),
        ] {
            let s = svg(&extra);
            assert_ne!(s, plain, "{extra:?} changed nothing");
            assert!(label.map_or(true, |l| s.contains(l)), "{extra:?}");
        }
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }

    #[test]
    fn flag_refusals_exit_2_and_input_failures_exit_1() {
        let go = |f: &PlotFlags| {
            run(f, &Env::default(), &mut Vec::new(), &mut Vec::new()).map_err(|e| e.exit_code())
        };
        assert_eq!(go(&flags("x=a,y=b", &["--geom", "line"])), Err(2));
        assert_eq!(go(&flags("y=b", &[])), Err(2));
        assert_eq!(go(&flags("x=a,y=b", &[])), Err(1)); // d.csv does not exist
    }

    #[test]
    fn run_draws_through_the_library_and_refuses_before_reading() {
        let dir = std::env::temp_dir().join(format!("plot-cli-flags-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let csv = dir.join("row1.csv");
        std::fs::write(&csv, "x,y\n1.5,-2.25\n2.25,0.5\n3,1.75\n4.75,3\n6.5,2.5\n")
            .expect("write csv");
        let path = csv.to_str().expect("utf8");
        let go = |argv: &[&str]| {
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let f = parse(argv).expect("parses");
            let r = run(&f, &Env::default(), &mut out, &mut err).map_err(|e| e.to_string());
            (r, String::from_utf8_lossy(&out).into_owned())
        };
        let (r, svg) = go(&["--data", path, "--aes", "x=x,y=y", "--output", "svg"]);
        assert_eq!(r, Ok(()));
        assert!(svg.starts_with("<svg"), "{svg:.200}");
        let (r, term) =
            go(&["--data", path, "--aes", "x=x,y=y", "--width", "40", "--height", "12"]);
        assert_eq!(r, Ok(()));
        assert_eq!(term.lines().count(), 12, "{term}");
        let (r, out) = go(&["--data", path, "--aes", "x=x,y=y", "--geom", "line"]);
        assert_eq!(r, Err("unsupported geom 'line'; supported: point".into()));
        assert!(out.is_empty());
        let (r, _) = go(&["--data", path, "--aes", "x=x,y=z"]);
        assert!(r.expect_err("unknown column").contains("unknown column 'z'"));
        std::fs::remove_dir_all(&dir).expect("rm tmp dir");
    }
}
