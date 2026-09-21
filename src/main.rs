use ptxlint::lints::{Severity, Thresholds};
use ptxlint::metrics::{Arch, Options};
use ptxlint::{analyse_source, report};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

const USAGE: &str = "\
ptxlint — static analysis for NVIDIA PTX

USAGE:
    ptxlint [OPTIONS] <FILE|DIR>...        (use - for stdin)

OPTIONS:
    --arch <sm_XX>       Override the target architecture (default: .target in the file)
    --block-size <N>     Threads per block for the occupancy model (default: launch bounds, else 256)
    --ptxas              Use `ptxas -v` for exact register counts and spills
    --ptxas-report <F>   Read a saved `ptxas -v` log instead of running ptxas, for when
                         the build machine has CUDA and the lint job does not
    --baseline <F|DIR>   Compare against an earlier build of the same kernels and report
                         what got better or worse, instead of linting in isolation
    --all                In --baseline mode, also list kernels that did not change
    --deny <LEVEL|CODE>  Exit non-zero on error|warning|info|all or a code such as PTX003,
                         or `regression` in --baseline mode
                         (repeatable; default: never fails)
    --json               Machine-readable output
    --no-color           Disable ANSI colors
    --list-lints         Describe every lint and exit
    -h, --help           Show this help

EXIT CODES:
    0  no denied lint fired
    1  a denied lint fired
    2  ptxlint could not run: bad arguments, or a file it could not read

LINTS:
    PTX001  local memory in use (dynamically indexed array)
    PTX002  register spills reported by ptxas
    PTX003  FP64 instructions (Rust float literals default to f64)
    PTX004  integer division or remainder
    PTX005  high register pressure
    PTX006  low estimated occupancy
    PTX007  shared memory over budget or capping residency
    PTX008  narrow, non-vectorised global accesses
    PTX009  no launch bounds (.maxntid/.reqntid)
    PTX010  non-inlined calls
";

struct Args {
    paths: Vec<String>,
    opts: Options,
    use_ptxas: bool,
    ptxas_report: Option<String>,
    baseline: Option<String>,
    show_all: bool,
    deny: Vec<String>,
    json: bool,
    color: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        paths: vec![],
        opts: Options {
            arch: None,
            block_size: None,
            ptxas: BTreeMap::new(),
        },
        use_ptxas: false,
        ptxas_report: None,
        baseline: None,
        show_all: false,
        deny: vec![],
        json: false,
        color: std::io::stdout().is_terminal(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--list-lints" => {
                print!("{}", USAGE.split("LINTS:").nth(1).unwrap_or(""));
                std::process::exit(0);
            }
            "--json" => a.json = true,
            "--no-color" => a.color = false,
            "--ptxas" => a.use_ptxas = true,
            "--all" => a.show_all = true,
            "--baseline" => a.baseline = Some(it.next().ok_or("--baseline needs a value")?),
            "--ptxas-report" => {
                a.ptxas_report = Some(it.next().ok_or("--ptxas-report needs a value")?)
            }
            "--arch" => {
                let v = it.next().ok_or("--arch needs a value")?;
                if Arch::lookup(&v).is_none() {
                    eprintln!("warning: unknown arch {v}, falling back to defaults");
                }
                a.opts.arch = Some(v);
            }
            "--block-size" => {
                let v = it.next().ok_or("--block-size needs a value")?;
                let n: u32 = v.parse().map_err(|_| format!("bad --block-size: {v}"))?;
                if n == 0 || n > 1024 {
                    return Err("--block-size must be 1..=1024".into());
                }
                a.opts.block_size = Some(n);
            }
            "--deny" => a.deny.push(it.next().ok_or("--deny needs a value")?),
            s if s.starts_with('-') && s != "-" => return Err(format!("unknown option {s}")),
            s => a.paths.push(s.to_string()),
        }
    }
    if a.paths.is_empty() {
        return Err("no input files".into());
    }
    Ok(a)
}

fn collect(paths: &[String]) -> Vec<PathBuf> {
    let mut out = vec![];
    for p in paths {
        let path = Path::new(p);
        if path.is_dir() {
            let mut stack = vec![path.to_path_buf()];
            while let Some(d) = stack.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else {
                    continue;
                };
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if p.extension().is_some_and(|x| x == "ptx") {
                        out.push(p);
                    }
                }
            }
        } else {
            out.push(path.to_path_buf());
        }
    }
    out.sort();
    out
}

fn denies_regressions(deny: &[String]) -> bool {
    deny.iter()
        .any(|d| d.eq_ignore_ascii_case("regression") || d == "all")
}

fn denied(deny: &[String], sev: Severity, code: &str) -> bool {
    deny.iter().any(|d| match d.as_str() {
        "all" => true,
        "error" => sev == Severity::Error,
        "warning" => sev >= Severity::Warning,
        "info" => true,
        c => c.eq_ignore_ascii_case(code),
    })
}

/// Read a PTX file (or stdin) and analyse it. `None` means it could not be read.
fn analyse_path(
    path: &Path,
    args: &Args,
    saved_report: &BTreeMap<String, (u32, u64)>,
    thresholds: &Thresholds,
) -> Option<report::FileReport> {
    let display = path.display().to_string();
    let src = if display == "-" {
        let mut s = String::new();
        match std::io::stdin().read_to_string(&mut s) {
            Ok(_) => s,
            Err(e) => {
                eprintln!("ptxlint: stdin: {e}");
                return None;
            }
        }
    } else {
        match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ptxlint: {display}: {e}");
                return None;
            }
        }
    };

    let mut opts = Options {
        arch: args.opts.arch.clone(),
        block_size: args.opts.block_size,
        ptxas: saved_report.clone(),
    };
    if args.use_ptxas && display != "-" {
        let target = args
            .opts
            .arch
            .clone()
            .or_else(|| ptxlint::parse::parse(&src).target)
            .unwrap_or_else(|| ptxlint::metrics::DEFAULT_ARCH.to_string());
        match ptxlint::ptxas::analyse(&display, &target) {
            Ok(info) => opts.ptxas = info,
            Err(e) => eprintln!("ptxlint: {display}: {e} (falling back to estimates)"),
        }
    }
    Some(analyse_source(&display, &src, &opts, thresholds))
}

/// The baseline file that corresponds to `current`: the same name inside a
/// baseline directory, or the baseline file itself when both are single files.
fn baseline_for(baseline: &Path, current: &Path, single: bool) -> Option<PathBuf> {
    if baseline.is_dir() {
        let candidate = baseline.join(current.file_name()?);
        candidate.exists().then_some(candidate)
    } else if single {
        Some(baseline.to_path_buf())
    } else {
        None
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ptxlint: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    let thresholds = Thresholds::default();
    let mut had_io_error = false;

    // A `ptxas -v` log saved on a machine that has CUDA, applied to every file.
    let mut saved_report = BTreeMap::new();
    if let Some(p) = &args.ptxas_report {
        match std::fs::read_to_string(p) {
            Ok(text) => {
                saved_report = ptxlint::ptxas::parse_verbose(&text);
                if saved_report.is_empty() {
                    eprintln!("ptxlint: {p}: no kernels found in the ptxas report");
                }
            }
            Err(e) => {
                eprintln!("ptxlint: {p}: {e}");
                std::process::exit(2);
            }
        }
    }

    let paths = collect(&args.paths);
    let mut reports = vec![];
    for path in &paths {
        match analyse_path(path, &args, &saved_report, &thresholds) {
            Some(r) => reports.push(r),
            None => had_io_error = true,
        }
    }

    // --baseline: report what changed instead of the state of one build.
    if let Some(base) = &args.baseline {
        let base = Path::new(base);
        let single = paths.len() == 1;
        let mut deltas = vec![];
        for (path, current) in paths.iter().zip(&reports) {
            let Some(bpath) = baseline_for(base, path, single) else {
                eprintln!("ptxlint: no baseline for {}", path.display());
                had_io_error = true;
                continue;
            };
            match analyse_path(&bpath, &args, &saved_report, &thresholds) {
                Some(b) => deltas.push(ptxlint::diff::compare(&b, current)),
                None => had_io_error = true,
            }
        }
        let out = if args.json {
            report::diff_json(&deltas)
        } else {
            report::diff_text(&deltas, args.color, args.show_all)
        };
        let _ = std::io::stdout().write_all(out.as_bytes());
        let blocked =
            denies_regressions(&args.deny) && deltas.iter().any(|d| d.blocking_regression());
        std::process::exit(if had_io_error {
            2
        } else if blocked {
            1
        } else {
            0
        });
    }

    let out = if args.json {
        report::json(&reports)
    } else {
        report::text(&reports, args.color)
    };
    let _ = std::io::stdout().write_all(out.as_bytes());

    let violated = reports
        .iter()
        .flat_map(|f| &f.kernels)
        .flat_map(|k| &k.findings)
        .any(|f| denied(&args.deny, f.severity, f.code));

    // 0 clean, 1 a denied lint fired, 2 ptxlint itself could not do its job.
    // Keeping those apart lets a CI check assert "this case still fires".
    std::process::exit(if had_io_error {
        2
    } else if violated {
        1
    } else {
        0
    });
}
