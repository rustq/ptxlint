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
    --deny <LEVEL|CODE>  Exit non-zero on error|warning|info|all or a code such as PTX003
                         (repeatable; default: never fails)
    --json               Machine-readable output
    --no-color           Disable ANSI colors
    --list-lints         Describe every lint and exit
    -h, --help           Show this help

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

fn denied(deny: &[String], sev: Severity, code: &str) -> bool {
    deny.iter().any(|d| match d.as_str() {
        "all" => true,
        "error" => sev == Severity::Error,
        "warning" => sev >= Severity::Warning,
        "info" => true,
        c => c.eq_ignore_ascii_case(code),
    })
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
    let mut reports = vec![];
    let mut had_io_error = false;

    for path in collect(&args.paths) {
        let display = path.display().to_string();
        let src = if display == "-" {
            let mut s = String::new();
            match std::io::stdin().read_to_string(&mut s) {
                Ok(_) => s,
                Err(e) => {
                    eprintln!("ptxlint: stdin: {e}");
                    had_io_error = true;
                    continue;
                }
            }
        } else {
            match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("ptxlint: {display}: {e}");
                    had_io_error = true;
                    continue;
                }
            }
        };

        let mut opts = Options {
            arch: args.opts.arch.clone(),
            block_size: args.opts.block_size,
            ptxas: BTreeMap::new(),
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
        reports.push(analyse_source(&display, &src, &opts, &thresholds));
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

    std::process::exit(if violated || had_io_error { 1 } else { 0 });
}
