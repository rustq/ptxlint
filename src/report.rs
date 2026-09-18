//! Human-readable and JSON reporting.

use crate::lints::{Finding, Severity};
use crate::metrics::{Metrics, Occupancy, RegSource};
use crate::parse::Kernel;
use std::fmt::Write as _;

pub struct KernelReport {
    pub kernel: Kernel,
    pub metrics: Metrics,
    pub occupancy: Occupancy,
    pub findings: Vec<Finding>,
}

pub struct FileReport {
    pub path: String,
    pub target: Option<String>,
    pub kernels: Vec<KernelReport>,
}

fn color(s: Severity, on: bool) -> &'static str {
    if !on {
        return "";
    }
    match s {
        Severity::Error => "\x1b[31m",
        Severity::Warning => "\x1b[33m",
        Severity::Info => "\x1b[36m",
    }
}

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";

pub fn text(files: &[FileReport], color_on: bool) -> String {
    let mut o = String::new();
    let (b, d, r) = if color_on {
        (BOLD, DIM, RESET)
    } else {
        ("", "", "")
    };
    for f in files {
        let _ = writeln!(o, "{b}{}{r}", f.path);
        if f.kernels.is_empty() {
            let _ = writeln!(o, "  {d}no .entry kernels found{r}\n");
            continue;
        }
        for kr in &f.kernels {
            let m = &kr.metrics;
            let occ = &kr.occupancy;
            let arch_note = if m.arch_known { "" } else { " (assumed)" };
            let _ = writeln!(
                o,
                "\n  {b}{}{r}  {d}line {}{r}",
                kr.kernel.name, kr.kernel.line
            );
            let reg_note = match m.reg_source {
                RegSource::Ptxas => "ptxas",
                RegSource::VirtualUpperBound => "virtual, upper bound",
            };
            let _ = writeln!(
                o,
                "    arch {}{arch_note}   regs/thread {} {d}({reg_note}){r}   shared {} B   local {} B",
                m.arch.name, m.regs_per_thread, m.shared_bytes, m.local_bytes
            );
            let block_note = if m.block_size_from_ptx {
                "from launch bounds"
            } else {
                "assumed"
            };
            let _ = writeln!(
                o,
                "    occupancy {:.0}% {d}({} of {} warps/SM, {} blocks/SM @ {} threads/block, {block_note}; limited by {}){r}",
                occ.ratio() * 100.0,
                occ.active_warps,
                occ.max_warps,
                occ.blocks_per_sm,
                m.block_size,
                occ.limiter.label(),
            );
            let mix = &m.mix;
            let mut parts = vec![format!("{} instructions", mix.total)];
            for (label, n) in [
                ("fp32", mix.fp32),
                ("fp64", mix.fp64),
                ("fp16", mix.fp16),
                ("int", mix.int),
                ("branch", mix.branch),
                ("sync", mix.sync),
                ("atomic", mix.atomic),
                ("tensor", mix.tensor),
            ] {
                if n > 0 {
                    parts.push(format!("{label} {n}"));
                }
            }
            for (space, n) in &mix.mem {
                parts.push(format!("{space} {n}"));
            }
            let _ = writeln!(o, "    {d}{}{r}", parts.join("  ·  "));

            if kr.findings.is_empty() {
                let _ = writeln!(o, "    {d}no findings{r}");
            }
            for find in &kr.findings {
                let c = color(find.severity, color_on);
                let loc = find.line.map(|l| format!(":{l}")).unwrap_or_default();
                let _ = writeln!(
                    o,
                    "    {c}{:<7}{r} {} {d}[{}{loc}]{r}",
                    find.severity.as_str(),
                    find.message,
                    find.code
                );
                let _ = writeln!(o, "            {d}{}{r}", find.help);
                if find.samples.len() > 1 {
                    let lines: Vec<String> = find.samples.iter().map(|l| l.to_string()).collect();
                    let _ = writeln!(o, "            {d}at lines {}{r}", lines.join(", "));
                }
            }
        }
        let _ = writeln!(o);
    }
    let (e, w, i) = counts(files);
    let _ = writeln!(o, "{e} error, {w} warning, {i} info");
    o
}

pub fn counts(files: &[FileReport]) -> (usize, usize, usize) {
    let all = files
        .iter()
        .flat_map(|f| &f.kernels)
        .flat_map(|k| &k.findings);
    let mut c = (0, 0, 0);
    for f in all {
        match f.severity {
            Severity::Error => c.0 += 1,
            Severity::Warning => c.1 += 1,
            Severity::Info => c.2 += 1,
        }
    }
    c
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}

pub fn json(files: &[FileReport]) -> String {
    let mut o = String::from("{\n  \"files\": [\n");
    for (fi, f) in files.iter().enumerate() {
        let _ = write!(o, "    {{\n      \"path\": \"{}\",\n", esc(&f.path));
        let _ = write!(
            o,
            "      \"target\": {},\n      \"kernels\": [\n",
            f.target
                .as_ref()
                .map(|t| format!("\"{}\"", esc(t)))
                .unwrap_or("null".into())
        );
        for (ki, k) in f.kernels.iter().enumerate() {
            let m = &k.metrics;
            let _ = write!(
                o,
                "        {{\n          \"name\": \"{}\",\n          \"line\": {},\n          \
                 \"arch\": \"{}\",\n          \"arch_known\": {},\n          \
                 \"regs_per_thread\": {},\n          \"reg_source\": \"{}\",\n          \
                 \"shared_bytes\": {},\n          \"local_bytes\": {},\n          \
                 \"block_size\": {},\n          \"occupancy\": {:.4},\n          \
                 \"blocks_per_sm\": {},\n          \"limiter\": \"{}\",\n          \
                 \"instructions\": {},\n          \"fp64\": {},\n          \"findings\": [\n",
                esc(&k.kernel.name),
                k.kernel.line,
                m.arch.name,
                m.arch_known,
                m.regs_per_thread,
                match m.reg_source {
                    RegSource::Ptxas => "ptxas",
                    RegSource::VirtualUpperBound => "virtual",
                },
                m.shared_bytes,
                m.local_bytes,
                m.block_size,
                k.occupancy.ratio(),
                k.occupancy.blocks_per_sm,
                k.occupancy.limiter.label(),
                m.mix.total,
                m.mix.fp64,
            );
            for (i, find) in k.findings.iter().enumerate() {
                let _ = writeln!(
                    o,
                    "            {{\"code\": \"{}\", \"severity\": \"{}\", \"line\": {}, \
                     \"message\": \"{}\", \"help\": \"{}\"}}{}",
                    find.code,
                    find.severity.as_str(),
                    find.line.map(|l| l.to_string()).unwrap_or("null".into()),
                    esc(&find.message),
                    esc(find.help),
                    if i + 1 < k.findings.len() { "," } else { "" }
                );
            }
            let _ = write!(
                o,
                "          ]\n        }}{}\n",
                if ki + 1 < f.kernels.len() { "," } else { "" }
            );
        }
        let _ = write!(
            o,
            "      ]\n    }}{}\n",
            if fi + 1 < files.len() { "," } else { "" }
        );
    }
    let (e, w, i) = counts(files);
    let _ = write!(
        o,
        "  ],\n  \"summary\": {{\"error\": {e}, \"warning\": {w}, \"info\": {i}}}\n}}\n"
    );
    o
}
