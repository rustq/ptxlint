//! Lint rules. Each rule looks at one kernel's parsed body plus its metrics.

use crate::metrics::{Limiter, Metrics, Occupancy, RegSource};
use crate::parse::Kernel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub code: &'static str,
    pub severity: Severity,
    pub kernel: String,
    /// Line in the PTX file, when the finding points at specific instructions.
    pub line: Option<u32>,
    pub message: String,
    pub help: &'static str,
    /// Up to a few example source lines.
    pub samples: Vec<u32>,
}

pub struct Thresholds {
    pub regs_warn: u32,
    pub occupancy_warn: f64,
    pub narrow_global_ratio: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            regs_warn: 64,
            occupancy_warn: 0.25,
            narrow_global_ratio: 0.9,
        }
    }
}

fn samples(k: &Kernel, pred: impl Fn(&crate::parse::Inst) -> bool) -> Vec<u32> {
    k.insts
        .iter()
        .filter(|i| pred(i))
        .map(|i| i.line)
        .take(3)
        .collect()
}

pub fn run(k: &Kernel, m: &Metrics, occ: &Occupancy, t: &Thresholds) -> Vec<Finding> {
    let mut out = Vec::new();
    let mix = &m.mix;
    let kernel = k.name.clone();

    // PTX001 — local memory traffic.
    if m.local_bytes > 0 || mix.mem.contains_key("local") {
        let lines = samples(k, |i| i.space() == Some("local"));
        let accesses = mix.mem.get("local").copied().unwrap_or(0);
        out.push(Finding {
            code: "PTX001",
            severity: Severity::Error,
            kernel: kernel.clone(),
            line: lines.first().copied(),
            message: format!(
                "{} bytes of local memory, {accesses} local accesses — this lives in DRAM, not registers",
                m.local_bytes
            ),
            help: "An array indexed by a runtime value cannot stay in registers. \
                   Use a fixed index, unroll the loop, or move the array to shared memory.",
            samples: lines,
        });
    }

    // PTX002 — spills reported by ptxas.
    if let Some(spill) = m.spill_bytes.filter(|&s| s > 0) {
        out.push(Finding {
            code: "PTX002",
            severity: Severity::Error,
            kernel: kernel.clone(),
            line: None,
            message: format!("ptxas spilled {spill} bytes of registers to local memory"),
            help: "Reduce live values, split the kernel, or cap registers with \
                   -C llvm-args=--nvptx-max-regs / __launch_bounds__.",
            samples: vec![],
        });
    }

    // PTX003 — FP64 on hardware that is bad at it.
    if mix.fp64 > 0 {
        let lines = samples(k, |i| i.quals.iter().any(|q| q == "f64"));
        let rate = if m.arch.weak_fp64 { "1/64" } else { "1/2" };
        out.push(Finding {
            code: "PTX003",
            severity: if m.arch.weak_fp64 {
                Severity::Error
            } else {
                Severity::Warning
            },
            kernel: kernel.clone(),
            line: lines.first().copied(),
            message: format!(
                "{} FP64 instructions ({} runs FP64 at {rate} of FP32 throughput)",
                mix.fp64, m.arch.name
            ),
            help: "In Rust a bare float literal is f64, so `x * 0.5` promotes the whole \
                   expression. Write `0.5f32`, and prefer f32 math functions.",
            samples: lines,
        });
    }

    // PTX004 — integer division and remainder.
    if mix.int_divrem > 0 {
        let lines = samples(k, |i| matches!(i.base.as_str(), "div" | "rem"));
        out.push(Finding {
            code: "PTX004",
            severity: Severity::Warning,
            kernel: kernel.clone(),
            line: lines.first().copied(),
            message: format!(
                "{} integer div/rem instructions (each expands to ~20 SASS instructions)",
                mix.int_divrem
            ),
            help: "GPUs have no integer divider. Use power-of-two sizes so the compiler \
                   can shift and mask, or hoist the division out of the hot loop.",
            samples: lines,
        });
    }

    // PTX005 — register pressure.
    if m.regs_per_thread > t.regs_warn {
        let (sev, note) = match m.reg_source {
            RegSource::Ptxas => (Severity::Warning, ""),
            RegSource::VirtualUpperBound => (
                Severity::Info,
                " (virtual-register upper bound; run with ptxas for the real count)",
            ),
        };
        out.push(Finding {
            code: "PTX005",
            severity: sev,
            kernel: kernel.clone(),
            line: None,
            message: format!("{} registers per thread{note}", m.regs_per_thread),
            help: "High register use caps how many warps fit on an SM. \
                   Shrink live ranges or set launch bounds.",
            samples: vec![],
        });
    }

    // PTX006 — low occupancy.
    if occ.ratio() < t.occupancy_warn && m.reg_source == RegSource::Ptxas {
        out.push(Finding {
            code: "PTX006",
            severity: Severity::Warning,
            kernel: kernel.clone(),
            line: None,
            message: format!(
                "estimated occupancy {:.0}% at {} threads/block, limited by {}",
                occ.ratio() * 100.0,
                m.block_size,
                occ.limiter.label()
            ),
            help: "Low occupancy hides memory latency poorly. Tune the block size or \
                   reduce the limiting resource.",
            samples: vec![],
        });
    }

    // PTX007 — shared memory that will not launch.
    if m.shared_bytes as u32 > m.arch.max_shared_per_block {
        out.push(Finding {
            code: "PTX007",
            severity: Severity::Error,
            kernel: kernel.clone(),
            line: None,
            message: format!(
                "{} bytes of static shared memory exceeds the {} limit of {} bytes per block",
                m.shared_bytes, m.arch.name, m.arch.max_shared_per_block
            ),
            help: "The launch will fail. Use dynamic shared memory with \
                   cudaFuncSetAttribute, or tile the data.",
            samples: vec![],
        });
    } else if occ.limiter == Limiter::SharedMemory && occ.ratio() < 0.5 {
        out.push(Finding {
            code: "PTX007",
            severity: Severity::Info,
            kernel: kernel.clone(),
            line: None,
            message: format!(
                "{} bytes of shared memory per block limits residency to {} block(s)/SM",
                m.shared_bytes, occ.blocks_per_sm
            ),
            help: "Shrink the tile or split it across more, smaller blocks.",
            samples: vec![],
        });
    }

    // PTX008 — narrow global accesses.
    if mix.global_access >= 4 {
        let ratio = mix.narrow_global as f64 / mix.global_access as f64;
        if ratio >= t.narrow_global_ratio {
            let lines = samples(k, |i| {
                i.space() == Some("global") && matches!(i.base.as_str(), "ld" | "st")
            });
            out.push(Finding {
                code: "PTX008",
                severity: Severity::Info,
                kernel: kernel.clone(),
                line: lines.first().copied(),
                message: format!(
                    "{}/{} global accesses move <= 4 bytes each; no vectorised (.v2/.v4) access",
                    mix.narrow_global, mix.global_access
                ),
                help: "Wider per-thread accesses cut instruction count and improve \
                       memory pipe utilisation — try processing 2 or 4 elements per thread.",
                samples: lines,
            });
        }
    }

    // PTX009 — no launch bounds. Only worth saying when occupancy is not already full.
    if k.maxntid.is_none() && k.reqntid.is_none() && occ.ratio() < 1.0 {
        out.push(Finding {
            code: "PTX009",
            severity: Severity::Info,
            kernel: kernel.clone(),
            line: Some(k.line),
            message: "no .maxntid/.reqntid launch bounds".to_string(),
            help: "Without launch bounds ptxas budgets registers for the largest \
                   possible block, which can cost occupancy.",
            samples: vec![],
        });
    }

    // PTX010 — calls that were not inlined.
    if mix.calls > 0 {
        let lines = samples(k, |i| i.base == "call");
        out.push(Finding {
            code: "PTX010",
            severity: Severity::Info,
            kernel: kernel.clone(),
            line: lines.first().copied(),
            message: format!(
                "{} non-inlined call(s) — ABI calls cost registers and stack",
                mix.calls
            ),
            help: "Mark the callee #[inline(always)], or check it is not an \
                   accidental libdevice/intrinsic call.",
            samples: lines,
        });
    }

    out.sort_by(|a, b| b.severity.cmp(&a.severity).then(a.code.cmp(b.code)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{analyse, occupancy, Options};
    use std::collections::BTreeMap;

    fn check(src: &str) -> Vec<Finding> {
        let m = crate::parse::parse(src);
        let opts = Options {
            arch: None,
            block_size: None,
            ptxas: BTreeMap::new(),
        };
        let k = &m.kernels[0];
        let met = analyse(k, m.target.as_deref(), m.global_shared_bytes, &opts);
        let occ = occupancy(&met);
        run(k, &met, &occ, &Thresholds::default())
    }

    fn has(f: &[Finding], code: &str) -> bool {
        f.iter().any(|x| x.code == code)
    }

    #[test]
    fn flags_local_memory() {
        let f = check(
            ".target sm_80 .visible .entry k() { .local .align 4 .b8 d[256]; \
             st.local.b32 [%rd1], %r1; ret; }",
        );
        assert!(has(&f, "PTX001"));
        assert_eq!(
            f.iter().find(|x| x.code == "PTX001").unwrap().severity,
            Severity::Error
        );
    }

    #[test]
    fn flags_fp64_harder_on_geforce() {
        let src = ".target sm_89 .visible .entry k() { add.f64 %fd1, %fd2, %fd3; ret; }";
        let f = check(src);
        assert_eq!(
            f.iter().find(|x| x.code == "PTX003").unwrap().severity,
            Severity::Error
        );
        // Same kernel on a datacentre part is only a warning.
        let f = check(&src.replace("sm_89", "sm_80"));
        assert_eq!(
            f.iter().find(|x| x.code == "PTX003").unwrap().severity,
            Severity::Warning
        );
    }

    #[test]
    fn clean_kernel_has_no_warnings() {
        let f = check(
            ".target sm_80 .visible .entry k() .maxntid 256, 1, 1 { \
             .reg .b32 %r<8>; ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1]; \
             fma.rn.f32 %f5, %f1, %f2, %f3; st.global.v4.f32 [%rd2], {%f5,%f5,%f5,%f5}; ret; }",
        );
        assert!(f.iter().all(|x| x.severity == Severity::Info), "{f:#?}");
        assert!(!has(&f, "PTX009"));
    }

    #[test]
    fn flags_integer_division() {
        let f = check(".target sm_80 .visible .entry k() { div.s32 %r1, %r2, %r3; ret; }");
        assert!(has(&f, "PTX004"));
    }

    #[test]
    fn flags_oversized_shared_memory() {
        let f = check(".target sm_86 .visible .entry k() { .shared .align 4 .b8 s[131072]; ret; }");
        let x = f.iter().find(|x| x.code == "PTX007").unwrap();
        assert_eq!(x.severity, Severity::Error);
    }

    #[test]
    fn vectorised_access_is_not_flagged() {
        let narrow = ".target sm_80 .visible .entry k() { ld.global.f32 %f1, [%rd1]; \
                      ld.global.f32 %f2, [%rd2]; st.global.f32 [%rd3], %f1; \
                      st.global.f32 [%rd4], %f2; ret; }";
        assert!(has(&check(narrow), "PTX008"));
        assert!(!has(
            &check(&narrow.replace("global.f32", "global.v4.f32")),
            "PTX008"
        ));
    }
}
