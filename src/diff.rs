//! Compare two builds of the same kernels.
//!
//! A single report answers "is this kernel bad?". The question that actually
//! comes up in review is "did my change make it worse?", which needs a
//! baseline. Kernels are matched by name; metrics that got worse are called
//! regressions, and the ones that are exact enough to gate CI on are marked
//! `blocking`.

use crate::lints::Severity;
use crate::metrics::RegSource;
use crate::report::{FileReport, KernelReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Regressed,
    Improved,
    Unchanged,
    /// The kernel is only in the new build.
    Added,
    /// The kernel was in the baseline and is gone now.
    Removed,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Regressed => "regressed",
            Verdict::Improved => "improved",
            Verdict::Unchanged => "unchanged",
            Verdict::Added => "added",
            Verdict::Removed => "removed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricDelta {
    pub label: &'static str,
    pub before: i64,
    pub after: i64,
    /// Whether a rise in this number is bad. Everything here is currently
    /// "lower is better" except occupancy.
    pub lower_is_better: bool,
    /// Exact enough to fail a build over.
    pub blocking: bool,
}

impl MetricDelta {
    pub fn change(&self) -> i64 {
        self.after - self.before
    }

    pub fn is_regression(&self) -> bool {
        if self.lower_is_better {
            self.after > self.before
        } else {
            self.after < self.before
        }
    }

    pub fn is_improvement(&self) -> bool {
        self.change() != 0 && !self.is_regression()
    }
}

#[derive(Debug, Clone)]
pub struct KernelDelta {
    pub name: String,
    pub verdict: Verdict,
    pub metrics: Vec<MetricDelta>,
    /// Lint codes that the baseline did not have.
    pub new_findings: Vec<(String, Severity)>,
    /// Lint codes that are gone.
    pub fixed_findings: Vec<String>,
}

impl KernelDelta {
    /// A regression in a metric exact enough to gate on, or a new error.
    pub fn blocking_regression(&self) -> bool {
        self.verdict == Verdict::Removed
            || self.metrics.iter().any(|m| m.blocking && m.is_regression())
            || self.new_findings.iter().any(|(_, s)| *s == Severity::Error)
    }

    pub fn changed(&self) -> bool {
        self.verdict != Verdict::Unchanged
    }
}

#[derive(Debug, Clone)]
pub struct FileDelta {
    pub baseline: String,
    pub current: String,
    pub kernels: Vec<KernelDelta>,
}

impl FileDelta {
    pub fn blocking_regression(&self) -> bool {
        self.kernels.iter().any(|k| k.blocking_regression())
    }
}

fn counts(k: &KernelReport) -> Vec<(String, Severity)> {
    k.findings
        .iter()
        .map(|f| (f.code.to_string(), f.severity))
        .collect()
}

fn metrics_of(before: &KernelReport, after: &KernelReport) -> Vec<MetricDelta> {
    let (b, a) = (&before.metrics, &after.metrics);
    // Registers only gate a build when both sides came from ptxas; the
    // virtual-register estimate moves around too much to fail on.
    let regs_exact = b.reg_source == RegSource::Ptxas && a.reg_source == RegSource::Ptxas;
    let mut out = vec![
        MetricDelta {
            label: "local memory (B)",
            before: b.local_bytes as i64,
            after: a.local_bytes as i64,
            lower_is_better: true,
            blocking: true,
        },
        MetricDelta {
            label: "spill (B)",
            before: b.spill_bytes.unwrap_or(0) as i64,
            after: a.spill_bytes.unwrap_or(0) as i64,
            lower_is_better: true,
            blocking: true,
        },
        MetricDelta {
            label: "shared memory (B)",
            before: b.shared_bytes as i64,
            after: a.shared_bytes as i64,
            lower_is_better: true,
            blocking: true,
        },
        MetricDelta {
            label: "registers/thread",
            before: b.regs_per_thread as i64,
            after: a.regs_per_thread as i64,
            lower_is_better: true,
            blocking: regs_exact,
        },
        MetricDelta {
            label: "occupancy (%)",
            before: (before.occupancy.ratio() * 100.0).round() as i64,
            after: (after.occupancy.ratio() * 100.0).round() as i64,
            lower_is_better: false,
            blocking: false,
        },
        MetricDelta {
            label: "instructions",
            before: b.mix.total as i64,
            after: a.mix.total as i64,
            lower_is_better: true,
            blocking: false,
        },
        MetricDelta {
            label: "fp64 ops",
            before: b.mix.fp64 as i64,
            after: a.mix.fp64 as i64,
            lower_is_better: true,
            blocking: false,
        },
    ];
    out.retain(|m| m.change() != 0);
    out
}

pub fn compare(baseline: &FileReport, current: &FileReport) -> FileDelta {
    let mut kernels = Vec::new();

    for after in &current.kernels {
        match baseline
            .kernels
            .iter()
            .find(|k| k.kernel.name == after.kernel.name)
        {
            None => kernels.push(KernelDelta {
                name: after.kernel.name.clone(),
                verdict: Verdict::Added,
                metrics: vec![],
                new_findings: counts(after),
                fixed_findings: vec![],
            }),
            Some(before) => {
                let metrics = metrics_of(before, after);
                let old = counts(before);
                let new = counts(after);
                let new_findings: Vec<_> = new
                    .iter()
                    .filter(|(c, _)| !old.iter().any(|(o, _)| o == c))
                    .cloned()
                    .collect();
                let fixed_findings: Vec<_> = old
                    .iter()
                    .filter(|(c, _)| !new.iter().any(|(n, _)| n == c))
                    .map(|(c, _)| c.clone())
                    .collect();

                let regressed =
                    metrics.iter().any(|m| m.is_regression()) || !new_findings.is_empty();
                let improved =
                    metrics.iter().any(|m| m.is_improvement()) || !fixed_findings.is_empty();
                let verdict = match (regressed, improved) {
                    (true, _) => Verdict::Regressed,
                    (false, true) => Verdict::Improved,
                    _ => Verdict::Unchanged,
                };
                kernels.push(KernelDelta {
                    name: after.kernel.name.clone(),
                    verdict,
                    metrics,
                    new_findings,
                    fixed_findings,
                });
            }
        }
    }

    for before in &baseline.kernels {
        if !current
            .kernels
            .iter()
            .any(|k| k.kernel.name == before.kernel.name)
        {
            kernels.push(KernelDelta {
                name: before.kernel.name.clone(),
                verdict: Verdict::Removed,
                metrics: vec![],
                new_findings: vec![],
                fixed_findings: counts(before).into_iter().map(|(c, _)| c).collect(),
            });
        }
    }

    kernels.sort_by(|a, b| a.name.cmp(&b.name));
    FileDelta {
        baseline: baseline.path.clone(),
        current: current.path.clone(),
        kernels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lints::Thresholds;
    use crate::metrics::Options;
    use std::collections::BTreeMap;

    fn report(src: &str) -> FileReport {
        let opts = Options {
            arch: None,
            block_size: None,
            ptxas: BTreeMap::new(),
        };
        crate::analyse_source("t.ptx", src, &opts, &Thresholds::default())
    }

    const DIRTY: &str = ".target sm_80 .visible .entry k() { .reg .b32 %r<4>; \
                         .local .align 4 .b8 d[256]; st.local.b32 [%rd1], %r1; ret; }";
    const CLEAN: &str = ".target sm_80 .visible .entry k() { .reg .b32 %r<4>; ret; }";

    #[test]
    fn removing_local_memory_is_a_blocking_improvement() {
        let d = compare(&report(DIRTY), &report(CLEAN));
        let k = &d.kernels[0];
        assert_eq!(k.verdict, Verdict::Improved);
        assert!(k.fixed_findings.contains(&"PTX001".to_string()));
        assert!(!d.blocking_regression());
    }

    #[test]
    fn adding_local_memory_blocks() {
        let d = compare(&report(CLEAN), &report(DIRTY));
        let k = &d.kernels[0];
        assert_eq!(k.verdict, Verdict::Regressed);
        assert!(k.blocking_regression(), "local memory must gate CI");
        let m = k
            .metrics
            .iter()
            .find(|m| m.label.starts_with("local"))
            .unwrap();
        assert_eq!((m.before, m.after), (0, 256));
        assert!(m.is_regression() && m.blocking);
    }

    #[test]
    fn identical_builds_are_unchanged() {
        let d = compare(&report(DIRTY), &report(DIRTY));
        assert_eq!(d.kernels[0].verdict, Verdict::Unchanged);
        assert!(
            d.kernels[0].metrics.is_empty(),
            "unchanged metrics are not listed"
        );
        assert!(!d.blocking_regression());
    }

    #[test]
    fn a_disappearing_kernel_blocks() {
        let two = ".target sm_80 .visible .entry a() { ret; } .visible .entry b() { ret; }";
        let one = ".target sm_80 .visible .entry a() { ret; }";
        let d = compare(&report(two), &report(one));
        let gone = d.kernels.iter().find(|k| k.name == "b").unwrap();
        assert_eq!(gone.verdict, Verdict::Removed);
        assert!(d.blocking_regression());

        // The other direction is an addition, which is not a regression.
        let d = compare(&report(one), &report(two));
        let added = d.kernels.iter().find(|k| k.name == "b").unwrap();
        assert_eq!(added.verdict, Verdict::Added);
        assert!(!d.blocking_regression());
    }

    #[test]
    fn soft_metrics_regress_without_blocking() {
        // One more instruction, nothing else: worth reporting, not worth failing.
        let a =
            ".target sm_80 .visible .entry k() { .reg .b32 %r<4>; add.s32 %r1, %r2, %r3; ret; }";
        let b = ".target sm_80 .visible .entry k() { .reg .b32 %r<4>; add.s32 %r1, %r2, %r3; \
                 add.s32 %r1, %r2, %r3; ret; }";
        let d = compare(&report(a), &report(b));
        assert_eq!(d.kernels[0].verdict, Verdict::Regressed);
        assert!(
            !d.blocking_regression(),
            "instruction count alone must not gate"
        );
    }

    #[test]
    fn fp64_creeping_in_is_reported() {
        let a =
            ".target sm_80 .visible .entry k() { .reg .f32 %f<4>; add.f32 %f1, %f2, %f3; ret; }";
        let b = ".target sm_80 .visible .entry k() { .reg .f64 %fd<4>; add.f64 %fd1, %fd2, %fd3; ret; }";
        let d = compare(&report(a), &report(b));
        let k = &d.kernels[0];
        assert!(k
            .metrics
            .iter()
            .any(|m| m.label.starts_with("fp64") && m.is_regression()));
        assert!(k.new_findings.iter().any(|(c, _)| c == "PTX003"));
    }
}
