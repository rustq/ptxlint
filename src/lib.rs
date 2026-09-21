//! `ptxlint` — a static analyser for NVIDIA PTX.
//!
//! Point it at the `.ptx` your Rust GPU kernels compile to and it reports
//! register pressure, local-memory traffic, shared-memory budget, estimated
//! occupancy and a set of performance lints. No GPU and no CUDA install
//! required; if `ptxas` happens to be on PATH, the register numbers become
//! exact instead of estimated.

pub mod diff;
pub mod lints;
pub mod metrics;
pub mod parse;
pub mod ptxas;
pub mod report;

use lints::Thresholds;
use metrics::Options;
use report::{FileReport, KernelReport};

/// Analyse one PTX source.
pub fn analyse_source(path: &str, src: &str, opts: &Options, t: &Thresholds) -> FileReport {
    let module = parse::parse(src);
    let kernels = module
        .kernels
        .into_iter()
        .map(|k| {
            let m = metrics::analyse(
                &k,
                module.target.as_deref(),
                module.global_shared_bytes,
                opts,
            );
            let occ = metrics::occupancy(&m);
            let findings = lints::run(&k, &m, &occ, t);
            KernelReport {
                kernel: k,
                metrics: m,
                occupancy: occ,
                findings,
            }
        })
        .collect();
    FileReport {
        path: path.to_string(),
        target: module.target,
        kernels,
    }
}
