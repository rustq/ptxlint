//! Optional exact numbers from `ptxas -v`.
//!
//! PTX only contains virtual registers, so the real per-thread register count
//! and any spills are only known after ptxas runs. When the CUDA toolkit is
//! installed we shell out to it; otherwise the estimate in `metrics` is used.

use std::collections::BTreeMap;
use std::process::Command;

/// kernel name -> (registers per thread, spill store bytes)
pub type RegInfo = BTreeMap<String, (u32, u64)>;

pub fn available() -> bool {
    Command::new("ptxas").arg("--version").output().is_ok()
}

pub fn analyse(path: &str, arch: &str) -> Result<RegInfo, String> {
    let out = Command::new("ptxas")
        .args([
            "-v",
            "-o",
            if cfg!(windows) { "NUL" } else { "/dev/null" },
            &format!("-arch={arch}"),
            path,
        ])
        .output()
        .map_err(|e| format!("running ptxas: {e}"))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    if !out.status.success() {
        return Err(format!(
            "ptxas failed: {}",
            text.lines().next().unwrap_or("").trim()
        ));
    }
    Ok(parse_verbose(&text))
}

/// Parse the `ptxas -v` report:
/// ```text
/// ptxas info    : Compiling entry function 'my_kernel' for 'sm_80'
/// ptxas info    : Used 32 registers, 8 bytes spill stores, 376 bytes cmem[0]
/// ```
pub fn parse_verbose(text: &str) -> RegInfo {
    let mut out = RegInfo::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some(i) = line.find("Compiling entry function") {
            let rest = &line[i..];
            if let (Some(a), Some(b)) = (
                rest.find('\''),
                rest[rest.find('\'').unwrap() + 1..].find('\''),
            ) {
                let start = a + 1;
                current = Some(rest[start..start + b].to_string());
            }
        } else if let Some(i) = line.find("Used ")
            && let Some(name) = current.clone()
        {
            let rest = &line[i + 5..];
            let regs = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u32>().ok())
                .unwrap_or(0);
            let spill = spill_bytes(rest);
            out.insert(name, (regs, spill));
        }
    }
    out
}

fn spill_bytes(rest: &str) -> u64 {
    let mut total = 0;
    for part in rest.split(',') {
        let p = part.trim();
        // Count stores only; loads mirror them.
        if p.contains("spill stores")
            && let Some(n) = p
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())
        {
            total += n;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'good_saxpy' for 'sm_80'
ptxas info    : Function properties for good_saxpy
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function 'bad_local_array' for 'sm_80'
ptxas info    : Function properties for bad_local_array
    256 bytes stack frame, 48 bytes spill stores, 52 bytes spill loads
ptxas info    : Used 40 registers, 48 bytes spill stores, 52 bytes spill loads, 380 bytes cmem[0]
";

    #[test]
    fn parses_ptxas_verbose_output() {
        let info = parse_verbose(SAMPLE);
        assert_eq!(info["good_saxpy"], (12, 0));
        assert_eq!(info["bad_local_array"], (40, 48));
    }
}
