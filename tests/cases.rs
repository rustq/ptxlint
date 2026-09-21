//! One test per lint, one file per lint.
//!
//! Every case lives in `cases/`. Most are compiled from the Rust kernel of the
//! same name in `fixtures/examples/` (see `fixtures/generate.sh`); the few that
//! cannot be expressed in Rust on the raw nvptx64 target say so in a comment at
//! the top of the `.ptx`.

use std::process::{Command, Stdio};

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ptxlint"));
    c.arg("--no-color");
    c
}

fn run(args: &[&str]) -> (String, i32) {
    let out = bin().args(args).output().expect("run ptxlint");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

fn case(name: &str) -> String {
    run(&[&format!("cases/{name}.ptx")]).0
}

/// A case that needs exact register counts replays a recorded ptxas log.
fn case_with_report(name: &str) -> String {
    run(&[
        "--ptxas-report",
        &format!("cases/{name}.ptxas.txt"),
        &format!("cases/{name}.ptx"),
    ])
    .0
}

fn codes(out: &str) -> Vec<String> {
    out.lines()
        .filter_map(|l| l.split('[').nth(1))
        .filter_map(|l| l.split([':', ']']).next())
        .filter(|c| c.starts_with("PTX"))
        .map(str::to_string)
        .collect()
}

fn fires(out: &str, code: &str) -> bool {
    codes(out).iter().any(|c| c == code)
}

// --- one test per lint -----------------------------------------------------

#[test]
fn ptx001_local_memory() {
    let out = case("ptx001_local_memory");
    assert!(fires(&out, "PTX001"), "{out}");
    assert!(out.contains("256 bytes of local memory"));
    assert!(out.contains("local 256 B"));
}

#[test]
fn ptx002_register_spills() {
    let out = case_with_report("ptx002_register_spills");
    assert!(fires(&out, "PTX002"), "{out}");
    assert!(out.contains("spilled 96 bytes"));
    // The replayed report replaces the estimate.
    assert!(out.contains("regs/thread 72 (ptxas)"));
    // Without the report the lint cannot fire at all.
    assert!(!fires(&case("ptx002_register_spills"), "PTX002"));
}

#[test]
fn ptx003_fp64_literals() {
    let out = case("ptx003_fp64_literals");
    assert!(fires(&out, "PTX003"), "{out}");
    assert!(out.contains("FP64 instructions"));
    assert!(!fires(&out, "PTX001"), "no local memory in this case");
}

#[test]
fn ptx004_integer_division() {
    let out = case("ptx004_integer_division");
    assert!(fires(&out, "PTX004"), "{out}");
    assert!(out.contains("integer div/rem"));
    // The divisor is forced odd, so Rust's divide-by-zero panic call is gone.
    assert!(!fires(&out, "PTX010"), "unexpected call:\n{out}");
}

#[test]
fn ptx005_register_pressure() {
    let out = case("ptx005_register_pressure");
    assert!(fires(&out, "PTX005"), "{out}");
    assert!(out.contains("registers per thread"));
    // Every index is a literal, so nothing was pushed into local memory.
    assert!(!fires(&out, "PTX001"), "{out}");
    assert!(out.contains("local 0 B"));
}

#[test]
fn ptx006_low_occupancy() {
    let out = case_with_report("ptx006_low_occupancy");
    assert!(fires(&out, "PTX006"), "{out}");
    assert!(out.contains("limited by registers"));
    // Occupancy is only reported when the register count is exact.
    assert!(!fires(&case("ptx006_low_occupancy"), "PTX006"));
}

#[test]
fn ptx007_shared_memory() {
    let out = case("ptx007_shared_memory");
    assert!(fires(&out, "PTX007"), "{out}");
    assert!(out.contains("exceeds the sm_86 limit"));
    assert!(out.contains("shared 131072 B"));
}

#[test]
fn ptx008_narrow_access() {
    let out = case("ptx008_narrow_access");
    assert!(fires(&out, "PTX008"), "{out}");
    assert!(out.contains("no vectorised"));
    // The PTX009 case does the same work with .v4 accesses and stays quiet.
    assert!(!fires(&case("ptx009_launch_bounds"), "PTX008"));
}

#[test]
fn ptx009_launch_bounds() {
    let out = case("ptx009_launch_bounds");
    assert!(fires(&out, "PTX009"), "{out}");
    assert!(out.contains("no .maxntid"));
}

#[test]
fn ptx010_uninlined_calls() {
    let out = case("ptx010_uninlined_calls");
    assert!(fires(&out, "PTX010"), "{out}");
    assert!(out.contains("non-inlined call"));
}

// --- the cases that are not lints ------------------------------------------

#[test]
fn clean_saxpy_reports_nothing() {
    let out = case("clean_saxpy");
    assert!(out.contains("no findings"), "{out}");
    assert_eq!(codes(&out), Vec::<String>::new());
    assert!(out.contains("0 error, 0 warning, 0 info"));
}

#[test]
fn modern_instructions_are_understood() {
    let out = case("modern_tensor_cores");
    assert!(out.contains("hgemm_tc"));
    assert!(
        out.contains("tensor 2"),
        "wmma should count as tensor ops:\n{out}"
    );
    assert!(out.contains("shared 16384 B"));
    assert!(out.contains("arch sm_90"));
}

#[test]
fn nanoid_kernel_has_no_local_memory_after_the_fix() {
    // Regression guard for the bug ptxlint found in its own sibling project:
    // the ChaCha20 state used to be spilled to local memory.
    let out = case("nanoid_regression");
    assert!(out.contains("nanoid_chacha"));
    assert!(!fires(&out, "PTX001"), "local memory is back:\n{out}");
    assert!(out.contains("local 0 B"));
}

/// Nothing in `cases/` may be forgotten by this file.
#[test]
fn every_case_file_is_covered() {
    // Counts both the case("name") helpers and paths written out in full.
    let src = std::fs::read_to_string(file!()).unwrap();
    let mut tested: Vec<String> = src
        .lines()
        .filter_map(|l| {
            l.split("case(\"")
                .nth(1)
                .or_else(|| l.split("report(\"").nth(1))
        })
        .filter_map(|l| l.split('"').next())
        .map(str::to_string)
        .collect();
    for part in src.split("cases/").skip(1) {
        if let Some(name) = part.split(".ptx").next() {
            tested.push(name.to_string());
        }
    }
    for entry in std::fs::read_dir("cases").unwrap().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "ptx") {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            assert!(tested.contains(&stem), "cases/{stem}.ptx has no test");
        }
    }
}

// --- baseline diff ---------------------------------------------------------

/// `diff_before` and `diff_after` export the same kernel, before and after the
/// local-memory fix, which is the shape of the bug ptxlint found in nanoid.
#[test]
fn a_fix_shows_up_as_an_improvement() {
    let (out, code) = run(&[
        "--deny",
        "regression",
        "--baseline",
        "cases/diff_before.ptx",
        "cases/diff_after.ptx",
    ]);
    assert!(out.contains("mix16"));
    assert!(out.contains("improved"), "{out}");
    assert!(out.contains("local memory (B)     64 \u{2192} 0"), "{out}");
    assert!(out.contains("fixed    PTX001"), "{out}");
    assert!(out.contains("0 regressed, 1 improved"));
    assert_eq!(code, 0, "an improvement must not fail the build");
}

#[test]
fn undoing_the_fix_blocks_the_build() {
    let (out, code) = run(&[
        "--deny",
        "regression",
        "--baseline",
        "cases/diff_after.ptx",
        "cases/diff_before.ptx",
    ]);
    assert!(out.contains("regressed (blocking)"), "{out}");
    assert!(out.contains("new      PTX001"), "{out}");
    assert_eq!(code, 1);

    // Without --deny the same diff only reports.
    let (_, code) = run(&[
        "--baseline",
        "cases/diff_after.ptx",
        "cases/diff_before.ptx",
    ]);
    assert_eq!(code, 0);
}

#[test]
fn an_unchanged_build_is_quiet() {
    let (out, code) = run(&[
        "--deny",
        "regression",
        "--baseline",
        "cases/diff_after.ptx",
        "cases/diff_after.ptx",
    ]);
    assert!(out.contains("no change"), "{out}");
    assert!(out.contains("0 regressed, 0 improved"));
    assert_eq!(code, 0);
    // --all lists it anyway.
    let (out, _) = run(&[
        "--all",
        "--baseline",
        "cases/diff_after.ptx",
        "cases/diff_after.ptx",
    ]);
    assert!(out.contains("unchanged"), "{out}");
}

#[test]
fn a_baseline_directory_pairs_by_file_name() {
    let (out, _) = run(&["--baseline", "cases", "cases/diff_after.ptx"]);
    // cases/diff_after.ptx against itself inside cases/: nothing moved.
    assert!(out.contains("no change"), "{out}");
}

#[test]
fn a_missing_baseline_is_fatal() {
    let (_, code) = run(&["--baseline", "nope.ptx", "cases/diff_after.ptx"]);
    assert_eq!(code, 2);
}

#[test]
fn diff_json_is_shaped_correctly() {
    let (out, _) = run(&[
        "--json",
        "--baseline",
        "cases/diff_after.ptx",
        "cases/diff_before.ptx",
    ]);
    assert!(out.starts_with('{') && out.trim_end().ends_with('}'));
    assert_eq!(out.matches('{').count(), out.matches('}').count());
    assert!(out.contains("\"verdict\": \"regressed\""));
    assert!(out.contains("\"blocking\": true"));
    assert!(out.contains("\"summary\": {\"regressed\": 1"));
}

// --- CLI behaviour ---------------------------------------------------------

#[test]
fn a_directory_is_walked() {
    let (out, _) = run(&["cases"]);
    for f in [
        "ptx001_local_memory.ptx",
        "clean_saxpy.ptx",
        "modern_tensor_cores.ptx",
    ] {
        assert!(out.contains(f), "missing {f}");
    }
}

#[test]
fn deny_controls_the_exit_code() {
    assert_eq!(
        run(&["cases/ptx001_local_memory.ptx"]).1,
        0,
        "silent by default"
    );
    assert_eq!(
        run(&["--deny", "error", "cases/ptx001_local_memory.ptx"]).1,
        1
    );
    assert_eq!(
        run(&["--deny", "PTX003", "cases/ptx003_fp64_literals.ptx"]).1,
        1
    );
    assert_eq!(
        run(&["--deny", "PTX003", "cases/ptx001_local_memory.ptx"]).1,
        0
    );
    assert_eq!(run(&["--deny", "error", "cases/clean_saxpy.ptx"]).1, 0);
}

#[test]
fn arch_override_changes_the_verdict() {
    let fp64 = |arch: &str| {
        run(&["--arch", arch, "cases/ptx003_fp64_literals.ptx"])
            .0
            .lines()
            .find(|l| l.contains("PTX003"))
            .unwrap_or_default()
            .to_string()
    };
    // Datacentre parts do FP64 at half rate; GeForce parts at 1/64.
    assert!(fp64("sm_80").contains("warning"), "{}", fp64("sm_80"));
    assert!(fp64("sm_89").contains("error"), "{}", fp64("sm_89"));
}

#[test]
fn block_size_changes_occupancy() {
    let occ = |n: &str| {
        run(&[
            "--arch",
            "sm_86",
            "--block-size",
            n,
            "cases/clean_saxpy.ptx",
        ])
        .0
        .lines()
        .find(|l| l.contains("occupancy"))
        .unwrap_or_default()
        .to_string()
    };
    assert_ne!(occ("32"), occ("256"));
    assert!(occ("32").contains("blocks/SM"));
}

#[test]
fn json_output_is_parseable_and_complete() {
    let (out, _) = run(&["--json", "cases/ptx001_local_memory.ptx"]);
    // No serde dependency, so check the shape by hand.
    assert!(out.starts_with('{') && out.trim_end().ends_with('}'));
    assert_eq!(out.matches("\"name\":").count(), 1);
    assert!(out.contains("\"summary\": {\"error\": 1"));
    assert!(out.contains("\"reg_source\": \"virtual\""));
    for q in [
        "\"code\": \"PTX001\"",
        "\"severity\": \"error\"",
        "\"occupancy\":",
    ] {
        assert!(out.contains(q), "missing {q}");
    }
    assert_eq!(out.matches('{').count(), out.matches('}').count());
}

#[test]
fn stdin_is_accepted() {
    let src = std::fs::read("cases/ptx001_local_memory.ptx").unwrap();
    let mut child = bin()
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child.stdin.as_mut().unwrap().write_all(&src).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("local_array"));
}

#[test]
fn a_missing_ptxas_report_is_fatal() {
    // Exit 2, not 1: a missing report is ptxlint failing, not a lint firing.
    let (_, code) = run(&["--ptxas-report", "nope.txt", "cases/clean_saxpy.ptx"]);
    assert_eq!(code, 2);
}

#[test]
fn bad_input_does_not_panic() {
    for junk in ["", "not ptx at all", "{{{{", ".entry broken(", "\u{0}\u{1}"] {
        let path = std::env::temp_dir().join(format!("ptxlint-junk-{}.ptx", junk.len()));
        std::fs::write(&path, junk).unwrap();
        let out = bin().arg(&path).output().unwrap();
        assert!(out.status.code().unwrap_or(-1) >= 0, "crashed on {junk:?}");
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("panicked"),
            "panicked on {junk:?}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn missing_file_is_an_error_not_a_panic() {
    let out = bin().arg("definitely/not/here.ptx").output().unwrap();
    assert_eq!(out.status.code(), Some(2), "an unreadable file is exit 2");
    assert!(String::from_utf8_lossy(&out.stderr).contains("No such file"));
}
