//! End-to-end tests. Every case has its own `.ptx`, generated from the Rust
//! kernel of the same name in `fixtures/examples/` (see `fixtures/generate.sh`),
//! so a fixture only ever triggers the lint it is meant to demonstrate.

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

fn lint(fixture: &str) -> String {
    run(&[&format!("tests/fixtures/{fixture}.ptx")]).0
}

/// Lint codes present in a report, in order.
fn codes(out: &str) -> Vec<String> {
    out.lines()
        .filter_map(|l| l.split('[').nth(1))
        .filter_map(|l| l.split([':', ']']).next())
        .filter(|c| c.starts_with("PTX"))
        .map(str::to_string)
        .collect()
}

// --- one file per case -----------------------------------------------------

#[test]
fn clean_saxpy_reports_nothing() {
    let out = lint("clean_saxpy");
    assert!(out.contains("clean_saxpy"));
    assert!(out.contains("no findings"), "{out}");
    assert_eq!(codes(&out), Vec::<String>::new());
    assert!(out.contains("0 error, 0 warning, 0 info"));
}

#[test]
fn fp64_literals_is_the_only_fp64_case() {
    let out = lint("fp64_literals");
    assert!(codes(&out).contains(&"PTX003".to_string()), "{out}");
    assert!(out.contains("FP64 instructions"));
    // The kernel is otherwise clean: no local memory, no spills.
    assert!(!codes(&out).contains(&"PTX001".to_string()));
    assert!(out.contains("local 0 B"));
}

#[test]
fn local_array_lands_in_local_memory() {
    let out = lint("local_array");
    assert!(codes(&out).contains(&"PTX001".to_string()), "{out}");
    assert!(out.contains("256 bytes of local memory"));
    assert!(out.contains("local 256 B"));
    // No doubles are involved, so PTX003 must stay quiet.
    assert!(!codes(&out).contains(&"PTX003".to_string()));
}

#[test]
fn libdevice_calls_are_not_inlined() {
    let out = lint("libdevice_calls");
    assert!(codes(&out).contains(&"PTX010".to_string()), "{out}");
    assert!(out.contains("non-inlined call"));
    assert!(out.contains("local 0 B"));
}

#[test]
fn modern_instructions_are_understood() {
    let out = lint("modern");
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
    let out = lint("nanoid_fixed");
    assert!(out.contains("nanoid_chacha"));
    assert!(
        !codes(&out).contains(&"PTX001".to_string()),
        "local memory is back:\n{out}"
    );
    assert!(out.contains("local 0 B"));
}

// --- CLI behaviour ---------------------------------------------------------

#[test]
fn a_directory_is_walked() {
    let (out, _) = run(&["tests/fixtures"]);
    for f in [
        "clean_saxpy.ptx",
        "fp64_literals.ptx",
        "local_array.ptx",
        "modern.ptx",
    ] {
        assert!(out.contains(f), "missing {f}");
    }
}

#[test]
fn deny_controls_the_exit_code() {
    assert_eq!(
        run(&["tests/fixtures/local_array.ptx"]).1,
        0,
        "silent by default"
    );
    assert_eq!(
        run(&["--deny", "error", "tests/fixtures/local_array.ptx"]).1,
        1
    );
    assert_eq!(
        run(&["--deny", "PTX003", "tests/fixtures/fp64_literals.ptx"]).1,
        1
    );
    assert_eq!(
        run(&["--deny", "PTX003", "tests/fixtures/local_array.ptx"]).1,
        0
    );
    assert_eq!(
        run(&["--deny", "error", "tests/fixtures/clean_saxpy.ptx"]).1,
        0
    );
}

#[test]
fn arch_override_changes_the_verdict() {
    let fp64 = |arch: &str| {
        run(&["--arch", arch, "tests/fixtures/fp64_literals.ptx"])
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
            "tests/fixtures/clean_saxpy.ptx",
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
    let (out, _) = run(&["--json", "tests/fixtures/local_array.ptx"]);
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
    // Braces must balance, i.e. the hand-rolled writer is not truncating.
    assert_eq!(out.matches('{').count(), out.matches('}').count());
}

#[test]
fn stdin_is_accepted() {
    let src = std::fs::read("tests/fixtures/local_array.ptx").unwrap();
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
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("No such file"));
}
