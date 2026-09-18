//! End-to-end tests against PTX produced by the real Rust NVPTX backend
//! (`tests/fixtures/*.ptx`, generated from `fixtures/src/lib.rs`).

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

const RUST_KERNELS: &str = "tests/fixtures/rust_kernels.ptx";

#[test]
fn reports_every_kernel_in_a_module() {
    let (out, code) = run(&[RUST_KERNELS]);
    for k in [
        "good_saxpy",
        "bad_f64_literals",
        "bad_local_array",
        "transcendental",
    ] {
        assert!(out.contains(k), "missing kernel {k} in:\n{out}");
    }
    assert_eq!(code, 0, "without --deny the exit code stays 0");
}

#[test]
fn finds_local_memory_and_fp64() {
    let (out, _) = run(&[RUST_KERNELS]);
    assert!(out.contains("PTX001"), "local memory lint missing:\n{out}");
    assert!(out.contains("PTX003"), "fp64 lint missing:\n{out}");
    assert!(out.contains("bytes of local memory"));
}

#[test]
fn clean_kernel_is_clean() {
    let (out, _) = run(&[RUST_KERNELS]);
    let section = out
        .split("good_saxpy")
        .nth(1)
        .unwrap()
        .split("\n\n")
        .next()
        .unwrap();
    assert!(
        !section.contains("error"),
        "good_saxpy should be clean:\n{section}"
    );
    assert!(
        !section.contains("warning"),
        "good_saxpy should be clean:\n{section}"
    );
}

#[test]
fn nanoid_kernel_has_no_local_memory_after_the_fix() {
    let (out, _) = run(&["tests/fixtures/nanoid_fixed.ptx"]);
    assert!(out.contains("nanoid_chacha"));
    assert!(
        !out.contains("PTX001"),
        "regression: local memory is back:\n{out}"
    );
    assert!(out.contains("local 0 B"));
}

#[test]
fn modern_instructions_are_understood() {
    let (out, _) = run(&["tests/fixtures/modern.ptx"]);
    assert!(out.contains("hgemm_tc"));
    assert!(
        out.contains("tensor 2"),
        "wmma should count as tensor ops:\n{out}"
    );
    assert!(out.contains("shared 16384 B"));
    assert!(out.contains("arch sm_90"));
}

#[test]
fn deny_controls_the_exit_code() {
    assert_eq!(run(&["--deny", "error", RUST_KERNELS]).1, 1);
    assert_eq!(run(&["--deny", "PTX003", RUST_KERNELS]).1, 1);
    assert_eq!(run(&["--deny", "PTX999", RUST_KERNELS]).1, 0);
    assert_eq!(run(&["--deny", "error", "tests/fixtures/modern.ptx"]).1, 0);
}

#[test]
fn arch_override_changes_the_verdict() {
    // sm_80 is a datacentre part: FP64 is only a warning there.
    let (a100, _) = run(&["--arch", "sm_80", RUST_KERNELS]);
    let (ada, _) = run(&["--arch", "sm_89", RUST_KERNELS]);
    let fp64_line = |s: &str| {
        s.lines()
            .find(|l| l.contains("PTX003"))
            .unwrap_or_default()
            .to_string()
    };
    assert!(fp64_line(&a100).contains("warning"), "{}", fp64_line(&a100));
    assert!(fp64_line(&ada).contains("error"), "{}", fp64_line(&ada));
}

#[test]
fn block_size_changes_occupancy() {
    let (small, _) = run(&["--arch", "sm_86", "--block-size", "32", RUST_KERNELS]);
    let (big, _) = run(&["--arch", "sm_86", "--block-size", "256", RUST_KERNELS]);
    assert!(small.contains("blocks/SM"), "{small}");
    assert_ne!(
        small.lines().find(|l| l.contains("occupancy")),
        big.lines().find(|l| l.contains("occupancy")),
    );
}

#[test]
fn json_output_is_parseable_and_complete() {
    let (out, _) = run(&["--json", RUST_KERNELS]);
    // No serde dependency, so check the shape by hand.
    assert!(out.starts_with('{') && out.trim_end().ends_with('}'));
    assert_eq!(out.matches("\"name\":").count(), 4);
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
fn reads_a_directory_and_stdin() {
    let (dir, _) = run(&["tests/fixtures"]);
    assert!(dir.contains("modern.ptx") && dir.contains("rust_kernels.ptx"));

    let src = std::fs::read(RUST_KERNELS).unwrap();
    let mut child = bin()
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child.stdin.as_mut().unwrap().write_all(&src).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("bad_local_array"));
}

#[test]
fn bad_input_does_not_panic() {
    for junk in ["", "not ptx at all", "{{{{", ".entry broken(", "\u{0}\u{1}"] {
        let dir = std::env::temp_dir().join(format!("ptxlint-junk-{}.ptx", junk.len()));
        std::fs::write(&dir, junk).unwrap();
        let out = bin().arg(&dir).output().unwrap();
        assert!(out.status.code().unwrap_or(-1) >= 0, "crashed on {junk:?}");
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("panicked"),
            "panicked on {junk:?}"
        );
        let _ = std::fs::remove_file(&dir);
    }
}

#[test]
fn missing_file_is_an_error_not_a_panic() {
    let out = bin().arg("definitely/not/here.ptx").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("No such file"));
}
