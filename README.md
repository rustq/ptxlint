# PTX Lint

[![license](https://img.shields.io/badge/license-MIT-cyan)](https://opensource.org/licenses/MIT) ![rust](https://img.shields.io/badge/rust-1.75%2B-lightgreen) ![dependencies](https://img.shields.io/badge/dependencies-0-purple) [![crates](https://img.shields.io/crates/v/ptxlint)](https://crates.io/crates/ptxlint) [![docs](https://img.shields.io/docsrs/ptxlint)](https://docs.rs/ptxlint) [![downloads](https://img.shields.io/crates/d/ptxlint)](https://crates.io/crates/ptxlint) [![CI](https://github.com/rustq/ptxlint/actions/workflows/ci.yml/badge.svg)](https://github.com/rustq/ptxlint/actions/workflows/ci.yml)

`ptxlint` reads the `.ptx` your GPU kernels compile to and reports local memory, FP64 use, register pressure and occupancy. No GPU, no CUDA install, no dependencies.

NVIDIA PTX 静态分析工具 —— 读取 kernel 编译出的 `.ptx`，报告 local memory、FP64、寄存器压力与占用率。不需要显卡和 CUDA。

---

## What It Is And Why It Exists

Rust can now compile kernels to PTX, but nothing tells you that the array you just wrote landed in DRAM instead of registers. That kind of mistake is invisible in the source: whether a scratch array stays in registers depends on how it is indexed and on whether the loop got unrolled, not on how the Rust reads. It usually surfaces much later, on a machine with a GPU, long after the change that caused it.

`ptxlint` moves that feedback earlier. PTX is the first point in the pipeline where the compiler has committed to its decisions — memory space, vector width, instruction types are all fixed — and it is still typed, readable text. Analysing it needs nothing from NVIDIA, so the same check runs on a laptop, in review, and on a CI runner with no GPU attached.

> [!NOTE]
> Every Rust GPU toolchain ends at PTX — the in-tree `nvptx64-nvidia-cuda` target, rust-cuda, and NVIDIA's cuda-oxide alike. A tool that reads the common output works regardless of which one you picked.

---

## What You Get

| Capability | What It Detects | Why It Matters |
| --- | --- | --- |
| Lints | Local memory, spills, FP64, integer division, register pressure, shared memory, narrow accesses, launch bounds, stray calls | Ten specific mistakes with a fix for each, not a wall of statistics |
| Occupancy model | Register, shared-memory, warp and block limits per architecture | Tells you which resource is capping residency, so you tune the right one |
| Baseline diff | Metric deltas between two builds of the same kernels | Answers "did my change make it worse", which is the question review actually asks |
| ptxas integration | Exact register counts and spill bytes | Turns the estimates into measurements when a CUDA toolkit is available |
| CI gating | Per-lint and per-regression exit codes | Fails the build on the findings you chose, and only those |

---

## Quick Start

### 1. Install

```shell
cargo install ptxlint
```

### 2. Build your kernels to PTX

```shell
cargo build --release --target nvptx64-nvidia-cuda
```

### 3. Analyse

```shell
ptxlint target/nvptx64-nvidia-cuda/release/my_kernels.ptx
```

```
  local_array  line 5
    arch sm_70   regs/thread 176 (virtual, upper bound)   shared 0 B   local 256 B
    occupancy 12% (8 of 64 warps/SM, 1 blocks/SM @ 256 threads/block, assumed; limited by registers)
    198 instructions  ·  fp32 10  ·  int 23  ·  branch 1  ·  global 65  ·  local 72  ·  param 3
    error   256 bytes of local memory, 72 local accesses — this lives in DRAM, not registers [PTX001:42]
            An array indexed by a runtime value cannot stay in registers. Use a fixed index, unroll the loop, or move the array to shared memory.
```

A file, a directory, or `-` for stdin. Add `--json` for machine-readable output, `--arch sm_89` to override the target, and `--ptxas` to get exact register counts when CUDA is installed.

> [!TIP]
> If your build machine has CUDA but your lint job does not, save the `ptxas -v` log there and replay it with `--ptxas-report build.log`.

---

## Lints

Every lint has a runnable case in [`cases/`](cases), compiled from the Rust kernel linked beside it. Try one with `ptxlint cases/ptx001_local_memory.ptx`.

| | What It Detects | Case | Kernel |
| --- | --- | --- | --- |
| `PTX001` | Local memory in use — an array indexed by a runtime value, backed by DRAM | [ptx001](cases/ptx001_local_memory.ptx) | [rs](fixtures/examples/ptx001_local_memory.rs) |
| `PTX002` | Register spills, needs `--ptxas` | [ptx002](cases/ptx002_register_spills.ptx) | [rs](fixtures/examples/ptx002_register_spills.rs) |
| `PTX003` | FP64 instructions — in Rust a bare `0.5` is `f64` | [ptx003](cases/ptx003_fp64_literals.ptx) | [rs](fixtures/examples/ptx003_fp64_literals.rs) |
| `PTX004` | Integer `div`/`rem` — GPUs have no integer divider | [ptx004](cases/ptx004_integer_division.ptx) | [rs](fixtures/examples/ptx004_integer_division.rs) |
| `PTX005` | High register pressure | [ptx005](cases/ptx005_register_pressure.ptx) | [rs](fixtures/examples/ptx005_register_pressure.rs) |
| `PTX006` | Low estimated occupancy | [ptx006](cases/ptx006_low_occupancy.ptx) | [rs](fixtures/examples/ptx006_low_occupancy.rs) |
| `PTX007` | Shared memory over budget, or capping residency | [ptx007](cases/ptx007_shared_memory.ptx) | hand-written |
| `PTX008` | Narrow, non-vectorised global accesses | [ptx008](cases/ptx008_narrow_access.ptx) | [rs](fixtures/examples/ptx008_narrow_access.rs) |
| `PTX009` | No `.maxntid`/`.reqntid` launch bounds | [ptx009](cases/ptx009_launch_bounds.ptx) | [rs](fixtures/examples/ptx009_launch_bounds.rs) |
| `PTX010` | Calls that were not inlined | [ptx010](cases/ptx010_uninlined_calls.ptx) | [rs](fixtures/examples/ptx010_uninlined_calls.rs) |

`cases/` also holds [clean_saxpy](cases/clean_saxpy.ptx), the control that must report nothing, [modern_tensor_cores](cases/modern_tensor_cores.ptx) for `wmma` and `cp.async`, and [nanoid_regression](cases/nanoid_regression.ptx), a kernel this tool found a real bug in.

---

## Baseline Diff

A report tells you whether a kernel is bad. Review usually asks something else: did this change make it worse? `--baseline` matches kernels by name across two builds and prints only what moved.

```shell
ptxlint --baseline cases/diff_before.ptx cases/diff_after.ptx
```

```
  mix16  improved
    ↓ local memory (B)     64 → 0 (-64)
    ↓ registers/thread     180 → 69 (-111)
    ↑ occupancy (%)        13 → 38 (+25)
    ↓ instructions         156 → 63 (-93)
    fixed    PTX001
```

Only the exact metrics can fail a build: local memory, spills, shared memory, a kernel that disappeared, a new error, and registers when both sides came from ptxas. Instruction counts and the virtual-register estimate move around too much to gate on, so they are reported and never block.

> [!TIP]
> Keep the previous build's `.ptx` as a CI artifact and point `--baseline` at the directory. Kernels are paired by file name, then by kernel name.

---

## CI Integration

```yaml
- run: cargo build --release --target nvptx64-nvidia-cuda
- run: ptxlint --deny error target/nvptx64-nvidia-cuda/release/
- run: ptxlint --deny regression --baseline baseline/ target/nvptx64-nvidia-cuda/release/
```

Exit codes are `0` for a clean run, `1` when a denied lint or regression fired, and `2` when ptxlint itself could not run. Keeping `1` and `2` apart lets a check assert that a lint still fires rather than silently passing on a missing file.

---

## Accuracy And Limits

PTX only carries virtual registers, which ptxas coalesces on the way to SASS. Local and shared memory, instruction mix, FP64 use and vector widths are therefore exact; registers and occupancy are an upper bound and are labelled as such in the report. Pass `--ptxas` or `--ptxas-report` to replace the estimate with the real number.

The scanner is deliberately not a full PTX grammar. An unknown opcode is recorded and matches no lint, rather than failing CI over an instruction NVIDIA shipped last month — `wmma`, `cp.async` and friends parse fine without the tool knowing what they mean.

> [!IMPORTANT]
> This is not a profiler. For real tuning use Nsight Compute, and for races use compute-sanitizer. `ptxlint` is the smoke alarm, not the fire brigade.

---

## Development

```shell
cargo test
cargo clippy --all-targets -- -D warnings
```

Each case in `fixtures/examples/` is a real Rust kernel that compiles to its own `.ptx`, so a fixture only ever triggers the lint it demonstrates. Regenerating them needs a nightly toolchain with the `nvptx64-nvidia-cuda` target.

```shell
./fixtures/generate.sh
```

`showcase.sh` walks through every feature against the cases, and is what the [Showcase](https://github.com/rustq/ptxlint/actions/workflows/ci.yml) step in CI runs — the log is a live demo on a runner with no GPU.

```shell
cargo build --release && ./showcase.sh
```

Prior art: [cuda-sage](https://github.com/hkevin01/cuda-sage) is a Python static PTX analyser covering similar ground, and the baseline diff idea came from it.

## License

[MIT](https://opensource.org/licenses/MIT)
