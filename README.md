# PTX Lint

[![license](https://img.shields.io/badge/license-MIT-cyan)](https://opensource.org/licenses/MIT) ![rust](https://img.shields.io/badge/rust-stable-lightgreen) ![dependencies](https://img.shields.io/badge/dependencies-0-purple) [![CI](https://github.com/meloalright/ptxlint/actions/workflows/ci.yml/badge.svg)](https://github.com/meloalright/ptxlint/actions/workflows/ci.yml)

The `ptxlint` is a static analyser for `NVIDIA PTX`. It reads the `.ptx` your GPU kernels already compile to and reports local memory traffic, `FP64` use, register pressure, shared memory budget and estimated occupancy. It needs no GPU, no `CUDA` install and no dependencies, so a kernel performance regression can fail `CI` on a machine that has never seen a graphics card.

针对 `NVIDIA PTX` 的静态分析工具 —— 直接读取 GPU kernel 编译出的 `.ptx`，报告 local memory 流量、`FP64` 使用、寄存器压力、shared memory 预算与估算占用率；不需要显卡、不需要安装 `CUDA`、零依赖，因此 kernel 的性能回归可以在一台从没见过显卡的机器上卡住 `CI`。

```
$ ptxlint kernels.ptx

  bad_local_array  line 61
    arch sm_70   regs/thread 176 (virtual, upper bound)   shared 0 B   local 256 B
    occupancy 12% (8 of 64 warps/SM, 1 blocks/SM @ 256 threads/block; limited by registers)
    198 instructions  ·  fp32 10  ·  int 23  ·  global 65  ·  local 72
    error   256 bytes of local memory, 72 local accesses — this lives in DRAM, not registers [PTX001:92]
            An array indexed by a runtime value cannot stay in registers.
```

## Usage

```shell
$ cargo install ptxlint
```

```shell
$ ptxlint kernels.ptx                   # a file, a directory, or - for stdin
$ ptxlint --deny error kernels.ptx      # exit 1 on errors, as a CI gate
$ ptxlint --json kernels.ptx            # machine-readable
$ ptxlint --ptxas kernels.ptx           # exact register counts, if CUDA is installed
```

## Lints

| | | |
|---|---|---|
| `PTX001` | local memory in use — an array indexed by a runtime value, backed by DRAM | 用到 local memory，运行时下标的数组实际落在显存里 |
| `PTX002` | register spills, needs `--ptxas` | 寄存器溢出，需要 `--ptxas` |
| `PTX003` | `FP64` instructions — in Rust a bare `0.5` is `f64` | `FP64` 指令 —— Rust 里裸写的 `0.5` 是 `f64` |
| `PTX004` | integer `div`/`rem` — GPUs have no integer divider | 整数除法取模 —— GPU 没有整数除法器 |
| `PTX005` | high register pressure | 寄存器压力过高 |
| `PTX006` | low estimated occupancy | 估算占用率过低 |
| `PTX007` | shared memory over budget, or capping residency | shared memory 超限或压制驻留块数 |
| `PTX008` | narrow, non-vectorised global accesses | 窄的、未向量化的 global 访问 |
| `PTX009` | no `.maxntid`/`.reqntid` launch bounds | 没有 launch bounds |
| `PTX010` | calls that were not inlined | 没有被内联的调用 |

## Accuracy

`PTX` only carries virtual registers, which `ptxas` coalesces on the way to `SASS`. Local and shared memory, instruction mix, `FP64` use and vector widths are therefore exact, while registers and occupancy are an upper bound, and are labelled as such. The scanner is deliberately not a full `PTX` grammar: an unknown opcode is recorded and matches no lint, rather than failing `CI` over an instruction NVIDIA shipped last month.

`PTX` 中只有虚拟寄存器，真实分配要等 `ptxas` 降级到 `SASS` 才确定。所以 local/shared memory、指令构成、`FP64` 使用与向量宽度是精确的，而寄存器数和占用率只是上界，报告中会明确标注。扫描器**故意不是**完整的 `PTX` 文法：不认识的指令原样记录、不匹配任何 lint，而不会因为 NVIDIA 上个月新加了一条指令就让 `CI` 崩掉。

It is not a profiler — for real tuning use `Nsight Compute`, for races use `compute-sanitizer`. This is the smoke alarm, not the fire brigade.

它不是 profiler —— 真要调优请用 `Nsight Compute`，查竞态请用 `compute-sanitizer`。这是烟雾报警器，不是消防队。

## Library Development

#### Lint Development

```shell
$ cargo test
$ cargo clippy --all-targets -- -D warnings
```

#### Fixture Kernel Development

The `.ptx` test fixtures are generated from real Rust kernels, and need a nightly toolchain.

`.ptx` 测试素材由真实的 Rust kernel 编译产生，需要 nightly 工具链。

```shell
$ cd fixtures
$ cargo build --release
$ cp target/nvptx64-nvidia-cuda/release/ptxlint_fixtures.ptx ../tests/fixtures/rust_kernels.ptx
```

## License

[MIT](https://opensource.org/licenses/MIT)
