# ptxlint

Static analysis and performance lints for NVIDIA PTX — **no GPU, no CUDA install, no dependencies**.

*[中文](#中文)*

```
$ ptxlint kernels.ptx

  bad_local_array  line 61
    arch sm_70   regs/thread 176 (virtual, upper bound)   shared 0 B   local 256 B
    occupancy 12% (8 of 64 warps/SM, 1 blocks/SM @ 256 threads/block; limited by registers)
    198 instructions  ·  fp32 10  ·  int 23  ·  global 65  ·  local 72
    error   256 bytes of local memory, 72 local accesses — this lives in DRAM, not registers [PTX001:92]
            An array indexed by a runtime value cannot stay in registers.
```

Rust can compile kernels to PTX, but nothing tells you the array you just wrote landed in DRAM
instead of registers. `ptxlint` reads the `.ptx` your build already produces, so kernel
regressions fail CI on a machine that has never seen a graphics card.

## Install & use

```bash
cargo install ptxlint

ptxlint kernels.ptx                   # file, directory, or - for stdin
ptxlint --deny error kernels.ptx      # exit 1 on errors — CI gate
ptxlint --json kernels.ptx            # machine-readable
ptxlint --ptxas kernels.ptx           # exact register counts, if CUDA is installed
```

## Lints

| | |
|---|---|
| `PTX001` | local memory in use — an array indexed by a runtime value, backed by DRAM |
| `PTX002` | register spills (needs `--ptxas`) |
| `PTX003` | FP64 instructions — in Rust a bare `0.5` is `f64` and promotes the whole expression |
| `PTX004` | integer `div`/`rem` — GPUs have no integer divider |
| `PTX005` | high register pressure |
| `PTX006` | low estimated occupancy |
| `PTX007` | shared memory over budget, or capping residency |
| `PTX008` | narrow, non-vectorised global accesses |
| `PTX009` | no `.maxntid`/`.reqntid` launch bounds |
| `PTX010` | calls that were not inlined |

## Accuracy

PTX only has *virtual* registers; ptxas coalesces them on the way to SASS. So local/shared memory,
instruction mix, FP64 use and vector widths are **exact**, while registers and occupancy are an
**upper bound** (labelled as such). `--ptxas` makes them exact and enables `PTX002`.

The scanner is deliberately not a full PTX grammar: unknown opcodes are recorded and match no
lint, rather than failing CI on an instruction NVIDIA shipped last month.

Not a profiler. For real tuning use Nsight Compute; for races use compute-sanitizer. This is the
smoke alarm, not the fire brigade.

License: MIT OR Apache-2.0

---

## 中文

给 NVIDIA PTX 做静态分析和性能 lint —— **不需要显卡、不需要装 CUDA、零依赖**。

Rust 现在能把 kernel 编译成 PTX，但没有任何东西会告诉你：你刚写的那个数组进了显存而不是寄存器。
`ptxlint` 直接读构建产物 `.ptx`，让 kernel 的性能回归能在**一台从没见过显卡的机器上**卡住 CI。

### 安装与使用

```bash
cargo install ptxlint

ptxlint kernels.ptx                   # 文件、目录，或 - 读 stdin
ptxlint --deny error kernels.ptx      # 有 error 就退出码 1 —— CI 门禁
ptxlint --json kernels.ptx            # 机器可读
ptxlint --ptxas kernels.ptx           # 装了 CUDA 的话，寄存器数变精确
```

### 十条 lint

| | |
|---|---|
| `PTX001` | 用到 local memory —— 运行时下标的数组，实际落在显存里 |
| `PTX002` | 寄存器溢出（需 `--ptxas`）|
| `PTX003` | FP64 指令 —— Rust 里裸写的 `0.5` 是 `f64`，会把整个表达式提升 |
| `PTX004` | 整数除法/取模 —— GPU 没有整数除法器 |
| `PTX005` | 寄存器压力过高 |
| `PTX006` | 估算占用率过低 |
| `PTX007` | shared memory 超限，或压制了驻留块数 |
| `PTX008` | 窄的、未向量化的 global 访问 |
| `PTX009` | 没有 `.maxntid`/`.reqntid` launch bounds |
| `PTX010` | 没有被内联的调用 |

### 准确度

PTX 里只有**虚拟寄存器**，真实分配要等 ptxas 降到 SASS 才定。所以 local/shared memory、指令构成、
FP64 使用、向量宽度是**精确的**，而寄存器数和占用率只是**上界**（报告里会标注）。加 `--ptxas`
可以变精确，并激活 `PTX002`。

扫描器**故意不是**完整的 PTX 文法：不认识的指令原样记录、不匹配任何 lint，而不是因为 NVIDIA
上个月加了条新指令就让 CI 崩掉。

它不是 profiler。真要调优请用 Nsight Compute，查竞态请用 compute-sanitizer。**这是烟雾报警器，不是消防队。**

许可证：MIT OR Apache-2.0
