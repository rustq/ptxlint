# ptxlint

Static analysis and performance lints for NVIDIA PTX — **no GPU, no CUDA install, no dependencies**.

Rust can compile kernels to PTX (`--target nvptx64-nvidia-cuda`, rust-cuda, cuda-oxide), but
nothing tells you that the array you just wrote landed in DRAM instead of registers. `ptxlint`
reads the `.ptx` your build already produces and reports it, so a kernel regression fails CI
on a machine that has never seen a graphics card.

```
$ ptxlint target/nvptx64-nvidia-cuda/release/my_kernels.ptx

  bad_local_array  line 61
    arch sm_70   regs/thread 176 (virtual, upper bound)   shared 0 B   local 256 B
    occupancy 12% (8 of 64 warps/SM, 1 blocks/SM @ 256 threads/block, assumed; limited by registers)
    198 instructions  ·  fp32 10  ·  int 23  ·  branch 1  ·  global 65  ·  local 72  ·  param 3
    error   256 bytes of local memory, 72 local accesses — this lives in DRAM, not registers [PTX001:92]
            An array indexed by a runtime value cannot stay in registers. Use a fixed index,
            unroll the loop, or move the array to shared memory.
            at lines 92, 94, 96
```

## Install

```bash
cargo install --path .
```

## Use

```bash
ptxlint kernels.ptx                      # one file
ptxlint target/nvptx64-nvidia-cuda/      # or a directory, recursively
rustc ... --emit asm -o - | ptxlint -    # or stdin

ptxlint --deny error kernels.ptx         # exit 1 on any error  (CI gate)
ptxlint --deny PTX003 kernels.ptx        # exit 1 on a specific lint
ptxlint --json kernels.ptx               # machine-readable
ptxlint --arch sm_89 --block-size 128 kernels.ptx
ptxlint --ptxas kernels.ptx              # exact register counts, if CUDA is installed
```

## Lints

| Code | Severity | What it catches |
|---|---|---|
| PTX001 | error | Local memory in use — an array indexed by a runtime value, backed by DRAM |
| PTX002 | error | Register spills (requires `--ptxas`) |
| PTX003 | error on GeForce, warning elsewhere | FP64 instructions — in Rust a bare `0.5` is `f64` and promotes the whole expression |
| PTX004 | warning | Integer `div`/`rem` — GPUs have no integer divider |
| PTX005 | warning / info | High register pressure |
| PTX006 | warning | Low estimated occupancy (requires `--ptxas` to be meaningful) |
| PTX007 | error / info | Shared memory over the per-block limit, or capping residency |
| PTX008 | info | Narrow, non-vectorised global accesses |
| PTX009 | info | No `.maxntid`/`.reqntid` launch bounds |
| PTX010 | info | Calls that were not inlined |

## How much to trust the numbers

PTX contains *virtual* registers in near-SSA form; ptxas coalesces them when it lowers PTX to
SASS. So:

- **Exact from the PTX:** local memory, shared memory, instruction mix, FP64 use, vector widths,
  launch bounds, calls. These are what the lints key on.
- **Upper bound only:** registers per thread, and therefore occupancy. The report labels these
  `(virtual, upper bound)`. Pass `--ptxas` when the CUDA toolkit is available and the counts
  become exact (`ptxas -v`), which also enables PTX002 and makes PTX006 meaningful.

The occupancy model is the textbook one — 256-register-per-warp allocation granularity,
128-byte shared memory granularity, per-architecture SM limits — and is unit-tested against
NVIDIA's occupancy calculator for sm_75/80/86.

## Does it actually find anything?

Yes — on its first run against a real kernel. A hand-written ChaCha20 nanoid generator looked
fine and was not:

```
error   64 bytes of local memory, 19 local accesses [PTX001]
```

The 16-word cipher state was being computed in registers, stored to local memory, and read back,
because the output loop handed the array to an iterator. Forcing the indices to be literals
removed it (`local 0 B`) and made the same code 12% faster on the CPU as a side effect.

## Design notes

The scanner is deliberately not a full PTX grammar. A CI tool must not hard-fail on an
instruction NVIDIA shipped last month, so unknown opcodes are recorded verbatim and simply match
no lint — `wmma`, `cp.async` and friends parse fine without the tool knowing what they mean.

## Roadmap

- `bar.sync` inside divergent control flow (needs a CFG)
- Shared-memory bank-conflict estimation
- SASS-level analysis via `nvdisasm`
- A `cargo ptxlint` subcommand that finds the PTX in `target/` on its own

## License

MIT OR Apache-2.0
