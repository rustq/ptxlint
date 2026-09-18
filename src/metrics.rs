//! Per-kernel resource metrics and an occupancy model.

use crate::parse::{Inst, Kernel, type_bits};
use std::collections::BTreeMap;

/// Hardware limits per SM. Numbers from the CUDA C Programming Guide's
/// "Compute Capabilities" table.
#[derive(Debug, Clone, Copy)]
pub struct Arch {
    pub name: &'static str,
    pub max_warps_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub regs_per_sm: u32,
    pub max_regs_per_thread: u32,
    pub shared_per_sm: u32,
    pub max_shared_per_block: u32,
    /// True for GeForce parts, where FP64 runs at 1/64 of FP32.
    pub weak_fp64: bool,
}

const ARCHS: &[Arch] = &[
    Arch {
        name: "sm_70",
        max_warps_per_sm: 64,
        max_blocks_per_sm: 32,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 98304,
        max_shared_per_block: 98304,
        weak_fp64: false,
    },
    Arch {
        name: "sm_72",
        max_warps_per_sm: 64,
        max_blocks_per_sm: 32,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 98304,
        max_shared_per_block: 98304,
        weak_fp64: true,
    },
    Arch {
        name: "sm_75",
        max_warps_per_sm: 32,
        max_blocks_per_sm: 16,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 65536,
        max_shared_per_block: 65536,
        weak_fp64: true,
    },
    Arch {
        name: "sm_80",
        max_warps_per_sm: 64,
        max_blocks_per_sm: 32,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 167936,
        max_shared_per_block: 166912,
        weak_fp64: false,
    },
    Arch {
        name: "sm_86",
        max_warps_per_sm: 48,
        max_blocks_per_sm: 16,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 102400,
        max_shared_per_block: 101376,
        weak_fp64: true,
    },
    Arch {
        name: "sm_87",
        max_warps_per_sm: 48,
        max_blocks_per_sm: 16,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 167936,
        max_shared_per_block: 166912,
        weak_fp64: true,
    },
    Arch {
        name: "sm_89",
        max_warps_per_sm: 48,
        max_blocks_per_sm: 24,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 102400,
        max_shared_per_block: 101376,
        weak_fp64: true,
    },
    Arch {
        name: "sm_90",
        max_warps_per_sm: 64,
        max_blocks_per_sm: 32,
        regs_per_sm: 65536,
        max_regs_per_thread: 255,
        shared_per_sm: 233472,
        max_shared_per_block: 232448,
        weak_fp64: false,
    },
];

pub const DEFAULT_ARCH: &str = "sm_80";

impl Arch {
    /// Look up by `sm_XX`, tolerating suffixes like `sm_90a`.
    pub fn lookup(target: &str) -> Option<Arch> {
        // `.target sm_52, debug` — the architecture is the first entry.
        let t = target.split(',').next().unwrap_or(target).trim();
        let t = t.strip_suffix(['a', 'f']).unwrap_or(t);
        ARCHS.iter().copied().find(|a| a.name == t)
    }

    pub fn lookup_or_default(target: Option<&str>) -> (Arch, bool) {
        match target.and_then(Arch::lookup) {
            Some(a) => (a, true),
            None => (Arch::lookup(DEFAULT_ARCH).unwrap(), false),
        }
    }
}

/// Where the register count came from — this changes how much you should trust it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegSource {
    /// Exact, from `ptxas -v`.
    Ptxas,
    /// Estimated from virtual register declarations in the PTX. Virtual
    /// registers are in SSA-ish form and get coalesced by ptxas, so this is an
    /// upper bound, often a loose one.
    VirtualUpperBound,
}

#[derive(Debug, Clone, Default)]
pub struct InstMix {
    pub total: u32,
    pub fp32: u32,
    pub fp64: u32,
    pub fp16: u32,
    pub int: u32,
    pub int_divrem: u32,
    pub branch: u32,
    pub predicated: u32,
    pub sync: u32,
    pub atomic: u32,
    pub tensor: u32,
    pub calls: u32,
    /// Loads/stores per state space.
    pub mem: BTreeMap<String, u32>,
    /// Global accesses that move <= 4 bytes per thread per instruction.
    pub narrow_global: u32,
    pub global_access: u32,
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub arch: Arch,
    pub arch_known: bool,
    pub regs_per_thread: u32,
    pub reg_source: RegSource,
    pub spill_bytes: Option<u64>,
    pub local_bytes: u64,
    pub shared_bytes: u64,
    pub block_size: u32,
    pub block_size_from_ptx: bool,
    pub mix: InstMix,
}

fn classify(insts: &[Inst]) -> InstMix {
    let mut m = InstMix::default();
    for i in insts {
        m.total += 1;
        if i.predicated {
            m.predicated += 1;
        }
        let ty = i.ty().unwrap_or("");
        let is_mem = matches!(
            i.base.as_str(),
            "ld" | "st" | "atom" | "red" | "cp" | "prefetch"
        );
        if matches!(i.base.as_str(), "bra" | "brx") {
            m.branch += 1;
        }
        if i.base == "bar" || i.base == "barrier" || i.base == "membar" || i.base == "fence" {
            m.sync += 1;
        }
        if i.base == "atom" || i.base == "red" {
            m.atomic += 1;
        }
        if matches!(
            i.base.as_str(),
            "wmma" | "mma" | "ldmatrix" | "stmatrix" | "wgmma"
        ) {
            m.tensor += 1;
        }
        if i.base == "call" {
            m.calls += 1;
        }
        if matches!(i.base.as_str(), "div" | "rem") && ty.starts_with(['s', 'u']) {
            m.int_divrem += 1;
        }
        if !is_mem && i.base != "cvt" && i.base != "mov" {
            match ty {
                "f64" => m.fp64 += 1,
                "f32" => m.fp32 += 1,
                "f16" | "f16x2" | "bf16" | "bf16x2" => m.fp16 += 1,
                t if t.starts_with(['s', 'u']) => m.int += 1,
                _ => {}
            }
        }
        // cvt.f64.* and mov.f64 still occupy the FP64 pipe.
        if (i.base == "cvt" || i.base == "mov") && i.quals.iter().any(|q| q == "f64") {
            m.fp64 += 1;
        }
        if is_mem && let Some(space) = i.space() {
            *m.mem.entry(space.to_string()).or_insert(0) += 1;
            if space == "global" && matches!(i.base.as_str(), "ld" | "st") {
                m.global_access += 1;
                let bytes = type_bits(i.ty().unwrap_or("b32")) / 8 * i.vector_width();
                if bytes <= 4 {
                    m.narrow_global += 1;
                }
            }
        }
    }
    m
}

/// Sum of declared virtual registers, converted to 32-bit register slots.
/// Predicate registers are allocated separately on real hardware, so they are
/// excluded.
pub fn virtual_regs(k: &Kernel) -> u32 {
    k.regs
        .iter()
        .filter(|(ty, _)| *ty != "pred")
        .map(|(ty, n)| {
            let slots = type_bits(ty).div_ceil(32).max(1);
            n * slots
        })
        .sum()
}

pub struct Options {
    pub arch: Option<String>,
    pub block_size: Option<u32>,
    /// Exact per-kernel register counts and spill bytes from `ptxas -v`.
    pub ptxas: BTreeMap<String, (u32, u64)>,
}

pub fn analyse(
    k: &Kernel,
    module_target: Option<&str>,
    module_shared: u64,
    opts: &Options,
) -> Metrics {
    let (arch, arch_known) = Arch::lookup_or_default(opts.arch.as_deref().or(module_target));
    let (regs_per_thread, reg_source, spill_bytes) = match opts.ptxas.get(&k.name) {
        Some(&(regs, spill)) => (regs, RegSource::Ptxas, Some(spill)),
        None => (
            virtual_regs(k).min(arch.max_regs_per_thread),
            RegSource::VirtualUpperBound,
            None,
        ),
    };
    let ptx_block = k.reqntid.or(k.maxntid).map(|(x, y, z)| x * y * z);
    let block_size = opts.block_size.or(ptx_block).unwrap_or(256);
    Metrics {
        arch,
        arch_known,
        regs_per_thread: regs_per_thread.max(1),
        reg_source,
        spill_bytes,
        local_bytes: k.local_bytes,
        shared_bytes: k.shared_bytes + module_shared,
        block_size,
        block_size_from_ptx: opts.block_size.is_none() && ptx_block.is_some(),
        mix: classify(&k.insts),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limiter {
    Registers,
    SharedMemory,
    BlocksPerSm,
    WarpsPerSm,
}

impl Limiter {
    pub fn label(self) -> &'static str {
        match self {
            Limiter::Registers => "registers",
            Limiter::SharedMemory => "shared memory",
            Limiter::BlocksPerSm => "blocks/SM",
            Limiter::WarpsPerSm => "warps/SM (already maxed)",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Occupancy {
    pub active_warps: u32,
    pub max_warps: u32,
    pub blocks_per_sm: u32,
    pub limiter: Limiter,
}

impl Occupancy {
    pub fn ratio(&self) -> f64 {
        self.active_warps as f64 / self.max_warps as f64
    }
}

/// Textbook CUDA occupancy calculation.
///
/// Registers are allocated per warp in units of 256 on Volta and later, and
/// shared memory in 128-byte units.
pub fn occupancy(m: &Metrics) -> Occupancy {
    let a = m.arch;
    let warps_per_block = m.block_size.div_ceil(32).max(1);

    let regs_per_warp = (m.regs_per_thread * 32).div_ceil(256) * 256;
    let warps_by_regs = a.regs_per_sm / regs_per_warp.max(1);
    let blocks_by_regs = warps_by_regs / warps_per_block;

    let shared_per_block = m.shared_bytes.div_ceil(128) * 128;
    let blocks_by_shared = (a.shared_per_sm as u64)
        .checked_div(shared_per_block)
        .map_or(u32::MAX, |n| n as u32);

    let blocks_by_warps = a.max_warps_per_sm / warps_per_block;
    let blocks_by_limit = a.max_blocks_per_sm;

    let blocks = blocks_by_regs
        .min(blocks_by_shared)
        .min(blocks_by_warps)
        .min(blocks_by_limit);
    let limiter = if blocks == blocks_by_regs {
        Limiter::Registers
    } else if blocks == blocks_by_shared {
        Limiter::SharedMemory
    } else if blocks == blocks_by_limit {
        Limiter::BlocksPerSm
    } else {
        Limiter::WarpsPerSm
    };
    Occupancy {
        active_warps: (blocks * warps_per_block).min(a.max_warps_per_sm),
        max_warps: a.max_warps_per_sm,
        blocks_per_sm: blocks,
        limiter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(arch: &str, regs: u32, shared: u64, block: u32) -> Metrics {
        Metrics {
            arch: Arch::lookup(arch).unwrap(),
            arch_known: true,
            regs_per_thread: regs,
            reg_source: RegSource::Ptxas,
            spill_bytes: Some(0),
            local_bytes: 0,
            shared_bytes: shared,
            block_size: block,
            block_size_from_ptx: false,
            mix: InstMix::default(),
        }
    }

    // Cross-checked against NVIDIA's occupancy calculator.
    #[test]
    fn a100_32_regs_is_full_occupancy() {
        let o = occupancy(&metrics("sm_80", 32, 0, 256));
        assert_eq!(o.active_warps, 64);
        assert_eq!(o.blocks_per_sm, 8);
    }

    #[test]
    fn a100_64_regs_halves_occupancy() {
        let o = occupancy(&metrics("sm_80", 64, 0, 256));
        // 64 regs -> 2048 regs/warp -> 32 warps/SM -> 4 blocks of 8 warps.
        assert_eq!(o.active_warps, 32);
        assert_eq!(o.blocks_per_sm, 4);
        assert_eq!(o.limiter, Limiter::Registers);
        assert!((o.ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn shared_memory_can_be_the_limiter() {
        // 64 KB per block on a 100 KB SM -> a single resident block.
        let o = occupancy(&metrics("sm_86", 32, 65536, 256));
        assert_eq!(o.blocks_per_sm, 1);
        assert_eq!(o.limiter, Limiter::SharedMemory);
    }

    #[test]
    fn turing_caps_at_32_warps() {
        let o = occupancy(&metrics("sm_75", 24, 0, 256));
        assert_eq!(o.max_warps, 32);
        assert_eq!(o.active_warps, 32);
    }

    #[test]
    fn small_blocks_hit_the_block_limit() {
        // 32-thread blocks on Ampere: 16 blocks max -> only 16 warps.
        let o = occupancy(&metrics("sm_86", 16, 0, 32));
        assert_eq!(o.blocks_per_sm, 16);
        assert_eq!(o.limiter, Limiter::BlocksPerSm);
    }

    #[test]
    fn arch_suffix_is_tolerated() {
        assert_eq!(Arch::lookup("sm_90a").unwrap().name, "sm_90");
        assert!(Arch::lookup("sm_999").is_none());
    }
}
