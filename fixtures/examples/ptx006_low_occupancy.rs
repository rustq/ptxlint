//! PTX006: so many registers per thread that only a fraction of the SM's warp
//! slots can be filled.
//!
//! Occupancy is only worth reporting when the register count is exact, which
//! means it needs ptxas. Like the PTX002 case this one ships with a recorded
//! `ptxas -v` report, replayed via `--ptxas-report`.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

macro_rules! load_all {
    ($x:expr, $i:expr, $($k:literal),+) => {
        [$( unsafe { *$x.add($i + $k * 1024) } ),+]
    };
}

macro_rules! dot_pairs {
    ($v:ident, $(($a:literal, $b:literal)),+) => {
        0.0f32 $( + $v[$a] * $v[$b] )+
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn low_occupancy(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let v = load_all!(
        x, i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    );
    let acc = dot_pairs!(
        v,
        (0, 23), (1, 22), (2, 21), (3, 20), (4, 19), (5, 18), (6, 17), (7, 16),
        (8, 15), (9, 14), (10, 13), (11, 12), (12, 11), (13, 10), (14, 9), (15, 8),
        (16, 7), (17, 6), (18, 5), (19, 4), (20, 3), (21, 2), (22, 1), (23, 0)
    );
    unsafe { *y.add(i) = acc };
}
