//! PTX005: many values live at once. Every one of these loads must survive
//! until the final reduction, so the compiler cannot reuse registers between
//! them. Unlike PTX001 nothing spills to local memory here — every index is a
//! literal — and the cost is paid in occupancy instead, because fewer warps
//! fit on an SM.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

/// 32 independent strided loads.
macro_rules! load_all {
    ($x:expr, $i:expr, $($k:literal),+) => {
        [$( unsafe { *$x.add($i + $k * 1024) } ),+]
    };
}

/// Pair the first load with the last one, and so on, so all 32 stay live.
macro_rules! dot_pairs {
    ($v:ident, $(($a:literal, $b:literal)),+) => {
        0.0f32 $( + $v[$a] * $v[$b] )+
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn register_pressure(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let v = load_all!(
        x, i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31
    );
    let acc = dot_pairs!(
        v,
        (0, 31), (1, 30), (2, 29), (3, 28), (4, 27), (5, 26), (6, 25), (7, 24),
        (8, 23), (9, 22), (10, 21), (11, 20), (12, 19), (13, 18), (14, 17), (15, 16),
        (16, 15), (17, 14), (18, 13), (19, 12), (20, 11), (21, 10), (22, 9), (23, 8),
        (24, 7), (25, 6), (26, 5), (27, 4), (28, 3), (29, 2), (30, 1), (31, 0)
    );
    unsafe { *y.add(i) = acc };
}
