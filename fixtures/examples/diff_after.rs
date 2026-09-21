//! The "after" half of the --baseline demo: the same `mix16` kernel with every
//! index a literal, so the state stays in registers and the local memory is
//! gone. Diffing this against `diff_before.ptx` is what the fix looked like.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

macro_rules! rounds {
    ($state:ident, $acc:ident, $($k:literal),+) => {$(
        $acc = $acc.wrapping_add($state[$k]).rotate_left(3);
    )+};
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mix16(x: *const u32, y: *mut u32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let state = [
        unsafe { *x.add(i) },
        unsafe { *x.add(i + 1) },
        unsafe { *x.add(i + 2) },
        unsafe { *x.add(i + 3) },
        unsafe { *x.add(i + 4) },
        unsafe { *x.add(i + 5) },
        unsafe { *x.add(i + 6) },
        unsafe { *x.add(i + 7) },
        unsafe { *x.add(i + 8) },
        unsafe { *x.add(i + 9) },
        unsafe { *x.add(i + 10) },
        unsafe { *x.add(i + 11) },
        unsafe { *x.add(i + 12) },
        unsafe { *x.add(i + 13) },
        unsafe { *x.add(i + 14) },
        unsafe { *x.add(i + 15) },
    ];
    let mut acc = 0u32;
    rounds!(state, acc, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    unsafe { *y.add(i) = acc };
}
