//! PTX004: integer division and remainder by a runtime value. A GPU has no
//! integer divider, so ptxas expands each of these into ~20 SASS
//! instructions. A power-of-two divisor would become a shift and a mask, and
//! no `div`/`rem` instruction would appear at all.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn integer_division(x: *const u32, y: *mut u32, n: u32, m: u32) {
    let i = tid();
    if i < n as usize {
        // Forcing the divisor odd removes Rust's divide-by-zero check, which
        // would otherwise add a panic call and trip PTX010 as well.
        let m = m | 1;
        let v = unsafe { *x.add(i) };
        unsafe { *y.add(i) = v / m + v % m };
    }
}
