//! The "before" half of the --baseline demo: a kernel that indexes its scratch
//! array with a runtime value, so the array lives in local memory. Its twin in
//! `diff_after.rs` exports a kernel of the same name that does the same work
//! with literal indices.
//!
//! This is the shape of the real bug ptxlint found in its own sibling project.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mix16(x: *const u32, y: *mut u32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let mut state = [0u32; 16];
    for (k, s) in state.iter_mut().enumerate() {
        *s = unsafe { *x.add(i + k) };
    }
    // A runtime index keeps the array out of registers.
    let mut acc = 0u32;
    for k in 0..16 {
        let j = (acc as usize).wrapping_add(k) & 15;
        acc = acc.wrapping_add(state[j]).rotate_left(3);
    }
    unsafe { *y.add(i) = acc };
}
