//! PTX001: an array reached through a data-dependent index cannot live in
//! registers, so the compiler puts it in local memory — which is DRAM.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn local_array(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let mut scratch = [0.0f32; 64];
    for (k, s) in scratch.iter_mut().enumerate() {
        *s = unsafe { *x.add(i + k) };
    }
    // The index depends on the data, which defeats register promotion.
    let j = (unsafe { *x.add(i) } as usize) & 63;
    let mut acc = 0.0f32;
    for k in 0..8 {
        acc += scratch[(j + k * 7) & 63];
    }
    unsafe { *y.add(i) = acc };
}
