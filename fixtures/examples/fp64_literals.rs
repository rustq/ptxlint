//! PTX003: the classic Rust footgun — float literals default to f64, so the
//! whole expression is promoted and runs on the FP64 units (1/64 rate on
//! GeForce parts).
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn fp64_literals(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i < n as usize {
        let v = unsafe { *x.add(i) } as f64;
        // `0.5` and `1.0` are f64; nothing here asked for double precision.
        let r = v * 0.5 + 1.0 / (1.0 + v * v);
        unsafe { *y.add(i) = r as f32 };
    }
}
