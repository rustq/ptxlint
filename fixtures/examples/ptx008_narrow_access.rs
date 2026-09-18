//! PTX008: every global access moves four bytes per thread. A warp then needs
//! one memory transaction per instruction; loading four elements per thread
//! (`.v4`) would move the same data in a quarter of the instructions.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn narrow_access(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    n: u32,
) {
    let i = tid();
    if i < n as usize {
        // Four separate scalar accesses; nothing here is vectorised.
        let x = unsafe { *a.add(i) };
        let y = unsafe { *b.add(i) };
        let z = unsafe { *c.add(i) };
        unsafe { *out.add(i) = x + y + z };
    }
}
