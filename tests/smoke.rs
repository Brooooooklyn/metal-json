//! M0 smoke tests: prove the full path works on real hardware —
//! build.rs AOT metallib (or runtime MSL compile) → MetalContext → Pipeline →
//! GpuBuffer → dispatch → exact CPU readback. Run with
//! `MTL_SHADER_VALIDATION=1` in CI.

use metal_json::metal::{Binding, Dispatch, GpuBuffer, MetalContext, MjParams, Pipeline};

/// Deliberately not a multiple of THREADGROUP_SIZE (256) to exercise the
/// per-thread `element_count` bound check and non-uniform threadgroups.
const N: usize = 4096 + 37;

/// GPU gating: in environments without a Metal device (containers, sandboxed
/// CI), skip with a loud message instead of failing — unless
/// `METAL_JSON_REQUIRE_GPU=1` (set in our CI) makes a missing device a hard
/// error, so the only GPU evidence cannot silently disappear.
fn ctx_or_skip(test: &str) -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
            }
            eprintln!("SKIP {test}: no usable Metal device here ({err})");
            None
        }
    }
}

#[test]
fn smoke_add_exact() {
    let Some(ctx) = ctx_or_skip("smoke_add_exact") else {
        return;
    };
    let pipeline = Pipeline::new(&ctx, "smoke_add").expect("pipeline smoke_add");

    let a_host: Vec<u32> = (0..N as u32).collect();
    let b_host: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2654435761)).collect();

    let mut a = GpuBuffer::alloc(&ctx, N * size_of::<u32>()).unwrap();
    let mut b = GpuBuffer::alloc(&ctx, N * size_of::<u32>()).unwrap();
    let mut out = GpuBuffer::alloc(&ctx, N * size_of::<u32>()).unwrap();
    a.write_from(&a_host);
    b.write_from(&b_host);

    let params = MjParams {
        input_len: (N * size_of::<u32>()) as u64,
        element_count: N as u64,
        ..Default::default()
    };
    ctx.dispatch(
        &pipeline,
        &[Binding::Read(&a), Binding::Read(&b), Binding::ReadWrite(&mut out)],
        Some(&params),
        Dispatch::Threads(N),
    )
    .expect("dispatch smoke_add");

    let got = out.as_slice::<u32>();
    for i in 0..N {
        assert_eq!(
            got[i],
            a_host[i].wrapping_add(b_host[i]),
            "smoke_add mismatch at index {i}"
        );
    }
}

#[test]
fn smoke_popcount64_exact() {
    let Some(ctx) = ctx_or_skip("smoke_popcount64_exact") else {
        return;
    };
    let pipeline = Pipeline::new(&ctx, "smoke_popcount64").expect("pipeline smoke_popcount64");

    // Mix edge cases with a deterministic pseudo-random pattern.
    let in_host: Vec<u64> = (0..N as u64)
        .map(|i| match i {
            0 => 0,
            1 => u64::MAX,
            2 => 1,
            3 => 1 << 63,
            4 => 0xAAAA_AAAA_AAAA_AAAA,
            _ => i.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left((i % 64) as u32),
        })
        .collect();

    let mut input = GpuBuffer::alloc(&ctx, N * size_of::<u64>()).unwrap();
    let mut out = GpuBuffer::alloc(&ctx, N * size_of::<u32>()).unwrap();
    input.write_from(&in_host);

    let params = MjParams {
        input_len: (N * size_of::<u64>()) as u64,
        element_count: N as u64,
        ..Default::default()
    };
    ctx.dispatch(
        &pipeline,
        &[Binding::Read(&input), Binding::ReadWrite(&mut out)],
        Some(&params),
        Dispatch::Threads(N),
    )
    .expect("dispatch smoke_popcount64");

    let got = out.as_slice::<u32>();
    for i in 0..N {
        assert_eq!(
            got[i],
            in_host[i].count_ones(),
            "smoke_popcount64 mismatch at index {i} (input {:#x})",
            in_host[i]
        );
    }
}
