//! Pipeline-state probe: build a PSO for every kernel in the shader library
//! and report each result. Run on a machine (or CI runner) where pipeline
//! creation misbehaves to see exactly which kernels the device's backend
//! compiler rejects and why:
//!
//! ```sh
//! cargo run --example pso_probe
//! MTL_SHADER_VALIDATION=1 cargo run --example pso_probe
//! ```
//!
//! Exits 0 even on failures — it is a diagnostic, not a gate.

use metal_json::metal::{MetalContext, Pipeline};

fn main() {
    println!(
        "MTL_SHADER_VALIDATION={}",
        std::env::var("MTL_SHADER_VALIDATION").unwrap_or_else(|_| "<unset>".into())
    );

    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(err) => {
            println!("FAIL MetalContext::new: {err}");
            return;
        }
    };
    println!("device: {}", ctx.device_name());

    let names = ctx.kernel_names();
    println!("library has {} kernels", names.len());
    let mut failures = 0usize;
    for name in &names {
        match Pipeline::new(&ctx, name) {
            Ok(p) => println!(
                "  OK   {name} (max_threads_per_tg {})",
                p.max_total_threads_per_threadgroup()
            ),
            Err(err) => {
                failures += 1;
                println!("  FAIL {name}: {err}");
            }
        }
    }
    println!("{} of {} kernels failed PSO creation", failures, names.len());
}
