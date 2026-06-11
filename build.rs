//! AOT shader pipeline: compile every `shaders/*.metal` to `.air` with
//! `xcrun -sdk macosx metal`, then link them into `$OUT_DIR/metal_json.metallib`
//! which `src/metal/context.rs` embeds via `include_bytes!`.
//!
//! Skipped entirely when the `runtime-shaders` feature is enabled (runtime MSL
//! compilation path, no Metal toolchain required).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let shader_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("shaders");

    // Re-run when any shader source or header changes (also covers the
    // runtime-shaders path, which embeds sources with include_str!).
    println!("cargo::rerun-if-changed={}", shader_dir.display());
    let mut metal_sources: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&shader_dir).expect("shaders/ directory must exist") {
        let path = entry.expect("readable shaders/ entry").path();
        match path.extension().and_then(|e| e.to_str()) {
            Some("metal") => {
                println!("cargo::rerun-if-changed={}", path.display());
                metal_sources.push(path);
            }
            Some("h") => {
                println!("cargo::rerun-if-changed={}", path.display());
            }
            _ => {}
        }
    }
    metal_sources.sort();

    if env::var_os("CARGO_FEATURE_RUNTIME_SHADERS").is_some() {
        // Runtime MSL compilation: no AOT step needed.
        return;
    }

    // Probe the Metal toolchain.
    let probe = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output();
    let toolchain_ok = matches!(probe, Ok(ref out) if out.status.success());
    if !toolchain_ok {
        panic!(
            "metal-json: the Metal shader toolchain was not found \
             (`xcrun -sdk macosx metal --version` failed).\n\
             Fix one of:\n\
             1. Install full Xcode, then run: xcodebuild -downloadComponent MetalToolchain\n\
             2. Build with runtime shader compilation instead: \
             cargo build --features runtime-shaders"
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let debug_profile = env::var("PROFILE").as_deref() == Ok("debug");

    // `-std=metal3.1` verified working with Apple metal 32023.883 on this
    // machine. If a different toolchain rejects it, fall back to metal3.0,
    // then to the compiler default.
    let std_flag = pick_std_flag(&out_dir);

    let mut air_files: Vec<PathBuf> = Vec::new();
    for src in &metal_sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let air = out_dir.join(format!("{stem}.air"));
        let mut cmd = Command::new("xcrun");
        cmd.args(["-sdk", "macosx", "metal"]);
        if let Some(flag) = std_flag {
            cmd.arg(flag);
        }
        if debug_profile {
            // Keep sources + line tables in the AIR for Xcode GPU capture.
            cmd.args(["-frecord-sources", "-gline-tables-only"]);
        }
        cmd.arg("-c").arg(src).arg("-o").arg(&air);
        run(cmd, &format!("compile {}", src.display()));
        air_files.push(air);
    }

    if air_files.is_empty() {
        panic!("metal-json: no .metal sources found in {}", shader_dir.display());
    }

    let metallib = out_dir.join("metal_json.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    cmd.args(&air_files);
    cmd.arg("-o").arg(&metallib);
    run(cmd, "link metal_json.metallib");
}

/// Choose the highest MSL -std flag the toolchain accepts.
/// Order: metal3.1 → metal3.0 → none (compiler default).
fn pick_std_flag(out_dir: &Path) -> Option<&'static str> {
    let probe_src = out_dir.join("mj_std_probe.metal");
    let probe_air = out_dir.join("mj_std_probe.air");
    std::fs::write(&probe_src, "kernel void mj_std_probe() {}\n").unwrap();
    for flag in ["-std=metal3.1", "-std=metal3.0"] {
        let ok = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", flag, "-c"])
            .arg(&probe_src)
            .arg("-o")
            .arg(&probe_air)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(flag);
        }
    }
    None
}

fn run(mut cmd: Command, what: &str) {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("metal-json build.rs: failed to spawn `{cmd:?}` ({what}): {e}"));
    if !out.status.success() {
        panic!(
            "metal-json build.rs: {what} failed ({}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
}
