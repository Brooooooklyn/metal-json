//! Compiles the vendored simdjson amalgamation + the extern "C" shim.
//!
//! Always -O3 / C++17 (even under `cargo test`, so the baseline is never an
//! unoptimized strawman). On aarch64 we additionally ask for the native CPU
//! so simdjson's ARM64 (NEON) kernel gets the best scheduling the local
//! compiler can do. The `parallel` feature of `cc` compiles the two
//! translation units concurrently.

fn main() {
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/vendor/simdjson.cpp");
    println!("cargo:rerun-if-changed=cpp/vendor/simdjson.h");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .define("NDEBUG", None)
        // NOTE: cpp/vendor is deliberately NOT on the include path — the
        // vendored VERSION file would shadow the C++ <version> header on
        // case-insensitive filesystems. Quoted includes ("simdjson.h",
        // "vendor/simdjson.h") resolve relative to the including file.
        .file("cpp/vendor/simdjson.cpp")
        .file("cpp/shim.cpp");

    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
        build.flag_if_supported("-mcpu=native");
    }

    build.compile("simdjson_shim");
}
