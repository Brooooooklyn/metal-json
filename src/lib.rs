//! # metal-json
//!
//! GPU JSON parser on Apple Metal: parses standard JSON documents to a
//! simdjson-equivalent typed tape on Apple Silicon GPUs, exploiting unified
//! memory (`bytesNoCopy`, zero copy) to beat CPU SIMD parsers on large
//! (MBs–GB) inputs.
//!
//! ## Status: M1 (work in progress)
//!
//! What exists today: the build pipeline (AOT `.metal` → metallib in
//! build.rs, or runtime MSL compilation behind the `runtime-shaders`
//! feature), the safe wrapper layer over `objc2-metal` in [`metal`], the
//! [`Error`] type, GPU smoke tests, and the **tape format v1** foundation in
//! [`tape`] (constants + encode/decode helpers + buffers, locked to
//! `shaders/tape_types.h` by a layout test; spec in `docs/tape-format.md`).
//! [`Parser`], [`Document`], [`Value`] and the `cpu-reference` oracle are
//! compiling M1 stubs being filled in now; the GPU pipeline lands over
//! M2–M5 — see `docs/superpowers/specs/2026-06-10-metal-json-design.md` for
//! the full design.
//!
//! ## Feature flags
//!
//! - `runtime-shaders` — compile MSL at runtime instead of embedding an
//!   AOT-built metallib; honors `METAL_JSON_SHADER_DIR` for hot reload.
//!   Also the fallback for machines without the Xcode Metal toolchain.
//! - `cpu-reference` — scalar CPU oracle backend (M1).
//! - `timing` — per-kernel GPU timing via `MTLCounterSampleBuffer` (M5).

mod error;

pub mod document;
pub mod metal;
pub mod parser;
pub mod tape;
pub mod value;

#[cfg(feature = "cpu-reference")]
pub mod reference;

pub use document::Document;
pub use error::{Error, Result, SyntaxErrorKind};
pub use parser::{Backend, Parser, ParserOptions};
pub use value::{Value, ValueKind};
