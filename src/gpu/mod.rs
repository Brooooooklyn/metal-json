//! GPU pipeline orchestration: composes the kernels in `shaders/` into the
//! command buffers of the parse pipeline.
//!
//! M2 scope: [`stage1`] — CB1 (classify → escape valve → spine → token mask
//! → spine) plus the K5 token scatter, producing bitmaps and the token
//! stream that the structural stages (M3) consume.
//!
//! Internal/unstable: exposed publicly so integration tests can drive the
//! pipeline directly (like [`crate::metal`] and [`crate::stage`]), but not
//! part of the supported API surface.

pub mod stage1;

pub use stage1::{ERR_STRING, ERR_UTF8, Stage1, Stage1Output, run_stage1};
