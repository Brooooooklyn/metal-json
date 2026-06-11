//! K10 on the GPU: number + literal parsing (`parse_numbers`,
//! `shaders/12_numbers.metal`) — the scalar half of M4, filling the tape's
//! number/literal holes left by the M3 structure stage.
//!
//! # What K10 does
//!
//! One thread per scalar-list entry (the K6b `scalar_tokens` work list):
//!
//! - literals `true` / `false` / `null` — K6 Layer-1 already byte-validated
//!   them, so the thread only writes the 1-word tape entry (never
//!   re-checks, never duplicates an error);
//! - numbers — full reference-stage-5 treatment
//!   (`reference::stage5_scalars`, the bit-exact spec): grammar validation
//!   over the whole scalar run, the integer fast path (`l` / `u` two-word
//!   entries with simdjson's type selection), and the Eisel-Lemire double
//!   path producing the correctly rounded f64 **bit pattern** as a `u64`
//!   over the 128-bit pow5 table in constant memory (`shaders/pow5_table.h`
//!   — generated and entry-by-entry verified by the `pow5` test module).
//!   Apple GPUs have no fp64: no `double` exists anywhere in the kernel.
//!
//! # The fixup contract (hard cases)
//!
//! Eisel-Lemire cannot certify a handful of roundings: the truncated-
//! product off-by-one ambiguity, truncated ≥ 20-significant-digit mantissas
//! whose `w` and `w+1` parses disagree, and roundings that land past the
//! largest finite exponent (would-be infinities the CPU must confirm).
//! Those threads append their TOKEN INDEX to a fixup list via a device
//! atomic counter — order does not matter, the runner sorts — and write a
//! `d` marker with a `0` placeholder value word. After the command buffer
//! completes, [`patch_number_fixups`] re-parses those few scalars on the
//! CPU with the reference oracle's double path (`str::parse::<f64>`,
//! correctly rounded; infinities rejected as `InvalidNumber` — simdjson
//! parity) and patches the 8-byte value words in the shared tape buffer
//! before any `Document` is constructed.
//!
//! # Error contract
//!
//! Every K10 error is the single class [`ERR_NUMBER`] (`MJ_ERR_NUMBER`,
//! reference `SyntaxErrorKind::InvalidNumber`) at the failing scalar's
//! token byte offset, so the kernel min-reduces just the offset with a
//! 32-bit `atomic_min` (the K1 UTF-8 protocol — 64-bit atomics are an
//! Apple9+ feature the embedded metallib cannot assume) and the runner
//! packs `(offset << 32) | ERR_NUMBER` on the CPU. Fixup re-parses that
//! reject contribute their own packed candidates; the merged verdict is the
//! packed minimum, which equals the reference's first-in-token-order
//! failing scalar (token positions strictly increase). Per the rejection
//! contract a number error short-circuits Document construction: the
//! [`NumbersOutput::tape`] of a rejected run is empty — the tape is never
//! observed.
//!
//! The string holes (`"` tokens) remain zero words here — they are K11's
//! (`gpu::strings`). This module is the standalone K10 runner (the M4
//! deliverable + its torture tests); the production composition that runs
//! K10 and K11 inside the structure CB3 is
//! [`crate::gpu::pipeline::GpuPipeline`], which shares
//! [`patch_number_fixups`] and the error protocol defined here.

use crate::error::Result;
use crate::metal::{Dispatch, GpuBuffer, MetalContext, MjParams};
use crate::stage::{Stage, WORD_BYTES};

use super::stage3::{Stage3, Stage3Output};

/// `MjErrorCode` for an invalid number (mirrors `MJ_ERR_NUMBER` in
/// `shaders/common.h` — keep in sync; a const test pins the value).
/// No same-offset tie-break is needed: K10 is the only stage that reports
/// at a number token's offset (rejection contract — earlier CBs short-
/// circuit), and every K10 error carries this one code.
pub const ERR_NUMBER: u32 = 5;

/// "No number error" sentinel the runner arms the 32-bit offset cell with
/// (any real token offset is smaller — input length is capped below
/// `u32::MAX`). Shared with the full pipeline runner (`crate::gpu::pipeline`).
pub(crate) const NO_NUMBER_ERROR: u32 = u32::MAX;

/// Pack a K10 verdict like the GPU error word: `(offset << 32) | code`.
pub(crate) const fn pack_number_error(offset: u64) -> u64 {
    (offset << 32) | ERR_NUMBER as u64
}

/// The K10 kernel plus the composed [`Stage3`] pipeline (which composes
/// stages 1–2), with lazily-built cached pipelines. Create once and reuse
/// across parses.
#[derive(Debug)]
pub struct Numbers {
    stage3: Stage3,
    parse_numbers: Stage,
}

/// Everything the K10 runner produces, mirroring [`Stage3Output`]'s
/// conventions.
///
/// # Rejection contract
///
/// When [`error`](Self::error) is `Some`, the pipeline rejected the input
/// and outputs after the failing stage are never produced:
///
/// - a stage-1/2/3 rejection carries [`structure`](Self::structure) with
///   its own rejection contract applied; K10 never ran, so
///   [`tape`](Self::tape) and [`fixup_tokens`](Self::fixup_tokens) are
///   empty;
/// - a K10 number error (grammar, overflow-to-infinity — whether detected
///   on the GPU or by a fixup re-parse) keeps the accepted structure
///   outputs but leaves [`tape`](Self::tape) empty: the tape is never
///   observed on a rejected input. [`fixup_tokens`](Self::fixup_tokens) is
///   still reported (diagnostic: which scalars took the slow path).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NumbersOutput {
    /// The stage-3 view of the same run (stage-1/2 views nested inside).
    pub structure: Stage3Output,
    /// The tape with every number/literal hole filled (and fixup value
    /// words CPU-patched): container/root words from M3, `l`/`u`/`d`
    /// two-word entries, `t`/`f`/`n` one-word entries. String holes stay
    /// zero words (K11's). Empty on rejected inputs.
    pub tape: Vec<u64>,
    /// Token indices that took the hard-case fixup path, sorted ascending
    /// (the GPU appends in nondeterministic order; the runner sorts).
    pub fixup_tokens: Vec<u32>,
    /// First error, packed `(byte_offset << 32) | code`, or `None`. Codes:
    /// everything stage 3 can report, plus [`ERR_NUMBER`].
    pub error: Option<u64>,
}

impl NumbersOutput {
    /// Decode [`error`](Self::error) as `(byte_offset, code)`.
    #[must_use]
    pub fn error_offset_code(&self) -> Option<(u64, u32)> {
        self.error.map(|e| (e >> 32, e as u32))
    }
}

impl Numbers {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stage3: Stage3::new(),
            parse_numbers: Stage::new("parse_numbers"),
        }
    }

    /// Run the pipeline through the M3 structure stage, then drive K10 over
    /// the scalar list and apply the CPU fixup patches: the full
    /// numbers-side M4 flow over `input` (default depth limit).
    ///
    /// The K10 inputs are re-uploaded from the stage-3 output snapshot
    /// (TestHarness-style buffers); wiring K10 into the live pipeline
    /// buffers is the parser integration's job (next phase).
    ///
    /// # Errors
    ///
    /// GPU plumbing failures only; input *content* problems are **data**,
    /// reported in [`NumbersOutput::error`].
    pub fn run(&self, ctx: &MetalContext, input: &[u8]) -> Result<NumbersOutput> {
        let structure = self.stage3.run(ctx, input)?;
        if structure.error.is_some() {
            // Stage 1/2/3 rejected: K10 never runs.
            let error = structure.error;
            return Ok(NumbersOutput {
                structure,
                error,
                ..NumbersOutput::default()
            });
        }
        let scalar_tokens = structure.stage2.scalar_tokens.clone();
        if scalar_tokens.is_empty() {
            // Nothing to parse: the M3 tape is already the M4 tape (string
            // holes are K11's).
            let tape = structure.tape.clone();
            return Ok(NumbersOutput {
                structure,
                tape,
                ..NumbersOutput::default()
            });
        }

        // --- K10 buffer set, re-uploaded from the stage-3 snapshot ---------
        // The input copy is space-padded to whole 64-byte words like
        // Stage1Buffers (defense in depth; K10 never reads >= input_len).
        let mut padded = input.to_vec();
        padded.resize(input.len().div_ceil(WORD_BYTES).max(1) * WORD_BYTES, b' ');
        let input_buf = upload(ctx, &padded)?;
        let scalars_buf = upload(ctx, &scalar_tokens)?;
        let tok_pos = &structure.stage2.stage1.tok_pos;
        let tok_pos_buf = upload(ctx, tok_pos)?;
        let tape_ofs = &structure.stage2.tape_ofs;
        let tape_ofs_buf = upload(ctx, tape_ofs)?;
        let mut tape_buf = upload(ctx, &structure.tape)?;
        // Accumulation targets get their preconditions established
        // explicitly (GpuBuffer::alloc makes no contents guarantee).
        let mut err_buf = GpuBuffer::alloc(ctx, size_of::<u32>())?;
        err_buf.as_mut_slice::<u32>()[0] = NO_NUMBER_ERROR;
        let mut fixup_count_buf = GpuBuffer::alloc(ctx, size_of::<u32>())?;
        fixup_count_buf.as_mut_slice::<u32>()[0] = 0;
        // Worst case: every scalar takes the fixup path.
        let mut fixup_buf = GpuBuffer::alloc(ctx, scalar_tokens.len() * size_of::<u32>())?;

        let params = MjParams {
            input_len: input.len() as u64,
            element_count: scalar_tokens.len() as u64,
            reserved0: 0,
            reserved1: 0,
        };
        {
            let mut batch = ctx.batch()?;
            let h_input = batch.bind_read(&input_buf);
            let h_scalars = batch.bind_read(&scalars_buf);
            let h_tok_pos = batch.bind_read(&tok_pos_buf);
            let h_tape_ofs = batch.bind_read(&tape_ofs_buf);
            let h_tape = batch.bind_write(&mut tape_buf);
            let h_err = batch.bind_write(&mut err_buf);
            let h_count = batch.bind_write(&mut fixup_count_buf);
            let h_fixups = batch.bind_write(&mut fixup_buf);
            self.parse_numbers.encode(
                &mut batch,
                &[
                    h_input, h_scalars, h_tok_pos, h_tape_ofs, h_tape, h_err, h_count, h_fixups,
                ],
                Some(&params),
                Dispatch::Threads(scalar_tokens.len()),
            )?;
            batch.commit_and_wait()?;
        }

        // --- CPU sync: number verdict + fixup patch -------------------------
        let gpu_err_pos = err_buf.as_slice::<u32>()[0];
        let fixup_total = (fixup_count_buf.as_slice::<u32>()[0] as usize).min(scalar_tokens.len()); // kernel appends at most once per thread
        let mut fixup_tokens = fixup_buf.as_slice::<u32>()[..fixup_total].to_vec();
        fixup_tokens.sort_unstable();

        // Patch the 8-byte value words in the SHARED tape buffer in place
        // (this is the production flow: the buffer is CPU-visible). Done
        // even when the GPU already found an error: a fixup re-parse can
        // reject at an EARLIER offset, and the reference reports the first
        // failing scalar in token order — the packed minimum below.
        let patch_error = patch_number_fixups(
            input,
            tok_pos,
            tape_ofs,
            &fixup_tokens,
            tape_buf.as_mut_slice::<u64>(),
        );
        let gpu_error =
            (gpu_err_pos != NO_NUMBER_ERROR).then(|| pack_number_error(u64::from(gpu_err_pos)));
        let error = match (gpu_error, patch_error) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };

        if error.is_some() {
            // Rejection contract: the tape is never observed.
            return Ok(NumbersOutput {
                structure,
                tape: Vec::new(),
                fixup_tokens,
                error,
            });
        }
        Ok(NumbersOutput {
            tape: tape_buf.as_slice::<u64>().to_vec(),
            structure,
            fixup_tokens,
            error: None,
        })
    }
}

impl Default for Numbers {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot convenience over [`Numbers::run`] (builds the pipelines each
/// call; tests that run many inputs should hold a [`Numbers`] instead).
///
/// # Errors
///
/// As [`Numbers::run`].
pub fn run_numbers(ctx: &MetalContext, input: &[u8]) -> Result<NumbersOutput> {
    Numbers::new().run(ctx, input)
}

/// The CPU half of the K10 fixup contract: re-parse every fixup token's
/// number with the reference oracle's double path and patch the 8-byte
/// value word at `tape[tape_ofs[token] + 1]` (the marker word was already
/// written by the kernel). The K10 grammar walk has already validated these
/// runs, and only double-path numbers can take the fixup path, so the
/// re-parse is exactly the reference's `str::parse::<f64>` (correctly
/// rounded) with overflow-to-infinity rejected as `InvalidNumber`
/// (simdjson parity; `src/reference/scalars.rs` defers to the same oracle).
///
/// Returns the earliest packed `(offset << 32) | ERR_NUMBER` among fixups
/// that reject, or `None` when every fixup patched cleanly. Patch order
/// does not matter (disjoint value words), so callers may pass the list in
/// any order.
///
/// # Panics
///
/// If a token index is out of range of `tok_pos` / `tape_ofs`, a patched
/// slot is out of range of `tape`, or the bytes at a token position are
/// not a grammar-valid JSON number — all internal-contract violations
/// (the fixup list is produced by K10 after grammar validation).
pub fn patch_number_fixups(
    input: &[u8],
    tok_pos: &[u32],
    tape_ofs: &[u32],
    fixup_tokens: &[u32],
    tape: &mut [u64],
) -> Option<u64> {
    let mut first_error: Option<u64> = None;
    for &tok in fixup_tokens {
        let pos = tok_pos[tok as usize] as usize;
        let end = scalar_run_end(input, pos);
        let text = std::str::from_utf8(&input[pos..end])
            .expect("K10 fixup runs passed the number grammar and are ASCII");
        let value: f64 = text
            .parse()
            .expect("K10 fixup runs passed the number grammar");
        if value.is_infinite() {
            // simdjson rejects values that overflow to infinity.
            let packed = pack_number_error(pos as u64);
            first_error = Some(first_error.map_or(packed, |e| e.min(packed)));
            continue;
        }
        let ofs = tape_ofs[tok as usize] as usize;
        tape[ofs + 1] = value.to_bits();
    }
    first_error
}

/// End (exclusive) of the scalar run starting at `pos`: the next
/// whitespace, structural operator, `"`, or end of input — exactly
/// `reference::scalars::scalar_run`'s stop set.
fn scalar_run_end(input: &[u8], pos: usize) -> usize {
    input[pos..]
        .iter()
        .position(|&b| {
            matches!(
                b,
                b' ' | b'\t' | b'\n' | b'\r' | b'{' | b'}' | b'[' | b']' | b':' | b',' | b'"'
            )
        })
        .map_or(input.len(), |n| pos + n)
}

/// Allocate a GPU buffer holding exactly `data`.
fn upload<T: crate::metal::Pod>(ctx: &MetalContext, data: &[T]) -> Result<GpuBuffer> {
    let mut buffer = GpuBuffer::alloc(ctx, size_of_val(data))?;
    buffer.write_from(data);
    Ok(buffer)
}

// --- pow5 table generator + big-int oracle (test-only) ------------------------

/// Deterministic generator for `shaders/pow5_table.h` and the exact big-int
/// arithmetic the verification tests recompute every entry with. The
/// checked-in header is regenerated via the `regenerate_pow5_table_header`
/// ignored test; `pow5_table_header_is_generated_and_bigint_exact` fails on
/// any drift. The entries are bit-identical to simdjson's
/// `internal::power_of_five_128` / Rust core's `POWER_OF_FIVE_128`
/// (`library/core/src/num/dec2flt/table.rs`) — known values are pinned from
/// the latter in [`tests`].
#[cfg(test)]
pub(crate) mod pow5 {
    /// Minimal arbitrary-precision unsigned integer: little-endian u64
    /// limbs, no trailing zero limbs. Just enough exact arithmetic for the
    /// pow5 entries and the halfway-decimal test fixtures.
    #[derive(Clone, PartialEq, Eq)]
    pub struct BigUint(Vec<u64>);

    impl BigUint {
        pub fn from_u64(v: u64) -> Self {
            Self(if v == 0 { vec![] } else { vec![v] })
        }

        pub fn is_zero(&self) -> bool {
            self.0.is_empty()
        }

        fn normalize(&mut self) {
            while self.0.last() == Some(&0) {
                self.0.pop();
            }
        }

        pub fn bit_length(&self) -> usize {
            match self.0.last() {
                None => 0,
                Some(top) => 64 * (self.0.len() - 1) + (64 - top.leading_zeros() as usize),
            }
        }

        fn set_bit(&mut self, i: usize) {
            let (limb, off) = (i / 64, i % 64);
            if self.0.len() <= limb {
                self.0.resize(limb + 1, 0);
            }
            self.0[limb] |= 1 << off;
        }

        pub fn shl_bits(&self, n: usize) -> Self {
            if self.is_zero() {
                return Self(vec![]);
            }
            let (limbs, bits) = (n / 64, n % 64);
            let mut out = vec![0u64; limbs];
            if bits == 0 {
                out.extend_from_slice(&self.0);
            } else {
                let mut carry = 0u64;
                for &l in &self.0 {
                    out.push((l << bits) | carry);
                    carry = l >> (64 - bits);
                }
                if carry != 0 {
                    out.push(carry);
                }
            }
            let mut r = Self(out);
            r.normalize();
            r
        }

        pub fn shr_bits(&self, n: usize) -> Self {
            let (limbs, bits) = (n / 64, n % 64);
            if limbs >= self.0.len() {
                return Self(vec![]);
            }
            let mut out = Vec::with_capacity(self.0.len() - limbs);
            if bits == 0 {
                out.extend_from_slice(&self.0[limbs..]);
            } else {
                for i in limbs..self.0.len() {
                    let lo = self.0[i] >> bits;
                    let hi = if i + 1 < self.0.len() {
                        self.0[i + 1] << (64 - bits)
                    } else {
                        0
                    };
                    out.push(lo | hi);
                }
            }
            let mut r = Self(out);
            r.normalize();
            r
        }

        pub fn mul_small(&self, m: u64) -> Self {
            let mut out = Vec::with_capacity(self.0.len() + 1);
            let mut carry = 0u128;
            for &l in &self.0 {
                let v = u128::from(l) * u128::from(m) + carry;
                out.push(v as u64);
                carry = v >> 64;
            }
            if carry != 0 {
                out.push(carry as u64);
            }
            let mut r = Self(out);
            r.normalize();
            r
        }

        pub fn add_small(&self, a: u64) -> Self {
            let mut out = self.0.clone();
            let mut carry = a;
            for l in &mut out {
                let (v, c) = l.overflowing_add(carry);
                *l = v;
                carry = u64::from(c);
                if carry == 0 {
                    break;
                }
            }
            if carry != 0 {
                out.push(carry);
            }
            Self(out)
        }

        fn ge(&self, other: &Self) -> bool {
            if self.0.len() != other.0.len() {
                return self.0.len() > other.0.len();
            }
            for i in (0..self.0.len()).rev() {
                if self.0[i] != other.0[i] {
                    return self.0[i] > other.0[i];
                }
            }
            true
        }

        fn sub_assign(&mut self, other: &Self) {
            let mut borrow = 0u64;
            for i in 0..self.0.len() {
                let rhs = other.0.get(i).copied().unwrap_or(0);
                let (v, b1) = self.0[i].overflowing_sub(rhs);
                let (v, b2) = v.overflowing_sub(borrow);
                self.0[i] = v;
                borrow = u64::from(b1) + u64::from(b2);
            }
            assert_eq!(borrow, 0, "BigUint subtraction underflow");
            self.normalize();
        }

        /// `floor(2^z / self)`; `self` must be nonzero. Restoring binary
        /// long division — the numerator has a single set bit.
        pub fn div_pow2(&self, z: usize) -> Self {
            assert!(!self.is_zero());
            let mut q = Self(vec![]);
            let mut r = Self(vec![]);
            for i in (0..=z).rev() {
                r = r.shl_bits(1);
                if i == z {
                    r = r.add_small(1);
                }
                if r.ge(self) {
                    r.sub_assign(self);
                    q.set_bit(i);
                }
            }
            q
        }

        fn divmod_small(&self, m: u64) -> (Self, u64) {
            let mut out = vec![0u64; self.0.len()];
            let mut rem = 0u128;
            for i in (0..self.0.len()).rev() {
                let cur = (rem << 64) | u128::from(self.0[i]);
                out[i] = (cur / u128::from(m)) as u64;
                rem = cur % u128::from(m);
            }
            let mut q = Self(out);
            q.normalize();
            (q, rem as u64)
        }

        pub fn to_decimal(&self) -> String {
            if self.is_zero() {
                return "0".to_owned();
            }
            let mut groups = Vec::new();
            let mut cur = self.clone();
            while !cur.is_zero() {
                let (q, r) = cur.divmod_small(10_000_000_000_000_000_000); // 10^19
                groups.push(r);
                cur = q;
            }
            let mut s = groups.pop().expect("nonzero value has digits").to_string();
            for g in groups.iter().rev() {
                s.push_str(&format!("{g:019}"));
            }
            s
        }

        pub fn limb(&self, i: usize) -> u64 {
            self.0.get(i).copied().unwrap_or(0)
        }
    }

    pub fn pow5(n: u32) -> BigUint {
        let mut v = BigUint::from_u64(1);
        for _ in 0..n {
            v = v.mul_small(5);
        }
        v
    }

    pub const MIN_EXP: i32 = -342;
    pub const MAX_EXP: i32 = 308;

    /// The `(hi, lo)` table entry for 5^q: the 128-bit truncated
    /// significand normalized into `[2^127, 2^128)`:
    ///
    /// - `q >= 0`: the top 128 bits of 5^q (shifted up if shorter);
    /// - `q < 0`: `floor(2^z / 5^-q)` with `z = bitlen(5^-q) + 127`, PLUS
    ///   ONE for `q in [-27, -1]` — exactly the negative powers whose
    ///   `5^-q` fits u64, where fast_float deliberately rounds the
    ///   reciprocal upward. Reproduces simdjson's `power_of_five_128` /
    ///   Rust core's `POWER_OF_FIVE_128` entry-for-entry.
    pub fn entry(q: i32) -> (u64, u64) {
        let c = if q >= 0 {
            let c = pow5(u32::try_from(q).expect("q >= 0"));
            let b = c.bit_length();
            if b <= 128 {
                c.shl_bits(128 - b)
            } else {
                c.shr_bits(b - 128)
            }
        } else {
            let d = pow5(u32::try_from(-q).expect("q < 0"));
            let z = d.bit_length() + 127;
            let fl = d.div_pow2(z);
            if q >= -27 { fl.add_small(1) } else { fl }
        };
        assert_eq!(c.bit_length(), 128, "entry for 5^{q} must be normalized");
        (c.limb(1), c.limb(0))
    }

    /// The full `shaders/pow5_table.h` text, byte-for-byte.
    pub fn generate_header() -> String {
        let mut out = String::with_capacity(64 * 1024);
        out.push_str(HEADER_PREFIX);
        for q in MIN_EXP..=MAX_EXP {
            let (hi, lo) = entry(q);
            out.push_str(&format!("    0x{hi:016X}, 0x{lo:016X}, // 5^{q}\n"));
        }
        out.push_str(HEADER_SUFFIX);
        out
    }

    const HEADER_PREFIX: &str = r#"// pow5_table.h - 128-bit truncated powers of five for the K10 number kernel.
//
// GENERATED FILE - do not edit by hand. Regenerate with:
//     cargo test -p metal-json --lib regenerate_pow5_table_header -- --ignored
// The generator and the entry-by-entry big-int verification test live in
// src/gpu/numbers.rs (test module `pow5`); `pow5_table_header_is_generated_
// and_bigint_exact` fails on any drift between this file and the generator.
//
// Layout mirrors simdjson's internal::power_of_five_128 table (simdjson is
// Apache-2.0; the table layout and the Eisel-Lemire algorithm consuming it
// in 12_numbers.metal are derived from simdjson/fast_float - see the
// attribution note in 12_numbers.metal). For every decimal exponent q in
// [MJ_POW5_MIN_EXP, MJ_POW5_MAX_EXP] the entry at index 2*(q + 342) is the
// HIGH 64 bits and index 2*(q + 342) + 1 the LOW 64 bits of the 128-bit
// truncated significand of 5^q, normalized into [2^127, 2^128):
//
//     q >= 0:  the most significant 128 bits of 5^q (5^q shifted left if it
//              has fewer than 128 bits; truncated otherwise);
//     q <  0:  floor(2^z / 5^-q) with z = bitlen(5^-q) + 127, plus one for
//              q in [-27, -1] (the negative powers whose 5^-q fits u64) -
//              fast_float's deliberate upward rounding for the small
//              negative powers.
//
// The entries are bit-identical to Rust core's POWER_OF_FIVE_128
// (library/core/src/num/dec2flt/table.rs) and simdjson's
// internal::power_of_five_128.
//
// 651 entries x 16 bytes = 10416 bytes, well within constant memory.
//
// Like the other headers this file is consumed both by the AOT Metal
// compiler and by the textual include inliner in src/metal/context.rs
// (runtime-shaders), so it stays self-contained apart from <metal_stdlib>.

#ifndef METAL_JSON_POW5_TABLE_H
#define METAL_JSON_POW5_TABLE_H

#include <metal_stdlib>
using namespace metal;

// Decimal-exponent coverage. Outside this range the value is exactly 0.0
// (q < -342 with a <= 19-digit mantissa underflows past the smallest
// subnormal's half) or infinite (q > 308 overflows; rejected as
// InvalidNumber per the tape contract).
constant constexpr int MJ_POW5_MIN_EXP = -342;
constant constexpr int MJ_POW5_MAX_EXP = 308;

// (hi, lo) pairs, indexed by 2 * (q - MJ_POW5_MIN_EXP).
constant constexpr ulong MJ_POW5_TABLE[1302] = {
"#;

    const HEADER_SUFFIX: &str = r#"};

#endif // METAL_JSON_POW5_TABLE_H
"#;
}

#[cfg(test)]
mod tests {
    use super::super::{ERR_INVALID_LITERAL, ERR_UNEXPECTED_TOKEN};
    use super::pow5;
    use super::*;
    use crate::tape::{
        TAG_DOUBLE, TAG_INT64, TAG_UINT64, make_false, make_final_root, make_null, make_root,
        make_true, tag,
    };

    const POW5_HEADER: &str = include_str!("../../shaders/pow5_table.h");

    /// The packed-code value is the `shaders/common.h` contract.
    #[test]
    fn err_number_matches_the_msl_error_code() {
        assert_eq!(ERR_NUMBER, 5, "MJ_ERR_NUMBER in shaders/common.h");
    }

    // --- pow5 table -------------------------------------------------------

    /// The checked-in header is exactly the generator's output AND every
    /// entry parsed back from it equals the big-int recomputation — the
    /// table cannot drift from the generator nor the generator from the
    /// header.
    #[test]
    fn pow5_table_header_is_generated_and_bigint_exact() {
        assert_eq!(
            POW5_HEADER,
            pow5::generate_header(),
            "shaders/pow5_table.h drifted from the generator; regenerate with \
             `cargo test -p metal-json --lib regenerate_pow5_table_header -- --ignored`"
        );

        // Independently parse the header's data section and recompute every
        // entry with the big-int oracle.
        let entries = parse_pow5_header(POW5_HEADER);
        assert_eq!(entries.len(), 651, "651 entries for q in [-342, 308]");
        for (i, &(q, hi, lo)) in entries.iter().enumerate() {
            assert_eq!(
                q,
                pow5::MIN_EXP + i32::try_from(i).unwrap(),
                "entries are in ascending q order"
            );
            assert_eq!(
                pow5::entry(q),
                (hi, lo),
                "big-int recomputation of the 5^{q} entry"
            );
        }
    }

    /// Known entries transcribed from Rust core's POWER_OF_FIVE_128
    /// (library/core/src/num/dec2flt/table.rs — bit-identical to simdjson's
    /// internal::power_of_five_128), pinning the generator against ground
    /// truth that is independent of this codebase.
    #[test]
    fn pow5_known_entries_match_rust_core_and_simdjson() {
        let pins: &[(i32, u64, u64)] = &[
            (-342, 0xeef4_53d6_923b_d65a, 0x113f_aa29_06a1_3b3f),
            (-341, 0x9558_b466_1b65_65f8, 0x4ac7_ca59_a424_c507),
            (-27, 0x9e74_d1b7_91e0_7e48, 0x775e_a264_cf55_347e),
            (-1, 0xcccc_cccc_cccc_cccc, 0xcccc_cccc_cccc_cccd),
            (0, 0x8000_0000_0000_0000, 0x0000_0000_0000_0000),
            (1, 0xa000_0000_0000_0000, 0x0000_0000_0000_0000),
            (308, 0x8e67_9c2f_5e44_ff8f, 0x570f_09ea_a7ea_7648),
        ];
        for &(q, hi, lo) in pins {
            assert_eq!(pow5::entry(q), (hi, lo), "5^{q}");
        }
    }

    /// Rewrites shaders/pow5_table.h from the generator. Run explicitly:
    /// `cargo test -p metal-json --lib regenerate_pow5_table_header -- --ignored`
    #[test]
    #[ignore = "writes shaders/pow5_table.h; run on purpose only"]
    fn regenerate_pow5_table_header() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/pow5_table.h");
        std::fs::write(&path, pow5::generate_header()).expect("writable shaders/pow5_table.h");
    }

    /// Parse the `(hi, lo) // 5^q` lines between `= {` and `};`.
    fn parse_pow5_header(text: &str) -> Vec<(i32, u64, u64)> {
        let body_start = text.find("= {").expect("array open") + 3;
        let body_end = text[body_start..].find("};").expect("array close") + body_start;
        let mut out = Vec::new();
        for line in text[body_start..body_end].lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (data, comment) = line.split_once("//").expect("data lines carry the 5^q tag");
            let q: i32 = comment
                .trim()
                .strip_prefix("5^")
                .expect("comment shape")
                .parse()
                .expect("q parses");
            let mut words = data.split(',').filter_map(|w| {
                let w = w.trim();
                w.strip_prefix("0x")
                    .map(|h| u64::from_str_radix(h, 16).expect("hex u64"))
            });
            let hi = words.next().expect("hi word");
            let lo = words.next().expect("lo word");
            assert!(words.next().is_none(), "two words per line");
            out.push((q, hi, lo));
        }
        out
    }

    // --- GPU harness --------------------------------------------------------

    /// GPU gating, as in stage1/2/3: skip without a device unless
    /// `METAL_JSON_REQUIRE_GPU=1` makes that a hard failure.
    fn gpu_or_skip(test: &str) -> Option<(MetalContext, Numbers)> {
        match MetalContext::new() {
            Ok(ctx) => Some((ctx, Numbers::new())),
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!("SKIP {test}: no usable Metal device here ({err})");
                None
            }
        }
    }

    fn run(numbers: &Numbers, ctx: &MetalContext, input: &[u8]) -> NumbersOutput {
        numbers
            .run(ctx, input)
            .unwrap_or_else(|e| panic!("GPU numbers run failed on {:?}: {e}", preview(input)))
    }

    fn preview(input: &[u8]) -> String {
        String::from_utf8_lossy(&input[..input.len().min(60)]).into_owned()
    }

    /// Parse `text` as a root number scalar: tape must be
    /// `[root, marker, value, final root]`; returns `(tag, value_word)`.
    fn root_number(numbers: &Numbers, ctx: &MetalContext, text: &str) -> (u8, u64) {
        let out = run(numbers, ctx, text.as_bytes());
        assert_eq!(out.error, None, "{text:?} must be accepted");
        assert_eq!(out.tape.len(), 4, "{text:?}: root number tape length");
        assert_eq!(out.tape[0], make_root(3), "{text:?}");
        assert_eq!(out.tape[3], make_final_root(), "{text:?}");
        assert_eq!(
            out.tape[1] & crate::tape::PAYLOAD_MASK,
            0,
            "{text:?}: marker payload is 0"
        );
        (tag(out.tape[1]), out.tape[2])
    }

    /// `text` must parse as a double whose bits equal the
    /// `str::parse::<f64>` oracle, bit-for-bit.
    fn assert_double_matches_oracle(numbers: &Numbers, ctx: &MetalContext, text: &str) {
        let oracle: f64 = text
            .parse()
            .unwrap_or_else(|e| panic!("oracle rejects fixture {text:?}: {e}"));
        assert!(
            oracle.is_finite(),
            "fixture bug: {text:?} is not finite — belongs in the reject table"
        );
        let (t, bits) = root_number(numbers, ctx, text);
        assert_eq!(t, TAG_DOUBLE, "{text:?} selects the double tape kind");
        assert_eq!(
            bits,
            oracle.to_bits(),
            "{text:?}: bits must equal str::parse::<f64> (got {:?}, want {oracle:?})",
            f64::from_bits(bits),
        );
    }

    fn assert_rejected_at(
        numbers: &Numbers,
        ctx: &MetalContext,
        input: &[u8],
        offset: u64,
        code: u32,
    ) {
        let out = run(numbers, ctx, input);
        assert_eq!(
            out.error_offset_code(),
            Some((offset, code)),
            "{:?}: rejection verdict",
            preview(input)
        );
        assert!(
            out.tape.is_empty(),
            "{:?}: rejection contract — the tape is never observed",
            preview(input)
        );
    }

    // --- (a) the tests/numbers.rs torture table, through K10 ----------------

    #[test]
    fn subnormal_normal_boundary_and_range_extremes() {
        let Some((ctx, numbers)) = gpu_or_skip("subnormal_normal_boundary_and_range_extremes")
        else {
            return;
        };
        for text in [
            "2.2250738585072011e-308", // largest subnormal-rounding literal (PHP/Java hang bug)
            "2.2250738585072014e-308", // smallest normal
            "2.2250738585072012e-308", // between: rounds to min normal
            "5e-324",                  // smallest subnormal
            "4.9e-324",
            "2.4703282292062327e-324", // just below half the smallest subnormal: 0.0
            "2.4703282292062328e-324", // just above: rounds up to 5e-324
            "1.7976931348623157e308",  // largest finite f64
            "1.7976931348623158e308",  // rounds back down to f64::MAX
            "1e308",
            "8.98846567431158e307", // 2^1023
            "1e-308",               // subnormal territory via plain exponent
        ] {
            assert_double_matches_oracle(&numbers, &ctx, text);
        }
    }

    #[test]
    fn seventeen_digit_round_trips() {
        let Some((ctx, numbers)) = gpu_or_skip("seventeen_digit_round_trips") else {
            return;
        };
        for text in [
            "0.1234567890123456",
            "0.12345678901234567",
            "1.7976931348623157",
            "17.976931348623157",
            "1797.6931348623157",
            "2.2250738585072014",
            "9007199254740993.0", // 2^53 + 1: not exactly representable
            "9007199254740992.0", // 2^53
            "0.3000000000000000444089209850062616169452667236328125", // exact 0.3
            "0.1",
            "0.2",
            "0.3",
            "123.456",
            "1e23", // famous half-way case
            "9.109383632e-31",
            "6.02214085774e23",
            "7.2057594037927933e16",
        ] {
            assert_double_matches_oracle(&numbers, &ctx, text);
        }
    }

    #[test]
    fn hundred_plus_digit_mantissas() {
        let Some((ctx, numbers)) = gpu_or_skip("hundred_plus_digit_mantissas") else {
            return;
        };
        let big_int = "1".repeat(120);
        assert_double_matches_oracle(&numbers, &ctx, &big_int);
        let big_int_frac = format!("{}.{}", "9".repeat(105), "9".repeat(40));
        assert_double_matches_oracle(&numbers, &ctx, &big_int_frac);
        let long_frac = format!("0.{}1", "0".repeat(100));
        assert_double_matches_oracle(&numbers, &ctx, &long_frac);
        let pi_ish = format!("3.{}", "1415926535897932384626433832795028841971".repeat(3));
        assert_double_matches_oracle(&numbers, &ctx, &pi_ish);
        let mixed = format!("{}e-200", "123456789".repeat(12));
        assert_double_matches_oracle(&numbers, &ctx, &mixed);
        assert_double_matches_oracle(
            &numbers,
            &ctx,
            "0.99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999",
        );
    }

    #[test]
    fn exponent_edge_forms_parse() {
        let Some((ctx, numbers)) = gpu_or_skip("exponent_edge_forms_parse") else {
            return;
        };
        for text in [
            "1e0", "1E0", "1e+0", "1e-0", "1e01", "1e+01",
            "1e-01", // leading zeros in the EXPONENT are legal
            "0e000", "1.5e3", "1.5E+3", "1.5e-3", "100e-2",
        ] {
            assert_double_matches_oracle(&numbers, &ctx, text);
        }
    }

    #[test]
    fn overflow_is_rejected_like_simdjson() {
        let Some((ctx, numbers)) = gpu_or_skip("overflow_is_rejected_like_simdjson") else {
            return;
        };
        for text in [
            "1e309",
            "-1e309",
            "1e400",
            "-1e400",
            "2e308",
            "1.7976931348623159e308", // first literal that rounds to inf
            "1e99999999",
            "123123e100000",
            "0.4e00669999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999969999999006",
        ] {
            assert!(
                text.parse::<f64>().unwrap().is_infinite(),
                "fixture bug: {text:?}"
            );
            assert_rejected_at(&numbers, &ctx, text.as_bytes(), 0, ERR_NUMBER);
        }
    }

    #[test]
    fn underflow_collapses_to_signed_zero() {
        let Some((ctx, numbers)) = gpu_or_skip("underflow_collapses_to_signed_zero") else {
            return;
        };
        for (text, want) in [
            ("1e-400", 0.0f64),
            ("-1e-400", -0.0),
            ("1e-99999999", 0.0),
            ("123e-10000000", 0.0),
            ("-123.456e-789", -0.0),
        ] {
            let (t, bits) = root_number(&numbers, &ctx, text);
            assert_eq!(t, TAG_DOUBLE, "{text:?}");
            assert_eq!(bits, want.to_bits(), "{text:?}: signed-zero underflow");
            assert_double_matches_oracle(&numbers, &ctx, text); // oracle agrees
        }
        // Just above the underflow cliff: subnormal, NOT zero.
        let (_, bits) = root_number(&numbers, &ctx, "1e-310");
        assert_ne!(f64::from_bits(bits), 0.0, "1e-310 is a subnormal");
        assert_double_matches_oracle(&numbers, &ctx, "1e-310");
    }

    #[test]
    fn negative_zero_keeps_its_sign_bit() {
        let Some((ctx, numbers)) = gpu_or_skip("negative_zero_keeps_its_sign_bit") else {
            return;
        };
        for text in ["-0.0", "-0e0", "-0.0e5", "-0E-2"] {
            let (t, bits) = root_number(&numbers, &ctx, text);
            assert_eq!(t, TAG_DOUBLE, "{text:?}");
            assert_eq!(bits, (-0.0f64).to_bits(), "{text:?} keeps the sign bit");
        }
        // Integer "-0" takes the integer fast path: Int64(0), like simdjson.
        assert_eq!(root_number(&numbers, &ctx, "-0"), (TAG_INT64, 0));
        // And "0.0" is plain positive zero.
        assert_eq!(
            root_number(&numbers, &ctx, "0.0"),
            (TAG_DOUBLE, 0.0f64.to_bits())
        );
    }

    // --- (d) type-selection edges -------------------------------------------

    #[test]
    fn integer_type_selection_boundaries() {
        let Some((ctx, numbers)) = gpu_or_skip("integer_type_selection_boundaries") else {
            return;
        };
        let cases: &[(&str, u8, u64)] = &[
            ("9223372036854775807", TAG_INT64, i64::MAX as u64),
            ("-9223372036854775808", TAG_INT64, i64::MIN as u64),
            ("9223372036854775808", TAG_UINT64, 9_223_372_036_854_775_808),
            ("18446744073709551615", TAG_UINT64, u64::MAX),
            (
                "18446744073709551616", // u64::MAX + 1
                TAG_DOUBLE,
                1.844_674_407_370_955_2e19_f64.to_bits(),
            ),
            (
                "-9223372036854775809", // i64::MIN - 1
                TAG_DOUBLE,
                (-9.223_372_036_854_776e18_f64).to_bits(),
            ),
            ("0", TAG_INT64, 0),
            ("-0", TAG_INT64, 0),
            ("-1", TAG_INT64, (-1i64) as u64),
            ("42", TAG_INT64, 42),
            ("0e0", TAG_DOUBLE, 0.0f64.to_bits()),
            (
                "9223372036854775807.0", // fraction forces double
                TAG_DOUBLE,
                9.223_372_036_854_776e18_f64.to_bits(),
            ),
            // Way past u128 too (>39 digits) — reference falls to double.
            (
                "99999999999999999999999999999999999999999999999999",
                TAG_DOUBLE,
                1e50_f64.to_bits(),
            ),
        ];
        for &(text, want_tag, want_word) in cases {
            assert_eq!(
                root_number(&numbers, &ctx, text),
                (want_tag, want_word),
                "{text:?}"
            );
        }
    }

    // --- (a) rejections: reference-exact code + offset -----------------------

    #[test]
    fn grammar_rejections_report_reference_code_and_offset() {
        let Some((ctx, numbers)) =
            gpu_or_skip("grammar_rejections_report_reference_code_and_offset")
        else {
            return;
        };
        // InvalidNumber (K10) for digit-led garbage; UnexpectedToken /
        // InvalidLiteral come from K6 Layer-1 (already pinned in stage 2,
        // re-checked here so the torture table's full verdict column holds
        // through the numbers runner).
        let invalid_number: &[&[u8]] = &[
            b"01",
            b"-01",
            b"00",
            b"012",
            b"0.e1",
            b"-",
            b"1.",
            b"-.",
            b"1e",
            b"1e+",
            b"1e-",
            b"0e",
            b"-x",
            b"--1",
            b"1eE2",
            b"1e1.5",
            b"1.2.3",
            b"0x1",
            b"0X42",
            b"-0x1",
            b"1x",
            b"123abc",
            b"-Infinity",
            b"-inf",
            b"-NaN",
            b"123\x00",
        ];
        for &input in invalid_number {
            assert_rejected_at(&numbers, &ctx, input, 0, ERR_NUMBER);
        }
        // 'n' looks like null -> InvalidLiteral; bytes that cannot start a
        // scalar -> UnexpectedToken (both K6's, carried through).
        assert_rejected_at(&numbers, &ctx, b"nan", 0, ERR_INVALID_LITERAL);
        for &input in &[&b"inf"[..], b"Infinity", b"NaN", b".5", b"+1", b"+0"] {
            assert_rejected_at(&numbers, &ctx, input, 0, ERR_UNEXPECTED_TOKEN);
        }
        // The reference's offset test: the error reports at the TOKEN
        // position of the offending scalar, inside a container.
        assert_rejected_at(&numbers, &ctx, br#"[1, 2.5, 0x1]"#, 9, ERR_NUMBER);
        // Earliest offset wins across multiple bad scalars.
        assert_rejected_at(&numbers, &ctx, b"[01, 0x2]", 1, ERR_NUMBER);
        // ... and the same rejections hold nested in an object.
        assert_rejected_at(&numbers, &ctx, br#"{"k": 1e}"#, 6, ERR_NUMBER);
    }

    // --- literals + the worked example ---------------------------------------

    #[test]
    fn literals_write_their_words() {
        let Some((ctx, numbers)) = gpu_or_skip("literals_write_their_words") else {
            return;
        };
        let cases: &[(&[u8], u64)] = &[
            (b"true", make_true()),
            (b"false", make_false()),
            (b"null", make_null()),
        ];
        for &(input, word) in cases {
            let out = run(&numbers, &ctx, input);
            assert_eq!(out.error, None, "{input:?}");
            assert_eq!(out.tape, vec![make_root(2), word, make_final_root()]);
            assert!(out.fixup_tokens.is_empty(), "literals never take fixups");
        }
        // Nested: [true,false,null] -> words at tape 2, 3, 4.
        let out = run(&numbers, &ctx, b"[true,false,null]");
        assert_eq!(out.error, None);
        assert_eq!(out.tape[2], make_true());
        assert_eq!(out.tape[3], make_false());
        assert_eq!(out.tape[4], make_null());
    }

    /// THE tape pin: the docs/tape-format.md worked example with K10's
    /// holes now filled — container/root words intact from M3, the number
    /// holes carry the exact `l 1` / `d 2.5` entries, and the string holes
    /// (K11's) stay zero words.
    #[test]
    fn worked_example_numbers_fill_their_holes() {
        let Some((ctx, numbers)) = gpu_or_skip("worked_example_numbers_fill_their_holes") else {
            return;
        };
        let out = run(&numbers, &ctx, br#"{"a":[1,2.5],"b":"x\n"}"#);
        assert_eq!(out.error, None);
        let expected: [u64; 13] = [
            0x7200_0000_0000_000C, // r -> 12
            0x7B00_0002_0000_000C, // { end=12 count=2
            0,                     // hole: "a" (K11)
            0x5B00_0002_0000_0009, // [ end=9 count=2
            0x6C00_0000_0000_0000, // l marker          (K10)
            1,                     // i64 bits of 1     (K10)
            0x6400_0000_0000_0000, // d marker          (K10)
            0x4004_0000_0000_0000, // f64 bits of 2.5   (K10)
            0x5D00_0000_0000_0003, // ] open=3
            0,                     // hole: "b" (K11)
            0,                     // hole: "x\n" (K11)
            0x7D00_0000_0000_0001, // } open=1
            0x7200_0000_0000_0000, // r -> 0
        ];
        assert_eq!(out.tape, expected);
        assert!(out.fixup_tokens.is_empty());
    }

    // --- (b) fixup-path coverage ---------------------------------------------

    /// Exact decimal JSON literal of the halfway point between the finite
    /// nonnegative f64 with these `bits` and the next value up (for
    /// `f64::MAX` that "next" is 2^1024, the overflow threshold). The
    /// halfway value is `(2m + 1) * 2^(e-1)`, printed exactly:
    /// `e-1 >= 0` gives a big integer; `e-1 < 0` gives
    /// `(2m+1) * 5^(1-e) * 10^(e-1)` — digits plus a decimal exponent.
    fn halfway_up_decimal(bits: u64) -> String {
        let exp_field = (bits >> 52) & 0x7FF;
        let frac = bits & ((1u64 << 52) - 1);
        let (m, e) = if exp_field == 0 {
            (frac, -1074i64) // subnormal (covers +0.0: halfway to 5e-324)
        } else {
            ((1u64 << 52) | frac, exp_field as i64 - 1075)
        };
        let m2 = pow5::BigUint::from_u64(2 * m + 1);
        let e2 = e - 1;
        if e2 >= 0 {
            m2.shl_bits(usize::try_from(e2).unwrap()).to_decimal()
        } else {
            let k = u32::try_from(-e2).unwrap();
            let mut digits = m2;
            for _ in 0..k {
                digits = digits.mul_small(5);
            }
            format!("{}e{}", digits.to_decimal(), e2)
        }
    }

    /// Inputs that MUST take the fixup path: exact halfway points between
    /// adjacent doubles, written with their full (>= 20 significant digit)
    /// decimal expansions. The kernel's truncated mantissa w satisfies
    /// `w*10^q < halfway < (w+1)*10^q`, so the `w` / `w+1` parses land on
    /// the two different neighbors and the kernel cannot decide — the
    /// fixup list must be non-empty and the CPU patch must produce the
    /// `str::parse` ties-to-even bits exactly.
    #[test]
    fn fixup_path_resolves_halfway_cases_bit_exactly() {
        let Some((ctx, numbers)) = gpu_or_skip("fixup_path_resolves_halfway_cases_bit_exactly")
        else {
            return;
        };
        let cases: &[(&str, u64)] = &[
            // halfway(1.0, 1.0 + ulp): ties to even -> 1.0 (54 digits).
            ("one", 1.0f64.to_bits()),
            // halfway(largest subnormal, smallest normal): the long-form
            // 2.2250738585072011e-308 family; ties to even -> min normal.
            ("min-normal boundary", 0x000F_FFFF_FFFF_FFFF),
            // halfway(0, smallest subnormal) = 2^-1075: ties to even -> 0.0.
            ("zero", 0.0f64.to_bits()),
            // halfway(1e22's neighbor pair) — the famous power-of-ten case.
            ("1e22", 1e22f64.to_bits()),
        ];
        for &(label, bits) in cases {
            let text = halfway_up_decimal(bits);
            let oracle: f64 = text.parse().expect("halfway literal parses");
            assert!(oracle.is_finite(), "{label}: finite fixture");
            let out = run(&numbers, &ctx, text.as_bytes());
            assert_eq!(out.error, None, "{label}: accepted");
            assert!(
                !out.fixup_tokens.is_empty(),
                "{label}: a halfway value with truncated digits MUST take the fixup path"
            );
            assert_eq!(tag(out.tape[1]), TAG_DOUBLE, "{label}");
            assert_eq!(
                out.tape[2],
                oracle.to_bits(),
                "{label}: CPU-patched bits must equal str::parse ({text})"
            );
        }

        // Long Pi digits and the classic literal family: bit-exact through
        // whichever path the kernel picks (fixup not asserted — these are
        // not provably ambiguous).
        let pi100 = format!(
            "3.{}",
            "14159265358979323846264338327950288419716939937510".repeat(2)
        );
        assert_double_matches_oracle(&numbers, &ctx, &pi100);
        assert_double_matches_oracle(
            &numbers,
            &ctx,
            "2.22507385850720113605740979670913197593481954635164564e-308",
        );
        assert_double_matches_oracle(
            &numbers,
            &ctx,
            "2.22507385850720088902458687608585988765042311224095946549352480256244000922823569517877588880375915526423097809504343120858773871583572918219930202943792242235598198275012420417889695713117910822610439719796040004548973919380791989360815256131133761498420432717510336273915497827315941438281362751138386040942494649422863166954291050802018159266421349966065178030950759130587198464239060686371020051087232827846788436319445158661350412234790147923695852083215976210663754016137365830441936037147783553066828345356340050740730401356029680463759185831631242245215992625464943008147621113864903953346516323261504496347", // 600+ digits in the family
        );

        // -0.0 sign is preserved through the FIXUP path too: the negated
        // halfway-below-min-subnormal rounds to -0.0.
        let neg_zero = format!("-{}", halfway_up_decimal(0));
        let out = run(&numbers, &ctx, neg_zero.as_bytes());
        assert_eq!(out.error, None);
        assert!(!out.fixup_tokens.is_empty(), "negated halfway: fixup fires");
        assert_eq!(out.tape[2], (-0.0f64).to_bits(), "sign bit preserved");
    }

    /// The fixup-driven rejection: the halfway point between f64::MAX and
    /// 2^1024 ties to even = infinity, which the CPU re-parse must reject
    /// with the reference's InvalidNumber at the token offset. One digit
    /// below it stays finite = f64::MAX.
    #[test]
    fn fixup_driven_overflow_rejects_like_the_reference() {
        let Some((ctx, numbers)) = gpu_or_skip("fixup_driven_overflow_rejects_like_the_reference")
        else {
            return;
        };
        let threshold = halfway_up_decimal(f64::MAX.to_bits());
        assert!(
            threshold.parse::<f64>().unwrap().is_infinite(),
            "the overflow threshold ties to infinity"
        );
        let out = run(&numbers, &ctx, threshold.as_bytes());
        assert!(
            !out.fixup_tokens.is_empty(),
            "the threshold straddles MAX/inf: the kernel must punt to the CPU"
        );
        assert_eq!(
            out.error_offset_code(),
            Some((0, ERR_NUMBER)),
            "CPU re-parse confirms infinity and rejects"
        );
        assert!(out.tape.is_empty(), "rejected: tape never observed");

        // One ulp of the decimal below: finite, exactly f64::MAX.
        let mut below = threshold.clone().into_bytes();
        let last = below.last_mut().expect("nonempty");
        assert!((b'1'..=b'9').contains(last), "integer form, nonzero tail");
        *last -= 1;
        let below = String::from_utf8(below).unwrap();
        assert_double_matches_oracle(&numbers, &ctx, &below);

        // Nested: the rejecting fixup reports at ITS token offset, and
        // beats a later GPU-detected grammar error (packed min merge).
        let nested = format!("[1, {threshold}, 01]");
        let out = run(&numbers, &ctx, nested.as_bytes());
        assert_eq!(
            out.error_offset_code(),
            Some((4, ERR_NUMBER)),
            "fixup rejection offset wins over the later grammar error"
        );
    }

    // --- random + canada-style spreads ----------------------------------------

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Deterministic random f64 bit patterns (finite only), shortest
    /// round-trip formatted — every one must round-trip bit-exactly
    /// through K10. Mirrors the tests/numbers.rs proptest on the GPU.
    #[test]
    fn random_f64_round_trips_bit_exactly() {
        let Some((ctx, numbers)) = gpu_or_skip("random_f64_round_trips_bit_exactly") else {
            return;
        };
        let mut state = 0x0123_4567_89AB_CDEF_u64;
        let mut texts = Vec::new();
        let mut expected = Vec::new();
        while texts.len() < 256 {
            let value = f64::from_bits(splitmix64(&mut state));
            if !value.is_finite() {
                continue;
            }
            let text = format!("{value:e}");
            assert_eq!(
                text.parse::<f64>().unwrap().to_bits(),
                value.to_bits(),
                "Rust shortest-form formatting round-trips"
            );
            texts.push(text);
            expected.push((TAG_DOUBLE, value.to_bits()));
        }
        // One document with all of them (exercises the multi-entry scalar
        // list), then spot-check a few as root scalars.
        let doc = format!("[{}]", texts.join(","));
        let out = run(&numbers, &ctx, doc.as_bytes());
        assert_eq!(out.error, None);
        assert_eq!(collect_number_entries(&out.tape), expected);
        for text in texts.iter().take(8) {
            assert_double_matches_oracle(&numbers, &ctx, text);
        }
    }

    /// Walk the tape and collect every number entry as (tag, value word),
    /// in document order.
    fn collect_number_entries(tape: &[u64]) -> Vec<(u8, u64)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < tape.len() {
            let t = tag(tape[i]);
            if t == TAG_INT64 || t == TAG_UINT64 || t == TAG_DOUBLE {
                out.push((t, tape[i + 1]));
                i += 2;
            } else {
                i += 1;
            }
        }
        out
    }

    /// A canada.json-shaped document: coordinate-like doubles with mixed
    /// formatting plus integer stragglers, in nested pair arrays. Returns
    /// the JSON and the expected (tag, value word) sequence in document
    /// order. 3000 scalars also pushes the token stream across multiple
    /// 1024-token chunks.
    fn canada_style_doc(n: usize) -> (Vec<u8>, Vec<(u8, u64)>) {
        let mut state = 0x6A09_E667_F3BC_C908_u64;
        let mut json = b"[".to_vec();
        let mut expected = Vec::new();
        for i in 0..n / 2 {
            if i > 0 {
                json.push(b',');
            }
            json.push(b'[');
            for j in 0..2 {
                if j > 0 {
                    json.push(b',');
                }
                let r = splitmix64(&mut state);
                let text = match i % 5 {
                    // coordinate-like: scaled fixed-point of varying precision
                    0 => format!("{:?}", ((r % 360_000_000) as f64 / 1e6) - 180.0),
                    1 => format!("{:?}", ((r % 180_000_000_000) as f64 / 1e9) - 90.0),
                    2 => format!("{:e}", f64::from_bits(r & 0x7FEF_FFFF_FFFF_FFFF)),
                    3 => format!("{:.9}", (r % 1_000_000) as f64 / 997.0),
                    // integer stragglers, both signs
                    _ => format!("{}", (r as i64) >> 16),
                };
                if text.contains('.') || text.contains('e') || text.contains('E') {
                    expected.push((TAG_DOUBLE, text.parse::<f64>().unwrap().to_bits()));
                } else {
                    let v: i64 = text.parse().unwrap();
                    expected.push((TAG_INT64, v as u64));
                }
                json.extend_from_slice(text.as_bytes());
            }
            json.push(b']');
        }
        json.push(b']');
        (json, expected)
    }

    #[test]
    fn canada_style_float_spread_matches_the_oracle() {
        let Some((ctx, numbers)) = gpu_or_skip("canada_style_float_spread_matches_the_oracle")
        else {
            return;
        };
        let (json, expected) = canada_style_doc(3000);
        let out = run(&numbers, &ctx, &json);
        assert_eq!(out.error, None);
        assert_eq!(
            collect_number_entries(&out.tape),
            expected,
            "canada-style spread: every number bit-exact in document order"
        );
    }

    /// EOF-terminated runs (no delimiter after the number) parse fine —
    /// the run ends at input_len, like the reference.
    #[test]
    fn numbers_at_end_of_input_parse() {
        let Some((ctx, numbers)) = gpu_or_skip("numbers_at_end_of_input_parse") else {
            return;
        };
        assert_eq!(root_number(&numbers, &ctx, "42"), (TAG_INT64, 42));
        assert_double_matches_oracle(&numbers, &ctx, "2.5");
    }

    // --- (c) vs the cpu-reference oracle ---------------------------------------

    #[cfg(feature = "cpu-reference")]
    mod vs_reference {
        use super::*;
        use crate::reference::{ScalarValue, stage1_classify, stage2_tokens, stage5_scalars};

        /// Diff every scalar of an ACCEPTED document against reference
        /// stage 5: the tape words at tape_ofs[token] must encode exactly
        /// the reference's parsed values.
        fn diff_scalars(numbers: &Numbers, ctx: &MetalContext, input: &[u8], label: &str) {
            let bitmaps = stage1_classify(input).expect("fixture passes stage 1");
            let tokens = stage2_tokens(&bitmaps, input);
            let scalars = stage5_scalars(&tokens, input)
                .unwrap_or_else(|e| panic!("{label}: fixture passes stage 5: {e}"));
            let out = run(numbers, ctx, input);
            assert_eq!(out.error, None, "{label}: accepted");
            for scalar in &scalars {
                let ofs = out.structure.stage2.tape_ofs[scalar.token_index as usize] as usize;
                let (want_marker, want_value) = match scalar.value {
                    ScalarValue::Int64(v) => (Some(crate::tape::make_int64_marker()), v as u64),
                    ScalarValue::UInt64(v) => (Some(crate::tape::make_uint64_marker()), v),
                    ScalarValue::Double(d) => {
                        (Some(crate::tape::make_double_marker()), d.to_bits())
                    }
                    ScalarValue::True => (None, make_true()),
                    ScalarValue::False => (None, make_false()),
                    ScalarValue::Null => (None, make_null()),
                };
                match want_marker {
                    Some(marker) => {
                        assert_eq!(
                            out.tape[ofs], marker,
                            "{label}: marker at tape[{ofs}] (token {})",
                            scalar.token_index
                        );
                        assert_eq!(
                            out.tape[ofs + 1],
                            want_value,
                            "{label}: value word at tape[{}] (token {})",
                            ofs + 1,
                            scalar.token_index
                        );
                    }
                    None => {
                        assert_eq!(
                            out.tape[ofs], want_value,
                            "{label}: literal word at tape[{ofs}]"
                        );
                    }
                }
            }
        }

        /// Every number in corpus/ diffed against reference stage 5.
        #[test]
        fn corpus_scalars_match_reference_stage5() {
            let Some((ctx, numbers)) = gpu_or_skip("corpus_scalars_match_reference_stage5") else {
                return;
            };
            let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
            let mut paths: Vec<_> = std::fs::read_dir(&corpus)
                .expect("corpus/ is checked in")
                .map(|e| e.expect("readable corpus entry").path())
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            paths.sort();
            assert!(!paths.is_empty(), "corpus/ must contain fixtures");
            for path in paths {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let bytes = std::fs::read(&path).expect("readable corpus fixture");
                diff_scalars(&numbers, &ctx, &bytes, &name);
            }
        }

        /// The canada-style spread through the reference oracle as well
        /// (the in-module oracle above is str::parse; this one is the full
        /// reference stage-5 path, type selection included).
        #[test]
        fn canada_style_spread_matches_reference_stage5() {
            let Some((ctx, numbers)) = gpu_or_skip("canada_style_spread_matches_reference_stage5")
            else {
                return;
            };
            let (json, _) = canada_style_doc(1000);
            diff_scalars(&numbers, &ctx, &json, "canada-style spread");
        }

        /// Rejection parity on scalar-content errors: the reference stage-5
        /// verdict (offset + InvalidNumber) equals the GPU's packed error.
        #[test]
        fn rejection_offsets_match_reference_stage5() {
            let Some((ctx, numbers)) = gpu_or_skip("rejection_offsets_match_reference_stage5")
            else {
                return;
            };
            for input in [
                &br#"[1, 2.5, 0x1]"#[..],
                b"[01]",
                br#"{"k":1e999}"#,
                br#"{"a":[1,2,-]}"#,
                b"[1.2.3, 4]",
            ] {
                let bitmaps = stage1_classify(input).expect("fixture passes stage 1");
                let tokens = stage2_tokens(&bitmaps, input);
                let err = stage5_scalars(&tokens, input)
                    .expect_err("fixture must fail reference stage 5");
                let crate::Error::Syntax { offset, kind } = err else {
                    panic!("unexpected reference error {err:?}");
                };
                assert_eq!(kind, crate::SyntaxErrorKind::InvalidNumber);
                let out = run(&numbers, &ctx, input);
                assert_eq!(
                    out.error_offset_code(),
                    Some((offset, ERR_NUMBER)),
                    "{:?}",
                    String::from_utf8_lossy(input)
                );
            }
        }
    }
}
