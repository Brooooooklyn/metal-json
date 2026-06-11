// 12_numbers.metal — K10 `parse_numbers`: number grammar validation + value
// parsing (the scalar half of M4; strings are K11's).
//
//   parse_numbers   1 thread / scalar-list entry (the K6b `scalar_tokens`
//                   work list): at tok_pos[token] either a literal first
//                   byte ('t'/'f'/'n' — VALIDATED BY K6 already, so K10
//                   only writes the 1-word tape entry, never re-checks or
//                   re-reports) or a number ('-'/digit), which gets the
//                   full reference-stage-5 treatment:
//
//                   1. grammar walk over the scalar run
//                      (-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?,
//                      consuming the ENTIRE run) — any violation reports
//                      MJ_ERR_NUMBER at the token's byte offset;
//                   2. integer fast path (no '.'/'e'/'E'): exact u64
//                      accumulation; fits i64 -> 'l' two-word entry, fits
//                      u64 -> 'u'; out of range falls to the double path;
//                   3. double path: <= 19 significant decimal digits into a
//                      u64 mantissa w (further digits only bump the decimal
//                      exponent; a dropped nonzero digit sets the
//                      `truncated` flag), then Eisel-Lemire over the
//                      128-bit pow5 table produces the correctly rounded
//                      f64 BIT PATTERN as a ulong — Apple GPUs have no
//                      fp64, so no `double` appears anywhere in this file.
//                      'd' marker word + raw bits word.
//
// HARD-CASE FIXUP CONTRACT: when Eisel-Lemire cannot certify the rounding —
// the truncated-product off-by-one ambiguity, a truncated mantissa whose
// w and w+1 parses disagree (the value lies between, so the kernel cannot
// know which side wins), or a rounding that lands past the largest finite
// exponent (would-be infinity, which the CPU must confirm and reject) —
// the thread appends the TOKEN INDEX to the fixup list with a device
// atomic counter (order does not matter; the CPU sorts) and writes the 'd'
// marker with a 0 placeholder value word. The CPU patch
// (`gpu::numbers::patch_number_fixups`) re-parses those few scalars with
// the reference oracle's double path (str::parse, correctly rounded;
// infinities rejected as InvalidNumber per simdjson parity) and overwrites
// the 8-byte value words in the shared tape buffer before Document
// construction.
//
// VERDICT POLICY (reference stage-5 parity, src/reference/scalars.rs):
//   - grammar violations ("01", "-", "1.", ".5"-style runs, "1e", "0x1",
//     trailing junk) -> MJ_ERR_NUMBER at the token offset;
//   - overflow to infinity (decimal exponent > 308 with a nonzero
//     mantissa) -> MJ_ERR_NUMBER (simdjson rejects infinities); near-MAX
//     roundings go through the fixup path and the CPU rejects the infinite
//     ones with the same code+offset;
//   - underflow (decimal exponent < -342, or a subnormal that rounds to
//     zero) -> ACCEPTED as a signed zero ("-1e-400" keeps its sign bit);
//   - "-0.0" -> -0.0 (sign preserved); integer "-0" -> Int64 0.
//
// ERROR PROTOCOL: every K10 error is the single code MJ_ERR_NUMBER, so the
// kernel reduces just the OFFSET with a 32-bit atomic_min (the stage-1 K1
// UTF-8 protocol — 64-bit atomics are an Apple9+ feature the embedded
// metallib cannot assume; offsets fit u32 because input_len is capped).
// The earliest offset IS the reference's first-in-token-order failing
// scalar (token positions strictly increase). The runner packs
// (offset << 32) | MJ_ERR_NUMBER on the CPU and applies the rejection
// contract: a number error short-circuits Document construction — the tape
// is never observed.
//
// 128-bit products use the explicit limbs64 umul128 — four (ulong)uint*uint
// partial products assembled with 64-bit adds/shifts, simdjson's portable
// umul128 shape — per the binding spike-A decision (docs/spikes.md):
// measured ~1.6x faster than mulhi(ulong)+x*y when both product words are
// needed. Mantissa/product state stays ulong (clean 2x32 lowering); the
// hot path has no 64-bit rotates/popcounts (clz runs on 32-bit halves).
//
// The Eisel-Lemire algorithm and the pow5 table layout are derived from
// simdjson's compute_float_64 / internal::power_of_five_128 (Apache-2.0,
// (c) the simdjson authors), itself from fast_float (Daniel Lemire et al.,
// "Number Parsing at a Gigabyte per Second"). pow5_table.h is generated +
// entry-by-entry verified in src/gpu/numbers.rs.
//
// KNOWN PERF CLIFF (documented, v1-acceptable — same valve story as K11):
// a thread walks its whole scalar run sequentially, so a pathological
// multi-KB digit string serializes one thread. If real workloads ever care,
// the threadgroup-per-scalar valve would go where the grammar walk loops
// over [int_begin, int_end) / [frac_begin, frac_end) — both are trivially
// chunkable scans — gated by a run-length histogram from K6.
//
// Dispatched as a plain thread grid (Dispatch::Threads) with a per-thread
// bound check; no cooperative scans, no barriers.

#include "common.h"
#include "tape_types.h"
#include "pow5_table.h"

// --- helpers -------------------------------------------------------------------

static inline bool mj_k10_is_digit(uchar b) {
    return b >= uchar('0') && b <= uchar('9');
}

// Does byte `b` terminate a scalar run? (reference::scalars::scalar_run
// stop set: whitespace, structural operator, or '"'.)
static inline bool mj_k10_run_ends(uchar b) {
    return mj_is_ws_byte(b) || mj_is_op_byte(b) || b == uchar('"');
}

// Report an invalid number at byte `pos` (32-bit offset atomic_min; the
// runner packs the MJ_ERR_NUMBER code on the CPU).
static inline void mj_k10_report(device atomic_uint* err_min_pos, ulong pos) {
    atomic_fetch_min_explicit(err_min_pos, uint(pos), memory_order_relaxed);
}

// 128-bit product of two u64s, explicit limbs64 umul128 (the spike-A
// binding decision; simdjson's portable Emulate64x64to128 shape).
struct MjU128 {
    ulong hi;
    ulong lo;
};

static inline MjU128 mj_k10_umul128(ulong a, ulong b) {
    uint a_lo = uint(a);
    uint a_hi = uint(a >> 32);
    uint b_lo = uint(b);
    uint b_hi = uint(b >> 32);
    ulong p_ll = ulong(a_lo) * b_lo;
    ulong p_lh = ulong(a_lo) * b_hi;
    ulong p_hl = ulong(a_hi) * b_lo;
    ulong p_hh = ulong(a_hi) * b_hi;
    // Neither mid sum can carry: (2^32-1)^2 + (2^32-1) < 2^64.
    ulong mid = p_hl + (p_ll >> 32);
    ulong mid2 = p_lh + (mid & 0xFFFFFFFFul);
    MjU128 r;
    r.hi = p_hh + (mid >> 32) + (mid2 >> 32);
    r.lo = (mid2 << 32) | (p_ll & 0xFFFFFFFFul);
    return r;
}

// mj_k10_eisel_lemire result statuses.
constant constexpr uint MJ_EL_OK = 0;    // *out_bits is the correctly rounded f64
constant constexpr uint MJ_EL_HARD = 1;  // cannot certify: take the fixup path

// Eisel-Lemire: the correctly rounded IEEE-754 binary64 bit pattern of
// (negative ? -1 : 1) * w * 10^q, computed in pure 64-bit integer math.
// Line-for-line port of simdjson's compute_float_64 (minus its fp64 fast
// path — no doubles exist here). Preconditions: w != 0 and q in
// [MJ_POW5_MIN_EXP, MJ_POW5_MAX_EXP] (the caller handles both overflow and
// underflow outside that window).
//
// MJ_EL_HARD covers (a) the truncated-product ambiguity (product.lo
// saturated after the second 64-bit refinement — the rounding cannot be
// proven) and (b) a rounded exponent past 2046 (the value rounds to
// infinity; the CPU re-parse confirms and rejects). Both are rare; the
// caller appends to the fixup list.
static inline uint mj_k10_eisel_lemire(ulong w, int q, bool negative,
                                       thread ulong* out_bits) {
    ulong sign_bit = negative ? 0x8000000000000000ul : 0ul;

    // Normalize w so its top bit is set; clz on 32-bit halves (no 64-bit
    // bit-scan in the hot loop — spike-A guidance).
    uint w_hi = uint(w >> 32);
    int lz = (w_hi != 0u) ? int(clz(w_hi)) : 32 + int(clz(uint(w)));
    w <<= ulong(lz);

    // Truncated 128-bit product w * 5^q (both operands have their top bit
    // set, so the product has 0 or 1 leading zeros). Unless the low 9 bits
    // of the high word are all ones, the top 55 bits are already exact;
    // otherwise refine with the second table word.
    uint index = 2u * uint(q - MJ_POW5_MIN_EXP);
    MjU128 product = mj_k10_umul128(w, MJ_POW5_TABLE[index]);
    if ((product.hi & 0x1FFul) == 0x1FFul) {
        MjU128 second = mj_k10_umul128(w, MJ_POW5_TABLE[index + 1u]);
        product.lo += second.hi;
        if (second.hi > product.lo) {
            product.hi += 1ul;
        }
        // One more unit could still flip product.hi if product.lo is
        // saturated — the famous "much more work" case: punt to the CPU.
        if (product.lo == 0xFFFFFFFFFFFFFFFFul) {
            return MJ_EL_HARD;
        }
    }

    ulong upper = product.hi;
    ulong lower = product.lo;
    // 54-bit mantissa with a leading 1 (53 + 1 rounding bit).
    ulong upperbit = upper >> 63;
    ulong mantissa = upper >> (upperbit + 9ul);
    lz += int(1ul ^ upperbit);

    // Binary exponent: (((152170 + 65536) * q) >> 16) + 1024 + 63 - lz.
    // Rewritten so the shifted operand is provably nonnegative over q in
    // [-342, 308] (no implementation-defined negative right shift):
    // 74514432 = 1137 << 16, and 217706 * -342 + 74514432 = 58980 >= 0.
    int real_exponent = (((217706 * q + 74514432) >> 16) - 1137) + 1087 - lz;

    if (real_exponent <= 0) { // subnormal (or rounds up into the min normal)
        if (-real_exponent + 1 >= 64) {
            // Smaller than half the smallest subnormal: a signed zero.
            *out_bits = sign_bit;
            return MJ_EL_OK;
        }
        mantissa >>= ulong(-real_exponent + 1);
        mantissa += (mantissa & 1ul); // round up
        mantissa >>= 1ul;
        // Rounding can carry into the min normal (the 2.2250738585072013e-308
        // shape): only subnormal while below 2^52.
        real_exponent = (mantissa < (1ul << 52)) ? 0 : 1;
        *out_bits = (mantissa & ~(1ul << 52)) | (ulong(real_exponent) << 52) | sign_bit;
        return MJ_EL_OK;
    }

    // Round-to-even guard: a candidate halfway value (only possible for
    // q in [-4, 23], where w * 5^q can be exact) must not round up.
    if (lower <= 1ul && q >= -4 && q <= 23 && (mantissa & 3ul) == 1ul) {
        if ((mantissa << (upperbit + 64ul - 53ul - 2ul)) == upper) {
            mantissa &= ~1ul; // round down to even
        }
    }

    mantissa += mantissa & 1ul;
    mantissa >>= 1ul;
    if (mantissa >= (1ul << 53)) { // rounding overflowed into 2^53
        mantissa = 1ul << 52;
        real_exponent += 1;
    }
    mantissa &= ~(1ul << 52);
    if (real_exponent > 2046) {
        // Rounds to infinity. The CPU re-parse confirms and rejects with
        // the reference's InvalidNumber verdict (fixup path).
        return MJ_EL_HARD;
    }
    *out_bits = mantissa | (ulong(real_exponent) << 52) | sign_bit;
    return MJ_EL_OK;
}

// --- K10: parse_numbers --------------------------------------------------------

kernel void parse_numbers(
    device const uchar* input        [[buffer(0)]],
    device const uint* scalar_tokens [[buffer(1)]],
    device const uint* tok_pos       [[buffer(2)]],
    device const uint* tape_ofs      [[buffer(3)]],
    device ulong* tape               [[buffer(4)]],
    device atomic_uint* err_min_pos  [[buffer(5)]],
    device atomic_uint* fixup_count  [[buffer(6)]],
    device uint* fixup_tokens        [[buffer(7)]],
    constant MjParams& params        [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (ulong(tid) >= params.element_count) { // element_count = scalar_total
        return;
    }
    uint tok = scalar_tokens[tid];
    ulong pos = ulong(tok_pos[tok]);
    uint ofs = tape_ofs[tok];
    ulong len = params.input_len;
    uchar first = input[pos];

    // Literals: K6 Layer-1 already byte-checked true/false/null and the
    // boundary rule — write the 1-word entry, never duplicate the error.
    if (first == uchar('t')) {
        tape[ofs] = mj_make_entry(MJ_TAG_TRUE, 0ul);
        return;
    }
    if (first == uchar('f')) {
        tape[ofs] = mj_make_entry(MJ_TAG_FALSE, 0ul);
        return;
    }
    if (first == uchar('n')) {
        tape[ofs] = mj_make_entry(MJ_TAG_NULL, 0ul);
        return;
    }
    if (first != uchar('-') && !mj_k10_is_digit(first)) {
        // Unreachable on accepted token streams: K6 rejected non-scalar
        // first bytes (UnexpectedToken). Inert for memory safety.
        return;
    }

    // --- grammar walk (reference parse_number, src/reference/scalars.rs) ---
    // -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)? consuming the entire
    // run; every violation reports at the TOKEN offset (reference parity).
    ulong p = pos;
    bool negative = first == uchar('-');
    if (negative) {
        p += 1ul;
    }
    ulong int_begin = p;
    while (p < len && mj_k10_is_digit(input[p])) {
        p += 1ul;
    }
    ulong int_end = p;
    // "-", "-x": no integer digits. "01", "-012", "00": leading zero.
    bool ok = int_end != int_begin
        && !(int_end - int_begin > 1ul && input[int_begin] == uchar('0'));

    bool is_double = false;
    ulong frac_begin = 0ul;
    ulong frac_end = 0ul;
    if (ok && p < len && input[p] == uchar('.')) {
        is_double = true;
        p += 1ul;
        frac_begin = p;
        while (p < len && mj_k10_is_digit(input[p])) {
            p += 1ul;
        }
        frac_end = p;
        ok = frac_end != frac_begin; // "1.", "1.e5"
    }

    long exp_val = 0l;
    if (ok && p < len && (input[p] == uchar('e') || input[p] == uchar('E'))) {
        is_double = true;
        p += 1ul;
        bool exp_neg = false;
        if (p < len && (input[p] == uchar('+') || input[p] == uchar('-'))) {
            exp_neg = input[p] == uchar('-');
            p += 1ul;
        }
        ulong exp_begin = p;
        long acc = 0l;
        while (p < len && mj_k10_is_digit(input[p])) {
            // Saturate far past the f64 range: longer exponents only ever
            // mean "overflow" or "underflow", both decided below.
            if (acc < 100000000000000000l) {
                acc = acc * 10l + long(input[p] - uchar('0'));
            }
            p += 1ul;
        }
        ok = ok && p != exp_begin; // "1e", "1e+"
        exp_val = exp_neg ? -acc : acc;
    }
    // The grammar must consume the entire scalar run: the next byte is a
    // run terminator or EOF, else trailing junk ("1x", "0x1", "1.2.3").
    ok = ok && (p >= len || mj_k10_run_ends(input[p]));
    if (!ok) {
        mj_k10_report(err_min_pos, pos);
        return;
    }

    // --- integer fast path (no '.', no exponent) ---------------------------
    if (!is_double) {
        ulong digits = int_end - int_begin;
        if (digits <= 20ul) { // u64::MAX has 20 digits; 21+ always overflows
            ulong w = 0ul;
            bool overflow = false;
            for (ulong i = int_begin; i < int_end; i += 1ul) {
                uint d = uint(input[i] - uchar('0'));
                // w * 10 + d > u64::MAX <=> w > (u64::MAX - d) / 10.
                if (w > 1844674407370955161ul
                    || (w == 1844674407370955161ul && d > 5u)) {
                    overflow = true;
                    break;
                }
                w = w * 10ul + ulong(d);
            }
            if (!overflow) {
                if (negative) {
                    if (w <= 0x8000000000000000ul) {
                        // -(2^63) == i64::MIN still fits; ~w + 1 is the
                        // two's complement for every w incl. 2^63.
                        tape[ofs] = mj_make_entry(MJ_TAG_INT64, 0ul);
                        tape[ofs + 1u] = ~w + 1ul;
                        return;
                    }
                    // Below i64::MIN: double path.
                } else if (w <= 0x7FFFFFFFFFFFFFFFul) {
                    tape[ofs] = mj_make_entry(MJ_TAG_INT64, 0ul);
                    tape[ofs + 1u] = w;
                    return;
                } else {
                    tape[ofs] = mj_make_entry(MJ_TAG_UINT64, 0ul);
                    tape[ofs + 1u] = w;
                    return;
                }
            }
        }
        // Out-of-range integer literal: falls through to the double path
        // (reference type-selection rule 3).
    }

    // --- double path: significant-digit extraction -------------------------
    // value = (w + tail/10^dropped) * 10^exp10 with 0 <= tail < 10^dropped:
    // w holds the first <= 19 significant digits, `dropped` counts the
    // significant digits beyond them (each bumps the decimal exponent), and
    // a dropped NONZERO digit sets `truncated` (all-zero drops keep the
    // value exact — no fixup needed).
    ulong w = 0ul;
    uint sig = 0u;
    bool nonzero_seen = false;
    ulong dropped = 0ul;
    bool truncated = false;
    for (uint part = 0u; part < 2u; part += 1u) {
        ulong begin = part == 0u ? int_begin : frac_begin;
        ulong end = part == 0u ? int_end : frac_end;
        for (ulong i = begin; i < end; i += 1ul) {
            uint d = uint(input[i] - uchar('0'));
            if (!nonzero_seen) {
                if (d == 0u) {
                    continue; // leading zeros carry no significance
                }
                nonzero_seen = true;
            }
            if (sig < 19u) {
                w = w * 10ul + ulong(d);
                sig += 1u;
            } else {
                dropped += 1ul;
                truncated = truncated || d != 0u;
            }
        }
    }
    long exp10 = exp_val - long(frac_end - frac_begin) + long(dropped);
    ulong sign_bit = negative ? 0x8000000000000000ul : 0ul;

    if (w == 0ul || exp10 < long(MJ_POW5_MIN_EXP)) {
        // All-zero digits, or underflow past half the smallest subnormal
        // (a <= 19-digit mantissa at q <= -343 is < 2^-1075): signed zero,
        // ACCEPTED ("-1e-400" -> -0.0).
        tape[ofs] = mj_make_entry(MJ_TAG_DOUBLE, 0ul);
        tape[ofs + 1u] = sign_bit;
        return;
    }
    if (exp10 > long(MJ_POW5_MAX_EXP)) {
        // >= 10^309 with a nonzero mantissa: infinite, rejected
        // (InvalidNumber — simdjson parity).
        mj_k10_report(err_min_pos, pos);
        return;
    }

    int q = int(exp10);
    ulong bits = 0ul;
    uint status = mj_k10_eisel_lemire(w, q, negative, &bits);
    if (status == MJ_EL_OK && truncated) {
        // Truncated mantissa: the true value lies in (w, w+1) * 10^q. If
        // both endpoints round identically the answer is certain
        // (rounding is monotone); otherwise the kernel cannot decide.
        ulong bits_up = 0ul;
        uint status_up = mj_k10_eisel_lemire(w + 1ul, q, negative, &bits_up);
        if (status_up != MJ_EL_OK || bits_up != bits) {
            status = MJ_EL_HARD;
        }
    }
    if (status != MJ_EL_OK) {
        // Hard case: append the token index for the CPU re-parse and leave
        // a deterministic placeholder value word.
        uint slot = atomic_fetch_add_explicit(fixup_count, 1u, memory_order_relaxed);
        fixup_tokens[slot] = tok;
        tape[ofs] = mj_make_entry(MJ_TAG_DOUBLE, 0ul);
        tape[ofs + 1u] = 0ul;
        return;
    }
    tape[ofs] = mj_make_entry(MJ_TAG_DOUBLE, 0ul);
    tape[ofs + 1u] = bits;
}
