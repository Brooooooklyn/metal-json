// 02_classify.metal — K1 `classify_escape_utf8` + the escape-carry valve
// kernel `escape_carry_fixup`.
//
// K1, one thread per 64-byte input word (the spike-C grain), loads its word
// as 4x uint4 and produces:
//   - bm_quote[word]: the escape-resolved real-quote bitmap (simdjson
//     find_escaped, written in uint2 per the binding spike-B decision);
//   - bm_cand[word]:  candidates = structural operators | scalar starts,
//     computed PRE in-string masking (K3 masks them);
//   - escape_info[word]: the carries the thread used + valve flags;
//   - per-chunk real-quote popcount partials (simd_sum + one atomic add per
//     simdgroup) for the K2 spine scan;
//   - UTF-8 validation errors via 64-bit atomic_min on the header.
//
// The bit-exact spec is the CPU oracle `reference::stage1_classify`
// (src/reference/classify.rs). Its two sequential dependences are made
// parallel exactly as the plan prescribes:
//
//   1. Escape state (is the word's first byte escaped?) = parity of the
//      backslash run ending right before the word, found by peeking
//      backward at the RAW INPUT, capped at MJ_ESCAPE_LOOKBACK_CAP bytes.
//      A run still going at the cap sets a flag; `escape_carry_fixup`
//      (below) repairs those words before K2/K3 consume the bitmaps.
//   2. Scalar-start state (may a scalar run start at the word's first
//      byte?) = the classification of the previous raw byte; when that byte
//      is a quote, its own escapedness comes from the same capped run scan.
//
// UTF-8 validation is two-tier, entered only for words containing
// non-ASCII bytes: a branch-light bitmask fast path (mj_utf8_fast_word)
// PROVES the well-formed common case in registers, and only inconclusive
// words run the per-word scalar walk of the reference's Lemire-style
// table (mj_utf8_validate_word) — the single source of error offsets. A
// 3-byte look-back (from the neighbor SIMD lane's registers when
// available, raw loads otherwise) tells the thread how many leading bytes
// belong to a sequence opened in the previous word (that word's thread
// validates it). Because every walking thread's determination is exact
// whenever all bytes before its word are valid, the atomic_min over all
// reported offsets equals the reference's first-error offset.
//
// `escape_carry_fixup` is THE VALVE for adversarial backslash walls
// (>= 4096 consecutive backslashes). It runs in CB1 between K1 and K2:
//   - benign path: every thread reads header->carry_overflow (uniform) and
//     returns immediately — one near-empty dispatch, no sync;
//   - valve path: for each flagged word, the true carry is recovered by
//     walking the flag chain backward in 64-word (= one cap) strides — a
//     capped scan proves the 4096 bytes before the word are all
//     backslashes, and an even-length all-backslash gap preserves escape
//     parity, so the word's state equals the state of the word one cap
//     earlier; the first unflagged ancestor's stored carry is exact. Words
//     whose guessed carry was wrong are reclassified with the shared
//     per-word routine and the chunk quote partial is patched atomically.
//   Cost is O(chain length) loads per flagged word (quadratic in the wall
//   length across threads, in parallel) — pathological inputs only; benign
//   inputs never pay more than the empty dispatch.

#include "common.h"
#include "bitmap_u2.h"

// --- byte-class bitmaps from vectorized loads ---------------------------------

// 0x80-flag per byte of `v` that equals the replicated byte `pat`.
//
// Exact zero-byte detect on x = v ^ pat: `(x & 0x7F7F7F7F) + 0x7F7F7F7F`
// sets a byte's high bit iff its low 7 bits are nonzero (per-byte sums max
// out at 0xFE, so nothing carries across lanes); OR-ing x back in covers
// bytes whose own high bit is set; inverting isolates the bytes that are
// exactly zero, i.e. equal to pat.
//
// Deliberately NOT the classic `(x - 0x01010101) & ~x & 0x80808080` trick:
// its subtraction borrows ACROSS byte lanes, so a byte equal to pat ^ 0x01
// right after a matching byte is falsely flagged (and the borrow ripples
// through runs). Real JSON hits that constantly: `]\` would flag the
// backslash as `]`, `"#` would flag `#` as a quote, `:;` `,-` `{z` `}|`
// `[Z` and space-`!` likewise. Found by the M2 differential fuzz suite
// (tests/kernels.rs `bit_trick_neighbor_bytes_match_the_reference` pins it).
static inline uint mj_zmask(uint v, uint pat) {
    uint x = v ^ pat;
    return ~(((x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | x) & 0x80808080u;
}

// Compress 0x80-per-byte flags (bits 7,15,23,31) to a 4-bit mask (bits 0-3).
static inline uint mj_compress80(uint m) {
    return ((m >> 7) & 1u) | ((m >> 14) & 2u) | ((m >> 21) & 4u) | ((m >> 28) & 8u);
}

// 16-bit equality mask of the 16 bytes in `u` against a replicated byte.
static inline uint mj_eq16(uint4 u, uint pat) {
    return mj_compress80(mj_zmask(u.x, pat))
        | (mj_compress80(mj_zmask(u.y, pat)) << 4)
        | (mj_compress80(mj_zmask(u.z, pat)) << 8)
        | (mj_compress80(mj_zmask(u.w, pat)) << 12);
}

// Structural operators: { } [ ] : ,
static inline uint mj_op4(uint v) {
    uint m = mj_zmask(v, 0x7B7B7B7Bu) | mj_zmask(v, 0x7D7D7D7Du)
           | mj_zmask(v, 0x5B5B5B5Bu) | mj_zmask(v, 0x5D5D5D5Du)
           | mj_zmask(v, 0x3A3A3A3Au) | mj_zmask(v, 0x2C2C2C2Cu);
    return mj_compress80(m);
}

static inline uint mj_op16(uint4 u) {
    return mj_op4(u.x) | (mj_op4(u.y) << 4) | (mj_op4(u.z) << 8) | (mj_op4(u.w) << 12);
}

// JSON insignificant whitespace: space, tab, \n, \r.
static inline uint mj_ws4(uint v) {
    uint m = mj_zmask(v, 0x20202020u) | mj_zmask(v, 0x09090909u)
           | mj_zmask(v, 0x0A0A0A0Au) | mj_zmask(v, 0x0D0D0D0Du);
    return mj_compress80(m);
}

static inline uint mj_ws16(uint4 u) {
    return mj_ws4(u.x) | (mj_ws4(u.y) << 4) | (mj_ws4(u.z) << 8) | (mj_ws4(u.w) << 12);
}

// Non-ASCII bytes (high bit set).
static inline uint mj_hi16(uint4 u) {
    return mj_compress80(u.x & 0x80808080u)
        | (mj_compress80(u.y & 0x80808080u) << 4)
        | (mj_compress80(u.z & 0x80808080u) << 8)
        | (mj_compress80(u.w & 0x80808080u) << 12);
}

// Raw per-word byte-class bitmaps, plus the valid-bit mask for the tail word.
struct MjWordClasses {
    uint2 backslash;
    uint2 quote; // raw `"` bytes, escapes NOT yet resolved
    uint2 op;
    uint2 ws;
    uint2 non_ascii;
    uint2 valid; // bits whose byte index is < input_len
};

// Vectorized classification of one 64-byte word. The input buffer is padded
// to a whole number of words with ASCII spaces (Stage1Buffers contract), so
// full uint4 loads never read past the buffer; the valid mask additionally
// clamps every output bit to input_len (defense in depth — the reference
// guarantees bits at positions >= input_len are zero).
// The raw 64 input bytes of one word as 4x uint4 (16 little-endian uints).
struct MjWordRaw {
    uint4 u0;
    uint4 u1;
    uint4 u2;
    uint4 u3;
};

static inline MjWordRaw mj_load_raw(device const uchar* input, ulong word) {
    device const uint4* in16 = (device const uint4*)input;
    MjWordRaw r;
    r.u0 = in16[word * 4 + 0];
    r.u1 = in16[word * 4 + 1];
    r.u2 = in16[word * 4 + 2];
    r.u3 = in16[word * 4 + 3];
    return r;
}

static inline MjWordClasses mj_classes_from_raw(
    thread const MjWordRaw& r,
    ulong word,
    ulong input_len)
{
    uint4 u0 = r.u0;
    uint4 u1 = r.u1;
    uint4 u2 = r.u2;
    uint4 u3 = r.u3;

    MjWordClasses c;
    c.backslash = make_u2(mj_eq16(u0, 0x5C5C5C5Cu) | (mj_eq16(u1, 0x5C5C5C5Cu) << 16),
                          mj_eq16(u2, 0x5C5C5C5Cu) | (mj_eq16(u3, 0x5C5C5C5Cu) << 16));
    c.quote = make_u2(mj_eq16(u0, 0x22222222u) | (mj_eq16(u1, 0x22222222u) << 16),
                      mj_eq16(u2, 0x22222222u) | (mj_eq16(u3, 0x22222222u) << 16));
    c.op = make_u2(mj_op16(u0) | (mj_op16(u1) << 16), mj_op16(u2) | (mj_op16(u3) << 16));
    c.ws = make_u2(mj_ws16(u0) | (mj_ws16(u1) << 16), mj_ws16(u2) | (mj_ws16(u3) << 16));
    c.non_ascii = make_u2(mj_hi16(u0) | (mj_hi16(u1) << 16), mj_hi16(u2) | (mj_hi16(u3) << 16));

    ulong start = word * ulong(MJ_WORD_BYTES);
    c.valid = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    if (input_len < start + ulong(MJ_WORD_BYTES)) {
        // 1..63 valid bytes in this word (words = ceil(len/64) keeps >= 1).
        c.valid = shr64_u2(c.valid, MJ_WORD_BYTES - uint(input_len - start));
    }
    c.backslash &= c.valid;
    c.quote &= c.valid;
    c.op &= c.valid;
    c.non_ascii &= c.valid;
    // ws is deliberately NOT masked: the space padding must keep counting as
    // whitespace so padded bytes can never look like scalar starts.
    return c;
}

// Load + classify in one call (the escape_carry_fixup path, which has no
// use for the raw vectors afterwards).
static inline MjWordClasses mj_load_classes(
    device const uchar* input,
    ulong word,
    ulong input_len)
{
    MjWordRaw r = mj_load_raw(input, word);
    return mj_classes_from_raw(r, word, input_len);
}

// --- escape resolution + candidates (shared with the fix-up kernel) -----------

// Resolve escapes and scalar starts for one word, given the two carries.
// Bit-exact uint2 port of simdjson find_escaped + the reference's
// prev_allows_start rule (see reference::stage1_classify).
static inline void mj_classify_word(
    thread const MjWordClasses& c,
    uint prev_escaped,
    uint prev_allows,
    thread uint2& quote_real,
    thread uint2& candidates)
{
    const uint2 EVEN_BITS = uint2(0x55555555u, 0x55555555u);
    const uint2 ODD_BITS = uint2(0xAAAAAAAAu, 0xAAAAAAAAu);

    // simdjson find_escaped: a backslash that is itself escaped starts
    // nothing; odd-position run starts + the run collapse, via a 64-bit add
    // with carries crossing the 32-bit seam (add64_u2), into a mask of
    // every odd-offset byte of each run — the escaped bytes.
    uint2 bs = c.backslash;
    bs.x &= ~prev_escaped;
    uint2 follows_escape = shl64_u2(bs, 1u);
    follows_escape.x |= prev_escaped;
    uint2 odd_starts = bs & ODD_BITS & ~follows_escape;
    uint carry_out; // unused: the next word derives its carry from raw look-back
    uint2 seq_even = add64_u2(odd_starts, bs, carry_out);
    uint2 invert_mask = shl64_u2(seq_even, 1u);
    uint2 escaped = (EVEN_BITS ^ invert_mask) & follows_escape;

    quote_real = c.quote & ~escaped;

    // candidates = operators | scalar starts. A byte is scalar-class when it
    // is not a real quote / operator / whitespace (escaped quotes and
    // backslashes included); it STARTS a run when the previous byte is
    // whitespace, an operator, or a real quote.
    uint2 scalar = ~quote_real & ~c.op & ~c.ws;
    uint2 allows = shl64_u2(c.ws | c.op | quote_real, 1u);
    allows.x |= prev_allows;
    candidates = (c.op | (scalar & allows)) & c.valid;
}

// Length of the backslash run ending at byte `end - 1`, scanning backward at
// most `MJ_ESCAPE_LOOKBACK_CAP` bytes and never before byte 0. `capped` is
// set when the run is still going at the cap (parity unknown — valve case).
static inline uint mj_backslash_run(
    device const uchar* input,
    ulong end,
    thread bool& capped)
{
    ulong lo = end > ulong(MJ_ESCAPE_LOOKBACK_CAP) ? end - ulong(MJ_ESCAPE_LOOKBACK_CAP) : 0;
    ulong r = end;
    while (r > lo && input[r - 1] == uchar(0x5C)) {
        r -= 1;
    }
    capped = (r == lo) && (lo != 0);
    return uint(end - r);
}

// (mj_is_ws_byte / mj_is_op_byte / mj_load_padded live in common.h: K6's
// literal checks share them, and the runtime-shaders path compiles all
// units as one translation unit, so they must have a single definition.)

// Compute the two carries for `word` by peeking backward at the raw input.
// Returns the escape_info record (carry bits + valve flags).
static inline uint mj_word_carries(
    device const uchar* input,
    ulong word,
    thread uint& prev_escaped,
    thread uint& prev_allows)
{
    if (word == 0) {
        prev_escaped = 0u;
        prev_allows = 1u; // start of input allows a scalar start
        return MJ_CARRY_PREV_ALLOWS;
    }
    ulong p = word * ulong(MJ_WORD_BYTES);
    uchar b = input[p - 1];
    uint info = 0u;
    bool capped = false;
    if (b == uchar(0x5C)) {
        // Run ending at p-1 decides whether byte p is escaped; a backslash
        // is scalar-class, so no scalar run can start at p.
        uint run = mj_backslash_run(input, p, capped);
        prev_escaped = capped ? 0u : (run & 1u); // capped: guess even
        prev_allows = 0u;
        if (capped) {
            info |= MJ_CARRY_FLAG_ESCAPE;
        }
    } else if (b == uchar('"')) {
        // A REAL closing/opening quote allows a scalar start; an escaped
        // quote is scalar-class and does not. Its escapedness is the parity
        // of the backslash run ending at p-2.
        prev_escaped = 0u;
        uint run = mj_backslash_run(input, p - 1, capped);
        prev_allows = capped ? 1u : ((run & 1u) ^ 1u); // capped: guess even => real quote
        if (capped) {
            info |= MJ_CARRY_FLAG_QUOTE;
        }
    } else {
        prev_escaped = 0u;
        prev_allows = (mj_is_ws_byte(b) || mj_is_op_byte(b)) ? 1u : 0u;
    }
    return info | prev_escaped | (prev_allows << 1);
}

// --- UTF-8 validation ----------------------------------------------------------

// Total sequence length implied by a lead-range byte (>= 0xC0); meaningful
// only for deciding how far a possibly-invalid lead reaches.
static inline uint mj_utf8_seq_len(uchar b) {
    if (b < uchar(0xE0)) {
        return 2u;
    }
    if (b < uchar(0xF0)) {
        return 3u;
    }
    return 4u;
}

// The reference's lead-byte table: continuation count + allowed range for
// the SECOND byte (which encodes overlong / surrogate / > U+10FFFF checks).
// Returns false for bytes that can never lead a sequence.
static inline bool mj_utf8_lead(uchar b, thread uint& cont, thread uchar& lo, thread uchar& hi) {
    if (b >= uchar(0xC2) && b <= uchar(0xDF)) {
        cont = 1u; lo = uchar(0x80); hi = uchar(0xBF); return true;
    }
    if (b == uchar(0xE0)) {
        cont = 2u; lo = uchar(0xA0); hi = uchar(0xBF); return true; // reject overlong 3-byte
    }
    if ((b >= uchar(0xE1) && b <= uchar(0xEC)) || b == uchar(0xEE) || b == uchar(0xEF)) {
        cont = 2u; lo = uchar(0x80); hi = uchar(0xBF); return true;
    }
    if (b == uchar(0xED)) {
        cont = 2u; lo = uchar(0x80); hi = uchar(0x9F); return true; // reject surrogates
    }
    if (b == uchar(0xF0)) {
        cont = 3u; lo = uchar(0x90); hi = uchar(0xBF); return true; // reject overlong 4-byte
    }
    if (b >= uchar(0xF1) && b <= uchar(0xF3)) {
        cont = 3u; lo = uchar(0x80); hi = uchar(0xBF); return true;
    }
    if (b == uchar(0xF4)) {
        cont = 3u; lo = uchar(0x80); hi = uchar(0x8F); return true; // reject > U+10FFFF
    }
    // 0x80..=0xC1 stray continuation / overlong 2-byte; 0xF5..=0xFF too big.
    return false;
}

// How many leading bytes of the word at byte offset `p` continue a UTF-8
// sequence opened in the previous word (that word's thread validates
// them)? A raw look-back of at most 3 bytes; result 0..3.
static inline uint mj_utf8_incoming_skip(device const uchar* input, ulong p) {
    uint skip = 0u;
    if (p > 0) {
        uchar b1 = input[p - 1];
        if (b1 >= uchar(0xC0)) {
            skip = mj_utf8_seq_len(b1) - 1u;
        } else if (p > 1 && b1 >= uchar(0x80)) {
            uchar b2 = input[p - 2];
            if (b2 >= uchar(0xE0)) {
                skip = mj_utf8_seq_len(b2) - 2u;
            } else if (p > 2 && b2 >= uchar(0x80) && b2 < uchar(0xC0)) {
                uchar b3 = input[p - 3];
                if (b3 >= uchar(0xF0)) {
                    skip = 1u; // 4-byte lead at p-3 covers byte p
                }
            }
        }
    }
    return skip;
}

// mj_utf8_incoming_skip with the previous word's last 4 bytes already in a
// register (little-endian uint, so byte p-1 is the top byte) — the K1 fast
// path gets them from the neighboring SIMD lane instead of raw loads. The
// caller guarantees the previous word exists, hence p >= 64 > 3 and the
// p > 1 / p > 2 guards of the load version hold vacuously.
static inline uint mj_utf8_skip_from_prev4(uint prev4) {
    uchar b1 = uchar(prev4 >> 24);
    if (b1 >= uchar(0xC0)) {
        return mj_utf8_seq_len(b1) - 1u;
    }
    if (b1 >= uchar(0x80)) {
        uchar b2 = uchar((prev4 >> 16) & 0xFFu);
        if (b2 >= uchar(0xE0)) {
            return mj_utf8_seq_len(b2) - 2u;
        }
        if (b2 >= uchar(0x80) && b2 < uchar(0xC0)) {
            uchar b3 = uchar((prev4 >> 8) & 0xFFu);
            if (b3 >= uchar(0xF0)) {
                return 1u; // 4-byte lead at p-3 covers byte p
            }
        }
    }
    return 0u;
}

// Validate every UTF-8 sequence that STARTS in `word`. The incoming skip
// excludes leading bytes that continue the previous word's sequence (that
// word's thread validates them). Sequences may extend past the word end;
// their tail bytes are read raw. Errors report the offset of the first
// byte of the first invalid sequence — identical to the reference.
static inline void mj_utf8_validate_word(
    device const uchar* input,
    ulong word,
    ulong padded_len,
    device MjHeaderDev* header)
{
    ulong p = word * ulong(MJ_WORD_BYTES);
    uint skip = mj_utf8_incoming_skip(input, p);

    ulong i = p + skip;
    ulong end = p + ulong(MJ_WORD_BYTES);
    while (i < end) {
        uchar b = input[i];
        if (b < uchar(0x80)) {
            i += 1;
            continue;
        }
        uint cont;
        uchar lo, hi;
        if (!mj_utf8_lead(b, cont, lo, hi)) {
            mj_report_utf8(header, i);
            return;
        }
        uchar second = mj_load_padded(input, i + 1, padded_len);
        if (second < lo || second > hi) {
            mj_report_utf8(header, i);
            return;
        }
        bool ok = true;
        for (uint k = 2u; k <= cont; ++k) {
            uchar ck = mj_load_padded(input, i + k, padded_len);
            ok = ok && ck >= uchar(0x80) && ck <= uchar(0xBF);
        }
        if (!ok) {
            mj_report_utf8(header, i);
            return;
        }
        i += ulong(cont) + 1;
    }
}

// --- UTF-8 bitmask fast path -----------------------------------------------------
//
// The scalar walk above costs ~64 dependent raw loads + branches per
// non-ASCII word; on CJK-heavy inputs it dominated K1 (75% of the kernel
// on twitter, where 30% of words contain non-ASCII bytes). This fast path
// proves the common case — every sequence starting in the word is
// well-formed and uses no special-cased lead — with pure register math on
// the already-loaded vectors, and returns false ("inconclusive") for
// everything else, sending the word to the scalar walk. The walk stays
// the single source of error offsets, so error reporting is bit-exact by
// construction; the proof obligation here is one-sided:
//
//                  FAST PASS  =>  THE WALK ACCEPTS THE WORD.
//
// Per 4-byte lane (one little-endian uint v, flags on bit 7 of each byte):
//   cont    = 10xxxxxx                  (continuation bytes)
//   lead    = 11xxxxxx (C0..FF)         (any lead)
//   lead34  = 111xxxxx (E0..FF)         lead4 = 1111xxxx (F0..FF)
//   special = leads the walk's table treats specially: C0/C1 (overlong
//             2-byte) and F5..FF (too big) are invalid outright; E0
//             (overlong 3-byte), ED (surrogates), F0 (overlong 4-byte) and
//             F4 (> U+10FFFF) restrict their second byte beyond the plain
//             80..BF continuation range. The only non-special members of
//             lead4 are F1/F2/F3; of the E-row, E0 and ED.
//
// Structural check: a lead at byte i requires continuations at exactly
// the next seqlen-1 bytes, so with byte-granular shifts carried across
// lanes
//     expected = lead << 1B  |  lead34 << 2B  |  lead4 << 3B
// the word is structurally valid iff expected == cont everywhere:
//   - stray continuation (no covering lead): cont 1, expected 0;
//   - truncated sequence (lead followed by a non-cont): expected 1 at the
//     missing position, cont 0 (a following lead/ASCII is never cont);
//   - nested lead (a lead inside another's span): that position has
//     expected 1 but is a lead byte, so cont 0.
// Equality therefore reproduces the walk's greedy left-to-right
// consumption; a standard lead's 80..BF second-byte rule IS the
// continuation-placement check, and special leads are excluded up front.
//
// Word boundaries:
//   - Incoming: the first `skip` bytes continue the previous word's
//     sequence. The walk starts at p + skip and never inspects them (the
//     previous word's thread validates them as its spill), so ALL class
//     flags below skip are cleared — cont so the equality ignores those
//     bytes, lead/special so a lead the walk would never process cannot
//     vouch for (or fail on) later continuations. On valid input the
//     skipped bytes are continuations and carry no lead flags anyway; on
//     mangled input clearing can only force a mismatch -> walk, never a
//     false accept: any in-word continuation the cleared lead "covered"
//     now has expected 0 -> mismatch.
//   - Outgoing: lead flags byte-shifted past byte 63 accumulate in
//     `carry` (expected continuations at bytes 64..66). They are checked
//     against the next word's first bytes; a truncated final sequence
//     fails against the space padding exactly like the walk's padded
//     probes.
//
// The boundary bytes come from the neighboring SIMD lanes: thread `word`
// already holds word-1's last uint (r.u3.w) and word+1's first uint
// (r.u0.x) in registers, so simd_shuffle_up/down replaces the raw
// look-back/probe loads. Only lane 0 (no previous lane), the last lane,
// and the grid edges fall back to raw loads. The kernel computes the
// shuffles unconditionally right after the vector loads — the `word <
// words` guard keeps the active lanes a prefix of each simdgroup, so
// every active lane's up/down neighbor needed here is active too.
static inline bool mj_utf8_fast_word(
    device const uchar* input,
    thread const MjWordRaw& r,
    ulong word,
    ulong padded_len,
    uint prev4,
    bool prev4_valid,
    uint next4,
    bool next4_valid)
{
    ulong p = word * ulong(MJ_WORD_BYTES);
    uint skip = prev4_valid ? mj_utf8_skip_from_prev4(prev4)
                            : mj_utf8_incoming_skip(input, p);

    uint carry = 0u;       // expected-cont flags entering the next lane
    uint mismatch = 0u;    // OR of (expected ^ cont) across all lanes
    uint special_any = 0u; // OR of special-lead flags across all lanes

    // One 4-byte lane; `keep` is a compile-time all-ones for every lane
    // except lane 0, where it clears the incoming-skip bytes (the
    // constant folds away everywhere else). Special detection works on
    // shared shifts: bit b of byte k lands on bit 7 of byte k when v is
    // shifted left by 7 - b.
    #define MJ_UTF8_LANE(v_expr, keep)                                            \
        {                                                                         \
            uint v = (v_expr);                                                    \
            uint hi = v & 0x80808080u;                                            \
            uint cont = (hi & ~(v << 1)) & (keep);                                \
            uint lead = (hi & (v << 1)) & (keep);                                 \
            uint lead34 = lead & (v << 2);                                        \
            uint lead4 = lead34 & (v << 3);                                       \
            uint expected = (lead << 8) | (lead34 << 16) | (lead4 << 24) | carry; \
            mismatch |= expected ^ cont;                                          \
            carry = (lead >> 24) | (lead34 >> 16) | (lead4 >> 8);                 \
            uint or41 = (v << 3) | (v << 4) | (v << 5) | (v << 6); /* bits 4..1 */\
            uint specialC = (lead ^ lead34) & ~or41;          /* C0/C1     */     \
            uint specialE0 = (lead34 ^ lead4) & ~(or41 | (v << 7)); /* E0  */     \
            uint specialF = lead4 & ((v << 4) | (v << 5) | ~((v << 6) | (v << 7)));\
            special_any |= (specialC | specialE0 | specialF                       \
                            | mj_zmask(v, 0xEDEDEDEDu)) & (keep);                 \
        }

    uint keep0 = skip == 0u ? 0xFFFFFFFFu : (0xFFFFFFFFu << (8u * skip));
    MJ_UTF8_LANE(r.u0.x, keep0)
    MJ_UTF8_LANE(r.u0.y, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u0.z, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u0.w, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u1.x, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u1.y, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u1.z, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u1.w, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u2.x, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u2.y, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u2.z, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u2.w, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u3.x, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u3.y, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u3.z, 0xFFFFFFFFu)
    MJ_UTF8_LANE(r.u3.w, 0xFFFFFFFFu)
    #undef MJ_UTF8_LANE

    if ((mismatch | special_any) != 0u) {
        return false; // structural break, or a special lead: walk decides
    }

    // Sequences spilling past byte 63: their continuations must open the
    // next word (bit 7 of carry byte k = expected cont at byte 64 + k).
    if (carry != 0u) {
        if (next4_valid) {
            uint cont_next = (next4 & 0x80808080u) & ~(next4 << 1);
            return (carry & cont_next) == carry;
        }
        while (carry != 0u) { // grid edge: probe raw, padded like the walk
            uint k = ctz(carry) >> 3;
            carry &= carry - 1u;
            uchar b =
                mj_load_padded(input, p + ulong(MJ_WORD_BYTES) + ulong(k), padded_len);
            if ((b & uchar(0xC0)) != uchar(0x80)) {
                return false;
            }
        }
    }
    return true;
}

// --- K1 -------------------------------------------------------------------------

// One thread per 64-byte word; dispatched as FULL threadgroups
// (Dispatch::Threadgroups) so the simd_sum below is convergent — bounds are
// checked per thread, never by early return.
kernel void classify_escape_utf8(
    device const uchar* input [[buffer(0)]],
    device uint2* bm_quote [[buffer(1)]],
    device uint2* bm_cand [[buffer(2)]],
    device uchar* escape_info [[buffer(3)]],
    device atomic_uint* chunk_quote_counts [[buffer(4)]],
    device MjHeaderDev* header [[buffer(5)]],
    constant MjParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    ulong words = params.element_count;
    ulong word = ulong(gid);
    uint quote_count = 0u;

    if (word < words) {
        ulong padded_len = words * ulong(MJ_WORD_BYTES);
        MjWordRaw raw = mj_load_raw(input, word);

        // Boundary bytes from the neighbor lanes (see mj_utf8_fast_word):
        // word - 1's last 4 bytes and word + 1's first 4. Computed before
        // any further divergence — `word < words` keeps the active lanes a
        // prefix of the simdgroup, so the source lane of every shuffle
        // whose validity flag is true is active.
        uint prev4 = simd_shuffle_up(raw.u3.w, 1u);
        bool prev4_valid = simd_lane != 0u;
        uint next4 = simd_shuffle_down(raw.u0.x, 1u);
        bool next4_valid = simd_lane + 1u < simd_size && word + 1u < words;

        MjWordClasses classes = mj_classes_from_raw(raw, word, params.input_len);

        uint prev_escaped, prev_allows;
        uint info = mj_word_carries(input, word, prev_escaped, prev_allows);
        escape_info[word] = uchar(info);
        if ((info & (MJ_CARRY_FLAG_ESCAPE | MJ_CARRY_FLAG_QUOTE)) != 0u) {
            atomic_fetch_add_explicit(&header->carry_overflow_lo, 1u, memory_order_relaxed);
        }

        uint2 quote_real, candidates;
        mj_classify_word(classes, prev_escaped, prev_allows, quote_real, candidates);
        bm_quote[word] = quote_real;
        bm_cand[word] = candidates;
        quote_count = popcount64_u2(quote_real);

        // ASCII fast path: only words containing non-ASCII bytes are
        // validated at all (a word with no high bit can neither start nor
        // continue a multi-byte sequence in a way IT must report). Of
        // those, the bitmask fast path proves the well-formed common case
        // in registers; only inconclusive words pay the scalar walk, the
        // single (reference-exact) source of error offsets.
        if ((classes.non_ascii.x | classes.non_ascii.y) != 0u
            && !mj_utf8_fast_word(input, raw, word, padded_len, prev4, prev4_valid,
                                  next4, next4_valid)) {
            mj_utf8_validate_word(input, word, padded_len, header);
        }
    }

    // Per-chunk quote partial: one simdgroup spans 32 consecutive words,
    // always inside a single 1024-word chunk, so one atomic add per
    // simdgroup accumulates the chunk popcount (buffer is zero-initialized).
    uint simd_total = simd_sum(quote_count);
    if (simd_lane == 0u && simd_total != 0u) {
        atomic_fetch_add_explicit(&chunk_quote_counts[gid / MJ_CHUNK_WORDS], simd_total,
                                  memory_order_relaxed);
    }
}

// --- escape-carry fix-up (the valve) ---------------------------------------------

// One thread per 4 words (a 256-thread group covers one 1024-word chunk).
// See the file header for the algorithm; runs between K1 and K2 in CB1.
kernel void escape_carry_fixup(
    device const uchar* input [[buffer(0)]],
    device const uchar* escape_info [[buffer(1)]],
    device uint2* bm_quote [[buffer(2)]],
    device uint2* bm_cand [[buffer(3)]],
    device atomic_uint* chunk_quote_counts [[buffer(4)]],
    device MjHeaderDev* header [[buffer(5)]],
    constant MjParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    // Benign fast path: uniform load, whole grid returns at once.
    if (atomic_load_explicit(&header->carry_overflow_lo, memory_order_relaxed) == 0u) {
        return;
    }

    ulong words = params.element_count;
    const uint cap_words = MJ_ESCAPE_LOOKBACK_CAP / MJ_WORD_BYTES; // 64

    for (uint j = 0u; j < 4u; ++j) {
        ulong word = ulong(gid) * 4 + ulong(j);
        if (word >= words) {
            break;
        }
        uint info = uint(escape_info[word]);
        if ((info & (MJ_CARRY_FLAG_ESCAPE | MJ_CARRY_FLAG_QUOTE)) == 0u) {
            continue;
        }

        // A flagged word has (at least) one full cap of backslashes ending
        // just before it, so its state equals the state one cap earlier;
        // chase flagged ancestors until a word whose stored carry is exact.
        // Flagging requires word > cap_words, so the walk never underflows.
        ulong q = word - ulong(cap_words);
        while ((uint(escape_info[q]) & MJ_CARRY_FLAG_ESCAPE) != 0u) {
            q -= ulong(cap_words);
        }
        uint anchor = uint(escape_info[q]) & MJ_CARRY_PREV_ESCAPED;

        uint true_escaped, true_allows;
        if ((info & MJ_CARRY_FLAG_ESCAPE) != 0u) {
            // Byte before the word is a backslash: never allows a start;
            // an even-length all-backslash gap preserves escape parity.
            true_escaped = anchor;
            true_allows = 0u;
        } else {
            // Byte before the word is a quote; the run before it is one
            // byte short of a whole cap (odd gap), so the quote's
            // escapedness is the anchor parity FLIPPED — and the quote is
            // real (allows a start) exactly when it is NOT escaped, i.e.
            // when the anchor parity is set.
            true_escaped = 0u;
            true_allows = anchor;
        }

        uint used_escaped = info & MJ_CARRY_PREV_ESCAPED;
        uint used_allows = (info >> 1) & 1u;
        if (true_escaped == used_escaped && true_allows == used_allows) {
            continue; // the guess was right; nothing to repair
        }

        MjWordClasses classes = mj_load_classes(input, word, params.input_len);
        uint2 quote_real, candidates;
        mj_classify_word(classes, true_escaped, true_allows, quote_real, candidates);
        uint old_count = popcount64_u2(bm_quote[word]);
        bm_quote[word] = quote_real;
        bm_cand[word] = candidates;
        uint new_count = popcount64_u2(quote_real);
        if (new_count != old_count) {
            // Patch the chunk partial; wrapping uint add handles negatives.
            atomic_fetch_add_explicit(&chunk_quote_counts[uint(word) / MJ_CHUNK_WORDS],
                                      new_count - old_count, memory_order_relaxed);
        }
    }
}
