// 10_pair_ctx.metal — K9: adjacent bracket pairing, segmented forward-fill
// of the enclosing opener, Layer-2 separator-context checks and per-opener
// child counts over the depth-sorted skeleton (reference stage 4 phases
// 3-5), plus the CB3 error fold into the header.
//
//   ctx_partials        1 threadgroup / 1024-element chunk of the SORTED
//                       order: per-chunk segmented-scan summary
//   ctx_spine           1 threadgroup: combine the summaries into the
//                       exclusive walk state entering each chunk
//   pair_ctx_apply      per sorted element: replay the reference group
//                       walk against the carried state — pair map, opener
//                       context, child counts, Layer-2 errors; min-folds
//                       its error candidates into chunk_error (on top of
//                       the depth scan's)
//   structure_finalize  1 threadgroup: min-fold chunk_error into
//                       header.error (single writer)
//
// The bit-exact spec is the group walk of `reference::stage4_structure`
// (src/reference/structure.rs). The walk's whole sequential state at any
// element reduces to three segmented quantities — the comma count, the
// colon count, and the LATEST BRACKET since the start of the depth group
// (the reference's `pending_open` is "latest bracket, if it is an open";
// after a close the latest bracket is that close, which is exactly
// pending == None) — so it carries across chunks with the standard
// reduce → spine → apply shape over the MjCtxState monoid below:
//
//   - close pairing is ADJACENT pairing: within a group, brackets strictly
//     alternate in document order (guaranteed by the sort's stability), so
//     a close's partner is the latest bracket, when that bracket is an
//     open. The open/close type check is `open ^ close == 0x06`.
//   - a close whose latest bracket is missing or a close, an open whose
//     latest bracket is an open (two-opens), and a group whose LAST
//     bracket is an open (unclosed container — checked by the group's last
//     element) are MJ_ERR_UNBALANCED, at the offsets the reference uses
//     (the close, the second open, and the unclosed open respectively).
//   - a separator whose latest bracket is missing or a close sits outside
//     any container: MJ_ERR_TRAILING_CONTENT (`1,2`, `{},{}`).
//   - a colon whose opener is `[` is MJ_ERR_UNEXPECTED_TOKEN (`[1,"a":2]`).
//   - a comma whose opener is `{` must be followed, within the group, by
//     the member's colon exactly 3 tokens later: MJ_ERR_MISSING_COLON
//     (`{"a":1,2}`, `{"foo":"bar", "a"}`).
//   - child counts per opener come from comma/colon-rank DIFFERENCES
//     between the open and its close (objects: colons; arrays: commas + 1,
//     or 0 when the close token immediately follows the open token). Rank
//     differences are exactly the reference's reset-at-open counters.
//     Counts are stored full-width; saturation is K12's job.
//
// Group boundaries need no histogram table: element j starts a group iff
// j == 0 or its mj_sort_key differs from sorted neighbor j-1's.
//
// OVERFLOW ELEMENTS ARE INERT: an element whose depth exceeds max_depth
// (mj_depth_overflows) shares the key_max sort key with the legal
// max_depth group (mj_sort_key clamps — the key range must not grow, see
// common.h), but it must not participate in that group's walk: it would
// pair closes with the wrong opens and suppress leftover-open errors the
// reference reports at earlier offsets (`[[1]` at max_depth=1). So every
// walk transition and rule evaluation here skips overflow elements; the
// boundary reset still applies (boundaries depend only on sort keys), and
// the group-last leftover check still runs on them (with the walk state
// the legal elements produced). The full first-error-preservation proof
// lives on mj_sort_key in common.h.
//
// Write coverage (why no buffer here needs pre-zeroing): every NON-inert
// element writes its own context_opener; separators and closes write
// their own match_index / child_counts; a close writes its open's
// match_index and child_counts. On clean inputs no element is inert and
// every open is paired, so all entries are written; unwritten entries can
// only exist on inputs this stage (or an earlier one) rejects — inert
// elements imply a DepthLimit rejection — whose CB3 outputs are never
// read (rejection contract).
//
// All kernels are dispatched as FULL 256-thread threadgroups.

#include "common.h"
#include "tg_scan.h"

// "no bracket since the group started" marker (also the NO_MATCH value of
// the match_index output — mirrors `reference::structure::NO_MATCH`).
constant constexpr uint MJ_CTX_NONE = 0xFFFFFFFFu;

// Segmented walk state / range summary. As a SUMMARY of a range, counts
// and bracket ranks are relative to the latest group boundary inside the
// range (or the range start if it has none) and bit 0 of `flags` records
// whether the range contains a boundary; as a carried STATE entering an
// element, they are relative to the group start. 8 x u32 = 32 bytes,
// layout-mirrored by `stage::Stage3Buffers::chunk_ctx` ("CTX_STATE_BYTES").
struct MjCtxState {
    uint flags;     // bit 0: range contains a group boundary (summaries)
    uint commas;    // commas since segment start
    uint colons;    // colons since segment start
    uint br_skel;   // skeleton index of the latest bracket, or MJ_CTX_NONE
    uint br_byte;   // its byte ({ } [ ]), valid when br_skel != MJ_CTX_NONE
    uint br_token;  // its stage-2 token index
    uint br_commas; // comma count AT that bracket (segment-relative)
    uint br_colons; // colon count AT that bracket (segment-relative)
};

static inline MjCtxState mj_ctx_identity() {
    return MjCtxState{ 0u, 0u, 0u, MJ_CTX_NONE, 0u, 0u, 0u, 0u };
}

// Monoid combine: `a` then `b` over consecutive ranges. If `b` contains a
// boundary its tail state is already absolute; otherwise counts add and
// `b`'s bracket ranks shift by `a`'s counts.
static inline MjCtxState mj_ctx_combine(MjCtxState a, MjCtxState b) {
    if ((b.flags & 1u) != 0u) {
        b.flags |= a.flags & 1u;
        return b;
    }
    MjCtxState r = a;
    r.commas = a.commas + b.commas;
    r.colons = a.colons + b.colons;
    if (b.br_skel != MJ_CTX_NONE) {
        r.br_skel = b.br_skel;
        r.br_byte = b.br_byte;
        r.br_token = b.br_token;
        r.br_commas = a.commas + b.br_commas;
        r.br_colons = a.colons + b.br_colons;
    }
    return r;
}

// Advance the state across one element (boundary reset first, then the
// element's delta) — the single transition both ctx_partials and
// pair_ctx_apply use, so summaries and replays can never disagree.
// Overflow elements are inert (`inert` true): the boundary reset still
// applies, but the element contributes no delta to the walk state.
static inline void mj_ctx_advance(
    thread MjCtxState& s,
    bool boundary,
    bool inert,
    uchar byte,
    uint skel_idx,
    uint token_idx)
{
    if (boundary) {
        uint f = s.flags | 1u;
        s = mj_ctx_identity();
        s.flags = f;
    }
    if (inert) {
        return;
    }
    if (byte == uchar(',')) {
        s.commas += 1u;
    } else if (byte == uchar(':')) {
        s.colons += 1u;
    } else {
        s.br_skel = skel_idx;
        s.br_byte = uint(byte);
        s.br_token = token_idx;
        s.br_commas = s.commas;
        s.br_colons = s.colons;
    }
}

// Does sorted position `t` start a new depth group?
static inline bool mj_group_boundary(
    device const uint* sorted,
    device const uint* depths,
    ulong t,
    uint key_max)
{
    if (t == 0) {
        return true;
    }
    return mj_sort_key(depths[sorted[t]], key_max)
        != mj_sort_key(depths[sorted[t - 1]], key_max);
}

// --- ctx_partials ---------------------------------------------------------------

kernel void ctx_partials(
    device const uint* sorted [[buffer(0)]],
    device const uint* depths [[buffer(1)]],
    device const uchar* skel_byte [[buffer(2)]],
    device const uint* skel_token_index [[buffer(3)]],
    device MjCtxState* chunk_ctx [[buffer(4)]],
    constant MjParams& params [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup MjCtxState tg_states[THREADGROUP_SIZE];

    ulong m = params.element_count; // skeleton_total
    uint64_t max_depth = params.reserved0;
    uint key_max = mj_key_max(max_depth);
    ulong base = ulong(tgid) * ulong(MJ_SKEL_CHUNK_ELEMS)
        + ulong(lid) * ulong(MJ_SKEL_PER_THREAD);

    MjCtxState s = mj_ctx_identity();
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < m) {
            uint e = sorted[t];
            bool boundary = mj_group_boundary(sorted, depths, t, key_max);
            bool inert = mj_depth_overflows(depths[e], max_depth);
            mj_ctx_advance(s, boundary, inert, skel_byte[e], e, skel_token_index[e]);
        }
    }
    tg_states[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        MjCtxState run = mj_ctx_identity();
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            run = mj_ctx_combine(run, tg_states[i]);
        }
        chunk_ctx[tgid] = run;
    }
}

// --- ctx_spine ------------------------------------------------------------------

// One threadgroup: rewrite the chunk summaries in place as the exclusive
// walk state entering each chunk (chunk 0 gets the identity). Same ladder
// shape as the other spines, with the monoid combine instead of +.
kernel void ctx_spine(
    device MjCtxState* chunk_ctx [[buffer(0)]],
    constant MjParams& params [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup MjCtxState lanes[THREADGROUP_SIZE];

    uint n = uint(params.element_count); // skeleton chunks
    uint per = (n + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;
    uint base = lid * per;

    MjCtxState s = mj_ctx_identity();
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            s = mj_ctx_combine(s, chunk_ctx[idx]);
        }
    }
    lanes[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        MjCtxState run = mj_ctx_identity();
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            MjCtxState t = lanes[i];
            lanes[i] = run;
            run = mj_ctx_combine(run, t);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MjCtxState run = lanes[lid];
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            MjCtxState c = chunk_ctx[idx];
            chunk_ctx[idx] = run;
            run = mj_ctx_combine(run, c);
        }
    }
}

// --- pair_ctx_apply -------------------------------------------------------------

// Evaluate one sorted element against the walk state `s` carried up to it
// (exclusive — the state BEFORE the element), mirroring the reference's
// per-element match arms. Returns the min of the error candidates it
// fired.
static inline uint64_t mj_pair_eval(
    ulong t,
    uint e,
    MjCtxState s,
    device const uint* sorted,
    device const uint* depths,
    device const uchar* skel_byte,
    device const uint* skel_pos,
    device const uint* skel_token_index,
    device uint* match_index,
    device uchar* context_opener,
    device uint* child_counts,
    ulong m,
    uint key_max)
{
    uint64_t err = MJ_HEADER_NO_ERROR;
    uchar b = skel_byte[e];
    ulong pos = ulong(skel_pos[e]);
    bool prev_open = s.br_skel != MJ_CTX_NONE && mj_is_open_byte(uchar(s.br_byte));

    if (mj_is_open_byte(b)) {
        if (prev_open) {
            // Two opens without a close in one depth group — only reachable
            // through depth-mangled (already-rejected) inputs, flagged at
            // the second open like the reference.
            err = min(err, mj_pack_error(pos, MJ_ERR_UNBALANCED));
        }
        context_opener[e] = uchar(0);
        // match_index[e] / child_counts[e] are written by the matching
        // close (adjacent pairing). Unpaired opens only exist on rejected
        // inputs, whose CB3 outputs are never read.
    } else if (mj_is_close_byte(b)) {
        if (!prev_open) {
            // Close with no pending open (group-0 stray closes land here,
            // already flagged by the depth scan at the same offset/code).
            err = min(err, mj_pack_error(pos, MJ_ERR_UNBALANCED));
            match_index[e] = MJ_CTX_NONE;
        } else {
            if ((uchar(s.br_byte) ^ b) != uchar(0x06)) {
                // `{` closed by `]` (or `[` by `}`): xor is 0x26, not 0x06.
                err = min(err, mj_pack_error(pos, MJ_ERR_UNBALANCED));
            }
            // Pair both directions (the reference does so even on the xor
            // mismatch) and write the open's child count.
            match_index[e] = s.br_skel;
            match_index[s.br_skel] = e;
            uint count;
            if (s.br_byte == uint('{')) {
                count = s.colons - s.br_colons;
            } else if (skel_token_index[e] == s.br_token + 1u) {
                count = 0u; // `[]`: close token immediately follows open
            } else {
                count = (s.commas - s.br_commas) + 1u;
            }
            child_counts[s.br_skel] = count;
        }
        context_opener[e] = uchar(0);
        child_counts[e] = 0u;
    } else { // ':' or ','
        match_index[e] = MJ_CTX_NONE;
        child_counts[e] = 0u;
        if (!prev_open) {
            // Separator with no enclosing container: a complete root value
            // already ended (`1,2`, `{},{}`, `[""],1`).
            context_opener[e] = uchar(0);
            err = min(err, mj_pack_error(pos, MJ_ERR_TRAILING_CONTENT));
        } else {
            context_opener[e] = uchar(s.br_byte); // the forward-fill result
            if (b == uchar(':')) {
                if (s.br_byte != uint('{')) {
                    // Colon inside an array: `[1,"a":2]`.
                    err = min(err, mj_pack_error(pos, MJ_ERR_UNEXPECTED_TOKEN));
                }
            } else if (s.br_byte == uint('{')) {
                // Object comma must introduce `"key":` — the next element
                // of this depth group is the member's colon, exactly 3
                // tokens later. sorted[t + 1] may be an inert overflow
                // element sharing the clamped key; treating it as "not the
                // member colon" is reference-exact, because a real member
                // colon (only quote tokens sit between `,` and a colon at
                // +3 — Layer 1 bans every other token before `:`) has the
                // comma's own depth and is its immediate doc-order
                // neighbor, so it would BE sorted[t + 1].
                bool ok = false;
                if (t + 1 < m) {
                    uint next = sorted[t + 1];
                    ok = mj_sort_key(depths[next], key_max)
                            == mj_sort_key(depths[e], key_max)
                        && skel_byte[next] == uchar(':')
                        && skel_token_index[next] == skel_token_index[e] + 3u;
                }
                if (!ok) {
                    err = min(err, mj_pack_error(pos, MJ_ERR_MISSING_COLON));
                }
            }
        }
    }
    return err;
}

kernel void pair_ctx_apply(
    device const uint* sorted [[buffer(0)]],
    device const uint* depths [[buffer(1)]],
    device const uchar* skel_byte [[buffer(2)]],
    device const uint* skel_pos [[buffer(3)]],
    device const uint* skel_token_index [[buffer(4)]],
    device const MjCtxState* chunk_ctx [[buffer(5)]], // exclusive carries
    device uint* match_index [[buffer(6)]],
    device uchar* context_opener [[buffer(7)]],
    device uint* child_counts [[buffer(8)]],
    device ulong* chunk_error [[buffer(9)]],
    constant MjParams& params [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup MjCtxState tg_states[THREADGROUP_SIZE];
    threadgroup ulong lanes[THREADGROUP_SIZE];

    ulong m = params.element_count; // skeleton_total
    uint64_t max_depth = params.reserved0;
    uint key_max = mj_key_max(max_depth);
    ulong base = ulong(tgid) * ulong(MJ_SKEL_CHUNK_ELEMS)
        + ulong(lid) * ulong(MJ_SKEL_PER_THREAD);

    // 1) Per-thread range summary (identical transitions to ctx_partials).
    MjCtxState sum = mj_ctx_identity();
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < m) {
            uint e = sorted[t];
            bool boundary = mj_group_boundary(sorted, depths, t, key_max);
            bool inert = mj_depth_overflows(depths[e], max_depth);
            mj_ctx_advance(sum, boundary, inert, skel_byte[e], e, skel_token_index[e]);
        }
    }
    tg_states[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Thread 0 rewrites the summaries in place as each thread's incoming
    //    state, seeded with the chunk carry (deterministic serial ladder).
    if (lid == 0u) {
        MjCtxState run = chunk_ctx[tgid];
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            MjCtxState t = tg_states[i];
            tg_states[i] = run;
            run = mj_ctx_combine(run, t);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Replay the walk over this thread's elements: boundary reset, then
    //    evaluate against the exclusive state, then advance — exactly the
    //    reference's per-element order. Overflow elements are inert: they
    //    reset on a boundary like everyone else but are neither evaluated
    //    nor advanced over (see the module header). The group's LAST
    //    element — inert or not — also runs the unclosed-container check
    //    on the inclusive state the non-inert elements produced.
    MjCtxState s = tg_states[lid];
    uint64_t err = MJ_HEADER_NO_ERROR;
    for (uint j = 0u; j < MJ_SKEL_PER_THREAD; ++j) {
        ulong t = base + ulong(j);
        if (t < m) {
            uint e = sorted[t];
            uchar b = skel_byte[e];
            if (mj_group_boundary(sorted, depths, t, key_max)) {
                s = mj_ctx_identity();
            }
            if (!mj_depth_overflows(depths[e], max_depth)) {
                err = min(err,
                          mj_pair_eval(t, e, s, sorted, depths, skel_byte, skel_pos,
                                       skel_token_index, match_index, context_opener,
                                       child_counts, m, key_max));
                mj_ctx_advance(s, false, false, b, e, skel_token_index[e]);
            }

            bool group_last = t + 1 == m
                || mj_sort_key(depths[sorted[t + 1]], key_max)
                    != mj_sort_key(depths[e], key_max);
            if (group_last && s.br_skel != MJ_CTX_NONE
                && mj_is_open_byte(uchar(s.br_byte))) {
                // Unclosed container: the group walk ends with a pending
                // open — error at the OPEN's own position (`[1`, `{"a":1`).
                err = min(err,
                          mj_pack_error(ulong(skel_pos[s.br_skel]), MJ_ERR_UNBALANCED));
            }
        }
    }

    lanes[lid] = err;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        uint64_t chunk_min = MJ_HEADER_NO_ERROR;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            chunk_min = min(chunk_min, lanes[i]);
        }
        // Fold onto the depth scan's candidates for the same chunk index
        // (single writer per entry; the serial encoder orders this after
        // depth_apply).
        chunk_error[tgid] = min(chunk_error[tgid], chunk_min);
    }
}

// --- structure_finalize ---------------------------------------------------------

// One threadgroup: min-fold the per-chunk CB3 error words into
// header.error (single writer; the serial encoder orders this last in
// CB3). On accepted inputs the header still holds MJ_HEADER_NO_ERROR from
// the CB2 sync, so a fold is exact.
kernel void structure_finalize(
    device const ulong* chunk_error [[buffer(0)]],
    device MjHeaderDev* header [[buffer(1)]],
    constant MjParams& params [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup ulong lanes[THREADGROUP_SIZE];

    uint n = uint(params.element_count); // skeleton chunks
    uint per = (n + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;
    uint base = lid * per;

    uint64_t emin = MJ_HEADER_NO_ERROR;
    for (uint k = 0u; k < per; ++k) {
        uint idx = base + k;
        if (idx < n) {
            emin = min(emin, chunk_error[idx]);
        }
    }
    lanes[lid] = emin;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        uint64_t folded = header->error;
        for (uint i = 0u; i < THREADGROUP_SIZE; ++i) {
            folded = min(folded, lanes[i]);
        }
        header->error = folded;
    }
}
