// 11_emit.metal — K12 `emit_container_words` + K13 `tape_root_words`: the
// CB3 tail that materializes the M3 tape (container + root words).
//
//   emit_container_words  1 thread / skeleton element (brackets do the
//                         work, separators no-op): gather the partner via
//                         the K9 pair map and both tape positions via the
//                         K6b tape_ofs map, then write the element's OWN
//                         tape word — open: tag | (one past the close's
//                         tape index) | child count << 32 (saturated);
//                         close: tag | the open's tape index
//   tape_root_words       1 thread: the root prologue word at tape[0]
//                         (payload = the final root's index) and the final
//                         root word at tape[len - 1] (payload 0)
//
// Both run inside CB3, after pair_ctx_apply (K12 reads its match_index /
// child_counts) and before/independent of structure_finalize — the tape
// buffer exists because tape_word_total was read at the CPU sync 2, so no
// extra sync is needed (the plan's CB3 shape).
//
// The bit-exact spec is `reference::emit_tape` (src/reference/emit.rs):
//
//   - a bracket's tape position is tape_ofs[token] = 1 + the exclusive
//     footprint prefix (K6b), exactly the reference's tape_pos vector;
//   - an open word is make_open(tag, partner_tape_pos + 1, child_count)
//     with the count saturated at MJ_CONTAINER_COUNT_MAX = 0xFFFFFF
//     (mj_make_open mirrors crate::tape::make_open's saturation);
//   - a close word is make_close(tag, partner_tape_pos);
//   - the TAG bytes are the skeleton bytes themselves: the tape contract's
//     container tags ARE ASCII '{' '}' '[' ']' (tape_types.h), so
//     skel_byte[e] is passed straight through as the tag;
//   - tape[0] = make_root(final_root_index) with final_root_index =
//     tape_word_total + 1 (the reference seeds its running position at 1
//     and appends the final root last); tape[final_root_index] =
//     make_final_root().
//
// HOLE CONVENTION (M3): scalar and string tape positions — everything that
// is not a container or root word — are HOLES the M4 kernels (K10 numbers,
// K11 strings) will fill. The orchestration zero-fills the tape buffer
// before CB3, and no M3 kernel touches a hole, so holes read as 0u64
// deterministically. Where the reference detects EmptyInput (stage 3, on an
// empty token stream) and TrailingContent (stage 4, depth-0 separators) is
// already mirrored upstream: EmptyInput is the CPU-side verdict at the CB1
// sync (src/gpu/stage2.rs), TrailingContent fires in pair_ctx_apply
// (10_pair_ctx.metal); the CB3 error reduce is structure_finalize. K13 is
// therefore exactly the root words.
//
// Memory safety on REJECTED inputs (whose CB3 outputs are never read — the
// rejection contract): match_index entries of unpaired opens are stale, so
// K12 skips any partner index >= skeleton_total (MJ_CTX_NONE always is) and
// otherwise only produces in-bounds reads/writes — a stale partner < m
// still names a real skeleton record, whose token index and tape offset are
// in range, and distinct brackets always have distinct tape offsets (the
// footprint prefix sum is strictly increasing across footprint-1 tokens),
// so no write conflicts either. Clean inputs pair every bracket, so every
// container word is written exactly once.
//
// Neither kernel uses cooperative scans or barriers, so both dispatch as
// plain thread grids (Dispatch::Threads) with per-thread bound checks.

#include "common.h"
#include "tape_types.h"

// --- K12: emit_container_words ----------------------------------------------------

kernel void emit_container_words(
    device const uint* skel_token_index [[buffer(0)]],
    device const uchar* skel_byte [[buffer(1)]],
    device const uint* match_index [[buffer(2)]],
    device const uint* child_counts [[buffer(3)]],
    device const uint* tape_ofs [[buffer(4)]],
    device ulong* tape [[buffer(5)]],
    constant MjParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    ulong m = params.element_count; // skeleton_total
    if (ulong(tid) >= m) {
        return;
    }
    uchar b = skel_byte[tid];
    if (!mj_is_open_byte(b) && !mj_is_close_byte(b)) {
        return; // separators emit no tape words
    }
    uint partner = match_index[tid];
    if (ulong(partner) >= m) {
        return; // unpaired bracket: rejected input, output never read
    }
    uint own_pos = tape_ofs[skel_token_index[tid]];
    uint partner_pos = tape_ofs[skel_token_index[partner]];
    if (mj_is_open_byte(b)) {
        // One past the matching close word; count saturates at 0xFFFFFF.
        tape[own_pos] = mj_make_open(b, partner_pos + 1u, child_counts[tid]);
    } else {
        tape[own_pos] = mj_make_close(b, partner_pos);
    }
}

// --- K13: tape_root_words ----------------------------------------------------------

// element_count = the total tape word count (tape_word_total + 2); the
// final root index is one less.
kernel void tape_root_words(
    device ulong* tape [[buffer(0)]],
    constant MjParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) {
        return;
    }
    ulong final_root = params.element_count - 1;
    tape[0] = mj_make_entry(MJ_TAG_ROOT, final_root);
    tape[final_root] = mj_make_entry(MJ_TAG_ROOT, 0);
}
