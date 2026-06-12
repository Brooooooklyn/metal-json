# metal-json: parsing huge JSON on an Apple GPU

JSON parsing is supposed to be the worst possible workload for a GPU. The grammar is a chain of data-dependent branches. Strings can contain escaped quotes that change the meaning of every byte after them. Brackets match by a stack, and a stack is the textbook example of sequential state. Every byte's meaning depends on every byte before it.

I built a JSON parser that runs on the GPU of a Mac anyway. It is called metal-json. It parses standard JSON to a simdjson-compatible 64-bit tape — the same flat output format the fastest CPU parser produces — using Metal compute kernels on Apple Silicon. On large documents it beats C++ simdjson, the parser it borrows its output format from. On small documents it loses, badly, and I want to show that part first:

```
dataset       size       metal-json   simdjson-cpp   vs simdjson   winner
twitter       0.6 MiB    1.287 ms     0.159 ms       0.12x         simdjson
twitter_1m    1.0 MiB    1.009 ms     0.328 ms       0.33x         simdjson
twitter_4m    4.0 MiB    1.435 ms     1.385 ms       0.96x         simdjson
twitter_8m    8.0 MiB    1.955 ms     2.615 ms       1.34x         metal-json
twitter_16m   16.0 MiB   3.383 ms     6.129 ms       1.81x         metal-json
twitter_64m   64.0 MiB   10.465 ms    23.852 ms      2.28x         metal-json
twitter_100m  100.0 MiB  16.386 ms    33.772 ms      2.06x         metal-json
twitter_256m  256.0 MiB  37.654 ms    82.308 ms      2.19x         metal-json
twitter_512m  512.0 MiB  71.447 ms    193.711 ms     2.71x         metal-json
```

Criterion medians, one Apple M5 Max (128 GiB unified memory, macOS 26.5.1), simdjson v4.6.4 built with `-O3 -mcpu=native`, both parsers timed doing the same work, both verified to produce bit-identical results before timing. The crossover on this sweep is about 4.3 MiB — the report says to treat it as a band that moves with document shape and machine, and the README rounds it to 4–5 MiB. Below it, the GPU's fixed dispatch and sync overhead dominates — at 0.6 MiB the GPU runs at 0.12x, meaning simdjson is about 8x faster. Above it, metal-json wins 2.1–2.7x on 100–512 MiB twitter-shaped inputs, and the gap widens with size. Document shape matters too: number-dense `canada.json` already favors the GPU at 2.1 MiB (1.42x), while `citm_catalog` at 1.6 MiB favors simdjson (0.44x).

If your documents are small, use a CPU parser. This library is for big ones. The rest of this article is how that table happened, including the parts that went wrong.

## Why this should not work

The standard objection has two halves.

First, JSON is branchy. A naive parser is a byte-at-a-time state machine, and GPUs execute 32 threads in lockstep — divergent branches serialize.

Second, even if you fix the branches, JSON is sequential. Whether byte *i* is inside a string depends on the parity of quotes before it. Whether a quote is real depends on the parity of backslashes before it. Whether a `}` is legal depends on a stack of every unclosed bracket before it.

simdjson answered the first half years ago for CPUs: don't branch over bytes, compute bitmaps over them. Its stage 1 builds 64-bit masks (quotes, backslashes, structural characters) with SIMD compares and resolves string state with bit tricks. The exact tricks vary by version and platform: the vendored v4.6.4 resolves escape runs with a borrow-propagating *subtraction* (`maybe_escaped_and_odd_bits - potential_escape`), and on Apple Silicon — the platform actually benchmarked here — the prefix-XOR that turns quote positions into an "inside a string" mask is a six-step shift-XOR ladder; the arm64 source rejects the carryless-multiply route with the comment "We could do this with PMULL, but it is apparently slow" (clmul survives only in simdjson's x86 kernels). Branch-free, data-parallel, exactly what a GPU wants. My stage 1 ports those tricks to Metal — using the older simdjson `find_escaped` formulation (a carry-propagating 64-bit addition) for escape runs and the same shift-XOR ladder for the prefix-XOR — with each 64-byte chunk owned by one GPU thread instead of one loop iteration.

The second half — bracket matching and grammar — simdjson handles with a sequential stage 2, a tight tape-building loop. That does not port. A GPU version needs a way to match brackets without a stack; the trick that makes the whole project work gets its own section below.

I am not the first to put JSON parsing on a GPU. [cuJSON](https://github.com/AutomataLab/cuJSON) (ASPLOS '26) offloads validation, tokenization, and structure recognition to CUDA and reports wins over simdjson and over earlier GPU parsers like GPJSON and cuDF — the "GPUs can't parse" objection has been falling for a while. What is different here is the platform and the output contract: cuJSON targets discrete NVIDIA cards and emits its own purpose-built output format, while metal-json targets Apple Silicon's unified memory and emits simdjson's exact tape, so the benchmark comparison is one materialized tape against another with no format discount.

There is a third objection specific to discrete GPUs: moving the input over PCIe costs more than parsing it. Apple Silicon removes it. The CPU and GPU share one physical memory. `MTLBuffer newBufferWithBytesNoCopy` lets the GPU read the input pages in place, and the finished tape is read by the CPU in place. Zero copies in either direction. That is why this project targets Metal on Apple Silicon and not CUDA on a discrete card.

And one objection I want to grant fully: the dispatch overhead is real and it never goes away. That is the crossover at 4.3 MiB. I measured it before writing any kernel, which brings me to the spikes.

## Measure first: the three spikes

Before committing to a design I wrote three throwaway benchmark programs ("spikes") against raw Metal, each answering one question the design depended on. The design spec listed three top risks, and the spikes were the named mitigation for the second of them — "64-bit integer perf on 32-bit Apple ALUs (bitmaps + Eisel-Lemire)": Apple GPUs have 32-bit arithmetic units, and the whole plan leans on 64-bit bitmaps and 128-bit multiplies. (The first-listed risk was parallel grammar-validation completeness — more on that in the testing section.) The spike files were committed in the first commit and never touched again; their results are labeled "binding design decisions" in `docs/spikes.md`.

All numbers below: Apple M5 Max, Metal toolchain 32023.883, medians over multiple runs (median of 7, 10, and 20 respectively), each spike reproduced across two or three independent process runs with the ranges published. Spikes A and B verified every GPU output against a CPU oracle — A against a u128 chain on a 65,536-element prefix, B bit-for-bit between variants plus spot-checks against a scalar CPU model. Spike C is a pure timing spike; it checks only that the command buffer did not error.

### Spike A: how to multiply 64x64→128 (for the float kernel)

Float parsing via the Eisel–Lemire algorithm needs full 128-bit products. Three candidate formulations, measured over 2^26 dependent multiply chains with the memory traffic cancelled out arithmetically:

```
variant                                    ALU-isolated mul128/s
mulhi(ulong) builtin + x*y                 3.76e11
limbs32  (pure 4-limb 32-bit, bool carry)  3.82e11
limbs64  ((ulong)uint*uint partials)       6.04e11   <- winner, ~1.6x
```

The winner is simdjson's own portable `umul128` fallback: four `(ulong)uint*uint` partial products assembled with 64-bit adds. The recorded reason: the two opaque builtins each redo partial-product work, while the explicit limb form lets the compiler compute the four partials once and derive both result words. The spike also recorded a caveat — at realistic memory pressure all three variants ran identically at ~546 GB/s, so "formulation choice likely won't move end-to-end numbers much — it is still free perf, take it."

A side measurement (Q3) shaped the rest of the kernels: a rotate/popcount-heavy 64-bit chain ran at 0.29–0.31x of the 32-bit version — 64-bit rotates lower to roughly 3x the cost of a 32-bit op, while plain add/xor lower cleanly to 2x. Consequence: keep state in `ulong`, but never do 64-bit popcounts or bit-scans in hot loops. The production code does popcounts and count-leading-zeros on 32-bit halves everywhere.

### Spike B: `ulong` bitmaps or `uint2` bitmaps?

The bitmap kernels keep one 64-bit mask per 64 input bytes. Should that mask be a native `ulong` (emulated on 32-bit ALUs) or an explicit `uint2` pair with hand-written carries? Two kernels doing identical simdjson-stage-1-style work over 1 GiB of input:

```
config                ulong                  uint2                  ratio
rounds=1 (K1 mix)     4.507 ms (238 GB/s)    2.198 ms (489 GB/s)    2.05x
rounds=8 (ALU-heavy)  5.351 ms (201 GB/s)    2.715 ms (395 GB/s)    1.97x
```

The `uint2` variant reaches ~550 GB/s of total memory traffic — it turns the kernel memory-bandwidth-bound — "while the ulong variant is ALU-bound at under half that, so the emulated 64-bit ops cost more than the entire memory traffic of the kernel." The spike doc also recorded the texture of the runs: the `ulong` medians wobbled ~10% between process runs while `uint2` was stable within 1%, and the two variants' outputs matched bit-for-bit, with identical XOR checksums across runs. Decision: every hot bitmap is `uint2`. The compromise is a hand-maintained helper header (`shl64_u2`, `add64_u2` with carry-out, `prefix_xor64_u2`, ...) instead of native operators.

The rejected alternative got a consolation job. The `uint2` self-test kernel deliberately uses `ulong` as its in-kernel oracle, with a comment I enjoy: Apple GPUs do support `ulong` — "it is merely ~2x slower in the hot bitmap mix, which is exactly what banished it from the real kernels and exactly what makes it a fine in-kernel oracle here."

### Spike C: what does a dispatch actually cost?

The planned pipeline was 3 command buffers (batches of GPU work submitted together), ~14 kernel dispatches, 2 CPU synchronization points. Measured fixed costs:

```
scenario                                   wall            GPU
empty command buffer, commit+wait          18.7–20.1 us    n/a
one CB, 14 tiny dispatches                 219–421 us      29–53 us
planned 3-CB shape + 3 waits               497–533 us      31–38 us
16Mi-thread no-op dispatch                 ~160 us         (~2.4 ns per 256-thread group)
```

Three rules fell out, and all three are quoted in source comments to this day:

1. **The overhead is CPU-side.** Nearly all of the ~0.5 ms is encode/commit/sync latency; each `waitUntilCompleted` round trip costs ~50–160 µs, and even within one command buffer each tiny dispatch costs ~14–29 µs of CPU encode time — which is why the batching layer's comment says it "exists precisely to pay the encoder + sync cost once per command buffer instead of once per kernel." GPU busy time for trivial work is ~30–50 µs. The sync number became the standing currency for architecture decisions — every time a kernel got its own command buffer, the comment prices it at "~50–160 µs per spike C."
2. **Never go thread-per-byte.** Scheduling alone costs ~2.4 ns per 256-thread threadgroup, so a giant no-op grid costs real time. The byte-domain bitmap kernels honor a 64-bytes-per-thread grain floor (`MJ_WORD_BYTES = 64`): the classifier processes 64 B per thread, the token-mask and token-scatter kernels 256 B. Kernels whose grids scale with tokens, scalars, or strings are not bound by that floor — they process 4 tokens, one scalar run, or one string per thread.
3. **Use medians.** Low-occupancy wall times jitter up to 4x from GPU power-state ramping. Every benchmark in the project reports medians for this reason.

Spike C also predicted the crossover: ~0.50–0.53 ms of fixed overhead against a 3–7 GB/s CPU baseline gives a break-even at 1.5–3.7 MB. The final measured crossover was ≈4.3 MiB — slightly above the band because simdjson runs faster than 7 GB/s on small hot inputs on this machine, but the same mechanism, predicted before any kernel existed. The spike also noted that at the ≥100 MB design target, 0.5 ms is under 1% — noise. A considered-and-deferred option is recorded with it: merging two of the command buffers would save ~0.1–0.3 ms, worth nothing at the design target, so it never happened.

## The output: a simdjson tape, deliberately

The parser's output is a *tape*: a flat array of 64-bit words plus a string buffer of unescaped string records. The layout is simdjson's tape layout, and the spec records why: "chosen deliberately so the M5 benchmark compares apples to apples: both parsers do the same amount of output work." No benchmark games where one parser materializes less than the other.

Every tape word is one little-endian u64:

```
bits 63..56   tag (ASCII byte)
bits 55..0    payload (56 bits)

word = ((tag as u64) << 56) | payload
```

The tags are simdjson's own ASCII tape characters: `r` root, `{` `}` `[` `]` containers, `"` string, `l`/`u`/`d` for i64/u64/f64 numbers (two words: marker, then raw value bits), `t` `f` `n` for literals. A container open word packs the index *one past* its matching close in bits 0..31 — that is the O(1) skip — and its direct-child count in bits 32..55, saturated at 0xFFFFFF (16,777,215), meaning "this many or more," exactly as simdjson does. Consumers needing the exact count of a saturated container must walk it. That is the first entry in the compromise ledger: 24 bits of count traded for fitting count and end-index into one word.

The format spec pins a worked example, byte for byte, and a test fails if the doc and the code ever drift. For the 23-byte input `{"a":[1,2.5],"b":"x\n"}` the tape is exactly 13 words:

```
idx  word (hex)              tag  meaning
0    7200_0000_0000_000C     r    final root word at index 12
1    7B00_0002_0000_000C     {    end = 12 (one past } at 11), count = 2
2    2200_0000_0000_0000     "    string buffer offset 0  -> key "a"
3    5B00_0002_0000_0009     [    end = 9 (one past ] at 8), count = 2
4    6C00_0000_0000_0000     l    i64 marker
5    0000_0000_0000_0001     -    i64 bits of 1
6    6400_0000_0000_0000     d    f64 marker
7    4004_0000_0000_0000     -    f64 bits of 2.5
8    5D00_0000_0000_0003     ]    matching [ at index 3
9    2200_0000_0000_0006     "    offset 6  -> key "b"
10   2200_0000_0000_000C     "    offset 12 -> value "x\n"
11   7D00_0000_0000_0001     }    matching { at index 1
12   7200_0000_0000_0000     r    points back at index 0
```

Number typing mirrors simdjson exactly: an integer literal that fits i64 gets tag `l`; one in the u64-only range gets `u`; anything fractional, exponential, or out of integer range gets `d` with the correctly rounded double's bits. Integers beyond u64 become f64 — a real loss of precision, accepted for simdjson parity, and it resurfaces later as a serde limitation.

### The one place the tape deviates: string offsets

A string record in the buffer is `[u32 LE length][content bytes][NUL]`. The explicit length exists because ` ` is legal JSON and produces an interior NUL. simdjson packs these records densely, in the order its sequential parse writes them. I could not: dense packing makes every offset depend on every previous string's *unescaped* length, which is not known until unescaping runs.

So the format pins a deviation: slots are allocated by **raw** input length.

```
raw_len(s)  = byte count between the quotes in the INPUT (escapes intact)
slot(s)     = raw_len + 5            (4-byte length prefix + NUL)
offset(s)   = exclusive prefix sum of slot, in document order
```

Now every offset is computable in parallel from token positions alone, before any unescaping happens. This scheme is not a design-time insight, though — it is an M1 adversarial-review fix, caught before any GPU code existed. The review pass flagged that the original spec's offsets were not computable by the planned kernels, and the fix commit re-specified them "to match what the GPU can compute: exclusive prefix sum of (raw_len + 5) in document order," with the policy pinned in the format doc.

The compromise: a record whose escapes shrank it does not fill its slot, leaving gap bytes (unescaping never grows a string, so slots never overflow). The gaps later caused a real security bug — covered in the war stories — and are now required to be zero-filled on both backends, at a cost proportional to escape shrinkage, never to buffer size.

The whole format is locked across three artifacts — Rust constants, a Metal header, and the prose spec — by two tests: `msl_header_layout_lock` parses the Metal header at test time and fails on any one-sided Rust/Metal constant edit, in either direction, and `worked_example_matches_tape_format_doc` pins the prose spec's worked example word-for-word and byte-for-byte.

## The pipeline

The shipped pipeline is thirteen kernel families across four command buffers (the plan said three; a fourth appeared for a reason below). Each CPU sync exists to read back an exact size so the next allocation is exact-fit — never a guess proportional to input length. Kernel numbers are the design doc's names (they survive in the shader file names); the order shown is dispatch order.

```
input bytes (page-aligned, space-padded to 64 B words, read in place)
   |
CB1: K1  classify_escape_utf8   quote + candidate bitmaps (uint2),
                                escape carries, UTF-8 validation
     K1b escape_carry_fixup     the backslash-wall valve
     K2  spine_quote_scan       chunk quote carries + total (odd = error)
     K3  token_mask             in-string mask via prefix-XOR; tokens
     K4  spine_token_scan       chunk token carries + token_total
   -- sync 1: read header; reject bad UTF-8 / odd quotes;
      allocate tok_pos/tok_kind at exactly token_total --
CB2: K5  token_scatter          dense-ranked token positions + kinds
     K6  token_validate_footprint  parallel grammar rules + tape footprints
     K7  spine3                 prefix sums of footprints/counts; totals
   -- sync 2: reject Layer-1 errors; allocate tape + work lists exact-fit --
CB2b: K6b apply_tape_offsets    tape offset per token; skeleton/string/
                                scalar work lists
   -- sync 3 --
CB3: depth triple               nesting depth per skeleton element
     K8  [counting sort] x passes   stable 5-bit LSD sort by depth
     K9  ctx triple             pairing + container context (segmented scan)
     K12/K13 emit               container + root tape words
     string_record_offsets, K11 strings_unescape, K10 parse_numbers
     structure_finalize         error fold
   -- sync 4: verdict; rare CPU fixups patch the shared tape --
Document (tape + strings read in place, zero copy)
```

### Stage 1: bytes to tokens, and the backslash wall

K1 gives each thread one 64-byte word of input (the spike-C grain). It builds byte-class bitmaps with SWAR (SIMD-within-a-register) compares, resolves escapes with the simdjson `find_escaped` trick ported to `uint2`, validates UTF-8, and emits quote and candidate bitmaps.

Two sequential dependences cross word boundaries, and both are broken the same way: by re-reading raw neighbor bytes instead of waiting for another thread. The escape state at a word's left edge is the parity of the backslash run ending just before it — so the thread peeks backward in the raw input. Unbounded look-back would make an adversarial input ("\\\\\\...", a backslash wall) quadratic, so the peek is capped at 4096 bytes. A thread whose run is still going at the cap *guesses* (even run, not escaped), sets a flag, and bumps a counter. K1b, the valve — the project's word for a bounded fast path with an escape hatch to a slower exact path — runs after K1 in the same command buffer: on benign input every thread reads the zero counter and exits — one near-empty dispatch costing 0.114 ms on a 512 MiB parse (0.3% of GPU time). On a wall, it repairs flagged words by walking the flag chain backward in 4096-byte strides, using the fact that an even-length all-backslash gap preserves parity. The subtle half is a quote whose own look-back capped: the run before it is one byte short of a whole cap — an odd gap — so the quote's escapedness is the anchor parity *flipped*, and the quote is real exactly when it is not escaped. The repair cost is written down too: O(chain length) loads per flagged word — quadratic in the wall length across threads, in parallel, pathological inputs only — and a test pins a flag chain that walks through 64 flagged words to reach its anchor. Pathological inputs pay; benign inputs don't. The valve's cost is inside every benchmark number.

The in-string mask itself is the prefix-XOR of real quote bits — bit *i* is the parity of quote bits at or before *i*. In `uint2` form, five of the six shift-ladder steps run on both 32-bit halves at once, and the cross-half step collapses to one line I find genuinely pretty:

```c
// after the in-word ladder, bit 31 of lo is the parity of ALL 32 low
// bits; if that parity is odd every hi bit flips. Broadcast and XOR.
x.y ^= 0u - (x.x >> 31);
```

Then K3 computes tokens per word, bit-exactly matching the CPU reference:

```
in_string = prefix_xor64(quote_real) ^ parity_carried_into_word
tokens    = (candidates & ~in_string & ~quote_real) | quote_real
```

Carries between 64 KiB chunks flow through a two-level scan: K1/K3 emit one partial per chunk, and K2/K4 each run as a *single* 256-thread threadgroup that rewrites the partials in place as an exclusive prefix sum. No decoupled look-back scans anywhere — Apple GPUs give no forward-progress guarantee between threadgroups, so the design never lets one threadgroup spin-wait on another. K5 then scatters tokens to their globally dense ranks: position and kind arrays, every slot written exactly once.

K5 sits in its own command buffer for a principled reason: its output buffers cannot exist until the CPU has read `token_total`. The cost is one extra sync round trip, priced at ~50–160 µs by spike C, accepted for exact-fit allocation. The same rule later added CB2b — the fourth command buffer the plan didn't have.

UTF-8 validation lives inside K1 with exact first-error offsets (the same offset Rust's `str::from_utf8` reports; a test pins the equivalence). The first version walked every non-ASCII word byte-by-byte, and that walk was 75% of K1 on twitter — where 30% of words contain non-ASCII bytes. The fix is a two-tier design: a register-only bitmask check *proves* the well-formed common case, and only inconclusive words fall to the scalar walk, which remains the single source of error offsets. The proof obligation is one-sided — fast-pass implies the walk accepts — so error behavior can't drift. That change cut K1 by about 60%.

Stage 1, for what it's worth, was never the bottleneck: at milestone 2 it already ran at 52.9 GB/s on a 256 MB input.

### Bracket matching by counting sort

Everything else in the pipeline is "port the bitmap tricks, add valves." This is the part with no CPU counterpart.

After stage 2, the *skeleton* is the list of structural elements: brackets, colons, commas. Assign weights (+1 open, −1 close, 0 separator) and take an exclusive prefix sum to get the depth before each element; an open records the depth *after* its increment, a close the depth *before* its decrement. The consequence: **an open bracket and its matching close share the depth of the container they delimit.**

Now the observation that replaces the stack. Within one depth, the brackets of distinct containers cannot interleave — to open a second container at depth *d* you must first return to depth *d−1*, which closes the first. `[..[..].. [..]..]` puts the inner pairs at the same depth, but one closes before the next opens. So within a depth group, in document order, brackets strictly alternate open/close. Which means: **a stable sort by depth makes every matching pair adjacent.**

Worked example, `[{"a":1},[2]]`:

```
skeleton (doc order):   [   {   :   }   ,   [   ]   ]
depth:                  1   2   2   2   1   2   2   1

stable sort by depth:
  depth-1 group:  [ , ]            <- outer array: open, comma, close
  depth-2 group:  { : } [ ]        <- the two inner containers, adjacent

pairing within a group: a close's partner is the latest preceding
bracket, when that bracket is an open.
  ]  (depth 1) -> [   ok
  }  (depth 2) -> {   ok          type check: open ^ close == 0x06
  ]  (depth 2) -> [   ok          ('{'^'}' == '['^']' == 0x06;
                                   '{' closed by ']' xors to 0x26: error)
```

One sort buys three things at once: pairing, balance errors (a leftover open or close in a group is an unbalanced-brackets error at exactly the offset the sequential reference reports), and *separator context* — a comma or colon's enclosing container is the latest open in its group, recovered by a segmented forward-fill. The forward-filled context drives the rules a stack would have checked: a colon whose enclosing opener is `[` is an error (`[1,"a":2]`), and an object comma must be chased by its member's colon exactly three tokens later (`{"a":1,2}`). Child counts come free too, as rank differences between an open and its close: an object's count is the colon-rank difference; an array's is the comma-rank difference plus one — or 0 when the close token immediately follows the open.

The sort is a least-significant-digit counting sort, 5-bit digits, 32 buckets, as the classic histogram → matrix scan → scatter triple. Stability is not an optimization here; the source shouts it: "STABILITY IS CORRECTNESS-CRITICAL." Document order within a group *is* the alternation guarantee. Stability comes from three nested orders — bucket-major matrix scan orders chunks, thread ID orders threads, each thread walks its four elements in order — each preserving document order within a bucket.

The pass count is a small contract with a big comment. Clean inputs only carry depths 1..=max_depth, so the sort key is `depth − 1`, one value narrower than raw depth — and at the default depth limit of 1024 (simdjson parity), keys 0..=1023 fit exactly two 5-bit passes where raw depth would need three:

```rust
pub fn sort_passes(max_depth: u32) -> usize {
    let key_max = max_depth.max(1) - 1;
    let bits = (32 - key_max.leading_zeros()).max(1) as usize;
    bits.div_ceil(5)
}
```

Error inputs are where this design earns its review scars. Depths past the limit *clamp into* the deepest legal key rather than getting a key of their own — a 1025th key value would need an eleventh bit and a third sort pass on every parse, paid by every clean document to handle inputs that get rejected anyway. The first implementation let clamped elements join that bucket's group walk, and the adversarial review pass after M3 found the counterexample: with `max_depth=1`, input `[[1]`, the overflow close would pair with the *inner* open, and the outer open's unbalanced-bracket error at offset 0 — the reference's first error — would vanish behind a depth-limit error at offset 1. The fix makes overflow elements **inert**: they keep their clamped sort key (no third pass) but never advance the walk, never evaluate rules, never write. A three-step proof that this preserves the reference's first-error verdict lives as a comment in `shaders/common.h`, including the observation that the first overflow element is always an open already flagged as DepthLimit at its own offset, which is the smallest error code in that stage, so dropping the rest changes nothing.

The depth scan itself carries a documented deviation of the same flavor: the reference *parks* underflowed closes at depth 0 while the GPU's prefix sum keeps going negative. The two agree on every depth up to and including the first underflow, and past that point the rejection contract discards every output of the rejected input anyway — a written soundness argument, not luck.

The pairing/context kernel (K9) then replays the sequential group walk in parallel. The whole walk state collapses to a small algebraic object — comma count, colon count, latest bracket — that combines associatively, so it scans with the standard reduce→spine→apply shape over 32-byte state structs. Its comments carry two opposite measured answers to the same questions. The per-simdgroup shuffle scan replaced a 256-step serial ladder that 255 threads sat at a barrier waiting for; but the *partials* kernel kept its serial ladder, because there nobody consumes the result in-kernel, the latency hides behind other threadgroups, and the shuffle-scan variant measured ~10% *slower*. Likewise the register cache that gathers each thread's four elements once (the compiler cannot CSE the reloads across the kernels' device writes) is used in the apply kernel and deliberately skipped in the partials kernel, where "its register pressure measurably hurts." Same primitives, two kernels, opposite answers, all measured and all written down.

There is also a measured shortcut: the depth kernel max-folds the clamped sort key into a single cell, and any sort pass whose digits are all zero — every parse of a document nested ≤ 32 deep, i.e. nearly all real JSON — degenerates to one coalesced stream copy. That made the mandatory second pass 78% cheaper.

### First-error parity without atomics

Errors everywhere in the pipeline are values, packed as `(byte_offset << 32) | code` and min-reduced — earliest offset wins, and numeric code order breaks same-offset ties. That tie-break is a real contract: the reference checks rules in a specific sequential order, the GPU evaluates all of them in parallel, and the only way both report the same error is if the code numbering encodes the reference's rule order. The header documents each required inequality with the input that forces it — MissingColon(16) < MissingComma(17) because of `{"a" "b"}`, Unbalanced(20) greatest of all because the reference's end-of-input check runs after every per-token rule.

Getting deterministic first-error parity out of a massively parallel pipeline is mostly this kind of bookkeeping, plus a discipline about atomics: every order-sensitive result is a single-writer plain store, ordered by Metal's serial encoder. The device atomics that do exist are all commutative reductions or counters whose ordering is unobservable or repaired afterward: a max-fold (the depth scan's max key), offset min-folds (the UTF-8 and number error offsets), counter adds (escape-carry overflow, per-chunk quote popcounts), and two fixup-list slot allocators (hard floats in K10, long strings in K11) whose scheduling-dependent slot order the CPU sorts away after the wait. The counting sort's histogram counts are threadgroup-scope atomics, not device atomics. And there is a hardware reason the error protocol looks the way it does: 64-bit device atomics are an Apple9+ feature the embedded shader library cannot assume — the ahead-of-time build targets generic macOS — so headers split 64-bit counters into an (atomic lo, plain hi) pair, and the two stages where many threads race to report an error (UTF-8 in K1, numbers in K10) min-reduce a 32-bit *offset only*. That works only because each of those stages has exactly one possible error code.

### Scalars: Eisel–Lemire with no doubles, strings with two valves

Apple GPUs have no 64-bit floating point. The number kernel K10 therefore computes f64 *bit patterns* using only integer math — no `double` appears anywhere in the file. It is a line-for-line port of simdjson's `compute_float_64` (minus its fp64 fast path, which can't exist here), over a 651-entry × 16-byte table of 128-bit powers of five covering exponents −342..308. The table is generated by a from-scratch big-integer implementation in the Rust test module and verified entry-by-entry; seven entries are additionally pinned against values transcribed from Rust core's `POWER_OF_FIVE_128` as codebase-independent ground truth. The 128-bit multiplies use the spike-A limbs64 form; normalization counts leading zeros on 32-bit halves per the spike-A side finding.

"Line-for-line port" hides the kind of detail the port actually surfaced. The binary-exponent formula is rewritten as `(((217706 * q + 74514432) >> 16) - 1137) + 1087 - lz` so the shifted operand is provably nonnegative over the whole exponent range — right-shifting a negative value is implementation-defined — and the comment carries the proof: `217706 * -342 + 74514432 = 58980 >= 0`. The exponent accumulator saturates at `100000000000000000` because "longer exponents only ever mean 'overflow' or 'underflow'."

Eisel–Lemire has well-known hard cases — the truncated-product off-by-one, truncated mantissas whose rounding is ambiguous, values that round to infinity. A CPU implementation falls back to a big-decimal path. The GPU does something simpler: the thread appends its token index to a **fixup list** via an atomic counter, writes a placeholder, and moves on. After the command buffer completes, the CPU re-parses just those literals with Rust's `str::parse::<f64>` (correctly rounded) and patches the value words in the shared buffer. One ambiguous-mantissa trick keeps the list short: when the mantissa was truncated, the kernel runs Eisel–Lemire on both `w` and `w+1`; if both endpoints round to the same double, the answer is certain and no fixup is needed. Tests force the list non-empty with exact decimal halfway points between adjacent doubles: the 54-digit halfway point between 1.0 and its successor, the subnormal/normal boundary, halfway between 0 and the smallest subnormal, and the halfway point of 1e22's neighbor pair (the source calls that one "the famous power-of-ten case"). The torture fixtures also pin `2.2250738585072011e-308` — the largest subnormal-rounding literal, the one that hung PHP and Java — and a 615-digit literal in the same family, tested bit-exact. The fixup merge has its own rule: a CPU re-parse can reject at an *earlier* offset than a GPU-detected error, so verdicts merge by packed minimum — a test pins a fixup rejection at offset 4 beating a later GPU grammar error. Semantics follow simdjson: underflow is a signed zero (`-1e-400` keeps its sign bit), overflow to infinity is a rejection.

K11, string unescaping, is one thread per string — and that shape has an obvious failure mode: one multi-megabyte string serializes the entire parse on a single GPU lane. The adversarial review after M4 demonstrated it: an 8 MB escaped string took 950 ms. The fix is the same fixup pattern: strings whose raw length exceeds 16,384 bytes (one 16 KiB page) go to a list and the CPU unescapes them after the wait, through the *same* unescape function the reference oracle uses — one implementation, divergence impossible. 950 ms became 83 ms, an 11.4x improvement on the adversarial case. The threshold's rationale is a doc comment: one page is long enough that real-world strings almost never cross it (JSON strings are overwhelmingly sub-KB, so the GPU keeps the whole hot path) yet short enough that the worst case one lane ever owns is a single page-sized walk, never megabytes. A Rust test parses the shader source to keep the constant in sync.

The same kernel got the project's biggest single optimization in the benchmark milestone. On twitter-shaped data ~93% of strings are clean (no escapes) and ~91% are ≤ 32 bytes, but the long tail owns about half the string bytes — and pre-split, every SIMD lane waited out the longest string in its group. The M5 shape runs three phases: short strings lane-parallel, long clean strings copied cooperatively by a whole simdgroup (32 coalesced bytes per step), and dirty strings compacted through a threadgroup queue into the fewest possible simdgroups before the sequential unescape walk runs. That cut K11 by 27%. It is still the most expensive kernel in the pipeline: 6.506 ms, 16.8% of GPU time, on the 512 MiB parse.

K10 has a documented, accepted cliff of the same species: a pathological multi-KB digit string serializes one thread. The comment sketches the valve that would fix it and the trigger for building it ("if real workloads ever care"). Compromise recorded, not hidden.

## Testing as a design tool

The test strategy was decided before the GPU code existed, and it shaped the GPU code.

Milestone 1 built a **CPU reference oracle**: a scalar, seven-stage pipeline that deliberately mirrors the GPU kernel structure — same stages, same inspectable intermediate artifacts, with one documented deviation, in the error model: the GPU min-reduces so the globally earliest-offset error wins, while the reference reports the first error in stage order, so on a multi-error document the two may disagree about *which* error is reported — never about *whether* parsing fails. The module doc admits it models the GPU formulation instead of the obvious recursive-descent one. That inversion is the point: because the oracle is stage-shaped, every GPU kernel can be differentially tested **bit-for-bit against its stage**, not just end-to-end. The oracle itself is validated against serde_json (configured with `preserve_order` and `arbitrary_precision`; doubles compared bit-exactly against `str::parse::<f64>` of the raw literal) and against JSONTestSuite: 95/95 `y` files parse, 188/188 `n` files error without panicking, and all 35 `i` files have pinned verdicts with written reasons, so any behavior change is noticed. One conformance file is a free advertisement for the stack-free design: `n_structure_100000_opening_arrays.json` — 100,000 unclosed `[`s, where recursive parsers stack-overflow — is rejected as unbalanced brackets by the Layer-1 open-bracket-then-EOF ban. There is no recursion anywhere in the pipeline.

The oracle is slow — 0.23–0.36 GB/s, some 7–15x slower than C++ simdjson (the pinned sample report's ratio column reads 0.07x, 0.08x, 0.15x). It exists for testing and backend parity, not speed, and it is not a runtime fallback for an explicitly selected GPU backend: "the reference backend stays what it is: the bit-exact oracle the GPU is diffed against, not a runtime escape hatch."

On top of the oracle:

```
layer                          what it pins
u2_selftest kernel             every uint2 helper vs in-kernel ulong oracle,
                               ~16k adversarial operand pairs, 15 named fail bits
per-stage differential         GPU stage outputs == reference, bit-for-bit,
                               over corpus + 318 JSONTestSuite files + seam/
                               wall/boundary fixtures + proptest
exhaustive model check         every token sequence of length <= 4 over a
                               12-symbol alphabet (22,620 inputs) + ~30k
                               stride-sampled length 5-6 sequences
end-to-end                     GPU tape words == reference tape words AND
                               whole string buffer byte-equal, gaps included
poison tests                   pool pre-fills buffers with 0xDB; no poison
                               byte may survive into any output
layout locks                   tests parse the .metal headers and fail on
                               any one-sided constant edit
```

Every suite runs under Metal's shader validation layer, and once more with runtime-compiled shaders to prove both shader build paths.

Property testing earned its keep twice, and both regressions are checked in. The first found a real kernel bug: my SWAR byte-classifier used the classic `(x - 0x01010101) & ~x & 0x80808080` zero-byte trick, whose subtraction borrows *across byte lanes* — a byte equal to `pattern ^ 0x01` right after a matching byte is falsely flagged. Real JSON hits that constantly: `]\` would flag the backslash as `]`, `"#` would flag `#` as a quote. Proptest shrank a failing case to a ~2.7 KB (2,797-byte) quote/backslash byte soup; the fix is a carry-free per-lane detect, and a regression test now enumerates every classified byte followed by its XOR-1 neighbor at every alignment. The second find was better still: a proptest seed shrank to `{"": 2449958197289549825}` — an i64 whose high byte collides with the `"` tape tag — and the bug was in the *test's own* tape-walking comparator. The test suite found a bug in itself.

At four tokens and below, you don't sample the grammar, you enumerate it. The 22,620 exhaustive inputs cover every possible local structure, which is exactly the completeness net the per-token rule table (a 10×10-bit allowed-pairs matrix mirroring the reference row for row) needs — this is the mitigation for the design spec's first-listed risk. The structure milestone's commit message records the outcome: "zero kernel bugs found" — the bugs that round did find were contract bugs (the inert-overflow ordering above), caught by review, not by fuzzing.

The process had a fixed rhythm, visible in the git history: milestone commits M0 through M4 are each immediately followed by an "address adversarial review findings" commit; M5's review-fix commit lands two commits later, with an MIT-license-and-CI-probe commit in between; the serde feature got two review-fix commits of its own. The rhythm paid off starting at M0, before any kernel existed: the first review pass found a soundness hole and fixed it with the type system — `Binding::Read(&GpuBuffer)` / `Binding::ReadWrite(&mut GpuBuffer)` replaced a plain buffer-slice argument so that an exclusive borrow statically prevents a live CPU view of a buffer across a GPU dispatch that mutates it. The review pass was adversarial by instruction — its job was to break the milestone, and the findings above (the inert-overflow counterexample, the 950 ms string, two high-severity safety findings below) are its output. One process artifact even shaped the git history: the review tool caps its input at 1 MB of diff, so the vendored simdjson amalgamation had to be split into its own commit to keep the benchmark milestone reviewable.

The timeline deserves stating plainly, because "milestones" suggests months: the whole project is seventeen commits spanning about two days. M0's scaffold landed 2026-06-11 at 10:45; M5's review fixes landed 2026-06-12 at 09:50 — about 23 hours of commit timeline — and the serde layer followed the same evening, on the repo's only pull request, opened and merged the same day. The design spec is dated the day before the first commit. Every commit ends "Co-Authored-By: Claude Fable 5."

## War stories

**The paravirtual Metal compiler.** CI runs on GitHub's macos-15 runners, which are Apple Silicon — but the GPU is an "Apple Paravirtual device," and from milestone 2 on, its backend shader compiler crashed while building this crate's pipeline state objects (PSOs — the compiled form of a kernel). The same kernels built fine on real hardware. Worse, the failures were nondeterministic: a diagnostic probe that builds every one of the 27 kernels individually reported 10/27 failures on one run, 3/27 on the next, a different first-crashing kernel each time — and once the compile service dies, every later PSO in the process fails. There is no fixing that from my side. The resolution is layered: a documented `METAL_JSON_DISABLE_GPU=1` escape hatch that CI sets (the alternative was failing 89 tests mid-suite); the full CPU-oracle suite still runs on CI, carrying the differential coverage; GPU suites run on real hardware; and the PSO probe stays in CI as a continue-on-error *canary*, with a comment stating the exit condition: when it reports 0/27 failures across a few runs, the VM Metal stack is fixed and the gating can be removed. The probe exits 0 even on failure — "it is a diagnostic, not a gate."

**The gap-byte leak.** The benchmark milestone removed whole-buffer zero-fills and added a buffer pool that recycles GPU memory between parses. The adversarial review found the consequence rated [high]: the string buffer's gap bytes — the slack left when escapes shrink a record — now leaked bytes of a *previously parsed document* through the safe `StringBuffer::as_bytes()` API. An optimization had quietly created an information leak through a safe interface. The fix moves zeroing to the producers: the unescape kernel zero-fills each shrunk record's slot tail as it finishes, and the long-string CPU valve does the same — so the cost tracks escape shrinkage (under 1% at 100 MiB), not buffer size, and the contract is enforced by tests that poison pooled buffers with 0xDB and assert gap bytes read zero.

**The SIGBUS hiding in a safe function.** Same review, second [high]: `parse_file` was a safe function over a copy-on-write mmap. A copy-on-write mapping does not protect against concurrent truncation — if the file shrinks while a kernel touches an unbacked page, the process takes SIGBUS. Undefined behavior reachable from safe code. Now `parse_file` copies (one read straight into a pooled page-aligned GPU buffer; a racing truncation surfaces as an I/O error), and the zero-copy mmap path is an explicitly `unsafe fn parse_file_mmap` with the stable-file contract spelled out. The benchmarks had used the aligned-input path all along, so no number moved.

**Error parity has limits, and they're written down.** On a multi-error input the reference and the GPU can legitimately disagree about *which* error to report — a review-round counterexample was the two-byte input `]"`, which the reference rejects as UnexpectedToken at offset 0 and the GPU as an odd-quote string error. The contract was relaxed to verdict parity on that class — both must reject, the class may differ — with the counterexample checked in as a fixture. Where parity *is* claimed (structural errors: code and offset both), it is tested per error class across a max_depth × inputs sweep.

## Benchmarks, and how to not fool yourself

The benchmark harness got as much paranoia as the parser. The specific fairness measures, each in the report's methodology section:

- **Same work, verified.** Both parsers run a stats walk over their tape inside the timed region — node count, total unescaped string bytes, XOR of all 64-bit number payloads — feeding `black_box` so nothing is dead code. Once per dataset, an *untimed* check asserts both parsers produced bit-identical stats, f64 bit patterns included. Because tape format v1 mirrors simdjson's, the mapping between tapes is the identity; any policy drift fails loudly.
- **No strawman.** The vendored simdjson v4.6.4 builds `-O3` C++17 even under `cargo test`, with `-mcpu=native` on aarch64 so its NEON kernel gets the best local scheduling. It runs through its DOM tape API — the apples-to-apples target for a materialized tape; On-Demand is a lazier contract and would not produce a comparable output (a stated scoping decision, not an omission). One build-system landmine is recorded in the harness: the vendored simdjson directory is deliberately *not* on the include path, because its `VERSION` file would shadow the C++ `<version>` header on case-insensitive filesystems — that is, on the Macs this project targets.
- **All my CPU costs timed.** metal-json's timed region includes every command buffer, every sync, every exact-fit allocation, the number-fixup re-parses, and the long-string valve. Input preparation (one page-aligned copy, made once per dataset) is outside for both sides; simdjson's parser object is reused so its tape allocation is warm, mirroring my buffer pool.
- **Medians, for the spike-C reason.** And one harness footgun documented in the source: criterion's `iter_with_large_drop` accumulates documents, which holds pooled buffers hostage and forces every parse onto cold allocations — measured +25% parse time at 256 MB. Per-iteration drops mirror steady state.

The full sweep, in decimal GB/s (Rust simd-json and serde_json included as context):

```
dataset        size      metal-json  simdjson-cpp  simd-json-rs  serde_json
twitter        0.6 MiB   0.491       3.960         2.783         0.697
twitter_1m     1.0 MiB   1.042       3.200         2.135         0.514
citm_catalog   1.6 MiB   1.901       4.362         3.245         1.101
canada         2.1 MiB   2.179       1.539         1.212         0.753
twitter_4m     4.0 MiB   2.924       3.030         2.158         0.538
twitter_8m     8.0 MiB   4.294       3.210         2.227         0.508
twitter_16m    16.0 MiB  4.961       2.738         2.056         0.526
twitter_64m    64.0 MiB  6.413       2.814         2.048         0.556
twitter_100m   100 MiB   6.400       3.105         2.051         0.536
twitter_256m   256 MiB   7.129       3.261         2.206         0.548
twitter_512m   512 MiB   7.514       2.772         2.162         0.566
```

The report states its own evidential limits in its third paragraph: every row at and above 1 MiB *of the twitter sweep* is a deterministic expansion of the twitter template, measured on this one M5 Max; the other shapes (citm_catalog at 1.6 MiB, canada at 2.1 MiB) were measured only at their canonical sizes; so the ≥100 MiB speedups read as "twitter-shaped documents on this machine," not a universal constant. (The expansion mechanics — records re-serialized by the Cargo.lock-pinned serde_json, cycled verbatim into one top-level JSON array, byte-identical across runs and machines, sha256-pinned — are documented in the report's Datasets section.) That scoping language was itself a review finding — the original headline claimed more than the sweep showed.

Where the time goes on the 512 MiB parse (parse call only, no stats walk):

```
phase                              wall ms   gpu ms   gap ms   % wall
cb1 (K1-K4, bitmaps/tokens)         9.938    3.069    6.869    20.6%
cb2 (K5-K7, scatter/validate)       8.880    8.681    0.199    18.4%
cb2b (K6b, offsets + lists)         3.862    3.564    0.297     8.0%
cb3 (structure+strings+numbers)    25.078   24.779    0.299    51.9%
syncs + allocs + recycle            ~0.03     0        ~0.03    0.1%
unaccounted (encode gaps)           0.530    0        0.530     1.1%
TOTAL                              48.322   40.093    8.229   100.0%

throughput: 11.110 GB/s wall | GPU-execution-only bound: 13.391 GB/s
```

Two readings. First, the sync rows are microseconds — the exact-fit allocation discipline costs nothing at this size. Second, CB1's 6.9 ms wall-vs-GPU gap is CPU encode latency, the largest remaining non-GPU cost and the obvious next target. Per kernel, the top four are `strings_unescape` 6.506 ms (16.8%), `pair_ctx_apply` 5.748 ms (14.9%), `token_validate_footprint` 5.383 ms (13.9%), `apply_tape_offsets` 3.601 ms (9.3%) — string handling and the structural machinery, not the bitmap tricks, which are nearly free (`token_mask` is 1.0%).

Getting here from milestone 4 was its own story: at M4 the parser delivered 2.12 GB/s of wall throughput while its kernels through M3 computed at 11.9 GB/s, because tape copy-out and defensive zero-fills dominated. The kernels were never the problem — stage 1 alone had clocked 52.9 GB/s back at milestone 2; the pipeline around them was. M5 removed both costs — the `Document` now owns the GPU buffers and reads them in place, held alive by an `Arc` pool handle so a live document's memory can never be recycled under it — plus the kernel rounds already mentioned (K11 −27%, K1 UTF-8 −60%, sort pass-2 skip −78%). Net effect on twitter_512m: 232 ms → 52 ms, about 4.5x, in one milestone.

## The serde chapter

After the benchmarks I added a serde deserializer — the Rust idiom where a parser hands values to compile-time-generated visitor code for your concrete types. The design is small and the module doc states it plainly: "This module does not parse JSON text." It deserializes from the already-built tape:

```
JSON bytes --GPU--> tape + unescaped strings --serde walk--> T
```

The document cursor type itself implements `Deserializer`; `deserialize_any` is one match on the tape tag. Strings arrive as `visit_borrowed_str`, so `&str` fields borrow straight from the document's unescaped string buffer with no per-string allocation. Ignored fields skip in O(1) via the container end-index payload — an ignored 10 MB sub-object costs one tape-word read.

The first version was 450 lines and passed its own 225-line test file. Then the review rhythm did its work, in two layers.

The adversarial review pass produced a fix commit listing six findings, each pinned by a test. The two best:

**Borrowed keys were silently impossible.** Object keys went through serde's stock `key.into_deserializer()` helper, which calls `visit_str` — not `visit_borrowed_str` — dropping the document lifetime. Values borrowed fine; keys could not, so `BTreeMap<&str, _>` failed to deserialize. The fix grew into a custom map-key deserializer that also mirrors serde_json's integer-key conventions.

**`size_hint` was an O(n) landmine.** The sequence-access constructor used `Value::len()`, which is O(1) — *except* on a saturated container count (≥ 16,777,215 children), where it walks the container for the exact count. So building the accessor for a giant array did a full tape pre-walk just to feed a size hint that serde's collection code caps anyway. The fix is a new O(1) accessor that returns the saturated count as-is and a hint of `None` when saturated: no hint beats a pre-walk. The tape-format compromise from section three came back as an API design constraint, months — well, hours — later.

The same review also forced a documentation fix rather than a code fix: integers beyond i64/u64 were stored as f64 at parse time (simdjson parity), so `i128`/`u128` targets reject them — unlike serde_json, which parses the full 128-bit range. The module doc now says exactly that.

Duplicate object keys are kept verbatim on the tape (simdjson parity), and then three consumers give three different answers, all pinned by tests: `Value::get` is first-match-wins like simdjson's DOM `at_key`, serde into a map is *last*-write-wins, and serde into a struct is a "duplicate field" error. The differential suite has to know this too — duplicate-key documents are excluded from the serde_json comparison and asserted on the raw tape instead.

Then the second layer. The review-fix commit had fixed tuples to reject extra array elements (`(i64,i64)` from `[1,2,3]` had silently returned `(1,2)`), and the *same commit* added serde_json-parity positional structs — without that arity check on the two new paths. An automated PR review (Codex) caught both: `[1,2,3]` into a two-field struct, and `{"V":[1,2,3]}` into a two-field struct variant, silently dropped the extras. The fix factored one `visit_seq_exact` helper used by all three positional paths. The first review found the bug class; the fix reintroduced two instances of it; an independent second reviewer caught those. One review layer was demonstrably not enough.

The architectural insight worth keeping: **static type information cannot reach the parse.** serde_json fuses parse and deserialize in one text pass, so knowing that a field is skipped can avoid work during parsing. Here the GPU builds the complete tape and unescaped string buffer for the whole document before any type-specific code runs — the parse is type-blind by construction, and `T`'s shape only accelerates the walk afterward. The benchmark shows what that costs at small sizes: on a deterministic ~4 MiB synthetic corpus (checksum-verified across all contenders; the PRNG seed spells "metal-js" in ASCII), end-to-end into typed structs on the M5 Max:

```
contender                    throughput
serde_json borrowed          ~972 MiB/s
serde_json owned             ~614 MiB/s
metal-json borrowed (GPU)    ~384 MiB/s
metal-json owned (GPU)       ~349 MiB/s
```

At 4 MiB, plain serde_json beats the GPU path about 2.5x end-to-end, like for like (borrowed vs borrowed; owned vs owned is about 1.8x). The PR description itself states no multiplier — just "at 4 MiB the GPU dispatch overhead dominates and serde_json wins end-to-end." It is the same curve as the main benchmark, told from the losing side; larger documents remain the design point. Note the deltas too: borrowing saves serde_json ~37% of its time but metal-json only ~9%, because metal-json's owned path only pays for `String` allocations during the walk — the parse is identical either way.

## The compromise ledger

Every system is its compromises. Here are this one's, collected:

```
compromise                         gave up                        got
u32 token/tape indices             inputs > 4 GiB - 65 B          half the index traffic; 32-bit
                                                                  atomics on every Apple GPU
child counts saturate at           O(1) exact len for huge        count + end index in one word;
16,777,215                         containers (must walk)         simdjson parity
big integers become f64            exactness past u64; serde      simdjson parity; 2-word numbers
                                   i128/u128 rejects beyond
string slots by raw length         gap bytes of buffer slack      offsets computable in parallel
(not dense, unlike simdjson)                                      before unescaping runs
gap zero-fill at producers         write cost ~ escape            no cross-parse data leak via a
                                   shrinkage (<1% at 100 MiB)     safe API over pooled buffers
escape look-back cap 4096 B        slow path on backslash walls   bounded look-back; near-free
+ fixup kernel                     (cost timed in benchmarks)     valve on benign input
strings >16 KiB to CPU valve       CPU work for rare giants       bounded GPU serial tail
                                                                  (8 MB string: 950 ms -> 83 ms)
hard floats to CPU fixup list      CPU re-parse of rare           no big-decimal path in-kernel
                                   literals (timed)
no fp64 anywhere                   (not a choice - hardware)      f64 bit patterns via int math
overflow depths clamp into         inert-element machinery        no 3rd sort pass on every parse
the legal sort-key range           + a written proof              at the 1024 default
K5/K6b get own command buffers     ~50-160 us extra sync each     exact-fit allocations, never
                                                                  input-proportional guesses
buffer pool never shrinks          high-water-mark memory         steady-state zero allocations
                                                                  (cold allocs cost +25% @256 MB)
safe parse_file copies;            one copy on the safe path      no SIGBUS reachable from
mmap path is unsafe fn                                            safe code
AOT shaders default                Xcode Metal toolchain to       no runtime compile; runtime-
                                   build from source              shaders feature for prebuilt
GPU tests off hosted CI            CI exercises CPU oracle only   no nondeterministic VM compiler
                                                                  crashes; PSO-probe canary watches
GPU at all                         everything below ~4.3 MiB      2.1-2.7x above 100 MiB
```

## What I learned, and what's next

Four short lessons, then the long one. Measure before you design: the three spikes cost a day, settled three arguments permanently, and spike C's ~50–160 µs per sync priced every architecture decision that followed. Shape the oracle like the thing you test: per-stage bit-for-bit diffing caught most GPU bugs as one stage's buffer diverging from the oracle's, not as a wrong parse three stages later. Valves, not limits: adversarial inputs — backslash walls, megabyte strings, 600-plus-digit floats — all get bounded fast paths whose costs sit inside the benchmark numbers, and either alternative — rejecting, or unbounded slowness — would be simpler and worse. And hostile review on a schedule finds real bugs, with the serde arity story adding a corollary: one review layer isn't enough, because the fix for a bug class reintroduced the class and only an independent second reviewer caught it.

The long lesson: **the crossover is the product spec.** A GPU parser is not a faster JSON parser; it is a different point on a latency/throughput curve, with a fixed cost floor around half a millisecond that no kernel work removes. Saying "use simdjson below ~4 MiB" in the README is not a confession, it's the documentation.

What's next, in rough order of expected value: shrink CB1's 6.9 ms CPU encode gap (the largest non-GPU cost at 512 MiB); large-size sweeps of non-twitter shapes, since shape dependence is measured fact at small sizes and asserted hope at large ones; possibly folding command buffers to move the crossover left, now that small-input latency has a user in the serde path; and watching the CI canary — the day the paravirtual Metal compiler stops crashing, 27/27 kernels build, and the GPU suite comes back to hosted CI.

The code, the spike programs, the bench harness, and the reports it generates are all in the repository. Every number in this article is in a committed document or a commit message; if you have an M-series Mac, `cargo run -p xtask -- bench-report` regenerates the whole table on your machine — medians, per spike C.
