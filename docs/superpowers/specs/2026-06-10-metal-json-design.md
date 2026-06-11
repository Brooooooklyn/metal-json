# metal-json: GPU JSON parser on Metal, faster than C++ simdjson

## Context

Build a Rust library at `/Users/brooklyn/workspace/github/metal-json` (currently an empty scaffold) that parses standard JSON documents on the Apple Silicon GPU via Metal and is **faster than C++ simdjson parse-to-tape on large inputs (MBs–1GB)**. Reference: `/Users/brooklyn/workspace/github/cuJSON` (CUDA research parser).

User decisions (confirmed):
- **Full tape, apples-to-apples**: GPU produces a simd-json-equivalent typed tape — parsed i64/u64/f64 numbers, validated + unescaped strings, full grammar validation. (cuJSON only emits a structural index; we go further.)
- **Baseline**: C++ simdjson (vendored amalgamation, in-process FFI bench).
- **Workload v1**: large standard JSON documents. JSON Lines later.
- **Surface**: Rust crate only.

Why this can win: Apple unified memory kills the H2D/D2H cost that cripples discrete-GPU parsers (`MTLBuffer bytesNoCopy`, zero copy). Pipeline memory traffic ≈ 9 bytes/input byte; on this machine (M5 Max, verified) realistic throughput is 24–40 GB/s vs simdjson's ~3–7 GB/s. Crossover vs CPU is around a few MB — the bench report shows it honestly.

## Architecture

```
mmap/aligned input ──> MTLBuffer bytesNoCopy (storageModeShared, 16KB pages)

CB1: K1 classify+escape+UTF8 (1 pass → quote/candidate bitmaps, chunk partials)
     K2 spine scan (quote parity)   K3 token mask (in-string via prefix-XOR)
     K4 spine scan (token counts)
  ── CPU sync: read token count t → exact-size allocations ──
CB2: K5 token scatter (positions + kinds)
     K6 local validation + tape footprints + skeleton/string/scalar lists
     K7 spine scan (tape words, stringbuf bytes, list counts)
  ── CPU sync: tape/stringbuf sizes ──
CB3: K8 counting sort of skeleton by depth (histogram→scan→stable scatter)
     K9 pair map + container context + child counts (segmented ops)
     K10 number parse (integer Eisel-Lemire → f64 BIT PATTERN; hard cases → fixup list)
     K11 string unescape (fast no-escape path; \uXXXX + surrogates)
     K12 container tape words (via pair map)   K13 finalize/root/error-reduce
CPU: check error word, fix up rare hard floats, return Document
```

Key algorithm decisions (from design review):
- **Scans**: hierarchical reduce→spine→apply, fused into producer/consumer kernels. NOT decoupled look-back (no forward-progress guarantee on Apple GPUs). Spine = 1 tiny threadgroup using `simd_prefix_exclusive_sum`.
- **Bracket matching**: NO comparison sort (cuJSON uses Thrust stable_sort). Depth ≤ 1024 → **stable counting sort by depth** (5-bit digits, 1–2 passes); within a depth group brackets alternate open/close → pairs adjacent. Sorted stream is the **skeleton** (brackets + colons + commas), so one sort gives pairing AND comma/colon container context.
- **Numbers, no fp64 on Apple GPUs**: Eisel-Lemire is pure 64-bit integer math; compute the f64 **bit pattern** as `ulong`. 128-bit products via `mulhi(ulong)` or 4×`mulhi(uint)` limbs (spike decides). Pow5 table ~10KB in `constant` memory. Truncated ≥20-digit mantissas / ambiguous cases → atomic-appended fixup list, CPU re-parses few stragglers (Rust `str::parse::<f64>` correctness path).
- **Strings**: open/close quotes are adjacent tokens → extent free. Unescaped len ≤ raw len → stringbuf offsets from raw-length prefix sum (gaps OK). Fast path: no backslash → vector copy + control-char check. Escape path: thread-per-string sequential unescape with full surrogate-pair validation.
- **Grammar validation** (simdjson parity): Layer 1 in K6 = local token-adjacency table + colon 4-token rule + literal `true/false/null` byte check. Layer 2 in K9 = comma context via segmented forward-fill of opener type per depth group. Layer 3 = number grammar (K10), escapes (K11), UTF-8 (K1), depth/balance (scan), trailing content (K13). Errors: `atomic_min` on packed u64 (offset<<32|code) → earliest error wins deterministically.
- **Error handling**: structured `Error` enum with byte offsets (Utf8, Syntax{kind}, DepthLimit, TrailingContent, InputTooLarge, NoDevice…). Never exit/panic on bad input (unlike cuJSON).

## Tech choices

- **Bindings**: `objc2-metal` 0.3.2 (+ objc2 0.6.4, objc2-foundation 0.3.2, block2 0.6.2, dispatch2 0.3.1). gfx-rs `metal` crate is officially deprecated. Confine raw bindings to `src/metal/` wrappers; may need audited `unsafe impl Send/Sync` wrappers (wgpu precedent).
- **Shaders**: AOT `.metal → .metallib` in build.rs via `xcrun -sdk macosx metal` (verified working on this machine), embedded with `include_bytes!`. `runtime-shaders` feature = runtime MSL compile + `METAL_JSON_SHADER_DIR` hot-reload for dev iteration, and fallback for machines without the Metal toolchain. Specialization via MTLFunctionConstantValues.
- **Workspace**: root package = core lib; members `bench/` (criterion + vendored simdjson via `cc`, publish=false) and `xtask/` (fetch-data, gen-data, bench-report).
- Other deps: memmap2 0.9.10, thiserror 2.0.18; dev: serde_json 1.0.150 (preserve_order, arbitrary_precision), proptest 1; bench: criterion 0.8.2, simd-json 0.17.0 (secondary baseline), cc 1.2.63.

## File tree (to create)

```
Cargo.toml            workspace + package        build.rs   AOT shader compile + toolchain probe
shaders/common.h, tape_types.h, 01_classify.metal … NN_tape.metal   (~10–15 numbered kernels)
src/lib.rs  error.rs  parser.rs (Parser, buffer pool, orchestration)
src/document.rs  value.rs (Value<'doc> cursor API)  tape.rs (u64 entries; mirrors tape_types.h)
src/input.rs (AlignedBuffer, mmap)  stage.rs (per-stage encode abstraction → kernel unit tests)
src/metal/{context,pipeline,buffer,timing}.rs
src/reference/        cpu-reference feature: scalar Rust oracle per stage + full pipeline backend
tests/{kernels,jsontestsuite,differential,numbers}.rs   corpus/  examples/
bench/{Cargo.toml,build.rs,cpp/{vendor/simdjson.{h,cpp},shim.cpp,main.cpp},src/lib.rs,benches/compare.rs}
xtask/src/main.rs     scripts/fetch_jsontestsuite.sh fetch_simdjson_data.sh update_simdjson_amalgamation.sh
data/ (.gitignored)
```

## Public API (keystone signatures)

```rust
let parser = Parser::new()?;                       // device + PSOs, reusable, Send+Sync
let doc = parser.parse(&bytes)?;                   // copies into pooled aligned buffer unless provably aligned
let doc = parser.parse_file("big.json")?;          // mmap, guaranteed zero-copy
let mut buf = parser.alloc_input(cap);             // page-aligned input the caller fills → parse_aligned()
let v = doc.root();                                 // Value<'doc>: kind/get/at/as_i64/u64/f64/str/bool/len/entries/elements
```

`Document` owns tape + stringbuf MTLBuffers (shared storage → CPU reads them directly), self-contained (never borrows input), buffers return to the parser pool on drop. Tape: u64 entries, type tag in high byte, containers store matching-index + count (O(1) skip/len), strings pack (offset:40,len:24) into stringbuf.

## Milestones (execute in order; each has its own verification)

**M0 — Scaffold + spikes (de-risk first).**
Workspace conversion, build.rs AOT pipeline, `MetalContext` (device/queue/library), one smoke kernel, CI skeleton (macos-15 Apple Silicon runners have real Metal GPUs; `MTL_SHADER_VALIDATION=1` in tests).
Spikes (decide before building on them): (a) `mulhi(ulong)` availability/codegen vs 4-limb `mulhi(uint)` microbench; (b) `ulong` bitmap ops vs 2×`uint` formulation; (c) command-buffer + 14-dispatch overhead measurement.
✓ Verify: `cargo test` runs a GPU dispatch; spike numbers recorded in `docs/spikes.md`.

**M1 — Tape format + CPU reference pipeline + API.**
`tape.rs` (+ layout test vs `shaders/tape_types.h`), `Document`/`Value`, `Error`, full scalar reference backend (`Backend::CpuReference`) producing the exact target tape. This is the oracle everything else diffs against.
✓ Verify: differential tests vs serde_json pass on corpus + JSONTestSuite with CPU backend; number torture table passes via the CPU fixup path logic.

**M2 — GPU stage 1: bitmaps + scans + token extraction (K1–K5).**
Classify/escape/UTF-8 fused kernel, spine scans, in-string mask, token scatter. Backslash-run carry: cap look-back at 4KB with sequential fix-up valve for the adversarial case.
✓ Verify: per-kernel diff vs reference on corpus + proptest-generated JSON + adversarial cases (quote/backslash walls, chunk-boundary straddles), under MTL_SHADER_VALIDATION.

**M3 — GPU structure: validation + sort + pairing + container tape (K6–K9, K12–K13).**
Local rule table (pre-verify rule completeness by enumerating short token sequences against a reference validator script), counting sort, segmented pair/context kernel, container emission.
✓ Verify: JSONTestSuite y_/n_ accept/reject parity (GPU backend) for structural cases; pair maps diff vs reference.

**M4 — GPU scalars: numbers + strings (K10–K11) → full GPU tape.**
Eisel-Lemire kernel + pow5 table + fixup list + CPU patch pass; string fast/escape paths.
✓ Verify: full JSONTestSuite parity, whole-document differential vs serde_json (f64 bit-exact vs `str::parse::<f64>`), proptest roundtrips, duplicate-key tape tests.

**M5 — Benchmarks + optimization until the claim holds.**
Vendored simdjson shim (in-process FFI, checksum to defeat DCE, reused parser), criterion compare (metal-json / C++ simdjson / Rust simd-json / serde_json), `xtask gen-data` (deterministic 100MB–1GB from twitter/canada templates; cuJSON samples at ../cuJSON/dataset), `xtask bench-report` → GB/s table + per-kernel breakdown (MTLCounterSampleBuffer behind `timing` feature) + size sweep exposing the CPU/GPU crossover. Optimize hot kernels with Xcode GPU capture / Metal System Trace until metal-json > simdjson on ≥100MB datasets.
✓ Verify: `cargo xtask bench-report` table shows metal-json GB/s > simdjson GB/s on large datasets on this machine; README documents crossover size honestly.

## Top risks & mitigations

1. **Parallel grammar-validation completeness** → model-check the adjacency rules against a reference validator before writing K6; JSONTestSuite + differential fuzz on every kernel change.
2. **64-bit integer perf on 32-bit Apple ALUs** (bitmaps + Eisel-Lemire) → M0 spikes; design ports to u32 words with constant changes only.
3. **Adversarial inputs** (`[[[[…` → skeleton ≈ n; backslash walls) → exact allocation after CB1 sync, look-back valve, synthetic worst-case benches early; depth limit 1024 = simdjson parity.

## Process notes

- Implementation via subagent-driven development (user preference: subagents must read full context — this plan, the cuJSON reference, and prior milestone code — before writing).
- Code review: run `/codex:adversarial-review` after each milestone.
- Commit design doc to `docs/superpowers/specs/2026-06-10-metal-json-design.md` as part of M0 (plan mode forbade writing it now).
