# metal-json benchmark report

Parse-to-tape throughput: metal-json (GPU) vs C++ simdjson (vendored, FFI) vs Rust simd-json vs serde_json. All numbers are **medians of criterion samples** in decimal GB/s (input bytes / 1e9 / seconds), measured back-to-back in one session by `cargo run -p xtask -- bench-report`. The timed region for the two headline contenders is *parse to tape + an identical shallow stats walk over the resulting tape* — see [Methodology](#methodology) for exactly what is and isn't timed.

**Headline:** on this machine metal-json parses the twitter_512m (512.0 MiB) document **2.71× faster** than C++ simdjson (71.447 ms vs 193.711 ms). The win holds across the ≥100 MiB sweep below. **Crossover caveat:** below 4.3 MiB of input, simdjson wins — the GPU pipeline carries a fixed dispatch/sync overhead that small documents cannot amortize.

**Scope of evidence:** every row at and above 1 MiB of the twitter sweep — including all ≥100 MiB rows — is a deterministic expansion of the *twitter template* measured on this one M5 Max; the other document shapes (citm_catalog, canada) were measured only at their canonical 1.6–2.1 MiB sizes. Shape already matters at those sizes (number-dense canada favors metal-json at 2.1 MiB, citm_catalog favors simdjson), so the ≥100 MiB speedups should be read as "twitter-shaped documents on this machine", not as a universal large-input constant; large-size coverage of other shapes is future work.

## Environment

- **Date**: 2026-06-12
- **Machine**: Apple M5 Max, 128 GiB unified memory
- **OS**: macOS 26.5.1
- **Rust**: rustc 1.95.0 (59807616e 2026-04-14)
- **Metal toolchain**: Apple metal version 32023.883 (metalfe-32023.883)
- **simdjson (vendored)**: v4.6.4 (amalgamation, cc -O3)
- **metal-json backend**: gpu (default)

## Datasets

Canonical files come from [simdjson-data](https://github.com/simdjson/simdjson-data) via `cargo run -p xtask -- fetch-data`. Large/sweep variants are deterministic expansions produced by `cargo run -p xtask -- gen-data`: records from the canonical template (twitter `statuses`), re-serialized by the Cargo.lock-pinned serde_json (sorted keys) and cycled verbatim into one top-level JSON array until the target size is reached — byte-identical across runs and machines, valid standard JSON at every size.

| dataset | bytes | sha256 | source |
|---|---:|---|---|
| twitter | 631515 | `30721e496a8d73cfc50658923c34eb2c0fbe15ee6835005e43ee624d8dedf200` | simdjson-data canonical (`xtask fetch-data`) |
| twitter_1m | 1051040 | `c43ecbda6c118400108a1d24f2c2c86e0464650f62114166b3251f1ac6ddfcdb` | generated: `xtask gen-data --template twitter --size 1m` |
| citm_catalog | 1727204 | `a73e7a883f6ea8de113dff59702975e60119b4b58d451d518a929f31c92e2059` | simdjson-data canonical (`xtask fetch-data`) |
| canada | 2251051 | `f83b3b354030d5dd58740c68ac4fecef64cb730a0d12a90362a7f23077f50d78` | simdjson-data canonical (`xtask fetch-data`) |
| twitter_4m | 4195935 | `426e953d03a5eb5b6a02c2d30b8e162666b58490d1e60b7b0d6daa6ef8f60ff6` | generated: `xtask gen-data --template twitter --size 4m` |
| twitter_8m | 8395011 | `a22d9e1c9be85fd4b076431c0a4fa760c6d5dc5650cda87656145f1b15b06577` | generated: `xtask gen-data --template twitter --size 8m` |
| twitter_16m | 16781382 | `c015a0615f58eb6031bfbf621f0a0d599162fdc3c892a9bfc63dded2433c2b7f` | generated: `xtask gen-data --template twitter --size 16m` |
| twitter_64m | 67112686 | `0063ddf7a49b60f7658316fafda772b1c9fd12e98fe6fed9d8a016912f8d1ddf` | generated: `xtask gen-data --template twitter --size 64m` |
| twitter_100m | 104861712 | `c05b2c677a4c70622c01820ac5c50cf2db02570e3982295836f67e19650d9457` | generated: `xtask gen-data --template twitter --size 100m` |
| twitter_256m | 268440412 | `730618522202c075cfb34ac56ab7cd805aa753dff4460a3d0cf5b3072580d9e0` | generated: `xtask gen-data --template twitter --size 256m` |
| twitter_512m | 536871026 | `498fa5aec36849e52de5da820a750f40c25b5be5947c1bc2cf96eccd80d3cd40` | generated: `xtask gen-data --template twitter --size 512m` |

## Results (median GB/s, higher is better)

| dataset | size | metal-json GB/s | simdjson-cpp GB/s | simd-json-rust GB/s | serde-json GB/s | metal-json / simdjson-cpp |
|---|---:|---:|---:|---:|---:|---:|
| twitter | 0.6 MiB | 0.491 | 3.960 | 2.783 | 0.697 | 0.12x |
| twitter_1m | 1.0 MiB | 1.042 | 3.200 | 2.135 | 0.514 | 0.33x |
| citm_catalog | 1.6 MiB | 1.901 | 4.362 | 3.245 | 1.101 | 0.44x |
| canada | 2.1 MiB | 2.179 | 1.539 | 1.212 | 0.753 | 1.42x |
| twitter_4m | 4.0 MiB | 2.924 | 3.030 | 2.158 | 0.538 | 0.96x |
| twitter_8m | 8.0 MiB | 4.294 | 3.210 | 2.227 | 0.508 | 1.34x |
| twitter_16m | 16.0 MiB | 4.961 | 2.738 | 2.056 | 0.526 | 1.81x |
| twitter_64m | 64.0 MiB | 6.413 | 2.814 | 2.048 | 0.556 | 2.28x |
| twitter_100m | 100.0 MiB | 6.400 | 3.105 | 2.051 | 0.536 | 2.06x |
| twitter_256m | 256.0 MiB | 7.129 | 3.261 | 2.206 | 0.548 | 2.19x |
| twitter_512m | 512.0 MiB | 7.514 | 2.772 | 2.162 | 0.566 | 2.71x |

Contenders present: metal-json, simdjson-cpp, simd-json-rust, serde-json.

## CPU/GPU crossover (twitter size sweep)

metal-json vs C++ simdjson on the twitter template across input sizes. The GPU pipeline pays a roughly fixed dispatch/sync overhead per parse (~0.5–0.9 ms on this machine), so small documents lose to the CPU; throughput grows with size until the pipeline is memory-bound.

| dataset | size | metal-json median | simdjson-cpp median | metal-json GB/s | simdjson-cpp GB/s | speedup | winner |
|---|---:|---:|---:|---:|---:|---:|---|
| twitter | 0.6 MiB | 1.287 ms | 0.159 ms | 0.491 | 3.960 | 0.12x | simdjson |
| twitter_1m | 1.0 MiB | 1.009 ms | 0.328 ms | 1.042 | 3.200 | 0.33x | simdjson |
| twitter_4m | 4.0 MiB | 1.435 ms | 1.385 ms | 2.924 | 3.030 | 0.96x | simdjson |
| twitter_8m | 8.0 MiB | 1.955 ms | 2.615 ms | 4.294 | 3.210 | 1.34x | **metal-json** |
| twitter_16m | 16.0 MiB | 3.383 ms | 6.129 ms | 4.961 | 2.738 | 1.81x | **metal-json** |
| twitter_64m | 64.0 MiB | 10.465 ms | 23.852 ms | 6.413 | 2.814 | 2.28x | **metal-json** |
| twitter_100m | 100.0 MiB | 16.386 ms | 33.772 ms | 6.400 | 3.105 | 2.06x | **metal-json** |
| twitter_256m | 256.0 MiB | 37.654 ms | 82.308 ms | 7.129 | 3.261 | 2.19x | **metal-json** |
| twitter_512m | 512.0 MiB | 71.447 ms | 193.711 ms | 7.514 | 2.772 | 2.71x | **metal-json** |

**Crossover: ≈ 4.3 MiB.** Below that size C++ simdjson wins (the GPU's fixed overhead dominates); above it metal-json wins, with the gap widening toward large inputs. The estimate interpolates linearly between the neighboring sweep sizes where the winner flips; treat it as a band around that value, not a sharp constant — it moves with document shape and machine.

## Where the time goes (largest input)

Phase-level wall/GPU split of `Parser::parse_aligned` on `twitter_512m` (`cargo run --release --features timing --example parse_breakdown`, median of 9 iterations after 3 warmups). This times **the parse call only** — the symmetric stats walk of the criterion harness is not included, so the total here is faster than the table above.

```text
/Users/brooklyn/workspace/github/metal-json/data/bench/twitter_512m.json — 536871026 bytes, 9 iters, input mode aligned (medians)

phase                                         wall ms     gpu ms     gap ms  % wall
stage1 alloc (zero-copy input)                  0.002      0.000      0.002    0.0%
cb1 (K1-K4)                                     9.938      3.069      6.869   20.6%
sync1: header + token/scratch alloc             0.003      0.000      0.003    0.0%
cb2 (K5-K7)                                     8.880      8.681      0.199   18.4%
sync2: header read                              0.001      0.000      0.001    0.0%
cb2b (K6b) + list alloc                         3.862      3.564      0.297    8.0%
sync2: tape/scratch alloc                       0.006      0.000      0.006    0.0%
cb3 (structure + strings + numbers)            25.078     24.779      0.299   51.9%
sync3: verdict + fixup patches                  0.002      0.000      0.002    0.0%
scratch recycle (stages 2-3)                    0.001      0.000      0.001    0.0%
scratch recycle (stage1)                        0.019      0.000      0.019    0.0%
(unaccounted: encode-call gaps etc.)            0.530      0.000      0.530    1.1%
TOTAL parse wall                               48.322     40.093      8.229  100.0%

throughput: 11.110 GB/s wall | GPU-execution-only bound: 13.391 GB/s
```

Per-kernel GPU execution times (`METAL_JSON_SPLIT_KERNELS=1` measurement mode: each dispatch gets its own command buffer + sync, so *wall* time inflates and only the GPU column is representative; phase numbers from that mode are therefore not shown):

```text
per-kernel GPU times (METAL_JSON_SPLIT_KERNELS=1, medians of 9):

#    kernel                               gpu ms    % gpu
0    classify_escape_utf8                  2.616     6.8%
1    escape_carry_fixup                    0.114     0.3%
2    spine_quote_scan                      0.011     0.0%
3    token_mask                            0.375     1.0%
4    spine_token_scan                      0.009     0.0%
5    token_scatter                         2.625     6.8%
6    token_validate_footprint              5.383    13.9%
7    spine3                                0.253     0.7%
8    apply_tape_offsets                    3.601     9.3%
9    depth_partials                        0.222     0.6%
10   depth_spine                           0.049     0.1%
11   depth_apply                           0.655     1.7%
12   sort_hist                             0.307     0.8%
13   sort_matrix_scan                      0.554     1.4%
14   sort_scatter                          1.161     3.0%
15   sort_hist                             0.078     0.2%
16   sort_matrix_scan                      0.002     0.0%
17   sort_scatter                          0.560     1.4%
18   ctx_partials                          1.620     4.2%
19   ctx_spine                             0.084     0.2%
20   pair_ctx_apply                        5.748    14.9%
21   emit_container_words                  1.507     3.9%
22   tape_root_words                       0.002     0.0%
23   structure_finalize                    0.041     0.1%
24   string_record_offsets                 1.479     3.8%
25   strings_unescape                      6.506    16.8%
26   structure_finalize                    0.060     0.2%
27   parse_numbers                         3.001     7.8%
     SUM                                  38.623   100.0%
```

## Methodology

Harness: `bench/benches/compare.rs` (criterion); helpers and the FFI shim
contract in `bench/src/lib.rs` + `bench/cpp/shim.cpp`.

**Timed region, metal-json** — one call to `Parser::parse_aligned` plus a
shallow stats walk over the produced tape (`metal_stats`):

- *Inside the timed region*: the whole parse — all GPU command buffers
  (encode + commit + `waitUntilCompleted`), the CPU syncs between them,
  exact-size output allocations, **all CPU fixup costs** (hard-rounding
  float re-parses; the >16 KiB long-string valve, which unescapes
  fixup-listed long strings on the CPU so one giant string cannot
  serialize a GPU lane), `Document` assembly, and the stats walk.
- *Outside the timed region*: input preparation (one page-aligned copy of
  the file, made once per dataset — the zero-copy `bytesNoCopy` input
  path then maps it straight into an `MTLBuffer`), and `Document` drop
  (which returns pooled buffers; the reused simdjson parser's tape is
  likewise never freed inside its timed call).

**Timed region, simdjson (C++)** — one call to `sj_parse_tape`: a reused
`simdjson::dom::parser` parses a pre-padded buffer to its tape, then walks
that tape linearly filling the same stats struct. Padding the input
(`SIMDJSON_PADDING`) happens once per dataset, untimed. The parser object
is reused across iterations so its tape allocation is warm — mirroring
metal-json's reused parser and buffer pool.

**Symmetric stats walk (DCE defeat + proof of equivalent work)** — both
contenders compute the same `SjStats` (node count, total unescaped string
bytes, XOR of all 64-bit number payloads) inside the timed region, and the
results feed `black_box`. Once per dataset, an **untimed** check asserts
both parsers produce bit-identical stats — both really parsed the same
document to an equivalent tape (bit-exact f64s included). Numbers quoted
as "parse-only" anywhere strip this walk and are labeled as such.

**Other contenders** — Rust `simd-json::to_tape` mutates its input, so
each iteration gets a fresh copy in untimed setup (the API has no
non-destructive tape parse); output drop is untimed. `serde_json` parses
to a DOM (`Value`) — an allocation-heavy floor, not a tape peer; drop is
untimed.

**Sampling** — criterion defaults (60 samples) for inputs <10 MiB;
20 samples / 10 s for 10–100 MiB; 10 samples / 20 s + per-iteration
batching for ≥100 MiB. Warmup precedes every measurement (criterion
default 3 s; 2 s on ≥100 MiB groups), which also absorbs GPU power-state
ramp and PSO/pool warming. **Medians** everywhere: low-occupancy GPU wall
times jitter up to 4× from power-state ramping (see `docs/spikes.md`), so
means would overweight outliers. All contenders for all datasets run
back-to-back in a single session. Background desktop activity during the
session is not controlled for beyond using medians (it hits both
contenders); the sub-4 MiB wall times and the exact crossover point are
the numbers most sensitive to it, the ≥100 MiB headline the least.

**Honesty notes** —

- Throughput is decimal (GB = 1e9 bytes); sizes in tables are binary MiB.
- metal-json numbers *include* every CPU-side cost of the hybrid design
  (syncs, allocations, fixups, copy-out); only input preparation is
  excluded, identically for both headline contenders.
- The per-kernel breakdown times the bare parse call (no stats walk) and
  says so; split-kernel mode adds one sync per kernel, so only its GPU
  column is meaningful.
- The crossover is a property of *this* machine and document shape;
  the report states the measured band rather than a universal constant.
- simdjson is run through its DOM tape API (`dom::parser`), the closest
  apples-to-apples target for a full materialized tape; On-Demand is a
  different (lazier) contract and would not produce a comparable tape.
