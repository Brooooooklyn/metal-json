# Tape format v1

The parse result of metal-json is a **tape** (a flat sequence of `u64` words)
plus a **string buffer** (a flat byte array of unescaped string records). The
layout is simdjson's tape layout, chosen deliberately so the M5 benchmark
compares apples to apples: both parsers do the same amount of output work.

Canonical definitions, kept in lock-step by tests:

| Artifact | Role |
|---|---|
| `src/tape.rs` | canonical constants + encode/decode helpers (Rust) |
| `shaders/tape_types.h` | the same constants/helpers for MSL kernels |
| this document | prose spec + worked example |

The `msl_header_layout_lock` test in `src/tape.rs` parses
`shaders/tape_types.h` at test time and fails on any constant mismatch (in
either direction). The `worked_example_matches_tape_format_doc` test pins the
worked example below word-for-word and byte-for-byte.

## Tape word encoding

Every tape entry is one little-endian `u64`:

```text
bits 63..56   tag (ASCII byte)
bits 55..0    payload (56 bits)

word = ((tag as u64) << 56) | payload
```

### Tags

| Tag | ASCII | Meaning | Payload |
|---|---|---|---|
| `r` | `0x72` | root (first and last word) | see [Root words](#root-words) |
| `{` | `0x7B` | object open | see [Containers](#containers) |
| `}` | `0x7D` | object close | see [Containers](#containers) |
| `[` | `0x5B` | array open | see [Containers](#containers) |
| `]` | `0x5D` | array close | see [Containers](#containers) |
| `"` | `0x22` | string (key or value) | string-buffer record offset (56 bits) |
| `l` | `0x6C` | `i64` number | 0; **next word** = the `i64` bits (two's complement) |
| `u` | `0x75` | `u64` number | 0; **next word** = the `u64` value |
| `d` | `0x64` | `f64` number | 0; **next word** = the IEEE-754 `f64` bits |
| `t` | `0x74` | `true` | 0 |
| `f` | `0x66` | `false` | 0 |
| `n` | `0x6E` | `null` | 0 |

Numbers occupy **two** tape words; everything else occupies one.

## Root words

- `tape[0]` has tag `r` and payload = the **index of the final root word**
  (equivalently: `tape.len() - 1`). This matches the simdjson tape
  documentation's example, where the leading root points at the trailing
  root entry.
- The final tape word has tag `r` and payload `0` (the index of `tape[0]`).

A valid tape therefore always has at least 3 words (e.g. `null` parses to
`r`, `n`, `r`).

## Containers

Object/array **open** words (`{`, `[`) pack two fields into the payload:

```text
bits 31..0    index ONE PAST the matching close word
bits 55..32   count of direct children, saturated at 0xFFFFFF
```

- "One past the matching close word" means: for `{` at index `i` whose `}`
  lands at index `j`, the open payload's low 32 bits hold `j + 1` — the index
  of the first tape word *after* the container. This gives O(1) skip.
- The **count** is the number of direct children only (not descendants):
  for objects, the number of key–value members; for arrays, the number of
  elements. Counts of `0xFFFFFF` (16,777,215) or more saturate to `0xFFFFFF`
  and mean "this many **or more**" — consumers needing the exact count of a
  saturated container must walk it (simdjson parity).

Object/array **close** words (`}`, `]`) hold the index of the matching open
word in payload bits 31..0 (upper payload bits are 0).

## Strings

A `"` word's payload is a byte offset (56 bits) into the **string buffer**
where this record starts:

```text
[u32 LE length][content bytes][NUL]
```

- `length` counts only the content bytes (not the header, not the NUL).
- The content is fully **unescaped** UTF-8: `\n`, `\uXXXX`, surrogate pairs
  etc. are already resolved. It may contain interior NUL bytes (from
  `\u0000`) — that is why the explicit length exists; the trailing NUL is a
  C-string convenience only.
- Object keys and string values use the same encoding; records appear in the
  buffer in document order.

### Offset allocation (pinned policy)

Record offsets are **not** densely packed by unescaped length. They are
allocated by raw input length, so the GPU can compute every offset from
token positions alone — *before* any unescaping runs:

```text
raw_len(s)  = byte count between the quotes of s in the INPUT (escapes
              still escaped)
slot(s)     = raw_len(s) + 5          (+5 = 4-byte length prefix + NUL)
offset(s)   = exclusive prefix sum of slot over strings in document order
buffer size = Σ slot(s)               (over all strings)
```

Unescaped content is always ≤ raw content, so a record whose escapes
shrank it does not fill its slot: the bytes between its NUL and the next
slot (or the buffer end, for the last string) are a **gap**.

- Gap bytes are **zero-filled on both backends**: the CPU reference
  zero-fills as it emits, and the GPU string kernel (plus the long-string
  CPU valve) zero-fills each shrunk record's slot tail as it finishes the
  record. The cost is proportional to actual escape shrinkage, never to
  buffer size — there is no whole-buffer memset.
- This is a hard requirement, not a convenience: GPU buffers come from a
  reuse pool without zeroing, and the whole buffer (gaps included) is
  reachable through the safe `StringBuffer::as_bytes` API — unspecified
  gaps would leak bytes of a previously parsed document.
- Equal documents therefore produce **byte-identical string buffers** on
  both backends; the differential tests compare whole buffers, and
  consumers still normally read through the offsets stored on the tape.

## Numbers

Two words: the marker (`l`/`u`/`d`, payload 0), then the raw value word.
Type selection mirrors simdjson:

1. **Integer fast path**: the literal contains no `.`, `e`, or `E` and its
   value fits `i64` → tag `l`, value word = the `i64` bits.
2. Otherwise, if it is still an integer literal and fits `u64` (i.e.
   `9223372036854775808 ..= 18446744073709551615`) → tag `u`, value word =
   the `u64` value.
3. Otherwise (fractional, exponent, or out of integer range) → tag `d`,
   value word = `f64::to_bits` of the correctly rounded double.

## Worked example

Input document (23 bytes of JSON text):

```json
{"a":[1,2.5],"b":"x\n"}
```

The value string contains the two source characters `\` `n`; its unescaped
content is `x` followed by a line feed (2 bytes).

String slot allocation (raw-length prefix sum):

| string | raw bytes between quotes | raw_len | slot = raw_len + 5 | offset |
|---|---|---|---|---|
| `"a"` | `a` | 1 | 6 | 0 |
| `"b"` | `b` | 1 | 6 | 6 |
| `"x\n"` | `x` `\` `n` | 3 | 8 | 12 |

Buffer size = 6 + 6 + 8 = **20 bytes**. The last record unescapes to 2
content bytes, so it occupies only 7 of its 8 slot bytes — its slot ends
with 1 gap byte (zero-filled on both backends).

### Tape (13 words)

| idx | word (hex) | tag | decoded payload |
|---|---|---|---|
| 0 | `0x7200_0000_0000_000C` | `r` | final root word is at index 12 |
| 1 | `0x7B00_0002_0000_000C` | `{` | end = 12 (one past `}` at 11), count = 2 members |
| 2 | `0x2200_0000_0000_0000` | `"` | stringbuf offset 0 → key `"a"` |
| 3 | `0x5B00_0002_0000_0009` | `[` | end = 9 (one past `]` at 8), count = 2 elements |
| 4 | `0x6C00_0000_0000_0000` | `l` | i64 marker |
| 5 | `0x0000_0000_0000_0001` | — | i64 bits of `1` |
| 6 | `0x6400_0000_0000_0000` | `d` | f64 marker |
| 7 | `0x4004_0000_0000_0000` | — | f64 bits of `2.5` |
| 8 | `0x5D00_0000_0000_0003` | `]` | matching `[` is at index 3 |
| 9 | `0x2200_0000_0000_0006` | `"` | stringbuf offset 6 → key `"b"` |
| 10 | `0x2200_0000_0000_000C` | `"` | stringbuf offset 12 → value `"x\n"` |
| 11 | `0x7D00_0000_0000_0001` | `}` | matching `{` is at index 1 |
| 12 | `0x7200_0000_0000_0000` | `r` | payload 0 (points back at index 0) |

### String buffer (20 bytes)

| offset | bytes (hex) | meaning |
|---|---|---|
| 0 | `01 00 00 00` | u32 LE length = 1 (record for `"a"`) |
| 4 | `61` | `a` |
| 5 | `00` | NUL |
| 6 | `01 00 00 00` | u32 LE length = 1 (record for `"b"`) |
| 10 | `62` | `b` |
| 11 | `00` | NUL |
| 12 | `02 00 00 00` | u32 LE length = 2 (record for `"x\n"`) |
| 16 | `78` | `x` |
| 17 | `0A` | line feed — the `\n` escape, resolved |
| 18 | `00` | NUL |
| 19 | `00` | gap (slot is 8 bytes, record used 7; zero-filled on both backends) |

## Notes / deviations

- Word encoding, container payloads, count saturation, string record
  encoding, number markers and tag characters all follow simdjson's
  documented tape layout.
- **String record offset allocation** deviates: simdjson packs records
  densely as its sequential parse writes them; format v1 allocates slots by
  the raw-length prefix sum (see [Offset allocation](#offset-allocation-pinned-policy))
  so every offset is computable in parallel from token positions alone,
  before unescaping. Consumers are unaffected — they only ever read through
  the offsets stored on the tape.
- Pinned interpretation: simdjson's docs describe `tape[0]`'s payload as
  pointing "one past the last node", which in its own worked example equals
  the index of the final `r` word. Format v1 pins it to exactly **the index
  of the final root word**; that is the contract both backends implement and
  the differential tests enforce.
- Duplicate object keys are kept on the tape verbatim, in document order
  (simdjson parity); the tape does no deduplication.
