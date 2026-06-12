// extern "C" shim over simdjson::dom::parser for the criterion harness.
//
// Contract (mirrored by metal-json-bench/src/lib.rs):
//   - sj_parser_new / sj_parser_free      reusable parser (tape buffers are
//                                         reused across parses, like the
//                                         metal-json buffer pool).
//   - sj_alloc_padded / sj_free_padded    input allocation with
//                                         SIMDJSON_PADDING trailing bytes so
//                                         padding is handled OUTSIDE the
//                                         timed call.
//   - sj_parse_tape                       parse-to-tape + a shallow walk of
//                                         the raw tape filling SjStats. The
//                                         walk touches every tape word once
//                                         (no DOM recursion), which defeats
//                                         dead-code elimination and proves
//                                         the parse actually happened.
//
// SjStats semantics (must match the serde_json oracle in the smoke test):
//   node_count   = every scalar, every object key, and every container
//                  opening ('{' or '[') counts 1; closers and the root
//                  sentinels count 0.
//   string_bytes = sum of unescaped byte lengths of all strings (keys and
//                  values).
//   number_xor   = XOR of the raw 64-bit payloads of all numbers (int64 /
//                  uint64 two's-complement bits, f64 IEEE-754 bits).

#include "vendor/simdjson.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" {

struct SjStats {
  uint64_t node_count;
  uint64_t string_bytes;
  uint64_t number_xor;
};

void *sj_parser_new(void) {
  return new (std::nothrow) simdjson::dom::parser();
}

void sj_parser_free(void *p) {
  delete static_cast<simdjson::dom::parser *>(p);
}

size_t sj_padding(void) { return simdjson::SIMDJSON_PADDING; }

// Allocates len + SIMDJSON_PADDING bytes; the harness copies the document in
// once, before timing starts.
uint8_t *sj_alloc_padded(size_t len) {
  return static_cast<uint8_t *>(std::malloc(len + simdjson::SIMDJSON_PADDING));
}

void sj_free_padded(uint8_t *p) { std::free(p); }

// Parse `ptr[0..len]` (which MUST have SIMDJSON_PADDING readable bytes after
// `len`, i.e. come from sj_alloc_padded) and fill `out`.
//
// Returns 0 on success, otherwise the simdjson::error_code as an int.
int sj_parse_tape(void *pv, const uint8_t *ptr, size_t len, SjStats *out) {
  auto &parser = *static_cast<simdjson::dom::parser *>(pv);

  simdjson::dom::element root;
  // realloc_if_needed = false: the buffer is already padded.
  auto err = parser.parse(ptr, len, false).get(root);
  if (err != simdjson::SUCCESS) {
    return static_cast<int>(err);
  }

  SjStats stats{0, 0, 0};

  // Shallow walk over the raw tape (one linear pass, no recursion).
  // Tape format: word 0 is the root sentinel 'r' whose payload is one past
  // the final tape word; each word has the type tag in the top byte and a
  // 56-bit payload. Numbers carry their 64-bit value in the following word.
  const uint64_t *tape = parser.doc.tape.get();
  const uint8_t *string_buf = parser.doc.string_buf.get();
  const uint64_t end = tape[0] & simdjson::internal::JSON_VALUE_MASK;

  for (uint64_t i = 1; i < end; i++) {
    const uint64_t word = tape[i];
    switch (static_cast<char>(word >> 56)) {
      case '"': {
        const uint64_t off = word & simdjson::internal::JSON_VALUE_MASK;
        uint32_t slen;
        std::memcpy(&slen, string_buf + off, sizeof(slen));
        stats.string_bytes += slen;
        stats.node_count++;
        break;
      }
      case 'l':  // int64
      case 'u':  // uint64
      case 'd':  // double
        i++;     // value lives in the next tape word
        stats.number_xor ^= tape[i];
        stats.node_count++;
        break;
      case 't':
      case 'f':
      case 'n':
      case '{':
      case '[':
        stats.node_count++;
        break;
      default:  // '}' ']' 'r'
        break;
    }
  }

  if (out != nullptr) {
    *out = stats;
  }
  return 0;
}

}  // extern "C"
