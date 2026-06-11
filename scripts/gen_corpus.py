#!/usr/bin/env python3
"""Regenerate the deterministic corpus fixtures that are awkward to type by
hand: corpus/escapes.json (literal backslash escapes must survive editors and
tooling that resolve \\uXXXX sequences) and corpus/twitter_like_100kb.json
(a ~100KB realistic record array).

Fully deterministic: no randomness APIs anywhere — every field is a fixed
iterative function of the record index, so the output is byte-identical on
every run. The outputs are checked in; this script only exists to make them
reproducible.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92)  # backslash, built programmatically so no tool resolves it


def u(hex4: str) -> str:
    """A literal \\uXXXX escape sequence (6 characters)."""
    return BS + "u" + hex4


def gen_escapes() -> None:
    """corpus/escapes.json: every escape form as literal text in the file."""
    q = BS + '"'
    lines = [
        "[",
        f'  "simple: {q} {BS}{BS} {BS}/ {BS}b {BS}f {BS}n {BS}r {BS}t",',
        f'  "hex escapes: {u("0041")} {u("00e9")} {u("00E9")} {u("4e2d")} {u("FFFD")}",',
        f'  "surrogate pairs: {u("D83D")}{u("DE00")} {u("d834")}{u("dd1e")} {u("DBFF")}{u("DFFF")}",',
        f'  "interior nul: a{u("0000")}b",',
        f'  "escaped controls: {u("0001")}{u("001f")}{u("007F")}",',
        f'  {{"key with {q}quotes{q} and {BS}{BS}backslash{BS}{BS}": "and {u("0000")} {u("0009")} nul"}},',
        f'  "mixed raw and escaped: é → {u("00e9")}, 😀 → {u("D83D")}{u("DE00")}"',
        "]",
        "",
    ]
    text = "\n".join(lines)
    data = json.loads(text)  # sanity: must be valid JSON
    assert data[1] == "hex escapes: A é é 中 �"
    assert data[2] == "surrogate pairs: \U0001f600 \U0001d11e \U0010ffff"
    assert data[3] == "interior nul: a\x00b"
    assert data[4] == "escaped controls: \x01\x1f\x7f"
    assert u("0041") in text  # the file really holds literal escapes
    with open(os.path.join(ROOT, "corpus", "escapes.json"), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote corpus/escapes.json")


# Fixed word/emoji/name pools, cycled by index — no randomness.
TEXTS = [
    "Just parsed a JSON document on the GPU",
    "unified memory makes zero-copy real",
    'benchmark says: "faster" (quotes included)',
    "tape format v1 — simdjson layout",
    "newline\nand\ttab live in this tweet",
    "backslash test C:\\metal\\json",
    "emoji payload 😀🎉🚀 end",
    "日本語のツイートです",
    "counting sort beats comparison sort",
    "Eisel-Lemire as pure integer math",
    "prefix-xor finds the strings",
]
NAMES = ["Ada", "Grace", "Edsger", "Barbara", "Donald", "Tony", "Leslie", "Frances"]
SURNAMES = ["Lovelace", "Hopper", "Dijkstra", "Liskov", "Knuth", "Hoare", "Lamport", "Allen"]
LANGS = ["en", "ja", "de", "fr", "es"]


def make_record(i: int) -> dict:
    rec = {
        "id": 90000000000000000 + i * 9973,  # > 2^53: exercises exact i64
        "id_str": str(90000000000000000 + i * 9973),
        "created_at": (
            f"2026-06-{(i % 28) + 1:02d}T{i % 24:02d}:{i % 60:02d}:{(i * 37) % 60:02d}Z"
        ),
        "text": f"{TEXTS[i % len(TEXTS)]} #{i}",
        "truncated": i % 9 == 0,
        "user": {
            "id": 1000 + (i * 31) % 9000,
            "screen_name": f"user_{i % 97}",
            "name": f"{NAMES[i % len(NAMES)]} {SURNAMES[(i // 3) % len(SURNAMES)]}",
            "followers_count": (i * i) % 100000,
            "verified": i % 13 == 0,
            "description": None if i % 6 == 0 else f"bio line {i % 41} ✨",
        },
        "retweet_count": i % 1500,
        "favorite_count": (i * 7) % 9000,
        "coordinates": (
            None
            if i % 4
            else [-122.0 + (i % 1000) * 0.001357, 37.0 + (i % 500) * 0.002113]
        ),
        "lang": LANGS[i % len(LANGS)],
        "entities": {
            "hashtags": (
                [{"text": f"tag{i % 50}", "indices": [0, 5 + i % 10]}]
                if i % 3 == 0
                else []
            ),
            "urls": (
                [{"url": f"https://example.com/{i}", "expanded": None}]
                if i % 5 == 0
                else []
            ),
        },
        "in_reply_to_status_id": None if i % 2 else 90000000000000000 + (i - 1) * 9973,
    }
    if i % 50 == 0:
        # Sprinkle number-torture edges through the realistic payload.
        rec["edge_numbers"] = {
            "u64_max": 18446744073709551615,
            "i64_min": -9223372036854775808,
            "subnormal": 5e-324,
            "neg_zero": -0.0,
            "f64_max": 1.7976931348623157e308,
            "seventeen": 0.1234567890123456,
        }
    return rec


def gen_twitter() -> None:
    """corpus/twitter_like_100kb.json: ~100KB record array, fixed pattern."""
    records = []
    i = 0
    size = 2  # "[]"
    while size < 100_000:
        blob = json.dumps(make_record(i), ensure_ascii=False, separators=(",", ":"))
        size += len(blob.encode("utf-8")) + 1  # +1 for the separating comma
        records.append(blob)
        i += 1
    text = "[" + ",".join(records) + "]\n"
    json.loads(text)  # sanity
    path = os.path.join(ROOT, "corpus", "twitter_like_100kb.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote corpus/twitter_like_100kb.json ({len(text.encode('utf-8'))} bytes, {i} records)")


if __name__ == "__main__":
    gen_escapes()
    gen_twitter()
