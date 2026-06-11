//! Stage 4 — structural validation: depth, pairing, container context.
//!
//! Scalar oracle for GPU kernels **K8** (counting sort of the skeleton by
//! depth) and **K9** (pair map + container context + child counts). The M3
//! kernel unit tests run K8/K9 and this function on identical skeletons and
//! diff [`Stage4Output`] field by field.
//!
//! The algorithm deliberately models the GPU formulation instead of the
//! obvious recursive-descent/stack one:
//!
//! 1. **Depth scan**: each skeleton element gets a depth — an open bracket
//!    and its close share the depth of the container they delimit (root
//!    container = 1); separators get the depth of the container they sit
//!    in. On the GPU this is a prefix sum over ±1 bracket weights; here a
//!    sequential scan. This is also where the depth limit
//!    ([`Error::DepthLimit`]) and close-below-root underflow are detected.
//! 2. **Stable counting sort by depth** (histogram → exclusive prefix sum →
//!    stable scatter). The GPU does the same sort in 5-bit digit passes;
//!    one full-width pass is used here — both are stable, so the outputs
//!    are identical and diffable.
//! 3. **Adjacent pairing within each depth group**: within a group the
//!    brackets strictly alternate open/close in document order, so pairs
//!    are adjacent. The open/close *type* check is `open ^ close == 0x06`
//!    (`{`^`}` == `[`^`]` == `0x06`; a mismatched `[`…`}` xors to `0x26`).
//!    Leftover opens (unclosed container) and closes (close without open)
//!    are balance errors.
//! 4. **Comma-context Layer-2 checks** via the segmented forward-fill of
//!    the opener type per depth group (the separators between an open and
//!    its close in a group always belong to exactly that container):
//!    - a separator with **no** enclosing opener sits at depth 0 — a
//!      separator after a complete root value. Reported as
//!      [`Error::TrailingContent`]; kills `1,2` / `{},{}`-style
//!      multi-value roots (a trailing separator at end of input, as in
//!      `n_array_comma_after_close.json`'s `[""],`, already dies in
//!      Layer 1's separator→end ban).
//!    - a `:` whose opener is `[` — kills `[1,"a":2]` (the colon 4-token
//!      rule of Layer 1 cannot see the container type).
//!    - a `,` whose opener is `{` must be followed (next element in the
//!      same depth group) by the member's `:` exactly 3 tokens later
//!      (`,` `"key…` `…"` `:`); kills `{"foo":"bar", "a"}`
//!      (n_object_with_single_string.json) and `{"a":1,2}`. Together with
//!      Layer 1's colon 4-token rule this is exactly simdjson's
//!      key-colon-value object grammar.
//! 5. **Per-container child counts**: an object's direct member count is
//!    the number of colons between its open and close in the depth group;
//!    an array's is commas + 1 (0 when the close token immediately follows
//!    the open token).
//!
//! Error precedence: candidates from all phases compete and the earliest
//! byte offset wins (ties: first recorded). This mirrors the GPU's
//! `atomic_min` on `(offset << 32) | code` *within* this stage.

use super::validate::SkeletonRecord;
use crate::error::{Error, Result, SyntaxErrorKind};

/// Marker for "no partner" in [`Stage4Output::match_index`] (separators).
pub const NO_MATCH: u32 = u32::MAX;

/// Stage 4 outputs, all keyed by skeleton index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage4Output {
    /// Depth of each skeleton element (root container = 1; a depth-0 entry
    /// is a separator or stray close after the root value).
    pub depths: Vec<u32>,
    /// Skeleton indices in (depth, document-order) order — the stable
    /// counting-sort output (what GPU kernel K8 produces).
    pub sorted_by_depth: Vec<u32>,
    /// For brackets: skeleton index of the matching bracket. [`NO_MATCH`]
    /// for separators (what GPU kernel K9 produces).
    pub match_index: Vec<u32>,
    /// For separators: the opener byte (`{` or `[`) of the enclosing
    /// container — the segmented forward-fill result. `0` for brackets.
    pub context_opener: Vec<u8>,
    /// For open brackets: number of direct children (object members /
    /// array elements). `0` for everything else.
    pub child_counts: Vec<u32>,
}

/// Keeps the earliest-offset error candidate, mirroring the GPU's
/// `atomic_min` error word within the stage.
struct EarliestError {
    best: Option<(u64, Error)>,
}

impl EarliestError {
    fn new() -> Self {
        Self { best: None }
    }

    fn record(&mut self, offset: u64, error: Error) {
        if self.best.as_ref().is_none_or(|(best, _)| offset < *best) {
            self.best = Some((offset, error));
        }
    }

    fn syntax(&mut self, pos: u32, kind: SyntaxErrorKind) {
        self.record(
            u64::from(pos),
            Error::Syntax {
                offset: u64::from(pos),
                kind,
            },
        );
    }

    fn into_result(self) -> Result<()> {
        match self.best {
            Some((_, error)) => Err(error),
            None => Ok(()),
        }
    }
}

/// Stage 4: depth scan + counting-sort pairing + Layer-2 context checks.
///
/// `skeleton` must come from
/// [`stage3_validate_local`](super::stage3_validate_local) (or be an
/// equivalent hand-built skeleton in kernel tests); `max_depth` is
/// [`ParserOptions::max_depth`](crate::ParserOptions::max_depth).
///
/// # Errors
///
/// The earliest-offset structural error: [`Error::DepthLimit`],
/// [`Error::TrailingContent`], or [`Error::Syntax`] with
/// `UnbalancedBrackets` / `UnexpectedToken` / `MissingColon`.
pub fn stage4_structure(skeleton: &[SkeletonRecord], max_depth: u32) -> Result<Stage4Output> {
    let n = skeleton.len();
    let mut errors = EarliestError::new();

    // --- Phase 1: depth scan ---------------------------------------------
    let mut depths = vec![0u32; n];
    let mut depth: u32 = 0;
    for (i, rec) in skeleton.iter().enumerate() {
        match rec.byte {
            b'{' | b'[' => {
                depth += 1;
                if depth > max_depth {
                    errors.record(
                        u64::from(rec.pos),
                        Error::DepthLimit {
                            offset: u64::from(rec.pos),
                            limit: max_depth,
                        },
                    );
                }
                depths[i] = depth;
            }
            b'}' | b']' => {
                if depth == 0 {
                    // Close with nothing open: `1]`, `{}}`. Kills
                    // n_structure_close_unopened_array.json and
                    // n_structure_object_followed_by_closing_object.json.
                    errors.syntax(rec.pos, SyntaxErrorKind::UnbalancedBrackets);
                    depths[i] = 0; // park it in group 0; pairing re-flags it
                } else {
                    depths[i] = depth;
                    depth -= 1;
                }
            }
            b':' | b',' => depths[i] = depth,
            other => unreachable!("non-structural skeleton byte {other:#04x}"),
        }
    }

    // --- Phase 2: stable counting sort by depth (models K8) ---------------
    let buckets = depths.iter().max().map_or(0, |&d| d as usize + 1);
    let mut offsets = vec![0u32; buckets];
    for &d in &depths {
        offsets[d as usize] += 1;
    }
    // histogram -> exclusive prefix sum, in place
    let mut running = 0u32;
    for slot in &mut offsets {
        let count = *slot;
        *slot = running;
        running += count;
    }
    // stable scatter
    let mut sorted_by_depth = vec![0u32; n];
    for (i, &d) in depths.iter().enumerate() {
        let slot = &mut offsets[d as usize];
        sorted_by_depth[*slot as usize] = u32::try_from(i).expect("skeleton too large");
        *slot += 1;
    }

    // --- Phases 3-5: per depth group (models K9) ---------------------------
    let mut match_index = vec![NO_MATCH; n];
    let mut context_opener = vec![0u8; n];
    let mut child_counts = vec![0u32; n];

    let mut group_start = 0;
    while group_start < n {
        let d = depths[sorted_by_depth[group_start] as usize];
        let mut group_end = group_start;
        while group_end < n && depths[sorted_by_depth[group_end] as usize] == d {
            group_end += 1;
        }

        // Walk the group in document order (counting sort is stable).
        let mut pending_open: Option<usize> = None; // skeleton index
        let mut commas = 0u32;
        let mut colons = 0u32;
        for j in group_start..group_end {
            let si = sorted_by_depth[j] as usize;
            let rec = &skeleton[si];
            match rec.byte {
                b'{' | b'[' => {
                    if pending_open.is_some() {
                        // Two opens without a close in one depth group can
                        // only come from a hand-built skeleton; flag it
                        // rather than panic.
                        errors.syntax(rec.pos, SyntaxErrorKind::UnbalancedBrackets);
                    }
                    pending_open = Some(si);
                    commas = 0;
                    colons = 0;
                }
                b'}' | b']' => match pending_open.take() {
                    None => {
                        // Group-0 stray closes land here (and were already
                        // flagged in the scan, at the same offset).
                        errors.syntax(rec.pos, SyntaxErrorKind::UnbalancedBrackets);
                    }
                    Some(open_si) => {
                        let open = &skeleton[open_si];
                        if open.byte ^ rec.byte != 0x06 {
                            // `{` closed by `]` (or `[` by `}`): xor is
                            // 0x26, not 0x06. Kills `{"a":1]`-style
                            // mismatches.
                            errors.syntax(rec.pos, SyntaxErrorKind::UnbalancedBrackets);
                        }
                        match_index[open_si] = u32::try_from(si).expect("skeleton too large");
                        match_index[si] = u32::try_from(open_si).expect("skeleton too large");
                        child_counts[open_si] = if open.byte == b'{' {
                            colons
                        } else if rec.token_index == open.token_index + 1 {
                            0 // `[]`
                        } else {
                            commas + 1
                        };
                    }
                },
                b':' | b',' => match pending_open {
                    None => {
                        // Separator with no enclosing container: a complete
                        // root value already ended (Layer 1 guarantees a
                        // separator's predecessor is a value end). Kills
                        // `1,2`, `{},{}` and the colon of `1,"a":2`-style
                        // trailing content.
                        errors.record(
                            u64::from(rec.pos),
                            Error::TrailingContent {
                                offset: u64::from(rec.pos),
                            },
                        );
                    }
                    Some(open_si) => {
                        let opener = skeleton[open_si].byte;
                        context_opener[si] = opener; // segmented forward-fill
                        if rec.byte == b':' {
                            colons += 1;
                            if opener != b'{' {
                                // Colon inside an array: `[1,"a":2]`.
                                errors.syntax(rec.pos, SyntaxErrorKind::UnexpectedToken);
                            }
                        } else {
                            commas += 1;
                            if opener == b'{' {
                                // Object comma must introduce `"key":`,
                                // i.e. the next element of this depth group
                                // is the member's colon, exactly 3 tokens
                                // later. Kills `{"foo":"bar", "a"}`
                                // (n_object_with_single_string.json) and
                                // `{"a":1,2}`.
                                let next_is_member_colon = j + 1 < group_end && {
                                    let next = &skeleton[sorted_by_depth[j + 1] as usize];
                                    next.byte == b':' && next.token_index == rec.token_index + 3
                                };
                                if !next_is_member_colon {
                                    errors.syntax(rec.pos, SyntaxErrorKind::MissingColon);
                                }
                            }
                        }
                    }
                },
                other => unreachable!("non-structural skeleton byte {other:#04x}"),
            }
        }
        if let Some(open_si) = pending_open {
            // Unclosed container: `[1`, `{"a":1`. Kills
            // n_structure_unclosed_array.json-style inputs whose last token
            // is a value end (lone trailing opens already died in Layer 1's
            // open→end ban).
            errors.syntax(skeleton[open_si].pos, SyntaxErrorKind::UnbalancedBrackets);
        }

        group_start = group_end;
    }

    errors.into_result()?;
    Ok(Stage4Output {
        depths,
        sorted_by_depth,
        match_index,
        context_opener,
        child_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::super::classify::stage1_classify;
    use super::super::tokens::stage2_tokens;
    use super::super::validate::stage3_validate_local;
    use super::*;
    use crate::parser::DEFAULT_MAX_DEPTH;

    /// Run stages 1-3 then stage 4 with the default depth limit.
    fn run(input: &[u8]) -> Result<Stage4Output> {
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        let s3 = stage3_validate_local(&tokens, input).expect("fixture must pass Layer 1");
        stage4_structure(&s3.skeleton, DEFAULT_MAX_DEPTH)
    }

    fn skeleton_of(input: &[u8]) -> Vec<SkeletonRecord> {
        let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
        stage3_validate_local(&tokens, input).unwrap().skeleton
    }

    #[test]
    fn depths_sort_pairs_and_counts_for_a_nested_doc() {
        // {"a":[{}]}  — skeleton: { : [ { } ] }
        let out = run(br#"{"a":[{}]}"#).unwrap();
        assert_eq!(out.depths, vec![1, 1, 2, 3, 3, 2, 1]);
        // Stable by (depth, doc order): d1: {(0) :(1) }(6); d2: [(2) ](5);
        // d3: {(3) }(4).
        assert_eq!(out.sorted_by_depth, vec![0, 1, 6, 2, 5, 3, 4]);
        assert_eq!(
            out.match_index,
            vec![6, NO_MATCH, 5, 4, 3, 2, 0],
            "pair map"
        );
        assert_eq!(out.context_opener, vec![0, b'{', 0, 0, 0, 0, 0]);
        // Outer object: 1 member; array: 1 element; inner object: empty.
        assert_eq!(out.child_counts, vec![1, 0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn child_counts() {
        // (input, skeleton index of the container, expected count)
        let cases: &[(&[u8], usize, u32)] = &[
            (b"[]", 0, 0),
            (b"{}", 0, 0),
            (b"[1]", 0, 1),
            (b"[1,2,3]", 0, 3),
            (br#"{"a":1}"#, 0, 1),
            (br#"{"a":1,"b":2}"#, 0, 2),
            (b"[[],[],[]]", 0, 3),
            (br#"{"a":1,"b":[1,2,3]}"#, 0, 2), // outer object
            (br#"{"a":1,"b":[1,2,3]}"#, 4, 3), // inner array
            (br#"[{"x":0},2]"#, 0, 2),
        ];
        for &(input, si, want) in cases {
            let out = run(input).unwrap();
            assert_eq!(
                out.child_counts[si],
                want,
                "child count of skeleton[{si}] in {:?}",
                String::from_utf8_lossy(input)
            );
        }
    }

    #[test]
    fn separator_context_is_the_enclosing_opener() {
        // [1,{"a":1,"b":2},3]  — commas at array depth get '[', object
        // separators get '{'.
        let out = run(br#"[1,{"a":1,"b":2},3]"#).unwrap();
        let skel = skeleton_of(br#"[1,{"a":1,"b":2},3]"#);
        for (si, rec) in skel.iter().enumerate() {
            match rec.byte {
                b':' => assert_eq!(out.context_opener[si], b'{', "colon {si}"),
                b',' => {
                    let want = if out.depths[si] == 1 { b'[' } else { b'{' };
                    assert_eq!(out.context_opener[si], want, "comma {si}");
                }
                _ => assert_eq!(out.context_opener[si], 0, "bracket {si}"),
            }
        }
    }

    #[test]
    fn unmatched_closes_are_unbalanced() {
        for (input, offset) in [(&b"1]"[..], 1), (b"{}}", 2), (b"[[]]]", 4)] {
            match run(input) {
                Err(Error::Syntax {
                    offset: o,
                    kind: SyntaxErrorKind::UnbalancedBrackets,
                }) => assert_eq!(o, offset, "{:?}", String::from_utf8_lossy(input)),
                other => panic!("expected UnbalancedBrackets, got {other:?}"),
            }
        }
    }

    #[test]
    fn unclosed_opens_are_unbalanced() {
        // Last token is a value end, so Layer 1 passes; the open container
        // never closes.
        for (input, offset) in [(&b"[1"[..], 0), (br#"{"a":1"#, 0), (b"[[1]", 0)] {
            match run(input) {
                Err(Error::Syntax {
                    offset: o,
                    kind: SyntaxErrorKind::UnbalancedBrackets,
                }) => assert_eq!(o, offset, "{:?}", String::from_utf8_lossy(input)),
                other => panic!("expected UnbalancedBrackets, got {other:?}"),
            }
        }
    }

    #[test]
    fn mismatched_bracket_types_fail_the_xor_check() {
        // `[1}` passes Layer 1 (scalar then `}` is adjacency-legal) but the
        // pair `[`/`}` xors to 0x26.
        match run(b"[1}") {
            Err(Error::Syntax {
                offset,
                kind: SyntaxErrorKind::UnbalancedBrackets,
            }) => assert_eq!(offset, 2),
            other => panic!("expected UnbalancedBrackets, got {other:?}"),
        }
        assert!(run(br#"{"a":1]"#).is_err());
    }

    #[test]
    fn depth_zero_separators_are_trailing_content() {
        // A separator after the root value completed. (A separator as the
        // very LAST token, e.g. `[""],`, dies in Layer 1's separator→end
        // ban instead, so these fixtures continue with another value.)
        for (input, offset) in [(&b"{},1"[..], 2), (b"1,2", 1), (br#"[""],1"#, 4)] {
            match run(input) {
                Err(Error::TrailingContent { offset: o }) => {
                    assert_eq!(o, offset, "{:?}", String::from_utf8_lossy(input));
                }
                other => panic!("expected TrailingContent, got {other:?}"),
            }
        }
        // `{},{}`: Layer 1 passes (close, comma, open are pairwise legal);
        // the depth-0 comma is the trailing-content signal.
        assert!(matches!(
            run(b"{},{}"),
            Err(Error::TrailingContent { offset: 2 })
        ));
    }

    #[test]
    fn colon_inside_an_array_is_rejected() {
        // Passes the Layer-1 colon 4-token rule (i-3 is a comma) — only the
        // container context can kill it.
        match run(br#"[1,"a":2]"#) {
            Err(Error::Syntax {
                offset,
                kind: SyntaxErrorKind::UnexpectedToken,
            }) => assert_eq!(offset, 6),
            other => panic!("expected UnexpectedToken, got {other:?}"),
        }
    }

    #[test]
    fn object_comma_must_introduce_a_member() {
        // n_object_with_single_string.json
        for input in [
            &br#"{ "foo" : "bar", "a" }"#[..],
            br#"{"a":1,2}"#,
            br#"{"a":1,"b"}"#,
            br#"{"a":1,[]}"#,
        ] {
            assert!(
                matches!(
                    run(input),
                    Err(Error::Syntax {
                        kind: SyntaxErrorKind::MissingColon,
                        ..
                    })
                ),
                "{:?}",
                String::from_utf8_lossy(input)
            );
        }
        // ... while a well-formed second member passes.
        assert!(run(br#"{"a":1,"b":2}"#).is_ok());
    }

    #[test]
    fn depth_limit_respects_max_depth() {
        let four_deep = skeleton_of(b"[[[[]]]]");
        assert!(stage4_structure(&four_deep, 4).is_ok());
        match stage4_structure(&four_deep, 3) {
            Err(Error::DepthLimit { offset, limit }) => {
                assert_eq!(offset, 3, "the 4th open bracket");
                assert_eq!(limit, 3);
            }
            other => panic!("expected DepthLimit, got {other:?}"),
        }
    }

    #[test]
    fn depth_limit_at_simdjson_parity_1024() {
        let nest = |depth: usize| {
            let mut s = "[".repeat(depth);
            s.push_str(&"]".repeat(depth));
            s.into_bytes()
        };
        assert!(
            run(&nest(1024)).is_ok(),
            "1024 deep is exactly at the limit"
        );
        match run(&nest(1025)) {
            Err(Error::DepthLimit { offset, limit }) => {
                assert_eq!(offset, 1024, "the 1025th open bracket");
                assert_eq!(limit, 1024);
            }
            other => panic!("expected DepthLimit, got {other:?}"),
        }
    }

    #[test]
    fn earliest_offset_error_wins_across_phases() {
        // Underflow close at 2 (scan phase) vs depth-0 comma at 3 vs
        // unclosed open at 4 (pairing phase): the earliest offset must win.
        match run(b"{}},[1") {
            Err(Error::Syntax {
                offset,
                kind: SyntaxErrorKind::UnbalancedBrackets,
            }) => assert_eq!(offset, 2),
            other => panic!("expected UnbalancedBrackets@2, got {other:?}"),
        }
    }

    #[test]
    fn counting_sort_is_stable_within_depth_groups() {
        // Two sibling arrays with members: their brackets and separators
        // must stay in document order inside each depth group.
        let out = run(b"[[1,2],[3,4]]").unwrap();
        let skel = skeleton_of(b"[[1,2],[3,4]]");
        // skeleton: [0 [1 ,2 ]3 ,4 [5 ,6 ]7 ]8
        assert_eq!(out.depths, vec![1, 2, 2, 2, 1, 2, 2, 2, 1]);
        assert_eq!(out.sorted_by_depth, vec![0, 4, 8, 1, 2, 3, 5, 6, 7]);
        // Pairing within the d2 group is adjacent: (1,3) and (5,7).
        assert_eq!(out.match_index[1], 3);
        assert_eq!(out.match_index[3], 1);
        assert_eq!(out.match_index[5], 7);
        assert_eq!(out.match_index[7], 5);
        assert_eq!(out.match_index[0], 8);
        assert_eq!(skel[out.match_index[0] as usize].byte, b']');
        assert_eq!(out.child_counts[0], 2);
    }

    #[test]
    fn empty_skeleton_is_fine() {
        // Root scalars have no structural tokens at all.
        let out = stage4_structure(&[], DEFAULT_MAX_DEPTH).unwrap();
        assert!(out.depths.is_empty());
        assert!(out.sorted_by_depth.is_empty());
    }
}
