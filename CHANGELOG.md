# Changelog

## Unreleased
### Added
- Persisted state now carries an explicit `schema_version` (current format: 1), stamped by `new_state` and every save. `load_state` migrates legacy version 0 files (the pre-ST1 format with no version field: topics keyed by display name, exam/deadline/weight stored on the topic) to the current format in memory, rebuilding exam bindings from the config so a topic shared by two exams keeps one memory state and both bindings. Migration is idempotent, and corrupt or unmigratable input raises a clear error without modifying the file on disk. This completes the ST1 foundation's schema-version and migration components.

### Tests
- 8 new tests in tests/test_stage3_studyforge_st1.py covering legacy migration against a fixture built field-for-field from the pre-85e175b state format (including the shared-topic-two-exams case), migration idempotence, round-trip serialization stability, unchanged loading of current-format files, and non-destructive failure on corrupt, unmigratable, and future-version state files.

### Fixed
- Dashboard topic data now carries every binding deadline: each plan_data topic entry includes exam_days and exam_day (the latest binding deadline), validated by topic_exam_day, which fails loudly on a missing or inconsistent deadline instead of plotting against a wrong one. The dashboard and demo GIF generators use that contract, so both run on the included exams.yaml again: the dashboard forgetting curves no longer crash on a missing exam_day, and the demo GIF looks up the topic by its normalized ID (rotation) instead of the display name (Rotation).
- The fitting demo is now labeled honestly: fit_demo_data carries provenance marking its three points as an illustrative fixture with zero logged records, and the dashboard fit chart labels them as illustrative example points instead of logged reviews. The fitted stability is unchanged.

### Tests
- 6 new regression tests in tests/test_stage3_studyforge_02.py covering the topic deadline schema, schema-mismatch failure, normalized-ID lookup, engine agreement for the forgetting curve, fit-demo provenance, and end-to-end execution of both generators (skipped where the plotly/matplotlib dashboard extras are not installed).

### Docs
- Centered the README header: uniform dark badge labels and three action buttons in the dashboard palette (Open Dashboard, Explore Method, Run Locally) with a black label side so they stand out on the page.

### Fixed
- Evaluation is now as-of disciplined (A01). planner.simulate computes each exam's recall from the memory state as of that exam's day: when a topic shared by two exams was reviewed after the earlier exam (while studying for the later one), the earlier estimate is rebuilt by replaying the recorded review history instead of reading today's state, which previously produced impossible recalls above 1 (2.14 for a day-7 exam evaluated after a day-10 review). When the recorded history cannot reconstruct the as-of state, the estimate is marked unavailable (None) rather than fabricated. Planning objectives take an explicit as-of day: only exams still in the future participate in forward replanning, so replan and what-if no longer count a past exam's contribution. state.readiness applies the same rule to past exams and reports None for unavailable ones; the status command prints those as "n/a (as-of history unavailable)". dashboard plan_data now credits a shared topic's sessions and minutes to an exam only when the session falls before that exam's deadline; the included exams.yaml has no shared topics, so its published figures are unchanged.

### Tests
- 8 new tests in tests/test_stage3_studyforge_01.py: the day7/day21 shared-topic counterexample with hand-derived as-of values, the no-reviews-before-exam case that distinguishes reconstruction from clipping (exactly 0, not 1), past-exam exclusion from the replan objective, historical readiness invariance to later reviews, unavailable marking for display-name legacy history, before-deadline session attribution in plan_data, a finite-probability sweep, and a no-lost-bindings guard. Written before the fix: 7 failed and 1 passed on the pre-fix code.

## v1.1.0

### Fixed
- Topics are now keyed by stable IDs (slug of the name) holding the shared
  memory state, separately from exam-topic bindings (topic ID + weight, with
  the deadline from the exam). A topic shared by two exams (e.g. "Probability"
  in a midterm and a final) keeps one memory state and contributes to both
  exams' objectives at their own deadlines, instead of the second exam
  silently overwriting the first's association and deadline.
- Duplicate topic names within a single exam are now rejected with a clear
  error instead of silently colliding.
- The planner sums each binding's weight * recall at its own exam day over
  the shared memory state; review sessions scheduled after an exam's deadline
  no longer count toward that exam's recall.
- `studyforge log --topic` accepts a topic ID or its display name.

### Tests
- 5 new regression tests in `tests/test_state.py`; full suite 61/61 green.
