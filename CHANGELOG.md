# Changelog

## Unreleased
### Docs
- Centered the README header: uniform dark badge labels and three action buttons in the dashboard palette (Open Dashboard, Explore Method, Run Locally) with a black label side so they stand out on the page.

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
