# Planner invariants and migration

The rules the planner promises to keep, in one place. Each invariant
names the code that enforces it and the test that pins it, so a future
change that breaks one fails a named test instead of silently changing
what a plan means. Nothing here is new behavior invented for this
document: items marked "already enforced" were verified on main before
this unit was written, and the one item that was missing (backup on
save) is called out as new in ST7.

## 1. Stable topic identity and exam bindings

A knowledge topic has one stable ID, the slug of its display name
(`state.topic_id`, `src/studyforge/state.py:185`), and one memory
state (`S`, `last_review`). Exams bind to that ID with a weight; the
deadline comes from the exam, never from the topic
(`state.new_state`, `state.py:281`). A topic shared by two exams
keeps one memory state and contributes weight times recall at EACH
bound exam's deadline (`planner.PlanTopic`, `src/studyforge/planner.py:132`).

Enforced by: `validate_config` rejects duplicate topic names within
an exam and names that slug to the same ID (`state.py:61`);
`validate_state` rejects bindings to unknown topics and topics bound
to no exam (`state.py:362`); `PlanTopic` rejects empty or duplicate
bindings (`planner.py:132`).

Pinned by: `tests/test_state.py` (stable IDs, shared-topic
bindings), `tests/test_stage3_studyforge_st1.py` (shared topic keeps
its memory and both bindings after migration).

## 2. Persisted schema version

Persisted state carries an explicit `schema_version`
(`state.SCHEMA_VERSION = 1`, `state.py:21`). Every save stamps it
(`state.save_state`, `state.py:655`) and every fresh state carries it
(`state.new_state`, `state.py:281`). A state with no version key is
legacy version 0. A version newer than the build understands is
rejected, never silently read (`state.migrate_state`, `state.py:326`).

Pinned by: `tests/test_stage3_studyforge_st1.py`
(`test_current_format_loads_unchanged`,
`test_future_schema_version_rejected`).

## 3. Migration from the pre-ST1 format

`state.load_state` migrates a legacy version 0 state in memory before
validating it (`state.py:673`). Migration rebuilds the exam bindings
from the config's exam topic lists, the same source `new_state` uses,
so a topic shared by two exams becomes one memory state with one
binding per exam and no binding is lost
(`state._migrate_v0`, `state.py:585`). A legacy topic no exam
references still keeps its memory state. Legacy history entries keep
naming topics by display name; they pass through unchanged and are
treated as unusable for as-of replay rather than rewritten
(see invariant 8).

Pinned by: `tests/test_stage3_studyforge_st1.py`
(`test_legacy_fresh_state_migrates_to_new_state_shape`,
`test_legacy_shared_topic_keeps_memory_and_both_bindings`).

## 4. Idempotent migration

Migrating a current state changes nothing, and migrating twice is the
same as migrating once. Migration never mutates its input dict and
never writes to disk (`state.migrate_state`, `state.py:326`).

Pinned by: `tests/test_stage3_studyforge_st1.py`
(`test_migration_is_idempotent`) and
`tests/test_stage3_studyforge_7.py`
(`test_migration_is_idempotent`).

## 5. Round-trip serialization

Saving a state and loading it back yields the same state
(`state.save_state` / `state.load_state`, `state.py:655` /
`state.py:673`). Loading a state and saving it again does not change
the file.

Pinned by: `tests/test_stage3_studyforge_st1.py`
(`test_round_trip_serialization_is_stable`),
`tests/test_memory.py` (`test_state_roundtrip`), and
`tests/test_stage3_studyforge_7.py`
(`test_round_trip_serialization_is_stable`).

## 6. Non-destructive failure

A state that cannot be loaded or migrated raises a clear error and
leaves the file on disk exactly as it was: corrupt JSON, a malformed
legacy state, and a future schema version all fail without modifying
the file (`state.migrate_state` and `state.load_state`,
`state.py:326` / `state.py:673`).

Pinned by: `tests/test_stage3_studyforge_st1.py`
(`test_corrupt_state_fails_and_leaves_file_untouched`,
`test_unmigratable_legacy_state_fails_and_leaves_file_untouched`,
`test_future_schema_version_rejected`) and
`tests/test_stage3_studyforge_7.py`
(`test_loading_legacy_state_leaves_file_untouched`).

## 7. Backup state is preserved (new in ST7)

Saving never destroys the last good state. Before `save_state`
overwrites a state file, it copies the previous file byte for byte to
`<path>.bak` in the same directory (`state.save_state`,
`state.py:655`). The backup always holds the immediately previous
saved state and nothing older: each save replaces it. A first save
creates no backup, since there is no previous state to keep. The
backup is a byte copy, so it loads exactly as the file it came from
did. The CLI commands (`plan`, `log`, `replan`) all save through
`save_state`, so every one of them preserves a backup.

Pinned by: `tests/test_stage3_studyforge_7.py`
(`test_save_preserves_backup_of_previous_state`,
`test_backup_holds_only_the_immediately_previous_state`,
`test_saved_state_still_round_trips_with_backup_present`).

## 8. Calculation semantics

Recall follows the Ebbinghaus curve `R(t) = exp(-t / S)` with
stability `S` in days (`memory.recall`, `src/studyforge/memory.py:91`).
A review at elapsed time `t` with retrievability `R` grows stability
by `S' = S * (1 + GROWTH * (1 - R))` with `GROWTH = 1.0`
(`memory.TopicState.review`, `memory.py:126`): reviewing at full
recall leaves `S` unchanged, reviewing after full decay doubles it.
Initial stability comes from the 1 to 5 difficulty rating as
`S0 = 3.0 / difficulty` (`memory.initial_stability`, `memory.py:98`).
Logged reviews with an observed recall fraction refit a topic's
personal stability by least squares on log recall once three usable
pairs exist, clamped to 0.1 to 60 days (`memory.fit_stability`,
`memory.py:58`).

The planning objective is the sum over topics and their exam bindings
of weight times predicted recall at the exam day
(`planner.objective`, `planner.py:345`). Greedy allocation adds the
single session with the largest marginal gain; ties break by topic
name then day, so the result is fully deterministic and the seed
parameter is accepted for interface stability only
(`planner.greedy`, `planner.py:384`). Local search applies the single
best move (relocate or drop a session) until no move improves the
objective (`planner.local_search`, `planner.py:423`). A move counts
only when it improves the objective by more than `MIN_GAIN = 1e-12`
(`planner.py:41`).

Pinned by: `tests/test_memory.py` (recall formula, review update
rule, initial stability, fitting),
`tests/test_stage3_studyforge_01.py` (as-of values hand-derived),
`tests/test_planner.py` (`test_deterministic_given_seed`,
`test_local_search_never_hurts`).

## 9. As-of evaluation discipline

Each exam's recall is computed from the memory state as of that
exam's day. Only logged reviews strictly before the exam day may
inform the estimate; otherwise the state is rebuilt by replaying the
recorded review history from the initial stability
(`planner._asof_base`, `planner.py:241`). When the recorded history
cannot reconstruct the as-of state (no history was supplied, or a
migrated legacy state's history names topics by display name), the
estimate is marked unavailable (`None`) instead of fabricated
(`planner.simulate`, `planner.py:277`). Planning objectives take an
explicit as-of day: only exams still in the future participate in
forward replanning (`planner.objective`, `planner.py:345`).
`state.readiness` applies the same rule to past exams.

Pinned by: `tests/test_stage3_studyforge_01.py` (shared-topic
day-7/day-21 counterexample, no-reviews-before-exam case, past-exam
exclusion, historical readiness invariance, unavailable marking for
display-name legacy history).

## 10. Same-day review behavior

A planned session on the same day as the last review (logged or
planned) changes nothing: a repeat review on the same day sees recall
1 and gains no stability (`planner.simulate`, `planner.py:277`; the
same rule in `memory.TopicState.review`, `memory.py:126`). At most one
session per topic per day affects the memory state, so the
two-per-day cap (`planner.MAX_PER_TOPIC_PER_DAY = 2`,
`planner.py:40`) is a scheduling guardrail against cramming, not two
learning events.

Pinned by: `tests/test_memory.py`
(`test_same_day_review_gives_no_gain`) and
`tests/test_stage3_studyforge_7.py`
(`test_planner_session_on_last_review_day_is_a_noop`,
`test_planner_second_same_day_session_is_a_noop`,
`test_planner_next_day_session_counts`).

## 11. Rest days and the no-review prior

No session is ever scheduled on a rest day, on or after a topic's
exam day, and no day exceeds its minutes cap
(`planner.greedy` / `planner.local_search`, `planner.py:384` /
`planner.py:423`). A topic never reviewed as of an exam day has
recall zero there, and its first review is treated as full learning
(`R = 0` at review, so stability doubles)
(`planner.simulate`, `planner.py:277`).

Pinned by: `tests/test_planner.py`
(`test_plan_respects_rest_days`,
`test_no_sessions_on_or_after_exam_day`,
`test_plan_respects_daily_cap`,
`test_simulate_no_sessions_is_baseline`),
`tests/test_memory.py` (`test_fresh_topic_has_zero_recall`,
`test_first_review_gets_full_gain`).

## 12. The example plan regenerates deterministically

Building the plan from the included `exams.yaml` twice, from fresh
state both times, yields the identical plan and the identical
objective: 72 sessions, objective 2.278051. The same figures are
published in the README evidence table.

Pinned by: `tests/test_stage3_studyforge_7.py`
(`test_example_plan_regenerates_deterministically`) and
`tests/test_dashboard.py` (dashboard objective against the same
figure).
