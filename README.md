# studyforge

**Exam study scheduling under a forgetting curve.** You list your exams, their topics, and how much each topic matters. studyforge builds a day by day plan that maximizes what you will actually remember on exam day.

I kept making study plans by gut feel. Front load everything, cram the night before, forget half of it by the exam. Spaced repetition research has known the fix for a century: review a topic right as you are about to forget it. So I built the planner I wanted. It models memory with a forgetting curve, then optimizes the schedule the way you would optimize anything else. Define the objective, respect the constraints, let the algorithm do the rest.

## The math

Each topic has a stability S measured in days. Recall t days after a review is:

    R(t) = exp(-t / S)

Harder topics start with lower stability. A difficulty rating from 1 to 5 maps to S0 = 3.0 / difficulty, so an easy topic starts at 3.0 days and a hard one at 0.6 days.

Reviews grow stability. The update rule is fully transparent, SM-2 style, no black box:

    S' = S * (1 + (1 - R))

R is retrievability at the moment of review. Review at full recall and stability does not move. Review after forgetting everything and stability doubles. The first ever review of a topic counts as full learning, and same day repeats gain nothing, which stops the planner from cramming one topic.

The planner maximizes the sum over topics of (exam weight * predicted recall at exam time). Constraints: a daily minutes cap, rest days with zero sessions, no sessions on or after a topic's exam day, at most two sessions of one topic per day. Algorithm: greedy by marginal gain per minute, then a local search pass that moves or drops sessions until nothing improves. Deterministic. Same inputs give the same plan.

### Your own forgetting rates

The defaults above are a starting guess. When you log a session, add --recall with the fraction you actually remembered when you sat down, like 0.7 for seventy percent. Each logged review with a recall score becomes one data point: the days since your last review of that topic, and what you remembered. Once a topic has three such data points, I fit its stability to your history with least squares on log recall, clamped to sane bounds. That fitted value replaces the default in plan and replan from then on, and every new logged review refits it. When you report a recall score, the stability update uses your number instead of the model's prediction. Topics without enough data keep the defaults, and status tells you which is which.

Real output. Three earlier reviews of Rotation were logged with recall scores, so this one triggers the fit:

```
$ python -m studyforge log --topic "Rotation" --minutes 25 --day 9 --recall 0.14
Logged 25 min on Rotation (day 9).
Retrievability at review: 14%. Stability: 4.35 -> 8.09 days.
Fitted your personal stability from 3 recall data points: 2.01 days.

$ python -m studyforge status --today 9
...
Rotation                         Physics I Midterm         2.01    100%       5      8%   fitted
...
Stability: 1/15 topics fitted from your logs. The rest use defaults. Log sessions with --recall to personalize more topics.
```

## Quickstart

```bash
pip install -e .
python -m studyforge plan --exams exams.yaml
python -m studyforge status
python -m studyforge log --topic "Rotation" --minutes 25
python -m studyforge replan
python -m studyforge what-if --day 3 --extra-minutes 60
python -m studyforge export --format md --out plan.md
```

State lives in `studyforge_state.json` in the working directory. Override with `--state`.

## Demo transcript

Real output from the included `exams.yaml`: three exams, 15 topics, 21 days, 120 minutes per day, Sundays off.

```
$ python -m studyforge plan --exams exams.yaml
Study plan: 72 sessions, objective 2.278
Budget: 120 min/day, sessions of 25 min
Day 0 (2026-10-10):
  Psychology Midterm / Brain anatomy: 25 min
  Psychology Midterm / Classical conditioning: 25 min
  Psychology Midterm / Memory models: 25 min
  Physics I Midterm / Rotation: 25 min
Day 1 (2026-10-11):
  Psychology Midterm / Memory models: 25 min
  Physics I Midterm / Momentum: 25 min
  Physics I Midterm / Newton's laws: 25 min
  Psychology Midterm / Sleep and dreams: 25 min
Day 2 (2026-10-12):
  Psychology Midterm / Brain anatomy: 25 min
  Psychology Midterm / Classical conditioning: 25 min
  Psychology Midterm / Research methods: 25 min
  Physics I Midterm / Work and energy: 25 min
...
Day 6 (2026-10-16): rest
...
Day 20 (2026-10-30): rest
```

Psychology topics cluster before the day 7 exam. Calculus topics ramp up toward day 21. Rest days stay empty.

`status` shows per topic recall and a per exam readiness score, which is weight normalized predicted recall at exam time with no further study:

```
$ python -m studyforge status --today 2
...
Per-exam readiness (weighted recall at exam time, no more study):
  Psychology Midterm: 3% (5 topics, 5 days left)
  Physics I Midterm: 0% (5 topics, 12 days left)
  Calculus III Final: 0% (5 topics, 19 days left)
```

Log what you actually study. Each log updates that topic's stability through the review rule:

```
$ python -m studyforge log --topic "Rotation" --minutes 25 --day 0
Logged 25 min on Rotation (day 0).
Retrievability at review: 0%. Stability: 0.75 -> 1.50 days.
```

Then replan from today. Logged sessions count as past reviews, so the optimizer redistributes the remaining work:

```
$ python -m studyforge log --topic "Memory models" --minutes 25 --day 0
$ python -m studyforge log --topic "Brain anatomy" --minutes 25 --day 0
$ python -m studyforge log --topic "Series convergence" --minutes 50 --day 1
$ python -m studyforge replan --today 2
Replanned from day 2: 64 sessions, objective 2.233
Added sessions:
  Day 2: Physics I Midterm / Momentum: 25 min
  Day 3: Psychology Midterm / Memory models: 25 min
  ...
Removed sessions:
  Day 2: Psychology Midterm / Brain anatomy: 25 min
  Day 3: Calculus III Final / Series convergence: 25 min
  ...
```

`what-if` prices extra time. Here an extra hour on day 3 barely moves the objective, because the plan is already saturated at 120 minutes per day. That is the honest answer and the useful one:

```
$ python -m studyforge what-if --day 3 --extra-minutes 60 --today 2
Baseline objective: 2.233
With +60 min on day 3: 2.235 (+0.002)
Sessions added by the extra time:
  Day 2: Calculus III Final / Multiple integrals: 25 min
  Day 2: Calculus III Final / Partial derivatives: 25 min
  Day 3: Psychology Midterm / Classical conditioning: 25 min
  Day 3: Calculus III Final / Lagrange multipliers: 25 min
  Day 3: Psychology Midterm / Research methods: 25 min
  Day 4: Calculus III Final / Series convergence: 25 min
  Day 14: Calculus III Final / Vector fields: 25 min
  Day 16: Calculus III Final / Multiple integrals: 25 min
Sessions displaced:
  Day 2: Psychology Midterm / Classical conditioning: 25 min
  Day 2: Psychology Midterm / Research methods: 25 min
  Day 4: Physics I Midterm / Work and energy: 25 min
  Day 14: Calculus III Final / Lagrange multipliers: 25 min
  Day 16: Calculus III Final / Lagrange multipliers: 25 min
```

`export` writes the plan to CSV or Markdown for printing or sharing.

## Honest limitations

The curve parameters start as population averages, not yours. Ebbinghaus ran his experiments on himself with nonsense syllables in the 1880s. The shape of forgetting is real. The exact numbers for your brain on organic chemistry are not in this repo, at least not until you log them. Log reviews with --recall and each topic gets its own fitted forgetting rate once you have enough data. Until then the defaults stand in, and status marks which topics are which. Treat the schedule as principled guidance, not a guarantee.

It optimizes recall, not understanding. A topic you can recite is not a topic you can use on a novel problem. For proof based exams, weight understanding work outside the tool.

Garbage topic weights in, garbage plan out. The optimizer trusts your weights completely. If you guess them badly, it will confidently optimize the wrong thing. Spend ten minutes getting weights right before you trust the plan.

## Config reference

```yaml
daily_budget_minutes: 120
session_minutes: 25
rest_days: [6, 13, 20]   # day offsets with no sessions
exams:
  - name: "Physics I Midterm"
    day: 14              # days from today
    topics:
      - { name: "Rotation", difficulty: 4, weight: 0.20 }
```

Difficulty runs 1 (easy) to 5 (hard). Weight is the topic's share of the exam.
