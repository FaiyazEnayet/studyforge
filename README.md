<p align="center"><img src="docs/readme-banner.svg?v=type-20261017b" alt="StudyForge: Study scheduling under time and exam constraints" width="1200"></p>

<h1 align="center">StudyForge</h1>

<p align="center">
  <a href="https://github.com/FaiyazEnayet/studyforge/actions/workflows/ci.yml"><img src="https://github.com/FaiyazEnayet/studyforge/actions/workflows/ci.yml/badge.svg?label=CI&labelColor=24292f" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-4f6549?labelColor=24292f" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?labelColor=24292f" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/built%20with-NumPy%20%2B%20planning-013243?labelColor=24292f" alt="Built with NumPy + planning">
</p>

<p align="center">
  <a href="https://faiyazenayet.github.io/studyforge/docs/dashboard.html"><img src="https://img.shields.io/badge/Open-Dashboard-4f6549?style=for-the-badge&labelColor=f4edda" alt="Open Dashboard"></a>
  <a href="#method"><img src="https://img.shields.io/badge/Explore-Method-363d30?style=for-the-badge&labelColor=f4edda" alt="Explore Method"></a>
  <a href="#run-locally"><img src="https://img.shields.io/badge/Run-Locally-626954?style=for-the-badge&labelColor=f4edda" alt="Run Locally"></a>
</p>

StudyForge allocates limited study time across exams. Its schedule makes assumptions about forgetting, topic importance and available time explicit, so a student can inspect the plan rather than accept an unexplained priority list.


<p align="center"><a href="https://faiyazenayet.github.io/studyforge/docs/dashboard.html"><img src="docs/demo.gif?v=showcase-20261017b" alt="StudyForge demo"></a></p>


The calendar makes the cost of competing deadlines concrete. An additional session for one topic consumes time that could support another exam, while spacing and daily limits restrict what can be scheduled. Read the resulting allocation beside predicted exam recall to judge whether the plan reflects the student’s priorities and available study time.

## Explore

- Inspect the actual calendar and the sessions assigned to each day.
- Compare predicted exam readiness and topic forgetting curves.
- Evaluate an extra-hour scenario using the same planning engine.

## Method

A transparent forgetting curve estimates recall at each exam. Greedy allocation and local search choose fixed-length sessions under daily limits, rest days, deadlines and spacing constraints. Logged recall can personalize stability estimates.

## Evidence at a glance

| Scope | Recorded result or contract |
| :--- | :--- |
| Example plan | Three exams and 15 topics over 21 days |
| Scheduled work | 72 sessions; 1,800 study minutes |
| Planning constraints | 120-minute daily cap; 25-minute sessions; three rest days |
| Readiness | Model estimates, not observed future exam scores |

## Updating this project

Dashboard source language: **Python**. Generator: `tools/make_dashboard.py`. The local interactive output is `docs/dashboard.html`.

## Run locally


```sh
python -m pip install -e ".[dashboard,dev]"
python -m pip install -r portfolio_ui/requirements.txt
python -m pytest tests -q
python -m studyforge plan --exams exams.yaml
python tools/make_dashboard.py
```

## Technical design

StudyForge models recall with an exponential forgetting curve and increases stability when a review occurs. Fixed-length sessions are allocated using greedy marginal gain followed by local improvement. Daily capacity, rest days, exam deadlines and a spacing guardrail constrain the plan.

```mermaid
flowchart LR
  N0["Exam weights and deadlines"] --> N1["Planning constraints"]
  N2["Recall model"] --> N3["Marginal study value"]
  N1["Planning constraints"] --> N4["Greedy and local search"]
  N3["Marginal study value"] --> N4["Greedy and local search"]
  N4["Greedy and local search"] --> N5["Study calendar"]
  N6["Review logs"] --> N2["Recall model"]
```

## Evaluation protocol

The objective is weighted predicted recall at the exam, not a predicted course grade. Logged recall can personalize stability, but until enough observations exist the parameters are defaults. What-if analysis reruns the same planner with changed time capacity rather than adding a fabricated readiness gain.

![Readiness under the example study plan](docs/results.png)

The example calendar is a real engine output for its supplied weights and constraints. Its quality depends on those inputs. For conceptual or proof-based exams, understanding-oriented tasks need to be represented separately rather than assuming recall alone measures readiness.

## Code and data

`memory.py` defines forgetting and review updates. `planner.py` handles allocation. `state.py` manages exam configuration and history. `tools/make_dashboard.py` presents actual planner output.


## Limitations

- Initial curve parameters are defaults until enough personal recall data is logged.
- Recall optimization is not a model of conceptual understanding.
- Topic names currently collide across exams and need stable exam/topic identifiers.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
