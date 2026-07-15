# Meeting Log and Decision Record

Record short notes after every supervisor or team meeting. Keep decisions here even if discussion happens elsewhere.

## Template

```md
## YYYY-MM-DD — Meeting title

**Attendees:**
**Purpose:**

### Decisions
-

### Actions
| ID | Action | Owner | Due date |
| --- | --- | --- | --- |
| | | | |

### Questions / risks
-
```

## 2026-07-15 — Workspace setup

**Attendees:** Team
**Purpose:** Create a shared local record based on the project brief.

### Decisions

- The initial core scope is a reproducible non-stationary/non-stochastic multi-armed-bandit benchmark.
- An RL extension and theoretical regret analysis remain optional until supervisor expectations are confirmed.

### Actions

| ID | Action | Owner | Due date |
| --- | --- | --- | --- |
| T01 | Confirm scope and assessment expectations | Zuting | TBD |
| T02 | Start literature scan | Xuantong | TBD |

## 2026-07-15 — Sprint 1 kickoff draft

**Attendees:** Team
**Purpose:** Convert the initial idea into a two-week executable evaluation sprint.

### Decisions

- First evaluate fixed UCB1, Thompson Sampling, Discounted UCB, Sliding-Window UCB, and EXP3 policies.
- Treat algorithm/environment matches as hypotheses to test, not conclusions.
- Use synthetic Bernoulli potential outcomes for controlled comparisons.
- Preserve a later practical recommendation case study and dynamic policy router as Phase 2.

### Actions

| ID | Action | Owner | Due date |
| --- | --- | --- | --- |
| T10 | Review protocol with team/supervisor | Zuting | 2026-07-16 |
| T11 | Implement UCB1 from the shared interface | Xuantong | 2026-07-17 |
| T13 | Implement Discounted UCB from the shared interface | Zuting | 2026-07-17 |
