# Notes, deviations and findings

## Layout

The workspace *is* the repository, so the project lives at the repo root
(`app.py`, `oncall/`, `data/`, ...) rather than in a nested `oncall-memory/`
directory. All commands in the README assume the repo root.

## Hindsight API — verified against the docs and the installed client

Installed: `hindsight-client==1.2.0` (official HTTP client). Verified signatures
by inspecting the package and https://hindsight.vectorize.io/sdks/python:

```python
Hindsight(base_url=..., api_key=..., timeout=...)
client.create_bank(bank_id, name=..., mission=..., disposition={...})
client.retain(bank_id, content, context=..., timestamp=..., document_id=..., metadata={...})
client.recall(bank_id, query, types=[...], budget="mid", max_tokens=1400, query_timestamp=...)
client.reflect(bank_id, query, budget="mid", context=...).text
client.list_memories(bank_id, limit=1000)
client.delete_bank(bank_id)
```

Differences from the original brief:

- **`recall` has no `k` parameter.** Retrieval is budgeted in tokens
  (`max_tokens`), so the wrapper asks for 1400 tokens and slices/group its
  results client-side.
- **Cloud auth is a bearer key** (`HINDSIGHT_API_KEY`); Cloud base URL is
  `https://api.hindsight.vectorize.io`. Self-hosted uses `http://localhost:8888`
  and usually no key.
- **`document_id` is an upsert key**, which gives idempotent seeding and
  duplicate-click protection for free.
- **Bank mission/disposition** are passed to `create_bank` (they shape `reflect`).

### Finding: consolidated observations lose provenance

Live API inspection showed that recalled `observation` results come back with
`metadata=None` and no `document_id`. Our whole attribution model (incident ID,
failure family, exact worked/failed counts) relies on the metadata attached at
retain time. Recall therefore requests `types=["world","experience"]` (raw facts)
and leaves observations to `reflect`, where a synthesized view is what we want.
This is a deliberate trade-off, noted here rather than silently accepted.

### Finding: cloud recall often returns `scores` of 0

The Cloud API returned rankings in relevance order but zero/empty numeric scores
for these facts. The wrapper keeps the returned order when scores are absent and
uses scores only to order groups when present. It never invents a score.

## Data and evaluation decisions

- **Eval incidents are stored unresolved**: `root_cause=""`, `fixes_tried=[]`,
  `resolved_by=""`. Public `incidents_eval.json` has no `true_fix` field. The
  simulator-only answer map is stored separately in `incidents_eval_answers.json`
  and is loaded only by the benchmark; the agent never receives it.
- **25 eval incidents** = 22 recurring patterns, 2 genuinely novel patterns
  (Kafka consumer lag, signing-key rotation), plus **one repeat of the Kafka
  pattern** (`INC-121`). The repeat is marked `novel=false`: it is not in the
  seeded history, but condition C can learn it from `INC-119`. It remains the
  clearest B-vs-C difference, because the five designed-ambiguity incidents are
  already answerable from the seeded postmortems (they separate A from B, not B
  from C).
- **History** is 32 incidents across the 5 required families. It contains three
  explicit traps where restart fails and a specific fix works, plus
  `increase_pool_size` working once (`INC-009`) and failing later (`INC-019`) so
  conflicting memory is real data, not a fixture. `INC-029`–`INC-032` are
  hand-written trails that record a constraint instead of a generic note (see
  "Dataset design and freeze").
- **Simulated minutes**: 6 base + 8 per failed attempt. The numbers in the CSV
  are computed from attempts, not hand-entered.
- **Banks**: demo bank comes from `HINDSIGHT_BANK_ID`; the benchmark refuses to
  use that ID and creates `oncall-eval-<run>-<condition>` banks, deleting them at
  the end unless `--keep-bank`.

## Dataset design and freeze

- **Authored, not captured.** The world in `data/` is simulated: services,
  deploys, logs, metrics and incidents are generated from hand-written family
  templates by `scripts/generate_data.py`, and the answers live in
  `data/incidents_eval_answers.json`, which the agent never reads. No figure in
  this repository comes from a real production system.
- **Ambiguity is deliberate, and the constraint lives only in history.**
  `INC-122`–`INC-126` were designed so that the fix their own logs suggest is
  wrong: a PCI audit freeze that forbids rolling the release back (`INC-029`), a
  settlement batch that must not be replayed (`INC-030`), an index that blocks
  writes while a backfill runs (`INC-031`), a shared Redis cluster at
  `maxclients` (`INC-032`). The alert, symptoms, log extract and deploy summary
  never state the constraint; the earlier postmortem does. The logs still contain
  the signature, so the incidents detect as their family and recall is not
  filtered — the ambiguity is inside a family, not between families.
- **Frozen before the first run.** The eval set and its answer map were authored
  and frozen before any benchmark run used them. No incident, log line or answer
  was edited after seeing results; the redesign in this section was committed
  before the first run on the new set.
- **Scoring was fixed in advance.** Top-1 is an exact id match against the single
  hidden `true_fix`, and simulated minutes are computed from attempts (6 base + 8
  per failure), never hand-entered. Neither the metric, the acceptable answers
  nor the prompt were adjusted once results existed.
- **One prompt revision, disclosed.** After the freeze, a smoke run on `INC-122`
  with memory ON showed the model following family-wide tallies over a recalled
  same-service postmortem that recorded a team constraint (and returning no
  evidence). The prompt and the outcome scoping were revised once in response:
  tallies are now split into same-service and family-wide and both are labelled
  in the context, rule 3 says a matching recorded constraint overrides the
  tallies, and `evidence` is now required with at least one line. That happened
  before any benchmark run on this set; the prompts are frozen from that commit.
- **The novel families stay novel.** `kafka_consumer_lag` and
  `signing_key_rotation` have no history records at all, so for `INC-119`/`121`
  and `INC-120` memory can only help through in-run learning (after `INC-119`),
  never through seeding.

## Reliability choices

- **Pattern gate.** `tools.detect_pattern` classifies the failure from distinctive
  strings in the alert/symptoms/logs (`data/families.json`). It is runbook-style
  keyword logic over visible evidence — **not** the incident's stored `family`
  label. It reproduces the stored family for 57/57 generated incidents, which
  `python scripts/validate_data.py` checks, including per-eval-incident so a
designed-ambiguity case cannot silently drift into a neighbouring family. Recall
  results from another pattern are dropped so a novel alert can honestly return
  "no relevant history" instead of citing a superficially similar old incident.
- **Memory guardrail.** After LLM validation, fixes with recorded failures are
  stably demoted below fixes with recorded successes; a recorded-success fix
  missing entirely is surfaced at the top. This encodes the brief's system-prompt
  rule in code so a single bad generation cannot ignore it.
- **Deterministic analyst.** If no `GROQ_API_KEY` is set, or every Groq attempt
  fails, a rule-based analyst answers from log signatures + recalled outcomes.
  It is labelled everywhere (`llm_model="heuristic (...)"`, UI caption, CSV
  column) so benchmark numbers cannot be mistaken for LLM output.
- **Local keyword memory.** If Hindsight is not configured/unreachable, a local
  BM25-ish store under `.memory_store/` keeps the demo alive. The UI says
  "degraded mode ... Not Hindsight" and every investigation records
  `memory_backend=local-keyword`. It is a fallback, not the submitted memory.

## Agentic mode

`AGENT_MODE=agentic` implements function-calling tool selection (logs, deploys,
metrics, memory), capped at 6 tool calls, then reuses the same validated final
step. Any exception (malformed arguments, unsupported tool, model error) is
caught and the deterministic pipeline runs instead, with the reason recorded.
The default remains `deterministic` because it is the mode that has been
exercised end-to-end repeatedly; agentic is opt-in.

## Benchmark results and caveats

No A/B/C numbers are reported. The benchmark could not be completed here: one
attempt hit the Groq organization's daily token quota, and a separate run lost
Hindsight Cloud connectivity (DNS failure), which sent the seeded and learning
banks to local keyword memory. A full run exists in the development history, but
its artifacts could not be reproduced from the files in this repository, so none
of its figures are quoted.

- The benchmark calls real Groq and real Hindsight when keys are configured, and
  the CSV records the model, memory backend and whether a row fell back.
- `llm.py` deliberately prefers a fast deterministic answer over a long stall, so
  on a free tier a large share of rows can be ended by the heuristic analyst.
  The CSV labels every such row `llm_backend=heuristic (fallback)`.
- The memory evidence reported in the README is Compare mode on `INC-103`, which
  is reproducible from the seeded demo bank.

## Environment / known issues

- Windows: use `py -3.11 -m venv .venv`, then `.venv\Scripts\activate`. Some
  consoles are cp1252; prefix long-running commands with
  `PYTHONIOENCODING=utf-8` if you see replacement characters in log output.
- The Hindsight SDK's async transport sometimes prints "Unclosed client session"
  at interpreter shutdown; scripts call `memory.close()` but the SDK can still
  emit it. It does not affect results.
- `.env` is present locally for live runs and is gitignored. No secret is
  committed; `check_connections.py` only ever prints booleans/counts.
- The initial `check_connections` probe bank is deleted after the round trip.
  If deletion fails (permissions/quota), the script prints its name for manual
  cleanup.
