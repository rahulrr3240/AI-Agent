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
- **20 eval incidents** = 17 recurring patterns, 2 genuinely novel patterns
  (Kafka consumer lag, signing-key rotation), plus **one repeat of the Kafka
  pattern** (`INC-121`). The repeat is marked `novel=false`: it is not in the
  seeded history, but condition C can learn it from `INC-119`. It is the one
  place where C can differ from B without changing the seeded families.
- **History** is 28 incidents across the 5 required families. It contains three
  explicit traps where restart fails and a specific fix works, plus
  `increase_pool_size` working once (`INC-009`) and failing later (`INC-019`) so
  conflicting memory is real data, not a fixture.
- **Simulated minutes**: 6 base + 8 per failed attempt. The numbers in the CSV
  are computed from attempts, not hand-entered.
- **Banks**: demo bank comes from `HINDSIGHT_BANK_ID`; the benchmark refuses to
  use that ID and creates `oncall-eval-<run>-<condition>` banks, deleting them at
  the end unless `--keep-bank`.

## Reliability choices

- **Pattern gate.** `tools.detect_pattern` classifies the failure from distinctive
  strings in the alert/symptoms/logs (`data/families.json`). It is runbook-style
  keyword logic over visible evidence — **not** the incident's stored `family`
  label. It reproduces the stored family for 48/48 generated incidents, which
  `python scripts/validate_data.py` checks. Recall
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

Measured run `20261106-161035-f7f9` (20 incidents × 3 conditions, Hindsight Cloud,
`openai/gpt-oss-120b`), 979 s total:

| Condition | Top-1 | Resolved | Avg attempts | Avg simulated min | Fallback runs |
|---|---|---|---|---|---|
| A · memory OFF | 35% | 60% | 2.20 | 18.8 | 10/20 |
| B · seeded | 85% | 90% | 1.20 | 8.4 | 11/20 |
| C · seeded + learning | 85% | 95% | 1.25 | 8.4 | 10/20 |

- The benchmark calls real Groq and real Hindsight when keys are configured.
  The CSV records the model, memory backend and whether any run fell back.
- **Groq free-tier TPM is the main confound.** At 8,000 tokens/minute, back-to-back
  investigations produced 429s; roughly half of all runs (in every condition)
  ended on the deterministic analyst after the retry chain. Those rows are
  flagged `llm_backend=heuristic (fallback)`. The fallback uses the same recalled
  memories, so the A/B gap is still a memory effect, but the absolute numbers
  are not all LLM output. Pacing the run or using a paid tier would give cleaner
  numbers; `llm.py` deliberately prefers a 4-second deterministic answer over a
  30-second stall.
- **C's top-1 is flat relative to B.** Seeded history already covers every
  recurring family, so learning can only help where a new pattern repeats. The
  visible case is `INC-121` (Kafka repeat): C recalls the `scale_up` outcome
  written for `INC-119` (1 memory, `no_relevant_history=false`) and gets it
  first. C also resolves `INC-116` on the second attempt, but that is because the
  LLM returned two candidates there, not because of a learned fix.
- **B's `INC-116` is unresolved** because the model returned a single ranked fix
  (`increase_pool_size`). The schema allows 1–5 fixes. A deterministic guardrail
  could pad the list, but inventing candidates just to keep the simulation going
  would hide real model behaviour, so it was left visible and is reported here.
- **Novel cases:** B and C correctly flag both `novel=true` incidents. `INC-120`
  (signing-key rotation) is still unsolved by B/C — conservative generic triage
  ranked `restart_service` and never reached the true fix, while A guessed right
  from the logs. Honest "no history" is not the same as correct.
- **Conflicting memory:** `INC-106` (config drift → `increase_pool_size`) and
  `INC-116` (deploy → `rollback_deploy_and_increase_pool`) share a family but
  need opposite fixes. B/C rank whichever variant memory favours first and mark
  the conflict with lower confidence.
- Novel-incident flag accuracy is scored only for the two `novel=true` incidents
  and only when memory is ON. Condition A has no recall to be calibrated about.

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
