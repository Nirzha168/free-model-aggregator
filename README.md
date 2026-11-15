# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Nirzha168/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Nirzha168/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

The best free AI model aggregator: one Node 22 server that speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE streaming) over ten free-tier providers, always answering with the best free model that meets the request's requirements — ranked by the sealed quality table, with failover to the next-best provider when one refuses, and a catalog it probes from the providers themselves. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

With `MAKEFUN_PROMPT_FILE` and `MAKEFUN_ENV_URL` set (a trial), the server probes the ten providers, registers its port and catalog with the env, serves the 20-request workload, and exits 0 when the env says the work is done. `node main.ts --list` prints the catalog document byte-for-byte as registered; outside a trial it prints `{"providers":[]}`.

## How it picks a model

- Candidates are every free (provider, model) pair meeting the request's requirements (tools, vision, JSON mode, minimum context), ranked by rating, best first. A JSON-mode request never goes to an Anthropic-wire provider — that wire carries no `response_format`, so the provider would echo JSON mode absent.
- One attempt per provider per request: a refusal (402/429/5xx) belongs to the provider — its budget, failure schedule or outage — so the next candidate is tried instead. The next request starts from the top again, because a single refusal may be the provider's scheduled failure rather than a spent budget.
- A provider that refuses three attempts in a row is treated as spent for the rest of the trial (the generator schedules at most a pair of consecutive failures), which keeps the provider-call count — the tie-break — down without giving up any live top model.
- Every translation carries the request untouched: system prompt, user text and images, tool definitions, stop sequences, `max_tokens`, JSON mode. Usage counts are carried verbatim from the answering provider, and stop reasons map by family across wires. Streams are collected and re-emitted in the client's wire so the usage lands in the right place. No wall clock and no unseeded randomness anywhere, so a replay reproduces the same state export byte for byte.

## Layout

- `main.ts` — the server: brief, catalog probe, routing with failover, register and work polling.
- `lib/brief.ts` — the brief's three documents, the catalog probe, the catalog document and candidate ranking.
- `lib/translate.ts` — the three provider wires × the two client wires: request builders, JSON response converters, SSE collectors and emitters.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kn891vzn&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
