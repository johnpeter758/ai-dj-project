# Gateway Mesh Readiness (Local-first + Cloud Failover)

## Routing shape

- **Lane A (primary):** local Ollama for lowest latency/cost and partial offline tolerance.
- **Lane B (fallback):** cloud model via LiteLLM for continuity when local is unhealthy.
- **Fail policy:** retry-in-lane first; after lane failures exceed threshold, cooldown lane and route to fallback.

## Concrete commands

```bash
# 1) Start local model lane
ollama serve
ollama pull llama3.2

# 2) Run LiteLLM gateway with reliability config
litellm --config config/litellm.reliability.yaml --port 4000

# 3) Probe health (expected 200 JSON)
curl -sS http://127.0.0.1:4000/health | jq .

# 4) Smoke request through router
curl -sS http://127.0.0.1:4000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vocalfusion_primary",
    "messages": [{"role": "user", "content": "phase-12 dry run"}]
  }' | jq '.model, .usage'
```

## Health-check steps

1. **Gateway liveness:** `curl /health` succeeds.
2. **Local lane readiness:** `ollama ps` shows loaded model, request latency in expected range.
3. **Fallback lane readiness:** run one forced cloud call (`model: vocalfusion_cloud_fallback`) and verify non-empty output.
4. **Autopilot state sanity:** `cat runs/autopilot/state.json | jq .status,.next_step`.
5. **Promotion gate sanity:** run dry gate and confirm deterministic machine JSON output.

## Failure-mode behavior

- **Ollama down/unresponsive:** retries occur; lane enters cooldown (`allowed_fails`/`cooldown_time`) and cloud fallback handles traffic.
- **Cloud transient errors:** bounded retries (`num_retries`) before cycle marked failed by orchestration state.
- **Both lanes unhealthy:** autopilot cycle records failure details in `runs/autopilot/state.json` with next-step pointers.
- **Operator halt:** create `AUTOPILOT_STOP` file; orchestrator exits cleanly on next checkpoint.
