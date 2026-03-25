# UI / server critic-surface result

## What changed
- Updated `server.py` so the simple product flow is explicitly modeled as **baseline-first** instead of just "direct output" wording.
- Added server-side summaries for:
  - `product_flow` (baseline-first contract / headline / baseline path)
  - `critic_loop` (headline, latest decision, next focus, listener-gate / benchmark / compare / listen rollup)
- Wired those summaries into `/api/status` and into direct-fusion result payloads used by the simple fuse job/share flow.
- Updated `templates/simple_fuse.html` to visibly describe the default path as baseline-first and to surface critic-loop context in the success card.
- Updated `templates/simple_fuse_share.html` to show baseline-first framing plus critic-loop headline / decision / next-focus summary on the share page.
- Updated `templates/status.html` to add dedicated **Product flow** and **Critic-loop decision** cards so the status page presents the new sprint direction directly.

## Tests updated
- Expanded `tests/test_debug_server.py` assertions to cover:
  - baseline-first wording on `/`
  - new status-page cards
  - new `/api/status` fields (`product_flow`, `critic_loop`)
  - baseline-first + critic-loop details on share output
  - baseline-first metadata on simple fuse async job / direct fuse upload payloads

## Validation
- Ran: `python -m pytest -q tests/test_debug_server.py`
- Result: `27 passed`

## Notes
- Kept scope to `server.py`, `templates/*`, and related tests only.
- Did not touch planner/render core logic.
- Critic-loop surfacing currently summarizes the latest available listener-agent / closed-loop / benchmark / compare / listen artifacts; it does not invent new audio-side evaluation work in the simple baseline path.