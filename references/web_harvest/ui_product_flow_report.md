# UI/Product Flow Harvest Report for VocalFusion

Target app reviewed: `/Users/johnpeter/Code/ai-dj-project`

Current baseline in target app:
- Backend already has the right primitive shape: `POST /fuse` → returns `job_id`, `status_url`, `share_url` in `server.py` (`start_simple_fuse_job`), then `GET /status/<job_id>` and `GET /share/<job_id>`.
- Frontend in `templates/simple_fuse.html` does simple polling every 1.5s and renders a success card.
- Share page exists in `templates/simple_fuse_share.html`, but it is very thin and single-result only.

The biggest gap is **not** basic functionality. It is the product flow between:
1. file selection/upload confidence,
2. async state visibility,
3. queue/retry/cancel affordances,
4. durable result/history/share UX.

---

## Best patterns worth borrowing

## 1) HeartMuLa Studio — best reference for music-generation app flow
Repo: `https://github.com/fspecii/HeartMuLa-Studio`

### Exact files/functions
- `frontend/src/api.ts`
  - `generateJob(...)`
  - `getJobStatus(jobId)`
  - `getDownloadUrl(jobId)`
  - `connectToEvents(onMessage)`
- `frontend/src/App.tsx`
  - SSE hookup around `api.connectToEvents(...)`
  - event handling for `job_update`, `job_progress`, `job_queued`, `job_queue`
- `frontend/src/components/HistoryFeed.tsx`
  - per-job progress state
  - stuck-job detection
  - card actions: retry, delete, play, download, like, playlist
- `backend/app/main.py`
  - `POST /generate/music`
  - `GET /jobs/{job_id}`
  - `GET /download_track/{job_id}`
  - `PATCH /jobs/{job_id}`
  - reference upload endpoint `POST /upload/ref_audio`

### What is good
- Uses **server-sent events** instead of only polling. This is better for “music render in progress” than raw interval fetch loops.
- Separates **job creation**, **job status**, **history**, and **download URL** cleanly.
- Keeps a **history feed** so results are durable and revisitable, instead of only showing the last completed job.
- `HistoryFeed.tsx` has an underrated good pattern: **stuck job detection**. If progress does not move for long enough, the UI escalates concern rather than pretending all is fine.
- Download is explicit and ergonomic: `getDownloadUrl(jobId)` instead of exposing storage paths directly.

### What is weak / not ideal for VocalFusion
- The UI is more “music generator library/feed” than “two-input fusion workflow.”
- It is broad and feature-heavy; copying whole components would overcomplicate the current simple fuse page.
- It assumes persistent job history/store earlier than VocalFusion may need.

### Best merge ideas for `ai-dj-project`
1. **Add SSE on top of the existing status endpoint**
   - Keep `/status/<job_id>` for fallback.
   - Add `/events/<job_id>` or `/api/fuse-events/<job_id>` that streams stage/progress/message updates.
   - Emit the same payload shape already stored by `_set_simple_fuse_job(...)` in `server.py`.
2. **Add a tiny run history rail** under the main fuse form
   - Show the last 10 jobs with status pill, created time, play/download/share buttons.
   - Your in-memory `_SIMPLE_FUSE_JOBS` is not durable enough for product UX; persist a compact index JSON under `runs/ui_uploads/` or `runs/ui_job_index.json`.
3. **Add explicit queue/running/done/error states**
   - HeartMuLa distinguishes queued vs processing vs completed vs failed. VocalFusion already has `stage`; expose it more clearly in UI.
4. **Use safe download routes, not raw path leakage**
   - Prefer stable URLs based on `job_id` for primary download/share CTAs.

---

## 2) Uppy — best reference for upload UX, progress math, and recovery
Repo: `https://github.com/transloadit/uppy`

### Exact files/functions
- `packages/@uppy/dashboard/src/Dashboard.tsx`
  - main upload dashboard shell
- `packages/@uppy/status-bar/src/StatusBar.tsx`
  - upload state machine and smoothed ETA
  - `getUploadingState(...)`
  - `#computeSmoothETA(...)`
- `packages/@uppy/xhr-upload/src/index.ts`
  - robust XHR upload transport with concurrency, hooks, retries
- `packages/@uppy/golden-retriever/src/index.ts`
  - crash/tab-close recovery for selected files/uploads

### What is good
- Best-in-class **upload confidence layer**:
  - drag/drop
  - visible selected files
  - resumable-ready transport
  - ETA/speed math
  - clear states: waiting / preprocessing / uploading / postprocessing / complete / error
- `GoldenRetriever` is especially relevant if users upload big stems or long WAVs and refresh/tab-crash.
- `XHRUpload` and status-bar code handle the ugly operational details that simple hand-written forms usually skip.

### What is weak / not ideal for VocalFusion
- It solves **upload UX**, not the whole music-job lifecycle.
- Pulling Uppy wholesale into a Flask/Jinja prototype may be too much if you want to stay lightweight.

### Best merge ideas for `ai-dj-project`
1. **Steal the state model, not necessarily the whole library**
   - Current `templates/simple_fuse.html` has one animated purple bar that mostly means “something is happening.”
   - Replace with explicit phases:
     - validating files
     - uploading song A
     - uploading song B
     - analyzing compatibility
     - planning arrangement
     - rendering output
     - packaging result
2. **Add client-side preflight before submit**
   - duration/file size/type checks
   - visible selected filenames + sizes
   - warning for non-MP3 support mismatch (UI currently says MP3 while backend supports more formats)
3. **Add upload error granularity**
   - Uppy’s model distinguishes transport error vs server processing error.
   - In your current page, both collapse into generic “Fuse failed.”
4. **If bigger uploads are expected, add resumable uploads later**
   - Especially if VocalFusion moves from MP3 demo input to full WAV/stems.

---

## 3) abogen — best reference for queue/results/details/download workflow
Repo: `https://github.com/denizsafak/abogen`

### Exact files/functions
- `abogen/webui/service.py`
  - job model fields: `status`, `progress`, `processed_characters`, `queue_position`
  - `estimated_time_remaining`
  - queue bookkeeping
- `abogen/webui/templates/partials/jobs.html`
  - split between **Active jobs** and **Recent results**
  - inline actions: Details / Pause / Resume / Cancel / Retry / Download / Remove
- `abogen/webui/templates/job_detail.html`
  - dedicated detail page for one job
- `abogen/webui/static/queue.js`
  - lightweight page behavior around queue interactions

### What is good
- Excellent **ops-oriented job board** pattern.
- Separates **active jobs** from **recent results**, which is exactly the missing middle layer between your `/` and `/status` pages.
- Shows **queue position**, **ETA**, and **details** without forcing a giant debug dashboard.
- Keeps downloads attached to the finished result card, which is cleaner than burying artifacts in a diagnostics page.

### What is weak / not ideal for VocalFusion
- It is designed for audiobook conversion, so its metrics are text/character based.
- The UI is more utilitarian than consumer-polished.
- No especially good “viral share page” pattern.

### Best merge ideas for `ai-dj-project`
1. **Introduce a 2-column result board under the fuse form**
   - Section A: Active fusions
   - Section B: Recent completed fusions
2. **Add a dedicated job details page**
   - Current share page is public-facing but too sparse.
   - Add a private `/job/<job_id>` page with:
     - input filenames
     - timestamps
     - stage timeline
     - stdout snippet / fallback reason
     - output links
     - share link
3. **Expose retry/cancel explicitly**
   - Your backend job thread model can support cancellation only if the subprocess can be interrupted; even if not implemented immediately, the UI structure is worth adopting.
4. **Show ETA only if you can estimate honestly**
   - If not enough signal exists, show stage + elapsed time instead of fake percentages.

---

## Direct review of current VocalFusion flow

## What is already good
- Clean primitive API shape in `server.py`.
- `simple_fuse.html` is fast to test and does not overcomplicate the prototype.
- `simple_fuse_share.html` already gives a separate share endpoint.
- Backend preserves a `stage`, `message`, `progress`, `result`, `share_url`, and `run_id`, which is enough to build a much better UI without a deep backend rewrite.

## What is currently weak
1. **Progress is cosmetically animated rather than meaningfully informative**
   - `templates/simple_fuse.html` uses a moving bar, but the user mostly gets one line of text.
2. **Polling is blunt**
   - `setInterval(..., 1500)` is fine for prototype use, but SSE/websocket would feel more alive and reduce wasted fetches.
3. **No durable history view on the main page**
   - Once a run finishes, the page becomes a single success card, not a reusable workbench.
4. **Share page is too thin**
   - It has playback and download, but no input metadata, no artwork/waveform, no “copy share link”, no compare/reference context.
5. **UI/backend file-type messaging is inconsistent**
   - UI asks for MP3 only, backend allows MP3/WAV/FLAC/M4A/AAC.
6. **No upload-level UX**
   - no drag/drop
   - no file size/duration preview
   - no separate upload state vs render state
7. **`/status` page is debug-heavy rather than job-centered**
   - useful for operator diagnostics, but not the right primary UX for a creator uploading two tracks.

---

## Concrete merge plan for `/Users/johnpeter/Code/ai-dj-project`

## Phase 1 — highest ROI, minimal backend risk
1. **Upgrade `templates/simple_fuse.html` into a proper staged job card**
   - Replace the single generic progress bar with:
     - file chips for Song A / Song B
     - stage label
     - elapsed time
     - percent
     - small checklist/timeline
   - Keep polling for now.
2. **Make the result card durable**
   - After success, append to a “Recent fusions” list instead of replacing the whole result area.
3. **Improve `templates/simple_fuse_share.html`**
   - Add:
     - copy-link button
     - input filenames
     - render timestamp
     - fallback/source badge (`selection_source`, `fallback_reason`)
     - large primary CTA: Download
4. **Fix file-type messaging**
   - Align UI copy and `<input accept>` with backend support.

## Phase 2 — product flow upgrade
5. **Add SSE for job updates**
   - Mirror the HeartMuLa approach.
   - Fallback to polling if EventSource fails.
6. **Add `/job/<job_id>` private detail page**
   - Borrow the abogen “job detail” pattern.
7. **Persist lightweight job index**
   - So refresh does not lose the local session story.

## Phase 3 — upload confidence and resilience
8. **Adopt Uppy or emulate its interaction model**
   - drag/drop area
   - selected file list
   - better upload progress/error states
9. **If larger source files become common, add resumable uploads**
   - especially for WAV/stems
10. **Add crash/refresh recovery for pending jobs**
   - even a simple `localStorage` restore of `activeJobId` is a big improvement

---

## Specific code areas to change first

### Backend
- `server.py`
  - `_set_simple_fuse_job(...)`
    - already the central state patch point; expand it into a canonical event payload
  - `start_simple_fuse_job()`
    - return richer accepted payload
  - `simple_fuse_job_status(job_id)`
    - keep as polling fallback
  - add new route: `/events/<job_id>` or `/api/fuse-events/<job_id>`
  - add new route: `/job/<job_id>`

### Frontend
- `templates/simple_fuse.html`
  - replace current one-box flow with:
    - upload panel
    - active job panel
    - recent results panel
- `templates/simple_fuse_share.html`
  - upgrade into a presentable result/share surface
- optional: split inline JS into `static/simple_fuse.js` once the page grows

---

## Best ideas to borrow exactly

### Borrow from HeartMuLa
- Event-driven status updates
- explicit job lifecycle vocabulary
- durable result/history cards
- stable download endpoints by job id

### Borrow from Uppy
- upload state machine
- ETA/progress treatment
- drag/drop + selected file visibility
- recovery mindset for larger uploads

### Borrow from abogen
- active jobs vs recent results split
- per-job detail page
- inline actions: retry/cancel/details/download
- queue position / elapsed / ETA patterns

---

## Recommended final product shape for VocalFusion

For this project, the strongest UX is probably:

1. **Home page = creation surface**
   - pick two tracks
   - validate compatibility inputs
   - start fuse
   - show live stage timeline

2. **Inline active job card**
   - stage
   - percent
   - elapsed time
   - logs/details expander

3. **Completed result card**
   - audio player
   - download
   - share
   - “open details”
   - maybe “compare against previous” later

4. **Share page**
   - stripped-down public playback/download page
   - polished, not debuggy

5. **Operator status page**
   - keep `/status`, but position it as internal diagnostics, not main user flow

That keeps the current prototype simple while moving it toward a real async audio-product UX instead of a debug form.
