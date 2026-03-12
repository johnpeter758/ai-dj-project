# Render resolver design (`src/core/render/resolver.py`)

## Goal
Turn a `ChildArrangementPlan` plus two `SongDNA` artifacts into a **deterministic render-ready timeline**.

`resolver.py` should **not** render audio. It should only:
- validate inputs
- snap source section boundaries to musically safe points
- compute exact target timeline durations
- choose per-section anchor timing
- emit explicit audio work orders for downstream transforms/mixing

That keeps planning separate from execution and makes render behavior testable.

---

## Recommended responsibility split

`resolver.py` owns:
- contract validation
- source section lookup
- beat snapping
- section duration math
- target timeline math
- deterministic work-order generation

It should **not** own:
- audio I/O
- time-stretch DSP
- pitch-shift DSP
- transition DSP
- loudness normalization

Those belong in transform/mix/export modules.

---

## Proposed input contract

## Primary entrypoint
```python
def resolve_render_plan(
    plan: ChildArrangementPlan,
    parent_a: SongDNA,
    parent_b: SongDNA,
    *,
    config: ResolverConfig | None = None,
) -> ResolvedRenderPlan:
    ...
```

## Minimum required planner fields
For each `PlannedSection`:
- `label: str`
- `start_bar: int`
- `bar_count: int`
- `source_parent: str` (`"A"` or `"B"`)
- `source_section_label: str | None`
- `transition_in: str | None`
- `transition_out: str | None`
- `target_energy: float | None`

## Minimum required analysis fields
Current `SongDNA` is dict-heavy, so resolver should read these defensively:
- `source_path`
- `tempo_bpm`
- `duration_seconds`
- `sample_rate`
- `structure.sections[*].label`
- `structure.sections[*].start`
- `structure.sections[*].end`
- optional `structure.tempo_reference_bpm`
- optional beat/bar artifacts if later added

If beat artifacts are missing, resolver should fall back to a **uniform 4/4 tempo grid** derived from `tempo_bpm`.

---

## Strongly recommended dataclasses

```python
from dataclasses import dataclass, field
from typing import Literal

ParentId = Literal["A", "B"]
TransitionType = Literal["blend", "swap", "drop", "lift"]
SnapUnit = Literal["beat", "bar", "phrase"]
WorkOrderType = Literal["section_base", "transition_tail", "transition_head"]
```

```python
@dataclass(slots=True)
class ResolverConfig:
    beats_per_bar: int = 4
    bars_per_phrase: int = 4
    snap_unit: SnapUnit = "beat"
    section_pad_beats: int = 0
    max_stretch_ratio_for_blend: float = 1.12
    min_stretch_ratio_for_blend: float = 0.88
    default_transition_beats: int = 8
```

```python
@dataclass(slots=True)
class SourceSectionRef:
    parent_id: ParentId
    source_path: str
    section_label: str
    raw_start_sec: float
    raw_end_sec: float
    snapped_start_sec: float
    snapped_end_sec: float
    available_duration_sec: float
    source_tempo_bpm: float
```

```python
@dataclass(slots=True)
class TargetSectionTiming:
    start_bar: int
    end_bar: int
    bar_count: int
    target_start_beat: int
    target_end_beat: int
    target_start_sec: float
    target_end_sec: float
    target_duration_sec: float
    anchor_tempo_bpm: float
```

```python
@dataclass(slots=True)
class ResolvedSection:
    index: int
    label: str
    parent_id: ParentId
    source_section_label: str
    source: SourceSectionRef
    timing: TargetSectionTiming
    stretch_ratio: float
    semitone_shift: float
    transition_in: TransitionType | None
    transition_out: TransitionType | None
    target_energy: float | None
```

```python
@dataclass(slots=True)
class AudioWorkOrder:
    order_id: str
    section_index: int
    order_type: WorkOrderType
    parent_id: ParentId
    source_path: str
    source_start_sec: float
    source_end_sec: float
    target_start_sec: float
    target_duration_sec: float
    anchor_tempo_bpm: float
    source_tempo_bpm: float
    stretch_ratio: float
    semitone_shift: float
    gain_db: float = 0.0
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    transition_type: str | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)
```

```python
@dataclass(slots=True)
class ResolvedRenderPlan:
    sample_rate: int
    total_bars: int
    total_beats: int
    total_duration_sec: float
    anchor_tempo_bpm: float
    sections: list[ResolvedSection]
    work_orders: list[AudioWorkOrder]
    warnings: list[str] = field(default_factory=list)
```

---

## Clean resolver algorithm

## 1) Validate plan timeline
Checks:
- sections sorted by `start_bar`
- `bar_count > 0`
- no overlaps
- no gaps unless explicitly allowed
- `source_parent in {"A", "B"}`
- `source_section_label` exists in selected parent

Recommend strict behavior:
```python
def validate_plan(plan: ChildArrangementPlan) -> None: ...
```
Fail early on malformed structure rather than silently guessing.

## 2) Build beat grid for each parent
Preferred future source:
- explicit beat/downbeat/bar artifacts from analysis

Current fallback:
- synthesize beat times from `tempo_bpm`
- assume constant tempo and 4/4

```python
def build_parent_grid(song: SongDNA, *, config: ResolverConfig) -> ParentGrid: ...
```

Suggested helper shape:
```python
@dataclass(slots=True)
class ParentGrid:
    parent_id: ParentId
    source_path: str
    tempo_bpm: float
    beat_times_sec: list[float]
    bar_start_times_sec: list[float]
    phrase_start_times_sec: list[float]
    duration_seconds: float
```

## 3) Resolve source section bounds
Lookup the section by label in `song.structure["sections"]`.

```python
def resolve_source_section_ref(
    song: SongDNA,
    parent_id: ParentId,
    section_label: str,
    grid: ParentGrid,
    *,
    config: ResolverConfig,
) -> SourceSectionRef:
    ...
```

## 4) Snap source boundaries
Current repo only guarantees section start/end in seconds. Resolver should snap these to nearest musical boundary before any trimming.

Recommended rule:
- default snap to nearest **beat**
- if later bar/downbeat confidence exists, prefer **bar** for section starts
- never snap beyond audio duration
- preserve non-negative duration

```python
def snap_time_to_grid(time_sec: float, grid_times_sec: list[float]) -> float: ...

def snap_section_bounds(
    raw_start_sec: float,
    raw_end_sec: float,
    grid: ParentGrid,
    *,
    config: ResolverConfig,
) -> tuple[float, float]:
    ...
```

Recommended behavior:
- start -> nearest grid point at or before raw start when possible
- end -> nearest grid point at or after raw end when possible
- if snap collapses interval, expand to at least 1 beat

That is safer than symmetric nearest-neighbor because it avoids truncating useful content.

## 5) Compute target section timing
The plan already defines timeline in bars. Resolver should treat that as canonical.

Math:
- `target_start_beat = start_bar * beats_per_bar`
- `target_end_beat = (start_bar + bar_count) * beats_per_bar`
- `target_duration_beats = bar_count * beats_per_bar`
- `seconds_per_beat = 60.0 / anchor_tempo_bpm`
- `target_duration_sec = target_duration_beats * seconds_per_beat`

For current milestone, the cleanest anchor choice is:
- per section, `anchor_tempo_bpm = source_parent tempo`

This matches the first-pass render research: one anchor per child section, one stretch ratio per section.

```python
def compute_target_section_timing(
    planned: PlannedSection,
    *,
    anchor_tempo_bpm: float,
    config: ResolverConfig,
) -> TargetSectionTiming:
    ...
```

## 6) Compute transform math
For the source audio used as the base layer of the section:
- if source parent is the anchor, `stretch_ratio = 1.0`
- if later donor overlays are added, `stretch_ratio = anchor_tempo_bpm / source_tempo_bpm`
- `semitone_shift` should be explicit, even if `0.0` for now

```python
def compute_stretch_ratio(source_tempo_bpm: float, anchor_tempo_bpm: float) -> float: ...

def compute_semitone_shift(parent_key: dict, target_key: dict | None = None) -> float: ...
```

For the current milestone, it is acceptable to keep target key implicit and return `0.0` until harmonic rendering is added.

## 7) Emit deterministic work orders
Resolver output should be a flat list of explicit operations, sorted by target time.

For current milestone, each section should emit at least one base order:
- `section_base`

If transitions are modeled as separate regions, also emit:
- `transition_tail`
- `transition_head`

```python
def build_section_work_orders(
    resolved_section: ResolvedSection,
    *,
    config: ResolverConfig,
) -> list[AudioWorkOrder]:
    ...
```

---

## Section duration math

## Canonical rule
**Child section duration comes from planner bars, not source seconds.**

That is the cleanest contract because:
- planner owns form
- resolver owns timing realization
- renderer conforms source audio to target timeline

So this is correct:
- planner says `8 bars`
- resolver computes exact target seconds from anchor BPM
- renderer trims/pads/stretch-conforms source audio to that duration

This is better than inheriting source section seconds directly, because source section lengths are analysis artifacts, not final arrangement truth.

---

## Beat snapping logic

## Current safest logic
When analysis lacks reliable bar/downbeat grids:
1. synthesize beat grid from constant tempo
2. snap section start/end to beat grid
3. compute target section duration from bars
4. trim/pad source to exact target length after any transform

## Preferred exact snap policy
```python
def snap_section_bounds(...) -> tuple[float, float]:
    # start snaps down, end snaps up
```

Reason:
- avoids shaving attacks/transients
- preserves enough material for fades
- deterministic and easy to test

## Edge-case rules
- if raw section missing start/end -> error
- if snapped interval exceeds file duration -> clamp end to duration
- if available source is shorter than target duration -> allow pad warning
- if available source is much longer than target duration -> trim to target duration later, do not alter timeline math

---

## Target timeline math

The resolver should build the child timeline only from bars.

For section `i`:
- `target_start_bar = planned.start_bar`
- `target_end_bar = planned.start_bar + planned.bar_count`
- `target_start_sec = target_start_beat * 60 / anchor_tempo_bpm`
- `target_end_sec = target_end_beat * 60 / anchor_tempo_bpm`

For whole render:
- `total_bars = max(section.end_bar)`
- `total_beats = total_bars * beats_per_bar`
- `total_duration_sec = sum(section.target_duration_sec)` only if contiguous and same anchor tempo assumptions hold

Better final rule:
```python
total_duration_sec = max(section.timing.target_end_sec for section in sections)
```

That survives future overlap/transition regions.

---

## Deterministic audio work-order model

The downstream renderer should not have to infer anything from planner fields. Every needed decision should be explicit in work orders.

Minimum base work order fields:
- source file
- source time window
- target placement time
- target duration
- stretch ratio
- pitch shift
- fade envelope values
- transition type
- gain preset

That makes the render stage a pure execution engine.

Example conceptual output for current stub plan:

```python
[
  AudioWorkOrder(
    order_id="sec0-base",
    section_index=0,
    order_type="section_base",
    parent_id="A",
    source_path="song_a.wav",
    source_start_sec=0.0,
    source_end_sec=16.2,
    target_start_sec=0.0,
    target_duration_sec=15.98,
    anchor_tempo_bpm=120.19,
    source_tempo_bpm=120.19,
    stretch_ratio=1.0,
    semitone_shift=0.0,
    transition_type="lift",
  ),
]
```

---

## Recommended function signatures

```python
def resolve_render_plan(
    plan: ChildArrangementPlan,
    parent_a: SongDNA,
    parent_b: SongDNA,
    *,
    config: ResolverConfig | None = None,
) -> ResolvedRenderPlan:
    ...
```

```python
def validate_plan(plan: ChildArrangementPlan) -> None:
    ...
```

```python
def build_parent_grid(
    song: SongDNA,
    *,
    parent_id: ParentId,
    config: ResolverConfig,
) -> ParentGrid:
    ...
```

```python
def resolve_source_section_ref(
    song: SongDNA,
    parent_id: ParentId,
    section_label: str,
    grid: ParentGrid,
    *,
    config: ResolverConfig,
) -> SourceSectionRef:
    ...
```

```python
def snap_time_to_grid(time_sec: float, grid_times_sec: list[float]) -> float:
    ...
```

```python
def snap_section_bounds(
    raw_start_sec: float,
    raw_end_sec: float,
    grid: ParentGrid,
    *,
    config: ResolverConfig,
) -> tuple[float, float]:
    ...
```

```python
def choose_anchor_tempo_bpm(
    planned: PlannedSection,
    source_ref: SourceSectionRef,
) -> float:
    ...
```

```python
def compute_target_section_timing(
    planned: PlannedSection,
    *,
    anchor_tempo_bpm: float,
    config: ResolverConfig,
) -> TargetSectionTiming:
    ...
```

```python
def compute_stretch_ratio(source_tempo_bpm: float, anchor_tempo_bpm: float) -> float:
    ...
```

```python
def compute_semitone_shift(
    source_key: dict,
    target_key: dict | None = None,
) -> float:
    ...
```

```python
def resolve_section(
    index: int,
    planned: PlannedSection,
    song: SongDNA,
    *,
    parent_id: ParentId,
    grid: ParentGrid,
    config: ResolverConfig,
) -> ResolvedSection:
    ...
```

```python
def build_section_work_orders(
    resolved_section: ResolvedSection,
    *,
    config: ResolverConfig,
) -> list[AudioWorkOrder]:
    ...
```

---

## Design decisions worth keeping
- **Planner bars are canonical.**
- **Resolver emits explicit work orders, not implicit render hints.**
- **Source boundaries are snapped before trimming/stretch decisions.**
- **One anchor tempo per section** is the cleanest first-pass contract.
- **Per-section deterministic output** is better than a renderer that infers hidden rules.

---

## Practical recommendation for current milestone
The cleanest `resolver.py` for this repo right now is a small pure module that:
1. reads `ChildArrangementPlan`
2. resolves `source_section_label` into snapped source windows
3. computes per-section target timing from bars
4. assigns per-section anchor tempo from the chosen parent
5. emits one `section_base` work order per section
6. optionally emits simple transition orders later

That gives the render milestone a stable seam:

**planner -> resolver -> transforms/mixer/exporter**

and avoids mixing musical timeline logic with DSP code.