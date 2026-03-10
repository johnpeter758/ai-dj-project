# Repo Triage

## Purpose
Turn `ai-dj-project` into a focused, professional VocalFusion codebase.

## Diagnosis
The repo currently has too much breadth and not enough canonical architecture.
Main symptoms:
- 120 flat Python modules in `src/`
- multiple overlapping product ideas
- speculative and placeholder modules mixed into the critical path
- weak separation between analysis, planning, rendering, and evaluation

## Target product core
The center of gravity should be:
1. analysis
2. song DNA schema
3. planner
4. render engine
5. evaluation

Anything outside that core should justify its existence.

## Classification buckets

### Keep / modernize
These are directionally aligned with the real product:
- `src/vocalfusion.py`
- `src/arrangement_generator.py`
- `src/fusion_v5.py`
- `src/quality_evaluator.py`
- selected analysis artifacts in `data/analyses/`
- architecture / planning markdown that supports the core system

### Archive / demote
These may contain ideas but should not stay in the active center:
- speculative product/platform modules
- UI/dashboard files not tied to core evaluation needs
- social/sharing/cloud sync features
- placeholder controller shells
- workflow wrappers that do not reflect the real product architecture

### Remove or replace
These should be aggressively questioned:
- duplicate utilities with unclear ownership
- shallow placeholder modules with hardcoded outputs
- modules that simulate capability without enabling a real music pipeline

## Immediate cleanup strategy
1. create explicit architecture docs inside the repo
2. create a plain-English repo status note
3. identify the canonical active path for analysis, planning, rendering, evaluation
4. avoid blind mass deletion until key files are mapped
5. archive low-value breadth after the canonical path is in place

## Standard for future files
A file should stay active only if it directly supports one of:
- extracting musically meaningful song DNA
- planning an intentional child arrangement
- rendering a coherent fusion result
- evaluating output quality in a reproducible way
