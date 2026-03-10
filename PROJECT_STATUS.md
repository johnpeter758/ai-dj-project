# Project Status

## Plain-English summary
This repo has useful ideas but is currently too broad and too flat to function like a professional music system.

## Current state
- GitHub repo is connected and available for active cleanup.
- The project contains many modules, but not enough architectural discipline.
- The strongest current direction is VocalFusion: song analysis -> song DNA -> arrangement planning -> rendering -> evaluation.

## Most important decision
The project will be narrowed around a professional core instead of trying to do every possible AI music feature at once.

## Active priorities
1. repo triage
2. canonical architecture
3. analysis pipeline
4. planner design
5. render/evaluation loop

## Risks
- too many speculative modules
- unclear ownership of many files
- likely dead code
- too much flat structure in `src/`

## What success looks like
A smaller, cleaner repo where the important path is obvious and each subsystem has a clear role.
