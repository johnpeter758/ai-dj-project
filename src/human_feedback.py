from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = "0.1.0"


def _utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class HumanFeedbackStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "feedback_id": event.get("feedback_id") or str(uuid4()),
            "created_at": event.get("created_at") or _utc_now_iso(),
            **event,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def list_events(
        self,
        *,
        feedback_type: str | None = None,
        run_dir: str | None = None,
        reviewer: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.events_path.exists():
            return rows
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if feedback_type and payload.get("type") != feedback_type:
                    continue
                if reviewer and payload.get("reviewer") != reviewer:
                    continue
                if run_dir:
                    candidates = {
                        payload.get("run_dir"),
                        payload.get("left_run_dir"),
                        payload.get("right_run_dir"),
                    }
                    if run_dir not in candidates:
                        continue
                rows.append(payload)
        rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return rows[: max(1, int(limit))]

    def summarize(self) -> dict[str, Any]:
        events = self.list_events(limit=5000)
        type_counts: dict[str, int] = {}
        decision_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        for event in events:
            event_type = str(event.get("type") or "unknown")
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            decision = str(event.get("overall_label") or event.get("winner") or "unknown")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            for tag in event.get("tags") or []:
                key = str(tag).strip()
                if not key:
                    continue
                tag_counts[key] = tag_counts.get(key, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        return {
            "total": len(events),
            "type_counts": type_counts,
            "decision_counts": decision_counts,
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
        }
