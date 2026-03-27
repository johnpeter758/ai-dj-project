#!/usr/bin/env python3
"""Generate a lightweight, repeatable music-tech research brief for VocalFusion.

Usage:
  python scripts/generate_music_research_brief.py

Outputs:
  docs/research/MUSIC_SOTA_TRACKER.md
"""

from __future__ import annotations

import datetime as dt
import json
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "research" / "MUSIC_SOTA_TRACKER.md"

USER_AGENT = "openclaw-vocalfusion-research/1.0"


def _http_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _slug(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def fetch_github_repos() -> list[dict[str, Any]]:
    repos = [
        {
            "name": "facebookresearch/demucs",
            "why": "Strong practical baseline for stem-quality separation and mashup-safe layer isolation.",
            "integration": "Use separated stems to build integrated support layers instead of section-level hard swaps.",
        },
        {
            "name": "openvpi/DiffSinger",
            "why": "Mature open-source singing synthesis pipeline for melody/vocal conditioning ideas.",
            "integration": "Borrow phrase-conditioned vocal continuity priors for payoff transitions.",
        },
        {
            "name": "facebookresearch/audiocraft",
            "why": "Reference implementation for modern controllable audio/music generation systems.",
            "integration": "Reuse conditioning interfaces and token-level control ideas for section-role constraints.",
        },
        {
            "name": "spotify/basic-pitch",
            "why": "Fast melody/pitch extraction suitable for phrase compatibility checks.",
            "integration": "Penalize donor candidates with conflicting melodic contour during dense vocal windows.",
        },
        {
            "name": "CPJKU/madmom",
            "why": "Robust beat/downbeat stack used in production-level MIR pipelines.",
            "integration": "Tighten section-entry/downbeat alignment before support-overlay admission.",
        },
    ]

    rows: list[dict[str, Any]] = []
    for repo in repos:
        url = f"https://api.github.com/repos/{repo['name']}"
        data = _http_json(url)
        rows.append(
            {
                "repo": repo["name"],
                "stars": int(data.get("stargazers_count") or 0),
                "updated": str(data.get("pushed_at") or ""),
                "url": str(data.get("html_url") or f"https://github.com/{repo['name']}"),
                "description": str(data.get("description") or "").strip(),
                "why": repo["why"],
                "integration": repo["integration"],
            }
        )
    rows.sort(key=lambda item: item["stars"], reverse=True)
    return rows


def fetch_openalex_papers() -> list[dict[str, Any]]:
    topics = [
        {
            "label": "music source separation",
            "query": "music source separation",
            "must": ["music", "separation"],
        },
        {
            "label": "music structure segmentation",
            "query": "music structure segmentation",
            "must": ["music", "segment"],
        },
        {
            "label": "beat/downbeat tracking",
            "query": "music beat tracking transformer",
            "must": ["beat"],
        },
        {
            "label": "audio mashup generation",
            "query": "music mashup generation",
            "must": ["mashup"],
        },
        {
            "label": "vocal mixture disentanglement",
            "query": "singing voice separation music",
            "must": ["separation"],
        },
    ]

    seen: set[str] = set()
    papers: list[dict[str, Any]] = []

    for topic in topics:
        q = urllib.parse.quote(topic["query"])
        url = (
            "https://api.openalex.org/works"
            f"?filter=title.search:{q},from_publication_date:2022-01-01"
            "&sort=publication_date:desc"
            "&per-page=10"
        )
        payload = _http_json(url)
        for item in list(payload.get("results") or []):
            title = str(item.get("display_name") or "").strip()
            if not title:
                continue
            title_l = title.lower()
            if not all(token in title_l for token in topic["must"]):
                continue
            key = _slug(title)
            if key in seen:
                continue
            seen.add(key)

            source_name = ""
            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}
            source_name = str(source.get("display_name") or "").strip()

            papers.append(
                {
                    "topic": topic["label"],
                    "title": title,
                    "year": int(item.get("publication_year") or 0),
                    "cites": int(item.get("cited_by_count") or 0),
                    "openalex": str(item.get("id") or ""),
                    "doi": str(item.get("doi") or ""),
                    "source": source_name,
                }
            )

    papers.sort(key=lambda p: (p["year"], p["cites"]), reverse=True)
    return papers[:18]


def build_markdown(repos: list[dict[str, Any]], papers: list[dict[str, Any]]) -> str:
    now = dt.datetime.now().astimezone()
    lines: list[str] = []
    lines.append("# VocalFusion Music R&D Tracker")
    lines.append("")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append("This report is generated automatically so implementation work stays anchored to current music-tech practice.")
    lines.append("")

    lines.append("## 1) High-signal open-source repos (engineering benchmarks)")
    lines.append("")
    for repo in repos:
        lines.append(f"- **{repo['repo']}** ({repo['stars']}★, updated {repo['updated'][:10]})")
        if repo["description"]:
            lines.append(f"  - What it is: {repo['description']}")
        lines.append(f"  - Why it matters: {repo['why']}")
        lines.append(f"  - VocalFusion integration: {repo['integration']}")
    lines.append("")

    lines.append("## 2) Recent paper scan (OpenAlex title-filtered)")
    lines.append("")
    if not papers:
        lines.append("- No filtered papers returned by API in this run.")
    else:
        by_topic: dict[str, list[dict[str, Any]]] = {}
        for paper in papers:
            by_topic.setdefault(paper["topic"], []).append(paper)
        for topic, rows in by_topic.items():
            lines.append(f"### {topic.title()}")
            for row in rows[:5]:
                ref = row["doi"] or row["openalex"]
                source_suffix = f" ({row['source']})" if row["source"] else ""
                lines.append(f"- {row['year']} — {row['title']}{source_suffix}; cites={row['cites']}; ref={ref}")
            lines.append("")

    lines.append("## 3) Direct coding implications for current bottleneck")
    lines.append("")
    lines.append(textwrap.dedent(
        """
        Current bottleneck: medley-like section alternation and adaptive gate reject despite occasional high song-likeness.

        Highest-value research-aligned actions:
        1. **Stem-aware integrated overlays by default for major sections**
           - Move beyond ownership swaps to simultaneous backbone+donor support layers.
        2. **Phrase/downbeat hard alignment before adaptive admission**
           - Reject adaptive swaps/supports that violate downbeat confidence windows.
        3. **Owner-switch minimization objective in shortlist ranking**
           - Penalize variants with high owner_switch_ratio unless integration ratio also rises.
        4. **Vocal-collision guard at payoff seams**
           - Use melody/voicing confidence to suppress donor vocal content when backbone lead is active.
        """
    ).strip())
    lines.append("")

    lines.append("## 4) Next implementation checkpoint")
    lines.append("")
    lines.append("- Implement adaptive transition-gate rescue with beat-locked support overlays and rerun pair2 benchmark.")
    lines.append("- Keep this file regenerated each major cycle (`python scripts/generate_music_research_brief.py`).")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    repos = fetch_github_repos()
    papers = fetch_openalex_papers()
    content = build_markdown(repos, papers)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
