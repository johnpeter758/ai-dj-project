#!/usr/bin/env python3
"""
Reporting System for AI DJ Project

Provides comprehensive reporting capabilities:
- Song generation reports
- Fusion/transition reports  
- Performance and analytics reports
- Usage and activity reports
- Export to multiple formats (JSON, CSV, HTML, Markdown)
"""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict

# Import existing modules
from logger import get_logger
from database import DB_PATH

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "/Users/johnpeter/ai-dj-project/src/output/reports"
    include_charts: bool = False
    date_format: str = "%Y-%m-%d %H:%M:%S"
    timezone: str = "UTC"


class ReportGenerator:
    """Main reporting engine for AI DJ system."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_db_data(self, query: str, params: tuple = ()) -> list:
        """Load data from SQLite database."""
        import sqlite3
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            return []
    
    def _get_table_count(self, table: str) -> int:
        """Get row count for a table."""
        data = self._load_db_data(f"SELECT COUNT(*) as count FROM {table}")
        return data[0]["count"] if data else 0
    
    def generate_summary_report(self, days: int = 7) -> dict:
        """Generate system summary report for specified days."""
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Song stats
        songs = self._load_db_data(
            "SELECT * FROM songs WHERE created_at >= ? ORDER BY created_at DESC",
            (since_date,)
        )
        
        # Fusion stats
        fusions = self._load_db_data(
            "SELECT * FROM fusions WHERE created_at >= ? ORDER BY created_at DESC",
            (since_date,)
        )
        
        # Analysis stats
        analyses = self._load_db_data(
            "SELECT * FROM analyses WHERE created_at >= ? ORDER BY created_at DESC",
            (since_date,)
        )
        
        # Calculate genre distribution
        genre_counts = {}
        for song in songs:
            genre = song.get("genre", "unknown")
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Calculate average BPM
        bpms = [s.get("bpm", 0) for s in songs if s.get("bpm")]
        avg_bpm = sum(bpms) / len(bpms) if bpms else 0
        
        report = {
            "report_type": "system_summary",
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "since_date": since_date,
            "statistics": {
                "total_songs": len(songs),
                "total_fusions": len(fusions),
                "total_analyses": len(analyses),
                "genre_distribution": genre_counts,
                "average_bpm": round(avg_bpm, 1),
            },
            "recent_songs": songs[:10] if songs else [],
            "recent_fusions": fusions[:10] if fusions else [],
        }
        
        return report
    
    def generate_song_report(self, limit: int = 100, genre: Optional[str] = None) -> dict:
        """Generate detailed song generation report."""
        if genre:
            songs = self._load_db_data(
                "SELECT * FROM songs WHERE genre = ? ORDER BY created_at DESC LIMIT ?",
                (genre, limit)
            )
        else:
            songs = self._load_db_data(
                "SELECT * FROM songs ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
        
        # Group by genre
        by_genre = {}
        by_mood = {}
        by_key = {}
        
        for song in songs:
            genre = song.get("genre", "unknown")
            mood = song.get("mood", "unknown")
            key = song.get("key", "unknown")
            
            by_genre[genre] = by_genre.get(genre, 0) + 1
            by_mood[mood] = by_mood.get(mood, 0) + 1
            by_key[key] = by_key.get(key, 0) + 1
        
        # Duration stats
        durations = [s.get("duration", 0) for s in songs if s.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Energy stats
        energies = [s.get("energy", 0) for s in songs if s.get("energy")]
        avg_energy = sum(energies) / len(energies) if energies else 0
        
        report = {
            "report_type": "song_report",
            "generated_at": datetime.now().isoformat(),
            "total_songs": len(songs),
            "filters": {"genre": genre, "limit": limit},
            "statistics": {
                "by_genre": by_genre,
                "by_mood": by_mood,
                "by_key": by_key,
                "average_duration_sec": round(avg_duration, 1),
                "average_energy": round(avg_energy, 2),
            },
            "songs": songs,
        }
        
        return report
    
    def generate_fusion_report(self, limit: int = 100) -> dict:
        """Generate detailed fusion/transition report."""
        fusions = self._load_db_data(
            "SELECT * FROM fusions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        # Group by genre
        by_genre = {}
        by_bpm = []
        
        for fusion in fusions:
            genre = fusion.get("genre", "unknown")
            by_genre[genre] = by_genre.get(genre, 0) + 1
            
            bpm = fusion.get("bpm", 0)
            if bpm:
                by_bpm.append(bpm)
        
        avg_bpm = sum(by_bpm) / len(by_bpm) if by_bpm else 0
        
        # Get song details for each fusion
        fusion_details = []
        for fusion in fusions:
            song1 = self._load_db_data(
                "SELECT * FROM songs WHERE id = ?",
                (fusion.get("song1_id"),)
            )
            song2 = self._load_db_data(
                "SELECT * FROM songs WHERE id = ?",
                (fusion.get("song2_id"),)
            )
            fusion_details.append({
                **fusion,
                "song1": song1[0] if song1 else None,
                "song2": song2[0] if song2 else None,
            })
        
        report = {
            "report_type": "fusion_report",
            "generated_at": datetime.now().isoformat(),
            "total_fusions": len(fusions),
            "statistics": {
                "by_genre": by_genre,
                "average_bpm": round(avg_bpm, 1),
            },
            "fusions": fusion_details,
        }
        
        return report
    
    def generate_performance_report(self, hours: int = 24) -> dict:
        """Generate system performance report."""
        since_time = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Load analyses for performance metrics
        analyses = self._load_db_data(
            "SELECT * FROM analyses WHERE created_at >= ? ORDER BY created_at DESC",
            (since_time,)
        )
        
        # Extract performance metrics
        energy_values = [a.get("energy", 0) for a in analyses if a.get("energy")]
        danceability_values = [a.get("danceability", 0) for a in analyses if a.get("danceability")]
        bpm_values = [a.get("bpm", 0) for a in analyses if a.get("bpm")]
        
        def stats(values: list) -> dict:
            if not values:
                return {"count": 0, "min": 0, "max": 0, "avg": 0}
            return {
                "count": len(values),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "avg": round(sum(values) / len(values), 2),
            }
        
        report = {
            "report_type": "performance_report",
            "generated_at": datetime.now().isoformat(),
            "period_hours": hours,
            "total_analyses": len(analyses),
            "metrics": {
                "energy": stats(energy_values),
                "danceability": stats(danceability_values),
                "bpm": stats(bpm_values),
            },
            "recent_analyses": analyses[:20] if analyses else [],
        }
        
        return report
    
    def generate_activity_report(self, days: int = 30) -> dict:
        """Generate activity/timeline report."""
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Daily counts
        songs = self._load_db_data(
            "SELECT created_at FROM songs WHERE created_at >= ?",
            (since_date,)
        )
        fusions = self._load_db_data(
            "SELECT created_at FROM fusions WHERE created_at >= ?",
            (since_date,)
        )
        
        # Group by day
        daily_activity = {}
        for item in songs + fusions:
            created_at = item.get("created_at", "")
            if created_at:
                day = created_at[:10]  # Extract YYYY-MM-DD
                if day not in daily_activity:
                    daily_activity[day] = {"songs": 0, "fusions": 0}
                # Check if it's a song (has genre field in basic query)
                if "genre" in item:
                    daily_activity[day]["songs"] += 1
                else:
                    daily_activity[day]["fusions"] += 1
        
        # Sort by date
        daily_activity = dict(sorted(daily_activity.items()))
        
        report = {
            "report_type": "activity_report",
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "since_date": since_date,
            "total_songs": len(songs),
            "total_fusions": len(fusions),
            "daily_activity": daily_activity,
        }
        
        return report
    
    def export_to_json(self, report: dict, filename: str) -> str:
        """Export report to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Exported JSON report: {output_path}")
        return str(output_path)
    
    def export_to_csv(self, data: list, filename: str) -> str:
        """Export report data to CSV file."""
        if not data:
            return ""
        
        output_path = self.output_dir / filename
        keys = data[0].keys()
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Exported CSV report: {output_path}")
        return str(output_path)
    
    def export_to_markdown(self, report: dict, filename: str) -> str:
        """Export report to Markdown file."""
        output_path = self.output_dir / filename
        
        md_lines = [
            f"# {report.get('report_type', 'Report').replace('_', ' ').title()} Report",
            "",
            f"**Generated:** {report.get('generated_at', 'N/A')}",
            "",
        ]
        
        # Add statistics section
        if "statistics" in report:
            md_lines.append("## Statistics")
            md_lines.append("")
            stats = report["statistics"]
            
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, dict):
                        md_lines.append(f"### {key.replace('_', ' ').title()}")
                        md_lines.append("")
                        for k, v in value.items():
                            md_lines.append(f"- **{k}:** {v}")
                        md_lines.append("")
                    else:
                        md_lines.append(f"- **{key}:** {value}")
            md_lines.append("")
        
        # Add recent items section
        if "songs" in report and report["songs"]:
            md_lines.append("## Recent Songs")
            md_lines.append("")
            for song in report["songs"][:10]:
                md_lines.append(f"- **{song.get('name', 'Untitled')}** - {song.get('genre', 'N/A')} @ {song.get('bpm', 'N/A')} BPM")
            md_lines.append("")
        
        if "fusions" in report and report["fusions"]:
            md_lines.append("## Recent Fusions")
            md_lines.append("")
            for fusion in report["fusions"][:10]:
                md_lines.append(f"- **{fusion.get('name', 'Untitled')}** - {fusion.get('genre', 'N/A')}")
            md_lines.append("")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(md_lines))
        
        logger.info(f"Exported Markdown report: {output_path}")
        return str(output_path)
    
    def export_to_html(self, report: dict, filename: str) -> str:
        """Export report to HTML file."""
        output_path = self.output_dir / filename
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.get('report_type', 'Report').replace('_', ' ').title()} Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #007bff; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .meta {{ color: #666; font-size: 0.9em; }}
        .stat-card {{ display: inline-block; padding: 15px 25px; margin: 10px; 
                      background: #f8f9fa; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>{report.get('report_type', 'Report').replace('_', ' ').title()} Report</h1>
    <p class="meta">Generated: {report.get('generated_at', 'N/A')}</p>
"""
        
        # Add statistics as stat cards
        if "statistics" in report:
            html += "<h2>Statistics</h2>\n"
            stats = report["statistics"]
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if not isinstance(value, dict):
                        html += f'<div class="stat-card"><strong>{key.replace("_", " ").title()}</strong><br>{value}</div>\n'
        
        # Add songs table if present
        if "songs" in report and report["songs"]:
            html += "<h2>Recent Songs</h2>\n<table>\n<tr><th>Name</th><th>Genre</th><th>BPM</th><th>Key</th><th>Duration</th></tr>\n"
            for song in report["songs"][:20]:
                html += f"<tr><td>{song.get('name', 'N/A')}</td><td>{song.get('genre', 'N/A')}</td>"
                html += f"<td>{song.get('bpm', 'N/A')}</td><td>{song.get('key', 'N/A')}</td>"
                html += f"<td>{song.get('duration', 'N/A')}s</td></tr>\n"
            html += "</table>\n"
        
        # Add fusions table if present
        if "fusions" in report and report["fusions"]:
            html += "<h2>Recent Fusions</h2>\n<table>\n<tr><th>Name</th><th>Genre</th><th>BPM</th><th>Duration</th></tr>\n"
            for fusion in report["fusions"][:20]:
                html += f"<tr><td>{fusion.get('name', 'N/A')}</td><td>{fusion.get('genre', 'N/A')}</td>"
                html += f"<td>{fusion.get('bpm', 'N/A')}</td><td>{fusion.get('duration', 'N/A')}s</td></tr>\n"
            html += "</table>\n"
        
        html += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Exported HTML report: {output_path}")
        return str(output_path)
    
    def generate_all_reports(self, days: int = 7) -> dict:
        """Generate and export all standard reports."""
        reports = {}
        
        # Summary report
        summary = self.generate_summary_report(days)
        reports["summary"] = {
            "json": self.export_to_json(summary, "summary_report.json"),
            "markdown": self.export_to_markdown(summary, "summary_report.md"),
            "html": self.export_to_html(summary, "summary_report.html"),
        }
        
        # Song report
        songs = self.generate_song_report(limit=50)
        reports["songs"] = {
            "json": self.export_to_json(songs, "song_report.json"),
            "csv": self.export_to_csv(songs["songs"], "songs.csv"),
            "html": self.export_to_html(songs, "song_report.html"),
        }
        
        # Fusion report
        fusions = self.generate_fusion_report(limit=50)
        reports["fusions"] = {
            "json": self.export_to_json(fusions, "fusion_report.json"),
            "csv": self.export_to_csv(fusions["fusions"], "fusions.csv"),
            "html": self.export_to_html(fusions, "fusion_report.html"),
        }
        
        # Activity report
        activity = self.generate_activity_report(days * 2)
        reports["activity"] = {
            "json": self.export_to_json(activity, "activity_report.json"),
            "html": self.export_to_html(activity, "activity_report.html"),
        }
        
        logger.info(f"Generated all reports: {len(reports)} report types")
        return reports


# CLI interface
def main():
    """Command-line interface for report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI DJ Reporting System")
    parser.add_argument("--type", choices=["summary", "songs", "fusions", "performance", "activity", "all"],
                        default="summary", help="Report type to generate")
    parser.add_argument("--days", type=int, default=7, help="Number of days for report")
    parser.add_argument("--hours", type=int, default=24, help="Number of hours for performance report")
    parser.add_argument("--format", choices=["json", "csv", "markdown", "html", "all"],
                        default="all", help="Export format")
    parser.add_argument("--output", type=str, help="Output filename (without extension)")
    
    args = parser.parse_args()
    
    generator = ReportGenerator()
    report = None
    
    # Generate report based on type
    if args.type == "summary":
        report = generator.generate_summary_report(args.days)
    elif args.type == "songs":
        report = generator.generate_song_report()
    elif args.type == "fusions":
        report = generator.generate_fusion_report()
    elif args.type == "performance":
        report = generator.generate_performance_report(args.hours)
    elif args.type == "activity":
        report = generator.generate_activity_report(args.days)
    elif args.type == "all":
        reports = generator.generate_all_reports(args.days)
        print(json.dumps(reports, indent=2))
        return
    
    if report is None:
        print("Error: No report generated")
        return
    
    # Export in requested formats
    output_name = args.output or f"{args.type}_report"
    
    if args.format == "json" or args.format == "all":
        print(generator.export_to_json(report, f"{output_name}.json"))
    if args.format == "csv" and "songs" in report:
        print(generator.export_to_csv(report["songs"], f"{output_name}.csv"))
    elif args.format == "markdown" or args.format == "all":
        print(generator.export_to_markdown(report, f"{output_name}.md"))
    if args.format == "html" or args.format == "all":
        print(generator.export_to_html(report, f"{output_name}.html"))


if __name__ == "__main__":
    main()
