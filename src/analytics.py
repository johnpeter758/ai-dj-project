"""
AI DJ Project Analytics Dashboard
Tracks stats, metrics, and performance for the music generation system.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class AnalyticsDashboard:
    """Main analytics dashboard for tracking AI DJ system metrics."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or "/Users/johnpeter/ai-dj-project/src")
        self.output_dir = self.data_dir / "output"
        self.reports_dir = self.data_dir / "analytics_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load latest test report if exists
        self.test_report_path = self.data_dir / "test_report.json"
        self.test_data = self._load_json(self.test_report_path) if self.test_report_path.exists() else None
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON file safely."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_json(self, data: dict, path: Path) -> None:
        """Save data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_test_metrics(self) -> dict:
        """Extract metrics from test report."""
        if not self.test_data:
            return {"status": "no_data", "message": "No test report found"}
        
        return {
            "timestamp": self.test_data.get("timestamp"),
            "total_tests": self.test_data.get("total_tests", 0),
            "passed": self.test_data.get("passed", 0),
            "failed": self.test_data.get("failed", 0),
            "pass_rate": round(self.test_data.get("passed", 0) / max(self.test_data.get("total_tests", 1), 1) * 100, 1),
            "duration_ms": round(self.test_data.get("duration_ms", 0), 2),
            "summary": self.test_data.get("summary", {})
        }
    
    def get_category_breakdown(self) -> dict:
        """Get breakdown by test category."""
        if not self.test_data:
            return {}
        
        summary = self.test_data.get("summary", {})
        breakdown = {}
        
        for category, stats in summary.items():
            breakdown[category] = {
                "total": stats.get("total", 0),
                "passed": stats.get("passed", 0),
                "failed": stats.get("failed", 0),
                "pass_rate": round(stats.get("passed", 0) / max(stats.get("total", 1), 1) * 100, 1)
            }
        
        return breakdown
    
    def get_timing_stats(self) -> dict:
        """Extract timing statistics from tests."""
        if not self.test_data:
            return {}
        
        all_results = []
        for category in ["generator_results", "audio_results", "quality_results"]:
            results = self.test_data.get(category, [])
            all_results.extend(results)
        
        if not all_results:
            return {}
        
        durations = [r.get("duration_ms", 0) for r in all_results if r.get("duration_ms")]
        
        if not durations:
            return {}
        
        durations.sort()
        
        return {
            "total_time_ms": round(sum(durations), 2),
            "average_ms": round(sum(durations) / len(durations), 3),
            "min_ms": round(min(durations), 4),
            "max_ms": round(max(durations), 2),
            "median_ms": round(durations[len(durations) // 2], 3) if durations else 0,
            "p95_ms": round(durations[int(len(durations) * 0.95)], 3) if durations else 0,
            "test_count": len(durations)
        }
    
    def get_slowest_tests(self, limit: int = 5) -> list:
        """Get the slowest tests by duration."""
        if not self.test_data:
            return []
        
        all_results = []
        for category in ["generator_results", "audio_results", "quality_results"]:
            results = self.test_data.get(category, [])
            all_results.extend(results)
        
        sorted_results = sorted(
            all_results, 
            key=lambda x: x.get("duration_ms", 0), 
            reverse=True
        )
        
        return [
            {
                "name": r.get("name"),
                "duration_ms": round(r.get("duration_ms", 0), 3),
                "details": r.get("details", "")
            }
            for r in sorted_results[:limit]
        ]
    
    def get_output_stats(self) -> dict:
        """Get statistics about generated output files."""
        if not self.output_dir.exists():
            return {"status": "no_output", "message": "Output directory not found"}
        
        output_files = list(self.output_dir.rglob("*"))
        files_by_ext = {}
        
        for f in output_files:
            if f.is_file():
                ext = f.suffix.lower() or "no_ext"
                files_by_ext[ext] = files_by_ext.get(ext, 0) + 1
        
        total_size = sum(f.stat().st_size for f in output_files if f.is_file())
        
        return {
            "total_files": len([f for f in output_files if f.is_file()]),
            "total_dirs": len([f for f in output_files if f.is_dir()]),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_extension": files_by_ext
        }
    
    def get_system_health(self) -> dict:
        """Get overall system health score."""
        metrics = self.get_test_metrics()
        output = self.get_output_stats()
        
        # Calculate health score
        health_score = 0
        factors = []
        
        # Test pass rate (40% weight)
        pass_rate = metrics.get("pass_rate", 0)
        health_score += pass_rate * 0.4
        factors.append(f"Test pass rate: {pass_rate}%")
        
        # Output generation (30% weight)
        if output.get("total_files", 0) > 0:
            health_score += 30
            factors.append(f"Output files: {output.get('total_files', 0)}")
        else:
            factors.append("No output files yet")
        
        # Test coverage (30% weight)
        total_tests = metrics.get("total_tests", 0)
        if total_tests >= 30:
            health_score += 30
            factors.append(f"Test coverage: {total_tests} tests")
        else:
            health_score += (total_tests / 30) * 30
            factors.append(f"Test coverage: {total_tests}/30")
        
        return {
            "health_score": round(health_score, 1),
            "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
            "factors": factors
        }
    
    def generate_dashboard_report(self) -> dict:
        """Generate a comprehensive dashboard report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_health": self.get_system_health(),
            "test_metrics": self.get_test_metrics(),
            "category_breakdown": self.get_category_breakdown(),
            "timing_stats": self.get_timing_stats(),
            "slowest_tests": self.get_slowest_tests(),
            "output_stats": self.get_output_stats()
        }
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """Save dashboard report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_{timestamp}.json"
        
        report = self.generate_dashboard_report()
        report_path = self.reports_dir / filename
        self._save_json(report, report_path)
        
        return str(report_path)
    
    def print_dashboard(self) -> None:
        """Print dashboard to console."""
        report = self.generate_dashboard_report()
        
        print("\n" + "=" * 60)
        print("🎧 AI DJ PROJECT ANALYTICS DASHBOARD")
        print("=" * 60)
        
        # System Health
        health = report["system_health"]
        status_emoji = "✅" if health["status"] == "healthy" else "⚠️" if health["status"] == "warning" else "❌"
        print(f"\n{status_emoji} System Health: {health['health_score']}/100 ({health['status'].upper()})")
        for factor in health["factors"]:
            print(f"   • {factor}")
        
        # Test Metrics
        tests = report["test_metrics"]
        print(f"\n📊 Test Results:")
        print(f"   Total: {tests.get('total_tests', 0)} | Passed: {tests.get('passed', 0)} | Failed: {tests.get('failed', 0)}")
        print(f"   Pass Rate: {tests.get('pass_rate', 0)}% | Duration: {tests.get('duration_ms', 0)}ms")
        
        # Category Breakdown
        print(f"\n📁 Category Breakdown:")
        for cat, data in report["category_breakdown"].items():
            print(f"   {cat.capitalize()}: {data['passed']}/{data['total']} ({data['pass_rate']}%)")
        
        # Timing Stats
        timing = report["timing_stats"]
        if timing:
            print(f"\n⏱️  Timing Statistics:")
            print(f"   Total: {timing.get('total_time_ms', 0)}ms | Avg: {timing.get('average_ms', 0)}ms")
            print(f"   Min: {timing.get('min_ms', 0)}ms | Max: {timing.get('max_ms', 0)}ms")
            print(f"   Median: {timing.get('median_ms', 0)}ms | P95: {timing.get('p95_ms', 0)}ms")
        
        # Slowest Tests
        slowest = report.get("slowest_tests", [])
        if slowest:
            print(f"\n🐢 Slowest Tests:")
            for t in slowest[:3]:
                print(f"   • {t['name']}: {t['duration_ms']}ms")
        
        # Output Stats
        output = report["output_stats"]
        if output.get("total_files", 0) > 0:
            print(f"\n📂 Output Files:")
            print(f"   Total: {output.get('total_files', 0)} files | Size: {output.get('total_size_mb', 0)} MB")
            if output.get("by_extension"):
                exts = ", ".join([f"{k}:{v}" for k, v in output.get("by_extension", {}).items()])
                print(f"   Types: {exts}")
        
        print("\n" + "=" * 60)


def run_dashboard():
    """Main entry point for running the dashboard."""
    dashboard = AnalyticsDashboard()
    dashboard.print_dashboard()
    
    # Optionally save report
    report_path = dashboard.save_report()
    print(f"\n📄 Report saved to: {report_path}")
    
    return dashboard.generate_dashboard_report()


if __name__ == "__main__":
    run_dashboard()
