#!/usr/bin/env python3
"""
AI DJ Project - System Monitoring Module

Monitors system resources, performance metrics, and health checks
for the AI DJ project.
"""

import os
import psutil
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class SystemMonitor:
    """Monitor system resources and health"""
    
    def __init__(self, project_root: str = "/Users/johnpeter/ai-dj-project"):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / "src" / "output"
        self.cache_dir = self.project_root / "src" / "cache"
        
        # Thresholds
        self.cpu_threshold = 85.0  # %
        self.memory_threshold = 85.0  # %
        self.disk_threshold = 85.0  # %
        
        # Cache for rate limiting alerts
        self._alert_cooldown = 300  # seconds
        self._last_alerts = {}
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """Get CPU usage metrics"""
        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "per_core": psutil.cpu_percent(interval=0.1, percpu=True),
            "count": psutil.cpu_count(),
            "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage metrics"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
        }
    
    def get_disk_usage(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get disk usage metrics"""
        if path is None:
            path = str(self.project_root)
        
        disk = psutil_disk_usage(path)
        return disk
    
    def get_process_list(self) -> List[Dict[str, Any]]:
        """Get running processes related to AI DJ"""
        processes = []
        dj_processes = ["python", "python3", "ffmpeg", "librosa"]
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                if any(p in proc.info['name'].lower() for p in dj_processes):
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": round(proc.info['memory_percent'], 2) if proc.info['memory_percent'] else 0,
                        "status": proc.info['status'],
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network I/O statistics"""
        net = psutil.net_io_counters()
        return {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
            "mb_sent": round(net.bytes_sent / (1024**2), 2),
            "mb_recv": round(net.bytes_recv / (1024**2), 2),
        }
    
    def check_project_dirs(self) -> Dict[str, Any]:
        """Check project directory health"""
        dirs_to_check = [
            self.output_dir,
            self.cache_dir,
            self.project_root / "src",
        ]
        
        results = {}
        for d in dirs_to_check:
            name = d.name
            exists = d.exists()
            size = 0
            file_count = 0
            
            if exists:
                try:
                    for f in d.rglob("*"):
                        if f.is_file():
                            size += f.stat().st_size
                            file_count += 1
                except PermissionError:
                    pass
            
            results[name] = {
                "exists": exists,
                "size_bytes": size,
                "size_mb": round(size / (1024**2), 2),
                "file_count": file_count,
            }
        
        return results
    
    def get_system_load(self) -> Dict[str, Any]:
        """Get system load averages"""
        load = psutil.getloadavg()
        return {
            "1min": load[0],
            "5min": load[1],
            "15min": load[2],
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        disk = self.get_disk_usage()
        load = self.get_system_load()
        dirs = self.check_project_dirs()
        
        # Determine health status
        issues = []
        warnings = []
        
        if cpu['percent'] > self.cpu_threshold:
            issues.append(f"High CPU: {cpu['percent']:.1f}%")
        
        if memory['percent'] > self.memory_threshold:
            issues.append(f"High Memory: {memory['percent']:.1f}%")
        
        if disk['percent'] > self.disk_threshold:
            issues.append(f"High Disk: {disk['percent']:.1f}%")
        
        if load['1min'] > psutil.cpu_count():
            warnings.append(f"High Load: {load['1min']:.2f}")
        
        for name, info in dirs.items():
            if not info['exists']:
                issues.append(f"Missing directory: {name}")
        
        status = "healthy"
        if issues:
            status = "critical"
        elif warnings:
            status = "warning"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
            "load": load,
            "directories": dirs,
            "issues": issues,
            "warnings": warnings,
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get full system status report"""
        health = self.check_health()
        processes = self.get_process_list()
        network = self.get_network_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "processes": processes,
            "network": network,
        }
    
    def should_alert(self, alert_type: str) -> bool:
        """Check if we should send an alert (rate limited)"""
        now = time.time()
        last = self._last_alerts.get(alert_type, 0)
        
        if now - last > self._alert_cooldown:
            self._last_alerts[alert_type] = now
            return True
        return False
    
    def export_status(self, filepath: Optional[str] = None) -> str:
        """Export status to JSON file"""
        if filepath is None:
            filepath = str(self.output_dir / "monitoring_status.json")
        
        status = self.get_full_status()
        
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2)
        
        return filepath


# Helper to avoid name conflict with method
def psutil_disk_usage(path):
    """Wrapper for psutil.disk_usage"""
    usage = psutil.disk_usage(path)
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "percent": usage.percent,
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


class PerformanceMonitor:
    """Monitor AI DJ performance metrics"""
    
    def __init__(self):
        self.metrics = []
        self.max_metrics = 1000  # Keep last 1000
    
    def record_metric(self, metric_type: str, value: float, metadata: Optional[Dict] = None):
        """Record a performance metric"""
        metric = {
            "type": metric_type,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.metrics.append(metric)
        
        # Trim if needed
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_metrics(self, metric_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recorded metrics"""
        if metric_type:
            filtered = [m for m in self.metrics if m['type'] == metric_type]
        else:
            filtered = self.metrics
        
        return filtered[-limit:]
    
    def get_average(self, metric_type: str, last_n: int = 10) -> Optional[float]:
        """Get average value for a metric type"""
        values = [m['value'] for m in self.metrics if m['type'] == metric_type]
        if not values:
            return None
        return sum(values[-last_n:]) / min(len(values), last_n)


def run_monitoring():
    """Run monitoring and print results"""
    monitor = SystemMonitor()
    
    print("=" * 50)
    print("🎛️ AI DJ System Monitor")
    print("=" * 50)
    
    health = monitor.check_health()
    
    print(f"\n📊 Status: {health['status'].upper()}")
    print(f"   Time: {health['timestamp']}")
    
    print(f"\n💻 CPU: {health['cpu']['percent']:.1f}%")
    print(f"   Cores: {health['cpu']['count']}")
    
    print(f"\n🧠 Memory: {health['memory']['percent']:.1f}%")
    print(f"   Used: {health['memory']['used_gb']:.1f}GB / {health['memory']['total_gb']:.1f}GB")
    
    print(f"\n💾 Disk: {health['disk']['percent']:.1f}%")
    print(f"   Used: {health['disk']['used_gb']:.1f}GB / {health['disk']['total_gb']:.1f}GB")
    
    print(f"\n⚖️ Load: {health['load']['1min']:.2f} / {health['load']['5min']:.2f} / {health['load']['15min']:.2f}")
    
    if health['issues']:
        print(f"\n❌ Issues:")
        for issue in health['issues']:
            print(f"   - {issue}")
    
    if health['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in health['warnings']:
            print(f"   - {warning}")
    
    print(f"\n📁 Directories:")
    for name, info in health['directories'].items():
        status = "✓" if info['exists'] else "✗"
        print(f"   {status} {name}: {info['size_mb']:.1f}MB ({info['file_count']} files)")
    
    # Export to file
    filepath = monitor.export_status()
    print(f"\n📄 Status exported to: {filepath}")


if __name__ == "__main__":
    run_monitoring()
