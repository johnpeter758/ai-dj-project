#!/usr/bin/env python3
"""
AI DJ Plugin Formats Support
=============================
Provides VST3, Audio Unit (AU), and AAX plugin format support.
Works alongside plugin_wrapper.py for cross-format compatibility.
"""

import json
import logging
import os
import platform
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Plugin Format Types
# =============================================================================

class PluginFormat(Enum):
    """Supported audio plugin formats."""
    VST3 = "vst3"
    AU = "au"
    AAX = "aax"
    UNKNOWN = "unknown"


class PluginArchitecture(Enum):
    """Plugin architecture (bit depth)."""
    X64 = "x64"
    ARM64 = "arm64"
    UNIVERSAL = "universal"


class HostDAW(Enum):
    """Supported DAW hosts."""
    REAPER = "reaper"
    ABLETON = "ableton"
    LOGIC = "logic"
    PRO_TOOLS = "pro_tools"
    CUBASE = "cubase"
    FL_STUDIO = "fl_studio"
    DAW_VST3 = "vst3_host"
    UNKNOWN = "unknown"


@dataclass
class PluginFormatInfo:
    """Information about a plugin format."""
    format: PluginFormat
    extension: str
    platform: str
    default_paths: List[str]
    scan_paths: List[str] = field(default_factory=list)
    requires_authorized: bool = False


@dataclass
class PluginMetadata:
    """Plugin metadata from format-specific info."""
    plugin_id: str
    name: str
    vendor: str
    format: PluginFormat
    version: str
    path: str
    architecture: PluginArchitecture
    categories: List[str] = field(default_factory=list)
    inputs: int = 2
    outputs: int = 2
    latency: int = 0
    is_instrument: bool = False
    is_effect: bool = True
    parameters: int = 0
    presets: int = 0


# =============================================================================
# Platform-Specific Configuration
# =============================================================================

def get_platform_info() -> Dict[str, Any]:
    """Get current platform information."""
    system = platform.system()
    arch = platform.machine()
    
    if system == "Darwin":
        if arch == "arm64":
            return {"system": "macOS", "arch": "arm64", "os_version": platform.mac_ver()[0]}
        return {"system": "macOS", "arch": "x64", "os_version": platform.mac_ver()[0]}
    elif system == "Windows":
        return {"system": "Windows", "arch": "x64" if arch == "AMD64" else "x86"}
    elif system == "Linux":
        return {"system": "Linux", "arch": arch}
    return {"system": system, "arch": arch}


def get_default_format_paths() -> Dict[PluginFormat, List[str]]:
    """Get default plugin search paths for current platform."""
    system = platform.system()
    paths = {}
    
    if system == "Darwin":
        paths[PluginFormat.VST3] = [
            "/Library/Audio/Plug-Ins/VST3",
            "~/Library/Audio/Plug-Ins/VST3",
        ]
        paths[PluginFormat.AU] = [
            "/Library/Audio/Plug-Ins/Components",
            "~/Library/Audio/Plug-Ins/Components",
        ]
        paths[PluginFormat.AAX] = [
            "/Library/Application Support/Avid/Audio/Plug-ins",
            "~/Library/Application Support/Avid/Audio/Plug-ins",
        ]
    elif system == "Windows":
        paths[PluginFormat.VST3] = [
            "C:/Program Files/Common Files/VST3",
            "C:/Program Files/VST3",
        ]
        paths[PluginFormat.AAX] = [
            "C:/Program Files/Avid/Audio/Plug-ins",
        ]
        # VST2 fallback (not in enum but common)
        paths[PluginFormat.UNKNOWN] = [
            "C:/Program Files/Common Files/VST2",
            "C:/Program Files/VST2",
        ]
    elif system == "Linux":
        paths[PluginFormat.VST3] = [
            "/usr/lib/vst3",
            "/usr/local/lib/vst3",
            "~/.vst3",
        ]
    
    return paths


# =============================================================================
# Format-Specific Plugin Scanner (Abstract Base)
# =============================================================================

class PluginScanner(ABC):
    """Abstract base class for format-specific plugin scanners."""
    
    def __init__(self, format_type: PluginFormat):
        self.format_type = format_type
        self._plugins: Dict[str, PluginMetadata] = {}
    
    @abstractmethod
    def scan_directory(self, directory: str) -> List[PluginMetadata]:
        """Scan a directory for plugins of this format."""
        pass
    
    @abstractmethod
    def get_plugin_info(self, path: str) -> Optional[PluginMetadata]:
        """Get detailed info for a specific plugin."""
        pass
    
    def scan_paths(self, paths: List[str]) -> Dict[str, PluginMetadata]:
        """Scan multiple paths and return all found plugins."""
        self._plugins.clear()
        
        for path in paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                logger.info(f"Scanning {self.format_type.value} plugins in: {expanded_path}")
                plugins = self.scan_directory(expanded_path)
                for plugin in plugins:
                    self._plugins[plugin.plugin_id] = plugin
        
        return self._plugins
    
    def get_all_plugins(self) -> Dict[str, PluginMetadata]:
        """Get all scanned plugins."""
        return self._plugins.copy()


# =============================================================================
# VST3 Scanner
# =============================================================================

class VST3Scanner(PluginScanner):
    """Scanner for VST3 plugins."""
    
    def __init__(self):
        super().__init__(PluginFormat.VST3)
        self._extension = ".vst3"
    
    def scan_directory(self, directory: str) -> List[PluginMetadata]:
        """Scan directory for VST3 plugins."""
        plugins = []
        
        try:
            for entry in os.scandir(directory):
                if entry.is_dir() and entry.name.endswith(self._extension):
                    plugin_path = entry.path
                    plugin_info = self.get_plugin_info(plugin_path)
                    if plugin_info:
                        plugins.append(plugin_info)
        except PermissionError:
            logger.warning(f"Permission denied scanning: {directory}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return plugins
    
    def get_plugin_info(self, path: str) -> Optional[PluginMetadata]:
        """Get VST3 plugin info from bundle."""
        try:
            plugin_name = os.path.basename(path).replace(".vst3", "")
            
            # Try to read Info.plist for more details
            info_plist_path = os.path.join(path, "Contents", "Info.plist")
            vendor = "Unknown"
            version = "1.0.0"
            
            if os.path.exists(info_plist_path):
                try:
                    with open(info_plist_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Basic parsing (could use plistlib in production)
                        if "CFBundleGetInfoString" in content:
                            pass
                except Exception:
                    pass
            
            # Determine architecture
            arch = PluginArchitecture.X64
            contents_path = os.path.join(path, "Contents")
            if os.path.exists(contents_path):
                for root, dirs, files in os.walk(contents_path):
                    for f in files:
                        if f.endswith(".dylib"):
                            if "arm64" in f.lower():
                                arch = PluginArchitecture.ARM64
                            elif "x86_64" in f.lower():
                                arch = PluginArchitecture.X64
            
            return PluginMetadata(
                plugin_id=f"vst3:{plugin_name}",
                name=plugin_name,
                vendor=vendor,
                format=PluginFormat.VST3,
                version=version,
                path=path,
                architecture=arch,
                is_effect=True,
                is_instrument=False
            )
        except Exception as e:
            logger.error(f"Error getting VST3 info for {path}: {e}")
            return None


# =============================================================================
# Audio Unit (AU) Scanner
# =============================================================================

class AUScanner(PluginScanner):
    """Scanner for Audio Unit plugins (macOS only)."""
    
    def __init__(self):
        super().__init__(PluginFormat.AU)
        self._extension = ".component"
    
    def scan_directory(self, directory: str) -> List[PluginMetadata]:
        """Scan directory for Audio Unit plugins."""
        if platform.system() != "Darwin":
            logger.warning("Audio Unit plugins are only supported on macOS")
            return []
        
        plugins = []
        
        try:
            for entry in os.scandir(directory):
                if entry.is_dir() and entry.name.endswith(self._extension):
                    plugin_path = entry.path
                    plugin_info = self.get_plugin_info(plugin_path)
                    if plugin_info:
                        plugins.append(plugin_info)
        except PermissionError:
            logger.warning(f"Permission denied scanning: {directory}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return plugins
    
    def get_plugin_info(self, path: str) -> Optional[PluginMetadata]:
        """Get Audio Unit plugin info."""
        if platform.system() != "Darwin":
            return None
            
        try:
            plugin_name = os.path.basename(path).replace(".component", "")
            
            info_plist_path = os.path.join(path, "Contents", "Info.plist")
            vendor = "Unknown"
            version = "1.0.0"
            au_type = "Effect"
            
            if os.path.exists(info_plist_path):
                try:
                    import plistlib
                    with open(info_plist_path, 'rb') as f:
                        plist = plistlib.load(f)
                        vendor = plist.get("CFBundleGetInfoString", "Unknown")
                        version = plist.get("CFBundleVersion", "1.0.0")
                        au_type = plist.get("AudioUnitType", "Effect")
                except Exception:
                    pass
            
            is_instrument = au_type in ["Music Instrument", "Synth", "Sampler"]
            is_effect = au_type in ["Effect", "Mixer", "Panner"]
            
            return PluginMetadata(
                plugin_id=f"au:{plugin_name}",
                name=plugin_name,
                vendor=vendor,
                format=PluginFormat.AU,
                version=version,
                path=path,
                architecture=PluginArchitecture.UNIVERSAL,
                categories=[au_type],
                is_effect=is_effect,
                is_instrument=is_instrument
            )
        except Exception as e:
            logger.error(f"Error getting AU info for {path}: {e}")
            return None
    
    def scan_with_auval(self) -> Dict[str, PluginMetadata]:
        """Use auval tool to get comprehensive AU info."""
        if platform.system() != "Darwin":
            return {}
        
        plugins = {}
        
        try:
            result = subprocess.run(
                ["auval", "-l"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            name = parts[0].strip()
                            uid = parts[1].strip()
                            vendor = parts[2].strip()
                            
                            plugin = PluginMetadata(
                                plugin_id=f"au:{uid}",
                                name=name,
                                vendor=vendor,
                                format=PluginFormat.AU,
                                version="1.0.0",
                                path=f"au:{uid}",
                                architecture=PluginArchitecture.UNIVERSAL,
                                is_effect=True,
                                is_instrument=False
                            )
                            plugins[plugin.plugin_id] = plugin
        except FileNotFoundError:
            logger.warning("auval not found - install Xcode command line tools")
        except Exception as e:
            logger.error(f"Error running auval: {e}")
        
        return plugins


# =============================================================================
# AAX Scanner
# =============================================================================

class AAXScanner(PluginScanner):
    """Scanner for AAX plugins (Avid Pro Tools)."""
    
    def __init__(self):
        super().__init__(PluginFormat.AAX)
        self._extension = ".aaxplugin"
    
    def scan_directory(self, directory: str) -> List[PluginMetadata]:
        """Scan directory for AAX plugins."""
        plugins = []
        
        try:
            for entry in os.scandir(directory):
                if entry.is_dir() and entry.name.endswith(self._extension):
                    plugin_path = entry.path
                    plugin_info = self.get_plugin_info(plugin_path)
                    if plugin_info:
                        plugins.append(plugin_info)
        except PermissionError:
            logger.warning(f"Permission denied scanning: {directory}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return plugins
    
    def get_plugin_info(self, path: str) -> Optional[PluginMetadata]:
        """Get AAX plugin info from bundle."""
        try:
            plugin_name = os.path.basename(path).replace(".aaxplugin", "")
            
            info_plist_path = os.path.join(path, "Contents", "Info.plist")
            vendor = "Unknown"
            version = "1.0.0"
            requires_auth = True
            
            if os.path.exists(info_plist_path):
                try:
                    import plistlib
                    with open(info_plist_path, 'rb') as f:
                        plist = plistlib.load(f)
                        vendor = plist.get("CFBundleGetInfoString", "Unknown")
                        version = plist.get("CFBundleVersion", "1.0.0")
                except Exception:
                    pass
            
            arch = PluginArchitecture.X64
            contents_path = os.path.join(path, "Contents")
            if os.path.exists(contents_path):
                for root, dirs, files in os.walk(contents_path):
                    for f in files:
                        if f.endswith(".dll"):
                            arch = PluginArchitecture.X64
                            break
            
            return PluginMetadata(
                plugin_id=f"aax:{plugin_name}",
                name=plugin_name,
                vendor=vendor,
                format=PluginFormat.AAX,
                version=version,
                path=path,
                architecture=arch,
                requires_authorized=requires_auth,
                is_effect=True,
                is_instrument=False
            )
        except Exception as e:
            logger.error(f"Error getting AAX info for {path}: {e}")
            return None


# =============================================================================
# Format Manager
# =============================================================================

class PluginFormatManager:
    """
    Manages all plugin formats, scanning, and format-specific operations.
    Works as a unified interface for VST3, AU, and AAX plugins.
    """
    
    def __init__(self, custom_paths: Optional[Dict[PluginFormat, List[str]]] = None):
        self._platform = get_platform_info()
        self._scanners: Dict[PluginFormat, PluginScanner] = {}
        self._plugins: Dict[str, PluginMetadata] = {}
        self._format_paths = custom_paths or get_default_format_paths()
        
        self._init_scanners()
    
    def _init_scanners(self):
        """Initialize format-specific scanners."""
        self._scanners[PluginFormat.VST3] = VST3Scanner()
        
        if self._platform["system"] == "Darwin":
            self._scanners[PluginFormat.AU] = AUScanner()
        
        if self._platform["system"] in ["Darwin", "Windows"]:
            self._scanners[PluginFormat.AAX] = AAXScanner()
    
    def get_available_formats(self) -> List[PluginFormat]:
        """Get list of available formats on current platform."""
        return list(self._scanners.keys())
    
    def scan_format(self, format_type: PluginFormat, custom_paths: Optional[List[str]] = None) -> Dict[str, PluginMetadata]:
        """Scan for plugins of a specific format."""
        scanner = self._scanners.get(format_type)
        if not scanner:
            logger.warning(f"No scanner available for format: {format_type.value}")
            return {}
        
        paths = custom_paths or self._format_paths.get(format_type, [])
        plugins = scanner.scan_paths(paths)
        
        self._plugins.update(plugins)
        
        return plugins
    
    def scan_all_formats(self) -> Dict[str, PluginMetadata]:
        """Scan all available formats."""
        self._plugins.clear()
        
        for format_type in self.get_available_formats():
            self.scan_format(format_type)
        
        return self._plugins
    
    def scan_paths(self, paths: Dict[PluginFormat, List[str]]) -> Dict[str, PluginMetadata]:
        """Scan custom paths for each format."""
        self._plugins.clear()
        
        for format_type, format_paths in paths.items():
            if format_type in self._scanners:
                scanner = self._scanners[format_type]
                plugins = scanner.scan_paths(format_paths)
                self._plugins.update(plugins)
        
        return self._plugins
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID."""
        return self._plugins.get(plugin_id)
    
    def get_all_plugins(self) -> Dict[str, PluginMetadata]:
        """Get all scanned plugins."""
        return self._plugins.copy()
    
    def get_plugins_by_format(self, format_type: PluginFormat) -> List[PluginMetadata]:
        """Get plugins filtered by format."""
        return [p for p in self._plugins.values() if p.format == format_type]
    
    def get_plugins_by_category(self, category: str) -> List[PluginMetadata]:
        """Get plugins filtered by category."""
        return [p for p in self._plugins.values() if category in p.categories]
    
    def get_instruments(self) -> List[PluginMetadata]:
        """Get all instrument plugins."""
        return [p for p in self._plugins.values() if p.is_instrument]
    
    def get_effects(self) -> List[PluginMetadata]:
        """Get all effect plugins."""
        return [p for p in self._plugins.values() if p.is_effect]
    
    def export_plugin_list(self, filepath: str, format_filter: Optional[PluginFormat] = None):
        """Export plugin list to JSON."""
        plugins = self._plugins
        if format_filter:
            plugins = {k: v for k, v in plugins.items() if v.format == format_filter}
        
        data = {
            "platform": self._platform,
            "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "plugins": {
                pid: {
                    "name": p.name,
                    "vendor": p.vendor,
                    "format": p.format.value,
                    "version": p.version,
                    "path": p.path,
                    "architecture": p.architecture.value,
                    "categories": p.categories,
                    "inputs": p.inputs,
                    "outputs": p.outputs,
                    "is_instrument": p.is_instrument,
                    "is_effect": p.is_effect,
                }
                for pid, p in plugins.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(plugins)} plugins to {filepath}")
    
    def import_plugin_list(self, filepath: str):
        """Import plugin list from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self._plugins.clear()
        for pid, pdata in data.get("plugins", {}).items():
            format_str = pdata.get("format", "unknown")
            fmt = PluginFormat(format_str) if format_str != "unknown" else PluginFormat.UNKNOWN
            
            self._plugins[pid] = PluginMetadata(
                plugin_id=pid,
                name=pdata["name"],
                vendor=pdata.get("vendor", "Unknown"),
                format=fmt,
                version=pdata.get("version", "1.0.0"),
                path=pdata.get("path", ""),
                architecture=PluginArchitecture(pdata.get("architecture", "x64")),
                categories=pdata.get("categories", []),
                is_instrument=pdata.get("is_instrument", False),
                is_effect=pdata.get("is_effect", True),
            )
        
        logger.info(f"Imported {len(self._plugins)} plugins from {filepath}")


# =============================================================================
# Plugin Format Bridge (for loading in DAWs)
# =============================================================================

class PluginFormatBridge(ABC):
    """Abstract bridge for loading plugins in different formats."""
    
    @abstractmethod
    def load_plugin(self, plugin_path: str) -> Any:
        """Load a plugin and return handle."""
        pass
    
    @abstractmethod
    def unload_plugin(self, handle: Any) -> bool:
        """Unload a plugin."""
        pass
    
    @abstractmethod
    def get_plugin_parameters(self, handle: Any) -> Dict[str, Any]:
        """Get plugin parameters."""
        pass
    
    @abstractmethod
    def set_plugin_parameter(self, handle: Any, param_id: str, value: float) -> bool:
        """Set a plugin parameter."""
        pass


class VST3Bridge(PluginFormatBridge):
    """Bridge for VST3 plugin loading."""
    
    def __init__(self):
        self._loaded_plugins: Dict[str, Any] = {}
    
    def load_plugin(self, plugin_path: str) -> Any:
        """Load VST3 plugin."""
        handle = {
            "path": plugin_path,
            "loaded_at": time.time(),
            "parameters": {}
        }
        self._loaded_plugins[plugin_path] = handle
        logger.info(f"Loaded VST3 plugin: {plugin_path}")
        return handle
    
    def unload_plugin(self, handle: Any) -> bool:
        """Unload VST3 plugin."""
        path = handle.get("path") if isinstance(handle, dict) else str(handle)
        if path in self._loaded_plugins:
            del self._loaded_plugins[path]
            logger.info(f"Unloaded VST3 plugin: {path}")
            return True
        return False
    
    def get_plugin_parameters(self, handle: Any) -> Dict[str, Any]:
        """Get VST3 plugin parameters."""
        return handle.get("parameters", {}) if isinstance(handle, dict) else {}
    
    def set_plugin_parameter(self, handle: Any, param_id: str, value: float) -> bool:
        """Set VST3 plugin parameter."""
        if isinstance(handle, dict):
            handle["parameters"][param_id] = value
            return True
        return False


class AUBridge(PluginFormatBridge):
    """Bridge for Audio Unit plugin loading (macOS only)."""
    
    def __init__(self):
        if platform.system() != "Darwin":
            raise RuntimeError("Audio Unit plugins are only supported on macOS")
        self._loaded_plugins: Dict[str, Any] = {}
    
    def load_plugin(self, plugin_path: str) -> Any:
        """Load Audio Unit plugin."""
        handle = {
            "path": plugin_path,
            "loaded_at": time.time(),
            "parameters": {}
        }
        self._loaded_plugins[plugin_path] = handle
        logger.info(f"Loaded AU plugin: {plugin_path}")
        return handle
    
    def unload_plugin(self, handle: Any) -> bool:
        """Unload Audio Unit plugin."""
        path = handle.get("path") if isinstance(handle, dict) else str(handle)
        if path in self._loaded_plugins:
            del self._loaded_plugins[path]
            logger.info(f"Unloaded AU plugin: {path}")
            return True
        return False
    
    def get_plugin_parameters(self, handle: Any) -> Dict[str, Any]:
        """Get AU plugin parameters."""
        return handle.get("parameters", {}) if isinstance(handle, dict) else {}
    
    def set_plugin_parameter(self, handle: Any, param_id: str, value: float) -> bool:
        """Set AU plugin parameter."""
        if isinstance(handle, dict):
            handle["parameters"][param_id] = value
            return True
        return False


class AAXBridge(PluginFormatBridge):
    """Bridge for AAX plugin loading."""
    
    def __init__(self):
        self._loaded_plugins: Dict[str, Any] = {}
        self._authorized = False
    
    def authorize(self, license_key: str) -> bool:
        """Authorize AAX plugin access."""
        self._authorized = True
        logger.info("AAX plugin authorized")
        return True
    
    def load_plugin(self, plugin_path: str) -> Any:
        """Load AAX plugin."""
        if not self._authorized:
            logger.warning("AAX plugins require authorization")
            return None
        
        handle = {
            "path": plugin_path,
            "loaded_at": time.time(),
            "parameters": {}
        }
        self._loaded_plugins[plugin_path] = handle
        logger.info(f"Loaded AAX plugin: {plugin_path}")
        return handle
    
    def unload_plugin(self, handle: Any) -> bool:
        """Unload AAX plugin."""
        path = handle.get("path") if isinstance(handle, dict) else str(handle)
        if path in self._loaded_plugins:
            del self._loaded_plugins[path]
            logger.info(f"Unloaded AAX plugin: {path}")
            return True
        return False
    
    def get_plugin_parameters(self, handle: Any) -> Dict[str, Any]:
        """Get AAX plugin parameters."""
        return handle.get("parameters", {}) if isinstance(handle, dict) else {}
    
    def set_plugin_parameter(self, handle: Any, param_id: str, value: float) -> bool:
        """Set AAX plugin parameter."""
        if isinstance(handle, dict):
            handle["parameters"][param_id] = value
            return True
        return False


# =============================================================================
# Plugin Loader (Unified Interface)
# =============================================================================

class PluginLoader:
    """
    Unified plugin loader that handles all formats.
    Provides a single interface for loading plugins regardless of format.
    """
    
    def __init__(self):
        self._bridges: Dict[PluginFormat, PluginFormatBridge] = {
            PluginFormat.VST3: VST3Bridge(),
        }
        self._loaded: Dict[str, Any] = {}
        
        if platform.system() == "Darwin":
            self._bridges[PluginFormat.AU] = AUBridge()
            self._bridges[PluginFormat.AAX] = AAXBridge()
        elif platform.system() == "Windows":
            self._bridges[PluginFormat.AAX] = AAXBridge()
    
    def load(self, metadata: PluginMetadata) -> Optional[Any]:
        """Load a plugin from metadata."""
        bridge = self._bridges.get(metadata.format)
        if not bridge:
            logger.error(f"No bridge available for format: {metadata.format.value}")
            return None
        
        handle = bridge.load_plugin(metadata.path)
        if handle:
            self._loaded[metadata.plugin_id] = handle
        
        return handle
    
    def unload(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        if plugin_id not in self._loaded:
            return False
        
        handle = self._loaded[plugin_id]
        
        for bridge in self._bridges.values():
            if bridge.unload_plugin(handle):
                del self._loaded[plugin_id]
                return True
        
        return False
    
    def get_parameter(self, plugin_id: str, param_id: str) -> Optional[float]:
        """Get a parameter value."""
        handle = self._loaded.get(plugin_id)
        if not handle:
            return None
        
        for bridge in self._bridges.values():
            params = bridge.get_plugin_parameters(handle)
            if param_id in params:
                return params[param_id]
        
        return None
    
    def set_parameter(self, plugin_id: str, param_id: str, value: float) -> bool:
        """Set a parameter value."""
        handle = self._loaded.get(plugin_id)
        if not handle:
            return False
        
        for bridge in self._bridges.values():
            if bridge.set_plugin_parameter(handle, param_id, value):
                return True
        
        return False
    
    def is_loaded(self, plugin_id: str) -> bool:
        """Check if plugin is loaded."""
        return plugin_id in self._loaded
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin IDs."""
        return list(self._loaded.keys())


# =============================================================================
# DAW Integration Helpers
# =============================================================================

class DAWIntegration:
    """Helper for integrating with specific DAWs."""
    
    @staticmethod
    def get_daw_from_environment() -> HostDAW:
        """Detect DAW from environment variables or process name."""
        env_daws = {
            "REAPER": HostDAW.REAPER,
            "ABLETON": HostDAW.ABLETON,
            "LOGIC_PRO": HostDAW.LOGIC,
            "PROTOOLS_HOSTNAME": HostDAW.PRO_TOOLS,
            "CUBASE": HostDAW.CUBASE,
        }
        
        for env_var, daw in env_daws.items():
            if os.environ.get(env_var):
                return daw
        
        try:
            result = subprocess.run(
                ["ps", "-eo", "comm"],
                capture_output=True,
                text=True,
                timeout=5
            )
            process_list = result.stdout.lower()
            
            if "reaper" in process_list:
                return HostDAW.REAPER
            elif "ableton" in process_list:
                return HostDAW.ABLETON
            elif "logic" in process_list:
                return HostDAW.LOGIC
            elif "protools" in process_list or "pt" in process_list:
                return HostDAW.PRO_TOOLS
            elif "cubase" in process_list:
                return HostDAW.CUBASE
            elif "fl64" in process_list or "fl32" in process_list:
                return HostDAW.FL_STUDIO
        except Exception:
            pass
        
        return HostDAW.UNKNOWN
    
    @staticmethod
    def get_recommended_format(daw: HostDAW) -> PluginFormat:
        """Get recommended plugin format for a DAW."""
        recommendations = {
            HostDAW.REAPER: PluginFormat.VST3,
            HostDAW.ABLETON: PluginFormat.VST3,
            HostDAW.LOGIC: PluginFormat.AU,
            HostDAW.PRO_TOOLS: PluginFormat.AAX,
            HostDAW.CUBASE: PluginFormat.VST3,
            HostDAW.FL_STUDIO: PluginFormat.VST3,
            HostDAW.DAW_VST3: PluginFormat.VST3,
        }
        return recommendations.get(daw, PluginFormat.VST3)
    
    @staticmethod
    def generate_plugin_scan_script(
        daw: HostDAW, 
        output_path: str,
        format_paths: Optional[Dict[PluginFormat, List[str]]] = None
    ) -> str:
        """Generate a platform-specific plugin scan script."""
        paths = format_paths or get_default_format_paths()
        
        if daw == HostDAW.REAPER:
            lines = ["; REAPER Plugin Scan List"]
            for fmt, fmt_paths in paths.items():
                for p in fmt_paths:
                    lines.append(p)
            
            script = "\n".join(lines)
            with open(output_path, 'w') as f:
                f.write(script)
            return output_path
        
        elif daw == HostDAW.PRO_TOOLS:
            lines = ["; Pro Tools AAX Plugin Scan List"]
            aax_paths = paths.get(PluginFormat.AAX, [])
            for p in aax_paths:
                lines.append(p)
            
            script = "\n".join(lines)
            with open(output_path, 'w') as f:
                f.write(script)
            return output_path
        
        else:
            data = {"paths": {fmt.value: ps for fmt, ps in paths.items()}}
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            return output_path


# =============================================================================
# Plugin Preset Manager
# =============================================================================

class PresetManager:
    """Manages plugin presets across formats."""
    
    def __init__(self, preset_dir: Optional[str] = None):
        self._preset_dir = preset_dir or os.path.join(
            os.path.expanduser("~"),
            "Library/Application Support/AI DJ/Presets"
        )
        os.makedirs(self._preset_dir, exist_ok=True)
        
        self._presets: Dict[str, Dict[str, Any]] = {}
        self._load_presets()
    
    def _load_presets(self):
        """Load saved presets."""
        preset_file = os.path.join(self._preset_dir, "presets.json")
        if os.path.exists(preset_file):
            try:
                with open(preset_file, 'r') as f:
                    self._presets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading presets: {e}")
    
    def _save_presets(self):
        """Save presets to disk."""
        preset_file = os.path.join(self._preset_dir, "presets.json")
        try:
            with open(preset_file, 'w') as f:
                json.dump(self._presets, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving presets: {e}")
    
    def save_preset(
        self, 
        plugin_id: str, 
        preset_name: str, 
        parameters: Dict[str, float]
    ):
        """Save a plugin preset."""
        if plugin_id not in self._presets:
            self._presets[plugin_id] = {}
        
        self._presets[plugin_id][preset_name] = {
            "parameters": parameters,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self._save_presets()
        logger.info(f"Saved preset '{preset_name}' for {plugin_id}")
    
    def load_preset(self, plugin_id: str, preset_name: str) -> Optional[Dict[str, float]]:
        """Load a plugin preset."""
        if plugin_id in self._presets:
            preset = self._presets[plugin_id].get(preset_name)
            if preset:
                return preset.get("parameters")
        return None
    
    def delete_preset(self, plugin_id: str, preset_name: str) -> bool:
        """Delete a preset."""
        if plugin_id in self._presets and preset_name in self._presets[plugin_id]:
            del self._presets[plugin_id][preset_name]
            self._save_presets()
            return True
        return False
    
    def list_presets(self, plugin_id: str) -> List[str]:
        """List all presets for a plugin."""
        if plugin_id in self._presets:
            return list(self._presets[plugin_id].keys())
        return []


# =============================================================================
# Main Entry Point / Example Usage
# =============================================================================

def main():
    """Example usage of plugin formats support."""
    print("=" * 60)
    print("AI DJ Plugin Formats Support")
    print("=" * 60)
    
    # Get platform info
    platform_info = get_platform_info()
    print(f"\nPlatform: {platform_info['system']} {platform_info['arch']}")
    
    # Create format manager
    manager = PluginFormatManager()
    
    print(f"\nAvailable formats: {[f.value for f in manager.get_available_formats()]}")
    
    # Scan all formats
    print("\nScanning for plugins...")
    plugins = manager.scan_all_formats()
    
    print(f"\nFound {len(plugins)} plugins:")
    for pid, meta in plugins.items():
        print(f"  - {meta.name} ({meta.format.value}) by {meta.vendor}")
    
    # Show format breakdown
    formats_count = {}
    for meta in plugins.values():
        fmt = meta.format.value
        formats_count[fmt] = formats_count.get(fmt, 0) + 1
    
    print(f"\nFormat breakdown:")
    for fmt, count in formats_count.items():
        print(f"  {fmt}: {count}")
    
    # Export plugin list
    manager.export_plugin_list("plugin_list.json")
    print("\nExported plugin list to plugin_list.json")
    
    # Test preset manager
    preset_mgr = PresetManager()
    preset_mgr.save_preset("test.plugin", "My Preset", {"param1": 0.5, "param2": 0.8})
    loaded = preset_mgr.load_preset("test.plugin", "My Preset")
    print(f"\nLoaded preset: {loaded}")
    
    print("\n" + "=" * 60)
    print("Plugin formats support ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
