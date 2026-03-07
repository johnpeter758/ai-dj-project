#!/usr/bin/env python3
"""
Theme System for AI DJ Project
Provides consistent UI theming across the application.
Supports dark/light modes, custom color schemes, and matplotlib styling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib as mpl


class ThemeMode(Enum):
    """Theme operating modes"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"  # Follows system preference


# ============================================================================
# Color Palettes
# ============================================================================

@dataclass
class ColorPalette:
    """Base color palette for a theme"""
    # Primary colors
    primary: str = "#6366F1"        # Indigo
    primary_variant: str = "#4F46E5"
    secondary: str = "#EC4899"      # Pink
    secondary_variant: str = "#DB2777"
    
    # Background colors
    background: str = "#0F172A"      # Dark slate
    surface: str = "#1E293B"         # Lighter slate
    surface_variant: str = "#334155"
    
    # Text colors
    text_primary: str = "#F8FAFC"
    text_secondary: str = "#94A3B8"
    text_disabled: str = "#64748B"
    
    # Semantic colors
    success: str = "#22C55E"         # Green
    warning: str = "#F59E0B"         # Amber
    error: str = "#EF4444"           # Red
    info: str = "#3B82F6"            # Blue
    
    # Waveform/visualization colors
    waveform: str = "#22D3EE"        # Cyan
    waveform_secondary: str = "#A78BFA"  # Purple
    meter_green: str = "#4ADE80"
    meter_yellow: str = "#FACC15"
    meter_red: str = "#F87171"
    
    # Graph/plot colors
    plot_line_1: str = "#22D3EE"     # Cyan
    plot_line_2: str = "#F472B6"     # Pink
    plot_line_3: str = "#A78BFA"     # Purple
    plot_line_4: str = "#34D399"     # Emerald
    plot_fill: str = "#22D3EE"
    
    # Grid
    grid: str = "#334155"
    grid_major: str = "#475569"
    
    # Borders
    border: str = "#475569"
    border_light: str = "#64748B"
    
    def to_list(self) -> List[str]:
        """Return color list for sequential plots"""
        return [
            self.plot_line_1,
            self.plot_line_2,
            self.plot_line_3,
            self.plot_line_4,
        ]


# Default dark palette
DEFAULT_DARK_PALETTE = ColorPalette()


# Light palette
LIGHT_PALETTE = ColorPalette(
    primary="#6366F1",
    primary_variant="#4F46E5",
    secondary="#EC4899",
    secondary_variant="#DB2777",
    background="#F8FAFC",
    surface="#FFFFFF",
    surface_variant="#F1F5F9",
    text_primary="#0F172A",
    text_secondary="#475569",
    text_disabled="#94A3B8",
    success="#16A34A",
    warning="#D97706",
    error="#DC2626",
    info="#2563EB",
    waveform="#0891B2",
    waveform_secondary="#7C3AED",
    meter_green="#22C55E",
    meter_yellow="#EAB308",
    meter_red="#EF4444",
    plot_line_1="#0891B2",
    plot_line_2="#DB2777",
    plot_line_3="#7C3AED",
    plot_line_4="#059669",
    plot_fill="#0891B2",
    grid="#E2E8F0",
    grid_major="#CBD5E1",
    border="#CBD5E1",
    border_light="#E2E8F0",
)


# Genre-specific accent colors
GENRE_COLORS: Dict[str, str] = {
    "pop": "#EC4899",
    "house": "#22D3EE",
    "techno": "#A78BFA",
    "trance": "#F472B6",
    "dubstep": "#FB923C",
    "hip-hop": "#FACC15",
    "rnb": "#FBBF24",
    "rock": "#EF4444",
    "edm": "#22C55E",
    "ambient": "#38BDF8",
    "default": "#6366F1",
}


# ============================================================================
# Typography
# ============================================================================

@dataclass
class Typography:
    """Font and text settings"""
    # Font families
    font_family: str = "sans-serif"
    font_family_monospace: str = "monospace"
    
    # Font sizes
    font_size_xs: int = 8
    font_size_sm: int = 10
    font_size_md: int = 12
    font_size_lg: int = 14
    font_size_xl: int = 18
    font_size_xxl: int = 24
    
    # Title sizes
    title_size: int = 16
    subtitle_size: int = 14
    axis_label_size: int = 11
    tick_label_size: int = 9
    
    # Weights
    font_weight_normal: str = "normal"
    font_weight_bold: str = "bold"


DEFAULT_TYPOGRAPHY = Typography()


# ============================================================================
# Spacing & Layout
# ============================================================================

@dataclass
class Layout:
    """Spacing and layout settings"""
    # Padding
    padding_xs: int = 4
    padding_sm: int = 8
    padding_md: int = 12
    padding_lg: int = 16
    padding_xl: int = 24
    
    # Margins
    margin_sm: int = 8
    margin_md: int = 16
    margin_lg: int = 24
    
    # Figure sizes
    figure_small: Tuple[int, int] = (6, 4)
    figure_medium: Tuple[int, int] = (10, 6)
    figure_large: Tuple[int, int] = (14, 8)
    figure_xlarge: Tuple[int, int] = (16, 10)
    
    # DPI
    dpi_default: int = 100
    dpi_high: int = 150
    
    # Grid
    grid_alpha: float = 0.3
    grid_linestyle: str = "--"


DEFAULT_LAYOUT = Layout()


# ============================================================================
# Complete Theme
# ============================================================================

@dataclass
class Theme:
    """
    Complete theme configuration including colors, typography, and layout.
    """
    name: str = "default"
    mode: ThemeMode = ThemeMode.DARK
    
    colors: ColorPalette = field(default_factory=lambda: DEFAULT_DARK_PALETTE)
    typography: Typography = field(default_factory=lambda: DEFAULT_TYPOGRAPHY)
    layout: Layout = field(default_factory=lambda: DEFAULT_LAYOUT)
    
    # Matplotlib rcParams overrides
    rc_params: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate matplotlib rcParams from theme"""
        self._update_rc_params()
    
    def _update_rc_params(self):
        """Update matplotlib parameters based on theme colors"""
        self.rc_params = {
            # Figure
            "figure.facecolor": self.colors.surface,
            "figure.edgecolor": self.colors.border,
            "figure.dpi": self.layout.dpi_default,
            
            # Axes
            "axes.facecolor": self.colors.surface,
            "axes.edgecolor": self.colors.border,
            "axes.labelcolor": self.colors.text_secondary,
            "axes.titlecolor": self.colors.text_primary,
            
            # Text
            "text.color": self.colors.text_primary,
            "text.usetex": False,
            
            # Lines
            "lines.color": self.colors.waveform,
            "lines.linewidth": 1.5,
            
            # Grid
            "grid.color": self.colors.grid,
            "grid.linestyle": self.layout.grid_linestyle,
            "grid.alpha": self.layout.grid_alpha,
            
            # Ticks
            "xtick.color": self.colors.text_secondary,
            "ytick.color": self.colors.text_secondary,
            "xtick.labelsize": self.typography.tick_label_size,
            "ytick.labelsize": self.typography.tick_label_size,
            
            # Legend
            "legend.facecolor": self.colors.surface_variant,
            "legend.edgecolor": self.colors.border,
            "legend.labelcolor": self.colors.text_secondary,
            "legend.fontsize": self.typography.font_size_sm,
            
            # Saving
            "savefig.dpi": self.layout.dpi_default,
            "savefig.facecolor": self.colors.surface,
            "savefig.edgecolor": "none",
        }
    
    def apply_matplotlib(self):
        """Apply theme to matplotlib"""
        mpl.rcParams.update(self.rc_params)
    
    def get_waveform_style(self) -> Dict:
        """Get matplotlib style dict for waveforms"""
        return {
            "color": self.colors.waveform,
            "linewidth": 1.5,
            "fill": True,
            "fill_color": self.colors.plot_fill,
            "fill_alpha": 0.3,
        }
    
    def get_meter_colors(self) -> Tuple[str, str, str]:
        """Get (green, yellow, red) meter colors"""
        return (self.colors.meter_green, 
                self.colors.meter_yellow, 
                self.colors.meter_red)
    
    def get_plot_cycle(self) -> List[str]:
        """Get color cycle for multi-line plots"""
        return self.colors.to_list()


# ============================================================================
# Theme Registry
# ============================================================================

class ThemeRegistry:
    """Registry of available themes"""
    
    _themes: Dict[str, Theme] = {}
    _current: Optional[Theme] = None
    _mode: ThemeMode = ThemeMode.DARK
    
    @classmethod
    def register(cls, theme: Theme):
        """Register a new theme"""
        cls._themes[theme.name] = theme
    
    @classmethod
    def get(cls, name: str) -> Optional[Theme]:
        """Get theme by name"""
        return cls._themes.get(name)
    
    @classmethod
    def list_themes(cls) -> List[str]:
        """List all registered theme names"""
        return list(cls._themes.keys())
    
    @classmethod
    def set_current(cls, name: str) -> Theme:
        """Set current theme by name"""
        theme = cls.get(name)
        if theme is None:
            raise ValueError(f"Theme '{name}' not found")
        cls._current = theme
        cls._current.apply_matplotlib()
        return theme
    
    @classmethod
    def get_current(cls) -> Theme:
        """Get current theme"""
        if cls._current is None:
            cls._current = Theme(name="default", mode=ThemeMode.DARK)
        return cls._current


# Register default themes
ThemeRegistry.register(Theme(
    name="dark",
    mode=ThemeMode.DARK,
    colors=DEFAULT_DARK_PALETTE,
))

ThemeRegistry.register(Theme(
    name="light", 
    mode=ThemeMode.LIGHT,
    colors=LIGHT_PALETTE,
))


# ============================================================================
# Convenience Functions
# ============================================================================

def get_theme(name: str = "dark") -> Theme:
    """
    Get a theme by name.
    
    Args:
        name: Theme name ('dark', 'light', or registered custom theme)
    
    Returns:
        Theme object
    """
    return ThemeRegistry.get(name) or ThemeRegistry.set_current(name)


def use_theme(name: str = "dark"):
    """
    Apply a theme to matplotlib.
    
    Args:
        name: Theme name to use
    """
    ThemeRegistry.set_current(name)


def get_color_for_genre(genre: str) -> str:
    """
    Get accent color for a music genre.
    
    Args:
        genre: Music genre name
    
    Returns:
        Hex color string
    """
    return GENRE_COLORS.get(genre.lower(), GENRE_COLORS["default"])


def style_waveform(ax, theme: Optional[Theme] = None):
    """
    Apply theme styling to a waveform axes.
    
    Args:
        ax: Matplotlib axes
        theme: Theme to use (uses current if None)
    """
    if theme is None:
        theme = ThemeRegistry.get_current()
    
    ax.set_facecolor(theme.colors.surface)
    ax.spines['bottom'].set_color(theme.colors.border)
    ax.spines['left'].set_color(theme.colors.border)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=theme.colors.text_secondary)
    ax.xaxis.label.set_color(theme.colors.text_secondary)
    ax.yaxis.label.set_color(theme.colors.text_secondary)
    ax.title.set_color(theme.colors.text_primary)
    ax.grid(True, alpha=0.3, color=theme.colors.grid)


def create_figure(theme: Optional[Theme] = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a themed matplotlib figure.
    
    Args:
        theme: Theme to use (uses current if None)
        **kwargs: Additional args passed to plt.subplots
    
    Returns:
        (figure, axes) tuple
    """
    if theme is None:
        theme = ThemeRegistry.get_current()
    
    defaults = {
        "figsize": theme.layout.figure_medium,
        "facecolor": theme.colors.surface,
    }
    defaults.update(kwargs)
    
    fig, ax = plt.subplots(**defaults)
    style_waveform(ax, theme)
    
    return fig, ax


# ============================================================================
# Global Theme Management
# ============================================================================

def init_theme(mode: str = "dark"):
    """
    Initialize the global theme.
    
    Args:
        mode: 'dark', 'light', or 'auto'
    """
    if mode == "auto":
        # Could implement system preference detection here
        mode = "dark"
    
    use_theme(mode)


# Auto-initialize default theme
init_theme("dark")
