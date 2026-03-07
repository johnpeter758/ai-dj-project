"""
Compressor GUI - Visualization and control for DynamicCompressor
Real-time visualization of compressor behavior with parameter controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from typing import Optional
import threading
import time


class CompressorVisualization:
    """
    Interactive compressor visualization with real-time parameter adjustment.
    Shows transfer curve, gain reduction meter, and live input/output waveforms.
    """
    
    def __init__(self, compressor=None, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.compressor = compressor
        
        # Internal state
        self._running = False
        self._thread = None
        self.input_buffer = []
        self.output_buffer = []
        self.gr_history = []
        
        # Visualization settings
        self.figsize = (14, 10)
        self.buffer_size = 2048
        
    def _create_compressor(self):
        """Create default compressor if none provided."""
        if self.compressor is None:
            from dynamic_compressor import DynamicCompressor
            self.compressor = DynamicCompressor(sample_rate=self.sample_rate)
    
    def _compute_transfer_curve(self) -> tuple:
        """Compute input/output transfer curve."""
        # Generate input levels from -60 to 0 dB
        input_db = np.linspace(-60, 0, 500)
        output_db = []
        
        for inp in input_db:
            # Compute gain reduction
            gr = self.compressor._compute_gain_db(inp)
            # Apply gain reduction and makeup gain
            out = inp + gr + self.compressor.makeup_gain_db
            output_db.append(out)
        
        return input_db, np.array(output_db)
    
    def _compute_gain_curve(self) -> tuple:
        """Compute the gain reduction curve."""
        input_db = np.linspace(-60, 0, 500)
        gain_reduction = [self.compressor._compute_gain_db(inp) for inp in input_db]
        return input_db, np.array(gain_reduction)
    
    def plot_static(self, show_gr_curve: bool = True):
        """
        Create static visualization of compressor characteristics.
        
        Args:
            show_gr_curve: Whether to show gain reduction curve
        """
        self._create_compressor()
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle('Compressor Visualization & Control', fontsize=14, fontweight='bold')
        
        # Subplot layout: 2x2 grid
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.6], hspace=0.35, wspace=0.25)
        
        # ==== 1. Transfer Curve (Input vs Output) ====
        self.ax_transfer = self.fig.add_subplot(gs[0, 0])
        input_db, output_db = self._compute_transfer_curve()
        
        # Plot transfer curve
        self.ax_transfer.plot(input_db, output_db, 'b-', linewidth=2, label='Transfer')
        
        # Plot threshold line
        thresh = self.compressor.threshold_db
        self.ax_transfer.axvline(x=thresh, color='r', linestyle='--', alpha=0.7, label=f'Threshold: {thresh:.1f} dB')
        
        # Plot 1:1 line (unity)
        self.ax_transfer.plot([-60, 0], [-60, 0], 'k--', alpha=0.4, label='Unity (1:1)')
        
        # Fill above threshold
        mask = input_db >= thresh
        if np.any(mask):
            self.ax_transfer.fill_between(input_db[mask], output_db[mask], input_db[mask], 
                                           alpha=0.3, color='blue', label='Compression')
        
        self.ax_transfer.set_xlabel('Input (dB)')
        self.ax_transfer.set_ylabel('Output (dB)')
        self.ax_transfer.set_title('Transfer Curve')
        self.ax_transfer.set_xlim(-60, 0)
        self.ax_transfer.set_ylim(-60, 6)
        self.ax_transfer.legend(loc='upper left', fontsize=8)
        self.ax_transfer.grid(True, alpha=0.3)
        
        # ==== 2. Gain Reduction Curve ====
        self.ax_gr = self.fig.add_subplot(gs[0, 1])
        input_db, gr = self._compute_gain_curve()
        
        self.ax_gr.plot(input_db, -gr, 'r-', linewidth=2)
        self.ax_gr.fill_between(input_db, 0, -gr, alpha=0.3, color='red')
        self.ax_gr.axvline(x=thresh, color='r', linestyle='--', alpha=0.7)
        
        self.ax_gr.set_xlabel('Input (dB)')
        self.ax_gr.set_ylabel('Gain Reduction (dB)')
        self.ax_gr.set_title('Gain Reduction')
        self.ax_gr.set_xlim(-60, 0)
        self.ax_gr.set_ylim(0, max(30, -min(gr) + 5))
        self.ax_gr.grid(True, alpha=0.3)
        
        # ==== 3. Parameter Controls ====
        self.ax_controls = self.fig.add_subplot(gs[1, :])
        self.ax_controls.set_title('Parameters', pad=10)
        self.ax_controls.axis('off')
        
        # Create sliders
        self.sliders = {}
        slider_config = [
            ('threshold', 'Threshold (dB)', -60, 0, self.compressor.threshold_db),
            ('ratio', 'Ratio', 1, 20, self.compressor.ratio),
            ('attack', 'Attack (ms)', 0.1, 100, self.compressor.attack_ms),
            ('release', 'Release (ms)', 10, 1000, self.compressor.release_ms),
            ('knee', 'Knee (dB)', 0, 24, self.compressor.knee_db),
            ('makeup', 'Makeup (dB)', -24, 24, self.compressor.makeup_gain_db),
        ]
        
        slider_positions = [
            (0.08, 0.65, 0.35, 0.08),
            (0.08, 0.45, 0.35, 0.08),
            (0.08, 0.25, 0.35, 0.08),
            (0.55, 0.65, 0.35, 0.08),
            (0.55, 0.45, 0.35, 0.08),
            (0.55, 0.25, 0.35, 0.08),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(slider_config):
            ax = self.fig.add_axes(slider_positions[i])
            self.sliders[key] = Slider(ax, label, min_val, max_val, valinit=default)
            self.sliders[key].on_changed(self._on_param_change)
        
        # ==== 4. Mode Selection ====
        self.ax_radio = self.fig.add_axes([0.78, 0.55, 0.15, 0.15])
        self.ax_radio.axis('off')
        
        # Detector mode radio
        self.radio_detector = RadioButtons(
            self.ax_radio, 
            ('Peak', 'RMS'),
            activecolor='blue'
        )
        self.radio_detector.on_clicked(self._on_detector_change)
        self.radio_detector.ax.set_title('Detector', fontsize=9, loc='left')
        
        # Knee type radio
        self.ax_radio2 = self.fig.add_axes([0.78, 0.35, 0.15, 0.15])
        self.ax_radio2.axis('off')
        
        self.radio_knee = RadioButtons(
            self.ax_radio2,
            ('Soft', 'Hard'),
            activecolor='green'
        )
        self.radio_knee.on_clicked(self._on_knee_change)
        self.radio_knee.ax.set_title('Knee', fontsize=9, loc='left')
        
        # ==== 5. Gain Reduction Meter ====
        self.ax_meter = self.fig.add_subplot(gs[2, 0])
        self.ax_meter.set_title('Gain Reduction Meter', pad=5)
        self.ax_meter.axis('off')
        
        # Create GR meter rectangle
        self.gr_meter_bg = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                              edgecolor='black', facecolor='#222')
        self.gr_meter_fg = patches.Rectangle((0, 0), 1, 0, linewidth=0,
                                              facecolor='#00ff00')
        self.ax_meter.add_patch(self.gr_meter_bg)
        self.ax_meter.add_patch(self.gr_meter_fg)
        self.ax_meter.set_xlim(-0.1, 1.1)
        self.ax_meter.set_ylim(-0.1, 1.1)
        
        # GR meter labels
        self.gr_meter_label = self.ax_meter.text(0.5, 0.5, '0.0 dB', 
                                                  ha='center', va='center',
                                                  fontsize=14, fontweight='bold',
                                                  color='white')
        
        # ==== 6. Info Panel ====
        self.ax_info = self.fig.add_subplot(gs[2, 1])
        self.ax_info.axis('off')
        
        self.info_text = self.ax_info.text(
            0.05, 0.95, 
            self._get_info_text(),
            transform=self.ax_info.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace'
        )
        
        # Connect close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        plt.show()
    
    def _get_info_text(self) -> str:
        """Get current parameter info string."""
        c = self.compressor
        return (
            f"┌─ Compressor Settings ─────────┐\n"
            f"│ Threshold:  {c.threshold_db:>6.1f} dB          │\n"
            f"│ Ratio:      {c.ratio:>6.1f}:1            │\n"
            f"│ Attack:     {c.attack_ms:>6.1f} ms          │\n"
            f"│ Release:    {c.release_ms:>6.1f} ms          │\n"
            f"│ Knee:       {c.knee_db:>6.1f} dB           │\n"
            f"│ Makeup:     {c.makeup_gain_db:>6.1f} dB          │\n"
            f"│ Detector:   {c.detector_mode.upper():<6}          │\n"
            f"│ Knee Type:  {c.knee_type.upper():<6}          │\n"
            f"└────────────────────────────────┘"
        )
    
    def _on_param_change(self, val):
        """Handle slider parameter changes."""
        self.compressor.set_parameters(
            threshold_db=self.sliders['threshold'].val,
            ratio=self.sliders['ratio'].val,
            attack_ms=self.sliders['attack'].val,
            release_ms=self.sliders['release'].val,
            knee_db=self.sliders['knee'].val,
            makeup_gain_db=self.sliders['makeup'].val
        )
        self._update_visualization()
    
    def _on_detector_change(self, label):
        """Handle detector mode change."""
        mode = 'peak' if label == 'Peak' else 'rms'
        self.compressor.set_parameters(detector_mode=mode)
        self._update_visualization()
    
    def _on_knee_change(self, label):
        """Handle knee type change."""
        knee = label.lower()
        self.compressor.set_parameters(knee_type=knee)
        self._update_visualization()
    
    def _update_visualization(self):
        """Update all visualization elements."""
        # Update transfer curve
        self.ax_transfer.clear()
        input_db, output_db = self._compute_transfer_curve()
        
        thresh = self.compressor.threshold_db
        self.ax_transfer.plot(input_db, output_db, 'b-', linewidth=2, label='Transfer')
        self.ax_transfer.axvline(x=thresh, color='r', linestyle='--', alpha=0.7, 
                                  label=f'Threshold: {thresh:.1f} dB')
        self.ax_transfer.plot([-60, 0], [-60, 0], 'k--', alpha=0.4, label='Unity (1:1)')
        
        mask = input_db >= thresh
        if np.any(mask):
            self.ax_transfer.fill_between(input_db[mask], output_db[mask], input_db[mask], 
                                           alpha=0.3, color='blue')
        
        self.ax_transfer.set_xlabel('Input (dB)')
        self.ax_transfer.set_ylabel('Output (dB)')
        self.ax_transfer.set_title('Transfer Curve')
        self.ax_transfer.set_xlim(-60, 0)
        self.ax_transfer.set_ylim(-60, 6)
        self.ax_transfer.legend(loc='upper left', fontsize=8)
        self.ax_transfer.grid(True, alpha=0.3)
        
        # Update GR curve
        self.ax_gr.clear()
        input_db, gr = self._compute_gain_curve()
        self.ax_gr.plot(input_db, -gr, 'r-', linewidth=2)
        self.ax_gr.fill_between(input_db, 0, -gr, alpha=0.3, color='red')
        self.ax_gr.axvline(x=thresh, color='r', linestyle='--', alpha=0.7)
        self.ax_gr.set_xlabel('Input (dB)')
        self.ax_gr.set_ylabel('Gain Reduction (dB)')
        self.ax_gr.set_title('Gain Reduction')
        self.ax_gr.set_xlim(-60, 0)
        self.ax_gr.set_ylim(0, max(30, -min(gr) + 5))
        self.ax_gr.grid(True, alpha=0.3)
        
        # Update info text
        self.info_text.set_text(self._get_info_text())
        
        self.fig.canvas.draw_idle()
    
    def _on_close(self, event):
        """Handle figure close event."""
        self._running = False
    
    def simulate_audio(self, duration_sec: float = 5.0, 
                       input_level_db: float = -12.0):
        """
        Simulate audio processing for visualization.
        
        Args:
            duration_sec: Duration to simulate
            input_level_db: Input level in dB
        """
        # Generate test signal
        num_samples = int(self.sample_rate * duration_sec)
        
        # Mix of frequencies for realistic signal
        t = np.arange(num_samples) / self.sample_rate
        signal_osc = (
            np.sin(2 * np.pi * 440 * t) * 0.3 +
            np.sin(2 * np.pi * 880 * t) * 0.2 +
            np.sin(2 * np.pi * 220 * t) * 0.2 +
            np.random.randn(num_samples) * 0.1
        )
        
        # Convert to dB (with some variation for dynamics)
        base_amplitude = 10 ** (input_level_db / 20)
        signal_amplitude = signal_osc * base_amplitude
        
        # Process through compressor
        output = np.zeros(num_samples)
        for i in range(num_samples):
            # Get input level
            if self.compressor.detector_mode == 'peak':
                input_level = np.abs(signal_amplitude[i])
            else:
                # RMS over small window
                start = max(0, i - 128)
                input_level = np.sqrt(np.mean(signal_amplitude[start:i+1]**2))
            
            input_level_db = 20 * np.log10(max(input_level, 1e-10))
            
            # Compute gain
            gr = self.compressor._compute_gain_db(input_level_db)
            gain = 10 ** ((gr + self.compressor.makeup_gain_db) / 20)
            
            output[i] = signal_amplitude[i] * gain
            
            # Track gain reduction
            self.gr_history.append(-gr)
            if len(self.gr_history) > 500:
                self.gr_history.pop(0)
        
        return signal_amplitude, output


class CompressorGUI:
    """
    Standalone Compressor GUI application.
    Launch with: python compressor_gui.py
    """
    
    def __init__(self):
        self.vis = None
        self.fig = None
    
    def launch(self):
        """Launch the compressor GUI."""
        print("🎛️  Starting Compressor GUI...")
        print("   Close the window to exit.")
        
        self.vis = CompressorVisualization()
        self.vis.plot_static()
    
    def launch_with_audio(self, audio_file: str = None):
        """
        Launch GUI with audio file processing visualization.
        
        Args:
            audio_file: Optional path to audio file
        """
        print("🎛️  Starting Compressor GUI with Audio...")
        
        self.vis = CompressorVisualization()
        
        if audio_file:
            print(f"   Loading: {audio_file}")
            # Would load and process audio here
            # For now, just show static visualization
        
        self.vis.plot_static()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compressor Visualization GUI')
    parser.add_argument('--audio', '-a', type=str, help='Audio file to process')
    args = parser.parse_args()
    
    gui = CompressorGUI()
    
    if args.audio:
        gui.launch_with_audio(args.audio)
    else:
        gui.launch()


if __name__ == '__main__':
    main()
