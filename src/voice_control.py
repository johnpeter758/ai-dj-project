#!/usr/bin/env python3
"""
Voice Control Module for AI DJ
Allows hands-free control of the AI DJ system via voice commands.
"""

import re
import speech_recognition as sr
import argparse
import sys
from typing import Optional, Callable, Dict, Any
from enum import Enum


class CommandType(Enum):
    GENERATE = "generate"
    ANALYZE = "analyze"
    FUSION = "fusion"
    PLAY = "play"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    VOLUME = "volume"
    HELP = "help"
    STATUS = "status"
    UNKNOWN = "unknown"


# Command patterns for voice recognition
COMMAND_PATTERNS = {
    CommandType.GENERATE: [
        r"generate(?: a)? (.+?)(?: at (\d+))?(?: bpm)?(?: in (.+?))?$",
        r"create(?: a)? (.+?)(?: at (\d+))?(?: bpm)?(?: in (.+?))?$",
        r"make(?: a)? (.+?)(?: at (\d+))?(?: bpm)?$",
        r"new track (.+?)(?: at (\d+))?(?: bpm)?$",
    ],
    CommandType.ANALYZE: [
        r"analyze (.+?)$",
        r"what is (.+?)\??$",
        r"tell me about (.+?)$",
    ],
    CommandType.FUSION: [
        r"fusion (.+?) (?:and|with) (.+?)$",
        r"mix (.+?) (?:and|with) (.+?)$",
        r"blend (.+?) (?:and|with) (.+?)$",
        r"fuse (.+?) (?:and|with) (.+?)$",
    ],
    CommandType.PLAY: [
        r"play(?:\s+)?$",
        r"start(?:\s+)?$",
        r"resume(?:\s+)?$",
    ],
    CommandType.STOP: [
        r"stop(?:\s+)?$",
        r"halt(?:\s+)?$",
        r"end(?:\s+)?$",
    ],
    CommandType.PAUSE: [
        r"pause(?:\s+)?$",
        r"wait(?:\s+)?$",
    ],
    CommandType.VOLUME: [
        r"volume(?: set)? (.+?)$",
        r"set volume (.+?)$",
        r"turn (?:it )?(?:up|down)$",
    ],
    CommandType.HELP: [
        r"help(?:\s+)?$",
        r"commands(?:\s+)?$",
        r"what can (?:you|i) say\??$",
    ],
    CommandType.STATUS: [
        r"status(?:\s+)?$",
        r"what'?s playing(?:\s+)?$",
        r"now playing(?:\s+)?$",
    ],
}


class VoiceCommand:
    """Parsed voice command with extracted parameters."""
    def __init__(self, command_type: CommandType, raw_text: str, params: Dict[str, Any] = None):
        self.command_type = command_type
        self.raw_text = raw_text
        self.params = params or {}


class VoiceController:
    """
    Voice control system for AI DJ.
    Listens for voice commands and executes them.
    """
    
    def __init__(self, ai_dj_module=None):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.ai_dj = ai_dj_module
        self.is_listening = False
        self.callbacks: Dict[CommandType, Callable] = {}
        
    def initialize_microphone(self):
        """Initialize the microphone for voice input."""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Microphone initialized successfully")
            return True
        except OSError as e:
            print(f"Error initializing microphone: {e}")
            return False
    
    def register_callback(self, command_type: CommandType, callback: Callable):
        """Register a callback function for a specific command type."""
        self.callbacks[command_type] = callback
    
    def parse_command(self, text: str) -> VoiceCommand:
        """Parse voice text into a structured command."""
        text = text.lower().strip()
        
        # Try each command type's patterns
        for cmd_type, patterns in COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    params = self._extract_params(cmd_type, match)
                    return VoiceCommand(cmd_type, text, params)
        
        return VoiceCommand(CommandType.UNKNOWN, text)
    
    def _extract_params(self, cmd_type: CommandType, match: re.Match) -> Dict[str, Any]:
        """Extract parameters from regex match based on command type."""
        params = {}
        
        if cmd_type == CommandType.GENERATE:
            params['genre'] = match.group(1).strip() if match.group(1) else None
            params['bpm'] = int(match.group(2)) if match.group(2) else None
            params['key'] = match.group(3).strip() if match.group(3) else None
            
        elif cmd_type == CommandType.ANALYZE:
            params['track'] = match.group(1).strip() if match.group(1) else None
            
        elif cmd_type == CommandType.FUSION:
            params['track1'] = match.group(1).strip() if match.group(1) else None
            params['track2'] = match.group(2).strip() if match.group(2) else None
            
        elif cmd_type == CommandType.VOLUME:
            vol_text = match.group(1).strip() if match.group(1) else "50"
            params['volume'] = self._parse_volume(vol_text)
            
        return params
    
    def _parse_volume(self, text: str) -> int:
        """Parse volume level from text."""
        text = text.lower()
        
        # Handle specific values
        if "max" in text or "full" in text:
            return 100
        if "min" in text or "zero" in text:
            return 0
        
        # Handle percentages
        if "%" in text:
            try:
                return int(text.replace("%", ""))
            except ValueError:
                pass
        
        # Handle numbers
        try:
            val = int(text)
            if val <= 10:
                return val * 10  # Convert 1-10 to 10-100
            return min(100, max(0, val))
        except ValueError:
            pass
        
        return 50  # Default
    
    def execute_command(self, command: VoiceCommand) -> Any:
        """Execute a parsed voice command."""
        if command.command_type == CommandType.UNKNOWN:
            print(f"Didn't understand: '{command.raw_text}'")
            return None
        
        # Check for registered callbacks first
        if command.command_type in self.callbacks:
            return self.callbacks[command.command_type](command)
        
        # Default implementations
        if command.command_type == CommandType.HELP:
            return self._show_help()
        
        elif command.command_type == CommandType.STATUS:
            return self._show_status()
        
        elif command.command_type == CommandType.PLAY:
            print("Playing...")
            return True
        
        elif command.command_type == CommandType.STOP:
            print("Stopping...")
            return True
        
        elif command.command_type == CommandType.PAUSE:
            print("Paused")
            return True
        
        elif command.command_type == CommandType.GENERATE:
            return self._handle_generate(command)
        
        elif command.command_type == CommandType.ANALYZE:
            return self._handle_analyze(command)
        
        elif command.command_type == CommandType.FUSION:
            return self._handle_fusion(command)
        
        return None
    
    def _handle_generate(self, command: VoiceCommand) -> Any:
        """Handle generate command."""
        genre = command.params.get('genre', 'house')
        bpm = command.params.get('bpm', 128)
        key = command.params.get('key', None)
        
        print(f"Generating {genre} track at {bpm} BPM...")
        if self.ai_dj:
            return self.ai_dj.generate(genre=genre, bpm=bpm, key=key)
        return True
    
    def _handle_analyze(self, command: VoiceCommand) -> Any:
        """Handle analyze command."""
        track = command.params.get('track')
        if not track:
            print("Please specify a track to analyze")
            return None
        
        print(f"Analyzing: {track}")
        if self.ai_dj:
            return self.ai_dj.analyze(track)
        return True
    
    def _handle_fusion(self, command: VoiceCommand) -> Any:
        """Handle fusion command."""
        track1 = command.params.get('track1')
        track2 = command.params.get('track2')
        
        if not track1 or not track2:
            print("Please specify two tracks to mix")
            return None
        
        print(f"Fusing {track1} with {track2}...")
        if self.ai_dj:
            return self.ai_dj.fusion(track1, track2)
        return True
    
    def _show_help(self) -> str:
        """Show available voice commands."""
        help_text = """
Available Voice Commands:
  
  Generate:
    • "generate [genre]" or "create [genre]"
    • "generate [genre] at [bpm] bpm"
    • "make a track in [key]"
  
  Analyze:
    • "analyze [track name]"
    • "what is [track name]"
  
  Fusion:
    • "fusion [track1] and [track2]"
    • "mix [track1] with [track2]"
  
  Playback:
    • "play" / "start" / "resume"
    • "stop" / "pause"
  
  Volume:
    • "volume [0-100]"
    • "set volume to [level]"
  
  Info:
    • "help" / "commands"
    • "status" / "now playing"

Examples:
    "generate house at 128 bpm"
    "analyze my_track.mp3"
    "fusion track1.mp3 and track2.mp3"
    "volume 75"
"""
        print(help_text)
        return help_text
    
    def _show_status(self) -> str:
        """Show current status."""
        status = "Ready - No track loaded"
        print(status)
        return status
    
    def listen_once(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for a single voice command."""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            # Use Google's speech recognition (requires internet)
            # Alternative: use offline recognition with whisper
            text = self.recognizer.recognize_google(audio)
            print(f"Heard: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("No speech detected")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
        
        return None
    
    def listen_continuous(self, duration: float = None):
        """Continuously listen for voice commands."""
        self.is_listening = True
        print("Voice control started. Say a command...")
        
        try:
            with self.microphone as source:
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=1.0)
                        text = self.recognizer.recognize_google(audio)
                        print(f"Command: {text}")
                        
                        command = self.parse_command(text)
                        self.execute_command(command)
                        
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                        
        except KeyboardInterrupt:
            print("\nStopping voice control...")
            self.is_listening = False
    
    def process_text_command(self, text: str) -> Any:
        """Process a text-based command (for testing or API)."""
        command = self.parse_command(text)
        return self.execute_command(command)


def create_voice_controller(ai_dj_module=None) -> VoiceController:
    """Factory function to create a voice controller."""
    controller = VoiceController(ai_dj_module)
    controller.initialize_microphone()
    return controller


# CLI entry point
def main():
    parser = argparse.ArgumentParser(
        description='AI DJ Voice Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  voice_control.py --listen          # Start continuous listening
  voice_control.py --command "generate house at 128 bpm"
  voice_control.py --test           # Test with simulated commands
'''
    )
    
    parser.add_argument('--listen', '-l', action='store_true',
                        help='Start continuous voice listening')
    parser.add_argument('--command', '-c', type=str,
                        help='Process a text command')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run test commands')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Timeout for single command (seconds)')
    
    args = parser.parse_args()
    
    # Create controller
    controller = create_voice_controller()
    
    if args.listen:
        controller.listen_continuous()
    
    elif args.command:
        result = controller.process_text_command(args.command)
        print(f"Result: {result}")
    
    elif args.test:
        test_commands = [
            "generate house at 128 bpm",
            "generate techno",
            "analyze my_track.mp3",
            "fusion track1.mp3 and track2.mp3",
            "volume 75",
            "help",
            "status",
        ]
        print("Running test commands...\n")
        for cmd in test_commands:
            print(f"> {cmd}")
            controller.process_text_command(cmd)
            print()
    
    else:
        # Single command mode
        text = controller.listen_once(timeout=args.timeout)
        if text:
            controller.process_text_command(text)


if __name__ == '__main__':
    main()
