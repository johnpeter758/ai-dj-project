from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class DemucsError(RuntimeError):
    pass


def demucs_available() -> bool:
    return shutil.which("demucs") is not None


def separate_stems(audio_path: str | Path, output_dir: str | Path, model: str = "htdemucs") -> dict:
    if not demucs_available():
        raise DemucsError("Demucs is not installed or not on PATH")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "demucs",
        "-n",
        model,
        "-o",
        str(output_dir),
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DemucsError(result.stderr.strip() or "Demucs stem separation failed")

    stem_root = output_dir / model / Path(audio_path).stem
    stem_paths = {
        stem_name: str((stem_root / f"{stem_name}.wav").resolve())
        for stem_name in ["vocals", "drums", "bass", "other"]
        if (stem_root / f"{stem_name}.wav").exists()
    }

    return {
        "provider": "demucs",
        "model": model,
        "output_root": str(stem_root.resolve()) if stem_root.exists() else str(output_dir.resolve()),
        "files": stem_paths,
    }
