from pathlib import Path

from scripts import two_input_one_output


def test_build_fusion_command_includes_expected_flags(tmp_path: Path):
    repo_root = tmp_path / "repo"
    cmd = two_input_one_output.build_fusion_command(
        repo_root=repo_root,
        song_a=tmp_path / "a.mp3",
        song_b=tmp_path / "b.mp3",
        output_dir=tmp_path / "out",
        arrangement_mode="pro",
    )

    assert cmd[1].endswith("ai_dj.py")
    assert "fusion" in cmd
    assert "--output" in cmd
    assert "--arrangement-mode" in cmd
    assert cmd[-1] == "pro"
