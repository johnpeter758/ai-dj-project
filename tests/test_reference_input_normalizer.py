from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reference_input_normalizer import ReferenceInputError, normalize_reference_inputs


def test_normalize_reference_inputs_flattens_csv_and_dedupes(tmp_path: Path):
    ref_a = tmp_path / 'ref_a.json'
    ref_b = tmp_path / 'ref_b.json'
    ref_a.write_text('{}', encoding='utf-8')
    ref_b.write_text('{}', encoding='utf-8')

    normalized = normalize_reference_inputs([
        f' {ref_a} , {ref_b} ',
        str(ref_a),
    ])

    assert normalized == [str(ref_a.resolve()), str(ref_b.resolve())]


def test_normalize_reference_inputs_loads_json_reference_collections_relative_to_file(tmp_path: Path):
    refs_dir = tmp_path / 'refs'
    refs_dir.mkdir()
    ref_a = refs_dir / 'a.json'
    ref_b = refs_dir / 'b.json'
    ref_a.write_text('{}', encoding='utf-8')
    ref_b.write_text('{}', encoding='utf-8')

    collection = tmp_path / 'references.json'
    collection.write_text(json.dumps({'references': ['refs/a.json', 'refs/b.json', 'refs/a.json']}), encoding='utf-8')

    normalized = normalize_reference_inputs([str(collection)])

    assert normalized == [str(ref_a.resolve()), str(ref_b.resolve())]


def test_normalize_reference_inputs_loads_text_reference_collections_and_skips_comments(tmp_path: Path):
    ref_a = tmp_path / 'a.json'
    ref_b = tmp_path / 'b.json'
    ref_a.write_text('{}', encoding='utf-8')
    ref_b.write_text('{}', encoding='utf-8')

    collection = tmp_path / 'references.txt'
    collection.write_text(f'# comment\n{ref_a.name}\n\n{ref_b.name}\n', encoding='utf-8')

    normalized = normalize_reference_inputs([str(collection)])

    assert normalized == [str(ref_a.resolve()), str(ref_b.resolve())]


def test_normalize_reference_inputs_requires_at_least_one_entry():
    with pytest.raises(ReferenceInputError):
        normalize_reference_inputs([])
