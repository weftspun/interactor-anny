"""Consistency check for `anny/data/phoneme_viseme_facial_action.json`.

Three set-membership assertions run at import so a typo in the JSON fails
CI rather than the lip-sync stage's first render:

    - every blendshape name inside a viseme's weight dict exists in
      FACIAL_ACTION_LABELS (typo would silently produce no motion)
    - every viseme referenced in phonemes' values exists in visemes' keys
      (typo would silently drop the phoneme)
    - every weight sits in [0.0, 1.0]

A negative control per rule fires under `test_self_test_catches_planted_typos`
so a check that passes on known-broken input surfaces its own defect per
CLAUDE.md rule 2 rather than certifying it.
"""
from __future__ import annotations

import json
from pathlib import Path

from anny.models.facial_actions import FACIAL_ACTION_LABELS

DATA = Path(__file__).parent.parent / "src" / "anny" / "data" / "phoneme_viseme_facial_action.json"


def _load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def _find_bad(data: dict) -> list[str]:
    bad = []
    label_set = set(FACIAL_ACTION_LABELS)
    viseme_keys = set(data["visemes"])
    for viseme, weights in data["visemes"].items():
        for label, weight in weights.items():
            if label not in label_set:
                bad.append(f"viseme {viseme!r} names blendshape {label!r} which is not in FACIAL_ACTION_LABELS")
            if not (0.0 <= weight <= 1.0):
                bad.append(f"viseme {viseme!r} label {label!r} weight {weight} out of [0.0, 1.0]")
    for phoneme, viseme in data["phonemes"].items():
        if viseme not in viseme_keys:
            bad.append(f"phoneme {phoneme!r} maps to viseme {viseme!r} which is not in visemes")
    return bad


def test_mapping_is_consistent():
    bad = _find_bad(_load())
    assert not bad, "\n".join(bad)


def test_self_test_catches_planted_typos():
    good = _load()

    typo_label = json.loads(json.dumps(good))
    first_viseme = next(v for v, w in typo_label["visemes"].items() if w)
    first_label = next(iter(typo_label["visemes"][first_viseme]))
    typo_label["visemes"][first_viseme]["typo_not_in_FACIAL_ACTION_LABELS"] = \
        typo_label["visemes"][first_viseme].pop(first_label)
    assert any("not in FACIAL_ACTION_LABELS" in b for b in _find_bad(typo_label))

    typo_viseme = json.loads(json.dumps(good))
    first_phoneme = next(iter(typo_viseme["phonemes"]))
    typo_viseme["phonemes"][first_phoneme] = "typo_not_a_viseme"
    assert any("not in visemes" in b for b in _find_bad(typo_viseme))

    bad_weight = json.loads(json.dumps(good))
    first_viseme = next(v for v, w in bad_weight["visemes"].items() if w)
    first_label = next(iter(bad_weight["visemes"][first_viseme]))
    bad_weight["visemes"][first_viseme][first_label] = 1.5
    assert any("out of [0.0, 1.0]" in b for b in _find_bad(bad_weight))
