"""Tests for generic vowel-interval derivation."""

import pandas as pd

from formant_benchmark.preparation.intervals import derive_vowel_intervals


def test_derive_vowels_preserves_source_intervals_and_is_idempotent() -> None:
    intervals = pd.DataFrame(
        [
            {"interval_id": "p1", "item_id": "i1", "interval_type": "phone", "label": "iy", "start_s": 0.0, "end_s": 0.1, "origin": "source"},
            {"interval_id": "p2", "item_id": "i1", "interval_type": "phone", "label": "t", "start_s": 0.1, "end_s": 0.2, "origin": "source"},
        ]
    )
    first = derive_vowel_intervals(intervals, {"iy"})
    second = derive_vowel_intervals(first, {"iy"})
    assert len(first) == 3
    assert len(second) == 3
    derived = first[first["interval_type"] == "vowel"].iloc[0]
    assert derived["interval_id"] == "p1:vowel"
    assert derived["origin"] == "derived"
