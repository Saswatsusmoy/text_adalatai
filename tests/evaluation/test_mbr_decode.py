"""Tests for the MBR pick objective (pure function; no model load)."""

import pytest

from src.evaluation.mbr_decode import mbr_pick


def test_mbr_pick_returns_consensus_candidate():
    consensus = 'the appellant filed a writ petition'
    outlier = 'unrelated sentence with different tokens entirely'
    assert mbr_pick([consensus, consensus, consensus, outlier]) == consensus


def test_mbr_pick_single_candidate_is_identity():
    only = 'only one hypothesis'
    assert mbr_pick([only]) == only


def test_mbr_pick_empty_raises():
    with pytest.raises(ValueError):
        mbr_pick([])
