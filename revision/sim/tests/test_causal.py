"""Legality checker (EVENT_MODEL R-X8; THEORY A-4)."""
import pytest

from sim.causal import check_legal, IllegalSchedule
from sim.tests.test_counting import e7_requests, E7_BRANCH1_FIRST, E7_BRANCH2_FIRST


def test_T_E7_both_orders_are_legal():
    reqs = e7_requests()
    check_legal(reqs, E7_BRANCH1_FIRST)
    check_legal(reqs, E7_BRANCH2_FIRST)


def test_T_E7_child_prefill_before_parent_decode_is_illegal():
    reqs = e7_requests()
    with pytest.raises(IllegalSchedule):
        check_legal(reqs, [("P", "r1"), ("P", "r2"), ("D", "r1"), ("P", "r3"), ("D", "r3")])


def test_decode_before_own_prefill_is_illegal():
    reqs = e7_requests()
    with pytest.raises(IllegalSchedule):
        check_legal(reqs, [("D", "r1"), ("P", "r1"), ("P", "r2"), ("P", "r3"), ("D", "r3")])


def test_every_event_exactly_once():
    reqs = e7_requests()
    with pytest.raises(IllegalSchedule):
        check_legal(reqs, E7_BRANCH1_FIRST[:-1])  # missing r3's decode
    with pytest.raises(IllegalSchedule):
        check_legal(reqs, E7_BRANCH1_FIRST + [("P", "r1")])  # duplicate


def test_request_without_decode_has_no_D_event():
    reqs = e7_requests()  # r2 has empty decode
    with pytest.raises(IllegalSchedule):
        check_legal(reqs, E7_BRANCH1_FIRST + [("D", "r2")])


def test_fine_grained_child_before_parent_completion_is_illegal():
    reqs = e7_requests()
    with pytest.raises(IllegalSchedule):
        check_legal(reqs, [("P", "r1"), ("P", "r2"), ("C", "r1"), ("C", "r2"), ("P", "r3"), ("Db", "r3", 1), ("C", "r3")])


def test_fine_grained_legal_stream_accepted():
    reqs = e7_requests()
    check_legal(reqs, [("P", "r1"), ("C", "r1"), ("P", "r2"), ("C", "r2"), ("P", "r3"), ("Db", "r3", 1), ("C", "r3")])
