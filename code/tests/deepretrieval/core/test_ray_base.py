"""Characterisation tests for pure helpers in verl/single_controller/ray/base.py (no Ray cluster needed)."""
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from verl.single_controller.ray.base import extract_pg_from_exist


def _extract(pools: dict[str, Any], names: list[str], request: Any) -> list:
    """extract_pg_from_exist only reads .get_placement_groups(), .store and .bundle_count: duck-typed fakes suffice."""
    return extract_pg_from_exist(cast(Any, pools), names, cast(Any, request))


@dataclass
class FakePlacementGroup:
    name: str
    bundle_count: int

    def __lt__(self, other: "FakePlacementGroup") -> bool:  # sorted() of (idx, pg) ties never compare pgs
        return self.name < other.name


@dataclass
class FakeResourcePool:
    store: list[int] = field(default_factory=list)
    pgs: list[FakePlacementGroup] = field(default_factory=list)

    def get_placement_groups(self) -> list[FakePlacementGroup]:
        return self.pgs


def test_extract_pg_from_exist_matches_largest_requests_to_largest_groups_in_original_order():
    big, small = FakePlacementGroup("big", 8), FakePlacementGroup("small", 4)
    pools = {"actor": FakeResourcePool(pgs=[small, big]), "critic": FakeResourcePool(pgs=[FakePlacementGroup("x", 9)])}
    request = FakeResourcePool(store=[4, 8])

    result = _extract(pools, ["actor"], request)

    assert result == [small, big]


def test_extract_pg_from_exist_rejects_more_nodes_than_available():
    pools = {"actor": FakeResourcePool(pgs=[FakePlacementGroup("only", 8)])}
    request = FakeResourcePool(store=[4, 4])

    with pytest.raises(AssertionError, match=r"no enough nodes for request: searching 1 th node"):
        _extract(pools, ["actor"], request)


def test_extract_pg_from_exist_rejects_request_larger_than_group():
    pools = {"actor": FakeResourcePool(pgs=[FakePlacementGroup("only", 2)])}
    request = FakeResourcePool(store=[4])

    with pytest.raises(AssertionError, match=r"requesting 4 processes, bundle count cannot satisfy"):
        _extract(pools, ["actor"], request)
