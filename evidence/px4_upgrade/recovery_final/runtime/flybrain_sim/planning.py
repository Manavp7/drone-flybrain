"""Finite, deterministic 3-D visibility planning for known box geometry.

This is a geometric planner for the reduced-order simulator. It is not an
aircraft trajectory optimizer, visual mapper, or flight-controller interface.
"""
from __future__ import annotations

import heapq
import math

from .contracts import Box, Scenario, Vec3
from .geometry import distance, segment_intersects_box, within_bounds


def finite_vec(value: Vec3) -> bool:
    return len(value) == 3 and all(math.isfinite(v) for v in value)


def projected_boxes(boxes: tuple[Box, ...], age: float,
                    horizon: float = 0.8) -> tuple[Box, ...]:
    """Conservative constant-velocity swept boxes from observed motion only.

    Dynamic-motion prediction is deliberately bounded in time. It cannot model
    unknown acceleration or guarantee the behaviour of a physical obstacle.
    """
    result = []
    for box in boxes:
        if box.velocity == (0.0, 0.0, 0.0):
            result.append(box)
            continue
        start, end = max(0.0, age), max(0.0, age) + horizon
        low = tuple(box.low[i] + min(box.velocity[i] * start,
                                    box.velocity[i] * end) for i in range(3))
        high = tuple(box.high[i] + max(box.velocity[i] * start,
                                      box.velocity[i] * end) for i in range(3))
        result.append(Box(box.id, low, high, box.velocity))
    return tuple(result)


class VisibilityPlanner:
    """A* on box corners and free-space portals, with a finite node budget.

    A portal is inferred only from mapped box faces. Pairwise gaps supply their
    own altitude and approach points, so an opening need not share the home's
    altitude. No mission-specific intermediate waypoints are needed.
    """

    MAX_MAP_BOXES = 128
    MAX_GRAPH_NODES = 512

    def __init__(self, scenario: Scenario, radius: float = 0.45,
                 margin: float = 0.5) -> None:
        self.bounds = scenario.bounds
        self.radius = radius
        self.clearance = radius + margin
        self.recovery_clearance = radius + 0.12 + math.sqrt(3.0) * scenario.sensor_noise
        self.preferred_z = scenario.home[2]
        self.static = tuple(box for box in scenario.obstacles
                            if box.velocity == (0.0, 0.0, 0.0))
        self._map_over_budget = len(self.static) > self.MAX_MAP_BOXES
        self._static_nodes = () if self._map_over_budget else tuple(
            point for point in sorted(set(self._corners(self.static))
                                      | set(self._portals(self.static)))
            if self.segment_clear(point, point)
        )
        self._static_visibility: dict[tuple[Vec3, Vec3], bool] = {}
        self.plans = 0

    def _corners(self, boxes: tuple[Box, ...]) -> tuple[Vec3, ...]:
        candidates: set[Vec3] = set()
        padding = self.clearance + 0.65
        for box in boxes:
            zs = (self.preferred_z, box.low[2] - padding,
                  box.high[2] + padding)
            for x in (box.low[0] - padding, box.high[0] + padding):
                for y in (box.low[1] - padding, box.high[1] + padding):
                    for z in zs:
                        p = (x, y, z)
                        if within_bounds(p, self.bounds, self.clearance):
                            candidates.add(p)
        return tuple(sorted(candidates))

    def _portals(self, boxes: tuple[Box, ...],
                 require_ids: set[str] | None = None) -> tuple[Vec3, ...]:
        """Infer points in gaps between opposite faces of mapped AABBs.

        For each separated pair, use the gap midpoint and the overlapping
        extents on the other two axes. Their midpoints describe the opening;
        points beyond their ends let the graph approach and leave it before
        turning. Apply the same construction on all three axes. Candidates
        blocked by any third box are discarded by the usual clearance check.

        This is still a finite candidate graph, not a completeness proof for
        arbitrary AABB arrangements. Each pair emits at most nine points per
        axis, and the existing 512-node search budget remains in force.
        """
        candidates: set[Vec3] = set()
        padding = self.clearance + 0.65
        for index, left in enumerate(boxes):
            for right in boxes[index + 1:]:
                if require_ids is not None and not ({left.id, right.id} & require_ids):
                    continue
                for axis in range(3):
                    first, second = ((left, right) if left.low[axis] <= right.low[axis]
                                     else (right, left))
                    if second.low[axis] - first.high[axis] <= 2.0 * self.clearance:
                        continue
                    others = [other for other in range(3) if other != axis]
                    overlap = [(max(left.low[other], right.low[other]),
                                min(left.high[other], right.high[other]))
                               for other in others]
                    if any(high <= low for low, high in overlap):
                        continue
                    position = [0.0, 0.0, 0.0]
                    position[axis] = (first.high[axis] + second.low[axis]) / 2.0
                    first_low, first_high = overlap[0]
                    second_low, second_high = overlap[1]
                    for a in (first_low - padding, (first_low + first_high) / 2.0,
                              first_high + padding):
                        for b in (second_low - padding, (second_low + second_high) / 2.0,
                                  second_high + padding):
                            position[others[0]], position[others[1]] = a, b
                            point = tuple(position)
                            if within_bounds(point, self.bounds, self.clearance):
                                candidates.add(point)
        return tuple(sorted(candidates))

    def segment_clear(self, a: Vec3, b: Vec3,
                      extra: tuple[Box, ...] = (),
                      clearance: float | None = None) -> bool:
        radius = self.clearance if clearance is None else clearance
        if not finite_vec(a) or not finite_vec(b):
            return False
        if not within_bounds(a, self.bounds, self.radius):
            return False
        if not within_bounds(b, self.bounds, self.radius):
            return False
        return not any(segment_intersects_box(a, b, box, radius)
                       for box in self.static + extra)

    def _visible(self, a: Vec3, b: Vec3,
                 extra: tuple[Box, ...]) -> bool:
        key = (a, b) if a < b else (b, a)
        visible = self._static_visibility.get(key)
        if visible is None:
            visible = self.segment_clear(a, b)
            # Bound persistent cache size even under repeated replanning.
            if len(self._static_visibility) < 30000:
                self._static_visibility[key] = visible
        return visible and not any(segment_intersects_box(a, b, box, self.clearance)
                                   for box in extra)

    def plan(self, start: Vec3, goal: Vec3,
             observed_boxes: tuple[Box, ...] = ()) -> list[Vec3] | None:
        """Return points after start, including goal; None means no found route.

        A failure means this finite candidate graph found no path. It does not
        establish geometric impossibility for every possible trajectory.
        """
        self.plans += 1
        if not finite_vec(start) or not finite_vec(goal):
            return None
        if self._map_over_budget:
            return None
        known_ids = {box.id for box in self.static}
        extra = tuple(box for box in observed_boxes if box.id not in known_ids)
        if len(self.static) + len(extra) > self.MAX_MAP_BOXES:
            return None
        if self.segment_clear(start, goal, extra):
            return [goal]
        # A measured start can enter the discretionary planning margin through
        # noise or wind. Permit only its initial edge to use the smaller guard
        # margin, so recovery can move back into ordinarily cleared space.
        start_margin = self.clearance
        if not self.segment_clear(start, start, extra):
            start_margin = self.recovery_clearance
        if not self.segment_clear(start, start, extra, start_margin):
            return None
        goal_margin = self.clearance
        if not self.segment_clear(goal, goal, extra):
            goal_margin = self.recovery_clearance
        if not self.segment_clear(goal, goal, extra, goal_margin):
            return None
        candidates = self._static_nodes + self._corners(extra)
        if extra:
            candidates += self._portals(self.static + extra,
                                        require_ids={box.id for box in extra})
        nodes = [start, goal]
        for point in dict.fromkeys(candidates):
            if point not in (start, goal) and self.segment_clear(point, point, extra):
                nodes.append(point)
        if len(nodes) > self.MAX_GRAPH_NODES:
            # Dynamic augmentation must not erase a usable static roadmap.
            # Keep all surviving static nodes and allocate only the remaining
            # slots to dynamic candidates, ranked by start-to-goal detour.
            # This limits candidate points, never observed collision geometry:
            # every edge still checks every extra box below.
            static_set = set(self._static_nodes)
            static_nodes = [point for point in nodes[2:] if point in static_set]
            if len(static_nodes) + 2 > self.MAX_GRAPH_NODES:
                return None
            dynamic_nodes = [point for point in nodes[2:] if point not in static_set]
            dynamic_nodes.sort(key=lambda point: (distance(start, point)
                                                 + distance(point, goal), point))
            remaining = self.MAX_GRAPH_NODES - len(static_nodes) - 2
            nodes = [start, goal] + static_nodes + dynamic_nodes[:remaining]
        costs = {0: 0.0}
        previous: dict[int, int] = {}
        queue = [(distance(start, goal), 0.0, 0)]
        visited: set[int] = set()
        while queue and len(visited) < self.MAX_GRAPH_NODES:
            _, cost, index = heapq.heappop(queue)
            if index in visited:
                continue
            if index == 1:
                result = [goal]
                while index != 0:
                    index = previous[index]
                    if index:
                        result.append(nodes[index])
                return list(reversed(result))
            visited.add(index)
            for other in range(1, len(nodes)):
                if other in visited or other == index:
                    continue
                candidate = cost + distance(nodes[index], nodes[other])
                if candidate >= costs.get(other, math.inf):
                    continue
                edge_margin = self.clearance
                if index == 0:
                    edge_margin = min(edge_margin, start_margin)
                if other == 1:
                    edge_margin = min(edge_margin, goal_margin)
                visible = (self.segment_clear(nodes[index], nodes[other], extra, edge_margin)
                           if edge_margin < self.clearance
                           else self._visible(nodes[index], nodes[other], extra))
                if visible:
                    costs[other] = candidate
                    previous[other] = index
                    priority = candidate + distance(nodes[other], goal)
                    heapq.heappush(queue, (priority, candidate, other))
        return None
