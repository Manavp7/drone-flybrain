"""Conservative, simulation-only stopping gate for registered optical-Z depth.

This gate accepts sensor measurements and an estimated state, never scene truth.
It authorizes only a nonnegative speed along the current level heading. The
expanded forward prism must be observable; a camera cannot certify unseen sides
or the rear. The current physical footprint is assumed initially unoccupied,
but any measured intrusion into it also causes a stop.
"""
from __future__ import annotations

import numpy as np

from experiments.flight_contracts import (
    BRAKING_ACCEL_M_S2, BRAKING_TRANSIENT_MARGIN_M, DEPTH_MAX_AGE_S,
    DEPTH_PERIOD_S, MAX_SPEED_M_S,
    SAFETY_MARGIN_M, VEHICLE_RADIUS_M, CameraFrame, FlightState,
)


def _convex_hull(points):
    """Small monotone-chain hull; avoids a renderer or geometry dependency."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


class DepthGuardian:
    """Return a brake unless calibrated current depth certifies stopping space.

    ``minimum_clearance_m`` is longitudinal free travel from the expanded body
    front to the nearest *observed* intrusion, conservatively enlarged by a
    pixel's angular footprint. It is not a globally mapped obstacle distance.
    ``valid_clearance`` means calibrated finite measurements support the
    decision; it can be true when a measured obstruction requires braking.
    """

    def __init__(self, *, braking_accel_m_s2=BRAKING_ACCEL_M_S2,
                 reaction_allowance_s=.10, max_depth_m=20.,
                 braking_transient_margin_m=BRAKING_TRANSIENT_MARGIN_M,
                 vehicle_radius_m=VEHICLE_RADIUS_M,
                 safety_margin_m=SAFETY_MARGIN_M):
        values = [braking_accel_m_s2, reaction_allowance_s, max_depth_m,
                  braking_transient_margin_m, vehicle_radius_m, safety_margin_m]
        if (not np.isfinite(values).all() or braking_accel_m_s2 <= 0
                or reaction_allowance_s < 0 or max_depth_m <= 0
                or braking_transient_margin_m < 0
                or vehicle_radius_m <= 0 or safety_margin_m < 0):
            raise ValueError('Invalid depth stopping configuration')
        self.braking_accel_m_s2 = float(braking_accel_m_s2)
        self.reaction_allowance_s = float(reaction_allowance_s)
        self.max_depth_m = float(max_depth_m)
        self.braking_transient_margin_m = float(braking_transient_margin_m)
        self.radius_m = float(vehicle_radius_m + safety_margin_m)
        self._ray_cache = {}

    def _rays(self, shape, intrinsics):
        key = (tuple(shape), tuple(intrinsics))
        if key not in self._ray_cache:
            h, w = shape
            fx, fy, cx, cy = intrinsics
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            # Keep only the most recent calibration to bound cache growth.
            self._ray_cache = {key: (u, v, (u-cx)/fx, (v-cy)/fy)}
        return self._ray_cache[key]

    def check(self, frame: CameraFrame | None, state: FlightState,
              requested_speed: float, now_s: float) -> dict:
        result = {
            'forward_speed': 0., 'reason': 'unknown_depth',
            'valid_clearance': False, 'minimum_clearance_m': None,
            'stopping_distance_m': None, 'coverage': 0.,
            'metadata': {
                'mode': 'registered_depth_forward_stop_only',
                'braking_accel_m_s2': self.braking_accel_m_s2,
                'braking_transient_margin_m': self.braking_transient_margin_m,
                'reaction_allowance_s': self.reaction_allowance_s,
                'sensor_period_s': DEPTH_PERIOD_S,
                'max_depth_age_s': DEPTH_MAX_AGE_S,
                'expanded_radius_m': self.radius_m,
                'max_sensor_optical_depth_m': self.max_depth_m,
                'allowed_reverse_or_side_drift_m_s': .03,
                'allowed_vertical_drift_m_s': .08,
                'all_pixels_inspected': True, 'scene_truth_used': False,
                'current_footprint_assumed_initially_free': True,
                'clearance_scope': 'observed forward prism; no sides or rear authority',
            },
        }
        metadata = result['metadata']
        def stop(reason):
            result['reason'] = reason
            return result

        try:
            request = float(requested_speed)
            now = float(now_s)
            position = np.asarray(state.position, dtype=float)
            velocity = np.asarray(state.velocity, dtype=float)
            rotation = np.asarray(state.rotation, dtype=float)
            state_time = float(state.time_s)
        except (AttributeError, TypeError, ValueError):
            return stop('invalid_state_or_request')
        if (not np.isfinite([request, now, state_time]).all() or now < 0
                or position.shape != (3,) or velocity.shape != (3,)
                or rotation.shape != (3, 3)
                or not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or abs(np.linalg.det(rotation)-1) > 1e-6
                or state_time > now + 1e-9 or now-state_time > DEPTH_MAX_AGE_S + 1e-9):
            return stop('invalid_state_or_request')
        if request < 0:
            return stop('unsupported_reverse_request')
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        heading = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        level_velocity = heading.T @ velocity
        metadata['level_velocity_m_s'] = level_velocity.tolist()
        state_age = max(0., now-state_time)
        metadata['state_age_s'] = state_age
        # Braking means request zero to the stabilizer; this gate never tries to
        # counteract a blind rear/side drift by issuing another direction.
        if level_velocity[0] < -.03:
            return stop('unknown_reverse_motion')
        if abs(level_velocity[1]) > .03 or abs(level_velocity[2]) > .08:
            return stop('unknown_side_or_vertical_motion')
        if frame is None:
            return stop('missing_depth')
        try:
            frame.validate()
        except (AttributeError, TypeError, ValueError):
            return stop('invalid_depth_calibration')
        if not frame.registration_verified:
            return stop('unregistered_depth')
        age = now - float(frame.capture_time_s)
        metadata['depth_age_s'] = float(age)
        if age < -1e-9:
            return stop('future_depth')
        if age > DEPTH_MAX_AGE_S + 1e-9:
            return stop('stale_depth')
        age = max(age, 0.)
        request = min(request, MAX_SPEED_M_S)
        speed = max(request, float(np.linalg.norm(velocity)))
        sensing_age = max(age, state_age)
        reaction_s = sensing_age + DEPTH_PERIOD_S + self.reaction_allowance_s
        stopping = (speed * reaction_s + speed*speed/(2*self.braking_accel_m_s2)
                    + self.braking_transient_margin_m)
        result['stopping_distance_m'] = float(stopping)
        metadata.update({'considered_speed_m_s': speed,
                         'total_reaction_s': reaction_s,
                         'requested_speed_clamped_m_s': request,
                         'stopping_formula': 'speed*(max(depth_age,state_age)+sensor_period+reaction_allowance)+speed^2/(2*braking_accel)+braking_transient_margin'})

        # A small finite prism is still checked while hovering. Radius includes
        # both vehicle hull and the declared safety margin, never just center ray.
        radius = self.radius_m
        horizon = radius + max(stopping, .05)
        corners = np.array([[x, y, z] for x in (radius, horizon)
                            for y in (-radius, radius) for z in (-radius, radius)])
        camera_rotation = np.asarray(frame.rotation_world_camera)
        camera_position = np.asarray(frame.position_world_camera)
        world_corners = corners @ heading.T + position
        optical_corners = (world_corners - camera_position) @ camera_rotation
        metadata['required_forward_interval_m'] = [radius, float(horizon)]
        metadata['camera_position_in_current_heading_m'] = ((camera_position-position) @ heading).tolist()
        if np.min(optical_corners[:, 2]) <= .01:
            return stop('unknown_corridor_behind_camera')
        fx, fy, cx, cy = map(float, frame.intrinsics)
        projected = optical_corners[:, :2] / optical_corners[:, 2:]
        projected[:, 0] = projected[:, 0]*fx + cx
        projected[:, 1] = projected[:, 1]*fy + cy
        h, w = frame.depth_m.shape
        if (np.any(projected[:, 0] < 0) or np.any(projected[:, 0] > w-1)
                or np.any(projected[:, 1] < 0) or np.any(projected[:, 1] > h-1)):
            return stop('unknown_outside_field_of_view')
        max_required_z = float(np.max(optical_corners[:, 2]))
        metadata['required_max_optical_z_m'] = max_required_z
        if max_required_z > self.max_depth_m:
            return stop('unknown_beyond_sensor_range')

        u, v, ray_x, ray_y = self._rays((h, w), frame.intrinsics)
        hull = _convex_hull(projected)
        required = np.ones((h, w), dtype=bool)
        # Test overlap with each complete pixel square, not only its center.
        # Expanding each hull half-plane by half a pixel is conservative at
        # polygon corners; it can reject extra pixels but cannot omit an edge.
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            dx, dy = b-a
            signed = dx*(v-a[1]) - dy*(u-a[0])
            required &= signed >= -.5*(abs(dx)+abs(dy)) - 1e-9
        required &= ((u+.5 >= projected[:, 0].min())
                     & (u-.5 <= projected[:, 0].max())
                     & (v+.5 >= projected[:, 1].min())
                     & (v-.5 <= projected[:, 1].max()))
        transform = heading.T @ camera_rotation
        origin = heading.T @ (camera_position-position)
        directions = [transform[axis, 0]*ray_x + transform[axis, 1]*ray_y
                      + transform[axis, 2] for axis in range(3)]
        footprint_per_depth = np.array([
            abs(transform[axis, 0])/(2*fx) + abs(transform[axis, 1])/(2*fy)
            for axis in range(3)])
        # Bound each pixel's entire cone by intersecting its center ray with
        # a prism enlarged by the maximum possible half-pixel footprint.
        # This supplies a per-pixel exit depth. A floor or wall beyond that
        # exit is acceptable even when nearer than another ray's far corner.
        lower = np.array([radius, -radius, -radius]) - max_required_z*footprint_per_depth
        upper = np.array([horizon, radius, radius]) + max_required_z*footprint_per_depth
        entry = np.full((h, w), -np.inf)
        leave = np.full((h, w), np.inf)
        possible = np.ones((h, w), dtype=bool)
        for axis, direction in enumerate(directions):
            nonparallel = abs(direction) > 1e-12
            lo = np.divide(lower[axis]-origin[axis], direction,
                           out=np.full((h, w), -np.inf), where=nonparallel)
            hi = np.divide(upper[axis]-origin[axis], direction,
                           out=np.full((h, w), np.inf), where=nonparallel)
            entry = np.maximum(entry, np.minimum(lo, hi))
            leave = np.minimum(leave, np.maximum(lo, hi))
            possible &= nonparallel | ((lower[axis] <= origin[axis]) & (origin[axis] <= upper[axis]))
        required &= possible & (leave >= np.maximum(entry, 0.))
        required_exit_depth = np.minimum(leave, max_required_z)
        required_count = int(required.sum())
        if required_count == 0:
            return stop('unknown_empty_corridor_projection')
        depth = np.asarray(frame.depth_m, dtype=float)
        valid = np.isfinite(depth) & (depth > .01) & (depth <= self.max_depth_m + 1e-6)
        covered_count = int((required & valid).sum())
        result['coverage'] = float(covered_count/required_count)
        metadata.update({'required_pixels': required_count,
                         'valid_required_pixels': covered_count,
                         'total_pixels': int(depth.size)})

        # Backproject EVERY finite depth pixel through its capture pose. The
        # half-pixel angular bounds also catch sampled poles/corners whose ray
        # center falls just outside the expanded body cross-section.
        finite_depth = np.where(valid, depth, 0.)
        coordinates, uncertainty = [], []
        for axis in range(3):
            coordinates.append(origin[axis] + finite_depth*directions[axis])
            uncertainty.append(finite_depth*footprint_per_depth[axis])
        px, py, pz = coordinates
        ux, uy, uz = uncertainty
        intersects = (valid & (abs(py)-uy <= radius) & (abs(pz)-uz <= radius)
                      & (px+ux >= -radius))
        if intersects.any():
            clearance = max(0., float(np.min(px[intersects]-ux[intersects])-radius))
            result['minimum_clearance_m'] = clearance
        else:
            # No observed surface in the tube is not an infinite-range claim.
            clearance = float(max(stopping, .05))
            result['minimum_clearance_m'] = clearance
        metadata['observed_tube_pixels'] = int(intersects.sum())
        blocked = bool(intersects.any() and clearance <= stopping + 1e-6)
        metadata['observed_blockage'] = blocked
        if covered_count != required_count:
            return stop('unknown_depth_in_corridor')
        if blocked:
            result['valid_clearance'] = True
            return stop('blocked_stopping_distance')
        # Finite foreground may still hide part of a required pixel cone.
        reaches_far_face = depth[required] >= required_exit_depth[required] + 1e-6
        if not reaches_far_face.all():
            metadata['occluded_required_pixels'] = int((~reaches_far_face).sum())
            return stop('unknown_occluded_corridor')
        result['valid_clearance'] = True
        result['forward_speed'] = float(request)
        result['reason'] = 'clear' if request > 0 else 'hold_clear'
        return result
