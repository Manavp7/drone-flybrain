"""Depth stopping geometry tests; pure arrays are not renderer qualification."""
import unittest

import numpy as np

from experiments.flight_contracts import (
    BRAKING_ACCEL_M_S2, BRAKING_TRANSIENT_MARGIN_M, CameraFrame, FlightState, R_BODY_CAMERA,
    rotation_from_euler,
)
from experiments.flight_safety import DepthGuardian


def make_state(*, yaw=0., pitch=0., speed=0., side=0., vertical=0., time_s=1.,
               position=(0., 0., 1.1)):
    heading = rotation_from_euler(yaw=yaw)
    return FlightState(time_s, np.array(position, float),
                       heading @ np.array([speed, side, vertical]),
                       rotation_from_euler(pitch=pitch, yaw=yaw),
                       np.zeros(3), np.ones(4))


def make_frame(state=None, *, fovy=140., camera_yaw=None, camera_pitch=0.,
               capture_time=1., depth=8., forward_offset=.25):
    if state is None:
        state = make_state()
    camera_yaw = state.yaw if camera_yaw is None else camera_yaw
    body_rotation = rotation_from_euler(pitch=camera_pitch, yaw=camera_yaw)
    optical_rotation = body_rotation @ R_BODY_CAMERA
    optical_position = state.position + body_rotation @ np.array([forward_offset, 0., 0.])
    focal = 101/(2*np.tan(np.deg2rad(fovy/2)))
    return CameraFrame(np.zeros((101, 101, 3), np.uint8),
                       np.full((101, 101), depth, np.float32),
                       (focal, focal, 50., 50.), optical_rotation,
                       optical_position, capture_time)


def replace_frame(frame, **updates):
    data = dict(frame.__dict__)
    data.update(updates)
    return CameraFrame(**data)


def wall_frame(state, distance, *, camera_pitch=0., capture_time=1.):
    """Analytically ray-cast a plane perpendicular to current level heading."""
    frame = make_frame(state, camera_pitch=camera_pitch, capture_time=capture_time)
    fx, fy, cx, cy = frame.intrinsics
    u, v = np.meshgrid(np.arange(101), np.arange(101))
    optical_ray = np.stack(((u-cx)/fx, (v-cy)/fy, np.ones_like(u)), axis=-1)
    normal = rotation_from_euler(yaw=state.yaw)[:, 0]
    directions = optical_ray @ frame.rotation_world_camera.T
    camera_x = np.dot(frame.position_world_camera-state.position, normal)
    denominator = directions @ normal
    optical_z = np.divide(distance-camera_x, denominator,
                          out=np.full_like(denominator, np.nan, dtype=float),
                          where=denominator > 0)
    optical_z[(optical_z <= .01) | (optical_z > 20.)] = np.nan
    return replace_frame(frame, depth_m=optical_z.astype(np.float32))


class DepthGeometryTests(unittest.TestCase):
    def test_wide_camera_clearance_and_speed_cap(self):
        result = DepthGuardian().check(make_frame(), make_state(), 3., 1.)
        self.assertEqual(result['reason'], 'clear')
        self.assertTrue(result['valid_clearance'])
        self.assertEqual(result['forward_speed'], .45)
        self.assertEqual(result['coverage'], 1.)
        self.assertAlmostEqual(result['minimum_clearance_m'], 8.+.25-.52, places=5)
        self.assertGreater(result['metadata']['required_pixels'], 100)
        self.assertEqual(result['metadata']['total_pixels'], 10201)

    def test_world_yaw_translation_and_camera_pitch_backproject_correctly(self):
        state = make_state(yaw=1.17, pitch=.08, speed=.25, position=(12., -8., 1.1))
        result = DepthGuardian().check(wall_frame(state, 2., camera_pitch=.08), state, .3, 1.)
        self.assertEqual(result['reason'], 'clear')
        # Pixel footprints conservatively reduce longitudinal clearance when tilted.
        self.assertGreater(result['minimum_clearance_m'], 1.46)
        self.assertLessEqual(result['minimum_clearance_m'], 1.48)
        self.assertAlmostEqual(result['metadata']['level_velocity_m_s'][0], .25)

    def test_close_plane_brakes_with_nonzero_yaw_and_pitch(self):
        state = make_state(yaw=-2.1, pitch=-.06, speed=.45)
        result = DepthGuardian().check(wall_frame(state, .78, camera_pitch=-.06), state, .45, 1.)
        self.assertEqual(result['reason'], 'blocked_stopping_distance')
        self.assertEqual(result['forward_speed'], 0.)
        self.assertTrue(result['valid_clearance'])
        self.assertLess(result['minimum_clearance_m'], .26)

    def test_one_pixel_thin_pole_cannot_disappear_in_percentiles_or_subsampling(self):
        frame = make_frame()
        frame.depth_m[53, 49] = .50
        result = DepthGuardian().check(frame, make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'blocked_stopping_distance')
        self.assertAlmostEqual(result['minimum_clearance_m'], .23, places=6)

    def test_pixel_footprint_catches_safety_radius_corner(self):
        state = make_state(speed=.45)
        frame = make_frame(capture_time=.9)
        # The center is about y=+.530 m,z=-.530 m, just outside both .52 m
        # bounds. Its complete measured pixel overlaps both safety boundaries.
        frame.depth_m[65, 35] = .65
        result = DepthGuardian().check(frame, state, .45, 1.)
        self.assertEqual(result['reason'], 'blocked_stopping_distance')
        self.assertAlmostEqual(result['minimum_clearance_m'], .38, places=6)

    def test_close_observed_intrusion_inside_current_front_margin_brakes(self):
        frame = make_frame()
        frame.depth_m[50, 50] = .2
        result = DepthGuardian().check(frame, make_state(), 0., 1.)
        self.assertEqual(result['reason'], 'blocked_stopping_distance')
        self.assertEqual(result['minimum_clearance_m'], 0.)

    def test_front_offset_is_included_in_body_clearance(self):
        guardian = DepthGuardian()
        first = guardian.check(make_frame(depth=1., forward_offset=.25), make_state(), .2, 1.)
        second = guardian.check(make_frame(depth=1., forward_offset=.1), make_state(), .2, 1.)
        self.assertAlmostEqual(first['minimum_clearance_m']-second['minimum_clearance_m'], .15)

    def test_narrow_camera_does_not_invent_clearance_at_corridor_edges(self):
        result = DepthGuardian().check(make_frame(fovy=70), make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'unknown_outside_field_of_view')
        self.assertFalse(result['valid_clearance'])

    def test_changed_heading_does_not_treat_old_forward_frame_as_side_sensor(self):
        state = make_state(yaw=np.pi/2)
        result = DepthGuardian().check(make_frame(camera_yaw=0.), state, .45, 1.)
        self.assertIn(result['reason'], ('unknown_corridor_behind_camera', 'unknown_outside_field_of_view'))
        self.assertEqual(result['forward_speed'], 0.)

    def test_reverse_and_lateral_motion_are_not_authorized(self):
        guardian = DepthGuardian()
        self.assertEqual(guardian.check(make_frame(), make_state(), -.1, 1.)['reason'],
                         'unsupported_reverse_request')
        self.assertEqual(guardian.check(make_frame(), make_state(speed=-.1), .2, 1.)['reason'],
                         'unknown_reverse_motion')
        self.assertEqual(guardian.check(make_frame(), make_state(side=.1), .2, 1.)['reason'],
                         'unknown_side_or_vertical_motion')
        self.assertEqual(guardian.check(make_frame(), make_state(vertical=.2), .2, 1.)['reason'],
                         'unknown_side_or_vertical_motion')

    def test_measured_speed_overrides_lower_requested_speed_in_stopping_distance(self):
        state = make_state(speed=.6)
        result = DepthGuardian().check(make_frame(capture_time=.94), state, .1, 1.)
        expected = .6*(.06+.05+.1) + .6**2/(2*BRAKING_ACCEL_M_S2) + BRAKING_TRANSIENT_MARGIN_M
        self.assertAlmostEqual(result['stopping_distance_m'], expected)
        self.assertEqual(result['metadata']['considered_speed_m_s'], .6)

    def test_freespace_is_bounded_by_sensor_range(self):
        frame = make_frame(depth=21.)
        result = DepthGuardian().check(frame, make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'unknown_depth_in_corridor')
        self.assertEqual(result['coverage'], 0.)
        fast = DepthGuardian(max_depth_m=.6).check(make_frame(depth=.6), make_state(speed=2.), .45, 1.)
        self.assertEqual(fast['reason'], 'unknown_beyond_sensor_range')

    def test_stopped_vehicle_still_checks_finite_forward_space(self):
        result = DepthGuardian().check(make_frame(), make_state(), 0., 1.)
        self.assertEqual(result['reason'], 'hold_clear')
        self.assertTrue(result['valid_clearance'])
        self.assertEqual(result['forward_speed'], 0.)

    def test_floor_beyond_each_ray_exit_does_not_block_level_flight(self):
        frame = make_frame()
        _, fy, _, cy = frame.intrinsics
        vertical_ray = (np.arange(101)[:, None]-cy)/fy
        floor_depth = np.divide(1.1, vertical_ray,
                                out=np.full((101, 1), np.inf), where=vertical_ray > 0)
        depth = np.minimum(frame.depth_m, floor_depth).astype(np.float32)
        result = DepthGuardian().check(replace_frame(frame, depth_m=depth), make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'clear')
        self.assertTrue(result['valid_clearance'])

    def test_motor_attitude_transient_margin_applies_even_at_low_measured_speed(self):
        result = DepthGuardian().check(make_frame(), make_state(speed=.0444), 0., 1.)
        # Actual physics development observed .0974 m of drift after this
        # low-speed braking trigger. A velocity-only formula is insufficient.
        self.assertGreater(result['stopping_distance_m'], .15)
        self.assertGreater(result['stopping_distance_m'], .0974)
        self.assertEqual(result['metadata']['braking_transient_margin_m'], .15)

    def test_state_sensor_age_cannot_be_hidden_by_a_newer_depth_frame(self):
        guardian = DepthGuardian()
        current = guardian.check(make_frame(), make_state(speed=.4), .4, 1.)
        older = guardian.check(make_frame(), make_state(speed=.4, time_s=.92), .4, 1.)
        self.assertAlmostEqual(older['stopping_distance_m']-current['stopping_distance_m'], .4*.08)


class DepthValidityTests(unittest.TestCase):
    def test_missing_unregistered_future_stale_depth_brakes(self):
        guardian, state = DepthGuardian(), make_state()
        for frame, expected in (
            (None, 'missing_depth'),
            (replace_frame(make_frame(), registration_verified=False), 'unregistered_depth'),
            (make_frame(capture_time=1.001), 'future_depth'),
            (make_frame(capture_time=.899), 'stale_depth'),
        ):
            with self.subTest(expected=expected):
                result = guardian.check(frame, state, .45, 1.)
                self.assertEqual(result['reason'], expected)
                self.assertEqual(result['forward_speed'], 0.)
                self.assertFalse(result['valid_clearance'])

    def test_invalid_calibration_pose_alignment_and_registration_type(self):
        frame = make_frame()
        updates = (
            {'intrinsics': (0., 20., 50., 50.)},
            {'intrinsics': (20., 20., float('nan'), 50.)},
            {'intrinsics': (20., 20., 120., 50.)},
            {'intrinsics': (20., 20., 50.)},
            {'depth_m': np.ones((100, 101), np.float32)},
            {'depth_m': np.ones((101, 101), np.uint8)},
            {'rotation_world_camera': np.eye(3)*2},
            {'position_world_camera': np.array([0., np.nan, 0.])},
            {'registration_verified': 1},
        )
        for update in updates:
            with self.subTest(update=next(iter(update))):
                result = DepthGuardian().check(replace_frame(frame, **update), make_state(), .45, 1.)
                self.assertEqual(result['reason'], 'invalid_depth_calibration')

    def test_sky_nan_outside_required_corridor_is_acceptable(self):
        frame = make_frame()
        frame.depth_m[:5, :] = np.nan
        result = DepthGuardian().check(frame, make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'clear')
        self.assertEqual(result['coverage'], 1.)

    def test_one_unknown_required_pixel_is_not_smoothed_or_ignored(self):
        for value in (np.nan, np.inf, 0., -1., 21.):
            with self.subTest(value=value):
                frame = make_frame()
                frame.depth_m[51, 49] = value
                result = DepthGuardian().check(frame, make_state(), .45, 1.)
                self.assertEqual(result['reason'], 'unknown_depth_in_corridor')
                self.assertLess(result['coverage'], 1.)
                self.assertGreater(result['coverage'], .99)
                self.assertFalse(result['valid_clearance'])

    def test_unknown_corridor_is_distinguished_even_if_a_second_pixel_sees_obstruction(self):
        frame = make_frame()
        frame.depth_m[50, 50] = .4
        frame.depth_m[51, 51] = np.nan
        result = DepthGuardian().check(frame, make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'unknown_depth_in_corridor')
        self.assertTrue(result['metadata']['observed_blockage'])

    def test_wall_outside_tube_is_clear_once_its_ray_has_left_corridor(self):
        frame = make_frame()
        # Near the projection border, this ray leaves the forward prism early.
        # Its wall is nearer than the farthest prism corner, but does not hide
        # any required space. A global far-depth comparison would fail falsely.
        frame.depth_m[15, 15] = .4
        result = DepthGuardian().check(frame, make_state(), .45, 1.)
        self.assertEqual(result['reason'], 'clear')
        self.assertEqual(result['coverage'], 1.)
        self.assertTrue(result['valid_clearance'])

    def test_invalid_state_request_and_ages_fail_closed(self):
        for state, request, now in (
            (make_state(), np.nan, 1.),
            (make_state(), .1, np.nan),
            (make_state(time_s=1.01), .1, 1.),
            (make_state(time_s=.8), .1, 1.),
            (make_state(position=(np.nan, 0., 1.1)), .1, 1.),
        ):
            with self.subTest(request=request, now=now):
                result = DepthGuardian().check(make_frame(), state, request, now)
                self.assertEqual(result['reason'], 'invalid_state_or_request')
                self.assertEqual(result['forward_speed'], 0.)

    def test_invalid_constructor_configurations_rejected(self):
        for kwargs in ({'braking_accel_m_s2': 0}, {'max_depth_m': 0},
                       {'reaction_allowance_s': -.1}, {'safety_margin_m': -.1},
                       {'vehicle_radius_m': 0}, {'max_depth_m': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DepthGuardian(**kwargs)


if __name__ == '__main__':
    unittest.main()
