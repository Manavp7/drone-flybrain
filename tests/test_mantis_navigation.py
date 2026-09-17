"""Sensor-only route proposals against independent ray-cast box depth."""
from dataclasses import replace
import os
import unittest

import numpy as np

from experiments.flight_contracts import (CameraFrame, FlightState, R_BODY_CAMERA,
                                          rotation_from_euler, wrap_angle)
from experiments.mantis_navigation import DetourNavigator


def state_at(time_s=0., position=(0., 0., 1.1), yaw=0., speed=0.):
    rotation = rotation_from_euler(yaw=yaw)
    return FlightState(time_s, np.asarray(position, float), rotation[:, 0]*speed,
                       rotation, np.zeros(3), np.ones(4))


def sensor_frame(state, obstacle=(2., .65, .7), size=(.1, .25, .7)):
    """Independent ray/AABB slab intersections, with finite far-wall depth."""
    focal = 192/(2*np.tan(np.deg2rad(75.)))
    origin = state.position + state.rotation @ np.array([.25, 0., 0.])
    rotation = state.rotation @ R_BODY_CAMERA
    u, v = np.meshgrid(np.arange(192), np.arange(192))
    optical = np.stack([(u-95.5)/focal, (v-95.5)/focal, np.ones_like(u)], axis=-1)
    directions = optical @ rotation.T
    depth = np.full((192, 192), 8., np.float32)
    if obstacle is not None:
        near, far = np.full((192, 192), -np.inf), np.full((192, 192), np.inf)
        possible = np.ones((192, 192), bool)
        for axis in range(3):
            minimum, maximum = obstacle[axis]-size[axis], obstacle[axis]+size[axis]
            ray = directions[..., axis]
            active = np.abs(ray) > 1e-12
            a = np.divide(minimum-origin[axis], ray, out=np.full_like(ray, -np.inf), where=active)
            b = np.divide(maximum-origin[axis], ray, out=np.full_like(ray, np.inf), where=active)
            near, far = np.maximum(near, np.minimum(a, b)), np.minimum(far, np.maximum(a, b))
            possible &= active | (minimum <= origin[axis] <= maximum)
        hits = possible & (near > .01) & (near <= far) & (near < depth)
        depth[hits] = near[hits]
    return CameraFrame(np.zeros((192, 192, 3), np.uint8), depth,
                       (focal, focal, 95.5, 95.5), rotation, origin, state.time_s)


def request(sequence=0, yaw=0., speed=.3, **extras):
    return dict(sequence=sequence, forward_speed=speed, range_speed=speed, yaw_target=yaw, **extras)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.navigator = DetourNavigator()
        self.state = state_at()

    def test_clear_route_keeps_selected_command_and_final_guardian(self):
        result = self.navigator.update(sensor_frame(self.state, None), self.state, request())
        self.assertEqual(result['phase'], 'following')
        self.assertEqual(result['forward_speed'], .3)
        self.assertEqual(result['guardian']['reason'], 'clear')
        self.assertFalse(result['scene_truth_used'])

    def test_offset_obstruction_produces_observed_alternative_not_motion(self):
        result = self.navigator.update(sensor_frame(self.state), self.state, request())
        self.assertEqual(result['phase'], 'turning', result)
        self.assertEqual(result['forward_speed'], 0.)
        self.assertLess(result['yaw_target'], 0.)
        self.assertTrue(any(c['observed_clear'] for c in result['candidates']))
        self.assertTrue(any(not c['observed_clear'] for c in result['candidates']))

    def test_moving_aircraft_brakes_before_turning(self):
        moving = state_at(speed=.2)
        result = self.navigator.update(sensor_frame(moving), moving, request())
        self.assertEqual(result['phase'], 'braking')
        self.assertEqual(result['forward_speed'], 0.)
        self.assertEqual(result['yaw_target'], moving.yaw)
        self.assertEqual(result['candidates'], [])

    def test_selected_route_rechecked_before_forward_motion(self):
        planned = self.navigator.update(sensor_frame(self.state), self.state, request())
        turned = state_at(.1, yaw=planned['yaw_target'])
        moving = self.navigator.update(sensor_frame(turned), turned, request(1))
        self.assertEqual(moving['phase'], 'passing', moving)
        self.assertGreater(moving['forward_speed'], 0.)
        obstructed = sensor_frame(turned, (1., -.2, 1.1), (.1, .2, .5))
        blocked = self.navigator.update(obstructed, turned, request(1))
        self.assertEqual(blocked['phase'], 'holding')
        self.assertEqual(blocked['forward_speed'], 0.)

    def test_centered_obstacle_correctly_has_no_route(self):
        result = self.navigator.update(sensor_frame(self.state, (2., 0., .7)), self.state, request())
        self.assertEqual(result['phase'], 'holding')
        self.assertEqual(result['reason'], 'no_observed_detour')
        self.assertEqual(result['forward_speed'], 0.)
        self.assertFalse(any(c['observed_clear'] for c in result['candidates']))

    def test_missing_stale_and_nan_depth_never_move_or_turn(self):
        good = sensor_frame(self.state, None)
        bad = replace(good, depth_m=np.full_like(good.depth_m, np.nan))
        old = replace(good, capture_time_s=0.)
        for frame, state in [(None, self.state), (bad, self.state), (old, state_at(.3))]:
            with self.subTest(frame=frame is None, time=state.time_s):
                output = DetourNavigator().update(frame, state, request())
                self.assertEqual(output['forward_speed'], 0.)
                self.assertEqual(output['yaw_target'], state.yaw)

    def test_no_selected_person_or_expired_selection_aborts_plan(self):
        self.navigator.update(sensor_frame(self.state), self.state, request())
        for selected in [request(None), request(0, valid_until_s=0.),
                         request(0, capture_time_s=1.), request(0, valid=False)]:
            result = self.navigator.update(sensor_frame(self.state), self.state, selected)
            self.assertEqual(result['forward_speed'], 0.)
            self.assertEqual(result['yaw_target'], self.state.yaw)

    def test_selected_bearing_leaving_envelope_stops_detour(self):
        self.navigator.update(sensor_frame(self.state), self.state, request())
        output = self.navigator.update(sensor_frame(self.state), self.state, request(yaw=.25))
        self.assertEqual(output['phase'], 'holding')
        self.assertEqual(output['reason'], 'selected_target_left_detour_view')

    def test_guidance_gaps_cannot_renew_detour_budget_or_unlock_failed_plan(self):
        self.navigator.update(sensor_frame(self.state), self.state, request())
        origin = self.navigator.origin.copy()
        for selected in [request(None), request(3, valid_until_s=.5)]:
            waiting = state_at(1.)
            output = self.navigator.update(sensor_frame(waiting), waiting, selected)
            self.assertEqual(output['forward_speed'], 0.)
            self.assertEqual(self.navigator.started_s, 0.)
            np.testing.assert_array_equal(self.navigator.origin, origin)
        expired = state_at(15.)
        self.navigator.update(sensor_frame(expired), expired, request(None))
        output = self.navigator.update(sensor_frame(expired), expired, request(4))
        self.assertEqual(output['reason'], 'detour_time_limit')
        self.assertEqual(output['forward_speed'], 0.)

        failed = DetourNavigator()
        blocked = sensor_frame(self.state, (2., 0., .7))
        failed.update(blocked, self.state, request())
        failed.update(sensor_frame(state_at(.1), None), state_at(.1), request(None))
        output = failed.update(sensor_frame(state_at(.2), None), state_at(.2), request(2))
        self.assertEqual(output['phase'], 'holding')
        self.assertEqual(output['reason'], 'no_observed_detour')
        self.assertEqual(output['forward_speed'], 0.)

    def test_rejoin_stops_then_turns_before_counting_completion(self):
        plan = self.navigator.update(sensor_frame(self.state), self.state, request())
        yaw = plan['yaw_target']
        # Clear current image after travelling along the observed heading.
        p = np.array([np.cos(yaw)*.6, np.sin(yaw)*.6, 1.1])
        moving = state_at(2., p, yaw, .2)
        output = self.navigator.update(sensor_frame(moving, None), moving, request(2))
        self.assertEqual(output['phase'], 'turning')  # Physical turn was never confirmed.
        self.assertEqual(output['forward_speed'], 0.)
        stopped = state_at(2.1, p, yaw)
        output = self.navigator.update(sensor_frame(stopped, None), stopped, request(3))
        self.assertEqual(output['phase'], 'rejoining')
        self.assertEqual(output['forward_speed'], 0.)
        output = self.navigator.update(sensor_frame(stopped, None), stopped, request(3))
        self.assertEqual(output['reason'], 'turn_to_selected_target')
        self.assertEqual(output['detours_completed'], 0)
        facing = state_at(2.2, p, 0.)
        output = self.navigator.update(sensor_frame(facing, None), facing, request(4))
        self.assertEqual(output['reason'], 'detour_complete')
        self.assertEqual(output['detours_completed'], 1)
        self.assertEqual(output['forward_speed'], 0.)

    def test_gradual_bearing_change_brakes_before_the_hard_view_limit(self):
        planned = self.navigator.update(sensor_frame(self.state), self.state, request())
        yaw = planned['yaw_target']
        turned = state_at(.1, yaw=yaw)
        self.navigator.update(sensor_frame(turned), turned, request(1))
        moving = state_at(.3, (.1, -.02, 1.1), yaw, .2)
        result = self.navigator.update(sensor_frame(moving), moving, request(2, yaw+.24))
        self.assertEqual(result['phase'], 'rejoining')
        self.assertEqual(result['forward_speed'], 0.)
        self.assertEqual(result['yaw_target'], moving.yaw)
        self.assertEqual(result['detours_completed'], 0)

    def test_unknown_rejoin_view_does_not_interrupt_a_certified_corridor(self):
        plan = self.navigator.update(sensor_frame(self.state), self.state, request())
        yaw = plan['yaw_target']
        turned = state_at(.1, yaw=yaw)
        self.navigator.update(sensor_frame(turned), turned, request(1))
        moving = state_at(2., (.60, -.11, 1.1), yaw, .3)
        frame = sensor_frame(moving)
        bearing = .04
        inspection = self.navigator._probe_heading(
            frame, replace(moving, velocity=np.zeros(3)), bearing, .3)
        self.assertEqual(inspection['reason'], 'unknown_outside_field_of_view')
        result = self.navigator.update(frame, moving, request(2, bearing))
        self.assertEqual(result['phase'], 'passing', result)
        self.assertEqual(result['reason'], 'observed_detour')
        self.assertGreater(result['forward_speed'], 0.)
        self.assertEqual(result['detours_completed'], 0)
        self.assertEqual(self.navigator.started_s, 0.)

    def test_finite_braking_and_turning_resume_and_physically_pass_fixed_barrier(self):
        # An independent sensor fixture with finite acceleration/braking and
        # yaw rate exposes the cost of unnecessary stop/inspect/turn cycles.
        # The fixed box remains rendered after completion; success also needs
        # resumed fresh guidance and translation beyond the obstacle, not just
        # an incremented planner counter. This is not a learned-model trial.
        position, yaw, speed = np.array([0., 0., 1.1]), 0., 0.
        completed_at, resumed = None, False
        box_min, box_max = np.array([1.9, .40]), np.array([2.1, .90])
        for index in range(281):
            now = index*.05
            state = state_at(now, position, yaw, speed)
            bearing = float(np.arctan2(-position[1], 5.3-position[0]) + .035)
            # One unavailable interval must neither command translation nor
            # renew the original fourteen-second attempt budget.
            selected = request(None) if 1.8 <= now < 2.3 else request(
                index, bearing, capture_time_s=now-.15 if now >= .15 else 0.,
                valid_until_s=now+.70)
            result = self.navigator.update(sensor_frame(state), state, selected)
            if selected['sequence'] is None:
                self.assertEqual(result['forward_speed'], 0.)
                self.assertEqual(self.navigator.started_s, 0.)
            if result['reason'] == 'detour_complete':
                completed_at = now
                self.assertEqual(result['forward_speed'], 0.)
            if completed_at is not None and result['phase'] == 'following':
                resumed |= result['forward_speed'] > 0
            closest = np.clip(position[:2], box_min, box_max)
            self.assertGreater(np.linalg.norm(position[:2]-closest), .52)
            yaw += np.clip(wrap_angle(result['yaw_target']-yaw), -.25*.05, .25*.05)
            speed += np.clip(result['forward_speed']-speed, -.35*.05, .6*.05)
            position += .05*speed*np.array([np.cos(yaw), np.sin(yaw), 0.])
            if resumed and position[0] > box_max[0]+.32:
                break
        self.assertIsNotNone(completed_at, result)
        self.assertLess(completed_at, 14.)
        self.assertEqual(self.navigator.completed, 1)
        self.assertTrue(resumed)
        self.assertGreater(position[0], box_max[0]+.32)

    def test_reverse_or_side_drift_and_bad_input_fail_closed(self):
        frame = sensor_frame(self.state, None)
        for selected in [request(speed=-.1), request(yaw=float('nan')), request(True),
                         {'sequence': 0}]:
            self.assertEqual(DetourNavigator().update(frame, self.state, selected)['forward_speed'], 0.)
        for velocity in [(-.1, 0., 0.), (0., .1, 0.)]:
            drifting = replace(self.state, velocity=np.asarray(velocity))
            result = DetourNavigator().update(sensor_frame(drifting, None), drifting, request())
            self.assertEqual(result['forward_speed'], 0.)
        invalid = replace(self.state, rotation=np.full((3, 3), np.nan))
        self.assertEqual(self.navigator.update(frame, invalid, request())['forward_speed'], 0.)

    def test_braking_overshoot_gets_no_motion_then_resumes_inspection(self):
        planned = self.navigator.update(sensor_frame(self.state), self.state, request())
        heading = planned['yaw_target']
        turned = state_at(.1, yaw=heading)
        self.navigator.update(sensor_frame(turned), turned, request(1))
        position = [np.cos(heading)*.6, np.sin(heading)*.6, 1.1]
        stopped = state_at(2., position, heading)
        # Trigger the retained early view-bound inspection, rather than the
        # removed inspection of an unknown corridor after .55 m.
        bearing = float(heading+.24)
        self.assertEqual(self.navigator.update(sensor_frame(stopped), stopped,
                                              request(2, bearing))['phase'], 'rejoining')
        overshoot = replace(stopped, velocity=stopped.rotation[:, 0]*-.04)
        held = self.navigator.update(sensor_frame(overshoot), overshoot, request(3, bearing))
        self.assertEqual(held['phase'], 'rejoining')
        self.assertEqual(held['guardian']['reason'], 'unknown_reverse_motion')
        self.assertEqual(held['forward_speed'], 0.)
        self.assertEqual(held['yaw_target'], overshoot.yaw)
        resumed = self.navigator.update(sensor_frame(stopped), stopped, request(4, bearing))
        self.assertEqual(resumed['reason'], 'turn_to_selected_target')
        self.assertEqual(resumed['forward_speed'], 0.)

    def test_independent_depth_fixture_completes_inspection_and_rejoin(self):
        # Ideal kinematic execution isolates planning from the separate native
        # motor test. The box remains fixed throughout; no fake guardian or
        # disappearing obstacle can manufacture the clear rejoin decision.
        position, yaw, speed = np.array([0., 0., 1.1]), 0., 0.
        seen_inspection = False
        for index in range(130):
            state = state_at(index*.1, position, yaw, speed)
            command_bearing = np.arctan2(-position[1], 5.3-position[0])
            output = self.navigator.update(sensor_frame(state), state,
                                           request(index, command_bearing))
            if output['phase'] == 'rejoining':
                seen_inspection = True
                self.assertEqual(output['forward_speed'], 0.)
                self.assertTrue(np.allclose(self.navigator.origin, [0., 0., 1.1]))
                self.assertEqual(self.navigator.started_s, 0.)
            yaw += np.clip(wrap_angle(output['yaw_target']-yaw), -.07, .07)
            speed = output['forward_speed']
            position += .1*speed*np.array([np.cos(yaw), np.sin(yaw), 0.])
            if self.navigator.completed:
                break
        self.assertTrue(seen_inspection)
        self.assertEqual(self.navigator.completed, 1)
        self.assertGreater(position[0], 1.)
        self.assertLess(position[1], -.15)
        self.assertEqual(output['forward_speed'], 0.)

    def test_blocked_view_bound_inspection_reprobes_without_renewing_bounds(self):
        plan = self.navigator.update(sensor_frame(self.state), self.state, request())
        heading = plan['yaw_target']
        turned = state_at(.1, yaw=heading)
        self.navigator.update(sensor_frame(turned), turned, request(1))
        bearing = heading+.24
        position = (.1, -.02, 1.1)
        near_limit = state_at(.3, position, heading)
        result = self.navigator.update(sensor_frame(near_limit), near_limit, request(2, bearing))
        self.assertEqual(result['phase'], 'rejoining')
        facing = state_at(1., position, bearing)
        result = self.navigator.update(sensor_frame(facing), facing, request(3, bearing))
        self.assertEqual(result['phase'], 'braking')
        self.assertEqual(result['reason'], 'target_route_still_blocked')
        self.assertEqual(result['forward_speed'], 0.)
        self.assertEqual(self.navigator.started_s, 0.)
        np.testing.assert_array_equal(self.navigator.origin, self.state.position)
        replanning = state_at(1.1, position, bearing)
        result = self.navigator.update(sensor_frame(replanning), replanning, request(4, bearing))
        self.assertEqual(result['phase'], 'turning')
        self.assertEqual(self.navigator.started_s, 0.)
        np.testing.assert_array_equal(self.navigator.origin, self.state.position)
        self.assertEqual(self.navigator.attempts, 1)

    def test_distance_limit_stops_even_with_clear_current_depth(self):
        planned = self.navigator.update(sensor_frame(self.state), self.state, request())
        yaw = planned['yaw_target']
        turned = state_at(.1, yaw=yaw)
        self.navigator.update(sensor_frame(turned), turned, request(1))
        moved = state_at(10., (2.1, -.2, 1.1), yaw)
        output = self.navigator.update(sensor_frame(moved, None), moved, request(2))
        self.assertEqual(output['reason'], 'detour_distance_limit')
        self.assertEqual(output['forward_speed'], 0.)

    def test_disabled_detours_still_keep_final_stop_gate(self):
        result = self.navigator.update(sensor_frame(self.state, (.8, 0., .7)),
                                       self.state, request(), enabled=False)
        self.assertEqual(result['forward_speed'], 0.)
        self.assertEqual(result['guardian']['reason'], 'blocked_stopping_distance')

    def test_bounded_timeout_does_not_keep_attempting(self):
        self.navigator.update(sensor_frame(self.state), self.state, request())
        later = state_at(15.)
        result = self.navigator.update(sensor_frame(later), later, request(1))
        self.assertEqual(result['reason'], 'detour_time_limit')
        self.assertEqual(result['forward_speed'], 0.)

    @unittest.skipUnless(os.environ.get('FLIGHT_RENDER_TESTS') == '1', 'opt-in native graphics')
    def test_native_offset_fixture_has_observed_corridor(self):
        from experiments.mantis_arena import ArenaWorld
        world = ArenaWorld(image_size=192)
        try:
            world.update_scene(0., trajectory='detour', obstacle=True)
            result = self.navigator.update(world.capture(safety=True), world.state(), request())
            self.assertEqual(result['phase'], 'turning', result)
            self.assertTrue(any(candidate['observed_clear'] for candidate in result['candidates']))
            self.assertEqual(result['forward_speed'], 0.)
            self.assertEqual(world.truth()['contact_count'], 0)
        finally:
            world.close()


if __name__ == '__main__':
    unittest.main()
