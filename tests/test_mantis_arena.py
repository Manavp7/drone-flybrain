"""Independent skinned people, unchanged aircraft and evaluator-only scoring."""
import os
import unittest

import numpy as np

from experiments.mantis_arena import ArenaWorld, actor_placements, evaluate_selection


class ArenaPlacementTests(unittest.TestCase):
    def test_crossing_is_repeatable_and_preserves_separate_lanes(self):
        before, crossing, after = [actor_placements(t, 'crossing', .1) for t in (0., 8.5, 17.)]
        self.assertLess(before[0][0][1], before[1][0][1])
        self.assertGreater(after[0][0][1], after[1][0][1])
        self.assertAlmostEqual(crossing[0][0][1], 0.)
        self.assertAlmostEqual(crossing[1][0][1], 0.)
        self.assertGreater(crossing[1][0][0]-crossing[0][0][0], .6)
        for a, b in zip(before, actor_placements(0., 'crossing', .1)):
            np.testing.assert_array_equal(a[0], b[0])

    def test_stationary_and_zero_speed_are_fixed_ground_anchors(self):
        for trajectory, speed in [('stationary', .1), ('crossing', 0.), ('walk', 0.)]:
            for a, b in zip(actor_placements(0., trajectory, speed), actor_placements(20., trajectory, speed)):
                np.testing.assert_array_equal(a[0], b[0])

    def test_occlusion_has_bounded_hidden_interval_and_separates_again(self):
        at_start = actor_placements(0., 'occlusion', .1)
        during = actor_placements(15., 'occlusion', .1)
        after = actor_placements(32., 'occlusion', .1)
        self.assertGreater(abs(at_start[0][0][1]), .5)
        self.assertEqual(during[0][0][1], during[1][0][1])
        self.assertGreater(abs(after[0][0][1]), .5)

    def test_detour_fixture_has_stationary_separated_people(self):
        initial = actor_placements(0., 'detour')
        later = actor_placements(20., 'detour')
        for a, b in zip(initial, later):
            np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(initial[0][0][1], 0.)
        self.assertGreater(abs(initial[1][0][1]), 1.)

    def test_invalid_inputs_are_rejected(self):
        for time, trajectory, speed in [(float('nan'), 'walk', .1), (-1, 'walk', .1),
                                        (0, 'unknown', .1), (0, 'walk', float('inf')),
                                        (0, 'walk', True), (True, 'walk', .1), (0, 'walk', .51)]:
            with self.subTest(time=time, trajectory=trajectory, speed=speed):
                with self.assertRaises(ValueError):
                    actor_placements(time, trajectory, speed)

    def test_evaluator_distinguishes_wrong_person_and_unscorable_overlap(self):
        blue = dict(actor_id='blue', visible=True, bbox_xyxy=[10, 10, 30, 90])
        orange = dict(actor_id='orange', visible=True, bbox_xyxy=[80, 10, 100, 90])
        observation = dict(valid=True, bbox_xyxy=orange['bbox_xyxy'])
        audit = evaluate_selection(observation, [blue, orange], 'blue')
        self.assertEqual(audit['actor_id'], 'orange')
        self.assertTrue(audit['wrong_person'])
        blue['bbox_xyxy'] = orange['bbox_xyxy']
        audit = evaluate_selection(observation, [blue, orange], 'blue')
        self.assertTrue(audit['ambiguous'])
        self.assertIsNone(audit['wrong_person'])
        self.assertIsNone(evaluate_selection(dict(valid=False), [blue, orange])['actor_id'])


class ArenaNativeTests(unittest.TestCase):
    def setUp(self):
        self.world = ArenaWorld(image_size=192)

    def tearDown(self):
        self.world.close()

    def test_two_independent_skins_preserve_aircraft_dofs(self):
        self.assertEqual((self.world.model.nq, self.world.model.nv, self.world.model.nu), (7, 6, 4))
        self.assertEqual(self.world.model.nskin, 8)
        self.assertEqual(self.world.model.nmocap, 41)
        first, second = self.world.actor, self.world.other_actor
        self.assertFalse(set(first.bone_names) & set(second.bone_names))
        self.assertFalse(set(first._mocap_ids) & set(second._mocap_ids))
        self.assertNotEqual(first._target_mocap, second._target_mocap)
        blue = self.world.model.material('mantis_actor_shirt').rgba
        orange = self.world.model.material('guest_actor_shirt').rgba
        self.assertGreater(float(np.linalg.norm(blue-orange)), .5)

    def test_changing_other_actor_does_not_touch_first_or_aircraft(self):
        world = self.world
        before = {name: getattr(world.data, name).copy() for name in ('qpos', 'qvel', 'ctrl')}
        first_bones = world.data.mocap_pos[world.actor._mocap_ids].copy()
        world.other_actor.animate(world.data, .8, (6., -.3, 0.), 1.)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(world.data, name), value)
        np.testing.assert_array_equal(first_bones, world.data.mocap_pos[world.actor._mocap_ids])
        self.assertEqual(world.data.time, 0.)

    def test_scene_updates_only_on_current_physics_clock(self):
        before = self.world.data.qpos.copy()
        self.world.update_scene(0., trajectory='stationary', obstacle=True)
        np.testing.assert_array_equal(self.world.data.qpos, before)
        self.assertTrue(self.world.truth()['obstacle_enabled'])
        self.assertEqual(len(self.world.truth()['people']), 2)
        with self.assertRaisesRegex(ValueError, 'physics time'):
            self.world.update_scene(1.)
        with self.assertRaises(ValueError):
            self.world.update_scene(0., obstacle='yes')

    @unittest.skipUnless(os.environ.get('FLIGHT_RENDER_TESTS') == '1', 'opt-in native graphics')
    def test_both_skins_render_depth_and_project_with_actual_camera(self):
        self.world.update_scene(0., trajectory='stationary')
        image = self.world.capture()
        people = self.world.evaluation_people(image)
        self.assertEqual(len(people), 2)
        self.assertTrue(all(person['visible'] for person in people))
        for person in people:
            x0, y0, x1, y1 = np.asarray(person['bbox_xyxy'], int)
            crop = image.rgb[max(0, y0):y1, max(0, x0):x1]
            depths = image.depth_m[max(0, y0):y1, max(0, x0):x1]
            # Real foreground depth and distinct shirt pixels inside both
            # geometric boxes; a background-only box cannot satisfy this.
            self.assertGreater(np.count_nonzero(depths < 7.), 50)
            rgb = crop.astype(int)
            clothing = (rgb[..., 2] > rgb[..., 0]+30) if person['actor_id'] == 'blue' else (
                rgb[..., 0] > rgb[..., 2]+30)
            self.assertGreater(np.count_nonzero(clothing), 30)


if __name__ == '__main__':
    unittest.main()
