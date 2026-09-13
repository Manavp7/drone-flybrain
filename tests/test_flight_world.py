"""Motor and camera acceptance checks for the new MuJoCo fixture."""
import os
from pathlib import Path
import unittest
import numpy as np

from experiments.flight_contracts import PHYSICS_DT, R_BODY_CAMERA, rotation_from_euler
from experiments.flight_world import (QuadWorld, MASS_KG, GRAVITY_M_S2, MAX_MOTOR_FORCE_N,
                                     MOTOR_TIME_CONSTANT_S, CAMERA_OFFSET_M)

PATCH = Path(__file__).resolve().parents[1]/'results/hybrid_flight_run01/source_patch.png'


class WorldPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.world = QuadWorld(PATCH)

    def tearDown(self):
        self.world.close()

    def test_hover_motor_forces_support_mass_without_kinematic_assignment(self):
        for _ in range(600):
            state = self.world.step(np.full(4,MASS_KG*GRAVITY_M_S2/4))
        np.testing.assert_allclose(state.position,[0,0,1.1],atol=1e-9)
        np.testing.assert_allclose(state.velocity,0,atol=1e-9)
        self.assertAlmostEqual(state.time_s,3.,places=10)
        self.assertEqual(self.world.model.nv,6)
        self.assertEqual(self.world.model.nu,4)
        self.assertTrue(np.all(self.world.data.qfrc_applied == 0))
        self.assertTrue(np.all(self.world.data.xfrc_applied == 0))

    def test_power_loss_falls_and_contacts_ground(self):
        for _ in range(200):
            state = self.world.step(np.zeros(4))
        self.assertLess(state.position[2],.08)
        self.assertGreater(self.world.truth()['contact_count'],0)
        self.assertLess(state.motor_forces.max(),1e-9)

    def test_rotor_lag_applies_force_instead_of_instant_command(self):
        initial = self.world.motor_forces.copy()
        state = self.world.step(np.full(4,MAX_MOTOR_FORCE_N))
        expected = initial+(1-np.exp(-PHYSICS_DT/MOTOR_TIME_CONSTANT_S))*(MAX_MOTOR_FORCE_N-initial)
        np.testing.assert_allclose(state.motor_forces,expected,atol=1e-12)
        self.assertLess(state.motor_forces.max(),MAX_MOTOR_FORCE_N)
        self.assertGreater(state.velocity[2],0)

    def test_single_rotor_produces_expected_body_torque(self):
        target = self.world.motor_forces.copy();target[0] += .5
        state = self.world.step(target)
        self.assertGreater(state.angular_velocity[0],0)
        self.assertLess(state.angular_velocity[1],0)
        self.assertGreater(state.angular_velocity[2],0)

    def test_state_arrays_are_copies_and_frame_conventions_are_explicit(self):
        state = self.world.reset(euler=(.1,-.08,.3))
        np.testing.assert_allclose(state.rotation,rotation_from_euler(.1,-.08,.3),atol=1e-12)
        state.position[:] = 100
        state.rotation[:] = 100
        state.motor_forces[:] = 100
        self.assertLess(self.world.state().position[0],1)
        np.testing.assert_allclose(self.world.state().rotation.T@self.world.state().rotation,np.eye(3),atol=1e-12)
        self.assertLess(self.world.state().motor_forces.max(),MAX_MOTOR_FORCE_N)

    def test_target_and_obstacle_move_only_mocap_objects(self):
        before = self.world.state()
        self.world.set_target([4,.5,.95]);self.world.set_obstacle([2,0,.7])
        after = self.world.state()
        np.testing.assert_array_equal(after.position,before.position)
        np.testing.assert_array_equal(after.velocity,before.velocity)
        self.assertEqual(self.world.truth()['target_position'],[4,.5,.95])
        self.assertTrue(self.world.truth()['obstacle_enabled'])
        np.testing.assert_array_equal(self.world.data.mocap_pos[self.world._target_mocap],[4,.5,.95])
        self.world.set_target([4,.5,.95],hidden=True)
        self.assertTrue(self.world.truth()['target_hidden'])

    def test_invalid_motor_targets_rejected(self):
        for request in ([1,2,3],[0,0,-1,0],[6,0,0,0],[np.nan,0,0,0]):
            with self.assertRaises(ValueError):self.world.step(request)


@unittest.skipUnless(os.environ.get('FLIGHT_RENDER_TESTS') == '1',
                     'Set FLIGHT_RENDER_TESTS=1 with native graphics access for calibrated render checks')
class WorldCameraTests(unittest.TestCase):
    def setUp(self):self.world = QuadWorld(PATCH)
    def tearDown(self):self.world.close()

    def test_primary_pose_depth_and_registration(self):
        frame = self.world.capture()
        self.assertEqual(frame.rgb.shape,(391,391,3))
        np.testing.assert_allclose(frame.rotation_world_camera,R_BODY_CAMERA,atol=1e-12)
        np.testing.assert_allclose(frame.position_world_camera,[.25,0,1.1],atol=1e-12)
        self.assertAlmostEqual(float(frame.depth_m[195,195]),4.74,delta=.0002)
        # The same visible board is near in both RGB and metric depth.
        column = frame.depth_m[:,195]
        ys = np.flatnonzero(np.abs(column-4.74)<.002)
        self.assertGreater(len(ys),100)
        self.assertLess(ys.min(),160)
        self.assertGreater(ys.max(),245)

    def test_both_calibrations_recover_known_plane_with_rotated_camera(self):
        self.world.set_target([5,0,.95],hidden=True)
        self.world.reset(position=(.2,.15,1.2),euler=(.13,0.,.17))
        state = self.world.state()
        for safety in [False,True]:
            frame = self.world.capture(safety=safety)
            np.testing.assert_allclose(frame.position_world_camera,state.position+state.rotation@CAMERA_OFFSET_M,atol=1e-12)
            np.testing.assert_allclose(frame.rotation_world_camera,state.rotation@R_BODY_CAMERA,atol=1e-12)
            fx,fy,cx,cy = frame.intrinsics
            for u,v in [(int(cx),int(cy)),(int(cx)+1,int(cy)+1)]:
                ray = frame.rotation_world_camera@np.array([(u-cx)/fx,(v-cy)/fy,1.])
                expected = (11.9-frame.position_world_camera[0])/ray[0]
                self.assertAlmostEqual(float(frame.depth_m[v,u]),expected,delta=.003)

    def test_body_camera_observation_changes_when_motors_rotate_body(self):
        first = self.world.capture()
        target = self.world.motor_forces.copy();target[[0,2]] += .12;target[[1,3]] -= .12
        for _ in range(60):self.world.step(target)
        second = self.world.capture()
        self.assertGreater(np.mean(np.abs(first.rgb.astype(float)-second.rgb)),1.)
        self.assertGreater(abs(self.world.state().yaw),.015)
        self.assertGreater(second.capture_time_s,first.capture_time_s)
