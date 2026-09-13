"""Independent trajectory acceptance for conventional stabilization."""
from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np

from experiments.flight_contracts import wrap_angle, BRAKING_ACCEL_M_S2, BRAKING_TRANSIENT_MARGIN_M
from experiments.flight_world import QuadWorld,MASS_KG,GRAVITY_M_S2,MAX_MOTOR_FORCE_N
from experiments.flight_autopilot import Autopilot,MIXER

PATCH = Path(__file__).resolve().parents[1]/'results/hybrid_flight_run01/source_patch.png'


class AutopilotTests(unittest.TestCase):
    def setUp(self):self.world=QuadWorld(PATCH);self.pilot=Autopilot()
    def tearDown(self):self.world.close()

    def advance(self,seconds,speed=0.,yaw=0.,altitude=1.1):
        states=[]
        for _ in range(round(seconds/.005)):
            state=self.world.state()
            states.append(self.world.step(self.pilot.command(state,speed,yaw,altitude)))
        return states

    def test_zero_speed_means_hover_not_motor_cut(self):
        forces=self.pilot.command(self.world.state(),0.,0.)
        np.testing.assert_allclose(forces,MASS_KG*GRAVITY_M_S2/4,atol=1e-12)
        states=self.advance(5)
        self.assertLess(max(abs(s.position[2]-1.1) for s in states),1e-6)
        self.assertLess(np.linalg.norm(states[-1].velocity),1e-6)

    def test_small_attitude_perturbations_recover_without_ground_contact(self):
        for roll,pitch in [( .15,0.),(-.15,0.),(0.,.15),(0.,-.15),(.10,-.10)]:
            self.world.reset(euler=(roll,pitch,.1))
            states=self.advance(6,yaw=.1)
            self.assertLess(max(abs(s.position[2]-1.1) for s in states),.025)
            self.assertGreater(states[-1].rotation[2,2],.99999)
            self.assertLess(np.linalg.norm(states[-1].velocity),.002)
            self.assertEqual(self.world.truth()['contact_count'],0)

    def test_heading_command_rotates_with_rotor_reaction_torque(self):
        states=self.advance(5,yaw=.45)
        self.assertLess(abs(wrap_angle(states[-1].yaw-.45)),.001)
        self.assertLess(np.linalg.norm(states[-1].velocity),.001)
        self.assertLess(max(abs(s.position[2]-1.1) for s in states),.005)

    def test_turning_speed_request_remains_along_current_heading(self):
        # An ideal level estimate already moving at the requested .45 m/s
        # needs no lateral acceleration while a new yaw request rotates it.
        # The requested translation is in the current certified corridor.
        state=replace(self.world.state(),velocity=np.array([.45,0.,0.]))
        forces=self.pilot.command(state,.45,np.pi/2)
        wrench=MIXER@forces
        self.assertAlmostEqual(wrench[1],0.,delta=1e-12)
        self.assertAlmostEqual(wrench[2],0.,delta=1e-12)
        self.assertGreater(wrench[3],.10)
        # Once already aligned, the same command should remain identical after
        # rotating the whole horizontal world frame by 90 degrees.
        original=self.pilot.command(self.world.state(),.45,0.)
        self.world.reset(euler=(0,0,np.pi/2))
        rotated=self.pilot.command(self.world.state(),.45,np.pi/2)
        np.testing.assert_allclose(rotated,original,atol=1e-12)

    def test_speed_and_altitude_request_move_motor_driven_aircraft(self):
        states=self.advance(5,speed=.45,yaw=.3,altitude=1.3)
        self.assertGreater(states[-1].position[0],1.6)
        self.assertGreater(states[-1].position[1],.3)
        self.assertLess(abs(states[-1].position[2]-1.3),.005)
        self.assertLess(abs(np.linalg.norm(states[-1].velocity[:2])-.45),.01)
        self.assertTrue(all(np.all((s.motor_forces>=0)&(s.motor_forces<=MAX_MOTOR_FORCE_N)) for s in states))

    def test_braking_bound_across_heading_and_initial_perturbations(self):
        for yaw,roll,pitch in [(0,0,0),(.35,.10,-.08),(-.35,-.10,.08)]:
            self.world.reset(euler=(roll,pitch,yaw))
            self.advance(6,speed=.45,yaw=yaw)
            initial=self.world.state();heading=initial.velocity[:2]/np.linalg.norm(initial.velocity[:2])
            initial_speed=np.linalg.norm(initial.velocity[:2])
            states=self.advance(5,speed=0.,yaw=yaw)
            max_distance=max((s.position[:2]-initial.position[:2])@heading for s in states)
            conservative_distance=initial_speed**2/(2*BRAKING_ACCEL_M_S2)
            self.assertLessEqual(max_distance,conservative_distance)
            self.assertLess(np.linalg.norm(states[-1].velocity),.002)
            self.assertLess(max(abs(s.position[2]-1.1) for s in states),.02)

    def test_transient_braking_bound_includes_motor_and_attitude_coast(self):
        # Brake is issued immediately at this physics tick; perception/depth
        # reaction delay belongs to the guardian, so this probe adds none.
        for acceleration_duration in [.2,.4,.6,.8,1.,1.5,2.,6.]:
            self.world.reset()
            self.advance(acceleration_duration,speed=.45)
            initial=self.world.state()
            initial_speed=np.linalg.norm(initial.velocity[:2])
            states=self.advance(5)
            displacement=max(np.linalg.norm(s.position[:2]-initial.position[:2]) for s in states)
            bound=BRAKING_TRANSIENT_MARGIN_M+initial_speed**2/(2*BRAKING_ACCEL_M_S2)
            self.assertLessEqual(displacement,bound)
            self.assertLess(np.linalg.norm(states[-1].velocity),.002)

    def test_mixer_supports_collective_without_roll_pitch_yaw(self):
        np.testing.assert_allclose(MIXER@np.full(4,2.),[8,0,0,0],atol=1e-12)

    def test_invalid_requests_rejected(self):
        for speed,yaw,altitude in [(-.1,0,1.1),(1.1,0,1.1),(0,np.nan,1.1),(0,0,0)]:
            with self.assertRaises(ValueError):self.pilot.command(self.world.state(),speed,yaw,altitude)
