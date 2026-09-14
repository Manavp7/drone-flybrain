"""Geometry, causality and optional actual native rendering of the 3D actor."""
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from experiments.mantis_actor import MantisActor, ACTOR_HEIGHT_M, ASSET_PATH
from experiments.flight_contracts import load_mujoco, rotation_from_euler
from experiments.flight_world import QuadWorld

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'results/hybrid_flight_run01/source_patch.png'


class ActorGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor = MantisActor()

    def test_asset_bytes_are_checked_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'modified.glb'
            raw = bytearray(ASSET_PATH.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'pinned'):
                MantisActor(path)

    def test_actual_volume_height_and_ground_anchor(self):
        vertices = self.actor.vertices_world(0., (0., 0., 0.))
        self.assertAlmostEqual(float(vertices[:, 2].min()), 0., places=8)
        self.assertAlmostEqual(float(np.ptp(vertices[:, 2])), ACTOR_HEIGHT_M, places=8)
        self.assertGreater(float(np.min(np.ptp(vertices[:, :2], axis=0))), .5)
        self.assertGreater(len(self.actor.faces), 4000)
        self.assertEqual(len(self.actor.bone_names), 19)

    def test_animation_changes_limb_shape(self):
        a = self.actor.vertices_world(0., (0., 0., 0.))
        b = self.actor.vertices_world(.5, (0., 0., 0.))
        motion = b - a
        # Shape changes remain after removing any rigid translation.
        self.assertGreater(float(np.max(np.linalg.norm(motion-motion.mean(0), axis=1))), .25)

    def test_animation_loops_and_is_deterministic(self):
        a = self.actor.vertices_world(.37)
        np.testing.assert_allclose(a, self.actor.vertices_world(.37+self.actor.duration_s), atol=1e-7)
        np.testing.assert_array_equal(a, self.actor.vertices_world(.37))

    def test_ground_anchor_translation_and_yaw_equivariance(self):
        a = self.actor.vertices_world(.35, (0., 0., 0.))
        translation = np.array([4., -.7, .25])
        rotation = rotation_from_euler(yaw=.8)
        b = self.actor.vertices_world(.35, translation, .8)
        np.testing.assert_allclose(b, a @ rotation.T + translation, atol=1e-12)

    def test_bad_pose_inputs_are_rejected(self):
        for time, position, yaw in [(-1, [0,0,0], 0), (0, [0,0], 0),
                                     (0, [0,0,np.nan], 0), (0, [0,0,0], np.inf)]:
            with self.subTest(time=time, position=position, yaw=yaw):
                with self.assertRaises(ValueError):
                    self.actor.pose(time, position, yaw)

    def test_clothing_preserves_all_source_triangles_and_vertices(self):
        parts = ET.fromstring('<root>'+self.actor.skin_xml+'</root>')
        count = sum(len(part.get('face').split())//3 for part in parts)
        self.assertEqual(count, len(self.actor.faces))
        self.assertEqual(len(parts), 4)
        np.testing.assert_array_equal(np.unique(self.actor.native_vertex_indices),
                                      np.arange(len(self.actor._vertices)))
        self.assertTrue(all('texcoord' not in part.attrib for part in parts))


class ActorNativeTests(unittest.TestCase):
    def setUp(self):
        self.mj = load_mujoco()
        self.actor = MantisActor()
        self.world = QuadWorld(PATCH)
        self.xml = self.actor.install_xml(self.world.mjcf)
        self.model = self.mj.MjModel.from_xml_string(self.xml, assets=self.actor.assets)
        self.data = self.mj.MjData(self.model)
        self.actor.bind(self.model)

    def tearDown(self):
        self.world.close()

    def test_photo_is_removed_and_aircraft_dofs_are_preserved(self):
        self.assertNotIn('person_photo', self.xml)
        self.assertNotIn('person_board', self.xml)
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (7, 6, 4))
        self.assertEqual(self.model.nskin, 4)
        self.assertEqual(self.model.nmocap, 21)
        self.assertEqual(self.model.geom('mantis_actor_collider').rgba[3], 0.)
        self.assertGreater(self.model.geom('mantis_actor_collider').contype[0], 0)

    def test_animation_cannot_change_aircraft_state_control_or_time(self):
        self.data.qpos[:3] = [1., 2., 3.]
        self.data.qvel[:] = .7
        self.data.ctrl[:] = 2.2
        self.data.time = .23
        before = {name: getattr(self.data, name).copy() for name in ('qpos','qvel','ctrl')}
        pose = self.actor.animate(self.data, .7, (5., .4, .2), .3)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(self.data, name), value)
        self.assertEqual(self.data.time, .23)
        self.mj.mj_forward(self.model, self.data)
        for index, name in enumerate(self.actor.bone_names):
            np.testing.assert_allclose(self.data.body(name).xpos, pose.bone_matrices[index,:3,3], atol=1e-12)
        np.testing.assert_allclose(self.data.body('target').xpos, [5.,.4,.2])

    def test_requires_binding_before_animation(self):
        with self.assertRaisesRegex(RuntimeError, 'bind'):
            MantisActor().animate(self.data, 0.)

    @unittest.skipUnless(os.environ.get('FLIGHT_RENDER_TESTS') == '1', 'opt-in native graphics')
    def test_native_skin_matches_gltf_animation_and_supplies_true_depth(self):
        renderer = self.mj.Renderer(self.model, height=391, width=391)
        try:
            for phase, yaw in [(0.,0.), (.31,.7), (.83,-.4)]:
                pose = self.actor.animate(self.data, phase, (4.,0.,0.), yaw)
                self.mj.mj_forward(self.model,self.data)
                renderer.update_scene(self.data,camera='front')
                native = np.asarray(renderer.scene.skinvert).reshape(-1,3)
                np.testing.assert_allclose(native, pose.vertices[self.actor.native_vertex_indices], atol=2e-6)
            # Independent central ray/triangle intersections test the actual
            # animated mesh depth, excluding the transparent coarse collider.
            pose = self.actor.animate(self.data, 0., (4.,0.,0.), 0.)
            self.mj.mj_forward(self.model,self.data)
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data,camera='front')
            depth = renderer.render()
            origin = self.data.camera('front').xpos.copy()
            triangles = pose.vertices[self.actor.faces]
            edge1, edge2 = triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]
            ray = np.array([1.,0.,0.])
            h = np.cross(ray, edge2)
            a = np.einsum('ij,ij->i',edge1,h)
            valid = np.abs(a)>1e-9
            inv = np.divide(1.,a,out=np.zeros_like(a),where=valid)
            s = origin-triangles[:,0]
            u = inv*np.einsum('ij,ij->i',s,h)
            q = np.cross(s,edge1)
            v = inv*(q@ray)
            t = inv*np.einsum('ij,ij->i',edge2,q)
            hits = t[valid & (u>=0) & (v>=0) & (u+v<=1) & (t>0)]
            expected = float(hits.min())
            self.assertAlmostEqual(float(depth[195,195]),expected,delta=3e-4)
            self.assertGreater(abs(expected-(4.-.3-origin[0])), .1)
        finally:
            renderer.close()


if __name__ == '__main__':
    unittest.main()
