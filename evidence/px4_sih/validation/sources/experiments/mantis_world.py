"""Mantis animated-actor extension of the unchanged motor-flight fixture."""
from __future__ import annotations

import numpy as np

from experiments.flight_contracts import ROOT
from experiments.flight_world import QuadWorld, _vector
from experiments.mantis_actor import MantisActor


class MantisWorld(QuadWorld):
    """Same aircraft and cameras, with a prescribed, genuinely 3D actor.

    Actor position is its ground anchor. Only actor/obstacle mocap transforms
    are prescribed; the free aircraft remains motor-driven. Rendering and depth
    come from the skinned mesh; contact uses its explicitly approximate capsule.
    """
    def __init__(self, image_size=391):
        self._mantis_ready = False
        super().__init__(ROOT/'results/hybrid_flight_run01/source_patch.png', image_size)
        self.actor = MantisActor()
        self.mjcf = self.actor.install_xml(self.mjcf)
        self.model = self.mj.MjModel.from_xml_string(self.mjcf, assets=self.actor.assets)
        self.data = self.mj.MjData(self.model)
        self._quad_id = self.model.body('quad').id
        self._target_mocap = int(self.model.body_mocapid[self.model.body('target').id])
        self._obstacle_mocap = int(self.model.body_mocapid[self.model.body('obstacle').id])
        self.actor.bind(self.model)
        self._target_position = np.array([4.9, 0., 0.])
        self._target_yaw = np.pi
        self._actor_pose = None
        self._mantis_ready = True
        self.reset()

    def reset(self, position=(0, 0, 1.1), euler=(0, 0, 0)):
        state = super().reset(position, euler)
        if self._mantis_ready:
            self.set_actor(self._target_position, self._target_yaw, hidden=self._target_hidden)
        return state

    def set_actor(self, position, yaw, hidden=False):
        if type(hidden) is not bool or not np.isfinite(yaw):
            raise ValueError('Actor visibility and heading must be valid')
        self._target_position = _vector(position, 'actor ground anchor').copy()
        self._target_yaw = float(yaw)
        self._target_hidden = hidden
        shown_position = [40., 40., 0.] if hidden else self._target_position
        self._actor_pose = self.actor.animate(self.data, float(self.data.time),
                                             shown_position, self._target_yaw)
        self.mj.mj_forward(self.model, self.data)

    def truth(self):
        # Keep the base collision audit, replacing all photograph geometry.
        result = super().truth()
        result.pop('target_corners_world', None)
        result.update(target_position=(self._target_position + [0., 0., .9]).tolist(),
                      target_ground_anchor=self._target_position.tolist(),
                      target_orientation='animated skinned 3D actor',
                      target_yaw_rad=self._target_yaw,
                      actor_pose_time_s=float(self._actor_pose.time_s),
                      collision_geometry='approximate capsule; no anatomical contact guarantee')
        return result

    def evaluation_projection(self, frame):
        """Geometric full-actor projection, scored only after sensor inference.

        This is not an occlusion/segmentation annotation. Occluding objects can
        hide part of this full box; hidden/offscreen actors have no valid truth.
        """
        frame.validate()
        if abs(self._actor_pose.time_s-frame.capture_time_s) > 1e-8:
            raise ValueError('Evaluator actor pose must match this capture time')
        if self._target_hidden:
            return dict(bbox_xyxy=None, heading_world_rad=None, visible=False)
        optical = (self._actor_pose.vertices-frame.position_world_camera) @ frame.rotation_world_camera
        if np.min(optical[:, 2]) <= .01:
            return dict(bbox_xyxy=None, heading_world_rad=None, visible=False)
        fx, fy, cx, cy = frame.intrinsics
        pixels = optical[:, :2]/optical[:, 2:] * [fx, fy] + [cx, cy]
        box = np.r_[pixels.min(0), pixels.max(0)]
        h, w = frame.rgb.shape[:2]
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            return dict(bbox_xyxy=box.tolist(), heading_world_rad=None, visible=False)
        u, v = (box[:2]+box[2:])/2
        ray = frame.rotation_world_camera @ np.array([(u-cx)/fx, (v-cy)/fy, 1.])
        return dict(bbox_xyxy=box.tolist(), heading_world_rad=float(np.arctan2(ray[1], ray[0])),
                    visible=True, annotation='projected full skinned mesh, not visible segmentation',
                    actor_pose_time_s=float(self._actor_pose.time_s))
