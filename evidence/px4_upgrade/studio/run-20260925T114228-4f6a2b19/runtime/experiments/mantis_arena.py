"""Two skinned people in the unchanged motor-driven Mantis flight fixture.

Actor names, poses, projections and contact records are evaluator-only. They
must never enter target selection or guidance, which consume camera detections.
"""
from __future__ import annotations

from numbers import Real
import xml.etree.ElementTree as ET

import numpy as np

from experiments.mantis_actor import MantisActor
from experiments.mantis_world import MantisWorld
from perception.pipeline import iou


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def actor_placements(time_s, trajectory='crossing', target_speed=.1):
    """Deterministic ground-anchor poses, for fixture setup/evaluation only.

    Crossing actors have separated X lanes; at the midpoint one really occludes
    the other in the forward image without their collision proxies intersecting.
    The occlusion scene places one person behind the other for a bounded period.
    """
    if not _finite(time_s) or time_s < 0:
        raise ValueError('time_s must be finite and nonnegative')
    if not _finite(target_speed) or not 0 <= target_speed <= .5:
        raise ValueError('target_speed must be finite in [0, .5] m/s')
    if trajectory not in ('crossing', 'walk', 'stationary', 'occlusion', 'detour'):
        raise ValueError('Unknown actor trajectory')
    distance = float(target_speed * time_s)
    if trajectory == 'detour':
        first, second = (5.3, 0., 0.), (6., -1.4, 0.)
    elif trajectory == 'stationary':
        first, second = (4.9, -.85, 0.), (5.8, .85, 0.)
    elif trajectory == 'walk':
        # Reflect rather than letting a long interactive run leave the room.
        along = 1.2 - abs((distance % 4.8) - 2.4)
        first, second = (4.9, along, 0.), (6., 1.35, 0.)
    else:
        lateral = .85 - abs((distance % 3.4) - 1.7)
        if trajectory == 'occlusion':
            # A true multi-second occlusion at ordinary speeds, with continuous
            # approach/departure rather than moving the actors offscreen.
            phase = distance % 4.4
            lateral = max(0., .85-phase) if phase < 2.2 else min(.85, phase-2.2)
        first, second = (4.9, lateral, 0.), (6., -lateral, 0.)
    return ((np.asarray(first), float(np.pi)), (np.asarray(second), float(np.pi)))


class _SecondActor(MantisActor):
    """Namespace the existing pinned skin; keep its geometry and animation."""
    def __init__(self):
        super().__init__()
        self.bone_names = tuple(name.replace('mantis_', 'guest_') for name in self.bone_names)
        self.assets = {name.replace('mantis_', 'guest_'): data for name, data in self.assets.items()}
        self.asset_xml = self.asset_xml.replace('mantis_', 'guest_').replace(
            'rgba=".12 .35 .62 1"', 'rgba=".78 .25 .11 1"')
        self.skin_xml = self.skin_xml.replace('mantis_', 'guest_')
        self.body_xml = self.body_xml.replace('mantis_', 'guest_').replace(
            'name="target"', 'name="guest_target"')

    def bind(self, model):
        ids = [int(model.body_mocapid[model.body(name).id]) for name in self.bone_names]
        target = int(model.body_mocapid[model.body('guest_target').id])
        if min(ids + [target]) < 0:
            raise ValueError('Every actor body must be mocap')
        self._mocap_ids, self._target_mocap = np.asarray(ids), target
        return self


class ArenaWorld(MantisWorld):
    """Mantis aircraft/cameras plus two independently animated clothed people."""
    def __init__(self, image_size=391):
        self._arena_ready = False
        super().__init__(image_size)
        self.other_actor = _SecondActor()
        root = ET.fromstring(self.mjcf)
        for section, xml in [('asset', self.other_actor.asset_xml),
                             ('deformable', self.other_actor.skin_xml),
                             ('worldbody', self.other_actor.body_xml)]:
            for node in ET.fromstring('<root>' + xml + '</root>'):
                root.find(section).append(node)
        self.mjcf = ET.tostring(root, encoding='unicode')
        self.model = self.mj.MjModel.from_xml_string(
            self.mjcf, assets={**self.actor.assets, **self.other_actor.assets})
        self.data = self.mj.MjData(self.model)
        self._quad_id = self.model.body('quad').id
        self._target_mocap = int(self.model.body_mocapid[self.model.body('target').id])
        self._obstacle_mocap = int(self.model.body_mocapid[self.model.body('obstacle').id])
        self.actor.bind(self.model)
        self.other_actor.bind(self.model)
        self._arena_ready = True
        self.reset()

    def reset(self, position=(0, 0, 1.1), euler=(0, 0, 0)):
        state = super().reset(position, euler)
        if self._arena_ready:
            self.update_scene(0.)
        return state

    def update_scene(self, time_s, trajectory='crossing', target_speed=.1, obstacle=False):
        poses = actor_placements(time_s, trajectory, target_speed)
        if type(obstacle) is not bool:
            raise ValueError('obstacle must be boolean')
        if abs(float(time_s) - float(self.data.time)) > 1e-8:
            raise ValueError('Scene animation must use the current physics time')
        self.set_actor(*poses[0])
        self._other_pose = self.other_actor.animate(self.data, time_s, *poses[1])
        obstacle_position = (2., .65, .7) if trajectory == 'detour' else (2.65, 0., .7)
        self.set_obstacle(obstacle_position, enabled=obstacle)
        self.mj.mj_forward(self.model, self.data)

    def truth(self):
        result = super().truth()
        if self._arena_ready:
            result['people'] = [
                dict(actor_id='blue', ground_anchor=self._actor_pose.position.tolist()),
                dict(actor_id='orange', ground_anchor=self._other_pose.position.tolist())]
            result['annotation_scope'] = 'evaluator only; never selected-track identity'
        return result

    def evaluation_people(self, frame):
        """Full-mesh image boxes, not visible-segmentation or detector inputs."""
        frame.validate()
        result = []
        for actor_id, pose in [('blue', self._actor_pose), ('orange', self._other_pose)]:
            if abs(pose.time_s - frame.capture_time_s) > 1e-8:
                raise ValueError('Evaluator poses must match camera capture time')
            optical = (pose.vertices - frame.position_world_camera) @ frame.rotation_world_camera
            item = dict(actor_id=actor_id, bbox_xyxy=None, visible=False,
                        annotation='full projected mesh; may be occluded')
            if np.min(optical[:, 2]) > .01:
                fx, fy, cx, cy = frame.intrinsics
                pixels = optical[:, :2] / optical[:, 2:] * [fx, fy] + [cx, cy]
                box = np.r_[pixels.min(0), pixels.max(0)]
                h, w = frame.rgb.shape[:2]
                item.update(bbox_xyxy=box.tolist(), visible=bool(
                    0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h))
            result.append(item)
        return result


def evaluate_selection(observation, projections, intended_actor_id=None):
    """Evaluator-only temporary identity association; overlap is unscorable.

    Bind the intended actor only once after an explicit user selection with a
    unique geometric match. Do not replace that reference when a track changes.
    """
    result = dict(actor_id=None, wrong_person=None, ambiguous=False, best_iou=None)
    if not observation.get('valid') or observation.get('bbox_xyxy') is None:
        return result
    matches = sorted([(iou(observation['bbox_xyxy'], p['bbox_xyxy']), p['actor_id'])
                      for p in projections if p.get('visible') and p.get('bbox_xyxy')], reverse=True)
    if not matches or matches[0][0] < .25:
        return result
    score, actor_id = matches[0]
    ambiguous = len(matches) > 1 and matches[1][0] >= max(.2, score - .2)
    result.update(best_iou=float(score), ambiguous=ambiguous)
    if not ambiguous:
        result.update(actor_id=actor_id, wrong_person=(actor_id != intended_actor_id)
                      if intended_actor_id is not None else None)
    return result
