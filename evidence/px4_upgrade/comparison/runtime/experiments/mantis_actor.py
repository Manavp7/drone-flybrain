"""Pinned CC BY 4.0 walking human rendered as a native MuJoCo skinned mesh.

The small bundled Cesium Man glTF is decoded locally; no Blender, network,
Torch, or external glTF library is needed. Animation drives 19 mocap bones,
never aircraft state. The skin contributes RGB and optical depth. Its collision
proxy is a separate transparent capsule, not an anatomically exact body.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ASSET_PATH = Path(__file__).resolve().parents[1] / 'assets/mantis_actor/CesiumMan.glb'
ASSET_SHA256 = 'b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9'
ACTOR_HEIGHT_M = 1.8


def _numbers(values):
    return ' '.join(format(float(x), '.9g') for x in np.asarray(values).ravel())


def _quat_wxyz(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def _trs(translation, quaternion, scale):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix() @ np.diag(scale)
    result[:3, 3] = translation
    return result


def _slerp(a, b, fraction):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        value = a + fraction * (b - a)
        return value / np.linalg.norm(value)
    angle = np.arccos(np.clip(dot, -1., 1.))
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)


@dataclass(frozen=True)
class ActorPose:
    """World-space animation result; vertices are evaluator-only ground truth."""
    vertices: np.ndarray
    bone_matrices: np.ndarray
    position: np.ndarray
    yaw: float
    time_s: float


class MantisActor:
    """True 3D walking character, normalized to 1.8 m in its initial pose.

    ``position`` is the ground anchor, unlike the original photo's center.
    ``yaw=0`` faces world +X and positive yaw turns towards world +Y.
    ``install_xml`` replaces only the original photo target in a QuadWorld XML.
    Compile with ``MjModel.from_xml_string(xml, assets=actor.assets)``; bind the
    model, then call ``animate`` before ``mj_forward``/camera capture. The
    animation is prescribed kinematics and is not a human dynamics model.
    """
    def __init__(self, asset_path=ASSET_PATH, *, appearance="clothing"):
        if appearance not in ("clothing", "source"):
            raise ValueError("Actor appearance must be clothing or source")
        self.appearance = appearance
        raw = Path(asset_path).read_bytes()
        if sha256(raw).hexdigest() != ASSET_SHA256:
            raise ValueError('Mantis actor must match the pinned licensed Cesium Man asset')
        magic, version, size = struct.unpack_from('<4sII', raw)
        if (magic, version, size) != (b'glTF', 2, len(raw)):
            raise ValueError('Invalid pinned actor GLB header')
        json_size, json_type = struct.unpack_from('<II', raw, 12)
        if json_type != 0x4E4F534A:
            raise ValueError('Missing GLB JSON chunk')
        self._gltf = json.loads(raw[20:20 + json_size])
        offset = 20 + json_size
        binary_size, binary_type = struct.unpack_from('<II', raw, offset)
        if binary_type != 0x004E4942 or offset + 8 + binary_size != len(raw):
            raise ValueError('Invalid GLB binary chunk')
        self._binary = raw[offset + 8:]
        primitive = self._gltf['meshes'][0]['primitives'][0]
        attrs = primitive['attributes']
        self._vertices = self._accessor(attrs['POSITION']).astype(float)
        self.faces = self._accessor(primitive['indices']).reshape(-1, 3).astype(int)
        self._joints = self._accessor(attrs['JOINTS_0']).astype(int)
        self._weights = self._accessor(attrs['WEIGHTS_0']).astype(float)
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        self._uv = self._accessor(attrs['TEXCOORD_0']).astype(float)
        skin = self._gltf['skins'][0]
        self._joint_nodes = skin['joints']
        self._inverse_bind = self._accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.bone_names = tuple(f'mantis_bone_{i}' for i in range(len(self._joint_nodes)))
        self._nodes = self._gltf['nodes']
        animation = self._gltf['animations'][0]
        self._channels = []
        for channel in animation['channels']:
            sampler = animation['samplers'][channel['sampler']]
            if sampler.get('interpolation', 'LINEAR') != 'LINEAR':
                raise ValueError('Only pinned LINEAR animation channels are supported')
            times, values = self._accessor(sampler['input']), self._accessor(sampler['output'])
            self._channels.append((channel['target']['node'], channel['target']['path'], times, values))
        self.duration_s = max(float(c[2][-1]) for c in self._channels)
        # glTF is Y-up, the room is Z-up; Cesium Man's front is glTF +Z.
        self._coordinate = np.eye(4)
        self._coordinate[:3, :3] = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        initial = self._skin_vertices(self._global_nodes(0.))
        initial_world = initial @ self._coordinate[:3, :3].T
        self.scale = ACTOR_HEIGHT_M / float(np.ptp(initial_world[:, 2]))
        self._coordinate[:3, :3] *= self.scale
        self._coordinate[2, 3] = -float(initial_world[:, 2].min()) * self.scale
        self._bind_vertices = self._vertices @ self._coordinate[:3, :3].T + self._coordinate[:3, 3]
        self._bind_bones = self._coordinate @ np.linalg.inv(self._inverse_bind)
        self._bind_bones[:, :3, :3] /= self.scale
        self._mocap_ids = None
        self._target_mocap = None
        self._make_assets()

    def _accessor(self, index):
        accessor = self._gltf['accessors'][index]
        view = self._gltf['bufferViews'][accessor['bufferView']]
        dtypes = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
        counts = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
        dtype, columns = np.dtype(dtypes[accessor['componentType']]), counts[accessor['type']]
        offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        stride = view.get('byteStride', dtype.itemsize * columns)
        result = np.ndarray((accessor['count'], columns), dtype=dtype, buffer=self._binary,
                            offset=offset, strides=(stride, dtype.itemsize)).copy()
        return result[:, 0] if columns == 1 else result

    def _global_nodes(self, time_s):
        local = []
        trs = []
        for node in self._nodes:
            trs.append({'translation': np.asarray(node.get('translation', [0., 0., 0.])),
                        'rotation': np.asarray(node.get('rotation', [0., 0., 0., 1.])),
                        'scale': np.asarray(node.get('scale', [1., 1., 1.]))})
        phase = float(time_s) % self.duration_s
        for node, path, times, values in self._channels:
            right = min(int(np.searchsorted(times, phase, side='right')), len(times) - 1)
            left = max(0, right - 1)
            fraction = 0. if right == left else np.clip((phase - times[left]) / (times[right] - times[left]), 0., 1.)
            trs[node][path] = (_slerp(values[left], values[right], fraction) if path == 'rotation'
                               else (1 - fraction) * values[left] + fraction * values[right])
        for node, values in zip(self._nodes, trs):
            local.append(np.array(node['matrix']).reshape(4, 4).T if 'matrix' in node
                         else _trs(values['translation'], values['rotation'], values['scale']))
        world = np.zeros((len(local), 4, 4))
        def visit(index, parent):
            world[index] = parent @ local[index]
            for child in self._nodes[index].get('children', []):
                visit(child, world[index])
        for index in self._gltf['scenes'][self._gltf.get('scene', 0)]['nodes']:
            visit(index, np.eye(4))
        return world

    def _skin_vertices(self, global_nodes):
        transforms = global_nodes[self._joint_nodes] @ self._inverse_bind
        homogeneous = np.column_stack((self._vertices, np.ones(len(self._vertices))))
        transformed = np.einsum('nvij,nj->nvi', transforms[self._joints], homogeneous)
        return np.einsum('nv,nvi->ni', self._weights, transformed)[:, :3]

    def _make_assets(self):
        # MuJoCo accepts PNG/JPEG bytes via its virtual asset filesystem.
        view = self._gltf['bufferViews'][self._gltf['images'][0]['bufferView']]
        texture = self._binary[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
        # Convert JPEG to PNG losslessly after decoding, and invert UV's vertical
        # axis because glTF texture v=0 is top, while MuJoCo uses bottom origin.
        from PIL import Image
        with Image.open(BytesIO(texture)) as image:
            output = BytesIO()
            image.convert('RGB').save(output, format='PNG')
        self.assets = {'mantis_actor_texture.png': output.getvalue()}
        self.asset_xml = ('<texture name="mantis_actor_texture" type="2d" file="mantis_actor_texture.png"/>'
                          '<material name="mantis_actor_material" texture="mantis_actor_texture" '
                          'texuniform="false" specular="0" shininess="0" emission=".15"/>')
        uv = self._uv.copy()
        uv[:, 1] = 1 - uv[:, 1]
        bones = []
        for index, name in enumerate(self.bone_names):
            influence = np.sum(np.where(self._joints == index, self._weights, 0.), axis=1)
            indices = np.flatnonzero(influence > 0)
            bind = self._bind_bones[index]
            bones.append(f'<bone body="{name}" bindpos="{_numbers(bind[:3, 3])}" '
                         f'bindquat="{_numbers(_quat_wxyz(bind[:3, :3]))}" '
                         f'vertid="{" ".join(map(str, indices))}" vertweight="{_numbers(influence[indices])}"/>')
        self.skin_xml = (f'<skin name="mantis_actor_skin" material="mantis_actor_material" '
                         f'vertex="{_numbers(self._bind_vertices)}" texcoord="{_numbers(uv)}" '
                         f'face="{" ".join(map(str, self.faces.ravel()))}">{"".join(bones)}</skin>')
        self.native_vertex_indices = np.arange(len(self._vertices))
        if self.appearance == 'clothing':
            # Recolor the licensed geometry into plain clothing. Assignment is
            # fixed in mesh space using rig influences, never camera detections.
            # This removes the separate trademarked logo from rendered frames.
            colors = {'shirt': '.12 .35 .62 1', 'trousers': '.12 .16 .22 1',
                      'skin': '.63 .40 .26 1', 'shoes': '.06 .07 .08 1'}
            categories = np.array(['shirt', 'shirt', 'shirt', 'skin', 'skin',
                                   'shirt', 'shirt', 'skin', 'skin', 'skin', 'skin',
                                   'trousers', 'trousers', 'trousers', 'trousers',
                                   'shoes', 'shoes', 'shoes', 'shoes'])
            strengths = np.stack([np.sum(np.where(categories[self._joints] == name,
                                                   self._weights, 0.), axis=1)
                                  for name in colors], axis=1)
            face_groups = np.argmax(strengths[self.faces].sum(axis=1), axis=1)
            skins, indices_all = [], []
            for group, (name, color) in enumerate(colors.items()):
                selected = self.faces[face_groups == group]
                vertices, remapped = np.unique(selected, return_inverse=True)
                indices_all.extend(vertices.tolist())
                self.asset_xml += (f'<material name="mantis_actor_{name}" rgba="{color}" '
                                   'specular="0" shininess="0" emission=".12"/>')
                group_bones = []
                for index, bone_name in enumerate(self.bone_names):
                    influence = np.sum(np.where(self._joints[vertices] == index,
                                                self._weights[vertices], 0.), axis=1)
                    ids = np.flatnonzero(influence > 0)
                    if not len(ids):
                        continue
                    bind = self._bind_bones[index]
                    group_bones.append(f'<bone body="{bone_name}" bindpos="{_numbers(bind[:3, 3])}" '
                                       f'bindquat="{_numbers(_quat_wxyz(bind[:3, :3]))}" '
                                       f'vertid="{" ".join(map(str, ids))}" vertweight="{_numbers(influence[ids])}"/>')
                skins.append(f'<skin name="mantis_actor_{name}_skin" material="mantis_actor_{name}" '
                             f'vertex="{_numbers(self._bind_vertices[vertices])}" '
                             f'face="{" ".join(map(str, remapped))}">{"".join(group_bones)}</skin>')
            self.skin_xml = ''.join(skins)
            self.native_vertex_indices = np.asarray(indices_all)
        # Alpha zero hides the approximate collider from RGB/depth; collision
        # remains active. Native rendered skin supplies the visible depth.
        self.body_xml = ('<body name="target" mocap="true" pos="5 0 0">'
                         '<geom name="mantis_actor_collider" type="capsule" '
                         'fromto="0 0 .3 0 0 1.5" size=".3" rgba="0 0 0 0"/></body>'
                         + ''.join(f'<body name="{name}" mocap="true"/>' for name in self.bone_names))

    def install_xml(self, base_mjcf):
        """Return QuadWorld XML with the photograph replaced by the walking skin."""
        root = ET.fromstring(base_mjcf)
        asset, world = root.find('asset'), root.find('worldbody')
        if asset is None or world is None:
            raise ValueError('Mantis actor requires asset and worldbody sections')
        targets = [body for body in world.findall('body') if body.get('name') == 'target']
        if len(targets) != 1:
            raise ValueError('Expected exactly one original target body')
        world.remove(targets[0])
        for item in list(asset):
            if item.get('name') in ('person_photo', 'photo', 'person_board'):
                asset.remove(item)
        for item in ET.fromstring(f'<root>{self.asset_xml}</root>'):
            asset.append(item)
        deformable = root.find('deformable')
        if deformable is None:
            deformable = ET.SubElement(root, 'deformable')
        for skin in ET.fromstring(f'<root>{self.skin_xml}</root>'):
            deformable.append(skin)
        for body in ET.fromstring(f'<root>{self.body_xml}</root>'):
            world.append(body)
        return ET.tostring(root, encoding='unicode')

    def bind(self, model):
        """Resolve this actor's mocap indices without changing the model/state."""
        ids = [int(model.body_mocapid[model.body(name).id]) for name in self.bone_names]
        target = int(model.body_mocapid[model.body('target').id])
        if min(ids + [target]) < 0:
            raise ValueError('Mantis actor and collider must all be mocap bodies')
        self._mocap_ids, self._target_mocap = np.array(ids), target
        return self

    def pose(self, time_s, position=(5., 0., 0.), yaw=0.):
        position = np.asarray(position, dtype=float)
        if (position.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite([time_s, yaw]).all() or time_s < 0):
            raise ValueError('Actor requires finite position/yaw and nonnegative time')
        global_nodes = self._global_nodes(time_s)
        c, s = np.cos(yaw), np.sin(yaw)
        placement = np.array([[c, -s, 0., position[0]], [s, c, 0., position[1]],
                              [0., 0., 1., position[2]], [0., 0., 0., 1.]])
        transform = placement @ self._coordinate
        bones = transform @ global_nodes[self._joint_nodes]
        bones[:, :3, :3] /= self.scale
        # Tiny (~1e-7) authoring scales are rounded away for native rigid bones.
        bones[:, :3, :3] = Rotation.from_matrix(bones[:, :3, :3]).as_matrix()
        raw_vertices = self._skin_vertices(global_nodes)
        vertices = raw_vertices @ transform[:3, :3].T + transform[:3, 3]
        return ActorPose(vertices, bones, position.copy(), float(yaw), float(time_s))

    def animate(self, data, time_s, position=(5., 0., 0.), yaw=0.):
        """Set actor mocap transforms only. Caller runs mj_forward afterwards."""
        if self._mocap_ids is None:
            raise RuntimeError('Call actor.bind(model) before animation')
        pose = self.pose(time_s, position, yaw)
        data.mocap_pos[self._mocap_ids] = pose.bone_matrices[:, :3, 3]
        quaternions = Rotation.from_matrix(pose.bone_matrices[:, :3, :3]).as_quat()
        data.mocap_quat[self._mocap_ids] = quaternions[:, [3, 0, 1, 2]]
        data.mocap_pos[self._target_mocap] = pose.position
        data.mocap_quat[self._target_mocap] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        return pose

    def vertices_world(self, time_s, position=(5., 0., 0.), yaw=0.):
        """Return animated mesh vertices for evaluator use, never guidance."""
        return self.pose(time_s, position, yaw).vertices
