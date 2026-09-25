"""Four-motor MuJoCo flight fixture with calibrated, body-mounted RGB-D.

This world is an explicit engineering fixture: a fixed-orientation photograph
on a vertical board, ideal state sensors, no wind and no aircraft interface.
Only rotor actuator forces move the free-joint aircraft after reset.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np

from experiments.flight_contracts import (CameraFrame, FlightState, PHYSICS_DT,
                                           load_mujoco, rotation_from_euler)

MASS_KG = .8
INERTIA_KG_M2 = np.array([.008, .008, .014])
ARM_XY_M = .14
ROTOR_POSITIONS = np.array([[.14,.14,0.], [.14,-.14,0.], [-.14,-.14,0.], [-.14,.14,0.]])
ROTOR_SPINS = np.array([1., -1., 1., -1.])
YAW_MOMENT_PER_N = .025
MAX_MOTOR_FORCE_N = 5.
MOTOR_TIME_CONSTANT_S = .04
GRAVITY_M_S2 = 9.81
CAMERA_OFFSET_M = np.array([.25,0.,0.])
CAMERA_FOV_DEG = 70.
SAFETY_FOV_DEG = 150.
SAFETY_IMAGE_SIZE = 192
TARGET_HEIGHT_M = 1.9


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite three-vector')
    return result


class QuadWorld:
    """Six-DOF rigid body with 200 Hz physics and a 40 ms rotor response.

    `state().angular_velocity` is expressed in the body frame. Velocity is
    expressed in world axes. `truth()` is evaluation-only, never a sensor API.
    Call `capture(safety=True)` for the separate 150-degree depth sensor;
    default capture uses the 70-degree person camera. Each has its own K matrix.
    """
    def __init__(self, source_patch_path, image_size=391):
        self.mj = load_mujoco()
        self.patch_path = Path(source_patch_path).resolve()
        if not self.patch_path.is_file():
            raise ValueError('Source photograph is missing')
        if isinstance(image_size, bool) or not isinstance(image_size, int) or not 32 <= image_size <= 1024:
            raise ValueError('image_size must be an integer in [32,1024]')
        from PIL import Image
        with Image.open(self.patch_path) as source:
            patch_w, patch_h = source.size
        self.image_size = image_size
        self.target_half_width = .5*TARGET_HEIGHT_M*patch_w/patch_h
        w, h, t = self.target_half_width, TARGET_HEIGHT_M/2, .01
        vertices = [(-t,w,-h),(-t,-w,-h),(-t,-w,h),(-t,w,h),
                    (t,w,-h),(t,-w,-h),(t,-w,h),(t,w,h)]
        faces = [(0,1,2),(0,2,3),(4,6,5),(4,7,6), (0,4,5),(0,5,1),
                 (1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)]
        verts = ' '.join(str(v) for xyz in vertices for v in xyz)
        triangles = ' '.join(str(v) for face in faces for v in face)
        rotor_sites = ''.join(f'<site name="rotor{i}" pos="{p[0]} {p[1]} 0" size=".015" rgba=".2 .3 .4 1"/>' for i,p in enumerate(ROTOR_POSITIONS))
        rotor_geoms = ''.join(f'<geom type="capsule" fromto="0 0 0 {p[0]} {p[1]} 0" size=".012" mass="0" rgba=".18 .2 .25 1"/><geom type="cylinder" pos="{p[0]} {p[1]} .013" size=".055 .004" mass="0" rgba=".1 .12 .14 1"/>' for p in ROTOR_POSITIONS)
        actuators = ''.join(f'<general name="motor{i}" site="rotor{i}" gear="0 0 1 0 0 {YAW_MOMENT_PER_N*spin}" ctrllimited="true" ctrlrange="0 {MAX_MOTOR_FORCE_N}"/>' for i,spin in enumerate(ROTOR_SPINS))
        self.mjcf = f'''<mujoco model="quad_camera_fixture">
  <compiler angle="radian"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -{GRAVITY_M_S2}" integrator="RK4"/>
  <statistic extent="8" center="3 0 1.5"/>
  <visual><global offwidth="{max(image_size,640)}" offheight="{max(image_size,480)}"/><map znear=".005" zfar="5"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .67 .8" rgb2=".88 .91 .94" width="256" height="1536"/>
    <texture name="person_photo" type="2d" file="{escape(str(self.patch_path), {'"':'&quot;'})}"/>
    <material name="photo" texture="person_photo" texuniform="false" reflectance="0" specular="0" shininess="0" emission=".55"/>
    <mesh name="person_board" vertex="{verts}" face="{triangles}" texcoord="0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 7" dir="0 0 -1" diffuse=".7 .7 .7" ambient=".45 .45 .45" castshadow="false"/>
    <geom name="floor" type="plane" size="15 8 .1" rgba=".52 .55 .57 1" friction=".8 .02 .002"/>
    <geom name="ceiling" type="box" pos="3 0 4" size="9 6 .1" rgba=".76 .79 .81 1"/>
    <geom name="end_wall" type="box" pos="12 0 2" size=".1 6 2" rgba=".63 .69 .73 1"/>
    <geom name="left_wall" type="box" pos="3 6 2" size="9 .1 2" rgba=".60 .66 .70 1"/>
    <geom name="right_wall" type="box" pos="3 -6 2" size="9 .1 2" rgba=".60 .66 .70 1"/>
    <body name="target" mocap="true" pos="5 0 .95">
      <geom name="target_board" type="mesh" mesh="person_board" material="photo"/>
    </body>
    <body name="obstacle" mocap="true" pos="40 0 .7">
      <geom name="obstacle_box" type="box" size=".10 .25 .7" rgba=".9 .27 .1 1"/>
    </body>
    <body name="quad" pos="0 0 1.1">
      <freejoint name="aircraft_free"/>
      <inertial pos="0 0 0" mass="{MASS_KG}" diaginertia="{' '.join(map(str,INERTIA_KG_M2))}"/>
      <geom name="fuselage" type="box" size=".11 .075 .03" mass="0" rgba=".1 .23 .35 1"/>
      {rotor_geoms}{rotor_sites}
      <camera name="front" pos=".25 0 0" xyaxes="0 -1 0 0 0 1" fovy="{CAMERA_FOV_DEG}"/>
      <camera name="safety" pos=".25 0 0" xyaxes="0 -1 0 0 0 1" fovy="{SAFETY_FOV_DEG}"/>
    </body>
    <camera name="overview" pos="-3 -6 4" xyaxes=".8 -.6 0 .3 .4 .8660254"/>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>'''
        self.model = self.mj.MjModel.from_xml_string(self.mjcf)
        self.data = self.mj.MjData(self.model)
        self._quad_id = self.model.body('quad').id
        self._target_mocap = int(self.model.body_mocapid[self.model.body('target').id])
        self._obstacle_mocap = int(self.model.body_mocapid[self.model.body('obstacle').id])
        self._renderers = {}
        self._target_position = np.array([5.,0.,.95])
        self._target_hidden = False
        self._obstacle_position = np.array([40.,0.,.7])
        self._obstacle_enabled = False
        self.motor_forces = np.full(4, MASS_KG*GRAVITY_M_S2/4)
        self.reset()

    def reset(self, position=(0,0,1.1), euler=(0,0,0)):
        position, euler = _vector(position,'position'), _vector(euler,'euler')
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = position
        quaternion = np.zeros(4)
        self.mj.mju_mat2Quat(quaternion, rotation_from_euler(*euler).ravel())
        self.data.qpos[3:7] = quaternion
        # Rotor pre-spin is an explicitly initialized hover fixture, not thrust
        # instantly created by the autopilot after loss of power.
        self.motor_forces[:] = MASS_KG*GRAVITY_M_S2/4
        self.data.ctrl[:] = self.motor_forces
        self.data.mocap_pos[self._target_mocap] = self._target_position if not self._target_hidden else [40,40,.95]
        self.data.mocap_pos[self._obstacle_mocap] = self._obstacle_position if self._obstacle_enabled else [40,-40,.7]
        self.mj.mj_forward(self.model,self.data)
        return self.state()

    def state(self):
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model,self.data,self.mj.mjtObj.mjOBJ_BODY,self._quad_id,velocity,0)
        rotation = self.data.xmat[self._quad_id].reshape(3,3).copy()
        return FlightState(float(self.data.time), self.data.xpos[self._quad_id].copy(),
                           velocity[3:].copy(), rotation, rotation.T@velocity[:3], self.motor_forces.copy())

    def step(self, motor_targets):
        target = np.asarray(motor_targets, dtype=float)
        if target.shape != (4,) or not np.isfinite(target).all():
            raise ValueError('Four finite motor force targets are required')
        if np.any(target < 0) or np.any(target > MAX_MOTOR_FORCE_N):
            raise ValueError('Motor targets must be inside the physical actuator bounds')
        self.motor_forces += (1-np.exp(-PHYSICS_DT/MOTOR_TIME_CONSTANT_S))*(target-self.motor_forces)
        self.data.ctrl[:] = self.motor_forces
        self.mj.mj_step(self.model,self.data)
        # mj_step leaves position-dependent sensors at the last RK stage;
        # forward exposes the new state/camera consistently at data.time.
        self.mj.mj_forward(self.model,self.data)
        return self.state()

    def _renderer(self, key, height, width):
        if key not in self._renderers:
            self._renderers[key] = self.mj.Renderer(self.model,height=height,width=width)
        return self._renderers[key]

    def capture(self, safety=False):
        if type(safety) is not bool:
            raise ValueError('safety must be boolean')
        name = 'safety' if safety else 'front'
        size = SAFETY_IMAGE_SIZE if safety else self.image_size
        fov = SAFETY_FOV_DEG if safety else CAMERA_FOV_DEG
        renderer = self._renderer(name,size,size)
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data,camera=name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data,camera=name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        camera_id = self.model.camera(name).id
        rotation = self.data.cam_xmat[camera_id].reshape(3,3).copy()@np.diag([1.,-1.,-1.])
        focal = .5*size/np.tan(.5*np.deg2rad(fov))
        # Integer indices identify pixel centers; the principal point is at
        # (W-1)/2, while the image outer edges are -0.5 and W-0.5.
        return CameraFrame(rgb,depth,(focal,focal,(size-1)/2,(size-1)/2),rotation,
                           self.data.cam_xpos[camera_id].copy(),float(self.data.time)).validate()

    def overview(self):
        renderer = self._renderer('overview',480,640)
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data,camera='overview')
        return renderer.render().copy()

    def set_target(self, position, hidden=False):
        if type(hidden) is not bool:
            raise ValueError('hidden must be boolean')
        self._target_position = _vector(position,'target position').copy()
        self._target_hidden = hidden
        self.data.mocap_pos[self._target_mocap] = [40,40,.95] if hidden else self._target_position
        self.mj.mj_forward(self.model,self.data)

    def set_obstacle(self, position, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        self._obstacle_position = _vector(position,'obstacle position').copy()
        self._obstacle_enabled = enabled
        self.data.mocap_pos[self._obstacle_mocap] = self._obstacle_position if enabled else [40,-40,.7]
        self.mj.mj_forward(self.model,self.data)

    def truth(self):
        aircraft_geoms = set(np.flatnonzero(self.model.geom_bodyid == self._quad_id).tolist())
        contacts = []
        for contact in self.data.contact[:self.data.ncon]:
            if contact.geom1 in aircraft_geoms or contact.geom2 in aircraft_geoms:
                contacts.append({'geom1': self.model.geom(int(contact.geom1)).name,
                                 'geom2': self.model.geom(int(contact.geom2)).name,
                                 'distance_m': float(contact.dist)})
        corners = self._target_position + np.array([[0,self.target_half_width,-.95], [0,-self.target_half_width,-.95],
                                                    [0,-self.target_half_width,.95], [0,self.target_half_width,.95]])
        return {'target_position':self._target_position.tolist(),'target_hidden':self._target_hidden,
                'target_corners_world':corners.tolist(),'target_orientation':'fixed vertical front faces world -X',
                'obstacle_position':self._obstacle_position.tolist(),'obstacle_enabled':self._obstacle_enabled,
                'contacts':contacts,'contact_count':len(contacts),'time_s':float(self.data.time)}

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
