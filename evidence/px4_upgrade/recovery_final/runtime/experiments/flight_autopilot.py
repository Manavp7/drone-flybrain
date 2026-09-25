"""Conventional velocity/altitude and SO(3) attitude stabilization.

This controller is engineered, not biological. It has no target coordinates,
images, depth maps or teleportation API: its output is four bounded rotor forces.
"""
from __future__ import annotations
import numpy as np
from experiments.flight_contracts import FlightState
from experiments.flight_world import (MASS_KG, INERTIA_KG_M2, ROTOR_POSITIONS,
                                     ROTOR_SPINS, YAW_MOMENT_PER_N,
                                     MAX_MOTOR_FORCE_N, GRAVITY_M_S2)

MIXER = np.vstack([np.ones(4),ROTOR_POSITIONS[:,1],-ROTOR_POSITIONS[:,0],
                   YAW_MOMENT_PER_N*ROTOR_SPINS])
MIXER_INVERSE = np.linalg.inv(MIXER)


def _vee(matrix):
    return np.array([matrix[2,1],matrix[0,2],matrix[1,0]])


class Autopilot:
    """Ideal 200 Hz state feedback; zero desired speed actively brakes/holds.

    Commands are horizontal speed along the current body heading, target heading in radians,
    and absolute altitude in metres. The controller never changes FlightState.
    """
    def command(self, state: FlightState, forward_speed, yaw_target, altitude=1.1):
        request = np.asarray([forward_speed,yaw_target,altitude],dtype=float)
        if not np.isfinite(request).all() or not 0 <= forward_speed <= 1. or not .35 <= altitude <= 3.:
            raise ValueError('Invalid bounded forward speed, heading, or altitude')
        if not all(np.isfinite(a).all() for a in (state.position,state.velocity,state.rotation,state.angular_velocity)):
            raise ValueError('Nonfinite aircraft estimate')
        desired_velocity = float(forward_speed)*np.array([np.cos(state.yaw),np.sin(state.yaw)])
        acceleration_xy = 2.8*(desired_velocity-state.velocity[:2])
        acceleration_xy *= min(1.,2./max(np.linalg.norm(acceleration_xy),1e-12))
        acceleration_z = float(np.clip(5.*(altitude-state.position[2])-3.8*state.velocity[2],-4.,4.))
        desired_force = MASS_KG*np.array([*acceleration_xy,GRAVITY_M_S2+acceleration_z])
        z_axis = desired_force/np.linalg.norm(desired_force)
        x_heading = np.array([np.cos(yaw_target),np.sin(yaw_target),0.])
        y_axis = np.cross(z_axis,x_heading)
        y_axis /= np.linalg.norm(y_axis)
        desired_rotation = np.column_stack([np.cross(y_axis,z_axis),y_axis,z_axis])
        rotation = state.rotation
        rotation_error = .5*_vee(desired_rotation.T@rotation-rotation.T@desired_rotation)
        omega = state.angular_velocity
        torque = -np.array([.5,.5,.16])*rotation_error-np.array([.12,.12,.08])*omega
        torque += np.cross(omega,INERTIA_KG_M2*omega)
        collective = float(desired_force@rotation[:,2])
        # Symmetric torque desaturation preserves collective hover support.
        collective = np.clip(collective,0.,4*MAX_MOTOR_FORCE_N)
        base = np.full(4,collective/4)
        differential = MIXER_INVERSE@np.r_[0.,torque]
        factor = 1.
        for center,delta in zip(base,differential):
            if delta > 0:
                factor = min(factor,(MAX_MOTOR_FORCE_N-center)/delta)
            elif delta < 0:
                factor = min(factor,-center/delta)
        return np.clip(base+max(0.,factor)*differential,0.,MAX_MOTOR_FORCE_N)
