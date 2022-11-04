import os
import argparse
import json
import time
import quaternion
import scipy.io as sio
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import pybullet as p
import pybullet_data

# for robot control
from utils.bullet_utils import draw_coordinate, xyzw2wxyz, wxyz2xyzw, get_matrix_from_pos_rot, get_pos_rot_from_matrix
from pybullet_robot_envs.panda_envs.panda_env import pandaEnv

def get_dense_waypoints(start_config : list or tuple or np.ndarray, end_config : list or tuple or np.ndarray, resolution : float=0.005):

    assert len(start_config) == 7 and len(end_config) == 7

    d12 = np.asarray(end_config[:3]) - np.asarray(start_config[:3])
    steps = int(np.ceil(np.linalg.norm(np.divide(d12, resolution), ord=2)))
    obj_init_quat = quaternion.as_quat_array(xyzw2wxyz(start_config[3:]))
    obj_tgt_quat = quaternion.as_quat_array(xyzw2wxyz(end_config[3:]))

    ret = []
    # plan trajectory in the same way in collision detection module
    for step in range(steps):
        ratio = (step + 1) / steps
        pos = ratio * d12 + np.asarray(start_config[:3])
        quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
        quat = wxyz2xyzw(quaternion.as_float_array(quat))
        position7d = tuple(pos) + tuple(quat)
        ret.append(position7d)

    return ret

def main(args):
    
    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    sim_timestep = 1.0 / 240
    physics_client_id = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.3,
        cameraYaw=90,
        cameraPitch=0,
        # cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 1.0]
        # cameraTargetPosition=[0.5, 0.0, 1.3]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, -9.8)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.1])
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    
    # wall
    # wall_pos = [0.5, -0.11, 1.0]
    # wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # wall_id = p.loadURDF("models/wall/wall.urdf", wall_pos, wall_orientation)

    # ---------------------------- #
    # --- Load object and hook --- #
    # ---------------------------- #

    # random initial object pose

    # --------------------- #
    # --- Initial robot --- #
    # --------------------- #

    robot = pandaEnv(physics_client_id, use_IK=1)
    gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    gripper_pose = list(gripper_pos) + list(gripper_rot)
    before_pose = [0.49670371413230896, 0.03478135168552399, 0.92857426404953, 
                   0.9896430969238281, -0.07126988470554352, 0.12207289785146713, -0.02500670962035656]

    draw_coordinate(before_pose)

    waypoints = get_dense_waypoints(gripper_pose, before_pose, resolution=0.005)
    for waypoint in waypoints:
        robot.apply_action(waypoint, max_vel=-1)
        for _ in range(10): 
            p.stepSimulation()
            time.sleep(sim_timestep)

    while True:
        # key callback
        keys = p.getKeyboardEvents()            
        if ord('q') in keys and keys[ord('q')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='3d_models/hanging_pose/Hook_60-mug.json')
    args = parser.parse_args()
    main(args)