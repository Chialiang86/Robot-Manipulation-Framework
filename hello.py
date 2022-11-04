# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import pybullet_data
from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv
from pybullet_object_models import ycb_objects

import time
import os
import math as m
import numpy as np


def render(robot):
    pos, rot, _, _, _, _ = p.getLinkState(robot.robot_id, linkIndex=robot.end_eff_idx, computeForwardKinematics=True)
    rot_matrix = p.getMatrixFromQuaternion(rot)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # camera params
    height = 640
    width = 480
    fx, fy = 596.6278076171875, 596.6278076171875
    cx, cy = 311.98663330078125, 236.76170349121094
    near, far = 0.1, 10

    camera_vector = rot_matrix.dot((0, 0, 1))
    up_vector = rot_matrix.dot((0, -1, 0))

    camera_eye_pos = np.array(pos)
    camera_target_position = camera_eye_pos + 0.2 * camera_vector

    view_matrix = p.computeViewMatrix(camera_eye_pos, camera_target_position, up_vector)

    proj_matrix = (2.0 * fx / width, 0.0, 0.0, 0.0,
                   0.0, 2.0 * fy / height, 0.0, 0.0,
                   1.0 - 2.0 * cx / width, 2.0 * cy / height - 1.0, (far + near) / (near - far), -1.0,
                   0.0, 0.0, 2.0 * far * near / (near - far), 0.0)

    p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                     renderer=p.ER_BULLET_HARDWARE_OPENGL)  # renderer=self._p.ER_TINY_RENDERER)

def robot_key_callback(robot : pandaEnv, keys : dict, object_id : int=None):

    move_offset = 0.01
    rot_offset = 0.1
    ret = None

    # move up
    if 65297 in keys and keys[65297] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): # up arrow
        # quat_1 = p.getQuaternionFromEuler([m.pi, 0, 0])
        tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
        new_pos = (tmp_pos[0], tmp_pos[1], tmp_pos[2] + move_offset)
        robot.apply_action(new_pos)
        ret = 'up'
    # move down
    elif 65298 in keys and keys[65298] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): # down arrow
        tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
        new_pos = (tmp_pos[0], tmp_pos[1], tmp_pos[2] - move_offset)
        robot.apply_action(new_pos)
        ret = 'down'
    # move left
    elif 65295 in keys and keys[65295] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN):  # left arrow
        tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
        new_pos = (tmp_pos[0], tmp_pos[1] - move_offset, tmp_pos[2])
        robot.apply_action(new_pos)
        ret = 'left'
    # move right
    elif 65296 in keys and keys[65296] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): # right arrow
        tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
        new_pos = (tmp_pos[0], tmp_pos[1] + move_offset, tmp_pos[2])
        robot.apply_action(new_pos)
        ret = 'right'
    # move front
    elif ord('x') in keys and keys[ord('x')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
        tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
        new_pos = (tmp_pos[0] + move_offset, tmp_pos[1], tmp_pos[2])
        robot.apply_action(new_pos)
        ret = 'front'
    # move back
    elif ord('d') in keys and keys[ord('d')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
        tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
        new_pos = (tmp_pos[0] - move_offset, tmp_pos[1], tmp_pos[2])
        robot.apply_action(new_pos)
        ret = 'back'
    # flip flop gripper
    elif 32 in keys and keys[32] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
        gripper_pos = robot.get_gripper_pos()
        robot.grasp(object_id)
        if gripper_pos[0] < 0.02 or gripper_pos[1] < 0.02:
            robot.pre_grasp()
            for _ in range(100):
                p.stepSimulation()
                time.sleep(0.005)
            ret = 'open'
        else :
            robot.grasp(object_id)
            for _ in range(100):
                p.stepSimulation()
                time.sleep(0.005)
            ret = 'close'
    # rotate gripper : counter clockwise
    elif ord('z') in keys and keys[ord('z')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
        joint_name_ids = robot.get_joint_name_ids()
        tmp_pos = p.getJointState(robot.robot_id, joint_name_ids['panda_joint7'])[0]
        p.setJointMotorControlArray(robot.robot_id, [joint_name_ids['panda_joint7']], p.POSITION_CONTROL, targetPositions=[tmp_pos-rot_offset])
        ret = 'turn_cw'
    # flip flop gripper : clockwise
    elif ord('c') in keys and keys[ord('c')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
        joint_name_ids = robot.get_joint_name_ids()
        tmp_pos = p.getJointState(robot.robot_id, joint_name_ids['panda_joint7'])[0]
        p.setJointMotorControlArray(robot.robot_id, [joint_name_ids['panda_joint7']], p.POSITION_CONTROL, targetPositions=[tmp_pos+rot_offset])
        ret = 'turn_ccw'
    elif ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        ret = 'quit'
    return ret

def update_debug_param(robot : pandaEnv):

    p.removeAllUserParameters()

    param_ids = []
    joint_ids = []
    num_joints = p.getNumJoints(robot.robot_id)

    joint_states = p.getJointStates(robot.robot_id, range(0, num_joints))
    joint_poses = [x[0] for x in joint_states]

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot.robot_id, i)
        joint_name = joint_info[1]
        joint_type = joint_info[2]

        if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
            joint_ids.append(i)
            param_ids.append(
                p.addUserDebugParameter(joint_name.decode("utf-8"), joint_info[8], joint_info[9], joint_poses[i]))
    
    return param_ids

def main():
    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(2.1, 90, -30, [0.0, -0.0, -0.0])
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)

    # Load plane contained in pybullet_data
    planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    # Set gravity for simulation
    p.setGravity(0, 0, -9.8)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    robot = pandaEnv(physics_client_id, use_IK=1)
    p.stepSimulation()

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])
    obj_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "lego/lego.urdf"), [0.5, 0.0, 0.8])
    # obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbBanana', "model.urdf"), [0.5, 0.0, 0.8])

    # # Run the world for a bit
    # for _ in range(100):
    #     p.stepSimulation()

    # # ------------------ #
    # # --- Start Demo --- #
    # # ------------------ #

    # robot.pre_grasp()
    # render(robot)
    # p.stepSimulation()
    # time.sleep(sim_timestep)

    # # 1: go above the object
    # pos_1 = [0.5, 0.0, 0.9]
    # quat_1 = p.getQuaternionFromEuler([m.pi, 0, 0])

    # robot.apply_action(pos_1 + list(quat_1))

    # for _ in range(100):
    #     p.stepSimulation()
    #     render(robot)
    #     time.sleep(sim_timestep)

    # # 2: go down toward the object
    # pos_2 = [0.5, 0.0, 0.67]
    # quat_2 = p.getQuaternionFromEuler([m.pi, 0, 0])

    # robot.apply_action(pos_2 + list(quat_2), max_vel=5)
    # robot.pre_grasp()

    # for _ in range(200):
    #     p.stepSimulation()
    #     render(robot)
    #     time.sleep(sim_timestep)

    # # 3: close fingers
    # robot.grasp(obj_id)

    # for _ in range(120):
    #     p.stepSimulation()
    #     render(robot)
    #     time.sleep(sim_timestep)

    # # 4: go up
    # pos_4 = [0.5, 0.0, 0.9]
    # quat_4 = p.getQuaternionFromEuler([m.pi, 0, 0])

    # robot.apply_action(pos_4 + list(quat_4), max_vel=5)
    # robot.grasp(obj_id)

    # for _ in range(200):
    #     p.stepSimulation()
    #     render(robot)
    #     time.sleep(sim_timestep)

    # # 6: open hand
    # robot.pre_grasp()

    # for _ in range(50):
    #     p.stepSimulation()
    #     render(robot)
    #     time.sleep(sim_timestep)

    # ------------------------ #
    # --- Play with joints --- #
    # ------------------------ #

    param_ids = []
    joint_ids = []
    num_joints = p.getNumJoints(robot.robot_id)

    joint_states = p.getJointStates(robot.robot_id, range(0, num_joints))
    joint_poses = [x[0] for x in joint_states]
    idx = 0
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot.robot_id, i)
        joint_name = joint_info[1]
        joint_type = joint_info[2]

        if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
            joint_ids.append(i)
            param_ids.append(
                p.addUserDebugParameter(joint_name.decode("utf-8"), joint_info[8], joint_info[9], joint_poses[i]))
            idx += 1

    close_flag = False
    while True:
        
        # key callback
        keys = p.getKeyboardEvents()
        if keys != {} :
            
            action = robot_key_callback(robot, keys, obj_id)
            if action == 'quit':
                break
            elif action == 'close':
                close_flag = True
            elif action == 'open':
                close_flag = False
            
            if close_flag:
                robot.grasp(obj_id)
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(0.005)

                
            param_ids = update_debug_param(robot)

        else :
            new_pos = []
            for i in param_ids:
                new_pos.append(p.readUserDebugParameter(i))
            p.setJointMotorControlArray(robot.robot_id, joint_ids, p.POSITION_CONTROL, targetPositions=new_pos)

        p.stepSimulation()
        time.sleep(sim_timestep)


if __name__ == '__main__':
    main()