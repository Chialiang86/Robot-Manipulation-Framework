# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.
import os, inspect
import argparse
import json
import time
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import pybullet as p
import pybullet_data

# for motion planners
from utils.motion_planning_utils import get_collision7d_fn

# for robot control
# from pybullet_planning.interfaces.robots.joint import get_custom_limits
from pybullet_robot_envs.panda_envs.panda_env import pandaEnv


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


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


def robot_key_callback(robot : pandaEnv, keys : dict, object_id : int=None):

    move_offset = 0.01
    rot_offset = 0.02
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


def refine_tgt_obj_pose(physicsClientId, body, obstacles=[]):
    collision7d_fn = get_collision7d_fn(physicsClientId, body, obstacles=obstacles)

    low_limit = [-0.005, -0.005, -0.005, -np.pi / 180, -np.pi / 180, -np.pi / 180]
    high_limit = [ 0.005,  0.005,  0.005,  np.pi / 180,  np.pi / 180,  np.pi / 180]

    obj_pos, obj_rot = p.getBasePositionAndOrientation(body)
    original_pose = np.asarray(obj_pos + obj_rot)
    refine_pose = original_pose
    while collision7d_fn(tuple(refine_pose)):
        refine_pose6d = np.concatenate((np.asarray(obj_pos), R.from_quat(obj_rot).as_rotvec())) + np.random.uniform(low_limit, high_limit)
        refine_pose = np.concatenate((refine_pose6d[:3], R.from_rotvec(refine_pose6d[3:]).as_quat()))
        # print(refine_pose)
    return refine_pose

def main():

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.7, 0.0, 1.0]
    )
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

    num_joints = p.getNumJoints(robot.robot_id)
    p.stepSimulation()

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])
   
    # grasping
    # robot.apply_action(contact_info['object_pose'])
    sim_timestep = 1.0 / 240.0
    p.setTimeStep(sim_timestep)

    # init_pose = (0.38231226801872253, -0.03931187838315964, 1.2498921155929565, 0.7112785577774048, -0.0645006000995636, 0.6992669105529785, 0.030794139951467514)
    init_pose = (0.437650821912322, 0.055437112660847596, 0.9476674736905905, 0.9959291764453291, -0.06561519070877571, 0.05020414209061421, 0.03604533770302429)
    robot.apply_action(init_pose)
    for i in range(100):
        p.stepSimulation()
        time.sleep(sim_timestep)
  
    # if manual contral needed
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
        print(f'id = {i} : {joint_name}')

        if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
            joint_ids.append(i)
            param_ids.append(
                p.addUserDebugParameter(joint_name.decode("utf-8"), joint_info[8], joint_info[9], joint_poses[i]))
            idx += 1
    
    print(f'joint ids = {joint_ids}')

    close_flag = False
    param_control = True
    testcase_max = 1000
    # max_cnt = 3
    # cnt = 0

    joint_poses_old = [0, 0, 0, 0, 0, 0, 0]

    fk_dict = {
        'joint_poses': [],
        'poses': []
    }

    while True:
        
        # key callback
        keys = p.getKeyboardEvents()
        if keys != {} :
            
            if ord('a') in keys and keys[ord('a')] & (p.KEY_WAS_TRIGGERED): 
                
                param_control = not param_control
                param_ids = update_debug_param(robot)

            else:
                action = robot_key_callback(robot, keys, -1)
                if action == 'quit':
                    break
                elif action == 'close':
                    close_flag = True
                elif action == 'open':
                    close_flag = False
                
                if close_flag:
                    robot.grasp(-1)
                    for _ in range(5):
                        p.stepSimulation()
                
                # get joint angles
                joint_states = p.getJointStates(robot.robot_id, range(0, 7))
                joint_poses = [x[0] for x in joint_states]

                # get 7d pose
                gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
                gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
                gripper_pose = list(gripper_pos) + list(gripper_rot)

                if np.linalg.norm((np.asarray(joint_poses_old) - np.asarray(joint_poses)), ord=2) > 0.1:
                    # print(f'joint_poses = {joint_poses}, gripper_pose = {gripper_pose}')
                    fk_dict['joint_poses'].append(joint_poses)
                    fk_dict['poses'].append(gripper_pose)
                    joint_poses_old = joint_poses

                    testcase_num = len(fk_dict['joint_poses'])
                    print(f'testcase num = {testcase_num} / {testcase_max}')
                    if testcase_num == 1000:
                        f = open('fk_testcase.json', 'w')
                        json.dump(fk_dict, f, indent=4)
                        f.close()
                        break


        elif param_control:
            new_pos = []
            for i in param_ids:
                new_pos.append(p.readUserDebugParameter(i))
            p.setJointMotorControlArray(robot.robot_id, joint_ids, p.POSITION_CONTROL, targetPositions=new_pos)

        p.stepSimulation()
        time.sleep(sim_timestep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', '-d', type=str, default='easy')
    args = parser.parse_args()
    main()
