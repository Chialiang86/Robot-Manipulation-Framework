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
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose

from fk import your_fk, get_panda_DH_params

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

    move_offset = 0.005
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

def main(args):

    assert args.type == 0 or args.type == 1, f'type id should be 0 (fk) or 1 (ik), but get {args.type}'

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

    table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.9, 0.0, 0.1])


    # object
    urdf_path = args.obj_urdf
    obj_name = urdf_path.split('/')[-2]
    assert os.path.exists(urdf_path), f'{urdf_path} not exists'
    obj_id = p.loadURDF(urdf_path)
    p.resetBasePositionAndOrientation(obj_id, [0.35, 0.1, 0.78], R.from_rotvec([np.pi / 2, 0, 0]).as_quat())
   
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

    joint_poses_old = [0, 0, 0, 0, 0, 0, 0]

    fk_testcase_max = 1000
    fk_dict = {
        'joint_poses': [],
        'poses': [],
        'jacobian': []
    }

    ik_testcase_max = 100
    ik_dict = {
        'current_joint_poses': [],
        'next_poses': []
    }

    dh_params = get_panda_DH_params()

    while True:
        
        # key callback
        keys = p.getKeyboardEvents()
        if keys != {} :
            
            if ord('a') in keys and keys[ord('a')] & (p.KEY_WAS_TRIGGERED): 
                
                param_control = not param_control
                param_ids = update_debug_param(robot)

            elif ord('p') in keys and keys[ord('p')] & (p.KEY_WAS_TRIGGERED): 
                
                param_control = not param_control
                param_ids = update_debug_param(robot)

                obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
                obj_pose = obj_pos + obj_rot
                obj_trans = get_matrix_from_pose(obj_pose)

                gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
                gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
                gripper_pose = list(gripper_pos) + list(gripper_rot)
                gripper_trans = get_matrix_from_pose(gripper_pose)

                obj2gripper_trans = np.linalg.inv(obj_trans) @ gripper_trans

                grasping_dict = {
                    'obj2gripper': obj2gripper_trans.tolist()
                }

                grasping_fname = f'3d_models/objects/{obj_name}/grasping.json'
                f = open(grasping_fname, 'w')
                json.dump(grasping_dict, f, indent=4)
                f.close()
                break

            else:
                action = robot_key_callback(robot, keys, obj_id)
                if action == 'quit':
                    break
                elif action == 'close':
                    close_flag = True
                elif action == 'open':
                    close_flag = False
                
                if close_flag:
                    robot.grasp(obj_id)
                    for _ in range(5):
                        p.stepSimulation()
                
                # get joint angles
                joint_states = p.getJointStates(robot.robot_id, range(0, 7))
                joint_poses = [x[0] for x in joint_states]

                # get 7d pose
                gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
                gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
                gripper_pose = list(gripper_pos) + list(gripper_rot)

                new_thresh =    (0.025 if 'easy' in args.difficulty \
                                else 0.05 if 'medium' in args.difficulty \
                                else 0.1) \
                            if args.type == 0 \
                            else \
                                (0.1 if 'easy' in args.difficulty \
                                else 0.2 if 'medium' in args.difficulty \
                                else 0.3) 

                if np.linalg.norm((np.asarray(joint_poses_old) - np.asarray(joint_poses)), ord=2) > new_thresh:

                    if args.type == 0: # fk
                        fk_dict['joint_poses'].append(joint_poses)
                        fk_dict['poses'].append(gripper_pose)
                        _, jacobian = your_fk(robot, dh_params, joint_poses)
                        fk_dict['jacobian'].append(jacobian.tolist())
                        fk_testcase_num = len(fk_dict['joint_poses'])
                        print(f'fk testcase num = {fk_testcase_num} / {fk_testcase_max}')
                        if fk_testcase_num == fk_testcase_max:
                            f = open(f'test_case/fk_testcase_{args.difficulty}.json', 'w')
                            json.dump(fk_dict, f, indent=4)
                            f.close()
                            break
                    
                    if args.type == 1: # ik
                        ik_dict['current_joint_poses'].append(joint_poses_old)
                        ik_dict['next_poses'].append(gripper_pose)
                        ik_testcase_num = len(ik_dict['current_joint_poses'])
                        if ik_testcase_num > 0:
                            print(f'ik testcase num = {ik_testcase_num - 1} / {ik_testcase_max}')
                        if ik_testcase_num - 1 == ik_testcase_max:
                            ik_dict['current_joint_poses'] = ik_dict['current_joint_poses'][1:]
                            ik_dict['next_poses'] = ik_dict['next_poses'][1:]
                            f = open(f'test_case/ik_testcase_{args.difficulty}.json', 'w')
                            json.dump(ik_dict, f, indent=4)
                            f.close()
                            break

                    joint_poses_old = joint_poses


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
    parser.add_argument('--type', '-t', type=int, default=0, help='0 for fk, 1 for ik')
    parser.add_argument('--obj-urdf', '-urdf', type=str, default='3d_models/objects/mug_67/base.urdf')
    args = parser.parse_args()
    main(args)
