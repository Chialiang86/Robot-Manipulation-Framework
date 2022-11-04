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
from PIL import Image
import sys
import quaternion

# for motion planners
from utils.motion_planning_utils import get_sample7d_fn, get_distance7d_fn, get_extend7d_fn, get_collision7d_fn
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import plan_joint_motion, check_initial_end
from pybullet_planning.interfaces.control.control import trajectory_controller
from pybullet_planning.motion_planners.rrt_connect import birrt

# for robot control
from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

def load_obj_urdf(urdf_path, pos=[0, 0, 0], rot=[0, 0, 0]):

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    center = np.array([float(i) for i in root[0].find(
        "inertial").find("origin").attrib['xyz'].split(' ')])
    
    obj_id = -1
    if len(rot) == 3:
        obj_id = p.loadURDF(urdf_path, pos, p.getQuaternionFromEuler(rot))
    elif len(rot) == 4:
        obj_id = p.loadURDF(urdf_path, pos, rot)
        
    return obj_id, center

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

def xyzw2wxyz(quat : np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

def get_matrix_from_pos_rot(pos : list or tuple, rot : list or tuple):
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    if len(rot) == 3:
        rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4:
        rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def get_pos_rot_from_matrix(pose : np.ndarray):
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, rot

def robot_apply_action(robot : pandaEnv, obj_id : int, action : tuple or list, gripper_action : str = 'nop', 
                        sim_timestep : float = 1.0 / 240.0, diff_thresh : float = 0.005, max_vel : float = 0.2, max_iter = 5000):

    assert gripper_action in ['nop', 'pre_grasp', 'grasp']

    if gripper_action == 'nop':
        assert len(action) == 7, 'action length should be 7'

        robot.apply_action(action, max_vel=max_vel)
        diff = 10.0
        iter = 0
        while diff > diff_thresh and iter < max_iter:       
            iter += 1

            p.stepSimulation()
            time.sleep(sim_timestep)

            tmp_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4] # position
            tmp_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5] # rotation
            diff = np.sum((np.array(tmp_pos + tmp_rot) - np.array(action)) ** 2) ** 0.5

    elif gripper_action == 'pre_grasp' :

        robot.pre_grasp()
        for _ in range(int(1.0 / sim_timestep) * 1): # 1 sec
            p.stepSimulation()
            time.sleep(sim_timestep)
    else:

        robot.grasp(obj_id)
        for _ in range(int(1.0 / sim_timestep)): # 1 sec
            p.stepSimulation()
            time.sleep(sim_timestep)

def get_target_gripper_pose(robot : pandaEnv, obj_id : int, tgt_obj_pos : tuple or list, tgt_obj_rot : tuple or list,):

    # extract current object pose
    obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
    origin_obj_pose = get_matrix_from_pos_rot(obj_pos, obj_rot)

    # extract current gripper pose
    gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    origin_gripper_pose = get_matrix_from_pos_rot(gripper_pos, gripper_rot)

    # extract target pose
    tgt_obj_pose = get_matrix_from_pos_rot(tgt_obj_pos, tgt_obj_rot)

    # relative transform from object to gripper
    # see this stack-overflow issue : https://stackoverflow.com/questions/67001118/relative-transform
    obj2gripper_pose = np.linalg.inv(origin_obj_pose) @ origin_gripper_pose

    # target gripper pose by relative transform
    tgt_gripper_pose = tgt_obj_pose @ obj2gripper_pose

    # extract robot action (3 dim for position and 4 dim for rotation)
    tgt_gripper_pos = tgt_gripper_pose[:3, 3]
    tgt_gripper_rot = R.from_matrix(tgt_gripper_pose[:3, :3]).as_quat()

    return tuple(tgt_gripper_pos), tuple(tgt_gripper_rot)

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

def draw_coordinate(pose : np.ndarray or tuple or list, size : float = 0.02):
    assert (type(pose) == np.ndarray and pose.shape == (4, 4)) or len(pose) == 7

    if len(pose) == 7:
        pose = get_matrix_from_pos_rot(pose[:3], pose[3:])

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, [1, 0, 0])
    p.addUserDebugLine(origin, y, [0, 1, 0])
    p.addUserDebugLine(origin, z, [0, 0, 1])

def draw_bbox(start : list or tuple or np.ndarray,
              end : list or tuple or np.ndarray):
    
    assert len(start) == 3 and len(end) == 3, f'infeasible size of position, len(position) must be 3'

    points_bb = [
        [start[0], start[1], start[2]],
        [end[0], start[1], start[2]],
        [end[0], end[1], start[2]],
        [start[0], end[1], start[2]],
        [start[0], start[1], end[2]],
        [end[0], start[1], end[2]],
        [end[0], end[1], end[2]],
        [start[0], end[1], end[2]],
    ]

    for i in range(4):
        p.addUserDebugLine(points_bb[i], points_bb[(i + 1) % 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i + 4], points_bb[(i + 1) % 4 + 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i], points_bb[i + 4], [1, 0, 0])

def hanging_by_ik(robot : pandaEnv, obj_id : int, target_conf: tuple or list, 
                    sim_timestep : float = 1.0 / 240.0, max_vel : float = 0.2):
    
    tgt_gripper_pos, tgt_gripper_rot = get_target_gripper_pose(robot, obj_id, target_conf[:3], target_conf[3:])

    # execution step 1 : go to the pose that nearby the target pose
    tgt_gripper_pre_pos = (tgt_gripper_pos[0], tgt_gripper_pos[1] + 0.1, tgt_gripper_pos[2])
    action = tgt_gripper_pre_pos + tgt_gripper_rot
    robot_apply_action(robot, obj_id, action, gripper_action='nop', 
        sim_timestep=sim_timestep, diff_thresh=0.005, max_vel=max_vel)

    # execution step 2 : go to the target pose
    # tgt_gripper_pos = (tgt_gripper_pos[0], tgt_gripper_pos[1] - 0.02, tgt_gripper_pos[2])
    action = tgt_gripper_pos + tgt_gripper_rot
    robot_apply_action(robot, obj_id, action, gripper_action='nop', 
        sim_timestep=sim_timestep, diff_thresh=0.005, max_vel=max_vel)

    # execution step 3 : release the gripper
    robot_apply_action(robot, obj_id, action, gripper_action='pre_grasp', 
        sim_timestep=sim_timestep, diff_thresh=0.005, max_vel=max_vel)

    # execution step 4 : go to the ending pose
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    gripper_rot_matrix = R.from_quat(gripper_rot).as_matrix()
    ending_gripper_pos = np.asarray(tgt_gripper_pos) + (gripper_rot_matrix @ np.array([[0], [0], [-0.2]])).reshape(3)
    action = tuple(ending_gripper_pos) + tgt_gripper_rot
    robot_apply_action(robot, obj_id, action, gripper_action='nop', 
        sim_timestep=sim_timestep, diff_thresh=0.005, max_vel=max_vel)

def rrt_connect_7d(physics_client_id, obj_id, start_conf, target_conf, 
                    obstacles : list = [], sim_timestep : float = 1.0 / 240.0, max_vel : float = 0.2, diagnosis=False, **kwargs):

    # https://github.com/yijiangh/pybullet_planning/blob/dev/src/pybullet_planning/motion_planners/rrt_connect.py

    # config sample location space
    start_pos = start_conf[:3]
    target_pos = target_conf[:3]
    
    low_limit = [0, 0, 0]
    high_limit = [0, 0, 0]
    padding_low  = [0.01, 0.01, 0.00]
    padding_high = [0.01, 0.1, 0.05]

    # xlim, ylim
    for i in range(3):
        if start_pos[i] < target_pos[i]:
            low_limit[i] = start_pos[i] - padding_low[i]
            high_limit[i] = target_pos[i] + padding_high[i] 
        else :
            low_limit[i] = target_pos[i] - padding_low[i]
            high_limit[i] = start_pos[i] + padding_high[i] 
    
    draw_bbox(low_limit, high_limit)

    sample7d_fn = get_sample7d_fn(target_conf, low_limit, high_limit)
    distance7d_fn = get_distance7d_fn()
    extend7d_fn = get_extend7d_fn(resolution=0.0005)
    collision_fn = get_collision7d_fn(physics_client_id, obj_id, obstacles=obstacles)

    if not check_initial_end(start_conf, target_conf, collision_fn, diagnosis=diagnosis):
        return None

    return birrt(start_conf, target_conf, distance7d_fn, sample7d_fn, extend7d_fn, collision_fn, **kwargs)

def hanging_by_rrt(physics_client_id : int, robot : pandaEnv, obj_id : int, target_conf : list or tuple, 
                    obstacles : list=[], sim_timestep : float=1.0 / 240.0, max_vel : float=0.2):

    obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
    start_conf = obj_pos + obj_rot
    waypoints, nodes = rrt_connect_7d(physics_client_id, obj_id, start_conf=start_conf, target_conf=target_conf, obstacles=obstacles)
    
    if waypoints is None:
        print("Oops, no solution!")
        return

    # visualization all nodes
    # for node in nodes:
    #     draw_coordinate(node.config)

    # extract current gripper pose
    gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    origin_gripper_pose = get_matrix_from_pos_rot(gripper_pos, gripper_rot)

    origin_obj_pose = get_matrix_from_pos_rot(start_conf[:3], start_conf[3:])

    # relative transform from object to gripper
    # see this stack-overflow issue : https://stackoverflow.com/questions/67001118/relative-transform
    obj2gripper_pose = np.linalg.inv(origin_obj_pose) @ origin_gripper_pose
    planning_resolution = 0.001

    # reset obj pose
    p.resetBasePositionAndOrientation(obj_id, start_conf[:3], start_conf[3:])

    imgs = []
    # execution step 1 : execute RRT trajectories
    for i in range(len(waypoints) - 1):

        d12 = np.asarray(waypoints[i+1][:3]) - np.asarray(waypoints[i][:3])
        r12 = np.asarray(waypoints[i+1][3:]) - np.asarray(waypoints[i][3:])
        diff_q1_q2 = np.concatenate((d12, r12))
        steps = int(np.ceil(np.linalg.norm(np.divide(diff_q1_q2, planning_resolution), ord=2)))
        obj_init_quat = quaternion.as_quat_array(xyzw2wxyz(waypoints[i][3:]))
        obj_tgt_quat = quaternion.as_quat_array(xyzw2wxyz(waypoints[i+1][3:]))

        # plan trajectory in the same way in collision detection module
        for step in range(steps):
            ratio = (step + 1) / steps
            pos = ratio * d12 + np.asarray(waypoints[i][:3])
            quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
            quat = wxyz2xyzw(quaternion.as_float_array(quat))
            positions7d = tuple(pos) + tuple(quat)
            # p.resetBasePositionAndOrientation(obj_id, positions7d[:3], positions7d[3:])
            # draw_coordinate(positions7d)

            gripper_pose = get_matrix_from_pos_rot(positions7d[:3], positions7d[3:]) @ obj2gripper_pose
            gripper_pos, gripper_rot = get_pos_rot_from_matrix(gripper_pose)
            gripper_action = np.concatenate((gripper_pos, gripper_rot))

            robot.apply_action(gripper_action)
            p.stepSimulation()
            
            robot.grasp()
            for _ in range(10): # 1 sec
                p.stepSimulation()
                time.sleep(sim_timestep)

            # img = p.getCameraImage(480, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            # imgs.append(img)
    
    # execution step 2 : release gripper
    robot_apply_action(robot, obj_id, waypoints[-1], gripper_action='pre_grasp', 
        sim_timestep=0.05, diff_thresh=0.01, max_vel=-1, max_iter=100)

    # execution step 3 : go to the ending pose
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    gripper_rot_matrix = R.from_quat(gripper_rot).as_matrix()
    ending_gripper_pos = np.asarray(waypoints[-1][:3]) + (gripper_rot_matrix @ np.array([[0], [0], [-0.2]])).reshape(3)
    action = tuple(ending_gripper_pos) + tuple(gripper_rot)
    robot_apply_action(robot, obj_id, action, gripper_action='nop', 
        sim_timestep=0.05, diff_thresh=0.005, max_vel=-1, max_iter=100)

    return [Image.fromarray(img) for img in imgs]

# def gripper_motion_planning(robot : pandaEnv, tgt_gripper_pos : tuple or list, tgt_gripper_rot : tuple or list, 
#                     obstacles : list = [], sim_timestep : float = 1.0 / 240.0, max_vel : float = 0.2):

#     # https://github.com/yijiangh/pybullet_planning/blob/dev/src/pybullet_planning/motion_planners/rrt_connect.py

#     original_gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
#     original_gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]

#     # num_joints = p.getNumJoints(robot.robot_id)
#     target_conf = robot.get_target_joint_conf(list(tgt_gripper_pos) + list(tgt_gripper_rot))
#     joints = list(range(0, len(target_conf) - 2)) + [9, 10] # joint 1-7 with gripper [9, 10]
#     birrt_res = plan_joint_motion(robot.robot_id, joints, target_conf, obstacles=obstacles, verbose=True)

#     if birrt_res is None:
#         print("Oops, no solution!")
#         return
    
#     # draw waypoints
#     for step, waypoint in enumerate(birrt_res):
#         print(f'{step + 1}/{len(birrt_res)}')
#         for i in range(0, len(waypoint)):
#             p.resetJointState(robot.robot_id, joints[i], waypoint[i], targetVelocity=0)
        
#         gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
#         gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
#         draw_coordinate(gripper_pos + gripper_rot, size=0.1)

#     # reset to original gripper configuration
#     original_conf = robot.get_target_joint_conf(list(original_gripper_pos) + list(original_gripper_rot))
#     for i in range(0, len(original_conf)):
#         p.resetJointState(robot.robot_id, joints[i], original_conf[i], targetVelocity=0.2)
    
#     # the acurate way to execute trajectory_controller
#     # damn! I'm genuous!
#     points = trajectory_controller(robot.robot_id, joints, birrt_res)
#     for point in points:
#         p.stepSimulation()
#         time.sleep(sim_timestep)

#     diff_thresh = 0.01
#     max_iter = 3000
#     for step, waypoint in enumerate(birrt_res):
#         print(f'{step + 1}/{len(birrt_res)}')
#         for i in range(0, len(waypoint)):
#             p.setJointMotorControl2(bodyUniqueId=robot.robot_id,
#                                     jointIndex=joints[i],
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=waypoint[i],
#                                     maxVelocity=max_vel)
#         diff = 1.0
#         iter = 0
#         while diff > diff_thresh and iter < max_iter:
#             iter += 1
#             joint_states = p.getJointStates(robot.robot_id, range(0, len(waypoint))) # only jointPosition needed
#             joint_poses = [x[0] for x in joint_states]
#             diff = np.sum((np.array(joint_poses) - np.array(waypoint)) ** 2)
#             print(diff)
#             p.stepSimulation()
#             time.sleep(sim_timestep)


def main(args):

    # extract args
    input_json = args.input_json
    if not os.path.exists(input_json):
        print(f'{input_json} not exists')

    json_dict = None
    with open(input_json, 'r') as f:
        json_dict = json.load(f)
    
    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    # physics_client_id = p.connect(p.DIRECT)
    # p.resetDebugVisualizerCamera(2.1, 90, -30, [0.0, -0.0, -0.0])
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=90,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 1.3]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, -9.8)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # Load plane contained in pybullet_data
    planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    robot = pandaEnv(physics_client_id, use_IK=1)

    num_joints = p.getNumJoints(robot.robot_id)
    p.stepSimulation()

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])

    # wall
    wall_pos = [0.7, -0.3, 0.8]
    wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # wall_id = p.loadURDF("models/wall/wall.urdf", wall_pos, wall_orientation)
    
    # hook initialization
    hook_pos = json_dict['hook_pose'][:3]
    hook_orientation = json_dict['hook_pose'][3:]
    hook_id = p.loadURDF(json_dict['hook_path'], hook_pos, hook_orientation)

    # get target hanging pose
    assert len(json_dict['contact_info']) > 0, 'contact info is empty'
    contact_index = 0
    contact_info = json_dict['contact_info'][contact_index]
    tgt_obj_pos = contact_info['object_pose'][:3]
    tgt_obj_rot = contact_info['object_pose'][3:]
    obj_id_target, _ = load_obj_urdf(json_dict['obj_path'])
    p.resetBasePositionAndOrientation(obj_id_target, tgt_obj_pos, tgt_obj_rot)
    tgt_pose = refine_tgt_obj_pose(physics_client_id, obj_id_target, obstacles=[hook_id, planeId])
    p.removeBody(obj_id_target)

    # config output path
    output_root = os.path.split(args.input_json)[0]
    output_dir = os.path.splitext(os.path.split(args.input_json)[1])[0]
    os.makedirs(os.path.join(output_root, output_dir), exist_ok=True)
    output_dir = os.path.join(output_root, output_dir)

    # random position noise
    # low = [-0.005, -0.02, -0.01]
    # high = [0.005, 0.02, 0.03]
    # pos_noise = np.zeros(3)
    # pos_noise = np.random.uniform(low=low, high=high, size=3)
    obj_id, center = load_obj_urdf(json_dict['obj_path'])
    collision_fn = get_collision7d_fn(physics_client_id, obj_id, obstacles=[hook_id])

    len_init_pose = len(json_dict['initial_pose'])
    # assert len_init_pose <= 3, f'initial poses need to be 3, not {len_init_pose}'
    max_cnt = len_init_pose if args.max_cnt == -1 else args.max_cnt
    assert max_cnt > 0 and max_cnt <= len_init_pose


    for index in range(max_cnt):
        # object initialization
        initial_info = json_dict['initial_pose'][index]
        obj_pos = initial_info['object_pose'][:3]
        obj_rot = initial_info['object_pose'][3:]
        # obj_pos = list(np.asarray(obj_pos) + pos_noise)
        # obj_transform = get_matrix_from_pos_rot(obj_pos, obj_rot)

        # grasping
        initial_info = json_dict['initial_pose'][index]
        robot_pos = initial_info['robot_pose'][:3]
        robot_rot = initial_info['robot_pose'][3:]
        # robot_pos = list(np.asarray(robot_pos) + pos_noise)
        robot_pose = robot_pos + robot_rot

        robot.apply_action(robot_pose, max_vel=-1)
        for _ in range(int(1.0 / sim_timestep) * 2): # 1 sec
            p.stepSimulation()
            time.sleep(sim_timestep)
        robot.grasp(obj_id=obj_id)
        for _ in range(int(1.0 / sim_timestep) * 2): # 1 sec
            p.resetBasePositionAndOrientation(obj_id, obj_pos, obj_rot)
            p.stepSimulation()
            time.sleep(sim_timestep)

        # ----------------------------- #
        # --- Setting inital motion --- #
        # ----------------------------- #

        ####################################################

        obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
        start_conf = obj_pos + obj_rot

        # extract current gripper pose
        gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
        gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
        origin_gripper_pose = get_matrix_from_pos_rot(gripper_pos, gripper_rot)

        origin_obj_pose = get_matrix_from_pos_rot(start_conf[:3], start_conf[3:])

        # relative transform from object to gripper
        # see this stack-overflow issue : https://stackoverflow.com/questions/67001118/relative-transform
        obj2gripper_pose = np.linalg.inv(origin_obj_pose) @ origin_gripper_pose
        planning_resolution = 0.001

        d12 = np.asarray(tgt_pose[:3]) - np.asarray(start_conf[:3])
        steps = int(np.ceil(np.linalg.norm(np.divide(d12, planning_resolution), ord=2)))
        obj_init_quat = quaternion.as_quat_array(xyzw2wxyz(start_conf[3:]))
        obj_tgt_quat = quaternion.as_quat_array(xyzw2wxyz(tgt_pose[3:]))

        # draw_coordinate(start_conf)
        # draw_coordinate(tgt_pose)
        # time.sleep(3)

        # imgs = []
        # # plan trajectory in the same way in collision detection module
        # for step in range(steps):
        #     ratio = (step + 1) / steps
        #     pos = ratio * d12 + np.asarray(start_conf[:3])
        #     quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
        #     quat = wxyz2xyzw(quaternion.as_float_array(quat))
        #     positions7d = tuple(pos) + tuple(quat)
        #     # draw_coordinate(positions7d)

        #     gripper_pose = get_matrix_from_pos_rot(positions7d[:3], positions7d[3:]) @ obj2gripper_pose
        #     gripper_pos, gripper_rot = get_pos_rot_from_matrix(gripper_pose)
        #     gripper_action = np.concatenate((gripper_pos, gripper_rot))

        #     robot.apply_action(gripper_action)
        #     p.stepSimulation()
            
        #     robot.grasp()
        #     for _ in range(10): # 1 sec
        #         p.stepSimulation()
        #         time.sleep(sim_timestep)

        #     img = p.getCameraImage(480, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        #     imgs.append(img)
        # imgs = [Image.fromarray(img) for img in imgs]
        
        # obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
        # obj_pose = obj_pos + obj_rot
        # contact = collision_fn(obj_pose)
        # print('[Success]|{}'.format(1 if contact else 0))
        # # save gif
        # if imgs is not None:
        #     status = 'success' if contact else 'failed'
        #     init_pose = ['easy', 'medium', 'hard'][index]
        #     gif_path = os.path.join(output_dir, f'{init_pose}_{args.id}_{status}_direct.gif')
        #     imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        
        # # execution step 2 : release gripper
        # robot_apply_action(robot, obj_id, obj_pose, gripper_action='pre_grasp', 
        #     sim_timestep=0.05, diff_thresh=0.01, max_vel=-1, max_iter=100)
        # _=input()
        # continue
        # ####################################################

        # run RRT algorithm
        imgs_array = hanging_by_rrt(physics_client_id, robot, obj_id, target_conf=tgt_pose, obstacles=[ table_id, hook_id], max_vel=0.1)
        
        # check success
        obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
        obj_pose = obj_pos + obj_rot
        contact = collision_fn(obj_pose)
        print('[Success]|{}'.format(1 if contact else 0))
        sys.stdout.flush()
        p.removeAllUserDebugItems()

        # save gif
        if imgs_array is not None:
            status = 'success' if contact else 'failed'
            # init_pose = ['easy', 'medium', 'hard'][index]
            gif_path = os.path.join(output_dir, f'{index}_{args.id}_{status}.gif')
            imgs_array[0].save(gif_path, save_all=True, append_images=imgs_array[1:], duration=50, loop=0)
        
    print('process complete...')        
    # while True:
        
    #     # key callback
    #     keys = p.getKeyboardEvents()            
    #     if ord('q') in keys and keys[ord('q')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
    #         break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', '-ij', type=str, default='data/Hook_90-hanging_exp/Hook_90-hanging_exp_wrench_1.json')
    parser.add_argument('--id', '-id', type=str, default='0')
    parser.add_argument('--max-cnt', '-mc', type=int, default=1)
    args = parser.parse_args()
    main(args)