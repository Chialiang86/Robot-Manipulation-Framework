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

# for pose matching via keypoint annotation
import webbrowser

# for robot control
from utils.bullet_utils import draw_coordinate, xyzw2wxyz, wxyz2xyzw, get_robot_joint_info, get_matrix_from_pos_rot, get_pos_rot_from_matrix
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

def render(width, height, view_matrix, projection_matrix, far=1000., near=0.01, obj_id=-1):
    far = 1000.
    near = 0.01
    img = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
    rgb_buffer = np.reshape(img[2], (height, width, 4))[:,:,:3]
    depth_buffer = np.reshape(img[3], [height, width])
    depth_buffer = far * near / (far - (far - near) * depth_buffer)
    seg_buffer = np.reshape(img[4], [height, width])
    
    # # background substraction
    # if obj_id != -1:
    #     cond = np.where(seg_buffer != obj_id)
    #     rgb_buffer[cond] = np.array([255, 255, 255])
    #     depth_buffer[cond] = 1000.
    
    return rgb_buffer, depth_buffer


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

    # -------------------------------- #
    # --- Config camera parameters --- #
    # -------------------------------- #

    f_intr = open('3d_models/objects/mug/cam_info.json', 'r')
    cam_info_dict = json.load(f_intr)

    width, height = cam_info_dict['width'], cam_info_dict['height']
    fx = cam_info_dict['intrinsic'][0][0]
    fy = cam_info_dict['intrinsic'][1][1]
    cx = 0.5 * width
    cy = 0.5 * height
    far = 1000.
    near = 0.01
    fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

    rgb_cam_param = {
        'cameraEyePosition': [0.5, -0.4, 0.8],
        'cameraTargetPosition': [0.5, 0.0, 0.8],
        'cameraUpVector': [0.0, 0.0, 1.0],
    }

    rgb_view_matrix = p.computeViewMatrix(
                    cameraEyePosition=rgb_cam_param['cameraEyePosition'],
                    cameraTargetPosition=rgb_cam_param['cameraTargetPosition'],
                    cameraUpVector=rgb_cam_param['cameraUpVector']
                    )
    
    projection_matrix = p.computeProjectionMatrixFOV(
                            fov=fov,
                            aspect=1.0,
                            nearVal=near,
                            farVal=far
                        )
    # intrinsic
    intrinsic = np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ])

    # ---------------------------- #
    # --- Load object and hook --- #
    # ---------------------------- #
    # wall
    # wall_pos = [0.5, -0.11, 1.0]
    # wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # wall_id = p.loadURDF("models/wall/wall.urdf", wall_pos, wall_orientation)

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.1])
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    # object
    obj_id = p.loadURDF('3d_models/objects/mug/base.urdf')
    standing_pos = [0.5, 0.0, 0.77] # only for mug_70
    standing_rot = R.from_rotvec([np.pi / 2, 0, 0]).as_quat()
    p.resetBasePositionAndOrientation(obj_id, standing_pos, standing_rot)

    # write RGB-D
    rgb_buffer, depth_buffer = render(width, height, rgb_view_matrix, projection_matrix, 
                                        far=far, near=near, obj_id=obj_id)

    rgb_img = Image.fromarray(rgb_buffer)
    rgb_path = f'3d_models/objects/mug/input/rgb_dst.jpg'
    rgb_img.save(rgb_path)

    depth_path = f'3d_models/objects/mug/input/depth_dst.npy'
    np.save(depth_path, depth_buffer)

    # hook
    hook_id = p.loadURDF('3d_models/hook/Hook_60/base.urdf')

    # ------------------------------------- #
    # --- Pose matching via 3 keypoints --- #
    # ------------------------------------- #

    obj_root_dir = f'3d_models/objects/mug'
    obj_output_dir = f'3d_models/objects/mug/output'
    os.system(f"rm {obj_output_dir}/*")
    os.system(f"python3 -m wat.run --data-dir {obj_root_dir} --port 1230 --maxtips 4")

    url = 'http://localhost:1230/'
    webbrowser.open(url)

    while os.listdir(obj_output_dir) == []:
        time.sleep(1.0)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    robot = pandaEnv(physics_client_id, use_IK=1)
    gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
    gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    gripper_pose = list(gripper_pos) + list(gripper_rot)
    before_pose = [0.49670371413230896, 0.03478135168552399, 0.92857426404953, 
                   0.9896430969238281, -0.07126988470554352, 0.12207289785146713, -0.02500670962035656]

    joint_names, joint_poses, joint_types = get_robot_joint_info(robot.robot_id)
    for i in range(len(joint_names)):
        print(f'{joint_names[i]} position={joint_poses[i]} type={joint_types[i]}')

    draw_coordinate(before_pose)

    waypoints = get_dense_waypoints(gripper_pose, before_pose, resolution=0.005)
    for waypoint in waypoints:
        robot.apply_action(waypoint, max_vel=-1)
        for _ in range(10): 
            p.stepSimulation()
            time.sleep(sim_timestep)

    
    joint_names, joint_poses, joint_types = get_robot_joint_info(robot.robot_id)
    for i in range(len(joint_names)):
        print(f'{joint_names[i]} position={joint_poses[i]} type={joint_types[i]}')

        

    # grasping

    # rrt


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