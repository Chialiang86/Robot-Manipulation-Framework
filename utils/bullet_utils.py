import numpy as np
import pybullet as p
import quaternion
from scipy.spatial.transform import Rotation as R

def xyzw2wxyz(quat : np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

def pose_6d_to_7d(pose : list or tuple or np.ndarray):
    assert len(pose) == 6, f'input array size should be 6d but got {len(pose)}d'

    pos = pose[:3]
    rot_vec = pose[3:]
    rot_quat = R.from_rotvec(rot_vec).as_quat()

    return list(pos) + list(rot_quat)

def pose_7d_to_6d(pose : list or tuple or np.ndarray):
    assert len(pose) == 7, f'input array size should be 7d but got {len(pose)}d'

    pos = pose[:3]
    rot_quat = pose[3:]
    rot_vec = R.from_quat(rot_quat).as_rotvec()

    return list(pos) + list(rot_vec)

def get_matrix_from_pos_rot(pos : list or tuple or np.ndarray, rot : list or tuple or np.ndarray):
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    if len(rot) == 3:
        rot_m = R.from_rotvec(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4: # x, y, z, w
        rot_m = R.from_quat(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def get_matrix_from_7d_pose(pose : list or tuple or np.ndarray):
    assert (len(pose) == 7)
    pos_m = np.asarray(pose[:3])
    rot_m = R.from_quat(pose[3:]).as_matrix()
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def get_pos_rot_from_matrix(pose : np.ndarray):
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, rot

def get_7d_pose_from_matrix(pose : np.ndarray):
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    pose = list(pos) + list(rot)
    return pose

def draw_coordinate(pose : np.ndarray or tuple or list, size : float = 0.1, color : np.ndarray=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    assert (type(pose) == np.ndarray and pose.shape == (4, 4)) or (len(pose) == 7)

    if len(pose) == 7:
        pose = get_matrix_from_pos_rot(pose[:3], pose[3:])

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, color[0])
    p.addUserDebugLine(origin, y, color[1])
    p.addUserDebugLine(origin, z, color[2])

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

def get_robot_joint_info(robot_id):
    num_joints = p.getNumJoints(robot_id)

    joint_states = p.getJointStates(robot_id, range(0, num_joints))
    joint_poses = [x[0] for x in joint_states]
    joint_names = []
    joint_types = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1]
        joint_type = joint_info[2]
        joint_names.append(joint_name)
        joint_types.append(joint_type)
    
    return joint_names, joint_poses, joint_types