import time, os, argparse, json
import numpy as np

from scipy.spatial.transform import Rotation as R

# for simulator
import pybullet as p
import pybullet_data

# for geometry information
from utils.bullet_utils import draw_coordinate, get_matrix_from_7d_pose, get_7d_pose_from_matrix

SIM_TIMESTEP = 1.0 / 240.0
JACOBIAN_SCORE_MAX = 15.0
JACOBIAN_ERROR_THRESH = 0.05
FK_SCORE_MAX = 15.0
FK_ERROR_THRESH = 0.005
TASK1_SCORE_MAX = JACOBIAN_SCORE_MAX + FK_SCORE_MAX

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

def get_panda_DH_params():

    # details : 
    # see "pybullet_robot_envs/panda_envs/robot_data/franka_panda/panda_model.urdf" in this project folder
    # official spec : https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
    
    dh_params = [
        {'a':  0,      'd': 0.333, 'alpha':  0,  }, # panda_joint1
        {'a':  0,      'd': 0,     'alpha': -np.pi/2}, # panda_joint2
        {'a':  0,      'd': 0.316, 'alpha':  np.pi/2}, # panda_joint3
        {'a':  0.0825, 'd': 0,     'alpha':  np.pi/2}, # panda_joint4
        {'a': -0.0825, 'd': 0.384, 'alpha': -np.pi/2}, # panda_joint5
        {'a':  0,      'd': 0,     'alpha':  np.pi/2}, # panda_joint6
        {'a':  0.088,  'd': 0.07, 'alpha':  np.pi/2}, # panda_joint7
        # {'a':  0.088,  'd': 0.107, 'alpha':  np.pi/2},
    ]

    return dh_params

def your_fk(robot, DH_params : dict, q : list or tuple or np.ndarray) -> np.ndarray:

    # robot initial position
    base_pos = robot._base_position
    base_pose = list(base_pos) + [0, 0, 0, 1]  

    assert len(DH_params) == 7 and len(q) == 7, f'Both DH_params and q should contain 7 values,\n' \
                                                f'but get len(DH_params) = {DH_params}, len(q) = {len(q)}'

    A = get_matrix_from_7d_pose(base_pose) # a 4x4 matrix, type should be np.ndarray
    jacobian = np.zeros((6, 7)) # a 6x7 matrix, type should be np.ndarray

    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 1 : Compute Forward-Kinematic and Jacobain of the robot by yourself --- #
    # ---          Try to implement `your_fk` function without using any pybullet  --- #
    # ---          API.                                                            --- #
    # --- Note : please modify the code in `your_ik`.                              --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####

    # A = ? # may be more than one line
    # jacobian = ? # may be more than one line

    z = np.zeros((7, 3))
    t = np.zeros((7, 3))

    for i in range(7):
        
        DH_i = DH_params[i]

        d = DH_i['d']
        a = DH_i['a']
        alpha_i = DH_i['alpha']
        theta_i = q[i]

        cq = np.cos(theta_i)
        sq = np.sin(theta_i)

        ca = np.cos(alpha_i)
        sa = np.sin(alpha_i)

        A_i = np.asarray([
            [cq, -sq, 0, a],
            [ca * sq, ca * cq, -sa, -d * sa],
            [sa * sq, cq * sa, ca, d * ca],
            [0, 0, 0, 1],
        ])

        A = A @ A_i

        t[i] = A[:3, 3] # ti
        z[i] = A[:3, 2] # zi

    # -45 degree adjustment along z axis
    # details : see "pybullet_robot_envs/panda_envs/robot_data/franka_panda/panda_model.urdf"
    adjustment = R.from_rotvec([0, 0, -0.785398163397]).as_matrix()
    A[:3, :3] = A[:3, :3] @ adjustment
    
    # see https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/
    t_7 = A[:3, 3]
    for i in range(7):
        jacobian[:3, i] = cross(z[i], (t_7 - t[i]))
        jacobian[3:, i] = z[i]

    ###################

    pose_7d = np.asarray(get_7d_pose_from_matrix(A))

    return pose_7d, jacobian

def score_fk(robot, testcase_files : str, visualize : bool=False):

    testcase_file_num = len(testcase_files)
    dh_params = get_panda_DH_params()
    fk_score = [FK_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    fk_error_cnt = [0 for _ in range(testcase_file_num)]
    jacobian_score = [JACOBIAN_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    jacobian_error_cnt = [0 for _ in range(testcase_file_num)]

    difficulty = ['easy  ', 'normal', 'hard  ', 'devil']

    print("============================ Task 1 : Forward Kinematic ============================")
    for file_id, testcase_file in enumerate(testcase_files):

        f_in = open(testcase_file, 'r')
        fk_dict = json.load(f_in)
        f_in.close()

        joint_poses = fk_dict['joint_poses']
        poses = fk_dict['poses']
        jacobians = fk_dict['jacobian']
        cases_num = len(fk_dict['joint_poses'])

        penalty = TASK1_SCORE_MAX / cases_num

        for i in range(cases_num):
            your_pose, your_jacobian = your_fk(robot, dh_params, joint_poses[i])
            gt_pose = poses[i]

            if visualize :
                color_yours = [[1,0,0], [1,0,0], [1,0,0]]
                color_gt = [[0,1,0], [0,1,0], [0,1,0]]
                draw_coordinate(your_pose, size=0.01, color=color_yours)
                draw_coordinate(gt_pose, size=0.01, color=color_gt)

            fk_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
            if fk_error > FK_ERROR_THRESH:
                fk_score[file_id] -= penalty
                fk_error_cnt[file_id] += 1

            jacobian_error = np.linalg.norm(your_jacobian - np.asarray(jacobians[i]), ord=2)
            if jacobian_error > JACOBIAN_ERROR_THRESH:
                jacobian_score[file_id] -= penalty
                jacobian_error_cnt[file_id] += 1
        
        fk_score[file_id] = 0.0 if fk_score[file_id] < 0.0 else fk_score[file_id]
        jacobian_score[file_id] = 0.0 if jacobian_score[file_id] < 0.0 else jacobian_score[file_id]
    
        print("- Your Score Of Forward Kinematic - {} : {:00.02f} / {:00.02f}, Error Count : {:4d} / {:4d}".format(
            difficulty[file_id], fk_score[file_id], FK_SCORE_MAX / testcase_file_num, fk_error_cnt[file_id], cases_num))
        print("- Your Score Of Jacobian Matrix   - {} : {:00.02f} / {:00.02f}, Error Count : {:4d} / {:4d}".format(
            difficulty[file_id], jacobian_score[file_id], JACOBIAN_SCORE_MAX / testcase_file_num, jacobian_error_cnt[file_id], cases_num))

    total_fk_score = 0.0
    total_jacobian_score = 0.0
    for file_id in range(testcase_file_num):
        total_fk_score += fk_score[file_id]
        total_jacobian_score += jacobian_score[file_id]
    
    print("====================================================================================")
    print("- Your Total Score : {:00.02f} / {:00.02f}".format(
        total_fk_score + total_jacobian_score, FK_SCORE_MAX + JACOBIAN_SCORE_MAX))
    print("====================================================================================")

def main(args):

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet env without GUI
    visualize = args.gui
    physics_client_id = p.connect(p.GUI if visualize else p.DIRECT)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.7, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    from pybullet_robot_envs.panda_envs.panda_env import pandaEnv
    robot = pandaEnv(physics_client_id, use_IK=1)

    # -------------------------------------------- #
    # --- Test your Forward Kinematic function --- #
    # -------------------------------------------- #

    testcase_files = [
        'test_case/fk_testcase_easy.json',
        'test_case/fk_testcase_medium.json',
        'test_case/fk_testcase_hard.json'
    ]

    # scoring your algorithm
    score_fk(robot, testcase_files, visualize=args.gui)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', '-g', action='store_true', default=False, help='gui : whether show the window')
    args = parser.parse_args()
    main(args)