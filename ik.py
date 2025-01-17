import argparse, time, os, json
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R

# for simulator
import pybullet as p
import pybullet_data

# for geometry information
from utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pose_from_matrix

# you may use your forward kinematic algorithm to compute 
from fk import your_fk, get_panda_DH_params

SIM_TIMESTEP = 1.0 / 240.0
TASK2_SCORE_MAX = 30
IK_ERROR_THRESH = 0.02

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

# this is the pybullet version
def pybullet_ik(robot, new_pose : list or tuple or np.ndarray, 
                max_iters : int=1000, stop_thresh : float=.001):

    new_pos, new_rot = new_pose[:3], new_pose[3:]
    joint_poses = p.calculateInverseKinematics(robot.robot_id, robot.end_eff_idx, new_pos, new_rot,
                                                maxNumIterations=max_iters,
                                                residualThreshold=stop_thresh,
                                                physicsClientId=robot._physics_client_id)
    
    return joint_poses

def your_ik(robot, new_pose : list or tuple or np.ndarray, 
                max_iters : int=1000, stop_thresh : float=.001):

    # you may use this params to avoid joint limit
    joint_limits = np.asarray([
                [-2.9671, 2.9671], # panda_joint1
                [-1.8326, 1.8326], # panda_joint2
                [-2.9671, 2.9671], # panda_joint3
                [-3.1416, 0.0],    # panda_joint4
                [-2.9671, 2.9671], # panda_joint5
                [-0.0873, 3.8223], # panda_joint6
                [-2.9671, 2.9671]  # panda_joint7
            ])
    
    # get current joint angles and gripper pos, (gripper pos is fixed)
    num_q = p.getNumJoints(robot.robot_id)
    q_states = p.getJointStates(robot.robot_id, range(0, num_q))
    tmp_q = np.asarray([x[0] for x in q_states[:7]]) # current joint angles 7d 
    gripper_pos = robot.get_gripper_pos() # current gripper position 2d (don't change this variable)
        
    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 2 : Compute Inverse-Kinematic Solver of the robot by yourself.      --- #
    # ---          Try to implement `your_ik` function WITHOUT using ANY pybullet  --- #
    # ---          API. (40% : 30% for accuracy and 10% for the report)            --- #
    # --- Note : please modify the code in `your_ik` function.                     --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####

    # TODO: update tmp_q
    # tmp_q = ? # may be more than one line

    # hint : 
    # 1. You may use `your_fk` function and jacobian matrix to do this
    # 2. Be careful when computing the delta x

    step_ratio = 0.1
    new_pose_trans = get_matrix_from_pose(new_pose)
    dh_params = get_panda_DH_params()
    # new_pose_6d = np.asarray(pose_7d_to_6d(new_pose))
    
    dist_min = np.inf
    tolerence_cnt = 0
    tolerence_max = 10

    for i in range(max_iters):

        tmp_pose, tmp_jacobian = your_fk(robot, dh_params, tmp_q)

        # tmp_pose_6d = np.asarray(pose_7d_to_6d(tmp_pose))
        # delta_x = new_pose_6d - tmp_pose_6d

        relative_transform = new_pose_trans @ np.linalg.inv(get_matrix_from_pose(tmp_pose))
        delta_x = get_pose_from_matrix(relative_transform, pose_size=6)

        dist = np.linalg.norm(delta_x, ord=2)
        if dist < stop_thresh:
            break

        if dist < dist_min:
            dist_min = dist
            tolerence_cnt = 0
        else:
            tolerence_cnt += 1
            if tolerence_cnt == tolerence_max:
                break
        
        step_raw = step_ratio * np.sqrt(dist)
        step = 0.015 if step_raw > 0.015 else step_raw

        tmp_jacobian /= np.linalg.norm(tmp_jacobian)
        delta_q = np.linalg.pinv(tmp_jacobian) @ (step * delta_x)

        tmp_q = tmp_q + delta_q

        # check joint limit
        # if not (tmp_q > joint_limits[:, 0]).all() or not (tmp_q < joint_limits[:, 1]).all():
        #     print('meet joint limit!')

        cond_low = np.where(tmp_q <= joint_limits[:, 0])
        cond_high = np.where(tmp_q >= joint_limits[:, 1])
        tmp_q[cond_low] = joint_limits[cond_low, 0] + 1e-2
        tmp_q[cond_high] = joint_limits[cond_high, 1] - 1e-2

    ###################

    return list(tmp_q) + list(gripper_pos) # 9 DoF


# TODO: [for your information]
# This function is the scoring function, we will use the same code 
# to score your algorithm using all the testcases
def score_ik(robot, testcase_files : str, visualize : bool=False):

    testcase_file_num = len(testcase_files)
    ik_score = [TASK2_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    ik_error_cnt = [0 for _ in range(testcase_file_num)]

    difficulty = ['easy  ', 'normal', 'hard  ', 'easy_ta  ', 'normal_ta', 'hard_ta  ']
    joint_ids = range(7)

    p.addUserDebugText(text = "Scoring Your Inverse Kinematic Algorithm ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)

    print("============================ Task 2 : Inverse Kinematic ============================\n")
    for file_id, testcase_file in enumerate(testcase_files):

        f_in = open(testcase_file, 'r')
        ik_dict = json.load(f_in)
        f_in.close()

        joint_poses = ik_dict['current_joint_poses']
        poses = ik_dict['next_poses']
        cases_num = len(ik_dict['current_joint_poses'])

        penalty = (TASK2_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)
        ik_errors = []

        # reset to initial state
        for joint_id in joint_ids:
            p.resetJointState(robot.robot_id, joint_id, joint_poses[0][joint_id], physicsClientId=robot._physics_client_id)
        p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                        jointIndices=joint_ids,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=joint_poses[0],
                                        positionGains=[0.2] * len(joint_poses[0]),
                                        velocityGains=[1] * len(joint_poses[0]),
                                        physicsClientId=robot._physics_client_id)
        for _ in range(int(1 / SIM_TIMESTEP * 1)):
            p.stepSimulation()
            time.sleep(SIM_TIMESTEP)

        for i in range(cases_num):

            # TODO: check your default arguments of `max_iters` and `stop_thresh` are your best parameters.
            #       We will only pass default arguments of your `max_iters` and `stop_thresh`.
            your_joint_poses = your_ik(robot, poses[i]) 

            # You can use `pybullet_ik` to see the correct version 
            # your_joint_poses = pybullet_ik(robot, poses[i]) 
            
            gt_pose = poses[i]

            p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                        jointIndices=robot._joint_name_to_ids.values(),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=your_joint_poses,
                                        positionGains=[0.2] * len(your_joint_poses),
                                        velocityGains=[1] * len(your_joint_poses),
                                        physicsClientId=robot._physics_client_id)
            
            # warmup for 0.1 sec
            for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                p.stepSimulation()
                time.sleep(SIM_TIMESTEP)

            gripper_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[0]
            gripper_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[1]
            your_pose = list(gripper_pos) + list(gripper_rot)

            if visualize:
                color_yours = [[1,0,0], [1,0,0], [1,0,0]]
                color_gt = [[0,1,0], [0,1,0], [0,1,0]]
                draw_coordinate(your_pose, size=0.01, color=color_yours)
                draw_coordinate(gt_pose, size=0.01, color=color_gt)

            ik_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
            ik_errors.append(ik_error)
            if ik_error > IK_ERROR_THRESH:
                ik_score[file_id] -= penalty
                ik_error_cnt[file_id] += 1

        ik_score[file_id] = 0.0 if ik_score[file_id] < 0.0 else ik_score[file_id]
        ik_errors = np.asarray(ik_errors)

        score_msg = "- Difficulty : {}\n".format(difficulty[file_id]) + \
                    "- Mean Error : {:0.06f}\n".format(np.mean(ik_errors)) + \
                    "- Error Count : {:3d} / {:3d}\n".format(ik_error_cnt[file_id], cases_num) + \
                    "- Your Score Of Inverse Kinematic : {:00.03f} / {:00.03f}\n".format(
                            ik_score[file_id], TASK2_SCORE_MAX / testcase_file_num)
        
        print(score_msg)
    p.removeAllUserDebugItems()

    total_ik_score = 0.0
    for file_id in range(testcase_file_num):
        total_ik_score += ik_score[file_id]
    
    print("====================================================================================")
    print("- Your Total Score : {:00.03f} / {:00.03f}".format(total_ik_score , TASK2_SCORE_MAX))
    print("====================================================================================")

def main(args):

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.5, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.9, 0.0, 0.0])

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    from pybullet_robot_envs.panda_envs.panda_env import pandaEnv
    robot = pandaEnv(physics_client_id, use_IK=1)

    # -------------------------------------------- #
    # --- Test your Forward Kinematic function --- #
    # -------------------------------------------- #
    
    # --- set joint control --- #
    joint_poses = [-0.02310, 0.08643, -0.11133, -2.04190, -0.06971, 2.23339, 0.8254, 0.029994093229595557, 0.030001628978402556]
    p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                jointIndices=robot._joint_name_to_ids.values(),
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                positionGains=[0.2] * len(joint_poses),
                                velocityGains=[1] * len(joint_poses),
                                physicsClientId=robot._physics_client_id)
    # warmup for 1 sec
    for _ in range(int(1 / SIM_TIMESTEP * 1)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # ------------------------------------------------------------------ #
    # --- Test your Inverse Kinematic function using one target pose --- #
    # ------------------------------------------------------------------ #
    
    # warmup for 2 secs
    p.addUserDebugText(text = "Warmup for 2 secs ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    p.removeAllUserDebugItems()

    # test your ik solver
    testcase_files = [
        'test_case/ik_testcase_easy.json',
        'test_case/ik_testcase_medium.json',
        'test_case/ik_testcase_hard.json',
        'test_case/ik_testcase_easy_ta.json',
        'test_case/ik_testcase_medium_ta.json',
        'test_case/ik_testcase_hard_ta.json',
    ]

    # ------------------------------------------------------------- #
    # --- Test your Inverse Kinematic function using test cases --- #
    # ------------------------------------------------------------- #

    # scoring your algorithm
    score_ik(robot, testcase_files, visualize=args.visualize_pose)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize-pose', '-vp', action='store_true', default=False, help='whether show the poses of end effector')
    args = parser.parse_args()
    main(args)