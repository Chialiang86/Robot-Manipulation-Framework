import time, os, quaternion
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R

# for simulator
import pybullet as p
import pybullet_data

# for geometry information
from utils.bullet_utils import draw_coordinate, get_dense_waypoints, pose_7d_to_6d

# you may use your forward kinematic algorithm to compute 
from fk import your_fk, get_panda_DH_params

SIM_TIMESTEP = 1.0 / 240.0

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

def pybullet_ik(robot, new_pose : list or tuple or np.ndarray, 
                max_iters : int=500, stop_thresh : float=.001):

    new_pos, new_rot = new_pose[:3], new_pose[3:]
    joint_poses = p.calculateInverseKinematics(robot.robot_id, robot.end_eff_idx, new_pos, new_rot,
                                                maxNumIterations=max_iters,
                                                residualThreshold=stop_thresh,
                                                physicsClientId=robot._physics_client_id)
    
    return joint_poses

def your_ik(robot, new_pose : list or tuple or np.ndarray, 
                max_iters : int=500, stop_thresh : float=.001):

    # you may use this params to avoid joint limit
    limits = np.asarray([
                [-2.9671, 2.9671], # panda_joint1
                [-1.8326, 1.8326], # panda_joint2
                [-2.9671, 2.9671], # panda_joint3
                [-3.1416, 0.0],    # panda_joint4
                [-2.9671, 2.9671], # panda_joint5
                [-0.0873, 3.8223], # panda_joint6
                [-2.9671, 2.9671]  # panda_joint7
            ])

    step = 0.01
    
    # get current joint angles and gripper pos, (gripper pos is fixed)
    num_q = p.getNumJoints(robot.robot_id)
    q_states = p.getJointStates(robot.robot_id, range(0, num_q))
    tmp_q = np.asarray([x[0] for x in q_states[:7]]) # current joint angles 7d (TODO:you should update this)
    gripper_pos = robot.get_gripper_pos() # current gripper position 2d (don't change this variable)
    
    #### your code ####
    # tmp_q = ? # may be more than one line

    # hint : you may use `your_fk` function and jacobian matrix to do this

    new_pose_6d = np.asarray(pose_7d_to_6d(new_pose))
    dh_params = get_panda_DH_params()
    
    for i in range(max_iters):

        tmp_pose, tmp_jacobian = your_fk(robot, dh_params, tmp_q)
        tmp_pose_6d = np.asarray(pose_7d_to_6d(tmp_pose))

        delta_x = new_pose_6d - tmp_pose_6d
        tmp_jacobian /= np.linalg.norm(tmp_jacobian)
        delta_q = np.linalg.pinv(tmp_jacobian) @ (step * delta_x)

        tmp_q = tmp_q + delta_q

        # check joint limit
        # if not (tmp_q > limits[:, 0]).all() or not (tmp_q < limits[:, 1]).all():
        #     print('meet joint limit!')

        cond_low = np.where(tmp_q <= limits[:, 0])
        cond_high = np.where(tmp_q >= limits[:, 1])
        tmp_q[cond_low] = limits[cond_low, 0] + 1e-2
        tmp_q[cond_high] = limits[cond_high, 1] - 1e-2

        dist = np.linalg.norm(new_pose_6d - tmp_pose_6d)
        if dist < stop_thresh:
            break
    
    ###################

    return list(tmp_q) + list(gripper_pos) # 9 DoF

def apply_action(robot, from_pose : list or tuple or np.ndarray, to_pose : list or tuple or np.ndarray):

        # ------------------ #
        # --- IK control --- #
        # ------------------ #

        if not len(from_pose) == 7 or not len(to_pose) == 7:
            raise AssertionError('number of action commands must be 7: (dx,dy,dz,qx,qy,qz,w)'
                                    '\ninstead it is: ', len(to_pose))

        # --- Constraint end-effector pose inside the workspace --- #

        dx, dy, dz = to_pose[:3]
        new_pos = [dx, dy, min(robot._workspace_lim[2][1], max(robot._workspace_lim[2][0], dz))]
        new_rot = to_pose[3:7]
        new_pose = list(new_pos) + list(new_rot)
        
        # -------------------------------------------------------------------------------- #
        # --- TODO: Read the task description                                          --- #
        # --- Task 2 : Compute Inverse-Kinematic Solver of the robot by yourself.      --- #
        # ---          Try to implement `your_ik` function without using any pybullet  --- #
        # ---          API.                                                            --- #
        # --- Note : please modify the code in `your_ik` function.                     --- #
        # -------------------------------------------------------------------------------- #

        #### your code ####
        
        # you can use this function to see the correct version
        # joint_poses = pybullet_ik(robot, new_pose,
        #                         max_iters=500, stop_thresh=.001)

        # this is the function you need to implement
        joint_poses = your_ik(robot, new_pose,
                                max_iters=500, stop_thresh=.001)

        ###################

        # --- set joint control --- #
        p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                    jointIndices=robot._joint_name_to_ids.values(),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    positionGains=[0.2] * len(joint_poses),
                                    velocityGains=[1] * len(joint_poses),
                                    physicsClientId=robot._physics_client_id)

def robot_dense_action(robot, obj_id : int, from_pose : list, to_pose : list, grasp : bool=False, resolution : float=0.001):
    
    # interpolation from the current pose to the target pose
    waypoints = get_dense_waypoints(from_pose, to_pose, resolution=resolution)
    
    apply_action(robot, from_pose, waypoints[0])
    
    for i in range(len(waypoints) - 1):
        
        apply_action(robot, waypoints[i], waypoints[i+1])

        for _ in range(5): 
            p.stepSimulation()
            if grasp:
                robot.grasp(obj_id=obj_id)
            time.sleep(SIM_TIMESTEP)

def main():

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=90,
        cameraPitch=0,
        # cameraTargetPosition=[0.088, 0.0, 0.926]
        cameraTargetPosition=[0.5, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.1])

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

    # get DH params
    gripper_start_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[0]
    gripper_start_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[1]
    gripper_start_pose = list(gripper_start_pos) + list(gripper_start_rot)

    # -------------------------------------------- #
    # --- Test your Inverse Kinematic function --- #
    # -------------------------------------------- #

    # warmup for 2 secs
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    p.removeAllUserDebugItems()

    # example target pose
    
    # gripper_end_pose = [0.5, 0.0, 1.20, # pos
    #                     0.996,-0.047, 0.060, 0.036] # rot 
    gripper_end_pose = [0.4775,  0.0762, 1.3017, # pos
                        0.7084, -0.2386, 0.6625, -0.0481] # rot 

    action_resolution = 0.002
    draw_coordinate(gripper_start_pose) # from
    draw_coordinate(gripper_end_pose) # to
    robot_dense_action(robot, obj_id=-1, from_pose=gripper_start_pose, to_pose=gripper_end_pose, resolution=action_resolution)

    # wait for 2 secs
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    p.removeAllUserDebugItems()

if __name__=="__main__":
    main()