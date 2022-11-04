import pybullet as p

def pybullet_ik(physicsClientId, robot_id, end_eff_idx, new_pos, new_quat_orn, 
                maxNumIterations=500, residualThreshold=.001):

    jointPoses = p.calculateInverseKinematics(robot_id, end_eff_idx, new_pos, new_quat_orn,
                                                maxNumIterations=500,
                                                residualThreshold=.001,
                                                physicsClientId=physicsClientId)
    
    return jointPoses

def your_ik(new_pos, new_quat_orn, maxNumIterations=500, residualThreshold=.001):
    
    # you need to implement by your self
    jointPoses = [0, 0, 0, 0, 0, 0, 0]

    return jointPoses
