import os, copy, argparse, json, time, quaternion
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
import pybullet as p
import pybullet_data

# for motion planning
from utils.motion_planning_utils import get_sample7d_fn, get_distance7d_fn, get_extend7d_fn, get_collision7d_fn
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import check_initial_end
from pybullet_planning.motion_planners.rrt_connect import birrt

# for robot control
from utils.bullet_utils import draw_coordinate, get_7d_pose_from_matrix, get_matrix_from_7d_pose, draw_bbox, get_robot_joint_info
from pybullet_robot_envs.panda_envs.panda_env import pandaEnv

# for custom IK 
from kinematics import robot_dense_action

SIM_TIMESTEP = 1.0 / 240

def get_cam_params(info_dict : dict, cam_param : dict):
    assert 'template_extrinsic' in info_dict.keys() and \
            'hanging_extrinsic' in info_dict.keys() and \
            'intrinsic' in info_dict.keys() and \
            'width' in info_dict.keys() and \
            'height' in info_dict.keys()
    assert 'cameraEyePosition' in cam_param.keys() and \
            'cameraTargetPosition' in cam_param.keys() and \
            'cameraUpVector' in cam_param.keys()

    width, height = info_dict['width'], info_dict['height']
    fx = info_dict['intrinsic'][0][0]
    fy = info_dict['intrinsic'][1][1]
    cx = 0.5 * width
    cy = 0.5 * height
    far = 1000.
    near = 0.01
    fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

    template_extrinsic = np.asarray(info_dict['template_extrinsic'])
    hanging_extrinsic = np.asarray(info_dict['hanging_extrinsic'])

    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=cam_param['cameraEyePosition'],
                        cameraTargetPosition=cam_param['cameraTargetPosition'],
                        cameraUpVector=cam_param['cameraUpVector']
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

    # rotation vector extrinsic
    z = np.asarray(cam_param['cameraTargetPosition']) - np.asarray(cam_param['cameraEyePosition'])
    z /= np.linalg.norm(z, ord=2)
    y = -np.asarray(cam_param['cameraUpVector'])
    y -= (np.dot(z, y)) * z
    y /= np.linalg.norm(y, ord=2)
    x = cross(y, z)

    # extrinsic
    init_extrinsic = np.identity(4)
    init_extrinsic[:3, 0] = x
    init_extrinsic[:3, 1] = y
    init_extrinsic[:3, 2] = z
    init_extrinsic[:3, 3] = np.asarray(cam_param['cameraEyePosition'])

    ret = {
        'intrinsic': intrinsic,
        'template_extrinsic':  template_extrinsic,
        'init_extrinsic': init_extrinsic,
        'hanging_extrinsic': hanging_extrinsic,
        'view_matrix': view_matrix,
        'projection_matrix': projection_matrix,
        'width': width,
        'height': height
    }

    return ret

def randomize_standing_pose(base_pos, base_rot):
    pos_low = np.array([-0.05, -0.1, 0.0])
    pos_high = np.array([0.05,  0.1, 0.0])
    target_pos = np.asarray(base_pos) + np.random.uniform(low=pos_low, high=pos_high)
    rot_low = np.array([0.0, 0.0, -5.0]) * np.pi / 180.0
    rot_high = np.array([0.0, 0.0, 5.0]) * np.pi / 180.0
    target_rot = np.asarray(base_rot) + np.random.uniform(low=rot_low, high=rot_high)
    target_rot =R.from_rotvec(target_rot).as_quat()
    return list(target_pos), list(target_rot)

def create_pcd_from_rgbd(rgb, depth, intr, extr, dscale, depth_threshold=2.0):
    assert rgb.shape[:2] == depth.shape, f'{rgb.shape[:2]} != {depth.shape}'
    (h, w) = depth.shape
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    ix, iy =  np.meshgrid(range(w), range(h))

    x_ratio = (ix.ravel() - cx) / fx
    y_ratio = (iy.ravel() - cy) / fy

    z = depth.ravel() / dscale
    x = z * x_ratio
    y = z * y_ratio

    cond = np.where(z < depth_threshold)
    x = x[cond]
    y = y[cond]
    z = z[cond]

    points = np.vstack((x, y, z)).T
    colors = np.reshape(rgb,(depth.shape[0] * depth.shape[1], 3)) /255.
    colors = colors[cond]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(extr)

    return pcd

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

def load_3d_kpts_from_depth(intrinsic, json_path, depth_path):
    f = open(json_path, 'r')
    json_dict = json.load(f)
    f.close()

    # load keypoints
    kpts_x = np.array([ele["x"] for ele in json_dict["tooltips"]])
    kpts_y = np.array([ele["y"] for ele in json_dict["tooltips"]])
    kpts_rc = (kpts_y, kpts_x)

    depth = np.load(depth_path)

    # convert to xyz

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    z = depth[kpts_rc]
    x = z * (kpts_x - cx) / fx
    y = z * (kpts_y - cy) / fy

    kpts_3d =  np.vstack((x, y, z))

    return kpts_3d

def checkpoint(num : int, msg  : str):
    print(f"[checkpoint {num} : {msg}")
    print("press y on PyBullet GUI to continue or press n on PyBullet GUI to redo ...")
    repeat = False
    while True:
        # key callback
        keys = p.getKeyboardEvents()            
        if ord('y') in keys and keys[ord('y')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
            break
        if ord('n') in keys and keys[ord('n')] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN): 
            repeat = True
            break
    
    return repeat

def web_annotator(obj_root_dir : str, port : int or str):
    os.system(f'python3 web_annotator.py {obj_root_dir} {port}')
    return

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def get_src2dst_transform_3kpt(src_3kpts_homo : np.ndarray, src_extrinsic : np.ndarray, 
                               dst_3kpts_homo : np.ndarray, dst_extrinsic : np.ndarray):

    assert src_3kpts_homo.shape == (4, 3) and dst_3kpts_homo.shape == (4, 3), \
                f'input array shape need to be (3, 3), but get {src_3kpts_homo.shape} and {dst_3kpts_homo.shape}'
    
    assert src_extrinsic.shape == (4, 4) and dst_extrinsic.shape == (4, 4), \
                f'input array shape need to be (4, 4), but get {src_extrinsic.shape} and {dst_extrinsic.shape}'
    
    src_3kpts_homo = (src_extrinsic @ src_3kpts_homo)[:3,:]
    dst_3kpts_homo = (dst_extrinsic @ dst_3kpts_homo)[:3,:]
    
    src_mean = np.mean(src_3kpts_homo, axis=1).reshape((3, 1))
    dst_mean = np.mean(dst_3kpts_homo, axis=1).reshape((3, 1))

    src_points_ = src_3kpts_homo - src_mean
    dst_points_ = dst_3kpts_homo - dst_mean

    W = dst_points_ @ src_points_.T
    
    u, s, vh = np.linalg.svd(W, full_matrices=True)
    R = u @ vh

    # SVD : reflective issue https://medium.com/machine-learning-world/linear-algebra-points-matching-with-svd-in-3d-space-2553173e8fed
    if np.linalg.det(R) < 0:
        print('fix reflective issue')
        R[:, 2] *= -1
    t = dst_mean - R @ src_mean

    transform = np.identity(4)
    transform[:3,:3] = R
    transform[:3, 3] = t.reshape((1, 3))

    return transform

def get_src2dst_transform_4kpt(src_4kpts : np.ndarray, dst_4kpts : np.ndarray):
    assert src_4kpts.shape == (4, 4) and dst_4kpts.shape == (4, 4), \
                f'input array shape need to be (4,4), but get {src_4kpts.shape} and {dst_4kpts.shape}'

    return dst_4kpts @ np.linalg.inv(src_4kpts)

def refine_obj_pose(physicsClientId, obj, original_pose, obstacles=[]):
    collision7d_fn = get_collision7d_fn(physicsClientId, obj, obstacles=obstacles)

    low_limit = [-0.005, -0.005, -0.005, -np.pi / 180, -np.pi / 180, -np.pi / 180]
    high_limit = [ 0.005,  0.005,  0.005,  np.pi / 180,  np.pi / 180,  np.pi / 180]

    obj_pos, obj_rot = original_pose[:3], original_pose[3:]
    refine_pose = original_pose
    while collision7d_fn(tuple(refine_pose)):
        refine_pose6d = np.concatenate((np.asarray(obj_pos), R.from_quat(obj_rot).as_rotvec())) + np.random.uniform(low_limit, high_limit)
        refine_pose = np.concatenate((refine_pose6d[:3], R.from_rotvec(refine_pose6d[3:]).as_quat()))
    return refine_pose

def rrt_connect_7d(physics_client_id, obj_id, start_conf, target_conf, 
                    obstacles : list = [], diagnosis=False, **kwargs):

    # https://github.com/yijiangh/pybullet_planning/blob/dev/src/pybullet_planning/motion_planners/rrt_connect.py

    # config sample location space
    start_pos = start_conf[:3]
    target_pos = target_conf[:3]
    
    low_limit = [0, 0, 0]
    high_limit = [0, 0, 0]
    padding_low  = [ 0.02,  0.01, 0.00]
    padding_high = [ 0.02,  0.01, 0.1]

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
    extend7d_fn = get_extend7d_fn(resolution=0.002)
    collision_fn = get_collision7d_fn(physics_client_id, obj_id, obstacles=obstacles)

    if not check_initial_end(start_conf, target_conf, collision_fn, diagnosis=diagnosis):
        return None

    return birrt(start_conf, target_conf, distance7d_fn, sample7d_fn, extend7d_fn, collision_fn, **kwargs)

def main(args):

    assert os.path.exists(args.input_dir), f'{args.input_dir} not exists'
    input_dir = args.input_dir
    
    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=90,
        cameraPitch=0,
        # cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)

    # -------------------------------- #
    # --- Config camera parameters --- #
    # -------------------------------- #

    input_info = f'{input_dir}/info.json'
    assert os.path.exists(input_info), f'{input_info} not exists'
    f_intr = open(input_info, 'r')
    info_dict = json.load(f_intr)

    cam_coordinate = {
        'cameraEyePosition': [0.35, 0.4, 0.76],
        'cameraTargetPosition': [0.35, 0.1, 0.76],
        'cameraUpVector': [0.0, 0.0, 1.0],
    }

    cam_params = get_cam_params(info_dict, cam_coordinate)

    intrinsic = cam_params['intrinsic']
    template_extrinsic = cam_params['template_extrinsic']
    init_extrinsic = cam_params['init_extrinsic']
    hanging_extrinsic = cam_params['hanging_extrinsic']
    view_matrix = cam_params['view_matrix']
    projection_matrix = cam_params['projection_matrix']
    width = cam_params['width']
    height = cam_params['height']
    far = 1000
    near = 0.01

    # ---------------------------- #
    # --- Load object and hook --- #
    # ---------------------------- #
    # wall
    # wall_pos = [0.5, -0.11, 1.0]
    # wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # wall_id = p.loadURDF("models/wall/wall.urdf", wall_pos, wall_orientation)

    # Load plane contained in pybullet_data
    table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.9, 0.0, 0.1])
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    # object
    urdf_path = info_dict['obj_path']
    assert os.path.exists(urdf_path), f'{urdf_path} not exists'
    obj_id = p.loadURDF(urdf_path)

    # hook
    urdf_path = info_dict['hook_path']
    assert os.path.exists(urdf_path), f'{urdf_path} not exists'
    hook_id = p.loadURDF(urdf_path, [0.5, -0.1, 1.3], [0.0, 0.7071067811865475, 0.7071067811865476, 0.0])

    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 2 : Pose matching via N keypoints and find the target grasping /    --- #
    # ---          hanging pose.                                                   --- #
    # ---   - Task 2-1 : Pose matching using N keypoints via closed-form ICP       --- #
    # ---                between 2 different pointclouds                           --- #
    # ---   - Task 2-2 : Learn how to transform gripper poses to object pose or    --- #
    # ---                vice versa                                                --- #
    # -------------------------------------------------------------------------------- #

    # keypoint annotation using web keypoint annotator 
    # Github : https://github.com/luiscarlosgph/keypoint-annotation-tool

    # randomly initialize the pose of the object
    standing_pos, standing_rot = randomize_standing_pose(base_pos=[0.35, 0.1, 0.76], base_rot=[np.pi / 2, 0, 0])
    p.resetBasePositionAndOrientation(obj_id, standing_pos, standing_rot)

    # render init RGB-D
    obj_input_dir = f'{input_dir}/input'
    obj_output_dir = f'{input_dir}/output'
    init_rgb, init_depth = render(width, height, view_matrix, projection_matrix, 
                                        far=far, near=near, obj_id=obj_id)
    
    # save rgb-d to files
    init_rgb_img = Image.fromarray(init_rgb)
    init_rgb_path = f'{obj_input_dir}/rgb_init.jpg'
    init_rgb_img.save(init_rgb_path)

    init_depth_path = f'{obj_input_dir}/depth_init.npy'
    np.save(init_depth_path, init_depth)

    # 2d keypoints information will be save to these json
    template_json_path = f'{obj_output_dir}/rgb_template.json'
    init_json_path = f'{obj_output_dir}/rgb_init.json'
    hanging_json_path = f'{obj_output_dir}/rgb_hanging.json'

    # f_hanging_pose = args.input_dir
    # f = open(f_hanging_pose, 'r')
    # hanging_pose_dict = json.load(f)
    # src_gripper_trans = np.asarray(hanging_pose_dict["obj2gripper"])
    template_gripper_trans = np.asarray([
                            [ 0.98954677, -0.10356444,  0.10035739,  0.00496532],
                            [ 0.1071792 ,  0.06254209, -0.99227068,  0.03851797],
                            [ 0.09648739,  0.99265447,  0.07298828, -0.03677358],
                            [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]
                        ])

    repeat = True
    while repeat:
        
        # -------------------------------------------------------------------- #
        # --- For Task 2-1                                                 --- #
        # --- TODO: annotate some keypoints on images for pose matching by --- #
        # ---       using the web annotator                                --- #
        # --- Note : you don't need to add any code in this TODO           --- #
        # -------------------------------------------------------------------- #

        # remove old files
        os.system(f"rm {obj_output_dir}/*")

        # run web annotator
        port = 1234
        os.system(f'python3 web_annotator.py {input_dir} {port}')
        # t = threading.Thread(target=web_annotator, args=(input_dir, port))
        # t.start()

        while not os.path.exists(template_json_path) or \
              not os.path.exists(init_json_path) or \
              not os.path.exists(hanging_json_path):
            time.sleep(0.1)

        os.system(f"mv {obj_output_dir}/*.jpg {obj_input_dir}/")


        # template view RGB-D
        template_rgb_path = f'{obj_input_dir}/rgb_template.jpg'
        template_rgb = np.asarray(Image.open(template_rgb_path))
        template_depth_path = f'{obj_input_dir}/depth_template.npy'
        template_depth = np.load(template_depth_path)

        # hanging view RGB-D
        hanging_rgb_path = f'{obj_input_dir}/rgb_hanging.jpg'
        hanging_rgb = np.asarray(Image.open(hanging_rgb_path))
        hanging_depth_path = f'{obj_input_dir}/depth_hanging.npy'
        hanging_depth = np.load(hanging_depth_path)

        # before matching
        coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        colors = np.array([
                        [255, 0, 0] for _ in range(template_rgb.shape[0] * template_rgb.shape[1])
                    ]).reshape((template_rgb.shape[0], template_rgb.shape[1], 3))
        
        # create original point cloud from RGB-D image
        template_pcd = create_pcd_from_rgbd(colors, template_depth, intrinsic, template_extrinsic, dscale=1)
        init_pcd = create_pcd_from_rgbd(init_rgb, init_depth, intrinsic, init_extrinsic, dscale=1)
        hanging_pcd = create_pcd_from_rgbd(hanging_rgb, hanging_depth, intrinsic, hanging_extrinsic, dscale=1)

        # get 3d keypoints relative to RGBD cameras from querying depth images using the annotated keypoints
        template_3d_kpts = load_3d_kpts_from_depth(intrinsic, template_json_path, template_depth_path)
        init_3d_kpts = load_3d_kpts_from_depth(intrinsic, init_json_path, init_depth_path)
        hanging_3d_kpts = load_3d_kpts_from_depth(intrinsic, hanging_json_path, hanging_depth_path)


        # -------------------------------------------------------------------- #
        # --- For Task 2-1                                                 --- #
        # --- TODO: use the annotated keypoints to find target mug poses.  --- #
        # ---       Given camera intrinsic and two extrinsics of src and   --- #
        # ---       dst camera, we need to find the pose of the mug in     --- #
        # ---       world coordinate.                                      --- #
        # -------------------------------------------------------------------- #

        #### your code ####

        # template2init = ? # a 4x4 transformation matrix (may be more than one line)
        # template2hanging = ? # a 4x4 transformation matrix (may be more than one line)

        template_3d_kpts_homo = np.vstack((template_3d_kpts, np.array([1.0, 1.0, 1.0])))
        init_3d_kpts_homo = np.vstack((init_3d_kpts, np.array([1.0, 1.0, 1.0])))
        template2init = get_src2dst_transform_3kpt(template_3d_kpts_homo, template_extrinsic, init_3d_kpts_homo, init_extrinsic)


        template_3d_kpts_homo = np.vstack((template_3d_kpts, np.array([1.0, 1.0, 1.0])))
        hanging_3d_kpts_homo = np.vstack((hanging_3d_kpts, np.array([1.0, 1.0, 1.0])))
        template2hanging = get_src2dst_transform_3kpt(template_3d_kpts_homo, template_extrinsic, hanging_3d_kpts_homo, hanging_extrinsic)
        

        template_tran = np.linalg.inv(init_extrinsic) @ template2init @ template_extrinsic @ template_3d_kpts_homo
        for i in range(template_tran.shape[1]):
            print('template2init result : {} <-> {} error = {}'.format(
                template_tran[:,i], init_3d_kpts_homo[:, i], np.linalg.norm(template_tran[:,i] - init_3d_kpts_homo[:, i], ord=2))
            )

        hanging_tran = np.linalg.inv(hanging_extrinsic) @ template2hanging @ template_extrinsic @ template_3d_kpts_homo
        for i in range(template_tran.shape[1]):
            print('template2hanging result : {} <-> {} error = {}'.format(
                hanging_tran[:,i], hanging_3d_kpts_homo[:, i], np.linalg.norm(hanging_tran[:,i] - hanging_3d_kpts_homo[:, i], ord=2))
            )

        ####################
        
        # template and initial view point cloud before and after alignment
        o3d.visualization.draw_geometries([coor, template_pcd, init_pcd])
        template2init_pcd = copy.deepcopy(template_pcd)
        template2init_pcd.transform(template2init)
        o3d.visualization.draw_geometries([coor, template2init_pcd, init_pcd])
        
        # template and hanging view point cloud before and after alignment
        o3d.visualization.draw_geometries([coor, template_pcd, hanging_pcd])
        template2hanging_pcd = copy.deepcopy(template_pcd)
        template2hanging_pcd.transform(template2hanging)
        o3d.visualization.draw_geometries([coor, template2hanging_pcd, hanging_pcd])

        # -------------------------------------------------------------------- #
        # --- For Task 2-2                                                 --- #
        # --- TODO: compute the target grasping poses via the mug poses    --- #
        # ---       and the given template grasping pose using Forward     --- #
        # ---       Kinematic (see variable `template_gripper_trans`)      --- #
        # -------------------------------------------------------------------- #

        #### your code ####

        # init_grasping_pose = ? (may be more than one line)
        # hanging_gripper_pose = ? (may be more than one line)

        # grasping pose
        gripper_grasping_trans = template2init @ template_gripper_trans
        gripper_grasping_pose = get_7d_pose_from_matrix(gripper_grasping_trans)

        # hanging pose
        gripper_hanging_trans = template2hanging @ template_gripper_trans
        gripper_hanging_pose = get_7d_pose_from_matrix(gripper_hanging_trans)

        ####################

        # checkpoint, repeat if repeat (pressing n) 
        repeat = checkpoint(1, 'After finishing pose matching and grasping pose finding, let\'s grasp the mug')

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    action_resolution = 0.005
    robot = pandaEnv(physics_client_id, use_IK=1)
    gripper_start_pos = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[4]
    gripper_start_rot = p.getLinkState(robot.robot_id, robot.end_eff_idx, physicsClientId=robot._physics_client_id)[5]
    gripper_start_pose = list(gripper_start_pos) + list(gripper_start_rot)
    
    # warmup for 2 secs
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # ------------------ #
    # --- Reaching 1 --- #
    # ------------------ #

    # go to the preparing pose
    gripper_prepare_pose = copy.copy(gripper_grasping_pose)
    gripper_prepare_pose[2] += 0.15
    draw_coordinate(gripper_prepare_pose)
    robot_dense_action(robot, obj_id, gripper_start_pose, gripper_prepare_pose, grasp=False, resolution=action_resolution)

    joint_names, joint_poses, joint_types = get_robot_joint_info(robot.robot_id)
    print(joint_poses)

    # ---------------- #
    # --- Grasping --- #
    # ---------------- #

    # go to the grasping pose
    robot_dense_action(robot, obj_id, gripper_prepare_pose, gripper_grasping_pose, grasp=False, resolution=action_resolution)

    # grasping
    robot.grasp(obj_id=obj_id)
    for _ in range(int(1 / SIM_TIMESTEP * 0.5)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # ------------------ #
    # --- Reaching 2 --- #
    # ------------------ #

    # grasp up to the specific pose
    gripper_key_pose = copy.copy(gripper_hanging_pose)
    gripper_key_pose[1] += 0.1
    gripper_key_pose[2] += 0.02
    print(f'keypose : {gripper_key_pose}')
    draw_coordinate(gripper_key_pose)
    robot_dense_action(robot, obj_id, gripper_grasping_pose, gripper_key_pose, grasp=True, resolution=action_resolution)
    
    # record current object pose
    obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
    key_obj_pose = obj_pos + obj_rot

    stop = checkpoint(2, 'After grasping the mug, let\'s start to hanging the mug on the hook')
    if stop:
        return

    # --------------- #
    # --- Hanging --- #
    # --------------- #

    # -------------------------------------------------------------------- #
    # --- For Task 2-2                                                 --- #
    # --- TODO: `gripper_key_pose` is the pose for the gripper.        --- #
    # ---       However, we will apply object pose for computing RRT-  --- #
    # ---       connect algorithm, so we need to transform the         --- #
    # ---       `gripper_key_pose` to `obj_start_pose` using Forward   --- #
    # ---       Kinematic (see variable `template_gripper_trans`)      --- #
    # -------------------------------------------------------------------- #

    #### your code ####

    # obj_start_pose = ? (may be more than one line)

    gripper_key_trans = get_matrix_from_7d_pose(gripper_key_pose)
    obj_start_pose = get_7d_pose_from_matrix(gripper_key_trans @ np.linalg.inv(template_gripper_trans))

    ###################

    # avoid collision in initial checking
    obstacles=[hook_id, table_id]
    obj_hanging_pose = get_7d_pose_from_matrix(template2hanging)
    obj_end_pose = refine_obj_pose(physics_client_id, obj_id, obj_hanging_pose, obstacles=obstacles)
    
    # run RRT-connect to find a path for the mug
    obj_waypoints = rrt_connect_7d(physics_client_id, obj_id, start_conf=obj_start_pose, target_conf=obj_end_pose, obstacles=obstacles)

    if obj_waypoints is None:
        print("Oops, no solution!")
        return

    # -------------------------------------------------------------------- #
    # --- For Task 2-2                                                 --- #
    # --- TODO: transform the waypoints of the mug to the waypoints    --- #
    # ---       of the gripper via Forward Kinematic (see variable     --- #
    # ---       `template_gripper_trans`)                              --- #
    # -------------------------------------------------------------------- #

    gripper_waypoints = []

    #### your code ####

    # gripper_waypoints = ? (may be more than one line)

    for obj_waypoint in obj_waypoints:
        obj_trans = get_matrix_from_7d_pose(obj_waypoint)
        gripper_waypoint = get_7d_pose_from_matrix(obj_trans @ template_gripper_trans)
        gripper_waypoints.append(gripper_waypoint)

    ###################

    # reset obj pose and execute the gripper waypoints for hanging
    p.resetBasePositionAndOrientation(obj_id, key_obj_pose[:3], key_obj_pose[3:])
    for i in range(len(gripper_waypoints) - 1):
        robot_dense_action(robot, obj_id, gripper_waypoints[i], gripper_waypoints[i + 1], grasp=True, resolution=action_resolution)

    # execution step 3 : open the gripper
    robot.pre_grasp()
    for _ in range(int(1 / SIM_TIMESTEP * 0.5)): 
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # execution step 4 : go to the ending pose
    gripper_pose = gripper_waypoints[-1]
    gripper_rot_matrix = R.from_quat(gripper_pose[3:]).as_matrix()
    gripper_ending_pos = np.asarray(gripper_pose[:3]) + (gripper_rot_matrix @ np.array([[0], [0], [-0.05]])).reshape(3)
    gripper_ending_pose = tuple(gripper_ending_pos) + tuple(gripper_pose[3:])
    robot_dense_action(robot, obj_id, gripper_pose, gripper_ending_pose, grasp=False, resolution=action_resolution)

    joint_names, joint_poses, joint_types = get_robot_joint_info(robot.robot_id)
    for i in range(len(joint_names)):
        print(f'{joint_names[i]} position={joint_poses[i]} type={joint_types[i]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=str, default='data/mug-Hook_skew')
    args = parser.parse_args()
    main(args)