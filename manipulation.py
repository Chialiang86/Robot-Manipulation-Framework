import os, copy, argparse, json, time, cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import pybullet as p
import pybullet_data

# for RRT-Connect motion planning
from utils.motion_planning_utils import get_sample7d_fn, get_distance7d_fn, get_extend7d_fn, get_collision7d_fn
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import check_initial_end
from pybullet_planning.motion_planners.rrt_connect import birrt

# for robot control
from utils.bullet_utils import draw_coordinate, get_7d_pose_from_matrix, get_matrix_from_7d_pose, draw_bbox, get_robot_joint_info
from pybullet_robot_envs.panda_envs.panda_env import pandaEnv

# for your custom IK solver
from ik import robot_dense_action

SIM_TIMESTEP = 1.0 / 240

def print_log(msg : str='', state : int=0):
    state = 0 if (state > 2 or state < 0) else state
    prefix = {
        0: 'INFORMATION',
        1: 'WARNNING',
        2: 'ERROR'
    }[state]
    print(f'[{prefix}: {msg}]')

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

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print_log(f'keypoint = ({x} {y})')
        if state == 0:
            template_3d_kpts.append([x, y])
            cv2.circle(template_rgb_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('template_rgb', template_rgb_copy)
        elif state == 1:
            init_3d_kpts.append([x, y])
            cv2.circle(init_rgb_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('init_rgb', init_rgb_copy)
        elif state == 2:
            hanging_3d_kpts.append([x, y])
            cv2.circle(hanging_rgb_copy, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('hanging_rgb', hanging_rgb_copy)

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

def load_kpts_from_depth(intrinsic : list or np.ndarray, keypoints : list or np.ndarray, depth_path : str):

    # load keypoints
    if type(keypoints) == list :
        keypoints = np.asarray(keypoints)
    kpts_x = np.array(keypoints[:,0])
    kpts_y = np.array(keypoints[:,1])
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
    print_log(f"checkpoint {num} : {msg}")
    print_log("press y on PyBullet GUI to continue or press n on PyBullet GUI to redo ...")
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

def get_src2dst_transform_from_kpts(src_3kpts_homo : np.ndarray, src_extrinsic : np.ndarray, 
                               dst_3kpts_homo : np.ndarray, dst_extrinsic : np.ndarray):

    assert src_3kpts_homo.shape[0] == 4 and dst_3kpts_homo.shape[0] == 4, \
                f'input array shape need to be (4, N), but get {src_3kpts_homo.shape} and {dst_3kpts_homo.shape}'
    
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
        print_log('fix reflective issue')
        R[:, 2] *= -1
    t = dst_mean - R @ src_mean

    transform = np.identity(4)
    transform[:3,:3] = R
    transform[:3, 3] = t.reshape((1, 3))

    return transform

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

    # keypoint annotation using web keypoint annotator 
    # Github : https://github.com/luiscarlosgph/keypoint-annotation-tool

    # randomly initialize the pose of the object
    standing_pos, standing_rot = randomize_standing_pose(base_pos=[0.35, 0.1, 0.76], base_rot=[np.pi / 2, 0, 0])
    p.resetBasePositionAndOrientation(obj_id, standing_pos, standing_rot)
    
    # keypoint list
    global state
    global template_3d_kpts
    global init_3d_kpts
    global hanging_3d_kpts
    global template_rgb_copy
    global init_rgb_copy
    global hanging_rgb_copy
    
    # render init RGB-D
    init_rgb, init_depth = render(width, height, view_matrix, projection_matrix, 
                                        far=far, near=near, obj_id=obj_id)
    
    # save init view RGB-D to files
    init_rgb_img = Image.fromarray(init_rgb)
    init_rgb_path = f'{input_dir}/rgb_init.jpg'
    init_rgb_img.save(init_rgb_path)

    init_depth_path = f'{input_dir}/depth_init.npy'
    np.save(init_depth_path, init_depth)

    # template view RGB-D
    template_rgb_path = f'{input_dir}/rgb_template.jpg'
    template_rgb = np.asarray(Image.open(template_rgb_path))
    template_depth_path = f'{input_dir}/depth_template.npy'
    template_depth = np.load(template_depth_path)

    # hanging view RGB-D
    hanging_rgb_path = f'{input_dir}/rgb_hanging.jpg'
    hanging_rgb = np.asarray(Image.open(hanging_rgb_path))
    hanging_depth_path = f'{input_dir}/depth_hanging.npy'
    hanging_depth = np.load(hanging_depth_path)


    template_gripper_trans = np.asarray([
                            [ 0.98954677, -0.10356444,  0.10035739,  0.00496532],
                            [ 0.1071792 ,  0.06254209, -0.99227068,  0.03851797],
                            [ 0.09648739,  0.99265447,  0.07298828, -0.03677358],
                            [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]
                        ])

    repeat = True

    while repeat:

        # keypoint annotation for template image
        print_log('please annotate several keypoints on template_rgb for template pose matching')
        state = 0
        template_3d_kpts = []
        template_rgb_copy = template_rgb.copy()
        cv2.imshow('template_rgb', template_rgb_copy)
        cv2.setMouseCallback('template_rgb', click_event)
        while len(template_3d_kpts) == 0:
            cv2.waitKey(0)
            if len(template_3d_kpts) == 0:
                print_log('please annotate at least one keypoint!!!', 1)
        
        # keypoint annotation for initial pose
        print_log('please annotate several keypoints on init_rgb for template pose matching')
        state = 1
        init_3d_kpts = []
        init_rgb_copy = init_rgb.copy()
        cv2.imshow('init_rgb', init_rgb_copy)
        cv2.setMouseCallback('init_rgb', click_event)
        while len(init_3d_kpts) == 0:
            cv2.waitKey(0)
            if len(init_3d_kpts) == 0:
                print_log('please annotate at least one keypoint!!!', 1)

        # keypoint annotation for target pose
        print_log('please annotate several keypoints on hanging_rgb for template pose matching')
        state = 2
        hanging_3d_kpts = []
        hanging_rgb_copy = hanging_rgb.copy()
        cv2.imshow('hanging_rgb', hanging_rgb_copy)
        cv2.setMouseCallback('hanging_rgb', click_event)
        while len(hanging_3d_kpts) == 0:
            cv2.waitKey(0)
            if len(hanging_3d_kpts) == 0:
                print_log('please annotate at least one keypoint!!!', 1)

        cv2.destroyAllWindows()

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
        template_3d_kpts = load_kpts_from_depth(intrinsic, template_3d_kpts, template_depth_path)
        init_3d_kpts = load_kpts_from_depth(intrinsic, init_3d_kpts, init_depth_path)
        hanging_3d_kpts = load_kpts_from_depth(intrinsic, hanging_3d_kpts, hanging_depth_path)

        num_kpts = template_3d_kpts.shape[1]
        template_3d_kpts_homo = np.vstack((template_3d_kpts, np.full((num_kpts,), 1.0)))
        init_3d_kpts_homo = np.vstack((init_3d_kpts, np.full((num_kpts,), 1.0)))
        template2init = get_src2dst_transform_from_kpts(template_3d_kpts_homo, template_extrinsic, init_3d_kpts_homo, init_extrinsic)

        template_3d_kpts_homo = np.vstack((template_3d_kpts, np.full((num_kpts,), 1.0)))
        hanging_3d_kpts_homo = np.vstack((hanging_3d_kpts, np.full((num_kpts,), 1.0)))
        template2hanging = get_src2dst_transform_from_kpts(template_3d_kpts_homo, template_extrinsic, hanging_3d_kpts_homo, hanging_extrinsic)

        # template_tran = np.linalg.inv(init_extrinsic) @ template2init @ template_extrinsic @ template_3d_kpts_homo
        # for i in range(template_tran.shape[1]):
        #     print_log('template2init result : {} <-> {} error = {}'.format(
        #         template_tran[:,i], init_3d_kpts_homo[:, i], np.linalg.norm(template_tran[:,i] - init_3d_kpts_homo[:, i], ord=2))
        #     )

        # hanging_tran = np.linalg.inv(hanging_extrinsic) @ template2hanging @ template_extrinsic @ template_3d_kpts_homo
        # for i in range(template_tran.shape[1]):
        #     print_log('template2hanging result : {} <-> {} error = {}'.format(
        #         hanging_tran[:,i], hanging_3d_kpts_homo[:, i], np.linalg.norm(hanging_tran[:,i] - hanging_3d_kpts_homo[:, i], ord=2))
        #     )
        
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

        # grasping pose
        gripper_grasping_trans = template2init @ template_gripper_trans
        gripper_grasping_pose = get_7d_pose_from_matrix(gripper_grasping_trans)

        # hanging pose
        gripper_hanging_trans = template2hanging @ template_gripper_trans
        gripper_hanging_pose = get_7d_pose_from_matrix(gripper_hanging_trans)

        # checkpoint, repeat if repeat (pressing n) 
        repeat = checkpoint(1, '[After finishing pose matching and grasping pose finding, let\'s grasp the mug]')

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
    # print_log(joint_poses)

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
    draw_coordinate(gripper_key_pose)
    robot_dense_action(robot, obj_id, gripper_grasping_pose, gripper_key_pose, grasp=True, resolution=action_resolution)
    
    # print_log(f'keypose : {gripper_key_pose}')
    # record current object pose
    obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
    key_obj_pose = obj_pos + obj_rot

    stop = checkpoint(2, 'After grasping the mug, let\'s start to hanging the mug on the hook')
    if stop:
        return

    # --------------- #
    # --- Hanging --- #
    # --------------- #

    gripper_key_trans = get_matrix_from_7d_pose(gripper_key_pose)
    obj_start_pose = get_7d_pose_from_matrix(gripper_key_trans @ np.linalg.inv(template_gripper_trans))

    # avoid collision in initial checking
    obstacles=[hook_id, table_id]
    obj_hanging_pose = get_7d_pose_from_matrix(template2hanging)
    obj_end_pose = refine_obj_pose(physics_client_id, obj_id, obj_hanging_pose, obstacles=obstacles)
    
    # run RRT-connect to find a path for the mug
    obj_waypoints = rrt_connect_7d(physics_client_id, obj_id, start_conf=obj_start_pose, target_conf=obj_end_pose, obstacles=obstacles)

    if obj_waypoints is None:
        print_log("Oops, no solution!", 2)
        return

    gripper_waypoints = []

    for obj_waypoint in obj_waypoints:
        obj_trans = get_matrix_from_7d_pose(obj_waypoint)
        gripper_waypoint = get_7d_pose_from_matrix(obj_trans @ template_gripper_trans)
        gripper_waypoints.append(gripper_waypoint)

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
    gripper_ending_pos = np.asarray(gripper_pose[:3]) + (gripper_rot_matrix @ np.array([[0], [0], [-0.15]])).reshape(3)
    gripper_ending_pose = tuple(gripper_ending_pos) + tuple(gripper_pose[3:])
    robot_dense_action(robot, obj_id, gripper_pose, gripper_ending_pose, grasp=False, resolution=action_resolution)

    print_log('process completed')
    for _ in range(int(1 / SIM_TIMESTEP * 3)): 
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # joint_names, joint_poses, joint_types = get_robot_joint_info(robot.robot_id)
    # for i in range(len(joint_names)):
    #     print_log(f'{joint_names[i]} position={joint_poses[i]} type={joint_types[i]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=str, default='data/mug-Hook_60')
    args = parser.parse_args()
    main(args)