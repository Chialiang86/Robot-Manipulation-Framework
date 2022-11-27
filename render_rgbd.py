import glob, os, time, json, argparse
import pybullet as p
import numpy as np
import pybullet_data
import open3d as o3d
import xml.etree.ElementTree as ET

from PIL import Image
from scipy.spatial.transform import Rotation as R

from utils.motion_planning_utils import get_collision7d_fn

SIM_TIMESTEP = 1.0 / 240

def load_obj_urdf(urdf_path, pos=[0, 0, 0], rot=[0, 0, 0]):

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    center = np.array(
      [
        float(i) for i in root[0].find(
          "inertial"
        ).find(
          "origin"
        ).attrib['xyz'].split(' ')
      ]
    )
    scale = np.array(
      [
        float(i) for i in root[0].find(
          "visual"
        ).find(
          "geometry"
        ).find(
          "mesh"
        ).attrib["scale"].split(" ")
      ]
    )
    assert scale[0] == scale[1] and scale[0] == scale[2] and scale[1] == scale[2], f"scale is not uniformed : {scale}"
    obj_id = p.loadURDF(urdf_path, pos, p.getQuaternionFromEuler(rot))
    return obj_id, center, scale[0]


def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def get_projmat_and_intrinsic(width, height, fx, fy, far, near):

  cx = width / 2
  cy = height / 2
  fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

  project_matrix = p.computeProjectionMatrixFOV(
                      fov=fov,
                      aspect=1.0,
                      nearVal=near,
                      farVal=far
                    )
  
  intrinsic = np.array([
                [ fx, 0.0,  cx],
                [0.0,  fy,  cy],
                [0.0, 0.0, 1.0],
              ])
  
  return project_matrix, intrinsic

def get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector):


    view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cameraEyePosition,
                    cameraTargetPosition=cameraTargetPosition,
                    cameraUpVector=cameraUpVector
                  )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, f'cameraTargetPosition and cameraEyePosition is at same location'
    z /= norm
   
    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, f'cameraUpVector is parallel to z axis'
    y /= norm
    
    x = cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return view_matrix, extrinsic

def write_rgbd(output_dir, width, height, view_matrix, projection_matrix, far, near, file_postfix='init'):

    assert os.path.exists(output_dir), f'{output_dir} not exists'

    anno_in_dir = f'{output_dir}'
    os.makedirs(anno_in_dir, exist_ok=True)

    img = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
    rgb_buffer = np.reshape(img[2], (height, width, 4))[:,:,:3]
    depth_buffer = np.reshape(img[3], [height, width])

    # get real depth
    depth_buffer = far * near / (far - (far - near) * depth_buffer)

    # write RGB-D
    rgb_img = Image.fromarray(rgb_buffer)
    rgb_path = f'{anno_in_dir}/rgb_{file_postfix}.jpg'
    rgb_img.save(rgb_path)

    depth_path = f'{anno_in_dir}/depth_{file_postfix}.npy'
    np.save(depth_path, depth_buffer)
  
def create_pcd_from_rgbd(rgb, depth, intr, extr, dscale, depth_threshold=2.0):
    assert rgb.shape[:2] == depth.shape
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
    center = np.mean(points, axis=0)
    colors = np.reshape(rgb,(depth.shape[0] * depth.shape[1], 3))
    colors = np.array([colors[:,2], colors[:,1], colors[:,0]]).T / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(extr)

    return pcd

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

def main(args):

  urdl_file = args.obj_urdf
  if not os.path.exists(urdl_file):
    print(f'{urdl_file} not exists')
    return 
  
  input_dir = args.input_dir
  if not os.path.exists(input_dir):
    print(f'{input_dir} not exists')
    return
  
  json_files = glob.glob(f'{input_dir}/*.json')

  physics_client_id = p.connect(p.GUI)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())

  p.resetDebugVisualizerCamera(
      cameraDistance=0.2,
      cameraYaw=90,
      cameraPitch=-30,
      cameraTargetPosition=[0.5, 0.0, 1.2]
  )
  p.resetSimulation()
  p.setPhysicsEngineParameter(numSolverIterations=150)
  p.setTimeStep(SIM_TIMESTEP)
  p.setGravity(0, 0, -9.8)

  # ----------------------#
  # --- Camera params --- #
  # ----------------------#

  # intrinsic matrix (fixed)
  # config camera params : https://www.intelrealsense.com/depth-camera-d435/
  # D435 serial_num=141722071222, fx=605, fy=605, cx=323, cy=240
  # fov = 2 * atan(480/605) ~= 1.34139295 rad ~= 76.8561560146
  width, height = 512, 512
  fx = fy = 605
  cx = 0.5 * width
  cy = 0.5 * height
  far = 1000.
  near = 0.01
  projection_matrix, intrinsic = get_projmat_and_intrinsic(width, height, fx, fy, far, near)

  # render the object in hanging pose 
  for json_file in json_files:
    f = open(json_file, 'r')
    json_dict = json.load(f)

    # for _ in range(int(1 / SIM_TIMESTEP * 1)):
    #     p.stepSimulation()
    #     time.sleep(SIM_TIMESTEP)

    obj_name = json_dict['obj_path'].split('/')[-2]
    hook_name = json_dict['hook_path'].split('/')[-2]
    output_dir = f'data/{obj_name}-{hook_name}'
    os.makedirs(output_dir, exist_ok=True)

    # extrinsic matrix for template
    obj_id = p.loadURDF(json_dict['obj_path'])
    p.resetBasePositionAndOrientation(obj_id, [0, 0, 0], [0, 0, 0, 1])

    time.sleep(1.0)

    cameraEyePosition = [0.0, 0.0, -0.2]
    cameraTargetPosition = [0.0, 0.0, 0.0]
    cameraUpVector = [0.0, 1.0, 0.0]
    view_matrix, template_extrinsic = get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector)
    write_rgbd(output_dir, width, height, view_matrix, projection_matrix, far, near, file_postfix='template')

    # extrinsic matrix for hanging pose
    hook_pos = json_dict['contact_info']['hook_pose'][:3]
    hook_rot = json_dict['contact_info']['hook_pose'][3:]
    hook_id = p.loadURDF(json_dict['hook_path'], hook_pos, hook_rot)
    obj_pos = json_dict['contact_info']['obj_pose'][:3]
    obj_rot = json_dict['contact_info']['obj_pose'][3:]
    obj_pose = obj_pos + obj_rot
    refined_obj_pose = refine_obj_pose(physics_client_id, obj_id, obj_pose, obstacles=[hook_id])
    p.resetBasePositionAndOrientation(obj_id, refined_obj_pose[:3], refined_obj_pose[3:])
    
    time.sleep(1.0)
    
    cameraEyePosition = [0.5, 0.2, 1.3]
    cameraTargetPosition = [0.5, -0.1, 1.3]
    cameraUpVector = [0.0, 0.0, 1.0]
    view_matrix, hanging_extrinsic = get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector)
    write_rgbd(output_dir, width, height, view_matrix, projection_matrix, far, near, file_postfix='hanging')

    p.removeBody(obj_id)
    p.removeBody(hook_id)

    # write to 3d_models/objects/mug_[id]/
    info_dict = {
      'obj_path': json_dict['obj_path'],
      'hook_path': json_dict['hook_path'],
      'hook_pose': json_dict['contact_info']['hook_pose'],
      'width': width,
      'height': height,
      'intrinsic': intrinsic.tolist(),
      'template_extrinsic': template_extrinsic.tolist(),
      'hanging_extrinsic': hanging_extrinsic.tolist()
    }

    # output camera information
    info_path = f'{output_dir}/info.json'
    with open(info_path, 'w') as f_out:
      json.dump(info_dict, f_out, indent=4)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', '-id', type=str, default='3d_models/hanging_pose')
  parser.add_argument('--obj_urdf', '-urdf', type=str, default='3d_models/objects/mug_67/base.urdf')
  args = parser.parse_args()

  main(args)