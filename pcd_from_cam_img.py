import glob, os, time, json, argparse
import pybullet as p
import numpy as np
import pybullet_data
import scipy.io as sio
import open3d as o3d
import xml.etree.ElementTree as ET

from PIL import Image
from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_matrix_from_pos_rot

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


def create_rgbd(rgb, depth, intr, extr, dscale, depth_threshold=2.0):
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

def main(args):

  urdl_file = args.input_urdf
  if not os.path.exists(urdl_file):
    print(f'{urdl_file} not exists')
    return 

  p.connect(p.GUI)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())

  p.resetDebugVisualizerCamera(
      cameraDistance=0.3,
      cameraYaw=120,
      cameraPitch=-30,
      cameraTargetPosition=[0.0, 0.0, 0.0]
  )
  p.resetSimulation()
  p.setPhysicsEngineParameter(numSolverIterations=150)
  sim_timestep = 1.0 / 240
  p.setTimeStep(sim_timestep)
  p.setGravity(0, 0, -9.8)
  rgb_cam_param = {
    'cameraEyePosition': [0.0, 0.0, 0.2],
    'cameraTargetPosition': [0.0, 0.0, 0.0],
    'cameraUpVector': [0.0, 1.0, 0.0],
  }

  # intrinsic matrix
  width, height = 512, 512
  fx = fy = 605
  cx = 0.5 * width
  cy = 0.5 * height

  # config camera params : https://www.intelrealsense.com/depth-camera-d435/
  # D435 serial_num=141722071222, fx=605, fy=605, cx=323, cy=240
  # fov = 2 * atan(480/605) ~= 1.34139295 rad ~= 76.8561560146
  far = 1000.
  near = 0.01
  fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi
  depth_threshold = 2.0

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
  
  origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

  print(f'processing {urdl_file}')
  obj_id = p.loadURDF(urdl_file, [0, 0, 0], [0, 0, 0, 1])

  # reset_camera(yaw, pitch, cameraDistance)

  time.sleep(0.05)
  img = p.getCameraImage(width, height, viewMatrix=rgb_view_matrix, projectionMatrix=projection_matrix)
  rgb_buffer = np.reshape(img[2], (height, width, 4))[:,:,:3]
  depth_buffer = np.reshape(img[3], [height, width])

  # get real depth
  depth_buffer = far * near / (far - (far - near) * depth_buffer)

  # write RGB-D
  rgb_img = Image.fromarray(rgb_buffer)
  rgb_path = f'3d_models/objects/mug/input/rgb_src.jpg'
  rgb_img.save(rgb_path)

  depth_path = f'3d_models/objects/mug/input/depth_src.npy'
  np.save(depth_path, depth_buffer)

  # intrinsic
  intrinsic = np.array([
    [fx, 0., cx],
    [0., fy, cy],
    [0., 0., 1.]
  ])

  # rotation vector extrinsic
  z = np.asarray(rgb_cam_param['cameraTargetPosition']) - \
      np.asarray(rgb_cam_param['cameraEyePosition'])
  z /= np.linalg.norm(z, ord=2)
  y = -np.asarray(rgb_cam_param['cameraUpVector'])
  y -= (np.dot(z, y)) * z
  y /= np.linalg.norm(y, ord=2)
  x = np.cross(y, z)

  # extrinsic
  extrinsic = np.identity(4)
  extrinsic[:3, 0] = x
  extrinsic[:3, 1] = y
  extrinsic[:3, 2] = z
  extrinsic[:3, 3] = np.asarray(rgb_cam_param['cameraEyePosition'])
  # extrinsic[:3, 3] = np.asarray(rgb_cam_param['cameraTargetPosition']) - \
  #                           np.asarray(rgb_cam_param['cameraEyePosition'])

  # write to 3d_models/objects/mug/
  intrinsic_dict = {
    'width': width,
    'height': height,
    'intrinsic': intrinsic.tolist(),
    'extrinsic': extrinsic.tolist()
  }

  # output camera information
  cam_info_path = f'3d_models/objects/mug/cam_info.json'
  with open(cam_info_path, 'w') as f_out:
    json.dump(intrinsic_dict, f_out, indent=4)

  # pcd = create_rgbd(rgb_buffer, depth_buffer, intrinsic, extrinsic, dscale=1, depth_threshold=depth_threshold)

  # mesh_file = os.path.splitext(urdl_file)[0] + '.obj'
  # pcd_ori = o3d.io.read_triangle_mesh(mesh_file)
  # pcd_ori.scale(scale, [0., 0., 0.,])
  
  # save ply
  # o3d.visualization.draw_geometries([origin, pcd_merged], point_show_normal=False)
  # output_ply_path = os.path.splitext(urdl_file)[0] + '.ply'
  # o3d.io.write_point_cloud(output_ply_path, pcd)

  # print(f'{output_ply_path} and {output_jpg_path} saved')
  # p.removeBody(obj_id)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_urdf', '-id', type=str, default='3d_models/objects/mug/base.urdf')
  args = parser.parse_args()

  main(args)