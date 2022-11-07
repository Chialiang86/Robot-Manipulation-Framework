import json
from PIL import Image
import numpy as np
import open3d as o3d


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

if __name__=="__main__":
    
    input_rgb_file = '3d_models/objects/mug/input/rgb.jpg'
    rgb = np.asarray(Image.open(input_rgb_file))
    
    input_depth_file = '3d_models/objects/mug/input/depth.npy'
    depth = np.load(input_depth_file)

    input_caminfo_file = '3d_models/objects/mug/cam_info.json'
    f = open(input_caminfo_file, 'r')
    caminfo_dict = json.load(f)

    gt_mesh_file = '3d_models/objects/mug/base.obj'
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file)

    extrinsic = np.identity(4)
    pcd = create_rgbd(rgb, depth, np.asarray(caminfo_dict["intrinsic"]), np.asarray(caminfo_dict["extrinsic"]), dscale=1)
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, coor, gt_mesh])
