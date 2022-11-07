import sys, os
import numpy as np
import torch
import open3d as o3d
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

def main(): 
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()

    # # Set the device
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")
    #     print("WARNING: CPU only, this will be slow!")

    obj_file = sys.argv[1]
    print(f'processing {obj_file} ...')


    verts, faces, aux = load_obj(obj_file)
    faces_idx = faces.verts_idx
    # faces_idx = faces.verts_idx.to(device)
    meshes = Meshes(verts=[verts], faces=[faces_idx])
    points = sample_points_from_meshes(meshes, 20000)
    points = points.detach().cpu().squeeze().numpy()
    # colors = np.array([[0.5, 0.8, 0.8] for i in range(points.shape[0])])

    output_pcd_file = os.path.splitext(obj_file)[0] + '.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coor_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coor_gripper.transform(np.asarray([[ 0.98954677, -0.10356444,  0.10035739,  0.00496532],
                                       [ 0.1071792 ,  0.06254209, -0.99227068,  0.03851797],
                                       [ 0.09648739,  0.99265447,  0.07298828, -0.03677358],
                                       [ 0.0       ,  0.0       ,  0.0       ,  1.0       ]]))
    o3d.visualization.draw_geometries([pcd, coor, coor_gripper])
    o3d.io.write_point_cloud(output_pcd_file, pcd)


if __name__=="__main__":
    main()