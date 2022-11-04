#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import trimesh
import pybullet as p

def open3d_to_trimesh(src):
    """Convert mesh from open3d to trimesh

    https://github.com/wkentaro/morefusion/blob/b8b892b3fbc384982a4929b1418ee29393069b11/morefusion/utils/open3d_to_trimesh.py

    Parameters
    ----------
    src : open3d.open3d.geometry.TriangleMesh

    Returns
    -------
    dst : trimesh.base.Trimesh

    Raises
    ------
    ValueError
        when type of src is not open3d.open3d.geometry.TriangleMesh
    """
    if isinstance(src, o3d.geometry.TriangleMesh):
        vertex_colors = None
        if src.has_vertex_colors:
            vertex_colors = np.asarray(src.vertex_colors)
        dst = trimesh.Trimesh(
            vertices=np.asarray(src.vertices),
            faces=np.asarray(src.triangles),
            vertex_normals=np.asarray(src.vertex_normals),
            vertex_colors=vertex_colors,
        )
    else:
        raise ValueError("Unsupported type of src: {}".format(type(src)))

    return dst

def centerize_mesh(mesh):
    """Move mesh to its center

    Parameters
    ----------
    mesh : trimesh.base.Trimesh or open3d.open3d.geometry.TriangleMesh

    Returns
    -------
    mesh : trimesh.base.Trimesh
    center : list[float]
    """
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh = open3d_to_trimesh(mesh)
    center = np.mean(mesh.vertices, axis=0)
    mesh.vertices -= center
    return mesh, center

def create_urdf(mesh, file):
    """Create urdf from mesh

    Parameters
    ----------
    mesh : trimesh.base.Trimesh or open3d.open3d.geometry.TriangleMesh
        input mesh
    output_path : str
        Ouput file where output mesh saved
    init_texture : bool
        If true, make the mesh texture the same as the base one.
        This is necessary if you want to change the texture when rendering with
        https://github.com/kosuke55/hanging_points_cnn/blob/master/hanging_points_cnn/create_dataset/renderer.py

    Returns
    -------
    output_file : str
        output file path
    """
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh = open3d_to_trimesh(mesh)
    center = np.mean(mesh.vertices, axis=0)
    # mesh.vertices -= center
    print(center)

    dirname, filename = os.path.split(file)

    base_urdf_path = 'models/base/base.urdf'
    tree = ET.parse(base_urdf_path)
    root = tree.getroot()
    center = ''.join(str(i) + ' ' for i in mesh.centroid.tolist()).strip()
    root[0].find('inertial').find('origin').attrib['xyz'] = center

    mesh.export(os.path.join(dirname, 'base.obj'), 'obj')

    p.vhacd(os.path.join(dirname, 'base.obj'), os.path.join(dirname, 'base.obj'), os.path.join(dirname, 'base.txt'))

    tree.write(os.path.join(dirname, 'base.urdf'), encoding='utf-8', xml_declaration=True)

def main(args):

    print('success in.')

    input_dir = args.input_dir
    # base_save_dir = args.save_dir
    # os.makedirs(base_save_dir, exist_ok=True)

    target_length = 0.1

    files = glob.glob(f'{input_dir}/*/*_normalized.obj')
    # ignore_list = ['bag', 'wrench']
    ignore_list = []

    for file in files:

        for ignore_item in ignore_list:
            if ignore_item in file:
                print(f'ignore : {ignore_item}')

        dirname, filename = os.path.split(file)
        filename_without_ext, ext = os.path.splitext(filename)

        try:
            mesh = o3d.io.read_triangle_mesh(file)
            mesh = mesh.simplify_vertex_clustering(
                voxel_size=0.01,
                contraction=o3d.geometry.SimplificationContraction.Average)
            mesh = open3d_to_trimesh(mesh)
            mesh_invert = mesh.copy()
            mesh_invert.invert()
            mesh += mesh_invert
            mesh.merge_vertices()
            size = np.array(np.max(mesh.vertices, axis=0)
                            - np.min(mesh.vertices, axis=0))
            length = np.max(size)
            mesh.vertices = mesh.vertices * target_length / length

        except Exception as e:
            print('skip {}'.format(file))
            continue
        
        if mesh.vertices.shape[0] > 1 and mesh.faces.shape[0] > 1:
            print(f'processing {file}')
            create_urdf(mesh, file)
            print(f'{dirname}/base.urdf saved')
        else:
            print('skip {}'.format(file))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input-dir',
        '-i',
        type=str,
        help='input directory',
        default='')
    # parser.add_argument(
    #     '--save-dir',
    #     '-s',
    #     type=str,
    #     help='save directory',
    #     default='')

    args = parser.parse_args()

    main(args)
