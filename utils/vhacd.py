import argparse
import os
import glob
from tqdm import tqdm
import pybullet as p


def main(args):
    input_dir = args.input_dir
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    pathes = glob.glob(f'{input_dir}/*/base.obj')
    for path in tqdm(pathes):
        if '_' in path:
            continue
        print(f'processing {path} ...')
        p.vhacd(path, path, f'{path}.txt')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=str)
    args = parser.parse_args()
    main(args)