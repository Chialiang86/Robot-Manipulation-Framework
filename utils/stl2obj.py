'''

1. Download blender

$ wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.93/blender-2.93.2-linux-x64.tar.xz
$ tar -xf blender-2.93.2-linux-x64.tar.xz

2. Run this script, please refer to run.sh

INPUT=('models/myhook/Hook_60/Hook_60.stl' 'models/myhook/Hook_90/Hook_90.stl' 'models/myhook/Hook_180/Hook_180.stl' 'models/myhook/Hook_bar/Hook_bar.stl' 'models/myhook/Hook_long/Hook_long.stl' 'models/myhook/Hook_skew/Hook_skew.stl')
BLENDER='/home/chialiang/Downloads/blender-2.93.2-linux-x64/blender'
for input in "${INPUT[@]}"
do
    $BLENDER -noaudio --background --python utils/stlobj.py -- $input
done

'''

import sys
import os
import bpy

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    input_file = argv[0]
    if not os.path.exists(input_file):
        print(f'{input_file} not exists')
        return

    extension = input_file.split('.')[-1]
    type = 'obj' if extension == 'obj' else 'stl' if extension == 'stl' else 'err'
    assert type != 'err', f'error file extension : {extension}'
    
    output_file = None
    if type == 'obj':
        output_file = input_file.split('.')[0] + '.stl'
    elif type == 'stl':
        output_file = input_file.split('.')[0] + '.obj'

    print(f'processing {input_file} ...')
    bpy.ops.import_mesh.stl(filepath=input_file, axis_forward='-Z', axis_up='Y')
    bpy.ops.export_scene.obj(filepath=output_file, axis_forward='-Z', axis_up='Y')
    print(f'{output_file} saved')
    

if __name__=='__main__':
    main()

