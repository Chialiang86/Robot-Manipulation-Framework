import sys, os
import webbrowser 

if __name__=="__main__":
    if len(sys.argv) != 3:
        print('argv should contain 3 args')
        exit(-1)
    
    obj_root_dir = sys.argv[1]
    port = sys.argv[2]

    url = f'http://localhost:{port}'
    webbrowser.open(url)
    os.system(f"python3 -m wat.run --data-dir {obj_root_dir} --port {port} --maxtips 4")