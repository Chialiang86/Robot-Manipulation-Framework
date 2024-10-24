# A Robot Manipulation Framework For PDMS-HW4

### Recommended Environment
- Ubuntu 20.04
- Python >= 3.7 (we test on 3.7 and 3.8)

### Setup
- Highly suggested to use conda virtual environment
```shell 
$ conda create --name [env_name] python=3.7
$ conda activate [env_name]
```
- Install required packages
```shell 
$ pip install -r requirements.txt
```

### Task 1. Forward Kinematic Algorithm and Jacobian Matrix
- Implement and test the forward kinematic algorithm and Jacobian matrix for each pose
```shell 
$ python3 fk.py [--gui] [--visualize-pose]

# --gui : set to show the Pybullet GUI
# --visualize-pose : set to visualize the end-effector pose on PyBullet GUI
```
- execution example
```shell 
$ python3 fk.py 
pybullet build time: May 20 2022 19:43:01
============================ Task 1 : Forward Kinematic ============================

- Difficulty : easy  
- Your Score Of Forward Kinematic : 2.500 / 2.500, Error Count :    0 /  300
- Your Score Of Jacobian Matrix   : 2.500 / 2.500, Error Count :    0 /  300

- Difficulty : normal
- Your Score Of Forward Kinematic : 2.500 / 2.500, Error Count :    0 /  300
- Your Score Of Jacobian Matrix   : 2.500 / 2.500, Error Count :    0 /  300

- Difficulty : hard  
- Your Score Of Forward Kinematic : 2.500 / 2.500, Error Count :    0 /  300
- Your Score Of Jacobian Matrix   : 2.500 / 2.500, Error Count :    0 /  300

- Difficulty : easy_ta  
- Your Score Of Forward Kinematic : 2.500 / 2.500, Error Count :    0 /  700
- Your Score Of Jacobian Matrix   : 2.500 / 2.500, Error Count :    0 /  700

- Difficulty : normal_ta
- Your Score Of Forward Kinematic : 2.500 / 2.500, Error Count :    0 /  700
- Your Score Of Jacobian Matrix   : 2.500 / 2.500, Error Count :    0 /  700

- Difficulty : hard_ta  
- Your Score Of Forward Kinematic : 2.500 / 2.500, Error Count :    0 /  700
- Your Score Of Jacobian Matrix   : 2.500 / 2.500, Error Count :    0 /  700

====================================================================================
- Your Total Score : 30.000 / 30.000
====================================================================================
```

### Task 2. Inverse Kinematic Algorithm
- Implement and test the inverse kinematic algorithm 
    - reference : [Jacobian methods for inverse
kinematics and planning](https://homes.cs.washington.edu/~todorov/courses/cseP590/06_JacobianMethods.pdf)
```shell 
$ python3 ik.py [--visualize-pose]

# --visualize-pose : set to visualize the end-effector pose on PyBullet GUI
```
- execution example
```shell 
$ python3 ik.py

### pybullet log ... ###

============================ Task 2 : Inverse Kinematic ============================

- Difficulty : easy  
- Mean Error : 0.001177
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 5.000 / 5.000

- Difficulty : normal
- Mean Error : 0.001443
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 5.000 / 5.000

- Difficulty : hard  
- Mean Error : 0.001811
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 5.000 / 5.000

- Difficulty : easy_ta  
- Mean Error : 0.001247
- Error Count :   0 / 200
- Your Score Of Inverse Kinematic : 5.000 / 5.000

- Difficulty : normal_ta
- Mean Error : 0.001456
- Error Count :   0 / 200
- Your Score Of Inverse Kinematic : 5.000 / 5.000

- Difficulty : hard_ta  
- Mean Error : 0.001969
- Error Count :   0 / 200
- Your Score Of Inverse Kinematic : 5.000 / 5.000

====================================================================================
- Your Total Score : 30.000 / 30.000
====================================================================================

### pybullet log ... ###

```
- PyBullet GUI (with end-effector pose visualization)
    - red : your pose
    - green : ground truth pose

![](https://i.imgur.com/vHT2Txl.png)

### The Whole Manipultion Pipline
- See `manipulation.py`
- This script will execute the whole manipulation pipeline (hanging task) which covers
    - 1. pose matching for object grasping pose and target pose by handcrafted keypoints
    - 2. control the robot to move to desired poses by using the inverse kinematic algorithm you designed in `ik.py`
    - 3. run RRT-Connect algorithm to find a collision free path to execute manipulation task
- Run this script
```shell 
$ python3 manipulation.p --input-dir [data/[obj_name]-[hook_name]]

# --input-dir: the task information defined in data/
# default : data/mug_19-Hook_60
```
- Keypoint annotation for three images (template, initial pose, target pose)

![](https://i.imgur.com/epDIBaN.jpg)

- Moving the robot  

![](https://i.imgur.com/CrVsbr8.png)

- Motion planning for manipulation

![](https://i.imgur.com/YbVYK7S.png)

