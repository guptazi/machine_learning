from numpy import asarray
import numpy as np
from os import listdir
from os.path import isdir
from numpy import savez_compressed
from numpy import asarray
import pickle
def pose_normalize(p):
    print('First person:\n', p)
    x = p[0][0]
    y = p[0][1]
    partsNum = 17
    for i in range( partsNum ):
        p[i][0] -=  x
        p[i][1] -=  y
        p[i][2] -=  x
        p[i][3] -=  y
    return p
def load_pose(filename):
    with open(filename, "rb") as fp:   #Pickling
        pose = pickle.load(fp)

    pose = pose_normalize(pose)    
    return pose
def load_all_poses(directory):
    poses = list()
    for filename in listdir(directory):
        path = directory + filename
        pose = load_pose(path)
        poses.append(pose)
    return poses
def load_posedata(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        pose = load_all_poses(path)
        labels = [subdir for _ in range(len(pose))] # generate label list
        X.extend(pose) # add faces to X
        y.extend(labels) # add label to y
    return asarray(X), asarray(y)
poseX, poseY = load_posedata('pose/')
print('First normalized pose:\n', poseX[0])
print("Pose label : \n", poseY)
savez_compressed('pose_database.npz', poseX, poseY)


