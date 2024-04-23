import os
import sys
import argparse
import cv2
import time
import numpy as np
from numpy import load
from config_reader import config_reader
from processingUpdate import extract_parts
from processingUpdate2 import draw2
from model.cmu_model import get_testing_model
from collections import Counter


def pose_measure(p1, p2):
    # find height of p1
    measure = 0
    neck1 = np.array((p1[0][0], p1[0][1]))
    rightAnkle1 = np.array((p1[10][2], p1[10][3]))
    leftAnkle1 = np.array((p1[11][2], p1[11][3]))
    is_rightAnkle1_zero = np.all((rightAnkle1 == 0))
    is_leftAnkle1_zero = np.all((leftAnkle1 == 0))
    if is_rightAnkle1_zero is True:
        if is_leftAnkle1_zero is True:
            measure = float('inf')
        else:            
            dist1 = np.linalg.norm(neck1 - leftAnkle1)
    else:
        dist1 = np.linalg.norm(neck1 - rightAnkle1)

    if measure == float('inf'):
        return measure
    
    # height of p2
    measure = 0
    neck2 = np.array((p2[0][0], p2[0][1]))
    rightAnkle2 = np.array((p2[10][2], p2[10][3]))
    leftAnkle2 = np.array((p2[11][2], p2[11][3]))
    is_rightAnkle2_zero = np.all((rightAnkle2 == 0))
    is_leftAnkle2_zero = np.all((leftAnkle2 == 0))
    if is_rightAnkle2_zero is True:
        if is_leftAnkle2_zero is True:
            measure = float('inf')
        else:            
            dist2 = np.linalg.norm(neck2 - leftAnkle2)
    else:
        dist2 = np.linalg.norm(neck2 - rightAnkle2)

    if measure == float('inf'):
        return measure

    k = dist1 / dist2
    
    partsNum = 17
    for i in range( partsNum ):
        p2[i][0] = p2[i][0] * k
        p2[i][1] = p2[i][1] * k
        p2[i][2] = p2[i][2] * k
        p2[i][3] = p2[i][3] * k
    
    partsNumUsed = 12 # neck to ankle
    measure = 0
    for i in range( partsNumUsed ):
        point1 = np.array((p1[i][0], p1[i][1]))
        point2 = np.array((p2[i][0], p2[i][1]))
        point3 = np.array((p1[i][2], p1[i][3]))
        point4 = np.array((p2[i][2], p2[i][3]))
        measure = measure + np.linalg.norm(point1 - point2)
        measure = measure + np.linalg.norm(point3 - point4)

    return measure

def pose_matching(pose_db, pose_detected): 
    p = pose_detected[0] # get the first person
    measure = []     
    len_db = len(pose_db)
    x = p[0][0]
    y = p[0][1]
    
    partsNum = 17
    for i in range( partsNum ):
        p[i][0] -=  x
        p[i][1] -=  y
        p[i][2] -=  x
        p[i][3] -=  y
    for i in range (len_db):
        m = pose_measure(pose_db[i], p)
        measure.append(m)
    
    # find min measure
    minpos = measure.index(min(measure))
    return minpos

def display_result(image, txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 350)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(image, txt, org, font, fontScale, \
                        color, thickness, cv2.LINE_AA)
    
data = load('pose_database.npz')
poseX, poseY = data['arr_0'], data['arr_1']

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

keras_weights_file = 'model/keras/model.h5'
frame_rate_ratio = 10
process_speed = 4
ending_frame = None

print('start processing...')

# Video input
video = 'c1-1-2.avi'
video_path = 'videos/'
video_file = video_path + video
print("Input video name: ", video_file)
    
# Output location
output_path = 'videos/outputs/'
output_format = '.mp4'
video_output = output_path + video + str(start_datetime) + output_format

model = get_testing_model()
model.load_weights(keras_weights_file)
params, model_params = config_reader()

cam = cv2.VideoCapture(video_file)
print('video_file:', video_file)
input_fps = cam.get(cv2.CAP_PROP_FPS)
ret_val, orig_image = cam.read()
video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

if ending_frame is None:
        ending_frame = video_length

output_fps = input_fps / frame_rate_ratio
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, output_fps, \
                      (orig_image.shape[1], orig_image.shape[0]))

scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
scale_search = scale_search[0:process_speed]
params['scale_search'] = scale_search
i = 0  # default is 0
result = []
while(cam.isOpened()) and ret_val is True and i < ending_frame:
    ret_val, orig_image = cam.read()
    if orig_image is None :
        break
    if i % frame_rate_ratio == 0:
        input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
        tic = time.time()
        body_parts, all_peaks, subset, candidate = extract_parts(input_image, \
                                            params, model, model_params)
        canvas, human = draw2(orig_image, all_peaks, subset, candidate)
        if human != [] and human[0][0][0] != 0 and human[0][0][1] != 0:
            index = pose_matching(poseX, human)
            res = poseY[index]
            result.append(res)
            resultStr = 'A person is ' + res
            display_result(canvas, resultStr)
        cv2.imshow('Canvas Image', canvas)
        cv2.waitKey(5)
        toc = time.time()
        print('processing time is %.5f' % (toc - tic))
        out.write(canvas)

    i += 1



