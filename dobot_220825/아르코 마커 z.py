from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import cv2
import numpy as np
import imutils
import threading
import math
import time
import DobotDllType as dType
from numpy.linalg import inv

import cv2.aruco as aruco
import os

from multiprocessing import Process, Pipe, Queue, Value, Array, Lock

import numpy.random as rnd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.dilate(cv2.erode(cb_th, None, iterations=2), None, iterations=2)
    #cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance(x, y, imagePoints):
    
    objectPoints = np.array([[33.5,65,0], #33
                            [33.5,75,0],
                            [43.5,75,0], #23
                            [43.5,65,0],],dtype = 'float32')

    fx = float(470.5961)
    fy = float(418.18176)
    cx = float(275.7626)
    cy = float(240.41246)
    k1 = float(0.06950)
    k2 = float(-0.07445)
    p1 = float(-0.01089)
    p2 = float(-0.01516)

    #cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    #distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')

    cameraMatrix = np.array([[470.5961,0,275.7626],[0,418.18176,240.41246],[0,0,1]],dtype = 'float32')
    distCoeffs = np.array([0.06950,-0.07445,-0.01089,-0.01516],dtype = 'float32')
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
            
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    px = Pw[0]
    py = Pw[1]

    #print("px: %f, py: %f" %(px,py))

    return px,py

def get_distance_z(x, y, imagePoints):
    
    objectPoints = np.array([[65,0,0], #33
                            [75,0,0],
                            [75,10,0], #23
                            [65,10,0],],dtype = 'float32')
    
    fx = float(266.90047006)
    fy = float(105.7428056)
    cx = float(310.93899886)
    cy = float(241.40478113)

    cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float32')
    #distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')

    #cameraMatrix = np.array([[470.5961,0,275.7626],[0,418.18176,240.41246],[0,0,1]],dtype = 'float32')
    distCoeffs = np.array([[-0.13826547], [-0.14284035], [0.1761695], [0.03215271], [0.04492665]], dtype='float32')
    
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
            
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    py = Pw[0]
    pz = Pw[1]

    #print("px: %f, py: %f" %(px,py))

    #print("Pw: ", Pw)

    return py,pz

def find_ball(frame,cb_th,box_points):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    px = None
    py = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            px,py = get_distance(center[0], center[1],box_points)
            
            text = " %f , %f" %(px,py)
            cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return px,py

def find_ball_z(frame,cb_th,box_points):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    py = None
    pz = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            py,pz = get_distance_z(center[0], center[1],box_points)
            
            text = " %f , %f" %(py,pz)
            cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            #print("py: %f, pz: %f" %(py,pz))
            #print("\n")

    return py,pz

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        cv2.aruco.drawDetectedMarkers(img,bboxs)
        #print(len(bboxs))

    if len(bboxs) > 0:
        return bboxs[0][0]
    else:
        return [0,0]

def main():

    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    while(1):

        _,frame = cap.read()
        _,frame2 = cap2.read()
        
        box_points = findArucoMarkers(frame)
        print('find1')
        print(len(box_points))
        box_points2 = findArucoMarkers(frame2)
        print('find2')
        print(len(box_points2))


        #cv2.imshow('1',frame)
        #cv2.imshow('2',frame2)

        if len(box_points) > 2: #and len(box_points2) > 2:
            break

    #cv2.destroyAllWindows()

    ball_x = []
    ball_y = []
    ball_y2 = []
    ball_z = []

    while(1):

        _,frame = cap.read()
        _,frame2 = cap2.read()
        
        cb_th = segmentaition(frame)
        cb_th2 = segmentaition(frame2)
        
        px,py = find_ball(frame,cb_th,box_points)
        py2,pz = find_ball_z(frame2,cb_th2,box_points2)

        #pz = find_ball_z(frame,cb_th,box_points)
        
        if py != None and pz != None:

            print("px: %f, py: %f, pz: %f" %(px,py,pz))
            
            ball_x.append(px)
            ball_y.append(py)
            ball_y2.append(py2)
            ball_z.append(pz)

        else:

            if len(ball_x) > 0 and len(ball_z) > 0:

                plt.subplot(2,1,1)
                plt.xlabel('Y-pose')
                plt.ylabel('X-pose')
                plt.ylim([0,67])
                plt.xlim([0,180])
                plt.plot(ball_y,ball_x,'b.-')

                plt.subplot(2,1,2)
                plt.xlabel('Y-pose')
                plt.ylabel('Z-pose')
                plt.xlim([0,165])
                plt.ylim([-10,30])
                plt.plot(ball_y,ball_z,'b.-')

                plt.show()

            ball_x.clear()
            ball_y.clear()
            ball_z.clear()
            ball_y2.clear()

        cv2.imshow('cam',frame)
        cv2.imshow('cam2',frame2)
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

main()

        
