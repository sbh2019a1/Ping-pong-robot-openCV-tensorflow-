import pyrealsense2 as rs
import numpy as np
import cv2
import time
import datetime as dt
import keyboard

import keyboard
import os,sys
from socket import *
import pickle

import logging
import threading

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

pixels_per_meter = 363
center_pixel_x = 935
center_pixel_y = 273

camera_height_centimeter = 192.0
gui_enabled = True

outputArray = []
detectedArray = []
ballNotSeenCounter = 0

outputArrayX = []
outputArrayY = []
outputArrayZ = []
outputArrayTime = []

linePos = 0
lastZ = 4.0
whiteIdx = [1]
realtime = []

img_mask = np.zeros(shape=[540, 960, 3], dtype=np.uint8)
img_print = np.zeros(shape=[540, 960, 3], dtype=np.uint8)
img_print_added = np.zeros(shape=[540, 960, 3], dtype=np.uint8)

image_initialize = True

print('vision_start')


def do_stuff(frames):
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return

    # resulting images are 960*540 (resolution of color image)
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    lower_orange = (10, 160, 160)
    upper_orange = (70, 255, 255)

    # get global variables
    global center_pixel_x
    global center_pixel_y
    global img_mask
    global whiteIdx

    image_center_X = center_pixel_x
    image_center_Y = center_pixel_y

    img_mask = cv2.inRange(color_image_hsv, lower_orange, upper_orange)
    img_mask = cv2.erode(img_mask, None, iterations=2)
    img_mask = cv2.dilate(img_mask, None, iterations=2)

    whiteIdx = cv2.findNonZero(img_mask)

    if whiteIdx is not None and len(whiteIdx) >= 20:
        ballDetected = True
        # print ('detected')

        cnts = cv2.findContours(img_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            largest_circle = max(cnts, key=cv2.contourArea)
            M = cv2.moments(largest_circle)
            image_center_X = int(M["m10"] / M["m00"])
            image_center_Y = int(M["m01"] / M["m00"])
            #cv2.circle(img_mask, (image_center_X, image_center_Y), 5, (0, 0, 255), -1)

        else:
            total_x = 0
            total_y = 0
            for i in range(len(whiteIdx)):
                total_x += whiteIdx[i][0][0]
                total_y += whiteIdx[i][0][1]
            image_center_X = int(round(total_x / len(whiteIdx) * 1.0))
            image_center_Y = int(round(total_y / len(whiteIdx) * 1.0))
            #cv2.circle(img_mask, (image_center_X, image_center_Y), 5, (0, 0, 255), -1)
    else:
        ballDetected = False

    sensor_z = depth_image[image_center_Y][image_center_X]

    world_x = (-image_center_Y + 270) * 100.0 / 363.0
    world_y = (-image_center_X + 480) * 100.0 / 363.0
    world_z = round(camera_height_centimeter - (sensor_z / 10.0), 1)

    global lastZ

    if world_z >= 160 or world_z <= 2:
        world_z = lastZ
        print('triggered')
    else:
        lastZ = world_z

    world_x_calibrated = round((camera_height_centimeter - world_z) * (world_x / camera_height_centimeter), 1)
    world_y_calibrated = round((camera_height_centimeter - world_z) * (world_y / camera_height_centimeter), 1)

    final_x = round(world_x_calibrated + (center_pixel_y - 270) * 100.0 / 363.0, 1)
    final_y = round(world_y_calibrated + (center_pixel_x - 480) * 100.0 / 363.0, 1)

    global realtime
    global linePos
    global outputArray

    global detectedArray

    global ballNotSeenCounter

    global outputArrayX
    global outputArrayY
    global outputArrayZ
    global outputArrayTime

    if ballDetected:
        ballNotSeenCounter = 0
        a = time.time()
        # if len(realtime) == 0:
        realtime.append(a)
        a -= realtime[0]

        row = [final_x, final_y, world_z, a]

        outputArrayX.append(final_x)
        outputArrayY.append(final_y)
        outputArrayZ.append(world_z)
        outputArrayTime.append(a)

        detectedArray.append('1')

        outputArray.append(row)
    else:
        if ballNotSeenCounter <= 10:
            ballNotSeenCounter += 1

        if ballNotSeenCounter >= 10:
            detectedArray = []
            outputArrayX = []
            outputArrayY = []
            outputArrayZ = []
            outputArrayTime = []
            outputArray = []
            realtime = []

    global img_print
    global img_print_added
    global image_initialize

    if image_initialize:
        img_print = img_mask
        image_initialize = False

    img_print_added = cv2.add(img_print, img_mask)

    img_print = img_print_added


while True:
    frames = pipeline.wait_for_frames()
    do_stuff(frames)

    # print(len(detectedArray))
    if len(outputArray) == 20:
        if len(detectedArray) <= 21:
            i = 0
            finalArray = [outputArrayX, outputArrayY, outputArrayZ, outputArrayTime]
            print(finalArray)
            '''
            clientSock = socket(AF_INET, SOCK_STREAM)
            clientSock.connect(('127.0.0.1', 8082))
            ##print('Connection checked')
            clientSock.send(pickle.dumps(finalArray))
            ##print('Data sended')
            '''

        outputArrayX = []
        outputArrayY = []
        outputArrayZ = []
        outputArrayTime = []
        finalArray = []
        outputArray = []
        realtime = []
        time.sleep(1)

    cv2.namedWindow('RealSenseImg')
    cv2.imshow('RealSenseImg', img_print)

    key1 = cv2.waitKey(1)
    if key1 == 27:
        cv2.destroyAllWindows()
        pipeline.stop()
        break

