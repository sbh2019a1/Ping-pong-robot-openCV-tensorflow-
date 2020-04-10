import pyrealsense2 as rs
import numpy as np
import cv2
import time
import datetime as dt
import csv
import keyboard
import os,sys

import logging
import threading

# Create realsense pipeline object
pipeline = rs.pipeline()

# Create config object and set resolutions
# supported resolutions
# depth: 1280*720(30Hz), 848*480(90Hz), 640*480(90Hz), 640*360(90Hz)
# color: 1280*720(30Hz), 960*540(60Hz), 848*480(60Hz), 640*360(60Hz)
# using higher FPS or unsupported resolutions will result in a crash
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
# pixels per 100cm
pixels_per_meter = 363
# center coordinates
center_pixel_x = 919
center_pixel_y = 267

calibrationHeight = 191.0

gui_enabled = True
outputArray = []
linePos = 0
lastZ = 1
whiteIdx = [1]

realtime = []

threads = list()
threadCounter = 1

image_print_01 = np.zeros(shape=[540, 960, 3], dtype=np.uint8)
image_initialize = True

print('start')


def do_stuff(frames):
    # print(time.time())
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # Validate that both frames are valid
    if not depth_frame or not color_frame:
        return

    # resulting images are 960*540 (resolution of color image)
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    lower_orange = (10, 160, 160)
    upper_orange = (70, 255, 255)

    img_mask = cv2.inRange(color_image_hsv, lower_orange, upper_orange)

    global center_pixel_x
    global center_pixel_y
    image_center_X = center_pixel_x
    image_center_Y = center_pixel_y

    whiteIdx = cv2.findNonZero(img_mask)

    if whiteIdx is not None and len(whiteIdx) >= 20:
        ballDetected = True
        #print('detected')
        # print(len(whiteIdx))  # number of white pixels in imageqq
        total_x = 0
        total_y = 0
        for i in range(len(whiteIdx)):
            total_x += whiteIdx[i][0][0]
            total_y += whiteIdx[i][0][1]
        image_center_X = int(round(total_x/len(whiteIdx)*1.0))
        image_center_Y = int(round(total_y/len(whiteIdx)*1.0))
    else:
        ballDetected = False

    if image_center_Y > 5 and image_center_Y < 535 and image_center_X > 5 and image_center_X < 955:
        sensor_z_list = [depth_image[image_center_Y][image_center_X],
                         depth_image[image_center_Y + 3][image_center_X],
                         depth_image[image_center_Y][image_center_X + 3],
                         depth_image[image_center_Y - 3][image_center_X],
                         depth_image[image_center_Y][image_center_X - 3],
                         10000]

        sensor_z_list = filter(lambda a: a != 0, sensor_z_list)
        # print(sensor_z_list)
        sensor_z_list.sort()
        sensor_z = sensor_z_list[0]
        # print(sensor_z)
        if sensor_z == 10000:
            sensor_z = 0
            # print('fuck')

    else:
        sensor_z = depth_image[image_center_Y][image_center_X]

    # print(sensor_z)

    world_x = (-image_center_Y + 270) * 100.0 / 363.0
    world_y = (-image_center_X + 480) * 100.0 / 363.0
    world_z = round(calibrationHeight - (sensor_z / 10.0), 1)

    global lastZ

    if world_z >= 160:
        world_z = lastZ
        # print('shit')
    else:
        lastZ = world_z

    # print(world_z)

    world_x_calibrated = round((calibrationHeight - world_z) * (world_x / calibrationHeight), 1)
    world_y_calibrated = round((calibrationHeight - world_z) * (world_y / calibrationHeight), 1)

    final_x = round(world_x_calibrated + (center_pixel_y - 270) * 100.0 / 363.0, 1)
    final_y = round(world_y_calibrated + (center_pixel_x - 480) * 100.0 / 363.0, 1)

    global realtime

    global linePos
    global outputArray

    if ballDetected:
        a = time.time()
        # if len(realtime) == 0:
        realtime.append(a)
        a -= realtime[0]
        row = [final_x, world_z, a]
        outputArray.append(row)
        linePos += 1


    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img_mask, str(final_x) + ' , ' + str(final_y) + ' , ' + str(world_z), (image_center_X, image_center_Y),
    #            font, 1, (100, 100, 100))
    # mark center
    #cv2.circle(img_mask, (center_pixel_x, center_pixel_y), 3, (100, 100, 100), -1)

    global image_print_01
    global image_initialize

    if image_initialize:
        image_print_01 = img_mask
        image_initialize = False

    image_added = cv2.add(image_print_01, img_mask)

    image_print_01 = image_added

    # Show images
    ####cv2.namedWindow('RealSenseImg')
    
    # cv2.namedWindow('RealSenseDepth')
    ####cv2.imshow('RealSenseImg', img_mask)
    # cv2.imshow('RealSenseDepth', depth_colormap)
    # cv2.setMouseCallback('RealSenseImg', pick_color

    

while True:
    frames = pipeline.wait_for_frames()

    #print(time.time())
    x = threading.Thread(target=do_stuff, args=(frames,))
    threads.append(x)
    x.start()

    #do_stuff(frames)

    if keyboard.is_pressed('q'):
        print('q_is_pressed')
        with open('pong_testdata.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            k = len(outputArray)
            if k < 10:
                print('data less than 10 points')
            else:
                for i in range(10):
                    writer.writerow(outputArray[i])
                writer.writerow(outputArray[k - 1])
        print(outputArray)
        outputArray = []
        realtime = []
        linePos = 0
        time.sleep(0.5)
        print('data saved')

    if keyboard.is_pressed('w'):
        print('w_is_pressed')
        outputArray = []
        realtime = []
        linePos = 0
        time.sleep(0.5)
        print('deleted last run')

    cv2.namedWindow('RealSenseImg')
    cv2.imshow('RealSenseImg', image_print_01)

    key2 = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key2 == 27:
        cv2.destroyAllWindows()
        pipeline.stop()
        break

#
