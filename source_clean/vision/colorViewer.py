import pyrealsense2 as rs
import numpy as np
import cv2
import time
import datetime as dt
import csv
import keyboard
import os,sys

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
whiteIdx = []
color_image_hsv = None

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi', fourcc, 60.0, (960, 540))


print('start')

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = color_image_hsv[y, x]

        # HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel[0], pixel[1], pixel[2])

        # A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS
        image_mask = cv2.inRange(color_image_hsv, lower, upper)
        cv2.imshow("Mask", image_mask)

try:
    while True:
        # print(time.time())
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # resulting images are 960*540 (resolution of color image)
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        writer.write(color_image_hsv)

        lower_orange = (23 - 5, 180, 180)
        upper_orange = (23 + 5, 255, 255)

        img_mask = cv2.inRange(color_image_hsv, lower_orange, upper_orange)

        #print(cv2.countNonZero(img_mask))

        M = cv2.moments(img_mask)

        # image_center_X and image_center_Y are pixel based coordinates
        image_center_X = center_pixel_x
        image_center_Y = center_pixel_y

        #cv2.findNonZero(img_mask, whiteIdx)

        #print(whiteIdx)


        if M["m00"] != 0:
            image_center_X = int(M["m10"] / M["m00"])
            image_center_Y = int(M["m01"] / M["m00"])
            ballDetected = True
            #print('detected')
        else:
            ballDetected = False


        if image_center_Y>5 and image_center_Y<535 and image_center_X>5 and image_center_X<955:
            sensor_z_list = [depth_image[image_center_Y][image_center_X],depth_image[image_center_Y+3][image_center_X],
                             depth_image[image_center_Y][image_center_X+3],depth_image[image_center_Y-3][image_center_X],
                             depth_image[image_center_Y][image_center_X-3], 10000]
            sensor_z_list = filter(lambda a: a != 0, sensor_z_list)
            #print(sensor_z_list)
            sensor_z_list.sort()
            sensor_z = sensor_z_list[0]
            #print(sensor_z)
            if sensor_z == 10000:
                sensor_z = 0
                #print('fuck')

        else:
            sensor_z = depth_image[image_center_Y][image_center_X]

        #print(sensor_z)

        world_x = (-image_center_Y+270)*100.0/363.0
        world_y = (-image_center_X+480)*100.0/363.0
        world_z = round(calibrationHeight - (sensor_z / 10.0), 1)

        if world_z >= 160:
            world_z = lastZ
            #print('shit')
        else:
            lastZ = world_z

        #print(world_z)

        world_x_calibrated = round((calibrationHeight - world_z) * (world_x / calibrationHeight), 1)
        world_y_calibrated = round((calibrationHeight - world_z) * (world_y / calibrationHeight), 1)

        final_x = round(world_x_calibrated+(center_pixel_y-270)*100.0/363.0, 1)
        final_y = round(world_y_calibrated+(center_pixel_x-480)*100.0/363.0, 1)

        if ballDetected:
            row = [final_x, final_y, world_z, dt.datetime.today().time().microsecond]
            outputArray.insert(linePos, row)
            linePos += 1

        if keyboard.is_pressed('q'):
            print('q_is_pressed')
            time.sleep(0.5)
            print('data saved')
        if keyboard.is_pressed('w'):
            print('w_is_pressed')
            outputArray = []
            linePos = 0
            time.sleep(0.5)
            print('deleted last run')



        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_mask, str(final_x) + ' , ' + str(final_y) + ' , ' + str(world_z), (image_center_X, image_center_Y), font, 1, (100, 100, 100))
        # mark center
        cv2.circle(img_mask, (center_pixel_x, center_pixel_y), 3, (100, 100, 100), -1)

        # Show images
        cv2.namedWindow('RealSenseImg')
        # cv2.namedWindow('RealSenseDepth')
        cv2.imshow('RealSenseImg', color_image_hsv)
        # cv2.imshow('RealSenseDepth', depth_colormap)
        #cv2.setMouseCallback('RealSenseImg', pick_color)
        cv2.setMouseCallback('RealSenseImg', pick_color)

        key = cv2.waitKey(1)


finally:
    pipeline.stop()
    writer.release()
    #cv2.destroyAllWindows()

#qqqq