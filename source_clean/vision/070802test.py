import pyrealsense2 as rs
import numpy as np
import cv2
import time
import datetime as dt
import keyboard
import csv

#start_time = dt.datetime.today().time()
#i = 0

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

lastZ = 0
ballDetected = False
calibrationHeight = 190.0

outputArray = []
linePos = 0

print('vision_system_start')
print('press_q_to_write_data')

try:
    while True:
        
        #print('start iteration')
        # print timestamps each iteration
        #qprint(dt.datetime.today().time())

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_orange = (20-5, 180, 180)
        upper_orange = (20+5, 255, 255)

        img_mask = cv2.inRange(color_image_hsv, lower_orange, upper_orange)

        M = cv2.moments(img_mask)

        # calculate x,y coordinate of center
        cX = 1
        cY = 1

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            ballDetected = True
        else:
            ballDetected = False

        #cv2.circle(img_mask, (cX, cY), 3, (100, 100, 100), -1)
        #cv2.circle(depth_colormap, (cX, cY), 3, (100, 100, 100), -1)

        worldX = ((cY-242.0)*0.3049)
        worldY = (cX*0.3049)-97.5
        worldZ = round(191.0 - (depth_image[cY][cX]/10.0), 1)
        #print(worldZ)
        if worldZ >= 180:
            worldZ = lastZ
        else:
            lastZ = worldZ

        worldX_calibrated = round((calibrationHeight - worldZ) * (worldX / calibrationHeight), 1)
        worldY_calibrated = round((calibrationHeight - worldZ) * (worldY / calibrationHeight), 1)

        # world coordinates calc: ~0.0003s

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_mask, str(worldX_calibrated)+' , '+str(worldY_calibrated)+' , ' + str(worldZ), (cX, cY), font, 1, (100, 100, 100))
        # putText method: under 0.0001s

        if ballDetected:
            row = [worldX_calibrated, worldY_calibrated, worldZ, dt.datetime.today().time().microsecond]
            outputArray.insert(linePos, row)
            linePos += 1
        #print('start')
        #print(outputArray)
        '''
        if keyboard.is_pressed('q'):
            print('q_is_pressed')
            with open('pong_testdata.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                for i in range(len(outputArray)):
                    writer.writerow(outputArray[i])
                writer.writerow('___')
            outputArray = []
            linePos = 0
            time.sleep(1)
            print('completed')
        '''

        '''
        with open('pong_testdata.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            if ballDetected:
                writer.writerow(row)
        '''

        # Show images
        cv2.namedWindow('RealSenseImg')
        #cv2.namedWindow('RealSenseDepth')
        cv2.imshow('RealSenseImg', img_mask)
        #cv2.imshow('RealSenseDepth', depth_colormap)
        cv2.waitKey(1)
        #GUI: 0.0025s~0.005s

    print('end')

finally:
    pipeline.stop()
    print(outputArray)
    #csvFile.close()qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
