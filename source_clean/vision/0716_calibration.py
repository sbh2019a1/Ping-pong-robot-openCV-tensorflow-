import pyrealsense2 as rs
import numpy as np
import cv2
import time
import datetime as dt

# Create realsense pipeline object
pipeline = rs.pipeline()

# Create config object and set resolutions
# supported resolutions
# depth: 1280*720(30Hz), 848*480(90Hz), 640*480(90Hz), 640*360(90Hz)
# color: 1280*720(30Hz), 960*540(60Hz), 848*480(60Hz), 640*360(60Hz)
# using higher FPS or unsupported resolutions will result in a crash
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

align = rs.align(rs.stream.color)

center_pixel_x = 1
center_pixel_y = 1

meter_pixel_x = 1
meter_pixel_y = 1

gui_enabled = True

try:
    while True:

        key = cv2.waitKey(1)

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

        # RED detection start
        lower_red = (-5, 150, 150)
        upper_red = (5, 255, 255)

        img_mask_red = cv2.inRange(color_image_hsv, lower_red, upper_red)

        Mr = cv2.moments(img_mask_red)

        # image_center_X and image_center_Y are pixel based coordinates
        red_image_center_X = 1
        red_image_center_Y = 1

        if Mr["m00"] != 0:
            red_image_center_X = int(Mr["m10"] / Mr["m00"])
            red_image_center_Y = int(Mr["m01"] / Mr["m00"])
            center_pixel_x = red_image_center_X
            center_pixel_y = red_image_center_Y
            ballDetected = True
        else:
            ballDetected = False

        # sensor_z = depth_image[red_image_center_Y][red_image_center_X]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_mask_red, str(red_image_center_X) + ' , ' + str(red_image_center_Y), (red_image_center_X, red_image_center_Y), font, 1, (100, 100, 100))
        # RED detection end

        # GREEN detection start
        lower_green = (20, 180, 180)
        upper_green = (40, 255, 255)

        img_mask_green = cv2.inRange(color_image_hsv, lower_green, upper_green)

        Mg = cv2.moments(img_mask_green)

        # image_center_X and image_center_Y are pixel based coordinates
        green_image_center_X = 1
        green_image_center_Y = 1

        if Mg["m00"] != 0:
            green_image_center_X = int(Mg["m10"] / Mg["m00"])
            green_image_center_Y = int(Mg["m01"] / Mg["m00"])
            meter_pixel_x = green_image_center_X
            meter_pixel_y = green_image_center_Y
            ballDetected = True
        else:
            ballDetected = False

        # sensor_z = depth_image[green_image_center_Y][green_image_center_X]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_mask_green, str(green_image_center_X) + ' , ' + str(green_image_center_Y),
                    (green_image_center_X, green_image_center_Y), font, 1, (100, 100, 100))
        # RED detection end


        # Show images
        cv2.namedWindow('RealSenseImg')
        # cv2.namedWindow('RealSenseDepth')
        cv2.imshow('RealSenseImg', img_mask_green)
        #cv2.imshow('RealSenseDepth', depth_colormap)

        # Press esc or 'q' to close the image window
        if key == 27:
            cv2.destroyAllWindows()
            print('center X = ' + str(center_pixel_x))
            print('center Y = ' + str(center_pixel_y))
            print('meter(yellow) X = ' + str(green_image_center_X))
            print('meter(yellow) Y = ' + str(green_image_center_Y))
            print('========================================')
            print('100cm = ' + str(center_pixel_x-meter_pixel_x) + ' pixels')
            print('centerCoordinates = ' + str(center_pixel_x) + ' , ' + str((center_pixel_y+green_image_center_Y)/2))
            break
finally:
    pipeline.stop()