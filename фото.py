import cv2 as cv
import mediapipe as mp
import time
import math
from build import utils
import numpy as np
def foto():
    # variables
    frame_counter = 0
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    # constants
    CLOSED_EYES_FRAME = 2
    FONTS = cv.FONT_HERSHEY_COMPLEX
    # [[[120 255 255]]]
    # face bounder indices
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # lips indices for Landmarks

    # Left eyes indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


    # right eyes indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


    map_face_mesh = mp.solutions.face_mesh
    # camera object
    camera = cv.VideoCapture(0)


    # landmark detection function
    def landmarksDetection(img, results, draw=False):
        img_height, img_width = img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.multi_face_landmarks[0].landmark]
        if draw:
            [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

        # returning the list of tuples for each landmarks
        return mesh_coord


    # Euclaidean distance
    def euclaideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance


    # Blinking Ratio
    def blinkRatio(img, landmarks, right_indices, left_indices):
        # Right eyes
        # horizontal line
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

        # LEFT_EYE
        # horizontal line
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        # vertical line
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rhDistance = euclaideanDistance(rh_right, rh_left)
        rvDistance = euclaideanDistance(rv_top, rv_bottom)

        lvDistance = euclaideanDistance(lv_top, lv_bottom)
        lhDistance = euclaideanDistance(lh_right, lh_left)

        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance


        return reRatio


    # Eyes Extrctor function,
    def eyesExtractor(img, right_eye_coords, left_eye_coords):
        # converting color image to  scale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # getting the dimension of image
        dim = gray.shape

        # creating mask from gray scale dim
        mask = np.zeros(dim, dtype=np.uint8)

        # drawing Eyes Shape on mask with white color
        cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

        # showing the mask
        # cv.imshow('mask', mask)

        # draw eyes image on mask, where white shape is
        eyes = cv.bitwise_and(gray, gray, mask=mask)
        # change black color to gray other than eys
        # cv.imshow('eyes draw', eyes)
        eyes[mask == 0] = 155

        # getting minium and maximum x and y  for right and left eyes
        # For Right Eye
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

        # For LEFT Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

        # croping the eyes from mask
        cropped_right = img[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = img[l_min_y: l_max_y, l_min_x: l_max_x]
        # returning the cropped eyes
        return cropped_right, cropped_left


    # Eyes Postion Estimator
    def positionEstimator(cropped_eye):
        # getting height and width of eye
        h, w = cropped_eye.shape

        # remove the noise from images
        gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
        median_blur = cv.medianBlur(gaussain_blur, 3)

        # applying thrsholding to convert binary_image
        ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

        # create fixd part for eye with
        piece = int(w / 3)

        # slicing the eyes into three parts
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece + piece]
        left_piece = threshed_eye[0:h, piece + piece:w]

        # calling pixel counter function
        eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

        return eye_position, color


    # creating pixel counter function
    def pixelCounter(first_piece, second_piece, third_piece):
        # counting black pixel in each part
        right_part = np.sum(first_piece == 0)
        center_part = np.sum(second_piece == 0)
        left_part = np.sum(third_piece == 0)
        # creating list of these values
        eye_parts = [right_part, center_part, left_part]

        # getting the index of max values in the list
        max_index = eye_parts.index(max(eye_parts))
        pos_eye = ''
        if max_index == 0:
            pos_eye = "RIGHT"
            color = [utils.BLACK, utils.GREEN]
        elif max_index == 1:
            pos_eye = 'CENTER'
            color = [utils.YELLOW, utils.PINK]
        elif max_index == 2:
            pos_eye = 'LEFT'
            color = [utils.GRAY, utils.YELLOW]
        else:
            pos_eye = "Closed"
            color = [utils.GRAY, utils.YELLOW]
        return pos_eye, color


    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # starting time here
        start_time = time.time()
        # starting Video loop here.
        while True:
            frame_counter += 1  # frame counter
            ret, frame = camera.read()  # getting frame from camera
            if not ret:
                break  # no more frames break
            #  resizing frame

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                reRatio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                utils.colorBackgroundText(frame, f'Ratio : {round(reRatio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                          utils.YELLOW)

                if reRatio > 3.5:
                    CEF_COUNTER += 1
                    # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                    # utils.colorBackgroundText(frame, f'Моргание', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                    #                           pad_x=6, pad_y=6, )

                else:
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        CEF_COUNTER = 0
                        TOTAL_BLINKS += 1
                #cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                utils.colorBackgroundText(frame, f'Моргания: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                             cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                             cv.LINE_AA)

                # Blink Detector Counter Completed
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                crop_right = cv.resize(crop_right, (340, 200))
                crop_left = cv.resize(crop_left, (340, 200))

                # cv.imshow('right.png', crop_right)
                # cv.imshow('left.png', crop_left)
                cv.imwrite(fr'right.png', crop_right)
                cv.imwrite(fr'left.png', crop_left)
                image = cv.imread('right.png')
                image1 = cv.imread('left.png')
                result = image.copy()
                result1 = image1.copy()
                image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                image1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
                lower = np.array([155, 25, 0])
                upper = np.array([179, 255, 255])
                mask = cv.inRange(image, lower, upper)
                mask1 = cv.inRange(image1, lower, upper)
                count = np.count_nonzero(mask)
                countleft = np.count_nonzero(mask1)
                print('Краснота правого глаза:', count)
                print('Краснота левого глаза:', countleft)
                result = cv.bitwise_and(result, result, mask=mask)
                result1 = cv.bitwise_and(result1, result1, mask=mask1)
                mask = cv.resize(mask, (340, 360))
                result = cv.resize(result, (340, 200))
                mask1 = cv.resize(mask1, (340, 360))
                result1 = cv.resize(result1, (340, 200))
                # cv.imshow('mask', mask)
                # cv.imshow('result', result)
                # # cv.imshow('maska', mask1)
                # cv.imshow('res', result1)
                cv.imwrite(fr'redright.png', result)
                cv.imwrite(fr'redleft.png', result1)
                return

        cv.destroyAllWindows()
        camera.release()