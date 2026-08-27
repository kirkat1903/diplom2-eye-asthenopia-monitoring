from tkinter import *
import cv2
from build import utils
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import math
import numpy as np
from фото import foto
from tkinter.messagebox import showinfo as _show
from custombutton import RoundedButton
import time
import random
from pathlib import Path
from tkinter import Tk

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\rukos\PycharmProjects\диплом2\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
camera = cv.VideoCapture(0)
width, height = 520, 360
camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
app = Tk()
app.bind('<Escape>', lambda e: app.quit())
app.geometry("710x380")
app.configure(bg="#6B96FF")
label_widget = Label(app)
label_widget.config(width=500, height=300, bg='#FFFFFF')
label_widget.place(relx=0.5, rely=0.4, anchor='center')
frame_counter = 0
TOTAL_BLINKS = 0
CEF_COUNTER = 0
def open_camera():
    global camera, label_widget, frame_counter, TOTAL_BLINKS
    # constants
    CLOSED_EYES_FRAME = 2
    FONTS = cv.FONT_HERSHEY_COMPLEX
    # [[[120 255 255]]]
    # face bounder indices
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # lips indices for Landmarks

    # Left eyes indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    # right eyes indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    map_face_mesh = mp.solutions.face_mesh

    # camera object

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

    def open_video():
        global camera, label_widget, frame_counter, TOTAL_BLINKS

        with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            # starting time here
            start_time = time.time()
            # starting Video loop here.
            while True:
                global frame_counter
                frame_counter += 1  # frame counter
                ret, frame1 = camera.read()  # getting frame from camera
                if not ret:
                    break  # no more frames break
                #  resizing frame

                frame1 = cv.resize(frame1, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)

                frame_height, frame_width = frame1.shape[:2]
                global CEF_COUNTER
                results = face_mesh.process(frame1)
                if results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(frame1, results, False)
                    reRatio = blinkRatio(frame1, mesh_coords, RIGHT_EYE, LEFT_EYE)
                    # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                    # utils.textWithBackground(frame, f'Ratio : {round(reRatio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK, utils.YELLOW)
                    print(reRatio)
                    if reRatio > 3.45:
                        CEF_COUNTER += 1
                    # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                    # utils.colorBackgroundText(frame, f'Моргание', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                    #                           pad_x=6, pad_y=6, )

                    else:
                        if CEF_COUNTER > CLOSED_EYES_FRAME:
                            CEF_COUNTER = 0
                            global TOTAL_BLINKS
                            TOTAL_BLINKS += 1
                    # cv.putText(frame1, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                    frame1 = utils.textWithBackground(frame1, f'Моргания: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)
                    # if TOTAL_BLINKS < 13:
                    #     _show('Title', 'Обнаружена нагрузка на глаза, нужен отдых')
                    cv.polylines(frame1, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True,
                                 utils.GREEN, 1,
                                 cv.LINE_AA)
                    cv.polylines(frame1, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True,
                                 utils.GREEN, 1,
                                 cv.LINE_AA)

                # eye_position, color = positionEstimator(crop_right)
                # utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                # eye_position_left, color = positionEstimator(crop_left)
                # utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8,
                #                           8)

                # calculating  frame per seconds FPS
                end_time = time.time() - start_time
                fps = frame_counter / end_time

                frame1 = utils.textWithBackground(frame1, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                                  textThickness=2)
                # writing image for thumbnail drawing shape

                frame1 = cv.resize(frame1, (400, 300))
                frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
                captured_image = Image.fromarray(frame1)

                photo_image = ImageTk.PhotoImage(image=captured_image)
                global label_widget
                # Displaying photoimage in the label
                label_widget.photo_image = photo_image

                # Configure image in the label
                label_widget.configure(image=photo_image)
                # Repeat the same process after every 10 seconds
                label_widget.after(10, open_video)
                cv.imshow('', photo_image)
                # # Convert image from one color space to other
            cv.destroyAllWindows()
    open_video()
    pass


# We send a signal that the other thread should stop.

def foto():
    global camera, label_widget
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
    def red_eye():
        camera = cv.VideoCapture(0)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while True:
                global frame_counter
                frame_counter += 1  # frame counter
                ret, frame = camera.read()  # getting frame from camera
                if not ret:
                    break  # no more frames break
                frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(frame, results, False)
                    reRatio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                        # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                    utils.colorBackgroundText(frame, f'Ratio : {round(reRatio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                              utils.YELLOW)

                    if reRatio > 3.5:
                        global CEF_COUNTER
                        CEF_COUNTER += 1
                            # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                            # utils.colorBackgroundText(frame, f'Моргание', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                            #                           pad_x=6, pad_y=6, )

                    else:
                        if CEF_COUNTER > CLOSED_EYES_FRAME:
                            CEF_COUNTER = 0
                            global TOTAL_BLINKS
                            TOTAL_BLINKS += 1
                        # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                    utils.colorBackgroundText(frame, f'Моргания: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)
                    cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN,
                                 1,
                                 cv.LINE_AA)
                    cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN,
                                 1,
                                 cv.LINE_AA)

                        # Blink Detector Counter Completed
                    right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                    left_coords = [mesh_coords[p] for p in LEFT_EYE]
                    crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                    crop_right = cv.resize(crop_right, (640, 400))
                    crop_left = cv.resize(crop_left, (640, 400))

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
                    lower = np.array([155,25,0])
                    upper = np.array([179,255,255])#светло-розовый
                    mask = cv.inRange(image, lower, upper)
                    mask1 = cv.inRange(image1, lower, upper)
                    result = cv.bitwise_and(result, result, mask=mask)
                    result1 = cv.bitwise_and(result1, result1, mask=mask1)
                    mask = cv.resize(mask, (640, 400))
                    result = cv.resize(result, (640, 400))
                    mask1 = cv.resize(mask1, (640, 400))
                    result1 = cv.resize(result1, (640, 400))
                    total = result.size
                    totalleft = result1.size
                    count = np.count_nonzero(result)
                    countleft = np.count_nonzero(result1)
                    percentage = round(count * 100 / total, 2)*10
                    percentageleft = round(countleft * 100 / totalleft, 2)*10
                        # cv.imshow('mask', mask)
                        # cv.imshow('result', result)
                        # # cv.imshow('maska', mask1)
                        # cv.imshow('res', result1)
                    cv.imwrite(fr'redright.png', result)
                    cv.imwrite(fr'redleft.png', result1)
                    if percentage>=5:
                        label.config(text='Правый глаз - '+str(percentage))
                    else:
                        red_eye()
                    if percentageleft >= 5:
                        label1.config(text='Левый глаз - '+str(percentageleft))
                    else:
                        red_eye()
                    camera.release()
            label.after(60000, lambda:[_show('Title', 'Посмотрите в камеру'), red_eye()])
    red_eye()

def upr():
    uprlist = ['№1 — Совиный глаз Для тренировки глазных мышц, улучшения обмена веществ и трофики. Сильно зажмурьтесь на 5 секунд, а затем на 5 секунд широко распахните веки. Повторите 10-15 раз. ', '№2 — Шторки. Для усиления кровообращения. Быстро моргайте в течение 1,5-2 минут.',
                   '№3 — Пальминг. Для расслабления. Опустите веки, сверху положите разогретые ладони, скрестив их на лбу. В глаза не должен проникать свет. Расслабьтесь, успокойтесь, внимательно вглядитесь в темноту. Дышите ровно, проведите в таком положении 5 минут, стараясь ни о чем не думать.',
                   '№4 — Окно. Для бодрости, укрепления цилиарной мышцы. Нарисуйте на оконном стекле точку. Поочередно фокусируйте взгляд — сначала на максимально удаленном объекте, затем на точке. Кратность — 10 раз.',
                   '№5 — Массаж. Для ускорения лимфотока, кровотока, снятия отечности тканей. На 2 секунды прижмите к сомкнутым векам указательный, средний, безымянный пальцы. Отпустите. Продублируйте упражнение 10 раз.',
                   '№6 — Контуринг. Для снятия гипертонуса глазодвигательных мышц. Подберите несколько объектов разной величины из окружающих предметов. Например, дверь, органайзер, стул и т. д. Скользите взглядом, как бы обводя объекты. Повторите 4-8 раз.',
                   '№7 — Маятник. Для восстановления фокуса при дальнозоркости. Возьмите карандаш, установите его стержень вровень с носом. Покачивайте карандашом в разные стороны, не отводя глаз от стержня. Продолжительность — 30-60 секунд.',
                   '№8 — Восьмерки. Для восстановления фокуса при близорукости. С открытыми глазами рисуйте взглядом цифру 8 в течение 30 секунд. Сомкните веки. Повторите упражнение.',
                   '№9 — Фокусировка. Для тренировки аккомодации. Вытяните вперед руку, выставите указательный палец, сосредоточьте на нем внимание. Медленно подводите палец к носу, не отрывая взгляда. Когда изображение начнет двоиться, опустите руку. Продублируйте 5-7 раз.',
                   '№10 — Круги. Для здоровья сосудов и мышц. Не двигая головой, совершите несколько круговых движений зрачками: сначала по часовой стрелке, затем в обратном направлении. Повторите 5-10 раз в каждую сторону.']
    uprag = random.choice(uprlist)
    _show('Упражнение', uprag)
def yellowglaz():
    image = cv2.imread('right.png')
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([16, 0, 99])
    upper = np.array([39, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    mask = cv2.resize(mask, (640, 360))
    result = cv2.resize(result, (640, 360))
    # cv2.imshow('maskyell', mask)
    # cv2.imshow('resultyell', result)
    image1 = cv2.imread('left.png')
    result1 = image1.copy()
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    lower = np.array([16, 0, 99])
    upper = np.array([39, 255, 255])
    mask1 = cv2.inRange(image1, lower, upper)
    result1 = cv2.bitwise_and(result1, result1, mask=mask1)
    mask1 = cv2.resize(mask, (640, 360))
    result1 = cv2.resize(result1, (640, 360))
    total = result.size
    totalleft = result1.size
    count = np.count_nonzero(result)
    countleft = np.count_nonzero(result1)
    percentage = round(count * 100 / total, 2)
    percentageleft = round(countleft * 100 / totalleft, 2)
    # cv2.imshow('maskyell1', mask1)
    # cv2.imshow('resultyell1', result1)
    cv2.imwrite(fr'yelright.png', result)
    cv2.imwrite(fr'yelleft.png', result1)
    label2.config(text='Правый глаз - '+str(percentage))
    label3.config(text='Левый глаз - '+str(percentageleft))
    return 5


label = Label(app, fg="#6477AB")
label.config(bg='#2F3046')
label.place(relx=0.5, rely=0.76, anchor='center')
label1 = Label(app, fg="#6477AB")
label1.config(bg='#2F3046')
label1.place(relx=0.5, rely=0.8, anchor='center')
label2 = Label(app, fg="#6477AB")
label2.config(bg='#2F3046')
label2.place(relx=0.77, rely=0.76, anchor='center')
label3 = Label(app, fg="#6477AB")
label3.config(bg='#2F3046')
label3.place(relx=0.77, rely=0.8, anchor='center')



def run_func_with_timeout():
    global start_time, timer_id, camera, label_widget, frame_counter, TOTAL_BLINKS
    timer_id = None
    timeout = 60
    start_time = time.time()

    def check_timeout():
        global timer_id, camera

        if timer_id:
            button1.after_cancel(timer_id)  # Cancel existing timer

        if time.time() - start_time >= timeout:
            _show('Title', f'Количество морганий: {TOTAL_BLINKS}')
            label_widget.place_forget()
            camera.release()
        else:
            timer_id = app.after(100, check_timeout)

    def reset_timer():
        global start_time, camera, label_widget, frame_counter, TOTAL_BLINKS
        start_time = time.time()

        camera = cv.VideoCapture(0)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        label_widget = Label(app)
        label_widget.config(width=400, height=300, bg='#FFFFFF')
        label_widget.place(relx=0.5, rely=0.4, anchor='center')
        frame_counter = 0
        TOTAL_BLINKS = 0

    def on_button_click():
        reset_timer()
        label_widget.after(180000, lambda: [_show('Title', 'Посмотрите в камеру(моргание)'), on_button_click()])
        check_timeout()
        open_camera()

    button1 = RoundedButton(app, text="Подсчет морганий", border_radius=4, padding=17,
                            command=on_button_click, color="#6477AB")
    button1.place(relx=0.025, rely=0.9, anchor='w')

    button2 = RoundedButton(app, text="Определить красноту", border_radius=4, padding=17,
                            command=foto, color="#6477AB")
    button2.place(relx=0.37, rely=0.9, anchor='center')

    button3 = RoundedButton(app, text="Определить желтизну", border_radius=4, padding=17,
                            command=yellowglaz, color="#6477AB")
    button3.place(relx=0.75, rely=0.9, anchor='e')

    button4 = RoundedButton(app, text="Упражнения для глаз", border_radius=4, padding=17,
                            command=upr, color="#6477AB")
    button4.place(relx=1.0, rely=0.9, anchor='e')
run_func_with_timeout()
app.mainloop()
