from tkinter import *
import cv2
from PIL import Image, ImageTk
import utils
import cv2 as cv
import mediapipe as mp
import math
import numpy as np
from tkinter.messagebox import showinfo as _show
import time
import random
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

camera = cv.VideoCapture(0)
width, height = 520, 360
camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)
app = Tk()
app.geometry("710x380")
app.configure(bg = "#6B96FF")
app.bind('<Escape>', lambda e: app.quit())
label_widget = Label(app)
label_widget.config(width=481, height=220, bg='#6B96FF')
label_widget.place(relx=0.5, rely=0.4, anchor='center')
frame_counter = 0
TOTAL_BLINKS = 0
CEF_COUNTER = 0
def open_camera():
    label4.config(text='Пожалуйста, убедитесь, что находитесь в хорошо')
    label5.config(text='освещённом помещении, это может повлиять на результат')
    global camera, label_widget, frame_counter, TOTAL_BLINKS
    # constants
    CLOSED_EYES_FRAME = 2
    FONTS = cv.FONT_HERSHEY_COMPLEX
    # [[[120 255 255]]]
    # face bounder indices
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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

    # Vertical (upper-lower eyelid) opening distance, kept per-eye (two
    # point-pairs, averaged, per eye). Turning the head makes the far eye's
    # reading degrade/shrink while the near eye stays normal — an
    # asymmetric change a real blink never produces, since both eyelids
    # close together. Keeping the eyes separate lets the caller require
    # *both* to read "closed" instead of averaging the asymmetry away.
    def eyeOpenness(raw_landmarks, right_indices, left_indices, img_width, img_height):
        def pt(i):
            lm = raw_landmarks[i]
            return lm.x * img_width, lm.y * img_height

        def dist(i, j):
            return euclaideanDistance(pt(i), pt(j))

        right_opening = (dist(right_indices[12], right_indices[4]) + dist(right_indices[13], right_indices[3])) / 2
        left_opening = (dist(left_indices[12], left_indices[4]) + dist(left_indices[13], left_indices[3])) / 2
        return right_opening, left_opening


    face_mesh_obj = map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_start_time = time.time()
    # Tracks the face's on-screen position between frames so a fast
    # reposition/lean can be told apart from an actual blink. Uses only the
    # eye *corner* landmarks (indices 0 and 8 of each eye list), which stay
    # put while blinking — the eyelid landmarks used by eyeOpenness() move
    # by design on a real blink, so including them here would make a real
    # blink look like "movement" and get skipped instead of counted.
    prev_face_center = {'pos': None}
    FACE_MOVE_THRESHOLD = 20  # pixels between frames; tune if too strict/loose

    # Dividing eyeOpenness by forehead-to-chin distance keeps the value
    # scale-invariant (moving closer/further from the camera). It does NOT
    # meaningfully cancel out nodding, though: measured on this setup,
    # nodding changes forehead-to-chin by only ~7% while it changes the
    # raw eyelid gap by 100%+ — the two don't foreshorten at the same
    # rate, so a separate gate (below) catches sustained nodding instead.
    face_height_baseline = {'value': None}
    PITCH_DEVIATION_THRESHOLD = 0.05  # relative deviation from its own baseline

    # Adaptive "eyes normally open" baseline for eyeOpenness()/face_height.
    eye_open_baseline = {'value': None}
    # Measured from real blinks on this setup: the normalized opening only
    # dips to ~0.68-0.87 of baseline (never near 0), so 0.62 never actually
    # triggered. 0.85 sits just under the observed dips with headroom above
    # the ~0.95-1.3 normal noise floor.
    BLINK_CLOSE_RATIO = 0.85   # current/baseline ratio below this = eyes closed
    BASELINE_EMA_ALPHA = 0.05  # how fast the baseline re-adapts (open frames only)

    def open_video():
        # Processes exactly one frame per call and reschedules itself via
        # label_widget.after so Tkinter's event loop stays responsive and
        # actually repaints the label (a blocking while-loop here prevents
        # the widget from ever being redrawn, even though frames are read).
        global camera, label_widget, frame_counter, TOTAL_BLINKS, CEF_COUNTER

        ret, frame1 = camera.read()  # getting frame from camera
        if not ret:
            return  # camera released/unavailable, stop rescheduling

        frame_counter += 1  # frame counter
        frame1 = cv.resize(frame1, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame1.shape[:2]

        results = face_mesh_obj.process(frame1)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame1, results, False)

            corner_points = [mesh_coords[RIGHT_EYE[0]], mesh_coords[RIGHT_EYE[8]],
                             mesh_coords[LEFT_EYE[0]], mesh_coords[LEFT_EYE[8]]]
            face_cx = sum(p[0] for p in corner_points) / len(corner_points)
            face_cy = sum(p[1] for p in corner_points) / len(corner_points)
            is_moving = False
            if prev_face_center['pos'] is not None:
                prev_x, prev_y = prev_face_center['pos']
                if math.hypot(face_cx - prev_x, face_cy - prev_y) > FACE_MOVE_THRESHOLD:
                    is_moving = True
            prev_face_center['pos'] = (face_cx, face_cy)

            face_height = euclaideanDistance(mesh_coords[10], mesh_coords[152])
            if face_height_baseline['value'] is None:
                face_height_baseline['value'] = face_height
            face_height_deviation = abs(face_height - face_height_baseline['value']) / face_height_baseline['value']
            if face_height_deviation > PITCH_DEVIATION_THRESHOLD:
                is_moving = True
            else:
                # Only adapt while roughly level, so a nod's own swing
                # doesn't drag the baseline along with it.
                face_height_baseline['value'] = (
                    (1 - BASELINE_EMA_ALPHA) * face_height_baseline['value']
                    + BASELINE_EMA_ALPHA * face_height
                )

            right_opening, left_opening = eyeOpenness(results.multi_face_landmarks[0].landmark, RIGHT_EYE, LEFT_EYE,
                                                       frame_width, frame_height)
            right_norm = right_opening / face_height if face_height > 0 else 0
            left_norm = left_opening / face_height if face_height > 0 else 0
            current_opening_norm = (right_norm + left_norm) / 2

            if not is_moving:
                if eye_open_baseline['value'] is None:
                    eye_open_baseline['value'] = current_opening_norm

                right_ratio = right_norm / eye_open_baseline['value']
                left_ratio = left_norm / eye_open_baseline['value']
                # Both eyes must independently read as closed: a real blink
                # is symmetric, a head turn only degrades the far eye.
                both_closed = right_ratio < BLINK_CLOSE_RATIO and left_ratio < BLINK_CLOSE_RATIO

                if both_closed:
                    CEF_COUNTER += 1
                else:
                    # Reset unconditionally whenever the eyes read as open, so
                    # unrelated brief spikes (noise, head movement) scattered
                    # across time can't slowly accumulate into a false blink.
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
                    # Only adapt the baseline while the eyes read clearly
                    # open, so a blink's own dip never drags it down too.
                    eye_open_baseline['value'] = (
                        (1 - BASELINE_EMA_ALPHA) * eye_open_baseline['value']
                        + BASELINE_EMA_ALPHA * current_opening_norm
                    )
            frame1 = utils.textWithBackground(frame1, f'Моргания: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2, utils.WHITE)
            cv.polylines(frame1, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True,
                         utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame1, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True,
                         utils.GREEN, 1,
                         cv.LINE_AA)

        # calculating  frame per seconds FPS
        end_time = time.time() - video_start_time
        fps = frame_counter / end_time if end_time > 0 else 0.0

        frame1 = utils.textWithBackground(frame1, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                          textThickness=2)
        frame1 = cv.resize(frame1, (400, 300))
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
        captured_image = Image.fromarray(frame1)

        photo_image = ImageTk.PhotoImage(image=captured_image)
        # Displaying photoimage in the label
        label_widget.photo_image = photo_image
        # Configure image in the label
        label_widget.configure(image=photo_image)
        # Repeat the same process after every 10 milliseconds
        label_widget.after(10, open_video)
    open_video()


# We send a signal that the other thread should stop.

def foto():
    global label_widget
    # variables
    frame_counter = 0
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    # constants
    CLOSED_EYES_FRAME = 2
    FONTS = cv.FONT_HERSHEY_COMPLEX
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


    def red_eye():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
                    dst = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                    dst1 = cv.fastNlMeansDenoisingColored(image1, None, 10, 10, 7, 21)
                    result = dst.copy()
                    result1 = dst1.copy()
                    image = cv.cvtColor(dst, cv.COLOR_BGR2HSV)
                    image1 = cv.cvtColor(dst1, cv.COLOR_BGR2HSV)
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
                    percentage = round(count * 100 / total, 2)
                    percentageleft = round(countleft * 100 / totalleft, 2)
                    if percentage >= 18 or percentageleft>=18:
                        _show('Рекомендация', 'Рекомендуем обратиться к специалисту. Нужен отдых, сделайте перерыв на 10 минут')
                        # cv.imshow('mask', mask)
                        # cv.imshow('result', result)
                        # # cv.imshow('maska', mask1)
                        # cv.imshow('res', result1)
                    cv.imwrite(fr'redright.png', result)
                    cv.imwrite(fr'redleft.png', result1)
                    label.config(text='Правый глаз - '+str(percentage))
                    label1.config(text='Левый глаз - '+str(percentageleft))
                    camera.release()
                    break
            label.after(60000, lambda:[_show('Предупреждение', 'Посмотрите в камеру'), red_eye()])
    red_eye()
    label4.pack_forget()

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
def pravila():
    _show('Инструкция', '1.Кнопка «Посмотрите в камеру»: нажмите кнопку «Посмотрите в камеру».'
    'Посмотрите в камеру в течение минуты.'
    'После окончания подсчёта моргнов появится результат.'
    'Через 5 минут появится напоминание о необходимости продолжить подсчёт морганий.'
    'Нажмите кнопку «Ok» для возобновления подсчёта.'
    '                         2.Кнопка «Определение красноты глаз»: нажмите кнопку «Определение красноты глаз».'
    'Посмотрите в камеру и дождитесь появления процента красноты на экране.'
    'Через 5 минут появится напоминание о необходимости повторить измерение красноты.'
    'Нажмите кнопку «Повторить измерение» для возобновления процесса.')
    _show('Инструкция', '3.Кнопка «Определение желтизны глаз»: нажмите кнопку «Определение желтизны глаз».'
    'Посмотрите в камеру и дождитесь появления процента желтизны на экране.'
          '        4. Кнопка «Упражнение для глаз» активирует специальный алгоритм, который предлагает пользователю различные упражнения для тренировки зрения. При каждом нажатии кнопки появляются новые упражнения, направленные на улучшение фокусировки, расслабление глазных мышц и снижение усталости. ')
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
    if percentage >= 5 or percentageleft >= 5:
        _show('Рекомендация', 'Рекомендуем обратиться к специалисту')
    return 5

canvas = Canvas(
    app,
    bg = "#6B96FF",
    height = 380,
    width = 710,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    720.0,
    380.0,
    fill="#6B96FF",
    outline="")
button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
global start_time, timer_id
timer_id = None
timeout = 60
start_time = time.time()

def check_timeout():
    global timer_id, camera

    if timer_id:
        button1.after_cancel(timer_id)  # Cancel existing timer

    if time.time() - start_time >= timeout:
        _show('Результат', f'Количество морганий: {TOTAL_BLINKS}')
        if TOTAL_BLINKS >= 30:
            _show('Рекомендация', 'Рекомендуем обратиться к специалисту. Обнаружена нагрузка на глаза, нужен отдых. Сделайте перерыв на 10 минут')
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
    label_widget.config(width=481, height=220, bg='#6B96FF')
    label_widget.place(relx=0.5, rely=0.4, anchor='center')
    frame_counter = 0
    TOTAL_BLINKS = 0
def on_button_click():
    reset_timer()
    label_widget.after(180000, lambda: [_show('Title', 'Посмотрите в камеру(моргание)'), on_button_click()])
    check_timeout()
    open_camera()
button1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command= on_button_click,
    relief="flat"
)
button1.place(
    x=7.0,
    y=332.0,
    width=146.0,
    height=41.0
)

button_image_2 = PhotoImage(
file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=upr,
    relief="flat"
)
button_2.place(
    x=540.0,
    y=331.5,
    width=181.0,
    height=41.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=foto,
    relief="flat"
)
button_3.place(
    x=153.0,
    y=332.0,
    width=197.0,
    height=41.0
)


button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=pravila,
    relief="flat"
)
button_7.place(
    x=633.0,
    y=17.0,
    width=53.0,
    height=42.0,
)

rot = Image.open(relative_to_assets("button_8.png"))
rotunda = ImageTk.PhotoImage(rot)
label_2 = Label(app, image=rotunda)
label_2.image = rotunda
label_2.place(x=256, y=59)
label_2.config(width=201, height=195, bg='#6B96FF')


button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=yellowglaz,
    relief="flat"
)
button_6.place(
    x=348.0,
    y=332.0,
    width=206.5,
    height=41.0
)
label = Label(app, fg="#000000")
label.config(bg='#6B96FF')
label.place(x=250, y=290, anchor='center')
label1 = Label(app, fg="#000000")
label1.config(bg='#6B96FF')
label1.place(x=250, y=305, anchor='center')
label2 = Label(app, fg="#000000")
label2.config(bg='#6B96FF')
label2.place(x=450, y=290, anchor='center')
label3 = Label(app, fg="#000000")
label3.config(bg='#6B96FF')
label3.place(x=450, y=305, anchor='center')
label4 = Label(app, fg="#000000")
label4.config(bg='#6B96FF')
label4.place(x=350, y=10, anchor='center')
label5 = Label(app, fg="#000000")
label5.config(bg='#6B96FF')
label5.place(x=350, y=30, anchor='center')
app.resizable(False, False)
app.mainloop()
