import cv2
import numpy as np
import win10toast
import matplotlib.pyplot as plt
import os
import shutil
def redglaz():
    image = cv2.imread('right.png')
    image1 = cv2.imread('left.png')
    result = image.copy()
    result1 = image1.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    lower = np.array([17, 15, 100])
    upper = np.array([50, 56, 200])
    mask = cv2.inRange(image, lower, upper)
    mask1 = cv2.inRange(image1, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    result1 = cv2.bitwise_and(result1, result1, mask=mask1)
    mask = cv2.resize(mask,(640, 360))
    result = cv2.resize(result,(640, 360))
    mask1 = cv2.resize(mask1, (640, 360))
    result1 = cv2.resize(result1, (640, 360))
    cv2.imshow('mask', mask)
    cv2.imshow('resultred', result)
    cv2.imshow('mask1', mask1)
    cv2.imshow('resultred1', result1)
    cv2.waitKey()
def yellowglaz(label2,label3):
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
    cv2.imshow('resultyell', result)
    image1 = cv2.imread('left.png')
    result1 = image1.copy()
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    lower = np.array([16, 0, 99])
    upper = np.array([39, 255, 255])
    mask1 = cv2.inRange(image1, lower, upper)
    result1 = cv2.bitwise_and(result1, result1, mask=mask1)
    mask1 = cv2.resize(mask, (640, 360))
    result1 = cv2.resize(result1, (640, 360))
    count = np.count_nonzero(result) / 1000
    countleft = np.count_nonzero(result1) / 1000
    # cv2.imshow('maskyell1', mask1)
    # cv2.imshow('resultyell1', result1)
    cv2.imwrite(fr'yelright.png', result)
    cv2.imwrite(fr'yelleft.png', result1)
    label2.config(text=str(count))
    label3.config(text=str(countleft))
    return 5
    # cv2.waitKey()









