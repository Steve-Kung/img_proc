# -*- coding: utf-8 -*-
'''
1、创建滑动条
cv2.createTrackbar(trackbarName, windowName, value, count, onChange)
    trackbarName：滑动条名称
    windowName：所在窗口名
    value：初始值
    count：最大值
    onChange：回调函数名称
2、获取滑动条数据
cv2.getTrackbarPos(trackbarname, winname)
    trackbarname：滑动条名称
    winname：所在窗口名
'''

import cv2
import numpy as np

# 滑动条的回调函数
def nothing(x):
    pass

# 创建滑动条
img = cv2.imread('002.jpg')

WindowName = 'img'  # 窗口名
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口
cv2.createTrackbar('value', WindowName, 0, 255, nothing)  # 选取阈值

while (1):
    threshold_value = cv2.getTrackbarPos('value', WindowName)  # 获取a1滑动条值

    cv2.imshow(WindowName, img)

    k = cv2.waitKey(5)
    if k == ord('q'):
        break
cv2.destroyAllWindows()