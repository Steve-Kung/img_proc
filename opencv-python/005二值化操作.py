'''
图像的阈值处理
阈值处理一般使得图像的像素值更单一、图像更简单。
简单阈值
选取一个全局阈值，然后就把整幅图像分成了非黑即白的二值图像了。函数为cv2.threshold()
这个函数有四个参数，第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，
第四个是一个方法选择参数，常用的有：
cv2.THRESH_BINARY（黑白二值）
cv2.THRESH_BINARY_INV（黑白二值反转）
opencv函数：腐蚀cv2.erode(), cv2.dilate(), cv2.morphologyEx()
'''
# 导入相应的包
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义相应的常量

# 定义相应的函数
# 滑动条的回调函数
def nothing(x):
    pass

# main函数
img = cv2.imread('002.jpg', 0)  # 直接读为灰度图像
# gaussian = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波
# median = cv2.medianBlur(img, 5)  # 中值滤波
cv2.imshow('blur', blur)

WindowName = 'img'  # 窗口名
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口
cv2.createTrackbar('value', WindowName, 0, 255, nothing)  # 选取阈值




while (1):
    threshold_value = cv2.getTrackbarPos('value', WindowName)  # 获取a1滑动条值
    # 这里把阈值设置成了127，对于BINARY方法，当图像中的灰度值大于127的重置像素值为255.
    # 经测试取100比较合适
    ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow(WindowName, thresh)

    k = cv2.waitKey(5)
    if k == ord('q'):
        break
cv2.destroyAllWindows()



'''
# ----------------------------------------------------------------------------------------------------------------

'''

'''
# ----------------------------------------------------------------------------------------------------------------

'''

'''
# ----------------------------------------------------------------------------------------------------------------

'''

'''
# ----------------------------------------------------------------------------------------------------------------

'''