'''
霍夫变换
使用霍夫线变换和圆变换检测图像中的直线和圆
opencv函数：cv2.matchTemplate(), cv2.minMaxLoc()
'''
# 导入相应的包
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义相应的常量

# 定义相应的函数

# main函数
# 读入图片和模板：
img = cv2.imread('002.jpg', 0)
template = cv2.imread('template.PNG', 0)
h, w = template.shape[:2]  # rows->h, cols->w
cv2.imshow('template', template)
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 30, 70)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

cv2.imshow('img', img)

cv2.waitKey(0)
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