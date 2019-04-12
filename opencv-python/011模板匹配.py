'''
使用模板匹配在图像中寻找物体
模板匹配就是用来在大图中找小图，也就是说在一副图像中寻找另外一张模板图像的位置：
不断地在原图中移动模板图像去比较
用cv2.matchTemplate()实现模板匹配
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
'''
# 匹配函数返回的是一副灰度图，最白的地方表示最大的匹配。
# 使用cv2.minMaxLoc()函数可以得到最大匹配值的坐标，
# 以这个点为左上角角点，模板的宽和高画矩形就是匹配的位置了
# 相关系数匹配方法：cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top = max_loc  # 左上角
right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
# cv2.imshow('img', img)
'''
# 匹配多个物体
# 前面我们是找最大匹配的点，所以只能匹配一次。
# 我们可以设定一个匹配阈值来匹配多次
# 标准相关模板匹配
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7

# Numpy的知识
# 因为loc是先y坐标再x坐标，所以用loc[::-1]翻转一下，然后再用zip函数拼接在一起。
loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
for pt in zip(*loc[::-1]):  # *号表示可选参数
    right_bottom = (pt[0] + w, pt[1] + h)
    # cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)
    cv2.rectangle(img, pt, right_bottom, 255, 2)
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