'''
计算轮廓特征，如面积、周长、最小外接矩形等

opencv函数：cv2.contourArea(), cv2.arcLength(), cv2.approxPolyDP()
'''
# 导入相应的包
import cv2
import numpy as np

# 定义相应的常量

# 定义相应的函数

# main函数

img = cv2.imread('002.jpg', 0)
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 30, 70)

image, contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img, contours[500], -1, (0, 0, 255), 2)


# 轮廓面积
area = cv2.contourArea(contours[500])
print(area)

# 轮廓周长
# 参数2表示轮廓是否封闭，显然我们的轮廓是封闭的，所以是True。
perimeter = cv2.arcLength(contours[500], True)
print(perimeter)

# 图像矩
# 矩可以理解为图像的各类几何特征
# M中包含了很多轮廓的特征信息，比如M[‘m00’]表示轮廓面积，与cv2.contourArea()计算一样。质心也可以用它来算
M = cv2.moments(contours[500])
# 质心
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
print(cx, cy)

# 外接矩形
# 形状的外接矩形有两种，外接矩形，表示不考虑旋转并且能包含整个轮廓的矩形。最小外接矩，考虑了旋转
x, y, w, h = cv2.boundingRect(contours[500])  # 外接矩形
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
rect = cv2.minAreaRect(contours[500])  # 最小外接矩形
# np.int0(x)是把x取整的操作，比如377.93就会变成377，也可以用x.astype(np.int
box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

# 最小外接圆
# 外接圆跟外接矩形一样，找到一个能包围物体的最小圆
(x, y), radius = cv2.minEnclosingCircle(contours[500])
(x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
cv2.circle(img, (x, y), radius, (0, 0, 255), 2)

# 拟合椭圆
ellipse = cv2.fitEllipse(contours[500])
cv2.ellipse(img, ellipse, (255, 255, 0), 2)

# 形状匹配
# cv2.matchShapes()可以检测两个形状之间的相似度，返回值越小，越相似
# 图形的旋转或缩放并没有影响
# 参数3是匹配方法
# 用形状匹配比较两个字母或数字（这相当于很简单的一个OCR）
print(cv2.matchShapes(contours[500], contours[500], 1, 0.0))  # 0.0
print(cv2.matchShapes(contours[500], contours[1000], 1, 0.0))  # 0.277

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