'''
霍夫变换
使用霍夫线变换和圆变换检测图像中的直线和圆
opencv函数：cv2.HoughLines(), cv2.HoughLinesP(), cv2.HoughCircles()
'''
# 导入相应的包
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义相应的常量

# 定义相应的函数

# main函数

img = cv2.imread('002.jpg', 0)
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 30, 70)

'''
# 霍夫直线变换
# OpenCV中用cv2.HoughLines()在二值图上实现霍夫变换，函数返回的是一组直线的(r,θ)数据：
# 参数1：要检测的二值图（一般是阈值分割或边缘检测后的图）
# 参数2：距离r的精度，值越大，考虑越多的线
# 参数3：角度θ的精度，值越小，考虑越多的线
# 参数4：累加数阈值，值越小，考虑越多的线
lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)

# 将检测的线画出来（注意是极坐标）
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255))
cv2.imshow('edges', edges)
'''
'''
# 统计概率霍夫直线变换
# 前面的方法又称为标准霍夫变换，它会计算图像中的每一个点，计算量比较大，
# 另外它得到的是整一条线（r和θ），并不知道原图中直线的端点。
# 所以提出了统计概率霍夫直线变换(Probabilistic Hough Transform)，是一种改进的霍夫变换
# 前面几个参数跟之前的一样，有两个可选参数：
# minLineLength：最短长度阈值，比这个长度短的线会被排除
# maxLineGap：同一直线两点之间的最大距离
lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 90,
                        minLineLength=50, maxLineGap=10)
# 将检测的线画出来
# cv2.LINE_AA表示抗锯齿线型
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(edges, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
cv2.imshow('edges', edges)
'''
# 霍夫圆变换
# 圆是用(x_center,y_center,r)来表示，从二维变成了三维，数据量变大了很多；
# 所以一般使用霍夫梯度法减少计算量
# 参数2：变换方法，一般使用霍夫梯度法
# 参数3 dp=1：表示霍夫梯度法中累加器图像的分辨率与原图一致
# 参数4：两个不同圆圆心的最短距离
# 参数5：param2跟霍夫直线变换中的累加数阈值一样
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=30, minRadius=25, maxRadius=65)
circles = np.int0(np.around(circles))
# 将检测的圆画出来
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
for i in circles[0, :]:
    cv2.circle(edges, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
    cv2.circle(edges, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心
cv2.imshow('edges', edges)

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