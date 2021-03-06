'''
寻找并绘制轮廓
轮廓是连续的，边缘并不全都连续
其实边缘主要是作为图像的特征使用，比如可以用边缘特征可以区分脸和手，
而轮廓主要用来分析物体的形态，比如物体的周长和面积等，可以说边缘包括轮廓。
寻找轮廓的操作一般用于二值化图，所以通常会使用阈值分割或Canny边缘检测先得到二值图。
寻找轮廓是针对白色物体的，一定要保证物体是白色，而背景是黑色，
不然很多人在寻找轮廓时会找到图片最外面的一个框。

绘制轮廓
轮廓找出来后，为了方便观看，可以图中用红色画出来：cv2.drawContours()
opencv函数：cv2.findContours(), cv2.drawContours()
'''
# 导入相应的包
import cv2
import numpy as np

# 定义相应的常量

# 定义相应的函数

# main函数

img0 = cv2.imread('002.jpg', 1)
# 画在灰度图和二值图上显然是没有彩色的
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 30, 70)

# 寻找二值化图中的轮廓
# 参数2：轮廓的查找方式，一般使用cv2.RETR_TREE，表示提取所有的轮廓并建立轮廓间的层级。
# 参数3：轮廓的近似方法。比如对于一条直线，我们可以存储该直线的所有像素点，也可以只存储起点和终点。
# 使用cv2.CHAIN_APPROX_SIMPLE就表示用尽可能少的像素点表示轮廓。
# 函数有3个返回值，image还是原来的二值化图片，hierarchy是轮廓间的层级关系
# contours，它就是找到的轮廓了，以数组形式存储，记录了每条轮廓的所有像素点的坐标(x,y)
image, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))  # 结果应该为725

# 轮廓找出来后，为了方便观看，可以图中用红色画出来
# 其中参数2就是得到的contours，参数3表示要绘制哪一条轮廓，-1表示绘制所有轮廓，
# 参数4是颜色（B/G/R通道，所以(0,0,255)表示红色），参数5是线宽
# 首先获得要操作的轮廓，再进行轮廓绘制及分析
# cnt = contours[1]
# cv2.drawContours(img0, cnt, 0, (0, 0, 255), 2)

cv2.drawContours(img0, contours, -1, (0, 0, 255), 2)

cv2.imshow('img0', img0)

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