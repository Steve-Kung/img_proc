'''
形态学操作
1.膨胀、腐蚀、开运算和闭运算等形态学操作
形态学操作一般作用于二值化图，来连接相邻的元素或分离成独立的元素。腐蚀和膨胀是针对图片中的白色部分
腐蚀：在原图的小区域内取局部最小值。因为是二值化图，只有0和255，所以小区域内有一个是0该像素点就为0，
        这样原图中边缘地方就会变成0，达到了瘦身目的
膨胀：膨胀与腐蚀相反，取的是局部最大值，效果是把图片”变胖”
开运算：先腐蚀后膨胀，其作用是：分离物体，消除小区域。这类形态学操作用cv2.morphologyEx()函数实现
闭运算：先膨胀后腐蚀，先膨胀会使白色的部分扩张，以至于消除/“闭合”物体里面的小黑洞，所以叫闭运算
形态学梯度：膨胀图减去腐蚀图，dilation - erosion，这样会得到物体的轮廓
顶帽：原图减去开运算后的图：src - opening
黑帽：闭运算后的图减去原图：closing - src

opencv函数：腐蚀cv2.erode(), cv2.dilate(), cv2.morphologyEx()
'''
# 导入相应的包
import cv2
import numpy as np

# 定义相应的常量

# 定义相应的函数

# main函数
# 读入图像并进行二值处理
img = cv2.imread('002.jpg', 0)
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
# 腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
# erosion = cv2.erode(thresh, kernel)  # 腐蚀
# cv2.imshow('erosion', erosion)
# dilation = cv2.dilate(thresh, kernel)  # 膨胀
# cv2.imshow('dilation', dilation)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算
# cv2.imshow('opening', opening)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭运算
# cv2.imshow('closing', closing)
# gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)  # 形态学梯度
# cv2.imshow('gradient', gradient)
# tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)  # 顶帽
# cv2.imshow('tophat', tophat)
blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)  # 黑帽
cv2.imshow('blackhat', blackhat)

cv2.waitKey(0)  # 不断刷新图像,返回值为当前键盘按键值，如果不按键一直等待,0表示无限等待
# 销毁所有窗口
cv2.destroyAllWindows()



'''
# ----------------------------------------------------------------------------------------------------------------
# gaussian = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
# blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波
# median = cv2.medianBlur(img, 5)  # 中值滤波
'''

'''
# ----------------------------------------------------------------------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构

'''

'''
# ----------------------------------------------------------------------------------------------------------------

'''

'''
# ----------------------------------------------------------------------------------------------------------------

'''