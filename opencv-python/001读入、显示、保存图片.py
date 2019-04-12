import numpy as np
import cv2

# 0显示灰度图 1显示彩色图
img = cv2.imread('001.jpg', 1)
print(img.shape)
# 第一个参数是窗口的名字，第二个参数是传入图像。可以创建多个窗口
cv2.imshow('image', img)
# 原图太大，所以缩放原图，fx/fy为缩放系数，interpolation为插入方式
img1 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

cv2.imshow('image1', img1)
# 不调用waitKey的话，窗口会一闪而逝，看不到显示的图片。
print(img1.shape)

cv2.waitKey(0)  # 不断刷新图像,返回值为当前键盘按键值，如果不按键一直等待,0表示无限等待
# 销毁所有窗口
cv2.destroyAllWindows()

# 第一个参数是要保存的文件名，第二个参数是要保存的图像。
cv2.imwrite('002.jpg',img1)