import cv2
import numpy as np  # 添加模块和矩阵模块

cap = cv2.VideoCapture(0)  # 打开摄像头


def detection(cnt):
    n = 0
    for i in range(len(cnt)):
        area = cv2.contourArea(cnt[i])
        (x, y), radius = cv2.minEnclosingCircle(cnt[i])
        s = np.pi * (radius ** 2)
        if (area / s) > 0.85:
            n += 1
    return n


# 设定提取红、绿色范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([35, 100, 100])
upper_green = np.array([77, 255, 255])

while (1):
    # 1、读取帧，高斯滤波，bgr2hsv,imrange提取色块
    ret, frame = cap.read()
    # frame = cv2.imread('green_light.jpg')
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # 转化为hsv通道

    mask = cv2.inRange(hsv, lower_red, upper_red)  # 利用inrange函数提取红色区域
    mask1 = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('mask', mask)
    cv2.imshow('mask1', mask1)

    # 2、形态学膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
    dilate = cv2.dilate(mask, kernel)  # 膨胀
    dilate1 = cv2.dilate(mask1, kernel)  # 膨胀
    edges = cv2.Canny(dilate, 30, 70)
    edges1 = cv2.Canny(dilate1, 30, 70)

    # 3、使用cv2.findContours()函数来查找检测物体的轮廓。
    binary, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary1, contours1, hierarchy1 = cv2.findContours(dilate1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 4、绘制轮廓
    cv2.drawContours(frame, contours, -1, (255, 0, 255), 3)
    cv2.drawContours(frame, contours1, -1, (255, 255, 0), 3)

    # 5、霍夫变换检测圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=20, minRadius=10, maxRadius=200)
    circles1 = cv2.HoughCircles(edges1, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=20, minRadius=10, maxRadius=200)
    # circles = np.int0(np.around(circles))
    # circles1 = np.int0(np.around(circles1))

    # # 将检测的圆画出来
    # for i in circles[0, :]:
    #     cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
    #     cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心
    cv2.imshow('Result', frame)

    # 6、判断是否为红绿灯
    if circles is not None and detection(contours) != 0:
        print("red light")
    if circles1 is not None and detection(contours1) != 0:
        print("green light")

    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
