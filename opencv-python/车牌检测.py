import cv2
import numpy as np

# 参数设定
# 蓝色范围
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])


def bwareaopen(binary, threshold):
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < threshold:
            cv2.drawContours(binary, [contours[i]], 0, 0, -1)
    return binary


img = cv2.imread('car_num.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(threshold, kernel)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
image, contours, hierarchy = cv2.findContours(
    mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
maxContour = contours[0]
for i in range(len(contours)):
    if len(contours[i]) >= len(maxContour):
        maxContour = contours[i]
rect = cv2.minAreaRect(maxContour)  # 找到最小矩形区域
box = cv2.boxPoints(rect)  # 找到最小矩形的顶点
box = np.int0(box)
mask = np.zeros_like(dilate)
cv2.fillPoly(mask, [box], 255)
masked_img = cv2.bitwise_and(dilate, mask)
bwareaopen = bwareaopen(masked_img, 700)

roi = bwareaopen[min(box[1][1], box[2][1]):max(box[0][1], box[3][1]),
                                           min(box[1][0], box[0][0]):max(box[2][0], box[3][0])]
# clossing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
# canny = cv2.Canny(clossing, 100, 200)
print(roi.shape)
cv2.imshow('roi', roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
