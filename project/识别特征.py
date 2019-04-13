import cv2
import numpy as np
import matplotlib.pyplot as plt

# 滑动条的回调函数
def nothing(x):
    pass

WindowName = 'img'
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('value_zoom', WindowName, 100, 100, nothing)
cv2.createTrackbar('value_threshold', WindowName, 70, 100, nothing)

# ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# edges = cv2.Canny(thresh, 30, 70)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img)

img = cv2.imread('002.jpg', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("template.PNG", 0)



while (1):
    value_zoom = cv2.getTrackbarPos('value_zoom', WindowName) / 100
    value_threshold = cv2.getTrackbarPos('value_threshold', WindowName) / 100
    template = cv2.resize(template, (0, 0), fx=value_zoom, fy=value_zoom, interpolation=cv2.INTER_NEAREST)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    locs = np.where(res >= value_threshold)
    for loc in zip(*locs[::-1]):
        img = cv2.rectangle(img, loc, (loc[0] + w, loc[1] + h), (0, 0, 255), 3)

    cv2.imshow(WindowName, img)

    k = cv2.waitKey(5)
    if k == ord('q'):
        break
cv2.destroyAllWindows()


