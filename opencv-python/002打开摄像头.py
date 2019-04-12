'''
1. 打开摄像头、并捕获照片、录制视频
2. 播放本地视频
opencv函数：cv2.VideoCapture(), cv2.VideoWriter()
'''
# 导入相应的包
import cv2

# 定义相应的常量

# 定义相应的函数

# main函数
'''
# ----------------------------------------------------------------------------------------------------------------
# 打开0号摄像头
capture = cv2.VideoCapture(0)
while(True):
    # 获取一帧
    # 函数返回的第1个参数ret(return value缩写)是一个布尔值，表示当前这一帧是否获取正确。
    ret, frame = capture.read()
    # 转换颜色，这里将彩色图转成灰度图。
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('camer', frame)

    if cv2.waitKey(1) == ord('q'):
        break
'''


# ----------------------------------------------------------------------------------------------------------------
# 播放本地视频
# 把摄像头的编号换成视频的路径就可以播放本地视频
# cv2.waitKey()，它的参数表示暂停时间，所以这个值越大，视频播放速度越慢，反之，播放速度越快，通常设置为25或30。
capture = cv2.VideoCapture('output.avi')

while(capture.isOpened()):
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    # orq返回的是对应的ASCII的十进制整数形式。
    if cv2.waitKey(30) == ord('q'):
        break

'''
# ----------------------------------------------------------------------------------------------------------------
# 录制视频
# 保存图片用的是cv2.imwrite()
# 保存视频，我们需要创建一个VideoWriter的对象
capture = cv2.VideoCapture(0)

# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# 第一个参数：输出文件名，第二个参数：编码方式FourCC码，第三个参数：帧率，第四个参数保存分辨率大小
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))

while(capture.isOpened()):
    ret, frame = capture.read()

    if ret:
        outfile.write(frame)  # 写入文件
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
'''