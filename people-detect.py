import cv2
from win10toast import ToastNotifier
import time

# 加载人脸识别分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
toaster = ToastNotifier()

# 打开摄像头
cap = cv2.VideoCapture(0)

start_time = None
detected_faces = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) >= 2:  # 如果检测到两个或更多人脸
        detected_faces += 1
        if detected_faces == 1:
            start_time = time.time()
        elif detected_faces >= 1 and time.time() - start_time > 1:  # 如果两个人停留时间超过2秒
            print("有两个人且停留时间超过1秒！")
            toaster.show_toast("提醒", "有人出现！", duration=5)

    else:
        detected_faces = 0

    # 显示视频流（可选）
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 按q退出该功能


cap.release()
cv2.destroyAllWindows()