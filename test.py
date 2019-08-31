import cv2
import numpy as np


# cap = cv2.VideoCapture(0)
#
# while(1):
#     # capture frame
#     ret, frame = cap.read()
#
#     if ret:
#         cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destoryAllWindows()


# 저장된 이미지를 여는 코드
# image = cv2.imread("D://python/santana.jpg", cv2.IMREAD_UNCHANGED)
# cv2.imshow("santana", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 내장카메라 or 외장카메라에서 영상을 받아온다
# 0 = 카메라의 장치번호. 노트북에서 내장카메라는 장치번호가 0이다
# 카메라를 추가연결하여 외장카메라를 사용하는 경우, 장치번호가 1~n까지 변화한다
capture = cv2.VideoCapture(0)

# 프레임의 속성 정의
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while 문을 이용, 영상 출력을 반복한다
while True:
    # ret = return: 카메라가 정상 작동 할 경우, True를 반환. 작동하지 않을 경우 False를 반환
    # cature.read(): 카메라의 상태 및 프레임을 받아옴
    # frame에 현재 프레임이 저장된다
    ret, frame = capture.read()

    # 윈도우 창에 이미지를 띄운다
    cv2.imshow("Title", frame)

    # 키보드 입력이 있을 때까지 while 문을 반복한다
    # waitKey = 키보드 입력을 대기하는 함수. 0이면 입력이 있을 때까지 대기한다
    if cv2.waitKey(1) > 0:
        break

# 카메라 장치에서 받아온 메모리를 해제한다
capture.release()
# 모든 윈도우 창 닫음
cv2.destroyAllWindows()
# cv2.destroyWindow("윈도우 창 제목") 도 가능